"""UOD feature-sharing/separation reference implementation.

Paper: Fan et al., Unifying RGB and thermal object detection in one detector,
Pattern Recognition 179 (2026), 113902, Sections 3.1--3.4.

The paper's parameter-sharing graph is implemented here. The concrete
backbone/neck/head choices are configurable baselines, not a drop-in copy of
the authors' training code.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import models

Pyramid = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class ModelSpec:
    num_classes: int
    backbone: str = "resnet50"
    neck: str = "simple_pafpn"
    channels: int = 256
    head_depth: int = 2
    normalization: str = "imagenet"
    nfe_init: str = "perturbed_copy"
    nfe_noise: float = 0.05

    def __post_init__(self) -> None:
        if self.num_classes < 1 or self.channels < 8 or self.head_depth < 1:
            raise ValueError("Invalid num_classes, channels, or head_depth")
        if self.backbone not in {"resnet18", "resnet50", "yolov5n", "yolov5s", "yolov5m", "yolov5l"}:
            raise ValueError("backbone must be resnet18, resnet50, or yolov5n/s/m/l")
        if self.neck not in {"simple_pafpn", "yolov5_pafpn"}:
            raise ValueError("neck must be simple_pafpn or yolov5_pafpn")
        if self.normalization not in {"imagenet", "none"}:
            raise ValueError("normalization must be imagenet or none")
        if self.nfe_init not in {"independent", "perturbed_copy"}:
            raise ValueError("nfe_init must be independent or perturbed_copy")
        if self.nfe_init == "perturbed_copy" and self.nfe_noise <= 0:
            raise ValueError("A positive perturbation avoids identical feature initialization")


def ensure_three_channels(x: Tensor) -> Tensor:
    """Accept BCHW float images in [0, 1]. No learned thermal adapter.

    Thermal's one channel is replicated BEFORE the common normalization.
    Channel replication itself is NOT what shares weights: using one module is.
    """
    if x.ndim != 4 or x.shape[1] not in (1, 3):
        raise ValueError(f"Expected [B,1/3,H,W], got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError("Images must be floating point, scaled to [0, 1]")
    if min(x.shape[-2:]) < 64 or any(v % 32 for v in x.shape[-2:]):
        raise ValueError("H/W must be multiples of 32 and at least 64; use letterbox")
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(math.ceil(value / divisor) * divisor))


YOLOV5_SCALES = {
    "yolov5n": (0.33, 0.25),
    "yolov5s": (0.33, 0.50),
    "yolov5m": (0.67, 0.75),
    "yolov5l": (1.00, 1.00),
}


def yolo_depth(backbone: str, repeats: int) -> int:
    depth, _ = YOLOV5_SCALES[backbone]
    return max(round(repeats * depth), 1) if repeats > 1 else repeats


def yolo_channels(backbone: str, channels: int) -> int:
    _, width = YOLOV5_SCALES[backbone]
    return make_divisible(channels * width, 8)


def autopad(kernel: int, padding: int | None = None) -> int:
    return kernel // 2 if padding is None else padding


class YOLOConv(nn.Module):
    """YOLOv5-style Conv-BN-SiLU block."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 1, stride: int = 1,
                 padding: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel, stride, autopad(kernel, padding), bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c_in: int, c_out: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden = int(c_out * expansion)
        self.cv1 = YOLOConv(c_in, hidden, 1)
        self.cv2 = YOLOConv(hidden, c_out, 3)
        self.add = shortcut and c_in == c_out

    def forward(self, x: Tensor) -> Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """YOLOv5 C3/CSP block."""

    def __init__(self, c_in: int, c_out: int, repeats: int = 1, shortcut: bool = True,
                 expansion: float = 0.5):
        super().__init__()
        hidden = int(c_out * expansion)
        self.cv1 = YOLOConv(c_in, hidden, 1)
        self.cv2 = YOLOConv(c_in, hidden, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden, hidden, shortcut, 1.0) for _ in range(repeats)])
        self.cv3 = YOLOConv(2 * hidden, c_out, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling-fast block used by YOLOv5."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 5):
        super().__init__()
        hidden = c_in // 2
        self.cv1 = YOLOConv(c_in, hidden, 1)
        self.cv2 = YOLOConv(hidden * 4, c_out, 1)
        self.pool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.cv2(torch.cat((x, y1, y2, self.pool(y2)), dim=1))


class YOLOv5Backbone(nn.Module):
    """CSPDarknet-style YOLOv5 backbone, outputs P3/P4/P5 inputs."""

    def __init__(self, backbone: str):
        super().__init__()
        c1 = yolo_channels(backbone, 64)
        c2 = yolo_channels(backbone, 128)
        c3 = yolo_channels(backbone, 256)
        c4 = yolo_channels(backbone, 512)
        c5 = yolo_channels(backbone, 1024)
        self.out_channels = (c3, c4, c5)
        self.stem = YOLOConv(3, c1, 6, 2, 2)
        self.stage2 = nn.Sequential(YOLOConv(c1, c2, 3, 2), C3(c2, c2, yolo_depth(backbone, 3)))
        self.stage3 = nn.Sequential(YOLOConv(c2, c3, 3, 2), C3(c3, c3, yolo_depth(backbone, 6)))
        self.stage4 = nn.Sequential(YOLOConv(c3, c4, 3, 2), C3(c4, c4, yolo_depth(backbone, 9)))
        self.stage5 = nn.Sequential(YOLOConv(c4, c5, 3, 2), C3(c5, c5, yolo_depth(backbone, 3)), SPPF(c5, c5))

    def forward(self, x: Tensor) -> Pyramid:
        x = self.stage2(self.stem(x))
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


class BFE(nn.Module):
    """ONE shared backbone, outputs C3/C4/C5 at strides 8/16/32."""

    def __init__(self, spec: ModelSpec, pretrained: bool = False) -> None:
        super().__init__()
        self.kind = "resnet"
        if spec.backbone in YOLOV5_SCALES:
            if pretrained:
                raise ValueError("pretrained=True is only implemented for torchvision ResNet backbones")
            self.kind = "yolov5"
            self.net = YOLOv5Backbone(spec.backbone)
            self.out_channels = self.net.out_channels
        else:
            if spec.backbone == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                net = models.resnet50(weights=weights)
                self.out_channels = (512, 1024, 2048)
            else:
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                net = models.resnet18(weights=weights)
                self.out_channels = (128, 256, 512)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1, self.layer2 = net.layer1, net.layer2
            self.layer3, self.layer4 = net.layer3, net.layer4
        if spec.normalization == "imagenet":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean, std = (0., 0., 0.), (1., 1., 1.)
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Pyramid:
        x = (ensure_three_channels(x) - self.mean) / self.std
        if self.kind == "yolov5":
            return self.net(x)
        x = self.layer1(self.stem(x))
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


def group_count(channels: int) -> int:
    # Keep >= 4 channels per group for small feature maps.
    return next(g for g in range(min(32, channels // 4), 0, -1) if channels % g == 0)


class ConvNormAct(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, kernel: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(c_in, c_out, kernel, stride, kernel // 2, bias=False),
            nn.GroupNorm(group_count(c_out), c_out),
            nn.SiLU(inplace=False),
        )


class FeatureNeck(nn.Module):
    """FPN top-down + PAN bottom-up; used IDENTICALLY for IFE and each NFE.

    No literal subtraction or reconstruction constraint is imposed. Eq. (1)
    in the paper is conceptual; the separation is learned through the losses.
    """

    def __init__(self, in_channels: Sequence[int], channels: int):
        super().__init__()
        self.lateral = nn.ModuleList([ConvNormAct(c, channels, 1) for c in in_channels])
        self.top3 = ConvNormAct(channels, channels)
        self.top4 = ConvNormAct(channels, channels)
        self.top5 = ConvNormAct(channels, channels)
        self.down3 = ConvNormAct(channels, channels, 3, 2)
        self.down4 = ConvNormAct(channels, channels, 3, 2)
        self.bottom4 = ConvNormAct(channels, channels)
        self.bottom5 = ConvNormAct(channels, channels)

    def forward(self, features: Pyramid) -> Pyramid:
        l3, l4, l5 = [layer(x) for layer, x in zip(self.lateral, features, strict=True)]
        p5 = self.top5(l5)
        p4 = self.top4(l4 + F.interpolate(p5, size=l4.shape[-2:], mode="nearest"))
        p3 = self.top3(l3 + F.interpolate(p4, size=l3.shape[-2:], mode="nearest"))
        n4 = self.bottom4(p4 + self.down3(p3))
        n5 = self.bottom5(p5 + self.down4(n4))
        return p3, n4, n5


class YOLOv5PAFPN(nn.Module):
    """YOLOv5-style PAN-FPN with concat + C3 blocks.

    The internal topology follows YOLOv5's neck pattern, while the outputs are
    projected to a common channel count so the shared decoupled IFD can remain
    identical across ResNet and YOLOv5-style backbones.
    """

    def __init__(self, in_channels: Sequence[int], channels: int, depth_multiple: float = 0.33):
        super().__init__()
        c3, c4, c5 = in_channels
        repeats = max(round(3 * depth_multiple), 1)
        self.reduce5 = YOLOConv(c5, channels, 1)
        self.c3_p4 = C3(channels + c4, channels, repeats, shortcut=False)
        self.reduce4 = YOLOConv(channels, channels, 1)
        self.c3_p3 = C3(channels + c3, channels, repeats, shortcut=False)
        self.down3 = YOLOConv(channels, channels, 3, 2)
        self.c3_n4 = C3(channels * 2, channels, repeats, shortcut=False)
        self.down4 = YOLOConv(channels, channels, 3, 2)
        self.c3_n5 = C3(channels * 2, channels, repeats, shortcut=False)

    def forward(self, features: Pyramid) -> Pyramid:
        c3, c4, c5 = features
        p5 = self.reduce5(c5)
        p4 = self.c3_p4(torch.cat((F.interpolate(p5, size=c4.shape[-2:], mode="nearest"), c4), dim=1))
        p4_reduced = self.reduce4(p4)
        p3 = self.c3_p3(torch.cat((F.interpolate(p4_reduced, size=c3.shape[-2:], mode="nearest"), c3), dim=1))
        n4 = self.c3_n4(torch.cat((self.down3(p3), p4_reduced), dim=1))
        n5 = self.c3_n5(torch.cat((self.down4(n4), p5), dim=1))
        return p3, n4, n5


def build_neck(spec: ModelSpec, in_channels: Sequence[int]) -> nn.Module:
    if spec.neck == "simple_pafpn":
        return FeatureNeck(in_channels, spec.channels)
    depth = YOLOV5_SCALES.get(spec.backbone, (0.33, 0.50))[0]
    return YOLOv5PAFPN(in_channels, spec.channels, depth)


@dataclass
class HeadOutput:
    cls_logits: tuple[Tensor, ...]  # [B, C+1, H_l, W_l]; background is class C
    distances: tuple[Tensor, ...]  # [B, 4, H_l, W_l], positive l/t/r/b in PIXELS
    strides: tuple[int, ...] = (8, 16, 32)

    def slice(self, start: int, end: int) -> "HeadOutput":
        return HeadOutput(tuple(t[start:end] for t in self.cls_logits),
                          tuple(t[start:end] for t in self.distances), self.strides)


class IFD(nn.Module):
    """Decoupled classification and regression towers.

    Towers have DIFFERENT parameters from each other, but this ONE head is
    shared by RGB and thermal. C+1 softmax permits categorical CE + background.
    This is not the official YOLOv8 Detect head.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.cls_tower = nn.Sequential(*[
            ConvNormAct(spec.channels, spec.channels) for _ in range(spec.head_depth)])
        self.reg_tower = nn.Sequential(*[
            ConvNormAct(spec.channels, spec.channels) for _ in range(spec.head_depth)])
        self.cls_pred = nn.Conv2d(spec.channels, spec.num_classes + 1, 3, padding=1)
        self.reg_pred = nn.Conv2d(spec.channels, 4, 3, padding=1)
        self.level_scales = nn.Parameter(torch.ones(3))
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.zeros_(self.reg_pred.bias)
        with torch.no_grad():
            self.cls_pred.bias[:spec.num_classes].fill_(math.log(0.01 / spec.num_classes))
            self.cls_pred.bias[spec.num_classes].fill_(math.log(0.99))

    def forward(self, features: Pyramid) -> HeadOutput:
        cls, reg = [], []
        for i, (x, stride) in enumerate(zip(features, (8, 16, 32), strict=True)):
            cls.append(self.cls_pred(self.cls_tower(x)))
            # FP32 softplus avoids exponential overflow in AMP; no DFL/IoU loss.
            r = self.reg_pred(self.reg_tower(x)).float()
            reg.append(F.softplus(r * self.level_scales[i]) * stride)
        return HeadOutput(tuple(cls), tuple(reg))


class UODDetector(nn.Module):
    """Deployment graph: BFE -> IFE -> IFD. Contains NO NFE modules."""

    def __init__(self, spec: ModelSpec, pretrained: bool = False):
        super().__init__()
        self.spec = spec
        self.bfe = BFE(spec, pretrained)
        self.ife = build_neck(spec, self.bfe.out_channels)
        self.ifd = IFD(spec)

    def forward(self, image: Tensor) -> HeadOutput:
        return self.ifd(self.ife(self.bfe(image)))

    def paired_predictions(self, rgb: Tensor, thermal: Tensor) -> tuple[HeadOutput, HeadOutput]:
        rgb, thermal = ensure_three_channels(rgb), ensure_three_channels(thermal)
        if rgb.shape != thermal.shape:
            raise ValueError("Paired RGB/thermal must have identical B,H,W")
        b = rgb.shape[0]
        pred = self(torch.cat([rgb, thermal], dim=0))
        return pred.slice(0, b), pred.slice(b, 2 * b)


class UOD(UODDetector):
    """Training graph. Shared modules physically exist once, NFE exists twice."""

    def __init__(self, spec: ModelSpec, pretrained: bool = False):
        super().__init__(spec, pretrained)
        self.nfe_rgb = build_neck(spec, self.bfe.out_channels)
        self.nfe_thermal = build_neck(spec, self.bfe.out_channels)
        if spec.nfe_init == "perturbed_copy":
            # Equal architecture, independent parameters, close but NOT identical
            # initial features. This initialization is our engineering choice.
            for nfe in (self.nfe_rgb, self.nfe_thermal):
                nfe.load_state_dict(self.ife.state_dict())
                with torch.no_grad():
                    for module in nfe.modules():
                        if isinstance(module, nn.Conv2d):
                            scale = module.weight.std().clamp_min(1e-4)
                            module.weight.add_(torch.randn_like(module.weight) * scale * spec.nfe_noise)

    def forward_pair(self, rgb: Tensor, thermal: Tensor) -> dict:
        """Train only. Mixing 2B images also balances shared BatchNorm statistics."""
        if not self.training:
            raise RuntimeError("Use paired_predictions() in eval; it never runs NFE")
        rgb, thermal = ensure_three_channels(rgb), ensure_three_channels(thermal)
        if rgb.shape != thermal.shape:
            raise ValueError("Paired RGB and thermal shapes must agree")
        b = rgb.shape[0]
        basic = self.bfe(torch.cat([rgb, thermal], dim=0))  # ONE call; one weight set
        intrinsic = self.ife(basic)                         # ONE shared IFE
        predictions = self.ifd(intrinsic)                    # ONE shared IFD
        basic_r = tuple(f[:b] for f in basic)
        basic_t = tuple(f[b:] for f in basic)
        return {
            "pred_rgb": predictions.slice(0, b),
            "pred_thermal": predictions.slice(b, 2 * b),
            "intrinsic_rgb": tuple(f[:b] for f in intrinsic),
            "intrinsic_thermal": tuple(f[b:] for f in intrinsic),
            "non_rgb": self.nfe_rgb(basic_r),
            "non_thermal": self.nfe_thermal(basic_t),
        }

    def to_deploy(self) -> UODDetector:
        # Avoid building a new pretrained backbone or allocating any extra NFE.
        deploy = UODDetector.__new__(UODDetector)
        nn.Module.__init__(deploy)
        deploy.spec = self.spec
        deploy.bfe = copy.deepcopy(self.bfe)
        deploy.ife = copy.deepcopy(self.ife)
        deploy.ifd = copy.deepcopy(self.ifd)
        return deploy.eval()


def model_spec_dict(model: UODDetector) -> dict:
    return asdict(model.spec)
