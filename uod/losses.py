"""Paper Eqs. (8)--(13), with explicit choices for dense detection targets.

Feature loss: pixelwise L2 normalization, squared Euclidean distance, separate
positive/hinge terms. NOT torch.nn.TripletMarginLoss. Multi-scale aggregation
and padding policy are explicitly configurable engineering choices.

Detection: categorical CE + L1. The paper does not specify dense assignment,
background weighting, hard-negative sampling, or coordinate normalization.
Those choices are exposed here; they are NOT claimed to reproduce YOLO loss.
"""
from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .model import HeadOutput
from .geometry import flatten_predictions, distances_to_boxes, valid_points


class FeatureSeparationLoss(nn.Module):
    def __init__(self, margin: float = 1., reduction: str = "pixel", mask_padding: bool = False):
        super().__init__()
        if not 0 < margin <= 4:
            raise ValueError("For L2-normalized features squared distance is <= 4; require 0 < margin <= 4")
        if reduction not in {"pixel", "level"}:
            raise ValueError("reduction must be pixel or level")
        self.margin, self.reduction, self.mask_padding = margin, reduction, mask_padding

    def forward(self, intrinsic_rgb: Sequence[Tensor], intrinsic_thermal: Sequence[Tensor],
                non_rgb: Sequence[Tensor], non_thermal: Sequence[Tensor],
                valid_mask: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        groups = (intrinsic_rgb, intrinsic_thermal, non_rgb, non_thermal)
        if not groups[0] or len({len(g) for g in groups}) != 1:
            raise ValueError("All four pyramids must contain the same nonzero number of levels")
        if self.mask_padding and valid_mask is None:
            raise ValueError("mask_padding=True needs a valid image mask")
        totals: dict[str, Tensor] = {}
        denominator = 0.
        for tensors in zip(*groups, strict=True):
            if len({tuple(t.shape) for t in tensors}) != 1 or tensors[0].ndim != 4:
                raise ValueError("Every compared feature must have the same BCHW shape")
            # Normalize each C-dimensional vector separately at every (b,y,x).
            # Cast to FP32 even when the backbone ran under AMP.
            ir, it, nr, nt = [F.normalize(t.float(), p=2, dim=1, eps=1e-6) for t in tensors]
            d_pos = (ir - it).square().sum(dim=1)  # [B,H,W], NOT mean across C
            d_r = (ir - nr).square().sum(dim=1)
            d_t = (it - nt).square().sum(dim=1)
            d_n = (nr - nt).square().sum(dim=1)
            neg_r, neg_t, neg_n = [(self.margin - d).clamp_min(0) for d in (d_r, d_t, d_n)]
            if self.mask_padding:
                mask = F.interpolate(valid_mask.float(), size=ir.shape[-2:], mode="nearest-exact")[:, 0]
            else:
                mask = torch.ones_like(d_pos)
            count = mask.sum().clamp_min(1)
            values = {
                "positive_d2": d_pos,
                "rgb_negative": neg_r,
                "thermal_negative": neg_t,
                "non_modal_negative": neg_n,
                "rgb_non_d2": d_r,
                "thermal_non_d2": d_t,
                "non_non_d2": d_n,
                "rgb_hinge_active": (d_r < self.margin).float(),
                "thermal_hinge_active": (d_t < self.margin).float(),
                "non_hinge_active": (d_n < self.margin).float(),
            }
            for key, value in values.items():
                amount = (value * mask).sum()
                if self.reduction == "level":
                    amount = amount / count
                totals[key] = totals.get(key, 0.) + amount
            denominator = denominator + (count if self.reduction == "pixel" else 1.)
        values = {k: v / denominator for k, v in totals.items()}
        # Eq. (12): the same positive term is present in BOTH L_C^RGB and L_C^T.
        loss_rgb = values["positive_d2"] + values["rgb_negative"]
        loss_thermal = values["positive_d2"] + values["thermal_negative"]
        loss_non = values["non_modal_negative"]
        loss = loss_rgb + loss_thermal + loss_non
        stats = {k: v.detach() for k, v in values.items()}
        stats.update(fs=loss.detach(), lc_rgb=loss_rgb.detach(), lc_thermal=loss_thermal.detach(),
                     ln=loss_non.detach())
        return loss, stats


class DetectionLoss(nn.Module):
    """Our dense realization of CE + L1, not official FCOS/YOLO training.

    Target assignment: points inside boxes, center sampling, size ranges,
    smallest-area GT in overlaps. Classes 0..C-1; background C; ignored -1.
    BCE/focal/CIoU/DFL are deliberately NOT substituted for paper Eq. (8).
    """

    def __init__(self, num_classes: int, center_radius: float = 1.5,
                 size_ranges: Sequence[Sequence[float]] = ((0, 64), (64, 128), (128, 1e8)),
                 background_weight: float = .1, negative_ratio: int = 3,
                 min_negatives: int = 32, box_normalization: str = "stride"):
        super().__init__()
        if len(size_ranges) != 3 or center_radius <= 0 or background_weight <= 0:
            raise ValueError("Invalid assignment/loss settings")
        if negative_ratio < 1 or min_negatives < 1 or box_normalization not in {"stride", "image", "none"}:
            raise ValueError("Invalid negative sampling or box normalization")
        self.num_classes, self.center_radius = num_classes, center_radius
        self.size_ranges = tuple(tuple(map(float, r)) for r in size_ranges)
        self.background_weight, self.negative_ratio = background_weight, negative_ratio
        self.min_negatives, self.box_normalization = min_negatives, box_normalization

    @torch.no_grad()
    def assign(self, points: Tensor, strides: Tensor, levels: Tensor,
               boxes: Tensor, labels: Tensor, valid: Tensor):
        n = points.shape[0]
        target_cls = torch.full((n,), self.num_classes, device=points.device, dtype=torch.long)
        target_cls[~valid] = -1
        target_boxes = torch.zeros(n, 4, device=points.device)
        if boxes.numel() == 0:
            return target_cls, target_boxes, 0
        if boxes.shape != (len(labels), 4):
            raise ValueError("GT boxes must be Nx4 and labels must be N")
        if (labels < 0).any() or (labels >= self.num_classes).any():
            raise ValueError("GT class id is outside configured class range")
        if ((boxes[:, 2:] - boxes[:, :2]) <= 0).any():
            raise ValueError("Degenerate ground-truth boxes")
        x, y = points[:, 0, None], points[:, 1, None]
        ltrb = torch.stack([x - boxes[None, :, 0], y - boxes[None, :, 1],
                            boxes[None, :, 2] - x, boxes[None, :, 3] - y], dim=-1)
        inside = ltrb.min(dim=-1).values > 0
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        in_center = ((points[:, None] - centers[None]).abs().amax(-1)
                     <= strides[:, None] * self.center_radius)
        ranges = torch.tensor(self.size_ranges, device=points.device)[levels]
        size = ltrb.max(dim=-1).values
        in_range = (size >= ranges[:, :1]) & (size < ranges[:, 1:])
        eligible = inside & in_center & in_range & valid[:, None]
        areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        cost = areas[None].expand(n, -1).clone().masked_fill_(~eligible, float("inf"))
        best_area, idx = cost.min(dim=1)
        positive = torch.isfinite(best_area)
        target_cls[positive] = labels[idx[positive]]
        target_boxes[positive] = boxes[idx[positive]]
        # Small objects can have no center at stride >=8; expose it, do not hide it.
        unmatched = len(boxes) - idx[positive].unique().numel()
        return target_cls, target_boxes, int(unmatched)

    def forward(self, out: HeadOutput, targets: list[dict[str, Tensor]],
                image_size: tuple[int, int], valid_mask: Tensor | None = None):
        logits, distances, points, strides, levels = flatten_predictions(out)
        decoded = distances_to_boxes(points, distances)
        valid = valid_points(out, valid_mask)
        if len(targets) != logits.shape[0]:
            raise ValueError("Number of targets must match batch size")
        cls_losses, reg_losses = [], []
        positive_count = unmatched_count = 0
        for i, target in enumerate(targets):
            classes, gt_boxes, unmatched = self.assign(points, strides, levels,
                target["boxes"], target["labels"], valid[i])
            positive = (classes >= 0) & (classes < self.num_classes)
            negative = classes == self.num_classes
            ce = F.cross_entropy(logits[i], classes, ignore_index=-1, reduction="none")
            n_pos, n_neg = int(positive.sum()), int(negative.sum())
            take = min(n_neg, max(self.min_negatives, self.negative_ratio * n_pos))
            neg_loss = ce[negative].topk(take).values.sum() if take else ce.sum() * 0.
            normalizer = max(1., n_pos + self.background_weight * take)
            cls_losses.append((ce[positive].sum() + self.background_weight * neg_loss) / normalizer)
            if n_pos:
                error = (decoded[i, positive] - gt_boxes[positive]).abs()
                if self.box_normalization == "stride":
                    error = error / strides[positive, None]
                elif self.box_normalization == "image":
                    h, w = image_size
                    error = error / error.new_tensor([w, h, w, h])
                reg_losses.append(error.mean())
            else:
                reg_losses.append(distances[i].sum() * 0.)
            positive_count += n_pos
            unmatched_count += unmatched
        cls_loss, reg_loss = torch.stack(cls_losses).mean(), torch.stack(reg_losses).mean()
        total = cls_loss + reg_loss
        return total, {"cls": cls_loss.detach(), "l1": reg_loss.detach(), "det": total.detach(),
                       "positives_per_image": total.new_tensor(positive_count / len(targets)),
                       "unmatched_gt_per_image": total.new_tensor(unmatched_count / len(targets))}


class UODLoss(nn.Module):
    def __init__(self, num_classes: int, alpha: float = 1., margin: float = 1.,
                 reduction: str = "pixel", mask_padding: bool = False, detection: dict | None = None):
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.separation = FeatureSeparationLoss(margin, reduction, mask_padding)
        self.detection = DetectionLoss(num_classes, **(detection or {}))

    def forward(self, outputs: dict, targets: list[dict[str, Tensor]],
                image_size: tuple[int, int], valid_mask: Tensor | None = None):
        det_r, stats_r = self.detection(outputs["pred_rgb"], targets, image_size, valid_mask)
        det_t, stats_t = self.detection(outputs["pred_thermal"], targets, image_size, valid_mask)
        fs, stats_fs = self.separation(outputs["intrinsic_rgb"], outputs["intrinsic_thermal"],
                                      outputs["non_rgb"], outputs["non_thermal"], valid_mask)
        # Each detector term is a batch mean. No hidden factor-of-B mismatch.
        loss = det_r + det_t + self.alpha * fs
        stats = {"loss": loss.detach(), "det_sum": (det_r + det_t).detach(), **stats_fs}
        stats.update({"rgb/" + k: v for k, v in stats_r.items()})
        stats.update({"thermal/" + k: v for k, v in stats_t.items()})
        return loss, stats
