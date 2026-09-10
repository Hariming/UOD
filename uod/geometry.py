"""Anchor-point decoding, letterbox inversion, and decision-level RGBT fusion."""
from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops import batched_nms
from .model import HeadOutput


def flatten_predictions(out: HeadOutput):
    cls, distances, points, strides, levels = [], [], [], [], []
    for level, (c, r, s) in enumerate(zip(out.cls_logits, out.distances, out.strides, strict=True)):
        b, _, h, w = c.shape
        cls.append(c.flatten(2).transpose(1, 2))
        distances.append(r.flatten(2).transpose(1, 2))
        yy, xx = torch.meshgrid(torch.arange(h, device=c.device, dtype=torch.float32),
                                torch.arange(w, device=c.device, dtype=torch.float32), indexing="ij")
        points.append(torch.stack([(xx + .5) * s, (yy + .5) * s], dim=-1).reshape(-1, 2))
        strides.append(torch.full((h * w,), float(s), device=c.device))
        levels.append(torch.full((h * w,), level, device=c.device, dtype=torch.long))
    return (torch.cat(cls, 1).float(), torch.cat(distances, 1).float(),
            torch.cat(points), torch.cat(strides), torch.cat(levels))


def distances_to_boxes(points: Tensor, distances: Tensor) -> Tensor:
    # points: [N,2], distances: [B,N,4] or [N,4], in pixel units
    return torch.cat([points - distances[..., :2], points + distances[..., 2:]], dim=-1)


def valid_points(out: HeadOutput, valid_mask: Tensor | None) -> Tensor:
    b = out.cls_logits[0].shape[0]
    if valid_mask is None:
        n = sum(c.shape[-1] * c.shape[-2] for c in out.cls_logits)
        return torch.ones(b, n, device=out.cls_logits[0].device, dtype=torch.bool)
    return torch.cat([F.interpolate(valid_mask.float(), size=c.shape[-2:], mode="nearest-exact")
                      .flatten(1).bool() for c in out.cls_logits], dim=1)


@torch.no_grad()
def candidates(out: HeadOutput, image_size: tuple[int, int], valid_mask: Tensor | None = None,
               score_threshold: float = .05, pre_nms_topk: int = 2000) -> list[dict[str, Tensor]]:
    if not 0 <= score_threshold <= 1 or pre_nms_topk < 1:
        raise ValueError("Invalid postprocessing thresholds")
    logits, distances, points, _, _ = flatten_predictions(out)
    probs = logits.softmax(-1)[..., :-1]  # last logit is background, NOT objectness
    scores, labels = probs.max(dim=-1)   # one predicted class per candidate
    boxes = distances_to_boxes(points, distances)
    h, w = image_size
    boxes[..., 0::2].clamp_(0, w)
    boxes[..., 1::2].clamp_(0, h)
    valid = valid_points(out, valid_mask)
    results = []
    for b in range(boxes.shape[0]):
        keep = valid[b] & (scores[b] >= score_threshold)
        keep &= (boxes[b, :, 2] > boxes[b, :, 0]) & (boxes[b, :, 3] > boxes[b, :, 1])
        keep &= torch.isfinite(boxes[b]).all(-1) & torch.isfinite(scores[b])
        idx = keep.nonzero(as_tuple=False).flatten()
        if idx.numel() > pre_nms_topk:
            idx = idx[scores[b, idx].topk(pre_nms_topk).indices]
        results.append({"boxes": boxes[b, idx], "scores": scores[b, idx], "labels": labels[b, idx]})
    return results


@torch.no_grad()
def suppress(result: dict[str, Tensor], iou_threshold: float = .5, max_detections: int = 100):
    if not 0 <= iou_threshold <= 1 or max_detections < 1:
        raise ValueError("Invalid NMS settings")
    keep = batched_nms(result["boxes"].float(), result["scores"].float(),
                       result["labels"], iou_threshold)[:max_detections]
    return {k: v[keep] for k, v in result.items()}


def fuse_candidates(rgb: dict[str, Tensor], thermal: dict[str, Tensor],
                    iou_threshold: float = .5, max_detections: int = 100):
    """Concatenate PRE-NMS candidates; then exactly one class-aware NMS."""
    merged = {k: torch.cat([rgb[k], thermal[k]], dim=0) for k in ("boxes", "scores", "labels")}
    return suppress(merged, iou_threshold, max_detections)


def undo_letterbox(boxes: Tensor, meta: dict) -> Tensor:
    boxes = boxes.clone()
    sx, sy = meta["scale_xy"]
    px, py = meta["pad_xy"]
    h, w = meta["original_hw"]
    boxes[:, 0::2] = (boxes[:, 0::2] - px) / sx
    boxes[:, 1::2] = (boxes[:, 1::2] - py) / sy
    boxes[:, 0::2].clamp_(0, w)
    boxes[:, 1::2].clamp_(0, h)
    return boxes
