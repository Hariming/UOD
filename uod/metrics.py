"""Transparent AP metric for fully labeled, non-crowd bounding boxes.

IoU .50:.05:.95; 101 recall samples; classes without GT excluded. No COCO crowd,
ignore-region, area-range, or difficult-object handling. NOT an official dataset
benchmark evaluator. Use the dataset's official evaluator for published results.
"""
from __future__ import annotations
import numpy as np
import torch
from torchvision.ops import box_iou


class DetectionAP:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.predictions, self.targets = [], []

    def update(self, predictions: list[dict], targets: list[dict]):
        if len(predictions) != len(targets):
            raise ValueError("Prediction/target lengths disagree")
        self.predictions.extend([{k: v.detach().cpu() for k, v in p.items()} for p in predictions])
        self.targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])

    def compute(self) -> dict:
        thresholds = np.linspace(.5, .95, 10)
        by_class = {}
        aps = []
        for cls in range(self.num_classes):
            n_gt, all_scores, all_tp = 0, [], []
            for pred, gt in zip(self.predictions, self.targets, strict=True):
                gt_boxes = gt["boxes"][gt["labels"] == cls].float()
                mask = pred["labels"] == cls
                scores, boxes = pred["scores"][mask], pred["boxes"][mask].float()
                order = scores.argsort(descending=True, stable=True)
                scores, boxes = scores[order], boxes[order]
                n_gt += len(gt_boxes)
                tp = np.zeros((len(boxes), len(thresholds)), dtype=np.float64)
                if len(boxes) and len(gt_boxes):
                    ious = box_iou(boxes, gt_boxes).numpy()
                    for j, threshold in enumerate(thresholds):
                        used = np.zeros(len(gt_boxes), dtype=bool)
                        for p in range(len(boxes)):
                            row = ious[p].copy()
                            row[used] = -1
                            g = int(row.argmax())
                            if row[g] >= threshold:
                                tp[p, j] = 1
                                used[g] = True
                all_scores.append(scores.numpy())
                all_tp.append(tp)
            if n_gt == 0:
                by_class[str(cls)] = {"num_gt": 0, "ap50": None, "ap50_95": None}
                continue
            scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
            tps = np.concatenate(all_tp) if all_tp else np.zeros((0, 10))
            order = np.argsort(-scores, kind="stable")
            tps = tps[order]
            tp_acc = tps.cumsum(0)
            fp_acc = (1 - tps).cumsum(0)
            recall = tp_acc / n_gt
            precision = tp_acc / np.maximum(tp_acc + fp_acc, 1e-12)
            ap = np.zeros(10)
            for j in range(10):
                p = precision[:, j]
                envelope = np.maximum.accumulate(p[::-1])[::-1] if len(p) else p
                indices = np.searchsorted(recall[:, j], np.linspace(0., 1., 101), side="left")
                samples = np.zeros(101)
                valid = indices < len(envelope)
                samples[valid] = envelope[indices[valid]]
                ap[j] = samples.mean()
            aps.append(ap)
            by_class[str(cls)] = {"num_gt": n_gt, "ap50": float(ap[0]), "ap50_95": float(ap.mean())}
        if not aps:
            return {"map50": None, "map50_95": None, "per_class": by_class,
                    "images": len(self.targets), "note": "No GT objects: AP is undefined"}
        means = np.stack(aps).mean(0)
        return {"map50": float(means[0]), "map50_95": float(means.mean()),
                "per_class": by_class, "images": len(self.targets)}
