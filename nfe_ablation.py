#!/usr/bin/env python3
"""Evaluate whether NFE features still carry detection signal.

This is a direct-head diagnostic: BFE -> NFE -> the trained shared IFD. If NFE
AP is high, non-discriminative branches likely retain task-discriminative
information. Low AP is encouraging, but not a formal mutual-information proof.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from uod.engine import build_dataset, evaluate, load_checkpoint, make_loader, move_batch, read_config, select_device
from uod.geometry import candidates, fuse_candidates, suppress, undo_letterbox
from uod.losses import FeatureSeparationLoss
from uod.metrics import DetectionAP
from uod.model import ModelSpec, UOD


def load_training_model(path: str | Path, device: torch.device) -> tuple[UOD, dict]:
    checkpoint = load_checkpoint(path)
    if checkpoint["kind"] != "uod_training":
        raise ValueError("NFE ablation needs a full training checkpoint; deploy checkpoints have no NFE")
    model = UOD(ModelSpec(**checkpoint["model_spec"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def evaluate_nfe_direct(model: UOD, loader, device: torch.device, postprocess: dict,
                        cross: bool = False) -> dict:
    pp = {"score_threshold": .001, "pre_nms_topk": 3000, "iou_threshold": .5, "max_detections": 100}
    pp.update(postprocess or {})
    modes = ["nfe_rgb", "nfe_thermal", "nfe_rgbt"]
    if cross:
        modes += ["nfe_rgb_on_thermal", "nfe_thermal_on_rgb"]
    metrics = {mode: DetectionAP(model.spec.num_classes) for mode in modes}
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        size = batch["rgb"].shape[-2:]
        basic_r = model.bfe(batch["rgb"])
        basic_t = model.bfe(batch["thermal"])
        out_nr = model.ifd(model.nfe_rgb(basic_r))
        out_nt = model.ifd(model.nfe_thermal(basic_t))
        cr = candidates(out_nr, size, batch["valid_mask"], pp["score_threshold"], pp["pre_nms_topk"])
        ct = candidates(out_nt, size, batch["valid_mask"], pp["score_threshold"], pp["pre_nms_topk"])
        pred_r, pred_t, pred_rt = [], [], []
        for r, t, meta in zip(cr, ct, batch["meta"], strict=True):
            pr = suppress(r, pp["iou_threshold"], pp["max_detections"])
            pt = suppress(t, pp["iou_threshold"], pp["max_detections"])
            pf = fuse_candidates(r, t, pp["iou_threshold"], pp["max_detections"])
            for pred in (pr, pt, pf):
                pred["boxes"] = undo_letterbox(pred["boxes"], meta)
            pred_r.append(pr); pred_t.append(pt); pred_rt.append(pf)
        metrics["nfe_rgb"].update(pred_r, raw_batch["original_targets"])
        metrics["nfe_thermal"].update(pred_t, raw_batch["original_targets"])
        metrics["nfe_rgbt"].update(pred_rt, raw_batch["original_targets"])
        if cross:
            out_r_on_t = model.ifd(model.nfe_rgb(basic_t))
            out_t_on_r = model.ifd(model.nfe_thermal(basic_r))
            cross_r = candidates(out_r_on_t, size, batch["valid_mask"], pp["score_threshold"], pp["pre_nms_topk"])
            cross_t = candidates(out_t_on_r, size, batch["valid_mask"], pp["score_threshold"], pp["pre_nms_topk"])
            pred_cross_r, pred_cross_t = [], []
            for r, t, meta in zip(cross_r, cross_t, batch["meta"], strict=True):
                pr = suppress(r, pp["iou_threshold"], pp["max_detections"])
                pt = suppress(t, pp["iou_threshold"], pp["max_detections"])
                pr["boxes"] = undo_letterbox(pr["boxes"], meta)
                pt["boxes"] = undo_letterbox(pt["boxes"], meta)
                pred_cross_r.append(pr); pred_cross_t.append(pt)
            metrics["nfe_rgb_on_thermal"].update(pred_cross_r, raw_batch["original_targets"])
            metrics["nfe_thermal_on_rgb"].update(pred_cross_t, raw_batch["original_targets"])
    return {mode: metric.compute() for mode, metric in metrics.items()}


@torch.no_grad()
def feature_distance_stats(model: UOD, loader, device: torch.device, loss_cfg: dict | None = None) -> dict:
    cfg = dict(loss_cfg or {})
    cfg.pop("alpha", None)
    cfg.pop("detection", None)
    separation = FeatureSeparationLoss(**cfg).to(device)
    totals, seen = {}, 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        basic_r = model.bfe(batch["rgb"])
        basic_t = model.bfe(batch["thermal"])
        _, stats = separation(model.ife(basic_r), model.ife(basic_t),
                              model.nfe_rgb(basic_r), model.nfe_thermal(basic_t),
                              batch["valid_mask"])
        b = batch["rgb"].shape[0]
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.) + float(value) * b
        seen += b
    return {key: value / max(seen, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="nfe_ablation.json")
    parser.add_argument("--cross", action="store_true", help="also test each modality with the opposite NFE branch")
    args = parser.parse_args()

    cfg = read_config(args.config)
    device = select_device(args.device)
    model, checkpoint = load_training_model(args.checkpoint, device)
    if checkpoint["class_names"] != cfg["data"]["class_names"]:
        raise ValueError("Checkpoint and data class mapping disagree")
    for key, value in checkpoint["preprocessing"].items():
        if key == "image_size":
            continue
        if cfg["data"].get(key, value) != value:
            raise ValueError(f"Preprocessing mismatch: {key}")

    dataset = build_dataset(cfg, "val", False)
    loader = make_loader(dataset, cfg["training"].get("val_batch_size", 4), args.workers, False, device)
    results = {
        "ife_baseline": evaluate(model, loader, device, cfg.get("postprocess", {})),
        "nfe_direct_head": evaluate_nfe_direct(model, loader, device, cfg.get("postprocess", {}), args.cross),
        "feature_distance_stats": feature_distance_stats(model, loader, device, cfg.get("loss", {})),
        "notes": {
            "nfe_direct_head": "Runs trained IFD on NFE features without retraining the head.",
            "interpretation": "High NFE AP suggests task information leakage; low NFE AP is useful evidence but not MI=0 proof.",
        },
    }
    text = json.dumps(results, indent=2, ensure_ascii=False)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
