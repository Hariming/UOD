#!/usr/bin/env python3
"""Inference with RGB-only, thermal-only, or aligned RGBT. No NFE instantiated."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from uod.data import read_rgb, read_thermal, letterbox_tensor
from uod.engine import load_detector, select_device
from uod.geometry import candidates, fuse_candidates, suppress, undo_letterbox


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["rgb", "thermal", "rgbt"], required=True)
    p.add_argument("--rgb"); p.add_argument("--thermal")
    p.add_argument("--device", default="cpu")
    p.add_argument("--score", type=float, default=.05)
    p.add_argument("--nms-iou", type=float, default=.5)
    p.add_argument("--max-detections", type=int, default=100)
    p.add_argument("--output", default="prediction")
    a = p.parse_args()
    if a.mode in {"rgb", "rgbt"} and not a.rgb: p.error("This mode requires --rgb")
    if a.mode in {"thermal", "rgbt"} and not a.thermal: p.error("This mode requires --thermal")
    device = select_device(a.device)
    model, ckpt = load_detector(a.checkpoint, device)
    prep = ckpt["preprocessing"]; size = int(prep["image_size"]); fill = prep["letterbox_fill"]
    raw_r = read_rgb(a.rgb) if a.rgb else None
    raw_t = read_thermal(a.thermal, prep["thermal_divisor"]) if a.thermal else None
    if a.mode == "rgbt" and raw_r.shape[-2:] != raw_t.shape[-2:]:
        raise ValueError("RGBT requires registered images in the same original coordinate frame")
    if a.mode == "rgbt":
        r, mask, meta = letterbox_tensor(raw_r, size, fill)
        t, _, meta_t = letterbox_tensor(raw_t, size, fill)
        if meta != meta_t: raise AssertionError("Different geometric transforms")
        pred_r, pred_t = model.paired_predictions(r[None].to(device), t[None].to(device))
        cr = candidates(pred_r, (size, size), mask[None].to(device), a.score)[0]
        ct = candidates(pred_t, (size, size), mask[None].to(device), a.score)[0]
        result = fuse_candidates(cr, ct, a.nms_iou, a.max_detections)
        canvas = raw_r
    else:
        canvas = raw_r if a.mode == "rgb" else raw_t
        x, mask, meta = letterbox_tensor(canvas, size, fill)
        pred = model(x[None].to(device))
        result = suppress(candidates(pred, (size,size), mask[None].to(device), a.score)[0],
                          a.nms_iou, a.max_detections)
    result["boxes"] = undo_letterbox(result["boxes"], meta)
    result = {k: v.cpu() for k, v in result.items()}
    records = [{"class_id": int(label), "class_name": ckpt["class_names"][int(label)],
                "score": float(score), "xyxy": box.tolist()}
               for box, score, label in zip(result["boxes"], result["scores"], result["labels"], strict=True)]
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"mode": a.mode,
        "class_names": ckpt["class_names"], "detections": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    if canvas.shape[0] == 1: canvas = canvas.repeat(3, 1, 1)
    image = Image.fromarray((canvas.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for r in records:
        box = r["xyxy"]
        draw.rectangle(box, outline=(255, 220, 0), width=2)
        draw.text((box[0], max(0, box[1]-12)), f"{r['class_id']} {r['score']:.3f}", fill=(255, 220, 0))
    image.save(out.with_suffix(".png"))
    print(f"{len(records)} detections -> {out.with_suffix('.json')} and {out.with_suffix('.png')}")
    print("NFE modules loaded:", [n for n, _ in model.named_modules() if n.startswith("nfe_")])


if __name__ == "__main__": main()
