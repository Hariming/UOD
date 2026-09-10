#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from uod.engine import build_dataset, evaluate, load_detector, make_loader, read_config, select_device


def main():
    p = argparse.ArgumentParser(description="Evaluate the same checkpoint on RGB, thermal and RGBT")
    p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", default="evaluation.json")
    a = p.parse_args(); cfg = read_config(a.config); device = select_device(a.device)
    model, checkpoint = load_detector(a.checkpoint, device)
    if checkpoint["class_names"] != cfg["data"]["class_names"]:
        raise ValueError("Checkpoint and data class mapping disagree")
    for key, value in checkpoint["preprocessing"].items():
        if key == "image_size": continue  # explicit alternate evaluation scale allowed
        if cfg["data"].get(key, value) != value: raise ValueError(f"Preprocessing mismatch: {key}")
    ds = build_dataset(cfg, "val", False)
    loader = make_loader(ds, cfg["training"].get("val_batch_size", 4), a.workers, False, device)
    results = evaluate(model, loader, device, cfg.get("postprocess", {}))
    text = json.dumps(results, indent=2, ensure_ascii=False)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(text, encoding="utf-8")
    print(text)
    print("AP assumes fully labeled non-crowd boxes; not a substitute for the official benchmark evaluator.")


if __name__ == "__main__": main()
