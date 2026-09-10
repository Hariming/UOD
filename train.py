#!/usr/bin/env python3
"""Single-device paired UOD training with resumable checkpoints.
Run: python train.py --config configs/uod_resnet50.yaml --device cuda:0
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil
import time
import torch
from uod.data import assert_disjoint
from uod.engine import (atomic_save, build_dataset, build_spec, cpu_state_dict, evaluate,
    freeze_backbone_bn, load_checkpoint, make_loader, move_batch, parameter_groups,
    read_config, seed_all, select_device)
from uod.losses import UODLoss
from uod.model import UOD, model_spec_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--resume", help="A full last.pt/best.pt, NOT deploy.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.device: cfg["device"] = args.device
    if args.epochs is not None: cfg["training"]["epochs"] = args.epochs
    if args.workers is not None: cfg["num_workers"] = args.workers
    seed_all(int(cfg.get("seed", 2026)))
    device = select_device(cfg.get("device", "cuda:0"))
    tr = cfg["training"]
    epochs, accum = int(tr["epochs"]), int(tr.get("accum_steps", 1))
    if epochs < 1 or accum < 1 or tr.get("eval_every", 1) < 1 or tr.get("save_every", 5) < 1:
        raise ValueError("Epoch/accum/evaluation/save intervals must be positive")
    out_dir = Path(tr["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out_dir / "last.pt").exists():
        raise FileExistsError(f"Existing run: {out_dir}. Use --resume or a new out_dir")
    train_ds, val_ds = build_dataset(cfg, "train", True), build_dataset(cfg, "val", False)
    assert_disjoint(train_ds, val_ds)
    loader_generator = torch.Generator().manual_seed(int(cfg.get("seed", 2026)))
    val_generator = torch.Generator().manual_seed(int(cfg.get("seed", 2026)) + 1)
    workers = int(cfg.get("num_workers", 4))
    train_loader = make_loader(train_ds, tr.get("batch_size", 4), workers, True, device, loader_generator)
    val_loader = make_loader(val_ds, tr.get("val_batch_size", 4), workers, False, device, val_generator)
    spec = build_spec(cfg)
    model = UOD(spec, pretrained=bool(cfg["model"].get("pretrained", False)) and not args.resume).to(device)
    criterion = UODLoss(spec.num_classes, **cfg.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, tr.get("weight_decay", 1e-4)),
                                  lr=float(tr.get("lr", 2e-4)))
    updates_per_epoch = math.ceil(len(train_loader) / accum)
    total_updates = epochs * updates_per_epoch
    warmup = int(float(tr.get("warmup_epochs", 3)) * updates_per_epoch)
    warmup = min(warmup, max(0, total_updates - 1))
    final_ratio = float(tr.get("final_lr_ratio", .1))
    if not 0 < final_ratio <= 1: raise ValueError("final_lr_ratio must be in (0,1]")
    def lr_multiplier(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1., (step - warmup) / max(1, total_updates - warmup))
        return final_ratio + (1 - final_ratio) * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    amp = bool(tr.get("amp", True)) and device.type == "cuda"
    amp_name = tr.get("amp_dtype", "float16")
    if amp_name not in {"float16", "bfloat16"}: raise ValueError("Invalid amp_dtype")
    amp_dtype = getattr(torch, amp_name)
    if amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 AMP unsupported; use float16 or amp:false")
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    start_epoch, best_score, global_update = 0, -1., 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint["kind"] != "uod_training":
            raise ValueError("Deployment checkpoint cannot resume training: NFE/optimizer were removed")
        if checkpoint["model_spec"] != model_spec_dict(model) or checkpoint["class_names"] != cfg["data"]["class_names"]:
            raise ValueError("Resume model architecture/class mapping differs")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_score = checkpoint["epoch"] + 1, float(checkpoint["best_score"])
        global_update = checkpoint["global_update"]
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        loader_generator.set_state(checkpoint["loader_rng"])
        val_generator.set_state(checkpoint["val_loader_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            saved = checkpoint["cuda_rng"]
            if len(saved) == torch.cuda.device_count(): torch.cuda.set_rng_state_all(saved)
        if checkpoint["config"]["training"]["epochs"] != epochs:
            print("NOTE: epochs changed on resume; the remaining cosine LR schedule changes.")
    if start_epoch >= epochs:
        raise ValueError(f"Already completed {start_epoch} epochs; epochs={epochs}")
    (out_dir / "config.resolved.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    total_params = sum(p.numel() for p in model.parameters())
    deploy_params = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("nfe_"))
    print(f"Pairs train/val={len(train_ds)}/{len(val_ds)}, device={device}, pair_batch={tr.get('batch_size', 4)}")
    print(f"Parameters train={total_params:,}, deploy={deploy_params:,}; AMP={amp}")
    print(f"This is a {spec.backbone}/{spec.neck} UOD reference baseline, NOT official UOD-v5/v8.")
    print("Class mapping:", dict(enumerate(cfg["data"]["class_names"])))
    log_path = out_dir / "history.csv"
    for epoch in range(start_epoch, epochs):
        begin = time.monotonic()
        model.train()
        if tr.get("freeze_backbone_bn", False): freeze_backbone_bn(model)
        optimizer.zero_grad(set_to_none=True)
        running, seen = {}, 0
        for step, raw_batch in enumerate(train_loader):
            batch = move_batch(raw_batch, device)
            # Correct divisor for the LAST partial accumulation group as well.
            group_start = (step // accum) * accum
            group_size = min(accum, len(train_loader) - group_start)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                outputs = model.forward_pair(batch["rgb"], batch["thermal"])
                loss, stats = criterion(outputs, batch["targets"], batch["rgb"].shape[-2:], batch["valid_mask"])
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite loss at epoch={epoch+1}, step={step}; ids={[m['id'] for m in batch['meta']]}")
            scaler.scale(loss / group_size).backward()
            boundary = (step + 1) % accum == 0 or step + 1 == len(train_loader)
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr.get("grad_clip", 5.),
                                                error_if_nonfinite=not scaler.is_enabled())
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.is_enabled() and scaler.get_scale() < old_scale
                if not skipped:
                    scheduler.step()
                    global_update += 1
                else:
                    print("AMP overflow: update skipped and scale reduced; LR was not advanced.")
                optimizer.zero_grad(set_to_none=True)
            b = batch["rgb"].shape[0]
            for key, value in stats.items(): running[key] = running.get(key, 0.) + float(value) * b
            seen += b
            log_every = int(tr.get("log_every", 20))
            if log_every > 0 and ((step + 1) % log_every == 0 or step + 1 == len(train_loader)):
                print(f"epoch {epoch+1}/{epochs} step {step+1}/{len(train_loader)} "
                      f"loss={float(stats['loss']):.4f} det={float(stats['det_sum']):.4f} "
                      f"fs={float(stats['fs']):.4f} dpos2={float(stats['positive_d2']):.4f}", flush=True)
        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
               **{k: v / seen for k, v in running.items()}}
        improved = False
        if (epoch + 1) % int(tr.get("eval_every", 1)) == 0 or epoch + 1 == epochs:
            results = evaluate(model, val_loader, device, cfg.get("postprocess", {}))
            (out_dir / f"val_epoch_{epoch+1:03d}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            maps = []
            for mode, values in results.items():
                for key in ("map50", "map50_95"): row[f"val/{mode}/{key}"] = values[key]
                if mode in ("rgb", "thermal") and values["map50"] is not None: maps.append(values["map50"])
            if maps:
                # Engineering choice: don't select only for the easier modality.
                score = sum(maps) / len(maps)
                improved = score > best_score
                if improved: best_score = score
            print("validation:", {m: {k: v[k] for k in ('map50','map50_95')} for m,v in results.items()}, flush=True)
        for mode in ("rgb", "thermal", "rgbt"):
            for key in ("map50", "map50_95"): row.setdefault(f"val/{mode}/{key}", "")
        row["seconds"] = time.monotonic() - begin
        fields = list(row)
        if log_path.exists():
            with log_path.open(newline="", encoding="utf-8") as f:
                existing = next(csv.reader(f))
            if set(existing) != set(fields): raise ValueError("history.csv schema mismatch; use a new output directory")
            fields = existing
        new_log = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new_log: writer.writeheader()
            writer.writerow(row)
        prep = {k: cfg["data"].get(k, default) for k, default in
                (("image_size", 640), ("thermal_divisor", 255.), ("letterbox_fill", 114 / 255))}
        payload = {"format_version": 1, "kind": "uod_training", "model_spec": model_spec_dict(model),
            "class_names": cfg["data"]["class_names"], "preprocessing": prep,
            "state_dict": cpu_state_dict(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_score": best_score, "global_update": global_update,
            "config": cfg, "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "loader_rng": loader_generator.get_state(), "val_loader_rng": val_generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        atomic_save(payload, out_dir / "last.pt")
        if improved: shutil.copy2(out_dir / "last.pt", out_dir / "best.pt")
        if (epoch + 1) % int(tr.get("save_every", 5)) == 0:
            shutil.copy2(out_dir / "last.pt", out_dir / f"epoch_{epoch+1:03d}.pt")
        print(f"epoch {epoch+1} saved; elapsed={row['seconds']:.1f}s; best mean single-modal AP50={best_score:.4f}")
    print(f"Finished. Training checkpoint: {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
