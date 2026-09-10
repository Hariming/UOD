#!/usr/bin/env python3
"""Run one real-data optimizer step to validate CUDA, AMP, and model wiring."""
from __future__ import annotations

import argparse

import torch

from uod.engine import build_dataset, build_spec, make_loader, move_batch, read_config, seed_all
from uod.losses import UODLoss
from uod.model import UOD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/llvip_yolov5s.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size < 1 or args.workers < 0 or args.max_attempts < 1:
        raise ValueError("batch-size/max-attempts must be positive and workers must be nonnegative")
    cfg = read_config(args.config)
    seed_all(int(cfg.get("seed", 2026)))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("gpu_smoke_test.py requires a CUDA device")

    dataset = build_dataset(cfg, "train", augment=True)
    loader = make_loader(dataset, args.batch_size, args.workers, True, device)
    batch = move_batch(next(iter(loader)), device)
    spec = build_spec(cfg)
    model = UOD(spec, pretrained=False).to(device).train()
    criterion = UODLoss(spec.num_classes, **cfg.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda")

    torch.cuda.reset_peak_memory_stats(device)
    succeeded = False
    for attempt in range(1, args.max_attempts + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model.forward_pair(batch["rgb"], batch["thermal"])
            loss, stats = criterion(
                outputs, batch["targets"], batch["rgb"].shape[-2:], batch["valid_mask"])
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite smoke loss: {float(loss)}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=False)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        if not skipped:
            succeeded = True
            break
        print(f"AMP overflow attempt={attempt}: scale {old_scale:g} -> {scaler.get_scale():g}")
    if not succeeded:
        raise RuntimeError(f"AMP optimizer step did not stabilize after {args.max_attempts} attempts")
    torch.cuda.synchronize(device)

    props = torch.cuda.get_device_properties(device)
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"PASS device={device} name={props.name} visible_devices={torch.cuda.device_count()}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} batch_pairs={args.batch_size}")
    print(f"loss={float(stats['loss']):.6f} det={float(stats['det_sum']):.6f} "
          f"fs={float(stats['fs']):.6f} grad_norm={float(grad_norm):.6f} "
          f"amp_scale={scaler.get_scale():g} peak_allocated_gib={peak_gib:.3f}")


if __name__ == "__main__":
    main()
