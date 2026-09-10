from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from .data import PairedDetectionDataset, collate_pairs
from .geometry import candidates, suppress, fuse_candidates, undo_letterbox
from .metrics import DetectionAP
from .model import ModelSpec, UOD, UODDetector, model_spec_dict


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def select_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a matching CUDA torch/torchvision pair or use --device cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"GPU {device.index} does not exist")
    elif device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable on this machine")
    elif device.type not in {"cuda", "cpu", "mps"}:
        raise ValueError("Supported devices: cpu, cuda:N, mps")
    return device


def read_config(path: str | Path) -> dict:
    p = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for field in ("train_manifest", "val_manifest"):
        if field in cfg["data"]:
            value = Path(cfg["data"][field]).expanduser()
            cfg["data"][field] = str((p.parent / value).resolve() if not value.is_absolute() else value)
    out = Path(cfg["training"]["out_dir"]).expanduser()
    cfg["training"]["out_dir"] = str((p.parent / out).resolve() if not out.is_absolute() else out)
    names = cfg["data"]["class_names"]
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("data.class_names must be a nonempty unique list in class-id order")
    if not all(isinstance(n, str) for n in names):
        raise ValueError("Class names must be strings")
    return cfg


def build_spec(cfg: dict) -> ModelSpec:
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "pretrained"}
    return ModelSpec(num_classes=len(cfg["data"]["class_names"]), **model_cfg)


def build_dataset(cfg: dict, split: str, augment: bool = False):
    d = cfg["data"]
    a = cfg.get("augmentation", {}) if augment else {}
    return PairedDetectionDataset(d[f"{split}_manifest"], len(d["class_names"]),
        d.get("image_size", 640), d.get("thermal_divisor", 255.), augment=augment,
        flip_probability=a.get("flip_probability", .5), rgb_brightness=a.get("rgb_brightness", 0.),
        fill=d.get("letterbox_fill", 114 / 255))


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool,
                device: torch.device, generator: torch.Generator | None = None):
    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid batch_size or num_workers")
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
        drop_last=False, collate_fn=collate_pairs, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, generator=generator, persistent_workers=False)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {**batch,
        "rgb": batch["rgb"].to(device, non_blocking=True),
        "thermal": batch["thermal"].to(device, non_blocking=True),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
        "targets": [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in batch["targets"]]}


def freeze_backbone_bn(model):
    for module in model.bfe.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()  # preserve running statistics; affine parameters may still learn


def parameter_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (no_decay if param.ndim < 2 or name.endswith("bias") else decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.}]


def cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def atomic_save(payload: dict, path: str | Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path):
    # Only primitive metadata + tensors are stored. Never unpickle a full model.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("kind") not in {"uod_training", "uod_deploy"}:
        raise ValueError("Not a supported UOD checkpoint")
    return checkpoint


def load_detector(path: str | Path, device: torch.device):
    checkpoint = load_checkpoint(path)
    model = UODDetector(ModelSpec(**checkpoint["model_spec"]), pretrained=False)
    state = checkpoint["state_dict"]
    if checkpoint["kind"] == "uod_training":
        state = {k: v for k, v in state.items() if not k.startswith(("nfe_rgb.", "nfe_thermal."))}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint


def export_checkpoint(source: str | Path, destination: str | Path):
    ckpt = load_checkpoint(source)
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("Do not overwrite the resumable training checkpoint")
    state = {k: v for k, v in ckpt["state_dict"].items()
             if not k.startswith(("nfe_rgb.", "nfe_thermal."))}
    deploy = UODDetector(ModelSpec(**ckpt["model_spec"]), pretrained=False)
    deploy.load_state_dict(state, strict=True)
    atomic_save({"format_version": 1, "kind": "uod_deploy", "model_spec": ckpt["model_spec"],
        "class_names": ckpt["class_names"], "preprocessing": ckpt["preprocessing"],
        "state_dict": state, "source_epoch": ckpt.get("epoch", ckpt.get("source_epoch", -1))}, destination)
    return sum(v.numel() for v in state.values())


@torch.no_grad()
def evaluate(model: UODDetector, loader, device: torch.device, postprocess: dict | None = None,
             modes: tuple[str, ...] = ("rgb", "thermal", "rgbt")) -> dict:
    pp = {"score_threshold": .001, "pre_nms_topk": 3000, "iou_threshold": .5, "max_detections": 100}
    pp.update(postprocess or {})
    if any(m not in {"rgb", "thermal", "rgbt"} for m in modes):
        raise ValueError("Invalid evaluation mode")
    model.eval()
    metrics = {m: DetectionAP(model.spec.num_classes) for m in modes}
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        # Uses only BFE/IFE/IFD even when model is a training UOD instance.
        out_r, out_t = model.paired_predictions(batch["rgb"], batch["thermal"])
        size = batch["rgb"].shape[-2:]
        cr = candidates(out_r, size, batch["valid_mask"], pp["score_threshold"], pp["pre_nms_topk"])
        ct = candidates(out_t, size, batch["valid_mask"], pp["score_threshold"], pp["pre_nms_topk"])
        for mode in modes:
            preds = []
            for r, t, meta in zip(cr, ct, batch["meta"], strict=True):
                if mode == "rgbt":
                    pred = fuse_candidates(r, t, pp["iou_threshold"], pp["max_detections"])
                else:
                    pred = suppress(r if mode == "rgb" else t, pp["iou_threshold"], pp["max_detections"])
                pred["boxes"] = undo_letterbox(pred["boxes"], meta)
                preds.append(pred)
            metrics[mode].update(preds, raw_batch["original_targets"])
    return {mode: metric.compute() for mode, metric in metrics.items()}
