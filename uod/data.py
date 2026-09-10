"""Strict aligned-pair loader: JSONL manifests + shared YOLO-format labels.

A pair must depict the SAME scene/object at corresponding pixels. Equal image
sizes alone do NOT establish registration. Registration must be done upstream.
"""
from __future__ import annotations
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


def read_rgb(path: str | Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1) / 255.


def read_thermal(path: str | Path, divisor: float = 255.) -> Tensor:
    if divisor <= 0:
        raise ValueError("thermal_divisor must be positive")
    with Image.open(path) as image:
        # Do not silently convert a uint16 sensor image to 8-bit L mode.
        array = np.asarray(image).copy()
    if array.ndim == 3 and array.shape[-1] == 3:
        if not (np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2])):
            raise ValueError(f"False-color thermal is not a single-channel image: {path}")
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a grayscale thermal image: {path}, shape={array.shape}")
    array = array.astype(np.float32)
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > divisor:
        raise ValueError(f"Thermal values outside [0,{divisor}]: {path}. "
                         "Set thermal_divisor for the sensor (e.g. 65535 for full-range uint16).")
    return torch.from_numpy(array).unsqueeze(0) / divisor


def letterbox_tensor(image: Tensor, size: int, fill: float = 114 / 255):
    if image.ndim != 3 or size < 64 or size % 32:
        raise ValueError("Require CHW image and size >=64 divisible by32")
    _, h, w = image.shape
    ratio = min(size / w, size / h)
    new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    px, py = (size - new_w) // 2, (size - new_h) // 2
    resized = F.interpolate(image[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
    output = image.new_full((image.shape[0], size, size), fill)
    output[:, py:py + new_h, px:px + new_w] = resized
    mask = torch.zeros(1, size, size, dtype=torch.bool)
    mask[:, py:py + new_h, px:px + new_w] = True
    meta = {"original_hw": [h, w], "scale_xy": [new_w / w, new_h / h],
            "pad_xy": [px, py], "resized_hw": [new_h, new_w]}
    return output, mask, meta


def transform_boxes(boxes: Tensor, meta: dict) -> Tensor:
    result = boxes.clone()
    sx, sy = meta["scale_xy"]
    px, py = meta["pad_xy"]
    result[:, 0::2] = result[:, 0::2] * sx + px
    result[:, 1::2] = result[:, 1::2] * sy + py
    return result


def read_labels(path: Path, original_hw: tuple[int, int], num_classes: int):
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file (create an empty file for a negative image): {path}")
    boxes, labels = [], []
    h, w = original_hw
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected class cx cy width height at {path}:{line_no}")
        values = list(map(float, parts))
        cls, cx, cy, bw, bh = values
        if not np.isfinite(values).all() or cls != int(cls) or not 0 <= cls < num_classes:
            raise ValueError(f"Invalid class/value at {path}:{line_no}")
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        if bw <= 0 or bh <= 0 or min(x1, y1) < -1e-4 or max(x2, y2) > 1.0001:
            raise ValueError(f"Invalid normalized YOLO box at {path}:{line_no}")
        boxes.append([max(0., x1) * w, max(0., y1) * h, min(1., x2) * w, min(1., y2) * h])
        labels.append(int(cls))
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(labels, dtype=torch.long)


class PairedDetectionDataset(Dataset):
    def __init__(self, manifest: str | Path, num_classes: int, image_size: int = 640,
                 thermal_divisor: float = 255., augment: bool = False, flip_probability: float = .5,
                 rgb_brightness: float = 0., fill: float = 114 / 255):
        self.manifest = Path(manifest).expanduser().resolve()
        self.num_classes, self.image_size, self.thermal_divisor = num_classes, image_size, thermal_divisor
        self.augment, self.flip_probability, self.rgb_brightness = augment, flip_probability, rgb_brightness
        self.fill = fill
        if not 0 <= flip_probability <= 1 or not 0 <= rgb_brightness <= 1:
            raise ValueError("Invalid augmentation probability/magnitude")
        self.samples = []
        seen_ids, seen_rgb, seen_thermal = set(), set(), set()
        for line_no, text in enumerate(self.manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not text.strip():
                continue
            sample = json.loads(text)
            for key in ("rgb", "thermal", "label"):
                if key not in sample:
                    raise ValueError(f"Missing {key} at {self.manifest}:{line_no}")
                p = Path(sample[key]).expanduser()
                p = (self.manifest.parent / p).resolve() if not p.is_absolute() else p.resolve()
                if not p.is_file():
                    raise FileNotFoundError(p)
                sample[key] = str(p)
            sample["id"] = str(sample.get("id", Path(sample["rgb"]).stem))
            if sample["id"] in seen_ids or sample["rgb"] in seen_rgb or sample["thermal"] in seen_thermal:
                raise ValueError(f"Duplicate sample/image in manifest: {sample['id']}")
            seen_ids.add(sample["id"]); seen_rgb.add(sample["rgb"]); seen_thermal.add(sample["thermal"])
            self.samples.append(sample)
        if not self.samples:
            raise ValueError(f"Empty manifest: {self.manifest}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        rgb, thermal = read_rgb(sample["rgb"]), read_thermal(sample["thermal"], self.thermal_divisor)
        if rgb.shape[-2:] != thermal.shape[-2:]:
            raise ValueError(f"Unaligned pair size for {sample['id']}: RGB {rgb.shape}, thermal {thermal.shape}. "
                             "Register the images first; independent resize is NOT registration.")
        original_boxes, classes = read_labels(Path(sample["label"]), rgb.shape[-2:], self.num_classes)
        rgb, mask, meta = letterbox_tensor(rgb, self.image_size, self.fill)
        thermal, thermal_mask, thermal_meta = letterbox_tensor(thermal, self.image_size, self.fill)
        if meta != thermal_meta or not torch.equal(mask, thermal_mask):
            raise AssertionError("Paired letterbox transform differs")
        boxes = transform_boxes(original_boxes, meta)
        if self.augment and random.random() < self.flip_probability:
            # One random decision, applied to BOTH images, the mask, AND labels.
            rgb, thermal, mask = rgb.flip(-1), thermal.flip(-1), mask.flip(-1)
            old_x1 = boxes[:, 0].clone()
            boxes[:, 0] = self.image_size - boxes[:, 2]
            boxes[:, 2] = self.image_size - old_x1
        if self.augment and self.rgb_brightness:
            factor = random.uniform(1 - self.rgb_brightness, 1 + self.rgb_brightness)
            rgb = (rgb * factor).clamp(0, 1)  # appearance only; geometry stays aligned
        meta.update(id=sample["id"], rgb_path=sample["rgb"], thermal_path=sample["thermal"])
        return {"rgb": rgb, "thermal": thermal, "valid_mask": mask,
                "target": {"boxes": boxes, "labels": classes},
                "original_target": {"boxes": original_boxes, "labels": classes.clone()}, "meta": meta}


def collate_pairs(samples: list[dict]) -> dict:
    return {"rgb": torch.stack([s["rgb"] for s in samples]),
            "thermal": torch.stack([s["thermal"] for s in samples]),
            "valid_mask": torch.stack([s["valid_mask"] for s in samples]),
            "targets": [s["target"] for s in samples],
            "original_targets": [s["original_target"] for s in samples],
            "meta": [s["meta"] for s in samples]}


def assert_disjoint(train: PairedDetectionDataset, val: PairedDetectionDataset) -> None:
    for key in ("rgb", "thermal"):
        overlap = {s[key] for s in train.samples} & {s[key] for s in val.samples}
        if overlap:
            raise ValueError(f"Train/val leakage ({key}): {next(iter(overlap))}")
    # Optional scene-level guard: a scene_id must be present for EVERY sample
    # when used, otherwise a partial scene check could give false confidence.
    samples = train.samples + val.samples
    if any("scene_id" in s for s in samples):
        if not all("scene_id" in s for s in samples):
            raise ValueError("scene_id must be supplied for every sample or for none")
        if {s["scene_id"] for s in train.samples} & {s["scene_id"] for s in val.samples}:
            raise ValueError("Train/val scene_id overlap")
