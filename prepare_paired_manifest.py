#!/usr/bin/env python3
"""Build JSONL manifests for aligned RGB/thermal detection datasets.

The training code expects one JSON object per aligned pair:
{"id": "...", "rgb": "...", "thermal": "...", "label": "..."}.

Examples:
  M3FD/TarDAL layout:
    python3 prepare_paired_manifest.py --root data/m3fd --rgb-dir vi --thermal-dir ir \
      --label-dir labels --split train:meta/train.txt --split val:meta/val.txt \
      --output data/m3fd_uod

  LLVIP split-directory layout:
    python3 prepare_paired_manifest.py --root data/LLVIP --rgb-dir visible \
      --thermal-dir infrared --label-dir Annotations --label-format voc \
      --class-names person --split train --split test \
      --output data/llvip_uod
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_split(value: str) -> tuple[str, str | None]:
    if ":" not in value:
        return value, None
    name, file_name = value.split(":", 1)
    if not name or not file_name:
        raise argparse.ArgumentTypeError("--split must be NAME or NAME:FILE")
    return name, file_name


def read_split_ids(root: Path, split_name: str, split_file: str | None,
                   rgb_base: Path, exts: tuple[str, ...]) -> list[str]:
    if split_file:
        path = root / split_file
        if not path.is_file():
            raise FileNotFoundError(path)
        ids = []
        for line in path.read_text(encoding="utf-8").splitlines():
            token = line.strip().split()
            if token:
                ids.append(Path(token[0]).stem)
        return ids
    split_dir = rgb_base / split_name
    if not split_dir.is_dir() and rgb_base.is_dir():
        split_dir = rgb_base
    if not split_dir.is_dir():
        raise FileNotFoundError(f"No split file was provided and split directory is missing: {split_dir}")
    stems = []
    for path in sorted(split_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            stems.append(path.stem)
    return stems


def find_file(base: Path, split_name: str, stem: str, exts: tuple[str, ...]) -> Path:
    candidates = []
    for parent in (base / split_name, base):
        for ext in exts:
            candidates += [parent / f"{stem}{ext}", parent / f"{stem}{ext.upper()}"]
    matches = [p for p in candidates if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"Missing file for id={stem} under {base}")
    if len({p.resolve() for p in matches}) > 1:
        raise ValueError(f"Ambiguous files for id={stem}: {matches}")
    return matches[0].resolve()


def voc_to_yolo(xml_path: Path, label_path: Path, class_to_id: dict[str, int]) -> int:
    tree = ET.parse(xml_path)
    annotation = tree.getroot()
    size = annotation.find("size")
    if size is None:
        raise ValueError(f"Missing size in {xml_path}")
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")
    rows = []
    for obj in annotation.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in class_to_id:
            raise ValueError(f"Unknown class {name!r} in {xml_path}; class_names={list(class_to_id)}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {xml_path}")
        x1 = max(0., min(width, float(box.findtext("xmin", "0"))))
        y1 = max(0., min(height, float(box.findtext("ymin", "0"))))
        x2 = max(0., min(width, float(box.findtext("xmax", "0"))))
        y2 = max(0., min(height, float(box.findtext("ymax", "0"))))
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        rows.append(f"{class_to_id[name]} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def build_records(root: Path, rgb_base: Path, thermal_base: Path, label_base: Path,
                  split_name: str, split_file: str | None, image_exts: tuple[str, ...],
                  label_ext: str, label_format: str, output: Path,
                  class_to_id: dict[str, int]) -> tuple[list[dict[str, str]], int]:
    ids = read_split_ids(root, split_name, split_file, rgb_base, image_exts)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids in split {split_name}")
    records = []
    objects = 0
    for stem in ids:
        rgb = find_file(rgb_base, split_name, stem, image_exts)
        thermal = find_file(thermal_base, split_name, stem, image_exts)
        source_label = find_file(label_base, split_name, stem, (label_ext,))
        if label_format == "voc":
            label = (output / "labels" / split_name / f"{stem}.txt").resolve()
            objects += voc_to_yolo(source_label, label, class_to_id)
        else:
            label = source_label
        records.append({"id": stem, "rgb": str(rgb), "thermal": str(thermal), "label": str(label)})
    if not records:
        raise ValueError(f"No records found for split {split_name}")
    return records, objects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="dataset root")
    parser.add_argument("--rgb-dir", required=True, help="RGB/visible image directory relative to root")
    parser.add_argument("--thermal-dir", required=True, help="thermal/infrared image directory relative to root")
    parser.add_argument("--label-dir", required=True, help="label directory relative to root")
    parser.add_argument("--split", action="append", type=parse_split, required=True,
                        help="NAME or NAME:FILE. Without FILE, ids are read from rgb-dir/NAME")
    parser.add_argument("--output", required=True, help="directory to write NAME.jsonl manifests")
    parser.add_argument("--image-exts", default=".jpg,.jpeg,.png,.bmp",
                        help="comma-separated image extensions")
    parser.add_argument("--label-ext", default=".txt")
    parser.add_argument("--label-format", choices=["yolo", "voc"], default="yolo")
    parser.add_argument("--class-names", default="",
                        help="comma-separated class names for VOC XML conversion")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    rgb_base = (root / args.rgb_dir).resolve()
    thermal_base = (root / args.thermal_dir).resolve()
    label_base = (root / args.label_dir).resolve()
    output = Path(args.output).expanduser().resolve()
    image_exts = tuple(ext if ext.startswith(".") else f".{ext}"
                       for ext in args.image_exts.lower().split(",") if ext)
    if args.label_format == "voc" and args.label_ext == ".txt":
        args.label_ext = ".xml"
    label_ext = args.label_ext if args.label_ext.startswith(".") else f".{args.label_ext}"
    class_names = [n.strip() for n in args.class_names.split(",") if n.strip()]
    if args.label_format == "voc" and not class_names:
        raise ValueError("--class-names is required for --label-format voc")
    class_to_id = {name: i for i, name in enumerate(class_names)}
    output.mkdir(parents=True, exist_ok=True)

    report = {}
    all_ids = {}
    for split_name, split_file in args.split:
        records, objects = build_records(root, rgb_base, thermal_base, label_base, split_name,
                                         split_file, image_exts, label_ext, args.label_format,
                                         output, class_to_id)
        path = output / f"{split_name}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                        encoding="utf-8")
        report[split_name] = {"records": len(records), "manifest": str(path)}
        if args.label_format == "voc":
            report[split_name]["objects"] = objects
            report[split_name]["converted_label_dir"] = str((output / "labels" / split_name).resolve())
        all_ids[split_name] = {r["id"] for r in records}

    overlaps = {}
    names = list(all_ids)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = sorted(all_ids[left] & all_ids[right])
            if shared:
                overlaps[f"{left}/{right}"] = shared[:10]
    if overlaps:
        report["warning"] = {"split_id_overlap_examples": overlaps}
    (output / "manifest_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
