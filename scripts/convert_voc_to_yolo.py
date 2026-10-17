#!/usr/bin/env python
"""Convert RDD2022 Pascal VOC annotations to YOLO format.

Expected input layout (as in the RDD2022 download):

    <src>/<Country>/train/images/<name>.jpg
    <src>/<Country>/train/annotations/xmls/<name>.xml

Output:

    <dst>/all/labels/<name>.txt   YOLO labels: "class cx cy w h" (normalised)
    <dst>/manifest.csv            one row per image, used by make_split.py

Images are NOT copied here. make_split.py copies them into the final
train/val/test folders, so the 47k images are only copied once.

Only the 4 standard classes are kept (D00, D10, D20, D40). Other labels that
appear in some countries are skipped and reported at the end.

Usage:
    python scripts/convert_voc_to_yolo.py --src data/RDD2022 --dst data/yolo
"""
from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from utils import CLASS_NAMES

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="data/RDD2022", help="RDD2022 root folder")
    p.add_argument("--dst", default="data/yolo", help="output folder")
    p.add_argument("--keep-empty", action=argparse.BooleanOptionalAction, default=True,
                   help="keep images with no kept boxes as background images (default: keep)")
    p.add_argument("--trust-xml-size", action="store_true",
                   help="use width/height from the XML instead of reading the image header "
                        "(faster, but wrong if the XML size does not match the image)")
    return p.parse_args()


def find_image(xml_path: Path) -> Path | None:
    stem = xml_path.stem
    img_dir = xml_path.parents[2] / "images" if len(xml_path.parents) > 2 else xml_path.parent
    for folder in (img_dir, xml_path.parent):
        for ext in IMAGE_EXTS:
            cand = folder / f"{stem}{ext}"
            if cand.exists():
                return cand
    return None


def get_country(xml_path: Path, src: Path) -> str:
    parts = xml_path.relative_to(src).parts
    if "train" in parts:
        i = parts.index("train")
        return parts[i - 1] if i > 0 else src.name
    return xml_path.parent.name


def parse_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    w = h = 0
    size = root.find("size")
    if size is not None:
        try:
            w = int(float(size.findtext("width") or 0))
            h = int(float(size.findtext("height") or 0))
        except ValueError:
            w = h = 0
    objects = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            box = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            continue
        objects.append((name, *box))
    return w, h, objects


def main():
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        sys.exit(f"Source folder not found: {src}\nDownload RDD2022 and place it there first.")

    xmls = sorted(p for p in src.rglob("*.xml") if "annotations" in p.relative_to(src).parts)
    if not xmls:
        sys.exit(f"No annotation XMLs found under {src}.\n"
                 f"Expected files like {src}/Japan/train/annotations/xmls/Japan_000001.xml")

    label_dir = dst / "all" / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    cls_map = {name: idx for idx, name in CLASS_NAMES.items()}
    class_cols = [f"n_{CLASS_NAMES[i]}" for i in sorted(CLASS_NAMES)]

    rows, seen = [], set()
    skipped_classes = Counter()
    missing_img = bad_xml = dup = dropped_empty = degenerate = 0
    per_country = defaultdict(lambda: Counter())

    for xml_path in tqdm(xmls, desc="Converting"):
        country = get_country(xml_path, src)
        img_path = find_image(xml_path)
        if img_path is None:
            missing_img += 1
            continue
        try:
            xml_w, xml_h, objects = parse_xml(xml_path)
        except ET.ParseError:
            bad_xml += 1
            continue

        if args.trust_xml_size and xml_w > 0 and xml_h > 0:
            w, h = xml_w, xml_h
        else:
            with Image.open(img_path) as im:
                w, h = im.size

        stem = img_path.stem
        if stem in seen:
            stem = f"{country}_{stem}"
            if stem in seen:
                dup += 1
                continue
        seen.add(stem)

        lines, counts = [], Counter()
        for name, xmin, ymin, xmax, ymax in objects:
            if name not in cls_map:
                skipped_classes[name] += 1
                continue
            xmin, xmax = max(0.0, min(xmin, w)), max(0.0, min(xmax, w))
            ymin, ymax = max(0.0, min(ymin, h)), max(0.0, min(ymax, h))
            if xmax - xmin < 1 or ymax - ymin < 1:
                degenerate += 1
                continue
            cx, cy = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
            bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
            cid = cls_map[name]
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            counts[CLASS_NAMES[cid]] += 1

        if not lines and not args.keep_empty:
            dropped_empty += 1
            continue

        label_path = label_dir / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        row = {"stem": stem, "country": country, "image_path": str(img_path.resolve()),
               "label_path": str(label_path.resolve()), "width": w, "height": h,
               "n_boxes": len(lines)}
        for cname in CLASS_NAMES.values():
            row[f"n_{cname}"] = counts[cname]
        rows.append(row)

        pc = per_country[country]
        pc["images"] += 1
        pc["empty"] += int(not lines)
        for cname, n in counts.items():
            pc[cname] += n

    manifest = dst / "manifest.csv"
    fields = ["stem", "country", "image_path", "label_path", "width", "height", "n_boxes"] + class_cols
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # ---- summary ----
    names = list(CLASS_NAMES.values())
    print(f"\nConverted {len(rows)} images -> {label_dir}")
    print(f"Manifest: {manifest}\n")
    header = f"{'country':<16}{'images':>8}{'empty':>8}" + "".join(f"{n:>8}" for n in names)
    print(header)
    print("-" * len(header))
    total = Counter()
    for country in sorted(per_country):
        pc = per_country[country]
        total.update(pc)
        print(f"{country:<16}{pc['images']:>8}{pc['empty']:>8}" + "".join(f"{pc[n]:>8}" for n in names))
    print("-" * len(header))
    print(f"{'TOTAL':<16}{total['images']:>8}{total['empty']:>8}" + "".join(f"{total[n]:>8}" for n in names))

    if skipped_classes:
        print("\nSkipped labels (not in the 4 target classes):",
              ", ".join(f"{k or '<empty>'}={v}" for k, v in skipped_classes.most_common()))
    problems = {"missing image": missing_img, "unparsable XML": bad_xml, "duplicate name": dup,
                "degenerate box": degenerate, "dropped empty image": dropped_empty}
    problems = {k: v for k, v in problems.items() if v}
    if problems:
        print("Notes:", ", ".join(f"{k}={v}" for k, v in problems.items()))
    print("\nNext: python scripts/make_split.py --root", dst)


if __name__ == "__main__":
    main()
