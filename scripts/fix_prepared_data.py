#!/usr/bin/env python
"""Fix an already-prepared data/{train,val,test}/{images,labels} copy of RDD2022.

Use this instead of convert_voc_to_yolo.py + make_split.py when the dataset was
downloaded pre-converted (e.g. from Kaggle). Two problems are fixed:

1. Class ids. The prepared labels use 5 ids: 0 D00, 1 D10, 2 D20, 3 "other
   corruption" (D43/D44/D50/Repair merged) and 4 D40 pothole. The per-class box
   totals (26016 / 11830 / 10617 / 6544 for ids 0/1/2/4) match the published RDD2022
   counts for D00 / D10 / D20 / D40. The project uses the 4 standard classes, so id 3
   boxes are dropped and id 4 becomes 3. This is only done when id 4 is present.

2. The split. The prepared split is random per image, so near-identical neighbouring
   frames land in train and val/test. Images are regrouped into blocks of consecutive
   file numbers per country (same heuristic as make_split.py) and whole blocks are
   assigned to one split.

In place (default): files are moved between the split folders, so no extra disk.
All original labels and the original split are first saved to
<root>/original_labels.zip. Running twice is refused unless --force is given.

    python scripts/fix_prepared_data.py --root data

To a new folder (--dst): the source is left untouched (e.g. a read-only Kaggle input).
Labels are written to <dst> and images are symlinked (copied if symlinks fail). If the
source was already fixed by this script, its split and classes are mirrored as they are.

    python scripts/fix_prepared_data.py --root /kaggle/input/rdd2022 --dst /kaggle/working/data
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from utils import IMG_EXTS, PROJECT_ROOT

SPLITS = ("train", "val", "test")
CLASS_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}  # source id -> project id; anything else is dropped
MARKER = ".fixed_by_fix_prepared_data"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data", help="folder holding train/, val/, test/")
    p.add_argument("--dst", default=None, help="write the fixed dataset here instead of changing --root")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--block-size", type=int, default=50, help="consecutive images treated as one sequence")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-classes", action="store_true", help="only re-split, do not remap class ids")
    p.add_argument("--force", action="store_true", help="run even if the marker file says it already ran")
    return p.parse_args()


def country_and_index(stem: str):
    m = re.match(r"(.+?)_(\d+)$", stem)
    return (m.group(1), int(m.group(2))) if m else (stem, None)


def remap_label_text(text: str, dropped: Counter) -> str:
    out = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        cid = int(float(fields[0]))
        if cid not in CLASS_REMAP:
            dropped[cid] += 1
            continue
        out.append(" ".join([str(CLASS_REMAP[cid])] + fields[1:]))
    return "\n".join(out) + ("\n" if out else "")


def has_class_id(label_dirs, class_id: int) -> bool:
    prefix = f"{class_id} "
    for d in label_dirs:
        for f in d.glob("*.txt"):
            with open(f, encoding="utf-8") as fh:
                if any(line.startswith(prefix) for line in fh):
                    return True
    return False


def link_or_copy(src: Path, dst: Path, state: dict) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if not state["copy"]:
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            state["copy"] = True
            print("Symlinks are not available here; copying images instead.")
    shutil.copy2(src, dst)


def block_split(items, args) -> dict:
    """Map each (country, block) group to a split, stratified by country."""
    groups = defaultdict(list)
    for item in items:
        country, idx = country_and_index(item[0])
        groups[(country, idx // args.block_size if idx is not None else item[0])].append(item)
    rng = random.Random(args.seed)
    by_country = defaultdict(list)
    for key in groups:
        by_country[key[0]].append(key)
    target = {}
    for country, keys in sorted(by_country.items()):
        keys = sorted(keys, key=str)
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(1, round(n * args.test_frac)) if n >= 3 else 0
        n_val = max(1, round(n * args.val_frac)) if n >= 3 else 0
        for i, key in enumerate(keys):
            target[key] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return groups, target


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    dst = Path(args.dst) if args.dst else root
    if not dst.is_absolute():
        dst = PROJECT_ROOT / dst
    in_place = dst.resolve() == root.resolve()
    if (dst / MARKER).exists() and not args.force:
        sys.exit(f"{dst / MARKER} exists: this data was already fixed. Use --force to run again.")

    # ---- inventory ----
    items = []  # (stem, current_split, image_path)
    for s in SPLITS:
        img_dir = root / s / "images"
        if img_dir.is_dir():
            items += [(p.stem, s, p) for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not items:
        sys.exit(f"No images found under {root}/<split>/images")
    stems = [i[0] for i in items]
    if len(set(stems)) != len(stems):
        sys.exit("The same image name appears in more than one split; cannot re-split safely.")

    # splits.csv also counts: uploads to Kaggle can drop dotfiles such as the marker
    source_fixed = ((root / MARKER).exists() or (root / "splits.csv").exists()) and not in_place
    if source_fixed:
        print(f"{root} was already fixed by this script: mirroring its split and classes unchanged.")
        remap = False
    elif args.keep_classes:
        remap = False
    else:
        remap = has_class_id([root / s / "labels" for s in SPLITS if (root / s / "labels").is_dir()], 4)
        print("Class id 4 (D40 pothole) found: remapping to the 4 project classes." if remap else
              "No class id 4 found: labels already use the 4 project classes, keeping them.")

    # ---- backup (in place only; --dst never touches the source) ----
    if in_place:
        backup = root / "original_labels.zip"
        if not backup.exists():
            with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("original_split.csv", "stem,split\n" + "".join(f"{st},{sp}\n" for st, sp, _ in items))
                for stem, s, _ in tqdm(items, desc="Backing up labels"):
                    lbl = root / s / "labels" / f"{stem}.txt"
                    if lbl.exists():
                        zf.write(lbl, f"{s}/labels/{stem}.txt")
            print(f"Backed up original labels and split to {backup}")

    # ---- split ----
    if source_fixed:
        groups = {(stem, None): [(stem, s, img)] for stem, s, img in items}
        target = {key: members[0][1] for key, members in groups.items()}
    else:
        groups, target = block_split(items, args)

    for s in SPLITS:
        (dst / s / "images").mkdir(parents=True, exist_ok=True)
        (dst / s / "labels").mkdir(parents=True, exist_ok=True)

    dropped, moved, state = Counter(), 0, {"copy": False}
    class_counts = {s: Counter() for s in SPLITS}
    rows = []
    for key, members in tqdm(sorted(groups.items(), key=lambda kv: str(kv[0])), desc="Writing split"):
        dst_split = target[key]
        for stem, src_split, img in members:
            src_lbl = root / src_split / "labels" / f"{stem}.txt"
            text = src_lbl.read_text(encoding="utf-8") if src_lbl.exists() else ""
            if remap:
                text = remap_label_text(text, dropped)
            (dst / dst_split / "labels" / f"{stem}.txt").write_text(text, encoding="utf-8")
            if in_place:
                if src_split != dst_split:
                    if src_lbl.exists():
                        src_lbl.unlink()
                    os.replace(img, dst / dst_split / "images" / img.name)
                    moved += 1
            else:
                link_or_copy(img, dst / dst_split / "images" / img.name, state)
                moved += src_split != dst_split
            for line in text.splitlines():
                class_counts[dst_split][int(float(line.split()[0]))] += 1
            rows.append(f"{stem},{key[0]},{key[0]}_{key[1]},{dst_split}\n")

    (dst / "splits.csv").write_text("stem,country,group,split\n" + "".join(sorted(rows)), encoding="utf-8")
    (dst / MARKER).write_text(f"block_size={args.block_size} seed={args.seed} remap={remap}\n", encoding="utf-8")

    # ---- summary ----
    print(f"\n{moved} images changed split. Dropped boxes by source class: {dict(dropped)}")
    per_split = Counter(target[k] for k, m in groups.items() for _ in m)
    for s in SPLITS:
        print(f"{s:>5}: {per_split[s]:>6} images | boxes per class {dict(sorted(class_counts[s].items()))}")
    print(f"\nWrote {dst / 'splits.csv'}")


if __name__ == "__main__":
    main()
