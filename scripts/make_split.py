#!/usr/bin/env python
"""Create leak-free train / val / test splits from the converted RDD2022 data.

Why not a random split? RDD images are frames captured along roads, and
consecutive frames look almost identical. A random split puts near-duplicates in
both train and val, which inflates the scores. Here, images are grouped into
blocks of consecutive file numbers per country (e.g. Japan_000100 ... Japan_000149)
and whole blocks are assigned to a single split. This is a heuristic, since the
dataset has no explicit sequence IDs: increase --block-size if you suspect
longer sequences.

Splits are stratified by country (each country contributes roughly the same
fractions). For a domain-shift study, use --holdout-countries to send entire
countries to the test split instead.

Run after convert_voc_to_yolo.py:

    python scripts/make_split.py --root data/yolo
    python scripts/make_split.py --root data/yolo --holdout-countries Norway   # cross-country test

Outputs under <root>:

    images/{train,val,test}/    labels/{train,val,test}/
    splits.csv                  image -> split assignment

It also writes configs/rdd2022.yaml (use --skip-yaml to leave it alone).
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from utils import CLASS_NAMES, PROJECT_ROOT

SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/yolo", help="folder produced by convert_voc_to_yolo.py")
    p.add_argument("--val-frac", type=float, default=0.15, help="fraction of blocks per country for val")
    p.add_argument("--test-frac", type=float, default=0.10, help="fraction of blocks per country for test "
                                                                  "(ignored with --holdout-countries)")
    p.add_argument("--block-size", type=int, default=50, help="consecutive images treated as one sequence")
    p.add_argument("--holdout-countries", nargs="*", default=[], help="countries sent entirely to test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--link", action="store_true", help="symlink images instead of copying (saves disk)")
    p.add_argument("--overwrite", action="store_true", help="replace existing split folders")
    p.add_argument("--yaml", default="configs/rdd2022.yaml", help="dataset yaml to write (relative to project root)")
    p.add_argument("--skip-yaml", action="store_true")
    return p.parse_args()


def parse_idx(stem: str):
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def assign_splits(df: pd.DataFrame, args) -> pd.Series:
    rng = random.Random(args.seed)
    holdout = set(args.holdout_countries)
    test_frac = 0.0 if holdout else args.test_frac
    group_split = {}

    for country, sub in df.groupby("country"):
        groups = sorted(sub["group"].unique())
        if country in holdout:
            group_split.update({g: "test" for g in groups})
            continue
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * args.val_frac))
        if test_frac > 0 and n >= 3:
            n_test = max(n_test, 1)
        if args.val_frac > 0 and n >= 3:
            n_val = max(n_val, 1)
        for i, g in enumerate(groups):
            group_split[g] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    return df["group"].map(group_split)


def place(src: Path, dst: Path, link: bool, state: dict) -> None:
    if link and not state["link_failed"]:
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            state["link_failed"] = True
            print("Symlinks are not available here (on Windows they need admin / developer mode). "
                  "Falling back to copying.")
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    if args.val_frac + args.test_frac >= 1:
        sys.exit("--val-frac + --test-frac must be below 1")
    root = Path(args.root)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"{manifest} not found. Run convert_voc_to_yolo.py first.")

    df = pd.read_csv(manifest)
    unknown = set(args.holdout_countries) - set(df["country"].unique())
    if unknown:
        sys.exit(f"Unknown countries: {sorted(unknown)}. Available: {sorted(df['country'].unique())}")
    if args.holdout_countries:
        print(f"Holding out {args.holdout_countries} as the test split (--test-frac ignored).")

    idx = df["stem"].map(parse_idx)
    df["group"] = [f"{c}_{int(i) // args.block_size}" if pd.notna(i) else s
                   for c, i, s in zip(df["country"], idx, df["stem"])]
    df["split"] = assign_splits(df, args)

    # sanity check: no block may appear in two splits
    assert df.groupby("group")["split"].nunique().max() == 1, "a block leaked across splits"

    used = [s for s in SPLITS if (df["split"] == s).any()]
    for s in used:
        for kind in ("images", "labels"):
            d = root / kind / s
            if d.exists() and any(d.iterdir()):
                if not args.overwrite:
                    sys.exit(f"{d} already has files. Re-run with --overwrite to replace the splits.")
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    state = {"link_failed": False}
    label_src_dir = root / "all" / "labels"
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Placing files"):
        img_src = Path(row.image_path)
        place(img_src, root / "images" / row.split / f"{row.stem}{img_src.suffix.lower()}", args.link, state)
        shutil.copy2(label_src_dir / f"{row.stem}.txt", root / "labels" / row.split / f"{row.stem}.txt")

    df.drop(columns=["image_path", "label_path"]).to_csv(root / "splits.csv", index=False)

    # ---- summary ----
    cols = [f"n_{n}" for n in CLASS_NAMES.values()]
    summary = df.groupby("split").agg(images=("stem", "count"), empty=("n_boxes", lambda s: int((s == 0).sum())),
                                      boxes=("n_boxes", "sum"), **{c: (c, "sum") for c in cols})
    print("\nPer split:\n", summary.reindex(used).to_string())
    print("\nImages per country and split:\n", pd.crosstab(df["country"], df["split"]).reindex(columns=used).to_string())

    if not args.skip_yaml:
        try:
            path_value = root.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            path_value = root.resolve().as_posix()
        cfg = {"path": path_value, "train": "images/train", "val": "images/val"}
        if "test" in used:
            cfg["test"] = "images/test"
        cfg["names"] = dict(CLASS_NAMES)
        yaml_path = PROJECT_ROOT / args.yaml
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"\nWrote {yaml_path}")
    print("\nNext: python scripts/train.py --model yolov8s.pt --imgsz 640 --epochs 100")


if __name__ == "__main__":
    main()
