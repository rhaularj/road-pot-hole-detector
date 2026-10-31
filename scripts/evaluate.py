#!/usr/bin/env python
"""Evaluate a trained model: F1 @ IoU 0.5, mAP, per-class metrics and speed.

Two evaluations are combined:
  1. Ultralytics validation: mAP@0.5, mAP@0.5:0.95, per-class AP, plots and confusion matrix.
  2. A custom detection evaluation: P / R / F1 at IoU 0.5 over a sweep of confidence
     thresholds (the RDD challenge metric), overall and per class.

Examples:
    python scripts/evaluate.py --weights runs/detect/yolov8s_640_s42/weights/best.pt

    # tune the threshold on val, then apply that exact threshold to the held-out test split
    python scripts/evaluate.py --weights best.pt --split val
    python scripts/evaluate.py --weights best.pt --split test --fixed-conf 0.30

The best-threshold F1 on the same split you tuned on is optimistic. For honest
numbers, pick the threshold on val and report the F1 at that --fixed-conf on test.

Outputs:
    results/evaluations/<run>_<split>.json   full results
    results/metrics.csv                      one row appended per evaluation

Speed numbers come from Ultralytics' per-image timing (preprocess + inference +
postprocess). Use export_model.py for a dedicated batch-size-1 latency benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (DEFAULT_THRESHOLDS, PROJECT_ROOT, DetectionEvaluator, append_metrics_row, get_split_dirs,
                   list_images, load_model, load_names, read_yolo_labels, resolve_data_yaml, resolve_weights,
                   save_json)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="configs/rdd2022.yaml")
    p.add_argument("--split", default="val", help="val or test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=8, help="dataloader processes for the validation pass")
    p.add_argument("--conf", type=float, default=0.001, help="minimum confidence kept for the sweep")
    p.add_argument("--nms-iou", type=float, default=0.6, help="NMS IoU threshold")
    p.add_argument("--fixed-conf", type=float, default=None,
                   help="also report F1 at this exact threshold (e.g. one tuned on val)")
    p.add_argument("--name", default=None, help="label for the results (default: run folder name)")
    p.add_argument("--max-images", type=int, default=None, help="evaluate on the first N images only (quick check)")
    p.add_argument("--skip-val", action="store_true", help="skip the Ultralytics validation (mAP, plots)")
    return p.parse_args()


def run_name(weights: str) -> str:
    p = Path(weights)
    return p.parent.parent.name if p.parent.name == "weights" else p.stem


def main():
    args = parse_args()
    weights = resolve_weights(args.weights)
    name = args.name or run_name(weights)
    data_yaml = resolve_data_yaml(args.data)
    names = load_names(data_yaml)
    nc = len(names)
    model = load_model(weights)

    # ---------- 1. Ultralytics validation (mAP, plots) ----------
    ul = {}
    if not args.skip_val:
        m = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, batch=args.batch,
                      conf=args.conf, iou=args.nms_iou, device=args.device, workers=args.workers, plots=True,
                      project=str(PROJECT_ROOT / "runs" / "val"), name=f"{name}_{args.split}",
                      exist_ok=True, verbose=False)
        ul = {"map50": float(m.box.map50), "map50_95": float(m.box.map),
              "precision": float(m.box.mp), "recall": float(m.box.mr),
              "per_class": {names[int(c)]: {"precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                                            "ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i])}
                            for i, c in enumerate(m.box.ap_class_index)}}

    # ---------- 2. Custom F1 sweep + speed ----------
    img_dir, lbl_dir = get_split_dirs(data_yaml, args.split)
    images = list_images(img_dir)
    if args.max_images:
        images = images[:args.max_images]
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    evaluator = DetectionEvaluator(nc, iou_thr=0.5)
    timings = []
    # Never pass a Python list: Ultralytics decodes every listed image into memory up front, which runs
    # out of RAM on a full split. A folder or a .txt of paths is read lazily, batch by batch.
    source = str(img_dir)
    if args.max_images:
        source = PROJECT_ROOT / "runs" / "_resolved" / f"{name}_{args.split}_images.txt"
        source.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
        source = str(source)
    stream = model.predict(source=source, imgsz=args.imgsz, conf=args.conf,
                           iou=args.nms_iou, device=args.device, batch=args.batch, stream=True, verbose=False)
    for r in tqdm(stream, total=len(images), desc=f"Predicting {args.split}"):
        gt_cls, gt_boxes = read_yolo_labels(lbl_dir / f"{Path(r.path).stem}.txt")
        b = r.boxes
        evaluator.add(b.xyxyn.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy(), gt_boxes, gt_cls)
        timings.append((r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]))

    best, best_pc, rows = evaluator.best(DEFAULT_THRESHOLDS)
    fixed = evaluator.at_threshold(args.fixed_conf) if args.fixed_conf is not None else None

    t = np.array(timings[5:] if len(timings) > 10 else timings)  # skip warm-up images
    ms_per_img = float(t.sum(axis=1).mean())
    fps = 1000.0 / ms_per_img if ms_per_img > 0 else 0.0
    try:
        params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    except Exception:
        params_m = None

    # ---------- report ----------
    print("\n" + "=" * 64)
    print(f"Model: {name} | split: {args.split} | images: {len(images)} | imgsz: {args.imgsz}")
    print("=" * 64)
    if ul:
        print(f"mAP@0.5 = {ul['map50']:.4f} | mAP@0.5:0.95 = {ul['map50_95']:.4f}")
    print(f"Best F1@IoU0.5 = {best['f1']:.4f} at conf {best['conf']:.2f} "
          f"(P {best['precision']:.3f}, R {best['recall']:.3f})")
    if fixed:
        print(f"F1 at fixed conf {fixed['conf']:.2f} = {fixed['f1']:.4f} "
              f"(P {fixed['precision']:.3f}, R {fixed['recall']:.3f})")
    print(f"Speed: {ms_per_img:.1f} ms/img ({fps:.1f} FPS)" + (f" | params: {params_m:.1f}M" if params_m else ""))
    print(f"\n{'class':<8}{'GT':>7}{'best conf':>11}{'P':>8}{'R':>8}{'F1':>8}{'AP50':>8}")
    for c in range(nc):
        pc = best_pc[c]
        ap50 = ul.get("per_class", {}).get(names[c], {}).get("ap50")
        print(f"{names[c]:<8}{pc['n_gt']:>7}{pc['conf']:>11.2f}{pc['precision']:>8.3f}{pc['recall']:>8.3f}"
              f"{pc['f1']:>8.3f}{(f'{ap50:.3f}' if ap50 is not None else '-'):>8}")

    # ---------- save ----------
    out = {"name": name, "weights": weights, "split": args.split, "imgsz": args.imgsz, "n_images": len(images),
           "ultralytics": ul, "best_f1": {k: v for k, v in best.items() if k != "per_class"},
           "best_per_class": {names[c]: best_pc[c] for c in range(nc)},
           "fixed_conf": ({k: v for k, v in fixed.items() if k != "per_class"} if fixed else None),
           "sweep": [{k: v for k, v in r.items() if k != "per_class"} for r in rows],
           "ms_per_img": ms_per_img, "fps": fps, "params_m": params_m}
    json_path = PROJECT_ROOT / "results" / "evaluations" / f"{name}_{args.split}.json"
    save_json(out, json_path)

    row = {"run": name, "split": args.split, "imgsz": args.imgsz, "params_M": params_m,
           "precision": best["precision"], "recall": best["recall"], "f1": best["f1"], "f1_conf": best["conf"],
           "map50": ul.get("map50"), "map50_95": ul.get("map50_95"), "ms_per_img": ms_per_img, "fps": fps}
    if fixed:
        row.update({"fixed_conf": fixed["conf"], "f1_at_fixed_conf": fixed["f1"]})
    for c in range(nc):
        row[f"ap50_{names[c]}"] = ul.get("per_class", {}).get(names[c], {}).get("ap50")
        row[f"f1_{names[c]}"] = best_pc[c]["f1"]
    csv_path = append_metrics_row(row)
    print(f"\nSaved {json_path}\nAppended a row to {csv_path}")


if __name__ == "__main__":
    main()
