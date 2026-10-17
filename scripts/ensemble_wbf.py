#!/usr/bin/env python
"""Ensemble several detectors with Weighted Boxes Fusion (WBF), optionally with TTA.

This is the trick most top CRDDC2022 teams relied on. Each model predicts on the
split, the predictions are fused with WBF, and F1 @ IoU 0.5 is computed for every
single model and for the ensemble so you can see the gain directly.

Examples:
    python scripts/ensemble_wbf.py \\
        --weights runs/detect/yolov8m_1024_s42/weights/best.pt runs/detect/yolo11m_1024_s42/weights/best.pt \\
        --imgsz 1024 1024 --split val

    # with Ultralytics TTA (flip + multi-scale) and a sweep of the WBF IoU threshold
    python scripts/ensemble_wbf.py --weights a.pt b.pt c.pt --tta --wbf-iou 0.45 0.55 0.65

    # tune on val, then report on test with the threshold chosen on val
    python scripts/ensemble_wbf.py --weights a.pt b.pt --split test --fixed-conf 0.30

Notes:
  * Models run at low confidence (0.001) so WBF has all candidates; the confidence
    threshold is chosen afterwards by the F1 sweep.
  * --tta has no effect on RT-DETR (Ultralytics falls back to single-scale).
  * Selecting the WBF IoU / threshold on val and reporting on a separate test split
    avoids over-tuning on one set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (DEFAULT_THRESHOLDS, PROJECT_ROOT, DetectionEvaluator, append_metrics_row, get_split_dirs,
                   list_images, load_model, load_names, read_yolo_labels, resolve_data_yaml, resolve_weights,
                   save_json)

try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError as e:  # pragma: no cover
    raise SystemExit("ensemble-boxes is not installed. Run: pip install ensemble-boxes") from e


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", nargs="+", required=True, help="two or more trained models")
    p.add_argument("--data", default="configs/rdd2022.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--imgsz", type=int, nargs="+", default=[640], help="one value, or one per model")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--nms-iou", type=float, default=0.6)
    p.add_argument("--tta", action="store_true", help="Ultralytics test-time augmentation for each model")
    p.add_argument("--wbf-iou", type=float, nargs="+", default=[0.55], help="WBF IoU threshold(s) to try")
    p.add_argument("--skip-thr", type=float, default=0.001, help="WBF: ignore boxes below this score")
    p.add_argument("--conf-type", default="avg", choices=["avg", "max", "box_and_model_avg", "absent_model_aware_avg"])
    p.add_argument("--model-weights", type=float, nargs="+", default=None, help="relative weight per model")
    p.add_argument("--fixed-conf", type=float, default=None, help="also report F1 at this exact threshold")
    p.add_argument("--name", default="ensemble")
    p.add_argument("--max-images", type=int, default=None)
    return p.parse_args()


def predict_all(weights: str, imgsz: int, images, args):
    """Run one model over all images. Returns {image_path: (boxes_xyxyn, scores, labels)}."""
    model = load_model(weights)
    out = {}
    stream = model.predict(source=[str(p) for p in images], imgsz=imgsz, conf=args.conf, iou=args.nms_iou,
                           device=args.device, batch=args.batch, augment=args.tta, stream=True, verbose=False)
    for r in tqdm(stream, total=len(images), desc=f"Predicting ({Path(weights).parent.parent.name or weights})"):
        b = r.boxes
        out[str(r.path)] = (b.xyxyn.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))
    return out


def fuse(per_model, key, args, iou_thr):
    boxes, scores, labels = [], [], []
    for pm in per_model:
        b, s, l = pm[key]
        boxes.append(np.clip(b, 0, 1).tolist())
        scores.append(s.tolist())
        labels.append(l.tolist())
    if all(len(s) == 0 for s in scores):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    fb, fs, fl = weighted_boxes_fusion(boxes, scores, labels, weights=args.model_weights, iou_thr=iou_thr,
                                       skip_box_thr=args.skip_thr, conf_type=args.conf_type)
    return fb, fs, np.asarray(fl).astype(int)


def evaluate(pred_fn, images, lbl_dir, nc):
    ev = DetectionEvaluator(nc, iou_thr=0.5)
    for p in images:
        gt_cls, gt_boxes = read_yolo_labels(lbl_dir / f"{p.stem}.txt")
        b, s, l = pred_fn(str(p))
        ev.add(b, s, l, gt_boxes, gt_cls)
    return ev


def main():
    args = parse_args()
    n = len(args.weights)
    if n < 2:
        raise SystemExit("Give at least two models to --weights.")
    imgsz = args.imgsz * n if len(args.imgsz) == 1 else args.imgsz
    if len(imgsz) != n:
        raise SystemExit("--imgsz must be one value or one value per model.")
    if args.model_weights and len(args.model_weights) != n:
        raise SystemExit("--model-weights must have one value per model.")

    data_yaml = resolve_data_yaml(args.data)
    nc = len(load_names(data_yaml))
    names = load_names(data_yaml)
    img_dir, lbl_dir = get_split_dirs(data_yaml, args.split)
    images = list_images(img_dir)
    if args.max_images:
        images = images[:args.max_images]
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    weights = [resolve_weights(w) for w in args.weights]
    per_model = [predict_all(w, s, images, args) for w, s in zip(weights, imgsz)]
    labels = [Path(w).parent.parent.name or Path(w).stem for w in weights]

    results = []  # (label, evaluator)
    for lab, pm in zip(labels, per_model):
        results.append((lab, evaluate(lambda k, pm=pm: pm[k], images, lbl_dir, nc)))
    for thr in args.wbf_iou:
        ev = evaluate(lambda k, thr=thr: fuse(per_model, k, args, thr), images, lbl_dir, nc)
        results.append((f"{args.name}_wbf{thr:g}" + ("_tta" if args.tta else ""), ev))

    # ---------- report ----------
    print("\n" + "=" * 78)
    print(f"Split: {args.split} | images: {len(images)} | TTA: {args.tta}")
    print("=" * 78)
    print(f"{'model':<44}{'best F1':>9}{'conf':>7}{'P':>7}{'R':>7}" + (f"{'F1@fixed':>10}" if args.fixed_conf else ""))
    summary = {}
    for lab, ev in results:
        best, best_pc, _ = ev.best(DEFAULT_THRESHOLDS)
        fixed = ev.at_threshold(args.fixed_conf) if args.fixed_conf is not None else None
        line = f"{lab:<44}{best['f1']:>9.4f}{best['conf']:>7.2f}{best['precision']:>7.3f}{best['recall']:>7.3f}"
        if fixed:
            line += f"{fixed['f1']:>10.4f}"
        print(line)
        summary[lab] = {"best": {k: v for k, v in best.items() if k != "per_class"},
                        "per_class_f1": {names[c]: best_pc[c]["f1"] for c in range(nc)},
                        "fixed": ({k: v for k, v in fixed.items() if k != "per_class"} if fixed else None)}
        if lab.startswith(args.name):
            row = {"run": lab, "split": args.split, "imgsz": "/".join(map(str, imgsz)), "precision": best["precision"],
                   "recall": best["recall"], "f1": best["f1"], "f1_conf": best["conf"]}
            if fixed:
                row.update({"fixed_conf": fixed["conf"], "f1_at_fixed_conf": fixed["f1"]})
            for c in range(nc):
                row[f"f1_{names[c]}"] = best_pc[c]["f1"]
            append_metrics_row(row)

    out_path = PROJECT_ROOT / "results" / "evaluations" / f"{args.name}_{args.split}_ensemble.json"
    save_json({"weights": weights, "imgsz": imgsz, "tta": args.tta, "wbf_iou": args.wbf_iou,
               "model_weights": args.model_weights, "conf_type": args.conf_type, "results": summary}, out_path)
    print(f"\nSaved {out_path}\nEnsemble rows appended to results/metrics.csv")


if __name__ == "__main__":
    main()
