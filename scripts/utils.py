"""Shared helpers for the road damage detection scripts.

Not meant to be run directly. The other scripts in this folder import from it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# --------------------------------------------------------------------------- #
# General helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def resolve_weights(weights) -> str:
    """Use the path as given, or relative to the project root. Otherwise return it
    unchanged (so names like 'yolov8s.pt' can be auto-downloaded by Ultralytics)."""
    p = Path(weights)
    if p.exists():
        return str(p)
    if (PROJECT_ROOT / p).exists():
        return str(PROJECT_ROOT / p)
    return str(weights)


def _is_rtdetr(weights: str) -> bool:
    if "rtdetr" in weights.lower() or "rt-detr" in weights.lower():
        return True
    p = Path(weights)
    if p.suffix == ".pt" and p.exists():
        try:
            import torch

            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            m = ckpt.get("ema") or ckpt.get("model")
            return type(m).__name__ == "RTDETRDetectionModel"
        except Exception:
            return False
    return False


def load_model(weights, task: str | None = "detect"):
    """Load a YOLO or RT-DETR model. RT-DETR needs its own class because it uses a
    different predictor, so it is detected from the file name or the checkpoint."""
    from ultralytics import RTDETR, YOLO

    weights = resolve_weights(weights)
    if _is_rtdetr(weights):
        return RTDETR(weights)
    return YOLO(weights, task=task) if task else YOLO(weights)


# --------------------------------------------------------------------------- #
# Dataset yaml helpers
# --------------------------------------------------------------------------- #
def resolve_data_yaml(data) -> Path:
    """Write a copy of the dataset yaml with an absolute `path:`.

    Ultralytics resolves relative dataset paths against its own datasets directory,
    not the current folder, so we always hand it an absolute path.
    """
    data_path = Path(data)
    if not data_path.exists():
        data_path = PROJECT_ROOT / data
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data}")
    with open(data_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_path.parent))
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    cfg["path"] = root.as_posix()
    out_dir = PROJECT_ROOT / "runs" / "_resolved"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / data_path.name
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def load_names(data_yaml) -> dict[int, str]:
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def get_split_dirs(data_yaml, split: str) -> tuple[Path, Path]:
    """Return (images_dir, labels_dir) for a split of a resolved dataset yaml."""
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if split not in cfg:
        raise KeyError(f"Split '{split}' is not defined in {data_yaml}")
    rel = cfg[split]
    if isinstance(rel, list):
        rel = rel[0]
    img_dir = Path(cfg["path"]) / rel
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        raise ValueError(f"Cannot derive the labels folder from {img_dir}")
    return img_dir, Path(*parts)


def list_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def read_yolo_labels(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO label file. Returns (class_ids[n], boxes_xyxy_normalised[n, 4])."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 4))
    arr = np.loadtxt(path, ndmin=2)
    cls = arr[:, 0].astype(int)
    cx, cy, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return cls, xyxy


# --------------------------------------------------------------------------- #
# Detection metrics (F1 at IoU 0.5, the RDD challenge metric)
# --------------------------------------------------------------------------- #
def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a[n, 4] and b[m, 4] in xyxy format."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


class DetectionEvaluator:
    """Accumulates predictions over a dataset and computes P / R / F1 at any
    confidence threshold.

    Predictions are matched to ground truth once (greedy, by descending confidence,
    same class, IoU >= iou_thr). Because greedy matching only depends on higher
    scoring boxes, the result for any threshold is just a filter on that single
    matching, so sweeping thresholds is cheap.

    IoU is computed on normalised coordinates, which gives the same value as pixel
    coordinates (IoU is invariant to scaling each axis).
    """

    def __init__(self, nc: int, iou_thr: float = 0.5):
        self.nc = nc
        self.iou_thr = iou_thr
        self.n_gt = np.zeros(nc, dtype=int)
        self._scores, self._cls, self._tp = [], [], []
        self._cache = None

    def add(self, pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls) -> None:
        self._cache = None
        pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
        pred_scores = np.asarray(pred_scores, dtype=float).reshape(-1)
        pred_cls = np.asarray(pred_cls).astype(int).reshape(-1)
        gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
        gt_cls = np.asarray(gt_cls).astype(int).reshape(-1)

        for c in range(self.nc):
            g = gt_boxes[gt_cls == c]
            self.n_gt[c] += len(g)
            m = pred_cls == c
            if not m.any():
                continue
            b, s = pred_boxes[m], pred_scores[m]
            order = np.argsort(-s)
            b, s = b[order], s[order]
            tp = np.zeros(len(b), dtype=bool)
            if len(g):
                ious = box_iou_np(b, g)
                matched = np.zeros(len(g), dtype=bool)
                for i in range(len(b)):
                    cand = np.where(matched, -1.0, ious[i])
                    j = int(cand.argmax())
                    if cand[j] >= self.iou_thr:
                        matched[j] = True
                        tp[i] = True
                        if matched.all():
                            break
            self._scores.append(s)
            self._cls.append(np.full(len(b), c, dtype=int))
            self._tp.append(tp)

    def _arrays(self):
        if self._cache is None:
            if self._scores:
                self._cache = (
                    np.concatenate(self._scores),
                    np.concatenate(self._cls),
                    np.concatenate(self._tp),
                )
            else:
                self._cache = (np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=bool))
        return self._cache

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    def at_threshold(self, conf: float) -> dict:
        scores, cls, tp = self._arrays()
        keep = scores >= conf
        tp_c = np.bincount(cls[keep & tp], minlength=self.nc)
        fp_c = np.bincount(cls[keep & ~tp], minlength=self.nc)
        fn_c = self.n_gt - tp_c
        per_class = []
        for c in range(self.nc):
            p, r, f1 = self._prf(int(tp_c[c]), int(fp_c[c]), int(fn_c[c]))
            per_class.append(
                {"class_id": c, "precision": p, "recall": r, "f1": f1,
                 "tp": int(tp_c[c]), "fp": int(fp_c[c]), "fn": int(fn_c[c]),
                 "n_gt": int(self.n_gt[c])}
            )
        p, r, f1 = self._prf(int(tp_c.sum()), int(fp_c.sum()), int(fn_c.sum()))
        return {"conf": float(conf), "precision": p, "recall": r, "f1": f1,
                "tp": int(tp_c.sum()), "fp": int(fp_c.sum()), "fn": int(fn_c.sum()),
                "per_class": per_class}

    def average_precision(self) -> np.ndarray:
        """Per-class AP at the evaluator's IoU threshold (101-point interpolation, as in COCO)."""
        scores, cls, tp = self._arrays()
        ap = np.zeros(self.nc)
        recall_points = np.linspace(0, 1, 101)
        for c in range(self.nc):
            if self.n_gt[c] == 0:
                continue
            m = cls == c
            order = np.argsort(-scores[m], kind="stable")
            hits = tp[m][order]
            tpc = np.cumsum(hits)
            fpc = np.cumsum(~hits)
            recall = tpc / self.n_gt[c]
            precision = tpc / np.maximum(tpc + fpc, 1)
            # precision envelope, then sample it at fixed recall points
            envelope = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
            idx = np.searchsorted(recall, recall_points, side="left")
            ap[c] = np.mean([envelope[i] if i < len(envelope) else 0.0 for i in idx])
        return ap

    def sweep(self, thresholds=DEFAULT_THRESHOLDS) -> list[dict]:
        return [self.at_threshold(t) for t in thresholds]

    def best(self, thresholds=DEFAULT_THRESHOLDS):
        """Returns (best_overall, best_per_class, all_rows)."""
        rows = self.sweep(thresholds)
        best = max(rows, key=lambda r: r["f1"])
        best_pc = []
        for c in range(self.nc):
            br = max(rows, key=lambda r: r["per_class"][c]["f1"])
            best_pc.append({"conf": br["conf"], **br["per_class"][c]})
        return best, best_pc, rows


# --------------------------------------------------------------------------- #
# Results logging
# --------------------------------------------------------------------------- #
def append_metrics_row(row: dict, path=None) -> Path:
    """Append one row to results/metrics.csv (created if missing)."""
    import pandas as pd

    path = Path(path) if path else PROJECT_ROOT / "results" / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists() and path.stat().st_size > 0:
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)
    return path
