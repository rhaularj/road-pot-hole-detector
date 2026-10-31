#!/usr/bin/env python
"""Train a YOLO or RT-DETR model on the road damage dataset (Ultralytics).

Examples:
    # baseline
    python scripts/train.py --model yolov8s.pt --imgsz 640 --epochs 100

    # stronger model, larger images
    python scripts/train.py --model yolov8m.pt --imgsz 1024 --epochs 100 --batch 8

    # RT-DETR comparison
    python scripts/train.py --model rtdetr-l.pt --epochs 100 --batch 8

    # from scratch (no COCO pretraining): use a model yaml instead of weights
    python scripts/train.py --model yolov8s.yaml --name scratch_yolov8s

    # ablation: extra Ultralytics arguments with --set
    python scripts/train.py --model yolov8s.pt --set mixup=0.1 --set degrees=5 --name aug_mixup

    # hyperparameter evolution
    python scripts/train.py --model yolov8s.pt --tune --tune-iterations 50 --tune-epochs 30

    # resume an interrupted run
    python scripts/train.py --resume runs/detect/yolov8s_640_s42/weights/last.pt

Every run saves its settings to <run_dir>/cli_args.json for reproducibility.
Use the same --seed, --epochs and --imgsz when comparing models.
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
import time
from pathlib import Path

from utils import PROJECT_ROOT, load_model, resolve_data_yaml, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolov8s.pt",
                   help="weights (.pt = pretrained) or a model yaml (e.g. yolov8s.yaml = from scratch)")
    p.add_argument("--data", default="configs/rdd2022.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16, help="-1 = auto batch size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="e.g. 0, 0,1 or cpu (default: auto)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=30, help="early-stopping patience (epochs)")
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--cos-lr", action="store_true", help="cosine learning-rate schedule")
    p.add_argument("--close-mosaic", type=int, default=10, help="disable mosaic for the last N epochs")
    p.add_argument("--freeze", type=int, default=None, help="freeze the first N layers")
    p.add_argument("--cache", choices=["ram", "disk"], default=None, help="cache images for faster epochs")
    p.add_argument("--no-deterministic", action="store_true", help="faster, but not bit-reproducible")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default=None, help="run name (default: <model>_<imgsz>_s<seed>)")
    p.add_argument("--resume", default=None, help="path to last.pt (or resume.pt) of an interrupted run")
    p.add_argument("--max-hours", type=float, default=None,
                   help="stop cleanly before this many hours (e.g. cloud session limits); "
                        "writes weights/resume.pt to continue with --resume")
    p.add_argument("--set", dest="extra", action="append", default=[], metavar="KEY=VALUE",
                   help="any extra Ultralytics argument, repeatable (e.g. --set mixup=0.1)")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (run `wandb login` first)")
    p.add_argument("--tune", action="store_true", help="run hyperparameter evolution instead of training")
    p.add_argument("--tune-iterations", type=int, default=50)
    p.add_argument("--tune-epochs", type=int, default=30)
    return p.parse_args()


def parse_extra(pairs):
    out = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def configure_wandb(enabled: bool) -> None:
    try:
        from ultralytics import settings

        settings.update({"wandb": bool(enabled)})
    except Exception as e:  # settings API differs between versions
        print(f"Could not toggle the W&B setting: {e}")
    if enabled:
        os.environ.setdefault("WANDB_PROJECT", "road-damage-detection")


def add_run_callbacks(model, max_hours: float | None) -> None:
    """Optional time budget, plus PAUSED / resume.pt bookkeeping for interrupted runs.

    Ultralytics strips the optimizer from last.pt whenever training ends, so a run
    stopped early could not be resumed. Before stopping for the time budget, last.pt
    (still complete at that point) is copied to resume.pt and a PAUSED file is written.
    When a run finishes normally both are removed.
    """
    start = time.time()
    state = {"paused": False}

    def on_fit_epoch_end(trainer):
        if not max_hours or state["paused"] or trainer.stop:  # also called again by final_eval
            return
        elapsed = time.time() - start
        epochs_run = trainer.epoch + 1 - trainer.start_epoch
        if elapsed + 1.2 * elapsed / max(epochs_run, 1) > max_hours * 3600:
            shutil.copy2(trainer.last, trainer.wdir / "resume.pt")
            trainer.stop = state["paused"] = True
            print(f"\nTime budget of {max_hours} h reached after epoch {trainer.epoch + 1}/{trainer.epochs}. "
                  f"Continue with: --resume {trainer.wdir / 'resume.pt'}", flush=True)

    def on_train_end(trainer):
        paused = Path(trainer.save_dir) / "PAUSED"
        if state["paused"]:
            paused.write_text(f"stopped after epoch {trainer.epoch + 1}\n", encoding="utf-8")
        else:
            paused.unlink(missing_ok=True)
            (trainer.wdir / "resume.pt").unlink(missing_ok=True)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)


def main():
    args = parse_args()
    set_seed(args.seed)
    configure_wandb(args.wandb)
    project = str((PROJECT_ROOT / args.project).resolve())

    if args.resume:
        resolve_data_yaml(args.data)  # a fresh (cloud) session needs the resolved yaml the run points to
        model = load_model(args.resume)
        add_run_callbacks(model, args.max_hours)
        model.train(resume=True)
        return

    data_yaml = str(resolve_data_yaml(args.data))
    name = args.name or f"{Path(args.model).stem}_{args.imgsz}_s{args.seed}"
    model = load_model(args.model)

    common = dict(data=data_yaml, imgsz=args.imgsz, device=args.device, project=project, name=name,
                  seed=args.seed, deterministic=not args.no_deterministic, workers=args.workers)
    extra = parse_extra(args.extra)

    if args.tune:
        print(f"Hyperparameter evolution: {args.tune_iterations} iterations x {args.tune_epochs} epochs")
        model.tune(epochs=args.tune_epochs, iterations=args.tune_iterations, optimizer="AdamW",
                   plots=False, save=False, val=False, **common, **extra)
        return

    kwargs = dict(common, epochs=args.epochs, batch=args.batch, patience=args.patience,
                  optimizer=args.optimizer, cos_lr=args.cos_lr, close_mosaic=args.close_mosaic,
                  lr0=args.lr0, freeze=args.freeze, cache=args.cache or False, **extra)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    add_run_callbacks(model, args.max_hours)
    results = model.train(**kwargs)

    save_dir = Path(model.trainer.save_dir)
    try:
        import torch
        import ultralytics

        versions = {"ultralytics": ultralytics.__version__, "torch": torch.__version__}
    except Exception:
        versions = {}
    save_json({"cli_args": vars(args), "train_kwargs": kwargs, "versions": versions}, save_dir / "cli_args.json")

    print("\n" + "=" * 60)
    print(f"Run folder : {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    try:
        print(f"Val mAP@0.5 = {results.box.map50:.4f} | mAP@0.5:0.95 = {results.box.map:.4f}")
    except Exception:
        pass
    print(f"Next: python scripts/evaluate.py --weights {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
