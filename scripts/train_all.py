#!/usr/bin/env python
"""Train every baseline model one after another, resuming where it left off.

    python scripts/train_all.py                      # all models, settings below
    python scripts/train_all.py --only yolov8s rtdetr_l
    python scripts/train_all.py --epochs 100         # same epoch count for every model
    python scripts/train_all.py --dry-run            # print the commands only

Safe to re-run after an interruption (sleep, reboot, crash): finished runs are
skipped (marker file DONE in the run folder) and interrupted runs are resumed from
their last.pt. Each run's console output goes to runs/logs/<name>.log.

Batch sizes are set for a 4 GB GPU (GTX 1650 Ti class) at 640 px. Every model uses
the same seed, image size and epoch count so the results are comparable.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# name -> (kind, weights, batch), fastest first so results arrive early.
# Measured on a GTX 1650 Ti (FP32, 640 px): ~18 / 6 / 4.5 / 3.2 train images per second.
MODELS = {
    "yolov8s": ("yolo", "weights/yolov8s.pt", 8),
    "yolov8m": ("yolo", "weights/yolov8m.pt", 4),
    "faster_rcnn": ("frcnn", None, 2),
    "rtdetr_l": ("yolo", "weights/rtdetr-l.pt", 2),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=2,
                   help="dataloader processes; each holds a copy of torch in RAM on Windows")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_command(key: str, args) -> tuple[list[str], Path]:
    kind, weights, batch = MODELS[key]
    name = f"{key}_{args.imgsz}_s{args.seed}"
    if kind == "frcnn":
        run_dir = PROJECT_ROOT / "runs" / "faster_rcnn" / name
        cmd = [PY, "scripts/train_faster_rcnn.py", "--imgsz", str(args.imgsz), "--epochs", str(args.epochs),
               "--batch", str(batch), "--device", args.device, "--workers", str(args.workers),
               "--seed", str(args.seed), "--name", name]  # no --amp: FP16 is unreliable on GTX 16xx
        if (run_dir / "last.pt").exists():
            cmd += ["--resume", str(run_dir / "last.pt")]
        return cmd, run_dir

    run_dir = PROJECT_ROOT / "runs" / "detect" / name
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        return [PY, "scripts/train.py", "--resume", str(last)], run_dir
    cmd = [PY, "scripts/train.py", "--model", weights, "--imgsz", str(args.imgsz), "--epochs", str(args.epochs),
           "--batch", str(batch), "--device", args.device, "--workers", str(args.workers),
           "--seed", str(args.seed), "--name", name, "--set", "exist_ok=True"]
    return cmd, run_dir


def main():
    args = parse_args()
    log_dir = PROJECT_ROOT / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for key in args.only:
        cmd, run_dir = build_command(key, args)
        if (run_dir / "DONE").exists():
            print(f"[skip] {key}: already finished ({run_dir})")
            continue
        print(f"[run ] {key}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        start = time.time()
        with open(log_dir / f"{run_dir.name}.log", "a", encoding="utf-8") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            log.flush()
            code = subprocess.call(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        hours = (time.time() - start) / 3600
        if code == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "DONE").write_text(f"finished {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
            print(f"[done] {key} in {hours:.1f} h", flush=True)
        else:
            failed.append(key)
            print(f"[FAIL] {key} exited with {code} after {hours:.1f} h; see {log_dir / (run_dir.name + '.log')}",
                  flush=True)
    if failed:
        sys.exit(f"Failed: {failed}. Re-run the same command to resume them.")


if __name__ == "__main__":
    main()
