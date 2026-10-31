#!/usr/bin/env python
"""Export a trained model (ONNX / TFLite / TensorRT / ...) and benchmark it.

For each requested format the script exports the model, then measures batch-size-1
latency (mean, p50, p95, FPS) on real images, plus file size. The original .pt model
is benchmarked too, as the reference. Results go to results/export_benchmark.csv.

Examples:
    python scripts/export_model.py --weights runs/detect/yolov8s_640_s42/weights/best.pt --formats onnx

    # several formats, with a real-image benchmark and an accuracy check on val
    python scripts/export_model.py --weights best.pt --formats onnx openvino tflite \\
        --source data/yolo/images/val --val-data configs/rdd2022.yaml

    # GPU / Jetson
    python scripts/export_model.py --weights best.pt --formats engine --half --device 0

Notes:
  * Exported CPU formats (onnx, openvino, tflite, ncnn) are benchmarked on CPU, which
    is the relevant number for edge devices. .pt, torchscript and engine use --device.
  * tflite needs `pip install tensorflow`; engine needs TensorRT and an NVIDIA GPU;
    openvino needs `pip install openvino`. A format that fails to export is reported
    and skipped; the others still run.
  * Run the benchmark on the device you will deploy to (e.g. the Jetson or Raspberry Pi
    itself); numbers from your laptop do not transfer.
  * --val-data checks that the exported model keeps its accuracy (important for int8/half).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils import PROJECT_ROOT, list_images, load_model, resolve_data_yaml, resolve_weights, save_json

DEVICE_FORMATS = {"pt", "torchscript", "engine"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--formats", nargs="+", default=["onnx"],
                   help="onnx, openvino, tflite, engine, torchscript, ncnn, coreml ...")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="device for export and for pt/torchscript/engine benchmarks")
    p.add_argument("--half", action="store_true", help="FP16 export (needs a GPU for most formats)")
    p.add_argument("--int8", action="store_true", help="INT8 quantisation (uses the dataset for calibration)")
    p.add_argument("--dynamic", action="store_true", help="dynamic input shapes (onnx / engine)")
    p.add_argument("--data", default="configs/rdd2022.yaml", help="dataset yaml (int8 calibration, --val-data)")
    p.add_argument("--source", default=None, help="folder of images for the benchmark (default: random noise image)")
    p.add_argument("--n-images", type=int, default=20, help="how many images from --source to cycle through")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--val-data", default=None, help="dataset yaml: also compute mAP@0.5 of each exported model on val")
    return p.parse_args()


def dir_or_file_size_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    return path.stat().st_size / 1e6


def load_bench_images(args):
    if args.source:
        files = list_images(args.source)[:args.n_images]
        imgs = [cv2.imread(str(f)) for f in files]
        imgs = [i for i in imgs if i is not None]
        if imgs:
            return imgs
        print(f"No readable images in {args.source}; using a random image instead.")
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)]


def benchmark(path: str, imgs, args, device) -> dict:
    model = load_model(path)
    for i in range(args.warmup):
        model.predict(imgs[i % len(imgs)], imgsz=args.imgsz, device=device, verbose=False)
    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        model.predict(imgs[i % len(imgs)], imgsz=args.imgsz, device=device, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"latency_ms_mean": float(t.mean()), "latency_ms_p50": float(np.percentile(t, 50)),
            "latency_ms_p95": float(np.percentile(t, 95)), "fps": float(1000 / t.mean())}


def quick_map(path: str, data_yaml, args, device):
    try:
        m = load_model(path).val(data=str(data_yaml), imgsz=args.imgsz, batch=1, device=device, conf=0.001,
                                 plots=False, verbose=False)
        return float(m.box.map50)
    except Exception as e:
        print(f"  accuracy check failed: {e}")
        return None


def main():
    args = parse_args()
    weights = resolve_weights(args.weights)
    imgs = load_bench_images(args)
    data_yaml = resolve_data_yaml(args.data)
    val_yaml = resolve_data_yaml(args.val_data) if args.val_data else None

    model = load_model(weights)
    try:
        params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    except Exception:
        params_m = None
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = float(get_flops(model.model, args.imgsz))
    except Exception:
        gflops = None

    rows = []
    jobs = [("pt", Path(weights))]
    for fmt in args.formats:
        print(f"\n--- Exporting to {fmt} ---")
        try:
            export_kwargs = dict(format=fmt, imgsz=args.imgsz, dynamic=args.dynamic, simplify=True, batch=1,
                                 device=args.device)
            if args.half:
                export_kwargs["half"] = True
            if args.int8:
                export_kwargs.update(int8=True, data=str(data_yaml))  # data is used for calibration
            exported = model.export(**export_kwargs)
            jobs.append((fmt, Path(str(exported))))
        except Exception as e:
            print(f"Export to {fmt} failed: {e}")
            rows.append({"format": fmt, "status": f"export failed: {type(e).__name__}"})

    for fmt, path in jobs:
        device = args.device if fmt in DEVICE_FORMATS else "cpu"
        print(f"\n--- Benchmarking {fmt}: {path.name} on {device or 'auto'} ---")
        row = {"model": Path(weights).parent.parent.name or Path(weights).stem, "format": fmt, "file": path.name,
               "imgsz": args.imgsz, "half": args.half, "int8": args.int8, "device": device or "auto",
               "size_mb": dir_or_file_size_mb(path), "params_M": params_m, "gflops": gflops, "status": "ok"}
        try:
            row.update(benchmark(str(path), imgs, args, device))
            if val_yaml:
                row["map50"] = quick_map(str(path), val_yaml, args, device)
        except Exception as e:
            row["status"] = f"benchmark failed: {type(e).__name__}: {e}"
            print(row["status"])
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = [c for c in ["model", "format", "size_mb", "latency_ms_mean", "latency_ms_p95", "fps", "map50", "status"]
            if c in df.columns]
    print("\n" + df[cols].round(2).to_string(index=False))

    out = PROJECT_ROOT / "results" / "export_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
