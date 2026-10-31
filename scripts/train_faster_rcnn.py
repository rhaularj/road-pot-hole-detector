#!/usr/bin/env python
"""Train Faster R-CNN (ResNet-50 FPN) on the prepared YOLO-format RDD2022 data.

Examples:
    python scripts/train_faster_rcnn.py --epochs 30 --imgsz 640 --batch 2 --device 0
    python scripts/train_faster_rcnn.py --resume runs/faster_rcnn/<name>/last.pt --max-hours 11

    # evaluate: pick the threshold on val, then report test F1 at that threshold
    python scripts/train_faster_rcnn.py --eval-weights runs/faster_rcnn/<name>/best.pt --split val
    python scripts/train_faster_rcnn.py --eval-weights runs/faster_rcnn/<name>/best.pt --split test --fixed-conf 0.3

The script converts normalized YOLO boxes to torchvision's pixel-space boxes and
uses COCO-pretrained Faster R-CNN weights with a new four-class prediction head.
Each epoch writes a resumable checkpoint to runs/faster_rcnn/<name>/last.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from utils import (PROJECT_ROOT, DetectionEvaluator, append_metrics_row, get_split_dirs, list_images, load_names,
                   resolve_data_yaml, save_json)


class RoadDamageDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, num_classes: int, fraction: float = 1.0):
        self.images = list_images(image_dir)
        if fraction < 1.0:
            self.images = random.Random(0).sample(self.images, max(1, int(len(self.images) * fraction)))
            self.images.sort()
        self.label_dir = label_dir
        self.num_classes = num_classes
        if not self.images:
            raise FileNotFoundError(f"No images found in {image_dir}")
        missing = [p.stem for p in self.images if not (label_dir / f"{p.stem}.txt").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} label files in {label_dir}; examples: {missing[:5]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            width, height = image.size
            image_tensor = pil_to_tensor(image).float().div_(255)

        boxes, labels = [], []
        label_path = self.label_dir / f"{image_path.stem}.txt"
        with label_path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields")
                try:
                    values = [float(value) for value in fields]
                except ValueError as exc:
                    raise ValueError(f"{label_path}:{line_number}: invalid numeric value") from exc
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"{label_path}:{line_number}: values must be finite")
                class_value, cx, cy, box_width, box_height = values
                class_id = int(class_value)
                if class_value != class_id or not 0 <= class_id < self.num_classes:
                    raise ValueError(f"{label_path}:{line_number}: invalid class id {class_value}")
                if box_width <= 0 or box_height <= 0:
                    raise ValueError(f"{label_path}:{line_number}: box width and height must be positive")

                x1 = max(0.0, (cx - box_width / 2) * width)
                y1 = max(0.0, (cy - box_height / 2) * height)
                x2 = min(float(width), (cx + box_width / 2) * width)
                y2 = min(float(height), (cy + box_height / 2) * height)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue  # degenerate after clipping; torchvision rejects zero-size boxes
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id + 1)  # torchvision reserves label 0 for background

        target = {
            "size": torch.tensor([height, width], dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        target["area"] = ((target["boxes"][:, 2] - target["boxes"][:, 0]) *
                          (target["boxes"][:, 3] - target["boxes"][:, 1]))
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int, imgsz: int, pretrained: bool):
    """Faster R-CNN with the same architecture whether or not COCO weights are loaded.

    With COCO weights torchvision freezes BatchNorm and the first two backbone stages;
    without weights it would build plain BatchNorm instead, which cannot load our
    checkpoints. The no-weights branch rebuilds the pretrained layout explicitly.
    """
    if pretrained:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                         min_size=imgsz, max_size=imgsz)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        return model
    backbone = _resnet_fpn_extractor(resnet50(weights=None, norm_layer=FrozenBatchNorm2d), trainable_layers=3)
    return FasterRCNN(backbone, num_classes=num_classes + 1, min_size=imgsz, max_size=imgsz)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, amp=False, return_evaluator=False):
    """F1 @ IoU 0.5 (best confidence threshold) and mAP@0.5, using the same matching as evaluate.py."""
    model.eval()
    evaluator = DetectionEvaluator(num_classes, iou_thr=0.5)
    for images, targets in tqdm(loader, desc="Validating", leave=False):
        images = [image.to(device, non_blocking=device.type == "cuda") for image in images]
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(images)
        for output, target in zip(outputs, targets):
            height, width = target["size"].tolist()
            scale = torch.tensor([width, height, width, height], dtype=torch.float32)
            evaluator.add((output["boxes"].float().cpu() / scale).numpy(), output["scores"].float().cpu().numpy(),
                          (output["labels"].cpu() - 1).numpy(), (target["boxes"] / scale).numpy(),
                          (target["labels"] - 1).numpy())
    best, _, _ = evaluator.best()
    ap50 = evaluator.average_precision()
    metrics = {"precision": best["precision"], "recall": best["recall"], "f1": best["f1"],
               "f1_conf": best["conf"], "map50": float(ap50.mean()), "ap50": ap50.tolist()}
    return (metrics, evaluator) if return_evaluator else metrics


def run_evaluation(args, device, data_yaml, names):
    """Evaluate saved weights on one split and log them like scripts/evaluate.py does."""
    weights = Path(args.eval_weights)
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    num_classes = len(names)
    if checkpoint["num_classes"] != num_classes:
        raise ValueError(f"{weights} has {checkpoint['num_classes']} classes, the dataset has {num_classes}")
    imgsz = checkpoint.get("imgsz", args.imgsz)
    model = build_model(num_classes, imgsz, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    images, labels = get_split_dirs(data_yaml, args.split)
    loader = DataLoader(RoadDamageDataset(images, labels, num_classes, args.fraction), batch_size=1,
                        shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
    start = time.time()
    metrics, evaluator = evaluate(model, loader, device, num_classes, args.amp, return_evaluator=True)
    ms_per_img = 1000 * (time.time() - start) / len(loader.dataset)
    fixed = evaluator.at_threshold(args.fixed_conf) if args.fixed_conf is not None else None
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    name = args.name or weights.parent.name
    print(f"\nModel: {name} | split: {args.split} | images: {len(loader.dataset)} | imgsz: {imgsz}")
    print(f"mAP@0.5 = {metrics['map50']:.4f}")
    print(f"Best F1@IoU0.5 = {metrics['f1']:.4f} at conf {metrics['f1_conf']:.2f} "
          f"(P {metrics['precision']:.3f}, R {metrics['recall']:.3f})")
    if fixed:
        print(f"F1 at fixed conf {fixed['conf']:.2f} = {fixed['f1']:.4f} "
              f"(P {fixed['precision']:.3f}, R {fixed['recall']:.3f})")
    print(f"Speed (incl. data loading): {ms_per_img:.1f} ms/img | params: {params_m:.1f}M")
    for c, n in names.items():
        print(f"  {n}: AP50 {metrics['ap50'][c]:.3f}")

    json_path = PROJECT_ROOT / "results" / "evaluations" / f"{name}_{args.split}.json"
    save_json({"name": name, "weights": str(weights), "split": args.split, "imgsz": imgsz, "metrics": metrics,
               "fixed_conf": ({k: v for k, v in fixed.items() if k != "per_class"} if fixed else None),
               "ms_per_img": ms_per_img, "params_m": params_m}, json_path)
    row = {"run": name, "split": args.split, "imgsz": imgsz, "params_M": params_m,
           "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
           "f1_conf": metrics["f1_conf"], "map50": metrics["map50"], "map50_95": None,
           "ms_per_img": ms_per_img, "fps": 1000 / ms_per_img}
    if fixed:
        row.update({"fixed_conf": fixed["conf"], "f1_at_fixed_conf": fixed["f1"]})
    for c, n in names.items():
        row[f"ap50_{n}"] = metrics["ap50"][c]
    print(f"Saved {json_path}\nAppended a row to {append_metrics_row(row)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="configs/rdd2022.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1, help="Use a small batch on GPUs with limited memory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. 0, cuda:0 or cpu (default: auto)")
    parser.add_argument("--lr", type=float, default=None,
                        help="base learning rate (default: 0.02 * batch / 16, the torchvision reference scaling)")
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=1, help="validate every N epochs (0 = never)")
    parser.add_argument("--fraction", type=float, default=1.0, help="train/val on this fraction of images")
    parser.add_argument("--patience", type=int, default=30, help="stop after N validations without mAP gain")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--amp", action="store_true", help="use CUDA automatic mixed precision")
    parser.add_argument("--project", default="runs/faster_rcnn")
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", default=None, help="resume from a checkpoint written by this script")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop after the epoch that would overrun this budget; continue with --resume last.pt")
    parser.add_argument("--eval-weights", default=None, help="only evaluate these weights (best.pt / model.pt)")
    parser.add_argument("--split", default="val", help="split for --eval-weights: val or test")
    parser.add_argument("--fixed-conf", type=float, default=None, help="with --eval-weights: also report F1 here")
    return parser.parse_args()


def save_checkpoint(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    if args.imgsz <= 0 or args.epochs <= 0 or args.batch <= 0 or args.workers < 0:
        raise SystemExit("--imgsz, --epochs and --batch must be positive; --workers cannot be negative")
    if args.lr is None:
        args.lr = 0.02 * args.batch / 16
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    requested_device = args.device
    if requested_device and requested_device.isdigit():
        requested_device = f"cuda:{requested_device}"
    if requested_device:
        device = torch.device(requested_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable in this PyTorch environment.")

    data_yaml = resolve_data_yaml(args.data)
    names = load_names(data_yaml)
    num_classes = len(names)
    if args.eval_weights:
        run_evaluation(args, device, data_yaml, names)
        return
    train_images, train_labels = get_split_dirs(data_yaml, "train")
    dataset = RoadDamageDataset(train_images, train_labels, num_classes, args.fraction)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", collate_fn=collate_fn,
                        persistent_workers=args.workers > 0)
    val_loader = None
    if args.val_every > 0:
        val_images, val_labels = get_split_dirs(data_yaml, "val")
        val_dataset = RoadDamageDataset(val_images, val_labels, num_classes, args.fraction)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=args.workers, collate_fn=collate_fn,
                                persistent_workers=args.workers > 0)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        model = build_model(num_classes, args.imgsz, pretrained=False)
    else:
        checkpoint = None
        model = build_model(num_classes, args.imgsz, pretrained=True)

    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # torchvision reference schedule: x0.1 at ~2/3 and ~8/9 of training, plus a short linear warmup
    milestones = [max(1, round(args.epochs * 2 / 3)), max(1, round(args.epochs * 8 / 9))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    run_name = args.name or f"fasterrcnn_resnet50fpn_{args.imgsz}_s{args.seed}"
    save_dir = (PROJECT_ROOT / args.project / run_name).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings.update({"device": str(device), "class_names": names, "train_images": len(dataset)})
    with (save_dir / "cli_args.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    start_epoch = 0
    if checkpoint is not None:
        if checkpoint["num_classes"] != num_classes or checkpoint["imgsz"] != args.imgsz:
            raise ValueError("Checkpoint class count or image size does not match this run.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
    best_map50 = float(checkpoint.get("best_map50", -1.0)) if checkpoint else -1.0
    stale = int(checkpoint.get("stale", 0)) if checkpoint else 0

    history_path = save_dir / "metrics.csv"
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "learning_rate", "precision", "recall", "f1",
                                    "f1_conf", "map50"] + [f"ap50_{n}" for n in names.values()])

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    warmup_iters = min(args.warmup_iters, len(loader)) if start_epoch == 0 else 0
    print(f"Training Faster R-CNN on {device}: {len(dataset)} images, {num_classes} classes, "
          f"batch={args.batch}, epochs={args.epochs}")

    session_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        loss_total = 0.0
        for step, (images, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            if epoch == start_epoch and step < warmup_iters:
                warmup = 0.001 + (1 - 0.001) * step / warmup_iters
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * warmup
            elif epoch == start_epoch and warmup_iters and step == warmup_iters:  # never on resume
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
            images = [image.to(device, non_blocking=device.type == "cuda") for image in images]
            targets = [{key: value.to(device, non_blocking=device.type == "cuda") for key, value in target.items()
                        if key != "size"} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.detach().item()

        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        mean_loss = loss_total / len(loader)

        metrics = None
        if val_loader is not None and ((epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs):
            del images, targets, loss_dict, loss  # free training tensors so eval fits on a 4 GB GPU
            if device.type == "cuda":
                torch.cuda.empty_cache()
            metrics = evaluate(model, val_loader, device, num_classes, args.amp)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if metrics["map50"] > best_map50:
                best_map50, stale = metrics["map50"], 0
                save_checkpoint({"model": model.state_dict(), "num_classes": num_classes, "imgsz": args.imgsz,
                                 "class_names": names, "epoch": epoch + 1, "val": metrics}, save_dir / "best.pt")
            else:
                stale += 1
        row = [epoch + 1, mean_loss, learning_rate]
        if metrics:
            row += [metrics[k] for k in ("precision", "recall", "f1", "f1_conf", "map50")] + metrics["ap50"]
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        checkpoint_data = {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "num_classes": num_classes, "imgsz": args.imgsz,
                           "class_names": names, "train_loss": mean_loss, "best_map50": best_map50, "stale": stale}
        save_checkpoint(checkpoint_data, save_dir / "last.pt")
        message = f"Epoch {epoch + 1}/{args.epochs}: train_loss={mean_loss:.4f}, lr={learning_rate:.6g}"
        if metrics:
            message += (f", val P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
                        f" mAP50={metrics['map50']:.4f} (best {best_map50:.4f})")
        print(message, flush=True)
        if val_loader is not None and stale >= args.patience:
            print(f"Early stopping: no mAP@0.5 improvement in {args.patience} validations.")
            break
        epoch_seconds = time.time() - epoch_start
        if (args.max_hours and epoch + 1 < args.epochs
                and time.time() - session_start + 1.2 * epoch_seconds > args.max_hours * 3600):
            (save_dir / "PAUSED").write_text(f"stopped after epoch {epoch + 1}\n", encoding="utf-8")
            print(f"Time budget of {args.max_hours} h reached after epoch {epoch + 1}/{args.epochs}. "
                  f"Continue with: --resume {save_dir / 'last.pt'}")
            return

    (save_dir / "PAUSED").unlink(missing_ok=True)
    save_checkpoint({"model": model.state_dict(), "num_classes": num_classes, "imgsz": args.imgsz,
                     "class_names": names}, save_dir / "model.pt")
    print(f"Training complete. Final weights: {save_dir / 'model.pt'}")
    if best_map50 >= 0:
        print(f"Best weights (val mAP@0.5 = {best_map50:.4f}): {save_dir / 'best.pt'}")
    print(f"Resumable checkpoint: {save_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
