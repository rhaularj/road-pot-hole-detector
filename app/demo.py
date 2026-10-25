#!/usr/bin/env python
"""Road damage detection demo: upload a road image, get potholes and cracks back.

Two tabs:
  Detect          the best baseline, RT-DETR-L trained on RDD2022 (test F1 0.614 at conf 0.50,
                  mAP@0.5 0.613). The default threshold is the one tuned on the validation split.
  Compare models  RT-DETR-L, YOLOv8m and Faster R-CNN side by side on the same image, each at
                  its own val-tuned threshold (or one shared threshold), with inference times and,
                  for the example images, how many ground-truth boxes each model found.

Examples:
    python app/demo.py                                    # http://127.0.0.1:7860
    python app/demo.py --weights path/to/best.pt --device cpu
    python app/demo.py --share                            # temporary public link

RT-DETR-L weights are looked up in this order: --weights, the ROAD_DAMAGE_WEIGHTS environment
variable, weights/rtdetr_l_rdd2022.pt, runs/detect/rtdetr_l_640_s42/weights/best.pt. The compare
tab also loads weights/yolov8m_rdd2022.pt and weights/faster_rcnn_rdd2022.pt when present.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_WEIGHTS = [WEIGHTS_DIR / "rtdetr_l_rdd2022.pt",
                   PROJECT_ROOT / "runs" / "detect" / "rtdetr_l_640_s42" / "weights" / "best.pt"]
DEFAULT_CONF = 0.50  # tuned on val for RT-DETR-L

# class id -> (code, readable name, box colour); ids follow configs/rdd2022.yaml
CLASSES = {
    0: ("D00", "Longitudinal crack", (255, 179, 0)),
    1: ("D10", "Transverse crack", (0, 176, 255)),
    2: ("D20", "Alligator crack", (170, 0, 255)),
    3: ("D40", "Pothole", (255, 23, 68)),
}
POTHOLE = 3
CHOICES = [f"{name} ({code})" for code, name, _ in CLASSES.values()]

# Test-split results from the README; conf is the threshold tuned on val
MODELS = {
    "RT-DETR-L": {"file": "rtdetr_l_rdd2022.pt", "conf": 0.50, "f1": 0.614, "map50": 0.613, "ap_pothole": 0.540},
    "YOLOv8m": {"file": "yolov8m_rdd2022.pt", "conf": 0.25, "f1": 0.590, "map50": 0.599, "ap_pothole": 0.509},
    "Faster R-CNN": {"file": "faster_rcnn_rdd2022.pt", "conf": 0.75, "f1": 0.530, "map50": 0.488,
                     "ap_pothole": 0.393},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=os.environ.get("ROAD_DAMAGE_WEIGHTS"))
    p.add_argument("--device", default=None, help="e.g. 0 or cpu (default: GPU if available)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    return p.parse_args()


def find_weights(weights: str | None) -> Path:
    candidates = [Path(weights)] if weights else DEFAULT_WEIGHTS
    for c in candidates:
        for p in (c, PROJECT_ROOT / c):
            if p.is_file():
                return p
    raise SystemExit("Trained RT-DETR-L weights not found. Looked for:\n  "
                     + "\n  ".join(str(c) for c in candidates)
                     + "\nDownload best.pt from the Kaggle notebook output (runs/rtdetr_l_640_s42/weights/) "
                       "and save it as weights/rtdetr_l_rdd2022.pt, or pass --weights.")


def resolve_device(device: str | None) -> str:
    """Device string in Ultralytics form: 'cpu' or a GPU index such as '0'."""
    if device is None:
        return "0" if torch.cuda.is_available() else "cpu"
    return device


def make_detection(class_id: int, confidence: float, box) -> dict:
    return {"class_id": class_id, "code": CLASSES[class_id][0], "name": CLASSES[class_id][1],
            "confidence": confidence, "box": [round(float(v)) for v in box]}


class UltralyticsDetector:
    """RT-DETR or YOLO checkpoint trained on the 4 RDD2022 classes."""

    def __init__(self, weights: Path, device: str, imgsz: int = 640):
        from ultralytics import RTDETR, YOLO

        self.model = RTDETR(str(weights)) if "rtdetr" in weights.name.lower() else YOLO(str(weights))
        names = self.model.names
        if [names[i] for i in sorted(names)] != [code for code, _, _ in CLASSES.values()]:
            raise SystemExit(f"{weights} was not trained on the 4 RDD2022 classes (its classes: {names}). "
                             "Use the fine-tuned best.pt, not a COCO-pretrained checkpoint.")
        self.device, self.imgsz = device, imgsz

    def __call__(self, image: Image.Image, conf: float, class_ids: list[int]) -> list[dict]:
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])  # Ultralytics expects BGR arrays
        r = self.model.predict(bgr, conf=conf, imgsz=self.imgsz, classes=class_ids, device=self.device,
                               verbose=False)[0]
        b = r.boxes
        return [make_detection(int(c), float(s), xyxy)
                for xyxy, s, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())]


class FasterRCNNDetector:
    """torchvision Faster R-CNN ResNet-50 FPN as trained by train_faster_rcnn (script or notebook)."""

    def __init__(self, weights: Path, device: str, imgsz: int = 640):
        from torchvision.models import resnet50
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
        from torchvision.ops.misc import FrozenBatchNorm2d

        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        if checkpoint.get("num_classes", len(CLASSES)) != len(CLASSES):
            raise SystemExit(f"{weights} has {checkpoint['num_classes']} classes, expected {len(CLASSES)}")
        imgsz = checkpoint.get("imgsz", imgsz)
        # same layout as build_model(pretrained=False) in scripts/train_faster_rcnn.py
        backbone = _resnet_fpn_extractor(resnet50(weights=None, norm_layer=FrozenBatchNorm2d), trainable_layers=3)
        self.model = FasterRCNN(backbone, num_classes=len(CLASSES) + 1, min_size=imgsz, max_size=imgsz)
        self.model.load_state_dict(checkpoint["model"])
        self.device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image: Image.Image, conf: float, class_ids: list[int]) -> list[dict]:
        from torchvision.transforms.functional import pil_to_tensor

        out = self.model([pil_to_tensor(image).float().div_(255).to(self.device)])[0]
        detections = []
        for box, score, label in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(),
                                     out["labels"].cpu().numpy()):
            class_id = int(label) - 1  # label 0 is background
            if score >= conf and class_id in class_ids:
                detections.append(make_detection(class_id, float(score), box))
        return detections


def load_detectors(main_weights: Path, device: str, imgsz: int) -> dict:
    """RT-DETR-L always; YOLOv8m and Faster R-CNN when their weights are in weights/."""
    detectors = {"RT-DETR-L": UltralyticsDetector(main_weights, device, imgsz)}
    for name, cls in (("YOLOv8m", UltralyticsDetector), ("Faster R-CNN", FasterRCNNDetector)):
        path = WEIGHTS_DIR / MODELS[name]["file"]
        if path.is_file():
            print(f"Loading {path}")
            detectors[name] = cls(path, device, imgsz)
        else:
            print(f"Compare tab: {name} skipped, {path} not found")
    warmup = Image.new("RGB", (imgsz, imgsz), (128, 128, 128))
    for detector in detectors.values():  # the first call is slow (CUDA init), keep it out of the timings
        detector(warmup, 0.5, list(CLASSES))
    return detectors


def detect(detector, image: Image.Image, conf: float, class_ids: list[int]) -> tuple[list[dict], float]:
    """Run one model on an image. Returns the detections (highest confidence first) and the time in ms."""
    if not class_ids:
        return [], 0.0
    image = image.convert("RGB")
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    detections = detector(image, conf, class_ids)
    if cuda:
        torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - start)
    return sorted(detections, key=lambda d: -d["confidence"]), ms


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: list[dict], show_conf: bool = True) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    scale = max(out.size) / 640
    width = max(2, round(3 * scale))
    font = _font(max(12, round(16 * scale)))
    # potholes last, so they are drawn on top of overlapping cracks
    for d in sorted(detections, key=lambda d: d["class_id"] == POTHOLE):
        colour = CLASSES[d["class_id"]][2]
        x1, y1, x2, y2 = d["box"]
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=width)
        label = f"{d['name']} {d['confidence']:.2f}" if show_conf else d["name"]
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), label, font=font)
        tw, th = tx2 - tx1, ty2 - ty1
        ty = y1 - th - 2 * width if y1 - th - 2 * width >= 0 else y1
        draw.rectangle((x1, ty, x1 + tw + 2 * width, ty + th + 2 * width), fill=colour)
        draw.text((x1 + width, ty + width - ty1), label, fill=(255, 255, 255), font=font)
    return out


def damage_counts(detections: list[dict]) -> tuple[int, str]:
    """Number of potholes and a short text listing the other damage types."""
    n_pot = sum(d["class_id"] == POTHOLE for d in detections)
    others = [f"{n} × {name}" for i, (_, name, _) in CLASSES.items()
              if i != POTHOLE and (n := sum(d["class_id"] == i for d in detections))]
    return n_pot, ", ".join(others)


def summarize(detections: list[dict]) -> str:
    if not detections:
        return "### No road damage found\nTry a lower confidence threshold if you expected some."
    n_pot, others = damage_counts(detections)
    head = (f"### ⚠️ {n_pot} pothole{'s' if n_pot != 1 else ''} detected" if n_pot
            else "### No potholes detected")
    return head + (f"\n\nOther damage: {others}" if others else "")


def load_ground_truth(image_path: str, size: tuple[int, int]) -> list[dict] | None:
    """Ground-truth boxes for the bundled example images (YOLO labels next to the image), else None."""
    label = EXAMPLES_DIR / f"{Path(image_path).stem}.txt"
    if not label.is_file():
        return None
    w, h = size
    boxes = []
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
        boxes.append(make_detection(c, 1.0, ((cx - bw / 2) * w, (cy - bh / 2) * h,
                                             (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return boxes


def _iou(a, b) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def count_matches(detections: list[dict], truth: list[dict]) -> tuple[int, int]:
    """Greedy same-class matching at IoU >= 0.5, highest confidence first (as in the evaluation).
    Returns (matched boxes, matched potholes)."""
    used, matched, matched_pot = set(), 0, 0
    for d in sorted(detections, key=lambda d: -d["confidence"]):
        best, best_iou = None, 0.5
        for j, t in enumerate(truth):
            if j not in used and t["class_id"] == d["class_id"] and (iou := _iou(d["box"], t["box"])) >= best_iou:
                best, best_iou = j, iou
        if best is not None:
            used.add(best)
            matched += 1
            matched_pot += d["class_id"] == POTHOLE
    return matched, matched_pot


def build_ui(detectors: dict) -> gr.Blocks:
    main = detectors["RT-DETR-L"]
    compared = list(detectors)

    def run(image, conf, selected):
        if image is None:
            return None, "", []
        detections, _ = detect(main, image, conf, [CHOICES.index(s) for s in selected])
        rows = [[i + 1, d["name"], d["code"], round(d["confidence"], 3), *d["box"]]
                for i, d in enumerate(detections)]
        return draw_detections(image, detections), summarize(detections), rows

    def compare(image_path, use_tuned, shared_conf, selected):
        if image_path is None:
            return [None] * len(compared) + [gr.update(value=None, visible=False), [], ""]
        image = Image.open(image_path).convert("RGB")
        class_ids = [CHOICES.index(s) for s in selected]
        truth = load_ground_truth(image_path, image.size)
        if truth is not None:
            truth = [t for t in truth if t["class_id"] in class_ids]
        # an idle GPU clocks down and the first model run afterwards takes ~200 ms longer, so wake it
        # with an untimed pass first; otherwise whichever model runs first looks slowest
        detect(detectors[compared[0]], image, 0.5, list(CLASSES))
        images, rows = [], []
        for name in compared:
            conf = MODELS[name]["conf"] if use_tuned else shared_conf
            detections, ms = detect(detectors[name], image, conf, class_ids)
            images.append(draw_detections(image, detections))
            n_pot, others = damage_counts(detections)
            row = [name, conf, n_pot, others or "–"]
            if truth is not None:
                matched, matched_pot = count_matches(detections, truth)
                n_truth_pot = sum(t["class_id"] == POTHOLE for t in truth)
                row += [f"{matched} / {len(truth)}", f"{matched_pot} / {n_truth_pot}", len(detections) - matched]
            else:
                row += ["–", "–", "–"]
            info = MODELS[name]
            rows.append(row + [round(ms), info["f1"], info["map50"], info["ap_pothole"]])
        gt_image = (gr.update(value=draw_detections(image, truth, show_conf=False), visible=True)
                    if truth is not None else gr.update(value=None, visible=False))
        note = ("Ground truth is available for this example image: *Found* counts detections that match a "
                "labelled box of the same class at IoU ≥ 0.5; *False alarms* are the rest."
                if truth is not None else
                "No ground truth for an uploaded image, so only the detections are compared. "
                "Pick an example image to also see how many labelled boxes each model found.")
        return images + [gt_image, rows, note]

    with gr.Blocks(title="Road Damage Detector") as demo:
        gr.Markdown("# Road Damage Detector\n"
                    "Find **potholes** and cracks in road photos with detectors trained on RDD2022.")
        examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg"))

        with gr.Tab("Detect"):
            gr.Markdown("Model: RT-DETR-L, the best of the baselines (test F1 0.614, mAP 0.613 at IoU 0.5).")
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(type="pil", label="Road image", sources=["upload", "clipboard", "webcam"])
                    conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence threshold",
                                     info="0.50 was tuned on the validation split")
                    classes = gr.CheckboxGroup(CHOICES, value=CHOICES, label="Damage types to show")
                    btn = gr.Button("Detect", variant="primary")
                with gr.Column():
                    out_img = gr.Image(type="pil", label="Detections", format="png")
                    summary = gr.Markdown()
            table = gr.Dataframe(headers=["#", "Damage", "Code", "Confidence", "x1", "y1", "x2", "y2"],
                                 label="Detections (box in pixels)", interactive=False)
            if examples:
                gr.Examples(examples, inputs=inp, label="Example images from the RDD2022 test split")
            # re-run on a new image (upload, webcam, example) and whenever the threshold or classes change
            args = ([inp, conf, classes], [out_img, summary, table])
            btn.click(run, *args)
            inp.change(run, *args)
            conf.release(run, *args)
            classes.change(run, *args)

        with gr.Tab("Compare models"):
            missing = [n for n in MODELS if n not in detectors]
            gr.Markdown("Runs " + ", ".join(compared) + " on the same image. By default each model uses its "
                        "own threshold tuned on the validation split. The test-split metrics come from the "
                        "evaluation on 5,820 images; one image only shows how the models behave. The example "
                        "images were chosen because RT-DETR-L finds their potholes, so they favour it."
                        + (f"\n\n*Not loaded (weights missing in weights/): {', '.join(missing)}.*"
                           if missing else ""))
            with gr.Row():
                with gr.Column(scale=1):
                    cmp_inp = gr.Image(type="filepath", label="Road image",
                                       sources=["upload", "clipboard", "webcam"])
                with gr.Column(scale=1):
                    use_tuned = gr.Checkbox(value=True, label="Use each model's tuned threshold",
                                            info=", ".join(f"{n} {MODELS[n]['conf']:.2f}" for n in compared))
                    shared_conf = gr.Slider(0.05, 0.95, value=0.5, step=0.05,
                                            label="Shared threshold (when the box above is unticked)")
                    cmp_classes = gr.CheckboxGroup(CHOICES, value=CHOICES, label="Damage types to show")
                    cmp_btn = gr.Button("Compare", variant="primary")
            with gr.Row():
                cmp_imgs = [gr.Image(type="pil", label=name, format="png") for name in compared]
                gt_img = gr.Image(type="pil", label="Ground truth", format="png", visible=False)
            cmp_table = gr.Dataframe(
                headers=["Model", "Threshold", "Potholes", "Other damage", "Found (all)", "Found (potholes)",
                         "False alarms", "Time (ms)", "Test F1", "Test mAP@0.5", "Test pothole AP@0.5"],
                label="Comparison on this image, plus each model's test-split metrics", interactive=False)
            cmp_note = gr.Markdown()
            if examples:
                gr.Examples(examples, inputs=cmp_inp, label="Example images from the RDD2022 test split")
            cmp_args = ([cmp_inp, use_tuned, shared_conf, cmp_classes], cmp_imgs + [gt_img, cmp_table, cmp_note])
            cmp_btn.click(compare, *cmp_args)
            cmp_inp.change(compare, *cmp_args)
            use_tuned.change(compare, *cmp_args)
            shared_conf.release(compare, *cmp_args)
            cmp_classes.change(compare, *cmp_args)
    return demo


def main():
    args = parse_args()
    weights = find_weights(args.weights)
    device = resolve_device(args.device)
    print(f"Loading {weights}")
    detectors = load_detectors(weights, device, args.imgsz)
    build_ui(detectors).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
