#!/usr/bin/env python
"""Road damage detection demo: upload a road image, get potholes and cracks back.

Uses the best baseline, RT-DETR-L trained on RDD2022 (test F1 0.614 at conf 0.50,
mAP@0.5 0.613). The default threshold is the one tuned on the validation split.

Examples:
    python app/demo.py                                    # http://127.0.0.1:7860
    python app/demo.py --weights path/to/best.pt --device cpu
    python app/demo.py --share                            # temporary public link

Weights are looked up in this order: --weights, the ROAD_DAMAGE_WEIGHTS environment
variable, weights/rtdetr_l_rdd2022.pt, runs/detect/rtdetr_l_640_s42/weights/best.pt.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
DEFAULT_WEIGHTS = [PROJECT_ROOT / "weights" / "rtdetr_l_rdd2022.pt",
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


def load_detector(weights: Path):
    from ultralytics import RTDETR

    model = RTDETR(str(weights))
    names = model.names
    if [names[i] for i in sorted(names)] != [code for code, _, _ in CLASSES.values()]:
        raise SystemExit(f"{weights} was not trained on the 4 RDD2022 classes (its classes: {names}). "
                         "Use the fine-tuned best.pt, not the COCO-pretrained rtdetr-l.pt.")
    return model


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
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
        label = f"{d['name']} {d['confidence']:.2f}"
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), label, font=font)
        tw, th = tx2 - tx1, ty2 - ty1
        ty = y1 - th - 2 * width if y1 - th - 2 * width >= 0 else y1
        draw.rectangle((x1, ty, x1 + tw + 2 * width, ty + th + 2 * width), fill=colour)
        draw.text((x1 + width, ty + width - ty1), label, fill=(255, 255, 255), font=font)
    return out


def detect(model, image: Image.Image, conf: float, class_ids: list[int], imgsz: int = 640,
           device=None) -> list[dict]:
    """Run the model on one image and return the kept detections, highest confidence first."""
    if not class_ids:
        return []
    bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])  # Ultralytics expects BGR arrays
    r = model.predict(bgr, conf=conf, imgsz=imgsz, classes=class_ids, device=device, verbose=False)[0]
    b = r.boxes
    detections = [{"class_id": int(c), "code": CLASSES[int(c)][0], "name": CLASSES[int(c)][1],
                   "confidence": float(s), "box": [round(float(v)) for v in xyxy]}
                  for xyxy, s, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())]
    return sorted(detections, key=lambda d: -d["confidence"])


def summarize(detections: list[dict]) -> str:
    if not detections:
        return "### No road damage found\nTry a lower confidence threshold if you expected some."
    n_pot = sum(d["class_id"] == POTHOLE for d in detections)
    head = (f"### ⚠️ {n_pot} pothole{'s' if n_pot != 1 else ''} detected" if n_pot
            else "### No potholes detected")
    others = [f"{n} × {name}" for i, (_, name, _) in CLASSES.items()
              if i != POTHOLE and (n := sum(d["class_id"] == i for d in detections))]
    return head + ("\n\nOther damage: " + ", ".join(others) if others else "")


def build_ui(model, imgsz: int, device) -> gr.Blocks:
    def run(image, conf, selected):
        if image is None:
            return None, "", []
        class_ids = [CHOICES.index(s) for s in selected]
        detections = detect(model, image, conf, class_ids, imgsz, device)
        rows = [[i + 1, d["name"], d["code"], round(d["confidence"], 3), *d["box"]]
                for i, d in enumerate(detections)]
        return draw_detections(image, detections), summarize(detections), rows

    with gr.Blocks(title="Road Damage Detector") as demo:
        gr.Markdown("# Road Damage Detector\n"
                    "Upload a road photo to find **potholes** and cracks. Model: RT-DETR-L trained on "
                    "RDD2022 (test F1 0.614, mAP 0.613 at IoU 0.5).")
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
        examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg"))
        if examples:
            gr.Examples(examples, inputs=inp, label="Example images from the RDD2022 test split")
        # re-run on a new image (upload, webcam, example) and whenever the threshold or classes change
        args = ([inp, conf, classes], [out_img, summary, table])
        btn.click(run, *args)
        inp.change(run, *args)
        conf.release(run, *args)
        classes.change(run, *args)
    return demo


def main():
    args = parse_args()
    weights = find_weights(args.weights)
    print(f"Loading {weights}")
    model = load_detector(weights)
    build_ui(model, args.imgsz, args.device).launch(server_name=args.host, server_port=args.port,
                                                    share=args.share)


if __name__ == "__main__":
    main()
