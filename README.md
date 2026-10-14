# Road Damage Detection

Detecting potholes and cracks in road images using deep learning object detection (YOLO), trained and evaluated on the RDD2022 benchmark dataset.

> Project for the ML Workshop. Status: work in progress.

---

## Overview

Road surface damage is costly and dangerous, and manual inspection does not scale. This project builds an object detection pipeline that finds and classifies road damage in images, and studies which training and inference choices actually improve results.

**Main goals**
- Train and compare several detectors on a road damage dataset
- Run controlled ablations (image size, augmentation, loss, attention, etc.)
- Evaluate honestly, with per-class metrics and error analysis
- Provide a simple demo for inference on images and video

## Damage classes

| Code | Class |
|------|-------|
| D00 | Longitudinal crack |
| D10 | Transverse crack |
| D20 | Alligator crack |
| D40 | Pothole |

## Dataset

- **RDD2022**: ~47k road images from 6 countries (Japan, India, Czech Republic, Norway, USA, China)
- Annotations converted from Pascal VOC XML to YOLO format
- Train/validation split done **by location/sequence**, not randomly, to avoid frame leakage

Download the dataset from the official RDD2022 source and place it under `data/`. The dataset is not included in this repository.

## Models

| Model | Role |
|-------|------|
| Faster R-CNN (ResNet-50 FPN) | Two-stage baseline |
| YOLOv8n / YOLOv8s | Fast baseline |
| YOLOv8m / YOLO11m | Stronger candidate |
| RT-DETR | Transformer-based comparison |

## Metrics

- F1 @ IoU 0.5 (standard for RDD challenges)
- mAP@0.5 and mAP@0.5:0.95
- Per-class precision, recall and AP
- Inference speed (FPS / latency) and model size
- Confusion matrix

## Results

Results will be filled in as experiments finish.

| Model | Img size | Precision | Recall | F1 | mAP@0.5 | FPS |
|-------|----------|-----------|--------|----|---------|-----|
| Faster R-CNN | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8s | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8m | TBD | TBD | TBD | TBD | TBD | TBD |
| RT-DETR | TBD | TBD | TBD | TBD | TBD | TBD |

## Project structure

```
road-pot-hole-detector/
├── data/                # dataset (not tracked in git)
├── notebooks/           # EDA and experiments
├── scripts/             # data conversion, training, evaluation
├── configs/             # dataset and training configs
├── runs/                # training outputs (not tracked in git)
├── app/                 # demo app (Gradio / Streamlit)
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/rhaularj/road-pot-hole-detector.git
cd road-pot-hole-detector
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the data

```bash
python scripts/convert_voc_to_yolo.py --src data/RDD2022 --dst data/yolo
```

### 3. Train

```bash
yolo detect train data=configs/rdd2022.yaml model=yolov8s.pt imgsz=640 epochs=100 seed=42
```

### 4. Evaluate

```bash
yolo detect val data=configs/rdd2022.yaml model=runs/detect/train/weights/best.pt
```

### 5. Run inference

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/images
```

## Experiments planned

- **Data:** image size (640 / 800 / 1024), augmentation, class imbalance handling, multi-country vs. single-country training
- **Model:** backbone size, attention modules (CBAM, SE), loss functions (CIoU, SIoU, WIoU)
- **Training:** hyperparameter tuning, learning-rate schedule, longer training
- **Inference:** test-time augmentation (TTA), Weighted Boxes Fusion (WBF) ensembling, per-class threshold tuning
- **Analysis:** error analysis, cross-country generalization, Grad-CAM explainability, multiple seeds (mean ± std)

## Possible extensions

- Severity estimation from crack segmentation
- GPS-tagged damage map
- Edge deployment (ONNX / TensorRT / TFLite)
- Video detection with tracking (ByteTrack)
- Demo web app for image and video upload

## Roadmap

- [ ] Literature review
- [ ] Data download, conversion and EDA
- [ ] Baseline models
- [ ] Ablation experiments
- [ ] Ensembling and TTA
- [ ] Error analysis
- [ ] Demo app
- [ ] Final report and slides

## References

- Maeda et al., road damage detection with deep neural networks (RDD2018)
- Arya et al., RDD2022 dataset and the Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)
- Ultralytics YOLO documentation

## Acknowledgements

Thanks to the RDD dataset authors and the open-source community behind Ultralytics, PyTorch and related tools.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Author

**kabilanpichaipilaai** **rhaularj** 
