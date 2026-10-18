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

Test split (5,820 images, location-blocked split). The confidence threshold is tuned on val
and F1 is reported on test at exactly that threshold; IoU 0.5 for F1/mAP@0.5. Seed 42, 640 px,
30 epochs, COCO-pretrained.

| Model | Trained on | Conf (val-tuned) | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | FPS |
|-------|------------|-----------------:|----------:|-------:|---:|--------:|-------------:|----:|
| YOLOv8s | local GTX 1650 Ti (batch 8, FP32) | 0.20 | 0.604 | 0.536 | 0.568 | 0.566 | 0.276 | 57 (GTX 1650 Ti) |
| YOLOv8m | Kaggle T4 (batch 16, AMP) | 0.25 | 0.647 | 0.542 | **0.590** | **0.599** | **0.303** | 42 (T4) |
| RT-DETR-L | Kaggle T4 (batch 8, AMP) | training (~58 min/epoch on T4) | | | | | | |
| Faster R-CNN | Kaggle T4 (batch 8, AMP) | queued | | | | | | |

Per-class AP@0.5 on test:

| Model | D00 longitudinal | D10 transverse | D20 alligator | D40 pothole |
|-------|-----------------:|---------------:|--------------:|------------:|
| YOLOv8s | 0.563 | 0.559 | 0.666 | 0.475 |
| YOLOv8m | 0.592 | 0.603 | 0.692 | 0.509 |

Potholes (D40) are the hardest class for both models; alligator cracks (D20) the easiest.
FPS includes pre/post-processing and was measured on different GPUs, so it is not directly
comparable between rows.

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

For the original RDD2022 download, convert the VOC annotations and create the
location-blocked train/validation/test split:

```bash
python scripts/convert_voc_to_yolo.py --src data/RDD2022 --dst data/yolo
python scripts/make_split.py --root data/yolo
```

If your data is already prepared as `data/{train,val,test}/{images,labels}` (for
example the Kaggle YOLO export), the included `configs/rdd2022.yaml` points to that
layout directly. Run the fix-up script once first:

```bash
python scripts/fix_prepared_data.py --root data
```

That export has two problems. It uses 5 class ids where id 3 is a merged "other
corruption" class and **id 4 is the pothole (D40)**. The per-class box totals match
the published RDD2022 counts for D00/D10/D20/D40. It is also split randomly per image,
so neighbouring frames leak between train and val/test. The script drops "other
corruption", maps 4 → 3, and moves whole 50-frame blocks per country into
train/val/test (~70/15/15). It backs up the original labels and split to
`data/original_labels.zip` and writes `data/splits.csv`.

| Split | Images | D00 | D10 | D20 | D40 |
|-------|-------:|----:|----:|----:|----:|
| train | 26,822 | 17,913 | 8,170 | 7,433 | 4,565 |
| val   |  5,743 |  4,142 | 1,801 | 1,593 |   995 |
| test  |  5,820 |  3,961 | 1,859 | 1,591 |   984 |

### 3. Train

Use the same seed, image size, and epoch count for comparable runs. Before a GPU run,
check that `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
On an older NVIDIA driver (e.g. 516, CUDA 11.7) install the cu118 build with Python 3.11:

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

Train all four baselines in one resumable queue: YOLOv8s, YOLOv8m, Faster R-CNN,
then RT-DETR-L. Re-running the command skips finished runs and resumes interrupted
ones from `last.pt`. Logs go to `runs/logs/`.

```bash
python scripts/train_all.py                  # 30 epochs each (default)
python scripts/train_all.py --epochs 100     # longer budget
```

Or one model at a time (batch sizes fit a 4 GB GPU at 640 px in FP32):

```bash
python scripts/train.py --model weights/yolov8s.pt --imgsz 640 --epochs 30 --batch 8 --workers 2 --device 0 --name yolov8s_640_s42
python scripts/train.py --model weights/yolov8m.pt --imgsz 640 --epochs 30 --batch 4 --workers 2 --device 0 --name yolov8m_640_s42
python scripts/train_faster_rcnn.py --imgsz 640 --epochs 30 --batch 2 --workers 2 --device 0 --name faster_rcnn_640_s42
python scripts/train.py --model weights/rtdetr-l.pt --imgsz 640 --epochs 30 --batch 2 --workers 2 --device 0 --name rtdetr_l_640_s42
```

Notes for a 4 GB laptop GPU (GTX 1650 Ti):

- **FP32 only.** Ultralytics' AMP check fails on GTX 16-series cards and disables it,
  so Faster R-CNN is also run without `--amp`.
- **Keep `--workers` at 2.** On Windows each dataloader process loads its own copy of
  torch. With more workers, or with a full system drive (the page file cannot grow),
  host RAM runs out.
- **Measured time per epoch** (train + val): YOLOv8s ~27 min, YOLOv8m ~80 min,
  Faster R-CNN ~110 min, RT-DETR-L ~150 min. At 30 epochs the whole queue takes about
  8 days of GPU time.

YOLO and RT-DETR checkpoints are saved under `runs/detect/<name>/weights/`. Faster R-CNN
writes `best.pt` (best val mAP@0.5), `last.pt` (resumable) and `model.pt` under
`runs/faster_rcnn/<name>/`, and logs per-epoch val P / R / F1 / mAP@0.5 to `metrics.csv`.

### 3b. Train in the cloud: Kaggle (recommended) or Google Colab

`notebooks/colab/` has one ready-to-run notebook per model: `train_yolov8s.ipynb`,
`train_yolov8m.ipynb`, `train_rtdetr_l.ipynb` and `train_faster_rcnn.ipynb`. Each one is
self-contained. All of its code (data preparation, metrics, training, evaluation) is in its own
cells, so no `.py` files are needed. It detects whether it runs on Kaggle or Colab.

**Kaggle** gives 30 free GPU hours per week and runs committed notebooks in the background:

1. *Create → New Notebook*, then *File → Import Notebook* (GitHub URL or upload).
2. In *Settings*, choose *Accelerator* **GPU T4 x2** (not P100, which current PyTorch no longer
   supports) and turn **Internet on**. Internet needs a phone-verified account.
3. Optional: *Add Input* → `aliabdelmenam/rdd-2022`. Otherwise the notebook downloads it.
4. *Save Version → Save & Run All (Commit)*. Outputs go to the version's *Output* tab.
5. A run longer than one 12 h session stops cleanly at `MAX_HOURS` (10.5 h) and prints
   "Paused". To continue, add the previous version's output as an input (*Add Input → Your Work
   → Notebook Output*) and commit again. The checkpoint is picked up automatically.

**Colab:** pick a T4 runtime and *Run all*. Checkpoints go to Google Drive (`DRIVE_DIR`), and
*Run all* again after a disconnect continues the run. The free Colab GPU allowance is only a few
hours per day.

Either way, the notebook fixes the data on the fly. Class 4 (pothole) is remapped to D40,
"other corruption" is dropped, and images are re-split into 50-frame blocks with seed 42, the
same split as the local runs. When training is done, the notebook tunes the confidence threshold
on val, reports test F1 at that threshold, and appends both rows to `results/metrics.csv`.

Approximate times on a T4, 30 epochs (YOLOv8s measured on Colab, the others estimated):

| Model | Per epoch | 30 epochs |
|-------|-----------|-----------|
| YOLOv8s | ~12 min | ~6 h |
| YOLOv8m | ~22 min | ~11 h |
| RT-DETR-L | ~32 min | ~16 h (2 sessions) |
| Faster R-CNN | ~40 min | ~20 h (2 sessions) |

To turn other scripts into notebooks, use `python scripts/py_to_ipynb.py scripts/evaluate.py`.
Converted copies of the evaluation, export, ensembling and data-prep scripts are in
`notebooks/converted/`.

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
- [x] Data download, conversion and EDA (class remap + leak-free block split)
- [ ] Baseline models (YOLOv8s and YOLOv8m done; RT-DETR-L training on Kaggle; Faster R-CNN next)
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
