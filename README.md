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

### 3b. Train in the cloud (Kaggle / Colab notebooks)

`notebooks/cloud/` has one self-contained notebook per model: `train_yolov8s.ipynb`,
`train_yolov8m.ipynb`, `train_rtdetr_l.ipynb` and `train_faster_rcnn.ipynb`. Each notebook
writes the project scripts itself with `%%writefile`, so nothing needs to be cloned. It then
finds the dataset input and fixes it on the fly (class remap + block split; data that is
already fixed is used as-is). It trains, plots the curves, picks the confidence threshold
on val, reports test F1 at that threshold, and saves everything to the persistent output
folder. Batch sizes are set for a 16 GB GPU (T4 / P100) with mixed precision. The intro
cell of each notebook has step-by-step Kaggle and Colab instructions.

Cloud sessions have time limits, so training stops cleanly after `MAX_HOURS` (default
10.5 h). Run the notebook again to continue:

- **Kaggle:** add the previous version's output as an input and set `RESUME_FROM`.
- **Colab:** set `DRIVE_DIR` so runs are saved on Drive, then *Run all* again; it resumes
  automatically.

The scripts support the same thing directly. `train.py --max-hours H` writes
`weights/resume.pt` and continues with `--resume`. `train_faster_rcnn.py --max-hours H`
continues from `last.pt`. `train_faster_rcnn.py --eval-weights best.pt --split test
--fixed-conf C` evaluates Faster R-CNN like `evaluate.py` does for YOLO.

The notebooks are generated from `scripts/`. After changing a script, rebuild them:

```bash
python scripts/build_cloud_notebooks.py
```

To turn any other script into a notebook:

```bash
python scripts/py_to_ipynb.py scripts/evaluate.py --args "--weights best.pt --split test"
```

The converter writes local imports such as `utils.py` with `%%writefile`. It turns the
`__main__` block into an editable `ARGS` cell, and it honours `# %%` cell markers if the
script has them. Converted copies of the evaluation, export, ensembling and data-prep
scripts are in `notebooks/converted/`.

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
- [ ] Baseline models (training queue in progress: YOLOv8s, YOLOv8m, Faster R-CNN, RT-DETR-L)
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
