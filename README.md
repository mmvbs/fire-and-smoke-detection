# Fire and Smoke Detection

This project trains a computer vision model to detect fire and smoke in images and videos using Ultralytics YOLO. The workflow is based on the notebooks `FireDetection.ipynb` and `val.ipynb`, which cover dataset acquisition, label conversion, model training, validation, and prediction on video clips.

## Overview

The repository uses a YOLO object detection pipeline to identify `fire`

The project downloads the public `dataclusterlabs/fire-and-smoke-dataset` dataset, converts the annotation files to YOLO format, and trains a detection model on the prepared images.

## Project structure

```text
.
├── dataset/
│   ├── dataset.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
├── runs/
│   └── detect/
├── FireDetection.ipynb
├── val.ipynb
└── README.md
```

## Dataset preparation

The dataset is downloaded from Kaggle using `kagglehub`, then organized into a standard YOLO structure.

Key steps in the notebook:

- Download the dataset from Kaggle.
- Locate the images and XML annotation files.
- Split the data into `train` and `val` folders.
- Convert XML annotations to YOLO `.txt` labels.
- Map the classes as:

```yaml
names:
  0: fire
  1: smoke
```

The dataset config is defined in `dataset/dataset.yaml`:

## Training

The model is trained using Ultralytics YOLO with a pretrained `yolo11n.pt` backbone.

Example command used in the notebook:

```bash
yolo task=detect mode=train model=yolo11n.pt data=dataset/dataset.yaml epochs=300 imgsz=960 patience=15
```

The project includes several training runs under `detect/`

These weights were used for validation and video inference experiments.

## Validation and evaluation

The notebook validates detection quality using YOLO's built-in evaluation pipeline:

```python
from ultralytics import YOLO

model = YOLO("detect/train-5/weights/best.pt")
metrics = model.val(data="dataset/dataset.yaml", imgsz=960)
```

The validation notebooks also compare different trained checkpoints

This allows quick assessment of which model performs best on the validation set.

## Inference on videos

The validation notebook loads a trained model and runs object detection on test videos.

## Environment and dependencies

The notebooks install the required dependencies for YOLO training and inference. The main package used is:

```bash
pip install ultralytics
```

Optional or additional dependencies used during experimentation include:

```bash
pip install kagglehub
pip install opencv-python
```