# Camera–LiDAR Traffic Perception Pipeline

End-to-end multi-sensor perception for mobile robotics: detect traffic cones in camera images, project LiDAR points into image space to estimate range, extract traffic sign crops, and classify signs with a trained ResNet18 model.

## Overview

| Stage | Module | Description |
|-------|--------|-------------|
| Cone detection | `ass2/detection/detect_cone.py` | Red HSV masking + contour geometry filters |
| Sensor fusion | `ass2/detection/detect_distance.py` | Calibrated LiDAR→camera projection, per-cone distance stats |
| Sign extraction | `ass2/detection/extract_sign.py` | Two-phase crop: cone ROI → sign plate detection |
| Classification | `ass2/classification/` | ResNet18 inference on extracted sign crops |
| Orchestration | `ass2/pipeline.py` | Full batch pipeline with ground-truth evaluation |

## Requirements

- Python 3.11 or 3.12 (Open3D has no Windows wheels for 3.13 yet)
- See `ass2/requirements.txt`

```bash
cd ass2
pip install -r requirements.txt
```

## Quick Start

```bash
cd ass2
python pipeline.py
```

Place synchronized image/LiDAR pairs in `traffic_cone_images/` and `traffic_cone_lidar/` (matching filenames). Calibration data lives in `calibration.json`.

## Traffic Sign Classifier Training

The ResNet18 classifier is trained separately from the GTSRB subset. See [Traffic-Sign-Student/Traffic-Sign/README.md](Traffic-Sign-Student/Traffic-Sign/README.md) for dataset download and training instructions.

After training, copy `checkpoint/best_model.pth` so `ass2/classification/model.py` can load it at inference time.

## Repository Structure

```
camera-lidar-traffic-perception/
├── ass2/                          # Main perception pipeline
│   ├── pipeline.py
│   ├── detection/                 # Cone detection, LiDAR fusion, sign extraction
│   ├── classification/            # ResNet18 model wrapper
│   ├── calibration.json
│   └── requirements.txt
└── Traffic-Sign-Student/Traffic-Sign/   # Classifier training code
```

## Tech Stack

Python, OpenCV, NumPy, Open3D, PyTorch, TorchVision, ResNet18
