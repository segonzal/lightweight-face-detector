# Lightweight Face Detector

A lightweight, production-ready face detector built on **MobileNetV2 + SSH** detection heads. 
Detects faces and 5 facial keypoints (eyes, nose, mouth corners) in real time.
Optimized for deployment via pruning, INT8 quantization, and ONNX export.

Built as a reusable module for personal Computer Vision projects.
```
lightweight-face-detector/
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   └── download.sh
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── transforms.py
│   └── export.py
├── optimize/
│   ├── prune.py
│   └── quantize.py
├── inference/
│   └── detector.py
├── configs/
│   └── base.yaml
├── README.md
└── .gitignore
```
