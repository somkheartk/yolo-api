# ตัวอย่างการใช้งาน YOLO API / YOLO API Examples

โฟลเดอร์นี้รวมตัวอย่างการใช้งาน YOLO API และ YOLO model

## โครงสร้างโฟลเดอร์

```
examples/
├── training/              # ตัวอย่างการ train model
│   ├── train_basic.py          # Training พื้นฐาน
│   └── train_advanced.py       # Training แบบละเอียด
├── inference/             # ตัวอย่างการใช้งานโมเดล
│   ├── api_client_examples.py  # ตัวอย่างการเรียกใช้ API
│   └── yolo_direct_examples.py # ตัวอย่างการใช้ YOLO โดยตรง
└── datasets/              # ตัวอย่างการจัดการข้อมูล
    ├── data_template.yaml      # Template สำหรับ data.yaml
    └── convert_coco_to_yolo.py # แปลง COCO เป็น YOLO format
```

## Training Examples

### 1. Basic Training (`training/train_basic.py`)

Training YOLO model แบบพื้นฐาน:

```bash
# เตรียม data.yaml
cp examples/datasets/data_template.yaml data.yaml
# แก้ไข data.yaml ให้ตรงกับ dataset ของคุณ

# รัน training
python examples/training/train_basic.py
```

### 2. Advanced Training (`training/train_advanced.py`)

Training แบบละเอียดพร้อม parameters ที่ปรับแต่งได้:

```bash
# Training ด้วย YOLOv8n, 100 epochs
python examples/training/train_advanced.py \
    --data data.yaml \
    --model yolov8n \
    --epochs 100 \
    --batch 16

# Training ด้วย YOLOv8s และ custom parameters
python examples/training/train_advanced.py \
    --data data.yaml \
    --model yolov8s \
    --epochs 200 \
    --batch 32 \
    --lr0 0.01 \
    --optimizer AdamW \
    --name my_custom_model
```

**Parameters:**
- `--data`: Path to data.yaml
- `--model`: Model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
- `--epochs`: จำนวน epochs
- `--batch`: Batch size
- `--imgsz`: ขนาดรูปภาพ (default: 640)
- `--device`: Device (0, 1, cpu, auto)
- `--lr0`: Initial learning rate
- `--optimizer`: Optimizer (SGD, Adam, AdamW, RMSProp)
- `--name`: ชื่อของการ train

## Inference Examples

### 1. API Client Examples (`inference/api_client_examples.py`)

ตัวอย่างการเรียกใช้ YOLO API:

```bash
# เริ่ม API ก่อน
docker-compose up -d

# รัน examples
python examples/inference/api_client_examples.py
```

**Features:**
- Basic API usage (health check, model info, detection)
- Batch processing (sequential)
- Batch processing (parallel with async)
- Detailed results parsing
- Error handling with retry

### 2. Direct YOLO Examples (`inference/yolo_direct_examples.py`)

ตัวอย่างการใช้ YOLO โดยตรง (ไม่ผ่าน API):

```bash
python examples/inference/yolo_direct_examples.py
```

**Features:**
- Image detection
- Video processing
- Webcam real-time detection
- Batch processing
- Custom trained model usage
- Object tracking
- Model export
- Model validation

## Dataset Examples

### 1. Data Template (`datasets/data_template.yaml`)

Template สำหรับสร้าง data.yaml:

```bash
# Copy template
cp examples/datasets/data_template.yaml data.yaml

# แก้ไขให้ตรงกับ dataset ของคุณ
vim data.yaml
```

### 2. COCO to YOLO Converter (`datasets/convert_coco_to_yolo.py`)

แปลง COCO format เป็น YOLO format:

```bash
# แปลง train set
python examples/datasets/convert_coco_to_yolo.py \
    --input /path/to/coco \
    --output /path/to/yolo \
    --split train

# แปลง validation set
python examples/datasets/convert_coco_to_yolo.py \
    --input /path/to/coco \
    --output /path/to/yolo \
    --split val
```

## การเตรียมข้อมูล

### Dataset Structure

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
└── val/
    ├── images/
    └── labels/
```

### Label Format (YOLO)

แต่ละไฟล์ label (.txt) มีรูปแบบ:
```
<class_id> <x_center> <y_center> <width> <height>
```

ค่าทั้งหมดเป็น normalized (0-1):
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

## Quick Start Guide

### 1. เตรียม Dataset

```bash
# สร้างโครงสร้างโฟลเดอร์
mkdir -p dataset/train/{images,labels}
mkdir -p dataset/val/{images,labels}

# Copy images และ labels
# ...

# สร้าง data.yaml
cp examples/datasets/data_template.yaml dataset/data.yaml
# แก้ไข data.yaml
```

### 2. Train Model

```bash
# Basic training
python examples/training/train_basic.py

# หรือ Advanced training
python examples/training/train_advanced.py \
    --data dataset/data.yaml \
    --model yolov8n \
    --epochs 100
```

### 3. ทดสอบ Model

```bash
# ทดสอบด้วย YOLO โดยตรง
python examples/inference/yolo_direct_examples.py

# หรือ Deploy ผ่าน API
cp runs/train/yolo_custom/weights/best.pt models/my_model.pt

# แก้ไข docker-compose.yml:
# MODEL_PATH=/app/models/my_model.pt

docker-compose up --build
```

### 4. ใช้งาน API

```bash
# ทดสอบ API
python examples/inference/api_client_examples.py

# หรือใช้ curl
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test.jpg" \
  -F "conf=0.3"
```

## Tips & Best Practices

### Training Tips
1. เริ่มจาก pretrained model เสมอ
2. ใช้โมเดลเล็ก (YOLOv8n) สำหรับทดสอบ
3. ตั้ง patience เพื่อ early stopping
4. Monitor training ด้วย TensorBoard

### Dataset Tips
1. มีข้อมูลอย่างน้อย 100-200 รูปต่อ class
2. Balance classes ให้ใกล้เคียงกัน
3. ตรวจสอบ labels ให้ถูกต้อง
4. ใช้ data augmentation

### Inference Tips
1. ปรับ confidence threshold ตามการใช้งาน
2. ใช้ batch processing สำหรับหลายรูป
3. Export เป็น ONNX สำหรับ production
4. ใช้ GPU สำหรับความเร็ว

## เอกสารเพิ่มเติม

- [Development Guide](../DEVELOPMENT.md) - คู่มือการพัฒนาอย่างละเอียด
- [Deployment Guide](../DEPLOYMENT.md) - คู่มือการ deploy บน Digital Ocean
- [Main README](../README.md) - README หลัก
- [Ultralytics Docs](https://docs.ultralytics.com) - เอกสาร YOLOv8
