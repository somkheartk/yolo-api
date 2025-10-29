# คู่มือการพัฒนา YOLO Object Detection API

คู่มือนี้จะอธิบายอย่างละเอียดเกี่ยวกับการพัฒนา การ train model และการใช้งาน YOLO API

## สารบัญ

1. [ภาพรวมของระบบ](#ภาพรวมของระบบ)
2. [การติดตั้งสภาพแวดล้อมสำหรับพัฒนา](#การติดตั้งสภาพแวดล้อมสำหรับพัฒนา)
3. [การเตรียมข้อมูลสำหรับ Training](#การเตรียมข้อมูลสำหรับ-training)
4. [การ Train โมเดล YOLO](#การ-train-โมเดล-yolo)
5. [ตัวอย่างโมเดลและการใช้งาน](#ตัวอย่างโมเดลและการใช้งาน)
6. [การทดสอบและ Evaluation](#การทดสอบและ-evaluation)
7. [การปรับแต่งประสิทธิภาพ](#การปรับแต่งประสิทธิภาพ)
8. [การใช้งาน API](#การใช้งาน-api)
9. [Best Practices](#best-practices)

---

## ภาพรวมของระบบ

YOLO API นี้สร้างด้วย:
- **FastAPI**: สำหรับสร้าง REST API ที่รวดเร็วและมีประสิทธิภาพ
- **Ultralytics YOLO**: สำหรับ object detection (YOLOv8)
- **Docker**: สำหรับการ deploy และรันระบบ
- **OpenCV & PIL**: สำหรับการประมวลผลภาพ

### สถาปัตยกรรมของระบบ

```
┌─────────────┐      HTTP POST      ┌──────────────┐      ┌─────────────┐
│   Client    │ ──────────────────> │  FastAPI     │ ───> │ YOLO Model  │
│ (User/App)  │                     │  Application │      │  (PyTorch)  │
└─────────────┘ <────────────────── └──────────────┘ <─── └─────────────┘
                     JSON Result
```

---

## การติดตั้งสภาพแวดล้อมสำหรับพัฒนา

### ข้อกำหนดของระบบ

**ฮาร์ดแวร์ที่แนะนำ:**
- CPU: 4+ cores
- RAM: 8GB+ (16GB แนะนำสำหรับ training)
- GPU: NVIDIA GPU with CUDA support (แนะนำสำหรับ training)
- Storage: 20GB+ free space

**ซอฟต์แวร์:**
- Python 3.10 หรือสูงกว่า
- pip (Python package manager)
- Git
- (Optional) CUDA Toolkit และ cuDNN สำหรับ GPU support

### ขั้นตอนการติดตั้ง

#### 1. Clone Repository

```bash
git clone https://github.com/somkheartk/yolo-api.git
cd yolo-api
```

#### 2. สร้าง Virtual Environment

**บน Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**บน Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

#### 3. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

#### 4. ติดตั้ง Dependencies เพิ่มเติมสำหรับ Training

```bash
# สำหรับ training และ data augmentation
pip install albumentations
pip install matplotlib
pip install tensorboard
pip install scikit-learn

# สำหรับ GPU support (ถ้ามี NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 5. ทดสอบการติดตั้ง

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## การเตรียมข้อมูลสำหรับ Training

### โครงสร้างข้อมูล

YOLO ใช้รูปแบบ YOLO format สำหรับ annotations โครงสร้างโฟลเดอร์ควรเป็นดังนี้:

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/  (optional)
    ├── images/
    └── labels/
```

### ไฟล์ data.yaml

สร้างไฟล์ `data.yaml` เพื่อกำหนดค่า dataset:

```yaml
# data.yaml
path: /path/to/dataset  # ตำแหน่งของ dataset
train: train/images     # path สำหรับ training images (relative to 'path')
val: val/images         # path สำหรับ validation images (relative to 'path')
test: test/images       # (optional) path สำหรับ test images

# Classes
names:
  0: person
  1: car
  2: bicycle
  3: dog
  4: cat
  # เพิ่ม classes ตามที่ต้องการ

nc: 5  # จำนวน classes
```

### รูปแบบของ Label Files

แต่ละรูปภาพจะมีไฟล์ label (.txt) ที่มีชื่อเดียวกัน แต่ละบรรทัดในไฟล์ label แสดงถึง object หนึ่งตัว:

```
<class_id> <x_center> <y_center> <width> <height>
```

**ตัวอย่าง (img1.txt):**
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

**หมายเหตุ:** 
- ค่าทั้งหมดเป็น normalized (0-1) relative to image dimensions
- `x_center` และ `y_center` คือจุดกึ่งกลางของ bounding box
- `width` และ `height` คือขนาดของ bounding box

### เครื่องมือสำหรับ Annotation

แนะนำเครื่องมือสำหรับการทำ annotation:

1. **Roboflow** (แนะนำ) - https://roboflow.com
   - Web-based, รองรับ auto-annotation
   - Export ในรูปแบบ YOLO format
   
2. **LabelImg** - https://github.com/heartexlabs/labelImg
   - Desktop application
   - รองรับ YOLO format

3. **CVAT** - https://www.cvat.ai
   - Web-based, open source
   - รองรับการทำงานแบบ team

### การแปลงจาก COCO/VOC Format

ถ้าคุณมีข้อมูลในรูปแบบอื่น สามารถแปลงได้:

**จาก COCO to YOLO:**
```python
from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir='path/to/coco/annotations',
    save_dir='path/to/yolo/dataset',
    use_segments=False  # True สำหรับ segmentation
)
```

**จาก VOC to YOLO:**
```python
from ultralytics.data.converter import convert_voc

convert_voc(
    labels_dir='path/to/voc/Annotations',
    save_dir='path/to/yolo/dataset'
)
```

### Data Augmentation

YOLO มี data augmentation built-in เช่น:
- Mosaic
- MixUp
- Random flip, rotate, scale
- HSV augmentation
- Random crop

สามารถปรับแต่งได้ในไฟล์ config หรือผ่าน parameters ตอน train

---

## การ Train โมเดล YOLO

### โมเดล YOLOv8 ที่มีให้เลือก

| Model | Size | mAP | Speed (ms) | Parameters |
|-------|------|-----|------------|------------|
| YOLOv8n | 640 | 37.3 | 1.0 | 3.2M |
| YOLOv8s | 640 | 44.9 | 1.6 | 11.2M |
| YOLOv8m | 640 | 50.2 | 2.9 | 25.9M |
| YOLOv8l | 640 | 52.9 | 4.1 | 43.7M |
| YOLOv8x | 640 | 53.9 | 6.5 | 68.2M |

**คำแนะนำการเลือก:**
- **YOLOv8n**: รวดเร็วที่สุด, เหมาะสำหรับ real-time บน mobile/edge devices
- **YOLOv8s/m**: สมดุลระหว่างความเร็วและความแม่นยำ
- **YOLOv8l/x**: ความแม่นยำสูงสุด, เหมาะสำหรับ applications ที่ต้องการความแม่นยำมากกว่าความเร็ว

### Training Script พื้นฐาน

สร้างไฟล์ `train.py`:

```python
from ultralytics import YOLO

# โหลด pre-trained model
model = YOLO('yolov8n.pt')  # เริ่มจาก YOLOv8n

# Train model
results = model.train(
    data='data.yaml',           # path to data.yaml
    epochs=100,                 # จำนวน epochs
    imgsz=640,                  # ขนาดรูปภาพ
    batch=16,                   # batch size
    name='yolo_custom',         # ชื่อของการ train
    project='runs/detect',      # โฟลเดอร์สำหรับบันทึกผลลัพธ์
    patience=50,                # early stopping patience
    save=True,                  # บันทึก checkpoints
    device=0,                   # 0 สำหรับ GPU, 'cpu' สำหรับ CPU
    workers=8,                  # จำนวน dataloader workers
    pretrained=True,            # ใช้ pretrained weights
    optimizer='AdamW',          # optimizer: 'SGD', 'Adam', 'AdamW'
    verbose=True,               # แสดงรายละเอียดขณะ train
    seed=42,                    # random seed สำหรับ reproducibility
    deterministic=True,         # deterministic mode
    single_cls=False,           # single class training
    rect=False,                 # rectangular training
    cos_lr=False,               # cosine learning rate scheduler
    close_mosaic=10,            # ปิด mosaic augmentation ในช่วงท้าย
    resume=False,               # resume จาก checkpoint
    amp=True,                   # Automatic Mixed Precision
    fraction=1.0,               # ใช้เฉพาะบาง fraction ของ dataset
    profile=False,              # profile ONNX and TensorRT speeds
    # Learning rate settings
    lr0=0.01,                   # initial learning rate
    lrf=0.01,                   # final learning rate (lr0 * lrf)
    momentum=0.937,             # SGD momentum
    weight_decay=0.0005,        # optimizer weight decay
    warmup_epochs=3.0,          # warmup epochs
    warmup_momentum=0.8,        # warmup momentum
    warmup_bias_lr=0.1,         # warmup bias learning rate
    # Augmentation parameters
    hsv_h=0.015,                # HSV-Hue augmentation
    hsv_s=0.7,                  # HSV-Saturation augmentation
    hsv_v=0.4,                  # HSV-Value augmentation
    degrees=0.0,                # rotation (+/- deg)
    translate=0.1,              # translation (+/- fraction)
    scale=0.5,                  # scaling (+/- gain)
    shear=0.0,                  # shear (+/- deg)
    perspective=0.0,            # perspective (+/- fraction)
    flipud=0.0,                 # flip up-down probability
    fliplr=0.5,                 # flip left-right probability
    mosaic=1.0,                 # mosaic augmentation probability
    mixup=0.0,                  # mixup augmentation probability
    copy_paste=0.0,             # copy-paste augmentation probability
)

# ผลลัพธ์จะถูกบันทึกใน runs/detect/yolo_custom/
print(f"Training completed. Results saved to {results.save_dir}")
```

### รัน Training

```bash
python train.py
```

**หรือใช้ Command Line:**

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16
```

### Training บน Multi-GPU

```bash
# ใช้ GPU หลายตัว
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 device=0,1,2,3
```

### Resume Training จาก Checkpoint

```bash
yolo detect train resume model=runs/detect/yolo_custom/weights/last.pt
```

### ตัวอย่าง Training Script แบบละเอียด

สร้างไฟล์ `train_advanced.py`:

```python
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

def train_yolo_model(
    data_yaml='data.yaml',
    model_size='n',  # n, s, m, l, x
    epochs=100,
    batch_size=16,
    img_size=640,
    project_name='yolo_project',
    device='0',
    resume_training=False
):
    """
    Train YOLOv8 model with custom settings
    
    Args:
        data_yaml: Path to data.yaml file
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        project_name: Project name for saving results
        device: Device to use ('0' for GPU, 'cpu' for CPU)
        resume_training: Resume from last checkpoint
    """
    
    # Check CUDA availability
    if device != 'cpu':
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = 'cpu'
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model_name = f'yolov8{model_size}.pt'
    
    if resume_training:
        # Resume from checkpoint
        checkpoint_path = f'runs/detect/{project_name}/weights/last.pt'
        if Path(checkpoint_path).exists():
            print(f"Resuming training from {checkpoint_path}")
            model = YOLO(checkpoint_path)
        else:
            print("Checkpoint not found, starting fresh training")
            model = YOLO(model_name)
    else:
        print(f"Loading pretrained model: {model_name}")
        model = YOLO(model_name)
    
    # Verify data.yaml exists
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Data yaml file not found: {data_yaml}")
    
    # Load and print dataset info
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    print(f"\nDataset configuration:")
    print(f"  Classes: {data_config.get('nc', 'Unknown')}")
    print(f"  Names: {data_config.get('names', 'Unknown')}")
    
    # Training parameters
    train_params = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'name': project_name,
        'project': 'runs/detect',
        'device': device,
        'workers': 8,
        'patience': 50,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': 42,
        'amp': True,  # Automatic Mixed Precision
        # Learning rate
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
    }
    
    print("\nStarting training with parameters:")
    for key, value in train_params.items():
        print(f"  {key}: {value}")
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60 + "\n")
    
    results = model.train(**train_params)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"\nResults saved to: {results.save_dir}")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Last model: {results.save_dir}/weights/last.pt")
    
    return model, results

if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    model, results = train_yolo_model(
        data_yaml='data.yaml',
        model_size='n',  # เริ่มจาก nano model
        epochs=100,
        batch_size=16,
        img_size=640,
        project_name='my_custom_detector',
        device='0',  # ใช้ GPU 0
        resume_training=False
    )
```

### การ Monitor Training

**1. TensorBoard:**

```bash
# เปิด TensorBoard
tensorboard --logdir runs/detect

# เปิดบน browser: http://localhost:6006
```

**2. Training Logs:**

ผลลัพธ์จะถูกบันทึกใน `runs/detect/yolo_custom/`:
- `weights/best.pt` - โมเดลที่ดีที่สุด
- `weights/last.pt` - โมเดลล่าสุด
- `results.png` - กราฟผลการ train
- `confusion_matrix.png` - confusion matrix
- `F1_curve.png`, `PR_curve.png`, `P_curve.png`, `R_curve.png` - curves
- `args.yaml` - parameters ที่ใช้

---

## ตัวอย่างโมเดลและการใช้งาน

### ตัวอย่างที่ 1: Object Detection พื้นฐาน (COCO Classes)

```python
from ultralytics import YOLO
from PIL import Image

# โหลดโมเดล pretrained
model = YOLO('yolov8n.pt')

# ทำนายบนรูปภาพ
results = model('image.jpg')

# แสดงผลลัพธ์
for result in results:
    # แสดงรูปภาพพร้อม bounding boxes
    result.show()
    
    # บันทึกรูปภาพ
    result.save('result.jpg')
    
    # ดึงข้อมูล detections
    boxes = result.boxes
    for box in boxes:
        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        # Confidence
        conf = box.conf[0].item()
        # Class ID
        cls = int(box.cls[0].item())
        # Class name
        class_name = model.names[cls]
        
        print(f"{class_name}: {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

### ตัวอย่างที่ 2: Custom Trained Model

```python
from ultralytics import YOLO

# โหลดโมเดลที่ train เอง
model = YOLO('runs/detect/yolo_custom/weights/best.pt')

# ทำนาย
results = model.predict(
    source='test_images/',  # สามารถเป็น folder, video, หรือ webcam
    conf=0.25,              # confidence threshold
    iou=0.45,               # NMS IOU threshold
    save=True,              # บันทึกผลลัพธ์
    save_txt=True,          # บันทึก labels
    save_conf=True,         # บันทึก confidence ใน labels
    project='runs/predict', # โฟลเดอร์สำหรับบันทึก
    name='test_run',        # ชื่อของการรัน
)
```

### ตัวอย่างที่ 3: Real-time Detection จาก Webcam

```python
from ultralytics import YOLO
import cv2

# โหลดโมเดล
model = YOLO('yolov8n.pt')

# เปิด webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ทำนาย
    results = model(frame, verbose=False)
    
    # วาด bounding boxes
    annotated_frame = results[0].plot()
    
    # แสดงผล
    cv2.imshow('YOLO Detection', annotated_frame)
    
    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### ตัวอย่างที่ 4: Video Processing

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# ประมวลผล video
results = model.predict(
    source='input_video.mp4',
    save=True,
    stream=True,  # stream mode สำหรับ video ขนาดใหญ่
    conf=0.3,
    iou=0.45,
)

# แสดงผลลัพธ์แต่ละ frame
for result in results:
    boxes = result.boxes
    print(f"Frame: {result.path}, Detections: {len(boxes)}")
```

### ตัวอย่างที่ 5: Batch Processing

```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO('yolov8n.pt')

# หา images ทั้งหมดในโฟลเดอร์
image_folder = Path('test_images')
image_files = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))

# ประมวลผลเป็น batch
results = model(image_files, batch=8)

# บันทึกผลลัพธ์
for i, result in enumerate(results):
    result.save(f'output/result_{i}.jpg')
```

---

## การทดสอบและ Evaluation

### Validate Model

```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolo_custom/weights/best.pt')

# Validate on validation set
metrics = model.val(
    data='data.yaml',
    split='val',  # 'val', 'test', or 'train'
    imgsz=640,
    batch=16,
    conf=0.001,
    iou=0.6,
    device='0',
    save_json=True,  # บันทึกผลลัพธ์เป็น COCO JSON format
    save_hybrid=False,
    plots=True,
)

# แสดง metrics
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
```

### Test on Single Image

```python
from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/yolo_custom/weights/best.pt')

# โหลดรูปภาพ
image = cv2.imread('test.jpg')

# ทำนาย
results = model(image)

# ดู detailed results
result = results[0]
print(f"Number of detections: {len(result.boxes)}")
print(f"Inference time: {result.speed['inference']:.1f}ms")
print(f"Image shape: {result.orig_shape}")

# แสดงแต่ละ detection
for i, box in enumerate(result.boxes):
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].tolist()
    
    print(f"\nDetection {i+1}:")
    print(f"  Class: {model.names[cls_id]}")
    print(f"  Confidence: {conf:.2%}")
    print(f"  BBox: {xyxy}")
```

### Export Metrics

```python
from ultralytics import YOLO
import json

model = YOLO('runs/detect/yolo_custom/weights/best.pt')
metrics = model.val()

# Export เป็น dictionary
metrics_dict = {
    'mAP50': float(metrics.box.map50),
    'mAP50-95': float(metrics.box.map),
    'precision': float(metrics.box.mp),
    'recall': float(metrics.box.mr),
    'fitness': float(metrics.fitness),
}

# บันทึกเป็น JSON
with open('metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)

print(json.dumps(metrics_dict, indent=2))
```

### Compare Multiple Models

```python
from ultralytics import YOLO
import pandas as pd

models = {
    'YOLOv8n': 'yolov8n.pt',
    'YOLOv8s': 'yolov8s.pt',
    'Custom': 'runs/detect/yolo_custom/weights/best.pt',
}

results = []

for name, model_path in models.items():
    model = YOLO(model_path)
    metrics = model.val(data='data.yaml', verbose=False)
    
    results.append({
        'Model': name,
        'mAP50': f"{metrics.box.map50:.3f}",
        'mAP50-95': f"{metrics.box.map:.3f}",
        'Precision': f"{metrics.box.mp:.3f}",
        'Recall': f"{metrics.box.mr:.3f}",
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

---

## การปรับแต่งประสิทธิภาพ

### 1. Model Optimization

**Export เป็น ONNX (เร็วกว่า PyTorch):**

```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolo_custom/weights/best.pt')

# Export เป็น ONNX
model.export(format='onnx', dynamic=True, simplify=True)

# ใช้งาน ONNX model
onnx_model = YOLO('runs/detect/yolo_custom/weights/best.onnx')
results = onnx_model('image.jpg')
```

**Export เป็น TensorRT (เร็วที่สุดสำหรับ NVIDIA GPU):**

```python
# Export เป็น TensorRT
model.export(format='engine', device=0, half=True)  # FP16 precision

# ใช้งาน TensorRT model
trt_model = YOLO('runs/detect/yolo_custom/weights/best.engine')
results = trt_model('image.jpg')
```

**Format ที่รองรับ:**
- PyTorch (`.pt`)
- ONNX (`.onnx`)
- TensorRT (`.engine`)
- CoreML (`.mlmodel`)
- TFLite (`.tflite`)
- OpenVINO

### 2. Inference Optimization

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# ใช้ half precision (FP16) สำหรับ GPU
results = model('image.jpg', half=True, device=0)

# ปรับ confidence threshold
results = model('image.jpg', conf=0.5)  # เพิ่มความเร็วโดยกรองผลที่ confidence ต่ำ

# ปรับขนาดรูปภาพ
results = model('image.jpg', imgsz=320)  # ใช้รูปที่เล็กกว่าจะเร็วกว่า
```

### 3. Batch Processing

```python
# ประมวลผลหลายรูปพร้อมกัน
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model(images, batch=3)  # ประมวลผล 3 รูปพร้อมกัน
```

### 4. Tips สำหรับการเพิ่มความเร็ว

1. **ใช้ GPU** แทน CPU
2. **ใช้ FP16 (half precision)** แทน FP32
3. **Export เป็น TensorRT** สำหรับ production
4. **เลือกโมเดลที่เล็กกว่า** (n, s) ถ้าไม่ต้องการความแม่นยำสูงมาก
5. **ลดขนาดรูปภาพ input** (320, 416 แทน 640)
6. **เพิ่ม confidence threshold** เพื่อกรองผลที่ไม่ต้องการ
7. **ใช้ batch processing** สำหรับหลายรูป

### 5. Tips สำหรับการเพิ่มความแม่นยำ

1. **เพิ่มข้อมูล training** - ยิ่งมีข้อมูลมากยิ่งดี
2. **Balance classes** - ให้แต่ละ class มีจำนวนใกล้เคียงกัน
3. **Data augmentation** - ปรับแต่ง augmentation parameters
4. **เพิ่ม epochs** - train นานขึ้น
5. **ใช้โมเดลที่ใหญ่ขึ้น** (m, l, x)
6. **Pre-trained weights** - ใช้ pretrained weights เสมอ
7. **Fine-tuning learning rate** - ลองปรับ learning rate
8. **Image size** - ใช้รูปที่ใหญ่ขึ้น (1280, 1920)

---

## การใช้งาน API

### การ Deploy Model ใน API

หลังจาก train model เสร็จแล้ว นำมาใช้กับ API:

```bash
# 1. Copy โมเดลไปยังโฟลเดอร์ models
cp runs/detect/yolo_custom/weights/best.pt models/my_model.pt

# 2. แก้ไข docker-compose.yml
# environment:
#   - MODEL_PATH=/app/models/my_model.pt

# 3. รัน API
docker-compose up --build
```

### ตัวอย่างการใช้งาน API

**1. Python Client:**

```python
import requests
from pathlib import Path

class YOLOClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """ตรวจสอบสถานะ API"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def detect(self, image_path, conf=0.25, iou=0.45):
        """ตรวจจับวัตถุในรูปภาพ"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'conf': conf, 'iou': iou}
            response = requests.post(
                f"{self.base_url}/detect",
                files=files,
                data=data
            )
        return response.json()
    
    def model_info(self):
        """ดูข้อมูลโมเดล"""
        response = requests.get(f"{self.base_url}/model-info")
        return response.json()

# ตัวอย่างการใช้งาน
client = YOLOClient()

# ตรวจสอบสถานะ
print(client.health_check())

# ตรวจจับวัตถุ
result = client.detect('test.jpg', conf=0.3, iou=0.5)
print(f"Found {result['count']} objects:")
for det in result['detections']:
    print(f"  - {det['class_name']}: {det['confidence']:.2%}")
```

**2. JavaScript/Node.js Client:**

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

class YOLOClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async healthCheck() {
        const response = await axios.get(`${this.baseUrl}/health`);
        return response.data;
    }
    
    async detect(imagePath, conf = 0.25, iou = 0.45) {
        const form = new FormData();
        form.append('file', fs.createReadStream(imagePath));
        form.append('conf', conf);
        form.append('iou', iou);
        
        const response = await axios.post(
            `${this.baseUrl}/detect`,
            form,
            { headers: form.getHeaders() }
        );
        return response.data;
    }
}

// ตัวอย่างการใช้งาน
const client = new YOLOClient();

client.detect('test.jpg', 0.3, 0.5)
    .then(result => {
        console.log(`Found ${result.count} objects:`);
        result.detections.forEach(det => {
            console.log(`  - ${det.class_name}: ${(det.confidence * 100).toFixed(1)}%`);
        });
    })
    .catch(err => console.error('Error:', err.message));
```

**3. cURL:**

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info

# Detect objects
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg" \
  -F "conf=0.3" \
  -F "iou=0.5"
```

**4. PHP Client:**

```php
<?php
class YOLOClient {
    private $baseUrl;
    
    public function __construct($baseUrl = 'http://localhost:8000') {
        $this->baseUrl = $baseUrl;
    }
    
    public function detect($imagePath, $conf = 0.25, $iou = 0.45) {
        $ch = curl_init();
        
        $postData = [
            'file' => new CURLFile($imagePath),
            'conf' => $conf,
            'iou' => $iou
        ];
        
        curl_setopt($ch, CURLOPT_URL, $this->baseUrl . '/detect');
        curl_setopt($ch, CURLOPT_POST, 1);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $postData);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        
        $response = curl_exec($ch);
        curl_close($ch);
        
        return json_decode($response, true);
    }
}

// ตัวอย่างการใช้งาน
$client = new YOLOClient();
$result = $client->detect('test.jpg', 0.3, 0.5);

echo "Found {$result['count']} objects:\n";
foreach ($result['detections'] as $det) {
    echo "  - {$det['class_name']}: " . 
         round($det['confidence'] * 100, 1) . "%\n";
}
?>
```

### Error Handling

```python
import requests
from requests.exceptions import RequestException

def detect_with_retry(image_path, max_retries=3):
    """ตรวจจับวัตถุพร้อม retry logic"""
    
    for attempt in range(max_retries):
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    'http://localhost:8000/detect',
                    files=files,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
                
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}")
            print(f"Retrying...")
    
# ตัวอย่างการใช้งาน
try:
    result = detect_with_retry('test.jpg')
    print(f"Success: {result['count']} objects detected")
except Exception as e:
    print(f"Failed after retries: {e}")
```

---

## Best Practices

### 1. Dataset Best Practices

- ✅ **มีข้อมูลเพียงพอ**: อย่างน้อย 100-200 รูปต่อ class
- ✅ **Balance classes**: พยายามให้แต่ละ class มีจำนวนใกล้เคียงกัน
- ✅ **Diverse data**: มีความหลากหลายในแง่มุม แสง พื้นหลัง
- ✅ **Quality labels**: ตรวจสอบ labels ให้ถูกต้อง
- ✅ **Split properly**: 70-80% train, 10-20% validation, 10% test

### 2. Training Best Practices

- ✅ **ใช้ pretrained weights**: เริ่มจาก pretrained model เสมอ
- ✅ **Start small**: เริ่มจากโมเดลเล็ก (YOLOv8n) ก่อน
- ✅ **Monitor training**: ดู loss และ metrics ระหว่าง train
- ✅ **Use early stopping**: ตั้ง patience ป้องกัน overfitting
- ✅ **Save checkpoints**: บันทึก checkpoints เป็นระยะ
- ✅ **Validate regularly**: validate หลัง train เสร็จ

### 3. Production Best Practices

- ✅ **Use Docker**: deploy ด้วย Docker เพื่อความสะดวก
- ✅ **Optimize model**: export เป็น ONNX/TensorRT สำหรับ production
- ✅ **Set proper thresholds**: ปรับ confidence threshold ตามการใช้งาน
- ✅ **Error handling**: จัดการ error และ edge cases
- ✅ **Monitoring**: ติดตาม performance และ errors
- ✅ **Logging**: บันทึก logs สำหรับ debugging

### 4. Security Best Practices

- ✅ **Validate inputs**: ตรวจสอบ file types และขนาด
- ✅ **Rate limiting**: จำกัดจำนวน requests
- ✅ **Authentication**: ใส่ authentication ถ้าจำเป็น
- ✅ **HTTPS**: ใช้ HTTPS ใน production
- ✅ **Update dependencies**: อัพเดท libraries เป็นประจำ

---

## สรุป

คู่มือนี้ครอบคลุม:
- ✅ การติดตั้งและตั้งค่าสภาพแวดล้อม
- ✅ การเตรียมข้อมูลและ annotation
- ✅ การ train โมเดล YOLO อย่างละเอียด
- ✅ ตัวอย่างโมเดลและการใช้งาน
- ✅ การทดสอบและ evaluation
- ✅ การปรับแต่งประสิทธิภาพ
- ✅ การใช้งาน API
- ✅ Best practices

สำหรับข้อมูลเพิ่มเติม:
- [Ultralytics Documentation](https://docs.ultralytics.com)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [YOLO Community](https://community.ultralytics.com)

สำหรับการ deploy บน Digital Ocean โปรดดูที่ [DEPLOYMENT.md](DEPLOYMENT.md)
