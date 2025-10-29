# YOLO Object Detection API

API สำหรับตรวจจับวัตถุด้วย YOLO ที่รันบน Docker / YOLO Object Detection API running on Docker

## คุณสมบัติ / Features

- 🚀 REST API สำหรับตรวจจับวัตถุด้วย YOLO / REST API for YOLO object detection
- 🐳 รองรับการรันบน Docker / Docker support
- 📦 ใช้ YOLOv8 จาก Ultralytics
- 🔄 สามารถใช้โมเดลที่ train เองได้ / Support custom trained models
- ⚡ Fast inference with FastAPI
- 📊 JSON response พร้อม bounding boxes และ class labels

## การติดตั้ง / Installation

### ข้อกำหนด / Requirements

- Docker และ Docker Compose
- หรือ Python 3.10+ (สำหรับรันแบบไม่ใช้ Docker)

### วิธีที่ 1: ใช้ Docker (แนะนำ) / Method 1: Using Docker (Recommended)

1. Clone repository:
```bash
git clone https://github.com/somkheartk/yolo-api.git
cd yolo-api
```

2. (ถ้ามีโมเดลที่ train เอง) วางไฟล์โมเดล .pt ในโฟลเดอร์ `models/`:
```bash
cp /path/to/your/model.pt models/your_model.pt
```

3. ถ้าใช้โมเดลของคุณเอง แก้ไขไฟล์ `docker-compose.yml`:
```yaml
environment:
  - MODEL_PATH=/app/models/your_model.pt
```

4. สร้างและรัน Docker container:
```bash
docker-compose up --build
```

API จะรันที่ `http://localhost:8000`

### วิธีที่ 2: รันโดยตรง / Method 2: Direct Run

1. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) ตั้งค่า path ของโมเดล:
```bash
export MODEL_PATH=models/your_model.pt
```

3. รันแอพพลิเคชัน:
```bash
python main.py
```

หรือ:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## การใช้งาน / Usage

### 1. ตรวจสอบสถานะ API / Check API Health

```bash
curl http://localhost:8000/health
```

### 2. ดูข้อมูลโมเดล / Get Model Information

```bash
curl http://localhost:8000/model-info
```

### 3. ตรวจจับวัตถุในรูปภาพ / Detect Objects in Image

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@your_image.jpg" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

ตัวอย่าง Response:
```json
{
  "success": true,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.92,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 300.7,
        "y2": 500.9
      }
    }
  ],
  "count": 1,
  "image_shape": {
    "height": 640,
    "width": 480
  }
}
```

### Parameters

- `file`: ไฟล์รูปภาพ (required)
- `conf`: ค่า confidence threshold (default: 0.25)
- `iou`: ค่า IOU threshold สำหรับ NMS (default: 0.45)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | ข้อมูล API และ endpoints ที่มี |
| `/health` | GET | ตรวจสอบสถานะ API |
| `/model-info` | GET | ข้อมูลโมเดลที่ใช้งาน |
| `/detect` | POST | ตรวจจับวัตถุในรูปภาพ |

## การใช้โมเดลที่ Train เอง / Using Custom Trained Models

1. วางไฟล์โมเดล `.pt` ในโฟลเดอร์ `models/`
2. แก้ไข environment variable `MODEL_PATH` ใน `docker-compose.yml` หรือตั้งค่าตอนรัน
3. รีสตาร์ท container:
```bash
docker-compose down
docker-compose up --build
```

## ตัวอย่างการใช้งานด้วย Python / Python Example

```python
import requests

# Upload image for detection
url = "http://localhost:8000/detect"
files = {"file": open("image.jpg", "rb")}
params = {"conf": 0.3, "iou": 0.5}

response = requests.post(url, files=files, data=params)
result = response.json()

print(f"Found {result['count']} objects")
for detection in result['detections']:
    print(f"- {detection['class_name']}: {detection['confidence']:.2f}")
```

## โครงสร้างโปรเจค / Project Structure

```
yolo-api/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker image configuration
├── docker-compose.yml     # Docker compose configuration
├── models/                # Directory for YOLO models
│   └── .gitkeep
└── README.md
```

## การพัฒนา / Development

### รัน Tests (ถ้ามี)
```bash
pytest
```

### Rebuild Docker Image
```bash
docker-compose build --no-cache
```

### ดู Logs
```bash
docker-compose logs -f
```

## Troubleshooting

### ปัญหา: Model ไม่ download
- ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
- ลอง download model ด้วยตัวเอง และวางในโฟลเดอร์ `models/`

### ปัญหา: Out of Memory
- ลดขนาดรูปภาพก่อนส่ง
- ใช้โมเดลที่เล็กกว่า (เช่น yolov8n แทน yolov8x)

### ปัญหา: Docker build ช้า
- ใช้ Docker cache: `docker-compose build`
- ตรวจสอบ network speed

## License

MIT License

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.