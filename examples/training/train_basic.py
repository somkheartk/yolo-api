#!/usr/bin/env python3
"""
ตัวอย่างการ Train YOLO Model พื้นฐาน
Basic YOLO Model Training Example

วิธีใช้งาน:
    python train_basic.py

หมายเหตุ:
    - ต้องมีไฟล์ data.yaml ที่กำหนด dataset
    - ต้องมี train/images และ train/labels
"""

from ultralytics import YOLO
import torch

def main():
    # ตรวจสอบ GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("⚠ No GPU detected, using CPU")
        device = 'cpu'
    
    # โหลดโมเดล pretrained
    print("\nLoading YOLOv8n pretrained model...")
    model = YOLO('yolov8n.pt')
    
    # Training parameters
    print("\nStarting training...")
    results = model.train(
        data='data.yaml',       # path to data configuration
        epochs=50,              # จำนวน epochs (เริ่มจาก 50 สำหรับทดสอบ)
        imgsz=640,              # ขนาดรูปภาพ
        batch=16,               # batch size (ปรับตาม RAM/VRAM)
        name='yolo_basic',      # ชื่อของการ train
        device=device,          # GPU or CPU
        project='runs/train',   # โฟลเดอร์สำหรับบันทึกผล
        patience=10,            # early stopping
        save=True,              # บันทึก checkpoints
        verbose=True,           # แสดงรายละเอียด
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Results saved to: {results.save_dir}")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Last model: {results.save_dir}/weights/last.pt")
    
    # Validate โมเดล
    print("\nValidating model...")
    metrics = model.val()
    
    print("\nMetrics:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall: {metrics.box.mr:.3f}")

if __name__ == "__main__":
    main()
