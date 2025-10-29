#!/usr/bin/env python3
"""
ตัวอย่างการใช้งาน YOLO โดยตรง (ไม่ผ่าน API)
Direct YOLO Usage Examples (without API)

รวมตัวอย่างการใช้งาน YOLO แบบต่างๆ เช่น:
- Image detection
- Video processing
- Webcam real-time detection
- Batch processing
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import time


def example_image_detection():
    """ตัวอย่างการตรวจจับวัตถุในรูปภาพ"""
    print("="*60)
    print("Example 1: Image Detection")
    print("="*60)
    
    # โหลดโมเดล
    model = YOLO('yolov8n.pt')
    
    # ทำนาย
    # results = model('test.jpg', conf=0.25, iou=0.45)
    
    # แสดงผลลัพธ์
    # for result in results:
    #     # แสดงรูปภาพ
    #     result.show()
    #     
    #     # บันทึกรูปภาพ
    #     result.save('result.jpg')
    #     
    #     # ดึงข้อมูล detections
    #     boxes = result.boxes
    #     for box in boxes:
    #         cls_name = model.names[int(box.cls[0])]
    #         conf = float(box.conf[0])
    #         print(f"  {cls_name}: {conf:.2%}")
    
    print("Example ready (uncomment to use)")


def example_video_processing():
    """ตัวอย่างการประมวลผล video"""
    print("\n" + "="*60)
    print("Example 2: Video Processing")
    print("="*60)
    
    # โหลดโมเดล
    model = YOLO('yolov8n.pt')
    
    # ประมวลผล video
    # results = model.predict(
    #     source='input_video.mp4',
    #     save=True,
    #     stream=True,
    #     conf=0.3,
    #     iou=0.45,
    # )
    # 
    # # แสดงผลลัพธ์แต่ละ frame
    # for i, result in enumerate(results):
    #     boxes = result.boxes
    #     print(f"Frame {i}: {len(boxes)} detections")
    
    print("Example ready (uncomment to use)")


def example_webcam_realtime():
    """ตัวอย่างการตรวจจับแบบ real-time จาก webcam"""
    print("\n" + "="*60)
    print("Example 3: Webcam Real-time Detection")
    print("="*60)
    
    # โหลดโมเดล
    model = YOLO('yolov8n.pt')
    
    # เปิด webcam
    # cap = cv2.VideoCapture(0)
    # 
    # print("Press 'q' to quit")
    # 
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     
    #     # ทำนาย
    #     results = model(frame, verbose=False)
    #     
    #     # วาด bounding boxes
    #     annotated_frame = results[0].plot()
    #     
    #     # แสดงผล
    #     cv2.imshow('YOLO Detection', annotated_frame)
    #     
    #     # กด 'q' เพื่อออก
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # 
    # cap.release()
    # cv2.destroyAllWindows()
    
    print("Example ready (uncomment to use)")


def example_batch_processing():
    """ตัวอย่างการประมวลผลหลายรูปพร้อมกัน"""
    print("\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)
    
    # โหลดโมเดล
    model = YOLO('yolov8n.pt')
    
    # รายการรูปภาพ
    # image_folder = Path('test_images')
    # image_files = list(image_folder.glob('*.jpg')) + \
    #               list(image_folder.glob('*.png'))
    # 
    # print(f"Processing {len(image_files)} images...")
    # 
    # # ประมวลผลเป็น batch
    # start_time = time.time()
    # results = model(image_files, batch=8)
    # elapsed_time = time.time() - start_time
    # 
    # # บันทึกผลลัพธ์
    # output_dir = Path('output')
    # output_dir.mkdir(exist_ok=True)
    # 
    # for i, result in enumerate(results):
    #     result.save(output_dir / f'result_{i}.jpg')
    # 
    # print(f"\nProcessed {len(image_files)} images in {elapsed_time:.2f}s")
    # print(f"Average: {elapsed_time/len(image_files):.3f}s per image")
    
    print("Example ready (uncomment to use)")


def example_custom_model():
    """ตัวอย่างการใช้งานโมเดลที่ train เอง"""
    print("\n" + "="*60)
    print("Example 5: Custom Trained Model")
    print("="*60)
    
    # โหลดโมเดลที่ train เอง
    # model = YOLO('runs/train/yolo_custom/weights/best.pt')
    # 
    # # ทำนาย
    # results = model.predict(
    #     source='test_images/',
    #     conf=0.25,
    #     iou=0.45,
    #     save=True,
    #     save_txt=True,
    #     save_conf=True,
    #     project='runs/predict',
    #     name='custom_test',
    # )
    # 
    # # แสดงข้อมูลโมเดล
    # print(f"Model classes: {model.names}")
    
    print("Example ready (uncomment to use)")


def example_tracking():
    """ตัวอย่างการ track objects ใน video"""
    print("\n" + "="*60)
    print("Example 6: Object Tracking")
    print("="*60)
    
    # โหลดโมเดล
    model = YOLO('yolov8n.pt')
    
    # Track objects ใน video
    # results = model.track(
    #     source='input_video.mp4',
    #     save=True,
    #     tracker='bytetrack.yaml',  # หรือ 'botsort.yaml'
    #     conf=0.3,
    #     iou=0.45,
    # )
    # 
    # # แสดงผลลัพธ์
    # for result in results:
    #     boxes = result.boxes
    #     if boxes.id is not None:
    #         # แสดง track IDs
    #         track_ids = boxes.id.int().cpu().tolist()
    #         print(f"Tracked objects: {track_ids}")
    
    print("Example ready (uncomment to use)")


def example_export_model():
    """ตัวอย่างการ export โมเดลเป็นรูปแบบอื่น"""
    print("\n" + "="*60)
    print("Example 7: Export Model")
    print("="*60)
    
    # โหลดโมเดล
    model = YOLO('yolov8n.pt')
    
    # Export เป็น ONNX
    # model.export(format='onnx', dynamic=True, simplify=True)
    # print("Model exported to ONNX format")
    
    # Export เป็น TensorRT (ต้องมี GPU)
    # model.export(format='engine', device=0, half=True)
    # print("Model exported to TensorRT format")
    
    # Format ที่รองรับ:
    # - onnx: ONNX format
    # - engine: TensorRT
    # - coreml: CoreML (iOS)
    # - tflite: TensorFlow Lite
    # - openvino: OpenVINO
    
    print("Example ready (uncomment to use)")


def example_validation():
    """ตัวอย่างการ validate โมเดล"""
    print("\n" + "="*60)
    print("Example 8: Model Validation")
    print("="*60)
    
    # โหลดโมเดล
    model = YOLO('yolov8n.pt')
    
    # Validate on validation set
    # metrics = model.val(
    #     data='data.yaml',
    #     split='val',
    #     imgsz=640,
    #     batch=16,
    #     conf=0.001,
    #     iou=0.6,
    #     device='0',
    # )
    # 
    # # แสดง metrics
    # print(f"\nValidation Metrics:")
    # print(f"  mAP50:     {metrics.box.map50:.3f}")
    # print(f"  mAP50-95:  {metrics.box.map:.3f}")
    # print(f"  Precision: {metrics.box.mp:.3f}")
    # print(f"  Recall:    {metrics.box.mr:.3f}")
    
    print("Example ready (uncomment to use)")


def main():
    """Main function"""
    print("YOLO Direct Usage Examples")
    print("="*60)
    
    # Example 1: Image detection
    example_image_detection()
    
    # Example 2: Video processing
    example_video_processing()
    
    # Example 3: Webcam real-time
    example_webcam_realtime()
    
    # Example 4: Batch processing
    example_batch_processing()
    
    # Example 5: Custom model
    example_custom_model()
    
    # Example 6: Object tracking
    example_tracking()
    
    # Example 7: Export model
    example_export_model()
    
    # Example 8: Validation
    example_validation()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("Uncomment the example code you want to try")
    print("="*60)


if __name__ == "__main__":
    main()
