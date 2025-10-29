#!/usr/bin/env python3
"""
Example script to test object detection with YOLO API
Creates a simple test image if none is provided
"""
import requests
import sys
import json
from PIL import Image, ImageDraw
import io

def create_test_image():
    """Create a simple test image with shapes"""
    print("Creating test image...")
    img = Image.new('RGB', (640, 480), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([50, 50, 200, 200], fill='red', outline='black', width=3)
    draw.ellipse([300, 100, 500, 300], fill='blue', outline='black', width=3)
    draw.rectangle([100, 300, 300, 450], fill='green', outline='black', width=3)
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

def test_detection(base_url, image_path=None, conf=0.25, iou=0.45):
    """Test object detection endpoint"""
    print(f"\nTesting object detection at {base_url}/detect")
    print(f"Parameters: conf={conf}, iou={iou}")
    
    try:
        # Prepare image
        if image_path:
            print(f"Using image: {image_path}")
            files = {"file": open(image_path, "rb")}
        else:
            print("No image provided, creating test image...")
            files = {"file": ("test.jpg", create_test_image(), "image/jpeg")}
        
        # Make request
        params = {"conf": conf, "iou": iou}
        response = requests.post(f"{base_url}/detect", files=files, data=params)
        response.raise_for_status()
        
        # Parse results
        data = response.json()
        
        print("\n" + "=" * 50)
        print("Detection Results:")
        print("=" * 50)
        print(f"Success: {data['success']}")
        print(f"Objects detected: {data['count']}")
        print(f"Image size: {data['image_shape']['width']}x{data['image_shape']['height']}")
        
        if data['detections']:
            print("\nDetected objects:")
            for i, det in enumerate(data['detections'], 1):
                print(f"\n{i}. {det['class_name']}")
                print(f"   Confidence: {det['confidence']:.2%}")
                print(f"   Bounding box: ({det['bbox']['x1']:.1f}, {det['bbox']['y1']:.1f}) "
                      f"to ({det['bbox']['x2']:.1f}, {det['bbox']['y2']:.1f})")
        else:
            print("\nNo objects detected")
        
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        return False

def main():
    base_url = "http://localhost:8000"
    image_path = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        base_url = sys.argv[2]
    
    print(f"YOLO Object Detection Test")
    print("=" * 50)
    
    success = test_detection(base_url, image_path)
    
    if success:
        print("\n✓ Test completed successfully!")
        return 0
    else:
        print("\n✗ Test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
