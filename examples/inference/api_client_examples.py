#!/usr/bin/env python3
"""
ตัวอย่างการใช้งาน API แบบต่างๆ
Various API Usage Examples

รวมตัวอย่างการใช้งาน YOLO API ด้วย Python, requests, และ async
"""

import requests
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any
import json
import time


class YOLOAPIClient:
    """YOLO API Client สำหรับเรียกใช้งาน API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize YOLO API Client
        
        Args:
            base_url: Base URL ของ API
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """
        ตรวจสอบสถานะ API
        
        Returns:
            Dictionary ของสถานะ API
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def model_info(self) -> Dict[str, Any]:
        """
        ดูข้อมูลโมเดลที่ใช้งาน
        
        Returns:
            Dictionary ของข้อมูลโมเดล
        """
        response = requests.get(f"{self.base_url}/model-info")
        response.raise_for_status()
        return response.json()
    
    def detect(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45
    ) -> Dict[str, Any]:
        """
        ตรวจจับวัตถุในรูปภาพ
        
        Args:
            image_path: Path ไปยังไฟล์รูปภาพ
            conf: Confidence threshold
            iou: IOU threshold
            
        Returns:
            Dictionary ของผลลัพธ์
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'conf': conf, 'iou': iou}
            response = requests.post(
                f"{self.base_url}/detect",
                files=files,
                data=data
            )
        response.raise_for_status()
        return response.json()
    
    def detect_batch(
        self,
        image_paths: List[str],
        conf: float = 0.25,
        iou: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        ตรวจจับวัตถุในหลายรูปภาพ (sequential)
        
        Args:
            image_paths: List ของ paths ไปยังไฟล์รูปภาพ
            conf: Confidence threshold
            iou: IOU threshold
            
        Returns:
            List ของผลลัพธ์
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.detect(image_path, conf, iou)
                results.append({
                    'image': image_path,
                    'success': True,
                    'data': result
                })
            except Exception as e:
                results.append({
                    'image': image_path,
                    'success': False,
                    'error': str(e)
                })
        return results


class AsyncYOLOAPIClient:
    """Async YOLO API Client สำหรับการเรียกใช้งานแบบ async"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize Async YOLO API Client
        
        Args:
            base_url: Base URL ของ API
        """
        self.base_url = base_url.rstrip('/')
    
    async def detect(
        self,
        session: aiohttp.ClientSession,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45
    ) -> Dict[str, Any]:
        """
        ตรวจจับวัตถุในรูปภาพ (async)
        
        Args:
            session: aiohttp ClientSession
            image_path: Path ไปยังไฟล์รูปภาพ
            conf: Confidence threshold
            iou: IOU threshold
            
        Returns:
            Dictionary ของผลลัพธ์
        """
        data = aiohttp.FormData()
        with open(image_path, 'rb') as f:
            data.add_field('file', f)
            data.add_field('conf', str(conf))
            data.add_field('iou', str(iou))
            
            async with session.post(
                f"{self.base_url}/detect",
                data=data
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def detect_batch(
        self,
        image_paths: List[str],
        conf: float = 0.25,
        iou: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        ตรวจจับวัตถุในหลายรูปภาพ (parallel)
        
        Args:
            image_paths: List ของ paths ไปยังไฟล์รูปภาพ
            conf: Confidence threshold
            iou: IOU threshold
            
        Returns:
            List ของผลลัพธ์
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.detect(session, image_path, conf, iou)
                for image_path in image_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # จัดรูปแบบผลลัพธ์
            formatted_results = []
            for image_path, result in zip(image_paths, results):
                if isinstance(result, Exception):
                    formatted_results.append({
                        'image': image_path,
                        'success': False,
                        'error': str(result)
                    })
                else:
                    formatted_results.append({
                        'image': image_path,
                        'success': True,
                        'data': result
                    })
            
            return formatted_results


# ตัวอย่างการใช้งาน
def example_basic():
    """ตัวอย่างการใช้งานพื้นฐาน"""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    client = YOLOAPIClient()
    
    # ตรวจสอบสถานะ
    health = client.health_check()
    print(f"API Status: {health['status']}")
    print(f"Model Loaded: {health['model_loaded']}")
    
    # ดูข้อมูลโมเดล
    model_info = client.model_info()
    print(f"Model Path: {model_info['model_path']}")
    
    # ตรวจจับวัตถุ
    # result = client.detect('test.jpg', conf=0.3, iou=0.5)
    # print(f"Detected {result['count']} objects")


def example_batch_sequential():
    """ตัวอย่างการประมวลผลหลายรูปแบบ sequential"""
    print("\n" + "="*60)
    print("Example 2: Batch Processing (Sequential)")
    print("="*60)
    
    client = YOLOAPIClient()
    
    # รายการรูปภาพ
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    
    # ประมวลผล
    start_time = time.time()
    results = client.detect_batch(image_paths, conf=0.3, iou=0.5)
    elapsed_time = time.time() - start_time
    
    # แสดงผลลัพธ์
    print(f"\nProcessed {len(image_paths)} images in {elapsed_time:.2f}s")
    for result in results:
        if result['success']:
            count = result['data']['count']
            print(f"  {result['image']}: {count} objects detected")
        else:
            print(f"  {result['image']}: Error - {result['error']}")


async def example_batch_parallel():
    """ตัวอย่างการประมวลผลหลายรูปแบบ parallel (async)"""
    print("\n" + "="*60)
    print("Example 3: Batch Processing (Parallel)")
    print("="*60)
    
    client = AsyncYOLOAPIClient()
    
    # รายการรูปภาพ
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    
    # ประมวลผล
    start_time = time.time()
    results = await client.detect_batch(image_paths, conf=0.3, iou=0.5)
    elapsed_time = time.time() - start_time
    
    # แสดงผลลัพธ์
    print(f"\nProcessed {len(image_paths)} images in {elapsed_time:.2f}s")
    for result in results:
        if result['success']:
            count = result['data']['count']
            print(f"  {result['image']}: {count} objects detected")
        else:
            print(f"  {result['image']}: Error - {result['error']}")


def example_detailed_results():
    """ตัวอย่างการดึงข้อมูลรายละเอียดจากผลลัพธ์"""
    print("\n" + "="*60)
    print("Example 4: Detailed Results")
    print("="*60)
    
    client = YOLOAPIClient()
    
    # ตรวจจับวัตถุ
    # result = client.detect('test.jpg', conf=0.25, iou=0.45)
    
    # แสดงรายละเอียด
    # print(f"\nImage Shape: {result['image_shape']}")
    # print(f"Total Detections: {result['count']}")
    # 
    # print("\nDetected Objects:")
    # for i, det in enumerate(result['detections'], 1):
    #     print(f"\n{i}. {det['class_name']}")
    #     print(f"   Confidence: {det['confidence']:.2%}")
    #     bbox = det['bbox']
    #     print(f"   BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) "
    #           f"to ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
    #     
    #     # คำนวณขนาดของ bounding box
    #     width = bbox['x2'] - bbox['x1']
    #     height = bbox['y2'] - bbox['y1']
    #     print(f"   Size: {width:.1f} x {height:.1f} pixels")


def example_error_handling():
    """ตัวอย่างการจัดการ errors"""
    print("\n" + "="*60)
    print("Example 5: Error Handling")
    print("="*60)
    
    client = YOLOAPIClient()
    
    def detect_with_retry(image_path, max_retries=3):
        """ตรวจจับวัตถุพร้อม retry logic"""
        for attempt in range(max_retries):
            try:
                result = client.detect(image_path)
                return result
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                print(f"  Attempt {attempt + 1} failed: {e}")
                print(f"  Retrying...")
                time.sleep(1)
    
    try:
        # result = detect_with_retry('test.jpg')
        # print(f"Success: {result['count']} objects detected")
        print("Example code ready (uncomment to use)")
    except Exception as e:
        print(f"Failed after retries: {e}")


def main():
    """Main function"""
    print("YOLO API Client Examples")
    print("="*60)
    
    # Example 1: Basic usage
    example_basic()
    
    # Example 2: Batch sequential
    # example_batch_sequential()
    
    # Example 3: Batch parallel (async)
    # asyncio.run(example_batch_parallel())
    
    # Example 4: Detailed results
    # example_detailed_results()
    
    # Example 5: Error handling
    # example_error_handling()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("Uncomment the example functions you want to try")
    print("="*60)


if __name__ == "__main__":
    main()
