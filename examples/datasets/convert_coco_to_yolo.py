#!/usr/bin/env python3
"""
สคริปต์สำหรับแปลงข้อมูลจาก COCO format เป็น YOLO format
Convert COCO format to YOLO format

วิธีใช้งาน:
    python convert_coco_to_yolo.py --input /path/to/coco --output /path/to/yolo
"""

import argparse
import json
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm


def convert_bbox_coco_to_yolo(bbox, img_width, img_height):
    """
    แปลง bounding box จาก COCO format เป็น YOLO format
    
    COCO format: [x_min, y_min, width, height]
    YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    
    Args:
        bbox: COCO bounding box [x_min, y_min, width, height]
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        YOLO bounding box [x_center, y_center, width, height]
    """
    x_min, y_min, width, height = bbox
    
    # คำนวณ center
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]


def convert_coco_to_yolo(coco_dir, output_dir, split='train'):
    """
    แปลง COCO dataset เป็น YOLO format
    
    Args:
        coco_dir: Path to COCO dataset
        output_dir: Output directory for YOLO format
        split: 'train' or 'val'
    """
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)
    
    # Paths
    ann_file = coco_dir / 'annotations' / f'instances_{split}2017.json'
    img_dir = coco_dir / f'{split}2017'
    
    # Output paths
    out_img_dir = output_dir / split / 'images'
    out_lbl_dir = output_dir / split / 'labels'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # โหลด COCO annotations
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # สร้าง mapping
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat for cat in coco_data['categories']}
    
    # จัดกลุ่ม annotations ตาม image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # แปลงแต่ละรูป
    print(f"Converting {len(images)} images...")
    for img_id, img_info in tqdm(images.items()):
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Copy image
        src_img = img_dir / img_filename
        dst_img = out_img_dir / img_filename
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        else:
            print(f"Warning: Image not found: {src_img}")
            continue
        
        # สร้าง label file
        label_filename = Path(img_filename).stem + '.txt'
        label_path = out_lbl_dir / label_filename
        
        # แปลง annotations
        if img_id in annotations_by_image:
            with open(label_path, 'w') as f:
                for ann in annotations_by_image[img_id]:
                    category_id = ann['category_id']
                    bbox = ann['bbox']
                    
                    # แปลง bbox เป็น YOLO format
                    yolo_bbox = convert_bbox_coco_to_yolo(
                        bbox, img_width, img_height
                    )
                    
                    # เขียน label (class_id x_center y_center width height)
                    # YOLO ใช้ class_id เริ่มจาก 0
                    class_id = category_id - 1  # COCO เริ่มจาก 1
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
    
    # สร้าง data.yaml
    data_yaml = output_dir / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write(f"path: {output_dir.absolute()}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n\n")
        f.write(f"nc: {len(categories)}\n")
        f.write(f"names:\n")
        for cat_id, cat in sorted(categories.items()):
            f.write(f"  {cat_id - 1}: {cat['name']}\n")
    
    print(f"\nConversion completed!")
    print(f"Output directory: {output_dir}")
    print(f"Data config: {data_yaml}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert COCO format to YOLO format'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to COCO dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for YOLO format'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val'],
        help='Dataset split to convert'
    )
    
    args = parser.parse_args()
    
    convert_coco_to_yolo(args.input, args.output, args.split)


if __name__ == "__main__":
    main()
