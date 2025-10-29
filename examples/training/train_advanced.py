#!/usr/bin/env python3
"""
ตัวอย่างการ Train YOLO Model แบบละเอียด
Advanced YOLO Model Training Example with Custom Parameters

วิธีใช้งาน:
    python train_advanced.py --data data.yaml --model yolov8n --epochs 100

Features:
    - Custom training parameters
    - Learning rate scheduling
    - Data augmentation settings
    - Callbacks for monitoring
    - Model evaluation
"""

import argparse
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path


def load_config(config_path):
    """โหลด configuration จากไฟล์ YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model(args):
    """
    Train YOLO model with advanced settings
    
    Args:
        args: Command line arguments
    """
    # ตรวจสอบ GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        device = args.device if args.device != 'auto' else 0
    else:
        print("⚠ No GPU detected, using CPU")
        device = 'cpu'
    
    # ตรวจสอบ data.yaml
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data config not found: {args.data}")
    
    # แสดง dataset info
    data_config = load_config(args.data)
    print("\n" + "="*60)
    print("Dataset Information:")
    print("="*60)
    print(f"  Path: {data_config.get('path', 'N/A')}")
    print(f"  Classes: {data_config.get('nc', 'N/A')}")
    print(f"  Names: {data_config.get('names', 'N/A')}")
    
    # โหลดโมเดล
    model_name = f"{args.model}.pt"
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Training parameters
    train_params = {
        # Basic settings
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'name': args.name,
        'project': args.project,
        'device': device,
        'workers': args.workers,
        'patience': args.patience,
        'save': True,
        'save_period': args.save_period,
        'pretrained': True,
        'optimizer': args.optimizer,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'amp': True,  # Automatic Mixed Precision
        
        # Learning rate settings
        'lr0': args.lr0,           # initial learning rate
        'lrf': args.lrf,           # final learning rate (lr0 * lrf)
        'momentum': 0.937,         # SGD momentum
        'weight_decay': 0.0005,    # optimizer weight decay
        'warmup_epochs': 3.0,      # warmup epochs
        'warmup_momentum': 0.8,    # warmup momentum
        'warmup_bias_lr': 0.1,     # warmup bias learning rate
        
        # Loss weights
        'box': 7.5,                # box loss gain
        'cls': 0.5,                # cls loss gain
        'dfl': 1.5,                # dfl loss gain
        
        # Augmentation parameters
        'hsv_h': 0.015,            # HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,              # HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,              # HSV-Value augmentation (fraction)
        'degrees': 0.0,            # rotation (+/- deg)
        'translate': 0.1,          # translation (+/- fraction)
        'scale': 0.5,              # scaling (+/- gain)
        'shear': 0.0,              # shear (+/- deg)
        'perspective': 0.0,        # perspective (+/- fraction)
        'flipud': 0.0,             # flip up-down probability
        'fliplr': 0.5,             # flip left-right probability
        'mosaic': 1.0,             # mosaic augmentation probability
        'mixup': 0.0,              # mixup augmentation probability
        'copy_paste': 0.0,         # copy-paste augmentation probability
        
        # Other settings
        'close_mosaic': 10,        # ปิด mosaic augmentation ในช่วงท้าย
        'resume': args.resume,     # resume from checkpoint
    }
    
    # แสดง training parameters
    print("\n" + "="*60)
    print("Training Parameters:")
    print("="*60)
    for key, value in train_params.items():
        print(f"  {key}: {value}")
    
    # เริ่ม training
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
    
    # Validate โมเดล
    print("\n" + "="*60)
    print("Model Validation")
    print("="*60)
    
    metrics = model.val()
    
    print("\nFinal Metrics:")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print(f"  Fitness:   {metrics.fitness:.4f}")
    
    # บันทึก metrics
    metrics_dict = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'fitness': float(metrics.fitness),
    }
    
    metrics_file = Path(results.save_dir) / 'final_metrics.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics_dict, f)
    print(f"\nMetrics saved to: {metrics_file}")
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description='Advanced YOLO Training')
    
    # Dataset
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to data.yaml')
    
    # Model
    parser.add_argument('--model', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='Model size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: 0, 1, 2, cpu, auto')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW', 'RMSProp'],
                        help='Optimizer')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='Final learning rate (lr0 * lrf)')
    
    # Output
    parser.add_argument('--name', type=str, default='yolo_advanced',
                        help='Experiment name')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    
    args = parser.parse_args()
    
    # Train model
    model, results = train_model(args)
    
    print("\n" + "="*60)
    print("✓ Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
