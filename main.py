from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import io
from PIL import Image
from typing import List, Dict, Any
import os

app = FastAPI(
    title="YOLO Object Detection API",
    description="API for object detection using YOLO models",
    version="1.0.0"
)

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")
model = None

def load_model():
    """Load YOLO model on startup"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            # Download default model if not exists
            print(f"Model not found at {MODEL_PATH}, downloading default YOLOv8n model...")
            os.makedirs("models", exist_ok=True)
            model = YOLO("yolov8n.pt")
            model.save(MODEL_PATH)
            print(f"Default model downloaded and saved to {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on API startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "YOLO Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "POST - Upload image for object detection",
            "/health": "GET - Check API health status",
            "/model-info": "GET - Get current model information"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": MODEL_PATH,
        "model_type": str(type(model)),
        "names": model.names if hasattr(model, 'names') else None
    }

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45
) -> JSONResponse:
    """
    Detect objects in an uploaded image
    
    Args:
        file: Image file to process
        conf: Confidence threshold (default: 0.25)
        iou: IOU threshold for NMS (default: 0.45)
    
    Returns:
        JSON response with detected objects
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run inference
        results = model.predict(
            source=image,
            conf=conf,
            iou=iou,
            verbose=False
        )
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                }
                detections.append(detection)
        
        return JSONResponse(content={
            "success": True,
            "detections": detections,
            "count": len(detections),
            "image_shape": {
                "height": image.shape[0],
                "width": image.shape[1]
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
