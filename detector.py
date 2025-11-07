import os
import glob 
import cv2
import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional
import numpy as np

class ObjectDetector:
    def __init__(self, confidence_threshold: float = 0.5, model_path: Optional[str] = None):
        """
        Initialize the object detector with YOLOv5.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        # Determine which model file to use. Prefer an explicit `model_path`,
        # otherwise try the shipped 'yolov5m.pt'. If that file is missing,
        # pick the first local file matching 'yolov5*.pt' in the project.
        if model_path:
            model_file = model_path
        else:
            model_file = 'yolov5m.pt'

        if not os.path.isabs(model_file):
            # allow relative paths from project directory
            model_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_file)

        if not os.path.exists(model_file):
            # Search for any local yolov5*.pt files in the project folder
            search_dir = os.path.abspath(os.path.dirname(__file__))
            candidates = glob.glob(os.path.join(search_dir, 'yolov5*.pt'))
            if candidates:
                model_file = candidates[0]
                print(f"Using local model file: {model_file}")
            else:
                # Fall back to the original name (the ultralytics package will try to resolve/download)
                model_file = 'yolov5m.pt'
                print(f"Model file not found locally. Falling back to '{model_file}' (may download automatically).")

        self.model = YOLO(model_file)  # Using chosen model
        
    def detect(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        """
        Detect objects in a frame.
        
        Args:
            frame (np.ndarray): Input frame from webcam
            
        Returns:
            List[Tuple[str, float, Tuple[float, float, float, float]]]: 
                List of (object_name, confidence, bbox) tuples
                bbox is (x1, y1, x2, y2) normalized coordinates
        """
        try:
            # Run inference
            results = self.model(frame, verbose=False)[0]
            
            # Get frame dimensions for normalization
            height, width = frame.shape[:2]
            
            # Process results
            detections = []
            for box in results.boxes:
                confidence = float(box.conf[0])
                if confidence >= self.confidence_threshold:
                    class_id = int(box.cls[0])
                    class_name = results.names[class_id]
                    
                    # Get normalized bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, x2 = x1 / width, x2 / width
                    y1, y2 = y1 / height, y2 / height
                    
                    detections.append((class_name, confidence, (x1, y1, x2, y2)))
            
            return detections
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            return []
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # YOLO model cleanup is handled automatically
        pass 