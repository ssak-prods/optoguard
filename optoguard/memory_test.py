import psutil
import os
import gc
import cv2
import numpy as np

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Initial memory: {get_memory_usage():.2f} MB")

from detector import ObjectDetector

print(f"After import: {get_memory_usage():.2f} MB")

detector = ObjectDetector()

print(f"After loading model: {get_memory_usage():.2f} MB")

# Create a dummy frame
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

print(f"Before inference: {get_memory_usage():.2f} MB")

detections = detector.detect(frame)

print(f"After inference: {get_memory_usage():.2f} MB")
print(f"Detections: {len(detections)}")

# Force garbage collection
gc.collect()

print(f"After GC: {get_memory_usage():.2f} MB")