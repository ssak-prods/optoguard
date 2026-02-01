import argparse
import os
import time
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np

from models.mc_dropout_yolov8 import MCDropoutYoloV8, MCDropoutConfig


def setup_output_path(results_dir: str, condition: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{condition}_{timestamp}.csv"
    return os.path.join(results_dir, filename)


def write_header(path: str) -> None:
    header = (
        "condition,frame_idx,class,confidence_mean,confidence_std,"
        "latency_ms,hardware,novelty_label\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)


def log_detections(
    path: str,
    condition: str,
    frame_idx: int,
    detections: List[Tuple[str, float, float, Tuple[float, float, float, float]]],
    latency_ms: float,
    hardware: str,
    novelty_label: str,
) -> None:
    if not detections:
        return
    with open(path, "a", encoding="utf-8") as f:
        for cls_name, mean_conf, std_conf, _ in detections:
            f.write(
                f"{condition},{frame_idx},{cls_name},{mean_conf:.4f},"
                f"{std_conf:.4f},{latency_ms:.2f},{hardware},{novelty_label}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Novel-objects condition with MC Dropout YOLOv8."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results"),
        help="Directory to store CSV results.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=100,
        help="Number of frames to process from the webcam.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="laptop",
        help="Hardware identifier to log (e.g., laptop, rpi5).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of MC Dropout samples per frame.",
    )
    parser.add_argument(
        "--novelty_label",
        type=str,
        default="novel_object",
        help="Label describing the out-of-distribution object(s) shown.",
    )
    args = parser.parse_args()

    condition = "novel_objects"
    results_dir = os.path.abspath(args.results_dir)
    output_path = setup_output_path(results_dir, condition)
    write_header(output_path)

    config = MCDropoutConfig(num_samples=args.num_samples)
    mc_model = MCDropoutYoloV8(config=config)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam")

    print(f"[{condition}] Experiment started. Logging to {output_path}")
    print("Show objects that are outside COCO's usual distribution or unusual for the environment.")

    frame_idx = 0
    try:
        while frame_idx < args.num_frames:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            start = time.time()
            detections = mc_model.infer(frame)
            end = time.time()
            latency_ms = (end - start) * 1000.0

            log_detections(
                output_path,
                condition,
                frame_idx,
                detections,
                latency_ms,
                args.hardware,
                args.novelty_label,
            )

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[{condition}] Experiment finished after {frame_idx} frames.")


if __name__ == "__main__":
    main()

