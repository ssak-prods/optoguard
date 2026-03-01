import os
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np

from models.mc_dropout_yolov8 import MCDropoutYoloV8, MCDropoutConfig


mc_config = MCDropoutConfig(num_samples=20)
mc_model = MCDropoutYoloV8(config=mc_config)


def predict(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Run MC Dropout YOLOv8 on an input image and return:
      - image with bounding boxes drawn
      - per-detection text summaries including mean and std.
    """
    if image is None:
        raise ValueError("No image provided")

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detections = mc_model.infer(img_bgr)

    h, w = image.shape[:2]
    annotated = image.copy()
    summaries: List[str] = []

    for cls_name, mean_conf, std_conf, (x1, y1, x2, y2) in detections:
        x1_px, y1_px = int(x1 * w), int(y1 * h)
        x2_px, y2_px = int(x2 * w), int(y2 * h)

        # Colour code by uncertainty: green=low, yellow=medium, red=high
        if std_conf < 0.05:
            color = (0, 255, 0)
            level = "low"
        elif std_conf < 0.12:
            color = (255, 255, 0)
            level = "medium"
        else:
            color = (255, 0, 0)
            level = "high"

        cv2.rectangle(annotated, (x1_px, y1_px), (x2_px, y2_px), color, 2)
        label = f"{cls_name} {mean_conf:.2f} (σ={std_conf:.2f})"
        cv2.putText(
            annotated,
            label,
            (x1_px, max(0, y1_px - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        summaries.append(
            f"{cls_name}: mean={mean_conf:.2f}, std={std_conf:.2f} (uncertainty {level})"
        )

    return annotated, summaries


title = "OptoGuard: Uncertainty-Aware Object Detection"
description = """
Upload an image to see YOLOv8n predictions with Monte Carlo Dropout.
Each detection includes a mean confidence and standard deviation (σ) across
multiple stochastic forward passes; σ is visualized as:

- Green: low uncertainty
- Yellow: medium uncertainty
- Red: high uncertainty
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy", label="Detections"), gr.JSON(label="Per-detection stats")],
    title=title,
    description=description,
)


if __name__ == "__main__":
    demo.launch()

