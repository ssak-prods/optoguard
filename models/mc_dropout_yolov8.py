import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class MCDropoutConfig:
    """
    Configuration for Monte Carlo Dropout inference with YOLOv8.

    This wrapper keeps dropout-like layers active at inference time and
    performs multiple stochastic forward passes to approximate a Bayesian
    posterior over detections.
    """
    model_path: str = "yolov8n.pt"
    num_samples: int = 30
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.5
    seed: int = 42


class MCDropoutYoloV8:
    """
    Monte Carlo Dropout wrapper around a YOLOv8 model.

    For each input frame:
      - run the detector `num_samples` times with dropout enabled
      - aggregate boxes across samples
      - compute mean confidence and standard deviation as an uncertainty proxy
    """

    def __init__(self, config: Optional[MCDropoutConfig] = None):
        self.config = config or MCDropoutConfig()

        # Resolve model path (allow both local file and model name)
        model_path = self.config.model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", model_path)
            model_path = os.path.abspath(model_path)

        # If the resolved path does not exist, fall back to the original string
        if not os.path.exists(model_path):
            model_path = self.config.model_path

        self.model = YOLO(model_path)

        # Put model in train mode to keep dropout active during inference
        self.model.model.train()

        # Disable gradient computation during inference
        for param in self.model.model.parameters():
            param.requires_grad = False

        # Fix random seeds for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _run_single(self, image: np.ndarray):
        """Run a single stochastic forward pass."""
        with torch.no_grad():
            results = self.model(
                image,
                verbose=False,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
            )[0]
        return results

    def infer(
        self, image: np.ndarray
    ) -> List[Tuple[str, float, float, Tuple[float, float, float, float]]]:
        """
        Run MC Dropout inference on a single image.

        Returns a list of detections in the form:
            (class_name, mean_confidence, std_confidence, (x1, y1, x2, y2))
        where the box coordinates are normalized to [0, 1].
        """
        height, width = image.shape[:2]

        # Collect all detections across samples
        per_sample_detections: List[List[Tuple[int, float, Tuple[float, float, float, float]]]] = []

        for _ in range(self.config.num_samples):
            results = self._run_single(image)
            sample_dets: List[Tuple[int, float, Tuple[float, float, float, float]]] = []

            for box in results.boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = x1 / width, x2 / width
                y1, y2 = y1 / height, y2 / height
                sample_dets.append((class_id, conf, (x1, y1, x2, y2)))

            per_sample_detections.append(sample_dets)

        if not per_sample_detections:
            return []

        # Simple aggregation strategy:
        # - assume detections are reasonably consistent across samples
        # - group by (class_id, rounded box coordinates)
        groups: Dict[Tuple[int, int, int, int, int], List[float]] = {}

        for sample in per_sample_detections:
            for class_id, conf, (x1, y1, x2, y2) in sample:
                key = (
                    class_id,
                    int(x1 * 1000),
                    int(y1 * 1000),
                    int(x2 * 1000),
                    int(y2 * 1000),
                )
                groups.setdefault(key, []).append(conf)

        detections: List[Tuple[str, float, float, Tuple[float, float, float, float]]] = []
        names = self.model.model.names

        for (class_id, x1k, y1k, x2k, y2k), confs in groups.items():
            if not confs:
                continue
            confs_arr = np.array(confs, dtype=np.float32)
            mean_conf = float(confs_arr.mean())
            std_conf = float(confs_arr.std())

            if mean_conf < self.config.confidence_threshold:
                continue

            x1 = x1k / 1000.0
            y1 = y1k / 1000.0
            x2 = x2k / 1000.0
            y2 = y2k / 1000.0

            class_name = names.get(class_id, str(class_id))
            detections.append((class_name, mean_conf, std_conf, (x1, y1, x2, y2)))

        return detections


__all__ = ["MCDropoutConfig", "MCDropoutYoloV8"]

