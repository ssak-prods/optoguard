# OptoGuard MC Dropout YOLOv8n — Model Card Notes

## Intended use

- Uncertainty-aware object detection in surveillance and assistive-vision style scenarios.
- Research on how prediction uncertainty behaves under distribution shift and edge deployment constraints.

## Model

- Base detector: YOLOv8n pretrained on COCO (80 classes).
- Inference modified with Monte Carlo Dropout:
  - Keep dropout-like layers active during inference.
  - Run N stochastic forward passes per image.
  - For each detection, compute mean confidence and standard deviation (σ) across passes.
  - Use σ as an epistemic uncertainty proxy.

## Data

- COCO dataset (pretraining; not modified in this project).
- Evaluation data consists of real-world webcam scenes under:
  - Baseline indoor conditions.
  - Lighting shift (reduced illumination, backlighting).
  - Occlusion (partial blocking of objects).
  - Novel objects (out-of-distribution for COCO).
  - Edge deployment on Raspberry Pi 5.

## Metrics

- Per-detection: mean confidence, standard deviation (uncertainty).
- Per-condition: aggregated uncertainty statistics and mean latency.

## Limitations

- MC Dropout is only an approximation to full Bayesian inference.
- No retraining or fine-tuning on domain-specific data.
- Latency on edge devices depends strongly on hardware and optimization (e.g. quantization).

## Safety and interpretation

- High σ indicates model disagreement and should be treated as \"model is unsure\".
- For safety-critical settings, combine uncertainty estimates with conservative decision rules or human-in-the-loop review.

