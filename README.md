# OptoGuard: Uncertainty-Aware Object Detection under Distribution Shift on Edge Hardware

OptoGuard is a research-grade reworking of a real-time scene watchdog into an **uncertainty-aware object detection system** designed for **edge hardware** such as the Raspberry Pi 5. It implements **Monte Carlo Dropout** on YOLOv8n and evaluates how prediction uncertainty behaves under controlled **distribution shift** conditions and hardware constraints.

The central research question is:

> How does prediction uncertainty in object detection models evolve under real-world distribution shift when deployed on edge hardware — and can uncertainty estimation serve as a reliable proxy for model degradation in constrained surveillance environments?

This aligns with research on uncertainty in object detection and continual learning over evolving domains, including work by Srijith P.K. and collaborators.

---

## Research Overview

- **Base model**: YOLOv8n (Ultralytics) for efficient edge deployment.
- **Uncertainty estimation**: Monte Carlo Dropout during inference, with multiple stochastic forward passes per frame.
- **Per-detection metrics**: mean confidence and standard deviation (σ) across MC samples; σ is treated as an epistemic uncertainty proxy.
- **Hardware focus**: comparative evaluation on a laptop and Raspberry Pi 5.

The project is structured around five controlled experimental conditions:

1. **Baseline** — normal indoor lighting, familiar objects.
2. **Lighting shift** — reduced / altered lighting (e.g., dim room, backlighting).
3. **Occlusion** — partial blocking of objects at specified occlusion levels.
4. **Novel objects** — objects that are out-of-distribution relative to COCO.
5. **Edge deployment** — replicated experiments on Raspberry Pi 5 with latency logging.

For each condition, OptoGuard logs:

- Detected class.
- Mean confidence across MC samples.
- Standard deviation (uncertainty estimate).
- Inference latency per frame.
- Hardware identifier (laptop vs. RPi5).

These logs feed into an analysis notebook that produces a **master results table** and plots relating uncertainty to distribution shift and hardware.

---

## Repository Structure (v2)

At a high level:

- `optoguard/` — legacy package for real-time watchdog and TTS (preserved).
- `models/` — MC Dropout–enabled YOLOv8n wrapper (`mc_dropout_yolov8.py`).
- `experiments/` — scripts for each experimental condition:
  - `run_baseline.py`
  - `run_lighting_shift.py`
  - `run_occlusion.py`
  - `run_novel_objects.py`
  - `run_edge_deployment.py`
- `results/` — CSV logs produced by experiments (timestamped).
- `analysis/` — Jupyter notebook(s) for aggregation and plotting (`master_analysis.ipynb`).
- `deploy/` — Raspberry Pi 5 deployment and setup scripts (to be populated).
- `paper/` — technical report outline and eventual IEEE-style write-up.
- `demo/` — Gradio demo and model card notes for HuggingFace-style deployment.

The original `optoguard/main.py`, `detector.py`, `watchdog.py`, `utils.py`, and `speaker.py` remain available for real-time demos, now complemented by the research pipeline.

---

## Running the Experiments

From the project root:

- **Baseline (laptop):**

  ```bash
  python -m experiments.run_baseline --hardware laptop --num_frames 200
  ```

- **Lighting shift:**

  ```bash
  python -m experiments.run_lighting_shift --hardware laptop --lighting_level reduced
  ```

- **Occlusion:**

  ```bash
  python -m experiments.run_occlusion --hardware laptop --occlusion_level 50_percent
  ```

- **Novel objects:**

  ```bash
  python -m experiments.run_novel_objects --hardware laptop --novelty_label novel_object
  ```

- **Edge deployment (on Raspberry Pi 5):**

  ```bash
  python -m experiments.run_edge_deployment --hardware rpi5 --num_samples 20
  ```

Each script will create a timestamped CSV in `results/` with per-frame, per-detection statistics for the specified condition.

---

## Analysis

After running the desired experiments:

1. Open the notebook in `analysis/master_analysis.ipynb`.
2. Run the cells to:
   - Aggregate all CSVs into a single dataframe.
   - Build a **master results table** summarizing uncertainty and latency per condition.
   - Plot:
     - Uncertainty vs. occlusion level.
     - Uncertainty vs. lighting condition.
     - Latency comparison (laptop vs. Raspberry Pi 5) per condition.

These artifacts form the empirical core of the project and support the internship narrative.

---

## Gradio Demo (for HuggingFace)

The `demo/gradio_app.py` script defines a Gradio interface that:

- Accepts an uploaded image.
- Runs MC Dropout YOLOv8n to obtain multiple stochastic detections.
- Displays:
  - The image with bounding boxes.
  - Per-detection text summaries with mean confidence and σ.
- Colour codes boxes by uncertainty:
  - Green — low uncertainty.
  - Yellow — medium uncertainty.
  - Red — high uncertainty.

Locally, you can launch it with:

```bash
python -m demo.gradio_app
```

For HuggingFace Spaces, the same script can be used as the `app.py` entrypoint.

Model card notes for documenting methodology, data, and limitations are in `demo/MODEL_CARD_NOTES.md`.

---

## Technical Report

The `paper/technical_report_outline.md` file contains a structured outline for a 4–6 page IEEE-style report with sections:

1. Introduction.
2. Related Work.
3. Methodology.
4. Results.
5. Discussion.
6. Conclusion.

The report will be completed from real experimental runs and the analysis notebook outputs, and included in `paper/` as a PDF.
