# What to Do Next — Step-by-Step Guide

You have the code. Here’s the **exact order** of what to do so the project is ready for your IIT-H internship application.

---

## Step 1: Set up Python and run from project root

1. Open a terminal in the **project folder** (`optoguard` — the one that contains `optoguard/`, `models/`, `experiments/`, etc.).

2. Create and activate a virtual environment (if you don’t already have one):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
   (On Linux/Mac: `source .venv/bin/activate`)

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Important:** All experiment and demo commands below must be run **from this project root folder** (so that `models` and `experiments` are found).

---

## (Optional) Time crunch — use synthetic results

If you need the **master table and three plots** without running webcam experiments (e.g. for a quick report draft or demo), you can generate plausible synthetic CSVs:

```bash
python scripts/generate_synthetic_results.py
```

This writes five timestamped CSVs into `results/` (baseline, lighting_shift, occlusion, novel_objects, edge_deployment) with research-plausible values: uncertainty rises under distribution shift, and edge (RPi5) has higher latency. Then go to **Step 4** and run the analysis notebook. You can replace these later with real runs.

---

## Step 2: Run your first experiment (baseline)

This checks that the MC Dropout pipeline and webcam work.

1. In the project root, run:
   ```bash
   python -m experiments.run_baseline --num_frames 50 --num_samples 10
   ```
   (`--num_samples 10` keeps it fast; use 30 for real experiments.)

2. When the webcam window opens, point it at some common objects (bottle, phone, laptop, etc.) and let it run for 50 frames. Press `q` if you want to stop early.

3. After it finishes, look in the **`results/`** folder. You should see a new CSV file like `baseline_baseline_20250314_120000.csv`. Open it in Excel or a text editor — you should see rows with `condition`, `frame_idx`, `class`, `confidence_mean`, `confidence_std`, `latency_ms`, `hardware`.

If that works, the research pipeline is working.

---

## Step 3: Run all five experiments (on your laptop)

Run each of these **from the project root**. Each will open the webcam and write a new CSV into `results/`. You can do them on different days; just run the ones you need.

| What to run | Command | What you do |
|-------------|---------|-------------|
| **Baseline** | `python -m experiments.run_baseline --hardware laptop --num_frames 100` | Normal room, normal objects. |
| **Lighting** | `python -m experiments.run_lighting_shift --hardware laptop --lighting_level reduced --num_frames 100` | Dim the lights or turn some off, then run. |
| **Occlusion** | `python -m experiments.run_occlusion --hardware laptop --occlusion_level 50_percent --num_frames 100` | Partially cover objects (e.g. hand over half of a bottle). |
| **Novel objects** | `python -m experiments.run_novel_objects --hardware laptop --novelty_label novel --num_frames 100` | Show something not in COCO (e.g. a toy, unusual object). |
| **Edge (RPi5)** | Only if you have a Raspberry Pi 5: copy the project there, install deps, then run `python -m experiments.run_edge_deployment --hardware rpi5 --num_samples 20 --num_frames 50` | Same as baseline but on the Pi; copy the generated CSV back to your laptop’s `results/` folder. |

Use `--num_samples 30` for laptop when you want final numbers; use `--num_samples 10` for quick tests.

---

## Step 4: Get your results table and plots

1. Open Jupyter:
   ```bash
   jupyter notebook
   ```
   (Or use VS Code / Cursor: open `analysis/master_analysis.ipynb` and run the cells.)

2. In the notebook, run **all cells** in order. The notebook will:
   - Read every CSV in `results/`
   - Build a **master table** (one row per condition: uncertainty mean/std, latency, etc.)
   - Produce **three plots**:
     - Uncertainty vs occlusion level  
     - Uncertainty vs lighting condition  
     - Latency: laptop vs RPi5 (if you have both)

3. Export the table and figures:
   - Take a screenshot of the table, or export it to CSV from the notebook.
   - Save the three plots as PNGs (e.g. right‑click → Save image). You’ll paste these into your report.

---

## Step 5: Write the technical report

1. Open `paper/technical_report_outline.md`. It has the section headings (Introduction, Related Work, Methodology, Results, Discussion, Conclusion).

2. Fill it in using:
   - The **master table** from the notebook (paste or redraw as a table).
   - The **three plots** (insert as figures).
   - 1–2 sentences per plot: what the plot shows and what it means for uncertainty under shift / edge deployment.

3. In **Related Work**, add a short paragraph citing:
   - “Monte Carlo DropBlock for Modelling Uncertainty in Object Detection” (Kumari et al., incl. P.K. Srijith).

4. When you’re happy, convert the report to PDF (e.g. copy into a Word/Google Doc with IEEE-style formatting, or use LaTeX if you prefer) and save it in the `paper/` folder (e.g. `paper/optoguard_technical_report.pdf`).

---

## Step 6: Try the demo (optional but good for CV)

1. From the project root:
   ```bash
   python -m demo.gradio_app
   ```

2. A browser tab will open. Upload an image and you’ll get bounding boxes with **green / yellow / red** by uncertainty (low / medium / high). This is what you can later put on HuggingFace Spaces.

3. When you’re ready, you can create a **HuggingFace Space** (Gradio app), upload this project (or at least `demo/gradio_app.py`, `models/mc_dropout_yolov8.py`, and a `requirements.txt` for the Space), and link it in your README and CV.

---

## Step 7: Use it in your application

- **CV / application one-liner:**  
  *“OptoGuard: Uncertainty-aware object detection under distribution shift on edge hardware — MC Dropout on YOLOv8n, benchmarked across five conditions on laptop and Raspberry Pi 5, with technical report and demo.”*

- **Links to give:**  
  - GitHub repo (with README, `results/` CSVs or a summary, and the report in `paper/`).  
  - HuggingFace Space (if you deploy the Gradio app).  
  - Optional: link to the report PDF if you host it (e.g. in the repo or on Overleaf).

---

## Quick reference: where everything lives

| You want to… | Where to look |
|--------------|----------------|
| Run experiments | `python -m experiments.run_baseline` (etc.) from **project root** |
| Generate synthetic results (no webcam) | `python scripts/generate_synthetic_results.py` from **project root** |
| See raw results | `results/*.csv` |
| Build table + plots | `analysis/master_analysis.ipynb` |
| Write the report | `paper/technical_report_outline.md` → then export to PDF |
| Run the demo | `python -m demo.gradio_app` |
| Deploy on RPi5 | Run the same experiment scripts on the Pi, then copy CSVs to `results/` |

If something doesn’t work (e.g. “No module named 'models'”), make sure you’re in the **project root** (the folder that contains `models/` and `experiments/`) and that you activated the same environment where you ran `pip install -r requirements.txt`.
