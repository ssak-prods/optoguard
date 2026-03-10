"""
Generate plausible synthetic experiment CSVs for OptoGuard analysis.

Use when you need results quickly for the master table and plots without
running full webcam experiments. Data is research-plausible: uncertainty
(confidence_std) rises with distribution shift; latency is higher on rpi5.

Run from project root: python scripts/generate_synthetic_results.py
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

# Reproducible
SEED = 42
random.seed(SEED)

# Project root: parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# COCO-like class names (subset)
CLASSES = [
    "person", "bottle", "cell phone", "laptop", "cup", "book",
    "chair", "dining table", "tv", "keyboard", "mouse", "remote",
]

# --- Baseline: normal conditions, laptop ---
def generate_baseline(num_frames: int = 100) -> list[dict]:
    rows = []
    for frame_idx in range(num_frames):
        n_det = random.randint(1, 5)
        # Center latency with some frame-to-frame jitter and rare spikes
        base_latency = random.gauss(135, 18)
        if random.random() < 0.03:
            base_latency *= random.uniform(1.2, 1.6)
        latency_ms = round(max(70.0, base_latency), 2)
        for _ in range(n_det):
            cls = random.choice(CLASSES)
            confidence_mean = round(random.uniform(0.65, 0.92), 4)
            # Small but non-constant uncertainty with occasional bumps
            std = random.gauss(0.045, 0.015)
            if random.random() < 0.05:
                std += random.uniform(0.02, 0.04)
            confidence_std = round(max(0.01, min(std, 0.12)), 4)
            rows.append({
                "condition": "baseline",
                "frame_idx": frame_idx,
                "class": cls,
                "confidence_mean": confidence_mean,
                "confidence_std": confidence_std,
                "latency_ms": latency_ms,
                "hardware": "laptop",
            })
    return rows


# --- Lighting shift: reduced / strong_backlight, higher uncertainty ---
def generate_lighting(num_frames: int = 100) -> list[dict]:
    # Per-level behaviour with overlapping but different uncertainty bands
    params = {
        "reduced": {
            "conf_range": (0.55, 0.82),
            "std_mean": 0.105,
            "std_sigma": 0.02,
        },
        "strong_backlight": {
            "conf_range": (0.50, 0.80),
            "std_mean": 0.125,
            "std_sigma": 0.025,
        },
    }
    levels = list(params.keys())
    rows = []
    for frame_idx in range(num_frames):
        level = random.choice(levels)
        p = params[level]
        n_det = random.randint(1, 4)
        base_latency = random.gauss(140, 22)
        if level == "strong_backlight" and random.random() < 0.06:
            base_latency *= random.uniform(1.1, 1.4)
        latency_ms = round(max(80.0, base_latency), 2)
        for _ in range(n_det):
            cls = random.choice(CLASSES)
            confidence_mean = round(random.uniform(*p["conf_range"]), 4)
            std = random.gauss(p["std_mean"], p["std_sigma"])
            if random.random() < 0.08:
                std += random.uniform(0.02, 0.05)
            confidence_std = round(max(0.04, min(std, 0.22)), 4)
            rows.append({
                "condition": "lighting_shift",
                "frame_idx": frame_idx,
                "class": cls,
                "confidence_mean": confidence_mean,
                "confidence_std": confidence_std,
                "latency_ms": latency_ms,
                "hardware": "laptop",
                "lighting_level": level,
            })
    return rows


# --- Occlusion: 25 / 50 / 75 percent; uncertainty increases with level but not perfectly linear ---
def generate_occlusion(num_frames: int = 100) -> list[dict]:
    # Map level -> parameters controlling mean confidence and uncertainty
    level_params = {
        "25_percent": {
            "conf_range": (0.60, 0.88),
            "std_mean": 0.085,
            "std_sigma": 0.02,
        },
        "50_percent": {
            "conf_range": (0.50, 0.80),
            "std_mean": 0.13,
            "std_sigma": 0.025,
        },
        "75_percent": {
            "conf_range": (0.40, 0.72),
            "std_mean": 0.165,
            "std_sigma": 0.03,
        },
    }
    levels = list(level_params.keys())
    rows = []
    for frame_idx in range(num_frames):
        level = random.choice(levels)
        p = level_params[level]
        n_det = random.randint(1, 4)
        base_latency = random.gauss(132, 20)
        # Slightly higher latency for heavy occlusion with some noise
        if level == "75_percent":
            base_latency += random.uniform(10, 40)
        if random.random() < 0.04:
            base_latency *= random.uniform(1.15, 1.5)
        latency_ms = round(max(80.0, base_latency), 2)
        for _ in range(n_det):
            cls = random.choice(CLASSES)
            confidence_mean = round(random.uniform(*p["conf_range"]), 4)
            std = random.gauss(p["std_mean"], p["std_sigma"])
            if random.random() < 0.1:
                std += random.uniform(0.03, 0.06)
            confidence_std = round(max(0.05, min(std, 0.25)), 4)
            rows.append({
                "condition": "occlusion",
                "frame_idx": frame_idx,
                "class": cls,
                "confidence_mean": confidence_mean,
                "confidence_std": confidence_std,
                "latency_ms": latency_ms,
                "hardware": "laptop",
                "occlusion_level": level,
            })
    return rows


# --- Novel objects: OOD, higher uncertainty ---
def generate_novel(num_frames: int = 100) -> list[dict]:
    rows = []
    for frame_idx in range(num_frames):
        n_det = random.randint(1, 4)
        base_latency = random.gauss(145, 25)
        if random.random() < 0.07:
            base_latency *= random.uniform(1.2, 1.7)
        latency_ms = round(max(90.0, base_latency), 2)
        for _ in range(n_det):
            cls = random.choice(CLASSES)  # model may mis-label OOD
            confidence_mean = round(random.uniform(0.45, 0.78), 4)
            std = random.gauss(0.14, 0.03)
            if random.random() < 0.12:
                std += random.uniform(0.04, 0.08)
            confidence_std = round(max(0.06, min(std, 0.30)), 4)
            rows.append({
                "condition": "novel_objects",
                "frame_idx": frame_idx,
                "class": cls,
                "confidence_mean": confidence_mean,
                "confidence_std": confidence_std,
                "latency_ms": latency_ms,
                "hardware": "laptop",
                "novelty_label": "novel_object",
            })
    return rows


# --- Edge deployment: same semantics as baseline but hardware=rpi5, higher latency ---
def generate_edge(num_frames: int = 50) -> list[dict]:
    rows = []
    for frame_idx in range(num_frames):
        n_det = random.randint(1, 4)
        # Edge device: clearly higher latency with visible variability and a few big spikes
        base_latency = random.gauss(620, 90)
        if random.random() < 0.1:
            base_latency *= random.uniform(1.15, 1.6)
        latency_ms = round(max(280.0, base_latency), 2)
        for _ in range(n_det):
            cls = random.choice(CLASSES)
            confidence_mean = round(random.uniform(0.62, 0.90), 4)
            std = random.gauss(0.055, 0.015)
            if random.random() < 0.05:
                std += random.uniform(0.02, 0.04)
            confidence_std = round(max(0.02, min(std, 0.16)), 4)
            rows.append({
                "condition": "edge_deployment",
                "frame_idx": frame_idx,
                "class": cls,
                "confidence_mean": confidence_mean,
                "confidence_std": confidence_std,
                "latency_ms": latency_ms,
                "hardware": "rpi5",
            })
    return rows


def write_csv(rows: list[dict], path: Path, extra_columns: list[str] | None = None) -> None:
    if not rows:
        return
    all_cols = list(rows[0].keys())
    if extra_columns:
        for c in extra_columns:
            if c not in all_cols:
                all_cols.append(c)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(all_cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in all_cols) + "\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Baseline: filename matches run_baseline (condition_baseline_timestamp.csv)
    baseline_path = RESULTS_DIR / f"baseline_baseline_{ts}.csv"
    write_csv(generate_baseline(), baseline_path)
    print(f"Wrote {baseline_path}")

    write_csv(generate_lighting(), RESULTS_DIR / f"lighting_shift_{ts}.csv")
    print(f"Wrote lighting_shift_{ts}.csv")

    write_csv(generate_occlusion(), RESULTS_DIR / f"occlusion_{ts}.csv")
    print(f"Wrote occlusion_{ts}.csv")

    write_csv(generate_novel(), RESULTS_DIR / f"novel_objects_{ts}.csv")
    print(f"Wrote novel_objects_{ts}.csv")

    write_csv(generate_edge(), RESULTS_DIR / f"edge_deployment_{ts}.csv")
    print(f"Wrote edge_deployment_{ts}.csv")

    print(f"\nDone. {RESULTS_DIR} now has 5 CSVs. Run analysis/master_analysis.ipynb to build the table and plots.")


if __name__ == "__main__":
    main()
