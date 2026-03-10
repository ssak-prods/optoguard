---
title: "OptoGuard: Uncertainty-Aware Object Detection under Distribution Shift on Edge Hardware"
author: "Suhaib Ahmed"
---

## 1. Introduction

- Motivation: overconfident object detectors in safety-critical, resource-constrained deployments (edge devices, surveillance, assistive tech).
- Problem framing: how prediction uncertainty behaves under real-world distribution shift and hardware constraints.
- Contribution summary (what OptoGuard evaluates and shows).

## 2. Related Work

- Monte Carlo Dropout and Monte Carlo DropBlock for uncertainty in object detection (including Srijith P.K.'s work).
- Uncertainty estimation frameworks and probabilistic object detection.
- Continual learning and evolving domains (e.g., EvoCL-style scenarios).

## 3. Methodology

- Base model: YOLOv8n for edge deployment.
- MC Dropout wrapper: enabling stochastic inference with N samples, computing mean and std per detection.
- Experimental protocol: five conditions (baseline, lighting shift, occlusion, novel objects, edge deployment on Raspberry Pi 5).
- Logged metrics and evaluation procedure.

## 4. Results

- Master results table summarizing uncertainty and latency across conditions and hardware.
- Plot 1: Uncertainty vs. occlusion percentage.
- Plot 2: Uncertainty vs. lighting condition.
- Plot 3: Latency comparison between laptop and Raspberry Pi 5.

## 5. Discussion

- Interpretation of how uncertainty tracks degradation under distribution shift.
- Implications for deploying object detectors on edge hardware.
- Limitations and potential extensions (e.g., MC-DropBlock, continual adaptation).

## 6. Conclusion

- Key findings.
- How this informs safer, uncertainty-aware edge vision systems.

