# Experiment results

CSV files here are produced by the experiment scripts in `../experiments/` (one per condition: baseline, lighting_shift, occlusion, novel_objects, edge_deployment).

You can also populate this folder quickly using **synthetic results** (no webcam):

```bash
python scripts/generate_synthetic_results.py
```

Then run `analysis/master_analysis.ipynb` to build the master table and plots.
