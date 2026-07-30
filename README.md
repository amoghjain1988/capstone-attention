# Is attention a faithful interaction graph?

Auditing a transformer's social attention against motion-only baselines (Constant
Velocity, EKF) on pedestrian trajectories. Read attention as an N×N interaction graph,
delete an edge, re-predict, and measure the shift Δ. **Faithfulness Index = corr(weight, Δ)**:
do heavy edges move the prediction (faithful) or merely decorate?

## Interface Contract
- `capstone.ipynb` — the one notebook; orchestration only, runs end-to-end on synthetic data.
- `src/` — all logic, swappable behind the Protocols in `src/models/base.py`:
  - `data/` loader · windows · synthetic generator (with a known interaction graph)
  - `models/` CV · EKF · mock · cached · **agentformer (the swap point)**
  - `metrics/` accuracy (ADE/FDE) · collision · calibration (ECE)
  - `faithfulness/` graph · ablation · index · validation (AUROC)
  - `stats/` scene-clustered bootstrap · H1–H4 hypothesis tests
  - `eda/` density / step-size / usable-window summaries
- `scripts/` AgentFormer setup + extract-once caching.


