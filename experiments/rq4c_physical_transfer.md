# RQ4c — Physical frozen response-shape transfer (10° ↔ 20°)

**Question.** Does a source response profile predict cylinder-axis rotation under a change in physical fixture geometry?

**Design.** Three operators contributed 50 upright, 61 ten-degree, and 52 twenty-degree demonstrations on a Franka platform. For each transfer direction, the mean normalized profile `ᾱ(s)` is estimated from the source demonstrations + upright reference; a disjoint target calibration set supplies the TCP-derived geometry `(c, u)`; the held-out target cylinder-axis response is then predicted as `ω̂(s) = u · c · ᾱ(s)` with the source profile frozen. The `10°→20°` direction uses 11 calibration + 41 evaluation episodes; the reverse uses 13 + 48.

**Data.** Shipped fully (~5 MB) under `real_relation_consistency/tcp_0_10_to_20/`:
- `fk_episodes.csv`, `metadata_audit.csv` — episode-level forward-kinematics metadata.
- `fk_cache/insert_jc_fix_3_*.npz` (50 files) — upright TCP FK cache.
- `observations/obs_insert_10degree.npz`, `obs_insert_20degree.npz` — tilted cylinder-axis observations.

The raw robot datasets (Franka joint trajectories and camera streams) are private data and are not distributed during anonymous review; the derived FK cache and observations above are sufficient for this analysis.

## Pipeline

The headline script has no command-line arguments; it reads its inputs relative to its own directory and writes everything under `calibration_split/`.

```bash
# analysis-only environment is sufficient (numpy/scipy/pandas/matplotlib)
python -u -B real_relation_consistency/tcp_0_10_to_20/calibration_split_transfer.py
```

Outputs (in `real_relation_consistency/tcp_0_10_to_20/calibration_split/`):
- `summary.csv` — per-method terminal and full-progress errors for both directions.
- `figure_curves.csv`, `predictions.npz` — the response curves and per-episode predictions.
- `fig_physical_transfer_single_column.png/.pdf` — the publication figure.
- `verification.json`, `provenance.json` — numerical checks and input hashes.

### Re-computing the FK cache / observations (if needed)

The upstream pipeline (`step1_audit_fk.py`, `step2_model.py`) re-derives the FK cache and observations from the raw robot datasets. It is not required for the shipped analysis and is not included in this anonymous release.

## Paper mapping

- Fig. 4 (physical frozen response-shape transfer): measured cylinder-axis responses vs. frozen-source and target-calibrated predictions, and the held-out terminal / full-progress response-vector errors.
- Table III (physical response-vector errors, degrees): frozen-source terminal `4.13 ± 2.04°` / `3.31 ± 1.65°`, full-progress `5.20°` / `4.48°`, vs. identity `6.66°` / `5.31°`, linear `8.59°` / `5.47°`, and target-calibrated `4.62°` / `4.26°`.
