# RQ4b — Relation-semantic mismatch control (circular → keyed)

**Question.** Where does profile reuse fail under semantic mismatch?

**Design.** A frozen **circular-insertion** profile is applied to **keyed** insertion. The geometric axis changes from a symmetry (yaw is a gauge) to a constraint (yaw is selective), so the yaw channel of the transferred profile is wrong while the non-yaw channels still transfer. The headline result is a `13.36°` full-orientation error on the yaw intervention.

**Data.** The frozen law `real_relation_consistency/full_se3_law/frozen_full_se3.npz` is shipped. The `keyed_seed_*.h5` rollouts (three seeds) are regenerated with `collect_se3_rollouts.py`; re-deriving the frozen law also needs `circular_honest_seed_*.h5`.

## Pipeline

### 1. (Optional) Re-derive the frozen circular law

Only needed if you want to re-fit `frozen_full_se3.npz` from scratch; otherwise use the shipped `.npz`.

```bash
# regenerate the three circular-honest seeds first (see rq2_se3_structure.md)
python -u -B real_relation_consistency/identify_full_se3.py
```

This fits `SE3SmoothFinitePDiagModel` on `phase_switch_symmetry_rollouts_se3/circular_honest_seed_*.h5` and writes `full_se3_law/frozen_full_se3.npz`.

### 2. Apply the frozen law to keyed insertion

```bash
# requires the three keyed seeds
python -u -B real_relation_consistency/experiment_a_transfer.py
```

Writes `real_relation_consistency/experiment_a_transfer/summary.json`, which contains the per-generator held-out errors. The `yaw` row reports `orient_rmse_deg ≈ 13.36`.

Both scripts import the shared simulation library via the sibling `phase_switch_symmetry/` directory; the analysis-only environment (`requirements.txt`) is sufficient once the rollouts and frozen law exist.

## Paper mapping

- RQ4b text: transferring the circular profile to keyed insertion preserves non-yaw responses, but the yaw intervention produces `13.36°` full-orientation error (`axis_rmse_deg ≈ 0.60°`); profile reuse requires matching generator semantics.
