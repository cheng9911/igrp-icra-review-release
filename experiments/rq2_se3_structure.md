# RQ2 — Task-local generator structure in full SE(3)

**Question.** Under what basis / relation / execution conditions does a compact generator-response structure hold?

**Benchmarks.** Full-`SE(3)` structural controls (six generators `[du, dv, dw, roll, pitch, yaw]`), the multi-generator control, basis-rotation ablation, and planar transport (`heading_push` / `free_yaw_push`). The controller/planner/speed/spawn robustness controls reuse the keyed relation.

**Data.** Full `phase_switch_symmetry_rollouts_se3/` and `phase_switch_symmetry_rollouts_planar_push/` archives (three seeds per task) are regenerated with the collectors below. The repository ships only tiny smoke samples (`smoke_keyed.h5`, `smoke_circular.h5`, `multigen_smoke_seed_20260818.h5`) plus the frozen manifests in `phase_switch_symmetry_multiseed/`.

## Pipeline

### 1. Collect rollouts (three seeds each)

```bash
# keyed full-SE(3) task
for seed in 20260818 20270818 20280818; do
  python -u -B phase_switch_symmetry/collect_se3_rollouts.py \
    --output phase_switch_symmetry_rollouts_se3/keyed_seed_${seed}.h5 \
    --yaw-mode keyed --seed ${seed}
done

# circular honest gauge (yaw released)
for seed in 20260818 20270818 20280818; do
  python -u -B phase_switch_symmetry/collect_se3_rollouts.py \
    --output phase_switch_symmetry_rollouts_se3/circular_honest_seed_${seed}.h5 \
    --yaw-mode honest --seed ${seed}
done

# multi-generator release
for seed in 20260818 20270818 20280818; do
  python -u -B phase_switch_symmetry/collect_se3_multigen_rollouts.py \
    --output phase_switch_symmetry_rollouts_se3/multigen_seed_${seed}.h5 --seed ${seed}
done

# planar transport
for seed in 20260818 20270818 20280818; do
  python -u -B phase_switch_symmetry/collect_planar_push_rollouts.py \
    --output phase_switch_symmetry_rollouts_planar_push/heading_push_seed_${seed}.h5 \
    --arm heading_push --seed ${seed}
  python -u -B phase_switch_symmetry/collect_planar_push_rollouts.py \
    --output phase_switch_symmetry_rollouts_planar_push/free_yaw_push_seed_${seed}.h5 \
    --arm free_yaw_push --seed ${seed}
done
```

The `--context-manifest` defaults to `phase_switch_symmetry_multiseed/se3_fixed_contexts.json` (shipped), so the frozen intervention design is reproduced exactly.

### 2. Benchmarks

```bash
# within-task generator selectivity (M1-M4) + SE(3) transfer
python -u -B phase_switch_symmetry/benchmark_se3_transfer.py

# multi-generator release
python -u -B phase_switch_symmetry/benchmark_se3_multigen.py

# basis-rotation ablation
python -u -B phase_switch_symmetry/benchmark_se3_basis_ablation.py

# planar transport
python -u -B phase_switch_symmetry/benchmark_planar_push.py
```

All four read their experiment/subset manifests from `phase_switch_symmetry_multiseed/` (shipped) and write their results (fits CSV, profiles, summary) to `phase_switch_symmetry_multiseed/se3_transfer|se3_multigen|se3_transfer_basis_ablation|planar_push/`. `benchmark_se3_multigen.py` additionally reuses `phase_switch_symmetry_multiseed/se3_transfer/se3_transfer_fits.csv`, so run `benchmark_se3_transfer.py` first.

## Paper mapping

- Fig. 3 (task-local generator structure): progress-mean relevance, insertion/yaw controls over 18 fits, empirical keyed-insert response matrices under basis rotation, phase-mean relevance under joint release.
- RQ2 text: yaw ranked first and transition recovered at `N = 8,15,30`; fixture rotation preserves task-axis profiles (correlation `0.938–0.985`); basis rotation raises the off-diagonal norm fraction `0.0503 → 0.3126`; planar heading-yaw relevance `0.696` (constrained) vs `0.050` (released).
