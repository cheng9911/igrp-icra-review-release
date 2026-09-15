# RQ1 — Heterogeneity and intervention response (keyed-to-circular insertion)

**Question.** Is generator response heterogeneous within a task relation, and does it follow intervention response rather than nominal trajectory motion?

**Benchmark.** A compound peg (rectangular key + circular shaft) inserts into a socket with a keyed gate. The socket intervention is `causal_delta = [dx_world, dy_world, dyaw_z]`. The fitted segment has four phases
`align → enter → unlock → insert`; yaw is task-relevant before key clearance and task-irrelevant afterwards. The profile is fitted from **30 mixed interventions** and evaluated on **eight disjoint isolated-generator interventions** plus one zero-intervention condition.

**Data.** Shipped fully in `phase_switch_symmetry_rollouts/` (`keyed_circular_phase_switch_physics_v2.h5` + geometry probes). This is the headline benchmark and is small enough to ship whole.

## Pipeline

Run every command from the repository root with the simulation environment active.

### 1. Collect rollouts

```bash
python -u -B phase_switch_symmetry/collect_phase_switch_rollouts.py \
  --output phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5 \
  --mixed-samples 30 --retries-per-condition 3
```

### 2. Analyze (strict)

```bash
python -u -B phase_switch_symmetry/analyze_phase_switch_rollouts.py \
  phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5 --strict
```

The strict check verifies translation propagation, keyed-yaw propagation, circular-yaw suppression, a held-out advantage over scalar frame weighting, real pairwise contact, a fixed kinematic socket, and full-rank mixed generator excitation.

### 3. Geometry probes (independent of the nominal planner targets)

```bash
python -u -B phase_switch_symmetry/collect_phase_switch_geometry_probes.py \
  --output phase_switch_symmetry_rollouts/keyed_circular_geometry_probes_v2.h5

python -u -B phase_switch_symmetry/validate_phase_switch_geometry_probes.py \
  phase_switch_symmetry_rollouts/keyed_circular_geometry_probes_v2.h5 --strict
```

### 4. Baseline benchmark

Fits all models on the same 30 mixed interventions and evaluates them on the same eight nonzero isolated interventions. Compared models: frame-weighted `w(s)I`, a phase-scalar GP, additive and rotation-aware SE(2) TP-GMM, a generic phase-local RBF regressor, a dense full operator, a pointwise diagonal ablation, and the smooth finite-action `Pdiag`.

```bash
python -u -B phase_switch_symmetry/benchmark_phase_switch_baselines.py \
  phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5 \
  --output-root phase_switch_symmetry_baselines
```

### 5. Validate benchmark

```bash
python -u -B phase_switch_symmetry/validate_phase_switch_baselines.py \
  phase_switch_symmetry_baselines --strict
```

## Paper mapping

- Translation relevance `0.9872` vs. axial-yaw relevance `0.9997 → 0.0022` across key clearance (Fig. 1b illustration, RQ1 text).
- Held-out task error `6.554` mm-equivalent (frame scalar) vs. `0.639` (generator-wise).
- The circular placebo (visible axial rotation, zero held-out yaw slope) is reproduced by the RQ2/`benchmark_se3_transfer.py` chain and the `phase_switch_symmetry_rollouts_se3/` circular-honest data — see [rq2_se3_structure.md](rq2_se3_structure.md).
