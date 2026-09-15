# Data

The anonymous review snapshot ships selected small samples and documents how to regenerate the remaining simulation data. Raw physical robot datasets are not distributed.

## What is shipped

| Experiment | Shipped | Notes |
|---|---|---|
| RQ1 keyed insertion | Full | `phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5` + geometry probes (~12 MB). Fully reproducible. |
| RQ2 full SE(3) | Regeneration scripts + manifests | HDF5 smoke samples are omitted from the anonymous Git snapshot because their attributes contain generation-machine provenance paths. Regenerate with the commands below. |
| RQ3 few-shot | Manifests + scripts | The one-seed HDF5 rollout is omitted from the anonymous Git snapshot for the same provenance-metadata reason. Regenerate with the commands below. |
| RQ4a LIBERO | Scripts + manifests | Drawer/relation-suite HDF5 rollouts are omitted from the anonymous Git snapshot for the same provenance-metadata reason. Regenerate below when needed. |
| RQ4b semantic mismatch | Frozen law | `real_relation_consistency/full_se3_law/frozen_full_se3.npz`. Rollouts regenerated below. |
| RQ4c physical transfer | Full derived data | `real_relation_consistency/tcp_0_10_to_20/{fk_episodes.csv, metadata_audit.csv, fk_cache/*, observations/*}` (~5 MB). Fully reproducible. |

Config manifests (`*_experiment.json`, `*_subsets.json`, `*_fixed_contexts.json`) are shipped under `phase_switch_symmetry_multiseed/` because the benchmark scripts read them as defaults.

> **Absolute paths in manifests.** The `context_manifest`, `experiment`, and `source_dataset` fields of these manifests were written by the `prepare_*` scripts as `str(path.resolve())`, so they may contain generation-machine absolute paths. In this anonymous release, such paths are replaced by placeholders such as `<workspace-root>` and `<release-root>`. They are provenance fields verified by sha256, not portable links. On a new machine, run the `prepare_*` step listed on each `experiments/*.md` page first — it regenerates these manifests with the local absolute paths and recomputes the matching hashes. The `benchmark_*` scripts do not read these fields (they load intervention contexts from the HDF5 rollouts directly), so benchmarks run on the shipped samples without this step.

## Regenerating simulation data

All simulation rollouts are reproduced by the `collect_*` scripts in `phase_switch_symmetry/`; see the corresponding `experiments/*.md` page for the exact commands. A summary:

```bash
conda activate maniskill_spectrum

# RQ2 full SE(3) — three seeds each of keyed / circular-honest / multigen
python -u -B phase_switch_symmetry/collect_se3_rollouts.py \
  --output phase_switch_symmetry_rollouts_se3/keyed_seed_20260818.h5 --yaw-mode keyed --seed 20260818
python -u -B phase_switch_symmetry/collect_se3_rollouts.py \
  --output phase_switch_symmetry_rollouts_se3/circular_honest_seed_20260818.h5 --yaw-mode honest --seed 20260818
python -u -B phase_switch_symmetry/collect_se3_multigen_rollouts.py \
  --output phase_switch_symmetry_rollouts_se3/multigen_seed_20260818.h5 --seed 20260818

# RQ2 planar transport
python -u -B phase_switch_symmetry/collect_planar_push_rollouts.py \
  --output phase_switch_symmetry_rollouts_planar_push/heading_push_seed_20260818.h5 \
  --arm heading_push --seed 20260818

# RQ3 multiseed
python -u -B phase_switch_symmetry/collect_phase_switch_multiseed.py --skip-complete
```

The intervention design is frozen in the shipped manifests (`se3_fixed_contexts.json`, `fixed_contexts.json`, etc.), so re-collection reproduces the exact conditions.

### RQ4a relation suite

The 10-task × 3-seed relation-suite rollouts require the LIBERO environment for collection:

```bash
python -u -B phase_switch_symmetry/collect_libero_relation_suite.py
python -u -B phase_switch_symmetry/collect_libero_drawer_probe.py
```

## Physical robot data

The raw Franka demonstrations (joint trajectories and camera streams) are private data and are **not** distributed during anonymous review. This repository ships only the derived, analysis-ready products used by RQ4c (FK cache and cylinder-axis observations). Raw-data access can be documented in the camera-ready release after review.
