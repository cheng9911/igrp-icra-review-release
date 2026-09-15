# RQ4a — Controlled matched-relation structural reuse (LIBERO / robosuite)

**Question.** Can an identified response profile be reused under matched relation semantics, with changed target geometry and no target-profile fitting?

**Design.** Five state-level matched pairs in LIBERO/robosuite scenes isolate reuse of a known response structure under changed target geometry. The frozen source profile is applied with **zero** target-intervention fitting; a target nominal trajectory supplies the geometry. Compared against target-trained `Pdiag`, TP-GMM, frame scalar, and phase scalar at a target budget of eight.

**Data.** The drawer probe is shipped fully (`phase_switch_symmetry_rollouts_libero_drawer/drawer_seed_*.h5`, 3 MB). The relation suite (10 tasks × 3 seeds, 26 MB) is regenerated or obtained separately — see [data/README.md](../data/README.md). Config manifests live in `phase_switch_symmetry_multiseed/libero_drawer/` and `libero_relation_suite/`.

## Pipeline

**The benchmark and prepare scripts read pre-collected HDF5 and do not need LIBERO.** LIBERO is required only for re-collection.

### 1. Benchmark (no LIBERO required)

```bash
# drawer probe (works on the shipped data)
python -u -B phase_switch_symmetry/benchmark_libero_drawer_probe.py

# relation suite (needs the 26 MB rollout archive)
python -u -B phase_switch_symmetry/benchmark_libero_relation_suite.py
```

Both write their fits, profiles, and summaries into `phase_switch_symmetry_multiseed/libero_drawer/` and `libero_relation_suite/` respectively.

### 2. Re-collection (requires LIBERO)

Install LIBERO (see `environment-libero.yml`), then:

```bash
python -u -B phase_switch_symmetry/collect_libero_drawer_probe.py
python -u -B phase_switch_symmetry/collect_libero_relation_suite.py
```

The subset manifests can be regenerated with `prepare_libero_drawer_subsets.py` and `prepare_libero_relation_suite_subsets.py`.

## Paper mapping

- Table II (state-level profile transfer at target budget eight): frozen source `M_sel = 1.000`, `E_α = 2.13 × 10⁻⁴`; target `Pdiag` `7.42 × 10⁻⁴`; target TP-GMM `2.62 × 10⁻²`; frame/phase scalar `1.28 × 10⁻¹` with `M_sel = 0`.
