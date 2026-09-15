# RQ3 — Identification from limited interventions

**Question.** How reliably can the generator profile be identified from limited interventions?

**Design.** Five independent execution seeds, with matched nested subsets `N ∈ {3, 5, 10, 20, 30}`. `N = 3` is a separate stress test (ten random subsets per seed); `N = 5,10,20` use five matched subsets per seed; `N = 30` is one shared full-data fit per seed. Subsets require full scaled rank and a scaled condition number below 10.

**Data.** One seed of the multiseed rollouts is shipped (`phase_switch_symmetry_multiseed/rollouts/seed_20260818.h5`) together with `fewshot_subsets.json`, `fixed_contexts.json`, and `experiment.json`. The remaining seeds are collected with `collect_phase_switch_multiseed.py`.

## Pipeline

### 1. Five-seed replication

```bash
# freeze the 30 mixed + 9 isolated contexts before collection
python -u -B phase_switch_symmetry/prepare_phase_switch_multiseed.py

# collect the remaining seeds (skip already-complete seeds)
python -u -B phase_switch_symmetry/collect_phase_switch_multiseed.py --skip-complete

# analyze + validate across all five seeds
python -u -B phase_switch_symmetry/analyze_phase_switch_multiseed.py
python -u -B phase_switch_symmetry/validate_phase_switch_multiseed.py
```

### 2. Few-shot and identifiability sweep

```bash
# freeze the subset manifest (reused verbatim for all seeds)
python -u -B phase_switch_symmetry/prepare_phase_switch_fewshot.py

# fit + audit for one seed (repeat per seed listed in experiment.json)
python -u -B phase_switch_symmetry/run_phase_switch_fewshot.py --seed 20260818
python -u -B phase_switch_symmetry/audit_phase_switch_fewshot_pdiag.py --seed 20260818

# TP-GMM matched few-shot comparison
python -u -B phase_switch_symmetry/run_phase_switch_tpgmm_fewshot.py

# analyze + validate
python -u -B phase_switch_symmetry/analyze_phase_switch_fewshot.py
python -u -B phase_switch_symmetry/validate_phase_switch_fewshot.py
```

Repeat the seed-specific `run`/`audit` commands for every seed in `phase_switch_symmetry_multiseed/experiment.json`. The independent audit refits `Pdiag`, records optimizer convergence, and checks exact reproduction of the saved profiles.

## Paper mapping

- Table I (held-out mean task error vs. `N`): `Pdiag` vs. TP-GMM / full operator / generic RBF. `Pdiag` achieves the largest advantage at small `N` and lower task error than TP-GMM in all matched seed-subset cells for `N = 5,10,20,30`.
- Paired mean task-error reduction over frame-scalar relevance `1.758 ± 0.465` mm-equivalent (hierarchical bootstrap 95% CI `[1.296, 2.230]`).
- `N = 3` recovery: `88%` random subsets, `92%` after excitation qualification.
- Progress-coordinate ablation (phase-normalized `4.60` vs. time `11.11` vs. DTW `3.19` mm-equivalent).
