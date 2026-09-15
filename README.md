# Interventional Generator Response Profiles for Matched-Relation Transfer in SE(3) Manipulation

Anonymous review code release for the paper **"Interventional Generator Response Profiles for Matched-Relation Transfer in SE(3) Manipulation"**.

Task-frame weighting does not directly identify *which* translational or rotational generators should propagate a change in task geometry. This repository identifies **progress-dependent, task-local generator response profiles** `P_r(s)` from controlled relation interventions, and separates the learned response from the known `SE(3)` geometric action of the relation. The factorization distinguishes (i) intervention response from (ii) nominal motion and (iii) geometry-induced pose coupling.

The reusable quantity is the normalized response shape, with relation semantics, generator correspondence and normalization, progress, and execution strategy defining the conditions for reuse.

---

## Research questions

The evaluation asks four questions, each with a dedicated reproduction page:

| | Question | Page |
|---|---|---|
| **RQ1** | Is generator response heterogeneous within a relation, and does it follow intervention response rather than nominal motion? | [experiments/rq1_keyed_insertion.md](experiments/rq1_keyed_insertion.md) |
| **RQ2** | Under what basis / relation / execution conditions does a compact generator-response structure hold (full SE(3))? | [experiments/rq2_se3_structure.md](experiments/rq2_se3_structure.md) |
| **RQ3** | How reliably is the profile identified from limited interventions? | [experiments/rq3_fewshot_identification.md](experiments/rq3_fewshot_identification.md) |
| **RQ4** | Can an identified profile be reused under matched relation semantics, and where does reuse fail? (a) state-level matched pairs, (b) semantic mismatch, (c) physical frozen response-shape transfer | [experiments/rq4a_libero_matched_pairs.md](experiments/rq4a_libero_matched_pairs.md) · [rq4b_semantic_mismatch.md](experiments/rq4b_semantic_mismatch.md) · [rq4c_physical_transfer.md](experiments/rq4c_physical_transfer.md) |

---

## Repository layout

```
release/
├── phase_switch_symmetry/          # simulation scripts + model library (flat namespace)
├── phase_switch_symmetry_*/        # shipped sample data (see data/README.md for full data)
├── real_relation_consistency/      # RQ4b + RQ4c scripts and derived physical data
├── experiments/                    # per-RQ reproduction instructions
├── assets/videos/                  # overview + experiment videos
├── data/                           # data policy and regeneration guide
├── environment.yml                 # simulation environment (ManiSkill)
├── environment-libero.yml          # optional: RQ4a re-collection only
└── requirements.txt                # light analysis environment (RQ4b + RQ4c)
```

All scripts are run **from the repository root** (`release/`). The simulation scripts use a flat namespace and import each other by top-level module name; running `python phase_switch_symmetry/<script>.py` from the root makes this work without any packaging.

---

## Installation

### 1. Simulation environment (RQ1–RQ3, RQ4b regeneration)

```bash
conda env create -f environment.yml
conda activate maniskill_spectrum
```

This installs [ManiSkill 3](https://github.com/haosulab/ManiSkill) (which pulls its own `torch` / `sapien` / `gymnasium`) plus the numerical stack (`numpy`, `scipy`, `pandas`, `matplotlib`, `h5py`, `scikit-learn`, `threadpoolctl`, `transforms3d`). Versions in `environment.yml` are the ones verified for this release.

### 2. Light analysis environment (RQ4b + RQ4c only)

Reproducing the physical-transfer results (RQ4c) and the semantic-mismatch transfer (RQ4b, from the shipped frozen law) does not need ManiSkill:

```bash
pip install -r requirements.txt
```

### 3. LIBERO (optional, RQ4a re-collection only)

The RQ4a benchmark reads pre-collected HDF5 and needs **no** LIBERO. LIBERO is required only to re-run the two `collect_*` scripts. Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) per its own instructions (see `environment-libero.yml` for the Python level).

---

## Quick start (smoke test)

The repository ships small sample data so you can verify the pipeline without downloading the full rollout archives. From the repository root:

```bash
conda activate maniskill_spectrum

# RQ1 — analyze the shipped keyed-insertion rollout
python -B phase_switch_symmetry/analyze_phase_switch_rollouts.py \
  phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5 --strict

# RQ1 — run the full baseline benchmark on the shipped rollout
python -B phase_switch_symmetry/benchmark_phase_switch_baselines.py \
  phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5 \
  --output-root phase_switch_symmetry_baselines

# RQ4c — reproduce the physical frozen response-shape transfer (analysis-only env is enough)
python -B real_relation_consistency/tcp_0_10_to_20/calibration_split_transfer.py
```

The RQ4c command writes the held-out comparison tables, the publication figure, and a `verification.json` into `real_relation_consistency/tcp_0_10_to_20/calibration_split/`.

---

## Reproducing the paper

Each page under [experiments/](experiments/) gives the exact commands, input data, and the mapping to the paper's tables and figures. In brief:

- **RQ1** (`rq1_keyed_insertion.md`) — keyed-to-circular insertion. Collect → analyze → geometry probes → baseline benchmark. Reproduces Fig. 1 (illustration) and the RQ1 heterogeneity numbers (translation `0.9872` vs. yaw `0.9997 → 0.0022`).
- **RQ2** (`rq2_se3_structure.md`) — full-SE(3) generator structure, basis ablation, planar transport. Reproduces Fig. 3 (task-local generator structure).
- **RQ3** (`rq3_fewshot_identification.md`) — few-shot identification sweep and five-seed replication. Reproduces Table I (held-out task error vs. `N`).
- **RQ4a** (`rq4a_libero_matched_pairs.md`) — five state-level matched pairs in LIBERO. Reproduces Table II (state-level profile transfer).
- **RQ4b** (`rq4b_semantic_mismatch.md`) — circular→keyed profile transfer; the yaw channel produces a `13.36°` full-orientation error.
- **RQ4c** (`rq4c_physical_transfer.md`) — physical `10°↔20°` cylinder-axis response-shape transfer with disjoint TCP calibration/evaluation. Reproduces Table III and Fig. 4.

---

## Data

The anonymous review snapshot ships the RQ1 sample rollouts and the complete RQ4c derived data. Some HDF5 smoke samples are omitted from the Git snapshot because their HDF5 attributes contain generation-machine provenance paths; they can be regenerated with the commands in [data/README.md](data/README.md) and the per-RQ pages under [experiments/](experiments/). Full multi-seed archives and the raw physical datasets are not bundled.

---

## Videos

Videos are in [assets/videos/](assets/videos/):

- `paper_video.mp4` — paper overview video.
- `0*_*.mp4`, `2*_*.mp4` — simulation experiment clips (keyed insertion, few-shot, SE(3) keyed-vs-circular, five-seed replication).
- `pitch_*.mp4` — physical robot tilt-sweep clips (RQ4c).

---

## Citation

If you use this code, please cite the paper (bibliographic details to be added on publication).

---

## License

MIT — see [LICENSE](LICENSE).
