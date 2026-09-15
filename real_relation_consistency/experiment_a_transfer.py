"""Experiment A (offline cross-instance transfer): apply the frozen CIRCULAR six-
generator law to the KEYED assembly instance and compare against the KEYED
teacher's actual responses.

The frozen law is the retrospective ``full_se3_law/frozen_full_se3.npz`` diagonal
P(s) identified on ``circular_honest`` (yaw is a gauge symmetry, alpha_yaw ~ 0.1).
The KEYED instance differs geometrically (key + keyway) and has a genuinely
selective yaw generator (oracle [1,1,0,0]).  A frozen circular law cannot predict
the keyed yaw response; du/dv/dw/roll/pitch should still transfer ([1,1,1,1] in
both).  This is the honest, informative negative result the review asks for: a
phase-dependent response law does NOT transfer across a symmetry-changing
geometry change.

Only the frozen law is used on the target (no keyed re-fit); the keyed baseline
provides the nominal X0 and frame C0, and every keyed intervention is held out.
"""

from pathlib import Path
import sys
import json

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "phase_switch_symmetry"))

from benchmark_phase_switch_baselines import usable, progress_grid  # noqa: E402
from benchmark_se3_transfer import (  # noqa: E402
    GENERATOR_NAMES,
    nominal_frame_se3,
    task_curve_se3,
)
from phase_switch_se3_baselines import (  # noqa: E402
    se3_exp_batched,
    se3_from_pose6,
    se3_from_pose6_batched,
    se3_inverse,
    se3_log,
)

FROZEN = HERE / "full_se3_law" / "frozen_full_se3.npz"
KEYED = str(ROOT / "phase_switch_symmetry_rollouts_se3" / "keyed_seed_{seed}.h5")
SEEDS = [20260818, 20270818, 20280818]
OUT = HERE / "experiment_a_transfer"


def apply_relation(X0, D, C0, alpha):
    """Xhat(s) = C0 Exp(alpha(s) . Log(D)) C0^-1 X0(s).

    D = se3_from_pose6(causal_delta) is the LOCAL (C0-relative) intervention:
    the env places the socket at ``C0 @ D``, so the response twist is ``Log(D)``
    in the nominal frame (NOT the conjugated ``Log(C0^-1 D C0)``).
    """
    C0inv = se3_inverse(C0)
    xi = se3_log(D)  # (6,) local intervention twist
    actions = se3_exp_batched(alpha * xi[None, :])  # (N,4,4)
    return C0 @ actions @ C0inv @ X0  # (N,4,4)


def metrics(pred, actual):
    # pred/actual: (N,4,4)
    dist = np.linalg.norm(pred[:, :3, 3] - actual[:, :3, 3], axis=-1) * 1000
    rel = np.swapaxes(pred[:, :3, :3], -1, -2) @ actual[:, :3, :3]
    rot = np.rad2deg(Rotation.from_matrix(rel).magnitude())
    u, v = pred[:, :3, 2], actual[:, :3, 2]
    axis = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1),
                                 np.sum(u * v, axis=-1)))
    return (np.sqrt(np.mean(dist ** 2)), np.sqrt(np.mean(rot ** 2)),
            np.sqrt(np.mean(axis ** 2)))


def load_keyed(seed):
    with h5py.File(KEYED.format(seed=seed), "r") as f:
        keys = sorted((k for k in f if k.startswith("episode_")),
                      key=lambda k: int(k.split("_")[-1]))
        baseline_key = None
        conds = []
        for k in keys:
            g = f[k]
            if not usable(g):
                continue
            gen = str(g.attrs["generator"])
            cd = np.asarray(g["causal_delta"], dtype=np.float64)
            if gen == "baseline":
                baseline_key = k
            else:
                conds.append((k, gen, cd))
        assert baseline_key is not None
        # nominal X0 + C0 from the baseline episode alone (target's one nominal).
        X0 = task_curve_se3(f[baseline_key], 25)  # (100,6)
        socket0 = np.asarray(f[baseline_key]["socket_pose"])[0]
        # C0 = baseline socket pose as the nominal frame (intervention = identity).
        from benchmark_se3_transfer import pose_to_pose6
        C0 = se3_from_pose6(pose_to_pose6(socket0))
        actuals = {k: task_curve_se3(f[k], 25) for k, _, _ in conds}
        return X0, C0, conds, actuals


def main():
    OUT.mkdir(exist_ok=True)
    frozen = np.load(FROZEN, allow_pickle=True)
    rows = []
    per_generator = []
    for seed in SEEDS:
        alpha = np.asarray(frozen[f"seed_{seed}_diagonal"])  # (100,6)
        X0, C0, conds, actuals = load_keyed(seed)
        X0_T = se3_from_pose6_batched(X0)
        for k, gen, cd in conds:
            D = se3_from_pose6(cd)
            pred = apply_relation(X0_T, D, C0, alpha)
            actual = se3_from_pose6_batched(actuals[k])
            pos, ori, ax = metrics(pred, actual)
            rows.append(dict(seed=seed, generator=gen, condition_key=k,
                             pos_rmse_mm=pos, orient_rmse_deg=ori, axis_rmse_deg=ax,
                             causal_delta=cd.tolist()))
    # aggregate per generator (mean over conditions x seeds)
    for gen in GENERATOR_NAMES:
        sub = [r for r in rows if r["generator"] == gen]
        if not sub:
            continue
        per_generator.append(dict(
            generator=gen,
            n_conditions=len(sub),
            pos_rmse_mm=float(np.mean([r["pos_rmse_mm"] for r in sub])),
            orient_rmse_deg=float(np.mean([r["orient_rmse_deg"] for r in sub])),
            axis_rmse_deg=float(np.mean([r["axis_rmse_deg"] for r in sub])),
        ))
    # mixed conditions are joint interventions; report separately.
    mixed = [r for r in rows if r["generator"] == "mixed"]
    overall = dict(
        pos_rmse_mm=float(np.mean([r["pos_rmse_mm"] for r in rows])),
        orient_rmse_deg=float(np.mean([r["orient_rmse_deg"] for r in rows])),
        axis_rmse_deg=float(np.mean([r["axis_rmse_deg"] for r in rows])),
        n_conditions=len(rows),
    )
    mixed_agg = dict(
        pos_rmse_mm=float(np.mean([r["pos_rmse_mm"] for r in mixed])),
        orient_rmse_deg=float(np.mean([r["orient_rmse_deg"] for r in mixed])),
        axis_rmse_deg=float(np.mean([r["axis_rmse_deg"] for r in mixed])),
        n_conditions=len(mixed),
    )

    print("=== per-generator cross-instance (frozen circular -> keyed) ===")
    hdr = f"{'generator':10s} {'n':>4s} {'pos mm':>9s} {'orient deg':>10s} {'axis deg':>9s}"
    print(hdr)
    for g in per_generator:
        print(f"{g['generator']:10s} {g['n_conditions']:4d} "
              f"{g['pos_rmse_mm']:9.3f} {g['orient_rmse_deg']:10.3f} {g['axis_rmse_deg']:9.3f}")
    print(f"\n{'ALL':10s} {overall['n_conditions']:4d} "
          f"{overall['pos_rmse_mm']:9.3f} {overall['orient_rmse_deg']:10.3f} {overall['axis_rmse_deg']:9.3f}")
    print(f"{'mixed':10s} {mixed_agg['n_conditions']:4d} "
          f"{mixed_agg['pos_rmse_mm']:9.3f} {mixed_agg['orient_rmse_deg']:10.3f} {mixed_agg['axis_rmse_deg']:9.3f}")

    summary = dict(
        note="Frozen CIRCULAR six-generator diagonal law applied to the KEYED "
             "instance (cross-instance, target provides one baseline nominal only). "
             "Frozen law yaw channel ~0 (circular gauge symmetry); keyed oracle yaw "
             "= [1,1,0,0]. No keyed re-fit.",
        frozen_source=str(FROZEN),
        within_instance_circular_frozen_mm_deg_axis=[3.806, 1.620, 0.902],
        per_generator=per_generator,
        mixed=mixed_agg,
        overall=overall,
        rows=rows,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
