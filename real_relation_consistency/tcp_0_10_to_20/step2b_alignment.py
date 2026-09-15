"""Step 2b: alignment-only (cylinder-axis direction) transfer validation.

Reframe the cross-condition transfer test onto the cylinder-axis ALIGNMENT
metric (unifying with tilt_sweep's hand-axis magnitude ~10.30 deg), instead of
the full SE(3) transform. The source law is the TILT rotation of the cylinder
axis — the minimal rotation mapping the upright axis to the tilted axis — with
the twist-about-axis and translation dropped. Prediction and evaluation use
only the cylinder axis direction a = -R[:,2].

Model:
    ahat_theta(s) = Rot( q * g(s) * theta_10 , u_10 ) @ a_0(s)
where theta_10 = angle(a_0_mean, a_1_10_mean), u_10 = normalize(a_0_mean x a_1_10_mean)
is the source (10-degree) tilt axis, and q = theta/10deg. g(s) is a smooth
scalar fitted on the source-train axis-angle residual.

This answers: does the 10-degree source law transfer to 20 degrees in the pure
tilt direction (dropping twist + translation)?
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

from se3_utils import resample_pose, cylinder_axis_from_rot, angle_between

N_POINTS = 151
S_GRID = np.linspace(0.0, 1.0, N_POINTS)
MID_LO, MID_HI = 0.2, 0.8
SPLIT_MOD = 5
METHODS = ["no_adapt", "full_transform", "constant_gain", "fixed_ramp", "learned_g"]


def _skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rotmat_about_axis(u, phi_rad):
    """Rotation matrices (Rodrigues) about a fixed unit axis u by angles phi_rad."""
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    phi = np.asarray(phi_rad, dtype=float)
    K = _skew(u)
    K2 = K @ K
    s = np.sin(phi)[..., None, None]
    c = np.cos(phi)[..., None, None]
    return np.eye(3) + s * K + (1.0 - np.cos(phi))[..., None, None] * K2


def rbf_basis(s_grid, n_basis, basis_width):
    centers = np.linspace(0.0, 1.0, n_basis)
    B = np.exp(-0.5 * (((s_grid[:, None] - centers[None, :]) / basis_width) ** 2))
    return B / B.sum(axis=1, keepdims=True)


def load_axes(out: Path):
    meta = pd.read_csv(out / "fk_episodes.csv")
    meta = meta[meta.extraction_status == "ok"]
    up = {}
    for _, r in meta[meta.dataset == "insert_jc_fix_3"].iterrows():
        z = np.load(out / "fk_cache" / f"insert_jc_fix_3_{int(r.episode_index):03d}.npz")
        n = int(r.window_end_exclusive) - int(r.grasp_start)
        p, R = resample_pose(z["tcp_position_m"], z["tcp_rotation"], N_POINTS, 0, n)
        up[int(r.episode_index)] = dict(axis=cylinder_axis_from_rot(R), pos=p)
    tilted = {}
    for ds in ["insert_10degree", "insert_20degree"]:
        z = np.load(out / "observations" / f"obs_{ds}.npz")
        tilted[ds] = dict(ep=z["episode_index"], nom=z["nominal_episode"],
                          axis=cylinder_axis_from_rot(z["rot"]), pos=z["pos"])
    return up, tilted


def source_tilt(up, tilted):
    """Source (10-degree) group terminal tilt: theta_10 and tilt axis u_10."""
    X10 = tilted["insert_10degree"]
    tr = np.array([int(e) % SPLIT_MOD != 0 for e in X10["ep"]])
    a0 = np.stack([up[int(nm)]["axis"][-1] for nm in X10["nom"][tr]])
    a1 = X10["axis"][tr, -1]
    a0m = a0.mean(0); a0m /= np.linalg.norm(a0m)
    a1m = a1.mean(0); a1m /= np.linalg.norm(a1m)
    theta = float(angle_between(a0m, a1m))
    u = np.cross(a0m, a1m)
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        raise ValueError("degenerate tilt axis (upright and 10-degree axes collinear)")
    u = u / nu
    return theta, u, a0m, a1m


def predict_axis(g_curve, q, theta_deg, u, a0):
    """ahat(s) = Rot(q*g(s)*theta, u) @ a0(s). a0: (n,151,3)."""
    phi_deg = q * np.asarray(g_curve) * theta_deg
    R = rotmat_about_axis(u, np.radians(phi_deg))
    return np.einsum("sjk,nsk->nsj", R, a0)


def fit_g_axis(a0, a1, theta_deg, u, n_basis, basis_width, smoothness_weight):
    basis = rbf_basis(S_GRID, n_basis, basis_width)
    n_data = a0.shape[0] * N_POINTS
    n_smooth = max(N_POINTS - 2, 1)
    scale = np.sqrt(smoothness_weight * n_data / n_smooth)

    def residual(beta):
        g = basis @ beta
        ahat = predict_axis(g, 1.0, theta_deg, u, a0)
        data = angle_between(ahat, a1).reshape(-1)          # degrees
        smooth = np.diff(g, n=2) * scale
        return np.concatenate([data, smooth])

    opt = least_squares(residual, np.ones(n_basis), method="trf", max_nfev=500,
                        ftol=1e-9, xtol=1e-9, gtol=1e-9)
    return basis @ opt.x, opt


def axis_rmse(ahat, a1):
    return angle_between(ahat, a1)  # (n,151) degrees


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    dest = out / "alignment"
    dest.mkdir(exist_ok=True)

    up, tilted = load_axes(out)
    X10 = tilted["insert_10degree"]
    X20 = tilted["insert_20degree"]

    # ---- source tilt + observed tilts (descriptive, vs upright) ----
    theta_10, u_10, a0m_10, a1m_10 = source_tilt(up, tilted)   # model param (train-only)

    def group_tilt(Xt):
        a0 = np.stack([up[int(nm)]["axis"][-1] for nm in Xt["nom"]])
        a1 = Xt["axis"][:, -1]
        a0m = a0.mean(0); a0m /= np.linalg.norm(a0m)
        a1m = a1.mean(0); a1m /= np.linalg.norm(a1m)
        th = float(angle_between(a0m, a1m))
        u = np.cross(a0m, a1m); u /= np.linalg.norm(u)
        return th, u, a0m, a1m
    # descriptive observed tilt over ALL episodes of each condition
    theta_10_obs, u_10_obs, a0m_10_all, a1m_10_all = group_tilt(X10)
    theta_20_obs, u_20, a0m_20, a1m_20 = group_tilt(X20)
    u_sep = float(angle_between(u_10_obs, u_20))

    # per-episode terminal axis-angle (mean +/- sd), vs matched upright
    def term_axis_angle(Xt):
        a0 = np.stack([up[int(nm)]["axis"] for nm in Xt["nom"]])       # (n,151,3)
        ang = angle_between(Xt["axis"], a0)                            # (n,151)
        return ang[:, -1]
    t10 = term_axis_angle(X10); t20 = term_axis_angle(X20)

    obs = dict(
        theta_10_group_deg=theta_10_obs, u_10_obs=u_10_obs.tolist(),
        theta_20_group_deg=theta_20_obs, u_20=u_20.tolist(),
        tilt_axis_separation_deg=u_sep,
        magnitude_ratio_20_over_10=theta_20_obs / theta_10_obs,
        model_theta_10_train_deg=theta_10, model_u_10_train=u_10.tolist(),
        term_axis_angle_10_deg=dict(mean=float(t10.mean()), sd=float(t10.std(ddof=1)),
                                    median=float(np.median(t10))),
        term_axis_angle_20_deg=dict(mean=float(t20.mean()), sd=float(t20.std(ddof=1)),
                                    median=float(np.median(t20))),
        note="axis angle = angle(a_tilt, a_upright) in deg; pure tilt, no twist/translation",
    )

    # ---- fit g(s) on source-train axis-angle residual ----
    tr = np.array([int(e) % SPLIT_MOD != 0 for e in X10["ep"]])
    va = ~tr
    a0_tr = np.stack([up[int(nm)]["axis"] for nm in X10["nom"][tr]])
    a1_tr = X10["axis"][tr]
    a0_va = np.stack([up[int(nm)]["axis"] for nm in X10["nom"][va]])
    a1_va = X10["axis"][va]

    n_basis, basis_width = 16, 0.08
    smooth_candidates = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    best = None
    for sw in smooth_candidates:
        g, opt = fit_g_axis(a0_tr, a1_tr, theta_10, u_10, n_basis, basis_width, sw)
        tr_rmse = float(np.mean(angle_between(predict_axis(g, 1.0, theta_10, u_10, a0_tr), a1_tr) ** 2) ** 0.5)
        va_rmse = float(np.mean(angle_between(predict_axis(g, 1.0, theta_10, u_10, a0_va), a1_va) ** 2) ** 0.5)
        if best is None or va_rmse < best[0]:
            best = (va_rmse, sw, g, opt)
    sel_sw = best[1]
    g_learned, opt_learned = fit_g_axis(a0_tr, a1_tr, theta_10, u_10, n_basis, basis_width, sel_sw)
    src_tr_rmse = float(np.mean(angle_between(predict_axis(g_learned, 1.0, theta_10, u_10, a0_tr), a1_tr) ** 2) ** 0.5)
    src_va_rmse = float(np.mean(angle_between(predict_axis(g_learned, 1.0, theta_10, u_10, a0_va), a1_va) ** 2) ** 0.5)
    c = float(minimize_scalar(
        lambda cc: np.mean(angle_between(predict_axis(np.full(N_POINTS, cc), 1.0, theta_10, u_10, a0_tr), a1_tr) ** 2),
        bounds=(-0.5, 2.5), method="bounded").x)

    g_functions = dict(no_adapt=np.zeros(N_POINTS), full_transform=np.ones(N_POINTS),
                       constant_gain=np.full(N_POINTS, c), fixed_ramp=S_GRID.copy(),
                       learned_g=g_learned)

    # ---- score 20-degree (q=2): alignment error of predicted axis vs observed ----
    a0_20 = np.stack([up[int(nm)]["axis"] for nm in X20["nom"]])
    a1_20 = X20["axis"]
    mid = (S_GRID >= MID_LO) & (S_GRID <= MID_HI)
    ep_rows = []
    summary_rows = []
    for m in METHODS:
        ahat = predict_axis(g_functions[m], 2.0, theta_10, u_10, a0_20)
        err = angle_between(ahat, a1_20)                       # (52,151) deg
        full = np.sqrt(np.mean(err ** 2, axis=1))
        mid_ = np.sqrt(np.mean(err[:, mid] ** 2, axis=1))
        term = err[:, -1]
        for i, ep in enumerate(X20["ep"]):
            ep_rows.append(dict(episode_index=int(ep), nominal_episode=int(X20["nom"][i]),
                                method=m, align_rmse_deg_full=float(full[i]),
                                align_rmse_deg_mid=float(mid_[i]), align_err_deg_terminal=float(term[i])))
        for col, seg in [(full, "full"), (mid_, "mid"), (term, "terminal")]:
            summary_rows.append(dict(method=m, metric=f"align_{'rmse_deg' if seg!='terminal' else 'err_deg'}_{seg}",
                                     mean=float(col.mean()), sd=float(col.std(ddof=1)),
                                     median=float(np.median(col)), n=int(len(col))))
    pd.DataFrame(ep_rows).to_csv(dest / "alignment_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(dest / "summary_alignment.csv", index=False)

    np.savez_compressed(dest / "alignment_model.npz", theta_10=theta_10, u_10=u_10,
                        **{f"g_{m}": g_functions[m] for m in METHODS}, s_grid=S_GRID)

    json.dump(dict(
        theta_10_group_deg=theta_10_obs, u_10_obs=u_10_obs.tolist(),
        theta_20_group_deg=theta_20_obs, u_20=u_20.tolist(),
        tilt_axis_separation_deg=u_sep,
        magnitude_ratio_20_over_10=theta_20_obs / theta_10_obs,
        model_theta_10_train_deg=theta_10, model_u_10_train=u_10.tolist(),
        constant_gain_c=c, learned_g_smoothness_weight=sel_sw,
        learned_g_terminal=float(g_learned[-1]), learned_g_min=float(g_learned.min()),
        learned_g_max=float(g_learned.max()),
        source_train_n=int(tr.sum()), source_val_n=int(va.sum()), eval_n=int(len(X20["ep"])),
        source_train_axis_rmse_deg=src_tr_rmse, source_val_axis_rmse_deg=src_va_rmse,
        metric="angle(predicted cylinder axis, observed cylinder axis), degrees",
        prediction="ahat_20(s) = Rot(2*g(s)*theta_10, u_10) @ a0_20(s)",
    ), open(dest / "alignment_protocol.json", "w"), indent=2)

    # ---- print summary ----
    print("=" * 72)
    print("OBSERVED (vs upright, axis-angle, pure tilt; all episodes):")
    print(f"  10 deg: group tilt {theta_10_obs:.2f} deg ; per-episode terminal {t10.mean():.2f} +/- {t10.std(ddof=1):.2f} deg")
    print(f"  20 deg: group tilt {theta_20_obs:.2f} deg ; per-episode terminal {t20.mean():.2f} +/- {t20.std(ddof=1):.2f} deg")
    print(f"  tilt-axis separation u_10 vs u_20 = {u_sep:.2f} deg ; magnitude ratio = {theta_20_obs/theta_10_obs:.3f}")
    print(f"  source model (train-only): theta_10 = {theta_10:.2f} deg ; fit RMSE train {src_tr_rmse:.2f} / val {src_va_rmse:.2f} deg")
    print("=" * 72)
    print("20-deg ALIGNMENT transfer (q=2, predicted axis vs observed axis):")
    print(f"{'method':16s} {'full RMSE(deg)':>15s} {'mid RMSE(deg)':>14s} {'terminal(deg)':>14s}")
    for m in METHODS:
        sub = [r for r in summary_rows if r["method"] == m]
        full = [r for r in sub if r["metric"] == "align_rmse_deg_full"][0]
        mid_ = [r for r in sub if r["metric"] == "align_rmse_deg_mid"][0]
        term = [r for r in sub if r["metric"] == "align_err_deg_terminal"][0]
        print(f"{m:16s} {full['mean']:15.2f} {mid_['mean']:14.2f} {term['mean']:14.2f}")
    print(f"\nconstant_gain c = {c:.3f}; learned_g terminal = {g_learned[-1]:.3f} "
          f"(min {g_learned.min():.3f}, max {g_learned.max():.3f})")
    print(f"learned_g smoothness weight = {sel_sw}")


if __name__ == "__main__":
    main()
