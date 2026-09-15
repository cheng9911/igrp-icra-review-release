"""Evaluate frozen response-shape transfer with disjoint TCP calibration."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step2b_alignment import load_axes, S_GRID
from se3_utils import angle_between

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "calibration_split"
METHODS = ["Direct reuse", "Identity", "Linear", "Frozen source", "Target calibration"]
COLORS = ["#777777", "#956A99", "#B48A38", "#007C83", "#C35D35"]


def unit(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    assert np.all(norm > 1e-10)
    return x / norm


def rotation_vectors(up, sample):
    baseline = unit(np.stack([up[int(i)]["axis"] for i in sample["nom"]]))
    observed = unit(sample["axis"])
    theta = np.degrees(np.arccos(np.clip((baseline * observed).sum(-1), -1, 1)))
    cross = np.cross(baseline, observed)
    norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    return theta[..., None] * cross / np.maximum(norm, 1e-12)


def geometry(upright, terminal_axes):
    target = unit(terminal_axes.mean(0))
    return float(angle_between(upright, target)), unit(np.cross(upright, target))


def predictions(source_profile, source_vectors, calibration_vectors, c, u):
    target_profile = (calibration_vectors @ u).mean(0) / c
    profiles = [np.ones_like(S_GRID), S_GRID, source_profile, target_profile]
    result = {"Direct reuse": source_vectors.mean(0)}
    result.update({method: profile[:, None] * c * u
                   for method, profile in zip(METHODS[1:], profiles)})
    return result



def draw(archive, scores):
    """
    Full-width publication figure: (a,b) bidirectional response profiles and
    (c) held-out vector errors, laid out as a clean 2x2 grid. Intended for a
    double-column ``figure*`` environment.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.titleweight": "normal",
        "axes.labelsize": 7.2,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    measured_color = "#3D4243"
    measured_band = "#C4CBCD"
    frozen_color = "#007C83"
    target_color = "#C35D35"
    direction_colors = ["#007C83", "#C35D35"]
    direction_markers = ["o", "s"]

    fig = plt.figure(figsize=(7.0, 3.9), facecolor="white")
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.085, right=0.985, top=0.875, bottom=0.145,
        hspace=0.55, wspace=0.30,
    )

    # ---- (a, b) response profiles ----
    for col, (a, b) in enumerate([(10, 20), (20, 10)]):
        key = f"{a}_to_{b}"
        ax = fig.add_subplot(gs[0, col])
        measured = archive[key + "_measured_projected"]
        mean = measured.mean(0)
        sd = measured.std(0, ddof=1)

        ax.fill_between(S_GRID, mean - sd, mean + sd,
                        color=measured_band, alpha=0.45, linewidth=0, zorder=1)
        ax.plot(S_GRID, mean, color=measured_color, lw=1.1,
                solid_capstyle="round", zorder=4)
        ax.plot(S_GRID, archive[key + "_frozen_projected"],
                color=frozen_color, ls="--", lw=1.3, zorder=5)
        ax.plot(S_GRID, archive[key + "_target_projected"],
                color=target_color, ls=":", lw=1.2, zorder=5)

        ax.set_xlim(0, 1)
        ax.margins(x=0)
        ax.set_xlabel(r"Normalized progress $s$", labelpad=2)
        ax.set_ylabel("Projected response (deg)", labelpad=2)
        ax.set_title(rf"({chr(97 + col)})  ${a}^\circ \rightarrow {b}^\circ$",
                     loc="left", pad=4)
        ax.tick_params(length=2.4, pad=1.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared legend above the two profiles.
    profile_handles = [
        Line2D([0], [0], color=measured_color, lw=1.1, label="Measured"),
        Line2D([0], [0], color=frozen_color, lw=1.3, ls="--", label="Frozen source"),
        Line2D([0], [0], color=target_color, lw=1.2, ls=":", label="Target-calibrated"),
    ]
    fig.legend(handles=profile_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.962), ncol=3, frameon=False,
               handlelength=2.0, columnspacing=1.1, handletextpad=0.5)

    # ---- (c) held-out vector errors ----
    primary = scores[(scores.fold == 0) & (scores.geometry == "Calibration")]
    methods = ["Direct reuse", "Identity", "Linear", "Frozen source", "Target calibration"]
    display_labels = ["Direct reuse", "Identity", "Linear", "Frozen source", "Target-calibrated"]
    metric_specs = [("terminal_deg", "Terminal"), ("whole_deg", "Mean over progress")]

    for col, (metric, sub_title) in enumerate(metric_specs):
        ax = fig.add_subplot(gs[1, col])
        for direction_idx, direction in enumerate(["10_to_20", "20_to_10"]):
            means, sds = [], []
            for method in methods:
                values = primary[(primary.direction == direction) &
                                 (primary.method == method)][metric]
                means.append(values.mean())
                sds.append(values.std(ddof=1))

            y = np.arange(len(methods)) + (direction_idx - 0.5) * 0.20
            ax.errorbar(means, y, xerr=sds,
                        fmt=direction_markers[direction_idx],
                        ms=3.4, capsize=3.0, elinewidth=0.9, capthick=1.2,
                        lw=0.9, color=direction_colors[direction_idx],
                        markeredgecolor=direction_colors[direction_idx],
                        zorder=3)

        ax.set(ylim=(4.6, -0.6), xlim=(0, 16), xticks=[0, 8, 16],
               xlabel="Error (deg)")
        ax.set_title(sub_title, loc="left", pad=3, fontsize=7.2)
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels(display_labels if col == 0 else [])
        ax.tick_params(length=2.4, pad=1.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # (c) row label spanning both bottom panels.
    fig.text(0.085, 0.475, "(c)  Held-out vector errors (mean ± episode SD)",
             fontsize=7.2, va="center", ha="left")

    # Shared transfer-direction legend beneath panel (c).
    direction_handles = [
        Line2D([0], [0], color=direction_colors[0], marker="o", lw=0.8, markersize=3.4,
               label=r"$10^\circ \rightarrow 20^\circ$"),
        Line2D([0], [0], color=direction_colors[1], marker="s", lw=0.8, markersize=3.4,
               label=r"$20^\circ \rightarrow 10^\circ$"),
    ]
    fig.legend(handles=direction_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.022), ncol=2, frameon=False,
               handlelength=1.8, columnspacing=1.6, handletextpad=0.5)

    # Export with a single consistent basename.
    basename = "fig_physical_transfer_single_column"
    fig.savefig(OUT / f"{basename}.pdf")
    fig.savefig(OUT / f"{basename}.svg")
    fig.savefig(OUT / f"{basename}.png", dpi=600)
    fig.savefig(OUT / f"{basename}.tiff", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    up, datasets = load_axes(ROOT)
    upright = unit(np.stack([up[i]["axis"][-1] for i in sorted(up)]).mean(0))
    data = {degree: datasets[f"insert_{degree}degree"] for degree in (10, 20)}
    vectors = {degree: rotation_vectors(up, sample) for degree, sample in data.items()}
    source_profiles = {}
    for degree in data:
        c, u = geometry(upright, data[degree]["axis"][:, -1])
        source_profiles[degree] = (vectors[degree] @ u).mean(0) / c
    scores, split_rows, geometry_rows, curves = [], [], [], []
    archive = {"progress": S_GRID}
    invariance_checks = []
    for source, target in [(10, 20), (20, 10)]:
        key = f"{source}_to_{target}"
        ids = data[target]["ep"].astype(int)
        assert len(np.unique(ids)) == len(ids)
        source_digest = hashlib.sha256(source_profiles[source].tobytes()).hexdigest()
        for fold in range(5):
            calibration = ids % 5 == fold
            evaluation = ~calibration
            assert calibration.any() and evaluation.any()
            assert not np.any(calibration & evaluation)
            for episode, is_cal in zip(ids, calibration):
                split_rows.append(dict(direction=key, fold=fold, episode=int(episode),
                                       role="calibration" if is_cal else "evaluation"))
            c, u = geometry(upright, data[target]["axis"][calibration, -1])
            current = predictions(source_profiles[source], vectors[source],
                                  vectors[target][calibration], c, u)
            # Evaluation endpoints cannot affect calibration or any prediction.
            perturbed = data[target]["axis"].copy()
            perturbed[evaluation] = np.array([1.0, 0.0, 0.0])
            cc, uu = geometry(upright, perturbed[calibration, -1])
            masked = vectors[target].copy()
            masked[evaluation] = 999.0
            check = predictions(source_profiles[source], vectors[source],
                                masked[calibration], cc, uu)
            assert all(np.array_equal(current[m], check[m]) for m in METHODS)
            invariance_checks.append(dict(direction=key, fold=fold, passed=True))
            all_c, all_u = geometry(upright, data[target]["axis"][:, -1])
            modes = [("Calibration", c, u),
                     ("Nominal angle + calibrated axis", float(target), u),
                     ("Pooled terminal (reference)", all_c, all_u)]
            for mode, amplitude, axis in modes:
                geometry_rows.append(dict(direction=key, fold=fold, geometry=mode,
                                          c_deg=amplitude, ux=axis[0], uy=axis[1], uz=axis[2],
                                          n_calibration=int(calibration.sum()),
                                          n_evaluation=int(evaluation.sum())))
                pred = predictions(source_profiles[source], vectors[source],
                                   vectors[target][calibration], amplitude, axis)
                for method in METHODS:
                    error = np.linalg.norm(vectors[target][evaluation] - pred[method], axis=-1)
                    for episode, values in zip(ids[evaluation], error):
                        scores.append(dict(direction=key, fold=fold, geometry=mode,
                                           method=method, episode=int(episode),
                                           terminal_deg=values[-1], whole_deg=values.mean()))
                    if fold == 0 and mode == "Calibration":
                        archive[key + "_" + method + "_prediction"] = pred[method]
                        archive[key + "_" + method + "_errors"] = error
                if fold == 0 and mode == "Calibration":
                    measured = vectors[target][evaluation] @ axis
                    archive[key + "_measured_vectors"] = vectors[target][evaluation]
                    archive[key + "_evaluation_ids"] = ids[evaluation]
                    archive[key + "_measured_projected"] = measured
                    archive[key + "_frozen_projected"] = pred["Frozen source"] @ axis
                    archive[key + "_target_projected"] = pred["Target calibration"] @ axis
                    for i, progress in enumerate(S_GRID):
                        curves.append(dict(direction=key, progress=progress,
                                           measured_mean=measured[:, i].mean(),
                                           measured_sd=measured[:, i].std(ddof=1),
                                           frozen=(pred["Frozen source"] @ axis)[i],
                                           target_calibration=(pred["Target calibration"] @ axis)[i]))
            assert hashlib.sha256(source_profiles[source].tobytes()).hexdigest() == source_digest
    scores = pd.DataFrame(scores)
    scores.to_csv(OUT / "episode_scores.csv", index=False)
    summary = scores.groupby(["direction", "fold", "geometry", "method"]).agg(
        n=("episode", "size"), terminal_mean=("terminal_deg", "mean"),
        terminal_sd=("terminal_deg", "std"), whole_mean=("whole_deg", "mean"),
        whole_sd=("whole_deg", "std")).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    paired = scores.pivot(index=["direction", "fold", "geometry", "episode"],
                          columns="method", values=["terminal_deg", "whole_deg"])
    delta_rows = []
    for metric in ["terminal_deg", "whole_deg"]:
        for method in METHODS:
            if method == "Frozen source":
                continue
            delta = paired[metric][method] - paired[metric]["Frozen source"]
            for idx, value in delta.items():
                delta_rows.append(dict(zip(["direction", "fold", "geometry", "episode"], idx),
                                       metric=metric, comparator=method,
                                       comparator_minus_frozen=value))
    pd.DataFrame(delta_rows).to_csv(OUT / "paired_differences.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "splits.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(OUT / "geometry.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "figure_curves.csv", index=False)
    np.savez_compressed(OUT / "predictions.npz", **archive)
    inputs = [Path(__file__), ROOT / "CALIBRATION_SPLIT_PROTOCOL.md",
              ROOT / "step2b_alignment.py", ROOT / "se3_utils.py",
              ROOT / "fk_episodes.csv", ROOT / "metadata_audit.csv"]
    inputs += sorted((ROOT / "observations").glob("obs_*.npz"))
    inputs += sorted((ROOT / "fk_cache").glob("insert_jc_fix_3_*.npz"))
    (OUT / "provenance.json").write_text(json.dumps({
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, indent=2))
    (OUT / "verification.json").write_text(json.dumps({
        "evaluation_mutation_invariance": invariance_checks,
        "source_profile_unchanged": True, "operators_author_reported": 3,
        "operator_episode_mapping": "not established",
        "independent_cad_camera_calibration": False,
        "all_input_episodes_retained_across_calibration_and_evaluation": True,
        "split_sensitivity_is_not_independent_replication": True,
    }, indent=2))
    draw(archive, scores)
    for destination in [ROOT.parents[1] / "paper_submission",
                        ROOT.parents[1] / "paper_submission/candidate_rq4_revision"]:
        if not destination.is_dir():
            print(f"skipping figure copy to {destination} (directory not present)", flush=True)
            continue
        for ext in ["png", "pdf", "svg", "tiff"]:
            shutil.copy2(OUT / f"fig_physical_transfer_single_column.{ext}", destination)
    print(summary[(summary.fold == 0) & (summary.geometry == "Calibration")].to_string(index=False))


if __name__ == "__main__":
    main()
