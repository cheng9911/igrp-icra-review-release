# Retrospective calibration/evaluation split

This protocol is fixed before computing this new analysis, but the datasets
and earlier pooled-terminal results have already been inspected.

- Reuse the existing 50 upright, 61 ten-degree and 52 twenty-degree episodes,
  established grasp-to-pre-release normalized progress, and nominal matches.
- For either direction, fit the source mean normalized profile using only
  upright and source-condition data. Keep it unchanged across target splits.
- Primary target calibration: episode_index modulo 5 equals 0. All remaining
  target episodes are evaluation only. Estimate one common target axis and
  amplitude from calibration terminal axes, relative to the upright mean.
- Never use evaluation terminal axes to fit geometry or any predictive profile.
  Existing target-based offline resampling and pickup matching remain explicit.
- Compare raw source-vector reuse, identity profile, linear progress profile,
  frozen source profile, and a target profile estimated on the calibration
  subset only. The last method has a nonzero target-law fitting budget and is
  not an oracle or a mathematical upper bound.
- Evaluate every method on identical target evaluation episodes. Report mean
  per-episode mean vector error and mean terminal error, with episode SD.
  Rotation-vector error describes cylinder-axis response, not full pose error.
- Geometry sensitivity: calibration-derived geometry; reported nominal angle
  plus calibration-derived axis (partial nominal geometry only); pooled
  all-target terminal geometry (retrospective non-held-out reference).
- Repeat the prespecified modulo split for residues 1 through 4 as sensitivity.
  Do not select the best split; the same targets recur across these runs.
- Three operators were reported by the author. Episode-to-operator/session
  mapping is not currently established. Episode SD is descriptive, not
  independent-subject uncertainty. Do not fabricate a hierarchical bootstrap.
- Show frozen prediction and empirical projected target response in each
  direction, an episode SD envelope, and terminal mean/SD method comparisons.
  Preserve source data and emit split IDs, predictions, scores and hashes.
- No independent CAD/camera geometry is available, as confirmed by the author.
  No new physical runs or online target-pose predictions are claimed.
