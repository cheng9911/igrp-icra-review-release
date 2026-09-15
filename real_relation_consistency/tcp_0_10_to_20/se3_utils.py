"""Self-contained Lie-group + Franka-FK utilities for the 0/10/20 cylinder-response run.

Lie-group functions are copied verbatim (semantics-preserving) from
`phase_switch_symmetry/phase_switch_se3_baselines.py` so that this run does not
pull in its sklearn dependency. FK and the terminal-window rule are copied from
`real_relation_consistency/analyze_fk.py` (the user-confirmed convention).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing
from scipy.spatial.transform import Rotation

# --- SO(3)/SE(3) Lie-group math (twist convention xi = [v(3); w(3)]) ----------


def _skew(w):
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=np.float64,
    )


def so3_exp(w):
    w = np.asarray(w, dtype=np.float64)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + _skew(w)
    W = _skew(w)
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * W
        + ((1.0 - np.cos(theta)) / (theta * theta)) * (W @ W)
    )


def so3_log(R):
    R = np.asarray(R, dtype=np.float64)
    theta = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if theta < 1e-12:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def se3_exp(xi):
    xi = np.asarray(xi, dtype=np.float64)
    v = xi[:3]
    w = xi[3:]
    theta = float(np.linalg.norm(w))
    R = so3_exp(w)
    if theta < 1e-12:
        V = np.eye(3) + 0.5 * _skew(w)
    else:
        W = _skew(w)
        V = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / (theta * theta)) * W
            + ((theta - np.sin(theta)) / (theta ** 3)) * (W @ W)
        )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    w = so3_log(R)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        Vinv = np.eye(3) - 0.5 * _skew(w)
    else:
        W = _skew(w)
        Vinv = (
            np.eye(3)
            - 0.5 * W
            + (1.0 / (theta * theta))
            * (1.0 - (theta * np.sin(theta)) / (2.0 * (1.0 - np.cos(theta))))
            * (W @ W)
        )
    v = Vinv @ t
    return np.concatenate([v, w])


def se3_exp_batched(twists):
    """Vectorized SE(3) exponential: (..., 6) -> (..., 4, 4)."""
    twists = np.asarray(twists, dtype=np.float64)
    v = twists[..., :3]
    w = twists[..., 3:]
    theta = np.linalg.norm(w, axis=-1)
    theta_safe = np.where(theta < 1e-12, 1.0, theta)
    small = theta < 1e-12

    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    W = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    W[..., 0, 1] = -wz
    W[..., 0, 2] = wy
    W[..., 1, 0] = wz
    W[..., 1, 2] = -wx
    W[..., 2, 0] = -wy
    W[..., 2, 1] = wx
    W2 = np.einsum("...ij,...jk->...ik", W, W)

    a = np.where(small, 1.0, np.sin(theta_safe) / theta_safe)
    b = np.where(small, 0.5, (1.0 - np.cos(theta_safe)) / (theta_safe ** 2))
    c = np.where(small, 1.0 / 6.0, (theta_safe - np.sin(theta_safe)) / (theta_safe ** 3))

    eye = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    eye[..., 0, 0] = eye[..., 1, 1] = eye[..., 2, 2] = 1.0
    R = eye + a[..., None, None] * W + b[..., None, None] * W2
    V = eye + b[..., None, None] * W + c[..., None, None] * W2
    t = np.einsum("...ij,...j->...i", V, v)

    T = np.zeros(w.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_inverse(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def se3_inverse_batched(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def so3_geodesic_deg(R1, R2):
    """Geodesic rotation angle (deg) between two SO(3) matrices = ||Log(R1^T R2)||."""
    return float(np.degrees(np.linalg.norm(so3_log(R1.T @ R2))))


def so3_log_batched(R):
    """Vectorized SO(3) log: (..., 3, 3) -> (..., 3)."""
    R = np.asarray(R, dtype=np.float64)
    cosang = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cosang)
    vee = np.stack(
        [R[..., 2, 1] - R[..., 1, 2],
         R[..., 0, 2] - R[..., 2, 0],
         R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    small = theta < 1e-12
    scale = np.where(small, 0.5, theta / (2.0 * np.sin(theta)))
    return vee * scale[..., None]


# --- Franka FK (from analyze_fk.py) -------------------------------------------

XYZ = [[0, 0, .333], [0, 0, 0], [0, -.316, 0], [.0825, 0, 0],
       [-.0825, .384, 0], [0, 0, 0], [.088, 0, 0]]
ROLL = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]


def fk(q):
    """Base -> default hand TCP. URDF origin transform precedes each Rz(q)."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
        raise ValueError('Expected finite (N,7) joint angles in radians')
    t = np.tile(np.eye(4), (len(q), 1, 1))
    for j in range(7):
        origin = np.eye(4)
        origin[:3, :3] = Rotation.from_euler('x', ROLL[j]).as_matrix()
        origin[:3, 3] = XYZ[j]
        z = np.tile(np.eye(4), (len(q), 1, 1))
        z[:, :3, :3] = Rotation.from_euler('z', q[:, j]).as_matrix()
        t = t @ origin @ z
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_euler('z', -np.pi / 4).as_matrix()
    tool[2, 3] = .107 + .1034
    return t @ tool


def cylinder_axis_from_rot(R):
    """Cylinder local axis in base frame: -R[:,2] (same as analyze_fk's upward_hand_axis)."""
    return -R[..., :, 2]


def terminal_window(state, pos, fps=30, threshold=.5, window=.5, guard=.2):
    """Grasp->pre-release segmentation (frozen from analyze_fk.py)."""
    raw = state[:, 7] < threshold
    closed = raw | binary_closing(raw, structure=np.ones(int(fps * .4) + 1))
    starts = np.flatnonzero(np.diff(np.r_[False, closed].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[closed, False].astype(int)) == -1) + 1
    segments = [(s, e) for s, e in zip(starts, ends) if e - s >= fps and e < len(state)]
    if not segments:
        return None
    s, e = max(segments, key=lambda se: np.linalg.norm(pos[se[1] - 1] - pos[se[0]]))
    end = e - round(guard * fps)
    start = end - round(window * fps)
    if start < s or end <= start:
        return None
    return dict(grasp_start=int(s), release_frame=int(e), window_start=int(start),
                window_end_exclusive=int(end), n_grasp_segments=len(segments),
                transport_distance_m=float(np.linalg.norm(pos[e - 1] - pos[s])))


def average_pose(pos, rot):
    """Per-episode terminal pose: mean position + quaternion chordal rotation mean."""
    pos = np.asarray(pos, dtype=np.float64)
    rot = np.asarray(rot, dtype=np.float64)
    p = pos.mean(axis=0)
    R = Rotation.from_matrix(rot).mean().as_matrix()
    return p, R


def resample_pose(pos, rot, n_points, frame_start, frame_end_exclusive):
    """Resample a pose curve over frame indices [start, end) to n_points samples.

    Position: linear interpolation. Rotation: sign-continuous quaternion nlerp.
    Returns (pos (n_points,3), rot (n_points,3,3)).
    """
    n = frame_end_exclusive - frame_start
    if n < 2:
        raise ValueError('too few frames to resample')
    idx = np.arange(frame_start, frame_end_exclusive)
    pos_seg = np.asarray(pos)[frame_start:frame_end_exclusive]
    rot_seg = np.asarray(rot)[frame_start:frame_end_exclusive]
    target = np.linspace(frame_start, frame_end_exclusive - 1, n_points)
    p = np.column_stack([np.interp(target, idx, pos_seg[:, c]) for c in range(3)])
    quat = Rotation.from_matrix(rot_seg).as_quat()  # xyzw
    # sign continuity
    for i in range(1, len(quat)):
        if quat[i] @ quat[i - 1] < 0:
            quat[i] *= -1
    q = np.column_stack([np.interp(target, idx, quat[:, c]) for c in range(4)])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = Rotation.from_quat(q).as_matrix()
    return p, R


def angle_between(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.degrees(np.arccos(np.clip(dot / np.maximum(denom, 1e-12), -1, 1)))


def pos_error_mm(p_obs, p_pred):
    return np.linalg.norm(np.asarray(p_obs) - np.asarray(p_pred), axis=-1) * 1000.0


def rot_error_deg(R_obs, R_pred):
    """Per-sample geodesic error = ||Log(R_obs^T R_pred)|| in degrees."""
    R_obs = np.asarray(R_obs, dtype=np.float64)
    R_pred = np.asarray(R_pred, dtype=np.float64)
    flat = (np.swapaxes(R_obs, -1, -2) @ R_pred).reshape(-1, 3, 3)
    out = np.empty(len(flat), dtype=np.float64)
    for i, R in enumerate(flat):
        out[i] = np.degrees(np.linalg.norm(so3_log(R)))
    return out.reshape(R_obs.shape[:-2])


def axis_error_deg(a_obs, a_pred):
    return angle_between(np.asarray(a_obs), np.asarray(a_pred))
