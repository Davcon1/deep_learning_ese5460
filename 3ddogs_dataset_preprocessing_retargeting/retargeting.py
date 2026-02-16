"""
Retargeting utilities for mapping dog motion capture keypoints to the Unitree Go2
URDF using Pinocchio.

This module is intentionally "early scaffolding":
- It knows how to load the Go2 URDF with Pinocchio.
- It parses optical-sync motion files into a structured array aligned with
  `KEY_JOINTS` from `nan_report.py`.
- It parses simple paw-contact schedules from the accompanying `_contact.txt`
  files.

The actual retargeting optimization (solving for Go2 generalized coordinates q
that best match a dog pose) will be built on top of these utilities.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from nan_report import KEY_JOINTS

try:
    # Prefer the canonical import if available.
    import pinocchio as pin  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - relaxed fallback
    import pin as pin  # type: ignore[import-not-found]


# ---------------------------
#  Contact schedule handling
# ---------------------------

PAW_NAMES = ("r_meta_carp", "l_meta_tars", "l_meta_carp", "r_meta_tars")

# Default global ground-plane height in the mocap coordinates (meters).
DEFAULT_GROUND_Y = -0.04

# Constant gravity for ballistic base motion (m/s^2, negative Y direction).
GRAVITY_Y = -9.81

# Rotation from mocap world (X-Z ground, Y up) to robot world (X-Y ground, Z up).
# We choose:
#   x_r = -x_m
#   y_r =  z_m
#   z_r =  y_m
# which is a proper rotation (determinant +1) and maps:
#   - mocap vertical (Y) -> robot vertical (Z)
#   - mocap ground plane XZ -> robot ground plane XY
R_MOCAP_TO_ROBOT = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


def mocap_to_robot_positions(positions: np.ndarray) -> np.ndarray:
    """
    Apply the fixed world-frame rotation from mocap (Y-up) to robot (Z-up).

    positions: array with shape (..., 3)
    """
    # positions are row vectors; we want p_r = R @ p_m, which is equivalent to
    # p_m^T @ R^T when treating them as row vectors. Hence the @ R.T.
    return positions @ R_MOCAP_TO_ROBOT.T


def estimate_base_pose_from_targets(
    go2: Go2Kinematics,
    targets_robot: np.ndarray,  # (N_key, 3) in robot world
    frame_names: Sequence[str],
) -> np.ndarray:
    """
    Estimate a good initial floating-base pose by rigidly aligning the
    keypoints of the neutral configuration to the target keypoints.
    """
    model, data = go2.model, go2.data
    q0 = go2.q_neutral().copy()

    # FK of keypoint frames at neutral pose.
    pos_dict = go2.frame_positions(q0, frame_names)
    P = np.stack([pos_dict[name] for name in frame_names], axis=0)  # (N, 3)
    Q = np.asarray(targets_robot, dtype=float)                      # (N, 3)

    # Center the point sets.
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    X = P - Pc
    Y = Q - Qc

    # Kabsch algorithm for best-fit rotation.
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = Qc - R @ Pc

    # Update base translation and orientation in q0.
    q0[:3] = t
    quat = pin.Quaternion(R)
    q0[3:7] = quat.coeffs()  # (x, y, z, w)
    return q0


@dataclass(frozen=True)
class ContactInterval:
    """Inclusive contact interval [start_frame, end_frame]."""

    start: int
    end: int

    def contains(self, frame_idx: int) -> bool:
        return self.start <= frame_idx <= self.end


class ContactSchedule:
    """
    Per-paw contact schedule.

    Backed by a dict:
        paw_name -> List[ContactInterval]
    """

    def __init__(self, intervals: Mapping[str, Sequence[ContactInterval]]):
        self._intervals: Dict[str, Tuple[ContactInterval, ...]] = {
            paw: tuple(seq) for paw, seq in intervals.items()
        }

    @classmethod
    def from_contact_file(cls, path: Path) -> "ContactSchedule":
        """
        Parse a `_contact.txt` file of the form:

            r_meta_carp (0, 6)
            l_meta_tars (0, 11)
            ...

        where the second entry is a Python-style tuple `(start, end)` with
        inclusive indices.
        """
        intervals: MutableMapping[str, List[ContactInterval]] = {}
        text = Path(path).read_text().strip().splitlines()
        for line in text:
            if not line.strip():
                continue
            # Example line: "r_meta_carp (0, 6)"
            name_part, range_part = line.split(None, 1)
            range_part = range_part.strip()
            if not (range_part.startswith("(") and range_part.endswith(")")):
                raise ValueError(f"Unexpected contact interval format: {line!r}")
            start_str, end_str = range_part[1:-1].split(",")
            start = int(start_str)
            end = int(end_str)
            intervals.setdefault(name_part, []).append(ContactInterval(start, end))

        return cls(intervals)

    def paws(self) -> Tuple[str, ...]:
        return tuple(self._intervals.keys())

    def intervals_for(self, paw: str) -> Tuple[ContactInterval, ...]:
        return self._intervals.get(paw, ())

    def in_contact(self, paw: str, frame_idx: int) -> bool:
        """Return True if `paw` is in contact at `frame_idx`."""
        return any(iv.contains(frame_idx) for iv in self._intervals.get(paw, ()))


def load_ground_estimates(path: Path) -> Dict[str, float]:
    """
    Parse an `*_ground_estimate.txt` file.

    Expected format (one line per paw, after a header line):

        RANSAC-estimated ground Y per joint:
        l_meta_carp: y≈-0.023120 m (inliers ...)
        ...

    Returns
    -------
    dict
        Mapping `joint_name` -> estimated ground y (float, meters).
    """
    path = Path(path)
    lines = path.read_text().strip().splitlines()
    if not lines:
        raise ValueError(f"Empty ground estimate file: {path}")

    estimates: Dict[str, float] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        # Example: "l_meta_carp: y≈-0.023120 m (inliers ...)"
        name_part, rest = line.split(":", 1)
        rest = rest.strip()
        try:
            y_str = rest.split("y≈", 1)[1].split()[0]
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not parse ground y from line: {line!r}") from exc
        estimates[name_part.strip()] = float(y_str)
    return estimates


def compute_paw_ground_deltas(
    ground_estimates: Mapping[str, float],
    plane_y: float = DEFAULT_GROUND_Y,
    paw_names: Sequence[str] = PAW_NAMES,
) -> Dict[str, float]:
    """
    Compute per-paw vertical deltas to move mocap markers onto a common ground plane.

    For each paw j, given an estimated "marker ground" height y_est[j], we assume the
    true ground plane is at `plane_y` and define:

        delta_y[j] = plane_y - y_est[j]

    Adding `delta_y[j]` to that paw's y-coordinate will place its average contact
    at `plane_y`, which better matches the URDF assumption that feet end exactly
    on the ground.
    """
    deltas: Dict[str, float] = {}
    for paw in paw_names:
        if paw not in ground_estimates:
            continue
        y_est = float(ground_estimates[paw])
        deltas[paw] = plane_y - y_est
    return deltas


def apply_paw_ground_deltas(
    motion: SourceMotion,
    deltas: Mapping[str, float],
) -> SourceMotion:
    """
    Shift each paw's y-coordinate by a fixed delta across the entire sequence.

    This is a simple, sequence-level correction that absorbs per-marker height
    offsets between mocap and the URDF's ground contact assumption.
    """
    if not deltas:
        return motion

    name_to_idx = {name: i for i, name in enumerate(motion.joint_names)}
    out = motion.positions.copy()

    for paw, dy in deltas.items():
        if paw not in name_to_idx:
            continue
        j = name_to_idx[paw]
        out[:, j, 1] += float(dy)  # y-axis is index 1

    return SourceMotion(fps=motion.fps, joint_names=motion.joint_names, positions=out)


def save_paw_ground_deltas(
    path: Path,
    deltas: Mapping[str, float],
    plane_y: float = DEFAULT_GROUND_Y,
) -> None:
    """
    Save per-paw ground offsets to a text file for reproducibility.

    Format:
        # plane_y = -0.040000
        l_meta_carp: dy=-0.016880
        ...
    """
    path = Path(path)
    lines: List[str] = []
    lines.append(f"# plane_y = {plane_y:.6f}\n")
    for paw in sorted(deltas.keys()):
        dy = float(deltas[paw])
        lines.append(f"{paw}: dy={dy:.6f}\n")
    path.write_text("".join(lines))


# ---------------------------
#  Source motion interface
# ---------------------------


@dataclass
class SourceMotion:
    """
    Dog motion sequence in a consistent coordinate frame.

    Attributes
    ----------
    fps:
        Frames per second.
    joint_names:
        Names of joints, aligned with `KEY_JOINTS`.
    positions:
        Array of shape (T, N, 3) giving 3D positions for each frame and joint.
    """

    fps: float
    joint_names: Tuple[str, ...]
    positions: np.ndarray  # (T, N, 3)

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.positions.shape[1])

    def frame(self, idx: int) -> np.ndarray:
        """Return positions for a single frame: shape (N, 3)."""
        return self.positions[idx]


def load_optical_sync_motion(path: Path) -> SourceMotion:
    """
    Load an `optical_sync_align_*.txt` file and return a `SourceMotion`
    containing just the joints in `KEY_JOINTS`, in that order.
    """
    path = Path(path)
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Unexpected optical sync file format: {path}")

    # First line: "fps 60.000000"
    header_tokens = lines[0].split()
    if header_tokens[0].lower() != "fps":
        raise ValueError(f"First line must start with 'fps', got: {lines[0]!r}")
    fps = float(header_tokens[1])

    # Header lines contain column layout; find the one whose FIRST TOKEN is "frame_num".
    header_idx: Optional[int] = None
    header_tokens: Optional[List[str]] = None
    for i, line in enumerate(lines):
        tokens = line.split()
        if tokens and tokens[0] == "frame_num":
            header_idx = i
            header_tokens = tokens
            break
    if header_idx is None or header_tokens is None:
        raise ValueError(f"No 'frame_num' header line found in {path}")

    joint_names_in_file = header_tokens[1:]

    # Map KEY_JOINTS to column indices.
    name_to_col = {name: i for i, name in enumerate(joint_names_in_file)}
    missing = [name for name in KEY_JOINTS if name not in name_to_col]
    if missing:
        raise KeyError(f"Joints {missing} not found in {path.name}")

    # Data lines: each line is "frame_idx x1 y1 z1 x2 y2 z2 ..."
    data_rows: List[List[float]] = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            continue
        tokens = line.split()
        # Drop frame index
        data = [float(tok) if tok != "NaN" else np.nan for tok in tokens[1:]]
        data_rows.append(data)

    data_arr = np.asarray(data_rows, dtype=float)  # shape (T, 3 * num_joints_total)
    num_frames = data_arr.shape[0]
    num_total_joints = len(joint_names_in_file)
    data_arr = data_arr.reshape(num_frames, num_total_joints, 3)

    # Stack the KEY_JOINTS in the canonical order.
    key_joint_positions = np.stack(
        [data_arr[:, name_to_col[j], :] for j in KEY_JOINTS], axis=1
    )  # (T, N_key, 3)

    return SourceMotion(
        fps=fps,
        joint_names=tuple(KEY_JOINTS),
        positions=key_joint_positions,
    )


# ---------------------------
#  Go2 kinematics via URDF
# ---------------------------


@dataclass
class Go2Kinematics:
    """
    Lightweight wrapper around Pinocchio's model/data for the Go2 URDF.

    This exposes:
    - `q_neutral`: a nominal standing configuration (all zeros by default).
    - `frame_positions(q, frame_names)`: FK for a set of frames.
    - `frame_jacobians(q, frame_names)`: spatial Jacobians.
    """

    model: "pin.Model"
    data: "pin.Data"
    frame_name_to_id: Dict[str, int]

    @classmethod
    def from_urdf(cls, urdf_path: Path) -> "Go2Kinematics":
        urdf_path = Path(urdf_path)

        # Build a model whose root joint is a free-flyer base. This matches the
        # intended generalized coordinates q = [p_b, h, θ, joint angles],
        # where p_b is base position and (h, θ) encode base orientation.
        root_joint = pin.JointModelFreeFlyer()
        model = pin.buildModelFromUrdf(str(urdf_path), root_joint)
        data = model.createData()

        frame_name_to_id = {f.name: i for i, f in enumerate(model.frames)}
        return cls(model=model, data=data, frame_name_to_id=frame_name_to_id)

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    def q_neutral(self) -> np.ndarray:
        """
        Nominal "neutral" configuration.

        For now this is just zero configuration in joint space, with base at
        origin and identity orientation. We can refine this later to a proper
        static-stance configuration if needed.
        """
        return pin.neutral(self.model)

    # ---- FK / Jacobians ----

    def _ensure_frame_ids(self, frame_names: Iterable[str]) -> List[int]:
        ids: List[int] = []
        for name in frame_names:
            try:
                ids.append(self.frame_name_to_id[name])
            except KeyError:
                raise KeyError(f"Frame '{name}' not found in Go2 model") from None
        return ids

    def frame_positions(
        self, q: np.ndarray, frame_names: Sequence[str]
    ) -> Dict[str, np.ndarray]:
        """
        Compute 3D positions of given frames in the world frame.

        Returns
        -------
        dict
            Mapping frame_name -> position (3,) expressed in world coordinates.
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        out: Dict[str, np.ndarray] = {}
        for name in frame_names:
            fid = self.frame_name_to_id[name]
            placement = self.data.oMf[fid]  # type: ignore[index]
            out[name] = np.array(placement.translation).copy()
        return out

    def frame_jacobians(
        self, q: np.ndarray, frame_names: Sequence[str], reference: str = "world"
    ) -> Dict[str, np.ndarray]:
        """
        Compute 6x`nv` Jacobians for given frames.

        Parameters
        ----------
        reference:
            Passed through to Pinocchio's `computeFrameJacobian`, typically
            one of {"world", "local", "local_world_aligned"}.
        """
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        jacobians: Dict[str, np.ndarray] = {}
        for name in frame_names:
            fid = self.frame_name_to_id[name]
            J = pin.computeFrameJacobian(
                self.model, self.data, q, fid, pin.ReferenceFrame[reference]
            )
            jacobians[name] = np.asarray(J)
        return jacobians


# ---------------------------
#  Dog limb rescaling (Unit Vector Method)
# ---------------------------

# Four 4-joint chains, in the same order as `KEY_JOINTS`.
LEG_KEYPOINTS: Dict[str, Tuple[str, str, str, str]] = {
    "LF": ("l_prox_scap", "l_shdr", "l_elb", "l_meta_carp"),
    "LH": ("l_iliac", "l_grt_trc", "l_hock", "l_meta_tars"),
    "RF": ("r_prox_scap", "r_shdr", "r_elb", "r_meta_carp"),
    "RH": ("r_iliac", "r_grt_trc", "r_hock", "r_meta_tars"),
}

# Corresponding Go2 frames for those 4 keypoints, read from the URDF.
GO2_LEG_FRAMES: Dict[str, Tuple[str, str, str, str]] = {
    "LF": ("FL_hip", "FL_thigh", "FL_calf", "FL_foot"),
    "LH": ("RL_hip", "RL_thigh", "RL_calf", "RL_foot"),
    "RF": ("FR_hip", "FR_thigh", "FR_calf", "FR_foot"),
    "RH": ("RR_hip", "RR_thigh", "RR_calf", "RR_foot"),
}


def _segment_lengths_from_positions(
    P: Mapping[str, np.ndarray],
    frames: Tuple[str, str, str, str],
) -> np.ndarray:
    """Compute three segment lengths along a 4-node chain."""
    a, b, c, d = frames
    p_a, p_b, p_c, p_d = P[a], P[b], P[c], P[d]
    return np.array(
        [
            np.linalg.norm(p_b - p_a),
            np.linalg.norm(p_c - p_b),
            np.linalg.norm(p_d - p_c),
        ],
        dtype=float,
    )


def compute_go2_segment_lengths(
    go2: Go2Kinematics, q_ref: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Measure Go2 leg segment lengths from the URDF at configuration `q_ref`.

    This gives, for each leg key ('LF', 'LH', 'RF', 'RH'), a length-3 array
    corresponding to hip→thigh, thigh→knee, knee→foot segments.
    """
    if q_ref is None:
        q_ref = go2.q_neutral()

    # Query all relevant frames in one FK call.
    all_frame_names = sorted({name for frames in GO2_LEG_FRAMES.values() for name in frames})
    P = go2.frame_positions(q_ref, all_frame_names)

    lengths: Dict[str, np.ndarray] = {}
    for leg, frames in GO2_LEG_FRAMES.items():
        lengths[leg] = _segment_lengths_from_positions(P, frames)
    return lengths


def _safe_unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < eps:
        return np.zeros_like(v)
    return v / norm


def rescale_limbs_to_go2_lengths(
    motion: SourceMotion,
    go2: Go2Kinematics,
    segment_lengths: Optional[Mapping[str, np.ndarray]] = None,
) -> SourceMotion:
    """
    Apply the "Unit Vector Method" to resize each dog limb to Go2 segment lengths.

    PAW-ANCHORED VERSION:
      For each 4-joint chain (hip/thigh/knee/foot), we:
        1. Keep the distal joint (paw) fixed in place.
        2. March *upwards* along the limb using Go2 segment lengths and the
           original bone directions, so that hip/thigh/knee move while the paw
           stays on the ground.
    """
    if segment_lengths is None:
        segment_lengths = compute_go2_segment_lengths(go2)

    name_to_idx = {name: i for i, name in enumerate(motion.joint_names)}
    out_positions = motion.positions.copy()

    for leg, joint_names in LEG_KEYPOINTS.items():
        idxs = [name_to_idx[j] for j in joint_names]
        L_leg = np.asarray(segment_lengths[leg], dtype=float)  # (3,) = [hip-thigh, thigh-knee, knee-foot]

        for t in range(motion.num_frames):
            p0, p1, p2, p3 = out_positions[t, idxs, :]

            # Skip if distal point is invalid.
            if not np.all(np.isfinite(p3)):
                continue

            # Directions from distal -> proximal along the original limb.
            d32 = _safe_unit(p2 - p3)  # foot -> knee
            d21 = _safe_unit(p1 - p2)  # knee -> thigh
            d10 = _safe_unit(p0 - p1)  # thigh -> hip

            # If any direction is zero (e.g. missing data), skip this leg/frame.
            if (
                not np.any(d32)
                or not np.any(d21)
                or not np.any(d10)
            ):
                continue

            n3 = p3
            n2 = n3 + L_leg[2] * d32
            n1 = n2 + L_leg[1] * d21
            n0 = n1 + L_leg[0] * d10

            out_positions[t, idxs, :] = np.stack([n0, n1, n2, n3], axis=0)

    return SourceMotion(
        fps=motion.fps,
        joint_names=motion.joint_names,
        positions=out_positions,
    )


def _clamp_to_joint_limits(model: "pin.Model", q: np.ndarray) -> np.ndarray:
    """
    Clamp configuration `q` to the joint limits encoded in the Pinocchio model.

    Notes
    -----
    - The free-flyer base (first 7 DoFs) typically has infinite limits; those
      entries are effectively left unchanged except for quaternion
      re-normalization.
    - Only finite lower/upper bounds are enforced, component-wise.
    """
    q_clamped = q.copy()
    lower = np.asarray(model.lowerPositionLimit, dtype=float)
    upper = np.asarray(model.upperPositionLimit, dtype=float)

    for i in range(model.nq):
        lo = lower[i]
        hi = upper[i]
        if np.isfinite(lo):
            if q_clamped[i] < lo:
                q_clamped[i] = lo
        if np.isfinite(hi):
            if q_clamped[i] > hi:
                q_clamped[i] = hi

    # Ensure base quaternion remains unit-length.
    try:
        quat = pin.Quaternion(q_clamped[3:7])
        quat.normalize()
        q_clamped[3:7] = quat.coeffs()
    except Exception:
        # Defensive: if quaternion is malformed, leave as-is.
        pass

    return q_clamped


def _compute_go2_hip_positions(
    go2: Go2Kinematics,
    q_ref: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Query Go2 hip frame positions for each leg in a reference configuration.

    Returns
    -------
    dict
        Mapping leg key ('LF', 'LH', 'RF', 'RH') -> position (3,) in world frame.
    """
    if q_ref is None:
        q_ref = go2.q_neutral()

    # First frame of each GO2_LEG_FRAMES chain is the hip link.
    hip_frame_names: Dict[str, str] = {
        leg: GO2_LEG_FRAMES[leg][0] for leg in ("LF", "LH", "RF", "RH")
    }
    pos_dict = go2.frame_positions(q_ref, list(hip_frame_names.values()))
    out: Dict[str, np.ndarray] = {}
    for leg, frame_name in hip_frame_names.items():
        out[leg] = pos_dict[frame_name]
    return out


def align_hips_to_go2(
    motion: SourceMotion,
    go2: Go2Kinematics,
    q_ref: Optional[np.ndarray] = None,
    debug: bool = False,
) -> SourceMotion:
    """
    Second-stage UVM that aligns the four hip joints to the Go2 URDF layout.

    Current limb UVM (`rescale_limbs_to_go2_lengths`) operates independently
    per leg (paw→hip) and does not enforce any relationship between the four
    hips. This function adds a *rigid* layout step:

      - Use the Go2 URDF to get the nominal 3D positions of {FL, FR, RL, RR}
        hip frames in a reference configuration.
      - Treat that set of four hips as a rigid shape.
      - For each mocap frame:
          * Keep the left-front hip (LF) exactly where the UVM paw→hip pass
            placed it.
          * Find the rotation R that best aligns the URDF hip layout to the
            current mocap hip layout (in a LS sense), with LF as the anchor.
          * Place a rigidly rotated copy of the URDF hip layout at the LF
            mocap hip.
          * For each leg, translate the *entire* 4-joint chain by a constant
            offset so that its hip coincides with the aligned URDF hip.

    This guarantees (up to numerical error) that the pairwise distances between
    the four hips match the URDF's rigid-body distances for every frame, while
    staying as close as possible in orientation to the original mocap layout.
    """
    # URDF hip layout (rigid reference).
    hip_go2 = _compute_go2_hip_positions(go2, q_ref=q_ref)
    anchor_leg = "LF"  # left-front as the global anchor

    # Relative URDF hip positions with LF at the origin.
    p0 = hip_go2[anchor_leg]
    legs_order = ("LF", "RF", "LH", "RH")
    rel_go2 = {leg: hip_go2[leg] - p0 for leg in legs_order}

    # Indices of hip joints in the mocap keypoint set for each leg.
    name_to_idx = {name: i for i, name in enumerate(motion.joint_names)}
    hip_joint_indices: Dict[str, int] = {}
    for leg, joint_names in LEG_KEYPOINTS.items():
        hip_name = joint_names[0]
        if hip_name not in name_to_idx:
            raise KeyError(f"Hip joint '{hip_name}' for leg '{leg}' not found in motion.joint_names")
        hip_joint_indices[leg] = name_to_idx[hip_name]

    out_positions = motion.positions.copy()
    T = motion.num_frames

    # Optional debug accumulators.
    max_err = 0.0
    if debug:
        # All pairwise URDF distances between hips for reference.
        urdf_dists: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(legs_order):
            for b in legs_order[i + 1 :]:
                d = float(np.linalg.norm(hip_go2[a] - hip_go2[b]))
                urdf_dists[(a, b)] = d

    for t in range(T):
        # Mocap/UVM hip positions this frame.
        hip_dog: Dict[str, np.ndarray] = {}
        valid = True
        for leg in legs_order:
            j = hip_joint_indices[leg]
            p = out_positions[t, j, :]
            if not np.all(np.isfinite(p)):
                valid = False
                break
            hip_dog[leg] = p
        if not valid:
            # Skip this frame; leave as-is.
            continue

        p_anchor = hip_dog[anchor_leg]

        # Build point sets for Kabsch: URDF layout vs current mocap layout,
        # both expressed relative to the anchor.
        X = np.stack([rel_go2[leg] for leg in legs_order[1:]], axis=0)  # (3, 3) rows
        Y = np.stack(
            [hip_dog[leg] - p_anchor for leg in legs_order[1:]],
            axis=0,
        )

        # Kabsch algorithm for best-fit rotation (no scaling).
        H = X.T @ Y
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0.0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T

        # New hip positions: rigid transform of URDF layout with LF anchored.
        new_hips: Dict[str, np.ndarray] = {}
        for leg in legs_order:
            rel = rel_go2[leg]
            new_hips[leg] = p_anchor + R @ rel

        # Translate each entire leg chain so its hip matches new_hips[leg].
        for leg in legs_order:
            j_hip = hip_joint_indices[leg]
            old_hip = out_positions[t, j_hip, :]
            delta = new_hips[leg] - old_hip

            idxs = [name_to_idx[name] for name in LEG_KEYPOINTS[leg]]
            out_positions[t, idxs, :] += delta

        if debug:
            # Check pairwise hip distances against URDF.
            for i, a in enumerate(legs_order):
                for b in legs_order[i + 1 :]:
                    d_cur = float(np.linalg.norm(new_hips[a] - new_hips[b]))
                    d_ref = float(np.linalg.norm(hip_go2[a] - hip_go2[b]))
                    max_err = max(max_err, abs(d_cur - d_ref))

    if debug:
        print(f"[align_hips_to_go2] max |distance error| over all frames: {max_err:.6e} m")

    return SourceMotion(
        fps=motion.fps,
        joint_names=motion.joint_names,
        positions=out_positions,
    )


# ---------------------------
#  STMR-style SMR from UVM
# ---------------------------


def _build_style_foot_anchors_from_uvm(
    targets_robot_feet: np.ndarray,  # (T, 4, 3)
    contact: np.ndarray,  # (T, num_feet) bool
    plane_y: float,
) -> np.ndarray:
    """
    Build per-foot anchor points purely from UVM foot trajectories.

    For each contact segment, we:
      - Take the mocap/UVM foot position at contact start.
      - Project it to the ground plane z = plane_y (robot vertical).
      - Keep that anchor fixed for the whole contact segment.
    """
    T, num_feet = contact.shape
    anchors = np.full((T, num_feet, 3), np.nan, dtype=float)

    for j in range(num_feet):
        current_anchor: Optional[np.ndarray] = None
        for t in range(T):
            if not contact[t, j]:
                current_anchor = None
                continue
            if current_anchor is None:
                p = targets_robot_feet[t, j].copy()
                if not np.all(np.isfinite(p)):
                    # Skip if the style target is invalid.
                    continue
                # Robot world uses Z as vertical.
                p[2] = plane_y
                current_anchor = p
            anchors[t, j, :] = current_anchor

    return anchors


@dataclass
class STMRResult:
    q: np.ndarray        # (T, nq)
    qdot: np.ndarray     # (T, nv)


# Mapping from dog keypoints to Go2 frames for SMR/keypoints.
KEYPOINT_TO_GO2_FRAME: Dict[str, str] = {}
for leg, dog_keys in LEG_KEYPOINTS.items():
    go2_frames = GO2_LEG_FRAMES[leg]
    for dog_key, frame_name in zip(dog_keys, go2_frames):
        KEYPOINT_TO_GO2_FRAME[dog_key] = frame_name

# Feet only (subset of KEY_JOINTS), in semantic order.
FOOT_KEYPOINTS: Tuple[str, ...] = (
    "l_meta_carp",
    "r_meta_carp",
    "l_meta_tars",
    "r_meta_tars",
)


def build_keypoint_frame_names() -> Tuple[str, ...]:
    """
    Return Go2 frame names corresponding to `KEY_JOINTS`, in the same order.
    """
    return tuple(KEYPOINT_TO_GO2_FRAME[name] for name in KEY_JOINTS)


def run_smr_from_uvm(
    go2: Go2Kinematics,
    motion_uvm: SourceMotion,
    contact_schedule: ContactSchedule,
    paw_names: Sequence[str],
    dt: float,
    plane_y: float = DEFAULT_GROUND_Y,
    lambda_kp: float = 1e-3,
    Kp_anchor: float = 50.0,
    Q_diag: Optional[np.ndarray] = None,
    rho_anchor: float = 100.0,
) -> STMRResult:
    """
    STMR-style SMR that works *directly* from UVM keypoints, without a
    separate full IK stage.

    Pipeline
    --------
    1. Transform UVM keypoints from mocap world (Y-up) to robot world (Z-up).
    2. Use Kabsch-based base pose initialization at frame 0 (all keypoints).
    3. For each frame i:
         - Build a "style" generalized velocity v_style by matching all
           keypoint velocities (via least squares using Jacobians).
         - Run a velocity QP with foot-anchor constraints to get v_i:

               min_v  1/2 ||v - v_style||_Q^2 + rho/2 ||A v - b||^2

         - Integrate q_{i+1} = integrate(q_i, v_i * dt).
    """
    T = motion_uvm.num_frames
    nq, nv = go2.nq, go2.nv

    # All keypoint frame names and UVM targets in robot world.
    frame_names_all = build_keypoint_frame_names()
    targets_robot_all = mocap_to_robot_positions(motion_uvm.positions)  # (T, N_key, 3)

    # Contact matrix from schedule.
    contact = build_contact_matrix(contact_schedule, T, paw_names)

    # Foot frames for anchors (one per paw).
    num_feet = len(paw_names)
    foot_frames = [DOG_TO_GO2_FRAMES[paw] for paw in paw_names]
    foot_frame_ids = [go2.frame_name_to_id[fname] for fname in foot_frames]

    # Anchors p_tilde[i, j]: per-frame, per-foot, defined *from robot FK* at
    # stance entry, then held fixed for the entire contact segment.
    anchors = np.full((T, num_feet, 3), np.nan, dtype=float)
    current_anchors: List[Optional[np.ndarray]] = [None] * num_feet

    # Initial base pose from all keypoints at frame 0.
    q0 = estimate_base_pose_from_targets(
        go2,
        targets_robot_all[0],
        frame_names_all,
    )
    q0 = _clamp_to_joint_limits(go2.model, q0)

    q_traj = np.zeros((T, nq), dtype=float)
    qdot_traj = np.zeros((T, nv), dtype=float)
    q_traj[0] = q0

    model, data = go2.model, go2.data

    for i in range(T - 1):
        q_i = q_traj[i]

        # --- Forward kinematics at current configuration ---
        pin.forwardKinematics(model, data, q_i)
        pin.updateFramePlacements(model, data)

        # --- Update stance foot anchors from robot FK, not from mocap ---
        for j, fid in enumerate(foot_frame_ids):
            if contact[i, j]:
                if current_anchors[j] is None:
                    placement = data.oMf[fid]
                    p = np.array(placement.translation)
                    # Project to ground plane: robot vertical is Z.
                    p[2] = plane_y
                    current_anchors[j] = p
                anchors[i, j, :] = current_anchors[j]
            else:
                current_anchors[j] = None

        J_blocks: List[np.ndarray] = []
        r_blocks: List[np.ndarray] = []

        targets_now = targets_robot_all[i]

        for k, fname in enumerate(frame_names_all):
            fid = go2.frame_name_to_id[fname]
            placement = data.oMf[fid]
            p = np.array(placement.translation)
            r = targets_now[k] - p  # (3,)
            r_blocks.append(r)

            J6 = pin.computeFrameJacobian(
                model, data, q_i, fid, pin.LOCAL_WORLD_ALIGNED
            )
            Jpos = np.asarray(J6[3:, :])  # 3 x nv
            J_blocks.append(Jpos)

        J = np.vstack(J_blocks)          # (3N, nv)
        r = np.concatenate(r_blocks, 0)  # (3N,)

        # Linearized objective:
        #   min_v 1/2 || J dt v - r ||^2 + (lambda_kp/2) ||v||^2
        # so that p(q_i + v dt) ≈ p(q_i) + J dt v tracks x_target.
        J_eff = J * dt
        nv = go2.nv
        JTJ = J_eff.T @ J_eff + lambda_kp * np.eye(nv)
        rhs = J_eff.T @ r

        try:
            v_style = np.linalg.solve(JTJ, rhs)
        except np.linalg.LinAlgError:
            eps = 1e-8
            v_style = np.linalg.solve(JTJ + eps * np.eye(nv), rhs)

        feet_in_contact = [j for j in range(len(paw_names)) if contact[i, j]]

        if feet_in_contact:
            v = _solve_velocity_qp(
                go2,
                q_i,
                v_style,
                feet_in_contact,
                paw_names,
                anchors[i],
                Kp=Kp_anchor,
                Q_diag=Q_diag,
                rho=rho_anchor,
            )
        else:
            # No contact: follow style velocity.
            v = v_style

        qdot_traj[i] = v
        q_next = pin.integrate(model, q_i, v * dt)
        q_next = _clamp_to_joint_limits(model, q_next)
        q_traj[i + 1] = q_next

    if T > 1:
        qdot_traj[-1] = qdot_traj[-2]

    return STMRResult(q=q_traj, qdot=qdot_traj)


# ---------------------------
#  SMR with contact anchors and ballistic flight
# ---------------------------


def build_contact_matrix(
    schedule: ContactSchedule,
    num_frames: int,
    paw_names: Sequence[str],
) -> np.ndarray:
    """
    Convert a `ContactSchedule` into a boolean matrix c[i, j].
    """
    c = np.zeros((num_frames, len(paw_names)), dtype=bool)
    for j, paw in enumerate(paw_names):
        for iv in schedule.intervals_for(paw):
            start = max(0, iv.start)
            end = min(num_frames - 1, iv.end)
            if end >= start:
                c[start : end + 1, j] = True
    return c


def _solve_velocity_qp(
    go2: Go2Kinematics,
    q: np.ndarray,
    delta_q_tilde: np.ndarray,  # (nv,)
    contact_feet: Sequence[int],
    paw_names: Sequence[str],
    anchors_t: np.ndarray,  # (num_feet, 3)
    Kp: float,
    Q_diag: Optional[np.ndarray] = None,
    rho: float = 100.0,
) -> np.ndarray:
    """
    Solve the per-frame velocity QP for stance feet using an ADMM-style method.

    We pose the problem as:

        minimize_v  1/2 ||v - delta_q_tilde||_Q^2
        subject to  A v = b

    where:
      - `delta_q_tilde` is the "style" velocity (from FK/keypoints),
      - Q is a positive-definite diagonal weighting matrix (defaults to I),
      - A, b encode foot-position constraints at the current frame.

    This follows the SMR idea of treating foot constraints as (approximately)
    hard while keeping the style term soft. ADMM gives a numerically robust
    solve even near singularities.
    """
    model, data = go2.model, go2.data
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    nv = go2.nv
    d = delta_q_tilde
    if Q_diag is None:
        Q = np.eye(nv)
    else:
        Q = np.diag(Q_diag)

    rows: List[np.ndarray] = []
    rhs: List[np.ndarray] = []

    for j in contact_feet:
        paw = paw_names[j]
        frame_name = DOG_TO_GO2_FRAMES[paw]
        fid = go2.frame_name_to_id[frame_name]
        placement = data.oMf[fid]
        p = np.array(placement.translation)
        p_anchor = anchors_t[j]
        if not np.all(np.isfinite(p_anchor)):
            continue
        err = p_anchor - p
        J6 = pin.computeFrameJacobian(
            model, data, q, fid, pin.LOCAL_WORLD_ALIGNED
        )
        Jpos = np.asarray(J6[3:, :])
        rows.append(Jpos)
        rhs.append(Kp * err)

    if not rows:
        # No active constraints for this frame.
        return d

    A = np.vstack(rows)          # (3 * n_feet_active, nv)
    b = np.concatenate(rhs, 0)   # (3 * n_feet_active,)

    # ADMM parameters.
    rho_admm = float(rho)
    max_iters = 50
    tol = 1e-4

    # Precompute matrices for v-update: (Q + rho A^T A) v = Q d + rho A^T (b - u).
    At = A.T
    M = Q + rho_admm * (At @ A)

    # Initialize.
    v = d.copy()
    u = np.zeros_like(b)

    for _ in range(max_iters):
        # v-update.
        rhs_v = Q @ d + rho_admm * (At @ (b - u))
        try:
            v = np.linalg.solve(M, rhs_v)
        except np.linalg.LinAlgError:
            eps = 1e-8
            v = np.linalg.solve(M + eps * np.eye(nv), rhs_v)

        # y is always equal to b (projection onto equality constraint set).
        Av = A @ v
        r_primal = Av - b

        # u-update.
        u = u + r_primal

        if np.linalg.norm(r_primal) < tol:
            break

    return v


# ---------------------------
#  Dog→Go2 mapping scaffold
# ---------------------------

# NOTE: This is a *placeholder* mapping of dog keypoints to Go2 frames,
# focusing on the feet for now. We will refine / extend this once we agree on
# the exact semantic correspondence between dog joints and robot links.
DOG_TO_GO2_FRAMES: Dict[str, str] = {
    # Fore feet
    "l_meta_carp": "FL_foot",
    "r_meta_carp": "FR_foot",
    # Hind feet
    "l_meta_tars": "RL_foot",
    "r_meta_tars": "RR_foot",
    # TODO: add shoulders / hips etc. once we lock down the mapping.
}


def dog_feet_to_go2_frames() -> Dict[str, str]:
    """Small helper exposing the current foot mapping."""
    return dict(DOG_TO_GO2_FRAMES)


__all__ = [
    "ContactInterval",
    "ContactSchedule",
    "DEFAULT_GROUND_Y",
    "load_ground_estimates",
    "compute_paw_ground_deltas",
    "apply_paw_ground_deltas",
    "save_paw_ground_deltas",
    "SourceMotion",
    "Go2Kinematics",
    "load_optical_sync_motion",
    "LEG_KEYPOINTS",
    "GO2_LEG_FRAMES",
    "compute_go2_segment_lengths",
    "rescale_limbs_to_go2_lengths",
    "align_hips_to_go2",
    "run_smr_from_uvm",
    "dog_feet_to_go2_frames",
]


