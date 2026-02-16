from __future__ import annotations

"""
Small helper to inspect hip–hip distances in a saved UVM file and compare
them to the rigid distances implied by the Go2 URDF.

Example:

    cd /mnt/e/3ddogs
    python debug_hip_distances.py \
        --uvm optical_sync_align_d29_t1_a_ready_go2_uvm.npz \
        --urdf UnitreeGo2/urdf/go2_description.urdf \
        --frames 0 10 20
"""

import argparse
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

from retargeting import (  # type: ignore[import]
    Go2Kinematics,
    LEG_KEYPOINTS,
    GO2_LEG_FRAMES,
)


def _compute_go2_hip_positions(
    go2: Go2Kinematics,
) -> dict[str, np.ndarray]:
    """Query Go2 hip frame positions for each leg in the neutral configuration."""
    q_ref = go2.q_neutral()
    hip_frame_names = {leg: GO2_LEG_FRAMES[leg][0] for leg in ("LF", "LH", "RF", "RH")}
    pos_dict = go2.frame_positions(q_ref, list(hip_frame_names.values()))
    out: dict[str, np.ndarray] = {}
    for leg, frame_name in hip_frame_names.items():
        out[leg] = pos_dict[frame_name]
    return out


def _pair_label(a: str, b: str) -> str:
    return f"{a}-{b}"


def print_hip_distances(
    uvm_path: Path,
    urdf_path: Path,
    frame_indices: Sequence[int],
) -> None:
    data = np.load(uvm_path, allow_pickle=True)
    positions = data["positions"]  # (T, N, 3)
    joint_names = [str(x) for x in data["joint_names"]]
    T, N, _ = positions.shape

    go2 = Go2Kinematics.from_urdf(urdf_path)
    hip_go2 = _compute_go2_hip_positions(go2)

    # Reference URDF distances between hips.
    legs_order: Tuple[str, ...] = ("LF", "RF", "LH", "RH")
    urdf_dists: dict[Tuple[str, str], float] = {}
    for i, a in enumerate(legs_order):
        for b in legs_order[i + 1 :]:
            d = float(np.linalg.norm(hip_go2[a] - hip_go2[b]))
            urdf_dists[(a, b)] = d

    # Indices of hip joints in the keypoint list.
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    hip_joint_indices: dict[str, int] = {}
    for leg, joint_names_leg in LEG_KEYPOINTS.items():
        hip_name = joint_names_leg[0]
        if hip_name not in name_to_idx:
            raise KeyError(
                f"Hip joint '{hip_name}' for leg '{leg}' not found in joint_names"
            )
        hip_joint_indices[leg] = name_to_idx[hip_name]

    print(f"UVM file: {uvm_path}")
    print(f"URDF:     {urdf_path}")
    print("\nReference URDF hip–hip distances (m):")
    for (a, b), d in sorted(urdf_dists.items(), key=lambda kv: kv[0]):
        print(f"  {a}-{b}: {d:.6f}")

    print("\nFrame-wise hip–hip distances from UVM positions:")
    for t in frame_indices:
        if t < 0 or t >= T:
            print(f"\n[frame {t}] (out of range, skipping)")
            continue
        hip_pos: dict[str, np.ndarray] = {}
        valid = True
        for leg in legs_order:
            j = hip_joint_indices[leg]
            p = positions[t, j, :]
            if not np.all(np.isfinite(p)):
                print(f"[frame {t}] leg {leg}: non-finite hip position, skipping frame")
                valid = False
                break
            hip_pos[leg] = p
        if not valid:
            continue

        print(f"\n[frame {t}]")
        for (a, b), d_ref in sorted(urdf_dists.items(), key=lambda kv: kv[0]):
            d_cur = float(np.linalg.norm(hip_pos[a] - hip_pos[b]))
            err = d_cur - d_ref
            print(f"  {a}-{b}: current={d_cur:.6f}  ref={d_ref:.6f}  err={err:+.3e}")


def _parse_frame_list(raw: Iterable[str], T: int) -> Sequence[int]:
    out: list[int] = []
    for tok in raw:
        if "-" in tok:
            s, e = tok.split("-", 1)
            start = int(s)
            end = int(e)
            out.extend(range(start, min(end + 1, T)))
        else:
            out.append(int(tok))
    if not out:
        out = list(range(min(5, T)))
    return sorted(sorted(set(i for i in out if 0 <= i < T)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print hip–hip distances from a UVM .npz and compare to URDF."
    )
    parser.add_argument(
        "--uvm",
        type=str,
        required=True,
        help="Path to <sequence>_go2_uvm.npz",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="UnitreeGo2/urdf/go2_description.urdf",
        help="Path to Go2 URDF.",
    )
    parser.add_argument(
        "--frames",
        type=str,
        nargs="*",
        default=["0", "1", "2", "10", "20"],
        help=(
            "Frame indices or ranges to inspect, e.g. '0 1 10-15'. "
            "Defaults to a few early frames."
        ),
    )
    args = parser.parse_args()

    uvm_path = Path(args.uvm)
    urdf_path = Path(args.urdf)
    if not uvm_path.is_file():
        raise FileNotFoundError(uvm_path)
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)

    # Peek at number of frames to clip ranges.
    data = np.load(uvm_path, allow_pickle=True)
    T = int(data["positions"].shape[0])
    frame_indices = _parse_frame_list(args.frames, T)

    print_hip_distances(uvm_path, urdf_path, frame_indices)


if __name__ == "__main__":
    main()


