import argparse
import os
from typing import List, Tuple

import numpy as np

from visualize import (
    find_optical_sync_files,
    load_optical_sync_file,
    pick_top_sequences,
)


RIGHT_SIDE_JOINTS = [
    "r_prox_scap",
    "r_caud_scap",
    "r_acro",
    "r_shdr",
    "r_elb",
    "r_dist_rad",
    "r_meta_carp",
    "r_iliac",
    "r_grt_trc",
    "r_isch_tub",
    "r_stifle",
    "r_hock",
    "r_meta_tars",
]

LEFT_FORELIMB_JOINTS = [
    "l_prox_scap",
    "l_caud_scap",
    "l_acro",
    "l_shdr",
    "l_elb",
    "l_dist_rad",
    "l_meta_carp",
]


def rigid_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the best-fit rigid transform that maps A -> B.
    Both inputs are [N, 3]. Returns (R, t).
    """
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def compute_per_frame_errors(frames: np.ndarray, joint_indices: List[int]) -> Tuple[np.ndarray, List[int]]:
    """
    Returns per-frame RMS errors and indices of frames used.
    """
    first_valid = None
    for idx in range(frames.shape[0]):
        pts = frames[idx, joint_indices, :]
        if not np.isnan(pts).any():
            first_valid = pts
            break
    if first_valid is None:
        raise RuntimeError("No valid frame found for the chosen joints.")

    errors = []
    used_frames = []
    ref_pts = first_valid
    for idx in range(frames.shape[0]):
        pts = frames[idx, joint_indices, :]
        if np.isnan(pts).any():
            continue
        R, t = rigid_transform(ref_pts, pts)
        pred = (R @ ref_pts.T).T + t
        err = np.linalg.norm(pred - pts, axis=1)
        errors.append(np.sqrt((err**2).mean()))
        used_frames.append(idx)
    return np.array(errors), used_frames


def fit_global_transform(frames: np.ndarray, joint_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Fit a single rigid transform over all valid frames and return:
      (R, t, per-frame RMS residuals, frame_indices_used)
    """
    ref_pts = frames[0, joint_indices, :]
    if np.isnan(ref_pts).any():
        raise RuntimeError("First frame has NaNs for the requested joints; pick a different reference.")

    stacked_ref = []
    stacked_cur = []
    frame_ids = []
    for idx in range(frames.shape[0]):
        pts = frames[idx, joint_indices, :]
        if np.isnan(pts).any():
            continue
        stacked_ref.append(ref_pts)
        stacked_cur.append(pts)
        frame_ids.append(idx)

    if not stacked_ref:
        raise RuntimeError("No valid frames found for the requested joints.")

    ref_stack = np.vstack(stacked_ref)
    cur_stack = np.vstack(stacked_cur)
    R, t = rigid_transform(ref_stack, cur_stack)

    per_frame_rms = []
    ptr = 0
    for idx in frame_ids:
        count = len(joint_indices)
        ref_block = ref_stack[ptr : ptr + count]
        cur_block = cur_stack[ptr : ptr + count]
        ptr += count
        pred = (R @ ref_block.T).T + t
        err = np.linalg.norm(pred - cur_block, axis=1)
        per_frame_rms.append(np.sqrt((err**2).mean()))

    return R, t, np.array(per_frame_rms), frame_ids


def main():
    parser = argparse.ArgumentParser(description="Rigid-link test for right-side joints.")
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/e/3ddogs",
        help="Dataset root containing Sync_Align* directory.",
    )
    parser.add_argument("--sequence-index", type=int, default=0, help="Index into the ranked top sequences.")
    parser.add_argument("--top-k", type=int, default=3, help="How many sequences to rank before selecting.")
    parser.add_argument(
        "--joints",
        type=str,
        nargs="*",
        default=RIGHT_SIDE_JOINTS,
        help="Names of joints to treat as a rigid body.",
    )
    parser.add_argument(
        "--global-fit",
        action="store_true",
        help="Fit a single rigid transform across all frames instead of per-frame.",
    )
    parser.add_argument(
        "--inspect",
        nargs="*",
        default=[],
        help="Optional list of joints to dump coordinates for debugging.",
    )
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=20,
        help="How many frames to print per inspected joint.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="Tolerance (meters) for declaring the link rigid.",
    )
    args = parser.parse_args()

    files = find_optical_sync_files(args.root)
    if not files:
        raise SystemExit("No optical sync files found.")
    ranked = pick_top_sequences(files, top_k=args.top_k)
    if not ranked:
        raise SystemExit("No ranked sequences produced.")
    seq_idx = max(0, min(args.sequence_index, len(ranked) - 1))
    seq_path = ranked[seq_idx][0]
    frames, joint_names = load_optical_sync_file(seq_path)
    print(f"Using sequence #{seq_idx}: {os.path.relpath(seq_path, args.root)}")

    idx_map = {name: i for i, name in enumerate(joint_names)}
    missing = [j for j in args.joints if j not in idx_map]
    if missing:
        raise SystemExit(f"Missing joints in file: {missing}")
    joint_indices = [idx_map[j] for j in args.joints]

    if args.inspect:
        print("\nJoint coordinate inspection:")
        for joint_name in args.inspect:
            if joint_name not in idx_map:
                print(f"  {joint_name}: NOT FOUND in this sequence.")
                continue
            j_idx = idx_map[joint_name]
            coords = frames[:, j_idx, :]
            valid_mask = ~np.isnan(coords).any(axis=1)
            print(
                f"  {joint_name}: valid {valid_mask.sum()}/{len(coords)} frames "
                f"({valid_mask.sum()/len(coords):.1%}). First {args.inspect_limit} frames:"
            )
            for frame_id in range(min(args.inspect_limit, frames.shape[0])):
                xyz = coords[frame_id]
                status = "NaN" if np.isnan(xyz).any() else f"{xyz}"
                print(f"    frame {frame_id:05d}: {status}")
        print("")

    if args.global_fit:
        R, t, errors, used_frames = fit_global_transform(frames, joint_indices)
        mode_desc = "global rigid fit"
    else:
        errors, used_frames = compute_per_frame_errors(frames, joint_indices)
        mode_desc = "per-frame rigid fit"

    if errors.size == 0:
        raise SystemExit("No frames with complete joint data; cannot evaluate.")

    max_err = errors.max()
    mean_err = errors.mean()
    perc99 = np.percentile(errors, 99)
    worst_idx = used_frames[int(np.argmax(errors))]
    print(f"Frames evaluated: {len(errors)} / {frames.shape[0]}")
    print(f"Mode: {mode_desc}")
    print(f"Mean RMS error: {mean_err:.5f} m")
    print(f"99th percentile RMS error: {perc99:.5f} m")
    print(f"Worst RMS error: {max_err:.5f} m at frame {worst_idx}")
    if max_err <= args.tol:
        print(f"Rigid constraint satisfied (max error <= {args.tol})")
    else:
        print(f"Rigid constraint violated (max error > {args.tol})")

    # Show a few sample errors to inspect temporal behavior
    sample = list(zip(used_frames, errors))[:10]
    print("Sample frame errors:")
    for frame_id, err in sample:
        print(f"  frame {frame_id:05d}: RMS {err:.5f} m")


if __name__ == "__main__":
    main()


