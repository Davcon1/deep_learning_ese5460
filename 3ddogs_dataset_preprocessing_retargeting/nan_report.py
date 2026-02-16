import argparse
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from visualize import find_optical_sync_files, load_optical_sync_file


# Joints we care about for "motion completeness" of the limbs.
KEY_JOINTS: Sequence[str] = (
    # Left forelimb
    "l_prox_scap",
    "l_shdr",
    "l_elb",
    "l_meta_carp",
    # Left hindlimb
    "l_iliac",
    "l_grt_trc",
    "l_hock",
    "l_meta_tars",
    # Right forelimb
    "r_prox_scap",
    "r_shdr",
    "r_elb",
    "r_meta_carp",
    # Right hindlimb
    "r_iliac",
    "r_grt_trc",
    "r_hock",
    "r_meta_tars",
)


def _longest_true_run(mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Return (start_idx, end_idx_exclusive, length) of the longest contiguous
    run of True values in a 1D boolean array. If no True values exist,
    returns (0, 0, 0).
    """
    assert mask.ndim == 1
    best_start = 0
    best_len = 0
    cur_start = None

    for i, val in enumerate(mask):
        if val:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                cur_len = i - cur_start
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                cur_start = None

    # Handle run that reaches the end.
    if cur_start is not None:
        cur_len = len(mask) - cur_start
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start

    return best_start, best_start + best_len, best_len


def analyze_key_joints_sequence(
    frames: np.ndarray,
    joint_names: Sequence[str],
    gap_tolerance: int = 5,
) -> Dict[str, object]:
    """
    Analyze how "complete" a sequence is for the key joints, allowing
    interpolation over short NaN gaps (<= gap_tolerance frames) and
    rigid-Transform-based reconstruction of shoulders when acro + elbow
    are present.
    """
    num_frames, num_joints, _ = frames.shape
    name_to_idx = {name: i for i, name in enumerate(joint_names)}

    missing_joints: List[str] = [j for j in KEY_JOINTS if j not in name_to_idx]
    present_key_joints: List[str] = [j for j in KEY_JOINTS if j in name_to_idx]
    if not present_key_joints:
        # Nothing useful for this file.
        return {
            "num_frames": int(num_frames),
            "missing_joints": missing_joints,
            "present_joints": present_key_joints,
            "best_run_start": 0,
            "best_run_end": 0,
            "best_run_length": 0,
            "good_frame_fraction": 0.0,
            "frames_using_interp": 0,
            "frames_using_shoulder_rigid": 0,
        }

    key_indices = [name_to_idx[j] for j in present_key_joints]
    # NaNs only in joints we care about.
    key_nans = np.isnan(frames[:, key_indices, :]).any(axis=2)  # [F, K]

    # Allow interpolation over short contiguous NaN gaps per joint.
    can_interp = np.zeros_like(key_nans, dtype=bool)  # [F, K]
    F, K = key_nans.shape
    for j in range(K):
        col = key_nans[:, j]
        i = 0
        while i < F:
            if not col[i]:
                i += 1
                continue
            start = i
            while i < F and col[i]:
                i += 1
            end = i  # [start, end) is a run of True
            run_len = end - start
            if run_len <= gap_tolerance:
                can_interp[start:end, j] = True

    effective_valid = (~key_nans) | can_interp  # per-joint, per-frame

    # Shoulder reconstruction using acro + elbow (rigid transform idea).
    shoulder_specs = [
        ("l_shdr", "l_acro", "l_elb"),
        ("r_shdr", "r_acro", "r_elb"),
    ]
    shoulder_rigid_frames = np.zeros(F, dtype=bool)

    for shoulder_name, acro_name, elb_name in shoulder_specs:
        if shoulder_name not in present_key_joints:
            continue
        if acro_name not in name_to_idx or elb_name not in name_to_idx:
            continue

        s_col = present_key_joints.index(shoulder_name)
        s_idx = name_to_idx[shoulder_name]
        acro_idx = name_to_idx[acro_name]
        elb_idx = name_to_idx[elb_name]

        shoulder_nan = key_nans[:, s_col]
        acro_valid = ~np.isnan(frames[:, acro_idx, :]).any(axis=1)
        elb_valid = ~np.isnan(frames[:, elb_idx, :]).any(axis=1)

        reconstructable = shoulder_nan & acro_valid & elb_valid
        if reconstructable.any():
            effective_valid[:, s_col] |= reconstructable
            shoulder_rigid_frames |= reconstructable

    frame_good = effective_valid.all(axis=1)  # all key joints OK (or fixable) in this frame
    best_start, best_end, best_len = _longest_true_run(frame_good)

    frames_using_interp = (key_nans & can_interp).any(axis=1).sum()
    frames_using_shoulder_rigid = int(shoulder_rigid_frames.sum())
    good_frame_fraction = float(frame_good.sum() / max(1, num_frames))

    return {
        "num_frames": int(num_frames),
        "missing_joints": missing_joints,
        "present_joints": present_key_joints,
        "best_run_start": int(best_start),
        "best_run_end": int(best_end),
        "best_run_length": int(best_len),
        "good_frame_fraction": good_frame_fraction,
        "frames_using_interp": int(frames_using_interp),
        "frames_using_shoulder_rigid": frames_using_shoulder_rigid,
    }


def detailed_key_joint_report(
    frames: np.ndarray,
    joint_names: Sequence[str],
    gap_tolerance: int = 5,
) -> None:
    """
    For a single file, print:
      - longest contiguous segment where all key joints are valid / fixable
      - per-frame list (within that segment) of joints that need interpolation
        or shoulder rigid reconstruction.
    """
    num_frames, _, _ = frames.shape
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    present = [j for j in KEY_JOINTS if j in name_to_idx]
    missing = [j for j in KEY_JOINTS if j not in name_to_idx]

    print("\nKey-joint detailed report:")
    print(f"  present key joints: {present}")
    if missing:
        print(f"  missing key joints: {missing}")

    if not present:
        print("  No key joints present in this file.")
        return

    key_idx = [name_to_idx[j] for j in present]
    key_nans = np.isnan(frames[:, key_idx, :]).any(axis=2)  # [F, K]

    # Mark NaN runs short enough for interpolation.
    can_interp = np.zeros_like(key_nans, dtype=bool)
    F, K = key_nans.shape
    for j in range(K):
        col = key_nans[:, j]
        i = 0
        while i < F:
            if not col[i]:
                i += 1
                continue
            start = i
            while i < F and col[i]:
                i += 1
            end = i
            if end - start <= gap_tolerance:
                can_interp[start:end, j] = True

    # Shoulder rigid reconstruction.
    shoulder_specs = [("l_shdr", "l_acro", "l_elb"), ("r_shdr", "r_acro", "r_elb")]
    shoulder_rigid = np.zeros_like(key_nans, dtype=bool)
    for shoulder_name, acro_name, elb_name in shoulder_specs:
        if shoulder_name not in present:
            continue
        if acro_name not in name_to_idx or elb_name not in name_to_idx:
            continue
        s_col = present.index(shoulder_name)
        s_idx = name_to_idx[shoulder_name]
        a_idx = name_to_idx[acro_name]
        e_idx = name_to_idx[elb_name]
        shoulder_nan = key_nans[:, s_col]
        acro_valid = ~np.isnan(frames[:, a_idx, :]).any(axis=1)
        elb_valid = ~np.isnan(frames[:, e_idx, :]).any(axis=1)
        recon = shoulder_nan & acro_valid & elb_valid
        shoulder_rigid[:, s_col] |= recon

    effective_valid = (~key_nans) | can_interp | shoulder_rigid
    frame_good = effective_valid.all(axis=1)
    best_start, best_end, best_len = _longest_true_run(frame_good)

    print(
        f"\n  best key-joint run: start={best_start}, end={best_end - 1} "
        f"(inclusive), length={best_len} frames"
    )
    if best_len <= 0:
        print("  No contiguous segment where all key joints are valid/fixable.")
        return

    print("\n  Frames within best run that need work:")
    any_work = False
    for f in range(best_start, best_end):
        interp_j = np.nonzero(key_nans[f] & can_interp[f])[0]
        rigid_j = np.nonzero(shoulder_rigid[f])[0]
        if not len(interp_j) and not len(rigid_j):
            continue
        any_work = True
        parts = []
        if len(interp_j):
            parts.append(
                "interpolate [" + ", ".join(present[j] for j in interp_j) + "]"
            )
        if len(rigid_j):
            parts.append(
                "rigid_shoulder [" + ", ".join(present[j] for j in rigid_j) + "]"
            )
        print(f"    frame {f}: " + "; ".join(parts))

    if not any_work:
        print("    (none; all key joints fully valid within this run)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "NaN analysis for optical sync files.\n"
            "- Default: detailed NaN report for a single file.\n"
            "- With --scan-key-joints: scan all files and find longest segments "
            "where key limb joints are usable (with small gaps allowed)."
        )
    )
    parser.add_argument(
        "file",
        type=str,
        nargs="?",
        default="",
        help="Path to optical_sync_align_*.txt (absolute or relative). "
        "If omitted and --scan-key-joints is given, all files are scanned.",
    )
    parser.add_argument(
        "--show-frames",
        type=int,
        default=5,
        help="How many NaN frame indices to print per joint in single-file mode (default 5).",
    )
    parser.add_argument(
        "--scan-key-joints",
        action="store_true",
        help=(
            "Scan all optical sync files under --root and find the longest contiguous "
            "segments where key limb joints are valid (allowing small NaN gaps and "
            "rigid shoulder reconstruction)."
        ),
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/e/3ddogs",
        help="Dataset root containing Sync_Align_v2023_11_16b (used with --scan-key-joints).",
    )
    parser.add_argument(
        "--detail-key-joints",
        action="store_true",
        help=(
            "For a single file: report the best key-joint run and, within that run, "
            "which frames/joints need interpolation or shoulder rigid reconstruction."
        ),
    )
    parser.add_argument(
        "--gap-tolerance",
        type=int,
        default=5,
        help="Maximum length (in frames) of contiguous NaN gaps per joint that we treat as interpolatable.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="How many most-complete files to print when using --scan-key-joints.",
    )
    args = parser.parse_args()

    if args.scan_key_joints and not args.file:
        # Global scan mode over all sync files.
        root = os.path.abspath(args.root)
        optical_files = find_optical_sync_files(root)
        if not optical_files:
            raise SystemExit(f"No optical sync alignment files found under root {root}")

        print(
            f"Scanning {len(optical_files)} files under {root} "
            f"for key-joint completeness (gap_tolerance={args.gap_tolerance} frames)..."
        )
        results: List[Tuple[str, Dict[str, object]]] = []
        for path in optical_files:
            try:
                frames, joints = load_optical_sync_file(path)
            except Exception as exc:
                print(f"[WARN] Failed to load {path}: {exc}")
                continue
            stats = analyze_key_joints_sequence(frames, joints, gap_tolerance=args.gap_tolerance)
            results.append((path, stats))

        # Sort by longest best_run_length, then by good_frame_fraction, then by num_frames.
        results.sort(
            key=lambda item: (
                -int(item[1]["best_run_length"]),
                -float(item[1]["good_frame_fraction"]),
                -int(item[1]["num_frames"]),
            )
        )

        print("\nMost complete motions for key joints:")
        for rank, (path, s) in enumerate(results[: args.top_k], start=1):
            rel = os.path.relpath(path, root)
            best_len = s["best_run_length"]
            if best_len <= 0:
                continue
            print(
                f"{rank:2d}. {rel:40s}  "
                f"best_run={best_len:4d} frames "
                f"(frames {s['best_run_start']}–{s['best_run_end'] - 1}), "
                f"good_frac={s['good_frame_fraction']:.3f}, "
                f"interp_frames={s['frames_using_interp']}, "
                f"shoulder_rigid_frames={s['frames_using_shoulder_rigid']}"
            )
            missing = s["missing_joints"]
            if missing:
                print(f"      missing key joints: {', '.join(missing)}")

        print(
            "\nTip: re-run this script on an individual file to see full per-joint NaN stats, e.g.\n"
            "  python nan_report.py Sync_Align_v2023_11_16b/optical_sync_align_d63_t6_a.txt"
        )
        return

    # Default: single-file detailed NaN report (all joints).
    if not args.file:
        raise SystemExit("Please provide a file path, or use --scan-key-joints to scan all files.")

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        raise SystemExit(f"File not found: {file_path}")

    frames, joint_names = load_optical_sync_file(file_path)
    print(f"Analyzing {file_path}")
    print(f"Total frames: {frames.shape[0]}, joints: {len(joint_names)}")

    if args.detail_key_joints:
        detailed_key_joint_report(
            frames,
            joint_names,
            gap_tolerance=args.gap_tolerance,
        )

    total_nan_frames = 0
    joint_nan_counts = []
    for j, name in enumerate(joint_names):
        mask = np.isnan(frames[:, j, :]).any(axis=1)
        count = int(mask.sum())
        joint_nan_counts.append((name, count, mask))
        if count:
            total_nan_frames += 1

    joint_nan_counts.sort(key=lambda x: x[1], reverse=True)

    print("\nPer-joint NaN counts (sorted descending):")
    for name, count, mask in joint_nan_counts:
        percent = count / frames.shape[0] * 100.0
        print(f"{name:>15}: {count:4d} frames ({percent:6.2f}%)")
        if count and args.show_frames > 0:
            idxs = np.nonzero(mask)[0][: args.show_frames]
            idx_list = ", ".join(str(i) for i in idxs)
            print(f"    first {len(idxs)} frames: {idx_list}")

    print("\nFrames with ANY joint NaN:")
    per_frame_mask = np.isnan(frames).any(axis=(1, 2))
    count_frames = int(per_frame_mask.sum())
    percent_frames = count_frames / frames.shape[0] * 100.0
    print(f"{count_frames} / {frames.shape[0]} frames ({percent_frames:.2f}%) contain at least one NaN")
    if count_frames and args.show_frames > 0:
        idxs = np.nonzero(per_frame_mask)[0][: args.show_frames]
        print("    first frames:", ", ".join(str(i) for i in idxs))


if __name__ == "__main__":
    main()



