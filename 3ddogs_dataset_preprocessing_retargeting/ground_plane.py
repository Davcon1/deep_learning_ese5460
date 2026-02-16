import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from visualize import load_optical_sync_file

# Default paw joints
PAW_JOINTS = ["l_meta_carp", "r_meta_carp", "l_meta_tars", "r_meta_tars"]
ContactRange = Tuple[int, int]


def parse_contact_file(path: str) -> Dict[str, List[ContactRange]]:
    joint_ranges: Dict[str, List[ContactRange]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if not parts:
                continue
            name = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            m = re.search(r"\((\d+)\s*,\s*(\d+)\)", rest)
            if not m:
                continue
            start = int(m.group(1))
            end = int(m.group(2))
            if end < start:
                start, end = end, start
            joint_ranges[name].append((start, end))
    return joint_ranges


def main():
    parser = argparse.ArgumentParser(description="Inspect paw coordinates or analyze contact ranges.")
    parser.add_argument("file", type=str, help="Path to optical_sync_align_*.txt file.")
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame index to inspect single-frame coordinates. If omitted, skips single frame inspection.",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=PAW_JOINTS,
        help="List of paw joints to consider.",
    )
    parser.add_argument(
        "--contact-file",
        type=str,
        default="",
        help="Optional path to *_contact.txt to analyze per-range medians.",
    )
    parser.add_argument(
        "--y-tol",
        type=float,
        default=0.005,
        help="Tolerance (meters) for Y deviation when reporting percentages (default 5mm).",
    )
    parser.add_argument(
        "--xyz-tol",
        type=float,
        default=0.01,
        help="Tolerance (meters) for 3D distance from median point (default 1cm).",
    )
    parser.add_argument(
        "--ransac-y-tol",
        type=float,
        default=0.005,
        help="Inlier tolerance for RANSAC Y estimation (default 5mm).",
    )
    parser.add_argument(
        "--ransac-iters",
        type=int,
        default=200,
        help="Number of RANSAC iterations when estimating per-joint Y (default 200).",
    )
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        raise SystemExit(f"File not found: {file_path}")

    frames, joint_names = load_optical_sync_file(file_path)
    if args.frame is not None:
        if args.frame < 0 or args.frame >= frames.shape[0]:
            raise SystemExit(f"Frame {args.frame} out of range (0..{frames.shape[0]-1})")

    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    paw_data = []
    for joint in args.joints:
        if joint not in name_to_idx:
            print(f"Joint {joint} not present; skipping.")
            continue
        idx = name_to_idx[joint]
        coord = frames[args.frame, idx, :]
        paw_data.append((joint, coord))

    if not paw_data:
        raise SystemExit("No joints available to inspect.")

    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

    if args.frame is not None:
        if args.frame < 0 or args.frame >= frames.shape[0]:
            raise SystemExit(f"Frame {args.frame} out of range (0..{frames.shape[0]-1})")
        paw_data = []
        for joint in args.joints:
            if joint not in name_to_idx:
                print(f"Joint {joint} not present; skipping.")
                continue
            idx = name_to_idx[joint]
            coord = frames[args.frame, idx, :]
            paw_data.append((joint, coord))

        if not paw_data:
            raise SystemExit("No joints available to inspect.")

        print(f"File: {file_path}")
        print(f"Frame {args.frame} paw coordinates:")
        for joint, coord in paw_data:
            status = "NaN" if np.isnan(coord).any() else f"{coord}"
            print(f"  {joint:>12}: {status}")

        valid_paws = [(joint, coord) for joint, coord in paw_data if not np.isnan(coord).any()]
        if len(valid_paws) >= 2:
            valid_paws.sort(key=lambda item: item[1][1])
            lowest_two = valid_paws[:2]
            print("\nLowest two paws by Y (ground contacts):")
            for joint, coord in lowest_two:
                print(f"  {joint:>12}: y={coord[1]:.6f}, x={coord[0]:.6f}, z={coord[2]:.6f}")

            front_idx = np.argmin([item[1][0] for item in lowest_two])
            back_idx = 1 - front_idx
            front = lowest_two[front_idx]
            back = lowest_two[back_idx]
            print(f"\nLikely front paw: {front[0]} (lower x).")
            print(f"Likely rear paw:  {back[0]} (higher x).")

            def expected_side(name: str) -> str:
                return "front" if "carp" in name else "rear"

            front_expected = expected_side(front[0])
            back_expected = expected_side(back[0])
            if front_expected != "front":
                print(f"[WARN] Heuristic says front, but joint '{front[0]}' name implies {front_expected}.")
            if back_expected != "rear":
                print(f"[WARN] Heuristic says rear, but joint '{back[0]}' name implies {back_expected}.")
        else:
            print("Not enough valid paw coordinates to locate ground contacts.")
    # name_to_idx already defined for following contact analysis

    if args.contact_file:
        contact_path = os.path.abspath(args.contact_file)
        if not os.path.exists(contact_path):
            raise SystemExit(f"Contact file not found: {contact_path}")
        ranges = parse_contact_file(contact_path)
        print(
            f"\nAnalyzing contact ranges from {contact_path} "
            f"(Y tolerance={args.y_tol:.4f} m, XYZ tolerance={args.xyz_tol:.4f} m)"
        )

        rng = np.random.default_rng(0)
        per_joint_ys: Dict[str, List[float]] = {joint: [] for joint in args.joints}

        for joint in args.joints:
            if joint not in name_to_idx:
                print(f"\nJoint {joint} not present; skipping contact analysis.")
                continue
            joint_ranges = ranges.get(joint, [])
            if not joint_ranges:
                print(f"\nJoint {joint}: no contact ranges provided.")
                continue
            idx = name_to_idx[joint]
            print(f"\nJoint {joint}:")
            for start, end in joint_ranges:
                s = max(start, 0)
                e = min(end, frames.shape[0] - 1)
                if e < s:
                    continue
                segment = frames[s : e + 1, idx, :]
                mask = ~np.isnan(segment).any(axis=1)
                pts = segment[mask]
                valid = pts.shape[0]
                if valid == 0:
                    print(f"  Range {start}->{end}: no valid frames (all NaNs).")
                    continue
                median_point = np.median(pts, axis=0)
                y_dev = np.abs(pts[:, 1] - median_point[1])
                xyz_dev = np.linalg.norm(pts - median_point, axis=1)
                pct_y = (y_dev <= args.y_tol).mean() * 100.0
                pct_xyz = (xyz_dev <= args.xyz_tol).mean() * 100.0
                per_joint_ys[joint].extend(pts[:, 1].tolist())
                print(f"  Range {start}->{end} (valid {valid} frames):")
                print(
                    f"    median point = [{median_point[0]:.6f}, {median_point[1]:.6f}, {median_point[2]:.6f}]"
                )
                print(
                    f"    Y deviation mean={y_dev.mean():.6f} m, max={y_dev.max():.6f} m, "
                    f"%<=tol={pct_y:.2f}%"
                )
                print(
                    f"    3D deviation mean={xyz_dev.mean():.6f} m, max={xyz_dev.max():.6f} m, "
                    f"%<=tol={pct_xyz:.2f}%"
                )

        print("\nRANSAC-estimated ground Y per joint:")
        for joint, ys_list in per_joint_ys.items():
            ys = np.asarray(ys_list, dtype=np.float64)
            if ys.size == 0:
                print(f"  {joint}: no samples.")
                continue
            best_inliers = np.array([])
            for _ in range(max(1, args.ransac_iters)):
                sample = rng.choice(ys)
                mask = np.abs(ys - sample) <= args.ransac_y_tol
                if mask.sum() > best_inliers.size:
                    best_inliers = ys[mask]
            if best_inliers.size == 0:
                estimate = float(np.median(ys))
                coverage = 0.0
                print(
                    f"  {joint}: insufficient inliers, fallback median={estimate:.6f} "
                    f"(samples {ys.size})"
                )
            else:
                estimate = float(np.median(best_inliers))
                coverage = best_inliers.size / ys.size * 100.0
                print(
                    f"  {joint}: y≈{estimate:.6f} m (inliers {best_inliers.size}/{ys.size}, "
                    f"{coverage:.2f}% within ±{args.ransac_y_tol:.4f})"
                )


if __name__ == "__main__":
    main()


