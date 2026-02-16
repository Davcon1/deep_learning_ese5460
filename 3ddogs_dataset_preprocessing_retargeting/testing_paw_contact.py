import argparse
import os
import re
from collections import defaultdict
from typing import DefaultDict, Dict, List, Sequence, Tuple

import numpy as np

from visualize import load_optical_sync_file


ContactRange = Tuple[int, int]


def parse_contact_file(path: str) -> Dict[str, List[ContactRange]]:
    """
    Parse a *_contact.txt file of the form:
        l_meta_carp (1, 14)
        r_meta_tars (3, 13)
        ...

    Returns a dict: joint_name -> list of (start_frame, end_frame), inclusive.
    Lines without a valid (start, end) pair are ignored.
    """
    joint_ranges: DefaultDict[str, List[ContactRange]] = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Expect "<name> (start, end)"
            parts = line.split(None, 1)
            if not parts:
                continue
            joint_name = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            m = re.search(r"\((\d+)\s*,\s*(\d+)\)", rest)
            if not m:
                # e.g., "l_meta_tars ()" – skip empty ranges
                continue
            start = int(m.group(1))
            end = int(m.group(2))
            if end < start:
                # Swap if accidentally reversed
                start, end = end, start
            joint_ranges[joint_name].append((start, end))

    return dict(joint_ranges)


def read_fps(path: str) -> float:
    """
    Read the FPS value from the optical_sync_align_*.txt header.
    Falls back to 60.0 if not found.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("fps"):
                    tokens = line.strip().split()
                    if len(tokens) >= 2:
                        return float(tokens[1])
                    break
    except FileNotFoundError:
        pass
    return 60.0


def compute_pairwise_speeds(
    frames: np.ndarray,
    joint_index: int,
    ranges: Sequence[ContactRange],
    fps: float,
) -> Dict[ContactRange, Dict[str, object]]:
    """
    For a given joint and a list of inclusive frame ranges, compute:
      - per-pair displacement (m) between frame k and k+1
      - per-pair velocity (m/s) using the sequence FPS
      - summary stats per range (mean, max, std) over valid pairs

    Additionally, for each range we also collect two context pairs
    immediately before and after the range (when available):
      - before:  (start-2 -> start-1), (start-1 -> start)
      - after:   (end   -> end+1),   (end+1   -> end+2)

    Returns:
        {
          (start, end): {
              "pairs": List[Tuple[int, int, float, float]],          # in-range (k, k+1, disp, vel)
              "context_before": List[Tuple[int, int, float, float]], # 2 pairs before, if valid
              "context_after": List[Tuple[int, int, float, float]],  # 2 pairs after, if valid
              "mean_speed": float,
              "max_speed": float,
              "std_speed": float,
              "num_pairs": int,
              "num_skipped": int,  # in-range pairs skipped due to NaNs
          },
          ...
        }
    """
    dt = 1.0 / max(fps, 1e-6)
    results: Dict[ContactRange, Dict[str, object]] = {}

    for start, end in ranges:
        # Ensure we stay within array bounds
        start_idx = max(start, 0)
        end_idx = min(end, frames.shape[0] - 1)
        if end_idx <= start_idx:
            results[(start, end)] = {
                "pairs": [],
                "context_before": [],
                "context_after": [],
                "mean_speed": float("nan"),
                "max_speed": float("nan"),
                "std_speed": float("nan"),
                "num_pairs": 0,
                "num_skipped": 0,
            }
            continue

        speeds: List[float] = []
        pair_records: List[Tuple[int, int, float, float]] = []
        context_before: List[Tuple[int, int, float, float]] = []
        context_after: List[Tuple[int, int, float, float]] = []
        num_skipped = 0

        for k in range(start_idx, end_idx):
            p0 = frames[k, joint_index, :]
            p1 = frames[k + 1, joint_index, :]
            if np.isnan(p0).any() or np.isnan(p1).any():
                num_skipped += 1
                continue
            disp = float(np.linalg.norm(p1 - p0))
            speed = disp / dt  # m/s
            pair_records.append((k, k + 1, disp, speed))
            speeds.append(speed)

        # Context: two pairs before and two after the range (indices in bounds)
        def add_context_pair(k: int, out_list: List[Tuple[int, int, float, float]]) -> None:
            if k < 0 or k + 1 >= frames.shape[0]:
                return
            p0 = frames[k, joint_index, :]
            p1 = frames[k + 1, joint_index, :]
            if np.isnan(p0).any() or np.isnan(p1).any():
                return
            disp = float(np.linalg.norm(p1 - p0))
            speed = disp / dt
            out_list.append((k, k + 1, disp, speed))

        # before: (start-2 -> start-1), (start-1 -> start)
        add_context_pair(start_idx - 2, context_before)
        add_context_pair(start_idx - 1, context_before)
        # after: (end -> end+1), (end+1 -> end+2)
        add_context_pair(end_idx, context_after)
        add_context_pair(end_idx + 1, context_after)

        if speeds:
            arr = np.asarray(speeds, dtype=np.float64)
            mean_speed = float(arr.mean())
            max_speed = float(arr.max())
            std_speed = float(arr.std(ddof=0))
        else:
            mean_speed = max_speed = std_speed = float("nan")

        results[(start, end)] = {
            "pairs": pair_records,
            "context_before": context_before,
            "context_after": context_after,
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "std_speed": std_speed,
            "num_pairs": len(pair_records),
            "num_skipped": num_skipped,
        }

    return results


def detect_stationary_sequences(
    frames: np.ndarray,
    joint_index: int,
    fps: float,
    base_ranges: Sequence[ContactRange],
    disp_thresh: float = 0.01,
    speed_thresh: float = 0.5,
    y_tol: float = 0.01,
    min_pairs: int = 3,
) -> Tuple[List[ContactRange], float]:
    """
    Automatically detect additional stationary sequences for a single joint.

    Criteria per frame pair k->k+1:
      - |Δx| < disp_thresh  (meters)
      - v < speed_thresh    (meters/second)
      - both Y coordinates close to a baseline contact height (|y - y_base| < y_tol)

    Existing annotated contact ranges (base_ranges) are excluded from detection.

    Returns:
        (ranges, y_base)
        where ranges is a list of inclusive frame ranges [start, end].
    """
    num_frames = frames.shape[0]
    if num_frames < 2:
        return [], float("nan")

    # Estimate baseline Y from provided contact ranges
    ys: List[float] = []
    for start, end in base_ranges:
        s = max(start, 0)
        e = min(end, num_frames - 1)
        if e < s:
            continue
        seg = frames[s : e + 1, joint_index, 1]  # Y coordinate
        seg = seg[~np.isnan(seg)]
        if seg.size:
            ys.append(float(seg.mean()))

    if ys:
        y_base = float(np.mean(ys))
    else:
        # Fallback: median Y over all valid frames
        all_y = frames[:, joint_index, 1]
        all_y = all_y[~np.isnan(all_y)]
        y_base = float(np.median(all_y)) if all_y.size else 0.0

    dt = 1.0 / max(fps, 1e-6)
    ok = np.zeros(num_frames - 1, dtype=bool)

    # Mask out indices that belong to annotated contact ranges
    exclude = np.zeros_like(ok)
    for start, end in base_ranges:
        s = max(start, 0)
        e = min(end, num_frames - 1)
        if e > s:
            exclude[s:e] = True  # pairs s->s+1 ... e-1->e

    for k in range(num_frames - 1):
        if exclude[k]:
            continue
        p0 = frames[k, joint_index, :]
        p1 = frames[k + 1, joint_index, :]
        if np.isnan(p0).any() or np.isnan(p1).any():
            continue
        disp = float(np.linalg.norm(p1 - p0))
        speed = disp / dt
        if disp >= disp_thresh or speed >= speed_thresh:
            continue
        y0 = float(p0[1])
        y1 = float(p1[1])
        if abs(y0 - y_base) > y_tol or abs(y1 - y_base) > y_tol:
            continue
        ok[k] = True

    # Convert contiguous runs of True into inclusive frame ranges [start_frame, end_frame]
    detected: List[ContactRange] = []
    k = 0
    while k < ok.size:
        if not ok[k]:
            k += 1
            continue
        run_start = k
        while k + 1 < ok.size and ok[k + 1]:
            k += 1
        run_end = k
        length = run_end - run_start + 1
        if length >= min_pairs:
            start_frame = run_start
            end_frame = run_end + 1
            detected.append((start_frame, end_frame))
        k += 1

    return detected, y_base


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-frame relative paw velocities for specified contact ranges. "
            "Intended for checking that contact phases are close to stationary."
        )
    )
    parser.add_argument(
        "optical_file",
        type=str,
        help="Path to optical_sync_align_*.txt file "
             "(e.g., /mnt/e/3ddogs/Sync_Align_v2023_11_16b/optical_sync_align_d12_t1_a.txt).",
    )
    parser.add_argument(
        "--contact-file",
        type=str,
        default="/mnt/e/3ddogs/Data/RGBD/Trimmed_Undist_FOV_Opt_2022_01/d12_t1a/Images/101/_contact.txt",
        help="Path to *_contact.txt defining (start, end) frame ranges for joints.",
    )
    parser.add_argument(
        "--detect-joint",
        type=str,
        default="l_meta_carp",
        help=(
            "Joint name to run automatic stationary-sequence detection for. "
            "Must appear in both the optical file and contact file. "
            "Set to empty string ('') to disable automatic detection."
        ),
    )
    parser.add_argument(
        "--detect-disp-thresh",
        type=float,
        default=0.01,
        help="Displacement threshold (meters) per frame pair for stationary detection (default 0.01).",
    )
    parser.add_argument(
        "--detect-speed-thresh",
        type=float,
        default=0.5,
        help="Speed threshold (meters/second) per frame pair for stationary detection (default 0.5).",
    )
    parser.add_argument(
        "--detect-y-tol",
        type=float,
        default=0.01,
        help="Allowed deviation in Y from baseline contact height (meters) for stationary detection (default 0.01).",
    )
    parser.add_argument(
        "--detect-min-pairs",
        type=int,
        default=3,
        help="Minimum number of consecutive pairs required to form a detected stationary sequence (default 3).",
    )
    parser.add_argument(
        "--debug-joint",
        type=str,
        default="",
        help=(
            "Optional: joint name to dump detailed per-frame pair motion for, "
            "independent of annotated contact ranges."
        ),
    )
    parser.add_argument(
        "--debug-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help=(
            "Optional: inclusive frame range [START, END] to inspect for --debug-joint. "
            "If omitted, the whole sequence is used."
        ),
    )
    args = parser.parse_args()

    optical_path = os.path.abspath(args.optical_file)
    contact_path = os.path.abspath(args.contact_file)

    if not os.path.exists(optical_path):
        raise SystemExit(f"Optical sync file not found: {optical_path}")
    if not os.path.exists(contact_path):
        raise SystemExit(f"Contact file not found: {contact_path}")

    print(f"Using optical file: {optical_path}")
    print(f"Using contact file: {contact_path}")

    # Load sequence and joint names.
    frames, joint_names = load_optical_sync_file(optical_path)
    fps = read_fps(optical_path)
    print(f"Sequence: {frames.shape[0]} frames, {len(joint_names)} joints, fps={fps:.3f}")

    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

    # Parse contact ranges from the provided _contact.txt
    contact_ranges = parse_contact_file(contact_path)
    if not contact_ranges:
        raise SystemExit("No valid (start, end) ranges found in contact file.")

    print("\nPer-joint relative velocities within specified contact ranges:")
    print("  (speeds are in meters/second; ranges are inclusive [start, end])\n")

    for joint_name, ranges in contact_ranges.items():
        if joint_name not in name_to_idx:
            print(f"- Joint '{joint_name}' not found in optical file; skipping.")
            continue

        j_idx = name_to_idx[joint_name]
        stats_by_range = compute_pairwise_speeds(frames, j_idx, ranges, fps=fps)

        print(f"Joint: {joint_name}")
        for (start, end), stats in stats_by_range.items():
            print(f"  Range {start}->{end}:")
            num_pairs = stats["num_pairs"]
            num_skipped = stats["num_skipped"]
            mean_speed = stats["mean_speed"]
            max_speed = stats["max_speed"]
            std_speed = stats["std_speed"]
            ctx_before = stats.get("context_before", [])
            ctx_after = stats.get("context_after", [])

            if num_pairs == 0:
                print("    No valid frame pairs (all NaNs or zero-length range).")
            else:
                print(f"    Pairs used: {num_pairs}  (skipped {num_skipped} with NaNs)")
                print(f"    Mean speed: {mean_speed:.6f} m/s")
                print(f"    Max  speed: {max_speed:.6f} m/s")
                print(f"    Std  speed: {std_speed:.6f} m/s")

                # In-range pairs
                print("    In-range frame-to-frame motion:")
                for k0, k1, disp, speed in stats["pairs"]:  # type: ignore[assignment]
                    print(f"      {k0:04d}->{k1:04d}: |Δx|={disp:.6f} m, v={speed:.6f} m/s")

            # Context pairs (before / after) to inspect phase transitions
            if ctx_before or ctx_after:
                print("    Context pairs (2 before, 2 after the range):")
                for k0, k1, disp, speed in ctx_before:
                    print(f"      BEFORE {k0:04d}->{k1:04d}: |Δx|={disp:.6f} m, v={speed:.6f} m/s")
                for k0, k1, disp, speed in ctx_after:
                    print(f"      AFTER  {k0:04d}->{k1:04d}: |Δx|={disp:.6f} m, v={speed:.6f} m/s")

        print("")  # blank line between joints

    # Optional detailed debug dump for an arbitrary joint and frame range
    if args.debug_joint:
        dbg_joint = args.debug_joint
        if dbg_joint not in name_to_idx:
            print(f"\n[DEBUG] Joint '{dbg_joint}' not found in optical file; cannot dump debug range.")
        else:
            dbg_idx = name_to_idx[dbg_joint]
            if args.debug_range is not None:
                dbg_start, dbg_end = args.debug_range
            else:
                dbg_start, dbg_end = 0, frames.shape[0] - 1
            dbg_start = max(dbg_start, 0)
            dbg_end = min(dbg_end, frames.shape[0] - 1)

            if dbg_end <= dbg_start:
                print(
                    f"\n[DEBUG] Invalid debug range [{dbg_start}, {dbg_end}] "
                    f"for joint '{dbg_joint}' (needs at least two frames)."
                )
            else:
                dt = 1.0 / max(fps, 1e-6)
                print(
                    f"\n[DEBUG] Detailed per-frame motion for joint '{dbg_joint}' "
                    f"in frames [{dbg_start}, {dbg_end}] (pairs {dbg_start}->{dbg_end}):"
                )
                for k in range(dbg_start, dbg_end):
                    p0 = frames[k, dbg_idx, :]
                    p1 = frames[k + 1, dbg_idx, :]
                    if np.isnan(p0).any() or np.isnan(p1).any():
                        print(f"  {k:04d}->{k+1:04d}: NaN present, skipping.")
                        continue
                    disp = float(np.linalg.norm(p1 - p0))
                    speed = disp / dt
                    y0 = float(p0[1])
                    y1 = float(p1[1])
                    print(
                        f"  {k:04d}->{k+1:04d}: |Δx|={disp:.6f} m, v={speed:.6f} m/s, "
                        f"y0={y0:.6f}, y1={y1:.6f}"
                    )

    # Optional automatic detection for a chosen joint: search for extra stationary phases
    target_joint = args.detect_joint
    if target_joint:
        if target_joint not in name_to_idx:
            print(
                f"Automatic detection: joint '{target_joint}' not found in optical file; "
                "skipping automatic detection."
            )
        elif target_joint not in contact_ranges:
            print(
                f"Automatic detection: joint '{target_joint}' has no base ranges in contact file; "
                "skipping automatic detection."
            )
        else:
            print(f"Automatic detection of additional stationary sequences for joint '{target_joint}':")
            base_ranges = contact_ranges[target_joint]
            j_idx = name_to_idx[target_joint]

            detected_ranges, y_base = detect_stationary_sequences(
                frames,
                j_idx,
                fps=fps,
                base_ranges=base_ranges,
                disp_thresh=args.detect_disp_thresh,
                speed_thresh=args.detect_speed_thresh,
                y_tol=args.detect_y_tol,
                min_pairs=args.detect_min_pairs,
            )

            if not detected_ranges:
                print("  No additional candidate sequences found with current thresholds.\n")
            else:
                print(f"  Baseline Y from annotated contact: {y_base:.6f}")
                print(f"  Found {len(detected_ranges)} additional candidate sequences.\n")

                extra_stats = compute_pairwise_speeds(frames, j_idx, detected_ranges, fps=fps)
                for (start, end), stats in extra_stats.items():
                    print(f"  Detected candidate range {start}->{end}:")
                    num_pairs = stats["num_pairs"]
                    num_skipped = stats["num_skipped"]
                    mean_speed = stats["mean_speed"]
                    max_speed = stats["max_speed"]
                    std_speed = stats["std_speed"]
                    ctx_before = stats.get("context_before", [])
                    ctx_after = stats.get("context_after", [])

                    if num_pairs == 0:
                        print("    No valid frame pairs (all NaNs or zero-length range).")
                    else:
                        print(f"    Pairs used: {num_pairs}  (skipped {num_skipped} with NaNs)")
                        print(f"    Mean speed: {mean_speed:.6f} m/s")
                        print(f"    Max  speed: {max_speed:.6f} m/s")
                        print(f"    Std  speed: {std_speed:.6f} m/s")
                        print("    In-range frame-to-frame motion:")
                        for k0, k1, disp, speed in stats["pairs"]:  # type: ignore[assignment]
                            print(f"      {k0:04d}->{k1:04d}: |Δx|={disp:.6f} m, v={speed:.6f} m/s")

                    if ctx_before or ctx_after:
                        print("    Context pairs (2 before, 2 after the range):")
                        for k0, k1, disp, speed in ctx_before:
                            print(f"      BEFORE {k0:04d}->{k1:04d}: |Δx|={disp:.6f} m, v={speed:.6f} m/s")
                        for k0, k1, disp, speed in ctx_after:
                            print(f"      AFTER  {k0:04d}->{k1:04d}: |Δx|={disp:.6f} m, v={speed:.6f} m/s")

                    print("")


if __name__ == "__main__":
    main()


