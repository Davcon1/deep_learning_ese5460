import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

from visualize import load_optical_sync_file

try:
    # Optional; used for shape-preserving cubic interpolation of short gaps.
    from scipy.interpolate import PchipInterpolator  # type: ignore[import]
except Exception:  # pragma: no cover
    PchipInterpolator = None


def rigid_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def write_augmented_file(
    original_path: str,
    output_path: str,
    frames: np.ndarray,
    joint_names,
):
    default_header = [
        "fps 60.000000",
        "frame_numbering_ref pass_start",
        "alignment_ref rgbd_gbl_ref",
        "frame_num " + " ".join(joint_names),
    ]
    header_lines = []
    try:
        with open(original_path, "r", encoding="utf-8") as f:
            for _ in range(4):
                line = f.readline()
                if not line:
                    break
                header_lines.append(line.rstrip("\n"))
    except FileNotFoundError:
        header_lines = []
    if len(header_lines) < 4:
        header_lines = default_header
    else:
        header_lines[3] = "frame_num " + " ".join(joint_names)
    if not any(line.startswith("frame_num") for line in header_lines):
        header_lines.append("frame_num " + " ".join(joint_names))
    with open(output_path, "w", encoding="utf-8") as fw:
        for line in header_lines:
            fw.write(line + "\n")
        for frame_idx in range(frames.shape[0]):
            parts = [str(frame_idx)]
            for joint_idx in range(len(joint_names)):
                xyz = frames[frame_idx, joint_idx, :]
                for coord in xyz:
                    if np.isnan(coord):
                        parts.append("NaN")
                    else:
                        parts.append(f"{coord:.6f}")
            fw.write(" ".join(parts) + "\n")


def reflect_point(point: np.ndarray, normal: np.ndarray, plane_point: np.ndarray) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    return point - 2.0 * n * np.dot(point - plane_point, n)


def gather_samples(frames: np.ndarray, joint_pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    src_samples = []
    tgt_samples = []
    for src_idx, tgt_idx in joint_pairs:
        src = frames[:, src_idx, :]
        tgt = frames[:, tgt_idx, :]
        mask = (~np.isnan(src).any(axis=1)) & (~np.isnan(tgt).any(axis=1))
        src_samples.append(src[mask])
        tgt_samples.append(tgt[mask])
    if not src_samples:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.vstack(src_samples), np.vstack(tgt_samples)


def interpolate_short_gaps(
    frames: np.ndarray,
    joint_names: List[str],
    joint_indices: List[int],
    max_gap: int = 5,
    root_joints: Tuple[str, str] = ("withers", "sacrum"),
) -> int:
    """
    Fill short NaN gaps (length <= max_gap) along time for the given joints.

    Strategy:
      1) Work in a trunk-stable frame when possible:
         - Compute a root as (withers + sacrum) / 2 per frame.
         - Subtract this root before interpolating coordinates.
      2) For each joint and coordinate axis independently:
         - For NaN runs of length 1..max_gap that have valid samples on both sides,
           fill using cubic, shape-preserving interpolation (PCHIP) if available,
           otherwise fall back to linear interpolation.
      3) If the root is not available for the whole gap (including endpoints),
         fall back to global coordinates for that gap or skip it.

    Returns:
        The number of gaps successfully filled (across all joints/axes).
    """
    num_frames, num_joints, _ = frames.shape
    if num_frames == 0 or not joint_indices:
        return 0

    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    root_idx: List[int] = []
    for r in root_joints:
        if r in name_to_idx:
            root_idx.append(name_to_idx[r])
    root_coords = None
    root_valid = None
    if len(root_idx) == 2:
        root_coords = frames[:, root_idx, :].mean(axis=1)  # [F, 3]
        root_valid = ~np.isnan(root_coords).any(axis=1)  # [F]

    filled_gaps = 0
    F = num_frames

    for j_idx in joint_indices:
        if j_idx < 0 or j_idx >= num_joints:
            continue
        for axis in range(3):
            coord = frames[:, j_idx, axis].copy()
            if np.all(np.isnan(coord)):
                continue

            # Build relative coordinates in trunk frame if possible.
            if root_coords is not None and root_valid is not None and root_valid.any():
                rel = np.full_like(coord, np.nan)
                for f in range(F):
                    if not np.isnan(coord[f]) and root_valid[f]:
                        rel[f] = coord[f] - root_coords[f, axis]
                base = rel
                use_root = True
            else:
                base = coord.copy()
                use_root = False

            isnan = np.isnan(base)
            i = 0
            while i < F:
                if not isnan[i]:
                    i += 1
                    continue
                start = i
                while i < F and isnan[i]:
                    i += 1
                end = i
                gap_len = end - start
                if gap_len == 0 or gap_len > max_gap:
                    continue
                if start == 0 or end >= F:
                    continue
                left = start - 1
                right = end

                # For this gap, decide whether we can stay in trunk frame.
                local_use_root = use_root
                if local_use_root:
                    assert root_coords is not None and root_valid is not None
                    if not (
                        root_valid[left]
                        and root_valid[right]
                        and bool(root_valid[start:end].all())
                    ):
                        local_use_root = False

                if local_use_root:
                    y_base = base
                else:
                    # Fall back to global coordinates for this gap.
                    y_base = coord

                if np.isnan(y_base[left]) or np.isnan(y_base[right]):
                    continue

                x_known = np.array([left, right], dtype=float)
                y_known = y_base[[left, right]]
                x_gap = np.arange(start, end, dtype=float)

                if PchipInterpolator is not None and len(x_known) >= 2:
                    try:
                        interp = PchipInterpolator(x_known, y_known)
                        y_gap = interp(x_gap)
                    except Exception:
                        # Fallback to linear if PCHIP fails for any reason.
                        y_gap = np.interp(x_gap, x_known, y_known)
                else:
                    y_gap = np.interp(x_gap, x_known, y_known)

                if local_use_root:
                    assert root_coords is not None
                    coord[start:end] = y_gap + root_coords[start:end, axis]
                else:
                    coord[start:end] = y_gap

                filled_gaps += 1

            frames[:, j_idx, axis] = coord

    return filled_gaps


def main():
    parser = argparse.ArgumentParser(description="Impute joints via rigid transform.")
    parser.add_argument("file", type=str, help="Path to optical_sync_align_*.txt file.")
    parser.add_argument("--output", type=str, default="", help="Output path (default <input>_augmented.txt).")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum paired samples to fit transform.")
    parser.add_argument("--report-only", action="store_true", help="Only print stats; do not write file.")
    parser.add_argument(
        "--fit-source",
        nargs="+",
        required=True,
        help="Joint names (source) used to fit the transform (e.g., r_acro).",
    )
    parser.add_argument(
        "--fit-target",
        nargs="+",
        required=True,
        help="Joint names (target) paired with --fit-source (e.g., r_shdr).",
    )
    parser.add_argument(
        "--predict-source",
        nargs="+",
        default=None,
        help="Joint names supplying inputs for imputation (default: --fit-source).",
    )
    parser.add_argument(
        "--predict-target",
        nargs="+",
        default=None,
        help="Joint names to impute (default: --fit-target).",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror transform across a plane before applying to prediction pairs.",
    )
    parser.add_argument(
        "--mirror-mode",
        choices=["constant", "auto"],
        default="constant",
        help="Constant plane uses fixed normal/point; auto derives plane per frame using axis joints.",
    )
    parser.add_argument(
        "--mirror-normal",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        help="Normal vector for constant mirror plane (default +Z).",
    )
    parser.add_argument(
        "--mirror-point",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        help="Point on constant mirror plane (default origin).",
    )
    parser.add_argument(
        "--mirror-axis-joints",
        nargs=2,
        default=("withers", "sacrum"),
        help="When --mirror-mode auto, use these two joints to define the spine axis.",
    )
    parser.add_argument(
        "--mirror-up",
        nargs=3,
        type=float,
        default=(0.0, 1.0, 0.0),
        help="Up direction used with auto mirror planes (default global Y).",
    )
    parser.add_argument(
        "--interp-max-gap",
        type=int,
        default=5,
        help="Maximum NaN gap length (in frames) to fill by time interpolation before rigid imputation.",
    )
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        raise SystemExit(f"File not found: {file_path}")

    default_out = os.path.join(
        os.getcwd(),
        os.path.basename(file_path).rsplit(".", 1)[0] + "_augmented.txt",
    )
    output_path = os.path.abspath(args.output) if args.output else default_out

    frames, joint_names = load_optical_sync_file(file_path)
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

    def joints_to_indices(joint_list):
        idxs = []
        for name in joint_list:
            if name not in name_to_idx:
                raise SystemExit(f"Joint not found in file: {name}")
            idxs.append(name_to_idx[name])
        return idxs

    fit_source_idx = joints_to_indices(args.fit_source)
    fit_target_idx = joints_to_indices(args.fit_target)
    if len(fit_source_idx) != len(fit_target_idx):
        raise SystemExit("--fit-source and --fit-target must have same length.")

    pred_source_names = args.predict_source or args.fit_source
    pred_target_names = args.predict_target or args.fit_target
    pred_source_idx = joints_to_indices(pred_source_names)
    pred_target_idx = joints_to_indices(pred_target_names)
    if len(pred_source_idx) != len(pred_target_idx):
        raise SystemExit("--predict-source and --predict-target must have same length.")

    fit_pairs = list(zip(fit_source_idx, fit_target_idx))
    src_samples, tgt_samples = gather_samples(frames, fit_pairs)
    total_samples = src_samples.shape[0]
    if total_samples < args.min_samples:
        raise SystemExit(f"Only {total_samples} paired samples available; need >= {args.min_samples}.")

    R, t = rigid_transform(src_samples, tgt_samples)
    fit_pred = (R @ src_samples.T).T + t
    residuals = np.linalg.norm(fit_pred - tgt_samples, axis=1)
    print(
        f"Fitted on {total_samples} samples. Residual mean {residuals.mean():.6f} m, "
        f"max {residuals.max():.6f} m"
    )

    if args.mirror:
        if args.mirror_mode == "constant":
            mirror_normal = np.asarray(args.mirror_normal, dtype=np.float64)
            if np.linalg.norm(mirror_normal) < 1e-8:
                raise SystemExit("Mirror normal cannot be zero vector.")
            mirror_point = np.asarray(args.mirror_point, dtype=np.float64)
            axis_indices = None
            up_vec = None
        else:
            joint_a, joint_b = args.mirror_axis_joints
            if joint_a not in name_to_idx or joint_b not in name_to_idx:
                raise SystemExit(f"Mirror axis joints not found: {joint_a}, {joint_b}")
            axis_indices = (name_to_idx[joint_a], name_to_idx[joint_b])
            up_vec = np.asarray(args.mirror_up, dtype=np.float64)
            if np.linalg.norm(up_vec) < 1e-8:
                raise SystemExit("Mirror up vector cannot be zero.")
            mirror_normal = np.asarray(args.mirror_normal, dtype=np.float64)
            mirror_point = np.asarray(args.mirror_point, dtype=np.float64)
    else:
        mirror_normal = mirror_point = None
        axis_indices = None
        up_vec = None

    # First, optionally interpolate short temporal gaps on the prediction targets.
    interp_gaps = interpolate_short_gaps(
        frames,
        joint_names,
        pred_target_idx,
        max_gap=max(1, args.interp_max_gap),
    )
    if interp_gaps > 0:
        print(
            f"Interpolated {interp_gaps} short NaN gaps "
            f"(<= {args.interp_max_gap} frames) on prediction targets before rigid imputation."
        )

    imputed_total = 0
    for src_idx, tgt_idx, src_name, tgt_name in zip(
        pred_source_idx, pred_target_idx, pred_source_names, pred_target_names
    ):
        src_pts = frames[:, src_idx, :]
        tgt_pts = frames[:, tgt_idx, :]
        original_tgt = tgt_pts.copy()
        src_available = ~np.isnan(src_pts).any(axis=1)
        tgt_missing = np.isnan(tgt_pts).any(axis=1)
        indices = np.where(src_available & tgt_missing)[0]
        if indices.size == 0:
            continue

        def plane_for_frame(frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
            if not args.mirror or args.mirror_mode == "constant" or axis_indices is None:
                return mirror_point, mirror_normal
            idx_a, idx_b = axis_indices
            joint_a = frames[frame_idx, idx_a, :]
            joint_b = frames[frame_idx, idx_b, :]
            if np.isnan(joint_a).any() or np.isnan(joint_b).any():
                return mirror_point, mirror_normal
            axis_vec = joint_b - joint_a
            if np.linalg.norm(axis_vec) < 1e-6:
                return mirror_point, mirror_normal
            normal = np.cross(axis_vec, up_vec)
            if np.linalg.norm(normal) < 1e-6:
                return mirror_point, mirror_normal
            normal = normal / np.linalg.norm(normal)
            plane_pt = 0.5 * (joint_a + joint_b)
            return plane_pt, normal

        new_preds: Dict[int, np.ndarray] = {}
        imputed_count = 0
        for frame_idx in indices:
            src = src_pts[frame_idx]
            if np.isnan(src).any():
                continue
            if args.mirror:
                plane_pt, plane_norm = plane_for_frame(frame_idx)
                if plane_norm is None or np.linalg.norm(plane_norm) < 1e-6:
                    plane_norm = np.array([0.0, 0.0, 1.0])
                    plane_pt = np.zeros(3)
                src_ref = reflect_point(src, plane_norm, plane_pt)
                pred = (R @ src_ref) + t
                pred = reflect_point(pred, plane_norm, plane_pt)
            else:
                pred = (R @ src) + t
            new_preds[frame_idx] = pred
            imputed_count += 1

        if imputed_count:
            imputed_total += imputed_count
            print(f"Imputed {imputed_count} frames for joint '{tgt_name}' using source '{src_name}'.")

        for frame_idx, pred in new_preds.items():
            frames[frame_idx, tgt_idx, :] = pred

    if imputed_total == 0:
        print("No frames required imputation with available sources.")

    if args.report_only:
        print("[INFO] Report-only mode; no file written.")
        return

    write_augmented_file(file_path, output_path, frames, joint_names)
    print(f"Wrote augmented file to {output_path}")


if __name__ == "__main__":
    main()


