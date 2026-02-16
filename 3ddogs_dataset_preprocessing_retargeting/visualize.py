import argparse
import glob
import json
import os
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


OpticalSeq = Tuple[np.ndarray, List[str]]  # (frames [F, J, 3], joint_names)


GROUND_PLANE_Y = {
    # RANSAC-estimated ground Y (meters) from external analysis:
    #   l_meta_carp: y≈-0.015695 m
    #   r_meta_carp: y≈-0.016545 m
    #   l_meta_tars: y≈-0.025354 m
    #   r_meta_tars: y≈-0.025432 m
    "l_meta_carp": -0.015695,
    "r_meta_carp": -0.016545,
    "l_meta_tars": -0.025354,
    "r_meta_tars": -0.025432,
}


def find_optical_sync_files(root_dir: str) -> List[str]:
    """
    Return absolute paths to all optical sync alignment files.
    """
    pattern = os.path.join(root_dir, "Sync_Align_v2023_11_16b", "optical_sync_align_*.txt")
    files = sorted(glob.glob(pattern))
    return [os.path.abspath(p) for p in files]


def _parse_joint_header(lines: List[str]) -> List[str]:
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        # Match the exact header line that begins with the token 'frame_num'
        if tokens[0] == "frame_num":
            return tokens[1:]
    raise RuntimeError("Could not find 'frame_num' header with joint names.")


def load_optical_sync_file(path: str) -> OpticalSeq:
    """
    Parse an optical sync alignment .txt file.
    Returns:
        frames: np.ndarray with shape [num_frames, num_joints, 3]
        joint_names: list[str] with length num_joints (order matches frames' second axis)
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

    joint_names = _parse_joint_header(lines)
    num_joints = len(joint_names)

    data_lines: List[str] = []
    for line in lines:
        if not line or line.startswith(("fps", "frame_numbering_ref", "alignment_ref", "frame_num")):
            continue
        data_lines.append(line)

    if not data_lines:
        raise RuntimeError(f"No frame data found in {path}")

    num_cols_expected = 1 + 3 * num_joints  # frame_idx + joints
    frames_list: List[np.ndarray] = []
    for raw in data_lines:
        parts = raw.strip().split()
        if len(parts) < num_cols_expected:
            # Pad missing columns as NaN if the line is truncated
            parts = parts + ["NaN"] * (num_cols_expected - len(parts))
        # Skip first token (frame index)
        coord_tokens = parts[1 : 1 + 3 * num_joints]
        coords = np.array([np.nan if t.lower() == "nan" else float(t) for t in coord_tokens], dtype=np.float64)
        frames_list.append(coords.reshape(num_joints, 3))

    frames = np.stack(frames_list, axis=0)  # [F, J, 3]
    return frames, joint_names


def compute_nan_stats(frames: np.ndarray) -> Dict[str, float]:
    """
    Compute NaN statistics for a sequence.
    Returns dictionary with:
        - num_frames
        - num_joints
        - num_nan_values
        - num_nan_joints  (count of (frame, joint) triplets with any NaN)
        - valid_joint_ratio
        - fully_valid_frames
    """
    num_frames, num_joints, _ = frames.shape
    is_nan_any = np.isnan(frames).any(axis=-1)  # [F, J]
    num_nan_values = int(np.isnan(frames).sum())
    num_nan_joints = int(is_nan_any.sum())
    fully_valid_frames = int((~is_nan_any).all(axis=1).sum())
    valid_joint_ratio = float((~is_nan_any).sum() / is_nan_any.size)
    return {
        "num_frames": int(num_frames),
        "num_joints": int(num_joints),
        "num_nan_values": num_nan_values,
        "num_nan_joints": num_nan_joints,
        "valid_joint_ratio": valid_joint_ratio,
        "fully_valid_frames": fully_valid_frames,
    }


def pick_top_sequences(file_paths: Sequence[str], top_k: int = 10) -> List[Tuple[str, Dict[str, float]]]:
    scored: List[Tuple[str, Dict[str, float]]] = []
    for p in file_paths:
        try:
            frames, _ = load_optical_sync_file(p)
            stats = compute_nan_stats(frames)
            scored.append((p, stats))
        except Exception as exc:  # keep scanning even if one file is malformed
            print(f"[WARN] Failed to parse {p}: {exc}")
            continue
    # Sort by fewest NaN joints; break ties by more fully-valid frames, then more total frames
    scored.sort(key=lambda item: (item[1]["num_nan_joints"], -item[1]["fully_valid_frames"], -item[1]["num_frames"]))
    return scored[:top_k]


def dog_skeleton_edges(joints: Iterable[str]) -> List[Tuple[int, int]]:
    """
    Build an edge list (as index pairs) based on joint names present in the file.
    """
    order = {name: idx for idx, name in enumerate(joints)}
    def has(a: str, b: str) -> bool:
        return a in order and b in order
    edges_named: List[Tuple[str, str]] = [
        ("poll", "withers"),
        ("withers", "sacrum"),
        # Left forelimb
        ("withers", "l_prox_scap"),
        ("l_prox_scap", "l_caud_scap"),
        ("l_caud_scap", "l_acro"),
        ("l_acro", "l_shdr"),
        ("l_shdr", "l_elb"),
        ("l_elb", "l_dist_rad"),
        ("l_dist_rad", "l_meta_carp"),
        # Right forelimb
        ("withers", "r_prox_scap"),
        ("r_prox_scap", "r_caud_scap"),
        ("r_caud_scap", "r_acro"),
        ("r_acro", "r_shdr"),
        ("r_shdr", "r_elb"),
        ("r_elb", "r_dist_rad"),
        ("r_dist_rad", "r_meta_carp"),
        # Left hindlimb
        ("sacrum", "l_iliac"),
        ("l_iliac", "l_grt_trc"),
        ("l_grt_trc", "l_isch_tub"),
        ("l_isch_tub", "l_stifle"),
        ("l_stifle", "l_hock"),
        ("l_hock", "l_meta_tars"),
        # Right hindlimb
        ("sacrum", "r_iliac"),
        ("r_iliac", "r_grt_trc"),
        ("r_grt_trc", "r_isch_tub"),
        ("r_isch_tub", "r_stifle"),
        ("r_stifle", "r_hock"),
        ("r_hock", "r_meta_tars"),
    ]
    edges: List[Tuple[int, int]] = []
    for a, b in edges_named:
        if has(a, b):
            edges.append((order[a], order[b]))
    return edges


def precompute_normalization(frames: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute center and scale for normalizing coordinates.
    """
    valid = ~np.isnan(frames).any(axis=-1)
    if valid.any():
        xyz = frames[valid]
        center = xyz.mean(axis=0)
        extent = np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
        extent = float(max(extent, 1e-3))
    else:
        center = np.zeros(3, dtype=np.float64)
        extent = 1.0
    return center, extent


def _visualize_with_viser(
    sequences: List[Dict[str, object]],
    fps: float = 60.0,
    debug: bool = False,
    ground_plane_spec: Sequence[Tuple[str, float]] | None = None,
) -> None:
    """
    Minimal Viser visualization with interactive sequence selection.
    Each sequence entry must provide frames, joints, edges, center, extent, and label.
    """
    try:
        import viser
    except Exception:
        print("[INFO] Viser not available; falling back to Matplotlib.")
        first = sequences[0]
        _visualize_with_matplotlib(
            first["frames"], first["joints"], first["edges"], fps=fps  # type: ignore[arg-type]
        )
        return

    server = viser.ViserServer()
    server.scene.world_axes.visible = True  # type: ignore[attr-defined]
    try:
        server.scene.world_axes.length = 0.15  # type: ignore[attr-defined]
        server.scene.world_axes.thickness = 0.01  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        server.camera.look_at(  # type: ignore[attr-defined]
            position=(0.0, 0.0, 1.5),  # viewer stands on +Z looking toward origin
            look_at=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
        )
    except Exception:
        pass

    point_rgb = (255, 140, 0)  # orange
    line_rgb = (180, 100, 255)  # purple

    labels = [seq["label"] for seq in sequences]
    option_values = [str(idx) for idx in range(len(sequences))]
    dropdown = None
    try:
        dropdown = server.gui.add_dropdown(  # type: ignore[attr-defined]
            label="Sequence",
            options={val: labels[int(val)] for val in option_values},
            initial_value=option_values[0],
        )
    except Exception:
        print("[WARN] Viser GUI dropdown unavailable; visualizing first sequence only.")
        dropdown = None

    speed_slider = None
    try:
        speed_slider = server.gui.add_slider(  # type: ignore[attr-defined]
            label="Slowdown (0-1)",
            min=0.0,
            max=1.0,
            step=0.05,
            initial_value=1.0,
        )
    except Exception:
        speed_slider = None

    axes_checkbox = None
    try:
        axes_checkbox = server.gui.add_checkbox(  # type: ignore[attr-defined]
            label="Show axes",
            initial_value=True,
        )
    except Exception:
        axes_checkbox = None

    # Optional: slider to jump to a particular frame index (0-based, applies to all sequences).
    frame_jump_slider = None
    max_frames = 0
    try:
        for seq in sequences:
            frames = seq["frames"]  # type: ignore[assignment]
            if isinstance(frames, np.ndarray):
                max_frames = max(max_frames, frames.shape[0])
        if max_frames > 0:
            frame_jump_slider = server.gui.add_slider(  # type: ignore[attr-defined]
                label="Jump to frame",
                min=0.0,
                max=float(max_frames - 1),
                step=1.0,
                initial_value=0.0,
            )
    except Exception:
        frame_jump_slider = None
        max_frames = 0

    # Optional: draw XZ planes at fixed world-space Y coordinates (no normalization).
    plane_checkboxes: List[object] = []
    plane_objects: List[object] = []
    plane_points: List[np.ndarray] = []
    if ground_plane_spec:
        # Compute global X/Z bounds over all sequences (using raw coordinates).
        all_xyz: List[np.ndarray] = []
        for seq in sequences:
            frames = seq["frames"]  # type: ignore[assignment]
            if not isinstance(frames, np.ndarray):
                continue
            coords = frames.reshape(-1, 3)
            mask = ~np.isnan(coords).any(axis=-1)
            if mask.any():
                all_xyz.append(coords[mask])
        if all_xyz:
            xyz_cat = np.concatenate(all_xyz, axis=0)
            xmin, ymin, zmin = xyz_cat.min(axis=0)
            xmax, ymax, zmax = xyz_cat.max(axis=0)
            dx = float(xmax - xmin)
            dz = float(zmax - zmin)
            span = max(dx, dz)
            if span <= 0.0:
                span = 1.0
            margin = 0.1 * span
            xmin_f = float(xmin) - margin
            xmax_f = float(xmax) + margin
            zmin_f = float(zmin) - margin
            zmax_f = float(zmax) + margin

            plane_colors = [
                (255, 0, 0),      # red
                (0, 255, 0),      # green
                (0, 128, 255),    # blue-cyan
                (255, 215, 0),    # gold
            ]

            for idx, ((joint_name, y_val), rgb) in enumerate(zip(ground_plane_spec, plane_colors)):
                try:
                    cb = server.gui.add_checkbox(  # type: ignore[attr-defined]
                        label=f"{joint_name} (y={y_val:.3f} m)",
                        initial_value=False,
                    )
                except Exception:
                    cb = None
                plane_checkboxes.append(cb)

                # Four corners of a rectangle in XZ at fixed Y (for a filled quad).
                y_f = float(y_val)
                quad_pts = np.array(
                    [
                        [xmin_f, y_f, zmin_f],
                        [xmax_f, y_f, zmin_f],
                        [xmax_f, y_f, zmax_f],
                        [xmin_f, y_f, zmax_f],
                    ],
                    dtype=np.float32,
                )
                plane_points.append(quad_pts)
                faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
                mesh = server.scene.add_mesh_simple(  # type: ignore[attr-defined]
                    name=f"ground_plane_{idx}",
                    vertices=quad_pts,
                    faces=faces,
                    color=rgb,  # RGB tuple
                    wireframe=False,
                    opacity=0.35,
                    side="double",
                )
                plane_objects.append(mesh)

    renderer_cache: Dict[int, Dict[str, object]] = {}

    def build_renderers(idx: int) -> Dict[str, object]:
        seq = sequences[idx]
        joints: List[str] = seq["joints"]  # type: ignore[assignment]
        edges = seq["edges"]  # type: ignore[assignment]
        joint_objs: List[object] = []
        edge_objs: List[object] = []
        for j, joint_name in enumerate(joints):
            safe_joint = joint_name.replace(" ", "_")
            name = f"seq{idx}_joint_{j}_{safe_joint}"
            try:
                sphere = server.scene.add_icosphere(  # type: ignore[attr-defined]
                    name=name,
                    position=(0.0, 0.0, 0.0),
                    radius=0.006,
                    color=point_rgb,  # type: ignore[arg-type]
                )
            except Exception:
                sphere = server.scene.add_icosphere(  # type: ignore[attr-defined]
                    name=name,
                    position=(0.0, 0.0, 0.0),
                    scale=0.012,  # type: ignore[arg-type]
                    color=point_rgb,  # type: ignore[arg-type]
                )
            joint_objs.append(sphere)
        for ei, (_a, _b) in enumerate(edges):
            seg = server.scene.add_line_segments(  # type: ignore[attr-defined]
                name=f"seq{idx}_edge_{ei}",
                points=np.zeros((1, 2, 3), dtype=np.float32),
                colors=np.tile(np.array(line_rgb, dtype=np.uint8)[None, None, :], (1, 2, 1)),
                line_width=2.0,
            )
            edge_objs.append(seg)
        return {"joint_objs": joint_objs, "edge_objs": edge_objs}

    def hide_renderers(renderers: Dict[str, object]) -> None:
        for obj in renderers["joint_objs"]:
            obj.visible = False  # type: ignore[attr-defined]
        for obj in renderers["edge_objs"]:
            obj.visible = False  # type: ignore[attr-defined]

    def ensure_renderers(idx: int) -> Dict[str, object]:
        if idx not in renderer_cache:
            renderer_cache[idx] = build_renderers(idx)
            if debug:
                seq = sequences[idx]
                print(
                    f"[DEBUG] built renderers for seq {idx} "
                    f"(joints={len(seq['joints'])}, edges={len(seq['edges'])})"
                )
        return renderer_cache[idx]

    print("[INFO] Streaming frames. Close the viewer tab to stop.")
    t_sleep = 1.0 / max(fps, 1.0)
    active_idx = None
    active_renderers: Dict[str, object] | None = None
    frame_idx = 0
    last_jump_value: float | None = None
    try:
        while True:
            selected_val = dropdown.value if dropdown else option_values[0]
            selected_idx = int(selected_val)
            if active_idx != selected_idx:
                if active_renderers is not None:
                    hide_renderers(active_renderers)
                active_renderers = ensure_renderers(selected_idx)
                active_idx = selected_idx
                frame_idx = 0
                if debug:
                    print(f"[DEBUG] switched to sequence {selected_idx}: {labels[selected_idx]}")

            seq = sequences[active_idx]
            frames = seq["frames"]  # type: ignore[assignment]
            joints = seq["joints"]  # type: ignore[assignment]
            edges = seq["edges"]  # type: ignore[assignment]

            coords = frames[frame_idx]
            joint_objs = active_renderers["joint_objs"]
            for j, obj in enumerate(joint_objs[: len(joints)]):
                p = coords[j]
                if np.any(np.isnan(p)):
                    obj.visible = False  # type: ignore[attr-defined]
                    continue
                pv = p.astype(np.float32)
                obj.visible = True  # type: ignore[attr-defined]
                obj.position = tuple(pv)  # type: ignore[attr-defined]

            edge_objs = active_renderers["edge_objs"]
            for (a, b), obj in zip(edges, edge_objs):
                pa, pb = coords[a], coords[b]
                if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
                    obj.visible = False  # type: ignore[attr-defined]
                else:
                    pts2 = np.stack([pa, pb], axis=0).astype(np.float32)
                    obj.visible = True  # type: ignore[attr-defined]
                    obj.points = pts2[None, ...]  # type: ignore[attr-defined]

            # Update ground planes visibility (if any).
            if plane_objects and plane_checkboxes and plane_points:
                for cb, mesh, quad_pts in zip(plane_checkboxes, plane_objects, plane_points):
                    if cb is not None and getattr(cb, "value", False):
                        mesh.visible = True  # type: ignore[attr-defined]
                    else:
                        mesh.visible = False  # type: ignore[attr-defined]

            slowdown = speed_slider.value if speed_slider is not None else 1.0
            slowdown = max(min(slowdown, 1.0), 0.0)

            # Optional frame jump via slider (does not affect playback rate).
            if frame_jump_slider is not None and max_frames > 0:
                try:
                    jump_val = float(frame_jump_slider.value)  # type: ignore[attr-defined]
                except Exception:
                    jump_val = None
                if jump_val is not None:
                    # Only act when the user changes the slider.
                    if last_jump_value is None or abs(jump_val - last_jump_value) >= 0.5:
                        target_idx = int(round(jump_val))
                        target_idx = max(0, min(target_idx, frames.shape[0] - 1))
                        frame_idx = target_idx
                        last_jump_value = jump_val

            if slowdown <= 1e-3:
                time.sleep(t_sleep)
                continue

            if axes_checkbox is not None:
                server.scene.world_axes.visible = bool(axes_checkbox.value)  # type: ignore[attr-defined]

            frame_idx = (frame_idx + 1) % frames.shape[0]
            effective_sleep = t_sleep / slowdown
            time.sleep(effective_sleep)
    except KeyboardInterrupt:
        pass


def _visualize_with_matplotlib(frames: np.ndarray, joints: List[str], edges: List[Tuple[int, int]], fps: float = 60.0) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    valid = ~np.isnan(frames).any(axis=-1)
    if valid.any():
        xyz = frames[valid]
        xmin, ymin, zmin = xyz.min(axis=0)
        xmax, ymax, zmax = xyz.max(axis=0)
    else:
        xmin = ymin = zmin = -1.0
        xmax = ymax = zmax = 1.0

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(float(xmin), float(xmax))
    ax.set_ylim(float(ymin), float(ymax))
    ax.set_zlim(float(zmin), float(zmax))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    scat = ax.scatter([], [], [], c="k", s=10)
    lines = [ax.plot([], [], [], c="tab:purple", lw=2)[0] for _ in edges]
    ax.set_title("3D Dog Motion")

    def init():
        scat._offsets3d = ([], [], [])
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return [scat, *lines]

    def update(frame_idx: int):
        pts = frames[frame_idx]
        mask = ~np.isnan(pts).any(axis=-1)
        x, y, z = pts[mask].T if mask.any() else (np.array([]), np.array([]), np.array([]))
        scat._offsets3d = (x, y, z)
        # Edges
        for ln, (a, b) in zip(lines, edges):
            pa, pb = pts[a], pts[b]
            if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
                ln.set_data([], [])
                ln.set_3d_properties([])
            else:
                ln.set_data([pa[0], pb[0]], [pa[1], pb[1]])
                ln.set_3d_properties([pa[2], pb[2]])
        return [scat, *lines]

    interval_ms = int(1000.0 / max(fps, 1.0))
    FuncAnimation(fig, update, init_func=init, frames=frames.shape[0], interval=interval_ms, blit=True, repeat=True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Scan optical sync files, select top-10 with fewest NaNs, and visualize in 3D.")
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/e/3ddogs",
        help="Path to dataset root (containing Sync_Align_v2023_11_16b).",
    )
    parser.add_argument("--top_k", type=int, default=10, help="How many sequences to select.")
    parser.add_argument("--visualize", action="store_true", help="Launch a viewer for the best sequence.")
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="Path to a single optical sync alignment file to analyze/visualize instead of scanning a directory.",
    )
    parser.add_argument("--fps", type=float, default=60.0, help="Playback FPS.")
    parser.add_argument("--dump_json", type=str, default="", help="Optional path to write selection/stats as JSON.")
    args = parser.parse_args()

    # If a specific file is given, operate on that file only.
    if args.file:
        path = os.path.abspath(args.file)
        try:
            frames, joints = load_optical_sync_file(path)
        except Exception as exc:
            print(f"[ERROR] Failed to parse {path}: {exc}")
            return

        stats = compute_nan_stats(frames)
        print("\nSingle-file NaN stats:")
        print(
            f"{os.path.relpath(path, args.root)}  "
            f"nan_joints={stats['num_nan_joints']}, "
            f"valid_joint_ratio={stats['valid_joint_ratio']:.4f}, "
            f"frames={stats['num_frames']}, "
            f"fully_valid_frames={stats['fully_valid_frames']}"
        )

        if args.visualize:
            edges = dog_skeleton_edges(joints)
            center, extent = precompute_normalization(frames)
            viz_entries: List[Dict[str, object]] = [
                {
                    "label": os.path.relpath(path, args.root),
                    "path": path,
                    "frames": frames,
                    "joints": joints,
                    "edges": edges,
                    "center": center,
                    "extent": extent,
                }
            ]
            _visualize_with_viser(
                viz_entries,
                fps=args.fps,
                debug=True,
                ground_plane_spec=list(GROUND_PLANE_Y.items()),
            )
        return

    optical_files = find_optical_sync_files(args.root)
    if not optical_files:
        print("[ERROR] No optical sync alignment files found.")
        return

    top = pick_top_sequences(optical_files, top_k=args.top_k)

    print("\nTop sequences with fewest NaNs:")
    for rank, (path, stats) in enumerate(top, start=1):
        print(f"{rank:2d}. {os.path.relpath(path, args.root)}  "
              f"nan_joints={stats['num_nan_joints']}, valid_joint_ratio={stats['valid_joint_ratio']:.4f}, "
              f"frames={stats['num_frames']}, fully_valid_frames={stats['fully_valid_frames']}")

    if args.dump_json:
        payload = [{"path": p, "stats": s} for p, s in top]
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Wrote selection to {args.dump_json}")

    if args.visualize and top:
        viz_entries: List[Dict[str, object]] = []
        for idx, (path, _stats) in enumerate(top):
            frames, joints = load_optical_sync_file(path)
            edges = dog_skeleton_edges(joints)
            center, extent = precompute_normalization(frames)
            viz_entries.append(
                {
                    "label": f"{idx}: {os.path.relpath(path, args.root)}",
                    "path": path,
                    "frames": frames,
                    "joints": joints,
                    "edges": edges,
                    "center": center,
                    "extent": extent,
                }
            )
        _visualize_with_viser(
            viz_entries,
            fps=args.fps,
            debug=True,
            ground_plane_spec=list(GROUND_PLANE_Y.items()),
        )


if __name__ == "__main__":
    main()


