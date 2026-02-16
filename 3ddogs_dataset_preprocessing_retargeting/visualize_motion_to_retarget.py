"""
Visualize dog motion and early retargeting stages (before IK) using Viser.

This script shows:
- the source dog keypoints (restricted to KEY_JOINTS), and
- optionally, the same keypoints after ground-offset + Unit Vector Method
  limb rescaling to Go2 segment lengths ("uvm" mode).

It also colors paw joints differently when they are in contact.

Example usage (from project root):

  # Raw dog motion only
  python visualize_motion_to_retarget.py \
      --sequence optical_sync_align_d29_t1_a_ready

  # Go2-sized limbs after ground-offset + UVM (no IK)
  python visualize_motion_to_retarget.py \
      --sequence optical_sync_align_d29_t1_a_ready \
      --mode uvm \
      --urdf UnitreeGo2/urdf/go2_description.urdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from nan_report import KEY_JOINTS
from retargeting import (
    SourceMotion,
    load_optical_sync_motion,
    load_ground_estimates,
    compute_paw_ground_deltas,
    apply_paw_ground_deltas,
    compute_go2_segment_lengths,
    rescale_limbs_to_go2_lengths,
    Go2Kinematics,
    ContactSchedule,
    DEFAULT_GROUND_Y,
)


PAW_JOINTS: Tuple[str, ...] = ("l_meta_carp", "r_meta_carp", "l_meta_tars", "r_meta_tars")


def build_leg_edges(joint_names: Sequence[str]) -> List[Tuple[int, int]]:
    """Edges for 4-limb chains restricted to KEY_JOINTS."""
    order = {name: idx for idx, name in enumerate(joint_names)}

    def has(a: str, b: str) -> bool:
        return a in order and b in order

    edges_named = [
        # Left forelimb
        ("l_prox_scap", "l_shdr"),
        ("l_shdr", "l_elb"),
        ("l_elb", "l_meta_carp"),
        # Left hindlimb
        ("l_iliac", "l_grt_trc"),
        ("l_grt_trc", "l_hock"),
        ("l_hock", "l_meta_tars"),
        # Right forelimb
        ("r_prox_scap", "r_shdr"),
        ("r_shdr", "r_elb"),
        ("r_elb", "r_meta_carp"),
        # Right hindlimb
        ("r_iliac", "r_grt_trc"),
        ("r_grt_trc", "r_hock"),
        ("r_hock", "r_meta_tars"),
    ]

    edges: List[Tuple[int, int]] = []
    for a, b in edges_named:
        if has(a, b):
            edges.append((order[a], order[b]))
    return edges


def build_contact_matrix(
    schedule: ContactSchedule, num_frames: int, paw_names: Sequence[str]
) -> np.ndarray:
    """Convert a ContactSchedule into [T, num_paws] bool matrix."""
    c = np.zeros((num_frames, len(paw_names)), dtype=bool)
    for j, paw in enumerate(paw_names):
        for iv in schedule.intervals_for(paw):
            start = max(0, iv.start)
            end = min(num_frames - 1, iv.end)
            if end >= start:
                c[start : end + 1, j] = True
    return c


def prepare_motion(
    sequence: str,
    mode: str,
    urdf_path: Path | None,
) -> Tuple[SourceMotion, np.ndarray]:
    """
    Load and optionally transform the motion for visualization.

    Returns:
        motion: SourceMotion containing KEY_JOINTS positions.
        contact: [T, 4] bool matrix for PAW_JOINTS.
    """
    base = Path(sequence)

    # Load source motion (KEY_JOINTS only).
    motion = load_optical_sync_motion(base.with_suffix(".txt"))

    # Contact schedule.
    schedule = ContactSchedule.from_contact_file(base.with_name(base.name + "_contact.txt"))
    contact = build_contact_matrix(schedule, motion.num_frames, PAW_JOINTS)

    if mode == "raw":
        return motion, contact

    if mode == "uvm":
        if urdf_path is None:
            raise ValueError("--urdf must be provided when mode='uvm'")

        go2 = Go2Kinematics.from_urdf(urdf_path)

        # 1) Original motion (for logging).
        motion_raw = motion

        # 2) Apply vertical offsets so paws sit at the common ground plane.
        ground = load_ground_estimates(base.with_name(base.name + "_ground_estimate.txt"))
        deltas = compute_paw_ground_deltas(ground, plane_y=DEFAULT_GROUND_Y)
        motion_offset = apply_paw_ground_deltas(motion_raw, deltas)

        # 3) Unit Vector Method with paw anchored.
        seg_lengths = compute_go2_segment_lengths(go2)
        motion_uvm = rescale_limbs_to_go2_lengths(motion_offset, go2, seg_lengths)

        # 4) Align hip layout to Go2 URDF distances (second-stage UVM).
        from retargeting import align_hips_to_go2  # local import to avoid cycles

        motion_uvm = align_hips_to_go2(motion_uvm, go2)

        # Debug logging: print paw Y before/after for a few contact frames.
        name_to_idx = {n: i for i, n in enumerate(motion_raw.joint_names)}
        paw_indices = [name_to_idx[p] for p in PAW_JOINTS if p in name_to_idx]
        print("\n[DEBUG] Paw heights (raw / offset / UVM) at first contact frames:")
        for pj, paw in enumerate(PAW_JOINTS):
            if paw not in name_to_idx:
                continue
            j = name_to_idx[paw]
            t_indices = np.where(contact[:, pj])[0][:5]
            for t in t_indices:
                y_raw = motion_raw.positions[t, j, 1]
                y_off = motion_offset.positions[t, j, 1]
                y_uvm = motion_uvm.positions[t, j, 1]
                print(
                    f"  {paw:10s} frame {t:3d}: "
                    f"raw_y={y_raw: .4f}, offset_y={y_off: .4f}, uvm_y={y_uvm: .4f}"
                )

        return motion_uvm, contact

    raise ValueError(f"Unknown mode: {mode}")


def visualize_with_viser(
    motion: SourceMotion,
    contact: np.ndarray,
) -> None:
    """Simple Viser viewer for a single motion sequence."""
    try:
        import viser
    except Exception as exc:
        print(f"[ERROR] viser not available: {exc}")
        return

    server = viser.ViserServer()
    server.scene.world_axes.visible = True  # type: ignore[attr-defined]

    frames = motion.positions  # [T, N, 3]
    joints = list(motion.joint_names)
    edges = build_leg_edges(joints)

    T, N, _ = frames.shape

    # Draw a simple ground plane at y = DEFAULT_GROUND_Y for reference.
    coords_flat = frames.reshape(-1, 3)
    mask = ~np.isnan(coords_flat).any(axis=-1)
    if mask.any():
        xyz = coords_flat[mask]
        xmin, ymin, zmin = xyz.min(axis=0)
        xmax, ymax, zmax = xyz.max(axis=0)
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
    else:
        xmin_f = zmin_f = -1.0
        xmax_f = zmax_f = 1.0

    y_plane = float(DEFAULT_GROUND_Y)
    quad_pts = np.array(
        [
            [xmin_f, y_plane, zmin_f],
            [xmax_f, y_plane, zmin_f],
            [xmax_f, y_plane, zmax_f],
            [xmin_f, y_plane, zmax_f],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    server.scene.add_mesh_simple(  # type: ignore[attr-defined]
        name="ground_plane",
        vertices=quad_pts,
        faces=faces,
        color=(0, 255, 0),
        wireframe=False,
        opacity=0.25,
        side="double",
    )

    # Base colors.
    default_rgb = (255, 140, 0)   # orange
    paw_contact_rgb = (0, 255, 0)  # green

    # Create joint spheres.
    joint_objs: List[object] = []
    for j, name in enumerate(joints):
        safe = name.replace(" ", "_")
        o = server.scene.add_icosphere(  # type: ignore[attr-defined]
            name=f"joint_{j}_{safe}",
            position=(0.0, 0.0, 0.0),
            radius=0.01,
            color=default_rgb,
        )
        joint_objs.append(o)

    # Create edge segments.
    edge_objs: List[object] = []
    for ei, (_a, _b) in enumerate(edges):
        seg = server.scene.add_line_segments(  # type: ignore[attr-defined]
            name=f"edge_{ei}",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=np.tile(np.array((180, 100, 255), dtype=np.uint8)[None, None, :], (1, 2, 1)),
            line_width=2.0,
        )
        edge_objs.append(seg)

    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(joints)}
    paw_indices = [name_to_idx[p] for p in PAW_JOINTS if p in name_to_idx]

    dt = 1.0 / float(motion.fps)

    try:
        frame_idx = 0
        while True:
            coords = frames[frame_idx]  # [N, 3]
            # Update joints.
            for j, obj in enumerate(joint_objs):
                p = coords[j]
                if np.any(np.isnan(p)):
                    obj.visible = False  # type: ignore[attr-defined]
                    continue
                obj.visible = True  # type: ignore[attr-defined]
                obj.position = tuple(p.astype(np.float32))  # type: ignore[attr-defined]

                # Color paws by contact.
                if j in paw_indices:
                    paw_j = paw_indices.index(j)
                    if contact[frame_idx, paw_j]:
                        obj.color = paw_contact_rgb  # type: ignore[attr-defined]
                    else:
                        obj.color = default_rgb  # type: ignore[attr-defined]
                else:
                    obj.color = default_rgb  # type: ignore[attr-defined]

            # Update edges.
            for (a, b), obj in zip(edges, edge_objs):
                pa, pb = coords[a], coords[b]
                if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
                    obj.visible = False  # type: ignore[attr-defined]
                else:
                    pts2 = np.stack([pa, pb], axis=0).astype(np.float32)
                    obj.visible = True  # type: ignore[attr-defined]
                    obj.points = pts2[None, ...]  # type: ignore[attr-defined]

            frame_idx = (frame_idx + 1) % T
            time_sleep = dt
            import time as _time

            _time.sleep(time_sleep)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Viser visualization of motion before IK retargeting.")
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Base name of optical sync sequence, e.g. optical_sync_align_d29_t1_a_ready",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("raw", "uvm"),
        default="raw",
        help="'raw' = source motion; 'uvm' = ground-offset + UVM limb rescaling.",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="UnitreeGo2/urdf/go2_description.urdf",
        help="Go2 URDF path (only used when mode='uvm').",
    )
    args = parser.parse_args()

    urdf_path = Path(args.urdf) if args.mode == "uvm" else None
    motion, contact = prepare_motion(args.sequence, args.mode, urdf_path)
    visualize_with_viser(motion, contact)


if __name__ == "__main__":
    main()


