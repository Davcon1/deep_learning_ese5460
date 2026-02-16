"""
Visualize the UVM-processed keypoints (before IK) together with the Go2 meshes.

Assumptions:
- Run this script from the `UnitreeGo2/urdf` directory.
- This file lives in the same directory as `go2_description.urdf`.
- The UVM keypoints were saved by `generate_go2_smr.py` as
  `<sequence>_go2_uvm.npz` in the project root.

Example:

    cd /mnt/e/3ddogs/UnitreeGo2/urdf
    python visualize_go2_uvm.py --uvm ../../optical_sync_align_d29_t1_a_ready_go2_uvm.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import meshcat.geometry as mg
import meshcat.transformations as mt
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

# Make project root importable so we can import `retargeting`.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # /mnt/e/3ddogs
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retargeting import Go2Kinematics, DEFAULT_GROUND_Y  # type: ignore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Go2 UVM keypoints with Meshcat")
    parser.add_argument(
        "--uvm",
        type=str,
        required=True,
        help="Path to <sequence>_go2_uvm.npz (relative to this directory).",
    )
    args = parser.parse_args()

    uvm_path = (THIS_FILE.parent / args.uvm).resolve()
    if not uvm_path.is_file():
        raise FileNotFoundError(uvm_path)

    data = np.load(uvm_path, allow_pickle=True)
    positions = data["positions"]  # [T, N, 3] in the same global frame as mocap
    joint_names = [str(x) for x in data["joint_names"]]
    dt = float(data.get("dt", 1.0 / 60.0))

    T, N, _ = positions.shape

    # Build Go2 model from local URDF.
    urdf_path = THIS_FILE.parent / "go2_description.urdf"
    go2 = Go2Kinematics.from_urdf(urdf_path)
    model = go2.model

    geom_model = pin.buildGeomFromUrdf(
        model,
        str(urdf_path),
        pin.GeometryType.VISUAL,
        None,
        [str(urdf_path.parent)],
    )

    viz = MeshcatVisualizer(model, None, geom_model)
    viz.initViewer(open=True)
    viz.loadViewerModel(rootNodeName="go2")

    # Align robot frame (URDF: +X forward, +Z up) to mocap world (Meshcat: +Y up,
    # motion facing roughly -X). This is purely for visualization.
    R = np.array(
        [
            [-1.0, 0.0, 0.0],  # e_x^r -> -X_w
            [0.0, 0.0, 1.0],   # e_y^r ->  Z_w
            [0.0, 1.0, 0.0],   # e_z^r ->  Y_w
        ]
    )
    T_align = np.eye(4)
    T_align[:3, :3] = R
    viz.viewer["go2"].set_transform(T_align)

    # Underlying meshcat viewer for keypoints.
    viewer = viz.viewer

    # Create spheres for each keypoint.
    sphere_geom = mg.Sphere(0.02)
    sphere_material = mg.MeshLambertMaterial(color=0xFF8C00)  # orange

    for j, name in enumerate(joint_names):
        safe = name.replace(" ", "_")
        viewer[f"uvm/joint_{j}_{safe}"].set_object(sphere_geom, sphere_material)

    # Ground plane for reference at DEFAULT_GROUND_Y.
    # Use a simple large quad in XZ.
    plane_half = 2.0
    plane_y = float(DEFAULT_GROUND_Y)
    plane_vertices = np.array(
        [
            [-plane_half, plane_y, -plane_half],
            [plane_half, plane_y, -plane_half],
            [plane_half, plane_y, plane_half],
            [-plane_half, plane_y, plane_half],
        ],
        dtype=float,
    )
    plane_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    viewer["uvm/ground"].set_object(
        mg.TriangularMeshGeometry(plane_vertices, plane_faces),
        mg.MeshLambertMaterial(color=0x00FF00, opacity=0.3),
    )

    # Show initial pose.
    viz.display(go2.q_neutral())
    print(f"Loaded UVM: {uvm_path}")
    print(f"Meshcat URL: {viewer.url()}")

    try:
        frame_idx = 0
        while True:
            # Keep robot in neutral pose; we only move the keypoint spheres.
            viz.display(go2.q_neutral())

            pts = positions[frame_idx]
            for j, name in enumerate(joint_names):
                safe = name.replace(" ", "_")
                p = pts[j]
                Tf = mt.translation_matrix(p)
                viewer[f"uvm/joint_{j}_{safe}"].set_transform(Tf)

            frame_idx = (frame_idx + 1) % T
            time.sleep(dt)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


