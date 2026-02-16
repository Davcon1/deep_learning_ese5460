from __future__ import annotations

"""
Step-by-step visualization of FK-target optimization for Go2.

This script is meant for *debugging* the STMR-style retargeting:
- It loads the Go2 URDF and a UVM keypoint file.
- For each frame:
    1) Shows the current robot configuration and the target keypoints.
    2) Runs a local FK-target optimization (Gauss-Newton in joint space).
    3) Displays the optimized configuration and reports residual errors.

You can:
- Control how many Gauss-Newton iterations to run per frame.
- Control the convergence tolerance.
- Control the sleep time between frames.

Example:

    cd /mnt/e/3ddogs/UnitreeGo2/urdf
    python go2_visualize_step_by_step.py \
        --uvm ../../optical_sync_align_d29_t1_a_ready_go2_uvm.npz \
        --max-iters 200 \
        --tol 1e-4 \
        --sleep 1.5
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

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

from retargeting import (  # type: ignore  # noqa: E402
    Go2Kinematics,
    mocap_to_robot_positions,
    build_keypoint_frame_names,
    estimate_base_pose_from_targets,
    _clamp_to_joint_limits,
    FOOT_KEYPOINTS,
    GO2_LEG_FRAMES,
    LEG_KEYPOINTS,
    DEFAULT_GROUND_Y,
)


def optimize_single_frame(
    go2: Go2Kinematics,
    q_init: np.ndarray,
    targets_robot: np.ndarray,  # (N_key, 3)
    frame_names: Sequence[str],
    weights: np.ndarray,  # (N_key,)
    q_prior: np.ndarray,
    max_iters: int,
    tol: float,
    lambda_damp: float = 1e-3,
    step_size_init: float = 1.0,
    ls_shrink: float = 0.5,
    ls_max: int = 8,
) -> Tuple[np.ndarray, float, int]:
    """
    Gauss-Newton FK-target optimization for a single frame:

        min_q 1/2 sum_k || FK_k(q) - x_k ||^2

    implemented in velocity space via repeated linearization.
    """
    q = q_init.copy()
    model, data = go2.model, go2.data

    # Best-so-far tracking so we never return something worse than we saw.
    best_q = q.copy()
    best_err = np.inf
    it = 0

    for it in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        res_blocks: List[np.ndarray] = []
        J_blocks: List[np.ndarray] = []

        # Keypoint tracking residuals (weighted).
        for k, (tgt, fname) in enumerate(zip(targets_robot, frame_names)):
            fid = go2.frame_name_to_id[fname]
            placement = data.oMf[fid]
            p = np.array(placement.translation)
            w = float(weights[k])
            s = np.sqrt(w)
            res_blocks.append(s * (tgt - p))

            J6 = pin.computeFrameJacobian(
                model, data, q, fid, pin.LOCAL_WORLD_ALIGNED
            )
            Jpos = np.asarray(J6[3:, :])  # translational part
            J_blocks.append(s * Jpos)

        # Hard-ish height tracking for all feet: keep feet near the ground plane.
        # (This is especially important for initial stance frames.)
        plane_z = float(DEFAULT_GROUND_Y)
        w_height = 5.0  # large relative weight on foot height
        s_h = np.sqrt(w_height)
        for leg, (_hip, _thigh, _calf, foot_frame) in GO2_LEG_FRAMES.items():
            try:
                fid = go2.frame_name_to_id[foot_frame]
            except KeyError:
                continue
            placement = data.oMf[fid]
            p = np.array(placement.translation)
            z_err = p[2] - plane_z
            res_blocks.append(s_h * np.array([0.0, 0.0, z_err], dtype=float))

            J6 = pin.computeFrameJacobian(
                model, data, q, fid, pin.LOCAL_WORLD_ALIGNED
            )
            Jpos = np.asarray(J6[3:, :])  # 3 x nv
            J_blocks.append(s_h * Jpos)

        # Regularization: keep configuration near a prior (q_prior).
        # This keeps joints near neutral and base pose near the initial guess.
        w_reg = 1e-2
        s_r = np.sqrt(w_reg)
        delta = pin.difference(model, q, q_prior)  # (nv,)
        res_blocks.append(s_r * delta)
        J_blocks.append(s_r * np.eye(go2.nv))

        r = np.concatenate(res_blocks, axis=0)  # (3N,)
        J = np.vstack(J_blocks)                 # (3N, nv)

        err = float(np.linalg.norm(r))
        if err < best_err:
            best_err = err
            best_q = q.copy()
        if err < tol:
            break

        nv = go2.nv
        JTJ = J.T @ J + lambda_damp * np.eye(nv)
        rhs = J.T @ r

        try:
            dq = np.linalg.solve(JTJ, rhs)
        except np.linalg.LinAlgError:
            eps = 1e-8
            dq = np.linalg.solve(JTJ + eps * np.eye(nv), rhs)

        # Simple backtracking line search to avoid increasing the residual and
        # to enforce basic height inequality constraints.
        alpha = step_size_init
        improved = False
        for _ in range(ls_max):
            q_candidate = pin.integrate(model, q, alpha * dq)
            q_candidate = _clamp_to_joint_limits(model, q_candidate)

            pin.forwardKinematics(model, data, q_candidate)
            pin.updateFramePlacements(model, data)

            # Reject configurations that violate simple height inequalities.
            if _violates_height_constraints(go2, data, eps=0.0):
                alpha *= ls_shrink
                continue

            res_ls: List[np.ndarray] = []
            for k, (tgt, fname) in enumerate(zip(targets_robot, frame_names)):
                fid = go2.frame_name_to_id[fname]
                placement = data.oMf[fid]
                p = np.array(placement.translation)
                w = float(weights[k])
                s = np.sqrt(w)
                res_ls.append(s * (tgt - p))
            r_ls = np.concatenate(res_ls, axis=0)
            err_ls = float(np.linalg.norm(r_ls))

            if err_ls < err:
                q = q_candidate
                err = err_ls
                improved = True
                break
            alpha *= ls_shrink

        if not improved:
            # Could not find a better step; stop early.
            break

    return best_q, best_err, it + 1


def _violates_height_constraints(
    go2: Go2Kinematics,
    data: pin.Data,
    eps: float = 0.0,
) -> bool:
    """
    Simple height-inequality constraints in robot world (Z-up):

    - For each leg, foot and "knee" (calf frame) must not be *above* the thigh.
    - Feet must not be above the base link.
    """
    model = go2.model

    # Base height.
    try:
        base_fid = go2.frame_name_to_id["base"]
        base_z = float(data.oMf[base_fid].translation[2])
    except KeyError:
        # Fallback: approximate from free-flyer translation part.
        base_z = 0.0

    for leg, (hip_name, thigh_name, calf_name, foot_name) in GO2_LEG_FRAMES.items():
        try:
            fid_thigh = go2.frame_name_to_id[thigh_name]
            fid_calf = go2.frame_name_to_id[calf_name]
            fid_foot = go2.frame_name_to_id[foot_name]
        except KeyError:
            continue

        z_thigh = float(data.oMf[fid_thigh].translation[2])
        z_calf = float(data.oMf[fid_calf].translation[2])
        z_foot = float(data.oMf[fid_foot].translation[2])

        # Feet and knees should not be above the thigh.
        if z_foot > z_thigh + eps:
            return True
        if z_calf > z_thigh + eps:
            return True

        # Feet should not be above the base.
        if z_foot > base_z + eps:
            return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-by-step FK-target optimization visualization for Go2."
    )
    parser.add_argument(
        "--uvm",
        type=str,
        required=True,
        help="Path to <sequence>_go2_uvm.npz (relative to this directory).",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="go2_description.urdf",
        help="Go2 URDF filename (relative to this directory).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=200,
        help="Max Gauss-Newton iterations per frame.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance on keypoint residual norm.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Seconds to sleep between frames.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=-1,
        help="Number of frames to visualize (default: all).",
    )
    args = parser.parse_args()

    uvm_path = (THIS_FILE.parent / args.uvm).resolve()
    if not uvm_path.is_file():
        raise FileNotFoundError(uvm_path)

    data = np.load(uvm_path, allow_pickle=True)
    positions_mocap = data["positions"]  # (T, N, 3) mocap world
    joint_names = [str(x) for x in data["joint_names"]]
    dt = float(data.get("dt", 1.0 / 60.0))

    T, N, _ = positions_mocap.shape

    # Transform keypoints to robot world (Z-up) to match Go2 kinematics.
    targets_robot_all = mocap_to_robot_positions(positions_mocap)  # (T, N, 3)

    # Build Go2 model from local URDF.
    urdf_path = (THIS_FILE.parent / args.urdf).resolve()
    go2 = Go2Kinematics.from_urdf(urdf_path)
    model = go2.model

    # Visual geometry model.
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

    viewer = viz.viewer

    # Target keypoint spheres (blue) in robot world coordinates.
    sphere_geom = mg.Sphere(0.02)
    sphere_mat = mg.MeshLambertMaterial(color=0x0000FF)  # blue

    keypoint_objs = []
    for j, name in enumerate(joint_names):
        safe = name.replace(" ", "_")
        obj = viewer[f"targets/joint_{j}_{safe}"]
        obj.set_object(sphere_geom, sphere_mat)
        keypoint_objs.append(obj)

    # Frame names for all keypoints, in same order as KEY_JOINTS.
    frame_names_all = build_keypoint_frame_names()

    # Map joint_names (KEY_JOINTS) to indices into targets_robot_all.
    # In UVM npz, joint_names should already be KEY_JOINTS order.
    if len(joint_names) != len(frame_names_all):
        print(
            f"[WARN] joint_names length {len(joint_names)} "
            f"!= frame_names_all length {len(frame_names_all)}"
        )

    # Build per-keypoint weights:
    #   - feet: 1.0
    #   - knee proxies: 0.2
    #   - everything else: 0.05
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    weights = np.full(len(joint_names), 0.05, dtype=float)
    for foot_name in FOOT_KEYPOINTS:
        if foot_name in name_to_idx:
            weights[name_to_idx[foot_name]] = 1.0

    # Knee proxies from LEG_KEYPOINTS (third element in each chain).
    for _leg, (_, _thigh, knee_name, _foot) in LEG_KEYPOINTS.items():
        if knee_name in name_to_idx:
            weights[name_to_idx[knee_name]] = 0.2

    # Initial pose: base from Kabsch on frame 0, joints neutral.
    q0 = estimate_base_pose_from_targets(
        go2,
        targets_robot_all[0],
        frame_names_all,
    )
    q0 = _clamp_to_joint_limits(go2.model, q0)

    q_curr = q0.copy()

    # Helper to update target spheres for a frame.
    def update_targets(t: int) -> None:
        pts = targets_robot_all[t]
        for j, p in enumerate(pts):
            Tf = mt.translation_matrix(p)
            keypoint_objs[j].set_transform(Tf)

    # Main loop over frames.
    num_frames = T if args.num_frames < 0 else min(T, args.num_frames)

    print(
        f"[INFO] Starting step-by-step optimization for {num_frames} frames "
        f"(dt={dt:.4f}, max_iters={args.max_iters}, tol={args.tol:.1e})"
    )

    t = 0
    try:
        while True:
            for t in range(num_frames):
                targets = targets_robot_all[t]

                # 1) Show current configuration and targets.
                viz.display(q_curr)
                update_targets(t)

                # Compute initial residual.
                model, data_local = go2.model, go2.data
                pin.forwardKinematics(model, data_local, q_curr)
                pin.updateFramePlacements(model, data_local)
                res_blocks: List[np.ndarray] = []
                for tgt, fname in zip(targets, frame_names_all):
                    fid = go2.frame_name_to_id[fname]
                    placement = data_local.oMf[fid]
                    p = np.array(placement.translation)
                    res_blocks.append(tgt - p)
                r0 = np.concatenate(res_blocks, axis=0)
                err0 = float(np.linalg.norm(r0))

                print(f"\n[FRAME {t}] initial residual norm: {err0:.6e}")

                # 2) Run FK-target optimization for this frame.
                q_opt, err_final, iters = optimize_single_frame(
                    go2,
                    q_curr,
                    targets,
                    frame_names_all,
                    weights,
                    q0,
                    max_iters=args.max_iters,
                    tol=args.tol,
                )
                print(
                    f"[FRAME {t}] optimization finished in {iters} iters, "
                    f"final residual: {err_final:.6e}"
                )

                # 3) Display optimized configuration.
                q_curr = q_opt
                viz.display(q_curr)
                update_targets(t)

                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, exiting.")


if __name__ == "__main__":
    main()


