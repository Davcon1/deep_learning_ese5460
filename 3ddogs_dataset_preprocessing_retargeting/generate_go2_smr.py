"""
Generate and save Go2 SMR trajectories for a given optical sync sequence.

This runs the full pipeline:
- load + preprocess mocap keypoints
- ground offsets per paw
- limb rescaling to Go2 segment lengths
- IK warm start
- SMR with contact anchors and ballistic base motion

The result is saved as a compressed .npz file containing:
    q_bar  : (T, nq) retargeted generalized coordinates
    qdot   : (T, nv) velocities from SMR
    dt     : time step in seconds
    fps    : source frame rate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from retargeting import (
    Go2Kinematics,
    load_optical_sync_motion,
    load_ground_estimates,
    compute_paw_ground_deltas,
    apply_paw_ground_deltas,
    compute_go2_segment_lengths,
    rescale_limbs_to_go2_lengths,
    align_hips_to_go2,
    run_smr_from_uvm,
    ContactSchedule,
    DEFAULT_GROUND_Y,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Go2 SMR trajectory and save to .npz")
    parser.add_argument(
        "--sequence",
        type=str,
        default="optical_sync_align_d29_t1_a_ready",
        help="Base name of the optical sync sequence (without suffixes).",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=str(Path(__file__).parent / "UnitreeGo2" / "urdf" / "go2_description.urdf"),
        help="Path to Go2 URDF with relative mesh paths resolved.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npz path. Defaults to <sequence>_go2_smr.npz",
    )
    args = parser.parse_args()

    base = Path(args.sequence)
    urdf_path = Path(args.urdf)
    if args.out is None:
        out_path = base.with_name(base.name + "_go2_smr.npz")
    else:
        out_path = Path(args.out)
    uvm_out_path = base.with_name(base.name + "_go2_uvm.npz")

    print(f"URDF:     {urdf_path}")
    print(f"Sequence: {base}")
    print(f"Output SMR: {out_path}")
    print(f"Output UVM: {uvm_out_path}")

    # --- Build kinematics ---
    go2 = Go2Kinematics.from_urdf(urdf_path)

    # --- Load mocap (KEY_JOINTS only) ---
    motion = load_optical_sync_motion(base.with_suffix(".txt"))

    # --- Vertical offsets: align each paw to the common ground plane ---
    ground = load_ground_estimates(base.with_name(base.name + "_ground_estimate.txt"))
    deltas = compute_paw_ground_deltas(ground, plane_y=DEFAULT_GROUND_Y)
    motion = apply_paw_ground_deltas(motion, deltas)

    # --- Unit Vector Method, stage 1: per-leg paw→hip rescaling to Go2 segment lengths ---
    seg_lengths = compute_go2_segment_lengths(go2)
    motion = rescale_limbs_to_go2_lengths(motion, go2, seg_lengths)

    # --- Unit Vector Method, stage 2: align hip layout to Go2 URDF distances ---
    # This enforces that the four hip joints behave like a rigid body matching
    # the Go2 shoulder/hip geometry, while preserving the local limb shapes
    # produced by the paw-anchored UVM above.
    motion = align_hips_to_go2(motion, go2)

    # --- Save UVM keypoints for debugging / Meshcat visualization ---
    np.savez_compressed(
        uvm_out_path,
        positions=motion.positions,
        joint_names=np.array(motion.joint_names),
        dt=1.0 / float(motion.fps),
        fps=motion.fps,
    )
    print(f"Saved UVM keypoints to {uvm_out_path}")

    # --- Contact schedule ---
    schedule = ContactSchedule.from_contact_file(
        base.with_name(base.name + "_contact.txt")
    )
    paw_names = ("l_meta_carp", "r_meta_carp", "l_meta_tars", "r_meta_tars")

    dt = 1.0 / motion.fps

    # --- STMR-style SMR driven directly from UVM keypoints ---
    smr = run_smr_from_uvm(
        go2,
        motion,
        schedule,
        paw_names,
        dt=dt,
        plane_y=DEFAULT_GROUND_Y,
    )

    np.savez_compressed(
        out_path,
        q_bar=smr.q,
        qdot=smr.qdot,
        dt=dt,
        fps=motion.fps,
    )
    print(f"Saved SMR trajectory to {out_path}")


if __name__ == "__main__":
    main()


