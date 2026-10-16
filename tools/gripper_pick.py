"""CONTRACT
Input: none required. --fps (default 25 video frames per sim second), --out-mp4
  (default work/gripper_pick.mp4), --out-gif (default docs/gripper_pick.gif),
  --out-png (default docs/gripper_mount.png). Builds the task layout
  (chem_rl_sim.scene) with the default Robotiq 2F-85 grippers; world frame of
  urgantry_sim (m, z up, +y away from the gantry column).
What it does:
  1. PNG: the left wrist at the home pose, Robotiq (default) next to the Wuji
     hand, same free camera. Shows the gripper bolted to the flange with no
     adapter cylinder, and the cylinder + hand it replaces.
  2. Pick: the left arm picks vial_a top-down, under PHYSICS (position servos,
     the gripper's tendon actuator, contacts), not kinematic playback. Damped
     least squares IK (6 arm DOFs, scratch MjData) puts site left_grip_pinch on
     Cartesian waypoints with tool +z -> world -z and the closing axis tool +y
     -> world +x (measured: the pads close along tool y). Waypoints, relative to
     vial_a's settled base: above (z + ABOVE), grasp (z + GRASP_Z), then lift
     (z + GRASP_Z + LIFT). Vertical moves are split into 1 cm IK steps so the
     servos follow a straight line rather than a joint-space arc.
     Phases: approach, descend, close (ctrl 0 -> 255), lift, hold.
Output: printed per-phase IK errors (mm, deg), the vial lift (mm) and the vial
  tilt (deg) at the end; an mp4 of two 640x480 panels (whole bench, close-up)
  with phase / gripper ctrl / vial height burned in; a GIF of it (ffmpeg
  palette, sierra dither, 12 fps, 900 px wide).
Consumer: humans; docs/gripper_pick.gif and docs/gripper_mount.png are shown
  in README.md.
Caveat: one scripted grasp on one IK branch; no path planning, no randomized
  props (task layout as written in LAYOUTS, no env jitter).
Reference output: docs/gripper_pick.gif, docs/gripper_mount.png (committed).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chem_rl_sim.robot import GRIPPER_CLOSED, GRIPPER_OPEN  # noqa: E402
from chem_rl_sim.scene import TARGET_VIAL, build_scene  # noqa: E402

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
SIDE = "left"
ABOVE = 0.15        # m, pinch above the vial base before descending
GRASP_Z = 0.035     # m, pinch above the vial base when closing (vial is ~57 mm tall)
LIFT = 0.12         # m
W, H = 640, 480
VIEWS = [((-0.05, 0.05, 0.95), 1.3, -90, -25), ((-0.12, 0.06, 0.90), 0.55, -60, -15)]
MOUNT_VIEW = (0.35, -150, -10)          # distance, azimuth, elevation at the flange


def solve_ik(m, d, site, body, qadr, dadr, target, iters=300):
    """Pinch site to `target`, tool z -> world -z, tool y -> world +x."""
    lo, hi = m.jnt_range[[m.dof_jntid[a] for a in dadr]].T
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p, R = d.site_xpos[site], d.site_xmat[site].reshape(3, 3)
        ep = target - p
        er = np.cross(R[:, 2], [0, 0, -1.0]) + np.cross(R[:, 1], [1.0, 0, 0])
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
        mujoco.mj_jac(m, d, jacp, jacr, p, body)
        J = np.vstack([jacp[:, dadr], jacr[:, dadr]])
        step = J.T @ np.linalg.solve(J @ J.T + 0.03 ** 2 * np.eye(6), np.concatenate([ep, 0.5 * er]))
        d.qpos[qadr] = np.clip(d.qpos[qadr] + np.clip(step, -0.2, 0.2), lo, hi)
    mujoco.mj_forward(m, d)
    R = d.site_xmat[site].reshape(3, 3)
    ang = np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))
    return d.qpos[qadr].copy(), np.linalg.norm(target - d.site_xpos[site]) * 1000, ang


def mount_png(path):
    panels = []
    for hand, label in (("robotiq", "Robotiq 2F-85 on the flange (default)"),
                        ("wuji", 'hand="wuji": adapter cylinder + hand')):
        m, d = build_scene("task", hand)
        r = mujoco.Renderer(m, 500, 600)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = d.body(f"{SIDE}_flange_adapter").xpos + [0, 0.06, 0]
        cam.distance, cam.azimuth, cam.elevation = MOUNT_VIEW
        r.update_scene(d, cam)
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, img.width, 24], fill=(0, 0, 0))
        dr.text((8, 6), label, fill=(255, 255, 255))
        panels.append(np.asarray(img))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.hstack(panels)).save(path)
    print(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--out-mp4", default=str(ROOT / "work" / "gripper_pick.mp4"))
    p.add_argument("--out-gif", default=str(ROOT / "docs" / "gripper_pick.gif"))
    p.add_argument("--out-png", default=str(ROOT / "docs" / "gripper_mount.png"))
    a = p.parse_args()
    mount_png(a.out_png)

    m, d = build_scene("task")
    for _ in range(int(0.5 / m.opt.timestep)):             # let the props settle
        mujoco.mj_step(m, d)
    vial0 = d.body(TARGET_VIAL).xpos.copy()     # upright vial: body origin = its base
    qadr = np.array([m.joint(f"{SIDE}_{j}_joint").qposadr[0] for j in ARM_JOINTS])
    dadr = np.array([m.joint(f"{SIDE}_{j}_joint").dofadr[0] for j in ARM_JOINTS])
    act = np.array([m.actuator(f"{SIDE}_{j}").id for j in ARM_JOINTS])
    grip = m.actuator(f"{SIDE}_grip_fingers_actuator").id
    site, body = m.site(f"{SIDE}_grip_pinch").id, m.body(f"{SIDE}_grip_base").id

    scratch = mujoco.MjData(m)
    scratch.qpos[:] = d.qpos
    top, z0 = vial0[:2], vial0[2]

    def line(z_from, z_to):
        out = []
        for z in np.linspace(z_from, z_to, max(2, int(abs(z_to - z_from) / 0.01) + 1)):
            out.append(solve_ik(m, scratch, site, body, qadr, dadr, np.array([*top, z])))
        return out

    approach = [solve_ik(m, scratch, site, body, qadr, dadr, np.array([*top, z0 + ABOVE]))]
    descend = line(z0 + ABOVE, z0 + GRASP_Z)
    lift = line(z0 + GRASP_Z, z0 + GRASP_Z + LIFT)
    for name, wps in (("approach", approach), ("descend", descend), ("lift", lift)):
        errs = np.array([(e, g) for _, e, g in wps])
        print(f"{name:9} IK max err {errs[:, 0].max():.2f} mm, {errs[:, 1].max():.2f} deg")

    # Phases: (label, list of arm targets, seconds, gripper ctrl ramp from->to)
    q_home = d.qpos[qadr].copy()
    phases = [
        ("approach", [q for q, _, _ in approach], 2.0, (GRIPPER_OPEN, GRIPPER_OPEN)),
        ("descend", [q for q, _, _ in descend], 1.5, (GRIPPER_OPEN, GRIPPER_OPEN)),
        ("close", [descend[-1][0]], 1.0, (GRIPPER_OPEN, GRIPPER_CLOSED)),
        ("lift", [q for q, _, _ in lift], 1.5, (GRIPPER_CLOSED, GRIPPER_CLOSED)),
        ("hold", [lift[-1][0]], 1.0, (GRIPPER_CLOSED, GRIPPER_CLOSED)),
    ]
    r = mujoco.Renderer(m, H, W)
    Path(a.out_mp4).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out_mp4, fps=a.fps, macro_block_size=1)
    every = max(1, round(1 / (a.fps * m.opt.timestep)))
    prev = q_home
    for label, targets, secs, (g0, g1) in phases:
        n = int(secs / m.opt.timestep)
        path = np.vstack([prev, *targets])
        for i in range(n):
            t = i / n * (len(path) - 1)
            k = min(int(t), len(path) - 2)
            d.ctrl[act] = path[k] + (t - k) * (path[k + 1] - path[k])
            d.ctrl[grip] = g0 + (g1 - g0) * min(1.0, 2 * i / n)
            mujoco.mj_step(m, d)
            if i % every == 0:
                panels = []
                for lookat, dist, az, el in VIEWS:
                    cam = mujoco.MjvCamera()
                    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
                    r.update_scene(d, cam)
                    panels.append(r.render())
                img = Image.fromarray(np.hstack(panels))
                dr = ImageDraw.Draw(img)
                dr.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
                lift_mm = (d.body(TARGET_VIAL).xpos[2] - vial0[2]) * 1000
                dr.text((8, 5), f"Robotiq 2F-85, {SIDE} arm, physics   phase: {label:8}   "
                                f"gripper ctrl {d.ctrl[grip]:5.0f}/255   vial lifted {lift_mm:6.1f} mm",
                        fill=(255, 255, 255))
                writer.append_data(np.asarray(img))
        prev = targets[-1]
    writer.close()
    v = d.body(TARGET_VIAL)
    tilt = np.degrees(np.arccos(np.clip(v.xmat.reshape(3, 3)[2, 2], -1, 1)))
    print(f"vial lifted {(v.xpos[2] - vial0[2]) * 1000:.1f} mm (target {LIFT * 1000:.0f}), tilt {tilt:.1f} deg")

    Path(a.out_gif).parent.mkdir(parents=True, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vf = "fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", a.out_mp4, "-vf", vf, a.out_gif], check=True)
    print(a.out_mp4, a.out_gif, f"{Path(a.out_gif).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
