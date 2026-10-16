"""CONTRACT
Input: --hand wuji|sharpa (default wuji), --out-mp4 (default work/ik_door_reach.mp4),
  --out-gif (default docs/ik_door_reach.gif), --fps (default 20). Builds the task
  layout (chem_rl_sim.scene) with props, world frame of urgantry_sim (m, z up,
  +y away from the gantry column).
What it does: for each arm, damped-least-squares IK (6 arm DOFs, scratch MjData)
  on a reach point 0.10 m along the palm's +z (the finger direction; measured:
  palm->finger3 tip = (-0.02, +-0.007, 0.166) m in palm frame), with palm +z held
  horizontal and pointing into the chamber (+x for the left arm through the left
  door, -x for the right arm through the right door), and palm-frame +y (left) /
  -y (right) pointing at world -z, which puts the thumb under the palm and the
  wrist above it. Waypoints (reach point):
    pre   = pan + (-+0.30, 0, 0.12)   outside the door
    door  = pan + (-+0.12, 0, 0.12)   in the door plane (balance side wall)
    pan   = pan + (0, 0, 0.12)        over the pan centre
  Height and roll come from a sweep (palm-down axis none/+-x/+-y, height 0.05-0.12
  m, contacts summed over the interpolated path): only this roll at >=0.10 m
  (right) / 0.12 m (left) is contact-free; lower, the thumb hits the pan/balance
  and the wrist hits vials and the Kimwipes/cardboard boxes.
  IK is warm-started along the chain home -> pre -> door -> pan. Playback is
  kinematic (qpos + mj_forward, no servos), joint-space linear between waypoints,
  with that arm's door opened to its 0.20 m limit.
Output: printed table per arm/waypoint: position error (mm), finger-axis error
  (deg), contacts between the arm subtree and anything outside it, and the
  deepest penetration (mm). An mp4 (1280x480, two 640x480 panels, both cameras
  on the +y edge facing the column: whole gantry, and a close-up of the chamber) with that state burned
  in, and a GIF of it (ffmpeg 256-colour palette, sierra dither, 8 fps, 800 px wide).
Consumer: humans; docs/ik_door_reach.gif is shown in README.md.
Caveat: one warm-started IK branch, no path planning; contacts are counted along
  the interpolated playback, so they are evidence about this path only.
Reference output: docs/ik_door_reach.gif (committed); work/ik_door_reach.mp4.
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
from chem_rl_sim.scene import build_scene  # noqa: E402

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
REACH = 0.10                     # m along palm +z
HEIGHT = 0.12                    # m, reach point above the pan top (see contract)
W, H = 640, 480
# Both cameras on the +y edge, opposite the column, which otherwise hides the chamber.
VIEWS = [((0.0, 0.0, 1.0), 2.0, -90, -25), ((0.10, 0.09, 0.95), 0.85, -90, -20)]


def palm_name(side, hand):
    return f"{side}_hand_palm_link" if hand == "wuji" else f"{side}_hand_palm"


def reach_point(d, palm):
    R = d.xmat[palm].reshape(3, 3)
    return d.xpos[palm] + R[:, 2] * REACH, R[:, 2]


def solve_ik(m, d, palm, qadr, dadr, target, axis_dir, down=None, iters=400):
    lo, hi = m.jnt_range[[m.dof_jntid[a] for a in dadr]].T
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p, z = reach_point(d, palm)
        ep = target - p
        er = np.cross(z, axis_dir)          # rotate palm z onto axis_dir
        if down is not None:                # and palm-frame `down` onto world -z
            R = d.xmat[palm].reshape(3, 3)
            er = er + np.cross(R @ down, [0.0, 0.0, -1.0])
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
        mujoco.mj_jac(m, d, jacp, jacr, p, palm)
        J = np.vstack([jacp[:, dadr], jacr[:, dadr]])
        e = np.concatenate([ep, 0.5 * er])
        step = J.T @ np.linalg.solve(J @ J.T + 0.03 ** 2 * np.eye(6), e)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + np.clip(step, -0.2, 0.2), lo, hi)
    mujoco.mj_forward(m, d)
    p, z = reach_point(d, palm)
    ang = np.degrees(np.arccos(np.clip(z @ axis_dir, -1, 1)))
    return d.qpos[qadr].copy(), np.linalg.norm(target - p) * 1000, ang


def subtree_mask(m, root):
    inside = np.zeros(m.nbody, bool)
    for b in range(m.nbody):
        x = b
        while x > 0 and x != root:
            x = m.body_parentid[x]
        inside[b] = x == root
    return inside[m.geom_bodyid]


def contacts(d, arm_geom):
    n = d.ncon
    g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
    hit = arm_geom[g1] != arm_geom[g2]
    if not hit.any():
        return 0, 0.0
    return int(hit.sum()), float(max(0.0, -d.contact.dist[:n][hit].min()) * 1000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand", choices=("wuji", "sharpa"), default="wuji")
    p.add_argument("--out-mp4", default=str(ROOT / "work" / "ik_door_reach.mp4"))
    p.add_argument("--out-gif", default=str(ROOT / "docs" / "ik_door_reach.gif"))
    p.add_argument("--fps", type=int, default=20)
    a = p.parse_args()

    m, d = build_scene("task", a.hand)
    home = d.qpos.copy()
    pan = d.site("balance_pan_site").xpos.copy()
    scratch = mujoco.MjData(m)
    plan = []                                # (side, door, [(name, q, err, ang)])
    for side, sgn in (("left", -1), ("right", 1)):
        qadr = np.array([m.joint(f"{side}_{j}_joint").qposadr[0] for j in ARM_JOINTS])
        dadr = np.array([m.joint(f"{side}_{j}_joint").dofadr[0] for j in ARM_JOINTS])
        palm = m.body(palm_name(side, a.hand)).id
        inward = np.array([-sgn, 0.0, 0.0])
        down = np.array([0.0, -sgn, 0.0])     # palm-frame axis to point at world -z
        scratch.qpos[:] = home
        wps = [("home", home[qadr].copy(), 0.0, 0.0)]
        for name, off in (("pre", (0.30, 0, HEIGHT)), ("door", (0.12, 0, HEIGHT)), ("pan", (0, 0, HEIGHT))):
            target = pan + np.array([sgn * off[0], off[1], off[2]])
            q, err, ang = solve_ik(m, scratch, palm, qadr, dadr, target, inward, down)
            wps.append((name, q, err, ang))
        plan.append((side, qadr, palm, wps))

    seq = []                                 # frames: (side, label, qpos, door side)
    for side, qadr, palm, wps in plan:
        order = [0, 1, 2, 3, 3, 2, 1, 0]
        for i, j in zip(order[:-1], order[1:]):
            n = 6 if i == j else 18
            for t in np.linspace(0, 1, n, endpoint=False):
                q = home.copy()
                q[qadr] = (1 - t) * wps[i][1] + t * wps[j][1]
                seq.append((side, wps[j][0], q))
        seq.append((side, "home", home.copy()))

    print(f"{'arm':6} {'wp':5} {'pos err mm':>10} {'axis err deg':>12} {'contacts':>8} {'pen mm':>7}")
    worst = {}
    r = mujoco.Renderer(m, H, W)
    Path(a.out_mp4).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out_mp4, fps=a.fps, macro_block_size=1)
    arm_geom = {s: subtree_mask(m, m.body(f"{s}_base").id) for s in ("left", "right")}
    door_adr = {s: m.joint(f"balance_door_{s}").qposadr[0] for s in ("left", "right")}
    errs = {(s, w[0]): (w[2], w[3]) for s, _, _, wps in plan for w in wps}
    for side, label, q in seq:
        d.qpos[:] = q
        d.qpos[door_adr[side]] = 0.20
        mujoco.mj_forward(m, d)
        nc, pen = contacts(d, arm_geom[side])
        k = (side, label)
        if label != "home" and (k not in worst or (nc, pen) > worst[k][:2]):
            worst[k] = (nc, pen)
        panels = []
        for lookat, dist, az, el in VIEWS:
            cam = mujoco.MjvCamera()
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
            r.update_scene(d, cam)
            panels.append(r.render())
        img = Image.fromarray(np.hstack(panels))
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
        e, ang = errs[k]
        dr.text((8, 5), f"{side} arm via {side} door -> {label}   IK err {e:.1f} mm / {ang:.1f} deg   "
                        f"contacts {nc}  max penetration {pen:.1f} mm", fill=(255, 255, 255))
        writer.append_data(np.asarray(img))
    writer.close()
    for side, _, _, wps in plan:
        for name, _, err, ang in wps[1:]:
            nc, pen = worst.get((side, name), (0, 0.0))
            print(f"{side:6} {name:5} {err:10.2f} {ang:12.2f} {nc:8d} {pen:7.1f}")
    Path(a.out_gif).parent.mkdir(parents=True, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vf = "fps=8,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", a.out_mp4, "-vf", vf, a.out_gif], check=True)
    print(a.out_mp4, a.out_gif, f"{Path(a.out_gif).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
