"""CONTRACT
Does a Robotiq grasp on vial_a survive TELEOP motion? Builds the env ("weigh" layout,
default Robotiq), grasps vial_a top-down with the left gripper (scripted IK pick from
tools/gripper_pick.py), then hands the arm to the real teleop controller
(chem_rl_sim.teleop.arm.ArmTeleop) and replays synthetic operator-hand motions at 20 Hz:
  Every motion except fast_lift first lifts 10 cm in 1 s, then for 3 s:
  lift      hold still
  fast_lift up 15 cm in 0.3 s from the bench (no pre-lift)
  sweep     left-right +-15 cm at 1 Hz (peak ~0.9 m/s, the speed cap)
  roll / pitch / yaw   wrist rotation +-40 deg at 1 Hz about the camera's optical /
            x / y axis (overhead camera: vertical / operator-right / operator-forward)
  jitter    hold with tracking noise on the pinch (sigma 8 mm) and hand (3 mm)
Each runs with the pinch held at `--pinch` (default 3.5 cm: a real, not perfectly closed
pinch). Output: per motion, the vial's max slip from the pinch site (mm), max tilt
relative to the gripper (deg), and DROPPED if it ended > 3 cm from the pinch site.
`--variant` applies a candidate fix to the compiled model / controller before the run.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from chem_rl_sim.env import VialWeighingEnv  # noqa: E402
from chem_rl_sim.scene import TARGET_VIAL  # noqa: E402
from chem_rl_sim.teleop import arm as teleop_arm  # noqa: E402
from chem_rl_sim.teleop.arm import ArmTeleop, operator_rotation  # noqa: E402
from gripper_pick import ARM_JOINTS, GRASP_Z, solve_ik  # noqa: E402

DT, SIDE = 0.05, "left"


def hand(p, R, pinch):
    kw = np.zeros((21, 3), np.float32)
    kw[8] = [pinch, 0, 0]
    return {"keypoints_cam": (kw + p).astype(np.float32), "keypoints_wrist": kw,
            "wrist_rot_cam": np.asarray(R, np.float32)}


def rot(axis, a):
    v, q, M = np.zeros(3), np.zeros(4), np.zeros(9)
    v[axis] = 1
    mujoco.mju_axisAngle2Quat(q, v, a)
    mujoco.mju_quat2Mat(M, q)
    return M.reshape(3, 3)


def grasp(env):
    """Scripted top-down grasp of vial_a; returns once the gripper has closed."""
    m, d = env.model, env.data
    qadr = np.array([m.joint(f"{SIDE}_{j}_joint").qposadr[0] for j in ARM_JOINTS])
    dadr = np.array([m.joint(f"{SIDE}_{j}_joint").dofadr[0] for j in ARM_JOINTS])
    act = np.array([m.actuator(f"{SIDE}_{j}").id for j in ARM_JOINTS])
    grip = m.actuator(f"{SIDE}_grip_fingers_actuator").id
    site, body = m.site(f"{SIDE}_grip_pinch").id, m.body(f"{SIDE}_grip_base").id
    v0 = d.body(TARGET_VIAL).xpos.copy()
    s = mujoco.MjData(m)
    s.qpos[:] = d.qpos
    path = [solve_ik(m, s, site, body, qadr, dadr, np.array([*v0[:2], z]))[0]
            for z in np.linspace(v0[2] + 0.15, v0[2] + GRASP_Z, 12)]
    for q in [path[0]] * 30 + path + [path[-1]] * 10:          # approach, descend, settle
        d.ctrl[act] = q
        env.step(d.ctrl.copy())
    for g in np.linspace(0, 255, 10).tolist() + [255] * 10:     # close
        d.ctrl[grip] = g
        env.step(d.ctrl.copy())


def vial_in_gripper(env):
    d = env.data
    site = d.site(f"{SIDE}_grip_pinch")
    R = site.xmat.reshape(3, 3)
    axis = d.body(TARGET_VIAL).xmat.reshape(3, 3)[:, 2]
    # vial axis vs the gripper's approach axis (vertical grasp: anti-parallel)
    tilt = np.degrees(np.arccos(np.clip(abs(axis @ R[:, 2]), -1, 1)))
    center = d.body(TARGET_VIAL).xpos + axis * GRASP_Z         # the grasped point on the vial
    return np.linalg.norm(center - site.xpos), tilt


LIFT_S = 1.0      # every motion first lifts 10 cm in 1 s; the motion starts after that


def lifted(f):
    def g(t):
        up = np.array([0, -0.10 * min(t / LIFT_S, 1), 0])
        if t < LIFT_S:
            return up, np.eye(3)
        dp, R = f(t - LIFT_S)
        return up + dp, R
    return g


MOTIONS = {
    "lift": lifted(lambda t: (np.zeros(3), np.eye(3))),
    "fast_lift": lambda t: (np.array([0, -0.15 * min(t / 0.3, 1), 0]), np.eye(3)),
    "sweep": lifted(lambda t: (np.array([0.15 * np.sin(2 * np.pi * t), 0, 0]), np.eye(3))),
    "roll": lifted(lambda t: (np.zeros(3), rot(2, np.deg2rad(40) * np.sin(2 * np.pi * t)))),
    "pitch": lifted(lambda t: (np.zeros(3), rot(0, np.deg2rad(40) * np.sin(2 * np.pi * t)))),
    "yaw": lifted(lambda t: (np.zeros(3), rot(1, np.deg2rad(40) * np.sin(2 * np.pi * t)))),
    "jitter": lifted(lambda t: (np.zeros(3), np.eye(3))),
}


def run(motion, variant, pinch, seed=0):
    env = VialWeighingEnv(layout="weigh", clutter=False, control_hz=1 / DT, max_episode_steps=10 ** 6)
    env.reset(seed=seed)
    m = env.model
    apply_variant(m, variant)
    grasp(env)
    a = ArmTeleop(m, env.data, SIDE, "robotiq", operator_rotation("overhead"), dt=DT)
    a.arm()
    rng = np.random.default_rng(seed)
    p0 = np.array([0.0, 0.0, 0.6])
    slip = tilt = 0.0
    secs = 1.5 if motion == "fast_lift" else LIFT_S + 3.0
    for k in range(int(secs / DT)):
        t = k * DT
        dp, R = MOTIONS[motion](t)       # dp in the overhead camera's frame (y = toward operator, -y up... see below)
        pin = pinch
        if motion == "jitter" and t >= LIFT_S:
            dp = dp + rng.normal(0, 0.003, 3)
            pin = pinch + rng.normal(0, 0.008)
        # overhead camera: -z_cam = up; motions above are written as (x right, y "up" via -z)
        dp_cam = np.array([dp[0], 0.0, dp[1]])
        a.step(hand(p0 + dp_cam, R, pin))
        ctrl = env.data.ctrl.copy()
        a.write(ctrl)
        env.step(ctrl)
        s, tl = vial_in_gripper(env)
        slip, tilt = max(slip, s), max(tilt, tl)
    end_slip, _ = vial_in_gripper(env)
    env.close()
    return slip * 1e3, tilt, end_slip > 0.03


def apply_variant(m, variant):
    pads = [i for i in range(m.ngeom) if m.geom_priority[i] == 1 and "grip" in (m.body(m.geom_bodyid[i]).name)]
    for v in variant.split("+"):
        if v == "condim6":
            m.geom_condim[pads] = 6
        elif v == "condim4":
            m.geom_condim[pads] = 4
        elif v.startswith("mu"):
            m.geom_friction[pads, 0] = float(v[2:])
        elif v.startswith("force"):
            for s in ("left", "right"):
                m.actuator_forcerange[m.actuator(f"{s}_grip_fingers_actuator").id] = [-float(v[5:]), float(v[5:])]
        elif v == "snap":
            pass                     # handled through the controller module constant
    teleop_arm.GRIP_SNAP = "snap" in variant


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base")
    ap.add_argument("--pinch", type=float, default=0.035)
    args = ap.parse_args()
    for variant in args.variants.split(","):
        rows = []
        for motion in MOTIONS:
            slip, tilt, dropped = run(motion, variant, args.pinch)
            rows.append(f"{motion}: slip {slip:5.1f} mm tilt {tilt:4.1f} deg{' DROPPED' if dropped else ''}")
        print(f"[{variant:18s}] " + " | ".join(rows), flush=True)
