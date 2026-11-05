"""CONTRACT
One arm's teleop controller: operator hand (wire.py hand dict, OpenCV camera frame) ->
that arm's actuator targets (6 arm joints + effector), written into a full ctrl vector.
Mapping (relative, in the operator's own frame):
  The operator sits where top1 looks from, so their frame (x = their right, y = forward,
  away from them, z = up) IS the world frame as seen in the top1 view (top1 looks along
  world +y with +x to the right). A hand delta in the real camera's OpenCV frame is
  rotated into that frame by operator_rotation(view):
      selfie    camera in front of the operator, facing them (e.g. on the monitor):
                image right = operator's LEFT, image down = down, optical axis = toward them
      overhead  camera above the hands looking down, image top pointing away from them
      tool_pos = ref_tool_pos + gain * R_wc @ (hand_pos - ref_hand_pos)
      tool_rot = (R_wc dR_cam R_wc^T) @ ref_tool_rot,  dR_cam = R_hand R_hand_ref^T
  so moving your hand right/forward/up moves the tool right/forward/up in the top1 view,
  and turning it turns the tool about the same axis. --camera-pitch tilts the camera
  frame about the operator's x (positive = optical axis tilted down / forward).
  Only deltas are used: camera position and the monocular depth offset drop out; a depth
  scale error becomes a gain on motion toward/away from the camera.
Clutch: `arm()` asks to engage on the next frame the hand is seen; engaging latches the
  hand pose and the tool's ACTUAL pose as the shared origin, so it never makes a step.
  `release()` holds the last command. While released NOTHING changes (arm or effector).
  A lost hand also holds; the reference survives, so tracking resumes in place.
Tool frame: Robotiq -> site {side}_grip_pinch (rotations pivot at the fingertips);
  Wuji -> body {side}_hand_palm_link.
Effector: Robotiq ctrl 0 (open) .. 255 (closed) from thumb-index tip distance, linear
  between PINCH_CLOSED_M and PINCH_OPEN_M and saturated outside, then latched (GRIP_SNAP):
  past GRAB_AT it commands fully closed until the pinch has stayed open RELEASE_HOLD_S; Wuji: the worker's 20 retargeted finger angles, clipped to ctrlrange.
Smoothing/limits ported from ~/WiLoR/teleop/teleop_sim.py (EMA expressed per second,
  speed cap, jump rejection with a timed recovery so a real large move is not refused
  forever), a reach shell around the shoulder so an unreachable target cannot pin IK,
  a world box over the bench (WORKSPACE_*) so a bad depth estimate cannot fold an arm
  up into the camera, and a joint speed cap (MAX_JOINT_SPEED) against IK branch flips.
"""
from __future__ import annotations

import numpy as np
import mujoco

from ..scene import BOARD_TOP

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")

REF_HZ = 30.0
POS_ALPHA = 0.35            # per-tick EMA weights at REF_HZ, converted with ema_alpha()
ROT_ALPHA = 0.35
EFF_ALPHA = 0.5
MAX_SPEED_MS = 0.9          # tool translation cap, m/s
REJECT_JUMP_M = 0.08        # target jump in one tick that reads as WiLoR flicker
REJECT_RECOVER_S = 0.5      # ... unless it persists this long, then slew to it
REACH_MIN_M, REACH_MAX_M = 0.20, 0.85   # tool distance from the shoulder (UR7e ~0.85 m reach)
# Tool target box, world frame: the bench top (vials sit at BOARD_TOP) up to the top of the
# draft shield and a little above. Lower z is per effector: the Robotiq TCP is at the
# fingertips, the Wuji palm is ~4 cm above its fingertips.
WORKSPACE_XY = np.array([[-0.65, -0.30], [0.65, 0.55]])
WORKSPACE_Z_ABOVE_BENCH = {"robotiq": (0.005, 0.55), "wuji": (0.04, 0.55)}
MAX_JOINT_SPEED = 3.0       # rad/s per joint; UR7e max is ~pi
PINCH_CLOSED_M, PINCH_OPEN_M = 0.03, 0.09
# Grasp latch (tools/grasp_hold_test.py): with the plain linear map a relaxed 5 cm pinch
# commands ~170/255, the gripper barely squeezes the 28 mm vial and it falls out on any
# motion, even a straight lift; at 255 the pads press ~40 N and it survives every test
# motion. So once the pinch passes GRAB_AT the gripper is commanded fully closed, and it
# lets go only after the pinch has stayed open past RELEASE_AT for RELEASE_HOLD_S: one
# noisy WiLoR frame must not drop the vial.
GRIP_SNAP = True
GRAB_AT, RELEASE_AT = 0.6 * 255, 0.25 * 255     # pinch < ~5.4 cm closes, > ~7.5 cm opens
RELEASE_HOLD_S = 0.2
THUMB_TIP, INDEX_TIP = 4, 8


def ema_alpha(alpha_ref: float, dt: float) -> float:
    """Per-tick EMA weight at REF_HZ -> the weight with the same time constant at dt."""
    tau = -1.0 / (REF_HZ * np.log(1.0 - alpha_ref))
    return float(1.0 - np.exp(-dt / tau))


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def slerp_mat(R0: np.ndarray, R1: np.ndarray, a: float) -> np.ndarray:
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, R0.flatten())
    mujoco.mju_mat2Quat(q1, R1.flatten())
    if q0 @ q1 < 0:
        q1 = -q1
    dq, v = np.zeros(4), np.zeros(3)
    q0inv = q0 * np.array([1, -1, -1, -1])
    mujoco.mju_mulQuat(dq, q1, q0inv)
    mujoco.mju_quat2Vel(v, dq, 1.0)
    step, out, M = np.zeros(4), np.zeros(4), np.zeros(9)
    mujoco.mju_axisAngle2Quat(step, v / (np.linalg.norm(v) + 1e-12), a * np.linalg.norm(v))
    mujoco.mju_mulQuat(out, step, q0)
    mujoco.mju_quat2Mat(M, out)
    return M.reshape(3, 3)


# World(= operator)-from-OpenCV-camera rotations; columns are the camera's x, y, z axes.
CAMERA_VIEWS = {
    "selfie": np.array([[-1.0, 0, 0], [0, 0, -1], [0, -1, 0]]),     # x->left, y->down, z->toward you
    "overhead": np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]]),    # x->right, y->toward you, z->down
}


def operator_rotation(view: str = "selfie", pitch_deg: float = 0.0) -> np.ndarray:
    """World-from-OpenCV-camera rotation for a real camera in `view` (see CONTRACT)."""
    p = np.deg2rad(pitch_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Rx @ CAMERA_VIEWS[view]


def pinch_to_gripper(keypoints_wrist: np.ndarray) -> float:
    d = float(np.linalg.norm(keypoints_wrist[THUMB_TIP] - keypoints_wrist[INDEX_TIP]))
    return 255.0 * float(np.clip((PINCH_OPEN_M - d) / (PINCH_OPEN_M - PINCH_CLOSED_M), 0.0, 1.0))


class ArmTeleop:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str, effector: str,
                 R_wc: np.ndarray, gain: float = 1.0, dt: float = 0.05):
        self.model, self.data, self.side, self.effector = model, data, side, effector
        self.R_wc, self.gain, self.dt = R_wc, gain, dt
        m = model
        self.arm_act = np.array([m.actuator(f"{side}_{j}").id for j in ARM_JOINTS])
        joints = [m.joint(f"{side}_{j}_joint") for j in ARM_JOINTS]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dofs = np.array([j.dofadr[0] for j in joints])
        self.q_lo = np.array([j.range[0] if j.limited else -2 * np.pi for j in joints])
        self.q_hi = np.array([j.range[1] if j.limited else 2 * np.pi for j in joints])
        if effector == "robotiq":
            self.eff_act = np.array([m.actuator(f"{side}_grip_fingers_actuator").id])
            self.site = m.site(f"{side}_grip_pinch").id
            self.body = int(m.site_bodyid[self.site])
        else:
            self.eff_act = np.array([m.actuator(f"{side}_hand_finger{f}_joint{j}").id
                                     for f in range(1, 6) for j in range(1, 5)])
            self.site, self.body = -1, m.body(f"{side}_hand_palm_link").id
        self.shoulder = m.body(f"{side}_robot_mount").id
        self._scratch = mujoco.MjData(m)
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        """Hold whatever the sim is commanding now; clutch open."""
        self.q_cmd = self.data.ctrl[self.arm_act].copy()
        self.eff_cmd = self.data.ctrl[self.eff_act].copy()
        self.engaged = self.armed = False
        self.gripping = False
        self._open_t = 0.0
        self.ref = None
        self.pos = self.rot = self._raw = None
        self._reject_t, self._recovering = 0.0, False
        self.last = {"seen": False, "ik_err_mm": 0.0}

    def arm(self) -> None:
        """Engage on the next frame this hand is seen."""
        if not self.engaged:
            self.armed = True

    def release(self) -> None:
        self.engaged = self.armed = False

    @property
    def active(self) -> bool:
        return self.engaged or self.armed

    def tool_pose(self, data: mujoco.MjData | None = None):
        d = data or self.data
        if self.site >= 0:
            return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()
        return d.xpos[self.body].copy(), d.xmat[self.body].reshape(3, 3).copy()

    # ---------------------------------------------------------------- tick
    def step(self, hand: dict | None) -> None:
        """Advance one control tick and update q_cmd / eff_cmd."""
        self.last["seen"] = hand is not None
        if hand is None:
            return
        p_hand = np.asarray(hand["keypoints_cam"][0], float)
        R_hand = orthonormalize(np.asarray(hand["wrist_rot_cam"], float))
        if self.armed:
            p_tool, R_tool = self.tool_pose()
            self.ref = (p_hand, R_hand, p_tool, R_tool)
            self.pos, self.rot, self._raw = p_tool, R_tool, p_tool
            self._reject_t, self._recovering = 0.0, False
            self.engaged, self.armed = True, False
        if not self.engaged:
            return

        ph0, Rh0, pt0, Rt0 = self.ref
        pos = pt0 + self.gain * self.R_wc @ (p_hand - ph0)
        dR = self.R_wc @ (R_hand @ Rh0.T) @ self.R_wc.T
        rot = orthonormalize(dR @ Rt0)
        shoulder = self.data.xpos[self.shoulder]
        r = pos - shoulder
        n = np.linalg.norm(r)
        pos = shoulder + r * np.clip(n, REACH_MIN_M, REACH_MAX_M) / max(n, 1e-9)
        pos[:2] = np.clip(pos[:2], *WORKSPACE_XY)
        pos[2] = np.clip(pos[2], *(BOARD_TOP + np.array(WORKSPACE_Z_ABOVE_BENCH[self.effector])))

        # Compared against the last ACCEPTED raw target, not the filtered pose: the EMA
        # lags a fast hand by v * tau (~4 cm at 0.5 m/s), which would read as a jump.
        if np.linalg.norm(pos - self._raw) > REJECT_JUMP_M and not self._recovering:
            self._reject_t += self.dt
            if self._reject_t < REJECT_RECOVER_S:
                return                                  # one bad frame: hold
            self._recovering = True                     # persistent: slew to it
        if np.linalg.norm(pos - self.pos) <= REJECT_JUMP_M:
            self._recovering = False                    # caught up
        self._reject_t, self._raw = 0.0, pos
        step = ema_alpha(POS_ALPHA, self.dt) * (pos - self.pos)
        n = np.linalg.norm(step)
        if n > MAX_SPEED_MS * self.dt:
            step *= MAX_SPEED_MS * self.dt / n
        self.pos = self.pos + step
        self.rot = slerp_mat(self.rot, rot, ema_alpha(ROT_ALPHA, self.dt))
        q, err = self._ik(self.pos, self.rot)
        lim = MAX_JOINT_SPEED * self.dt
        self.q_cmd = self.q_cmd + np.clip(q - self.q_cmd, -lim, lim)
        self.last["ik_err_mm"] = 1e3 * err

        a = ema_alpha(EFF_ALPHA, self.dt)
        if self.effector == "robotiq":
            g = pinch_to_gripper(np.asarray(hand["keypoints_wrist"]))
            if GRIP_SNAP:
                if not self.gripping:
                    self.gripping = g >= GRAB_AT
                else:
                    self._open_t = self._open_t + self.dt if g < RELEASE_AT else 0.0
                    if self._open_t >= RELEASE_HOLD_S:
                        self.gripping, self._open_t = False, 0.0
                g = 255.0 if self.gripping else min(g, GRAB_AT)
            target = np.array([g])
        elif "fingers" in hand:
            target = np.asarray(hand["fingers"], float)
        else:
            return
        lo, hi = self.model.actuator_ctrlrange[self.eff_act].T
        self.eff_cmd = np.clip(self.eff_cmd + a * (target - self.eff_cmd), lo, hi)

    def write(self, ctrl: np.ndarray) -> None:
        ctrl[self.arm_act] = self.q_cmd
        ctrl[self.eff_act] = self.eff_cmd

    # ---------------------------------------------------------------- IK
    def _ik(self, pos, rot, iters: int = 30, damping: float = 0.05, rot_weight: float = 0.5,
            max_step: float = 0.2):
        """Damped least squares on the 6 arm joints, warm-started from the last command
        so it stays on the current branch. Runs on a scratch MjData."""
        m, s = self.model, self._scratch
        s.qpos[:] = self.data.qpos
        q = self.q_cmd.copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        qt, qc, qd, w = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_mat2Quat(qt, rot.flatten())
        err_p = np.inf
        for _ in range(iters):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, s)
            mujoco.mj_comPos(m, s)
            p, R = self.tool_pose(s)
            err_p = pos - p
            mujoco.mju_mat2Quat(qc, R.flatten())
            mujoco.mju_negQuat(qc, qc)
            mujoco.mju_mulQuat(qd, qt, qc)
            mujoco.mju_quat2Vel(w, qd, 1.0)
            if np.linalg.norm(err_p) < 1e-3 and np.linalg.norm(w) < 0.01:
                break
            mujoco.mj_jac(m, s, jacp, jacr, p, self.body)
            J = np.vstack([jacp[:, self.dofs], rot_weight * jacr[:, self.dofs]])
            e = np.concatenate([err_p, rot_weight * w])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -max_step, max_step), self.q_lo, self.q_hi)
        return q, float(np.linalg.norm(err_p))
