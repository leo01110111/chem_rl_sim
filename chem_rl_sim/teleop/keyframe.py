"""CONTRACT
Keyframe teleop core (no GUI): the operator sets a target pose per arm while the sim is
frozen, sees a planned and checked preview, then executes it; only execution is
simulated and recorded. Driven by keyframe_app.py (mouse/keyboard) and by tests.
  Session(env, recorder=None)
    targets[side] = [pos (3,), rot (3, 3)]   tool target (Robotiq pinch site), world frame
    plan()          -> Plan for moving every arm from its current tool pose to its target
    execute(force)  queue that plan (refused if blocked, unless force)
    grip(side)      queue a gripper toggle (0 open .. 255 closed, ramped, then settles)
    follow = True   live mode: arms slew toward their targets every tick, speed-capped
    tick() -> bool  advance one control tick (dt = 1 / env control_hz) if there is work;
                    False = nothing to do, the sim stays frozen and nothing is recorded
    undo()          restore the sim, targets and recorder to before the last motion
    begin_episode() / save_episode() / discard_episode()
Plan: min-jerk Cartesian path (position lerp, rotation slerp) sampled at the control rate,
  each sample solved by damped least squares IK warm-started from the previous sample, so
  the preview IS the joint trajectory that runs. Duration from LIN_SPEED / ANG_SPEED
  (average speeds), at least MIN_MOVE_S; stretched if a joint would exceed MAX_JOINT_SPEED.
  status: "ok" | "contact" (arm or carried object touches a movable thing: a prop, a door,
          the pan -- allowed, that is how doors get pushed) | "blocked" (IK misses the
          target by > IK_TOL_MM / IK_TOL_DEG, or an arm or carried object penetrates static
          geometry or the other arm by > PEN_TOL). Contacts already present at the start
          are not counted against the move.
  A closed gripper carries the free bodies it touches: the check moves them with the tool.
Servo targets: ctrl = q_cmd + gravity_ff() (static gravity torque / kp), so the
  position servos settle ON the keyframe instead of ~9 mm below it. The recorded action
  is that ctrl, i.e. exactly what was sent to the env.
Recording: every executed tick is env.step + Recorder.add (OGPO/robomimic layout, see
  recorder.py) with operator/ = goal_pos (2,3), goal_quat (2,4) wxyz, segment (int32,
  one id per executed keyframe / gripper action / follow burst), kind (0 move, 1 grip,
  2 follow). Frozen time is never recorded, so demos carry no idle stretches.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..robot import GRIPPER_CLOSED, GRIPPER_OPEN
from ..scene import BOARD_TOP, TARGET_VIAL
from .arm import ARM_JOINTS, WORKSPACE_XY, orthonormalize, slerp_mat
from .recorder import observe

SIDES = ("left", "right")
LIN_SPEED = 0.12            # m/s, average along a keyframe move
ANG_SPEED = np.deg2rad(70)  # rad/s, average
MIN_MOVE_S = 0.5
MAX_JOINT_SPEED = 2.0       # rad/s per joint along a plan
SETTLE_MAX_S = 1.0          # after a move: run until the arm is on its command, at most this long
SETTLE_QVEL = 0.03          # rad/s
SETTLE_QERR = 1e-3          # rad, joints vs command (gravity_ff removes the static sag)
GRIP_RAMP_S = 0.4
GRIP_SETTLE_MAX_S = 1.0
FOLLOW_SPEED = 0.15         # m/s, follow mode
FOLLOW_ANG_SPEED = np.deg2rad(90)
IK_TOL_MM, IK_TOL_DEG = 3.0, 2.0
PEN_TOL = 0.001             # m of penetration that counts as a collision
CHECK_EVERY = 2             # collision-check every N plan samples (and the last)
WORKSPACE_Z = (BOARD_TOP + 0.005, BOARD_TOP + 0.6)
MAX_UNDO = 64
# When the straight path touches or hits something, try going up, across and down instead:
# the extra height above the higher end, tried in this order (first clean one wins).
ROUTE_LIFTS = (0.0, 0.05, 0.10, 0.18, 0.28)
ROUTE_MIN_XY = 0.02         # arms moving less than this sideways keep the straight path
ROUTE_LEG_MIN_S = 0.3
KIND = {"move": 0, "grip": 1, "follow": 2}

# Object-centric snap poses (pinch site relative to the vial base / pan top), see snap().
ABOVE = 0.12                # pre-grasp / pre-place height above the grasp height
GRASP_Z = 0.035             # pinch above the vial base when closing (tools/gripper_pick.py)
PLACE_CLEAR = 0.003         # vial base above the pan when placing
DOOR_HANDLE = "balance_door_top_handle"
HANDLE_DZ = 0.016           # pinch above the handle's centre: lower and the pads drag the glass
HANDLE_ABOVE = 0.10
SNAP_AT = 0.015             # gripper this close to a snap's first stage -> the next press goes on


def min_jerk(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def rot_angle(R0: np.ndarray, R1: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1.0, 1.0)))


def mat2quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def top_down(yaw: float) -> np.ndarray:
    """Tool z -> world -z, closing axis tool y -> (cos yaw, sin yaw, 0)."""
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


def nearest_top_down(R: np.ndarray) -> np.ndarray:
    """The top-down pose closest to R (the 2F-85 is symmetric under a 180 deg yaw)."""
    yaws = np.deg2rad(np.arange(0, 360, 5))
    return min((top_down(a) for a in yaws), key=lambda T: rot_angle(R, T))


def axis_rot(axis, ang: float) -> np.ndarray:
    q, M = np.zeros(4), np.zeros(9)
    a = np.asarray(axis, float)
    mujoco.mju_axisAngle2Quat(q, a / np.linalg.norm(a), ang)
    mujoco.mju_quat2Mat(M, q)
    return M.reshape(3, 3)


def _subtree(m: mujoco.MjModel, root: int) -> set[int]:
    out = {root}
    for b in range(root + 1, m.nbody):
        if m.body_parentid[b] in out:
            out.add(b)
    return out


class Arm:
    """One UR7e + Robotiq: kinematics, IK and the ids observe() and the recorder need."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str):
        m = model
        self.model, self.data, self.side = model, data, side
        self.arm_act = np.array([m.actuator(f"{side}_{j}").id for j in ARM_JOINTS])
        joints = [m.joint(f"{side}_{j}_joint") for j in ARM_JOINTS]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dofs = np.array([j.dofadr[0] for j in joints])
        self.q_lo = np.array([j.range[0] for j in joints])
        self.q_hi = np.array([j.range[1] for j in joints])
        self.eff_act = np.array([m.actuator(f"{side}_grip_fingers_actuator").id])
        self.site = m.site(f"{side}_grip_pinch").id
        self.body = int(m.site_bodyid[self.site])
        self.bodies = _subtree(m, int(m.jnt_bodyid[joints[0].id]))
        self.geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in self.bodies}
        self.grip_bodies = _subtree(m, m.body(f"{side}_grip_base").id)
        self.finger_dof = m.joint(f"{side}_grip_right_driver_joint").dofadr[0]

    def tool_pose(self, data: mujoco.MjData | None = None):
        d = data or self.data
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, s: mujoco.MjData, pos, rot, q0, iters: int = 30, damping: float = 0.05,
           rot_weight: float = 0.5, max_step: float = 0.2):
        """DLS IK of the 6 arm joints on scratch data `s` (other joints as already set).
        Returns (q, position error m, rotation error rad)."""
        m = self.model
        q = np.array(q0, float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        qt, qc, qd, w = mat2quat(rot), np.zeros(4), np.zeros(4), np.zeros(3)
        err_p, err_r = np.full(3, np.inf), np.full(3, np.inf)
        for i in range(iters + 1):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, s)
            p, R = self.tool_pose(s)
            err_p = pos - p
            mujoco.mju_mat2Quat(qc, R.flatten())
            mujoco.mju_negQuat(qc, qc)
            mujoco.mju_mulQuat(qd, qt, qc)
            mujoco.mju_quat2Vel(w, qd, 1.0)
            err_r = w
            if (np.linalg.norm(err_p) < 2e-4 and np.linalg.norm(w) < 2e-3) or i == iters:
                break
            mujoco.mj_comPos(m, s)
            mujoco.mj_jac(m, s, jacp, jacr, p, self.body)
            J = np.vstack([jacp[:, self.dofs], rot_weight * jacr[:, self.dofs]])
            e = np.concatenate([err_p, rot_weight * w])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -max_step, max_step), self.q_lo, self.q_hi)
        return q, float(np.linalg.norm(err_p)), float(np.linalg.norm(err_r))


@dataclass
class Plan:
    q: dict                     # side -> (T, 6) joint targets, one row per control tick
    goals: dict                 # side -> (pos, rot)
    status: str = "ok"          # ok | contact | blocked
    reasons: list = field(default_factory=list)
    final_q: dict = field(default_factory=dict)     # side -> (6,) for the preview ghost
    ik_err: dict = field(default_factory=dict)      # side -> (mm, deg)
    vias: dict = field(default_factory=dict)        # side -> [via positions] (routed plans)
    path: dict = field(default_factory=dict)        # side -> (n, 3) pinch positions along it
    hit_i: int | None = None                        # first tick with a new contact
    hit_pos: dict = field(default_factory=dict)     # side -> pinch position at hit_i

    @property
    def T(self) -> int:
        return len(next(iter(self.q.values())))


class Session:
    def __init__(self, env, recorder=None, speed: float = 1.0):
        self.env, self.model, self.data = env, env.model, env.data
        if env.hand != "robotiq":
            raise ValueError("keyframe teleop drives Robotiq grippers (hand='robotiq')")
        self.rec = recorder
        self.dt = env.n_substeps * self.model.opt.timestep
        self.speed = speed
        self.arms = {s: Arm(self.model, self.data, s) for s in SIDES}
        self.lo, self.hi = self.model.actuator_ctrlrange.T.copy()
        self._scratch = mujoco.MjData(self.model)
        self._ff_data = mujoco.MjData(self.model)
        self._ff_out = np.zeros(self.model.nv)
        self._classify_bodies()
        self.follow = False
        self.recording = False
        self.obs, self.info = None, env.task_status()
        self._reset_state()

    # ------------------------------------------------------------ setup
    def _classify_bodies(self):
        """Per geom: 'left'/'right' (arm), 'movable' (free props, doors, the pan),
        or 'static' (bench, balance housing and frame, gantry)."""
        m = self.model
        robot = set().union(*(a.bodies for a in self.arms.values()))
        pan = {m.body("balance_pan").id}
        self.geom_owner = []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            owner = "static"
            for s, a in self.arms.items():
                if b in a.bodies:
                    owner = s
            if owner == "static" and b not in robot:
                c = b
                while c > 0:
                    if m.body_jntnum[c] > 0 or c in pan:
                        owner = "movable"
                        break
                    c = int(m.body_parentid[c])
            self.geom_owner.append(owner)
        self.free_bodies = {int(m.jnt_bodyid[j]): int(m.jnt_qposadr[j]) for j in range(m.njnt)
                            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}

    def _reset_state(self):
        d = self.data
        # The joints where the arm actually rests: with gravity_ff added, commanding them
        # reproduces the current ctrl, so the first tick does not jump.
        self.q_cmd = {s: d.qpos[a.qadr].copy() for s, a in self.arms.items()}
        self.grip_cmd = {s: float(d.ctrl[a.eff_act[0]]) for s, a in self.arms.items()}
        self.targets = {s: list(a.tool_pose()) for s, a in self.arms.items()}
        self.home = {s: (p.copy(), R.copy()) for s, (p, R) in self.targets.items()}
        self.queue: list[dict] = []
        self.current: dict | None = None
        self.undo_stack: list[dict] = []
        self.segment = 0
        self._following = False
        self._follow_settle = 0
        self._plan_cache = None

    def reset(self, seed=None):
        self.obs, self.info = self.env.reset(seed=seed)
        self._reset_state()

    # ------------------------------------------------------------ targets
    def set_target(self, side: str, pos=None, rot=None):
        t = self.targets[side]
        if pos is not None:
            p = np.array(pos, float)
            p[:2] = np.clip(p[:2], *WORKSPACE_XY)
            p[2] = np.clip(p[2], *WORKSPACE_Z)
            t[0] = p
        if rot is not None:
            t[1] = orthonormalize(np.asarray(rot, float))
        self._plan_cache = None

    def point_down(self, side: str):
        """Turn the target to point straight down, keeping its position and yaw."""
        self.set_target(side, rot=nearest_top_down(self.targets[side][1]))

    def gap_below(self, side: str):
        """What is under the target, straight down: (gap m, name, hit point) from the pinch
        -- or from the base of a carried object -- to the first surface, or None."""
        m, d = self.model, self.data
        p = self.targets[side][0]
        carried = [c for c in self._carried() if c[0] == side]
        skip = set(self.arms[side].geoms) | set().union(*(a.geoms for a in self.arms.values()))
        drop = 0.0
        for _, b, _, rp, rR in carried:
            skip |= set(np.flatnonzero(m.body_rootid[m.geom_bodyid] == b).tolist())
            drop = max(drop, -(self.targets[side][1] @ rp)[2])   # the object's origin is its base
        start, geomid = p.astype(float).copy(), np.zeros(1, np.int32)
        travelled = 0.0
        for _ in range(8):
            dist = mujoco.mj_ray(m, d, start, np.array([0, 0, -1.0]), None, 1, -1, geomid)
            if dist < 0:
                return None
            g = int(geomid[0])
            if g in skip:                               # the arms themselves: look past them
                start = start - [0, 0, dist + 1e-4]
                travelled += dist + 1e-4
                continue
            hit = start - [0, 0, dist]
            return travelled + dist - drop, self._label(g), hit
        return None

    def target_to_current(self, side: str):
        self.set_target(side, *self.arms[side].tool_pose())

    def gripping(self, side: str) -> bool:
        return self.grip_cmd[side] > 0.5 * GRIPPER_CLOSED

    def snap(self, side: str, what: str) -> str:
        """Object-centric target presets; pressing again steps to the next stage.
        vial: above the vial (top-down) -> grasp height.  pan: above the pan -> place
        height (vial base PLACE_CLEAR above the pan).  door: above the top door's handle,
        fingers across it -> grasp height.  home: the episode start pose.
        The next stage comes once the gripper itself is at the first one (pressing twice
        before Enter doesn't skip ahead)."""
        _, R = self.targets[side]
        p = self.arms[side].tool_pose()[0]
        if what == "home":
            self.set_target(side, *self.home[side])
            return "home"
        rot = nearest_top_down(R)
        if what == "door":
            low = self.data.geom(DOOR_HANDLE).xpos + [0, 0, HANDLE_DZ]
            high = low + [0, 0, HANDLE_ABOVE]
            at_high = np.linalg.norm(p - high) < SNAP_AT
            self.set_target(side, low if at_high else high, top_down(np.pi / 2))
            return f"door handle {'grasp' if at_high else 'above'}"
        if what == "vial":
            base = self.data.body(TARGET_VIAL).xpos
            low = base + [0, 0, GRASP_Z]
        elif what == "pan":
            low = self.data.site("balance_pan_site").xpos + [0, 0, GRASP_Z + PLACE_CLEAR]
        else:
            raise ValueError(what)
        high = low + [0, 0, ABOVE]
        at_high = np.linalg.norm(p - high) < SNAP_AT
        self.set_target(side, low if at_high else high, rot)
        return f"{what} {'grasp' if what == 'vial' and at_high else 'place' if at_high else 'above'}"

    # ------------------------------------------------------------ planning
    def plan(self) -> Plan:
        if self._plan_cache is None:
            self._plan_cache = self._plan()
        return self._plan_cache

    def _start(self):
        """Scratch data at the current sim state, and the per-arm start configuration."""
        s = self._scratch
        s.qpos[:] = self.data.qpos
        q0 = {}
        for side, a in self.arms.items():
            # Start from the sim's actual joints (the servo lags its command a little).
            q0[side] = self.data.qpos[a.qadr].copy()
        return s, q0

    def _carried(self):
        """Free bodies a closed gripper is touching -> (side, body, qposadr, rel pos, rel rot)."""
        d, out = self.data, []
        for side, a in self.arms.items():
            if not self.gripping(side):
                continue
            p, R = a.tool_pose()
            seen = set()
            for c in d.contact[:d.ncon]:
                for g, h in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if self.model.geom_bodyid[g] in a.grip_bodies:
                        b = int(self.model.body_rootid[self.model.geom_bodyid[h]])
                        if b in self.free_bodies and b not in seen:
                            seen.add(b)
                            bp, bR = d.xpos[b], d.xmat[b].reshape(3, 3)
                            out.append((side, b, self.free_bodies[b], R.T @ (bp - p), R.T @ bR))
        return out

    def _contacts(self, s: mujoco.MjData, carried, tol: float = PEN_TOL) -> set[tuple]:
        """Offending contact pairs in scratch data `s`, penetrating deeper than `tol`:
        (kind, name_a, name_b)."""
        m = self.model
        # A carried body belongs to the arm carrying it.
        owner = list(self.geom_owner)
        for side, b, *_ in carried:
            for g in np.flatnonzero(m.geom_bodyid == b):
                owner[g] = side
        out = set()
        for c in s.contact[:s.ncon]:
            if c.dist > -tol:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            o1, o2 = owner[g1], owner[g2]
            if o1 not in SIDES and o2 not in SIDES:
                continue                                # props resting on things
            if o1 == o2:
                continue                                # an arm with itself or its load
            if o1 not in SIDES:
                g1, g2, o1, o2 = g2, g1, o2, o1          # the arm's side first
            kind = "contact" if o2 == "movable" else "blocked"   # static, or the other arm
            out.add((kind, self._label(g1, carried), self._label(g2, carried)))
        return out

    def _label(self, g: int, carried=()) -> str:
        """'left gripper', 'right arm', 'vial_a', 'balance_door_left', ... (body level)."""
        m = self.model
        b = int(m.geom_bodyid[g])
        for side, a in self.arms.items():
            if b in a.bodies:
                return f"{side} {'gripper' if b in a.grip_bodies else 'arm'}"
        if b == 0:
            return "table"                              # the bench is part of the world body
        root = int(m.body_rootid[b])
        if any(root == cb for _, cb, *_ in carried) or root in self.free_bodies:
            return m.body(root).name
        return m.body(b).name.replace("_", " ")

    def _place_carried(self, s, carried):
        for side, b, adr, rp, rR in carried:
            p, R = self.arms[side].tool_pose(s)
            s.qpos[adr:adr + 3] = p + R @ rp
            s.qpos[adr + 3:adr + 7] = mat2quat(R @ rR)

    def _plan(self) -> Plan:
        s, q0 = self._start()
        starts = {side: a.tool_pose() for side, a in self.arms.items()}
        plan = self._route(s, q0, starts, {side: [] for side in self.arms})
        if plan.status == "ok":
            return plan
        # Up, across, down: for each arm that moves sideways, climb above the higher end
        # (plus a lift), traverse, then descend onto the target.
        rank = {"ok": 0, "contact": 1, "blocked": 2}
        for lift in ROUTE_LIFTS:
            vias = {side: self._vias(starts[side], self.targets[side], lift) for side in self.arms}
            if not any(vias.values()):
                break
            alt = self._route(s, q0, starts, vias)
            if rank[alt.status] < rank[plan.status]:
                plan = alt
            if plan.status == "ok":
                break
        return plan

    @staticmethod
    def _vias(start, target, lift):
        (p0, _), (p1, _) = start, target
        if np.linalg.norm((p1 - p0)[:2]) < ROUTE_MIN_XY:
            return []
        z = min(max(p0[2], p1[2]) + lift, WORKSPACE_Z[1])
        out = []
        if z - p0[2] > 0.005:
            out.append(np.array([p0[0], p0[1], z]))
        if z - p1[2] > 0.005:
            out.append(np.array([p1[0], p1[1], z]))
        return out

    def _route(self, s, q0, starts, vias) -> Plan:
        """Plan every arm through its via points (positions; the rotation turns on the
        legs across) to its target, then collision-check the whole thing."""
        plan = Plan(q={}, goals={side: (t[0].copy(), t[1].copy()) for side, t in self.targets.items()},
                    vias={side: [v.copy() for v in vs] for side, vs in vias.items() if vs})
        for side, a in self.arms.items():
            p0, R0 = starts[side]
            p1, R1 = self.targets[side]
            pts = [p0] + vias[side] + [p1]
            # Turn on the across leg (or spread over the one straight leg).
            rots = [R0] * (len(pts) - 1) + [R1]
            if len(pts) == 4:
                rots[2] = R1
            elif len(pts) == 3:
                rots[1] = R0 if np.linalg.norm((pts[1] - pts[0])[:2]) < 1e-6 else R1
            s.qpos[:] = self.data.qpos
            qi, legs = q0[side], []
            for k in range(len(pts) - 1):
                dur = max(ROUTE_LEG_MIN_S if vias[side] else MIN_MOVE_S,
                          np.linalg.norm(pts[k + 1] - pts[k]) / LIN_SPEED,
                          rot_angle(rots[k], rots[k + 1]) / ANG_SPEED) / self.speed
                for _ in range(3):      # stretch the leg if a joint would go too fast
                    leg, ep, er = self._leg(a, s, qi, (pts[k], rots[k]), (pts[k + 1], rots[k + 1]), dur)
                    vmax = np.abs(np.diff(np.vstack([qi, leg]), axis=0)).max() / self.dt
                    if vmax <= MAX_JOINT_SPEED:
                        break
                    dur *= min(4.0, 1.05 * vmax / MAX_JOINT_SPEED)
                else:
                    plan.status = "blocked"
                    plan.reasons.append(f"{side}: IK jumps ({vmax:.1f} rad/s), a singularity or a wrist flip")
                legs.append(leg)
                qi = leg[-1]
            plan.q[side] = np.vstack(legs)
            plan.final_q[side] = qi
            plan.ik_err[side] = (1e3 * ep, np.degrees(er))
            if 1e3 * ep > IK_TOL_MM or np.degrees(er) > IK_TOL_DEG:
                plan.status = "blocked"
                plan.reasons.append(f"{side} can't reach: off by {1e3 * ep:.0f} mm, {np.degrees(er):.0f} deg")
        T = max(len(q) for q in plan.q.values())        # arms that finish first hold still
        for side, q in plan.q.items():
            if len(q) < T:
                plan.q[side] = np.vstack([q, np.repeat(q[-1:], T - len(q), axis=0)])
        self._check(plan, s, q0)
        return plan

    def _leg(self, a, s, qi, start, end, dur):
        (p0, R0), (p1, R1) = start, end
        n = max(2, int(round(dur / self.dt)) + 1)
        sched = min_jerk(n)[1:]
        out = np.zeros((n - 1, 6))
        for i, u in enumerate(sched):
            last = i == len(sched) - 1
            qi, ep, er = a.ik(s, p0 + u * (p1 - p0), slerp_mat(R0, R1, u), qi, iters=60 if last else 8)
            out[i] = qi
        return out, ep, er

    def _check(self, plan: Plan, s, q0):
        carried = self._carried()
        m = self.model
        s.qpos[:] = self.data.qpos
        mujoco.mj_kinematics(m, s)
        mujoco.mj_collision(m, s)
        # Every pair already touching at the start, however lightly, is not news.
        before = {(a, b) for _, a, b in self._contacts(s, carried, tol=-np.inf)}
        found = set()
        idx = list(range(CHECK_EVERY - 1, plan.T, CHECK_EVERY))
        if not idx or idx[-1] != plan.T - 1:
            idx.append(plan.T - 1)
        path = {side: [] for side in self.arms}
        for i in idx:
            for side, a in self.arms.items():
                s.qpos[a.qadr] = plan.q[side][i]
            mujoco.mj_kinematics(m, s)
            for side, a in self.arms.items():
                path[side].append(a.tool_pose(s)[0])
            self._place_carried(s, carried)
            mujoco.mj_kinematics(m, s)
            mujoco.mj_collision(m, s)
            new = {c for c in self._contacts(s, carried) if (c[1], c[2]) not in before}
            if new and plan.hit_i is None:
                plan.hit_i = i
                plan.hit_pos = {side: p[-1].copy() for side, p in path.items()}
            found |= new
        plan.path = {side: np.array(p) for side, p in path.items()}
        blocked = sorted({(a, b) for k, a, b in found if k == "blocked"})
        touch = sorted({(a, b) for k, a, b in found if k == "contact"})
        if blocked:
            plan.status = "blocked"
            plan.reasons += [f"{a} hits {b}" for a, b in blocked[:2]]
        elif touch and plan.status == "ok":
            plan.status = "contact"
            plan.reasons += [f"{a} touches {b}" for a, b in touch[:2]]

    # ------------------------------------------------------------ commands
    @property
    def busy(self) -> bool:
        return self.current is not None or bool(self.queue)

    def execute(self, force: bool = False) -> tuple[bool, str]:
        if self.busy:
            return False, "busy"
        plan = self.plan()
        if plan.status == "blocked" and not force:
            return False, "blocked: " + "; ".join(plan.reasons)
        self.queue.append({"kind": "move", "plan": plan})
        self._plan_cache = None
        return True, f"moving ({plan.T * self.dt:.1f} s)"

    def grip(self, side: str, close: bool | None = None) -> tuple[bool, str]:
        if self.busy:
            return False, "busy"
        close = (not self.gripping(side)) if close is None else close
        self.queue.append({"kind": "grip", "side": side,
                           "to": GRIPPER_CLOSED if close else GRIPPER_OPEN})
        return True, f"{side} {'close' if close else 'open'}"

    # ------------------------------------------------------------ undo
    def _checkpoint(self):
        d = self.data
        self.undo_stack.append({
            "qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "act": d.act.copy(),
            "ctrl": d.ctrl.copy(), "time": d.time, "warm": d.qacc_warmstart.copy(),
            "q_cmd": {k: v.copy() for k, v in self.q_cmd.items()}, "grip": dict(self.grip_cmd),
            "targets": {k: [v[0].copy(), v[1].copy()] for k, v in self.targets.items()},
            "rec": len(self.rec) if (self.rec is not None and self.recording) else None,
            "steps": self.env._step_count, "segment": self.segment, "obs": self.obs,
            "info": self.info})
        del self.undo_stack[:-MAX_UNDO]

    def undo(self) -> tuple[bool, str]:
        if self.busy:
            return False, "busy"
        if not self.undo_stack:
            return False, "nothing to undo"
        c = self.undo_stack.pop()
        d = self.data
        d.qpos[:], d.qvel[:], d.act[:], d.ctrl[:] = c["qpos"], c["qvel"], c["act"], c["ctrl"]
        d.time, d.qacc_warmstart[:] = c["time"], c["warm"]
        mujoco.mj_forward(self.model, d)
        self.q_cmd, self.grip_cmd, self.targets = c["q_cmd"], c["grip"], c["targets"]

        self.env._step_count, self.segment = c["steps"], c["segment"]
        self.obs, self.info = c["obs"], c["info"]
        if self.recording and c["rec"] is not None:
            self.rec.truncate(c["rec"])
        self._following = False
        self._plan_cache = None
        return True, "undone"

    # ------------------------------------------------------------ episodes
    def begin_episode(self, seed=None):
        self.reset(seed)
        if self.rec is not None:
            self.rec.begin(self._observe())
        self.recording = self.rec is not None

    def save_episode(self) -> int:
        self.recording = False
        self.undo_stack.clear()
        return self.rec.save(success=bool(self.info["success"]))

    def discard_episode(self):
        self.recording = False
        self.undo_stack.clear()
        self.rec.discard()

    def _observe(self):
        return observe(self.env, self.arms, self.obs["image"] if self.obs else
                       self.env.render_cameras(["top1"])["top1"])

    # ------------------------------------------------------------ tick
    def tick(self) -> bool:
        """One control tick if anything is moving; returns whether the sim stepped."""
        if self.current is None and self.queue:
            self._checkpoint()
            self.segment += 1
            self.current = dict(self.queue.pop(0), i=0, settle=0)
        cur = self.current
        if cur is not None:
            kind = cur["kind"]
            if kind == "move":
                self._tick_move(cur)
            else:
                self._tick_grip(cur)
        elif self.follow:
            if not self._tick_follow():
                return False
            kind = "follow"
        else:
            return False
        self._step(kind)
        return True

    def _tick_move(self, cur):
        plan = cur["plan"]
        if cur["i"] < plan.T:
            for side in self.arms:
                self.q_cmd[side] = plan.q[side][cur["i"]]
            cur["i"] += 1
            return
        cur["settle"] += 1
        if self._settled() or cur["settle"] * self.dt >= SETTLE_MAX_S:
            self.current = None

    def _tick_grip(self, cur):
        side = cur["side"]
        if cur["i"] == 0:
            cur["from"] = self.grip_cmd[side]
        n = max(1, int(round(GRIP_RAMP_S / self.dt)))
        cur["i"] += 1
        u = min(1.0, cur["i"] / n)
        self.grip_cmd[side] = cur["from"] + u * (cur["to"] - cur["from"])
        if u >= 1.0:
            v = abs(self.data.qvel[self.arms[side].finger_dof])
            if (cur["i"] > n + 2 and v < 0.05) or (cur["i"] - n) * self.dt >= GRIP_SETTLE_MAX_S:
                self.current = None

    def _settled(self) -> bool:
        d = self.data
        return all(np.abs(d.qvel[a.dofs]).max() < SETTLE_QVEL
                   and np.abs(d.qpos[a.qadr] - self.q_cmd[s]).max() < SETTLE_QERR
                   for s, a in self.arms.items())

    def _tick_follow(self) -> bool:
        s = self._scratch
        s.qpos[:] = self.data.qpos
        moving = False
        for side, a in self.arms.items():
            s.qpos[a.qadr] = self.q_cmd[side]   # slew from the commanded pose, not the lagging actual one
            mujoco.mj_kinematics(self.model, s)
            p, R = a.tool_pose(s)
            pt, Rt = self.targets[side]
            e, ang = pt - p, rot_angle(R, Rt)
            if np.linalg.norm(e) < 1e-3 and ang < np.deg2rad(0.5):
                continue
            moving = True
            n = np.linalg.norm(e)
            p_next = p + e * min(1.0, FOLLOW_SPEED * self.speed * self.dt / max(n, 1e-9))
            R_next = slerp_mat(R, Rt, min(1.0, FOLLOW_ANG_SPEED * self.speed * self.dt / max(ang, 1e-9)))
            q, _, _ = a.ik(s, p_next, R_next, self.q_cmd[side], iters=15)
            lim = MAX_JOINT_SPEED * self.dt
            self.q_cmd[side] = self.q_cmd[side] + np.clip(q - self.q_cmd[side], -lim, lim)
            s.qpos[a.qadr] = self.q_cmd[side]
        if moving:
            self._follow_settle = 0
        else:
            self._follow_settle += 1
            if self._settled() or self._follow_settle * self.dt >= SETTLE_MAX_S:
                self._following = False
                return False
        if not self._following:             # a new follow burst: one undo step, one segment
            self._checkpoint()
            self.segment += 1
            self._following = True
        return True

    def gravity_ff(self) -> dict:
        """Per arm, the servo-target offset that holds q_cmd against gravity:
        static gravity torque at q_cmd / servo kp. Without it the arm sags ~10 mrad
        per joint (~9 mm at the tool) below every keyframe."""
        m, f = self.model, self._ff_data
        f.qpos[:] = self.data.qpos
        f.qvel[:] = 0
        for side, a in self.arms.items():
            f.qpos[a.qadr] = self.q_cmd[side]
        mujoco.mj_kinematics(m, f)
        mujoco.mj_comPos(m, f)
        mujoco.mj_rne(m, f, 0, self._ff_out)
        tau = {side: self._ff_out[a.dofs].copy() for side, a in self.arms.items()}
        # A held object is a free body, outside the arm's chain: add its weight at its CoM.
        jac = np.zeros((3, m.nv))
        for side, b, *_ in self._carried():
            mujoco.mj_jac(m, f, jac, None, self.data.xipos[b], self.arms[side].body)
            tau[side] += jac[:, self.arms[side].dofs].T @ (-m.opt.gravity * m.body_mass[b])
        return {side: tau[side] / m.actuator_gainprm[a.arm_act, 0] for side, a in self.arms.items()}

    def _step(self, kind: str):
        d = self.data
        state = np.concatenate([[d.time], d.qpos, d.qvel])
        ctrl = d.ctrl.copy()
        ff = self.gravity_ff()
        for side, a in self.arms.items():
            ctrl[a.arm_act] = self.q_cmd[side] + ff[side]
            ctrl[a.eff_act] = self.grip_cmd[side]
        ctrl = np.clip(ctrl, self.lo, self.hi)
        self.obs, _, _, _, self.info = self.env.step(ctrl)
        self._plan_cache = None             # the start state moved
        if not self.recording:
            return
        goal_p = np.stack([self.targets[s][0] for s in SIDES]).astype(np.float32)
        if self.current is not None and self.current["kind"] == "move":
            goal_p = np.stack([self.current["plan"].goals[s][0] for s in SIDES]).astype(np.float32)
            goal_q = np.stack([mat2quat(self.current["plan"].goals[s][1]) for s in SIDES])
        else:
            goal_q = np.stack([mat2quat(self.targets[s][1]) for s in SIDES])
        op = {"goal_pos": goal_p, "goal_quat": goal_q.astype(np.float32),
              "segment": np.int32(self.segment), "kind": np.uint8(KIND[kind])}
        action = 2.0 * (ctrl - self.lo) / (self.hi - self.lo) - 1.0
        self.rec.add(state, action, ctrl, float(self.info["success"]), self._observe(), op)
