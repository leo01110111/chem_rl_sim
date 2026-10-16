"""CONTRACT
Gymnasium env for the vial-weighing bench (video chem_lab_vial_weighing_trial1,
2:50-4:10) on the gantry UR7e robot from the urgantry_sim submodule, with Robotiq
2F-85 grippers (default, hand="robotiq") or 5-finger hands (hand="wuji"|"sharpa").
Interface deliberately the same as urgantry_sim.env.SimGantryUR7eEnv (MaxLab) --
same methods, obs keys and action layout; with hand="wuji" the shapes match it
exactly (52) -- so OGPO (in-process) and OpenPi (websocket client via
get_openpi_observation) can use it unchanged:
  Action  (nu,) actuator targets in actuator order: left arm 6, left effector H,
          right arm 6, right effector H (H = 1 robotiq / 20 wuji / 22 sharpa;
          nu = 14 / 52 / 56). Arm and hand targets in radians; the Robotiq target
          is 0 = open .. 255 = closed. normalized_actions=True -> Box(-1, 1)
          mapped onto ctrlrange.
  Obs     {'state': (nu,) float32 actuated joint positions (rad), one per
           actuator; a Robotiq's slot is its right_driver_joint (the first joint
           of the tendon it drives), 0 = open .. 0.8 = closed,
           'image': (S, S, 3) uint8 RGB from camera 'top1'}
          state_mode="proprio+task" appends 4 floats to 'state':
          [balance reading in grams, left, right, top door opening in meters].
  Info    success (0/1), balance_g (float, grams on the pan), balance_display
          (str, 4 decimals like the ME204), doors (3,) meters, vial_pos (3,) world m,
          vial_on_pan, doors_closed, vial_upright.
Task (from the video): put vial_a (an open 20 mL vial, 20 g) onto the balance pan
  and close every draft-shield door. Doors start closed, as filmed; a door must be
  opened (slid back) to get the vial in. success = vial_on_pan (vial center within
  PAN_TOL of the pan center in xy, resting: |vial z - pan top| < 5 mm) and
  vial_upright (axis . +z > 0.95) and doors_closed (all < 5 mm) and the reading
  is within READING_TOL_G of the vial's 20 g. Terminates on success.
Reward: 1.0 on success, else 0.1 * shaping in [0, 1] (reach from the nearer
  grasp point -- Robotiq pinch site or palm -- to the vial, vial
  height, vial-to-pan distance); only for logging -- the env is meant for eval
  and demonstration-driven training, like MaxLab's.
World frame: urgantry_sim's (see chem_rl_sim/scene.py). 2 ms physics,
  implicitfast; control_hz sets substeps.
Reference output: tests/test_env.py (random-action episode, shapes, a scripted
  placement that must report success).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import mujoco
import mujoco.viewer

from urgantry_sim.build_urgantry import apply_initial_view

from .balance import PAN_R, pan_reading_g
from .robot import DEFAULT_EFFECTOR, tool_point
from .scene import LAYOUTS, TARGET_VIAL, build_model, set_initial_pose

DOORS = ("left", "right", "top")
VIAL_MASS_G = 20.0
PAN_TOL = PAN_R - 0.010
DOOR_CLOSED_TOL = 0.005
READING_TOL_G = 1.0
RANDOM_XY = 0.015          # per-episode jitter of bench props (m), task layout only
SETTLE_STEPS = 50

PROMPT = ("weigh the vial: open the draft shield, put the open vial on the balance pan, "
          "and close the draft shield")


def _driven_joint(m: mujoco.MjModel, act: int) -> int:
    """Joint an actuator moves: its target joint, or for a tendon actuator (the
    Robotiq) the tendon's first joint."""
    if m.actuator_trntype[act] == mujoco.mjtTrn.mjTRN_TENDON:
        return int(m.wrap_objid[m.tendon_adr[m.actuator_trnid[act, 0]]])
    return int(m.actuator_trnid[act, 0])


class VialWeighingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        image_size: int = 224,
        control_hz: float = 20.0,
        max_episode_steps: int = 600,
        normalized_actions: bool = False,
        prompt: str = PROMPT,
        show_viewer: bool = False,
        hand: str = DEFAULT_EFFECTOR,
        layout: str = "task",
        clutter: bool = True,
        randomize: bool = True,
        state_mode: str = "proprio",
        start_door_open: str | None = None,
    ):
        super().__init__()
        if state_mode not in ("proprio", "proprio+task"):
            raise ValueError(f"state_mode must be 'proprio' or 'proprio+task', got {state_mode!r}")
        self.layout, self.hand, self.prompt = layout, hand, prompt
        self.randomize, self.state_mode = randomize, state_mode
        self.start_door_open = start_door_open
        self.model = build_model(layout, hand, clutter)
        self.data = mujoco.MjData(self.model)
        self.image_size = image_size
        self.normalized_actions = normalized_actions
        self.max_episode_steps = max_episode_steps
        self.n_substeps = max(1, round((1.0 / control_hz) / self.model.opt.timestep))
        self._step_count = 0
        self._renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)
        self.cameras = {"top1": "top1"}

        m = self.model
        self._ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = m.actuator_ctrlrange[:, 1].copy()
        self._act_qadr = m.jnt_qposadr[[_driven_joint(m, i) for i in range(m.nu)]]
        self._door_qadr = np.array([m.joint(f"balance_door_{d}").qposadr[0] for d in DOORS])
        self._vial_qadr = m.joint(f"{TARGET_VIAL}_joint").qposadr[0]
        self._prop_qadr = {p: m.joint(f"{p}_joint").qposadr[0] for p in LAYOUTS[layout]}

        nu = m.nu
        if normalized_actions:
            self.action_space = Box(-1.0, 1.0, shape=(nu,), dtype=np.float32)
        else:
            self.action_space = Box(self._ctrl_low.astype(np.float32),
                                    self._ctrl_high.astype(np.float32), shape=(nu,), dtype=np.float32)
        n_state = nu + (4 if state_mode == "proprio+task" else 0)
        self.observation_space = Dict({
            "state": Box(-np.inf, np.inf, shape=(n_state,), dtype=np.float32),
            "image": Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8),
        })

        self._viewer = None
        if show_viewer:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data,
                                                        show_left_ui=False, show_right_ui=False)
            apply_initial_view(self._viewer)
            self._viewer.sync()

    # ------------------------------------------------------------------ readouts
    def balance_g(self) -> float:
        return pan_reading_g(self.model, self.data)

    def doors(self) -> np.ndarray:
        return self.data.qpos[self._door_qadr].copy()

    def vial_pos(self) -> np.ndarray:
        return self.data.body(TARGET_VIAL).xpos.copy()

    def _vial_axis(self) -> np.ndarray:
        return self.data.body(TARGET_VIAL).xmat.reshape(3, 3)[:, 2]

    def task_status(self) -> dict:
        pan = self.data.site("balance_pan_site").xpos
        v = self.vial_pos()
        reading = self.balance_g()
        on_pan = bool(np.linalg.norm(v[:2] - pan[:2]) < PAN_TOL and abs(v[2] - pan[2]) < 0.005)
        upright = bool(self._vial_axis()[2] > 0.95)
        closed = bool(np.all(self.doors() < DOOR_CLOSED_TOL))
        reading_ok = abs(reading - VIAL_MASS_G) < READING_TOL_G
        return {"vial_on_pan": on_pan, "vial_upright": upright, "doors_closed": closed,
                "success": int(on_pan and upright and closed and reading_ok),
                "balance_g": reading, "balance_display": f"{reading:.4f} g",
                "doors": self.doors(), "vial_pos": v}

    # ------------------------------------------------------------------ obs
    def render_cameras(self, roles=None) -> dict[str, np.ndarray]:
        roles = roles if roles is not None else self.cameras.keys()
        frames = {}
        for role in roles:
            self._renderer.update_scene(self.data, camera=self.cameras[role])
            frames[role] = self._renderer.render()
        return frames

    def _proprio(self) -> np.ndarray:
        return self.data.qpos[self._act_qadr].astype(np.float32)

    def _state(self) -> np.ndarray:
        s = self._proprio()
        if self.state_mode == "proprio+task":
            s = np.concatenate([s, [self.balance_g()], self.doors()]).astype(np.float32)
        return s

    def _get_obs(self) -> dict:
        return {"state": self._state(), "image": self.render_cameras(["top1"])["top1"]}

    def get_openpi_observation(self, prompt: str | None = None) -> dict:
        frames = self.render_cameras()
        return {"observation/image": frames["top1"], "observation/state": self._state(),
                "prompt": prompt if prompt is not None else self.prompt}

    # ------------------------------------------------------------------ reward
    def _shaping(self) -> float:
        v = self.vial_pos()
        pan = self.data.site("balance_pan_site").xpos
        reach = min(np.linalg.norm(tool_point(self.model, self.data, s) - v)
                    for s in ("left", "right"))
        lift = np.clip((v[2] - LAYOUTS[self.layout][TARGET_VIAL][1][2]) / 0.15, 0, 1)
        to_pan = np.linalg.norm(v - pan)
        return float((1 - np.tanh(4 * reach)) / 3 + lift / 3 + (1 - np.tanh(4 * to_pan)) / 3)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        set_initial_pose(self.model, self.data, self.layout)
        if self.randomize and self.layout == "task":
            for adr in self._prop_qadr.values():
                self.data.qpos[adr:adr + 2] += self.np_random.uniform(-RANDOM_XY, RANDOM_XY, 2)
        if self.start_door_open:
            j = self.model.joint(f"balance_door_{self.start_door_open}")
            self.data.qpos[j.qposadr[0]] = j.range[1]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)
        self._step_count = 0
        self._sync_viewer()
        return self._get_obs(), self.task_status()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(self.model.nu)
        if self.normalized_actions:
            a = np.clip(action, -1.0, 1.0)
            ctrl = self._ctrl_low + (a + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)
        else:
            ctrl = np.clip(action, self._ctrl_low, self._ctrl_high)
        self.data.ctrl[:] = ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        self._sync_viewer()
        info = self.task_status()
        reward = 1.0 if info["success"] else 0.1 * self._shaping()
        terminated = bool(info["success"])
        truncated = self._step_count >= self.max_episode_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _sync_viewer(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self):
        self._renderer.close()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
