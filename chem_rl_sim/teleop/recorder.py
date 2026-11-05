"""CONTRACT
Demonstration recorder in OGPO's dataset format, which is robomimic's HDF5 layout
(read by OGPO envs/robomimic_utils.get_dataset). One file, demos appended:
  data.attrs: total (int, all samples), env_args (JSON: env_name, type, env_kwargs),
              control_hz, effector
  data/demo_<k>/   (k counts up from the largest existing index; T = samples)
    actions     (T, nu) float32  normalized [-1, 1] (env normalized_actions mapping) --
                OGPO's action space
    actions_abs (T, nu) float32  the same targets in ctrl units (rad, Robotiq 0..255)
    rewards     (T,) float32     sparse: 1 on steps where the env reports success
    dones       (T,) uint8       = rewards (the env terminates on success)
    states      (T, 1+nq+nv)     [time, qpos, qvel] before each action, for replay
    obs/<key>, next_obs/<key>    (T, ...) -- next_obs[t] = obs[t+1]; the last one is the
                                  observation after the final action
      top1_image          (T, S, S, 3) uint8   the env's policy image
      robot0_eye_in_hand_image, robot1_eye_in_hand_image  (T, S, S, 3) uint8  left / right
                          wrist cameras (Robotiq only; robomimic's names for wrist views)
      joint_pos           (T, nu)              the env's proprio `state`
      robot0_eef_pos/_quat, robot0_gripper_qpos   left tool (Robotiq pinch site / Wuji
      robot1_eef_pos/_quat, robot1_gripper_qpos   palm) world pose (quat wxyz) and the
                                                  effector's slice of joint_pos
      object              (T, 11)  vial pos (3) + quat wxyz (4), balance g (1), doors m (3)
    operator/  the human side; keys come from the collector's add(operator=...):
      hand teleop (app.py), to re-retarget later without re-running WiLoR:
        keypoints_cam (T, 2, 21, 3) [left, right], NaN where that hand was unseen;
        keypoints_wrist (same); engaged (T, 2) bool; perception_seq (T,);
        perception_age_ms (T,) camera capture -> action applied
      keyframe teleop (keyframe.py): goal_pos (T, 2, 3), goal_quat (T, 2, 4),
        segment (T,) int32, kind (T,) uint8 (0 move, 1 grip, 2 follow)
    attrs: num_samples, success (0/1), date, duration_s
Suggested OGPO TASK_CONFIG entry (envs/robomimic_utils.py):
  'chem_vial': {'image_keys': ['top1_image'],
                'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                                 'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos'],
                'low_dim_keys': [<the six above>, 'object'], 'image_size': (S, S)}
Saving runs on a background thread (gzip of images takes ~1 s per minute of demo);
`wait()` joins it.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path

import numpy as np
import mujoco

from ..scene import TARGET_VIAL

IMAGE_KEY = "top1_image"
WRIST_KEYS = {"left_wrist": "robot0_eye_in_hand_image", "right_wrist": "robot1_eye_in_hand_image"}


def observe(env, arms: dict, image: np.ndarray) -> dict:
    """OGPO obs dict for the current sim state. `arms` = {'left': ArmTeleop, 'right': ...}."""
    m, d = env.model, env.data
    joint = env._proprio()
    o = {IMAGE_KEY: image, "joint_pos": joint}
    wrists = [r for r in WRIST_KEYS if r in env.cameras]
    for role, img in env.render_cameras(wrists).items():
        o[WRIST_KEYS[role]] = img
    for i, side in enumerate(("left", "right")):
        a = arms[side]
        p, R = a.tool_pose()
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        o[f"robot{i}_eef_pos"] = p
        o[f"robot{i}_eef_quat"] = q
        o[f"robot{i}_gripper_qpos"] = joint[a.eff_act]
    vb = d.body(TARGET_VIAL)
    o["object"] = np.concatenate([vb.xpos, vb.xquat, [env.balance_g()], env.doors()])
    return {k: (v if k.endswith("_image") else np.asarray(v, np.float32)) for k, v in o.items()}


class Recorder:
    def __init__(self, path: str | Path, env_args: dict, control_hz: float, effector: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.env_args, self.control_hz, self.effector = env_args, control_hz, effector
        self._thread: threading.Thread | None = None
        self.last_saved: str | None = None
        self.last_error: str | None = None
        self.active = False
        self._clear()

    def _clear(self):
        self.obs, self.actions, self.actions_abs, self.rewards, self.states = [], [], [], [], []
        self.op = {}

    def __len__(self):
        return len(self.actions)

    # ---------------------------------------------------------------- episode
    def begin(self, first_obs: dict) -> None:
        self.wait()
        self._clear()
        self.obs.append(first_obs)
        self.active = True

    def add(self, state, action, action_abs, reward, next_obs, operator: dict) -> None:
        """One control tick: the state/obs before `action` is already in self.obs."""
        self.states.append(state)
        self.actions.append(np.asarray(action, np.float32))
        self.actions_abs.append(np.asarray(action_abs, np.float32))
        self.rewards.append(float(reward))
        self.obs.append(next_obs)
        for k, v in operator.items():
            self.op.setdefault(k, []).append(v)

    def truncate(self, n: int) -> None:
        """Drop every sample after the first n (undo)."""
        del self.states[n:], self.actions[n:], self.actions_abs[n:], self.rewards[n:]
        del self.obs[n + 1:]
        for v in self.op.values():
            del v[n:]

    def discard(self) -> None:
        self.active = False
        self._clear()

    def save(self, success: bool) -> int:
        """Write the episode in the background; returns its sample count (0 = nothing)."""
        self.active = False
        n = len(self)
        if n == 0:
            return 0
        self.wait()
        payload = (self.obs, self.actions, self.actions_abs, self.rewards, self.states,
                   self.op, bool(success))
        self._clear()
        self._thread = threading.Thread(target=self._write, args=payload, daemon=False)
        self._thread.start()
        return n

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    @property
    def saving(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---------------------------------------------------------------- file
    def num_demos(self) -> int:
        if not self.path.exists():
            return 0
        import h5py
        with h5py.File(self.path, "r") as f:
            return len(f["data"]) if "data" in f else 0

    def delete_last(self) -> str | None:
        import h5py
        self.wait()
        if not self.path.exists():
            return None
        with h5py.File(self.path, "a") as f:
            demos = sorted(f["data"], key=lambda k: int(k[5:]))
            if not demos:
                return None
            name = demos[-1]
            f["data"].attrs["total"] = int(f["data"].attrs["total"]) - int(f[f"data/{name}"].attrs["num_samples"])
            del f[f"data/{name}"]
        return name     # HDF5 does not shrink the file; h5repack reclaims the space

    def _write(self, obs, actions, actions_abs, rewards, states, op, success):
        import h5py
        try:
            T = len(actions)
            r = np.asarray(rewards, np.float32)
            with h5py.File(self.path, "a") as f:
                data = f.require_group("data")
                if "env_args" not in data.attrs:
                    data.attrs["env_args"] = json.dumps(self.env_args)
                    data.attrs["control_hz"] = self.control_hz
                    data.attrs["effector"] = self.effector
                    data.attrs["total"] = 0
                k = 1 + max((int(n[5:]) for n in data), default=-1)
                g = data.create_group(f"demo_{k}")
                g.create_dataset("actions", data=np.stack(actions))
                g.create_dataset("actions_abs", data=np.stack(actions_abs))
                g.create_dataset("rewards", data=r)
                g.create_dataset("dones", data=(r > 0).astype(np.uint8))
                g.create_dataset("states", data=np.stack(states))
                for name, rows in (("obs", obs[:-1]), ("next_obs", obs[1:])):
                    og = g.create_group(name)
                    for key in obs[0]:
                        arr = np.stack([o[key] for o in rows])
                        if key.endswith("_image"):
                            og.create_dataset(key, data=arr, compression="gzip", compression_opts=1,
                                              chunks=(1, *arr.shape[1:]))
                        else:
                            og.create_dataset(key, data=arr)
                opg = g.create_group("operator")
                for key, rows in op.items():
                    opg.create_dataset(key, data=np.stack(rows))
                g.attrs["num_samples"] = T
                g.attrs["success"] = int(success)
                g.attrs["date"] = dt.datetime.now().isoformat(timespec="seconds")
                g.attrs["duration_s"] = T / self.control_hz
                data.attrs["total"] = int(data.attrs["total"]) + T
            self.last_saved = f"demo_{k}"
        except Exception as e:
            self.last_error = f"save failed: {e!r}"
