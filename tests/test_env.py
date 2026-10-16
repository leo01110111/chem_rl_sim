"""CONTRACT
Env interface + task logic; run: uv run pytest tests/ -q
  - gym.make("ChemVialWeighing-v0") with the default Robotiq grippers: state (14,),
    image (224,224,3) uint8, action (14,); hand="wuji" gives MaxLab
    SimGantryUR7e-v0's (52,); state_mode "proprio+task" adds 4
  - closing a Robotiq moves its state slot from ~0 to ~0.8 rad (the driver joint,
    not some arm joint)
  - 20 random normalized actions: finite obs, no exception
  - success logic: vial teleported onto the pan with doors closed -> success 1 and
    the reading ~20 g; same with the top door open -> success 0
"""
import gymnasium as gym
import mujoco
import numpy as np

import chem_rl_sim  # noqa: F401
from chem_rl_sim.scene import TARGET_VIAL


def test_spaces_and_random_steps():
    env = gym.make("ChemVialWeighing-v0", normalized_actions=True)
    obs, info = env.reset(seed=0)
    assert obs["state"].shape == (14,) and obs["image"].shape == (224, 224, 3)
    assert obs["image"].dtype == np.uint8
    assert env.action_space.shape == (14,)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs["state"]))
    env.close()
    env = gym.make("ChemVialWeighing-v0", state_mode="proprio+task")
    obs, _ = env.reset(seed=0)
    assert obs["state"].shape == (18,)
    env.close()
    env = gym.make("ChemVialWeighing-v0", hand="wuji")
    obs, _ = env.reset(seed=0)
    assert obs["state"].shape == (52,) and env.action_space.shape == (52,)
    env.close()


def test_gripper_state_slot():
    env = gym.make("ChemVialWeighing-v0")
    obs, _ = env.reset(seed=0)
    m = env.unwrapped.model
    i = m.actuator("left_grip_fingers_actuator").id
    assert abs(obs["state"][i]) < 0.02
    act = env.unwrapped.data.ctrl.copy()
    act[i] = 255.0
    for _ in range(20):
        obs, *_ = env.step(act)
    assert 0.7 < obs["state"][i] < 0.81, obs["state"][i]
    env.close()


def _place_vial_on_pan(env, top_open: bool):
    u = env.unwrapped
    m, d = u.model, u.data
    pan = d.site("balance_pan_site").xpos.copy()
    adr = m.joint(f"{TARGET_VIAL}_joint").qposadr[0]
    d.qpos[adr:adr + 7] = [*pan, 1, 0, 0, 0]
    d.qvel[m.joint(f"{TARGET_VIAL}_joint").dofadr[0]:][:6] = 0
    if top_open:
        d.qpos[m.joint("balance_door_top").qposadr[0]] = 0.2
    mujoco.mj_forward(m, d)
    hold = d.ctrl.copy()
    for _ in range(15):
        _, _, term, _, info = env.step(hold)
    return term, info


def test_success_logic():
    env = gym.make("ChemVialWeighing-v0")
    env.reset(seed=1)
    term, info = _place_vial_on_pan(env, top_open=False)
    assert info["success"] == 1 and term
    assert abs(info["balance_g"] - 20.0) < 0.2, info["balance_g"]
    env.reset(seed=1)
    term, info = _place_vial_on_pan(env, top_open=True)
    assert info["success"] == 0 and info["vial_on_pan"] and not info["doors_closed"]
    env.close()
