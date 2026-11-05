"""Teleop mapping, IK and recorder checks against the real chem model; no camera, no WiLoR.
A synthetic operator hand moves in the OpenCV camera frame of a selfie or overhead camera,
and the tool must move the same way in the operator's frame (= world as seen from top1:
x right, y forward, z up) and in the rendered top1 image, track within mm, and pinch must close the
gripper. The recorder must write a file OGPO's robomimic loader can read."""
import json

import h5py
import mujoco
import numpy as np
import pytest

from chem_rl_sim.env import VialWeighingEnv
from chem_rl_sim.teleop.arm import ArmTeleop, operator_rotation, pinch_to_gripper
from chem_rl_sim.teleop.recorder import Recorder, observe
from chem_rl_sim.teleop import wire

DT = 0.05


def project(model, data, p, cam="top1", size=480):
    """World point -> (u, v) pixels in a MuJoCo camera (v down)."""
    c = model.camera(cam).id
    R = data.cam_xmat[c].reshape(3, 3)
    pc = R.T @ (p - data.cam_xpos[c])          # MuJoCo cam frame: x right, y up, -z forward
    f = 0.5 * size / np.tan(np.deg2rad(model.cam_fovy[c]) / 2)
    return np.array([size / 2 + f * pc[0] / -pc[2], size / 2 - f * pc[1] / -pc[2]])


def hand(p, R=np.eye(3), pinch=0.10):
    kw = np.zeros((21, 3), np.float32)
    kw[8] = [pinch, 0, 0]
    kc = kw + np.asarray(p, np.float32)
    return {"keypoints_cam": kc, "keypoints_wrist": kw, "wrist_rot_cam": np.asarray(R, np.float32)}


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@pytest.fixture(scope="module")
def setup():
    env = VialWeighingEnv(control_hz=1 / DT, max_episode_steps=10 ** 6)
    env.reset(seed=0)
    R_wc = operator_rotation("selfie")
    arms = {s: ArmTeleop(env.model, env.data, s, "robotiq", R_wc, dt=DT) for s in ("left", "right")}
    return env, arms


def drive(env, arms, hands, n):
    for _ in range(n):
        ctrl = env.data.ctrl.copy()
        for s, a in arms.items():
            a.step(hands.get(s))
            a.write(ctrl)
        env.step(ctrl)


# Operator motion -> the hand's delta in each camera's OpenCV frame (x right in image,
# y down in image, z along the optical axis).
OPERATOR_MOVES = {
    "selfie":   {"right": [-1, 0, 0], "forward": [0, 0, -1], "up": [0, -1, 0]},
    "overhead": {"right": [1, 0, 0], "forward": [0, -1, 0], "up": [0, 0, -1]},
}
WORLD_AXIS = {"right": 0, "forward": 1, "up": 2}
# The home pose is ~0.81 m from the base, near the 0.85 m reach clamp, so the forward
# axis is tested by moving back toward the operator.
SIGN = {"right": 1, "forward": -1, "up": 1}


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("view", ["selfie", "overhead"])
@pytest.mark.parametrize("move", ["right", "forward", "up"])
def test_operator_frame_translation(setup, side, view, move):
    env, arms = setup
    env.reset(seed=0)
    for a in arms.values():
        a.reset()
        a.R_wc = operator_rotation(view)
    a = arms[side]
    p0 = np.array([0.0, 0.0, 0.6])
    a.arm()
    drive(env, arms, {side: hand(p0)}, 1)
    assert a.engaged
    tool0 = a.tool_pose()[0]
    uv0 = project(env.model, env.data, tool0)
    drive(env, arms, {side: hand(p0 + SIGN[move] * 0.08 * np.array(OPERATOR_MOVES[view][move]))}, 40)
    tool1 = a.tool_pose()[0]
    want = np.zeros(3)
    want[WORLD_AXIS[move]] = SIGN[move] * 0.08
    assert np.linalg.norm((tool1 - tool0) - want) < 0.015, f"{view} {move}: tool moved {tool1 - tool0}"
    if move == "right":      # and right on screen
        assert project(env.model, env.data, tool1)[0] - uv0[0] > 20


def test_rotation_follows_hand(setup):
    env, arms = setup
    env.reset(seed=0)
    for a in arms.values():
        a.reset()
    a = arms["left"]
    a.arm()
    p0 = np.array([0.0, 0.0, 0.6])
    drive(env, arms, {"left": hand(p0)}, 1)
    R0 = a.tool_pose()[1]
    a.R_wc = operator_rotation("selfie")
    drive(env, arms, {"left": hand(p0, rotz(0.4))}, 40)   # roll about the camera's optical axis
    R1 = a.tool_pose()[1]
    dR = R1 @ R0.T
    axis_world = a.R_wc @ np.array([0, 0, 1.0])
    rv = np.zeros(3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, dR.flatten())
    mujoco.mju_quat2Vel(rv, q, 1.0)
    assert abs(np.linalg.norm(rv) - 0.4) < 0.05
    assert rv @ axis_world / np.linalg.norm(rv) > 0.97


def test_clutch_holds_and_lost_hand_holds(setup):
    env, arms = setup
    env.reset(seed=0)
    for a in arms.values():
        a.reset()
    a = arms["right"]
    q_home = a.q_cmd.copy()
    drive(env, arms, {"right": hand([0, 0, 0.6])}, 5)               # not armed: nothing
    assert np.allclose(a.q_cmd, q_home)
    a.arm()
    drive(env, arms, {"right": hand([0, 0, 0.6])}, 2)
    q_engaged = a.q_cmd.copy()
    drive(env, arms, {}, 10)                                        # hand lost: hold
    assert np.allclose(a.q_cmd, q_engaged)
    a.release()
    drive(env, arms, {"right": hand([0.1, 0, 0.6])}, 10)            # released: hold
    assert np.allclose(a.q_cmd, q_engaged)


def test_single_frame_glitch_rejected(setup):
    env, arms = setup
    env.reset(seed=0)
    for a in arms.values():
        a.reset()
    a = arms["left"]
    a.arm()
    drive(env, arms, {"left": hand([0, 0, 0.6])}, 3)
    q = a.q_cmd.copy()
    drive(env, arms, {"left": hand([0.4, 0, 0.6])}, 1)              # 40 cm flicker
    assert np.allclose(a.q_cmd, q)
    drive(env, arms, {"left": hand([0.15, 0, 0.6])}, 40)            # real move, held 2 s
    assert not np.allclose(a.q_cmd, q)


def test_pinch_mapping():
    assert pinch_to_gripper(hand([0, 0, 0], pinch=0.12)["keypoints_wrist"]) == 0.0
    assert pinch_to_gripper(hand([0, 0, 0], pinch=0.01)["keypoints_wrist"]) == 255.0
    mid = pinch_to_gripper(hand([0, 0, 0], pinch=0.06)["keypoints_wrist"])
    assert 100 < mid < 155


def test_wire_roundtrip():
    msg = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": [np.int64(3), "x"], "c": b"jpg"}
    out = wire.unpack(wire.pack(msg))
    assert np.array_equal(out["a"], msg["a"]) and out["b"] == [3, "x"] and out["c"] == b"jpg"


def test_recorder_ogpo_layout(setup, tmp_path):
    env, arms = setup
    env.reset(seed=0)
    for a in arms.values():
        a.reset()
    rec = Recorder(tmp_path / "d.hdf5", {"env_name": "ChemVialWeighing-v0"}, 20, "robotiq")
    obs, _ = env.reset(seed=0)
    lo, hi = env.model.actuator_ctrlrange.T
    for ep in range(2):
        rec.begin(observe(env, arms, obs["image"]))
        for t in range(5):
            ctrl = env.data.ctrl.copy()
            state = np.concatenate([[env.data.time], env.data.qpos, env.data.qvel])
            obs, _, _, _, info = env.step(ctrl)
            op = {"keypoints_cam": np.full((2, 21, 3), np.nan, np.float32),
                  "keypoints_wrist": np.full((2, 21, 3), np.nan, np.float32),
                  "engaged": np.zeros(2, bool), "perception_seq": t, "perception_age_ms": 30.0}
            rec.add(state, 2 * (ctrl - lo) / (hi - lo) - 1, ctrl, 0.0, observe(env, arms, obs["image"]), op)
        rec.save(success=False)
    rec.wait()
    assert rec.last_error is None
    with h5py.File(tmp_path / "d.hdf5") as f:
        assert sorted(f["data"]) == ["demo_0", "demo_1"]
        assert f["data"].attrs["total"] == 10
        json.loads(f["data"].attrs["env_args"])
        g = f["data/demo_1"]
        # exactly the reads OGPO's robomimic get_dataset performs
        assert g["actions"].shape == (5, 14) and np.all(np.abs(g["actions"][:]) <= 1)
        assert g["rewards"].shape == g["dones"].shape == (5,)
        for k in ("top1_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image"):
            assert g[f"obs/{k}"].shape == (5, 224, 224, 3) and g[f"obs/{k}"].dtype == np.uint8
            assert g[f"next_obs/{k}"].shape == (5, 224, 224, 3)
        # the two wrist views are different cameras, not copies of one
        assert not np.array_equal(g["obs/robot0_eye_in_hand_image"][0], g["obs/robot1_eye_in_hand_image"][0])
        for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot1_eef_pos", "object"):
            assert g[f"obs/{k}"].shape[0] == g[f"next_obs/{k}"].shape[0] == 5
        assert np.allclose(g["obs/joint_pos"][1:], g["next_obs/joint_pos"][:-1])
    assert rec.delete_last() == "demo_1"
    with h5py.File(tmp_path / "d.hdf5") as f:
        assert sorted(f["data"]) == ["demo_0"] and f["data"].attrs["total"] == 5


def test_workspace_box_keeps_tool_over_bench(setup):
    """A hand yanked 1 m toward the camera (a bad depth estimate) must not lift the tool
    out of the bench volume, and no joint may move faster than MAX_JOINT_SPEED."""
    from chem_rl_sim.scene import BOARD_TOP
    from chem_rl_sim.teleop.arm import MAX_JOINT_SPEED, WORKSPACE_Z_ABOVE_BENCH
    env, arms = setup
    env.reset(seed=0)
    for a in arms.values():
        a.reset()
    a = arms["left"]
    a.arm()
    drive(env, arms, {"left": hand([0, 0, 1.2])}, 1)
    q_prev = a.q_cmd.copy()
    for _ in range(60):
        drive(env, arms, {"left": hand([0, -0.5, 0.2])}, 1)       # toward camera and up
        assert np.all(np.abs(a.q_cmd - q_prev) <= MAX_JOINT_SPEED * DT + 1e-9)
        q_prev = a.q_cmd.copy()
    assert a.pos[2] <= BOARD_TOP + WORKSPACE_Z_ABOVE_BENCH["robotiq"][1] + 1e-9
    assert a.tool_pose()[0][2] < BOARD_TOP + 0.6


def test_grip_latch_holds_through_noise_and_releases_on_open(setup):
    """A relaxed pinch (5 cm) must command a full squeeze, a single noisy open frame must
    not release it, and a sustained open hand must (tools/grasp_hold_test.py has the
    physics side: linear 170/255 drops the vial, latched 255 holds it)."""
    from chem_rl_sim.teleop.arm import RELEASE_HOLD_S
    env, arms = setup
    env.reset(seed=0)
    for a in arms.values():
        a.reset()
    a = arms["left"]
    a.arm()
    p0 = [0.0, 0.0, 0.6]
    drive(env, arms, {"left": hand(p0, pinch=0.05)}, 20)
    assert a.gripping and a.eff_cmd[0] > 250
    drive(env, arms, {"left": hand(p0, pinch=0.10)}, 1)          # one flicker open
    drive(env, arms, {"left": hand(p0, pinch=0.05)}, 5)
    assert a.gripping and a.eff_cmd[0] > 250
    drive(env, arms, {"left": hand(p0, pinch=0.10)}, int(RELEASE_HOLD_S / DT) + 20)
    assert not a.gripping and a.eff_cmd[0] < 5
