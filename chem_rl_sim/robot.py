"""CONTRACT
Input: gantry_spec(effector) with effector in EFFECTORS = ("robotiq", "wuji", "sharpa");
  DEFAULT_EFFECTOR = "robotiq".
Output: the urgantry_sim gantry mjSpec (submodule, unmodified) with that end effector
  on both UR7e flanges.
  - "wuji" / "sharpa": urgantry_sim unchanged. A 63 mm cylinder adapter sits on the
    flange and the 5-finger hand sits on the adapter.
  - "robotiq": no adapter cylinder and no hand. A Robotiq 2F-85 (mujoco_menagerie
    robotiq_2f85 @ 04674c8, verbatim in chem_rl_sim/robotiq_2f85/) is bolted
    straight onto the UR attachment_site, the same way MaxLab's urtable_sim mounts it.
    `{side}_flange_adapter` is still a body (an empty frame at the flange, no
    geom, no mass) so the gantry's wrist F/T sensors `{side}_ft_force/_torque`
    keep their names. Each gripper has one actuator
    `{side}_grip_fingers_actuator`, ctrl 0 = open .. 255 = closed (tendon driven,
    force +-5). The TCP is site `{side}_grip_pinch`, 0.145 m past the gripper base
    along the flange +z.
Actuator order (what the env's action/state index): left arm 6, left effector,
  right arm 6, right effector. Effector = 1 (robotiq), 20 (wuji), 22 (sharpa), so
  nu = 14 / 52 / 56. The robotiq order matches urtable_sim's SimUR7e env.
Downstream reads: tool_body(effector, side) / tool_point(model, data, side), plus
  the actuator names above, from chem_rl_sim/env.py and tools/.
Convention recovered: urgantry_sim.build_spec always mounts a hand through its
  private `_arm_with_hand(side, hand)`; for robotiq we swap that function for the
  duration of one build_spec call. That puts each gripper's actuator right after
  its own arm's six; deleting the adapters and attaching grippers after build_spec
  would append both grippers at the end and drop the F/T sensors. Checked by
  tests/test_scene.py (actuator names and order, F/T sensors present).
Reference output: docs/env.png (tools/render_overview.py).
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import mujoco
import numpy as np

from urgantry_sim import build_urgantry as gantry

ROBOTIQ_PATH = str(Path(__file__).resolve().parent / "robotiq_2f85" / "2f85.xml")
EFFECTORS = ("robotiq", *gantry.HANDS)
DEFAULT_EFFECTOR = "robotiq"
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 255.0


def _arm_with_robotiq(side: str, hand: str) -> mujoco.MjSpec:
    """UR7e with a Robotiq 2F-85 on its flange. Same signature as
    gantry._arm_with_hand so it can stand in for it; `hand` is ignored."""
    arm = gantry._load_part(gantry.UR7E_PATH)
    flange = arm.site("attachment_site")
    # Geomless stand-in for the adapter cylinder: keeps the F/T site's parent name.
    mount = arm.body("wrist_3_link").add_body(name="flange_adapter",
                                              pos=flange.pos, quat=flange.quat)
    gripper = gantry._load_part(ROBOTIQ_PATH)
    mount.add_site(name="gripper_mount").attach_body(gripper.body("base_mount"), "grip_", "")
    return arm


def gantry_spec(effector: str = DEFAULT_EFFECTOR) -> mujoco.MjSpec:
    if effector not in EFFECTORS:
        raise ValueError(f"effector must be one of {EFFECTORS}, got {effector!r}")
    if effector != "robotiq":
        return gantry.build_spec(spawn_props=False, hand=effector)
    with mock.patch.object(gantry, "_arm_with_hand", _arm_with_robotiq):
        return gantry.build_spec(spawn_props=False, hand=gantry.DEFAULT_HAND)


def effector_type(model: mujoco.MjModel) -> str:
    """'robotiq', 'wuji' or 'sharpa', read off the compiled model's actuator names."""
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_grip_fingers_actuator") >= 0:
        return "robotiq"
    return gantry.hand_type(model)


def tool_body(effector: str, side: str) -> str:
    """Body standing for the hand / gripper in reach shaping and IK."""
    return {"robotiq": f"{side}_grip_base", "wuji": f"{side}_hand_palm_link",
            "sharpa": f"{side}_hand_hand_C_MC"}[effector]


def tool_point(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    """World position (m) of the grasp point: the Robotiq pinch site, or the palm
    body origin for the hands."""
    if effector_type(model) == "robotiq":
        return data.site(f"{side}_grip_pinch").xpos.copy()
    return data.body(tool_body(effector_type(model), side)).xpos.copy()


def set_effectors_open(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Command both effectors open (hands flat, grippers at ctrl 0)."""
    for side in ("left", "right"):
        if effector_type(model) == "robotiq":
            data.ctrl[model.actuator(f"{side}_grip_fingers_actuator").id] = GRIPPER_OPEN
        else:
            gantry.set_hand(model, data, side, gantry.HAND_OPEN)
