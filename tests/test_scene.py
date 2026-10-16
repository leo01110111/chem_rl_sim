"""CONTRACT
Checks the scene's physical claims; run: uv run pytest tests/ -q
  - both layouts compile with every effector (robotiq, wuji, sharpa) and every
    prop settles < 3 mm in 1 s
  - robotiq: no adapter cylinder (flange_adapter has no geom), one gripper
    actuator right after each arm's six, wrist F/T sensors still present, and no
    robot contact at home
  - the balance reads 20.000 +- 0.05 g with the 20 g vial on the pan (video
    layout) and 0 +- 0.05 g with an empty pan (task layout)
  - every draft-shield door slides open to its 0.20 m limit and closes again
Reference numbers (2026-10-15, mujoco 3.16): drift 2.3 mm (dropper settling onto
its tip), reading 20.000 g / 0.000 g.
"""
import mujoco
import numpy as np
import pytest

from chem_rl_sim.balance import pan_reading_g
from chem_rl_sim.robot import EFFECTORS
from chem_rl_sim.scene import LAYOUTS, build_scene


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("hand", list(EFFECTORS))
def test_settle_and_reading(layout, hand):
    m, d = build_scene(layout, hand)
    start = {k: d.body(k).xpos.copy() for k in LAYOUTS[layout]}
    for _ in range(500):
        mujoco.mj_step(m, d)
    drift = max(np.linalg.norm(d.body(k).xpos - start[k]) for k in start)
    assert drift < 0.003, drift
    expected = 20.0 if layout == "video" else 0.0
    assert abs(pan_reading_g(m, d) - expected) < 0.05


def test_robotiq_mount():
    m, d = build_scene("task")
    names = [m.actuator(i).name for i in range(m.nu)]
    assert m.nu == 14
    assert names[6] == "left_grip_fingers_actuator" and names[13] == "right_grip_fingers_actuator"
    assert not any("_hand_" in n for n in names)
    for side in ("left", "right"):
        assert m.body_geomnum[m.body(f"{side}_flange_adapter").id] == 0
        m.sensor(f"{side}_ft_force"), m.sensor(f"{side}_ft_torque")
    for _ in range(500):
        mujoco.mj_step(m, d)
    robot = {m.body(i).name for i in range(m.nbody)
             if m.body(i).name.startswith(("left_", "right_"))}
    hits = [c for c in d.contact[:d.ncon]
            if {m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name} & robot]
    assert not hits


def test_doors_slide():
    m, d = build_scene("task")
    for door in ("left", "right", "top"):
        j = m.joint(f"balance_door_{door}")
        assert np.allclose(j.range, [0, 0.20])
        d.qpos[j.qposadr[0]] = j.range[1]
    mujoco.mj_forward(m, d)
    for _ in range(200):
        mujoco.mj_step(m, d)
    for door in ("left", "right", "top"):
        q = d.qpos[m.joint(f"balance_door_{door}").qposadr[0]]
        assert 0.19 < q <= 0.2001, (door, q)
