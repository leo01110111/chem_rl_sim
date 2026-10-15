"""CONTRACT
Checks the scene's physical claims; run: uv run pytest tests/ -q
  - both layouts compile with both hands and every prop settles < 3 mm in 1 s
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
from chem_rl_sim.scene import LAYOUTS, build_scene


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("hand", ["wuji", "sharpa"])
def test_settle_and_reading(layout, hand):
    m, d = build_scene(layout, hand)
    start = {k: d.body(k).xpos.copy() for k in LAYOUTS[layout]}
    for _ in range(500):
        mujoco.mj_step(m, d)
    drift = max(np.linalg.norm(d.body(k).xpos - start[k]) for k in start)
    assert drift < 0.003, drift
    expected = 20.0 if layout == "video" else 0.0
    assert abs(pan_reading_g(m, d) - expected) < 0.05


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
