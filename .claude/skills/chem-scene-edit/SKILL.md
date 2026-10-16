---
name: chem-scene-edit
description: Change the chem_rl_sim weighing bench - move, add or remove props (vials, caps, dropper, clutter), edit the balance, or change a layout - and verify the result. Use whenever editing chem_rl_sim/scene.py, chem_rl_sim/balance.py or LAYOUTS, or when the user says "move the objects", "add a vial", "change the layout", or asks what parts of the balance move.
---

# Editing the chem_rl_sim scene

Builds on the global `mujoco-scene` skill (physics and measurement traps). This file
holds what is specific to this repo.

## Run anything

```bash
export MUJOCO_GL=egl                                   # without it pytest segfaults at GL init
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q      # ROS Jazzy plugins on PYTHONPATH break pytest otherwise
```
11 tests, ~2 s. The "Attach conflict ... ur5e" and "parent has ... keeping parent value" warnings are normal.

End effector: `hand="robotiq"` (default, Robotiq 2F-85 bolted to the flange, built in
`chem_rl_sim/robot.py`) or `"wuji"`/`"sharpa"` (urgantry_sim hands on the adapter cylinder).

## Frame and bench limits (verified 2026-10-15)

- World frame is urgantry_sim's: origin on the floor at the exposed table's center,
  +x right along the mount edge, +y away from the gantry column, +z up, meters.
- Exposed bench: x in [-0.56, 0.56], y in [-0.2925, 0.2925], top z = `BOARD_TOP` = 0.775.
  The column stands behind y = -0.2925. "Away from the column" means +y.
- Balance bottom-center at `BALANCE_POS` = (0.10, 0.04); display faces -y (toward the robot).

## Layouts

`LAYOUTS` in `chem_rl_sim/scene.py` has two starting arrangements of the props:
- `task`: start of the weighing step, pan empty, vials upright on the bench. The env's
  default and what `docs/env.png` shows. Change this one unless told otherwise.
- `video`: reproduces frame 3:57 of the source footage (vial_a on the pan). Keep it
  matched to the footage; only edit it when the user asks for both.

Users may not know these names; explain them in one line before asking which to change.

## Moving props: checklist

1. Edit positions in `LAYOUTS[...]`; leave a comment saying what moved and why.
2. Long props: `dropper` lies ~15 cm along +y from its tip (the body origin). At
   y = 0.17 it already overhangs the far edge; y = 0.27 made it fall to the floor.
   Measure a prop's extent before pushing it toward an edge.
3. Settle 1 s and print each prop's drift. Anything over 3 mm (the test threshold) or a
   z near 0 means a collision or a fall. Snippet:
   ```python
   m, d = build_scene("task", "wuji")
   s = {k: d.body(k).xpos.copy() for k in LAYOUTS["task"]}
   for _ in range(int(1 / m.opt.timestep)): mujoco.mj_step(m, d)
   ```
4. Check against the clutter in `_clutter()` (Kimwipes box near (-0.42, 0.18),
   cardboard box (0.40, 0.14), cork ring (0.33, -0.12)) and the balance footprint
   (210 x 344 mm around `BALANCE_POS`).
5. The env jitters task props by `RANDOM_XY` = 15 mm per episode; leave that margin
   to edges and neighbours.
6. Run the tests, then re-render with the `chem-render-share` skill and look at the image.

## The balance (chem_rl_sim/balance.py)

What moves: the three draft-shield doors (`balance_door_{left,right,top}`), slide
joints, 0 = closed, open up to 0.20 m toward +y, with handles, damping and frictionloss.
What is fixed: housing, posts, front/rear glass, and the pan (welded body).
Reading: force sensor `balance_pan_force` under the pan, minus the pan's weight; the
20 g vial reads 20.000 g. The display and keypad are visual-only flat boxes; there is
no pressable tare/power button and the number appears only in `info["balance_display"]`.
If you change names here, the env (`chem_rl_sim/env.py`) reads them; update both.

## Contracts

Every module opens with a CONTRACT docstring (global rule). Update it when positions,
names, or outputs change.
