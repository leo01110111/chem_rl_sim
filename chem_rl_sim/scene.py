"""CONTRACT
Input: build_spec(layout="video"|"task", hand="wuji"|"sharpa", clutter=True).
Output: an mjSpec = urgantry_sim's gantry scene (bare: no block/tray; imported from
  the third_party/urgantry_sim submodule so robot updates come from upstream) plus
  the weighing bench from the video at 2:50-4:10 (chem_lab_vial_weighing_trial1):
    balance (chem_rl_sim/balance.py), 20 mL vials, loose caps, Pasteur dropper,
    and, with clutter=True, the cork ring, Kimwipes box, cardboard box and green
    floor tape seen around the balance.
  World frame is urgantry_sim's: origin on the floor at the center of the exposed
  table, +x right along the mount edge, +y away from the column, +z up, meters;
  bench top z = BOARD_TOP = 0.775. Integrator set to implicitfast (stiff servos
  chatter under Euler; see the mujoco-scene skill).
Fixed API (same shape as urgantry_sim.build_urgantry):
  build_spec / build_model / build_scene / set_initial_pose(model, data, layout)
Names the env reads: free joints "<prop>_joint" for every prop in LAYOUTS, plus
  balance names (balance.py). TARGET_VIAL is the vial the task moves.
Layout mapping from the video (operator = robot side, -y):
  balance in front of the operator, display toward them; vials, caps, dropper and
  Kimwipes to the operator's LEFT (-x); cork ring and cardboard box to the RIGHT
  (+x). In the video frame the bench edge with drawers is the image's left edge,
  so the phone was rolled 90 deg; "operator's left" = image top (frame 3:57).
  Positions are eyeballed from that frame at the vial/pan scale (28 mm / 90 mm),
  not reconstructed metrically -- expect +-3 cm.
Reference output: work/scene_video.png, work/scene_task.png (tools/render_scene.py).
"""
from __future__ import annotations

import numpy as np
import mujoco

from urgantry_sim.build_urgantry import (
    BOARD_TOP, DEFAULT_HAND, build_spec as gantry_spec, set_initial_pose as gantry_pose,
)

from .assets import add_asset
from .balance import CH_D, CH_Y0, PAN_Z, add_balance

BALANCE_POS = (0.10, 0.04)          # bench xy of the balance's bottom-center
PAN_XY = (BALANCE_POS[0], BALANCE_POS[1] + CH_Y0 + CH_D / 2)

VIAL_R = 0.014
DROPPER_TIP_R = 0.004       # lying, the tip end (body origin) rests on the 8 mm stem
TARGET_VIAL = "vial_a"

Q_UP = (1.0, 0.0, 0.0, 0.0)


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return tuple(out.tolist())


# Lying vial: axis -> +x, then turned 15 deg about its own axis so one of the 12
# collision wall facets (make_vial.N_WALL) faces down; resting on a facet edge it
# rolled 6 mm while settling.
Q_LIE_X = _quat_mul((np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0),
                    (np.cos(-np.pi / 24), 0.0, 0.0, np.sin(-np.pi / 24)))
Q_LIE_Y = (np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0, 0.0)   # axis -> +y

# prop -> (asset, xyz, quat). z for upright props is the surface height; lying
# props are lifted by their radius.
_BENCH = BOARD_TOP
LAYOUTS = {
    # Frame 3:57: vial a is on the pan, doors closed, rest of the bench as filmed.
    "video": {
        "vial_a": ("vial_20ml_open", (*PAN_XY, _BENCH + PAN_Z), Q_UP),
        "vial_b": ("vial_20ml_open", (-0.25, 0.03, _BENCH + VIAL_R), Q_LIE_X),
        "vial_c": ("vial_20ml_whitecap", (-0.13, -0.08, _BENCH + VIAL_R), Q_LIE_X),
        "vial_red": ("vial_20ml_redcap", (-0.20, 0.10, _BENCH), Q_UP),
        "cap_1": ("cap_24_400_white", (-0.31, -0.03, _BENCH), Q_UP),
        "cap_2": ("cap_24_400_white", (-0.33, 0.07, _BENCH), Q_UP),
        "dropper": ("pasteur_dropper", (-0.10, 0.09, _BENCH + DROPPER_TIP_R), Q_LIE_Y),
    },
    # Start of the weighing step (2:50): pan empty, every vial upright on the bench
    # within both arms' reach so a policy can pick vial a and set it on the pan.
    "task": {
        "vial_a": ("vial_20ml_open", (-0.12, -0.04, _BENCH), Q_UP),
        "vial_b": ("vial_20ml_open", (-0.20, 0.04, _BENCH), Q_UP),
        "vial_c": ("vial_20ml_whitecap", (-0.13, 0.10, _BENCH), Q_UP),
        "vial_red": ("vial_20ml_redcap", (-0.27, 0.12, _BENCH), Q_UP),
        "cap_1": ("cap_24_400_white", (-0.31, -0.03, _BENCH), Q_UP),
        "cap_2": ("cap_24_400_white", (-0.33, 0.05, _BENCH), Q_UP),
        "dropper": ("pasteur_dropper", (-0.04, 0.17, _BENCH + DROPPER_TIP_R), Q_LIE_Y),
    },
}
LAYOUT_NAMES = tuple(LAYOUTS)

CORK_RGBA = [0.42, 0.24, 0.13, 1]
CARDBOARD_RGBA = [0.80, 0.66, 0.50, 1]
KIMWIPES_RGBA = [0.20, 0.62, 0.42, 1]
TAPE_RGBA = [0.45, 0.85, 0.20, 1]


def _clutter(spec: mujoco.MjSpec) -> None:
    wb = spec.worldbody
    # Cork ring stand (~110 mm OD, 40 mm hole), right of the balance.
    cork = wb.add_body(name="cork_ring", pos=[0.33, -0.12, _BENCH])
    for k in range(12):
        t = 2 * np.pi * k / 12
        tangent = np.array([-np.sin(t), np.cos(t), 0.0])
        radial = np.array([np.cos(t), np.sin(t), 0.0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.column_stack([radial, [0, 0, 1], tangent]).flatten())
        cork.add_geom(name=f"cork_ring_{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      size=[0.017, 0.011, 0], pos=list(0.037 * radial + [0, 0, 0.017]),
                      quat=quat.tolist(), rgba=CORK_RGBA)
    # Fisherbrand cardboard box and Kimwipes box (sizes eyeballed against the balance).
    wb.add_geom(name="cardboard_box_fisher", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.09, 0.13, 0.05], pos=[0.40, 0.14, _BENCH + 0.05],
                quat=[np.cos(0.15), 0, 0, np.sin(0.15)], rgba=CARDBOARD_RGBA)
    wb.add_geom(name="kimwipes_box", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.058, 0.108, 0.06], pos=[-0.42, 0.18, _BENCH + 0.06],
                quat=[np.cos(-0.2), 0, 0, np.sin(-0.2)], rgba=KIMWIPES_RGBA)
    # Green tape L marking the balance's spot (visual only).
    bx, by = BALANCE_POS
    for name, size, pos in (
        ("tape_front", [0.13, 0.008, 0.0004], [bx, by - 0.19, _BENCH + 0.0004]),
        ("tape_left", [0.008, 0.20, 0.0004], [bx - 0.125, by, _BENCH + 0.0004]),
    ):
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                    rgba=TAPE_RGBA, contype=0, conaffinity=0)


def build_spec(layout: str = "task", hand: str = DEFAULT_HAND, clutter: bool = True) -> mujoco.MjSpec:
    spec = gantry_spec(spawn_props=False, hand=hand)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Small glass on a smooth pan: default 2-cone friction slips; elliptic +
    # noslip keeps a settled vial from creeping and the balance reading still.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 3
    add_balance(spec, spec.worldbody, (*BALANCE_POS, _BENCH))
    for prop, (asset, pos, quat) in LAYOUTS[layout].items():
        add_asset(spec, spec.worldbody, asset, prop, pos, quat)
    if clutter:
        _clutter(spec)
    return spec


def build_model(layout: str = "task", hand: str = DEFAULT_HAND, clutter: bool = True) -> mujoco.MjModel:
    return build_spec(layout, hand, clutter).compile()


def set_initial_pose(model: mujoco.MjModel, data: mujoco.MjData, layout: str = "task") -> None:
    gantry_pose(model, data)
    for prop, (_, pos, quat) in LAYOUTS[layout].items():
        adr = model.joint(f"{prop}_joint").qposadr[0]
        data.qpos[adr:adr + 7] = [*pos, *quat]
    for door in ("left", "right", "top"):
        data.qpos[model.joint(f"balance_door_{door}").qposadr[0]] = 0.0


def build_scene(layout: str = "task", hand: str = DEFAULT_HAND, clutter: bool = True):
    model = build_model(layout, hand, clutter)
    data = mujoco.MjData(model)
    set_initial_pose(model, data, layout)
    mujoco.mj_forward(model, data)
    return model, data
