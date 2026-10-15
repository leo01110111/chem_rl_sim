"""CONTRACT
Input: add_balance(spec, parent, pos, yaw) on an mjSpec; pos = world position (m)
  of the balance's bottom-center on the bench, yaw (rad) about +z. At yaw=0 the
  terminal/display faces -y (toward the gantry column, i.e. the robot), +x is the
  balance's right side.
Output (names fixed, the env reads them):
  bodies  balance (fixed), balance_pan (fixed, child of balance), balance_door_left,
          balance_door_right, balance_door_top
  joints  balance_door_{left,right,top}: slide, 0 = closed, + = open (meters,
          doors slide toward +y local, i.e. backward, like the real ME draft shield)
  site    balance_pan_site at the pan's top-center; sensor balance_pan_force
          (mjSENS_FORCE, newtons, in the site frame; +z = support force on the pan)
  site    balance_chamber_center
Reading: pan_reading_g(model, data, tare_N=PAN_MASS*g) = (Fz - tare_N) / g * 1000
  -- grams on the pan. Default tare is the pan's own weight, which equals the
  empty-pan Fz at rest (measured 1.4715 N = 0.15 kg * 9.81). Checked in
  tests/test_scene.py: the 20 g vial reads 20.000 g after 1 s of settling.
Dimensions: Mettler Toledo ME204 class (the balance in the video, frame 3:57:
  "METTLER TOLEDO ... ME", 0.1 mg display). Overall 210 x 344 x 344 mm (W x D x H)
  and 90 mm pan are reseller-listed (ME204TE: 210x344x344; successor MA204:
  209x354x354, pan 90 mm). Chamber split, pan height and post size are estimated
  from the video, not from a drawing.
Reference output: work/scene_*.png renders from tools/render_scene.py.
"""
from __future__ import annotations

import mujoco
import numpy as np

W, D, H = 0.210, 0.344, 0.344
BASE_H = 0.075                 # housing top = chamber floor (estimated)
TERMINAL_D = 0.100             # front strip with the display (estimated)
CH_Y0 = -D / 2 + TERMINAL_D    # chamber front edge, local y
CH_Y1 = D / 2
CH_D = CH_Y1 - CH_Y0
GLASS_T = 0.004
POST = 0.012
PAN_R = 0.045
PAN_T = 0.003
PAN_Z = BASE_H + 0.014         # pan top height above the bench (estimated)
PAN_MASS = 0.15
DOOR_TRAVEL = 0.20

HOUSING_RGBA = [0.93, 0.91, 0.84, 1]
LCD_RGBA = [0.62, 0.74, 0.82, 1]
KEYPAD_RGBA = [0.45, 0.45, 0.47, 1]
STEEL_RGBA = [0.72, 0.73, 0.75, 1]
GLASS_RGBA = [0.80, 0.92, 0.88, 0.18]
FRAME_RGBA = [0.70, 0.71, 0.72, 1]
HANDLE_RGBA = [0.15, 0.15, 0.16, 1]

G = 9.81


def _box(body, name, half, pos, rgba, collide=True, **kw):
    return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half),
                         pos=list(pos), rgba=rgba,
                         contype=1 if collide else 0, conaffinity=1 if collide else 0, **kw)


def add_balance(spec: mujoco.MjSpec, parent: mujoco.MjsBody, pos, yaw: float = 0.0) -> mujoco.MjsBody:
    quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    b = parent.add_body(name="balance", pos=list(pos), quat=quat)
    hw = W / 2

    _box(b, "balance_housing", (hw, D / 2, BASE_H / 2), (0, 0, BASE_H / 2), HOUSING_RGBA)
    # Sloped terminal: display panel tilted up toward the operator.
    tilt = np.deg2rad(20)
    tq = [np.cos(tilt / 2), np.sin(tilt / 2), 0, 0]
    ty = -D / 2 + TERMINAL_D / 2
    _box(b, "balance_lcd", (0.07, 0.022, 0.001), (0, ty + 0.012, BASE_H + 0.002), LCD_RGBA,
         collide=False, quat=tq)
    _box(b, "balance_keypad", (0.085, 0.012, 0.001), (0, ty - 0.03, BASE_H + 0.001), KEYPAD_RGBA,
         collide=False)

    # Pan: its own (welded) body so the force sensor sees exactly what rests on it.
    pan = b.add_body(name="balance_pan", pos=[0, CH_Y0 + CH_D / 2, PAN_Z - PAN_T / 2])
    pan.add_geom(name="balance_pan", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[PAN_R, PAN_T / 2, 0], rgba=STEEL_RGBA, mass=PAN_MASS,
                 friction=[1.0, 0.005, 0.0001])
    pan.add_site(name="balance_pan_site", pos=[0, 0, PAN_T / 2])
    spec.add_sensor(name="balance_pan_force", type=mujoco.mjtSensor.mjSENS_FORCE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="balance_pan_site")
    # Draft ring around the pan (visual; the real one is a low steel collar).
    b.add_geom(name="balance_draft_ring", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
               size=[PAN_R + 0.012, 0.004, 0], pos=[0, CH_Y0 + CH_D / 2, BASE_H + 0.004],
               rgba=STEEL_RGBA, contype=0, conaffinity=0)

    top_z = H
    mid_z = (BASE_H + top_z) / 2
    half_h = (top_z - BASE_H) / 2
    # Corner posts (the grey extrusions with dark caps in the video).
    for sx, side in ((-1, "l"), (1, "r")):
        for py, end in ((CH_Y0 + POST / 2, "f"), (CH_Y1 - POST / 2, "b")):
            _box(b, f"balance_post_{side}_{end}", (POST / 2, POST / 2, half_h),
                 (sx * (hw - POST / 2), py, mid_z), FRAME_RGBA)
    # Fixed glass: front (between terminal and chamber) and rear.
    _box(b, "balance_glass_front", (hw - POST, GLASS_T / 2, half_h), (0, CH_Y0 + GLASS_T / 2, mid_z),
         GLASS_RGBA, group=1)
    _box(b, "balance_glass_rear", (hw - POST, GLASS_T / 2, half_h), (0, CH_Y1 - GLASS_T / 2, mid_z),
         GLASS_RGBA, group=1)

    def door(name, pos, half, axis_y=True):
        d = b.add_body(name=f"balance_door_{name}", pos=list(pos))
        d.add_joint(name=f"balance_door_{name}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0],
                    range=[0, DOOR_TRAVEL], damping=2.0, frictionloss=0.3, armature=0.05)
        _box(d, f"balance_door_{name}_glass", half, (0, 0, 0), GLASS_RGBA, group=1, mass=0.25)
        return d

    side_half = (GLASS_T / 2, CH_D / 2 - POST, half_h - 0.003)
    for name, sx in (("left", -1), ("right", 1)):
        d = door(name, (sx * (hw + GLASS_T / 2 + 0.001), CH_Y0 + CH_D / 2, mid_z), side_half)
        # Handle at the front edge, outside the glass.
        _box(d, f"balance_door_{name}_handle", (0.006, 0.004, 0.03),
             (sx * 0.007, -CH_D / 2 + POST + 0.012, 0.0), HANDLE_RGBA, mass=0.01)
    top = door("top", (0, CH_Y0 + CH_D / 2, top_z + GLASS_T / 2 + 0.001),
               (hw, CH_D / 2 - POST / 2, GLASS_T / 2))
    _box(top, "balance_door_top_handle", (0.03, 0.005, 0.005), (0, -CH_D / 2 + 0.012, 0.007),
         HANDLE_RGBA, mass=0.01)
    b.add_site(name="balance_chamber_center", pos=[0, CH_Y0 + CH_D / 2, mid_z])
    return b


def pan_force_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    adr = model.sensor("balance_pan_force").adr[0]
    return float(data.sensordata[adr + 2])


def pan_reading_g(model: mujoco.MjModel, data: mujoco.MjData, tare_n: float = PAN_MASS * G) -> float:
    return (pan_force_z(model, data) - tare_n) / G * 1000.0


def pan_world(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return data.site("balance_pan_site").xpos.copy()
