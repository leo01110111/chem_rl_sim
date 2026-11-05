"""CONTRACT
Keyframe BC data collection for ChemVialWeighing with a mouse and keyboard (no camera,
no GPU model): drag a target pose for a gripper in a free 3D view, see the planned arm
as a ghost (green ok / orange touches a movable thing / red blocked), press Enter and the
arm runs there under physics while the episode records. Frozen time is never recorded.
  env -u PYTHONPATH uv run python -m chem_rl_sim.teleop.keyframe_app --dataset weigh_vial
Core logic (planning, checks, execution, undo, recording) is keyframe.Session; this file
is the window, mouse, keys and drawing.
Mouse (3D view):
  triad arrow drag     move the active target along that axis (red x, green y, blue z)
  triad ring drag      turn it about that axis (the ring's colour = its axis)
  left drag elsewhere  move the active target horizontally (also on the centre dot)
  shift + left drag    move it up / down
  ctrl + left drag     yaw it (horizontal mouse) and tilt it (vertical mouse)
  right drag           orbit the view;  middle drag: pan;  wheel: zoom (left drag never moves the view)
Keys:
  tab | 1 | 2          active arm: toggle | left | right
  w                    triad axes: world frame / the gripper's own frame
  arrows, pgup/pgdn    nudge the target (view-relative horizontal / vertical) by the step
  j l / i k / u o      yaw / tilt / roll the target by the step angle
  [ ]                  step size down / up  (2 mm 2 deg .. 3 cm 45 deg)
  v                    snap: above vial_a, top-down (again: grasp height)
  b                    snap: above the pan (again: place height)
  a                    snap: above the top door's handle, fingers across it (again: grasp)
  s                    point the gripper straight down (keeps position and heading)
  h / c                target to the episode's start pose / to where the gripper is now
  enter                execute the plan (when it is red, enter again = execute anyway)
  g                    toggle the active gripper (close / open)
  f                    follow mode on/off: arms track their targets live while you drag
  z | backspace        undo the last motion (sim, targets and recording)
  r                    new episode: reset the scene and start recording
  e / d                save / discard the episode;   x: delete the last saved episode
  0                    reset the scene without recording;  ?  help;  q / esc quit
  F11 | shift+f        fullscreen on / off (the window can also be resized or maximized)
View (the target edits stay view-relative, so turn the view to where you can see):
  n m / t y            orbit the view left / right;  tilt it up / down
  - =                  zoom out / in;  p: cycle the preset views
  3 4 5 6 7            view: operator | left side | right side | top | back
  8                    close-up on the active gripper's target
When the straight path would touch something, the plan goes up, across and down instead
("via above"); the drawn path shows the route and an X marks where a plan first touches.
The line under the target ends on whatever is below it; the top bar gives that gap
(from a carried vial's base when holding one).
An episode also saves itself when the env reports success, or after --max-episode-s
seconds of recorded motion.
Output: data/teleop/<dataset>.hdf5, OGPO/robomimic layout (recorder.py), with
  operator/ = goal_pos, goal_quat, segment, kind (keyframe.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from ..env import VialWeighingEnv
from ..scene import BOARD_TOP
from .keyframe import SIDES, Session, axis_rot
from .recorder import Recorder

WINDOW = "chem_rl_sim keyframe teleop"
COLORS = {"left": (255, 160, 0), "right": (0, 160, 255)}          # BGR, as in app.py
ARM_RGBA = {"left": (0.0, 0.63, 1.0, 1), "right": (1.0, 0.63, 0.0, 1)}
GHOST_RGBA = {"ok": (0.2, 0.9, 0.3, 0.35), "contact": (1.0, 0.6, 0.0, 0.4),
              "blocked": (1.0, 0.1, 0.1, 0.45)}
STATUS_BGR = {"ok": (80, 220, 80), "contact": (0, 160, 255), "blocked": (60, 60, 255)}
STEPS = [(0.002, 2.0), (0.005, 5.0), (0.01, 15.0), (0.03, 45.0)]   # (m, deg)
KEY = {"left": 65361, "up": 65362, "right": 65363, "down": 65364, "pgup": 65365, "f11": 65480,
       "pgdn": 65366, "tab": 9, "enter": 13, "backspace": 8, "esc": 27}
# Depending on the backend these arrive as X keysyms instead of ASCII.
KEY_ALIAS = {65289: 9, 65293: 13, 65421: 13, 10: 13, 65288: 8, 65307: 27}
DRAG_RAD_PER_PX = np.deg2rad(0.4)
# Triad gizmo on the active target, drawn flat over the 3D view (never hidden by the arm):
# arrow length and ring radius in pixels, pick distance, BGR colours.
TRIAD_PX, RING_PX, PICK_PX = 95, 62, 9
AXIS_BGR = ((70, 70, 240), (70, 210, 70), (240, 130, 50))
HOVER_BGR = (0, 230, 255)
# Gantry parts between the operator's viewpoint and the bench, drawn see-through in the
# 3D view only (the recorded cameras are untouched).
SEE_THROUGH = ("column", "column_foot", "gantry_head")
SEE_THROUGH_ALPHA = 0.12


# name: (azimuth, elevation, distance, lookat or None = the default); azimuth 0 looks along
# +x, i.e. from the left arm's side.
VIEWS = {"operator": (90.0, -42.0, 1.6, None), "left side": (0.0, -50.0, 2.0, None),
         "right side": (180.0, -50.0, 2.0, None), "top": (90.0, -89.0, 1.4, None),
         "back": (270.0, -30.0, 1.6, None)}
ORBIT_DEG, TILT_DEG = 15.0, 10.0


def drop_free(model):
    """Free-camera default: from the operator's side (like top1), over the balance."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.06, BOARD_TOP + 0.10)
    cam.distance, cam.azimuth, cam.elevation = 1.6, 90.0, -42.0
    return cam


class View:
    """Free orbit camera + screen <-> world rays for the main 3D view."""

    def __init__(self, model, width: int, height: int):
        self.model, self.renderer = model, None
        self.cam = drop_free(model)
        self.opt = mujoco.MjvOption()
        self.resize(width, height)

    def resize(self, width: int, height: int):
        if self.renderer is not None:
            close_renderer(self.renderer)
        self.W, self.H = width, height
        self.renderer = new_renderer(self.model, height, width)

    def basis(self):
        """Camera position, forward, right, up (world)."""
        az, el = np.deg2rad(self.cam.azimuth), np.deg2rad(self.cam.elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return np.array(self.cam.lookat) - self.cam.distance * fwd, fwd, right, up

    def ray(self, u: float, v: float):
        pos, fwd, right, up = self.basis()
        t = np.tan(np.deg2rad(self.model.vis.global_.fovy) / 2)
        x = (2 * u / self.W - 1) * t * self.W / self.H
        y = (1 - 2 * v / self.H) * t
        d = fwd + x * right + y * up
        return pos, d / np.linalg.norm(d)

    def project(self, p):
        pos, fwd, right, up = self.basis()
        r = np.asarray(p) - pos
        z = r @ fwd
        t = np.tan(np.deg2rad(self.model.vis.global_.fovy) / 2)
        return np.array([(r @ right / z / t * self.H / self.W + 1) * self.W / 2,
                         (1 - r @ up / z / t) * self.H / 2])

    def set(self, azimuth, elevation, distance, lookat=None):
        self.cam.azimuth, self.cam.elevation, self.cam.distance = azimuth, elevation, distance
        self.cam.lookat[:] = drop_free(self.model).lookat if lookat is None else lookat

    def orbit(self, d_az: float, d_el: float):
        self.cam.azimuth = (self.cam.azimuth + d_az) % 360.0
        self.cam.elevation = float(np.clip(self.cam.elevation + d_el, -89.0, 10.0))

    def move(self, action, dx: float, dy: float):
        mujoco.mjv_moveCamera(self.model, action, dx / self.H, dy / self.H, self.cam)


def close_renderer(r):
    """mujoco.Renderer.close() destroys its GL context before freeing its MjrContext, so
    the free runs in whichever context is current -- another renderer's -- and deletes that
    one's framebuffers by id (its images go black). Free the buffers in their own context."""
    if r._gl_context is not None and r._mjr_context is not None:
        r._gl_context.make_current()
        r._mjr_context.free()
    r.close()


def new_renderer(model, height: int, width: int):
    # The offscreen buffer is sized from the model; grow it for a big (fullscreen) window.
    g = model.vis.global_
    g.offwidth, g.offheight = max(g.offwidth, width), max(g.offheight, height)
    return mujoco.Renderer(model, height, width)


def seg_dist(q, a, b) -> float:
    ab, aq = b - a, q - a
    t = np.clip(aq @ ab / max(ab @ ab, 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(aq - t * ab))


def ring_points(p, axis, r, n=48):
    u = np.cross(axis, [0, 0, 1.0])
    if np.linalg.norm(u) < 1e-3:
        u = np.cross(axis, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    t = np.linspace(0, 2 * np.pi, n + 1)[:, None]
    return p + r * (np.cos(t) * u + np.sin(t) * np.cross(axis, u))


def axis_param(o, d, p0, a):
    """t of the point on the line p0 + t a nearest to the ray o + s d (a, d unit), or None
    when the two are near parallel."""
    b = a @ d
    if 1 - b * b < 0.02:
        return None
    w = p0 - o
    return (b * (d @ w) - a @ w) / (1 - b * b)


def plane_hit(origin, d, point, normal):
    den = d @ normal
    if abs(den) < 1e-6:
        return None
    t = ((point - origin) @ normal) / den
    return origin + t * d if t > 0 else None


class App:
    def __init__(self, args):
        self.args = args
        env_kwargs = dict(image_size=args.image_size, control_hz=args.hz, hand="robotiq",
                          layout=args.layout, clutter=args.clutter, max_episode_steps=10 ** 9,
                          start_door_open=args.start_door_open)
        self.env = VialWeighingEnv(**env_kwargs)
        out = Path(args.out) if args.out else Path("data/teleop") / f"{args.dataset}.hdf5"
        self.rec = Recorder(out, {"env_name": "ChemVialWeighing-v0", "type": "chem_rl_sim",
                                  "env_kwargs": {**env_kwargs, "normalized_actions": True,
                                                 "max_episode_steps": int(args.max_episode_s * args.hz)},
                                  "collector": "keyframe"},
                            args.hz, "robotiq")
        self.sess = Session(self.env, self.rec, speed=args.speed)
        self.sess.reset(seed=args.seed)
        m = self.env.model
        self.view = View(m, args.width, args.height)
        self.top1_r = self.wrist_r = None
        self._layout_panel(args.height)
        self.fullscreen = False
        self.window = None                              # TkWindow while run() is going
        self._ghost = mujoco.MjData(m)
        self._hit_ghost = mujoco.MjData(m)
        self.gaps = {}                                  # side -> Session.gap_below, per frame
        self.see_through = {m.geom(n).id for n in SEE_THROUGH
                            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0}
        self.visual = {s: [g for g in sorted(a.geoms) if m.geom_group[g] < 3 and m.geom_rgba[g, 3] > 0]
                       for s, a in self.sess.arms.items()}
        self.active = "left"
        self.step_i = 2
        self.drag = None
        self.force_armed = False
        self.help = False
        self.view_i = 0                                 # index into VIEWS for p
        self.triad_world = True                         # triad axes: world or tool frame
        self.hover = None                               # triad handle under the mouse
        self.message, self.message_t = "", 0.0
        self.n_saved = self.rec.num_demos()
        self.running = True

    # ------------------------------------------------------------ feedback
    def say(self, text: str):
        self.message, self.message_t = text, time.time()
        print(f"[keyframe] {text}", flush=True)

    # ------------------------------------------------------------ target edits
    def _edited(self):
        self.force_armed = False

    def nudge(self, d_view: np.ndarray):
        """d_view = (right, forward, up) in view-relative horizontal axes, in steps."""
        _, fwd, right, _ = self.view.basis()
        f = np.array([fwd[0], fwd[1], 0.0])
        f /= max(np.linalg.norm(f), 1e-9)
        r = np.array([right[0], right[1], 0.0])
        step = STEPS[self.step_i][0]
        p, _ = self.sess.targets[self.active]
        self.sess.set_target(self.active, p + step * (d_view[0] * r + d_view[1] * f + d_view[2] * np.array([0, 0, 1.0])))
        self._edited()

    def turn(self, axis: str, sign: int):
        p, R = self.sess.targets[self.active]
        ang = sign * np.deg2rad(STEPS[self.step_i][1])
        if axis == "yaw":
            dR = axis_rot([0, 0, 1], ang)
        elif axis == "tilt":
            dR = axis_rot(self.view.basis()[2], ang)
        else:                                           # roll about the tool's approach axis
            dR = axis_rot(R[:, 2], ang)
        self.sess.set_target(self.active, rot=dR @ R)
        self._edited()

    # ------------------------------------------------------------ mouse
    def on_mouse(self, event, x, y, flags, _param=None):
        if x >= self.view.W:
            return
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
        if event == cv2.EVENT_LBUTTONDOWN and not (shift or ctrl) and self.pick(x, y):
            self.drag = self._grab(self.pick(x, y), x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            p, R = self.sess.targets[self.active]
            mode = "rotate" if ctrl else "lift" if shift else "slide"
            o, d = self.view.ray(x, y)
            _, fwd, _, _ = self.view.basis()
            if mode == "slide":
                normal = np.array([0, 0, 1.0])
            else:
                normal = np.array([fwd[0], fwd[1], 0.0])
                normal /= max(np.linalg.norm(normal), 1e-9)
            self.drag = {"mode": mode, "xy": (x, y), "p": p.copy(), "R": R.copy(), "normal": normal,
                         "hit": plane_hit(o, d, p, normal)}
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            self.drag = {"mode": "orbit" if event == cv2.EVENT_RBUTTONDOWN else "pan", "xy": (x, y)}
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self.drag = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = np.int32(flags) >> 16                  # Qt packs the wheel delta in the high word
            self.view.move(mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * np.sign(delta) * self.view.H)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self._drag_to(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.hover = self.pick(x, y)

    # ------------------------------------------------------------ triad
    def triad(self):
        """Active target's triad: origin, axes as columns (world or tool frame), metres per
        pixel at the target (so it keeps its size on screen), or None behind the camera."""
        p, R = self.sess.targets[self.active]
        pos, fwd, _, _ = self.view.basis()
        depth = (p - pos) @ fwd
        if depth < 0.05:
            return None
        f = self.view.H / (2 * np.tan(np.deg2rad(self.env.model.vis.global_.fovy) / 2))
        return p, (np.eye(3) if self.triad_world else R.copy()), depth / f

    def _handles(self):
        """Screen polylines of the triad's handles: {('axis'|'ring', i): Nx2 pixels}."""
        t = self.triad()
        if t is None:
            return {}
        p, A, mpp = t
        P = lambda q: self.view.project(q)
        out = {}
        for i in range(3):
            out["ring", i] = np.array([P(q) for q in ring_points(p, A[:, i], RING_PX * mpp)])
            out["axis", i] = np.array([P(p + 0.2 * TRIAD_PX * mpp * A[:, i]), P(p + TRIAD_PX * mpp * A[:, i])])
        return out

    def pick(self, x, y):
        q, best, hit = np.array([x, y], float), PICK_PX, None
        t = self.triad()
        if t is None or np.linalg.norm(self.view.project(t[0]) - q) < 12:
            return None                                 # the centre dot: plain slide
        for h, pts in self._handles().items():
            d = min(seg_dist(q, a, b) for a, b in zip(pts[:-1], pts[1:]))
            if d < best or (d <= best + 2 and h[0] == "axis"):     # arrows win close calls
                best, hit = d, h
        return hit

    def _grab(self, handle, x, y):
        p, A, mpp = self.triad()
        kind, i = handle
        _, R = self.sess.targets[self.active]
        dr = {"mode": kind, "handle": handle, "xy": (x, y), "p": p.copy(), "R": R.copy(), "a": A[:, i].copy()}
        if kind == "axis":
            dr["t0"] = axis_param(*self.view.ray(x, y), p, dr["a"])
        else:
            # Face-on rings follow the mouse on the ring's plane; edge-on ones turn by
            # dragging along the ring's tangent at the grab point.
            fwd = self.view.basis()[1]
            dr["h0"] = plane_hit(*self.view.ray(x, y), p, dr["a"]) if abs(dr["a"] @ fwd) > 0.35 else None
            pts = ring_points(p, dr["a"], RING_PX * mpp)
            g = pts[np.argmin([np.linalg.norm(self.view.project(q) - [x, y]) for q in pts])]
            tan = np.cross(dr["a"], g - p)
            t2 = self.view.project(g + 0.1 * tan) - self.view.project(g)
            dr["t2"] = t2 / max(np.linalg.norm(t2), 1e-9)
        return dr

    def _drag_to(self, x, y):
        dr = self.drag
        x0, y0 = dr["xy"]
        if dr["mode"] == "axis":
            t = axis_param(*self.view.ray(x, y), dr["p"], dr["a"])
            if t is not None and dr["t0"] is not None:
                self.sess.set_target(self.active, dr["p"] + (t - dr["t0"]) * dr["a"], dr["R"])
                self._edited()
        elif dr["mode"] == "ring":
            h = plane_hit(*self.view.ray(x, y), dr["p"], dr["a"]) if dr["h0"] is not None else None
            if h is not None:
                r0, r1 = dr["h0"] - dr["p"], h - dr["p"]
                ang = np.arctan2(dr["a"] @ np.cross(r0, r1), r0 @ r1)
            elif dr["h0"] is not None:
                return
            else:
                ang = (np.array([x - x0, y - y0]) @ dr["t2"]) / RING_PX
            self.sess.set_target(self.active, dr["p"], axis_rot(dr["a"], ang) @ dr["R"])
            self._edited()
        elif dr["mode"] == "orbit":
            self.view.move(mujoco.mjtMouse.mjMOUSE_ROTATE_V, x - x0, y - y0)
            dr["xy"] = (x, y)
        elif dr["mode"] == "pan":
            self.view.move(mujoco.mjtMouse.mjMOUSE_MOVE_V, x - x0, y - y0)
            dr["xy"] = (x, y)
        elif dr["mode"] == "rotate":
            right = self.view.basis()[2]
            dR = axis_rot([0, 0, 1], -(x - x0) * DRAG_RAD_PER_PX) @ axis_rot(right, -(y - y0) * DRAG_RAD_PER_PX)
            self.sess.set_target(self.active, rot=dR @ dr["R"])
            self._edited()
        elif dr["hit"] is not None:
            o, d = self.view.ray(x, y)
            hit = plane_hit(o, d, dr["p"], dr["normal"])
            if hit is None:
                return
            delta = hit - dr["hit"]
            if dr["mode"] == "slide":
                delta[2] = 0.0
            else:
                delta = np.array([0.0, 0.0, delta[2]])
            self.sess.set_target(self.active, dr["p"] + delta)
            self._edited()

    # ------------------------------------------------------------ keys
    def on_key(self, k: int):
        s = self.sess
        k = KEY_ALIAS.get(k, k)
        ch = chr(k) if 32 <= k < 127 else "\0"
        moves = {KEY["up"]: (0, 1, 0), KEY["down"]: (0, -1, 0), KEY["left"]: (-1, 0, 0),
                 KEY["right"]: (1, 0, 0), KEY["pgup"]: (0, 0, 1), KEY["pgdn"]: (0, 0, -1)}
        turns = {"j": ("yaw", 1), "l": ("yaw", -1), "i": ("tilt", 1), "k": ("tilt", -1),
                 "u": ("roll", 1), "o": ("roll", -1)}
        if k in moves:
            self.nudge(np.array(moves[k], float))
        elif ch in turns:
            self.turn(*turns[ch])
        elif k == KEY["tab"]:
            self.active = "right" if self.active == "left" else "left"
        elif ch in "12":
            self.active = SIDES[int(ch) - 1]
        elif ch == "w":
            self.triad_world = not self.triad_world
            self.say(f"triad: {'world' if self.triad_world else 'gripper'} axes")
        elif ch == "[":
            self.step_i = max(0, self.step_i - 1)
        elif ch == "]":
            self.step_i = min(len(STEPS) - 1, self.step_i + 1)
        elif ch in "vb":
            self.say(f"{self.active}: " + s.snap(self.active, "vial" if ch == "v" else "pan"))
            self._edited()
        elif ch == "a":
            self.say(f"{self.active}: " + s.snap(self.active, "door"))
            self._edited()
        elif ch == "s":
            s.point_down(self.active)
            self._edited()
        elif ch == "h":
            s.snap(self.active, "home")
            self._edited()
        elif ch == "c":
            s.target_to_current(self.active)
            self._edited()
        elif k == KEY["enter"]:
            self.execute()
        elif ch == "g":
            ok, msg = s.grip(self.active)
            self.say(msg)
        elif ch == "f":
            s.follow = not s.follow
            self.say(f"follow {'on: arms track their targets live' if s.follow else 'off'}")
        elif ch == "z" or k == KEY["backspace"]:
            self.say(s.undo()[1])
        elif ch == "r":
            if s.recording:
                s.discard_episode()
            self.rec.wait()
            s.begin_episode()
            self.say("recording: new episode")
        elif ch == "e":
            self.finish(True)
        elif ch == "d":
            self.finish(False)
        elif ch == "x":
            if s.recording:
                return self.say("save or discard first")
            name = self.rec.delete_last()
            if name:
                self.n_saved -= 1
            self.say(f"deleted {name}" if name else "nothing to delete")
        elif ch == "0":
            if s.recording:
                s.discard_episode()
            s.reset()
            self.say("scene reset (not recording)")
        elif ch == "-":
            self.view.move(mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * self.view.H)
        elif ch == "=":
            self.view.move(mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * self.view.H)
        elif ch in "nmty":
            d = {"n": (-ORBIT_DEG, 0), "m": (ORBIT_DEG, 0), "t": (0, TILT_DEG), "y": (0, -TILT_DEG)}[ch]
            self.view.orbit(*d)
        elif ch == "p":
            names = list(VIEWS)
            self.view_i = (self.view_i + 1) % len(names)
            self.view.set(*VIEWS[names[self.view_i]])
            self.say(f"view: {names[self.view_i]}")
        elif ch in "34567":
            self.view_i = int(ch) - 3
            name = list(VIEWS)[self.view_i]
            self.view.set(*VIEWS[name])
            self.say(f"view: {name}")
        elif ch == "8":
            self.view.set(90.0, -60.0, 0.55, self.sess.targets[self.active][0])   # from the operator, steep
            self.say(f"view: close-up on the {self.active} target")
        elif k == KEY["f11"] or ch == "F":
            self.toggle_fullscreen()
        elif ch == "?":
            self.help = not self.help
        elif ch == "q" or k == KEY["esc"]:
            self.running = False

    def execute(self):
        s = self.sess
        if s.busy:
            return self.say("busy")
        plan = s.plan()
        if plan.status == "blocked" and not self.force_armed:
            self.force_armed = True
            return self.say("BLOCKED: " + "; ".join(plan.reasons) + "  (enter again to run anyway)")
        ok, msg = s.execute(force=True)
        self.force_armed = False
        self.say(msg)

    def finish(self, save: bool, why: str = ""):
        s = self.sess
        if not s.recording:
            return self.say("not recording (r starts an episode)")
        if save:
            n = s.save_episode()
            if n:
                self.n_saved += 1
            self.say(f"saved{why}: {n} steps ({n / self.args.hz:.1f} s), "
                     f"{'success' if s.info['success'] else 'no success'}")
        else:
            s.discard_episode()
            self.say("discarded")

    # ------------------------------------------------------------ loop pieces
    def tick(self):
        s = self.sess
        if not s.tick():
            return False
        if s.recording and s.info["success"] and not s.busy:
            self.finish(True, " (task success)")
        elif s.recording and len(self.rec) >= self.args.max_episode_s * self.args.hz:
            self.finish(True, " (time limit)")
        return True

    # ------------------------------------------------------------ drawing
    def _add_geom(self, scn, gtype, size, pos, mat, rgba):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                            np.asarray(mat, float).flatten(), np.asarray(rgba, np.float32))
        scn.ngeom += 1
        return g

    def _line(self, scn, a, b, rgba, width=0.0025):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1

    def _decorate(self, scn, plan):
        m, s = self.env.model, self.sess
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in self.see_through:
                g.rgba[3] = SEE_THROUGH_ALPHA
        gd, hd = self._ghost, self._hit_ghost
        show_ghost = plan is not None and not s.busy and not s.follow
        show_hit = show_ghost and plan.status != "ok" and plan.hit_i is not None
        if show_ghost:
            gd.qpos[:] = self.env.data.qpos
            for side, q in plan.final_q.items():
                gd.qpos[s.arms[side].qadr] = q
            mujoco.mj_kinematics(m, gd)
        if show_hit:                                    # the arm where the plan first touches
            hd.qpos[:] = self.env.data.qpos
            for side in s.arms:
                hd.qpos[s.arms[side].qadr] = plan.q[side][plan.hit_i]
            mujoco.mj_kinematics(m, hd)
        for side, a in s.arms.items():
            p, R = s.targets[side]
            cur = a.tool_pose()[0]
            moved = np.linalg.norm(p - cur) > 0.002 or np.trace(R.T @ a.tool_pose()[1]) < 2.999
            act = side == self.active
            base = np.array(ARM_RGBA[side])
            if show_ghost and moved:
                rgba = GHOST_RGBA[plan.status]
                for g in self.visual[side]:
                    vg = self._add_geom(scn, m.geom_type[g], m.geom_size[g], gd.geom_xpos[g],
                                        gd.geom_xmat[g], rgba)
                    if vg is not None and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                        vg.dataid = 2 * m.geom_dataid[g]
            if show_hit and moved:
                rgba = (*GHOST_RGBA[plan.status][:3], 0.16)
                for g in self.visual[side]:
                    vg = self._add_geom(scn, m.geom_type[g], m.geom_size[g], hd.geom_xpos[g],
                                        hd.geom_xmat[g], rgba)
                    if vg is not None and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                        vg.dataid = 2 * m.geom_dataid[g]
            alpha = 1.0 if act else 0.45
            self._add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.009 if act else 0.006, 0, 0], p, np.eye(3),
                           (*base[:3], alpha))
            for i, col in enumerate(((1, 0.2, 0.2), (0.2, 1, 0.2), (0.3, 0.4, 1))):
                if not act:                             # the active one gets the triad
                    self._line(scn, p, p + 0.06 * R[:, i], (*col, alpha), 0.002)
            w = 0.016 if s.gripping(side) else 0.042
            for sgn in (-1, 1):
                self._add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.004, 0.006, 0.02],
                               p + sgn * w * R[:, 1] - 0.005 * R[:, 2], R, (*base[:3], 0.6 * alpha))
            below = self.gaps.get(side)
            floor = below[2] + [0, 0, 0.001] if below else np.array([p[0], p[1], BOARD_TOP + 0.001])
            self._line(scn, p, floor, (0.9, 0.9, 0.9, 0.5 * alpha), 0.0012)
            self._add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.012, 0.0005, 0], floor, np.eye(3),
                           (*base[:3], 0.5 * alpha))
            if moved:
                self._line(scn, cur, p, (*base[:3], 0.35), 0.0015)

    def draw(self) -> np.ndarray:
        s, a = self.sess, self.args
        plan = None if s.busy else s.plan()
        v = self.view
        v.renderer.update_scene(self.env.data, camera=v.cam, scene_option=v.opt)
        self.gaps = {side: s.gap_below(side) for side in s.arms}
        self._decorate(v.renderer.scene, plan)
        img = cv2.cvtColor(v.renderer.render(), cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        if plan is not None and not s.follow:
            self._draw_paths(img, plan)
        self._draw_triad(img)

        cv2.rectangle(img, (0, 0), (W, 46), (25, 25, 25), -1)
        if s.recording:
            t = len(self.rec) / a.hz
            label, col = f"REC {int(t) // 60}:{t % 60:04.1f}", (40, 40, 230)
        else:
            label, col = "not recording (r)", (170, 170, 170)
        cv2.putText(img, label, (10, 30), font, 0.75, col, 2)
        step_m, step_d = STEPS[self.step_i]
        cv2.putText(img, f"{self.active.upper()} arm", (250, 30), font, 0.75, COLORS[self.active], 2)
        grip = " ".join(f"{x[0].upper()}:{'closed' if s.gripping(x) else 'open'}" for x in SIDES)
        info = (f"step {1e3 * step_m:.0f} mm / {step_d:.0f} deg   {grip}   "
                f"axes {'world' if self.triad_world else 'gripper'}   "
                f"{'FOLLOW  ' if s.follow else ''}undo {len(s.undo_stack)}   demos {self.n_saved}")
        cv2.putText(img, info, (400, 20), font, 0.45, (220, 220, 220), 1)
        below = self.gaps.get(self.active)
        gap = f"   below: {1e3 * below[0]:.0f} mm to {below[1]}" if below else ""
        cv2.putText(img, f"balance {s.info['balance_display']}   doors (mm) "
                         + " ".join(f"{max(0.0, 1e3 * x):.0f}" for x in s.info["doors"]) + gap, (400, 38), font,
                    0.45, (220, 220, 220), 1)
        # How far the active target is from its gripper (what Enter will move), per axis.
        d = s.targets[self.active][0] - s.arms[self.active].tool_pose()[0]
        if np.linalg.norm(d) > 0.002 and not s.busy:
            cv2.putText(img, f"move {1e2 * np.linalg.norm(d):.1f} cm  (x {1e2 * d[0]:+.1f}  y {1e2 * d[1]:+.1f}"
                             f"  z {1e2 * d[2]:+.1f})", (10, 92), font, 0.5, (235, 235, 235), 1)

        if s.busy:
            st, col = "executing ...", (230, 230, 230)
        elif s.follow:
            st, col = "follow mode: the arms track their targets live", (0, 200, 230)
        else:
            st = {"ok": "plan OK", "contact": "plan touches", "blocked": "plan BLOCKED"}[plan.status]
            if plan.vias:
                st += " via above"
            if plan.reasons:
                st += ": " + "; ".join(plan.reasons)
            st += f"   ({plan.T / a.hz:.1f} s)" if plan.status != "blocked" else ""
            col = STATUS_BGR[plan.status]
        cv2.putText(img, st, (10, 70), font, 0.55, col, 2)
        lines = [self.message] if time.time() - self.message_t < 4.0 else []
        if self.rec.last_error:
            lines.append(f"ERROR {self.rec.last_error}")
        if self.rec.saving:
            lines.append("saving ...")
        for i, ln in enumerate(lines):
            cv2.putText(img, ln, (10, H - 44 - 22 * i), font, 0.55, (255, 255, 255), 2)
        cv2.putText(img, "drag arrow/ring: move/turn | right-drag: view | 3-8: views | enter: go | g: grip | "
                         "v/b/a: vial/pan/door | s: point down | tab: arm | z: undo | r/e: rec/save | ?: help",
                    (10, H - 14), font, 0.42, (230, 230, 230), 1)
        if self.help:
            self._help(img)

        # Side panel: what gets recorded (top1) and the wrist cameras.
        panel = np.full((H, self.panel_w, 3), 25, np.uint8)
        n = self.top1_r.height
        cv2.putText(panel, "top1 (policy camera)", (10, 18), font, 0.5, (200, 200, 200), 1)
        self.top1_r.update_scene(self.env.data, camera="top1")
        panel[26:26 + n, 10:10 + n] = cv2.cvtColor(self.top1_r.render(), cv2.COLOR_RGB2BGR)
        y, k = 26 + n + 28, self.wrist_n
        for i, side in enumerate(SIDES):
            x = 10 + i * (k + 10)
            cv2.putText(panel, f"{side} wrist", (x, y - 8), font, 0.5, COLORS[side], 1)
            self.wrist_r.update_scene(self.env.data, camera=self.env.cameras[f"{side}_wrist"])
            if y + k <= H:
                panel[y:y + k, x:x + k] = cv2.cvtColor(self.wrist_r.render(), cv2.COLOR_RGB2BGR)
        return np.hstack([img, panel])

    def _draw_paths(self, img, plan):
        """The planned pinch path of each arm that moves, its via points, and an X where
        a touching / blocked plan first touches."""
        col = STATUS_BGR[plan.status]
        P = self.view.project
        for side, path in plan.path.items():
            if len(path) < 2 or np.linalg.norm(path[-1] - path[0]) < 0.003:
                continue
            pts = np.round([P(q) for q in path]).astype(np.int32)
            cv2.polylines(img, [pts], False, col, 2 if side == self.active else 1, cv2.LINE_AA)
            for v in plan.vias.get(side, []):
                cv2.circle(img, tuple(np.round(P(v)).astype(int)), 4, col, -1, cv2.LINE_AA)
            if plan.status != "ok" and side in plan.hit_pos:
                x, y = np.round(P(plan.hit_pos[side])).astype(int)
                for d in (-1, 1):
                    cv2.line(img, (x - 8, y - 8 * d), (x + 8, y + 8 * d), (40, 40, 255), 3, cv2.LINE_AA)

    def _draw_triad(self, img):
        live = self.drag["handle"] if self.drag and "handle" in self.drag else self.hover
        hs = self._handles()
        base, img = img, img.copy()                     # drawn translucent, so the grasp shows
        for (kind, i), pts in sorted(hs.items(), key=lambda kv: kv[0][0] == "axis"):   # rings first
            hot = live == (kind, i)
            col, th = (HOVER_BGR, 4) if hot else (AXIS_BGR[i], 2)
            pts = np.round(pts).astype(np.int32)
            if kind == "ring":
                cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA)
            else:
                cv2.arrowedLine(img, tuple(pts[0]), tuple(pts[1]), col, th + 1, cv2.LINE_AA, tipLength=0.22)
        if hs:
            c = tuple(np.round(self.view.project(self.sess.targets[self.active][0])).astype(int))
            cv2.circle(img, c, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(img, 0.7, base, 0.3, 0, base)

    def _help(self, img):
        # The Mouse / Keys tables of the module docstring, as two columns.
        body = __doc__.split("Mouse (3D view):")[1].split("An episode")[0]
        rows = [ln.strip().split("  ", 1) for ln in body.splitlines()
                if ln.strip() and not ln.strip().endswith(":")]
        h = 26 + 19 * len(rows)
        over = img.copy()
        cv2.rectangle(over, (20, 90), (img.shape[1] - 20, 90 + h), (0, 0, 0), -1)
        cv2.addWeighted(over, 0.75, img, 0.25, 0, img)
        for i, row in enumerate(rows):
            for x, cell in zip((34, 230), row):
                cv2.putText(img, cell.strip(), (x, 112 + 19 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                            (235, 235, 235), 1)

    # ------------------------------------------------------------ window size
    def _layout_panel(self, height: int):
        side = min(height // 2 - 8, 540)
        if self.top1_r is not None and self.top1_r.height == side:
            return
        for r in (self.top1_r, self.wrist_r):
            if r is not None:
                close_renderer(r)
        self.panel_w = side + 20
        self.top1_r = new_renderer(self.env.model, side, side)
        self.wrist_n = side // 2 - 5
        self.wrist_r = new_renderer(self.env.model, self.wrist_n, self.wrist_n)

    def resize(self, width: int, height: int):
        """Re-render at the window's size (so the image is never scaled and mouse pixels
        stay world-exact): the side panel keeps its proportions, the 3D view gets the rest."""
        height = max(height, 360)
        self._layout_panel(height)
        width = max(width - self.panel_w, 320)
        if (width, height) != (self.view.W, self.view.H):
            self.view.resize(width, height)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.window is not None:
            self.window.fullscreen(self.fullscreen)

    def _fit_window(self):
        w, h = self.window.size()
        if w <= 1 or h <= 1 or (w, h) == (self.view.W + self.panel_w, self.view.H):
            self._pending_size = None
            return
        # Wait for the size to hold for two frames, so a drag-resize doesn't rebuild every frame.
        if self._pending_size == (w, h):
            self.resize(w, h)
            self._pending_size = None
        else:
            self._pending_size = (w, h)

    # ------------------------------------------------------------ main loop
    def run(self):
        a = self.args
        self.window = TkWindow(self, self.view.W + self.panel_w, self.view.H)
        self._pending_size = None
        if a.fullscreen:
            self.toggle_fullscreen()
        self.say(f"ready: {self.n_saved} demos in {self.rec.path}. r starts an episode, ? shows keys")
        dt = 1.0 / (a.hz * a.playback)
        t_next = time.perf_counter()
        try:
            while self.running:
                now = time.perf_counter()
                while now >= t_next:                    # sim ticks due (catch up at most 2)
                    self.tick()
                    t_next += dt
                    if now - t_next > 2 * dt:
                        t_next = now + dt
                self._fit_window()
                self.window.show(self.draw())           # also delivers mouse / key events
                if self.window.closed:
                    self.running = False
        except KeyboardInterrupt:                       # ctrl-c in the terminal: quit (and save)
            pass
        finally:
            if self.sess.recording and len(self.rec):
                self.finish(True, " (on exit)")
            self.rec.wait()
            self.window.close()
        print(f"[keyframe] {self.n_saved} demos in {self.rec.path}", flush=True)


class TkWindow:
    """The app's window. Tk rather than cv2.imshow: OpenCV's Qt window zooms the picture
    on the mouse wheel and then pans it on left drag, on top of the app's own handling.
    Tk events are turned into the cv2-style calls App.on_mouse / App.on_key take (keys
    as X keysyms, which on_key already understands)."""
    # Tk event.state bits
    SHIFT, CTRL = 0x1, 0x4
    BUTTONS = {1: (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_FLAG_LBUTTON),
               2: (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_MBUTTONUP, cv2.EVENT_FLAG_MBUTTON),
               3: (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP, cv2.EVENT_FLAG_RBUTTON)}

    def __init__(self, app, width: int, height: int):
        import tkinter as tk
        self.app, self.closed, self.held = app, False, 0
        self.root = tk.Tk()
        self.root.title(WINDOW)
        self.root.geometry(f"{width}x{height}")
        self.root.configure(background="black")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.photo = tk.PhotoImage(width=width, height=height)
        self.label = tk.Label(self.root, image=self.photo, borderwidth=0, highlightthickness=0,
                              anchor="nw", background="black")
        self.label.pack(fill="both", expand=True)
        for b in (1, 2, 3):
            self.label.bind(f"<ButtonPress-{b}>", lambda e, b=b: self._button(e, b, True))
            self.label.bind(f"<ButtonRelease-{b}>", lambda e, b=b: self._button(e, b, False))
        self.label.bind("<Button-4>", lambda e: self._wheel(e, 120))      # X11 wheel
        self.label.bind("<Button-5>", lambda e: self._wheel(e, -120))
        self.label.bind("<MouseWheel>", lambda e: self._wheel(e, e.delta))  # Windows / macOS
        self.label.bind("<Motion>", self._motion)
        self.root.bind("<KeyPress>", self._key)
        self.root.focus_force()

    def _flags(self, e) -> int:
        f = self.held
        if e.state & self.SHIFT:
            f |= cv2.EVENT_FLAG_SHIFTKEY
        if e.state & self.CTRL:
            f |= cv2.EVENT_FLAG_CTRLKEY
        return f

    def _button(self, e, b, down):
        ev_down, ev_up, flag = self.BUTTONS[b]
        self.held = self.held | flag if down else self.held & ~flag
        self.app.on_mouse(ev_down if down else ev_up, e.x, e.y, self._flags(e))

    def _motion(self, e):
        self.app.on_mouse(cv2.EVENT_MOUSEMOVE, e.x, e.y, self._flags(e))

    def _wheel(self, e, delta):
        self.app.on_mouse(cv2.EVENT_MOUSEWHEEL, e.x, e.y, (int(np.sign(delta)) * 120) << 16 | self._flags(e))

    def _key(self, e):
        if e.keysym_num and not e.keysym.startswith(("Shift", "Control", "Alt", "Super", "Meta")):
            self.app.on_key(e.keysym_num)

    def size(self):
        return self.label.winfo_width(), self.label.winfo_height()

    def fullscreen(self, on: bool):
        self.root.attributes("-fullscreen", on)

    def show(self, bgr: np.ndarray):
        if self.closed:
            return
        h, w = bgr.shape[:2]
        ppm = f"P6 {w} {h} 255 ".encode() + np.ascontiguousarray(bgr[:, :, ::-1]).tobytes()
        self.photo.configure(data=ppm, format="ppm", width=w, height=h)
        try:
            self.root.update()                          # runs the event handlers
        except Exception:                               # the window went away mid-update
            self.closed = True

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="chem_vial_keyframe", help="data/teleop/<dataset>.hdf5")
    ap.add_argument("--out", default=None, help="explicit .hdf5 path (overrides --dataset)")
    ap.add_argument("--layout", default="weigh", choices=("weigh", "task", "video"))
    ap.add_argument("--clutter", action="store_true")
    ap.add_argument("--start-door-open", default=None, choices=(None, "left", "right", "top"))
    ap.add_argument("--hz", type=float, default=20.0, help="control = obs = action rate")
    ap.add_argument("--image-size", type=int, default=224, help="recorded image size")
    ap.add_argument("--speed", type=float, default=1.0, help="keyframe move speed multiplier")
    ap.add_argument("--playback", type=float, default=1.0,
                    help="execution speed vs real time on screen (recorded data is unaffected)")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=760)
    ap.add_argument("--fullscreen", action="store_true", help="start fullscreen (F11 toggles)")
    ap.add_argument("--max-episode-s", type=float, default=300.0, help="recorded seconds")
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    App(parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
