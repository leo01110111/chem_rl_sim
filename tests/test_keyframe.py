"""Keyframe teleop (chem_rl_sim/teleop/keyframe.py, keyframe_app.py) against the real chem
model, driven through the same calls the window makes. The whole weighing task -- open the
top door, pick vial_a, place it on the pan, close the door -- must reach env success with
keyframes, gripper toggles and snaps only; plans must flag what they hit; undo must
restore the sim and the recording; the 3D view's mouse rays must land where they point."""
import cv2
import h5py
import numpy as np
import pytest

from chem_rl_sim.env import VialWeighingEnv
from chem_rl_sim.teleop.keyframe import Session
from chem_rl_sim.teleop.keyframe_script import run_all as run, weigh
from chem_rl_sim.teleop.recorder import Recorder

L, R = "left", "right"


def go(s, side, pos, rot=None):
    s.set_target(side, pos, rot)
    plan = s.plan()
    assert s.execute(force=True)[0]
    run(s)
    return plan


@pytest.fixture
def env():
    e = VialWeighingEnv(layout="weigh", clutter=False, max_episode_steps=10 ** 9)
    yield e
    e.close()


def test_whole_task_by_keyframes(env):
    s = Session(env)
    s.reset(seed=0)
    st = weigh(s)
    # Snaps + the planner's own up-across-down routing: every step plans clean.
    assert all(v == "ok" for v in st.values()), st
    info = s.info
    assert info["vial_on_pan"] and info["vial_upright"] and info["doors_closed"], info
    assert info["success"] == 1, info
    assert abs(info["balance_g"] - 20.0) < 0.05


def test_servo_holds_the_keyframe(env):
    """Gravity feed-forward: the pinch site settles on the target, not ~9 mm below it."""
    s = Session(env)
    s.reset(seed=0)
    s.snap(L, "vial")
    s.execute()
    run(s)
    err = np.linalg.norm(s.targets[L][0] - s.arms[L].tool_pose()[0])
    assert err < 1.5e-3, f"{1e3 * err:.1f} mm"


def test_plan_flags_what_it_hits(env):
    s = Session(env)
    s.reset(seed=0)
    s.snap(L, "vial")
    assert s.plan().status == "ok"
    # Right gripper onto the left gripper: blocked (the other arm / the balance frame).
    s.target_to_current(L)
    s.set_target(R, s.arms[L].tool_pose()[0])
    plan = s.plan()
    assert plan.status == "blocked" and plan.reasons
    ok, msg = s.execute()
    assert not ok and msg.startswith("blocked")
    # Out of reach: blocked by IK.
    s.target_to_current(R)
    s.set_target(L, [0.6, 0.5, 1.3])
    assert any("can't reach" in r for r in s.plan().reasons)


def test_undo_restores_sim_and_recording(env, tmp_path):
    rec = Recorder(tmp_path / "kf.hdf5", {"env_name": "ChemVialWeighing-v0"}, 20.0, "robotiq")
    s = Session(env, rec)
    s.begin_episode(seed=0)
    s.snap(L, "vial"); s.execute(); run(s)
    s.snap(L, "vial"); s.execute(); run(s)
    s.grip(L, True); run(s)
    n, qpos = len(rec), env.data.qpos.copy()
    go(s, L, s.arms[L].tool_pose()[0] + [0, 0, 0.12])
    assert env.vial_pos()[2] > 0.775 + 0.1 and len(rec) > n
    assert s.undo()[0]
    assert len(rec) == n and len(rec.obs) == n + 1
    assert np.allclose(env.data.qpos, qpos)
    # Follow mode: the arm tracks a target edit live, as one recorded segment.
    s.follow = True
    s.set_target(L, s.targets[L][0] + [0, 0, 0.03])
    m = run(s)
    assert m > 0 and np.linalg.norm(s.targets[L][0] - s.arms[L].tool_pose()[0]) < 2e-3
    assert s.save_episode() == n + m
    rec.wait()
    assert rec.last_error is None
    with h5py.File(tmp_path / "kf.hdf5") as f:
        g = f["data/demo_0"]
        T = g["actions"].shape[0]
        assert T == n + m and g["actions"].shape[1] == 14
        assert np.abs(g["actions"][:]).max() <= 1.0
        op = g["operator"]
        assert op["goal_pos"].shape == (T, 2, 3) and op["goal_quat"].shape == (T, 2, 4)
        assert set(np.unique(op["kind"][:])) == {0, 1, 2}
        assert np.all(np.diff(op["segment"][:]) >= 0)
        assert g["obs/top1_image"].shape[0] == T and g["next_obs/top1_image"].shape[0] == T


def test_mouse_ray_and_drag(tmp_path):
    from chem_rl_sim.teleop.keyframe_app import App, parse_args
    app = App(parse_args(["--out", str(tmp_path / "x.hdf5"), "--seed", "0"]))
    s = app.sess
    p = s.targets[L][0].copy()
    uv = app.view.project(p)
    o, d = app.view.ray(*uv)
    assert np.linalg.norm(np.cross(p - o, d)) < 1e-9
    u, v = (int(round(x)) for x in uv)
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, u, v, 0)
    app.on_mouse(cv2.EVENT_MOUSEMOVE, u + 150, v, cv2.EVENT_FLAG_LBUTTON)
    app.on_mouse(cv2.EVENT_LBUTTONUP, u + 150, v, 0)
    q = s.targets[L][0]
    assert q[2] == pytest.approx(p[2]) and q[0] > p[0] + 0.05      # right on screen = +x
    assert np.allclose(app.view.project(q), [u + 150, v], atol=1.0)
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, u, v, cv2.EVENT_FLAG_SHIFTKEY)
    app.on_mouse(cv2.EVENT_MOUSEMOVE, u, v - 40, cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_SHIFTKEY)
    assert s.targets[L][0][2] > q[2] + 0.02 and np.allclose(s.targets[L][0][:2], q[:2])
    frame = app.draw()
    assert frame.shape[0] == app.args.height and frame.shape[1] == app.args.width + app.panel_w


def test_view_controls(tmp_path):
    """Orbit / pan / zoom by mouse and keys, preset views, and drags still land under the
    cursor from any view (mjv_moveCamera's signature changed in MuJoCo 3.x)."""
    from chem_rl_sim.teleop.keyframe_app import VIEWS, App, parse_args
    app = App(parse_args(["--out", str(tmp_path / "x.hdf5"), "--seed", "0"]))
    c = app.view.cam
    az, el, dist, look = c.azimuth, c.elevation, c.distance, np.array(c.lookat)
    app.on_mouse(cv2.EVENT_RBUTTONDOWN, 300, 300, 0)
    app.on_mouse(cv2.EVENT_MOUSEMOVE, 360, 330, cv2.EVENT_FLAG_RBUTTON)
    app.on_mouse(cv2.EVENT_RBUTTONUP, 360, 330, 0)
    assert c.azimuth != az and c.elevation != el
    app.on_mouse(cv2.EVENT_MBUTTONDOWN, 300, 300, 0)
    app.on_mouse(cv2.EVENT_MOUSEMOVE, 340, 300, cv2.EVENT_FLAG_MBUTTON)
    assert not np.allclose(c.lookat, look)
    app.on_mouse(cv2.EVENT_MOUSEWHEEL, 300, 300, 120 << 16)          # wheel up: zoom in
    assert c.distance < dist
    app.on_key(ord("-"))
    app.on_key(ord("-"))
    assert c.distance > dist
    a0 = c.azimuth
    app.on_key(ord("m"))
    assert c.azimuth == pytest.approx((a0 + 15) % 360)
    cam = (c.azimuth, c.elevation, c.distance, tuple(c.lookat))
    for flags in (0, cv2.EVENT_FLAG_ALTKEY, cv2.EVENT_FLAG_SHIFTKEY, cv2.EVENT_FLAG_CTRLKEY):
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, 300, 300, flags)       # left drag: never the view
        app.on_mouse(cv2.EVENT_MOUSEMOVE, 360, 330, cv2.EVENT_FLAG_LBUTTON | flags)
        app.on_mouse(cv2.EVENT_LBUTTONUP, 360, 330, flags)
    assert (c.azimuth, c.elevation, c.distance, tuple(c.lookat)) == cam
    for i, name in enumerate(VIEWS):
        app.on_key(ord(str(3 + i)))
        assert (c.azimuth, c.elevation) == VIEWS[name][:2]
        p = app.sess.targets[L][0].copy()
        u, v = (int(round(x)) for x in app.view.project(p))
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, u, v, 0)
        app.on_mouse(cv2.EVENT_MOUSEMOVE, u + 60, v, cv2.EVENT_FLAG_LBUTTON)
        app.on_mouse(cv2.EVENT_LBUTTONUP, u + 60, v, 0)
        q = app.sess.targets[L][0]
        if not np.allclose(q, p):                    # skipped only if clipped to the workspace
            assert np.allclose(app.view.project(q), [u + 60, v], atol=1.5), name
    app.on_key(ord("8"))
    assert np.allclose(c.lookat, app.sess.targets[L][0])


def test_window_resize(tmp_path):
    """Fullscreen / resize re-renders at the window's size, so mouse pixels stay exact."""
    from chem_rl_sim.teleop.keyframe_app import App, parse_args
    app = App(parse_args(["--out", str(tmp_path / "x.hdf5"), "--seed", "0"]))
    for w, h in [(1920, 1080), (800, 500)]:
        app.resize(w, h)
        assert app.draw().shape == (h, w, 3)
        p = app.sess.targets[L][0].copy()
        u, v = (int(round(x)) for x in app.view.project(p))
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, u, v, 0)
        app.on_mouse(cv2.EVENT_MOUSEMOVE, u + 50, v, cv2.EVENT_FLAG_LBUTTON)
        app.on_mouse(cv2.EVENT_LBUTTONUP, u + 50, v, 0)
        assert np.allclose(app.view.project(app.sess.targets[L][0]), [u + 50, v], atol=1.5)


def _drag(app, pts, flags=0):
    (x, y), *rest = [tuple(int(round(v)) for v in q) for q in pts]
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, flags)
    for x, y in rest:
        app.on_mouse(cv2.EVENT_MOUSEMOVE, x, y, cv2.EVENT_FLAG_LBUTTON | flags)
    app.on_mouse(cv2.EVENT_LBUTTONUP, x, y, flags)


def test_triad(tmp_path):
    """Arrows translate along their axis only; rings turn about their axis only (world
    and gripper frames); the pick follows what is drawn."""
    from chem_rl_sim.teleop.keyframe_app import App, parse_args
    app = App(parse_args(["--out", str(tmp_path / "x.hdf5"), "--seed", "0"]))
    s = app.sess
    p0, R0 = (x.copy() for x in s.targets[L])
    for world in (True, False):
        app.triad_world = world
        axes = np.eye(3) if world else R0
        for i in range(3):
            s.set_target(L, p0, R0)
            a, b = app._handles()["axis", i]
            u = (b - a) / np.linalg.norm(b - a)
            assert app.pick(*np.round(b - 5 * u)) == ("axis", i)
            _drag(app, [b - 5 * u, b + 35 * u])
            d = s.targets[L][0] - p0
            assert d @ axes[:, i] > 0.02 and np.linalg.norm(np.cross(d, axes[:, i])) < 1e-6
            assert np.allclose(s.targets[L][1], R0)
        for i in range(3):
            s.set_target(L, p0, R0)
            ring = app._handles()["ring", i]
            k = max(range(len(ring) - 12), key=lambda j: np.linalg.norm(ring[j + 6] - ring[j]))
            if app.pick(*np.round(ring[k])) != ("ring", i):
                continue                                # hidden behind another handle here
            _drag(app, ring[k:k + 7])
            dR = s.targets[L][1] @ R0.T
            w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            assert np.linalg.norm(w) > 0.1 and abs(w @ axes[:, i]) / np.linalg.norm(w) > 0.999
            assert np.allclose(s.targets[L][0], p0)


def test_tk_events_reach_the_app(tmp_path):
    """TkWindow's translation of Tk events into on_mouse / on_key (no display needed)."""
    from types import SimpleNamespace as E
    from chem_rl_sim.teleop.keyframe_app import TkWindow
    calls = []
    app = E(on_mouse=lambda *a: calls.append(("m", *a)), on_key=lambda k: calls.append(("k", k)))
    w = object.__new__(TkWindow)
    w.app, w.held = app, 0
    w._button(E(x=5, y=6, state=0x1), 1, True)
    w._motion(E(x=7, y=8, state=0x1))
    w._button(E(x=7, y=8, state=0), 1, False)
    w._button(E(x=1, y=1, state=0), 3, True)
    w._wheel(E(x=1, y=1, state=0), -120)
    w._key(E(keysym_num=65293, keysym="Return"))
    w._key(E(keysym_num=65505, keysym="Shift_L"))
    w._key(E(keysym_num=ord("F"), keysym="F"))
    assert calls[0] == ("m", cv2.EVENT_LBUTTONDOWN, 5, 6, cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_SHIFTKEY)
    assert calls[1] == ("m", cv2.EVENT_MOUSEMOVE, 7, 8, cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_SHIFTKEY)
    assert calls[2] == ("m", cv2.EVENT_LBUTTONUP, 7, 8, 0)
    assert calls[3] == ("m", cv2.EVENT_RBUTTONDOWN, 1, 1, cv2.EVENT_FLAG_RBUTTON)
    assert calls[4][1] == cv2.EVENT_MOUSEWHEEL and np.int32(calls[4][4]) >> 16 < 0
    assert calls[5:] == [("k", 65293), ("k", ord("F"))]


def test_snaps_route_and_gap(env):
    """Operator aids: the door snap grasps high enough that releasing doesn't drag the
    door; a move that would clip the enclosure is routed up-across-down; the readout
    measures from a carried vial's base to what is below it; s points straight down."""
    s = Session(env)
    s.reset(seed=0)
    s.snap(R, "door")
    gap, name, _ = s.gap_below(R)
    assert "balance door top" in name and gap > 0.05
    s.set_target(L, s.targets[L][0], np.eye(3))
    s.point_down(L)
    assert s.targets[L][1][:, 2] == pytest.approx([0, 0, -1], abs=1e-6)
    s.target_to_current(L)
    weigh_steps = weigh(s)                              # the whole task, snaps only
    assert s.info["success"] == 1 and all(v == "ok" for v in weigh_steps.values())
    # Door open, vial in hand, from home: the way into the balance is routed up and over.
    e2 = VialWeighingEnv(layout="weigh", clutter=False, start_door_open="top", max_episode_steps=10 ** 9)
    s = Session(e2)
    s.reset(seed=0)
    s.snap(L, "vial")
    s.execute(); run(s)
    s.snap(L, "vial")
    s.execute(); run(s)
    s.grip(L, True); run(s)
    s.snap(L, "home")
    plan = s.plan()
    assert plan.status == "ok", plan.reasons
    s.execute(); run(s)
    s.snap(L, "pan")
    gap, name, _ = s.gap_below(L)
    assert name == "balance pan" and gap == pytest.approx(0.12 + 0.003, abs=0.01)
    plan = s.plan()
    assert plan.status == "ok" and len(plan.vias[L]) == 2
    e2.close()
