"""CONTRACT
A scripted operator for the keyframe teleop: the whole weighing task (open the top door
with the right gripper, pick vial_a with the left, set it on the pan, close the door)
issued through keyframe.Session exactly as the window's keys would -- snaps (a / v / b /
h), one drag per door slide, gripper toggles and Enter. Used by tests/test_keyframe.py
(must reach env success) and tools/keyframe_demo.py (README GIF); it is also a worked
example of a keyframe sequence that works.
  weigh(session, run=run_all, preview=None) -> {step: plan status}
    run(session)   executes queued work (default: tick until idle)
    preview()      called after each target edit, before Enter (the demo renders the ghost)
Moves between places (home -> door, vial -> pan, ...) rely on the planner's own
up-across-down routing when the straight path would touch something.
"""
from __future__ import annotations

import numpy as np

from .keyframe import Session

L, R = "left", "right"
DOOR_TRAVEL = 0.20


def run_all(s: Session) -> int:
    n = 0
    while s.tick():
        n += 1
    return n


def weigh(s: Session, run=run_all, preview=None) -> dict:
    st = {}
    preview = preview or (lambda: None)

    def enter(label):
        preview()
        st[label] = s.plan().status
        assert s.execute(force=True)[0]
        run(s)

    def snap(side, what, label):
        s.snap(side, what)
        enter(label)

    def grip(side, close):
        s.grip(side, close)
        run(s)

    def slide(dy, label):
        snap(R, "door", f"above handle ({label})")
        snap(R, "door", f"grasp handle ({label})")
        grip(R, True)
        s.set_target(R, s.targets[R][0] + [0, dy, 0])
        enter(f"door {label}")
        grip(R, False)
        snap(R, "home", f"right home after {label}")

    slide(DOOR_TRAVEL, "open")
    snap(L, "vial", "above vial")
    snap(L, "vial", "grasp vial")
    grip(L, True)
    snap(L, "pan", "above pan")
    snap(L, "pan", "place")
    grip(L, False)
    snap(L, "home", "left home")
    slide(-DOOR_TRAVEL, "close")
    return st
