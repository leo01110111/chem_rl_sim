"""CONTRACT
Input: --layout video|task (default both), --settle SECONDS of physics before the
  render (default 1.0), --hand wuji|sharpa.
Output: work/scene_<layout>.png, three panels, 720 px tall: the video frame at
  3:57 (work/frames/f_068.jpg, if present), the sim from a free camera placed by
  eye near the phone viewpoint (operator side = -y, 60 deg down, swung 30 deg to
  +x to clear the gantry column that occupies the operator spot) and rotated
  90 deg clockwise because the phone was held rolled (bench front = image left),
  and the gantry's top1 camera. Overlay: layout, balance reading
  (g, tared on an empty pan before props are placed... see note), max prop drift
  during settling (mm).
  Note: the reading shown is Fz(pan) minus the empty-pan Fz measured on a
  separate build without props, so it is the true mass on the pan.
Consumer: a human comparing against work/frames/f_068.jpg.
Reference output: work/scene_video.png, work/scene_task.png (not committed).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chem_rl_sim.balance import pan_force_z, pan_reading_g  # noqa: E402
from chem_rl_sim.scene import LAYOUTS, build_scene  # noqa: E402


def empty_pan_force(hand: str) -> float:
    m, d = build_scene("task", hand, clutter=False)
    for _ in range(200):
        mujoco.mj_step(m, d)
    return pan_force_z(m, d)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layout", choices=list(LAYOUTS), nargs="*", default=list(LAYOUTS))
    p.add_argument("--settle", type=float, default=1.0)
    p.add_argument("--hand", default="wuji")
    a = p.parse_args()
    tare = empty_pan_force(a.hand)
    for layout in a.layout:
        m, d = build_scene(layout, a.hand)
        start = {k: d.body(k).xpos.copy() for k in LAYOUTS[layout]}
        for _ in range(int(a.settle / m.opt.timestep)):
            mujoco.mj_step(m, d)
        drift = max(np.linalg.norm(d.body(k).xpos - start[k]) for k in start) * 1000
        r = mujoco.Renderer(m, 720, 800)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.02, 0.06, 0.80]
        # Swung to the +x side: the real phone position (straight in front of the
        # display) is inside the gantry column and head in the sim.
        cam.distance, cam.azimuth, cam.elevation = 0.8, 120, -60
        r.update_scene(d, cam)
        phone = Image.fromarray(r.render()).rotate(-90, expand=True).resize((648, 720))
        r.update_scene(d, "top1")
        top = Image.fromarray(r.render())
        panels = [phone, top]
        frame = ROOT / "work" / "frames" / "f_068.jpg"
        if frame.exists():
            panels.insert(0, Image.open(frame).resize((405, 720)))
        img = Image.new("RGB", (sum(p.width for p in panels), 720))
        x = 0
        for pnl in panels:
            img.paste(pnl, (x, 0))
            x += pnl.width
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, img.width, 20], fill=(0, 0, 0))
        dr.text((6, 4), f"layout={layout}  balance={pan_reading_g(m, d, tare):.2f} g  "
                        f"max prop drift over {a.settle:.1f}s settle = {drift:.1f} mm   "
                        f"(video 3:57 | sim, phone view | sim top1)", fill=(255, 255, 255))
        out = ROOT / "work" / f"scene_{layout}.png"
        img.save(out)
        print(out, f"reading {pan_reading_g(m, d, tare):.3f} g drift {drift:.1f} mm")


if __name__ == "__main__":
    main()
