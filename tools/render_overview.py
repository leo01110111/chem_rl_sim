"""CONTRACT
Input: --layout task|video (default task), --hand robotiq|wuji|sharpa (default robotiq),
  --settle SECONDS of physics before the render (default 0.5), --out PATH.
Output: one PNG (default docs/env.png), 1600x800 RGB, two 800x800 panels: a wide
  free-camera shot of the whole gantry (both UR7e arms, bench, balance) and a
  close-up of the balance and props on the bench. Not normalized; plain 8-bit.
Consumer: humans reading README.md, which embeds docs/env.png.
Conventions: world Z up, the bench is in front of the gantry toward -y; camera
  lookat points were chosen by eye from the render, no measured check.
Reference output: docs/env.png (committed).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chem_rl_sim.robot import DEFAULT_EFFECTOR, EFFECTORS  # noqa: E402
from chem_rl_sim.scene import LAYOUTS, build_scene  # noqa: E402

# (lookat, distance, azimuth deg, elevation deg)
VIEWS = [
    ((0.0, 0.0, 0.9), 2.6, 135, -25),
    ((-0.05, 0.0, 0.80), 0.9, -60, -40),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layout", choices=list(LAYOUTS), default="task")
    p.add_argument("--hand", choices=EFFECTORS, default=DEFAULT_EFFECTOR)
    p.add_argument("--settle", type=float, default=0.5)
    p.add_argument("--out", default=str(ROOT / "docs" / "env.png"))
    a = p.parse_args()
    m, d = build_scene(a.layout, a.hand)
    for _ in range(int(a.settle / m.opt.timestep)):
        mujoco.mj_step(m, d)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, 800)
    m.vis.global_.offheight = max(m.vis.global_.offheight, 800)
    r = mujoco.Renderer(m, 800, 800)
    img = Image.new("RGB", (800 * len(VIEWS), 800))
    for i, (lookat, dist, az, el) in enumerate(VIEWS):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        r.update_scene(d, cam)
        img.paste(Image.fromarray(r.render()), (800 * i, 0))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
