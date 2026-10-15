"""CONTRACT
Input: asset names on argv (folders under chem_rl_sim/assets/), optional --out DIR.
Output: <out>/<name>_preview.png, 1200x400: three views (front, side, 45 deg above)
  of the asset dropped on a grey plane with a 1 cm checker, visual geoms on the left
  half of each panel and the CoACD collision hulls (opaque, random colors) on the
  right. Title bar: name, extents in mm, hull count, mass. Default out:
  work/previews/.
Consumer: a human (or agent reading the PNG) checks upright orientation, plausible
  size against the 1 cm checker, and that the hulls keep a cup's cavity open.
Reference output: work/previews/*_preview.png (not committed).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chem_rl_sim.assets import COLLISION_GROUP, add_asset, load_meta  # noqa: E402

W, H = 400, 360


def render(name: str) -> Image.Image:
    meta = load_meta(name)
    ext = np.array(meta["extents_m"])
    size = float(ext.max())
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.stat.extent = size * 2
    spec.add_texture(name="chk", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.55, 0.55, 0.55], rgb2=[0.7, 0.7, 0.7], width=64, height=64)
    m = spec.add_material(name="chk", texrepeat=[50, 50])
    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "chk"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.5, 0.5, 0.01],
                            material="chk")
    spec.worldbody.add_light(pos=[0.3, -0.3, 1], dir=[-0.3, 0.3, -1])
    add_asset(spec, spec.worldbody, name, "vis", pos=[-ext[0] * 0.7, 0, 0], free=False)
    hulls = add_asset(spec, spec.worldbody, name, "col", pos=[ext[0] * 0.7, 0, 0], free=False)
    rng = np.random.default_rng(0)
    for g in hulls.geoms:
        if g.group == COLLISION_GROUP:
            g.rgba = [*rng.uniform(0.2, 1, 3), 1]
            g.group = 0
        else:
            g.rgba = [0, 0, 0, 0]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, ext[2] / 2]
    cam.distance = max(size * 3.2, ext[0] * 3)
    panels = []
    for az, el in ((90, -5), (0, -5), (45, -45)):
        cam.azimuth, cam.elevation = az, el
        r.update_scene(data, cam)
        panels.append(r.render())
    img = Image.fromarray(np.hstack(panels))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    d.text((6, 3), f"{name}  extents {np.round(ext * 1000).astype(int)} mm  "
                   f"hulls {len(meta['collisions'])}  mass {meta['mass']} kg  "
                   f"(left: visual, right: collision; checker 1 cm)", fill=(255, 255, 255))
    return img


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "work" / "previews")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for n in a.names:
        path = a.out / f"{n}_preview.png"
        render(n).save(path)
        print(path)
