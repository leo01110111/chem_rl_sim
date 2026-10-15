"""CONTRACT
Input: .glb paths on argv (raw downloads, any units, glTF +Y up), --out PNG.
Output: one PNG grid, 320x320 per model, each model normalized to unit size
  (largest extent = 1), rotated Z-up, rendered from 30 deg above at 3/4 view with
  its base colors. Label: file stem, part count, raw extents in authored units.
Consumer: a human/agent triaging search hits before running mesh_to_mjcf.py
  (is it the right object, is it upright, is it one object or a pack).
Reference output: none committed (work/ is scratch).
"""
import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw

S = 320


def render_one(path: Path) -> Image.Image:
    scene = trimesh.load(path, force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        # Planar parts (decals, labels) make MuJoCo's convex-hull step fail at compile.
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0 or g.extents.min() < 1e-7 * g.extents.max():
            continue
        g = g.copy()
        g.apply_transform(T)
        parts.append(g)
    raw = np.ptp(np.vstack([p.vertices for p in parts]), axis=0)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    allv = np.vstack([trimesh.transform_points(p.vertices, rot) for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    s = 1.0 / (hi - lo).max()
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = spec.visual.global_.offheight = S
    spec.worldbody.add_light(pos=[1, -1, 2], dir=[-1, 1, -2])
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(parts):
            v = (trimesh.transform_points(p.vertices, rot) - [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]) * s
            f = Path(tmp) / f"p{i}.obj"
            trimesh.Trimesh(v, p.faces, process=False).export(f)
            spec.add_mesh(name=f"p{i}", file=str(f))
            rgba = [0.75, 0.75, 0.75, 1]
            mat = getattr(p.visual, "material", None)
            if getattr(mat, "baseColorFactor", None) is not None:
                rgba = (np.asarray(mat.baseColorFactor) / 255).tolist()
                rgba[3] = max(rgba[3], 0.35)
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"p{i}", rgba=rgba)
        model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, S, S)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, (hi[2] - lo[2]) * s / 2]
    cam.distance, cam.azimuth, cam.elevation = 2.2, 120, -30
    r.update_scene(data, cam)
    img = Image.fromarray(r.render())
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, S, 28], fill=(0, 0, 0))
    d.text((4, 2), f"{path.stem[:16]} parts={len(parts)}", fill=(255, 255, 0))
    d.text((4, 15), f"raw ext {np.round(raw, 3).tolist()}", fill=(255, 255, 255))
    return img


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("glbs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cols", type=int, default=4)
    a = p.parse_args()
    imgs = [render_one(g) for g in a.glbs]
    rows = (len(imgs) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (S * a.cols, S * rows))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % a.cols) * S, (i // a.cols) * S))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(a.out)
