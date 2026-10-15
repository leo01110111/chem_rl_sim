"""CONTRACT
Input: one source mesh file (.glb/.gltf/.obj/.stl; glTF is +Y up) plus
  --name           asset name -> output folder chem_rl_sim/assets/<name>/
  --keep/--drop    regexes on scene node names, to cut one object out of a pack
  --up {y,z}       source up axis (default y, the glTF convention)
  --fit AXIS=M     uniform rescale so the object's world extent along AXIS
                   (x|y|z, after the up-axis fix) equals M meters. This is the
                   only scale that is trusted; the authored units are not (seen
                   m, cm and inch in the same search).
  --fit-xy M       optional second, horizontal fit: rescale x,y (not z) so the
                   larger horizontal extent is M meters -- for meshes whose
                   proportions are wrong (the dropper bulb was 1.7x too fat)
  --mass KG, --rgba "r g b a" (override every visual color), --glass (alpha 0.25)
  --part-rgba RE=r,g,b,a  color override for parts whose label matches RE (repeatable)
  --hull-per-part  skip CoACD; one convex hull per source part (right when each
                   part is itself convex, e.g. bulb + stem: CoACD merged those
                   into one cone-shaped hull far fatter than the stem)
  --hulls N        CoACD max convex hulls (default 16); --threshold (default 0.04,
                   lower = more faithful, more hulls)
Output: chem_rl_sim/assets/<name>/
  visual_<i>.obj (+ texture_<i>.png when the material has one), Z up, meters,
    origin at the bottom-center of the bounding box (object rests on z=0)
  collision_<i>.obj  convex hulls from CoACD in the same frame
  asset.json  {name, source, scale, extents_m, mass, visuals:[{mesh, rgba, texture}],
               collisions:[...], provenance:{...from assets/raw/<uid>.json}}
Consumer: chem_rl_sim/assets.py:add_asset reads asset.json and attaches the
  meshes to an mjSpec body (visual geoms contype=0, collision geoms group 3).
Convention check: after --fit the reported extents_m along the fitted axis equals
  the requested value to <1e-6 (asserted below).
Reference output: chem_rl_sim/assets/*/asset.json, committed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import coacd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "chem_rl_sim" / "assets"
RAW = ROOT / "assets" / "raw"

Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])


def load_parts(path: Path, keep: str | None, drop: str | None):
    scene = trimesh.load(path, force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        T, geom_name = scene.graph[node]
        label = f"{node}|{geom_name}"
        if keep and not re.search(keep, label):
            continue
        if drop and re.search(drop, label):
            continue
        g = scene.geometry[geom_name].copy()
        g.apply_transform(T)
        parts.append((label, g))
    if not parts:
        names = [f"{n}|{scene.graph[n][1]}" for n in scene.graph.nodes_geometry]
        raise SystemExit(f"no parts matched; nodes are:\n  " + "\n  ".join(names))
    return parts


def material_of(g: trimesh.Trimesh):
    rgba, tex = [0.8, 0.8, 0.8, 1.0], None
    mat = getattr(g.visual, "material", None)
    if mat is None:
        return rgba, tex
    if hasattr(mat, "baseColorFactor") and mat.baseColorFactor is not None:
        rgba = (np.asarray(mat.baseColorFactor, float) / 255.0).tolist()
    elif hasattr(mat, "diffuse"):
        rgba = (np.asarray(mat.diffuse, float) / 255.0).tolist()
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is not None and getattr(g.visual, "uv", None) is not None:
        tex = img
    return rgba, tex


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--keep")
    p.add_argument("--drop")
    p.add_argument("--up", choices=["y", "z"], default="y")
    p.add_argument("--fit", required=True, help="AXIS=meters, e.g. z=0.055")
    p.add_argument("--fit-xy", type=float)
    p.add_argument("--part-rgba", action="append", default=[])
    p.add_argument("--mass", type=float, required=True)
    p.add_argument("--rgba")
    p.add_argument("--glass", action="store_true")
    p.add_argument("--hulls", type=int, default=16)
    p.add_argument("--hull-per-part", action="store_true")
    p.add_argument("--threshold", type=float, default=0.04)
    a = p.parse_args()

    parts = load_parts(a.src, a.keep, a.drop)
    if a.up == "y":
        for _, g in parts:
            g.apply_transform(Y_UP_TO_Z_UP)

    allv = np.vstack([g.vertices for _, g in parts])
    lo, hi = allv.min(0), allv.max(0)
    axis, target = a.fit.split("=")
    ax = "xyz".index(axis)
    scale = float(target) / (hi - lo)[ax]
    offset = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    for _, g in parts:
        g.apply_translation(offset)
        g.apply_scale(scale)
    if a.fit_xy:
        allv = np.vstack([g.vertices for _, g in parts])
        sxy = a.fit_xy / np.ptp(allv[:, :2], axis=0).max()
        for _, g in parts:
            g.vertices[:, :2] *= sxy
    allv = np.vstack([g.vertices for _, g in parts])
    extents = (allv.max(0) - allv.min(0))
    assert abs(extents[ax] - float(target)) < 1e-6, extents

    out = OUT_ROOT / a.name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()

    visuals = []
    for i, (label, g) in enumerate(parts):
        rgba, tex = material_of(g)
        if a.rgba:
            rgba, tex = [float(v) for v in a.rgba.split()], None
        if a.glass:
            rgba = [*rgba[:3], 0.25]
            tex = None
        for rule in a.part_rgba:
            pat, col = rule.split("=")
            if re.search(pat, label):
                rgba, tex = [float(v) for v in col.split(",")], None
        mesh_file = f"visual_{i}.obj"
        entry = {"mesh": mesh_file, "rgba": rgba, "texture": None, "part": label}
        if tex is not None:
            tex_file = f"texture_{i}.png"
            tex.convert("RGB").save(out / tex_file)
            entry["texture"] = tex_file
            plain = trimesh.Trimesh(g.vertices, g.faces, process=False,
                                    visual=trimesh.visual.TextureVisuals(uv=g.visual.uv))
        else:
            plain = trimesh.Trimesh(g.vertices, g.faces, process=False)
        plain.export(out / mesh_file, include_texture=False)
        visuals.append(entry)

    if a.hull_per_part:
        hulls = [(h.vertices, h.faces) for h in
                 (trimesh.Trimesh(g.vertices, g.faces).convex_hull for _, g in parts)]
    else:
        whole = trimesh.util.concatenate([trimesh.Trimesh(g.vertices, g.faces) for _, g in parts])
        whole.merge_vertices()
        hulls = coacd.run_coacd(coacd.Mesh(whole.vertices, whole.faces),
                                threshold=a.threshold, max_convex_hull=a.hulls,
                                preprocess_mode="auto")
    collisions = []
    for i, (v, f) in enumerate(hulls):
        name = f"collision_{i}.obj"
        trimesh.Trimesh(v, f).export(out / name)
        collisions.append(name)

    prov = {}
    raw_meta = RAW / f"{a.src.stem}.json"
    if raw_meta.exists():
        prov = json.loads(raw_meta.read_text())
    meta = {"name": a.name, "source": str(a.src.relative_to(ROOT)) if a.src.is_absolute() else str(a.src),
            "parts": [lbl for lbl, _ in parts], "up": a.up, "scale": scale,
            "extents_m": extents.round(5).tolist(), "mass": a.mass,
            "visuals": visuals, "collisions": collisions, "provenance": prov}
    (out / "asset.json").write_text(json.dumps(meta, indent=2))
    print(f"{a.name}: scale {scale:.5g}, extents {extents.round(4)} m, "
          f"{len(visuals)} visual parts, {len(collisions)} hulls -> {out}")


if __name__ == "__main__":
    main()
