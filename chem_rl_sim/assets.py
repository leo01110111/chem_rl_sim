"""CONTRACT
Input: an asset folder chem_rl_sim/assets/<name>/ written by tools/mesh_to_mjcf.py
  (asset.json + visual_*.obj [+ texture_*.png] + collision_*.obj; Z up, meters,
  origin at the bottom-center of the bounding box).
Output: add_asset(spec, parent_body, name, prefix, pos, quat, free) mutates an
  mjSpec: registers the meshes/textures/materials once per spec (names prefixed by
  the asset name) and adds a body `prefix` under parent_body at pos/quat (world
  frame when parent is the worldbody), optionally with a free joint `<prefix>_joint`.
  Visual geoms: contype=conaffinity=0, group 2. Collision geoms: group 3, rgba
  alpha 0, mass split evenly over the hulls so the body mass equals asset.json mass.
Consumer: chem_rl_sim/scene.py places every prop through this; the env reads the
  body/joint names (`<prefix>`, `<prefix>_joint`).
Convention: object rests on its own z=0, so pos z = surface height puts it on the
  surface (checked in tests/test_scene.py: settled z drift < 2 mm).
Reference output: none; see tests/test_scene.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco

ASSET_ROOT = Path(__file__).resolve().parent / "assets"

VISUAL_GROUP = 2
COLLISION_GROUP = 3


def load_meta(name: str) -> dict:
    return json.loads((ASSET_ROOT / name / "asset.json").read_text())


def _register(spec: mujoco.MjSpec, name: str, meta: dict) -> None:
    if any(m.name == f"{name}_visual_0" for m in spec.meshes):
        return
    folder = ASSET_ROOT / name
    for i, v in enumerate(meta["visuals"]):
        spec.add_mesh(name=f"{name}_visual_{i}", file=str(folder / v["mesh"]))
        mat = spec.add_material(name=f"{name}_mat_{i}", rgba=v["rgba"],
                                **meta.get("material", {}))
        if v.get("texture"):
            spec.add_texture(name=f"{name}_tex_{i}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                             file=str(folder / v["texture"]))
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_tex_{i}"
            mat.rgba = [1, 1, 1, v["rgba"][3]]
    for i, c in enumerate(meta["collisions"]):
        spec.add_mesh(name=f"{name}_collision_{i}", file=str(folder / c))


def add_asset(spec: mujoco.MjSpec, parent: mujoco.MjsBody, name: str, prefix: str,
              pos, quat=(1, 0, 0, 0), free: bool = True,
              friction=(1.0, 0.005, 0.0001)) -> mujoco.MjsBody:
    meta = load_meta(name)
    _register(spec, name, meta)
    body = parent.add_body(name=prefix, pos=list(pos), quat=list(quat))
    if free:
        body.add_freejoint(name=f"{prefix}_joint")
    for i, _ in enumerate(meta["visuals"]):
        body.add_geom(name=f"{prefix}_visual_{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=f"{name}_visual_{i}", material=f"{name}_mat_{i}",
                      contype=0, conaffinity=0, group=VISUAL_GROUP, mass=0)
    hull_mass = meta["mass"] / max(1, len(meta["collisions"]))
    for i, _ in enumerate(meta["collisions"]):
        body.add_geom(name=f"{prefix}_collision_{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=f"{name}_collision_{i}", group=COLLISION_GROUP,
                      rgba=[0.9, 0.3, 0.3, 0.0], mass=hull_mass, friction=list(friction),
                      condim=4, solref=[0.005, 1])
    return body
