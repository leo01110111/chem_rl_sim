"""CONTRACT
Input: none (dimensions are the constants below; edit and re-run).
Output: three asset folders under chem_rl_sim/assets/, same layout and asset.json
  schema as tools/mesh_to_mjcf.py writes (Z up, meters, base at z=0, centered):
    vial_20ml_open      open glass vial (cavity free for a dropper tip)
    vial_20ml_whitecap  vial + white 24-400 cap
    vial_20ml_redcap    vial + red 24-400 cap
    cap_24_400_white    loose cap, open side down
  Visual: lathe-revolved surfaces (outer + inner wall, 48 segments).
  Collision: N_WALL convex wall segments + one bottom disk (+ one cap cylinder),
    each written as its own convex OBJ, so the cavity stays open (a single convex
    hull or a coarse CoACD run would fill it).
Consumer: chem_rl_sim/assets.py:add_asset.
Dimensions: 20 mL borosilicate scintillation vial, 28 mm OD x 61 mm with cap,
  24-400 neck (Wheaton/DWK 986546 class). In the video (frame 3:57, work/frames/f_068.jpg)
  the vial OD is ~1/3 of the draft-ring diameter around the 90 mm pan -> ~30 mm,
  consistent with 28 mm. Masses: glass 0.020 kg, cap 0.004 kg (typical catalog).
Reference output: chem_rl_sim/assets/vial_*/asset.json (committed).
"""
import json
from pathlib import Path

import numpy as np
import trimesh

OUT = Path(__file__).resolve().parents[1] / "chem_rl_sim" / "assets"

OD = 0.028
WALL = 0.0012
BOTTOM_T = 0.002
BODY_H = 0.047           # straight body to shoulder
SHOULDER_H = 0.004
NECK_OD = 0.0245
NECK_H = 0.006           # threaded finish above the shoulder
VIAL_H = BODY_H + SHOULDER_H + NECK_H   # 0.057 glass height
CAP_OD = 0.0265
CAP_H = 0.0125
CAP_SEAT = 0.0045        # cap bottom sits this far below the lip (over the threads)
GLASS_MASS = 0.020
CAP_MASS = 0.004
N_WALL = 12
SEG = 48

GLASS_RGBA = [0.86, 0.93, 0.96, 0.28]
WHITE_RGBA = [0.95, 0.94, 0.90, 1.0]
RED_RGBA = [0.86, 0.08, 0.12, 1.0]


def lathe(profile: np.ndarray, seg: int = SEG) -> trimesh.Trimesh:
    """Revolve an (r, z) polyline about +z into a closed surface."""
    n = len(profile)
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    v = np.array([[r * np.cos(t), r * np.sin(t), z] for r, z in profile for t in th])
    f = []
    for i in range(n - 1):
        for j in range(seg):
            a, b = i * seg + j, i * seg + (j + 1) % seg
            c, d = a + seg, b + seg
            f += [[a, b, d], [a, d, c]]
    return trimesh.Trimesh(v, np.array(f), process=False)


def vial_profile() -> np.ndarray:
    r_o, r_n = OD / 2, NECK_OD / 2
    r_in = r_n - WALL
    return np.array([
        [0.0, 0.0], [r_o - 0.001, 0.0], [r_o, 0.001], [r_o, BODY_H],
        [r_n, BODY_H + SHOULDER_H], [r_n, VIAL_H], [r_in, VIAL_H],
        [r_in, BODY_H + SHOULDER_H], [r_o - WALL, BODY_H], [r_o - WALL, BOTTOM_T], [0.0, BOTTOM_T],
    ])


def cap_mesh(z0: float) -> trimesh.Trimesh:
    c = trimesh.creation.cylinder(radius=CAP_OD / 2, height=CAP_H, sections=SEG)
    c.apply_translation([0, 0, z0 + CAP_H / 2])
    return c


def wall_hulls() -> list[trimesh.Trimesh]:
    hulls = []
    r_o, r_i = OD / 2, OD / 2 - WALL
    dth = 2 * np.pi / N_WALL
    for k in range(N_WALL):
        t0, t1 = k * dth, (k + 1) * dth
        pts = []
        for r in (r_i, r_o):
            for t in (t0, t1):
                for z in (BOTTOM_T, VIAL_H):
                    rr = r / np.cos(dth / 2) if r == r_o else r
                    pts.append([rr * np.cos(t), rr * np.sin(t), z])
        hulls.append(trimesh.convex.convex_hull(np.array(pts)))
    bottom = trimesh.creation.cylinder(radius=OD / 2, height=BOTTOM_T, sections=24)
    bottom.apply_translation([0, 0, BOTTOM_T / 2])
    hulls.append(bottom)
    return hulls


def write(name: str, visuals: list, collisions: list, mass: float, note: str) -> None:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    vis_meta = []
    for i, (mesh, rgba) in enumerate(visuals):
        mesh.export(out / f"visual_{i}.obj")
        vis_meta.append({"mesh": f"visual_{i}.obj", "rgba": rgba, "texture": None, "part": f"part{i}"})
    col_names = []
    for i, h in enumerate(collisions):
        h.export(out / f"collision_{i}.obj")
        col_names.append(f"collision_{i}.obj")
    allv = np.vstack([m.vertices for m, _ in visuals])
    meta = {"name": name, "source": "tools/make_vial.py (parametric)", "scale": 1.0,
            "extents_m": np.ptp(allv, axis=0).round(5).tolist(), "mass": mass,
            "visuals": vis_meta, "collisions": col_names,
            "material": {"specular": 0.8, "shininess": 0.9, "reflectance": 0.1},
            "provenance": {"license": "generated", "note": note}}
    (out / "asset.json").write_text(json.dumps(meta, indent=2))
    print(name, meta["extents_m"], len(col_names), "hulls")


def main() -> None:
    glass = lathe(vial_profile())
    walls = wall_hulls()
    cap_z0 = VIAL_H - CAP_H + CAP_SEAT
    note = "20 mL scintillation vial, 28 mm OD, 24-400 finish"
    write("vial_20ml_open", [(glass, GLASS_RGBA)], walls, GLASS_MASS, note)
    for color, rgba in (("white", WHITE_RGBA), ("red", RED_RGBA)):
        cap = cap_mesh(cap_z0)
        write(f"vial_20ml_{color}cap", [(glass, GLASS_RGBA), (cap, rgba)],
              walls + [cap.convex_hull], GLASS_MASS + CAP_MASS, note + f", {color} cap")
    loose = cap_mesh(0.0)
    write("cap_24_400_white", [(loose, WHITE_RGBA)], [loose.convex_hull], CAP_MASS,
          "24-400 PP screw cap")


if __name__ == "__main__":
    main()
