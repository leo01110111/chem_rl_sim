# Assets

Every folder holds one prop in the same format: `asset.json`, `visual_*.obj`,
`collision_*.obj`. The meshes are Z up, in meters, with the origin at the
bottom-center. `chem_rl_sim/assets.py` loads them.

| asset | source | license | how it was made |
|---|---|---|---|
| `pasteur_dropper` | [Pipet/Pipette](https://sketchfab.com/3d-models/7e17299c2cdd42ea9686e48a1089790c) by **denis98** (Sketchfab, through Objaverse 1.0) | **CC-BY 4.0**: attribution required | `tools/mesh_to_mjcf.py ... --fit z=0.17 --fit-xy 0.016 --hull-per-part`, with the bulb recolored red and the stem made clear glass to match the video |
| `vial_20ml_open`, `vial_20ml_whitecap`, `vial_20ml_redcap`, `cap_24_400_white` | `tools/make_vial.py` (parametric lathe) | generated | 20 mL scintillation vial: 28 mm OD × 61 mm, 24-400 cap |
| balance | `chem_rl_sim/balance.py` (mjSpec primitives, not a mesh) | generated | Mettler Toledo ME204 class: 210 × 344 × 344 mm, 90 mm pan |

The raw downloads go in `assets/raw/` (the `*.glb` files are not tracked). To fetch
them again, run `uv run python tools/fetch_objaverse.py 7e17299c2cdd42ea9686e48a1089790c`.

These were searched for and rejected (details in the `video-to-sim-scene` skill):
- **Balance:** Sketchfab and Objaverse have no analytical balance with a draft
  shield. The "scale" hits were bathroom and kitchen scales.
- **Vials:** none of the downloadable vials is a threaded scintillation vial, and
  the closest one ("Short Glass Vial") is not in Objaverse 1.0.
- **Dropper:** the model listed as "Dropper." (`bdaf19fb…`) is an IV stand.
