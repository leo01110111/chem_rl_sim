"""CONTRACT
Input: Objaverse 1.0 / Sketchfab uids on argv (32 hex chars each).
Output: for each uid that is in Objaverse 1.0,
  assets/raw/<uid>.glb   the original glTF binary (Y-up, units as authored -- NOT meters)
  assets/raw/<uid>.json  {uid, name, license, author, url, face_count}
  Uids not in Objaverse 1.0 are reported on stderr and skipped.
Consumer: tools/mesh_to_mjcf.py reads the .glb; the .json is the provenance record
  that the per-asset README in chem_rl_sim/assets/ copies its license line from.
Convention recovered: glTF is +Y up (glTF 2.0 spec section 3.4); authored scale is
  arbitrary (seen 1 unit = 1 m, 1 cm, and 1 inch), so the consumer rescales from
  a measured real dimension, never from the file.
Reference output: assets/raw/*.json for the uids listed in chem_rl_sim/assets/README.md.
"""
import json
import shutil
import sys
from pathlib import Path

import objaverse

RAW = Path(__file__).resolve().parents[1] / "assets" / "raw"


def main(uids: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    # load_annotations raises KeyError on a uid outside the 2022 snapshot, so
    # filter against the object-path index first.
    known = objaverse._load_object_paths()
    for u in uids:
        if u not in known:
            print(f"{u}: not in Objaverse 1.0", file=sys.stderr)
    uids = [u for u in uids if u in known]
    ann = objaverse.load_annotations(uids)
    paths = objaverse.load_objects(uids, download_processes=4)
    for uid, src in paths.items():
        a = ann[uid]
        shutil.copy(src, RAW / f"{uid}.glb")
        meta = {"uid": uid, "name": a.get("name"), "license": a.get("license"),
                "author": a.get("user", {}).get("username"), "url": a.get("viewerUrl"),
                "face_count": a.get("faceCount")}
        (RAW / f"{uid}.json").write_text(json.dumps(meta, indent=2))
        print(f"{uid}: {meta['name']!r} license={meta['license']} author={meta['author']}")


if __name__ == "__main__":
    main(sys.argv[1:])
