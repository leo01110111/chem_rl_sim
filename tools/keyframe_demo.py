"""CONTRACT
Input: none required. --out-mp4 (default work/keyframe_demo.mp4), --out-gif (default
  docs/keyframe_teleop.gif), --every (render every Nth control tick, default 8),
  --preview-frames (frames the ghost is shown before each Enter, default 5).
What it does: opens the keyframe teleop window headless (chem_rl_sim.teleop.keyframe_app,
  default layout "weigh") and drives it with the scripted operator
  (chem_rl_sim.teleop.keyframe_script.weigh): before each Enter it renders the window with
  the planned ghost arm, then renders the execution under physics. The frames are the
  real window, HUD and side panel included.
Output: printed per-step plan statuses and the final task status; an mp4 at 10 fps (so
  --every 8 plays 4x real time); a GIF of it (ffmpeg palette, sierra dither, 10 fps,
  640 px wide).
Consumer: humans; docs/keyframe_teleop.gif is shown in README.md.
Caveat: a script, not a person: it shows what the interface looks like and that the task
  is doable with it, not how long an operator takes.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import imageio
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chem_rl_sim.teleop.keyframe_app import App, parse_args  # noqa: E402
from chem_rl_sim.teleop.keyframe_script import weigh  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-mp4", default=str(ROOT / "work" / "keyframe_demo.mp4"))
    p.add_argument("--out-gif", default=str(ROOT / "docs" / "keyframe_teleop.gif"))
    p.add_argument("--every", type=int, default=8)
    p.add_argument("--preview-frames", type=int, default=5)
    a = p.parse_args()

    tmp = tempfile.mkdtemp()
    app = App(parse_args(["--out", f"{tmp}/demo.hdf5", "--seed", "0", "--width", "960", "--height", "640"]))
    s = app.sess
    Path(a.out_mp4).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out_mp4, fps=10, macro_block_size=1)   # 20 Hz / every x speed-up
    frame = lambda: writer.append_data(cv2.cvtColor(app.draw(), cv2.COLOR_BGR2RGB))

    def preview():
        app.active = next(side for side in s.arms
                          if np.linalg.norm(s.targets[side][0] - s.arms[side].tool_pose()[0]) > 1e-4)
        for _ in range(a.preview_frames):
            frame()

    def run(sess):
        n = 0
        while sess.tick():
            if n % a.every == 0:
                frame()
            n += 1

    app.on_key(ord("r"))                    # new episode: reset and record, as an operator would
    statuses = weigh(s, run=run, preview=preview)
    for _ in range(10):
        frame()
    writer.close()
    for k, v in statuses.items():
        print(f"  {k:24s} {v}")
    print({k: s.info[k] for k in ("success", "vial_on_pan", "vial_upright", "doors_closed", "balance_display")})

    Path(a.out_gif).parent.mkdir(parents=True, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vf = "fps=10,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", a.out_mp4, "-vf", vf, a.out_gif], check=True)
    print(a.out_mp4, a.out_gif, f"{Path(a.out_gif).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
