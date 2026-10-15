---
name: chem-render-share
description: Render images of the chem_rl_sim environment and publish changes to the public GitHub repo. Use when the user asks for "an image of the env", a screenshot or render of the scene, to update docs/env.png, or to push, share, or invite someone to leo01110111/chem_rl_sim.
---

# Rendering and sharing chem_rl_sim

## Render

```bash
export MUJOCO_GL=egl
uv run python tools/render_overview.py     # -> docs/env.png, 1600x800, task layout
uv run python tools/render_scene.py        # -> work/scene_{video,task}.png, compared to the footage
```
- `render_overview.py` has two free-camera views in `VIEWS`: (lookat, distance,
  azimuth, elevation). Left = whole gantry (az 135). Right = bench close-up
  (lookat (-0.05, 0, 0.80), az -60, el -40).
- The UR7e arms hang over the bench in the home pose and block many close-up
  angles (az 60-120 fill the frame with an arm or the column). Before changing a
  view, try several azimuths in a scratch script at 400x400 and pick from a contact
  sheet; always Read the PNG before reporting it.
- `docs/env.png` is embedded in README.md. Re-render it after any layout change and
  commit it with that change.

## GitHub

- Remote: https://github.com/leo01110111/chem_rl_sim, PUBLIC (made public 2026-10-15; everything pushed is world-readable). Submodule
  `third_party/urgantry_sim` points at the public leo01110111/urgantry_sim, so
  collaborators can clone with `--recurse-submodules`.
- Collaborator: Servo97 (Sarvesh Patil) has write access (invited 2026-10-15).
- Auto mode blocks permission grants (adding collaborators) even when the user asks,
  and blocked the first `gh repo create`. Don't retry or work around it: give the
  user the exact command to run as `! <command>`, and tell them the `!` must be the
  first character (a leading space turns it into a chat message), or point them to
  https://github.com/leo01110111/chem_rl_sim/settings/access.
- Verify invites read-only: `gh api repos/leo01110111/chem_rl_sim/invitations`.
- Commit locally after a change, but ask before pushing unless the user already
  asked for the push in this request.
- Never commit videos (`*.mov`, `*.mp4`), `work/`, or `assets/raw/*.glb`;
  `.gitignore` covers them.
