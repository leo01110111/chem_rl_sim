"""Left hands from WiLoR must not come out mirrored (perception_worker.unmirror_left_hands).
Run in the wilor env from ~/WiLoR:
    env -u PYTHONPATH ~/miniconda3/envs/wilor/bin/python ~/chem_rl_sim/tools/check_wilor_left_hand.py
Reprojection check: 3D keypoints (camera frame) projected with WiLoR's assumed focal
must land on WiLoR's own 2D keypoints, for left AND right hands; also mirror test:
a horizontally flipped image turns each hand into the other side, and the fixed left-hand
keypoints must equal the mirrored right-hand ones."""
import sys, os, glob
sys.path[:0] = [os.getcwd(), os.getcwd() + "/teleop", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chem_rl_sim", "teleop")]
import cv2, numpy as np
from teleop.hand_source import WiLoRHandSource
import perception_worker as pw

def reproj(o, W, H):
    kc = o.keypoints_cam
    uv = np.stack([o.focal_length * kc[:, 0] / kc[:, 2] + W / 2, o.focal_length * kc[:, 1] / kc[:, 2] + H / 2], 1)
    return np.linalg.norm(uv - o.keypoints_uv, axis=1).mean()

for name, cls in (("original", WiLoRHandSource), ("fixed", pw.unmirror_left_hands(WiLoRHandSource))):
    src = cls(right_only=False)
    errs = {True: [], False: []}
    flip_err = []
    for f in sorted(glob.glob("demo_img/*")):
        img = cv2.imread(f)
        if img is None: continue
        H, W = img.shape[:2]
        obs = src(img)
        for o in obs:
            errs[o.is_right].append(reproj(o, W, H))
        # mirror test: flip the image; each right hand becomes a left hand at mirrored pixels
        obs_f = src(cv2.flip(img, 1))
        for o in obs:
            if not o.is_right: continue
            cands = [q for q in obs_f if not q.is_right]
            if not cands: continue
            q = min(cands, key=lambda q: abs((W - 1 - q.wrist_uv[0]) - o.wrist_uv[0]) + abs(q.wrist_uv[1] - o.wrist_uv[1]))
            uv_m = q.keypoints_uv.copy(); uv_m[:, 0] = W - 1 - uv_m[:, 0]
            flip_err.append(np.linalg.norm(uv_m - o.keypoints_uv, axis=1).mean())
    print(f"{name}: reprojection px  right {np.mean(errs[True]):.1f} (n={len(errs[True])})  left {np.mean(errs[False]):.1f} (n={len(errs[False])})"
          f"  | mirrored-left vs right keypoints px {np.median(flip_err):.1f} (n={len(flip_err)})")
