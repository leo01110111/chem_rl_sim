"""CONTRACT
WiLoR perception worker. Runs in the `wilor` conda env (it needs torch, ultralytics,
WiLoR and the vendored Wuji retargeter from ~/WiLoR/teleop), started by
chem_rl_sim.teleop.app as a script -- never imported as part of the chem_rl_sim
package, whose dependencies that env does not have.
Input: --source realsense | camera index | video path; --address host:port of the app's
  multiprocessing Listener and the hex --authkey it expects.
Output: one wire.py perception message per camera frame, pushed to the app.
  Hand -> side: WiLoR's handedness label. When both detections carry the same label
  (the classifier's usual failure), image position decides: with --view overhead the
  operator's left hand is image-left, with --view selfie (camera facing them) image-right.
  The selfie preview is mirrored so it reads like a mirror.
  keypoints_cam are metric-rescaled through the real intrinsics
  (teleop.hand_source.with_metric_depth); a webcam or video has no reported intrinsics,
  so they come from --hfov.
Exits when the app closes the connection or the source runs out.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wire  # noqa: E402

PREVIEW_W = 480
JPEG_QUALITY = 70


def rotate_frame(frame, k: int):
    """Rotate by k * 90 degrees clockwise."""
    import cv2
    return frame if k == 0 else cv2.rotate(frame, [None, cv2.ROTATE_90_CLOCKWISE,
                                                    cv2.ROTATE_180,
                                                    cv2.ROTATE_90_COUNTERCLOCKWISE][k])


def rotate_K(K: np.ndarray, w: int, h: int, k: int) -> np.ndarray:
    """Intrinsics of a (w, h) image after rotating it k * 90 degrees clockwise."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if k == 1:
        fx, fy, cx, cy = fy, fx, h - 1 - cy, cx
    elif k == 2:
        cx, cy = w - 1 - cx, h - 1 - cy
    elif k == 3:
        fx, fy, cx, cy = fy, fx, cy, w - 1 - cx
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def open_source(spec: str, width: int, height: int):
    """(reader with .read()/.release(), K or None, frame period for paced playback)."""
    import cv2
    if spec.lower() in ("realsense", "rs"):
        from teleop.realsense_source import RealSenseSource
        # imu=False: with the accel stream on, wait_for_frames() sometimes returns a
        # frameset with no colour frame, which read() reports as end-of-stream.
        src = RealSenseSource(width=width, height=height, imu=False)
        i = src.intrinsics
        K = np.array([[i["fx"], 0, i["ppx"]], [0, i["fy"], i["ppy"]], [0, 0, 1.0]])
        return src, K, 0.0
    try:
        cap = cv2.VideoCapture(int(spec))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        period = 0.0
    except ValueError:
        cap = cv2.VideoCapture(spec)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / fps            # play a file back at its own rate, not WiLoR's
    if not cap.isOpened():
        raise RuntimeError(f"could not open source {spec!r}")
    return cap, None, period


def pick_sides(obs, selfie: bool) -> dict:
    """{'left'|'right': HandObservation}, at most one per side."""
    best = {}
    for o in sorted(obs, key=lambda o: -o.confidence):
        side = "right" if o.is_right else "left"
        if side not in best:
            best[side] = o
        else:
            other = "left" if side == "right" else "right"
            if other not in best:
                best[other] = o
    if len(best) == 2 and best["left"].is_right == best["right"].is_right:
        a, b = best["left"], best["right"]
        if (a.wrist_uv[0] > b.wrist_uv[0]) != selfie:
            best["left"], best["right"] = b, a
    return best


def unmirror_left_hands(source_cls):
    """WiLoRHandSource that also handles LEFT hands correctly.

    WiLoR only models right hands: ViTDetDataset mirrors every left-hand crop, so for a
    left hand the network's outputs live in that mirrored crop. WiLoR's demo.py undoes it
    with x -> -x on pred_cam's x-shift, the 3D joints and (implicitly, by reprojecting)
    the 2D keypoints. ~/WiLoR/teleop/hand_source.py skips that because it ran right-only
    (the multiplier is +1 there); without it a left hand comes out mirrored about its own
    wrist -- keypoints drawn on the wrong side of the hand, a mirrored palm frame.
    """
    class Source(source_cls):
        def _build(self, out, batch, boxes, sides, confs):
            m = 2 * batch["right"].float() - 1                         # +1 right, -1 left
            out["pred_cam"][:, 1] *= m
            out["pred_keypoints_3d"][..., 0] *= m[:, None]
            out["pred_keypoints_2d"][..., 0] *= m[:, None]
            return super()._build(out, batch, boxes, sides, confs)
    return Source


def warmup(hands) -> None:
    """Run WiLoR once per batch size (1 and 2 hands) on fake boxes, so compilation and
    cuDNN autotuning happen before the app reports ready, not mid-demo."""
    import torch
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    img = np.zeros((480, 640, 3), np.uint8)
    for n in (1, 2, 1, 2):
        ds = ViTDetDataset(hands.model_cfg, img, np.tile([[200, 140, 440, 380]], (n, 1)).astype(np.float32),
                           np.arange(n, dtype=np.float32) % 2, rescale_factor=hands.rescale_factor,
                           fp16=hands.fp16)
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=n)))
        with torch.no_grad():
            hands.model(recursive_to(batch, hands.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="realsense")
    ap.add_argument("--address", required=True)
    ap.add_argument("--authkey", required=True)
    ap.add_argument("--wilor-root", default=str(Path.home() / "WiLoR"))
    ap.add_argument("--effector", default="robotiq")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--view", default="selfie", choices=("selfie", "overhead"))
    ap.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270))
    ap.add_argument("--hfov", type=float, default=69.0,
                    help="horizontal FOV (deg) when the source reports no intrinsics")
    ap.add_argument("--depth-scale", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    root = Path(args.wilor_root).resolve()
    for p in (str(root), str(root / "teleop")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(root)
    import cv2
    import torch
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    torch.set_num_threads(n_threads)
    cv2.setNumThreads(n_threads)
    from teleop.hand_source import WiLoRHandSource, with_metric_depth

    host, port = args.address.rsplit(":", 1)
    conn = Client((host, int(port)), authkey=bytes.fromhex(args.authkey))

    src, K_src, period = open_source(args.source, args.width, args.height)
    hands = unmirror_left_hands(WiLoRHandSource)(checkpoint_path=str(root / "pretrained_models/wilor_final.ckpt"),
                                                  cfg_path=str(root / "pretrained_models/model_config.yaml"),
                                                  detector_path=str(root / "pretrained_models/detector.pt"),
                                                  conf=args.conf, fp16=args.fast, right_only=False)
    # One operator has two hands. Capping the detector at 2 bounds WiLoR's batch to the
    # sizes warmed up below; with --fast, every new batch size otherwise triggers a
    # torch.compile that freezes perception for seconds mid-demo.
    detect = hands.detector
    hands.detector = lambda *a, **k: detect(*a, max_det=2, **k)
    warmup(hands)
    retarget = {}
    if args.effector == "wuji":
        from wuji_retargeting import Retargeter
        cfg = str(root / "teleop/config/adaptive_analytical_mano.yaml")
        retarget = {s: Retargeter.from_yaml(cfg, hand_side=s) for s in ("left", "right")}
    k_rot = args.rotate // 90
    conn.send(wire.pack({"ready": True}))

    seq, fps, t_prev = 0, 0.0, time.perf_counter()
    try:
        while True:
            t_read = time.perf_counter()
            ok, frame = src.read()
            if not ok:
                break
            t_capture = time.time()
            h0, w0 = frame.shape[:2]
            if K_src is None:
                f = 0.5 * w0 / np.tan(np.deg2rad(args.hfov) / 2)
                K_src = np.array([[f, 0, (w0 - 1) / 2], [0, f, (h0 - 1) / 2], [0, 0, 1.0]])
            frame = rotate_frame(frame, k_rot)
            K = rotate_K(K_src, w0, h0, k_rot)

            selfie = args.view == "selfie"
            if selfie:
                frame_raw, frame = frame, cv2.flip(frame, 1)     # preview reads like a mirror
            out = {}
            for side, o in pick_sides(hands(frame_raw if selfie else frame), selfie).items():
                try:
                    o = with_metric_depth(o, K, depth_scale=args.depth_scale)
                except ValueError:
                    continue
                h = {"keypoints_cam": o.keypoints_cam.astype(np.float32),
                     "keypoints_wrist": o.keypoints_wrist.astype(np.float32),
                     "wrist_rot_cam": o.wrist_rot_cam.astype(np.float32),
                     "uv": o.keypoints_uv.astype(np.float32),
                     "confidence": float(o.confidence)}
                if side in retarget:
                    h["fingers"] = np.asarray(
                        retarget[side].retarget(o.keypoints_wrist.astype(float)), np.float32)
                out[side] = h
                color = (255, 160, 0) if side == "left" else (0, 160, 255)
                uv = o.keypoints_uv.copy()
                if selfie:
                    uv[:, 0] = frame.shape[1] - 1 - uv[:, 0]
                for u, v in uv:
                    cv2.circle(frame, (int(u), int(v)), 3, color, -1)
                u, v = uv[0]
                cv2.putText(frame, side[0].upper(), (int(u) + 6, int(v)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            s = PREVIEW_W / frame.shape[1]
            small = cv2.resize(frame, (PREVIEW_W, int(round(frame.shape[0] * s))))
            jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()

            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            t_prev = now
            seq += 1
            msg = {"seq": seq, "t_capture": t_capture, "fps": fps, "hands": out,
                   "preview": jpeg, "stats": dict(hands.last_stats)}
            conn.send(wire.pack(msg))
            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - t_read)))
    except (BrokenPipeError, ConnectionResetError, EOFError):
        pass                                     # the app went away: just exit
    finally:
        src.release()


if __name__ == "__main__":
    main()
