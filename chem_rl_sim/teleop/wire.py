"""CONTRACT
Message format between the WiLoR worker (wilor conda env: py3.10, numpy 1.26) and the
teleop app (this repo's env: py3.13, numpy 2). Both import this file, so it may use only
the stdlib and numpy.
  Arrays travel as {"__nd__": (dtype str, shape, raw bytes)}, never as pickled ndarrays:
  numpy 2 pickles name `numpy._core`, which numpy 1.26 cannot unpickle, and the reverse
  only works by alias. Everything else in a message is plain dict/list/str/float/bytes,
  which multiprocessing.connection pickles portably across the two Pythons.
One perception message (worker -> app), sent once per camera frame:
  {"seq": int, "t_capture": float (time.time() at frame read), "fps": float,
   "hands": {"left"|"right": hand} (a side is absent when that hand was not seen),
   "preview": JPEG bytes (camera frame with detections drawn, ~480 px wide),
   "stats": {"detected", "kept"}}
  hand = {"keypoints_cam": (21,3) m, OpenCV camera frame, metric-rescaled;
          "keypoints_wrist": (21,3) m, wrist-relative, MediaPipe order;
          "wrist_rot_cam": (3,3) palm frame (teleop.hand_source.wrist_frame);
          "uv": (21,2) px; "confidence": float;
          "fingers": (20,) Wuji joint angles, only when the worker retargets (--effector wuji)}
"""
from __future__ import annotations

import numpy as np


def pack(obj):
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        return {"__nd__": (a.dtype.str, a.shape, a.tobytes())}
    if isinstance(obj, dict):
        return {k: pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [pack(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def unpack(obj):
    if isinstance(obj, dict):
        if "__nd__" in obj:
            dtype, shape, buf = obj["__nd__"]
            return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape).copy()
        return {k: unpack(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unpack(v) for v in obj]
    return obj
