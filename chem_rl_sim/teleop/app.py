"""CONTRACT
Bimanual BC data collection for ChemVialWeighing: the operator's hands under a real
camera (mounted like top1: overhead, image top pointing away from them) drive both UR7e
arms and their Robotiq grippers (default) or Wuji hands, seen live from top1.
  python -m chem_rl_sim.teleop --dataset weigh_vial            (RealSense)
  python -m chem_rl_sim.teleop --source 0 --hand wuji         (webcam, Wuji hands)
Processes: this one (sim, control, display, voice, recording) + the WiLoR worker in the
  `wilor` conda env (bridge.py). One fixed-rate loop at --hz (20): read newest hands ->
  per-arm controller (arm.py) -> env.step -> record -> draw. obs and action share that
  clock; a tick that runs late is not caught up (no bursts), it just starts the next one.
Modes: IDLE (sim live, robot drivable with "go" for practice, nothing recorded),
  RECORDING, PAUSED (sim frozen, nothing recorded, both clutches open: reposition your
  hands, then "resume" re-anchors them where they are).
Voice (voice.py) / keys:
  start|record  s    reset the env, start an episode, engage both arms
  stop|save     e    save the episode (success flag from the env)
  discard|cancel d   drop the episode
  delete        x    delete the last saved episode (IDLE only)
  pause | "P" / resume  SPACE   freeze / unfreeze
  go / hold     g / h   engage / release both arms
  left go|hold, right go|hold   1 / 2 (toggle)   one arm
  reset         r    discard and reset the env
  quit          q, ESC
  The env's success ends and saves the episode by itself; so does --max-episode-s.
Output: data/teleop/<dataset>.hdf5 in OGPO format (recorder.py).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import mujoco

from ..env import VialWeighingEnv
from ..robot import DEFAULT_EFFECTOR
from .arm import CAMERA_VIEWS, ArmTeleop, operator_rotation
from .bridge import DEFAULT_PYTHON, DEFAULT_WILOR, PerceptionBridge
from .recorder import Recorder, observe

WINDOW = "chem_rl_sim teleop (top1)"
SIDES = ("left", "right")
STALE_S = 0.3            # a perception result older than this counts as no hand
SPEECH_TAIL_S = 0.3      # ignore the mic this long after a spoken confirmation ends
KEYS = {ord("s"): "start", ord("e"): "save", ord("d"): "discard", ord("x"): "delete",
        ord(" "): "toggle_pause", ord("g"): "go", ord("h"): "hold", ord("1"): "left_toggle",
        ord("2"): "right_toggle", ord("r"): "reset", ord("q"): "quit", 27: "quit"}
COLORS = {"left": (255, 160, 0), "right": (0, 160, 255)}    # BGR, match the worker's


def panel_width(scene_width: int) -> int:
    return int(0.6 * scene_width)


def WRIST_TOP(camera_width: int) -> int:
    """Top row of the wrist views in the panel: below the 4:3 camera preview."""
    return 44 + camera_width * 3 // 4 + 40


class App:
    def __init__(self, args):
        self.args = args
        self.dt = 1.0 / args.hz
        env_kwargs = dict(image_size=args.image_size, control_hz=args.hz, hand=args.hand,
                          layout=args.layout, clutter=args.clutter, max_episode_steps=10 ** 9,
                          start_door_open=args.start_door_open)
        self.env = VialWeighingEnv(**env_kwargs)
        self.obs, self.info = self.env.reset(seed=args.seed)
        m, d = self.env.model, self.env.data
        R_wc = operator_rotation(args.view, args.camera_pitch)
        self.arms = {s: ArmTeleop(m, d, s, args.hand, R_wc, gain=args.gain, dt=self.dt) for s in SIDES}
        self.lo, self.hi = m.actuator_ctrlrange.T.copy()
        self.view = mujoco.Renderer(m, args.display_size, args.display_size)
        self.top1 = m.camera("top1").id
        pw = panel_width(args.display_size) - 20
        self.wrist_size = min((pw - 10) // 2, args.display_size - WRIST_TOP(pw) - 10)
        self.wrist_view = mujoco.Renderer(m, self.wrist_size, self.wrist_size)

        out = Path(args.out) if args.out else Path("data/teleop") / f"{args.dataset}.hdf5"
        self.rec = Recorder(out, {"env_name": "ChemVialWeighing-v0", "type": "chem_rl_sim",
                                  "env_kwargs": {**env_kwargs, "normalized_actions": True,
                                                 "max_episode_steps": int(args.max_episode_s * args.hz)}},
                            args.hz, args.hand)
        self.mode, self.paused_from = "idle", "idle"
        self.message, self.message_t = "", 0.0
        self.msg = None
        self.timeline = sorted((float(t), c) for t, c in
                               (x.split("@")[::-1] for x in args.commands.split(","))) if args.commands else []
        self.n_saved = self.rec.num_demos()
        self.tick_ms = 0.0
        self.voice = self.bridge = None

    # ------------------------------------------------------------ feedback
    def say(self, text: str, speak: bool = True):
        self.message, self.message_t = text, time.time()
        print(f"[teleop] {text}", flush=True)
        if speak and not self.args.no_speech and shutil.which("spd-say"):
            threading.Thread(target=self._speak, args=(text.split(":")[0],), daemon=True).start()

    def _speak(self, text: str):
        """Mute the mic only while the confirmation is actually playing (+ a short tail);
        a fixed window used to swallow a quick 'pause ... resume'."""
        if self.voice is not None:
            self.voice.mute_until = float("inf")
        subprocess.run(["spd-say", "-w", "-r", "30", text],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if self.voice is not None:
            self.voice.mute_until = time.time() + SPEECH_TAIL_S

    # ------------------------------------------------------------ episodes
    def _reset_env(self):
        self.obs, self.info = self.env.reset()
        for a in self.arms.values():
            a.reset()

    def _finish(self, save: bool, why: str = ""):
        if save:
            n = self.rec.save(success=bool(self.info["success"]))
            if n:
                self.n_saved += 1
                ok = "success" if self.info["success"] else "no success"
                self.say(f"saved{why}: {n} steps, {ok}")
        else:
            self.rec.discard()
            self.say("discarded")
        self.mode = "idle"
        for a in self.arms.values():
            a.release()

    def handle(self, cmd: str):
        recording = self.mode == "recording" or (self.mode == "paused" and self.paused_from == "recording")
        if cmd == "toggle_pause":
            cmd = "resume" if self.mode == "paused" else "pause"
        if cmd.endswith("_toggle"):
            side = cmd.split("_")[0]
            cmd = f"{side}_hold" if self.arms[side].active else f"{side}_go"

        if cmd == "pause" and self.mode == "paused":
            return self.say("already paused")
        if cmd == "start":
            if recording:
                return self.say("already recording")
            self.rec.wait()
            self._reset_env()
            self.rec.begin(observe(self.env, self.arms, self.obs["image"]))
            self.mode = "recording"
            for a in self.arms.values():
                a.arm()
            self.say("recording")
        elif cmd == "save":
            if recording:
                self._finish(True)
        elif cmd == "discard":
            if recording:
                self._finish(False)
        elif cmd == "delete":
            if self.mode != "idle":
                return self.say("save or discard first")
            name = self.rec.delete_last()
            if name:
                self.n_saved -= 1
            self.say(f"deleted {name}" if name else "nothing to delete")
        elif cmd == "pause" and self.mode != "paused":
            self.paused_from, self.mode = self.mode, "paused"
            for a in self.arms.values():
                a.release()
            self.say("paused")
        elif cmd == "resume":
            if self.mode == "paused":
                self.mode = self.paused_from
                self.say("resumed")
            else:
                self.say("not paused, engaging arms")
            for a in self.arms.values():
                a.arm()
        elif cmd in ("go", "hold") and self.mode != "paused":
            for a in self.arms.values():
                a.arm() if cmd == "go" else a.release()
            self.say("go" if cmd == "go" else "holding")
        elif cmd in ("left_go", "left_hold", "right_go", "right_hold") and self.mode != "paused":
            side, what = cmd.split("_")
            self.arms[side].arm() if what == "go" else self.arms[side].release()
            self.say(f"{side} {'on' if what == 'go' else 'off'}")
        elif cmd == "reset":
            if recording:
                self.rec.discard()
            self._reset_env()
            self.mode = "idle"
            self.say("reset")
        elif cmd == "quit":
            self.running = False

    # ------------------------------------------------------------ control tick
    def tick(self):
        msg = self.bridge.latest()
        self.msg = msg
        hands = {}
        if msg is not None and time.time() - msg["t_capture"] < STALE_S:
            hands = msg["hands"]
        if self.mode == "paused":
            return

        env, d = self.env, self.env.data
        state = np.concatenate([[d.time], d.qpos, d.qvel])
        for s, a in self.arms.items():
            a.step(hands.get(s))
        ctrl = d.ctrl.copy()
        for a in self.arms.values():
            a.write(ctrl)
        ctrl = np.clip(ctrl, self.lo, self.hi)
        self.obs, _, terminated, _, self.info = env.step(ctrl)

        if self.mode != "recording":
            return
        nan = np.full((21, 3), np.nan, np.float32)
        op = {"keypoints_cam": np.stack([hands[s]["keypoints_cam"] if s in hands else nan for s in SIDES]),
              "keypoints_wrist": np.stack([hands[s]["keypoints_wrist"] if s in hands else nan for s in SIDES]),
              "engaged": np.array([self.arms[s].engaged for s in SIDES]),
              "perception_seq": msg["seq"] if msg else -1,
              "perception_age_ms": 1e3 * (time.time() - msg["t_capture"]) if msg else np.nan}
        action = 2.0 * (ctrl - self.lo) / (self.hi - self.lo) - 1.0
        self.rec.add(state, action, ctrl, float(self.info["success"]),
                     observe(env, self.arms, self.obs["image"]), op)
        if terminated:
            self._finish(True, " (task success)")
        elif len(self.rec) >= self.args.max_episode_s * self.args.hz:
            self._finish(True, " (time limit)")

    # ------------------------------------------------------------ display
    def draw(self) -> np.ndarray:
        # Wider than the policy camera so the operator sees both arms whole; the recorded
        # top1_image keeps top1's own fovy (the env's renderer never sees this change).
        fovy = self.env.model.cam_fovy[self.top1]
        self.env.model.cam_fovy[self.top1] = self.args.view_fovy
        self.view.update_scene(self.env.data, camera="top1")
        self.env.model.cam_fovy[self.top1] = fovy
        img = cv2.cvtColor(self.view.render(), cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(img, (0, 0), (W, 40), (25, 25, 25), -1)
        if self.mode == "recording":
            t = len(self.rec) / self.args.hz
            label, col = f"REC {int(t) // 60}:{t % 60:04.1f}", (40, 40, 230)
        elif self.mode == "paused":
            label, col = "PAUSED" + (" (rec)" if self.paused_from == "recording" else ""), (0, 200, 230)
        else:
            label, col = "IDLE", (170, 170, 170)
        cv2.putText(img, label, (10, 28), font, 0.8, col, 2)
        x = 210
        for s, a in self.arms.items():
            st = "ON" if a.engaged else ("ARMED" if a.armed else "hold")
            c = COLORS[s] if a.engaged else ((0, 200, 230) if a.armed else (120, 120, 120))
            seen = "*" if a.last["seen"] else " "
            eff = ("GRIP" if a.gripping else f"{a.eff_cmd[0]:3.0f}") if a.effector == "robotiq" else ""
            cv2.putText(img, f"{s[0].upper()}{seen}{st} {eff}", (x, 28), font, 0.6, c, 2)
            x += 150
        fps = self.msg["fps"] if self.msg else 0.0
        cv2.putText(img, f"cam {fps:4.1f}Hz  tick {self.tick_ms:4.1f}ms  demos {self.n_saved}",
                    (x, 18), font, 0.42, (200, 200, 200), 1)
        cv2.putText(img, f"balance {self.info['balance_display']}", (x, 34), font, 0.42, (200, 200, 200), 1)

        lines = []
        if time.time() - self.message_t < 3.0:
            lines.append(self.message)
        if self.voice is not None:
            lines.append(f"heard: {self.voice.last_heard}")
        for err in (self.bridge.error, self.voice and self.voice.error, self.rec.last_error):
            if err:
                lines.append(f"ERROR {err}")
        if self.rec.saving:
            lines.append("saving...")
        for i, ln in enumerate(lines):
            cv2.putText(img, ln, (10, H - 40 - 22 * i), font, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "say: start | stop | discard | pause (or P) | resume | go | hold | "
                         "left/right go/hold | delete | reset | quit", (10, H - 12), font, 0.42,
                    (230, 230, 230), 1)

        # Operator camera in its own panel beside the scene, never over it.
        panel = np.full((H, panel_width(W), 3), 25, np.uint8)
        cv2.putText(panel, "your camera", (10, 28), font, 0.7, (200, 200, 200), 2)
        if self.msg is not None and self.msg.get("preview"):
            cam = cv2.imdecode(np.frombuffer(self.msg["preview"], np.uint8), cv2.IMREAD_COLOR)
            w = panel.shape[1] - 20
            h = min(int(cam.shape[0] * w / cam.shape[1]), H - 54)
            panel[44:44 + h, 10:10 + w] = cv2.resize(cam, (w, int(cam.shape[0] * w / cam.shape[1])))[:h]
        # Wrist cameras (left, right) below the operator camera, outside the top1 view.
        y, n = WRIST_TOP(panel.shape[1] - 20), self.wrist_size
        for i, side in enumerate(SIDES):
            x = 10 + i * (n + 10)
            cv2.putText(panel, f"{side} wrist", (x, y - 12), font, 0.6, COLORS[side], 2)
            if f"{side}_wrist" in self.env.cameras:
                self.wrist_view.update_scene(self.env.data, camera=self.env.cameras[f"{side}_wrist"])
                panel[y:y + n, x:x + n] = cv2.cvtColor(self.wrist_view.render(), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(panel, "no wrist camera", (x + 10, y + n // 2), font, 0.5, (150, 150, 150), 1)
        return np.hstack([img, panel])

    # ------------------------------------------------------------ main loop
    def run(self):
        a = self.args
        extra = ["--view", a.view, "--rotate", str(a.rotate), "--hfov", str(a.hfov), "--conf", str(a.conf),
                 "--depth-scale", str(a.depth_scale)] + ([] if a.no_fast else ["--fast"])
        print("[teleop] starting WiLoR worker (first load takes ~20 s) ...", flush=True)
        self.bridge = PerceptionBridge(a.source, a.hand, python=a.wilor_python, wilor_root=a.wilor_root,
                                       log_path=a.worker_log, extra_args=extra).start()
        self.voice = None
        if not a.no_voice:
            from .voice import VoiceCommands
            try:
                self.voice = VoiceCommands(device=a.mic).start()
            except Exception as e:
                print(f"[teleop] voice disabled: {e!r}", flush=True)
        if not a.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, a.display_size + panel_width(a.display_size), a.display_size)
        self.say(f"ready, {self.n_saved} demos in {self.rec.path}", speak=False)

        self.running, t0 = True, time.perf_counter()
        t_next, n = t0, 0
        try:
            while self.running:
                t_start = time.perf_counter()
                cmds = []
                for c, heard, note in (self.voice.poll() if self.voice else []):
                    print(f"[teleop] voice: {heard!r} -> {c or '-'} ({note})", flush=True)
                    if c:
                        cmds.append(c)
                while self.timeline and self.timeline[0][0] <= t_start - t0:
                    cmds.append(self.timeline.pop(0)[1])
                for c in cmds:
                    self.handle(c)
                self.tick()
                frame = self.draw()
                if not a.headless:
                    cv2.imshow(WINDOW, frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k in KEYS:
                        self.handle(KEYS[k])
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        self.running = False
                elif a.snapshot and n % int(a.hz) == 0:
                    cv2.imwrite(a.snapshot, frame)
                self.tick_ms = 0.9 * self.tick_ms + 0.1 * 1e3 * (time.perf_counter() - t_start)
                n += 1
                if a.frames and n >= a.frames:
                    break
                t_next += self.dt
                lag = t_next - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
                else:
                    t_next = time.perf_counter()     # late: start over, never burst
        finally:
            if self.mode == "recording" or (self.mode == "paused" and self.paused_from == "recording"):
                self._finish(True, " (on exit)")
            self.rec.wait()
            self.bridge.close()
            if self.voice:
                self.voice.close()
            cv2.destroyAllWindows()
        print(f"[teleop] {self.n_saved} demos in {self.rec.path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="chem_vial", help="data/teleop/<dataset>.hdf5")
    ap.add_argument("--out", default=None, help="explicit .hdf5 path (overrides --dataset)")
    ap.add_argument("--hand", default=DEFAULT_EFFECTOR, choices=("robotiq", "wuji"))
    ap.add_argument("--source", default="realsense", help="realsense | camera index | video file")
    ap.add_argument("--view", default="selfie", choices=tuple(CAMERA_VIEWS),
                    help="selfie: camera in front of you, facing you; overhead: above your "
                         "hands looking down, image top pointing away from you")
    ap.add_argument("--camera-pitch", type=float, default=0.0,
                    help="camera tilt in degrees, positive = looking further down")
    ap.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270),
                    help="rotate the camera image clockwise until it is upright")
    ap.add_argument("--hfov", type=float, default=69.0, help="webcam/video horizontal FOV, deg")
    ap.add_argument("--hz", type=float, default=20.0, help="control = obs = action rate")
    ap.add_argument("--gain", type=float, default=1.0, help="robot metres per hand metre")
    ap.add_argument("--depth-scale", type=float, default=1.0)
    ap.add_argument("--conf", type=float, default=0.3, help="hand detector confidence")
    ap.add_argument("--no-fast", action="store_true",
                    help="WiLoR in fp32 without torch.compile (~15 Hz instead of ~25 Hz)")
    ap.add_argument("--image-size", type=int, default=224, help="recorded top1 image size")
    ap.add_argument("--display-size", type=int, default=800)
    ap.add_argument("--view-fovy", type=float, default=100.0,
                    help="top1 field of view on screen, deg (the recorded image keeps top1's 80)")
    ap.add_argument("--layout", default="weigh", choices=("weigh", "task", "video"),
                    help="weigh: only the target vial and the balance; task: the full bench")
    ap.add_argument("--clutter", action="store_true",
                    help="add the Kimwipes box, cardboard box, cork ring and floor tape")
    ap.add_argument("--start-door-open", default=None, choices=(None, "left", "right", "top"))
    ap.add_argument("--max-episode-s", type=float, default=180.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-speech", action="store_true", help="no spoken confirmations")
    ap.add_argument("--mic", default=None, help="sounddevice input device (name or index)")
    ap.add_argument("--wilor-python", default=str(DEFAULT_PYTHON))
    ap.add_argument("--wilor-root", default=str(DEFAULT_WILOR))
    ap.add_argument("--worker-log", default="teleop_worker.log")
    # testing
    ap.add_argument("--headless", action="store_true", help="no window (testing)")
    ap.add_argument("--snapshot", default=None, help="headless: write the HUD frame here every second")
    ap.add_argument("--frames", type=int, default=0, help="stop after N ticks")
    ap.add_argument("--commands", default="", help="timed commands, e.g. 'start@2,save@20'")
    args = ap.parse_args(argv)
    if args.mic is not None and args.mic.isdigit():
        args.mic = int(args.mic)
    App(args).run()


if __name__ == "__main__":
    sys.exit(main())
