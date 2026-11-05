"""CONTRACT
App side of the perception link: starts perception_worker.py under the `wilor` conda
python and keeps only its newest message.
  PerceptionBridge(source, effector, ...).start() -> blocks until the worker has loaded
  WiLoR (or raises with the worker's log tail); .latest() -> newest unpacked wire.py
  message or None; .error -> set if the worker died; .close() stops it.
The worker's stdout/stderr go to <log_path> so a crash in torch is readable afterwards.
PYTHONPATH is cleared for the worker: ROS on it shadows the env's pinocchio (see
~/WiLoR/teleop/README.md).
"""
from __future__ import annotations

import os
import secrets
import subprocess
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path

from . import wire

WORKER = Path(__file__).resolve().parent / "perception_worker.py"
DEFAULT_PYTHON = Path.home() / "miniconda3/envs/wilor/bin/python"
DEFAULT_WILOR = Path.home() / "WiLoR"
READY_TIMEOUT_S = 180.0
WORKER_THREADS = 4


class PerceptionBridge:
    def __init__(self, source: str, effector: str, *, python: str | Path = DEFAULT_PYTHON,
                 wilor_root: str | Path = DEFAULT_WILOR, log_path: str | Path = "teleop_worker.log",
                 extra_args: list[str] | None = None):
        self.source, self.effector = source, effector
        self.python, self.wilor_root = Path(python), Path(wilor_root)
        self.log_path = Path(log_path)
        self.extra_args = extra_args or []
        self._lock = threading.Lock()
        self._latest = None
        self.error: str | None = None
        self.proc: subprocess.Popen | None = None
        self._listener: Listener | None = None
        self._conn = None

    def start(self) -> "PerceptionBridge":
        authkey = secrets.token_bytes(16)
        self._listener = Listener(("127.0.0.1", 0), authkey=authkey)
        host, port = self._listener.address
        cmd = [str(self.python), str(WORKER), "--source", self.source,
               "--address", f"{host}:{port}", "--authkey", authkey.hex(),
               "--wilor-root", str(self.wilor_root), "--effector", self.effector,
               *self.extra_args]
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        # torch/OpenCV default to one thread per core and spin-wait: on a 128-thread
        # machine the worker burned ~11 cores for no gain. The GPU does the real work.
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(k, str(WORKER_THREADS))
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env)

        # accept() blocks with no timeout, so watch the process from a helper thread
        # and poke the listener if the worker dies before connecting.
        accepted = threading.Event()

        def watchdog():
            t0 = time.time()
            while not accepted.is_set():
                if self.proc.poll() is not None or time.time() - t0 > READY_TIMEOUT_S:
                    self._listener.close()
                    return
                time.sleep(0.2)
        threading.Thread(target=watchdog, daemon=True).start()
        try:
            self._conn = self._listener.accept()
            accepted.set()
            hello = wire.unpack(self._conn.recv())    # sent after WiLoR has loaded
            assert hello.get("ready")
        except Exception as e:
            accepted.set()
            self.close()
            raise RuntimeError(f"perception worker failed to start ({e!r}). "
                               f"Last lines of {self.log_path}:\n{self._log_tail()}") from None
        threading.Thread(target=self._recv_loop, daemon=True).start()
        return self

    def _recv_loop(self):
        try:
            while True:
                msg = wire.unpack(self._conn.recv())
                msg["t_received"] = time.time()
                with self._lock:
                    self._latest = msg
        except Exception as e:
            code = self.proc.poll() if self.proc else None
            self.error = (f"perception worker stopped (exit {code}, {type(e).__name__}). "
                          f"See {self.log_path}")

    def latest(self):
        with self._lock:
            return self._latest

    def _log_tail(self, n: int = 25) -> str:
        try:
            return "".join(open(self.log_path).readlines()[-n:])
        except OSError:
            return ""

    def close(self):
        for c in (self._conn, self._listener):
            try:
                c and c.close()
            except OSError:
                pass
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
