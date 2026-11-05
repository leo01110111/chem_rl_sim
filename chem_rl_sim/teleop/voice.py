"""CONTRACT
Voice commands: microphone -> Vosk (offline, CPU), open vocabulary -> command names on
a queue.
  VoiceCommands(model_dir).start(); .poll() -> list of (command | None, heard_text, note):
  EVERY transcript is reported, with command None and a note saying why when it was
  not accepted (not a command / low confidence / muted), so the UI never shows a word as
  heard without saying what became of it.
  An utterance counts only when its WHOLE free-form transcript is one of PHRASES (or a
  known mishearing in ALIASES) and every word clears MIN_CONF. Say the command on its
  own, with a pause before and after.
  Why not a closed grammar: tried first, and ambient talk fired "hold" during testing.
  A grammar forces every sound onto the nearest command with confidence ~1, so no
  threshold separates real commands; with the full vocabulary, ordinary speech
  transcribes as other words and never exactly matches. Commands are dropped while
  `mute_until` is in the future (set by the app around its own spoken feedback so it cannot hear itself).
  Latency is Vosk's end-of-utterance detection, ~0.3-0.5 s after you stop talking.
Model: vosk-model-small-en-us-0.15 (40 MB) in ~/.cache/vosk, downloaded by the app on
  first run.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

PHRASES = {
    "start": "start", "record": "start",
    "stop": "save", "save": "save",
    "discard": "discard", "cancel": "discard",
    "delete": "delete",
    "pause": "pause", "resume": "resume",
    "go": "go", "hold": "hold",
    "left go": "left_go", "left hold": "left_hold",
    "right go": "right_go", "right hold": "right_hold",
    "reset": "reset",
    "quit": "quit",
}
# Mishearings of isolated commands by the small model.
ALIASES = {"paws": "pause", "pose": "pause", "resumed": "resume", "holds": "hold",
           "left goal": "left go", "right goal": "right go",
           # The letter "P" said on its own also pauses; the recognizer spells it any of
           # these ways. Whole-utterance match, so "pee" inside a sentence never fires.
           "p": "pause", "pee": "pause", "pea": "pause"}
# Low on purpose: the whole-utterance exact match is the real filter, and the small model
# scores clear isolated words like "resume" well under 0.6.
MIN_CONF = 0.3
RATE = 16000
BLOCK = 1600                     # 100 ms of audio per callback
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
DEFAULT_MODEL = Path.home() / ".cache/vosk/vosk-model-small-en-us-0.15"


def ensure_model(path: Path = DEFAULT_MODEL) -> Path:
    if path.exists():
        return path
    import io
    import urllib.request
    import zipfile
    print(f"downloading Vosk model to {path.parent} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MODEL_URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(path.parent)
    return path


class VoiceCommands:
    def __init__(self, model_dir: Path = DEFAULT_MODEL, device=None):
        self.model_dir, self.device = Path(model_dir), device
        self._q: queue.Queue = queue.Queue()
        self._audio: queue.Queue = queue.Queue()
        self.mute_until = 0.0
        self.last_heard = ""
        self.error: str | None = None

    def start(self) -> "VoiceCommands":
        import sounddevice as sd
        import vosk
        vosk.SetLogLevel(-1)
        model = vosk.Model(str(ensure_model(self.model_dir)))
        self._rec = vosk.KaldiRecognizer(model, RATE)
        self._rec.SetWords(True)
        self._stream = sd.RawInputStream(samplerate=RATE, blocksize=BLOCK, dtype="int16",
                                         channels=1, device=self.device,
                                         callback=lambda d, *_: self._audio.put(bytes(d)))
        self._stream.start()
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        try:
            while True:
                if not self._rec.AcceptWaveform(self._audio.get()):
                    continue
                r = json.loads(self._rec.Result())
                text = r.get("text", "").strip()
                if not text:
                    continue
                heard = text
                text = ALIASES.get(text, text)
                words = r.get("result", [])
                conf = min((w["conf"] for w in words), default=0.0)
                if text not in PHRASES:
                    item = (None, heard, "not a command")
                elif conf < MIN_CONF:
                    item = (None, heard, f"ignored: low confidence {conf:.2f}")
                elif time.time() < self.mute_until:
                    item = (None, heard, "ignored: was speaking")
                else:
                    item = (PHRASES[text], heard, f"conf {conf:.2f}")
                self.last_heard = f"'{heard}' -> {item[0] or item[2]}"
                self._q.put(item)
        except Exception as e:          # surface on the UI instead of dying silently
            self.error = f"voice: {e!r}"

    def poll(self) -> list[tuple[str, str]]:
        out = []
        while not self._q.empty():
            out.append(self._q.get_nowait())
        return out

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
