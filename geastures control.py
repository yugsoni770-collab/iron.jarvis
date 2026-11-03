r"""
HOLOGRAPHIC HAND TRACKING AR INTERFACE  -  AI edition (single file)
=====================================================================
Real-time webcam hand tracking, gesture recognition, holographic UI, air drawing
and AR object control. Python 3.9 - 3.12 + OpenCV + MediaPipe + NumPy. 100% local.

SETUP (Windows PowerShell, inside the folder that contains this file; use Python 3.11):
    py -3.11 -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install opencv-contrib-python==4.10.0.84 mediapipe==0.10.14 numpy==1.26.4
    python hand_tracking_ar.py

OPTIONAL AI VOICE ASSISTANT (offline, free):   python -m pip install vosk sounddevice
    First run downloads a 40 MB speech model once into ./models, then works fully offline.
    Say the wake word + a command:  "Jarvis, open drawing"  /  "Jarvis, change theme to matrix"
    Press I in the window to TYPE a command instead (works without any extra package).
    Optional local chat AI: install Ollama (ollama.com), run `ollama pull llama3.2:1b`, then enable
    Advanced Settings -> AI -> LOCAL LLM. Unknown questions are then answered by that local model.

OPTIONS:  --camera 1  --width 960 --height 540  --fullscreen  --no-effects  --no-mirror  --list-cameras

MAIN MENU:  HOME | DRAWING | OBJECT CONTROL | ADVANCED SETTINGS | SYSTEM INFORMATION | EXIT

AI FEATURES
    VOICE ASSISTANT  wake word, ~60 spoken commands (modes, themes, drawing, objects, recording,
                     settings, status report, time, spoken maths), spoken replies, mic level orb
    LOCAL LLM        optional Ollama chat model for open questions (no cloud, no API key)
    SHAPE SNAP       draw a rough circle / triangle / rectangle / line -> it becomes a perfect shape
    CUSTOM GESTURES  teach 3 of your own hand poses (kNN learner) and bind each to an action

ADVANCED SETTINGS (8 tabs, ~65 options, saved automatically to ./output/profile.json)
    GESTURE   sensitivity, stability, pinch threshold, cooldown, fist hold, swipe distance
    TRACKING  smoothing, model lite/full, max hands, detection/tracking confidence, inference
              width, hand-loss grace, mirror camera, camera resolution
    DISPLAY   6 colour themes, landmark style, panel opacity, brightness, scanlines, vignette,
              glass blur, fullscreen
    EFFECTS   scan rings, glow + intensity, fingertip trails, particles, cursor size/smoothing
    FEATURES  sound, object throw physics, drawing symmetry, rainbow brush, stroke smoothing,
              auto standby, FPS graph, hints
    AI        voice assistant, wake word, voice replies, local LLM, shape snap
    CUSTOM    train / bind / clear your own gestures
    SYSTEM    save/load/reset profile, snapshot, MP4 recording, switch camera, open output folder

GESTURES
    Open palm ........ show interface        Closed fist (hold) .. hide interface
    Index up ......... pointer / cursor      Pinch ............... click / grab object
    Two fingers ...... navigation (swipe left/right moves menu focus, Thumb up confirms)
    Thumb up ......... confirm               Three fingers (hold)  drawing mode
    Release a pinched object while moving = throw it (it bounces off the edges)
    Both hands pinching one object ... scale + rotate it

KEYS: I type a command to the assistant, 1-5 modes, H UI, T next theme, K snapshot, V record video, M sound, D FPS graph,
      F fullscreen, R resolution, L landmark style, G effects, Q/Esc quit
  Drawing: C clear, S save PNG, E eraser, P pen, [ ] brush, N colour, U undo, Y redo,
           Z symmetry, W rainbow
  Objects: A add, Del/X delete, + - scale, Tab next, , . rotate, arrows move
Files are written to the ./output folder next to this script.
"""
from __future__ import annotations

import argparse
import ast
import colorsys
import dataclasses
import difflib
import json
import math
import operator
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple


def check_python_version() -> None:
    """MediaPipe 0.10.14 only ships wheels for Python 3.8 - 3.12."""
    if sys.version_info >= (3, 13) or sys.version_info < (3, 8):
        v = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"\n[ERROR] You are running Python {v}, which MediaPipe does not support.")
        print("Install Python 3.11 (64-bit) and create the virtual environment with it:")
        print("    winget install Python.Python.3.11")
        print("    py -3.11 -m venv .venv")
        print("    .venv\\Scripts\\activate")
        print("    python -m pip install --upgrade pip")
        print("    python -m pip install opencv-contrib-python==4.10.0.84 mediapipe==0.10.14 numpy==1.26.4")
        print("Then in VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter' -> choose .venv\n")
        sys.exit(1)


def check_dependencies() -> None:
    """Fail early with a friendly message if a required package is missing."""
    required = (("cv2", "opencv-contrib-python==4.10.0.84"), ("numpy", "numpy==1.26.4"),
                ("mediapipe", "mediapipe==0.10.14"))
    missing = []
    for module, package in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("\n[ERROR] Missing Python packages: " + ", ".join(missing))
        print("Activate your virtual environment and run:")
        print("    python -m pip install " + " ".join(missing) + "\n")
        sys.exit(1)


check_python_version()
check_dependencies()

import cv2  # noqa: E402
import numpy as np  # noqa: E402



# ==========================================================================
# CONFIGURATION
# ==========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

APP_TITLE = "HOLOGRAPHIC AR INTERFACE"
APP_VERSION = "1.0.0"

Color = Tuple[int, int, int]


def hex_to_bgr(value: str) -> Color:
    """Convert '#RRGGBB' to an OpenCV (B, G, R) tuple."""
    value = value.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


class Palette:
    """Design system colours (BGR order, ready for OpenCV)."""

    BACKGROUND = hex_to_bgr("#050816")
    PRIMARY = hex_to_bgr("#00E5FF")
    SECONDARY = hex_to_bgr("#1677FF")
    ACCENT = hex_to_bgr("#7B61FF")
    TEXT = hex_to_bgr("#EAF7FF")
    DIM_TEXT = hex_to_bgr("#7FA6BF")
    SUCCESS = hex_to_bgr("#2CFF9A")
    WARNING = hex_to_bgr("#FFC857")
    DANGER = hex_to_bgr("#FF4D6D")


THEMES = {
    "CYAN": ("#00E5FF", "#1677FF", "#7B61FF"),
    "MATRIX": ("#2CFF6A", "#14B866", "#C6FF3D"),
    "CRIMSON": ("#FF3D5A", "#FF7A1A", "#FF3DCB"),
    "AMBER": ("#FFB300", "#FF6D00", "#FFE066"),
    "ULTRAVIOLET": ("#B388FF", "#6C4DFF", "#FF4081"),
    "ICE": ("#DFF7FF", "#4FC3F7", "#7B9BFF"),
}
THEME_NAMES = list(THEMES.keys())


def apply_theme(name: str) -> str:
    """Re-colour the whole interface by mutating the shared palette. Returns the applied theme name."""
    if name not in THEMES:
        name = "CYAN"
    primary, secondary, accent = THEMES[name]
    Palette.PRIMARY = hex_to_bgr(primary)
    Palette.SECONDARY = hex_to_bgr(secondary)
    Palette.ACCENT = hex_to_bgr(accent)
    return name


# Camera resolutions that can be cycled through at runtime.
RESOLUTIONS: List[Tuple[int, int]] = [(640, 480), (960, 540), (1280, 720), (1920, 1080)]


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True              # selfie view (also gives correct left/right labels)
    fourcc: str = "MJPG"             # MJPG gives full frame rate on most USB webcams
    reconnect_delay: float = 2.0     # seconds between reconnection attempts
    stall_timeout: float = 2.0       # seconds without frames before the camera counts as lost
    probe_indices: int = 4           # how many camera indices to probe when opening fails


@dataclass
class TrackerConfig:
    max_hands: int = 2
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.6
    model_complexity: int = 0        # 0 = fastest, 1 = more accurate
    process_width: int = 640         # frames are downscaled to this width before inference
    smoothing_level: int = 6         # 1 (raw / responsive) .. 10 (very smooth)
    beta: float = 12.0               # One Euro speed coefficient (higher = less lag when moving fast)
    lost_grace_frames: int = 8       # frames a hand is kept alive after detection is lost
    still_speed_fraction: float = 0.12  # fraction of frame width per second counted as "still"

    @property
    def min_cutoff(self) -> float:
        return max(0.4, 4.5 - 0.4 * self.smoothing_level)


@dataclass
class GestureConfig:
    sensitivity: float = 1.0         # scales all geometric thresholds (0.5 .. 1.6)
    ext_ratio_on: float = 1.50       # |wrist-tip| / |wrist-mcp| above which a finger counts as extended
    ext_ratio_off: float = 1.35      # ... and below which it counts as curled again (hysteresis)
    ext_angle_on: float = 145.0      # PIP joint angle (deg) for extended
    ext_angle_off: float = 125.0
    thumb_on: float = 0.60           # |thumb tip - index mcp| / hand size for extended thumb
    thumb_off: float = 0.45
    pinch_enter: float = 0.28        # |thumb tip - index tip| / hand size to start a pinch
    pinch_exit: float = 0.42         # ... and to release it
    thumb_up_cos: float = 0.60       # how vertical the thumb must point for "thumb up"
    stable_frames: int = 4           # frames a gesture must persist before it is accepted
    pinch_stable_frames: int = 2
    none_stable_frames: int = 6
    action_cooldown: float = 0.6     # seconds between repeated activations of the same action
    palm_hold: float = 0.25          # seconds an open palm must be held to show the interface
    fist_hold: float = 0.60          # seconds a fist must be held to hide the interface
    mode_switch_hold: float = 0.50   # seconds three fingers must be held to enter drawing mode
    swipe_window: float = 0.45       # seconds over which a swipe is measured
    swipe_distance: float = 0.20     # fraction of frame size a swipe must travel
    swipe_cooldown: float = 0.80


@dataclass
class UIConfig:
    window_name: str = APP_TITLE
    fullscreen: bool = False
    theme: str = "CYAN"
    landmark_style: str = "FULL"     # FULL / LINES / DOTS / OFF
    scan_effects: bool = True
    glow: bool = True
    glow_scale: float = 0.25         # glow layer resolution relative to the frame
    glow_gain: float = 1.8           # glow intensity
    glass_blur: bool = True
    panel_opacity: float = 0.55
    scanlines: bool = True
    vignette: bool = True
    feed_dim: float = 0.80           # brightness of the camera feed (1.0 = unchanged)
    boot_seconds: float = 2.2
    toast_seconds: float = 2.6


@dataclass
class FeatureConfig:
    sound: bool = True               # short beeps on click / confirm (Windows only)
    object_physics: bool = True      # throw objects and let them bounce off the screen edges
    symmetry: str = "OFF"            # drawing symmetry: OFF / MIRROR X / 4-WAY
    rainbow: bool = False            # rainbow brush
    stroke_smoothing: int = 6        # 1 (raw) .. 10 (very smooth)
    auto_standby: int = 0            # seconds without a hand before the UI hides (0 = off)
    fps_graph: bool = False          # performance graph overlay
    show_hints: bool = True
    trails: bool = True              # glowing fingertip trails
    trail_length: int = 18
    particles: bool = True           # spark particles on click / draw / grab
    cursor_size: float = 1.0
    cursor_smooth: int = 6           # 1 (snappy) .. 10 (floaty)
    autosave_profile: bool = True    # save settings on exit, load on start


CUSTOM_ACTIONS = ["NONE", "SNAPSHOT", "RECORD", "UNDO", "REDO", "CLEAR CANVAS", "NEXT THEME", "ADD OBJECT",
                  "TOGGLE UI", "SPEAK STATUS"]
WAKE_WORDS = ["JARVIS", "COMPUTER", "HOLO", "OFF"]
LLM_MODELS = ["llama3.2:1b", "llama3.2:3b", "phi3:mini", "gemma2:2b", "mistral"]


@dataclass
class AIConfig:
    voice_enabled: bool = True       # offline voice assistant (needs: pip install vosk sounddevice)
    wake_word: str = "JARVIS"        # say it before a command; OFF = always listening
    voice_replies: bool = True       # spoken answers (system text-to-speech)
    llm_enabled: bool = False        # send unknown questions to a LOCAL Ollama model
    llm_model: str = "llama3.2:1b"
    shape_snap: bool = True          # recognise drawn circles / triangles / rectangles / lines and perfect them
    custom_enabled: bool = True      # trainable custom hand gestures
    custom_action_1: str = "SNAPSHOT"
    custom_action_2: str = "UNDO"
    custom_action_3: str = "NEXT THEME"


@dataclass
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ai: AIConfig = field(default_factory=AIConfig)


settings = Settings()


# ==========================================================================
# SOUND EFFECTS
# ==========================================================================

PATTERNS: Dict[str, List[Tuple[int, int]]] = {
    "click": [(1250, 25)],
    "confirm": [(900, 40), (1400, 60)],
    "on": [(600, 45), (900, 45), (1300, 70)],
    "off": [(1300, 45), (900, 45), (600, 70)],
    "mode": [(1000, 30), (1500, 30)],
    "grab": [(700, 30)],
    "save": [(800, 40), (1200, 40), (1700, 70)],
    "error": [(300, 140)],
}


class SoundFx:
    """Plays short beeps on a background thread; silently does nothing on non-Windows systems."""

    def __init__(self) -> None:
        self._busy = False
        try:
            import winsound  # type: ignore
            self._winsound = winsound
        except ImportError:
            self._winsound = None

    def play(self, kind: str) -> None:
        if self._winsound is None or self._busy or not settings.features.sound:
            return
        pattern = PATTERNS.get(kind)
        if not pattern:
            return
        self._busy = True
        threading.Thread(target=self._run, args=(pattern,), daemon=True).start()

    def _run(self, pattern: List[Tuple[int, int]]) -> None:
        try:
            for freq, ms in pattern:
                self._winsound.Beep(freq, ms)
        except Exception:  # noqa: BLE001 - never let audio problems affect the app
            pass
        finally:
            self._busy = False


# ==========================================================================
# SETTINGS PROFILE
# ==========================================================================

PROFILE_PATH = os.path.join(OUTPUT_DIR, "profile.json")
_SECTIONS = ("gesture", "tracker", "ui", "features", "ai")
_SKIP = {"window_name", "fullscreen"}


def _copy_into(target: Any, data: Dict[str, Any]) -> None:
    """Copy values from ``data`` into a dataclass instance, ignoring unknown keys and wrong types."""
    for f in dataclasses.fields(target):
        if f.name in _SKIP or f.name not in data:
            continue
        cur, val = getattr(target, f.name), data[f.name]
        if isinstance(cur, bool):
            if isinstance(val, bool):
                setattr(target, f.name, val)
        elif isinstance(cur, (int, float)):
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                setattr(target, f.name, type(cur)(val))
        elif isinstance(cur, str) and isinstance(val, str):
            setattr(target, f.name, val)


def _sanitize() -> None:
    g, t, f = settings.gesture, settings.tracker, settings.features
    g.sensitivity = min(1.6, max(0.5, g.sensitivity))
    g.stable_frames = min(10, max(2, g.stable_frames))
    t.smoothing_level = min(10, max(1, t.smoothing_level))
    t.max_hands = min(2, max(1, t.max_hands))
    t.model_complexity = 1 if t.model_complexity else 0
    f.trail_length = min(40, max(5, f.trail_length))
    f.stroke_smoothing = min(10, max(1, f.stroke_smoothing))
    f.cursor_smooth = min(10, max(1, f.cursor_smooth))
    settings.ui.theme = apply_theme(settings.ui.theme)
    a = settings.ai
    if a.wake_word not in WAKE_WORDS:
        a.wake_word = "JARVIS"
    for n in (1, 2, 3):
        if getattr(a, f"custom_action_{n}") not in CUSTOM_ACTIONS:
            setattr(a, f"custom_action_{n}", "NONE")


def save_profile(path: str = PROFILE_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data: Dict[str, Any] = {name: dataclasses.asdict(getattr(settings, name)) for name in _SECTIONS}
    data["camera"] = {"mirror": settings.camera.mirror}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path


def load_profile(path: str = PROFILE_PATH) -> bool:
    """Load a profile. Returns False if none exists. Raises ValueError for a corrupt file."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Profile unreadable: {exc}") from exc
    for name in _SECTIONS:
        if isinstance(data.get(name), dict):
            _copy_into(getattr(settings, name), data[name])
    cam = data.get("camera")
    if isinstance(cam, dict) and isinstance(cam.get("mirror"), bool):
        settings.camera.mirror = cam["mirror"]
    _sanitize()
    return True


def reset_defaults() -> None:
    """Restore factory settings in place (other modules hold references to these objects)."""
    fresh = Settings()
    for name in _SECTIONS:
        target = getattr(settings, name)
        for f in dataclasses.fields(target):
            if f.name not in _SKIP:
                setattr(target, f.name, getattr(getattr(fresh, name), f.name))
    settings.camera.mirror = True
    _sanitize()


# ==========================================================================
# VOICE INPUT AND SPOKEN REPLIES
# ==========================================================================

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
MODELS_DIR = os.path.join(BASE_DIR, "models")


class VoiceListener:
    """Continuously listens to the default microphone on a background thread."""

    def __init__(self, on_final: Callable[[str], None], on_partial: Optional[Callable[[str], None]] = None) -> None:
        self.on_final = on_final
        self.on_partial = on_partial
        self.status = "OFF"          # OFF / NOT INSTALLED / DOWNLOADING n% / LOADING MODEL / LISTENING / NO MICROPHONE / ERROR
        self.error = ""
        self.level = 0.0             # microphone level 0..1 (for the UI meter)
        self.muted_until = 0.0       # ignore audio while the assistant itself is speaking
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @staticmethod
    def dependencies_ok() -> Tuple[bool, str]:
        try:
            import sounddevice  # noqa: F401
            import vosk  # noqa: F401
        except (ImportError, OSError) as exc:
            return False, f"pip install vosk sounddevice   ({exc})"
        return True, ""

    def start(self) -> None:
        if self.running:
            return
        ok, msg = self.dependencies_ok()
        if not ok:
            self.status, self.error = "NOT INSTALLED", msg
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="voice-listener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self._thread = None
        if self.status not in ("NOT INSTALLED", "ERROR", "NO MICROPHONE"):
            self.status = "OFF"
        self.level = 0.0

    # ----------------------------------------------------------------- model
    @staticmethod
    def model_path() -> str:
        return os.path.join(MODELS_DIR, MODEL_NAME)

    def _ensure_model(self) -> str:
        path = self.model_path()
        if os.path.isdir(os.path.join(path, "am")):
            return path
        os.makedirs(MODELS_DIR, exist_ok=True)
        zip_path = path + ".zip"
        self.status = "DOWNLOADING 0%"

        def hook(blocks: int, block_size: int, total: int) -> None:
            if total > 0:
                self.status = f"DOWNLOADING {min(100, int(blocks * block_size * 100 / total))}%"

        urllib.request.urlretrieve(MODEL_URL, zip_path, hook)
        self.status = "INSTALLING MODEL"
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(MODELS_DIR)
        os.remove(zip_path)
        if not os.path.isdir(os.path.join(path, "am")):
            raise RuntimeError("downloaded voice model is invalid")
        return path

    # ------------------------------------------------------------------ thread
    def _run(self) -> None:
        try:
            import sounddevice as sd
            import vosk
            vosk.SetLogLevel(-1)
            model_dir = self._ensure_model()
            self.status = "LOADING MODEL"
            recognizer = vosk.KaldiRecognizer(vosk.Model(model_dir), 16000)
            audio: "queue.Queue[bytes]" = queue.Queue()

            def callback(indata, _frames, _time, _status) -> None:
                audio.put(bytes(indata))

            was_muted = False
            with sd.RawInputStream(samplerate=16000, blocksize=3200, dtype="int16", channels=1, callback=callback):
                self.status = "LISTENING"
                while self._running:
                    try:
                        data = audio.get(timeout=0.3)
                    except queue.Empty:
                        continue
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt(np.mean(samples * samples))) / 6000.0 if len(samples) else 0.0
                    self.level = 0.6 * self.level + 0.4 * min(1.0, rms)
                    if time.monotonic() < self.muted_until:
                        was_muted = True
                        continue
                    if was_muted:
                        recognizer.Reset()
                        was_muted = False
                    if recognizer.AcceptWaveform(data):
                        text = json.loads(recognizer.Result()).get("text", "").strip()
                        if text:
                            self.on_final(text)
                    elif self.on_partial is not None:
                        self.on_partial(json.loads(recognizer.PartialResult()).get("partial", ""))
        except Exception as exc:  # noqa: BLE001 - audio stacks raise many different errors
            name = type(exc).__name__
            self.status = "NO MICROPHONE" if "PortAudio" in name or "portaudio" in str(exc).lower() else "ERROR"
            self.error = f"{name}: {exc}"[:120]
        finally:
            self._running = False
            self.level = 0.0
            if self.status in ("LISTENING", "LOADING MODEL", "INSTALLING MODEL") or self.status.startswith("DOWNLOADING"):
                self.status = "OFF"


class Speaker:
    """Speaks short replies with the OS text-to-speech (PowerShell/SAPI on Windows, `say`, `espeak`)."""

    def __init__(self, listener: Optional[VoiceListener] = None) -> None:
        self.listener = listener
        self.speaking = False
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None

    def say(self, text: str) -> None:
        if not settings.ai.voice_replies or not text:
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="speaker", daemon=True)
            self._thread.start()
        self._queue.put(text[:400])

    def stop(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            self.speaking = True
            if self.listener is not None:
                self.listener.muted_until = time.monotonic() + 120.0
            try:
                self._speak(text)
            except (OSError, subprocess.SubprocessError):
                pass
            finally:
                self.speaking = False
                if self.listener is not None:
                    self.listener.muted_until = time.monotonic() + 0.5

    def _speak(self, text: str) -> None:
        clean = text.encode("ascii", "ignore").decode("ascii")
        if sys.platform.startswith("win"):
            script = ("Add-Type -AssemblyName System.Speech; "
                      "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 1; "
                      "$s.Speak([Console]::In.ReadToEnd())")
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                self._proc.communicate(clean.encode("ascii"), timeout=40)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            return
        command = None
        if sys.platform == "darwin" and shutil.which("say"):
            command = ["say", clean]
        elif shutil.which("espeak"):
            command = ["espeak", clean]
        elif shutil.which("spd-say"):
            command = ["spd-say", "-w", clean]
        if command:
            self._proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                self._proc.wait(timeout=40)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ==========================================================================
# HAND TRACKER
# ==========================================================================

# ---------------------------------------------------------------- landmark ids
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_NAMES = ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")
FINGER_CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
FINGERTIPS = (4, 8, 12, 16, 20)
PALM_POINTS = (0, 5, 9, 13, 17)
LANDMARK_FINGER = tuple([0] + [(i - 1) // 4 for i in range(1, 21)])

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

_DIRECTIONS = ("RIGHT", "UP-RIGHT", "UP", "UP-LEFT", "LEFT", "DOWN-LEFT", "DOWN", "DOWN-RIGHT")


class TrackerError(RuntimeError):
    """Raised when MediaPipe cannot be initialised."""


# ------------------------------------------------------------------ filtering
class OneEuroFilter:
    """One Euro filter for numpy arrays: strong smoothing when slow, little lag when fast."""

    def __init__(self) -> None:
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev = 0.0

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


# ------------------------------------------------------------------ data type
@dataclass
class HandData:
    """Everything the rest of the application needs to know about one hand."""

    label: str                      # "Left" / "Right" (as seen by the user)
    confidence: float               # handedness classification score 0..1
    landmarks: np.ndarray           # (21, 2) smoothed pixel coordinates
    depth: np.ndarray               # (21,) MediaPipe relative depth (kept for future 3D work)
    hand_size: float                # palm size in pixels (scale reference)
    palm_center: np.ndarray         # (2,) pixels
    bbox: Tuple[int, int, int, int]
    velocity: np.ndarray            # (2,) palm velocity in px/s
    speed: float
    direction: str
    stale: bool = False             # True while the hand is only being "coasted" after a detection loss

    def tip(self, finger: int) -> np.ndarray:
        return self.landmarks[FINGERTIPS[finger]]

    @property
    def index_tip(self) -> np.ndarray:
        return self.landmarks[INDEX_TIP]

    @property
    def thumb_tip(self) -> np.ndarray:
        return self.landmarks[THUMB_TIP]

    @property
    def roll(self) -> float:
        """Rotation of the hand in the image plane (radians)."""
        v = self.landmarks[MIDDLE_MCP] - self.landmarks[WRIST]
        return math.atan2(float(v[1]), float(v[0]))


class _Track:
    def __init__(self) -> None:
        self.filter = OneEuroFilter()
        self.missed = 0
        self.last: Optional[HandData] = None
        self.prev_center: Optional[np.ndarray] = None
        self.prev_t = 0.0
        self.velocity = np.zeros(2, dtype=np.float32)


# -------------------------------------------------------------------- tracker
class HandTracker:
    """MediaPipe Hands wrapper that returns smoothed, enriched HandData objects."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise TrackerError(
                "MediaPipe is not installed. Run:  pip install -r requirements.txt"
            ) from exc

        if not hasattr(mp, "solutions"):
            raise TrackerError(
                "The installed MediaPipe version no longer ships the 'solutions' API.\n"
                "Install the tested version:  pip install mediapipe==0.10.14"
            )
        self._mp = mp
        self._hands = self._create_hands()
        self._tracks: Dict[str, _Track] = {}
        self.inference_ms = 0.0

    def _create_hands(self):
        cfg = self.cfg
        try:
            return self._mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=cfg.max_hands,
                model_complexity=cfg.model_complexity,
                min_detection_confidence=cfg.detection_confidence,
                min_tracking_confidence=cfg.tracking_confidence,
            )
        except Exception as exc:  # noqa: BLE001 - MediaPipe raises many exception types
            raise TrackerError(f"MediaPipe failed to initialise: {exc}") from exc

    def reconfigure(self) -> None:
        """Re-create the MediaPipe model after detection settings changed (complexity, confidence, max hands)."""
        new = self._create_hands()
        old, self._hands = self._hands, new
        try:
            old.close()
        except Exception:  # noqa: BLE001
            pass
        self._tracks.clear()

    # ------------------------------------------------------------- processing
    def process(self, frame_bgr: np.ndarray, now: float, mirrored: bool = True) -> List[HandData]:
        """Detect hands in a BGR frame. Returns hands sorted Right first, Left second."""
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.cfg.process_width / float(w))
        small = frame_bgr if scale >= 1.0 else cv2.resize(
            frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        t0 = cv2.getTickCount()
        try:
            results = self._hands.process(rgb)
        except Exception as exc:  # noqa: BLE001
            raise TrackerError(f"Hand tracking failed: {exc}") from exc
        self.inference_ms = (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()

        detections: List[Tuple[float, str, np.ndarray]] = []
        if results.multi_hand_landmarks:
            handed = results.multi_handedness or []
            for i, hl in enumerate(results.multi_hand_landmarks):
                label, score = "Right", 0.0
                if i < len(handed):
                    cls = handed[i].classification[0]
                    label, score = cls.label, float(cls.score)
                if not mirrored:  # MediaPipe assumes a selfie (mirrored) image
                    label = "Left" if label == "Right" else "Right"
                arr = np.fromiter(
                    (v for p in hl.landmark for v in (p.x, p.y, p.z)), dtype=np.float32, count=63
                ).reshape(21, 3)
                detections.append((score, label, arr))

        detections.sort(key=lambda d: -d[0])
        assigned: List[Tuple[str, float, np.ndarray]] = []
        used = set()
        for score, label, arr in detections:
            if label in used:  # MediaPipe occasionally reports two hands with the same label
                label = "Left" if label == "Right" else "Right"
            if label in used:
                continue
            used.add(label)
            assigned.append((label, score, arr))

        hands: List[HandData] = []
        size_arr = np.array([w, h], dtype=np.float32)
        for label, score, arr in assigned:
            track = self._tracks.setdefault(label, _Track())
            track.missed = 0
            filtered = track.filter(arr, now, self.cfg.min_cutoff, self.cfg.beta)
            px = (filtered[:, :2] * size_arr).astype(np.float32)
            hand = self._build_hand(track, label, score, px, filtered[:, 2].copy(), now, w)
            track.last = hand
            hands.append(hand)

        for label in list(self._tracks.keys()):
            if label in used:
                continue
            track = self._tracks[label]
            track.missed += 1
            if track.last is not None and track.missed <= self.cfg.lost_grace_frames:
                hands.append(
                    dataclasses.replace(
                        track.last,
                        stale=True,
                        velocity=np.zeros(2, dtype=np.float32),
                        speed=0.0,
                        direction="STILL",
                    )
                )
            else:
                del self._tracks[label]

        hands.sort(key=lambda hd: 0 if hd.label == "Right" else 1)
        return hands

    def _build_hand(self, track: _Track, label: str, score: float, px: np.ndarray,
                    depth: np.ndarray, now: float, frame_w: int) -> HandData:
        palm = px[list(PALM_POINTS)].mean(axis=0)
        size = max(
            float(np.linalg.norm(px[WRIST] - px[MIDDLE_MCP])),
            float(np.linalg.norm(px[INDEX_MCP] - px[PINKY_MCP])),
            1.0,
        )
        mn, mx = px.min(axis=0), px.max(axis=0)
        bbox = (int(mn[0]), int(mn[1]), int(mx[0]), int(mx[1]))

        if track.prev_center is not None:
            dt = max(now - track.prev_t, 1e-3)
            raw = (palm - track.prev_center) / dt
            track.velocity = 0.6 * track.velocity + 0.4 * raw
        track.prev_center, track.prev_t = palm.copy(), now
        speed = float(np.linalg.norm(track.velocity))
        direction = self._direction(track.velocity, speed, self.cfg.still_speed_fraction * frame_w)

        return HandData(
            label=label, confidence=score, landmarks=px, depth=depth, hand_size=size,
            palm_center=palm.astype(np.float32), bbox=bbox, velocity=track.velocity.copy(),
            speed=speed, direction=direction,
        )

    @staticmethod
    def _direction(vel: np.ndarray, speed: float, still_threshold: float) -> str:
        if speed < still_threshold:
            return "STILL"
        angle = math.degrees(math.atan2(-float(vel[1]), float(vel[0])))
        return _DIRECTIONS[int(((angle + 22.5) % 360.0) // 45.0)]

    def reset(self) -> None:
        self._tracks.clear()

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:  # noqa: BLE001
            pass


# ==========================================================================
# CUSTOM GESTURE LEARNING
# ==========================================================================

class CustomGestureLearner:
    SLOTS = 3
    COUNTDOWN = 3.0
    RECORD_FRAMES = 45
    MATCH_DISTANCE = 0.17          # RMS landmark distance (in hand-size units) that still counts as a match
    VOTES_NEEDED = 6               # consecutive matching frames before an action fires
    COOLDOWN = 1.5

    def __init__(self, output_dir: str) -> None:
        self.path = os.path.join(output_dir, "custom_gestures.json")
        self.samples: Dict[int, np.ndarray] = {}
        self.training_slot: Optional[int] = None
        self._train_start = 0.0
        self._record_start: Optional[float] = None
        self._buffer: List[np.ndarray] = []
        self.current: Optional[int] = None       # slot matched in the current frame (for the UI)
        self._votes = 0
        self._vote_slot: Optional[int] = None
        self._cool_until = 0.0
        self._armed = True
        self._fired_slot: Optional[int] = None
        self._last_match = -10.0
        self.load()

    # ----------------------------------------------------------------- features
    @staticmethod
    def features(hand: HandData) -> np.ndarray:
        """Translation-, scale- and rotation-invariant description of the pose (40 numbers)."""
        lm = hand.landmarks
        origin = lm[0]
        v = lm[9] - origin
        rot = -math.pi / 2 - math.atan2(float(v[1]), float(v[0]))      # rotate so the hand points up
        c, s = math.cos(rot), math.sin(rot)
        rot_m = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = ((lm - origin) @ rot_m.T) / max(hand.hand_size, 1.0)
        return pts[1:].reshape(-1).astype(np.float32)

    # ----------------------------------------------------------------- training
    def start_training(self, slot: int, now: float) -> None:
        self.training_slot = int(min(max(slot, 1), self.SLOTS))
        self._train_start = now
        self._record_start = None
        self._buffer = []

    def cancel(self) -> None:
        self.training_slot = None
        self._record_start = None
        self._buffer = []

    @property
    def training(self) -> bool:
        return self.training_slot is not None

    def training_status(self, now: float) -> Optional[tuple]:
        """(label, progress 0..1) for the on-screen ring, or None."""
        if self.training_slot is None:
            return None
        if self._record_start is None:
            left = max(0.0, self.COUNTDOWN - (now - self._train_start))
            return (f"GESTURE {self.training_slot}: POSE IN {int(math.ceil(left))}", 1.0 - left / self.COUNTDOWN)
        return (f"RECORDING GESTURE {self.training_slot}", len(self._buffer) / self.RECORD_FRAMES)

    def feed(self, hand: Optional[HandData], now: float) -> Optional[str]:
        """Advance training. Returns a message when training finished or timed out."""
        if self.training_slot is None:
            return None
        if now - self._train_start > self.COUNTDOWN + 12.0:
            self.cancel()
            return "TRAINING TIMED OUT - NO HAND"
        if now - self._train_start < self.COUNTDOWN or hand is None or hand.stale:
            return None
        if self._record_start is None:
            self._record_start = now
        self._buffer.append(self.features(hand))
        if len(self._buffer) >= self.RECORD_FRAMES:
            slot = self.training_slot
            self.samples[slot] = np.stack(self._buffer)
            self.cancel()
            self.save()
            return f"GESTURE {slot} TRAINED"
        return None

    # ---------------------------------------------------------------- detection
    def score(self, feature: np.ndarray) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for slot, data in self.samples.items():
            d = np.linalg.norm(data - feature[None, :], axis=1) / math.sqrt(len(feature) / 2)
            out[slot] = float(np.mean(np.sort(d)[:3]))
        return out

    def detect(self, hand: Optional[HandData], now: float) -> Optional[int]:
        """Return a slot number only on the frame a custom gesture fires (debounced, then re-armed)."""
        self.current = None
        matched: Optional[int] = None
        if hand is not None and not hand.stale and self.samples and self.training_slot is None:
            scores = self.score(self.features(hand))
            best = min(scores, key=scores.get)
            ordered = sorted(scores.values())
            ambiguous = len(ordered) > 1 and (ordered[1] - ordered[0]) < 0.02
            if scores[best] <= self.MATCH_DISTANCE and not ambiguous:
                matched = best

        if matched is None:
            self._votes, self._vote_slot = 0, None
            if now - self._last_match > 0.5:
                self._armed = True               # pose was released: allow the next trigger
            return None

        self.current = matched
        self._last_match = now
        if matched == self._vote_slot:
            self._votes += 1
        else:
            self._vote_slot, self._votes = matched, 1
        can_fire = self._armed or matched != self._fired_slot     # switching straight to another pose is allowed
        if self._votes >= self.VOTES_NEEDED and can_fire and now >= self._cool_until:
            self._armed = False
            self._fired_slot = matched
            self._cool_until = now + self.COOLDOWN
            return matched
        return None

    # -------------------------------------------------------------- persistence
    def clear(self) -> None:
        self.samples.clear()
        self.current = None
        self.save()

    def trained(self, slot: int) -> bool:
        return slot in self.samples

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            data = {str(k): np.round(v, 4).tolist() for k, v in self.samples.items()}
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"slots": data}, fh)
        except OSError:
            pass

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh).get("slots", {})
            for key, rows in data.items():
                arr = np.asarray(rows, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 40 and 1 <= int(key) <= self.SLOTS:
                    self.samples[int(key)] = arr
        except (OSError, ValueError, json.JSONDecodeError):
            self.samples = {}


# ==========================================================================
# CAMERA
# ==========================================================================

class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened."""


class Camera:
    """Threaded webcam reader that always exposes the most recent frame.

    Reading in a separate thread keeps the main loop from blocking on the
    driver and removes buffered (stale) frames, which lowers latency.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._last_seq = 0
        self._last_ok = 0.0
        self.actual_size: Tuple[int, int] = (0, 0)
        self.last_error = ""

    # ------------------------------------------------------------------ state
    @property
    def connected(self) -> bool:
        if self._cap is None or not self._running:
            return False
        return (time.monotonic() - self._last_ok) < self.cfg.stall_timeout

    # ---------------------------------------------------------------- opening
    @staticmethod
    def _backends() -> List[Optional[int]]:
        if sys.platform.startswith("win"):
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
        if sys.platform == "darwin":
            return [cv2.CAP_AVFOUNDATION, None]
        return [cv2.CAP_V4L2, None]

    def open(self, index: Optional[int] = None, probe_on_failure: bool = False) -> None:
        """Open the webcam. Raises CameraError with a readable message."""
        self.release()
        if index is not None:
            self.cfg.index = index
        idx = self.cfg.index

        if idx < 0:
            self._fail(f"Invalid camera index {idx}. The index must be 0 or higher.")

        for backend in self._backends():
            cap = None
            try:
                cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue
                if self.cfg.fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
                cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                first = None
                for _ in range(15):  # some cameras need a moment to deliver the first frame
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        first = frame
                        break
                    time.sleep(0.05)
                if first is None:
                    cap.release()
                    continue

                self._cap = cap
                self.actual_size = (first.shape[1], first.shape[0])
                self._frame = first
                self._seq = 1
                self._last_seq = 0
                self._last_ok = time.monotonic()
                self._running = True
                self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
                self._thread.start()
                self.last_error = ""
                return
            except cv2.error:
                if cap is not None:
                    cap.release()
                continue

        message = (
            f"Could not open camera index {idx}.\n"
            "  - Close other programs that use the webcam (Teams, Zoom, browser, Camera app).\n"
            "  - Windows: Settings > Privacy & security > Camera > allow camera access and\n"
            "    'Let desktop apps access your camera'.\n"
            "  - Try another camera index:  python main.py --camera 1\n"
            "  - Check that the webcam is plugged in and visible in Device Manager."
        )
        if probe_on_failure:
            found = self.probe()
            if found:
                message += f"\n  - Working camera indices detected: {found}"
            else:
                message += "\n  - No working camera was detected on indices 0-%d." % (self.cfg.probe_indices - 1)
        self._fail(message)

    def _fail(self, message: str) -> None:
        self.last_error = message
        raise CameraError(message)

    def probe(self) -> List[int]:
        """Return the camera indices that can be opened."""
        found: List[int] = []
        for i in range(self.cfg.probe_indices):
            try:
                cap = cv2.VideoCapture(i, self._backends()[0]) if self._backends()[0] is not None else cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    found.append(i)
                if cap is not None:
                    cap.release()
            except cv2.error:
                pass
        return found

    def set_resolution(self, width: int, height: int) -> None:
        """Re-open the camera with a new requested resolution."""
        old = (self.cfg.width, self.cfg.height)
        self.cfg.width, self.cfg.height = width, height
        try:
            self.open()
        except CameraError:
            self.cfg.width, self.cfg.height = old
            self.open()
            raise

    # ---------------------------------------------------------------- reading
    def _reader(self) -> None:
        cap = self._cap
        while self._running and cap is not None:
            try:
                ok, frame = cap.read()
            except cv2.error:
                ok, frame = False, None
            if ok and frame is not None:
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._last_ok = time.monotonic()
                    self._cond.notify_all()
            else:
                time.sleep(0.01)

    def read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Return the newest frame, or None if no new frame arrived in time."""
        with self._cond:
            if self._seq == self._last_seq:
                self._cond.wait(timeout)
            if self._seq == self._last_seq or self._frame is None:
                return None
            self._last_seq = self._seq
            return self._frame

    # ---------------------------------------------------------------- closing
    def release(self) -> None:
        """Stop the reader thread and free the camera device."""
        self._running = False
        thread, cap = self._thread, self._cap
        self._thread, self._cap = None, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        if cap is not None and (thread is None or not thread.is_alive()):
            try:
                cap.release()
            except cv2.error:
                pass
        self._frame = None


# ==========================================================================
# GESTURE RECOGNITION
# ==========================================================================

class Gesture(Enum):
    NONE = "NONE"
    OPEN_PALM = "OPEN PALM"
    FIST = "CLOSED FIST"
    INDEX_UP = "INDEX UP"
    TWO_FINGERS = "TWO FINGERS"
    THREE_FINGERS = "THREE FINGERS"
    PINCH = "PINCH"
    THUMB_UP = "THUMB UP"


@dataclass
class GestureResult:
    label: str
    raw: Gesture = Gesture.NONE            # classification of the current frame
    gesture: Gesture = Gesture.NONE        # stabilised gesture
    previous: Gesture = Gesture.NONE
    changed: bool = False                  # True only on the frame the stable gesture changed
    held: float = 0.0                      # seconds the stable gesture has been held
    fingers: Tuple[bool, bool, bool, bool, bool] = (False,) * 5
    pinch_ratio: float = 1.0
    pinch_active: bool = False             # hysteresis-filtered pinch state (fast, un-debounced)
    pinch_point: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    stability: float = 0.0                 # 0..1 agreement of the recent raw classifications


class ActionGate:
    """Cooldown helper: ``ready(key, now, cooldown)`` is True at most once per cooldown period."""

    def __init__(self) -> None:
        self._next: Dict[str, float] = {}

    def ready(self, key: str, now: float, cooldown: float) -> bool:
        if now >= self._next.get(key, 0.0):
            self._next[key] = now + cooldown
            return True
        return False

    def reset(self) -> None:
        self._next.clear()


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b (degrees) formed by a-b-c."""
    v1, v2 = a - b, c - b
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / n))))


class _HandState:
    def __init__(self, label: str) -> None:
        self.label = label
        self.fingers = [False] * 5
        self.pinch = False
        self.candidate = Gesture.NONE
        self.candidate_count = 0
        self.stable = Gesture.NONE
        self.since = 0.0
        self.history: Deque[Gesture] = deque(maxlen=10)
        self.last = GestureResult(label=label)


class GestureRecognizer:
    """Classifies every tracked hand once per frame."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._states: Dict[str, _HandState] = {}

    def reset(self) -> None:
        self._states.clear()

    # ----------------------------------------------------------------- public
    def update(self, hands: List[HandData], now: float) -> Dict[str, GestureResult]:
        results: Dict[str, GestureResult] = {}
        present = {h.label for h in hands}
        for label in list(self._states):
            if label not in present:
                del self._states[label]

        for hand in hands:
            st = self._states.setdefault(hand.label, _HandState(hand.label))
            if hand.stale:  # coasting after a detection loss: keep the last decision
                last = st.last
                last.changed = False
                last.held = now - st.since
                results[hand.label] = last
                continue
            results[hand.label] = self._classify_hand(st, hand, now)
        return results

    # --------------------------------------------------------------- internals
    def _classify_hand(self, st: _HandState, hand: HandData, now: float) -> GestureResult:
        cfg = self.cfg
        sens = max(0.3, cfg.sensitivity)
        lm, size = hand.landmarks, hand.hand_size
        wrist = lm[WRIST]

        # --- finger extension -------------------------------------------------
        fingers = list(st.fingers)

        tip, ip = lm[THUMB_TIP], lm[THUMB_IP]
        d_idx = float(np.linalg.norm(tip - lm[INDEX_MCP])) / size
        thumb_open = float(np.linalg.norm(tip - lm[PINKY_MCP])) > 0.98 * float(np.linalg.norm(ip - lm[PINKY_MCP]))
        thr = (cfg.thumb_off if st.fingers[0] else cfg.thumb_on) / sens
        fingers[0] = d_idx > thr and thumb_open

        for k in range(1, 5):
            mcp, pip, _dip, tip_i = FINGER_CHAINS[k]
            base = max(float(np.linalg.norm(lm[mcp] - wrist)), 1e-6)
            ratio = float(np.linalg.norm(lm[tip_i] - wrist)) / base
            if k == 4:  # the little finger is shorter
                ratio *= 1.05
            ang = _angle(lm[mcp], lm[pip], lm[tip_i])
            shift = (sens - 1.0) * 20.0
            if st.fingers[k]:
                fingers[k] = ratio > cfg.ext_ratio_off / sens and ang > cfg.ext_angle_off - shift
            else:
                fingers[k] = ratio > cfg.ext_ratio_on / sens and ang > cfg.ext_angle_on - shift
        st.fingers = fingers

        # --- pinch --------------------------------------------------------------
        pinch_ratio = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP])) / size
        limit = (cfg.pinch_exit if st.pinch else cfg.pinch_enter) * sens
        pinch_point = ((lm[THUMB_TIP] + lm[INDEX_TIP]) * 0.5).astype(np.float32)
        # a fist often tucks the thumb against the index finger: a real pinch happens away from the palm
        away_from_palm = float(np.linalg.norm(pinch_point - hand.palm_center)) / size > 0.45
        st.pinch = pinch_ratio < limit and (away_from_palm or st.pinch)

        # --- thumb direction (for THUMB UP) -------------------------------------
        v = lm[THUMB_TIP] - lm[THUMB_MCP]
        n = float(np.linalg.norm(v))
        thumb_points_up = n > 1e-6 and (-float(v[1]) / n) > cfg.thumb_up_cos

        raw = self._classify(fingers, st.pinch, thumb_points_up)

        # --- temporal stabilisation --------------------------------------------
        st.history.append(raw)
        if raw == st.candidate:
            st.candidate_count += 1
        else:
            st.candidate, st.candidate_count = raw, 1

        if raw is Gesture.PINCH:
            needed = cfg.pinch_stable_frames
        elif raw is Gesture.NONE:
            needed = cfg.none_stable_frames
        else:
            needed = cfg.stable_frames

        previous = st.stable
        changed = False
        if st.candidate != st.stable and st.candidate_count >= needed:
            st.stable = st.candidate
            st.since = now
            changed = True

        agree = sum(1 for g in st.history if g == st.stable) / max(1, len(st.history))
        result = GestureResult(
            label=hand.label, raw=raw, gesture=st.stable, previous=previous if changed else st.last.previous,
            changed=changed, held=now - st.since, fingers=tuple(fingers),  # type: ignore[arg-type]
            pinch_ratio=pinch_ratio, pinch_active=st.pinch, pinch_point=pinch_point, stability=agree,
        )
        st.last = result
        return result

    @staticmethod
    def _classify(fingers: List[bool], pinch: bool, thumb_up: bool) -> Gesture:
        thumb, index, middle, ring, pinky = fingers
        if pinch:
            return Gesture.PINCH
        none_up = not (index or middle or ring or pinky)
        if none_up and thumb and thumb_up:
            return Gesture.THUMB_UP
        if index and middle and ring and pinky:
            return Gesture.OPEN_PALM
        if index and middle and ring and not pinky:
            return Gesture.THREE_FINGERS
        if index and middle and not ring and not pinky:
            return Gesture.TWO_FINGERS
        if index and not middle and not ring and not pinky:
            return Gesture.INDEX_UP
        if none_up:
            return Gesture.FIST
        return Gesture.NONE


# ==========================================================================
# VISUAL EFFECTS
# ==========================================================================

FONT = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA


# ------------------------------------------------------------------- helpers
def ip(p) -> Tuple[int, int]:
    """Convert any 2-vector to a tuple of Python ints (OpenCV requires this)."""
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def fade(color: Color, a: float) -> Color:
    """Scale a colour toward black (== transparency on a dark HUD)."""
    a = clamp(a, 0.0, 1.0)
    return (int(color[0] * a), int(color[1] * a), int(color[2] * a))


def mix(c1: Color, c2: Color, t: float) -> Color:
    t = clamp(t, 0.0, 1.0)
    return (int(c1[0] + (c2[0] - c1[0]) * t), int(c1[1] + (c2[1] - c1[1]) * t), int(c1[2] + (c2[2] - c1[2]) * t))


def ui_scale(frame_h: int) -> float:
    return clamp(frame_h / 720.0, 0.6, 1.8)


# --------------------------------------------------------------------- text
@lru_cache(maxsize=2048)
def _char_width(ch: str, scale: float, thickness: int) -> int:
    return cv2.getTextSize(ch, FONT, scale, thickness)[0][0]


def text_width(text: str, scale: float, thickness: int = 1, tracking: float = 0.0) -> int:
    if tracking == 0:
        return cv2.getTextSize(text, FONT, scale, thickness)[0][0]
    return int(sum(_char_width(c, scale, thickness) for c in text) + tracking * max(0, len(text) - 1))


def put_text(img: np.ndarray, text: str, org: Tuple[float, float], scale: float = 0.5,
             color: Color = Palette.TEXT, thickness: int = 1, tracking: float = 0.0,
             anchor: str = "left") -> None:
    """Draw ASCII text. ``tracking`` adds letter spacing for a futuristic look."""
    text = text.encode("ascii", "replace").decode("ascii")
    if not text or max(color) <= 0:
        return
    x, y = float(org[0]), float(org[1])
    width = text_width(text, scale, thickness, tracking)
    if anchor == "center":
        x -= width / 2.0
    elif anchor == "right":
        x -= width
    if tracking == 0:
        cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, thickness, AA)
        return
    for ch in text:
        cv2.putText(img, ch, (int(x), int(y)), FONT, scale, color, thickness, AA)
        x += _char_width(ch, scale, thickness) + tracking


# ------------------------------------------------------------ shapes / panels
def fill_rect_alpha(frame: np.ndarray, rect: Tuple[float, float, float, float], color: Color, alpha: float) -> None:
    x, y, w, h = (int(v) for v in rect)
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(roi, 1.0 - alpha, overlay, alpha, 0, roi)


def glass_panel(frame: np.ndarray, rect: Tuple[float, float, float, float], alpha: float = 0.55,
                tint: Color = Palette.BACKGROUND, blur: bool = True) -> None:
    """Semi-transparent 'frosted glass' rectangle."""
    x, y, w, h = (int(v) for v in rect)
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    roi = frame[y0:y1, x0:x1]
    if blur and settings.ui.glass_blur:
        roi[:] = cv2.blur(roi, (9, 9))
    overlay = np.empty_like(roi)
    overlay[:] = tint
    cv2.addWeighted(roi, 1.0 - alpha, overlay, alpha, 0, roi)


def draw_brackets(frame: np.ndarray, rect: Tuple[float, float, float, float], color: Color,
                  length: float = 14, thickness: int = 2, glow: Optional["GlowLayer"] = None) -> None:
    x, y, w, h = (int(v) for v in rect)
    L = int(length)
    corners = (
        ((x, y), (x + L, y), (x, y + L)),
        ((x + w, y), (x + w - L, y), (x + w, y + L)),
        ((x, y + h), (x + L, y + h), (x, y + h - L)),
        ((x + w, y + h), (x + w - L, y + h), (x + w, y + h - L)),
    )
    for c, a, b in corners:
        cv2.line(frame, c, a, color, thickness, AA)
        cv2.line(frame, c, b, color, thickness, AA)
        if glow is not None:
            glow.line(c, a, fade(color, 0.6), 4)
            glow.line(c, b, fade(color, 0.6), 4)


def draw_panel(frame: np.ndarray, glow: "GlowLayer", rect: Tuple[float, float, float, float], title: str,
               alpha: float, s: float, accent: Optional[Color] = None, bg_alpha: Optional[float] = None) -> None:
    """Holographic panel: glass background, thin border, corner brackets and a title bar."""
    if alpha <= 0.01:
        return
    accent = Palette.PRIMARY if accent is None else accent
    bg_alpha = settings.ui.panel_opacity if bg_alpha is None else bg_alpha
    x, y, w, h = (int(v) for v in rect)
    glass_panel(frame, (x, y, w, h), bg_alpha * alpha)
    cv2.rectangle(frame, (x, y), (x + w, y + h), fade(accent, 0.35 * alpha), 1, AA)
    draw_brackets(frame, (x, y, w, h), fade(accent, alpha), 14 * s, 2, glow if alpha > 0.5 else None)
    if title:
        pad = int(14 * s)
        cv2.rectangle(frame, (x + pad, y + int(13 * s)), (x + pad + int(6 * s), y + int(19 * s)), fade(accent, alpha), -1)
        put_text(frame, title, (x + pad + 14 * s, y + 22 * s), 0.48 * s, fade(accent, alpha), 1, 2.0 * s)
        cv2.line(frame, (x + pad, y + int(31 * s)), (x + w - pad, y + int(31 * s)), fade(accent, 0.45 * alpha), 1, AA)


def draw_dashed_circle(img: np.ndarray, center, radius: float, color: Color, angle0: float = 0.0,
                       segments: int = 24, fill_ratio: float = 0.55, thickness: int = 1) -> None:
    c, r = ip(center), int(radius)
    step = 360.0 / segments
    for k in range(segments):
        a = angle0 + k * step
        cv2.ellipse(img, c, (r, r), 0, a, a + step * fill_ratio, color, thickness, AA)


def draw_hex(img: np.ndarray, center, radius: float, color: Color, angle: float = 0.0, thickness: int = 1) -> None:
    pts = [
        (center[0] + radius * math.cos(angle + k * math.pi / 3), center[1] + radius * math.sin(angle + k * math.pi / 3))
        for k in range(6)
    ]
    cv2.polylines(img, [np.array([ip(p) for p in pts], dtype=np.int32)], True, color, thickness, AA)


def draw_progress_ring(img: np.ndarray, center, radius: float, progress: float, color: Color,
                       thickness: int = 3) -> None:
    c, r = ip(center), int(radius)
    cv2.circle(img, c, r, fade(color, 0.25), 1, AA)
    if progress > 0:
        cv2.ellipse(img, c, (r, r), -90, 0, 360.0 * clamp(progress, 0, 1), color, thickness, AA)


# ---------------------------------------------------------------- glow layer
class GlowLayer:
    """Low-resolution additive layer that is blurred and added on top of the frame.

    Drawing the glow at 1/4 resolution keeps the cost very small while still
    producing a convincing neon bloom.
    """

    def __init__(self) -> None:
        self._buf: Optional[np.ndarray] = None
        self._w = self._h = 0
        self._scale = 0.25
        self.enabled = True

    def begin(self, w: int, h: int) -> None:
        self.enabled = settings.ui.glow
        self._scale = settings.ui.glow_scale
        self._w, self._h = w, h
        if not self.enabled:
            return
        sw, sh = max(8, int(w * self._scale)), max(8, int(h * self._scale))
        if self._buf is None or self._buf.shape[0] != sh or self._buf.shape[1] != sw:
            self._buf = np.zeros((sh, sw, 3), dtype=np.uint8)
        else:
            self._buf.fill(0)

    def _p(self, p) -> Tuple[int, int]:
        return (int(float(p[0]) * self._scale), int(float(p[1]) * self._scale))

    def _t(self, thickness: float) -> int:
        return max(1, int(thickness * self._scale + 0.5))

    def line(self, p1, p2, color: Color, thickness: float = 8) -> None:
        if self.enabled and self._buf is not None:
            cv2.line(self._buf, self._p(p1), self._p(p2), color, self._t(thickness), AA)

    def circle(self, c, radius: float, color: Color, thickness: float = -1) -> None:
        if self.enabled and self._buf is not None:
            t = -1 if thickness < 0 else self._t(thickness)
            cv2.circle(self._buf, self._p(c), max(1, int(radius * self._scale)), color, t, AA)

    def ellipse(self, c, axes, angle: float, start: float, end: float, color: Color, thickness: float = 8) -> None:
        if self.enabled and self._buf is not None:
            ax = (max(1, int(axes[0] * self._scale)), max(1, int(axes[1] * self._scale)))
            cv2.ellipse(self._buf, self._p(c), ax, angle, start, end, color, self._t(thickness), AA)

    def fill_poly(self, pts: np.ndarray, color: Color) -> None:
        if self.enabled and self._buf is not None and len(pts) >= 3:
            scaled = (np.asarray(pts, dtype=np.float32) * self._scale).astype(np.int32)
            cv2.fillConvexPoly(self._buf, scaled, color, AA)

    def rect(self, rect, color: Color, thickness: float = 6) -> None:
        if self.enabled and self._buf is not None:
            x, y, w, h = rect
            cv2.rectangle(self._buf, self._p((x, y)), self._p((x + w, y + h)), color, self._t(thickness), AA)

    def compose(self, frame: np.ndarray) -> None:
        if not self.enabled or self._buf is None:
            return
        blurred = cv2.GaussianBlur(self._buf, (0, 0), 2.2)
        blurred = cv2.convertScaleAbs(blurred, alpha=settings.ui.glow_gain)
        up = cv2.resize(blurred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.add(frame, up, frame)


# ------------------------------------------------------------- ambient filter
class Ambient:
    """Pre-computed vignette + scanline + cool tint applied with a single multiply."""

    def __init__(self) -> None:
        self._mask: Optional[np.ndarray] = None
        self._size = (0, 0)
        self._key: tuple = ()

    @staticmethod
    def _current_key() -> tuple:
        u = settings.ui
        return (u.feed_dim, u.scanlines, u.vignette)

    def _build(self, w: int, h: int) -> None:
        dim = settings.ui.feed_dim
        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        if settings.ui.vignette:
            vignette = np.clip(1.0 - 0.32 * (xs ** 2 + ys ** 2), 0.45, 1.0)
        else:
            vignette = np.ones((h, w), dtype=np.float32)
        scan = np.ones((h, 1), dtype=np.float32)
        if settings.ui.scanlines:
            scan[::4] = 0.90
        tint = np.array([1.00, 0.93, 0.80], dtype=np.float32)  # B, G, R multipliers (cool blue)
        mask = dim * vignette[:, :, None] * scan[:, :, None] * tint[None, None, :]
        self._mask = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        self._size, self._key = (w, h), self._current_key()

    def apply(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._mask is None or self._size != (w, h) or self._key != self._current_key():
            self._build(w, h)
        cv2.multiply(frame, self._mask, frame, scale=1.0 / 255.0)


# ---------------------------------------------------------------- hand overlay
def _finger_color(finger: int, base: Color) -> Color:
    return mix(base, Palette.ACCENT, finger / 6.0)


def draw_hand_overlay(frame: np.ndarray, glow: GlowLayer, hand: HandData, result: Optional[GestureResult],
                      now: float, s: float, alpha: float = 1.0) -> None:
    """Glowing skeleton, joints, scanning rings and tracking brackets for one hand."""
    ui = settings.ui
    base = Palette.PRIMARY if hand.label == "Right" else Palette.ACCENT
    a = alpha * (0.45 if hand.stale else 1.0)
    lm = hand.landmarks

    if ui.scan_effects:
        _draw_scan(frame, glow, hand, base, now, s, a)

    style = ui.landmark_style
    if style in ("FULL", "LINES"):
        for p, q in HAND_CONNECTIONS:
            col = _finger_color(LANDMARK_FINGER[q], base)
            cv2.line(frame, ip(lm[p]), ip(lm[q]), fade(col, a), max(1, int(2 * s)), AA)
            glow.line(lm[p], lm[q], fade(col, 0.8 * a), 9 * s)
    if style in ("FULL", "DOTS"):
        for i in range(21):
            col = fade(_finger_color(LANDMARK_FINGER[i], base), a)
            if i in FINGERTIPS:
                pulse = 0.5 + 0.5 * math.sin(now * 6.0 + i)
                r = int((5 + 1.5 * pulse) * s)
                cv2.circle(frame, ip(lm[i]), r, col, max(1, int(2 * s)), AA)
                cv2.circle(frame, ip(lm[i]), max(1, int(2 * s)), fade(Palette.TEXT, a), -1, AA)
                glow.circle(lm[i], 11 * s, col)
            else:
                cv2.circle(frame, ip(lm[i]), max(1, int(3 * s)), fade(Palette.TEXT, 0.9 * a), -1, AA)

    # pinch indicator: line between thumb and index with distance read-out
    if result is not None and not hand.stale and result.pinch_ratio < 0.9:
        t, i = lm[THUMB_TIP], lm[INDEX_TIP]
        col = Palette.SUCCESS if result.pinch_active else Palette.WARNING
        cv2.line(frame, ip(t), ip(i), fade(col, a), max(1, int(2 * s)), AA)
        glow.line(t, i, fade(col, a), 8 * s)
        mid = (t + i) * 0.5
        cv2.circle(frame, ip(mid), int((7 + (4 if result.pinch_active else 0)) * s), fade(col, a), 2, AA)
        if result.pinch_active:
            glow.circle(mid, 18 * s, fade(col, a))


def _draw_scan(frame: np.ndarray, glow: GlowLayer, hand: HandData, base: Color, now: float, s: float,
               a: float) -> None:
    c = hand.palm_center
    R = hand.hand_size * 1.55
    col = fade(base, a)
    dim = fade(base, 0.45 * a)

    draw_dashed_circle(frame, c, R, dim, angle0=now * 35.0, segments=28, fill_ratio=0.5)
    r2 = int(R * 0.9)
    cv2.ellipse(frame, ip(c), (r2, r2), 0, now * 110.0, now * 110.0 + 70, col, max(1, int(2 * s)), AA)
    cv2.ellipse(frame, ip(c), (r2, r2), 0, 180 - now * 80.0, 180 - now * 80.0 + 50, col, max(1, int(2 * s)), AA)
    glow.ellipse(c, (r2, r2), 0, now * 110.0, now * 110.0 + 70, col, 9 * s)

    for k in range(12):  # tick marks
        ang = math.radians(k * 30 + now * 12.0)
        p1 = (c[0] + math.cos(ang) * R * 1.03, c[1] + math.sin(ang) * R * 1.03)
        p2 = (c[0] + math.cos(ang) * R * 1.11, c[1] + math.sin(ang) * R * 1.11)
        cv2.line(frame, ip(p1), ip(p2), dim, 1, AA)

    sweep = math.radians((now * 140.0) % 360.0)  # radar sweep with a fading trail
    for k in range(5):
        ang = sweep - k * 0.09
        p = (c[0] + math.cos(ang) * R * 0.88, c[1] + math.sin(ang) * R * 0.88)
        cv2.line(frame, ip(c), ip(p), fade(base, a * (0.8 - k * 0.15)), 1, AA)

    phase = (now * 0.9) % 1.0  # horizontal scanning line across the hand circle
    dy = (phase * 2.0 - 1.0) * R
    half = math.sqrt(max(R * R - dy * dy, 0.0))
    cv2.line(frame, ip((c[0] - half, c[1] + dy)), ip((c[0] + half, c[1] + dy)), col, 1, AA)
    glow.line((c[0] - half, c[1] + dy), (c[0] + half, c[1] + dy), col, 7 * s)

    draw_hex(frame, c, hand.hand_size * 0.22, col, now * 0.8, 1)

    x1, y1, x2, y2 = hand.bbox
    m = int(14 * s)
    draw_brackets(frame, (x1 - m, y1 - m, (x2 - x1) + 2 * m, (y2 - y1) + 2 * m), col, 16 * s, 2, glow)
    tag = f"{hand.label.upper()} HAND  {int(hand.confidence * 100):d}%" + ("  REACQUIRING" if hand.stale else "")
    put_text(frame, tag, (x1 - m, y1 - m - 8 * s), 0.42 * s, col, 1, 1.5 * s)


def draw_cursor_ring(frame: np.ndarray, center, radius: float, color: Color, now: float, thickness: int = 2) -> None:
    c = ip(center)
    r = int(radius)
    cv2.circle(frame, c, r, color, thickness, AA)
    for k in range(4):
        ang = math.pi / 2 * k + now * 1.5
        p1 = (c[0] + math.cos(ang) * (r + 3), c[1] + math.sin(ang) * (r + 3))
        p2 = (c[0] + math.cos(ang) * (r + 10), c[1] + math.sin(ang) * (r + 10))
        cv2.line(frame, ip(p1), ip(p2), color, 1, AA)


# ------------------------------------------------------------------ particles
class ParticleSystem:
    """Small numpy-based spark system (clicks, drawing, grabbing)."""

    MAX = 240

    def __init__(self) -> None:
        self.pos = np.zeros((0, 2), dtype=np.float32)
        self.vel = np.zeros((0, 2), dtype=np.float32)
        self.life = np.zeros(0, dtype=np.float32)
        self.max_life = np.zeros(0, dtype=np.float32)
        self.color = np.zeros((0, 3), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.life)

    def emit(self, pos, count: int, color: Color, speed: float = 170.0, life: float = 0.6) -> None:
        if not settings.features.particles:
            return
        n = min(count, self.MAX - len(self.life))
        if n <= 0:
            return
        ang = np.random.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)
        sp = (np.random.uniform(0.3, 1.0, n) * speed).astype(np.float32)
        vel = np.stack([np.cos(ang) * sp, np.sin(ang) * sp], axis=1)
        lf = (np.random.uniform(0.5, 1.0, n) * life).astype(np.float32)
        self.pos = np.vstack([self.pos, np.tile(np.asarray(pos, dtype=np.float32), (n, 1))])
        self.vel = np.vstack([self.vel, vel])
        self.life = np.concatenate([self.life, lf])
        self.max_life = np.concatenate([self.max_life, lf])
        self.color = np.vstack([self.color, np.tile(np.array(color, dtype=np.int32), (n, 1))])

    def update(self, dt: float) -> None:
        if not len(self.life):
            return
        self.life -= dt
        keep = self.life > 0
        if not keep.all():
            self.pos, self.vel, self.life = self.pos[keep], self.vel[keep], self.life[keep]
            self.max_life, self.color = self.max_life[keep], self.color[keep]
        self.pos += self.vel * dt
        self.vel *= max(0.0, 1.0 - 2.2 * dt)

    def draw(self, frame: np.ndarray, glow: "GlowLayer", s: float) -> None:
        for p, life, ml, c in zip(self.pos, self.life, self.max_life, self.color):
            a = float(life / ml)
            col = fade((int(c[0]), int(c[1]), int(c[2])), a)
            cv2.circle(frame, ip(p), max(1, int((1.0 + 2.0 * a) * s)), col, -1, AA)
            glow.circle(p, 7 * s, fade(col, 0.8))


# --------------------------------------------------------------------- trails
class TrailManager:
    """Fading comet trails behind each index fingertip."""

    def __init__(self) -> None:
        self._trails: Dict[str, deque] = {}

    def update(self, hands, now: float) -> None:
        present = set()
        for hand in hands:
            if hand.stale:
                continue
            present.add(hand.label)
            dq = self._trails.setdefault(hand.label, deque(maxlen=60))
            dq.append((now, float(hand.landmarks[INDEX_TIP][0]), float(hand.landmarks[INDEX_TIP][1])))
        for label in list(self._trails):
            dq = self._trails[label]
            while dq and now - dq[0][0] > 0.9:
                dq.popleft()
            if label not in present and not dq:
                del self._trails[label]

    def draw(self, frame: np.ndarray, glow: "GlowLayer", s: float) -> None:
        if not settings.features.trails:
            return
        n = settings.features.trail_length
        for label, dq in self._trails.items():
            pts = list(dq)[-n:]
            base = Palette.PRIMARY if label == "Right" else Palette.ACCENT
            for i in range(1, len(pts)):
                t = i / len(pts)
                a, b = (pts[i - 1][1], pts[i - 1][2]), (pts[i][1], pts[i][2])
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 300:   # tracking jump
                    continue
                col = fade(base, t)
                cv2.line(frame, ip(a), ip(b), col, max(1, int((1 + 4 * t) * s)), AA)
                glow.line(a, b, fade(col, 0.8), (4 + 8 * t) * s)

    def clear(self) -> None:
        self._trails.clear()


def draw_sparkline(frame: np.ndarray, rect, values, vmin: float, vmax: float, color: Color, alpha: float) -> None:
    """Draw a small line graph of ``values`` inside ``rect``."""
    x, y, w, h = (float(v) for v in rect)
    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), fade(color, 0.25 * alpha), 1, AA)
    n = len(values)
    if n < 2 or vmax <= vmin:
        return
    pts = np.array([
        (x + i * w / (n - 1), y + h - clamp((v - vmin) / (vmax - vmin), 0.0, 1.0) * h)
        for i, v in enumerate(values)
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], False, fade(color, alpha), 1, AA)


# ==========================================================================
# VIRTUAL CURSOR
# ==========================================================================

class VirtualCursor:
    """Smoothed cursor with hover / pressed / locked visual states and a click ripple."""

    def __init__(self) -> None:
        self.pos = np.array([640.0, 360.0], dtype=np.float32)
        self.target = self.pos.copy()
        self.visible = False
        self.locked = False        # hand is not in a pointing pose: cursor holds its position
        self.pressed = False       # pinch is active
        self.hovering = False      # over an interactive element
        self._ripple = 0.0
        self._ripple_pos = self.pos.copy()

    def set_target(self, pt) -> None:
        self.target = np.asarray(pt, dtype=np.float32)
        self.locked = False

    def click(self) -> None:
        self._ripple = 1.0
        self._ripple_pos = self.pos.copy()

    def update(self, dt: float) -> None:
        delta = self.target - self.pos
        dist = float(np.hypot(delta[0], delta[1]))
        k = 6.0 + 2.0 * (11 - settings.features.cursor_smooth) + min(dist, 300.0) / 300.0 * 22.0  # faster for big jumps
        self.pos += delta * (1.0 - math.exp(-dt * k))
        self._ripple = max(0.0, self._ripple - dt * 2.2)

    # ------------------------------------------------------------------ drawing
    def draw_glow(self, glow: GlowLayer, now: float, alpha: float, s: float) -> None:
        if not self.visible:
            return
        col = self._color(alpha)
        glow.circle(self.pos, 22 * s, col)

    def _color(self, alpha: float):
        if self.locked:
            return fade(Palette.PRIMARY, 0.45 * max(alpha, 0.4))
        if self.pressed:
            return Palette.ACCENT
        if self.hovering:
            return Palette.TEXT
        return Palette.PRIMARY

    def draw(self, frame: np.ndarray, now: float, alpha: float, s: float) -> None:
        if not self.visible:
            return
        col = self._color(alpha)
        s = s * settings.features.cursor_size
        radius = (10 if self.hovering or self.pressed else 14) * s
        draw_cursor_ring(frame, self.pos, radius, col, now, max(1, int(2 * s)))
        cv2.circle(frame, ip(self.pos), max(2, int(3 * s)), col, -1, AA)
        if self.pressed:
            cv2.circle(frame, ip(self.pos), int(radius * 0.7), col, -1, AA)
        if self._ripple > 0.01:
            rr = int(radius + (1.0 - self._ripple) * 46 * s)
            cv2.circle(frame, ip(self._ripple_pos), rr, fade(Palette.PRIMARY, self._ripple), max(1, int(2 * s)), AA)


# ==========================================================================
# MENU AND WIDGETS
# ==========================================================================

Rect = Tuple[float, float, float, float]

MENU_ITEMS = (
    ("home", "HOME"),
    ("drawing", "DRAWING"),
    ("objects", "OBJECT CONTROL"),
    ("settings", "ADVANCED SETTINGS"),
    ("sysinfo", "SYSTEM INFORMATION"),
    ("exit", "EXIT"),
)


class HoloButton:
    """A rectangular holographic button with hover, press-flash and active states."""

    def __init__(self, bid: str, label: str = "", swatch: Optional[Color] = None) -> None:
        self.bid = bid
        self.label = label
        self.swatch = swatch
        self.rect: Rect = (0, 0, 0, 0)
        self.hover = 0.0
        self.flash = 0.0
        self.active = False
        self.focused = False       # highlighted by gesture navigation
        self.enabled = True

    def contains(self, pt, margin: float = 0.0) -> bool:
        x, y, w, h = self.rect
        return (x - margin) <= pt[0] <= (x + w + margin) and (y - margin) <= pt[1] <= (y + h + margin)

    def step(self, dt: float, hovered: bool) -> None:
        self.hover += ((1.0 if hovered else 0.0) - self.hover) * min(1.0, dt * 14.0)
        self.flash = max(0.0, self.flash - dt * 3.0)

    def trigger(self) -> None:
        self.flash = 1.0

    def draw(self, frame: np.ndarray, glow: GlowLayer, alpha: float, dx: float, s: float, now: float,
             label_override: Optional[str] = None) -> None:
        x, y, w, h = self.rect
        x += dx
        rect = (x, y, w, h)
        hov = self.hover
        base = Palette.PRIMARY

        if self.swatch is not None:
            fill_rect_alpha(frame, rect, self.swatch, (0.55 + 0.35 * hov + 0.4 * self.flash) * alpha)
        else:
            fill_rect_alpha(frame, rect, mix(Palette.BACKGROUND, base, 0.10 + 0.25 * hov + 0.5 * self.flash),
                            (0.55 + 0.25 * hov) * alpha)

        border = mix(fade(base, 0.45), Palette.TEXT, hov) if self.enabled else fade(base, 0.2)
        if self.active:
            border = Palette.ACCENT
        if self.focused:
            pulse = 0.6 + 0.4 * math.sin(now * 8.0)
            border = mix(Palette.WARNING, Palette.TEXT, pulse)
        thick = 2 if (hov > 0.5 or self.active or self.focused) else 1
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), fade(border, alpha), thick, AA)
        if hov > 0.5 or self.active or self.focused:
            glow.rect(rect, fade(border, alpha), 8 * s)

        if self.active:  # little marker on the left edge
            cv2.rectangle(frame, (int(x), int(y + h * 0.2)), (int(x + 4 * s), int(y + h * 0.8)),
                          fade(Palette.ACCENT, alpha), -1)

        text = label_override if label_override is not None else self.label
        if text:
            scale = 0.46 * s
            tw = text_width(text, scale, 1, 1.2 * s)
            while tw > w - 8 and scale > 0.28 * s:
                scale -= 0.02
                tw = text_width(text, scale, 1, 1.2 * s)
            col = Palette.TEXT if (self.enabled and self.swatch is None) else Palette.TEXT
            put_text(frame, text, (x + w / 2, y + h / 2 + 5 * s), scale, fade(col, alpha), 1, 1.2 * s, "center")


class ToolRow:
    """A label / value row, optionally with small buttons on the right (for settings)."""

    def __init__(self, label: str, value_fn: Optional[Callable[[], str]] = None,
                 buttons: Optional[List[HoloButton]] = None, value_color: Optional[Color] = None) -> None:
        self.label = label
        self.value_fn = value_fn
        self.buttons = buttons or []
        self.value_color = value_color
        self.y = 0.0
        self.h = 0.0


class ToolPanel:
    """Left-hand contextual panel: a grid of buttons followed by label/value rows."""

    def __init__(self, key: str, title: str, buttons: Optional[List[HoloButton]] = None, columns: int = 3,
                 cell_h: float = 40, rows: Optional[List[ToolRow]] = None, row_h: float = 26,
                 accent: Optional[Color] = None, min_height: float = 0) -> None:
        self.min_height = min_height
        self.key = key
        self.title = title
        self.buttons = buttons or []
        self.columns = columns
        self.cell_h = cell_h
        self.rows = rows or []
        self.row_h = row_h
        self.accent = accent
        self.rect: Rect = (0, 0, 0, 0)
        self._s = 1.0

    def all_buttons(self) -> List[HoloButton]:
        out = list(self.buttons)
        for row in self.rows:
            out.extend(row.buttons)
        return out

    def layout(self, x: float, y: float, w: float, s: float) -> None:
        self._s = s
        pad, gap = 12 * s, 8 * s
        cy = y + 40 * s
        if self.buttons:
            bw = (w - 2 * pad - (self.columns - 1) * gap) / self.columns
            bh = self.cell_h * s
            for i, b in enumerate(self.buttons):
                r, c = divmod(i, self.columns)
                b.rect = (x + pad + c * (bw + gap), cy + r * (bh + gap), bw, bh)
            n_rows = (len(self.buttons) + self.columns - 1) // self.columns
            cy += n_rows * (bh + gap)
        for row in self.rows:
            row.y, row.h = cy, self.row_h * s
            bx = x + w - pad
            for b in reversed(row.buttons):
                bw = (74 if len(b.label) > 2 else 34) * s
                b.rect = (bx - bw, cy + 2 * s, bw, row.h - 4 * s)
                bx -= bw + 5 * s
            cy += row.h
        self.rect = (x, y, w, max(cy - y + pad, self.min_height * s))

    def contains(self, pt) -> bool:
        x, y, w, h = self.rect
        return x <= pt[0] <= x + w and y <= pt[1] <= y + h

    def draw(self, frame: np.ndarray, glow: GlowLayer, now: float, alpha: float, dx: float, s: float) -> None:
        x, y, w, h = self.rect
        draw_panel(frame, glow, (x + dx, y, w, h), self.title, alpha, s, self.accent)
        for b in self.buttons:
            b.draw(frame, glow, alpha, dx, s, now)
        pad = 14 * s
        for row in self.rows:
            ty = row.y + row.h * 0.5 + 5 * s
            put_text(frame, row.label, (x + dx + pad, ty), 0.40 * s, fade(Palette.DIM_TEXT, alpha), 1, 1.0 * s)
            vx = x + dx + w - pad
            if row.buttons:
                vx = min(b.rect[0] for b in row.buttons) + dx - 8 * s
            if row.value_fn is not None:
                text = row.value_fn()
                if row.value_color is not None:
                    vcol = row.value_color
                elif text in ("ON", "RECORDING", "ACTIVE"):
                    vcol = Palette.SUCCESS
                elif text == "OFF":
                    vcol = Palette.DIM_TEXT
                else:
                    vcol = Palette.PRIMARY
                put_text(frame, text, (vx, ty), 0.42 * s, fade(vcol, alpha), 1, 0.8 * s, "right")
            for b in row.buttons:
                b.draw(frame, glow, alpha, dx, s, now)


class TabbedPanel(ToolPanel):
    """Wide control-centre panel: a row of tab buttons above the rows of the selected tab."""

    def __init__(self, key: str, title: str, tabs: "Dict[str, List[ToolRow]]", row_h: float = 36,
                 min_height: float = 0) -> None:
        self.tabs = tabs
        self.tab_names = list(tabs.keys())
        self.tab_index = 0
        tab_buttons = [HoloButton(f"set:tab{i}", name) for i, name in enumerate(self.tab_names)]
        super().__init__(key, title, tab_buttons, len(tab_buttons), 32, tabs[self.tab_names[0]], row_h,
                         min_height=min_height)
        self._layout_args: Optional[tuple] = None

    def layout(self, x: float, y: float, w: float, s: float) -> None:
        self._layout_args = (x, y, w, s)
        super().layout(x, y, w, s)

    def select_tab(self, index: int) -> None:
        self.tab_index = max(0, min(index, len(self.tab_names) - 1))
        self.rows = self.tabs[self.tab_names[self.tab_index]]
        if self._layout_args is not None:
            self.layout(*self._layout_args)

    def draw(self, frame: np.ndarray, glow: GlowLayer, now: float, alpha: float, dx: float, s: float) -> None:
        for i, b in enumerate(self.buttons):
            b.active = i == self.tab_index
        super().draw(frame, glow, now, alpha, dx, s)


class HoloMenu:
    """Main holographic menu on the right-hand side."""

    def __init__(self) -> None:
        self.buttons = [HoloButton(f"menu:{key}", label) for key, label in MENU_ITEMS]
        self.rect: Rect = (0, 0, 0, 0)
        self._s = 1.0

    def layout(self, frame_w: int, frame_h: int, s: float) -> None:
        self._s = s
        m, w = 20 * s, 270 * s
        x, y = frame_w - m - w, m
        pad, gap, bh = 14 * s, 8 * s, 46 * s
        cy = y + 42 * s
        for b in self.buttons:
            b.rect = (x + pad, cy, w - 2 * pad, bh)
            cy += bh + gap
        self.rect = (x, y, w, cy - y + 6 * s)

    def contains(self, pt) -> bool:
        x, y, w, h = self.rect
        return x <= pt[0] <= x + w and y <= pt[1] <= y + h

    def index_of(self, key: str) -> int:
        for i, (k, _label) in enumerate(MENU_ITEMS):
            if k == key:
                return i
        return 0

    def draw(self, frame: np.ndarray, glow: GlowLayer, now: float, alpha: float, dx: float, s: float,
             active_key: str, focus_index: Optional[int], exit_armed: bool) -> None:
        x, y, w, h = self.rect
        draw_panel(frame, glow, (x + dx, y, w, h), "MAIN MENU", alpha, s)
        for i, b in enumerate(self.buttons):
            key = MENU_ITEMS[i][0]
            b.active = key == active_key
            b.focused = focus_index is not None and focus_index == i
            override = "CONFIRM EXIT?" if (key == "exit" and exit_armed) else None
            b.draw(frame, glow, alpha, dx, s, now, override)


# ==========================================================================
# AI SHAPE RECOGNITION
# ==========================================================================

def _path_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def recognize_shape(points: List[np.ndarray]) -> Optional[Tuple[str, np.ndarray]]:
    """Return (name, polyline) for a recognised shape, or None to keep the freehand stroke.

    The returned polyline is an (N, 2) array; closed shapes repeat their first point at the end.
    """
    if len(points) < 12:
        return None
    pts = np.asarray(points, dtype=np.float32)
    length = _path_length(pts)
    if length < 90:
        return None
    start, end = pts[0], pts[-1]
    chord = float(np.linalg.norm(end - start))

    # --- straight line ---------------------------------------------------------
    if chord / length > 0.90:
        direction = (end - start) / max(chord, 1e-6)
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        deviation = float(np.max(np.abs((pts - start) @ normal)))
        if deviation < 0.07 * length:
            return "LINE", np.array([start, end], dtype=np.float32)
        return None

    # --- closed shapes -------------------------------------------------------------
    if chord > 0.28 * length:
        return None
    contour = pts.reshape(-1, 1, 2)
    hull = cv2.convexHull(contour)
    area = float(cv2.contourArea(hull))
    perimeter = float(cv2.arcLength(hull, True))
    if area < 600 or perimeter < 1:
        return None
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)

    if circularity > 0.84:
        (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
        a, b = max(w, h) / 2.0, min(w, h) / 2.0
        if a / max(b, 1e-6) < 1.18:
            a = b = (a + b) / 2.0
            angle = 0.0
        t = np.linspace(0, 2 * math.pi, 73, dtype=np.float32)
        rad = math.radians(angle if w >= h else angle + 90)
        x = a * np.cos(t)
        y = b * np.sin(t)
        poly = np.stack([cx + x * math.cos(rad) - y * math.sin(rad), cy + x * math.sin(rad) + y * math.cos(rad)], axis=1)
        return ("CIRCLE" if a == b else "ELLIPSE"), poly.astype(np.float32)

    approx = cv2.approxPolyDP(hull, 0.045 * perimeter, True).reshape(-1, 2).astype(np.float32)
    if len(approx) == 3:
        return "TRIANGLE", np.vstack([approx, approx[:1]])
    if len(approx) == 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        fill = area / max(rect[1][0] * rect[1][1], 1.0)
        if fill > 0.80:
            return "RECTANGLE", np.vstack([box, box[:1]])
        return "QUADRILATERAL", np.vstack([approx, approx[:1]])
    return None


# ==========================================================================
# AIR DRAWING
# ==========================================================================

DRAW_COLORS: List[Tuple[str, Color]] = [
    ("CYAN", hex_to_bgr("#00E5FF")),
    ("BLUE", hex_to_bgr("#1677FF")),
    ("VIOLET", hex_to_bgr("#7B61FF")),
    ("MAGENTA", hex_to_bgr("#FF3DCB")),
    ("LIME", hex_to_bgr("#2CFF9A")),
    ("WHITE", hex_to_bgr("#EAF7FF")),
]
BRUSH_SIZES = [3, 5, 8, 12, 18, 26]
HALO_SCALE = 0.25
SYMMETRY_MODES = ["OFF", "MIRROR X", "4-WAY"]
UNDO_LIMIT = 12


class AirDrawing:
    """Transparent drawing canvas with glowing strokes, eraser, clear and PNG export."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.size = (0, 0)
        self.core: Optional[np.ndarray] = None     # BGR stroke colours
        self.mask: Optional[np.ndarray] = None     # 255 where a stroke exists
        self.halo: Optional[np.ndarray] = None     # low-res glow canvas
        self._halo_up: Optional[np.ndarray] = None # cached blurred + upscaled halo
        self._halo_dirty = False
        self.has_content = False

        self.color_index = 0
        self.brush_index = 3
        self.tool = "pen"            # "pen" or "eraser"
        self.pen_down = False
        self.last_save = ""

        self._last: Optional[np.ndarray] = None
        self._smooth: Optional[np.ndarray] = None
        self._gap_start: Optional[float] = None
        self._hue = 0.0
        self._stroke_pts: List[np.ndarray] = []
        self.shape_event: Optional[str] = None
        self._undo: List[tuple] = []
        self._redo: List[tuple] = []

    # ---------------------------------------------------------------- properties
    @property
    def color(self) -> Color:
        if settings.features.rainbow and self.tool == "pen":
            r, g, b = colorsys.hsv_to_rgb(self._hue % 1.0, 0.85, 1.0)
            return (int(b * 255), int(g * 255), int(r * 255))
        return DRAW_COLORS[self.color_index][1]

    @property
    def mode_text(self) -> str:
        parts = [settings.features.symmetry]
        if settings.features.rainbow:
            parts.append("RAINBOW")
        return " + ".join(p for p in parts if p != "OFF") or "STANDARD"

    # --------------------------------------------------------------- undo / redo
    def _snapshot(self) -> tuple:
        return (self.core.copy(), self.mask.copy(), self.halo.copy(), self.has_content)

    def _restore(self, snap: tuple) -> None:
        self.core, self.mask, self.halo, self.has_content = (snap[0].copy(), snap[1].copy(), snap[2].copy(), snap[3])
        self._halo_dirty = True
        self._last = None
        self._smooth = None

    def _push_undo(self) -> None:
        if self.core is None:
            return
        self._undo.append(self._snapshot())
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo or self.core is None:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo or self.core is None:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    @property
    def undo_depth(self) -> int:
        return len(self._undo)

    @property
    def brush(self) -> int:
        return BRUSH_SIZES[self.brush_index]

    # ------------------------------------------------------------------- canvas
    def ensure_size(self, w: int, h: int) -> None:
        if self.size == (w, h) and self.core is not None:
            return
        old = (self.core, self.mask, self.size)
        self._undo.clear()
        self._redo.clear()
        self.size = (w, h)
        self.core = np.zeros((h, w, 3), dtype=np.uint8)
        self.mask = np.zeros((h, w), dtype=np.uint8)
        hw, hh = max(8, int(w * HALO_SCALE)), max(8, int(h * HALO_SCALE))
        self.halo = np.zeros((hh, hw, 3), dtype=np.uint8)
        self._halo_up = None
        self._halo_dirty = False
        if old[0] is not None and self.has_content:  # keep the drawing when the resolution changes
            self.core = cv2.resize(old[0], (w, h), interpolation=cv2.INTER_NEAREST)
            self.mask = cv2.resize(old[1], (w, h), interpolation=cv2.INTER_NEAREST)
            self.halo = cv2.resize(self.core, (hw, hh), interpolation=cv2.INTER_AREA)
            self._halo_dirty = True
        self._last = None
        self._smooth = None
        self._undo.clear()
        self._redo.clear()

    # ----------------------------------------------------------------- controls
    def set_color(self, index: int) -> None:
        self.color_index = int(np.clip(index, 0, len(DRAW_COLORS) - 1))
        self.tool = "pen"

    def next_color(self) -> None:
        self.set_color((self.color_index + 1) % len(DRAW_COLORS))

    def change_brush(self, delta: int) -> None:
        self.brush_index = int(np.clip(self.brush_index + delta, 0, len(BRUSH_SIZES) - 1))

    def set_tool(self, tool: str) -> None:
        self.tool = tool

    def toggle_eraser(self) -> None:
        self.tool = "pen" if self.tool == "eraser" else "eraser"

    def clear(self) -> None:
        if self.core is None:
            return
        if self.has_content:
            self._push_undo()
        self.core.fill(0)
        self.mask.fill(0)
        self.halo.fill(0)
        self._halo_up = None
        self._halo_dirty = False
        self.has_content = False
        self._last = None
        self._smooth = None

    # ------------------------------------------------------------------ drawing
    def pop_shape_event(self) -> Optional[str]:
        name, self.shape_event = self.shape_event, None
        return name

    def _finish_stroke(self) -> None:
        """AI shape snapping: replace a roughly drawn circle / triangle / rectangle / line with a perfect one."""
        pts, self._stroke_pts = self._stroke_pts, []
        if (not pts or not settings.ai.shape_snap or self.tool != "pen" or settings.features.symmetry != "OFF"
                or not self._undo):
            return
        result = recognize_shape(pts)
        if result is None:
            return
        name, poly = result
        snapshot = self._undo.pop()          # canvas as it was before the freehand stroke
        self._restore(snapshot)
        self._undo.append(snapshot)
        self._redo.clear()
        for a, b in zip(poly[:-1], poly[1:]):
            self._segment((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
        self.has_content = True
        self._halo_dirty = True
        self.shape_event = name

    def pen_up(self) -> None:
        self._finish_stroke()
        self.pen_down = False
        self._last = None
        self._smooth = None
        self._gap_start = None

    def update(self, point: Optional[np.ndarray], pen_down: bool, now: float) -> None:
        """Feed the current fingertip position. ``pen_down`` decides whether it paints."""
        if self.core is None:
            return
        self.pen_down = bool(pen_down and point is not None)

        if self.pen_down:
            self._gap_start = None
            p = np.asarray(point, dtype=np.float32)
            if self._smooth is None:
                self._smooth = p.copy()
            else:
                follow = 1.0 - 0.085 * settings.features.stroke_smoothing   # level 1 = raw, 10 = very smooth
                self._smooth += (p - self._smooth) * follow
            cur = self._smooth.copy()
            w, h = self.size
            if self._last is not None and float(np.linalg.norm(cur - self._last)) > 0.30 * w:
                self._finish_stroke()
                self._last = None                               # tracking glitch: start a new stroke
            if self._last is None:
                self._push_undo()                               # one undo step per stroke
                self._stroke_pts = []
            self._hue += 0.004
            self._paint(self._last, cur)
            self._stroke_pts.append(cur.copy())
            self._last = cur
        elif self._last is not None:
            # short tolerance so a one-frame gesture flicker does not cut the stroke
            if self._gap_start is None:
                self._gap_start = now
            elif now - self._gap_start > 0.12:
                self._finish_stroke()
                self._last = None
                self._smooth = None
                self._gap_start = None

    def _symmetric(self, p: Optional[np.ndarray]) -> List[Optional[np.ndarray]]:
        """The point plus its mirror images for the active symmetry mode."""
        if p is None:
            return []
        w, h = self.size
        mode = settings.features.symmetry
        out = [p]
        if mode in ("MIRROR X", "4-WAY"):
            out.append(np.array([w - p[0], p[1]], dtype=np.float32))
        if mode == "4-WAY":
            out.append(np.array([p[0], h - p[1]], dtype=np.float32))
            out.append(np.array([w - p[0], h - p[1]], dtype=np.float32))
        return out

    def _paint(self, p0: Optional[np.ndarray], p1: np.ndarray) -> None:
        ends = self._symmetric(p1)
        starts = self._symmetric(p0) if p0 is not None else ends
        for b_pt, a_pt in zip(starts, ends):
            self._segment((int(b_pt[0]), int(b_pt[1])), (int(a_pt[0]), int(a_pt[1])))
        self._halo_dirty = True

    def _segment(self, b: Tuple[int, int], a: Tuple[int, int]) -> None:
        size = self.brush
        hb = (int(b[0] * HALO_SCALE), int(b[1] * HALO_SCALE))
        ha = (int(a[0] * HALO_SCALE), int(a[1] * HALO_SCALE))
        if self.tool == "eraser":
            r = size * 2 + 6
            cv2.line(self.core, b, a, (0, 0, 0), r * 2, cv2.LINE_AA)
            cv2.line(self.mask, b, a, 0, r * 2, cv2.LINE_AA)
            cv2.line(self.halo, hb, ha, (0, 0, 0), max(1, int(r * 2 * HALO_SCALE)))
            return
        color = self.color
        cv2.line(self.core, b, a, color, size, cv2.LINE_AA)
        cv2.line(self.mask, b, a, 255, size, cv2.LINE_AA)
        cv2.line(self.halo, hb, ha, color, max(2, int((size + 10) * HALO_SCALE)))
        self.has_content = True

    # ---------------------------------------------------------------- rendering
    def _refresh_halo(self) -> None:
        if self._halo_dirty or self._halo_up is None:
            w, h = self.size
            blurred = cv2.GaussianBlur(self.halo, (0, 0), 2.5)
            blurred = cv2.convertScaleAbs(blurred, alpha=1.6)
            self._halo_up = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
            self._halo_dirty = False

    def render(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self.ensure_size(w, h)
        if not self.has_content:
            return
        self._refresh_halo()
        cv2.add(frame, self._halo_up, frame)
        cv2.copyTo(self.core, self.mask, frame)

    # --------------------------------------------------------------------- save
    def save(self) -> str:
        """Write the drawing as PNG files (transparent + on dark background). Returns the main path."""
        if self.core is None or not self.has_content:
            raise ValueError("Nothing to save - draw something first.")
        os.makedirs(self.output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dark_path = os.path.join(self.output_dir, f"drawing_{stamp}.png")
        alpha_path = os.path.join(self.output_dir, f"drawing_{stamp}_transparent.png")

        bgra = np.dstack([self.core, self.mask])
        dark = np.empty_like(self.core)
        dark[:] = Palette.BACKGROUND
        self._refresh_halo()
        cv2.add(dark, self._halo_up, dark)
        cv2.copyTo(self.core, self.mask, dark)

        if not (cv2.imwrite(dark_path, dark) and cv2.imwrite(alpha_path, bgra)):
            raise OSError(f"Could not write PNG files to {self.output_dir}")
        self.last_save = dark_path
        return dark_path


# ==========================================================================
# OBJECT CONTROL
# ==========================================================================

# ------------------------------------------------------------------- geometry
class Mesh:
    def __init__(self, name: str, vertices, edges) -> None:
        v = np.array(vertices, dtype=np.float32)
        v /= float(np.max(np.linalg.norm(v, axis=1)))     # normalise to a unit sphere
        self.name = name
        self.vertices = v
        self.edges = tuple(edges)


def _cube() -> Mesh:
    verts = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if sum(abs(a - b) for a, b in zip(verts[i], verts[j])) == 2]
    return Mesh("CUBE", verts, edges)


def _pyramid() -> Mesh:
    verts = [(-1, -1, -0.8), (1, -1, -0.8), (1, 1, -0.8), (-1, 1, -0.8), (0, 0, 1.3)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
    return Mesh("PYRAMID", verts, edges)


def _octahedron() -> Mesh:
    verts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    edges = [(i, j) for i in range(6) for j in range(i + 1, 6)
             if float(np.dot(verts[i], verts[j])) == 0.0]
    return Mesh("OCTAHEDRON", verts, edges)


def _prism() -> Mesh:
    verts = []
    for z in (-0.6, 0.6):
        for k in range(6):
            a = k * math.pi / 3
            verts.append((math.cos(a), math.sin(a), z))
    edges = []
    for base in (0, 6):
        edges += [(base + k, base + (k + 1) % 6) for k in range(6)]
    edges += [(k, k + 6) for k in range(6)]
    return Mesh("HEX PRISM", verts, edges)


MESHES: Dict[str, Mesh] = {"cube": _cube(), "pyramid": _pyramid(), "octahedron": _octahedron(), "prism": _prism()}
KINDS = tuple(MESHES.keys())


def object_color(slot: int) -> Color:
    """Object colours follow the active theme."""
    return (Palette.PRIMARY, Palette.SECONDARY, Palette.ACCENT, Palette.SUCCESS)[slot % 4]


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# -------------------------------------------------------------------- objects
class VirtualObject:
    _next_id = 1

    def __init__(self, kind: str, pos, radius: float, slot: int) -> None:
        self.id = VirtualObject._next_id
        VirtualObject._next_id += 1
        self.kind = kind
        self.pos = np.array(pos, dtype=np.float32)
        self.radius = float(radius)
        self.scale = 1.0
        self.rot = np.array([0.5, 0.6, 0.0], dtype=np.float32)
        self.spin = np.array([0.18, 0.45, 0.0], dtype=np.float32)
        self.slot = slot
        self.vel = np.zeros(2, dtype=np.float32)   # px/s, used by the throw physics
        self.age = 0.0
        self.two: Optional[dict] = None   # two-hand manipulation state

    @property
    def color(self) -> Color:
        return object_color(self.slot)

    @property
    def mesh(self) -> Mesh:
        return MESHES[self.kind]

    @property
    def r(self) -> float:
        ease = clamp(self.age / 0.35, 0.0, 1.0)
        ease = 1.0 - (1.0 - ease) ** 3
        return self.radius * self.scale * ease

    @property
    def name(self) -> str:
        return f"OBJ-{self.id:02d} {self.mesh.name}"


@dataclass
class _Grab:
    obj_id: int
    offset: np.ndarray
    last_roll: float


class ObjectControl:
    MAX_OBJECTS = 8
    IDLE_SPIN = np.array([0.18, 0.45, 0.0], dtype=np.float32)

    def __init__(self) -> None:
        self.objects: List[VirtualObject] = []
        self.selected_id: Optional[int] = None
        self.hover_id: Optional[int] = None
        self.size = (1280, 720)
        self._grabs: Dict[str, _Grab] = {}
        self._initialised = False
        self._next_kind = 0
        self.events: List[Tuple[str, np.ndarray]] = []   # ("grab"/"release"/"bounce", position) for FX / sound

    # ----------------------------------------------------------------- queries
    def get(self, obj_id: Optional[int]) -> Optional[VirtualObject]:
        for o in self.objects:
            if o.id == obj_id:
                return o
        return None

    @property
    def selected(self) -> Optional[VirtualObject]:
        return self.get(self.selected_id)

    def is_grabbed(self, obj_id: int) -> bool:
        return any(g.obj_id == obj_id for g in self._grabs.values())

    # --------------------------------------------------------------- management
    def _base_radius(self) -> float:
        return 0.11 * self.size[1]

    def ensure_init(self, size: Tuple[int, int]) -> None:
        self.size = size
        if self._initialised:
            return
        self._initialised = True
        self.reset()

    def reset(self) -> None:
        self.objects.clear()
        self._grabs.clear()
        self.selected_id = None
        w, h = self.size
        for i, kind in enumerate(("cube", "pyramid", "octahedron")):
            self.add(kind, (w * (0.40 + 0.17 * i), h * 0.52))

    def add(self, kind: Optional[str] = None, pos=None) -> Optional[VirtualObject]:
        if not self._initialised:
            self.ensure_init(self.size)        # create the default objects first so they are not wiped later
        if len(self.objects) >= self.MAX_OBJECTS:
            return None
        if kind is None:
            kind = KINDS[self._next_kind % len(KINDS)]
            self._next_kind += 1
        if pos is None:
            w, h = self.size
            n = len(self.objects)
            pos = (w * (0.35 + 0.1 * (n % 5)), h * (0.30 + 0.25 * ((n // 5) % 2)))
        obj = VirtualObject(kind, pos, self._base_radius(), len(self.objects))
        self.objects.append(obj)
        self.selected_id = obj.id
        return obj

    def delete_selected(self) -> bool:
        if not self._initialised:
            self.ensure_init(self.size)
        obj = self.selected
        if obj is None:
            return False
        self.objects.remove(obj)
        for label in [l for l, g in self._grabs.items() if g.obj_id == obj.id]:
            del self._grabs[label]
        self.selected_id = None
        return True

    def cycle_selection(self) -> None:
        if not self.objects:
            return
        ids = [o.id for o in self.objects]
        if self.selected_id in ids:
            self.selected_id = ids[(ids.index(self.selected_id) + 1) % len(ids)]
        else:
            self.selected_id = ids[0]

    def scale_selected(self, factor: float) -> None:
        obj = self.selected
        if obj:
            obj.scale = clamp(obj.scale * factor, 0.35, 3.0)

    def rotate_selected(self, dz: float = 0.0, dy: float = 0.0, dx: float = 0.0) -> None:
        obj = self.selected
        if obj:
            obj.rot += np.array([dx, dy, dz], dtype=np.float32)

    def move_selected(self, dx: float, dy: float) -> None:
        obj = self.selected
        if obj:
            obj.pos += np.array([dx, dy], dtype=np.float32)
            self._clamp_to_frame(obj)

    def release_all(self) -> None:
        self._grabs.clear()
        self.events.clear()
        for o in self.objects:
            o.two = None

    def _clamp_to_frame(self, obj: VirtualObject) -> None:
        w, h = self.size
        r = obj.radius * obj.scale * 0.6
        obj.pos[0] = clamp(float(obj.pos[0]), r, w - r)
        obj.pos[1] = clamp(float(obj.pos[1]), r, h - r)

    def _pick(self, pt) -> Optional[VirtualObject]:
        pad = 0.035 * self.size[1]
        for o in reversed(self.objects):
            if float(np.hypot(o.pos[0] - pt[0], o.pos[1] - pt[1])) <= o.r * 1.2 + pad:
                return o
        return None

    # ------------------------------------------------------------------ update
    def update(self, size: Tuple[int, int], now: float, dt: float,
               holders: Dict[str, Tuple[HandData, GestureResult]], pinch_events: set,
               hover_pos: Optional[np.ndarray]) -> None:
        """Advance object physics and process grabbing.

        holders        label -> (hand, gesture result) for every tracked hand
        pinch_events   hand labels whose pinch just started (and was not used by the UI)
        hover_pos      pointer position used for the hover indicator, or None
        """
        self.ensure_init(size)
        for o in self.objects:
            o.age += dt

        # 1. releases ------------------------------------------------------------
        for label in list(self._grabs.keys()):
            hand, res = holders.get(label, (None, None))
            if hand is None or (not hand.stale and not res.pinch_active):
                grab = self._grabs.pop(label)
                obj = self.get(grab.obj_id)
                if obj is not None:
                    self.events.append(("release", obj.pos.copy()))
                    if hand is not None and settings.features.object_physics and not self.is_grabbed(obj.id):
                        speed = float(np.linalg.norm(hand.velocity))
                        limit = 2.5 * size[0]
                        obj.vel = hand.velocity * (min(speed, limit) / speed * 0.8) if speed > 1e-3 else obj.vel * 0

        # 2. new grabs -----------------------------------------------------------
        for label in sorted(pinch_events):
            if label in self._grabs or label not in holders:
                continue
            hand, res = holders[label]
            obj = self._pick(res.pinch_point)
            if obj is None:
                self.selected_id = None
                continue
            self._grabs[label] = _Grab(obj.id, obj.pos - res.pinch_point, hand.roll)
            obj.vel[:] = 0
            self.events.append(("grab", obj.pos.copy()))
            self.selected_id = obj.id
            self.objects.remove(obj)       # bring to front
            self.objects.append(obj)

        # 3. manipulation ----------------------------------------------------------
        by_obj: Dict[int, List[str]] = {}
        for label, grab in self._grabs.items():
            by_obj.setdefault(grab.obj_id, []).append(label)
        w, h = size
        follow = 1.0 - math.exp(-dt * 26.0)

        for obj_id, labels in by_obj.items():
            obj = self.get(obj_id)
            if obj is None:
                continue
            labels.sort()
            live = [l for l in labels if not holders[l][0].stale]
            if len(live) >= 2:
                self._two_hand(obj, holders, live[:2], follow)
            elif len(live) == 1:
                self._one_hand(obj, holders, live[0], follow, w, h)
            if len(live) < 2:
                obj.two = None

        # 4. idle physics ------------------------------------------------------------
        held_ids = set(by_obj.keys())
        for o in self.objects:
            if o.id not in held_ids:
                o.spin += (self.IDLE_SPIN - o.spin) * (1.0 - math.exp(-dt * 1.5))
                if settings.features.object_physics and float(np.hypot(o.vel[0], o.vel[1])) > 4.0:
                    self._physics(o, dt)
                else:
                    o.vel[:] = 0
            o.rot += o.spin * dt
            self._clamp_to_frame(o)

        # 5. hover -------------------------------------------------------------------
        self.hover_id = None
        if hover_pos is not None and not self._grabs:
            o = self._pick(hover_pos)
            if o is not None:
                self.hover_id = o.id

    def _physics(self, o: VirtualObject, dt: float) -> None:
        """Free flight after a throw: friction + bouncing off the frame edges."""
        w, h = self.size
        o.pos += o.vel * dt
        o.vel *= math.exp(-dt * 1.4)
        r = o.radius * o.scale * 0.6
        for axis, limit in ((0, w), (1, h)):
            if o.pos[axis] < r:
                o.pos[axis], o.vel[axis] = r, abs(o.vel[axis]) * 0.8
                self.events.append(("bounce", o.pos.copy()))
            elif o.pos[axis] > limit - r:
                o.pos[axis], o.vel[axis] = limit - r, -abs(o.vel[axis]) * 0.8
                self.events.append(("bounce", o.pos.copy()))
        o.spin[1] += float(o.vel[0]) / w * 0.5      # a thrown object tumbles

    def _one_hand(self, obj: VirtualObject, holders, label: str, follow: float, w: int, h: int) -> None:
        hand, res = holders[label]
        grab = self._grabs[label]
        if obj.two is not None:                       # just dropped from two-hand mode
            grab.offset = obj.pos - res.pinch_point
            grab.last_roll = hand.roll
            obj.two = None
        target = res.pinch_point + grab.offset
        obj.pos += (target - obj.pos) * follow

        roll = hand.roll                               # hand roll -> rotation around the view axis
        delta = clamp(wrap_angle(roll - grab.last_roll), -0.35, 0.35)
        grab.last_roll = roll
        obj.rot[2] += delta

        vx, vy = float(hand.velocity[0]) / w, float(hand.velocity[1]) / h   # movement -> tumbling
        obj.spin[1] += (clamp(vx * 5.0, -6.0, 6.0) - obj.spin[1]) * 0.2
        obj.spin[0] += (clamp(-vy * 5.0, -6.0, 6.0) - obj.spin[0]) * 0.2

    def _two_hand(self, obj: VirtualObject, holders, labels: List[str], follow: float) -> None:
        p1, p2 = holders[labels[0]][1].pinch_point, holders[labels[1]][1].pinch_point
        mid = (p1 + p2) * 0.5
        dist = float(np.linalg.norm(p2 - p1))
        ang = math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0]))
        if obj.two is None:
            obj.two = {"d0": max(dist, 30.0), "s0": obj.scale, "last": ang, "acc": 0.0,
                       "rz0": float(obj.rot[2]), "off": obj.pos - mid}
        t = obj.two
        t["acc"] += wrap_angle(ang - t["last"])
        t["last"] = ang
        obj.scale = clamp(t["s0"] * dist / t["d0"], 0.35, 3.0)    # pinch distance between the hands -> scale
        obj.rot[2] = t["rz0"] + t["acc"]
        obj.pos += ((mid + t["off"]) - obj.pos) * follow
        obj.spin[:2] *= 0.8

    # ------------------------------------------------------------------ render
    def render(self, frame: np.ndarray, glow: GlowLayer, now: float, alpha: float, s: float) -> None:
        for obj in self.objects:
            self._draw(frame, glow, obj, now, alpha, s)

    def _draw(self, frame: np.ndarray, glow: GlowLayer, obj: VirtualObject, now: float, alpha: float,
              s: float) -> None:
        r = obj.r
        if r < 2:
            return
        mesh = obj.mesh
        v = mesh.vertices @ rotation_matrix(*obj.rot).T
        persp = 3.2 / (3.2 - v[:, 2])
        pts = obj.pos[None, :] + v[:, :2] * persp[:, None] * r

        grabbed = self.is_grabbed(obj.id)
        selected = obj.id == self.selected_id
        hover = obj.id == self.hover_id
        col = obj.color
        if grabbed:
            col = Palette.TEXT

        hull = cv2.convexHull(pts.astype(np.int32))[:, 0, :]          # translucent holographic body
        glow.fill_poly(hull, fade(col, 0.14 * alpha))

        for a, b in mesh.edges:
            depth = (float(v[a, 2]) + float(v[b, 2])) * 0.5
            bright = 0.40 + 0.60 * (depth + 1.0) * 0.5
            c = fade(col, bright * alpha)
            cv2.line(frame, ip(pts[a]), ip(pts[b]), c, 2 if depth > 0 else 1, AA)
            glow.line(pts[a], pts[b], c, 9 * s)
        for i in range(len(pts)):
            cv2.circle(frame, ip(pts[i]), max(1, int((3 if v[i, 2] > 0 else 2) * s)), fade(Palette.TEXT, alpha), -1, AA)

        # projector base
        base_c = (obj.pos[0], obj.pos[1] + r * 1.30)
        for k in range(3):
            a0 = now * 70 + k * 120
            cv2.ellipse(frame, ip(base_c), (int(r * 0.95), max(2, int(r * 0.22))), 0, a0, a0 + 70,
                        fade(col, 0.7 * alpha), 1, AA)

        if grabbed or selected or hover:
            ring_col = Palette.TEXT if grabbed else (Palette.PRIMARY if selected else Palette.DIM_TEXT)
            ring_r = r * (1.42 if grabbed else 1.35)
            draw_dashed_circle(frame, obj.pos, ring_r, fade(ring_col, alpha), now * (60 if grabbed else 30), 20, 0.6,
                               2 if grabbed else 1)
            glow.circle(obj.pos, ring_r, fade(ring_col, 0.25 * alpha), 6 * s)
            m = r * 1.15
            draw_brackets(frame, (obj.pos[0] - m, obj.pos[1] - m, 2 * m, 2 * m), fade(ring_col, alpha), 12 * s, 2)
            state = "GRABBED" if grabbed else ("SELECTED" if selected else "HOVER")
            put_text(frame, f"{obj.name}  {state}", (obj.pos[0] - m, obj.pos[1] - m - 8 * s), 0.42 * s,
                     fade(ring_col, alpha), 1, 1.5 * s)
            put_text(frame, f"SCALE {obj.scale:.2f}X", (obj.pos[0] - m, obj.pos[1] + m + 18 * s), 0.40 * s,
                     fade(Palette.DIM_TEXT, alpha), 1, 1.2 * s)

        if obj.age < 0.6:                                              # spawn flash
            f = 1.0 - obj.age / 0.6
            cv2.circle(frame, ip(obj.pos), int(r * (1.0 + (1 - f) * 1.2)), fade(Palette.PRIMARY, f * alpha), 2, AA)


# ==========================================================================
# GESTURE NAVIGATION
# ==========================================================================

class SwipeDetector:
    """Detects fast, mostly straight palm movements in four directions."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._hist: Deque[Tuple[float, float, float]] = deque()
        self._cooldown_until = 0.0

    def reset(self) -> None:
        self._hist.clear()

    def update(self, palm: np.ndarray, now: float, frame_w: int, frame_h: int) -> Optional[str]:
        """Return "LEFT", "RIGHT", "UP", "DOWN" or None."""
        self._hist.append((now, float(palm[0]), float(palm[1])))
        while self._hist and now - self._hist[0][0] > self.cfg.swipe_window:
            self._hist.popleft()
        if now < self._cooldown_until or len(self._hist) < 4:
            return None

        t0, x0, y0 = self._hist[0]
        dx, dy = float(palm[0]) - x0, float(palm[1]) - y0
        duration = max(now - t0, 1e-3)
        need_x = self.cfg.swipe_distance * frame_w
        need_y = self.cfg.swipe_distance * frame_h * 1.2

        direction = None
        if abs(dx) >= need_x and abs(dy) < 0.6 * abs(dx) and abs(dx) / duration > 0.6 * frame_w:
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) >= need_y and abs(dx) < 0.6 * abs(dy) and abs(dy) / duration > 0.6 * frame_h:
            direction = "DOWN" if dy > 0 else "UP"

        if direction:
            self._cooldown_until = now + self.cfg.swipe_cooldown
            self._hist.clear()
        return direction


class GestureNavigator:
    """Keeps the menu focus that two-finger swipes move around (thumb-up then confirms it)."""

    FOCUS_TIMEOUT = 5.0

    def __init__(self, cfg: GestureConfig, item_count: int) -> None:
        self.swipe = SwipeDetector(cfg)
        self.item_count = item_count
        self.focus = 0
        self.focus_until = 0.0

    def begin(self, current_index: int, now: float) -> None:
        """Called when navigation mode starts (two fingers raised)."""
        if now >= self.focus_until:
            self.focus = current_index
        self.focus_until = now + self.FOCUS_TIMEOUT
        self.swipe.reset()

    def move_focus(self, delta: int, now: float) -> None:
        self.focus = (self.focus + delta) % self.item_count
        self.focus_until = now + self.FOCUS_TIMEOUT

    def active_focus(self, now: float) -> Optional[int]:
        return self.focus if now < self.focus_until else None

    def clear(self) -> None:
        self.focus_until = 0.0
        self.swipe.reset()


# ==========================================================================
# AI VOICE ASSISTANT
# ==========================================================================

OLLAMA_URL = "http://127.0.0.1:11434"
SYSTEM_PROMPT = ("You are HOLO, the voice assistant of a futuristic holographic hand-tracking interface. "
                 "Answer in one or two short plain sentences, without markdown or lists.")

WAKE_VARIANTS = {
    "JARVIS": ["jarvis", "jarvas", "service", "jarvus"],
    "COMPUTER": ["computer"],
    "HOLO": ["holo", "hollow", "halo", "hello"],
}

THEME_WORDS = {"cyan": "CYAN", "matrix": "MATRIX", "green": "MATRIX", "crimson": "CRIMSON", "red": "CRIMSON",
               "amber": "AMBER", "gold": "AMBER", "orange": "AMBER", "yellow": "AMBER", "ultraviolet": "ULTRAVIOLET",
               "violet": "ULTRAVIOLET", "purple": "ULTRAVIOLET", "ice": "ICE", "blue": "ICE", "white": "ICE"}
COLOR_WORDS = {"cyan": 0, "blue": 1, "violet": 2, "purple": 2, "magenta": 3, "pink": 3, "green": 4, "lime": 4,
               "white": 5}
OBJECT_WORDS = {"cube": "cube", "box": "cube", "pyramid": "pyramid", "octahedron": "octahedron", "octa": "octahedron",
                "prism": "prism", "hexagon": "prism"}
MODE_WORDS = [("system information", "sysinfo"), ("system info", "sysinfo"), ("diagnostics", "sysinfo"),
              ("object control", "objects"), ("objects", "objects"), ("object", "objects"),
              ("advanced settings", "settings"), ("settings", "settings"), ("drawing", "drawing"),
              ("draw", "drawing"), ("home", "home")]
TAB_WORDS = {"gesture": 0, "tracking": 1, "display": 2, "effects": 3, "effect": 3, "features": 4, "feature": 4,
             "ai": 5, "custom": 6, "system": 7}
TOGGLES = {  # spoken name -> (settings section, attribute)
    "glow": ("ui", "glow"), "scan effects": ("ui", "scan_effects"), "scanlines": ("ui", "scanlines"),
    "vignette": ("ui", "vignette"), "blur": ("ui", "glass_blur"), "sound": ("features", "sound"),
    "physics": ("features", "object_physics"), "trails": ("features", "trails"),
    "particles": ("features", "particles"), "hints": ("features", "show_hints"),
    "graph": ("features", "fps_graph"), "shape snap": ("ai", "shape_snap"), "rainbow": ("features", "rainbow"),
    "voice replies": ("ai", "voice_replies"), "custom gestures": ("ai", "custom_enabled"),
}
NUMBER_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
NUMBER_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
OPERATORS = {"plus": "+", "minus": "-", "times": "*", "multiplied": "*", "x": "*", "divided": "/", "over": "/"}
_SAFE_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


# ---------------------------------------------------------------------- helpers
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 .+\-*/]", " ", text.lower())).strip()


def evaluate_math(text: str) -> Optional[str]:
    """Spoken or typed arithmetic ("what is twenty five times four") -> answer text, or None."""
    words = re.sub(r"[^a-z0-9 .+\-*/]", " ", text.lower()).split()
    expr: List[str] = []
    current: Optional[float] = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            expr.append(repr(current))
            current = None

    seen_number = False
    for w in words:
        if re.fullmatch(r"\d+(\.\d+)?", w):
            flush()
            current = float(w)
            seen_number = True
        elif w in NUMBER_UNITS:
            current = (current or 0) + NUMBER_UNITS[w]
            seen_number = True
        elif w in NUMBER_TENS:
            current = (current or 0) + NUMBER_TENS[w]
            seen_number = True
        elif w == "hundred":
            current = (current or 1) * 100
        elif w == "thousand":
            current = (current or 1) * 1000
        elif w in OPERATORS or w in "+-*/":
            flush()
            expr.append(OPERATORS.get(w, w))
        # every other word ("what", "is", "by", "and" ...) is ignored
    flush()
    if not seen_number or not any(t in "+-*/" for t in expr) or len(expr) < 3:
        return None
    try:
        tree = ast.parse(" ".join(expr), mode="eval")
        value = _eval_node(tree.body)
    except (SyntaxError, ZeroDivisionError, ValueError):
        return None
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval_node(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    raise ValueError("unsupported expression")


class LocalLLM:
    """Minimal client for a local Ollama server (no third-party packages)."""

    def available(self, timeout: float = 0.6) -> bool:
        try:
            with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=timeout):
                return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def ask(self, prompt: str, model: str, timeout: float = 90.0) -> str:
        body = json.dumps({"model": model, "prompt": prompt, "system": SYSTEM_PROMPT, "stream": False,
                           "options": {"num_predict": 90, "temperature": 0.6}}).encode("utf-8")
        request = urllib.request.Request(OLLAMA_URL + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(f"Model {model} is not installed. Run: ollama pull {model}") from exc
            raise RuntimeError(f"Local AI error {exc.code}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RuntimeError("Local AI is not running. Install Ollama and start it.") from exc
        return " ".join(str(data.get("response", "")).split())[:420] or "I have no answer for that."


# -------------------------------------------------------------------- assistant
class Assistant:
    """Understands commands and controls the whole application through the interaction engine."""

    AWAKE_SECONDS = 8.0

    def __init__(self, engine) -> None:
        self.engine = engine
        self.listener = VoiceListener(self._on_final, self._on_partial)
        self.speaker = Speaker(self.listener)
        self.llm = LocalLLM()
        self.state = "IDLE"                    # IDLE / LISTENING / THINKING / SPEAKING
        self.heard = ""
        self.partial = ""
        self.reply = ""
        self.reply_time = -100.0
        self.heard_time = -100.0
        self.awake_until = 0.0
        self.typing = False
        self.buffer = ""
        self.history: Deque[Tuple[str, str]] = deque(maxlen=20)
        self.llm_status = "OFF"
        self._pending: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._llm_replies: "queue.Queue[str]" = queue.Queue()
        self._thinking = 0
        self._last_llm_check = -100.0
        self._hint_printed = False
        self._rules = self._build_rules()

    # ------------------------------------------------------------ life cycle
    def start(self) -> None:
        if settings.ai.voice_enabled:
            self.listener.start()
            if self.listener.status == "NOT INSTALLED" and not self._hint_printed:
                self._hint_printed = True
                print("[INFO] Voice assistant needs two optional packages:  python -m pip install vosk sounddevice\n"
                      "       (typed commands still work: press I in the application window)")

    def stop(self) -> None:
        self.listener.stop()
        self.speaker.stop()

    @property
    def voice_status(self) -> str:
        if not settings.ai.voice_enabled:
            return "OFF"
        return self.listener.status

    # ------------------------------------------------------- thread callbacks
    def _on_final(self, text: str) -> None:
        self._pending.put((text, "voice"))

    def _on_partial(self, text: str) -> None:
        self.partial = text

    # --------------------------------------------------------- per-frame update
    def update(self, now: float) -> None:
        ai = settings.ai
        if ai.voice_enabled and not self.listener.running and self.listener.status in ("OFF", "LISTENING"):
            self.listener.start()
        elif not ai.voice_enabled and self.listener.running:
            self.listener.stop()

        while True:
            try:
                text, source = self._pending.get_nowait()
            except queue.Empty:
                break
            self.handle(text, now, source)
        while True:
            try:
                answer = self._llm_replies.get_nowait()
            except queue.Empty:
                break
            self._thinking = max(0, self._thinking - 1)
            self._say(answer, now)

        if ai.llm_enabled and now - self._last_llm_check > 20.0:
            self._last_llm_check = now
            threading.Thread(target=self._check_llm, daemon=True).start()
        elif not ai.llm_enabled:
            self.llm_status = "OFF"

        if self.speaker.speaking:
            self.state = "SPEAKING"
        elif self._thinking:
            self.state = "THINKING"
        elif self.listener.running and (now < self.awake_until or ai.wake_word == "OFF"):
            self.state = "LISTENING"
        else:
            self.state = "IDLE"
        if not self.listener.running:
            self.partial = ""

    def _check_llm(self) -> None:
        self.llm_status = "ONLINE" if self.llm.available() else "OFFLINE"

    # ------------------------------------------------------------ typed input
    def begin_typing(self) -> None:
        self.typing = True
        self.buffer = ""

    def handle_typing_key(self, key: str, now: float) -> bool:
        """Consume a key press while the command box is open. Returns True if consumed."""
        if not self.typing:
            return False
        if key == "esc":
            self.typing = False
        elif key == "enter":
            self.typing = False
            if self.buffer.strip():
                self.handle(self.buffer, now, "typed")
        elif key == "backspace":
            self.buffer = self.buffer[:-1]
        elif key == "space":
            self.buffer += " "
        elif len(key) == 1 and len(self.buffer) < 80:
            self.buffer += key
        return True

    # ------------------------------------------------------------ understanding
    @staticmethod
    def _strip_wake(norm: str, wake: str) -> Optional[str]:
        """Return the text after the wake word, or None if the wake word is absent."""
        variants = WAKE_VARIANTS.get(wake, [wake.lower()])
        tokens = norm.split()
        for i, tok in enumerate(tokens[:4]):          # the wake word must be at the start of the sentence
            if any(difflib.SequenceMatcher(None, tok, v).ratio() >= 0.78 for v in variants):
                return " ".join(tokens[i + 1:])
        return None

    def handle(self, text: str, now: float, source: str = "voice") -> None:
        norm = normalize(text)
        if not norm:
            return
        self.heard, self.heard_time = norm, now
        wake = settings.ai.wake_word
        if source == "voice" and wake != "OFF":
            rest = self._strip_wake(norm, wake)
            if now >= self.awake_until:
                if rest is None:
                    return                                   # ordinary talking: ignore it
                norm = rest
            elif rest is not None:
                norm = rest
            self.awake_until = now + self.AWAKE_SECONDS
            if not norm:
                self.engine.sound.play("on")
                self._say("Yes?", now)
                return
        self.history.append(("you", norm))
        self.process(norm, now)

    def process(self, norm: str, now: float) -> None:
        for pattern, handler in self._rules:
            match = pattern.search(norm)
            if match:
                reply = handler(match, norm, now)
                if reply is not None:
                    self._say(reply, now)
                return
        if settings.ai.llm_enabled:
            self._ask_llm(norm, now)
        else:
            self._say("Sorry, I did not understand. Say help for examples, or enable the local AI in settings.", now)

    def _say(self, text: str, now: float) -> None:
        self.reply, self.reply_time = text, now
        self.history.append(("holo", text))
        self.speaker.say(text)

    def speak_status(self, now: float) -> str:
        e = self.engine
        live = [h for h in e.hands if not h.stale]
        gesture = e.primary_result.gesture.value.lower() if e.primary_result else "none"
        text = (f"Running at {e.fps:.0f} frames per second. {len(live)} hand{'s' if len(live) != 1 else ''} tracked. "
                f"Mode {e.mode.value.lower()}. Current gesture {gesture}.")
        self._say(text, now)
        return text

    def _ask_llm(self, prompt: str, now: float) -> None:
        self._thinking += 1
        model = settings.ai.llm_model

        def work() -> None:
            try:
                self._llm_replies.put(self.llm.ask(prompt, model))
            except RuntimeError as exc:
                self._llm_replies.put(str(exc))

        threading.Thread(target=work, name="llm", daemon=True).start()

    # ------------------------------------------------------------------ rules
    def _build_rules(self) -> List[Tuple["re.Pattern", Callable]]:
        e = self.engine
        perform = lambda bid, now=0.0: e.perform(bid)  # noqa: E731

        def rule(pattern: str, fn: Callable) -> Tuple["re.Pattern", Callable]:
            return re.compile(pattern), fn

        def onoff(norm: str) -> Optional[bool]:
            if re.search(r"\b(off|disable|hide|stop|deactivate)\b", norm):
                return False
            if re.search(r"\b(on|enable|show|start|activate)\b", norm):
                return True
            return None

        def sleep(m, n, now):
            self.awake_until = 0.0
            return "Okay."

        def shutdown(m, n, now):
            e.request_exit()
            return "Shutting down. Goodbye."

        def show_ui(m, n, now):
            e.set_interface(True, now)
            return "Interface online."

        def hide_ui(m, n, now):
            e.set_interface(False, now)
            return "Interface standby."

        def tab(m, n, now):
            e.set_mode(e.mode_for("settings"), now)
            e.select_settings_tab(TAB_WORDS[m.group(1)])
            return f"{m.group(1)} settings."

        def mode(m, n, now):
            for word, key in MODE_WORDS:
                if word in n:
                    e.set_interface(True, now)
                    e.set_mode(e.mode_for(key), now)
                    return f"Opening {word}."
            return None

        def theme_named(m, n, now):
            name = THEME_WORDS[m.group(1)]
            settings.ui.theme = apply_theme(name)
            return f"Theme {name.lower()}."

        def theme_next(m, n, now):
            perform("set:theme")
            return f"Theme {settings.ui.theme.lower()}."

        def snapshot(m, n, now):
            e.request("snapshot")
            return "Taking a screenshot."

        def record(m, n, now):
            stopping = bool(re.search(r"\b(stop|end|finish)\b", n))
            if stopping and not e.recording:
                return "I am not recording."
            if not stopping and e.recording:
                return "Already recording."
            e.request("record_toggle")
            return "Recording stopped." if stopping else "Recording started."

        def clear(m, n, now):
            perform("draw:clear")
            return "Canvas cleared."

        def undo(m, n, now):
            perform("draw:undo")
            return "Undone."

        def redo(m, n, now):
            perform("draw:redo")
            return "Redone."

        def save_drawing(m, n, now):
            perform("draw:save")
            return "Saving your drawing."

        def eraser(m, n, now):
            perform("draw:eraser")
            return "Eraser selected."

        def pen(m, n, now):
            perform("draw:pen")
            return "Pen selected."

        def brush(m, n, now):
            bigger = bool(re.search(r"\b(bigger|larger|thicker|increase|up|more)\b", n))
            perform("draw:size_up" if bigger else "draw:size_dn")
            return "Brush bigger." if bigger else "Brush smaller."

        def colour(m, n, now):
            idx = COLOR_WORDS[m.group(2)]
            perform(f"draw:color{idx}")
            return f"{m.group(2)} ink."

        def symmetry(m, n, now):
            if re.search(r"\b(off|none|disable)\b", n):
                settings.features.symmetry = "OFF"
            elif re.search(r"\b(four|4|quad)\b", n):
                settings.features.symmetry = "4-WAY"
            else:
                settings.features.symmetry = "MIRROR X"
            return f"Symmetry {settings.features.symmetry.lower()}."

        def add_obj(m, n, now):
            kind = OBJECT_WORDS.get(m.group(2))
            if kind is None:
                e.objects.add()
            else:
                perform(f"obj:add_{kind}")
            e.set_mode(e.mode_for("objects"), now)
            return "Object added."

        def delete_obj(m, n, now):
            perform("obj:delete")
            return "Deleted."

        def reset_obj(m, n, now):
            perform("obj:reset")
            return "Objects reset."

        def obj_scale(m, n, now):
            bigger = bool(re.search(r"\b(bigger|larger|enlarge|grow)\b", n))
            e.objects.scale_selected(1.25 if bigger else 0.8)
            return "Bigger." if bigger else "Smaller."

        def adjust(setting: str):
            def fn(m, n, now):
                up = bool(re.search(r"\b(increase|raise|more|higher|up)\b", n))
                perform(f"set:{setting}_{'up' if up else 'dn'}")
                return f"{'Increased' if up else 'Decreased'} {setting}."
            return fn

        def toggle(m, n, now):
            name = m.group(1)
            for key in sorted(TOGGLES, key=len, reverse=True):
                if key in n:
                    section, attr = TOGGLES[key]
                    obj = getattr(settings, section)
                    state = onoff(n)
                    setattr(obj, attr, (not getattr(obj, attr)) if state is None else state)
                    return f"{key} {'on' if getattr(obj, attr) else 'off'}."
            return None

        def effects(m, n, now):
            state = onoff(n)
            on = (not (settings.ui.glow and settings.ui.scan_effects)) if state is None else state
            settings.ui.glow = settings.ui.scan_effects = on
            return f"Effects {'on' if on else 'off'}."

        def mute(m, n, now):
            settings.ai.voice_replies = False
            self.speaker.stop()
            return None

        def unmute(m, n, now):
            settings.ai.voice_replies = True
            return "Voice replies on."

        def landmarks(m, n, now):
            state = onoff(n)
            settings.ui.landmark_style = "OFF" if state is False else "FULL"
            return f"Landmarks {settings.ui.landmark_style.lower()}."

        def fullscreen(m, n, now):
            e.request("fullscreen")
            return "Toggling fullscreen."

        def camera(m, n, now):
            e.request("camera_next")
            return "Switching camera."

        def resolution(m, n, now):
            e.request("resolution_next")
            return "Changing resolution."

        def profile(m, n, now):
            if "save" in n:
                perform("set:profsave")
            elif "load" in n:
                perform("set:profload")
            else:
                perform("set:profreset")
            return "Done."

        def status(m, n, now):
            self.speak_status(now)
            return None

        def clock(m, n, now):
            return time.strftime("It is %I:%M %p.").replace(" 0", " ", 1)

        def date(m, n, now):
            return time.strftime("Today is %A, %B %d.")

        def who(m, n, now):
            return "I am HOLO, the assistant of this holographic interface."

        def helper(m, n, now):
            return ("Try: open drawing, change theme to matrix, take a screenshot, add a cube, undo, "
                    "start recording, what is twelve times eight, or status report.")

        def math(m, n, now):
            result = evaluate_math(n)
            return None if result is None else f"The answer is {result}."

        def train(m, n, now):
            word = m.group(2)
            slot = {"one": 1, "1": 1, "two": 2, "2": 2, "three": 3, "3": 3}[word]
            e.start_custom_training(slot, now)
            return f"Training gesture {slot}. Hold your pose after the countdown."

        def forget(m, n, now):
            e.custom.clear()
            return "Custom gestures cleared."

        themes = "|".join(THEME_WORDS)
        colors = "|".join(COLOR_WORDS)
        objects = "|".join(OBJECT_WORDS)
        toggles = "|".join(re.escape(k) for k in sorted(TOGGLES, key=len, reverse=True))
        return [
            rule(r"\b(stop listening|go to sleep|never mind|cancel that|that is all)\b", sleep),
            rule(r"\b(stop talking|be quiet|shut up|silence)\b", mute),
            rule(r"\b(unmute|talk to me|voice replies on)\b", unmute),
            rule(r"\b(shut ?down|exit|quit)( the)?( app| application| program| everything)?$", shutdown),
            rule(r"\b(show|open|activate|wake|enable|turn on|start)( the)? (interface|ui|hologram|hud)\b", show_ui),
            rule(r"\b(hide|close|deactivate|disable|turn off)( the)? (interface|ui|hologram|hud)\b|\bstandby\b", hide_ui),
            rule(r"\b(gesture|tracking|display|effects?|features?|ai|custom|system) (settings|tab|page)\b", tab),
            rule(r"\b(train|teach|learn|record) (?:custom )?(?:gesture|pose) (one|two|three|1|2|3)\b", train),
            rule(r"\b(clear|forget|delete|remove) (all )?(my )?custom gestures\b", forget),
            rule(r"\b(theme|colou?r scheme)\b.*\b(" + themes + r")\b", lambda m, n, now: theme_named(
                re.search(r"\b(" + themes + r")\b", n), n, now)),
            rule(r"\b(next|change|switch|cycle|new) (the )?(theme|colou?r scheme)\b", theme_next),
            rule(r"\b(screenshot|snapshot|take a picture|take a photo|capture the screen)\b", snapshot),
            rule(r"\b(start|begin|stop|end|finish) (the )?(video )?recording\b|\brecord (a )?video\b", record),
            rule(r"\b(clear|wipe|erase everything|erase all)( the)?( canvas| drawing| board| screen)?$|\bclear (the )?(canvas|drawing|board)\b", clear),
            rule(r"\bundo\b", undo),
            rule(r"\bredo\b", redo),
            rule(r"\bsave (the |my )?(drawing|picture|image|sketch)\b", save_drawing),
            rule(r"\b(eraser|erase mode|rubber)\b", eraser),
            rule(r"\b(pen|pencil)\b", pen),
            rule(r"\b(bigger|larger|thicker|increase|smaller|thinner|decrease|reduce)( the)? (brush|pen|line)\b|\bbrush (bigger|larger|smaller|thinner|up|down)\b", brush),
            rule(r"\b(draw|colou?r|ink|paint|use|switch to) (?:with |in )?(?:the )?(" + colors + r")\b", colour),
            rule(r"\b(symmetry|mirror)\b", symmetry),
            rule(r"\badd (a |an |the |another )?(" + objects + r"|object|shape)\b", add_obj),
            rule(r"\b(delete|remove) (the )?(selected )?(object|shape|cube|pyramid)\b", delete_obj),
            rule(r"\breset (the )?objects\b", reset_obj),
            rule(r"\b(make it|make that|scale it)? ?(bigger|larger|enlarge|smaller|shrink)\b", obj_scale),
            rule(r"\b(increase|raise|decrease|lower|reduce|more|less)( the)? sensitivity\b", adjust("sens")),
            rule(r"\b(increase|raise|decrease|lower|reduce|more|less)( the)? smoothing\b", adjust("smooth")),
            rule(r"\b(effects|holo effects)\b.*\b(on|off|enable|disable)\b|\b(enable|disable|turn on|turn off) (the )?effects\b", effects),
            rule(r"\b(skeleton|landmarks|hand points)\b", landmarks),
            rule(r"\b(" + toggles + r")\b", toggle),
            rule(r"\bfull ?screen\b", fullscreen),
            rule(r"\b(next|switch|change) (the )?camera\b", camera),
            rule(r"\b(next|change|switch) (the )?resolution\b", resolution),
            rule(r"\b(save|load|reset|restore) (my |the )?(profile|settings|defaults)\b", profile),
            rule(r"\b(status|report|diagnostics report|how are you|how am i doing|performance)\b", status),
            rule(r"\bwhat time\b|\btime is it\b|\bcurrent time\b", clock),
            rule(r"\bwhat (is )?(the )?(date|day)\b|\bwhat day\b|\btoday s date\b", date),
            rule(r"\b(who are you|what is your name|what are you)\b", who),
            rule(r"\b(help|what can you do|commands|examples)\b", helper),
            rule(r"^(what is|what s|whats|calculate|compute|how much is)\b", math),
            rule(r"\b(go to|open|show|switch to|take me to|launch)( the)? (system information|system info|diagnostics|object control|objects?|advanced settings|settings|drawing|home)\b|^(drawing|home|objects|settings)( mode)?$", mode),
        ]


# ==========================================================================
# INTERACTION ENGINE
# ==========================================================================

class AppMode(Enum):
    HOME = "HOME"
    DRAWING = "DRAWING"
    OBJECTS = "OBJECT CONTROL"
    SETTINGS = "ADVANCED SETTINGS"
    SYSINFO = "SYSTEM INFORMATION"


MODE_BY_KEY = {
    "home": AppMode.HOME,
    "drawing": AppMode.DRAWING,
    "objects": AppMode.OBJECTS,
    "settings": AppMode.SETTINGS,
    "sysinfo": AppMode.SYSINFO,
}
KEY_BY_MODE = {v: k for k, v in MODE_BY_KEY.items()}
POINTER_GESTURES = (Gesture.INDEX_UP, Gesture.TWO_FINGERS, Gesture.THREE_FINGERS)


@dataclass
class Toast:
    text: str
    kind: str
    t0: float
    duration: float


class InteractionEngine:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.gate = ActionGate()

        # state ---------------------------------------------------------------
        self.mode = AppMode.HOME
        self.interface_active = False
        self.ui_alpha = 0.0
        self.frame_size = (0, 0)
        self.s = 1.0
        self.running = True
        self.requests: List[tuple] = []
        self.toasts: List[Toast] = []
        self.hold: Optional[Tuple[str, float, np.ndarray]] = None   # (label, progress, position)
        self.exit_armed_until = 0.0
        self.hover_id: Optional[str] = None
        self.nav_active = False

        # inputs --------------------------------------------------------------
        self.hands: List[HandData] = []
        self.results: Dict[str, GestureResult] = {}
        self.primary: Optional[HandData] = None
        self.primary_result: Optional[GestureResult] = None
        self.mouse_pos: Optional[np.ndarray] = None
        self.mouse_time = -100.0

        # system info (filled in by main) -------------------------------------
        self.fps = 0.0
        self.inference_ms = 0.0
        self.camera_connected = False
        self.camera_index = 0
        self.camera_resolution = "N/A"
        self.recording = False
        self.fps_hist: deque = deque(maxlen=120)
        self.inf_hist: deque = deque(maxlen=120)
        self.last_hand_time = 0.0
        self._versions = self._collect_versions()

        # components ----------------------------------------------------------
        self._actions: Dict[str, Callable[[], None]] = {}
        self.sound = SoundFx()
        self.particles = ParticleSystem()
        self.trails = TrailManager()
        self.custom = CustomGestureLearner(output_dir)
        self.custom_label = ""
        self._custom_flash = 0.0
        self.cursor = VirtualCursor()
        self.menu = HoloMenu()
        self.drawing = AirDrawing(output_dir)
        self.objects = ObjectControl()
        self.navigator = GestureNavigator(settings.gesture, len(MENU_ITEMS))
        self.assistant = Assistant(self)
        self.panels: Dict[AppMode, ToolPanel] = {}
        self.status_rect = (0, 0, 0, 0)
        self.gesture_rect = (0, 0, 0, 0)
        self._build_panels()

    # =====================================================================
    # construction helpers
    # =====================================================================
    @staticmethod
    def _collect_versions() -> Dict[str, str]:
        import cv2
        try:
            import mediapipe as mp
            mp_version = getattr(mp, "__version__", "?")
        except Exception:  # noqa: BLE001
            mp_version = "N/A"
        return {
            "python": platform.python_version(), "opencv": cv2.__version__,
            "numpy": np.__version__, "mediapipe": mp_version,
        }

    def _request(self, name: str) -> None:
        if (name,) not in self.requests:
            self.requests.append((name,))

    # --- row factories (each registers its button actions in self._actions) ---
    def _stepper(self, label: str, key: str, getter, setter, lo, hi, step, fmt, post=None) -> ToolRow:
        def make(direction: int):
            def fn() -> None:
                cur = getter()
                val = min(hi, max(lo, cur + direction * step))
                val = int(round(val)) if isinstance(cur, int) else round(val, 3)
                setter(val)
                if post:
                    post()
            return fn
        self._actions[key + "_dn"] = make(-1)
        self._actions[key + "_up"] = make(1)
        return ToolRow(label, lambda: fmt(getter()),
                       [HoloButton(f"set:{key}_dn", "-"), HoloButton(f"set:{key}_up", "+")])

    def _toggle(self, label: str, key: str, getter, setter, post=None) -> ToolRow:
        def fn() -> None:
            setter(not getter())
            if post:
                post()
        self._actions[key] = fn
        return ToolRow(label, lambda: "ON" if getter() else "OFF", [HoloButton(f"set:{key}", "TOGGLE")])

    def _cycle(self, label: str, key: str, options: Sequence, getter, setter, display=None, post=None,
               button: str = "NEXT") -> ToolRow:
        def fn() -> None:
            cur = getter()
            idx = (list(options).index(cur) + 1) % len(options) if cur in options else 0
            setter(options[idx])
            if post:
                post()
        self._actions[key] = fn
        return ToolRow(label, lambda: str(display(getter()) if display else getter()), [HoloButton(f"set:{key}", button)])

    def _action(self, label: str, key: str, button: str, fn, value_fn=None) -> ToolRow:
        self._actions[key] = fn
        return ToolRow(label, value_fn, [HoloButton(f"set:{key}", button)])

    def _set_theme(self, name: str) -> None:
        settings.ui.theme = apply_theme(name)

    def _build_panels(self) -> None:
        d, o = self.drawing, self.objects
        g, t, u, f, cam = settings.gesture, settings.tracker, settings.ui, settings.features, settings.camera
        reconf = lambda: self._request("reconfigure_tracker")  # noqa: E731

        # --- drawing ---------------------------------------------------------
        draw_btns = [HoloButton(f"draw:color{i}", "", swatch=c) for i, (_n, c) in enumerate(DRAW_COLORS)]
        draw_btns += [HoloButton("draw:pen", "PEN"), HoloButton("draw:eraser", "ERASER"),
                      HoloButton("draw:clear", "CLEAR"), HoloButton("draw:save", "SAVE PNG"),
                      HoloButton("draw:size_dn", "SIZE -"), HoloButton("draw:size_up", "SIZE +"),
                      HoloButton("draw:undo", "UNDO"), HoloButton("draw:redo", "REDO")]
        self.panels[AppMode.DRAWING] = ToolPanel(
            "drawing", "DRAWING TOOLS", draw_btns, 3, 34,
            [ToolRow("BRUSH", lambda: f"{d.brush} PX"), ToolRow("STYLE", lambda: d.mode_text)],
        )

        # --- objects ---------------------------------------------------------
        obj_btns = [HoloButton("obj:add_cube", "ADD CUBE"), HoloButton("obj:add_pyramid", "ADD PYRAMID"),
                    HoloButton("obj:add_octahedron", "ADD OCTA"), HoloButton("obj:add_prism", "ADD PRISM"),
                    HoloButton("obj:delete", "DELETE"), HoloButton("obj:reset", "RESET")]
        self.panels[AppMode.OBJECTS] = ToolPanel(
            "objects", "OBJECT CONTROL", obj_btns, 2, 38,
            [ToolRow("OBJECTS", lambda: f"{len(o.objects)} / {o.MAX_OBJECTS}"),
             ToolRow("SELECTED", lambda: o.selected.name if o.selected else "NONE"),
             ToolRow("SCALE", lambda: f"{o.selected.scale:.2f}X" if o.selected else "-"),
             ToolRow("PHYSICS", lambda: "ON" if f.object_physics else "OFF")],
        )

        # --- advanced settings (tabbed control centre) ---------------------------
        def set_pinch(v: float) -> None:
            g.pinch_enter, g.pinch_exit = v, round(v * 1.5, 3)

        pct = lambda v: f"{int(round(v * 100))}%"  # noqa: E731
        tabs: Dict[str, List[ToolRow]] = {
            "GESTURE": [
                self._stepper("SENSITIVITY", "sens", lambda: g.sensitivity, lambda v: setattr(g, "sensitivity", v),
                              0.5, 1.6, 0.1, lambda v: f"{v:.1f}X"),
                self._stepper("STABILITY", "stable", lambda: g.stable_frames, lambda v: setattr(g, "stable_frames", v),
                              2, 10, 1, lambda v: f"{v} FRAMES"),
                self._stepper("PINCH THRESHOLD", "pinch", lambda: g.pinch_enter, set_pinch, 0.16, 0.40, 0.02,
                              lambda v: f"{v:.2f}"),
                self._stepper("ACTION COOLDOWN", "cool", lambda: g.action_cooldown,
                              lambda v: setattr(g, "action_cooldown", v), 0.2, 1.5, 0.1, lambda v: f"{v:.1f} S"),
                self._stepper("FIST HOLD TIME", "fist", lambda: g.fist_hold, lambda v: setattr(g, "fist_hold", v),
                              0.3, 1.5, 0.1, lambda v: f"{v:.1f} S"),
                self._stepper("SWIPE DISTANCE", "swipe", lambda: g.swipe_distance,
                              lambda v: setattr(g, "swipe_distance", v), 0.10, 0.40, 0.02, pct),
            ],
            "TRACK": [
                self._stepper("SMOOTHING", "smooth", lambda: t.smoothing_level,
                              lambda v: setattr(t, "smoothing_level", v), 1, 10, 1, lambda v: f"{v} / 10"),
                self._cycle("MODEL", "model", [0, 1], lambda: t.model_complexity,
                            lambda v: setattr(t, "model_complexity", v), lambda v: "FULL" if v else "LITE", reconf),
                self._cycle("MAX HANDS", "maxhands", [1, 2], lambda: t.max_hands,
                            lambda v: setattr(t, "max_hands", v), None, reconf),
                self._stepper("DETECTION CONF", "detconf", lambda: t.detection_confidence,
                              lambda v: setattr(t, "detection_confidence", v), 0.3, 0.95, 0.05, pct, reconf),
                self._stepper("TRACKING CONF", "trkconf", lambda: t.tracking_confidence,
                              lambda v: setattr(t, "tracking_confidence", v), 0.3, 0.95, 0.05, pct, reconf),
                self._cycle("INFERENCE WIDTH", "infw", [320, 480, 640, 960], lambda: t.process_width,
                            lambda v: setattr(t, "process_width", v), lambda v: f"{v} PX"),
                self._stepper("HAND LOSS GRACE", "grace", lambda: t.lost_grace_frames,
                              lambda v: setattr(t, "lost_grace_frames", v), 2, 20, 2, lambda v: f"{v} FRAMES"),
                self._toggle("MIRROR CAMERA", "mirror", lambda: cam.mirror, lambda v: setattr(cam, "mirror", v)),
                self._action("RESOLUTION", "res", "NEXT", lambda: self._request("resolution_next"),
                             lambda: self.camera_resolution),
            ],
            "DISPLAY": [
                self._cycle("THEME", "theme", THEME_NAMES, lambda: u.theme, self._set_theme),
                self._cycle("LANDMARKS", "landmarks", ["FULL", "LINES", "DOTS", "OFF"], lambda: u.landmark_style,
                            lambda v: setattr(u, "landmark_style", v)),
                self._stepper("PANEL OPACITY", "opacity", lambda: u.panel_opacity,
                              lambda v: setattr(u, "panel_opacity", v), 0.2, 0.95, 0.05, pct),
                self._stepper("FEED BRIGHTNESS", "bright", lambda: u.feed_dim, lambda v: setattr(u, "feed_dim", v),
                              0.4, 1.0, 0.05, pct),
                self._toggle("SCANLINES", "scanlines", lambda: u.scanlines, lambda v: setattr(u, "scanlines", v)),
                self._toggle("VIGNETTE", "vignette", lambda: u.vignette, lambda v: setattr(u, "vignette", v)),
                self._toggle("GLASS BLUR", "blur", lambda: u.glass_blur, lambda v: setattr(u, "glass_blur", v)),
                self._action("FULLSCREEN", "fullscreen", "TOGGLE", lambda: self._request("fullscreen"),
                             lambda: "ON" if u.fullscreen else "OFF"),
            ],
            "FX": [
                self._toggle("SCAN EFFECTS", "scan", lambda: u.scan_effects, lambda v: setattr(u, "scan_effects", v)),
                self._toggle("GLOW", "glow", lambda: u.glow, lambda v: setattr(u, "glow", v)),
                self._stepper("GLOW INTENSITY", "gain", lambda: u.glow_gain, lambda v: setattr(u, "glow_gain", v),
                              0.6, 3.0, 0.2, lambda v: f"{v:.1f}X"),
                self._toggle("FINGER TRAILS", "trails", lambda: f.trails, lambda v: setattr(f, "trails", v)),
                self._stepper("TRAIL LENGTH", "traillen", lambda: f.trail_length,
                              lambda v: setattr(f, "trail_length", v), 5, 40, 5, lambda v: f"{v} PTS"),
                self._toggle("PARTICLES", "particles", lambda: f.particles, lambda v: setattr(f, "particles", v)),
                self._stepper("CURSOR SIZE", "cursize", lambda: f.cursor_size, lambda v: setattr(f, "cursor_size", v),
                              0.6, 2.0, 0.2, lambda v: f"{v:.1f}X"),
                self._stepper("CURSOR SMOOTHING", "cursmooth", lambda: f.cursor_smooth,
                              lambda v: setattr(f, "cursor_smooth", v), 1, 10, 1, lambda v: f"{v} / 10"),
            ],
            "EXTRAS": [
                self._toggle("SOUND EFFECTS", "sound", lambda: f.sound, lambda v: setattr(f, "sound", v)),
                self._toggle("OBJECT PHYSICS", "physics", lambda: f.object_physics,
                             lambda v: setattr(f, "object_physics", v)),
                self._cycle("DRAW SYMMETRY", "symmetry", SYMMETRY_MODES, lambda: f.symmetry,
                            lambda v: setattr(f, "symmetry", v)),
                self._toggle("RAINBOW BRUSH", "rainbow", lambda: f.rainbow, lambda v: setattr(f, "rainbow", v)),
                self._stepper("STROKE SMOOTHING", "strokes", lambda: f.stroke_smoothing,
                              lambda v: setattr(f, "stroke_smoothing", v), 1, 10, 1, lambda v: f"{v} / 10"),
                self._cycle("AUTO STANDBY", "standby", [0, 10, 30, 60], lambda: f.auto_standby,
                            lambda v: setattr(f, "auto_standby", v), lambda v: "OFF" if v == 0 else f"{v} S"),
                self._toggle("FPS GRAPH", "fpsgraph", lambda: f.fps_graph, lambda v: setattr(f, "fps_graph", v)),
                self._toggle("SHOW HINTS", "hints", lambda: f.show_hints, lambda v: setattr(f, "show_hints", v)),
            ],
            "AI": [
                self._toggle("VOICE ASSISTANT", "voice", lambda: settings.ai.voice_enabled,
                             lambda v: setattr(settings.ai, "voice_enabled", v)),
                self._action("VOICE STATUS", "voicestat", "TYPE", self.assistant.begin_typing,
                             lambda: self.assistant.voice_status),
                self._cycle("WAKE WORD", "wake", WAKE_WORDS, lambda: settings.ai.wake_word,
                            lambda v: setattr(settings.ai, "wake_word", v)),
                self._toggle("VOICE REPLIES", "replies", lambda: settings.ai.voice_replies,
                             lambda v: setattr(settings.ai, "voice_replies", v)),
                self._toggle("LOCAL LLM (OLLAMA)", "llm", lambda: settings.ai.llm_enabled,
                             lambda v: setattr(settings.ai, "llm_enabled", v)),
                self._cycle("LLM MODEL", "llmmodel", LLM_MODELS, lambda: settings.ai.llm_model,
                            lambda v: setattr(settings.ai, "llm_model", v)),
                self._action("LLM STATUS", "llmstat", "STATUS", self._noop, lambda: self.assistant.llm_status),
                self._toggle("SHAPE SNAP", "shapesnap", lambda: settings.ai.shape_snap,
                             lambda v: setattr(settings.ai, "shape_snap", v)),
            ],
            "CUSTOM": self._custom_rows(),
            "SYSTEM": [
                self._action("SAVE PROFILE", "profsave", "SAVE", self._profile_save,
                             lambda: "FOUND" if os.path.isfile(PROFILE_PATH) else "NONE"),
                self._action("LOAD PROFILE", "profload", "LOAD", self._profile_load),
                self._action("RESET DEFAULTS", "profreset", "RESET", self._profile_reset),
                self._toggle("AUTOSAVE ON EXIT", "autosave", lambda: f.autosave_profile,
                             lambda v: setattr(f, "autosave_profile", v)),
                self._action("SNAPSHOT", "snap", "CAPTURE", lambda: self._request("snapshot")),
                self._action("RECORD VIDEO", "record", "TOGGLE", lambda: self._request("record_toggle"),
                             lambda: "RECORDING" if self.recording else "IDLE"),
                self._action("CAMERA", "camnext", "NEXT", lambda: self._request("camera_next"),
                             lambda: f"#{self.camera_index}"),
                self._action("OUTPUT FOLDER", "openout", "OPEN", self._open_output),
            ],
        }
        self.panels[AppMode.SETTINGS] = TabbedPanel("settings", "ADVANCED SETTINGS", tabs, 36, 400)

        # --- home: gesture guide -----------------------------------------------
        guide = [("OPEN PALM", "SHOW UI"), ("CLOSED FIST", "HIDE UI (HOLD)"), ("INDEX UP", "POINTER"),
                 ("TWO FINGERS", "NAVIGATE / SWIPE"), ("PINCH", "SELECT / GRAB"), ("THUMB UP", "CONFIRM"),
                 ("THREE FINGERS", "DRAW MODE"), ("HAND MOVE", "MOVE / THROW OBJECT"), ("FINGER MOVE", "AR CURSOR")]
        self.panels[AppMode.HOME] = ToolPanel(
            "home", "GESTURE GUIDE", None, 3, 38,
            [ToolRow(a, (lambda b=b: b), None, Palette.TEXT) for a, b in guide], 22)

        # --- system information -------------------------------------------------
        v = self._versions
        info = [
            ("APPLICATION", lambda: f"HAND-AR {APP_VERSION}"),
            ("PYTHON", lambda: v["python"]), ("OPENCV", lambda: v["opencv"]),
            ("MEDIAPIPE", lambda: v["mediapipe"]), ("NUMPY", lambda: v["numpy"]),
            ("CAMERA", lambda: f"#{self.camera_index}  {self.camera_resolution}"),
            ("FPS", lambda: f"{self.fps:.1f}"), ("INFERENCE", lambda: f"{self.inference_ms:.1f} MS"),
            ("HANDS TRACKED", lambda: str(sum(1 for h in self.hands if not h.stale))),
            ("THEME", lambda: settings.ui.theme), ("RENDERER", lambda: "CPU / OPENCV"),
        ]
        self.panels[AppMode.SYSINFO] = ToolPanel(
            "sysinfo", "SYSTEM INFORMATION", None, 3, 38, [ToolRow(a, fn) for a, fn in info], 24)

    def _noop(self) -> None:
        return None

    def _custom_rows(self) -> List[ToolRow]:
        rows: List[ToolRow] = [self._toggle("CUSTOM GESTURES", "customon", lambda: settings.ai.custom_enabled,
                                            lambda v: setattr(settings.ai, "custom_enabled", v))]
        for n in (1, 2, 3):
            rows.append(self._action(f"TRAIN GESTURE {n}", f"train{n}", "TRAIN",
                                     lambda n=n: self.start_custom_training(n),
                                     lambda n=n: "TRAINED" if self.custom.trained(n) else "EMPTY"))
            rows.append(self._cycle(f"GESTURE {n} ACTION", f"cact{n}", CUSTOM_ACTIONS,
                                    lambda n=n: getattr(settings.ai, f"custom_action_{n}"),
                                    lambda v, n=n: setattr(settings.ai, f"custom_action_{n}", v)))
        rows.append(self._action("FORGET ALL", "customclear", "CLEAR", self.custom.clear))
        return rows

    # --- assistant / automation API -----------------------------------------------
    def perform(self, bid: str) -> None:
        """Activate any button by id ("draw:undo", "set:theme", "menu:drawing" ...) - used by the assistant."""
        import time
        self._activate(bid, time.perf_counter())

    def request(self, name: str) -> None:
        self._request(name)

    def request_exit(self) -> None:
        self.requests.append(("exit",))
        self.running = False

    def mode_for(self, key: str) -> "AppMode":
        return MODE_BY_KEY[key]

    def select_settings_tab(self, index: int) -> None:
        panel = self.panels[AppMode.SETTINGS]
        if isinstance(panel, TabbedPanel):
            panel.select_tab(index)

    def start_custom_training(self, slot: int, now: Optional[float] = None) -> None:
        import time
        self.custom.start_training(slot, time.perf_counter() if now is None else now)
        self.toast(f"TRAINING GESTURE {slot} - GET READY", "info")

    def _run_custom_action(self, slot: int, now: float) -> None:
        name = getattr(settings.ai, f"custom_action_{slot}")
        if name == "NONE":
            return
        self.toast(f"CUSTOM GESTURE {slot}: {name}", "ok", now)
        self.sound.play("confirm")
        self._custom_flash = now
        if name == "SNAPSHOT":
            self._request("snapshot")
        elif name == "RECORD":
            self._request("record_toggle")
        elif name == "UNDO":
            self._draw_action("undo", now)
        elif name == "REDO":
            self._draw_action("redo", now)
        elif name == "CLEAR CANVAS":
            self._draw_action("clear", now)
        elif name == "NEXT THEME":
            self._set_action("theme", now)
        elif name == "ADD OBJECT":
            self.objects.add()
        elif name == "TOGGLE UI":
            self.set_interface(not self.interface_active, now)
        elif name == "SPEAK STATUS":
            self.assistant.speak_status(now)

    # --- profile / misc actions ------------------------------------------------
    def _profile_save(self) -> None:
        try:
            save_profile()
            self.toast("PROFILE SAVED", "ok")
            self.sound.play("save")
        except OSError as exc:
            self.toast(f"SAVE FAILED: {exc}"[:42], "error")

    def _profile_load(self) -> None:
        try:
            if load_profile():
                self._request("reconfigure_tracker")
                self.toast("PROFILE LOADED", "ok")
            else:
                self.toast("NO SAVED PROFILE FOUND", "warn")
        except ValueError as exc:
            self.toast(str(exc)[:42], "error")

    def _profile_reset(self) -> None:
        reset_defaults()
        self._request("reconfigure_tracker")
        self.toast("DEFAULTS RESTORED", "info")

    def _open_output(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            os.startfile(self.output_dir)  # type: ignore[attr-defined]  # Windows only
        except (AttributeError, OSError):
            self.toast("OUTPUT: " + os.path.basename(self.output_dir), "info")

    def _relayout(self, w: int, h: int) -> None:
        s = ui_scale(h)
        self.s, self.frame_size = s, (w, h)
        self.menu.layout(w, h, s)
        x, pw = 20 * s, 310 * s
        self.status_rect = (x, 20 * s, pw, 190 * s)
        self.gesture_rect = (x, h - 20 * s - 140 * s, pw, 140 * s)
        left, right = 350 * s, w - 310 * s
        for mode, panel in self.panels.items():
            if mode is AppMode.SETTINGS:      # wide control centre in the middle of the screen
                panel.layout(left, 100 * s, max(right - left, 440 * s), s)
            else:
                panel.layout(x, 20 * s + 190 * s + 12 * s, pw, s)
        self.drawing.ensure_size(w, h)
        self.objects.size = (w, h)

    # =====================================================================
    # public helpers
    # =====================================================================
    def toast(self, text: str, kind: str = "info", now: Optional[float] = None) -> None:
        import time
        t = time.perf_counter() if now is None else now
        self.toasts.append(Toast(text.upper(), kind, t, settings.ui.toast_seconds))
        self.toasts = self.toasts[-4:]

    def take_requests(self) -> List[tuple]:
        out, self.requests = self.requests, []
        return out

    def active_panel(self) -> ToolPanel:
        return self.panels[self.mode]

    def active_buttons(self) -> List[HoloButton]:
        return self.menu.buttons + self.active_panel().all_buttons()

    def over_ui(self, pt) -> bool:
        if self.ui_alpha < 0.5:
            return False
        return self.menu.contains(pt) or self.active_panel().contains(pt)

    def nav_focus(self, now: float) -> Optional[int]:
        return self.navigator.active_focus(now)

    # =====================================================================
    # state changes
    # =====================================================================
    def set_interface(self, on: bool, now: Optional[float] = None) -> None:
        if on == self.interface_active:
            return
        self.interface_active = on
        self.sound.play("on" if on else "off")
        if on:
            self.toast("INTERFACE ONLINE", "ok", now)
        else:
            self.toast("INTERFACE STANDBY", "info", now)
            self.drawing.pen_up()
            self.objects.release_all()
            self.navigator.clear()
            self.cursor.visible = False
            self.hover_id = None

    def set_mode(self, mode: AppMode, now: Optional[float] = None) -> None:
        if mode is self.mode:
            return
        self.mode = mode
        self.drawing.pen_up()
        self.objects.release_all()
        self.sound.play("mode")
        self.toast(f"MODE: {mode.value}", "info", now)

    # =====================================================================
    # per-frame update
    # =====================================================================
    def update(self, size: Tuple[int, int], hands: List[HandData], results: Dict[str, GestureResult],
               now: float, dt: float) -> None:
        if size != self.frame_size:
            self._relayout(*size)
        self.hands, self.results = hands, results
        live = [h for h in hands if not h.stale and h.label in results]
        if live or self.last_hand_time == 0.0:
            self.last_hand_time = now
        self.fps_hist.append(self.fps)
        self.inf_hist.append(self.inference_ms)
        self.trails.update(hands, now)
        self.particles.update(dt)
        limit = settings.features.auto_standby
        if limit and self.interface_active and now - self.last_hand_time > limit:
            self.set_interface(False, now)
            self.toast("AUTO STANDBY - NO HAND", "info", now)
        self.primary = live[0] if live else None
        self.primary_result = results.get(self.primary.label) if self.primary else None
        self.hold = None

        self.assistant.update(now)
        self._update_custom_gestures(live, now)
        shape = self.drawing.pop_shape_event()
        if shape:
            self.toast(f"SHAPE RECOGNISED: {shape}", "ok", now)
            self.sound.play("save")
            self.particles.emit(self.cursor.pos, 30, Palette.SUCCESS)

        self._update_interface_toggle(live, now)
        target = 1.0 if self.interface_active else 0.0
        self.ui_alpha += (target - self.ui_alpha) * min(1.0, dt * 7.0)
        if abs(target - self.ui_alpha) < 0.01:
            self.ui_alpha = target

        if self.interface_active:
            self._update_active(live, now, dt)
        else:
            self.nav_active = False
            self.hover_id = None
            self.cursor.visible = False
        self.cursor.update(dt)
        self.toasts = [t for t in self.toasts if now - t.t0 < t.duration]

    def _update_custom_gestures(self, live: List[HandData], now: float) -> None:
        """Training of / detection of user-defined gestures (primary hand only)."""
        hand = self.primary
        if self.custom.training:
            msg = self.custom.feed(hand, now)
            status = self.custom.training_status(now)
            if status is not None and hand is not None:
                self.hold = ("TRAIN:" + status[0], status[1], hand.palm_center)
            if msg:
                self.toast(msg, "ok" if "TRAINED" in msg else "warn", now)
                self.sound.play("save" if "TRAINED" in msg else "error")
            self.custom_label = ""
            return
        if not settings.ai.custom_enabled:
            self.custom_label = ""
            return
        slot = self.custom.detect(hand, now)
        self.custom_label = f"CUSTOM {self.custom.current}" if self.custom.current else ""
        if slot is not None:
            self._run_custom_action(slot, now)

    # ---------------------------------------------------------------- show / hide
    def _update_interface_toggle(self, live: List[HandData], now: float) -> None:
        cfg = settings.gesture
        if not self.interface_active:
            for hd in live:
                res = self.results[hd.label]
                if res.gesture is Gesture.OPEN_PALM:
                    if res.held >= cfg.palm_hold:
                        if self.gate.ready("show", now, cfg.action_cooldown):
                            self.set_interface(True, now)
                    else:
                        self.hold = ("ACTIVATING", res.held / cfg.palm_hold, hd.palm_center)
                    break
        else:
            for hd in live:
                res = self.results[hd.label]
                if res.gesture is Gesture.FIST:
                    if res.held >= cfg.fist_hold:
                        if self.gate.ready("hide", now, cfg.action_cooldown):
                            self.set_interface(False, now)
                    elif res.held > 0.12:
                        self.hold = ("HIDING", res.held / cfg.fist_hold, hd.palm_center)
                    break

    # ------------------------------------------------------------ active interface
    def _update_active(self, live: List[HandData], now: float, dt: float) -> None:
        cfg = settings.gesture
        w, h = self.frame_size
        prim, pres = self.primary, self.primary_result

        self._update_cursor(now)
        hover = None
        if self.cursor.visible and self.ui_alpha > 0.6:
            for b in self.active_buttons():
                if b.enabled and b.contains(self.cursor.pos):
                    hover = b.bid
                    break
        self.hover_id = hover
        self.cursor.hovering = hover is not None
        for b in self.active_buttons():
            b.step(dt, b.bid == hover)
        self._sync_button_states(now)

        # ---- discrete events from the primary hand ------------------------------
        consumed = set()
        if prim is not None and pres is not None and pres.changed:
            if pres.gesture is Gesture.PINCH:
                if self._click_at(self.cursor.pos, now):
                    consumed.add(prim.label)
            elif pres.gesture is Gesture.THUMB_UP:
                self._confirm(now)
            elif pres.gesture is Gesture.TWO_FINGERS:
                self.navigator.begin(self.menu.index_of(KEY_BY_MODE[self.mode]), now)

        # ---- navigation (two fingers + swipe) -------------------------------------
        nav = None
        for hd in live:
            r = self.results[hd.label]
            if r.gesture is Gesture.TWO_FINGERS and r.held > 0.15:
                nav = hd
                break
        self.nav_active = nav is not None
        if nav is not None:
            ev = self.navigator.swipe.update(nav.palm_center, now, w, h)
            if ev:
                self._on_swipe(ev, now)
        else:
            self.navigator.swipe.reset()

        # ---- drawing mode switch / pen --------------------------------------------
        pen_hand: Optional[HandData] = None
        for hd in live:
            r = self.results[hd.label]
            if r.gesture is Gesture.THREE_FINGERS:
                if self.mode is AppMode.DRAWING:
                    pen_hand = hd
                elif r.held >= cfg.mode_switch_hold:
                    self.set_mode(AppMode.DRAWING, now)
                    pen_hand = hd
                else:
                    self.hold = ("ENTERING DRAW MODE", r.held / cfg.mode_switch_hold, hd.palm_center)
                break

        if self.mode is AppMode.DRAWING:
            point, down = None, False
            if pen_hand is not None:
                point = pen_hand.landmarks[8]
                down = not self.over_ui(point)
            self.drawing.update(point, down, now)
            if down and point is not None:
                self.particles.emit(point, 1, self.drawing.color, 80.0, 0.5)
        elif self.drawing.pen_down:
            self.drawing.pen_up()

        # ---- object manipulation -----------------------------------------------------
        if self.mode is AppMode.OBJECTS:
            holders = {hd.label: (hd, self.results[hd.label]) for hd in self.hands if hd.label in self.results}
            events = {
                label for label, (hd, res) in holders.items()
                if not hd.stale and res.gesture is Gesture.PINCH and res.changed
                and label not in consumed and not self.over_ui(res.pinch_point)
            }
            hover_pos = None
            if self.cursor.visible and pres is not None and pres.gesture in POINTER_GESTURES:
                hover_pos = self.cursor.pos
            self.objects.update(self.frame_size, now, dt, holders, events, hover_pos)
            for kind, pos in self.objects.events:
                if kind == "grab":
                    self.particles.emit(pos, 26, Palette.TEXT)
                    self.sound.play("grab")
                elif kind == "bounce":
                    self.particles.emit(pos, 8, Palette.PRIMARY, 110.0, 0.4)
            self.objects.events.clear()

    def _update_cursor(self, now: float) -> None:
        cur, prim, pres = self.cursor, self.primary, self.primary_result
        cur.pressed = False
        if prim is not None and pres is not None:
            cur.visible = True
            if pres.pinch_active:
                cur.pressed = True
                cur.set_target(pres.pinch_point)
            elif pres.gesture in POINTER_GESTURES:
                cur.set_target(prim.landmarks[8])
            else:
                cur.locked = True
        elif self.mouse_pos is not None and now - self.mouse_time < 3.0:
            cur.visible = True
            cur.set_target(self.mouse_pos)
        else:
            cur.visible = False

    def _sync_button_states(self, now: float) -> None:
        d = self.drawing
        panel = self.active_panel()
        if self.mode is AppMode.DRAWING:
            for b in panel.buttons:
                if b.bid.startswith("draw:color"):
                    b.active = d.tool == "pen" and int(b.bid[10:]) == d.color_index
                elif b.bid == "draw:pen":
                    b.active = d.tool == "pen"
                elif b.bid == "draw:eraser":
                    b.active = d.tool == "eraser"

    # ---------------------------------------------------------------- UI actions
    def _click_at(self, pos, now: float) -> bool:
        """Click whatever button lies under ``pos``. Returns True if a button consumed the click."""
        if self.ui_alpha < 0.6:
            return False
        for b in self.active_buttons():
            if b.enabled and b.contains(pos):
                if self.gate.ready("click:" + b.bid, now, settings.gesture.action_cooldown * 0.5):
                    b.trigger()
                    self.cursor.click()
                    self.particles.emit(pos, 22, Palette.PRIMARY)
                    self.sound.play("click")
                    self._activate(b.bid, now)
                return True
        return False

    def _confirm(self, now: float) -> None:
        """THUMB UP: confirm the navigation focus or the button under the (frozen) cursor."""
        if not self.gate.ready("confirm", now, settings.gesture.action_cooldown):
            return
        focus = self.navigator.active_focus(now)
        if focus is not None:
            key = MENU_ITEMS[focus][0]
            self.navigator.clear()
            self.cursor.click()
            self.sound.play("confirm")
            self._activate(f"menu:{key}", now)
            return
        if self.hover_id is not None:
            for b in self.active_buttons():
                if b.bid == self.hover_id:
                    b.trigger()
            self.cursor.click()
            self.particles.emit(self.cursor.pos, 22, Palette.SUCCESS)
            self.sound.play("confirm")
            self._activate(self.hover_id, now)
            return
        self.sound.play("confirm")
        self.toast("CONFIRMED", "ok", now)

    def _on_swipe(self, direction: str, now: float) -> None:
        if self.navigator.active_focus(now) is None:
            self.navigator.begin(self.menu.index_of(KEY_BY_MODE[self.mode]), now)
        if direction in ("LEFT", "RIGHT"):
            self.navigator.move_focus(1 if direction == "RIGHT" else -1, now)
        elif self.mode is AppMode.DRAWING:
            self.drawing.change_brush(1 if direction == "UP" else -1)
            self.toast(f"BRUSH {self.drawing.brush} PX", "info", now)
        elif self.mode is AppMode.OBJECTS:
            self.objects.scale_selected(1.15 if direction == "UP" else 1 / 1.15)
        else:
            self.navigator.move_focus(1 if direction == "DOWN" else -1, now)

    def _activate(self, bid: str, now: float) -> None:
        kind, _, name = bid.partition(":")
        if kind == "menu":
            self._menu_action(name, now)
        elif kind == "draw":
            self._draw_action(name, now)
        elif kind == "obj":
            self._object_action(name, now)
        elif kind == "set":
            self._set_action(name, now)

    def _menu_action(self, name: str, now: float) -> None:
        if name == "exit":
            if now < self.exit_armed_until:
                self.requests.append(("exit",))
                self.running = False
            else:
                self.exit_armed_until = now + 3.0
                self.toast("SELECT EXIT AGAIN TO CONFIRM", "warn", now)
            return
        self.exit_armed_until = 0.0
        self.set_mode(MODE_BY_KEY[name], now)

    def _draw_action(self, name: str, now: float) -> None:
        d = self.drawing
        if name.startswith("color"):
            d.set_color(int(name[5:]))
        elif name == "pen":
            d.set_tool("pen")
        elif name == "eraser":
            d.set_tool("eraser")
        elif name == "clear":
            d.clear()
            self.toast("CANVAS CLEARED", "info", now)
        elif name == "save":
            self._save_drawing(now)
        elif name == "size_dn":
            d.change_brush(-1)
        elif name == "size_up":
            d.change_brush(1)
        elif name == "undo":
            if not d.undo():
                self.toast("NOTHING TO UNDO", "warn", now)
        elif name == "redo":
            if not d.redo():
                self.toast("NOTHING TO REDO", "warn", now)

    def _save_drawing(self, now: Optional[float] = None) -> None:
        try:
            path = self.drawing.save()
            self.sound.play("save")
            self.toast(f"SAVED {os.path.basename(path)}", "ok", now)
        except ValueError as exc:
            self.toast(str(exc)[:42], "warn", now)
        except OSError as exc:
            self.toast(f"SAVE FAILED: {exc}"[:42], "error", now)

    def _object_action(self, name: str, now: float) -> None:
        o = self.objects
        if name.startswith("add_"):
            if o.add(name[4:]) is None:
                self.toast("OBJECT LIMIT REACHED", "warn", now)
        elif name == "delete":
            if not o.delete_selected():
                self.toast("NO OBJECT SELECTED", "warn", now)
        elif name == "reset":
            o.reset()

    def _set_action(self, name: str, now: Optional[float]) -> None:
        if name.startswith("tab"):
            panel = self.panels[AppMode.SETTINGS]
            if isinstance(panel, TabbedPanel):
                panel.select_tab(int(name[3:]))
            return
        fn = self._actions.get(name)
        if fn is not None:
            fn()

    # ---------------------------------------------------------------- mouse input
    def mouse_move(self, x: int, y: int, now: float) -> None:
        self.mouse_pos = np.array([x, y], dtype=np.float32)
        self.mouse_time = now

    def mouse_click(self, x: int, y: int, now: float) -> None:
        self.mouse_move(x, y, now)
        if self.interface_active:
            self._click_at(self.mouse_pos, now)

    # ---------------------------------------------------------------- keyboard
    def handle_key(self, key: str, now: float) -> None:
        """Keyboard fallback for everything that can be done with gestures."""
        if self.assistant.typing and self.assistant.handle_typing_key(key, now):
            return
        if key == "i":
            self.assistant.begin_typing()
        elif key in ("q", "esc"):
            self.requests.append(("exit",))
            self.running = False
        elif key == "f":
            self.requests.append(("fullscreen",))
        elif key == "r":
            self.requests.append(("resolution_next",))
        elif key == "h":
            self.set_interface(not self.interface_active, now)
        elif key == "l":
            self._set_action("landmarks", now)
        elif key == "g":
            on = not (settings.ui.scan_effects and settings.ui.glow)
            settings.ui.scan_effects = settings.ui.glow = on
        elif key == "t":
            self._set_action("theme", now)
            self.toast(f"THEME {settings.ui.theme}", "info", now)
        elif key == "k":
            self._request("snapshot")
        elif key == "v":
            self._request("record_toggle")
        elif key == "m":
            settings.features.sound = not settings.features.sound
            self.toast("SOUND " + ("ON" if settings.features.sound else "OFF"), "info", now)
        elif key == "d":
            settings.features.fps_graph = not settings.features.fps_graph
        elif key in ("1", "2", "3", "4", "5"):
            self.set_mode(list(AppMode)[int(key) - 1], now)
        elif self.mode is AppMode.DRAWING:
            d = self.drawing
            if key == "c":
                self._draw_action("clear", now)
            elif key == "s":
                self._save_drawing(now)
            elif key == "e":
                d.toggle_eraser()
            elif key == "p":
                d.set_tool("pen")
            elif key == "[":
                d.change_brush(-1)
            elif key == "]":
                d.change_brush(1)
            elif key == "n":
                d.next_color()
            elif key == "u":
                self._draw_action("undo", now)
            elif key == "y":
                self._draw_action("redo", now)
            elif key == "z":
                self._set_action("symmetry", now)
                self.toast(f"SYMMETRY {settings.features.symmetry}", "info", now)
            elif key == "w":
                self._set_action("rainbow", now)
        elif self.mode is AppMode.OBJECTS:
            o = self.objects
            if key == "a":
                if o.add() is None:
                    self.toast("OBJECT LIMIT REACHED", "warn", now)
            elif key in ("delete", "x"):
                o.delete_selected()
            elif key in ("+", "="):
                o.scale_selected(1.1)
            elif key == "-":
                o.scale_selected(1 / 1.1)
            elif key == "tab":
                o.cycle_selection()
            elif key == ",":
                o.rotate_selected(dz=-0.15)
            elif key == ".":
                o.rotate_selected(dz=0.15)
            elif key in ("left", "right", "up", "down"):
                step = 0.02 * self.frame_size[0]
                o.move_selected({"left": -step, "right": step}.get(key, 0.0), {"up": -step, "down": step}.get(key, 0.0))


# ==========================================================================
# HOLOGRAPHIC UI
# ==========================================================================

def toast_color(kind: str):
    """Toast colours follow the active theme."""
    return {"ok": Palette.SUCCESS, "warn": Palette.WARNING, "error": Palette.DANGER}.get(kind, Palette.PRIMARY)


MODE_HINTS = {
    AppMode.HOME: "INDEX UP: POINT  PINCH: CLICK  THUMB UP: CONFIRM  2 FINGERS+SWIPE: NAV",
    AppMode.DRAWING: "3 FINGERS: DRAW  INDEX UP: POINT  PINCH: CLICK  2 FINGERS+SWIPE UP/DN: BRUSH",
    AppMode.OBJECTS: "PINCH: GRAB  MOVE/ROLL HAND  BOTH HANDS PINCH: SCALE+ROTATE",
    AppMode.SETTINGS: "POINT + PINCH: ADJUST  THUMB UP: CONFIRM  CLICK TABS TO SWITCH PAGES",
    AppMode.SYSINFO: "2 FINGERS+SWIPE: NAVIGATE MENU  THUMB UP: CONFIRM",
}


class HolographicUI:
    def __init__(self, engine: InteractionEngine) -> None:
        self.engine = engine
        self.glow = GlowLayer()
        self.ambient = Ambient()
        self.boot_start = time.perf_counter()

    # ================================================================== frame
    def render(self, frame: np.ndarray, now: float) -> None:
        e = self.engine
        h, w = frame.shape[:2]
        s = ui_scale(h)
        a = e.ui_alpha

        self.glow.begin(w, h)
        self.ambient.apply(frame)
        e.drawing.render(frame)

        if e.mode is AppMode.OBJECTS and a > 0.02:
            e.objects.render(frame, self.glow, now, a, s)

        e.trails.draw(frame, self.glow, s)
        for hand in e.hands:
            draw_hand_overlay(frame, self.glow, hand, e.results.get(hand.label), now, s)
        e.particles.draw(frame, self.glow, s)

        if a > 0.01:
            self._draw_interface(frame, now, a, s)
        self._draw_assistant(frame, now, a, s)
        if a < 0.99:
            self._draw_standby(frame, now, 1.0 - a, s)

        if a > 0.3 and (settings.features.fps_graph or e.mode is AppMode.SYSINFO):
            self._draw_graphs(frame, now, a, s)
        self._draw_hold(frame, now, s)
        if e.recording:
            self._draw_rec(frame, now, s)
        e.cursor.draw_glow(self.glow, now, a, s)
        self.glow.compose(frame)

        if a > 0.3 and e.mode is AppMode.DRAWING and e.cursor.visible and not e.cursor.locked:
            self._draw_brush_preview(frame, s)
        e.cursor.draw(frame, now, a, s)
        self._draw_toasts(frame, now, s)

        boot = now - self.boot_start
        if boot < settings.ui.boot_seconds:
            self._draw_boot(frame, boot, s)

    # ============================================================ interface
    def _draw_interface(self, frame: np.ndarray, now: float, a: float, s: float) -> None:
        e = self.engine
        h, w = frame.shape[:2]
        slide = (1.0 - a) * 60 * s

        self._draw_status_panel(frame, now, a, -slide, s)
        e.active_panel().draw(frame, self.glow, now, a, -slide, s)
        self._draw_gesture_panel(frame, now, a, -slide, s)

        focus = e.nav_focus(now)
        e.menu.draw(frame, self.glow, now, a, slide, s, KEY_BY_MODE[e.mode], focus,
                    now < e.exit_armed_until)

        # title bar and hint line
        title = f"// {e.mode.value}"
        put_text(frame, APP_TITLE, (w / 2, 34 * s), 0.62 * s, fade(Palette.TEXT, a), 1, 4 * s, "center")
        put_text(frame, title, (w / 2, 58 * s), 0.46 * s, fade(Palette.PRIMARY, a), 1, 3 * s, "center")
        cv2.line(frame, (int(w / 2 - 150 * s), int(68 * s)), (int(w / 2 + 150 * s), int(68 * s)),
                 fade(Palette.PRIMARY, 0.5 * a), 1, AA)
        if settings.features.show_hints:
            put_text(frame, MODE_HINTS[e.mode], (w / 2, h - 44 * s), 0.36 * s, fade(Palette.DIM_TEXT, a), 1, 1.0 * s,
                     "center")
            put_text(frame, "KEYS: 1-5 MODE  H UI  T THEME  K SNAPSHOT  V RECORD  F FULL  R RES  Q QUIT",
                     (w / 2, h - 22 * s), 0.34 * s, fade(Palette.DIM_TEXT, 0.7 * a), 1, 1.0 * s, "center")

        if e.nav_active:
            put_text(frame, "NAVIGATION MODE - SWIPE, THEN THUMB UP", (w / 2, 92 * s), 0.44 * s,
                     fade(Palette.WARNING, a), 1, 2 * s, "center")

    def _kv(self, frame: np.ndarray, x: float, y: float, w: float, label: str, value: str, color,
            a: float, s: float) -> None:
        put_text(frame, label, (x, y), 0.40 * s, fade(Palette.DIM_TEXT, a), 1, 1.0 * s)
        put_text(frame, value, (x + w, y), 0.42 * s, fade(color, a), 1, 0.8 * s, "right")

    def _draw_status_panel(self, frame: np.ndarray, now: float, a: float, dx: float, s: float) -> None:
        e = self.engine
        x, y, w, h = e.status_rect
        x += dx
        draw_panel(frame, self.glow, (x, y, w, h), "SYSTEM STATUS", a, s)
        live = [hd for hd in e.hands if not hd.stale]
        gesture = e.primary_result.gesture.value if e.primary_result else "NONE"
        conf = sum(hd.confidence for hd in live) / len(live) if live else 0.0
        fps_col = Palette.SUCCESS if e.fps >= 24 else (Palette.WARNING if e.fps >= 15 else Palette.DANGER)

        pad, row, y0 = 16 * s, 23.5 * s, y + 54 * s
        iw = w - 2 * pad
        rows = [
            ("SYSTEM", "ONLINE" if e.interface_active else "STANDBY", Palette.SUCCESS),
            ("CAMERA", "CONNECTED" if e.camera_connected else "LOST", Palette.SUCCESS if e.camera_connected else Palette.DANGER),
            ("HAND DETECTION", f"DETECTED ({len(live)})" if live else "NO HAND", Palette.SUCCESS if live else Palette.WARNING),
            ("GESTURE", gesture, Palette.PRIMARY),
            ("CONFIDENCE", f"{int(conf * 100)}%", Palette.PRIMARY),
            ("FPS", f"{e.fps:.0f}", fps_col),
        ]
        for i, (label, value, col) in enumerate(rows):
            self._kv(frame, x + pad, y0 + i * row, iw, label, value, col, a, s)
        # confidence bar
        by = int(y0 + 4 * row + 7 * s)
        bx, bw = int(x + pad), int(iw)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + max(2, int(3 * s))), fade(Palette.PRIMARY, 0.25 * a), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * conf), by + max(2, int(3 * s))), fade(Palette.PRIMARY, a), -1)

    def _draw_gesture_panel(self, frame: np.ndarray, now: float, a: float, dx: float, s: float) -> None:
        e = self.engine
        x, y, w, h = e.gesture_rect
        x += dx
        draw_panel(frame, self.glow, (x, y, w, h), "GESTURE TRACKING", a, s)
        pad = 16 * s
        y0 = y + 54 * s
        live = [hd for hd in e.hands if not hd.stale]
        if not live:
            put_text(frame, "NO HAND DETECTED", (x + pad, y0 + 18 * s), 0.46 * s,
                     fade(Palette.WARNING, a * (0.6 + 0.4 * math.sin(now * 4))), 1, 2 * s)
            return
        for i, hd in enumerate(live[:2]):
            res = e.results.get(hd.label)
            yy = y0 + i * 40 * s
            col = Palette.PRIMARY if hd.label == "Right" else Palette.ACCENT
            put_text(frame, f"{hd.label.upper()}  X {int(hd.palm_center[0]):4d}  Y {int(hd.palm_center[1]):4d}",
                     (x + pad, yy), 0.42 * s, fade(col, a), 1, 1.0 * s)
            gname = res.gesture.value if res else "NONE"
            extra = f"  {e.custom_label}" if (e.custom_label and i == 0) else ""
            put_text(frame, f"{gname}  {hd.direction}  {int(hd.confidence * 100)}%{extra}",
                     (x + pad, yy + 18 * s), 0.38 * s, fade(Palette.TEXT if not extra else Palette.SUCCESS, a), 1,
                     1.0 * s)

    # ====================================================== AI assistant
    @staticmethod
    def _fit(text: str, scale: float, max_w: float) -> str:
        while text and text_width(text, scale, 1, 0.8) > max_w:
            text = text[:-2]
            if text_width(text + "...", scale, 1, 0.8) <= max_w:
                return text + "..."
        return text

    def _orb(self, frame: np.ndarray, center, r: float, now: float, a: float, s: float) -> None:
        asst = self.engine.assistant
        state = asst.state
        col = {"IDLE": Palette.DIM_TEXT, "LISTENING": Palette.PRIMARY, "THINKING": Palette.ACCENT,
               "SPEAKING": Palette.SUCCESS}[state]
        c = ip(center)
        cv2.circle(frame, c, int(r * 0.45), fade(col, a), -1, AA)
        self.glow.circle(center, r * 0.9, fade(col, 0.6 * a))
        if state == "LISTENING":
            level = asst.listener.level
            for k in range(3):
                phase = (now * 0.9 + k / 3.0) % 1.0
                cv2.circle(frame, c, int(r * (0.6 + 0.7 * phase + 0.5 * level)), fade(col, a * (1 - phase) * 0.8), 1, AA)
        elif state == "THINKING":
            for k in range(3):
                ang = now * 220 + k * 120
                cv2.ellipse(frame, c, (int(r), int(r)), 0, ang, ang + 60, fade(col, a), 2, AA)
        elif state == "SPEAKING":
            for k in range(5):
                hgt = (0.25 + 0.75 * abs(math.sin(now * 9 + k * 1.3))) * r * 0.9
                x = c[0] + int((k - 2) * r * 0.32)
                cv2.line(frame, (x, int(c[1] - hgt)), (x, int(c[1] + hgt)), fade(Palette.TEXT, a), max(1, int(2 * s)), AA)
        else:
            cv2.circle(frame, c, int(r), fade(col, 0.5 * a), 1, AA)

    def _draw_assistant(self, frame: np.ndarray, now: float, a_ui: float, s: float) -> None:
        e = self.engine
        asst = e.assistant
        h, w = frame.shape[:2]
        busy = asst.state != "IDLE" or asst.typing
        if a_ui < 0.5:                                    # standby: just a small indicator
            if busy:
                self._orb(frame, (w - 44 * s, h - 40 * s), 14 * s, now, 1.0, s)
                put_text(frame, asst.state, (w - 70 * s, h - 35 * s), 0.4 * s, Palette.DIM_TEXT, 1, 2 * s, "right")
            return
        a = a_ui
        pw, ph = 460 * s, 92 * s
        x, y = w / 2 - pw / 2, h - 62 * s - ph
        draw_panel(frame, self.glow, (x, y, pw, ph), "", a, s, bg_alpha=0.5)
        self._orb(frame, (x + 46 * s, y + 44 * s), 24 * s, now, a, s)
        tx, maxw = x + 92 * s, pw - 108 * s
        status = asst.voice_status
        title = f"HOLO ASSISTANT  -  {asst.state}"
        put_text(frame, title, (tx, y + 24 * s), 0.44 * s, fade(Palette.PRIMARY, a), 1, 2 * s)
        if asst.typing:
            blink = "_" if int(now * 2) % 2 == 0 else " "
            put_text(frame, self._fit("> " + asst.buffer + blink, 0.46 * s, maxw), (tx, y + 48 * s), 0.46 * s,
                     fade(Palette.WARNING, a), 1, 0.8 * s)
        elif asst.partial and asst.state == "LISTENING":
            put_text(frame, self._fit("> " + asst.partial, 0.44 * s, maxw), (tx, y + 48 * s), 0.44 * s,
                     fade(Palette.TEXT, 0.8 * a), 1, 0.8 * s)
        elif now - asst.heard_time < 7:
            put_text(frame, self._fit("> " + asst.heard, 0.44 * s, maxw), (tx, y + 48 * s), 0.44 * s,
                     fade(Palette.TEXT, a), 1, 0.8 * s)
        if now - asst.reply_time < 9 and not asst.typing:
            put_text(frame, self._fit(asst.reply, 0.42 * s, maxw), (tx, y + 68 * s), 0.42 * s,
                     fade(Palette.SUCCESS, a), 1, 0.6 * s)
        elif not asst.typing and not (now - asst.heard_time < 7):
            wake = settings.ai.wake_word
            if status in ("LISTENING",):
                hint = f"SAY '{wake}' + COMMAND   OR PRESS I TO TYPE" if wake != "OFF" else "LISTENING - SAY A COMMAND"
            elif status == "NOT INSTALLED":
                hint = "VOICE OFF: pip install vosk sounddevice  (PRESS I TO TYPE)"
            else:
                hint = f"VOICE: {status}   PRESS I TO TYPE"
            put_text(frame, self._fit(hint, 0.38 * s, maxw), (tx, y + 68 * s), 0.38 * s, fade(Palette.DIM_TEXT, a), 1, 0.8 * s)
        lvl = clamp(asst.listener.level, 0.0, 1.0)
        bx, by, bw = int(tx), int(y + ph - 12 * s), int(maxw)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + max(2, int(3 * s))), fade(Palette.PRIMARY, 0.2 * a), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * lvl), by + max(2, int(3 * s))), fade(Palette.PRIMARY, a), -1)

    # ======================================================== diagnostics
    def _draw_graphs(self, frame: np.ndarray, now: float, a: float, s: float) -> None:
        e = self.engine
        h, w = frame.shape[:2]
        pw, ph = 270 * s, 150 * s
        x, y = w - 20 * s - pw, h - 70 * s - ph
        draw_panel(frame, self.glow, (x, y, pw, ph), "DIAGNOSTICS", a, s)
        pad = 14 * s
        gx, gw = x + pad, pw - 2 * pad
        fps = list(e.fps_hist)
        inf = list(e.inf_hist)
        put_text(frame, f"FPS {e.fps:.0f}", (gx, y + 50 * s), 0.38 * s, fade(Palette.SUCCESS, a), 1, 1 * s)
        draw_sparkline(frame, (gx, y + 56 * s, gw, 26 * s), fps, 0, 60, Palette.SUCCESS, a)
        put_text(frame, f"INFERENCE {e.inference_ms:.1f} MS", (gx, y + 100 * s), 0.38 * s,
                 fade(Palette.WARNING, a), 1, 1 * s)
        draw_sparkline(frame, (gx, y + 106 * s, gw, 26 * s), inf, 0, 60, Palette.WARNING, a)

    def _draw_rec(self, frame: np.ndarray, now: float, s: float) -> None:
        h, w = frame.shape[:2]
        if int(now * 2) % 2 == 0:
            cv2.circle(frame, ip((w / 2 - 300 * s, 28 * s)), int(7 * s), Palette.DANGER, -1, AA)
        put_text(frame, "REC", (w / 2 - 286 * s, 34 * s), 0.5 * s, Palette.DANGER, 1, 3 * s)

    # ============================================================== standby
    def _draw_standby(self, frame: np.ndarray, now: float, a: float, s: float) -> None:
        e = self.engine
        h, w = frame.shape[:2]
        x, y, bw, bh = 20 * s, 20 * s, 330 * s, 62 * s
        draw_panel(frame, self.glow, (x, y, bw, bh), "", a, s, bg_alpha=0.45)
        live = [hd for hd in e.hands if not hd.stale]
        gesture = e.primary_result.gesture.value if e.primary_result else "NONE"
        put_text(frame, "HOLO-AR  STANDBY", (x + 16 * s, y + 24 * s), 0.5 * s, fade(Palette.PRIMARY, a), 1, 2 * s)
        put_text(frame, f"FPS {e.fps:.0f}   HANDS {len(live)}   {gesture}", (x + 16 * s, y + 46 * s), 0.4 * s,
                 fade(Palette.TEXT, a), 1, 1 * s)
        if not e.camera_connected:
            return
        pulse = 0.55 + 0.45 * math.sin(now * 3.0)
        put_text(frame, "SHOW AN OPEN PALM TO ACTIVATE THE INTERFACE", (w / 2, h - 50 * s), 0.55 * s,
                 fade(Palette.PRIMARY, a * pulse), 1, 3 * s, "center")
        put_text(frame, "(KEYBOARD: PRESS H)", (w / 2, h - 26 * s), 0.38 * s, fade(Palette.DIM_TEXT, a), 1, 2 * s, "center")

    def _draw_hold(self, frame: np.ndarray, now: float, s: float) -> None:
        hold = self.engine.hold
        if hold is None:
            return
        label, progress, pos = hold
        col = Palette.WARNING if label == "HIDING" else Palette.SUCCESS
        if label.startswith("TRAIN:"):
            label, col = label[6:], Palette.ACCENT
        r = 46 * s
        draw_progress_ring(frame, pos, r, progress, col, max(2, int(4 * s)))
        self.glow.circle(pos, r, fade(col, 0.2), 8 * s)
        put_text(frame, label, (pos[0], pos[1] - r - 12 * s), 0.42 * s, col, 1, 2 * s, "center")

    def _draw_brush_preview(self, frame: np.ndarray, s: float) -> None:
        e = self.engine
        d = e.drawing
        pos = e.cursor.pos
        radius = max(2, d.brush * (2 if d.tool == "eraser" else 1) // 2 + (6 if d.tool == "eraser" else 0))
        col = Palette.TEXT if d.tool == "eraser" else d.color
        cv2.circle(frame, ip(pos), int(radius), col, 1, AA)

    # =============================================================== toasts
    def _draw_toasts(self, frame: np.ndarray, now: float, s: float) -> None:
        e = self.engine
        h, w = frame.shape[:2]
        y = h - 200 * s
        for toast in e.toasts[-3:]:
            age = now - toast.t0
            fade_in = clamp(age / 0.2, 0.0, 1.0)
            fade_out = clamp((toast.duration - age) / 0.4, 0.0, 1.0)
            a = fade_in * fade_out
            if a <= 0.02:
                continue
            col = toast_color(toast.kind)
            tw = text_width(toast.text, 0.46 * s, 1, 2 * s)
            bw, bh = tw + 40 * s, 30 * s
            x = w / 2 - bw / 2
            glass_panel(frame, (x, y, bw, bh), 0.6 * a)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), fade(col, 0.7 * a), 1, AA)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + 4 * s), int(y + bh)), fade(col, a), -1)
            put_text(frame, toast.text, (w / 2, y + bh / 2 + 5 * s), 0.46 * s, fade(Palette.TEXT, a), 1, 2 * s, "center")
            y -= bh + 8 * s

    # ================================================================ boot
    def _draw_boot(self, frame: np.ndarray, t: float, s: float) -> None:
        h, w = frame.shape[:2]
        total = settings.ui.boot_seconds
        fade_out = clamp((total - t) / 0.5, 0.0, 1.0)
        progress = clamp(t / (total - 0.4), 0.0, 1.0)
        glass_panel(frame, (0, 0, w, h), 0.85 * fade_out, blur=False)
        c = (w / 2, h / 2 - 20 * s)
        R = 90 * s
        col = fade(Palette.PRIMARY, fade_out)
        draw_dashed_circle(frame, c, R, col, t * 80, 32, 0.55, 2)
        draw_dashed_circle(frame, c, R * 0.7, fade(Palette.ACCENT, fade_out), -t * 120, 20, 0.6, 2)
        draw_hex(frame, c, R * 0.35, col, t * 1.5, 2)
        cv2.ellipse(frame, ip(c), (int(R * 1.15), int(R * 1.15)), 0, t * 200, t * 200 + 60, col, 3, AA)
        draw_progress_ring(frame, c, R * 1.3, progress, col, 3)
        put_text(frame, "INITIALIZING HOLOGRAPHIC INTERFACE", (w / 2, c[1] + R * 1.3 + 50 * s), 0.6 * s, col, 1, 4 * s, "center")
        put_text(frame, f"{int(progress * 100)}%", (w / 2, c[1] + 8 * s), 0.7 * s, fade(Palette.TEXT, fade_out), 1, 2 * s, "center")

    # ========================================================= error screens
    def render_error(self, frame: np.ndarray, title: str, lines: List[str], now: float, footer: str = "") -> None:
        """Full-screen error / waiting message (e.g. camera lost). ``frame`` is overwritten."""
        h, w = frame.shape[:2]
        s = ui_scale(h)
        frame[:] = Palette.BACKGROUND
        self.glow.begin(w, h)
        c = (w / 2, h * 0.30)
        draw_dashed_circle(frame, c, 60 * s, Palette.DANGER, now * 60, 24, 0.55, 2)
        cv2.ellipse(frame, ip(c), (int(44 * s), int(44 * s)), 0, -now * 150, -now * 150 + 80, Palette.WARNING, 3, AA)
        self.glow.circle(c, 60 * s, fade(Palette.DANGER, 0.5), 10 * s)
        put_text(frame, title, (w / 2, h * 0.30 + 110 * s), 0.8 * s, Palette.DANGER, 1, 4 * s, "center")
        y = h * 0.30 + 150 * s
        for line in lines:
            put_text(frame, line, (w / 2, y), 0.46 * s, Palette.TEXT, 1, 1 * s, "center")
            y += 26 * s
        if footer:
            pulse = 0.55 + 0.45 * math.sin(now * 3.0)
            put_text(frame, footer, (w / 2, y + 20 * s), 0.5 * s, fade(Palette.PRIMARY, pulse), 1, 2 * s, "center")
        self.glow.compose(frame)


# ==========================================================================
# APPLICATION / MAIN
# ==========================================================================

KEY_NAMES = {
    27: "esc", 9: "tab", 13: "enter", 32: "space", 8: "backspace", 127: "delete", 65535: "delete",
    3014656: "delete", 2424832: "left", 2555904: "right", 2490368: "up", 2621440: "down",
    65361: "left", 65363: "right", 65362: "up", 65364: "down",
}


def key_name(code: int):
    if code < 0:
        return None
    if code in KEY_NAMES:
        return KEY_NAMES[code]
    c = code & 0xFF
    return chr(c).lower() if 32 < c < 127 else None


def show_fatal(title: str, message: str) -> None:
    """Print a fatal error and also show it in a window (for users who miss the console)."""
    print(f"\n[ERROR] {title}\n{message}\n", file=sys.stderr)
    try:
        img = np.empty((420, 980, 3), dtype=np.uint8)
        img[:] = Palette.BACKGROUND
        cv2.putText(img, title, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, Palette.DANGER, 1, cv2.LINE_AA)
        y = 110
        for raw in message.splitlines():
            for line in textwrap.wrap(raw, 95) or [""]:
                cv2.putText(img, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, Palette.TEXT, 1, cv2.LINE_AA)
                y += 28
        cv2.putText(img, "Press any key to close", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    Palette.PRIMARY, 1, cv2.LINE_AA)
        cv2.imshow("Hand Tracking AR - Error", img)
        cv2.waitKey(15000)
        cv2.destroyAllWindows()
    except cv2.error:
        pass


class Application:
    def __init__(self, args: argparse.Namespace) -> None:
        try:                                   # restore the saved settings profile (if any)
            load_profile()
        except ValueError as exc:
            print(f"[WARNING] {exc} - using default settings")
        apply_theme(settings.ui.theme)
        cam = settings.camera
        cam.index, cam.width, cam.height = args.camera, args.width, args.height
        if args.no_mirror:
            cam.mirror = False
        settings.ui.fullscreen = args.fullscreen
        if args.no_effects:
            settings.ui.glow = settings.ui.scan_effects = False

        self.camera = Camera(cam)
        self.tracker = HandTracker(settings.tracker)          # may raise TrackerError
        self.recognizer = GestureRecognizer(settings.gesture)
        self.engine = InteractionEngine(OUTPUT_DIR)
        self.ui = HolographicUI(self.engine)

        self.window = settings.ui.window_name
        self.camera_error = ""
        self.reconnect_attempts = 0
        self.next_reconnect = 0.0
        self.was_connected = False
        self.fps = 0.0
        self._frames = 0
        self.writer = None
        self.writer_size = (0, 0)
        self._snapshot_pending = False

    # ------------------------------------------------------------------ window
    def _open_window(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window, 1280, 720)
        if settings.ui.fullscreen:
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _toggle_fullscreen(self) -> None:
        settings.ui.fullscreen = not settings.ui.fullscreen
        mode = cv2.WINDOW_FULLSCREEN if settings.ui.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, mode)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        now = time.perf_counter()
        if event == cv2.EVENT_MOUSEMOVE:
            self.engine.mouse_move(x, y, now)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.engine.mouse_click(x, y, now)

    # ------------------------------------------------------------------ camera
    def _connect_camera(self, first: bool = False) -> bool:
        try:
            self.camera.open(probe_on_failure=first)
        except CameraError as exc:
            self.camera_error = str(exc)
            return False
        self.camera_error = ""
        w, h = self.camera.actual_size
        self.engine.camera_resolution = f"{w}x{h}"
        self.engine.camera_index = settings.camera.index
        return True

    def _next_resolution(self) -> None:
        cam = settings.camera
        current = (cam.width, cam.height)
        idx = RESOLUTIONS.index(current) if current in RESOLUTIONS else -1
        width, height = RESOLUTIONS[(idx + 1) % len(RESOLUTIONS)]
        try:
            self.camera.set_resolution(width, height)
        except CameraError as exc:
            self.camera_error = str(exc)
            self.engine.toast("RESOLUTION CHANGE FAILED", "error")
            return
        self.tracker.reset()
        self.recognizer.reset()
        w, h = self.camera.actual_size
        self.engine.camera_resolution = f"{w}x{h}"
        self.engine.toast(f"RESOLUTION {w}X{h}", "ok")

    def _handle_requests(self) -> None:
        for req in self.engine.take_requests():
            if req[0] == "exit":
                self.engine.running = False
            elif req[0] == "fullscreen":
                self._toggle_fullscreen()
            elif req[0] == "resolution_next":
                self._next_resolution()
            elif req[0] == "reconfigure_tracker":
                self._reconfigure_tracker()
            elif req[0] == "snapshot":
                self._snapshot_pending = True
            elif req[0] == "record_toggle":
                self._toggle_recording()
            elif req[0] == "camera_next":
                self._next_camera()

    def _reconfigure_tracker(self) -> None:
        try:
            self.tracker.reconfigure()
            self.recognizer.reset()
        except TrackerError as exc:
            self.engine.toast("TRACKER RESTART FAILED", "error")
            print(f"[ERROR] {exc}")

    def _next_camera(self) -> None:
        cam = settings.camera
        previous = cam.index
        for step in range(1, cam.probe_indices + 1):
            candidate = (previous + step) % cam.probe_indices
            try:
                self.camera.open(index=candidate)
            except CameraError:
                continue
            self.tracker.reset()
            self.recognizer.reset()
            w, h = self.camera.actual_size
            self.engine.camera_index = candidate
            self.engine.camera_resolution = f"{w}x{h}"
            self.engine.toast(f"CAMERA #{candidate}", "ok")
            return
        try:
            self.camera.open(index=previous)
        except CameraError:
            pass
        self.engine.toast("NO OTHER CAMERA FOUND", "warn")

    # ------------------------------------------------------- snapshot / recording
    def _toggle_recording(self) -> None:
        if self.writer is not None:
            self._stop_recording()
            self.engine.toast("RECORDING SAVED", "ok")
            return
        w, h = self.camera.actual_size
        if w <= 0:
            self.engine.toast("NO CAMERA FRAME YET", "warn")
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, time.strftime("recording_%Y%m%d_%H%M%S.mp4"))
        fps = float(min(30.0, max(15.0, self.fps or 30.0)))
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            self.engine.toast("RECORDER UNAVAILABLE", "error")
            return
        self.writer, self.writer_size = writer, (w, h)
        self.engine.recording = True
        self.engine.toast("RECORDING STARTED", "ok")

    def _stop_recording(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.engine.recording = False

    def _save_snapshot(self, frame: np.ndarray) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, time.strftime("snapshot_%Y%m%d_%H%M%S.png"))
        if cv2.imwrite(path, frame):
            self.engine.toast("SNAPSHOT " + os.path.basename(path), "ok")
            self.engine.sound.play("save")
        else:
            self.engine.toast("SNAPSHOT FAILED", "error")

    # --------------------------------------------------------------------- loop
    def run(self) -> int:
        self._open_window()
        if not self._connect_camera(first=True):
            print("\n[WARNING] " + self.camera_error + "\nThe application keeps retrying automatically.\n")
        self.was_connected = self.camera.connected
        self.engine.assistant.start()                      # background voice assistant (optional packages)
        prev = time.perf_counter()

        try:
            while self.engine.running:
                now = time.perf_counter()
                dt = min(max(now - prev, 1e-4), 0.1)
                prev = now

                if not self.camera.connected:
                    self._show_camera_error(now)
                else:
                    if not self.was_connected:
                        self.was_connected = True
                        self.engine.toast("CAMERA ONLINE", "ok", now)
                    frame = self.camera.read(0.05)
                    if frame is not None:
                        self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1.0 / dt
                        self._process_frame(frame, now, dt)

                code = cv2.waitKeyEx(1)
                name = key_name(code)
                if name:
                    self.engine.handle_key(name, time.perf_counter())
                self._handle_requests()

                self._frames += 1
                if self._frames > 10 and cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except TrackerError as exc:
            show_fatal("HAND TRACKING FAILED", str(exc))
            return 2
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return 0

    def _process_frame(self, frame: np.ndarray, now: float, dt: float) -> None:
        if settings.camera.mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        hands = self.tracker.process(frame, now, settings.camera.mirror)
        results = self.recognizer.update(hands, now)

        e = self.engine
        e.fps = self.fps
        e.inference_ms = self.tracker.inference_ms
        e.camera_connected = True
        e.update((w, h), hands, results, now, dt)
        self.ui.render(frame, now)
        if self._snapshot_pending:
            self._snapshot_pending = False
            self._save_snapshot(frame)
        if self.writer is not None:
            if (w, h) == self.writer_size:
                self.writer.write(frame)
            else:                              # resolution changed while recording
                self._stop_recording()
                self.engine.toast("RECORDING STOPPED (RESOLUTION CHANGED)", "warn")
        cv2.imshow(self.window, frame)

    def _show_camera_error(self, now: float) -> None:
        """Camera missing / lost: show a message and retry the connection periodically."""
        self.engine.camera_connected = False
        if self.was_connected:
            self.was_connected = False
            self.camera_error = "The camera stopped delivering frames (unplugged, or used by another app)."
            self.tracker.reset()
            self.recognizer.reset()
        if now >= self.next_reconnect:
            self.camera.release()
            self.reconnect_attempts += 1
            if self._connect_camera():
                self.reconnect_attempts = 0
                return
            self.next_reconnect = time.perf_counter() + settings.camera.reconnect_delay

        frame = np.empty((settings.camera.height, settings.camera.width, 3), dtype=np.uint8)
        lines: list = []
        for raw in (self.camera_error or "Waiting for camera...").splitlines():
            lines.extend(textwrap.wrap(raw, 90) or [""])
        title = "CAMERA UNAVAILABLE" if self.reconnect_attempts else "CAMERA DISCONNECTED"
        self.ui.render_error(frame, title, lines[:8], now, f"RECONNECTING...  ATTEMPT {self.reconnect_attempts}")
        cv2.imshow(self.window, frame)
        time.sleep(0.03)

    def shutdown(self) -> None:
        self.engine.assistant.stop()
        self._stop_recording()
        if settings.features.autosave_profile:
            try:
                save_profile()
            except OSError as exc:
                print(f"[WARNING] Could not save profile: {exc}")
        self.camera.release()
        self.tracker.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=APP_TITLE)
    p.add_argument("--camera", type=int, default=settings.camera.index, help="camera index (default 0)")
    p.add_argument("--width", type=int, default=settings.camera.width, help="requested camera width")
    p.add_argument("--height", type=int, default=settings.camera.height, help="requested camera height")
    p.add_argument("--fullscreen", action="store_true", help="start in fullscreen mode")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the camera image")
    p.add_argument("--no-effects", action="store_true", help="disable glow and scan effects (faster)")
    p.add_argument("--list-cameras", action="store_true", help="list working camera indices and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_cameras:
        found = Camera(settings.camera).probe()
        print("Working camera indices:", found if found else "none found")
        return 0
    try:
        app = Application(args)
    except TrackerError as exc:
        show_fatal("HAND TRACKING UNAVAILABLE", str(exc))
        return 2
    return app.run()


if __name__ == "__main__":
    sys.exit(main())