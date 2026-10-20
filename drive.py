"""
drive.py  -  Autonomous + Voice-Controlled Socket.IO server
============================================================
Uses Flask + Eventlet (standard Udacity stack).

Voice pipeline (Push-to-Talk):
  Press SPACEBAR → Whisper records for 3s → keyword match → steer/throttle override

Voice Commands:
  "brake" / "stop"   → throttle=0, steering=0
  "left"             → steer hard left
  "right"            → steer hard right
  "reverse" / "back" → back up (negative throttle)
  "auto" / silence   → resume autonomous model
"""

import eventlet
eventlet.monkey_patch()

# ── Grab real OS threading BEFORE eventlet shadows it ────────────────────────
import eventlet.patcher as _ep
_RealThread = _ep.original("threading").Thread
_RealLock   = _ep.original("threading").Lock
_real_time  = _ep.original("time")
# ─────────────────────────────────────────────────────────────────────────────

import base64
import os
import sys
from io import BytesIO

sys.stdout.reconfigure(line_buffering=True)

import cv2
import h5py
import numpy as np
from PIL import Image

import socketio
from flask import Flask

# ---------------------------------------------------------------------------
# Voice-control imports (graceful fallback)
# ---------------------------------------------------------------------------
try:
    import whisper as _whisper  # type: ignore  # installed as openai-whisper
    WHISPER_AVAILABLE = True
    print("[VOICE] [OK] whisper imported OK")
except ImportError:
    WHISPER_AVAILABLE = False
    print("[WARN] openai-whisper not found - run: pip install openai-whisper")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
    print("[VOICE] [OK] groq imported OK")
except ImportError:
    GROQ_AVAILABLE = False
    print("[WARN] groq not found - run: pip install groq")

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
    print("[VOICE] [OK] sounddevice imported OK")
except ImportError:
    AUDIO_AVAILABLE = False
    print("[WARN] sounddevice not found - run: pip install sounddevice")

try:
    import ctypes
    CTYPES_AVAILABLE = True
    print("[VOICE] [OK] ctypes loaded")
except ImportError:
    CTYPES_AVAILABLE = False
    print("[WARN] ctypes not available")

VOICE_CONTROL_READY = (WHISPER_AVAILABLE or GROQ_AVAILABLE) and AUDIO_AVAILABLE and CTYPES_AVAILABLE
print(f"[VOICE] Voice control ready: {VOICE_CONTROL_READY}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
PORT       = 4567

# Load Groq API Key
GROQ_API_KEY = None
groq_key_path = os.path.join(BASE_DIR, "groq_api_key.txt")
if os.path.exists(groq_key_path):
    try:
        with open(groq_key_path, "r", encoding="utf-8") as f:
            GROQ_API_KEY = f.read().strip()
            print("[VOICE] [OK] Loaded Groq API key from groq_api_key.txt")
    except Exception as e:
        print(f"[VOICE] Error reading groq_api_key.txt: {e}")

_groq_client = None
if GROQ_AVAILABLE and GROQ_API_KEY:
    try:
        _groq_client = Groq(api_key=GROQ_API_KEY)
        print("[VOICE] [OK] Groq client initialized successfully.")
    except Exception as e:
        print(f"[VOICE] Error initializing Groq client: {e}")

INPUT_W       = 64
INPUT_H       = 32
CROP_TOP_FRAC = 0.30
CROP_BOT_FRAC = 0.85
TARGET_SPEED  =15.0

# Voice config
RECORD_SECONDS    = 3        # seconds of audio captured on each spacebar press
COMMAND_HOLD_SECS = 3.0      # seconds command stays active before reverting to auto
WHISPER_MODEL_SIZE = "base"  # tiny | base | small | medium | large
SAMPLE_RATE        = 16000   # Hz — Whisper requires 16 kHz mono

# Voice commands state options: "start" or "stop"
# By default, "start" runs autonomous model control for both steering and throttle.
# "stop" overrides throttle to 0.0, but still uses autonomous model for steering.

# ---------------------------------------------------------------------------
# Socket.IO server
# ---------------------------------------------------------------------------
sio = socketio.Server(cors_allowed_origins="*")

# ---------------------------------------------------------------------------
# MLP model (loaded from model.h5)
# ---------------------------------------------------------------------------
_coefs      = []
_intercepts = []
_model_ok   = False

def _relu(x):
    return np.maximum(0.0, x)

def load_model(path):
    global _coefs, _intercepts, _model_ok
    if not os.path.exists(path):
        print("ERROR: model.h5 not found at:", path)
        sys.exit(1)
    print("Loading model from:", path)
    with h5py.File(path, "r") as hf:
        n = int(hf.attrs["num_layers"])
        for i in range(n):
            _coefs.append(np.array(hf["coefs"][f"w_{i}"], dtype=np.float32))
            _intercepts.append(np.array(hf["intercepts"][f"b_{i}"], dtype=np.float32))
    _model_ok = True
    print(f"Model ready: {len(_coefs)}-layer MLP loaded.")

# ---------------------------------------------------------------------------
# Image preprocessing & steering inference
# ---------------------------------------------------------------------------
def preprocess(img_bgr):
    h, w = img_bgr.shape[:2]
    t = int(h * CROP_TOP_FRAC)
    b = int(h * CROP_BOT_FRAC)
    cropped  = img_bgr[t:b, :]
    resized  = cv2.resize(cropped, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    yuv      = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
    features = (yuv.astype(np.float32) / 127.5) - 1.0
    return features.flatten()

def predict_steering(features):
    if not _model_ok:
        return 0.0
    x = features
    for i in range(len(_coefs) - 1):
        x = _relu(x @ _coefs[i] + _intercepts[i])
    return float(np.clip(float(x @ _coefs[-1] + _intercepts[-1]), -1.0, 1.0))

def compute_throttle(steering, speed):
    speed_err = TARGET_SPEED - speed
    if speed_err > 0:
        return float(np.clip(0.3 + 0.025 * speed_err, 0.3, 1.0))
    return 0.1

# ---------------------------------------------------------------------------
# Voice command — global state
# ---------------------------------------------------------------------------
_voice_command       = "start"
_voice_command_until = 0.0
_voice_lock          = _RealLock()   # real OS lock
_is_recording        = False          # tracks whether we are currently recording
_recording_start_time = 0.0
_audio_buffer        = None
_max_recording_time  = 20.0          # max recording duration in seconds
_whisper_model       = None

def get_voice_command() -> str:
    """Thread-safe read; returns 'start' if a transient command has expired."""
    with _voice_lock:
        if _real_time.time() < _voice_command_until:
            return _voice_command
        # If the active command was left or right and it expired, revert to start
        if _voice_command in ("left", "right"):
            return "start"
        return _voice_command

def set_voice_command(cmd: str):
    """Thread-safe write with conditional expiry."""
    global _voice_command, _voice_command_until
    with _voice_lock:
        _voice_command = cmd
        if cmd in ("left", "right"):
            _voice_command_until = _real_time.time() + 3.0  # holds for 3 seconds
        else:
            _voice_command_until = 0.0  # persistent
    print(f"\n[VOICE] [COMMAND] -> {cmd.upper()}\n")

# ---------------------------------------------------------------------------
# Whisper model loader
# ---------------------------------------------------------------------------
def load_whisper_model():
    global _whisper_model
    if not WHISPER_AVAILABLE:
        return
    if _whisper_model is not None:
        return
    print(f"[VOICE] Loading Whisper '{WHISPER_MODEL_SIZE}' model (downloads on first run)...")
    _whisper_model = _whisper.load_model(WHISPER_MODEL_SIZE)
    print("[VOICE] [OK] Whisper model ready.")

# ---------------------------------------------------------------------------
# Hold-to-talk: Recording processing
# ---------------------------------------------------------------------------
def _process_audio(audio_chunk):
    """
    Runs in a real OS thread to process transcription and trigger the matched command.
    """
    try:
        text = ""
        transcription_method = ""
        
        # 1. Try Groq Cloud Whisper API if available
        if _groq_client is not None:
            try:
                print("[VOICE] [AUDIO] Transcribing via Groq Cloud Whisper API...")
                import io
                from scipy.io import wavfile
                
                # Write audio chunk to in-memory WAV buffer
                wav_buf = io.BytesIO()
                # sounddevice records floats between -1.0 and 1.0, convert/write as 16-bit PCM wav
                # Whisper API requires file-like audio format (wav)
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                wavfile.write(wav_buf, SAMPLE_RATE, audio_int16)
                wav_buf.seek(0)
                
                # Request transcription
                transcription = _groq_client.audio.transcriptions.create(
                    file=("audio.wav", wav_buf.read(), "audio/wav"),
                    model="whisper-large-v3",
                    language="en"
                )
                text = transcription.text.strip().lower()
                transcription_method = "Groq Cloud"
            except Exception as groq_err:
                print(f"[VOICE] Groq API call failed: {groq_err}. Falling back to local Whisper...")
        
        # 2. Fallback to Local Whisper Model
        if not text:
            print("[VOICE] [AUDIO] Transcribing via local Whisper model...")
            if not WHISPER_AVAILABLE:
                print("[VOICE] [ERROR] Local Whisper is not available!")
                return
            load_whisper_model()
            if _whisper_model is None:
                print("[VOICE] [ERROR] Whisper model could not be loaded!")
                return
            result = _whisper_model.transcribe(audio_chunk, fp16=False, language="en")
            text   = result.get("text", "").strip().lower()
            transcription_method = "local Whisper"

        if text:
            print(f'[VOICE] [HEARD] ({transcription_method}) Heard: "{text}"')
        else:
            print(f"[VOICE] [HEARD] ({transcription_method}) (nothing detected)")
            return

        # Keyword match
        cmd = _match_command(text)
        print(f"[VOICE] [MATCH] Command matched: {cmd}")

        if cmd not in ("none", ""):
            set_voice_command(cmd)
        else:
            print("[VOICE] [INFO] No known command - state unchanged")

    except Exception as exc:
        print(f"[VOICE] [ERROR] Error during processing: {exc}")

def _match_command(text: str) -> str:
    """Map transcript -> start / stop / left / right / none."""
    if not text:
        return "none"
    if any(w in text for w in ["stop", "brake", "halt", "slow"]):
        return "stop"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if any(w in text for w in ["start", "go", "reset", "resume", "continue", "auto"]):
        return "start"
    return "none"

# ---------------------------------------------------------------------------
# Spacebar key listener (ctypes GetAsyncKeyState polling)
# ---------------------------------------------------------------------------
def spacebar_polling_loop():
    """
    Poller running in a real OS thread. Checks the state of the Spacebar.
    Directly bypasses eventlet and pynput compatibility issues.
    """
    import ctypes
    global _is_recording, _recording_start_time, _audio_buffer
    VK_SPACE = 0x20
    user32 = ctypes.windll.user32

    was_down = False
    print("[VOICE] [OK] Keyboard polling listener active - hold SPACEBAR to speak, release to stop.")

    while True:
        try:
            # Check most significant bit to see if spacebar is down
            is_down = bool(user32.GetAsyncKeyState(VK_SPACE) & 0x8000)

            if is_down and not was_down:
                # Key Press event
                if not _is_recording:
                    _is_recording = True
                    _recording_start_time = _real_time.time()
                    print("\n[VOICE] [REC] Recording... (HOLD Spacebar, RELEASE to stop recording)")
                    _audio_buffer = sd.rec(
                        int(_max_recording_time * SAMPLE_RATE),
                        samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype="float32",
                    )
                was_down = True

            elif not is_down and was_down:
                # Key Release event
                if _is_recording:
                    sd.stop()
                    _is_recording = False
                    duration = _real_time.time() - _recording_start_time
                    print(f"[VOICE] [REC] Recording stopped. Duration: {duration:.2f}s")

                    if duration < 0.5:
                        print("[VOICE] [WARN] Recording too short (< 0.5s), ignoring.")
                    else:
                        num_frames = int(min(duration, _max_recording_time) * SAMPLE_RATE)
                        audio_chunk = _audio_buffer[:num_frames].flatten()

                        # Process in a separate real OS thread
                        t = _RealThread(target=_process_audio, args=(audio_chunk,), daemon=True)
                        t.start()
                was_down = False

        except Exception as exc:
            print(f"[VOICE] [ERROR] Keyboard polling error: {exc}")

        # Poll every 10ms (using real_time to avoid eventlet interference)
        _real_time.sleep(0.01)

# ---------------------------------------------------------------------------
# Socket.IO handlers
# ---------------------------------------------------------------------------
@sio.on("connect")
def on_connect(sid, environ):
    print(f"[SIO] Simulator connected (sid={sid})")
    sio.emit("steer", data={"steering_angle": "0", "throttle": "0"}, to=sid)

@sio.on("disconnect")
def on_disconnect(sid):
    print(f"[SIO] Simulator disconnected (sid={sid})")

@sio.on("telemetry")
def on_telemetry(sid, data):
    if not data:
        sio.emit("manual", data={}, skip_sid=sid)
        return

    speed   = float(data.get("speed", 0.0))
    img_b64 = data.get("image", "")
    cmd     = get_voice_command()  # "start" / "stop" / "left" / "right"

    try:
        # 1. Image Preprocessing & Steering Prediction (always run so we remain on the road/track!)
        raw     = base64.b64decode(img_b64)
        pil_img = Image.open(BytesIO(raw))
        img_rgb = np.asarray(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        features = preprocess(img_bgr)
        steering = predict_steering(features)

        # 2. Control decision based on voice command state and safety check
        if cmd == "stop":
            throttle = 0.0
            mode_label = "[VOICE STOP]"
        elif cmd == "left":
            # Steer left. Override if model suggests sharp right curve to avoid going off track.
            if steering > 0.3:
                # Keep steering from model to stay on track
                mode_label = "[VOICE L-SAFE]"
            else:
                steering = -0.7
                mode_label = "[VOICE LEFT]"
            throttle = compute_throttle(steering, speed)
        elif cmd == "right":
            # Steer right. Override if model suggests sharp left curve to avoid going off track.
            if steering < -0.3:
                # Keep steering from model to stay on track
                mode_label = "[VOICE R-SAFE]"
            else:
                steering = 0.7
                mode_label = "[VOICE RIGHT]"
            throttle = compute_throttle(steering, speed)
        else:  # "start" / default autonomous
            throttle = compute_throttle(steering, speed)
            mode_label = "[AUTO START]"

        # 3. Emit control to simulator (always send brake=False so we drop speed with physical friction deceleration)
        sio.emit("steer", data={
            "steering_angle": str(steering),
            "throttle":       str(throttle),
            "brake":          "False",
        }, to=sid)

        print(f"  {mode_label:<14} steer={steering:+.4f}  throttle={throttle:.3f}  speed={speed:.1f} mph")

    except Exception as exc:
        print(f"[WARN] telemetry error: {exc}")
        sio.emit("steer", data={"steering_angle": "0", "throttle": "0.1", "brake": "False"}, to=sid)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def free_port(port):
    import subprocess
    try:
        result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
                print(f"Freed port {port} (killed PID {pid}).")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    free_port(PORT)
    load_model(MODEL_PATH)

    if VOICE_CONTROL_READY:
        if _groq_client is None:
            load_whisper_model()
        else:
            print("[VOICE] [OK] Groq transcription is ready. Local Whisper model will load on-demand as a fallback if needed.")
        # Start keyboard polling loop in real OS thread
        t = _RealThread(target=spacebar_polling_loop, daemon=True)
        t.start()
    else:
        print("[WARN] Voice control NOT started (missing dependencies).")
        print("       Run: pip install groq sounddevice")

    print()
    print("+--------------------------------------------------------------+")
    print("|   Autonomous + Voice-Controlled Driver                       |")
    print(f"|   ws://localhost:{PORT}                                      |")
    print("|                                                              |")
    print("|   Open simulator -> choose track -> AUTONOMOUS MODE           |")
    print("|                                                              |")
    if VOICE_CONTROL_READY:
        print("|   HOLD-TO-TALK: Hold SPACEBAR to speak, release to process   |")
        print("|                                                              |")
        print("|   Commands:                                                  |")
        print("|     'stop'             -> Stop gradually (keeps steering)    |")
        print("|     'start'            -> Start driving autonomously         |")
        print("|     'left'             -> Turn left (safety-assisted)        |")
        print("|     'right'            -> Turn right (safety-assisted)       |")
    else:
        print("|   !! Voice control DISABLED - install missing packages !!   |")
    print("+--------------------------------------------------------------+")
    print()

    wsgi_app = socketio.Middleware(sio, Flask(__name__))
    eventlet.wsgi.server(eventlet.listen(("", PORT)), wsgi_app, log_output=False)

if __name__ == "__main__":
    main()
