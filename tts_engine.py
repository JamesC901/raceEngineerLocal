"""
tts_engine.py — Local TTS for the AC race engineer using Kokoro.

Kokoro is MIT-licensed, fully offline, and GPU-accelerated.
On a 4080 Super it generates audio significantly faster than real-time.

Install:
    pip install kokoro-onnx sounddevice numpy
    # The model weights are downloaded automatically on first run (~330 MB).
    # If you want a specific voice, see VOICE constant below.

Voice options (British male recommended for race engineer feel):
    "bm_lewis"  — British male (default, sounds the most like a real engineer)
    "bm_daniel" — British male, slightly deeper
    "am_adam"   — American male
    "af_sky"    — American female
"""

import queue
import re
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# Kokoro setup
# ---------------------------------------------------------------------------

VOICE = "bm_daniel"       # Change to taste — see docstring above
SAMPLE_RATE = 24000      # Kokoro's native output sample rate
SPEED = 0.95             # Slightly slower — gives prosody room to breathe

_kokoro = None
_kokoro_lock = threading.Lock()


def _get_kokoro():
    """Lazy-initialise Kokoro so import is cheap."""
    global _kokoro
    if _kokoro is None:
        with _kokoro_lock:
            if _kokoro is None:
                from kokoro_onnx import Kokoro  # noqa: PLC0415
                _kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
                print(f"[TTS] Kokoro loaded. Voice: {VOICE}, speed: {SPEED}")
    return _kokoro


# ---------------------------------------------------------------------------
# Audio post-processing — takes the edge off the synthetic sound
# ---------------------------------------------------------------------------

def _humanise(audio: np.ndarray) -> np.ndarray:
    """
    Apply a light chain of DSP to make Kokoro output sound less robotic:

    1. Gentle lowpass (8 kHz)  — rolls off the brittle synthetic highs
    2. Subtle pitch micro-variation — breaks up the perfectly flat pitch
       that our ears instantly flag as synthetic
    3. Normalise to 90% peak   — consistent loudness across sentences
    """
    audio = audio.astype(np.float32)

    # 1. Lowpass at 8 kHz — removes harshness without dulling intelligibility
    sos = butter(4, 8000, btype="low", fs=SAMPLE_RATE, output="sos")
    audio = sosfilt(sos, audio).astype(np.float32)

    # 2. Pitch micro-variation via very slow LFO on playback rate
    #    We resample with a sinusoidal time-warp (~±0.4% over ~3 s cycle)
    #    Subtle enough to be subliminal but breaks the robotic flatness.
    n = len(audio)
    t = np.linspace(0, n / SAMPLE_RATE, n, dtype=np.float32)
    lfo = 1.0 + 0.004 * np.sin(2 * np.pi * 0.33 * t)   # 0.33 Hz, ±0.4%
    warped_positions = np.clip(
        np.cumsum(lfo) - 1, 0, n - 1
    ).astype(np.float32)
    indices_floor = warped_positions.astype(np.int32)
    indices_ceil  = np.clip(indices_floor + 1, 0, n - 1)
    frac          = warped_positions - indices_floor
    audio = audio[indices_floor] * (1 - frac) + audio[indices_ceil] * frac

    # 3. Normalise to 90% peak — consistent loudness across sentences
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.90 / peak)

    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Audio playback — dedicated thread so TTS never blocks Ollama streaming
# ---------------------------------------------------------------------------

_audio_queue: queue.Queue = queue.Queue()
_tts_thread: threading.Thread | None = None
_tts_running = False


def _audio_worker():
    """Consume audio arrays from the queue and play them sequentially."""
    while _tts_running or not _audio_queue.empty():
        try:
            audio = _audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if audio is None:          # sentinel — exit signal
            break
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()                  # block until this chunk finishes
        _audio_queue.task_done()


def start_tts():
    """Start the background audio playback thread. Call once at startup."""
    global _tts_thread, _tts_running
    _tts_running = True
    _get_kokoro()                  # warm up the model now, not mid-race
    _tts_thread = threading.Thread(target=_audio_worker, daemon=True)
    _tts_thread.start()
    print("[TTS] Audio playback thread started.")


def stop_tts():
    """Gracefully stop the TTS thread."""
    global _tts_running
    _tts_running = False
    _audio_queue.put(None)         # sentinel to unblock the worker
    if _tts_thread:
        _tts_thread.join(timeout=3)
    print("[TTS] Audio playback thread stopped.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speak(text: str) -> None:
    """
    Convert `text` to speech and enqueue it for playback.
    Returns immediately — audio plays in the background thread.
    """
    text = text.strip()
    if not text:
        return

    kokoro = _get_kokoro()
    try:
        samples, _ = kokoro.create(text, voice=VOICE, speed=SPEED, lang="en-us")
        # samples is a float32 numpy array at SAMPLE_RATE
        samples = _humanise(samples)
        _audio_queue.put(samples)
    except Exception as exc:
        print(f"[TTS] Error generating speech: {exc}")


def speak_streaming(text_iterator) -> str:
    """
    Consume an iterator of string tokens (from Ollama streaming), accumulate
    them into sentences, and speak each sentence as soon as it's complete.

    This gives the lowest latency: the driver hears the first sentence while
    the rest of the response is still being generated.

    Returns the full response text.
    """
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    buffer = ""
    full_text = ""

    for token in text_iterator:
        buffer += token
        full_text += token

        # Split on sentence boundaries — speak complete sentences immediately
        parts = sentence_endings.split(buffer)
        if len(parts) > 1:
            # All parts except the last are complete sentences
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    speak(sentence)
            buffer = parts[-1]    # keep the incomplete tail

    # Speak any remaining text
    if buffer.strip():
        speak(buffer.strip())

    return full_text