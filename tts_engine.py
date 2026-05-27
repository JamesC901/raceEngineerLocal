"""
tts_engine.py — Local TTS for the AC race engineer using Kokoro (hexgrad/kokoro).

Kokoro is Apache-licensed, fully offline, and GPU-accelerated.
On a 4080 Super it generates audio significantly faster than real-time.

Install:
    pip install kokoro>=0.9.4 soundfile misaki[en]
    # Windows only — needed for fallback G2P (grapheme-to-phoneme):
    # Install espeak-ng manually from https://github.com/espeak-ng/espeak-ng/releases
    #
    # Model weights (~330 MB) are downloaded automatically from HuggingFace on first run.
    # No manual .onnx / .bin files required.

Voice options (British male recommended for race engineer feel):
    "bm_daniel" — British male, slightly deeper (default)
    "bm_lewis"  — British male
    "am_adam"   — American male
    "af_sky"    — American female

Lang codes:
    'a' — American English (en-us)
    'b' — British English  (en-gb)  ← used here to match the bm_* voices
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

VOICE = "bm_daniel"   # Change to taste — see docstring above
LANG_CODE = "b"       # 'b' = British English; matches bm_* voices
SAMPLE_RATE = 24_000  # Kokoro's native output sample rate
SPEED = 0.95          # Slightly slower — gives prosody room to breathe

_pipeline = None
_pipeline_lock = threading.Lock()


def _get_pipeline():
    """Lazy-initialise KPipeline so import is cheap."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from kokoro import KPipeline  # noqa: PLC0415
                _pipeline = KPipeline(lang_code=LANG_CODE)
                print(f"[TTS] Kokoro KPipeline loaded. Voice: {VOICE}, speed: {SPEED}")
    return _pipeline


# ---------------------------------------------------------------------------
# Audio post-processing — takes the edge off the synthetic sound
# ---------------------------------------------------------------------------

def _humanise(audio: np.ndarray) -> np.ndarray:
    """
    DSP chain to make Kokoro sound warmer, lower, and more human:

    1. Lowpass at 5.5 kHz     — cuts the harsh synthetic highs aggressively
    2. Low-mid shelf boost     — adds body/warmth in the 150–400 Hz range
                                 where a real male voice resonates
    3. High-pass at 80 Hz     — removes any muddy sub-bass rumble
    4. Pitch micro-variation   — slow LFO time-warp breaks robotic flat pitch
    5. Soft knee compressor    — evens out the unnaturally flat TTS dynamics
    6. Normalise to 85% peak  — consistent loudness
    """
    audio = audio.astype(np.float32)

    # 1. Lowpass at 5.5 kHz — roll off brittle synthetic highs
    sos_lp = butter(5, 5500, btype="low", fs=SAMPLE_RATE, output="sos")
    audio = sosfilt(sos_lp, audio).astype(np.float32)

    # 2. Low-mid warmth boost — gentle 2nd-order peak at 250 Hz, +3 dB, Q=0.7
    #    Adds the chest resonance that makes a voice sound grounded and human.
    #    Implemented as a biquad peaking EQ via bilinear transform.
    f0, gain_db, Q = 250.0, 3.0, 0.7
    A  = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / SAMPLE_RATE
    alpha = np.sin(w0) / (2 * Q)
    b0 =  1 + alpha * A;  b1 = -2 * np.cos(w0);  b2 = 1 - alpha * A
    a0 =  1 + alpha / A;  a1 = -2 * np.cos(w0);  a2 = 1 - alpha / A
    sos_warm = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    audio = sosfilt(sos_warm, audio).astype(np.float32)

    # 3. High-pass at 80 Hz — remove sub-bass mud without touching the voice
    sos_hp = butter(2, 80, btype="high", fs=SAMPLE_RATE, output="sos")
    audio = sosfilt(sos_hp, audio).astype(np.float32)

    # 4. Pitch micro-variation — sinusoidal time-warp, ±1.2% over a ~4 s cycle
    #    Noticeably breaks up the robotic perfect-pitch flatness of TTS.
    n = len(audio)
    t = np.linspace(0, n / SAMPLE_RATE, n, dtype=np.float32)
    lfo = 1.0 + 0.012 * np.sin(2 * np.pi * 0.25 * t)   # 0.25 Hz, ±1.2%
    warped_positions = np.clip(
        np.cumsum(lfo) - 1, 0, n - 1
    ).astype(np.float32)
    indices_floor = warped_positions.astype(np.int32)
    indices_ceil  = np.clip(indices_floor + 1, 0, n - 1)
    frac          = warped_positions - indices_floor
    audio = audio[indices_floor] * (1 - frac) + audio[indices_ceil] * frac

    # 5. Soft-knee downward compression — ratio 3:1, threshold at 40% of peak
    #    Tames loud bursts and lifts quieter parts, mimicking natural speech dynamics.
    threshold = 0.40
    ratio     = 3.0
    knee      = 0.10          # soft-knee half-width around threshold
    abs_audio = np.abs(audio)
    # Smooth gain with a soft knee
    in_knee   = (abs_audio > threshold - knee) & (abs_audio < threshold + knee)
    above     = abs_audio >= threshold + knee
    gain      = np.ones_like(audio)
    # Knee region — blend linearly between 1 and ratio-reduced gain
    blend = (abs_audio[in_knee] - (threshold - knee)) / (2 * knee)
    gain[in_knee] = 1.0 - blend * (1.0 - 1.0 / ratio)
    # Above threshold — full ratio reduction
    gain[above] = (threshold + (abs_audio[above] - threshold) / ratio) / np.maximum(abs_audio[above], 1e-9)
    audio = (audio * gain).astype(np.float32)

    # 6. Normalise to 85% peak
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.85 / peak)

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
    _get_pipeline()                # warm up the model now, not mid-race
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
# Internal synthesis helper
# ---------------------------------------------------------------------------

def _synthesise(text: str) -> np.ndarray | None:
    """
    Run Kokoro on `text` and return a single concatenated float32 audio array,
    or None on failure.

    KPipeline.__call__ returns a generator of Result objects. Each Result
    has an `.audio` attribute (a torch.Tensor or numpy array). We concatenate
    all chunks so callers get a single contiguous array — identical behaviour
    to the old kokoro-onnx .create() call.
    """
    pipeline = _get_pipeline()
    chunks = []
    try:
        for result in pipeline(text, voice=VOICE, speed=SPEED):
            audio = result.audio
            # KPipeline may return a torch.Tensor; convert to numpy if needed
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()
            chunks.append(audio)
    except Exception as exc:
        print(f"[TTS] Error generating speech: {exc}")
        return None

    if not chunks:
        return None
    return np.concatenate(chunks)


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

    samples = _synthesise(text)
    if samples is not None:
        samples = _humanise(samples)
        _audio_queue.put(samples)


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