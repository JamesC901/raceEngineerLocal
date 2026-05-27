import argparse
import queue
import signal
import threading
import time
import numpy as np
import whisper
import requests
import json
import os
from dotenv import load_dotenv
from game_server import start_game_server, stop_game_server
from ac_shared_memory import get_shared_memory_data
from gap_calculator import calculate_gaps
from tts_engine import start_tts, stop_tts, speak, speak_streaming

try:
    import sounddevice as sd
except ImportError:
    sd = None

load_dotenv()
MODEL_NAME   = os.getenv("MODEL_NAME")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_URL   = os.getenv("OLLAMA_URL")
_running = True

# Whisper commonly hallucinates these on silence/noise
HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching",
    "please subscribe", "you", ".", "..", "...", "bye", "bye bye",
    "subtitles by", "transcribed by",
}

# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def handle_sigint(sig, frame):
    global _running
    print("\n\nStopping...")
    _running = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_lap_time(ms: int) -> str:
    if ms > 0:
        minutes = ms // 60000
        seconds = (ms % 60000) / 1000
        return f"{minutes}:{seconds:06.3f}"
    return "N/A"


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0
    return audio.astype(np.float32)


def is_speech(audio: np.ndarray, threshold: float = 0.005) -> bool:
    audio = normalize_audio(audio)
    rms = np.sqrt(np.mean(np.square(audio), axis=-1))
    return rms >= threshold


def is_hallucination(text: str) -> bool:
    return text.lower().strip(".,!? ") in HALLUCINATIONS


def transcribe_audio_array(audio: np.ndarray, model_name: str = MODEL_NAME, model=None) -> str:
    audio = normalize_audio(audio)
    if model is None:
        model = whisper.load_model(model_name)
    result = model.transcribe(audio, language="en", verbose=False, no_speech_threshold=0.6, fp16=False)
    return result["text"].strip()

# ---------------------------------------------------------------------------
# Telemetry formatting — 100 % from shared memory
# ---------------------------------------------------------------------------

def format_telemetry(sm: dict) -> str:
    """
    Build a human-readable telemetry block from a shared memory snapshot.
    `sm` is the dict returned by get_shared_memory_data().
    """
    if not sm:
        return "[No telemetry — AC not connected]"

    lines = ["[Current Telemetry]"]

    # --- Session / static ---
    lines.append(f"  Track        : {sm.get('track', 'N/A')} "
                 f"({sm.get('tire_compound', 'N/A')} tyres)")
    lines.append(f"  Car          : {sm.get('car_model', 'N/A')}")
    lines.append(f"  Cars on grid : {sm.get('num_cars', 'N/A')}")

    session_map = {-1: "Unknown", 0: "Practice", 1: "Qualifying",
                   2: "Race", 3: "Hotlap", 4: "Time Attack",
                   5: "Drift", 6: "Drag"}
    session_label = session_map.get(sm.get('session_type', -1), "Unknown")
    lines.append(f"  Session      : {session_label}")

    # --- Driving ---
    lines.append(f"  Speed        : {sm.get('speed_kmh', 0):.1f} km/h  |  "
                 f"RPM: {sm.get('engine_rpm', 0):.0f}  |  "
                 f"Gear: {sm.get('gear', 0)}")
    lines.append(f"  Throttle     : {sm.get('gas', 0)*100:.1f}%  |  "
                 f"Brake: {sm.get('brake', 0)*100:.1f}%  |  "
                 f"Fuel: {sm.get('fuel', 0):.2f} L")

    # Active aids
    aids = []
    if sm.get('is_tc_in_action'):
        aids.append("TC")
    if sm.get('is_abs_in_action'):
        aids.append("ABS")
    if sm.get('is_engine_limiter_on'):
        aids.append("PIT-LIMITER")
    if aids:
        lines.append(f"  Active aids  : {', '.join(aids)}")

    # --- Timing ---
    lines.append(f"  Current lap (Incomplete Lap, still in progress) : {format_lap_time(sm.get('current_lap_ms', 0))}")
    lines.append(f"  Last lap     : {format_lap_time(sm.get('last_lap_ms', 0))}")
    lines.append(f"  Best lap     : {format_lap_time(sm.get('best_lap_ms', 0))}")
    lines.append(f"  Laps done    : {sm.get('completed_laps', 0)}")
    lines.append(f"  Position     : P{sm.get('position', '?')}")
    lines.append(f"  Flag         : {sm.get('flag', 'none').upper()}")
    lines.append(f"  In pit lane  : {sm.get('is_in_pit_lane', False)}")

    # --- Tyres ---
    tyre_wear = sm.get('tyre_wear', [0, 0, 0, 0])
    lines.append(
        f"  Tyre wear (FL/FR/RL/RR): "
        f"{tyre_wear[0]:.3f} / {tyre_wear[1]:.3f} / "
        f"{tyre_wear[2]:.3f} / {tyre_wear[3]:.3f}"
    )
    tyre_temp = sm.get('tyre_core_temp', [0, 0, 0, 0])
    lines.append(
        f"  Tyre temp °C  (FL/FR/RL/RR): "
        f"{tyre_temp[0]:.1f} / {tyre_temp[1]:.1f} / "
        f"{tyre_temp[2]:.1f} / {tyre_temp[3]:.1f}"
    )
    tyre_dirty = sm.get('tyre_dirty', [0, 0, 0, 0])
    lines.append(
        f"  Tyre dirt     (FL/FR/RL/RR): "
        f"{tyre_dirty[0]:.3f} / {tyre_dirty[1]:.3f} / "
        f"{tyre_dirty[2]:.3f} / {tyre_dirty[3]:.3f}"
    )

    # --- Damage ---
    dmg = sm.get('car_damage', [0, 0, 0, 0, 0])
    lines.append(
        f"  Damage (F/R/L/R/C): "
        f"{dmg[0]:.3f} / {dmg[1]:.3f} / {dmg[2]:.3f} / "
        f"{dmg[3]:.3f} / {dmg[4]:.3f}"
    )

    # --- Gaps (from acgaps file writer) ---
    gaps = calculate_gaps()
    def _fmt_gap(val):
        return f"{val:.2f}s" if val is not None else "N/A"
    lines.append(f"  Gap to car ahead  : {_fmt_gap(gaps.get('gap_ahead_s'))}")
    lines.append(f"  Gap to car behind : {_fmt_gap(gaps.get('gap_behind_s'))}")
    lines.append(f"  Gap to leader     : {_fmt_gap(gaps.get('gap_to_leader_s'))}")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Ollama interface
# ---------------------------------------------------------------------------

def _stream_ollama_tokens(messages: list):
    """
    Generator that posts to Ollama with stream=True and yields each text token.
    Also prints tokens to stdout as they arrive.
    """
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
    }, stream=True)

    if response.status_code != 200:
        yield f"[Ollama error {response.status_code}]"
        return

    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        if token:
            print(token, end="", flush=True)
            yield token
        if chunk.get("done"):
            break


def _send_to_ollama(messages: list, stream: bool = True) -> str:
    """
    Send a messages list to Ollama and return the full reply string.
    Streams tokens to stdout and through TTS sentence-by-sentence.
    """
    print("Ollama: ", end="", flush=True)
    full_reply = speak_streaming(_stream_ollama_tokens(messages))
    print()
    return full_reply


def ask_ollama(user_text: str, conversation_history: list) -> str:
    """Voice-triggered exchange. Appends telemetry to the user message."""
    sm = get_shared_memory_data()
    telemetry_block = format_telemetry(sm)
    full_message = f"{user_text}\n\n{telemetry_block}"

    conversation_history.append({"role": "user", "content": full_message})
    print(f"\n[Sending to Ollama]\n{full_message}\n")

    reply = _send_to_ollama(conversation_history)

    conversation_history.append({"role": "assistant", "content": reply})
    return reply

# ---------------------------------------------------------------------------
# Proactive 1-second telemetry loop
# ---------------------------------------------------------------------------

# System prompt for the proactive monitor — separate from the voice conversation
_MONITOR_SYSTEM = (
    "You are an experienced F1 race engineer monitoring live telemetry for a driver "
    "in Assetto Corsa. Every second you receive a full telemetry snapshot. "
    "Your job is to call out anything that genuinely warrants driver attention — "
    "examples: significant tyre wear difference across axles, dangerously high temps, "
    "car damage detected, flag changes (yellow/blue/black/checkered), pit-lane entry/exit, "
    "big lap-time delta vs best lap, or fuel running low.\n\n"
    "Rules:\n"
    "- If nothing is noteworthy, respond with exactly: SILENT\n"
    "- Never comment on normal, expected driving data.\n"
    "- Keep radio calls short, punchy, and realistic (1-2 sentences max).\n"
    "- Do NOT repeat an observation you already made unless the situation has worsened.\n"
    "- Do NOT hallucinate or invent data not present in the telemetry."
)


def _telemetry_monitor_loop(conversation_history: list, interval: float = 1.0):
    """
    Background thread: every `interval` seconds, pull shared memory, format it,
    and ask Ollama whether anything is worth reporting. If the reply is not
    'SILENT', print it so the driver can hear/read it.

    Uses its own short message list (only system + current snapshot) so it
    doesn't pollute the voice conversation history. Notable observations are
    injected into conversation_history as assistant messages so the voice
    assistant has context.
    """
    global _running

    # Track last-seen values so we can detect changes
    prev = {}

    while _running:
        time.sleep(interval)

        sm = get_shared_memory_data()
        if not sm or not sm.get('connected'):
            continue

        # Only monitor while a session is live (status == 2)
        if sm.get('status', 0) != 2:
            continue

        telemetry_block = format_telemetry(sm)

        # print(telemetry_block)

        monitor_messages = [
            {"role": "system",    "content": _MONITOR_SYSTEM},
            {"role": "user",      "content": telemetry_block},
        ]

        try:
            response = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "messages": monitor_messages,
                "stream": False,
            })
            if response.status_code != 200:
                continue

            data = response.json()
            reply = data.get("message", {}).get("content", "").strip()

            if reply and reply.upper() != "SILENT":
                print(f"\n[Engineer] {reply}\n", flush=True)
                speak(reply)
                # Inject into voice conversation so the driver's next question has context
                conversation_history.append({
                    "role": "assistant",
                    "content": f"[Proactive observation] {reply}"
                })

        except Exception as exc:
            print(f"[Monitor] Ollama request failed: {exc}")


# ---------------------------------------------------------------------------
# Audio transcription
# ---------------------------------------------------------------------------

def transcribe_live_microphone(
    chunk_duration:   float = 0.5,
    silence_duration: float = 1.2,
    model_name:       str   = MODEL_NAME,
    system_prompt:    str   = None,
) -> None:
    """
    Accumulates audio chunks while speech is detected.
    When silence_duration seconds of silence passes after speech,
    the accumulated buffer is transcribed as one complete utterance.
    """
    global _running

    if sd is None:
        raise RuntimeError("sounddevice is required. Install it with: pip install sounddevice")

    signal.signal(signal.SIGINT, handle_sigint)

    sample_rate   = 16000
    model         = whisper.load_model(model_name)
    print(f"Loaded Whisper model: {model_name}")
    print(f"Connected to Ollama model: {OLLAMA_MODEL}")

    chunk_frames  = int(sample_rate * chunk_duration)
    audio_queue   = queue.Queue()
    conversation_history = []
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})

    # Start proactive telemetry monitor on a daemon thread
    monitor_thread = threading.Thread(
        target=_telemetry_monitor_loop,
        args=(conversation_history,),
        daemon=True,
    )
    monitor_thread.start()
    print("[Monitor] Proactive telemetry monitor started (1 s interval).")

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio input status: {status}")
        audio_queue.put(indata.copy())

    print("Listening... Press Ctrl+C to stop.\n")

    utterance_index = 1
    speech_buffer   = []
    silent_chunks   = 0
    max_silent_chunks = int(silence_duration / chunk_duration)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                        blocksize=chunk_frames, callback=callback):
        while _running:
            try:
                data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            audio          = data.flatten()
            speech_detected = is_speech(audio)

            if speech_detected:
                speech_buffer.append(audio)
                silent_chunks = 0
            elif speech_buffer:
                speech_buffer.append(audio)
                silent_chunks += 1

                if silent_chunks >= max_silent_chunks:
                    full_audio    = np.concatenate(speech_buffer)
                    speech_buffer = []
                    silent_chunks = 0

                    text = transcribe_audio_array(full_audio, model=model)

                    if not text or is_hallucination(text):
                        continue

                    print(f"\nYou [{utterance_index}]: {text}")
                    ask_ollama(text, conversation_history)
                    utterance_index += 1

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    start_game_server()
    start_tts()

    parser = argparse.ArgumentParser(description="Whisper + Ollama AC race engineer")
    parser.add_argument("--chunk-duration",   type=float, default=0.5,
                        help="Audio polling interval in seconds")
    parser.add_argument("--silence-duration", type=float, default=1.2,
                        help="Seconds of silence before transcribing")
    parser.add_argument("--whisper-model",    default=MODEL_NAME,
                        help="Whisper model (tiny, base, small, medium, large)")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are an experienced F1 race engineer providing expert technical advice "
            "about vehicle performance, setup, and racing strategy. "
            "You are communicating with a driver currently racing in Assetto Corsa. "
            "You will receive a live telemetry snapshot with every driver message.\n\n"
            "Guidelines:\n"
            "- Keep responses concise and realistic, like real F1 team radio.\n"
            "- Only reference data that is actually present in the telemetry.\n"
            "- If telemetry is missing or AC is not connected, say so.\n"
            "- If the driver's message is unclear, ask them to repeat it.\n"
            "- Do NOT hallucinate lap times, positions, or any other data."
        ),
        help="System prompt for the voice assistant."
    )
    args = parser.parse_args()

    try:
        transcribe_live_microphone(
            chunk_duration=args.chunk_duration,
            silence_duration=args.silence_duration,
            model_name=args.whisper_model,
            system_prompt=args.system_prompt,
        )
    finally:
        stop_tts()
        stop_game_server()


if __name__ == "__main__":
    main()