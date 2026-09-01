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
# Proactive 1-second telemetry loop — threshold checks in Python, not the LLM
# ---------------------------------------------------------------------------

# Thresholds — edit these to taste
TYRE_TEMP_HIGH    = 110.0   # °C — report if any tyre exceeds this
TYRE_TEMP_LOW     =  75.0   # °C — report if any tyre drops below this
TYRE_WEAR_DELTA   =   0.05  # report if any corner differs from average by this much
TYRE_DIRT_HIGH    =   0.5   # report if any tyre dirt exceeds this
FUEL_LOW_PCT      =   5.0   # report if fuel % drops below this
LAP_DELTA_S       =   3.0   # report if current lap is this many seconds off best

# LLM only phrases the alert — it never decides whether to fire one
_MONITOR_SYSTEM = (
    "You are an F1 race engineer. You will be given a specific telemetry alert that has "
    "already been confirmed as genuine by the engineering system. Your only job is to "
    "rephrase it as a single short team-radio call — calm, assertive, no preamble. "
    "Include the exact numeric values provided. Do not add any commentary, caveats, or "
    "additional observations beyond what is given to you."
)


def _check_alerts(sm: dict, gaps: dict, prev: dict) -> list[str]:
    """
    Pure Python threshold checks. Returns a list of alert strings to speak,
    or an empty list if everything is nominal. Never calls the LLM.
    """
    alerts = []

    tyre_temp  = sm.get('tyre_core_temp', [0, 0, 0, 0])
    tyre_wear  = sm.get('tyre_wear',      [0, 0, 0, 0])
    tyre_dirty = sm.get('tyre_dirty',     [0, 0, 0, 0])
    dmg        = sm.get('car_damage',     [0, 0, 0, 0, 0])
    fuel       = sm.get('fuel', 0)
    flag       = sm.get('flag', 'none').lower()
    in_pit     = sm.get('is_in_pit_lane', False)
    best_ms    = sm.get('best_lap_ms', 0)
    current_ms = sm.get('current_lap_ms', 0)
    labels     = ['FL', 'FR', 'RL', 'RR']

    # --- Tyre temperatures ---
    for i, (label, temp) in enumerate(zip(labels, tyre_temp)):
        prev_temp = prev.get(f'tyre_temp_{i}', temp)
        if temp > TYRE_TEMP_HIGH and prev_temp <= TYRE_TEMP_HIGH:
            alerts.append(f"Tyre temp alert: {label} at {temp:.1f}°C, above {TYRE_TEMP_HIGH:.0f}°C limit.")
        elif temp < TYRE_TEMP_LOW and prev_temp >= TYRE_TEMP_LOW:
            alerts.append(f"Tyre temp alert: {label} at {temp:.1f}°C, below {TYRE_TEMP_LOW:.0f}°C minimum.")
        prev[f'tyre_temp_{i}'] = temp

    # --- Tyre wear imbalance ---
    avg_wear = sum(tyre_wear) / 4
    for label, wear in zip(labels, tyre_wear):
        key = f'wear_alerted_{label}'
        if abs(wear - avg_wear) > TYRE_WEAR_DELTA and not prev.get(key):
            vals = ' / '.join(f'{w:.3f}' for w in tyre_wear)
            alerts.append(f"Tyre wear imbalance: FL/FR/RL/RR = {vals}.")
            prev[key] = True
            break  # one alert covers all four corners
        elif abs(wear - avg_wear) <= TYRE_WEAR_DELTA:
            prev[f'wear_alerted_{label}'] = False

    # --- Tyre dirt ---
    for i, (label, dirt) in enumerate(zip(labels, tyre_dirty)):
        key = f'dirt_alerted_{i}'
        if dirt > TYRE_DIRT_HIGH and not prev.get(key):
            alerts.append(f"Tyre dirt: {label} at {dirt:.2f}.")
            prev[key] = True
        elif dirt <= TYRE_DIRT_HIGH:
            prev[key] = False

    # --- Damage ---
    dmg_labels = ['front', 'rear', 'left', 'right', 'centre']
    for i, (part, val) in enumerate(zip(dmg_labels, dmg)):
        key = f'dmg_alerted_{i}'
        if val > 0.0 and not prev.get(key):
            alerts.append(f"Damage detected: {part} at {val:.3f}.")
            prev[key] = True

    # --- Fuel ---
    if fuel < FUEL_LOW_PCT and not prev.get('fuel_alerted'):
        alerts.append(f"Fuel low: {fuel:.1f}%.")
        prev['fuel_alerted'] = True
    elif fuel >= FUEL_LOW_PCT:
        prev['fuel_alerted'] = False

    # --- Flags ---
    reportable_flags = {'yellow', 'blue', 'black', 'checkered'}
    if flag in reportable_flags and flag != prev.get('last_flag'):
        alerts.append(f"{flag.upper()} flag.")
        prev['last_flag'] = flag
    elif flag not in reportable_flags:
        prev['last_flag'] = None

    # --- Pit lane ---
    if in_pit and not prev.get('in_pit'):
        alerts.append("Pit lane entry.")
    elif not in_pit and prev.get('in_pit'):
        alerts.append("Pit lane exit.")
    prev['in_pit'] = in_pit

    # --- Lap time delta ---
    if best_ms > 0 and current_ms > 0:
        delta_s = (current_ms - best_ms) / 1000
        if delta_s > LAP_DELTA_S and not prev.get('lap_delta_alerted'):
            alerts.append(f"Lap time {delta_s:.1f}s off best.")
            prev['lap_delta_alerted'] = True
        elif delta_s <= LAP_DELTA_S:
            prev['lap_delta_alerted'] = False

    # --- Gap to leader (only report meaningful increases) ---
    gap_leader = gaps.get('gap_to_leader_s')
    if gap_leader is not None:
        prev_gap = prev.get('gap_to_leader')
        if prev_gap is not None and (gap_leader - prev_gap) >= 1.0:
            alerts.append(f"Gap to leader {gap_leader:.2f}s.")
        prev['gap_to_leader'] = gap_leader

    return alerts


def _phrase_alert(raw_alert: str) -> str:
    """Ask the LLM to rephrase a pre-validated alert as team radio."""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": _MONITOR_SYSTEM},
                {"role": "user",   "content": raw_alert},
            ],
            "stream": False,
        })
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "").strip()
    except Exception as exc:
        print(f"[Monitor] Ollama request failed: {exc}")
    return raw_alert  # fall back to the raw string if LLM fails


def _telemetry_monitor_loop(conversation_history: list, interval: float = 1.0):
    global _running
    prev = {}  # holds last-seen values and alert-fired flags

    while _running:
        time.sleep(interval)

        sm = get_shared_memory_data()
        if not sm or not sm.get('connected'):
            continue
        if sm.get('status', 0) != 2:
            continue

        gaps = calculate_gaps()
        alerts = _check_alerts(sm, gaps, prev)

        for raw_alert in alerts:
            reply = _phrase_alert(raw_alert)
            if reply:
                print(f"\n[Engineer] {reply}\n", flush=True)
                speak(reply)
                conversation_history.append({
                    "role": "assistant",
                    "content": f"[Proactive observation] {reply}"
                })

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
        default= (
            "You are an experienced F1 race engineer monitoring live telemetry for a driver "
            "in Assetto Corsa. Every second you receive a full telemetry snapshot. "
            "Your job is to call out anything that genuinely warrants driver attention — "
            "examples: significant tyre wear difference across axles, dangerously high temps, "
            "car damage detected, flag changes (yellow/blue/black/checkered), pit-lane entry/exit, "
            "big lap-time delta vs best lap, or fuel running low.\n\n"
            "Rules:\n"
            "- Use concise, calm, assertive statements.\n"
            "- Avoid excitement, encouragement, or emotional language.\n"
            "- If nothing is noteworthy, respond with only the single word SILENT and nothing else.\n"  # <-- tells it what to actually output
            "- Never comment on normal, expected driving data.\n"
            "- One short punchy sentence, like real F1 team radio. No preamble.\n"
            "- Do NOT repeat an observation you already made unless the situation has worsened.\n"
            "- Do NOT hallucinate or invent data not present in the telemetry.\n"          # <-- \n added
            "- Tyre temperatures: SILENT unless at least one tyre is ABOVE 110°C or BELOW 40°C. "
            "The range 40–110°C is completely normal. 71°C is normal. Do not report it.\n"
            "- Never describe values as 'approaching limit' unless an explicit limit is provided in telemetry.\n"
            "- Damage values of 0.0 means NO damage.\n"                                    # <-- \n added
            "- Fuel value is a percentage. Respond SILENT for fuel unless it is below 5%.\n"  # <-- explicit SILENT instruction
            "- Do not speak for anything you have already mentioned unless it has significantly worsened.\n"  # <-- \n added
        ))
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