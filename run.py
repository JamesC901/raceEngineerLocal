import argparse
import queue
import signal
import numpy as np
import whisper
import requests
import json
import os
from dotenv import load_dotenv
from game_server import start_game_server, get_game_data, get_handshake_data, get_lap_data
from ac_shared_memory import get_shared_memory_data
from gap_calculator import calculate_gaps


try:
    import sounddevice as sd
except ImportError:
    sd = None

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL")
_running = True

# Whisper commonly hallucinates these on silence/noise
HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching",
    "please subscribe", "you", ".", "..", "...", "bye", "bye bye",
    "subtitles by", "transcribed by",
}


def handle_sigint(sig, frame):
    global _running
    print("\n\nStopping transcription...")
    _running = False

def format_lap_time(time):
    if time > 0:
        minutes = time // 60000
        seconds = (time % 60000) / 1000
        return f" {minutes}:{seconds:06.3f}"
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


def format_telemetry(game_data: dict) -> str:
    """Format telemetry data into readable text for the race engineer."""
    if not game_data:
        return ""
    
    telemetry_text = "\n[Current Car Telemetry]:\n"
    telemetry_text += f"  Speed: {game_data.get('speed_kmh', 0):.1f} km/h | RPM: {game_data.get('engine_rpm', 0):.0f}\n"
    telemetry_text += f"  Gear: {game_data.get('gear', 0)} | Throttle: {game_data.get('gas', 0)*100:.1f}% | Brake: {game_data.get('brake', 0)*100:.1f}%\n"
    
    if game_data.get('is_tc_in_action'):
        telemetry_text += f"  TC IS ACTIVE\n"
    if game_data.get('is_abs_in_action'):
        telemetry_text += f"  ABS IS ACTIVE\n"
    if game_data.get('is_engine_limiter_on'):
        telemetry_text += f"  ENGINE LIMITER ON\n"
    
    # lap_time = game_data.get('lap_time', 0)
    # if lap_time > 0:
    #     minutes = lap_time // 60000
    #     seconds = (lap_time % 60000) / 1000
    #     telemetry_text += f"  Current Lap: {minutes}:{seconds:06.3f}\n"

    lap_time = game_data.get('lap_time', 0)
    telemetry_text += f"  Current Lap Time: " + format_lap_time(lap_time)

    best_lap = game_data.get('best_lap', 0)
    telemetry_text += f"  Best Lap Time: " + format_lap_time(best_lap)

    last_lap = game_data.get('last_lap', 0)
    telemetry_text += f"  Last Lap Time: " + format_lap_time(last_lap)

    
    telemetry_text += f"  Lap Count: {game_data.get('lap_count', 0)}\n"
    
    
    #Grab shared memory data
    sm=get_shared_memory_data()
    print("Shared memory: ", sm)
    
    #Track
    telemetry_text += f"  Race Track: {sm.get('track', 'None')}\n"
    #Car Model
    telemetry_text += f"  Car Model: {sm.get('car_model', 'None')}\n"
    #Number of Cars
    telemetry_text += f"  Number of Cars: {sm.get('num_cars', 'None')}\n"
    #Tyre Compound
    telemetry_text += f"  Current Tyre Compound: {sm.get('tire_compound', 'None')}\n"
    #Fuel level:
    telemetry_text += f"  Fuel level: {sm.get('fuel', 'None')}\n"
    #Flag Data
    telemetry_text += f"  Flag: {sm.get('flag', 'None')}\n"
    #Position
    telemetry_text += f"  Race Position: {sm.get('position', 'None')}\n"
    #Pit lane
    telemetry_text +=  f"  Is Driver in Pit Lane?: {sm.get('is_in_pit_lane', 'No')}\n"


    # Tyre wear — FL, FR, RL, RR
    tyre_wear = sm.get('tyre_wear', [0, 0, 0, 0])
    telemetry_text += (
        f"  Tyre Wear (FL/FR/RL/RR): "
        f"{tyre_wear[0]:.3f} / {tyre_wear[1]:.3f} / "
        f"{tyre_wear[2]:.3f} / {tyre_wear[3]:.3f}\n"
    )

    # Tyre temps — FL, FR, RL, RR
    tyre_core_temp = sm.get('tyre_core_temp', [0, 0, 0, 0])
    telemetry_text += (
        f"  Tyre Temps (FL/FR/RL/RR): "
        f"{tyre_core_temp[0]:.3f} / {tyre_core_temp[1]:.3f} / "
        f"{tyre_core_temp[2]:.3f} / {tyre_core_temp[3]:.3f}\n"
    )

    # Tyre Dirtyness — FL, FR, RL, RR
    tyre_dirty = sm.get('tyre_dirty', [0, 0, 0, 0])
    telemetry_text += (
        f"  Tyre Dirtiness (FL/FR/RL/RR): "
        f"{tyre_dirty[0]:.3f} / {tyre_dirty[1]:.3f} / "
        f"{tyre_dirty[2]:.3f} / {tyre_dirty[3]:.3f}\n"
    )

    # Car damage — front, rear, left, right, centre
    dmg = sm.get('car_damage', [0, 0, 0, 0, 0])
    telemetry_text += (
        f"  Car Damage (Front/Rear/Left/Right/Centre): "
        f"{dmg[0]:.3f} / {dmg[1]:.3f} / {dmg[2]:.3f} / "
        f"{dmg[3]:.3f} / {dmg[4]:.3f}\n"
    )

    #Gaps
    gaps = calculate_gaps()
    telemetry_text +=  f"  Gap (seconds) between the player car and the car in front of it: {gaps.get('gap_ahead_s', 'N/A')}\n"
    telemetry_text +=  f"  Gap (seconds) between the player car and the car behind of it: {gaps.get('gap_behind_s', 'N/A')}\n"
    telemetry_text +=  f"  Gap (seconds) between the player car and the car leading the race: {gaps.get('gap_to_leader_s', 'N/A')}\n"



    return telemetry_text
def format_handshake_data(handshake_data: dict) -> str:
    if not handshake_data:
        return ""

    text = "\n[Session Info]:\n"
    text += f"  Driver: {handshake_data.get('driver_name', 'Unknown')}\n"
    text += f"  Car: {handshake_data.get('car_name', 'Unknown')}\n"
    text += f"  Track: {handshake_data.get('track_name', 'Unknown')}\n"

    track_config = handshake_data.get('track_config')
    if track_config:
        text += f"  Track Config: {track_config}\n"

    text += f"  Session ID: {handshake_data.get('identifier', 0)}\n"
    text += f"  Protocol Version: {handshake_data.get('version', 0)}\n"

    return text


def format_lap_data(lap_data: dict) -> str:
    if not lap_data:
        return ""

    text = "\n[Last Lap Info]:\n"
    text += f"  Driver: {lap_data.get('driver_name', 'Unknown')}\n"
    text += f"  Car: {lap_data.get('car_name', 'Unknown')}\n"
    text += f"  Lap Number: {lap_data.get('lap', 0)}\n"

    lap_time = lap_data.get('time', 0)
    if lap_time > 0:
        minutes = lap_time // 60000
        seconds = (lap_time % 60000) / 1000
        text += f"  Lap Time: {minutes}:{seconds:06.3f}\n"

    return text

def ask_ollama(text: str, conversation_history: list, ollama_model: str = OLLAMA_MODEL) -> str:
    # Enhance message with game data if available
    game_data = get_game_data()
    handshake_data = get_handshake_data()
    lap_data = get_lap_data()
    
    message_content = text
    if game_data:
        telemetry_text = format_telemetry(game_data)
        lap_text = format_lap_data(lap_data)

        if telemetry_text:
            message_content = text + "\n\n" + telemetry_text + lap_text
        print(message_content)

    conversation_history.append({"role": "user", "content": message_content})
    print("Sending request to Ollama...")

    response = requests.post(OLLAMA_URL, json={
        "model": ollama_model,
        "messages": conversation_history,
        "stream": True,
    }, stream=True)

    if response.status_code != 200:
        return f"[Ollama error {response.status_code}]"

    full_reply = ""
    print(f"\nOllama: ", end="", flush=True)
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        print(token, end="", flush=True)
        full_reply += token
        if chunk.get("done"):
            break

    print()
    conversation_history.append({"role": "assistant", "content": full_reply})
    return full_reply


def transcribe_live_microphone(
    chunk_duration: float = 0.5,
    silence_duration: float = 1.2,
    model_name: str = MODEL_NAME,
    ollama_model: str = OLLAMA_MODEL,
    system_prompt: str = None,
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

    sample_rate = 16000
    model = whisper.load_model(model_name)
    print(f"Loaded Whisper model: {model_name}")
    print(f"Connected to Ollama model: {ollama_model}")

    chunk_frames = int(sample_rate * chunk_duration)
    audio_queue = queue.Queue()
    conversation_history = []
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio input status: {status}")
        audio_queue.put(indata.copy())

    print("Listening... Press Ctrl+C to stop.\n")

    utterance_index = 1
    speech_buffer = []       # accumulates chunks during an utterance
    silent_chunks = 0        # consecutive silent chunks after speech
    max_silent_chunks = int(silence_duration / chunk_duration)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", blocksize=chunk_frames, callback=callback):
        while _running:
            try:
                data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            audio = data.flatten()
            speech_detected = is_speech(audio)

            if speech_detected:
                speech_buffer.append(audio)
                silent_chunks = 0
            elif speech_buffer:
                # We're in silence after speech — count down
                speech_buffer.append(audio)  # include trailing silence for natural endings
                silent_chunks += 1

                if silent_chunks >= max_silent_chunks:
                    # Silence threshold reached — transcribe the whole utterance
                    full_audio = np.concatenate(speech_buffer)
                    speech_buffer = []
                    silent_chunks = 0

                    text = transcribe_audio_array(full_audio, model=model)

                    if not text or is_hallucination(text):
                        continue

                    print(f"\nYou [{utterance_index}]: {text}")
                    ask_ollama(text, conversation_history, ollama_model)
                    utterance_index += 1


def main() -> None:
    # Start game data server in background
    server_thread = start_game_server()
    if server_thread is None:
        print("[Warning] Game server did not start (port may be in use). Continuing without telemetry.")
    
    parser = argparse.ArgumentParser(description="Whisper + Ollama voice assistant")
    parser.add_argument("--chunk-duration", type=float, default=0.5, help="Audio polling interval in seconds")
    parser.add_argument("--silence-duration", type=float, default=1.2, help="Seconds of silence before transcribing")
    parser.add_argument("--whisper-model", default=MODEL_NAME, help="Whisper model (tiny, base, small, medium, large)")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL, help="Ollama model name (e.g. llama3.1, mistral)")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are an F1 race engineer providing expert technical advice about "
            "vehicle performance, setup, and racing strategy. "
            "You are communicating with a driver currently racing in Assetto Corsa. "
            "You will receive telemetry data in the following format:\n\n"
            "{\n"
            "    'speed_kmh': speed_kmh,\n"
            "    'engine_rpm': engine_rpm,\n"
            "    'gear': gear,\n"
            "    'gas': gas,\n"
            "    'brake': brake,\n"
            "    'lap_time': lap_time,\n"
            "    'lap_count': lap_count,\n"
            "    'is_in_pit': bool(in_pit),\n"
            "    'is_abs_in_action': bool(abs_in_action),\n"
            "    'is_tc_in_action': bool(tc_in_action),\n"
            "    'is_engine_limiter_on': bool(engine_limiter)\n"
            "}\n\n"
            "Keep responses concise and realistic, similar to a real F1 race engineer "
            "communicating over team radio. Focus only on relevant in-game telemetry "
            "and driving information.Only use data from the telemetry data passed to you, don't hallucinate any data or try to roleplay. If you have no data, then say you have no data to make any responses. If the confidence in the driver's message is low "
            "or unclear, ask them to repeat it."
        ),
        help="System prompt to set the model's role and behavior."
    )
    args = parser.parse_args()

    transcribe_live_microphone(
        chunk_duration=args.chunk_duration,
        silence_duration=args.silence_duration,
        model_name=args.whisper_model,
        ollama_model=args.ollama_model,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()