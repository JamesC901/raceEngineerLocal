import argparse
import queue
import signal
import time
import numpy as np
import whisper

try:
    import sounddevice as sd
except ImportError:
    sd = None

MODEL_NAME = "small"
_running = True


def handle_sigint(sig, frame):
    global _running
    print("\n\nStopping transcription...")
    _running = False


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to float32 in the range [-1.0, 1.0]."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0
    return audio.astype(np.float32)


def is_speech(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """Return True when audio contains enough energy to be likely speech."""
    audio = normalize_audio(audio)
    rms = np.sqrt(np.mean(np.square(audio), axis=-1))
    return rms >= threshold


def transcribe_audio_array(audio: np.ndarray, model_name: str = MODEL_NAME, model=None) -> str:
    """Transcribe raw audio samples using Whisper."""
    audio = normalize_audio(audio)
    if model is None:
        model = whisper.load_model(model_name)
        print(f"Loaded Whisper model: {model_name}")

    result = model.transcribe(audio, language="en", verbose=False, no_speech_threshold=0.6, fp16=False)
    return result["text"].strip()


def transcribe_live_microphone(chunk_duration: float = 2.0, model_name: str = MODEL_NAME) -> None:
    global _running

    if sd is None:
        raise RuntimeError("sounddevice is required. Install it with: pip install sounddevice")

    signal.signal(signal.SIGINT, handle_sigint)

    sample_rate = 16000
    model = whisper.load_model(model_name)
    print(f"Loaded Whisper model: {model_name}")

    chunk_frames = int(sample_rate * chunk_duration)
    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio input status: {status}")
        audio_queue.put(indata.copy())

    print("Listening... Press Ctrl+C to stop.\n")
    chunk_index = 1

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", blocksize=chunk_frames, callback=callback):
        while _running:
            try:
                data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                # print("[debug] queue empty, waiting...")
                continue

            audio = data.flatten()
            rms = np.sqrt(np.mean(np.square(normalize_audio(audio))))
            # print(f"[debug] got chunk, RMS={rms:.4f}, samples={len(audio)}")

            if not is_speech(audio, threshold=0.005):
                # print(f"[debug] skipped (below threshold)")
                continue

            # print(f"[debug] Transcribing...")
            text = transcribe_audio_array(audio, model=model)
            # print(f"[debug] raw result: '{text}'")

            if not text:
                continue

            print(f"[{chunk_index}] {text}")
            chunk_index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper continuous microphone speech-to-text")
    parser.add_argument("--chunk-duration", type=float, default=2.0, help="Chunk size in seconds")
    parser.add_argument("--model", default=MODEL_NAME, help="Whisper model (tiny, base, small, medium, large)")
    args = parser.parse_args()

    transcribe_live_microphone(chunk_duration=args.chunk_duration, model_name=args.model)


if __name__ == "__main__":
    main()