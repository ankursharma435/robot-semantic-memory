"""
riva_voice.py

Voice in/out via NVIDIA's hosted Riva Speech NIM (ASR + TTS), reached over
gRPC rather than plain REST — this is Riva's actual documented protocol,
not a simplification. Needs three things from your .env:
  NVIDIA_API_KEY        (same key as nvidia_nim.py)
  RIVA_ASR_FUNCTION_ID   (model-specific — copy from your chosen ASR
                          model's "API" tab on build.nvidia.com)
  RIVA_TTS_FUNCTION_ID   (same, for your chosen TTS model)

Records a fixed-length clip (simplest, most demo-reliable option — no
push-to-talk edge cases to debug live) rather than streaming. Good enough
for a controlled demo; streaming is a real upgrade but not needed here.

Untested against a live key (written without network access) — run this
file directly first, in isolation, before wiring it into demo_live.py.
"""

from __future__ import annotations
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
ASR_FUNCTION_ID = os.environ.get("RIVA_ASR_FUNCTION_ID", "")
TTS_FUNCTION_ID = os.environ.get("RIVA_TTS_FUNCTION_ID", "")
RIVA_SERVER = os.environ.get("RIVA_GRPC_SERVER", "grpc.nvcf.nvidia.com:443")
TTS_VOICE = os.environ.get("RIVA_TTS_VOICE", "Magpie-Multilingual.EN-US.Aria")

SAMPLE_RATE = 16000  # matches what most hosted Riva ASR models expect


def _require_config():
    missing = [n for n, v in [
        ("NVIDIA_API_KEY", API_KEY),
        ("RIVA_ASR_FUNCTION_ID", ASR_FUNCTION_ID),
        ("RIVA_TTS_FUNCTION_ID", TTS_FUNCTION_ID),
    ] if not v]
    if missing:
        raise RuntimeError(f"Missing from .env: {', '.join(missing)}")


def _auth(function_id: str):
    import riva.client
    return riva.client.Auth(
        uri=RIVA_SERVER,
        use_ssl=True,
        metadata_args=[
            ["function-id", function_id],
            ["authorization", f"Bearer {API_KEY}"],
        ],
    )


def record_clip(seconds: float = 4.0) -> np.ndarray:
    """Record `seconds` of mono audio from the default mic at 16kHz."""
    print(f"Recording for {seconds:.1f}s... speak now.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    print("Done recording.")
    return audio.flatten()


CHUNK_BYTES = 3200  # 1600 int16 samples = 100ms at 16kHz


def transcribe(audio: np.ndarray) -> tuple[str, float]:
    """Send a recorded clip to hosted ASR, return (text, latency_ms).

    Changed 2026-09-03: this used offline_recognize(), which this function
    can never serve. GetRivaSpeechRecognitionConfig on
    RIVA_ASR_FUNCTION_ID reports model `parakeet-0.6b-en-US-asr-streaming`
    with `offline = False, type = online, streaming = True`, so every
    offline call came back INVALID_ARGUMENT ("Unavailable model requested
    given these parameters: ... type=offline"). We now feed the same clip
    through the streaming API in 100ms chunks and concatenate the final
    results — measured at ~130ms for a 1.3s clip.

    Two parameters are NOT free to change here: language_code must be
    exactly "en-US" (plain "en" is rejected) and sample_rate_hertz must be
    16000 — those are the only values this function advertises.
    """
    _require_config()
    import riva.client

    auth = _auth(ASR_FUNCTION_ID)
    asr_service = riva.client.ASRService(auth)

    streaming_config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
        ),
        interim_results=False,
    )

    audio_bytes = audio.astype(np.int16).tobytes()
    chunks = [audio_bytes[i:i + CHUNK_BYTES]
              for i in range(0, len(audio_bytes), CHUNK_BYTES)]

    t0 = time.perf_counter()
    pieces = []
    for response in asr_service.streaming_response_generator(
            audio_chunks=chunks, streaming_config=streaming_config):
        for result in response.results:
            if result.is_final and result.alternatives:
                pieces.append(result.alternatives[0].transcript)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return "".join(pieces).strip(), elapsed_ms


def speak(text: str, play: bool = True) -> tuple[np.ndarray, float]:
    """Synthesize `text` via hosted TTS, optionally play it, return
    (audio_samples, latency_ms).

    Voice comes from RIVA_TTS_VOICE in .env — Magpie uses named voices
    (e.g. Magpie-Multilingual.EN-US.Aria), not a single default string.
    Run --list-voices (see .env.example for the exact command) against
    your TTS function-id to confirm the exact name before relying on it.
    """
    _require_config()
    import riva.client

    auth = _auth(TTS_FUNCTION_ID)
    tts_service = riva.client.SpeechSynthesisService(auth)

    t0 = time.perf_counter()
    response = tts_service.synthesize(
        text=text,
        voice_name=TTS_VOICE,
        language_code="en-US",
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hz=SAMPLE_RATE,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    audio = np.frombuffer(response.audio, dtype=np.int16)
    if play:
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()
    return audio, elapsed_ms


if __name__ == "__main__":
    # Standalone smoke test. Run this file directly first.
    print("=== ASR test ===")
    try:
        clip = record_clip(4.0)
        text, ms = transcribe(clip)
        print(f"  transcribed in {ms:.1f}ms: '{text}'")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("=== TTS test ===")
    try:
        _, ms = speak("Hello. This is a test of the text to speech system.")
        print(f"  synthesized + played in {ms:.1f}ms")
    except Exception as e:
        print(f"  FAILED: {e}")
