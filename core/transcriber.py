import os
import tempfile
from pathlib import Path

import requests
import whisper
from dotenv import load_dotenv
from pydub import AudioSegment

from utils.audio_pocessor import find_ffmpeg

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_PIECE_SECONDS = 25

_whisper_model = None


def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        ffmpeg_dir = str(Path(find_ffmpeg()).parent)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully.")
    return _whisper_model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    audio_path = Path(chunk_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Transcription chunk not found: {audio_path}")

    result = load_whisper_model().transcribe(str(audio_path), task="transcribe")
    return result["text"].strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Split a chunk into pieces accepted by Sarvam's synchronous API.
    """
    audio_path = Path(chunk_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Transcription chunk not found: {audio_path}")
    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is missing. Add it to the project .env file."
        )

    audio = AudioSegment.from_wav(str(audio_path))
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms
    transcripts: list[str] = []
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with tempfile.TemporaryDirectory(prefix="sarvam_") as temp_dir:
        for i, start in enumerate(range(0, len(audio), piece_ms)):
            piece = audio[start : start + piece_ms]
            piece_path = Path(temp_dir) / f"{audio_path.stem}_sv_{i}.wav"
            piece.export(str(piece_path), format="wav")
            print(f"  -> Sarvam piece {i + 1}/{total_pieces}")

            payload = {
                "model": "saaras:v3",
                "mode": "translate",
                "language_code": "unknown",
            }
            with piece_path.open("rb") as audio_file:
                response = requests.post(
                    SARVAM_STT_URL,
                    headers=headers,
                    data=payload,
                    files={"file": (piece_path.name, audio_file, "audio/wav")},
                    timeout=300,
                )

            if not response.ok:
                raise RuntimeError(
                    f"Sarvam transcription failed for piece {i + 1} "
                    f"({response.status_code}): {response.text[:500]}"
                )

            transcript = response.json().get("transcript")
            if not isinstance(transcript, str):
                raise RuntimeError(
                    f"Sarvam response for piece {i + 1} did not contain "
                    "a transcript."
                )
            if transcript.strip():
                transcripts.append(transcript.strip())

    return " ".join(transcripts)


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english -> Whisper (local model)
    - hinglish -> Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list[str], language: str = "english") -> str:
    transcriptions = []
    for i, chunk in enumerate(chunks):
        print(
            f"Transcribing chunk {i + 1}/{len(chunks)} "
            f"using {language.lower()}..."
        )
        transcriptions.append(transcribe_chunk(chunk, language=language))
    return " ".join(text for text in transcriptions if text)
