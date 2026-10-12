import yt_dlp
import os
import sys
import shutil
from pathlib import Path

DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def add_runtime_to_path(executable: str, search_root: Path) -> None:
    if shutil.which(executable):
        return

    matches = list(search_root.glob(f"**/{executable}.exe"))
    if matches:
        os.environ["PATH"] = (
            str(matches[0].parent) + os.pathsep + os.environ["PATH"]
        )


def find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    winget_dir = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Microsoft"
        / "WinGet"
        / "Packages"
    )
    matches = list(winget_dir.glob("**/ffmpeg.exe"))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(
        "FFmpeg was not found. Install FFmpeg before converting audio."
    )


def download_audio_from_youtube(url: str) -> str:
    add_runtime_to_path(
        "deno",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
    )
    ffmpeg_path = find_ffmpeg()
    output_path = str(DOWNLOAD_DIR / "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "ffmpeg_location": str(Path(ffmpeg_path).parent),
        "cookiefile": str(Path(__file__).resolve().parents[2] / "cookies.txt"),
        "extractor_retries": 3,
        "retries": 5,
        "fragment_retries": 5,
        "sleep_interval_requests": 1,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = str(Path(ydl.prepare_filename(info)).with_suffix(".wav"))
    return filename


def convert_to_wav(input_path: str) -> str:
    ffmpeg_path = find_ffmpeg()
    os.environ["PATH"] = str(Path(ffmpeg_path).parent) + os.pathsep + os.environ["PATH"]
    from pydub import AudioSegment

    AudioSegment.converter = ffmpeg_path
    output_path = str(Path(input_path).with_suffix("")) + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path



   
def chunk_audio(wav_path: str, chunk_length_minutes: float = 10) -> list[str]:
    from pydub import AudioSegment

    source_path = Path(wav_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {source_path}")

    audio = AudioSegment.from_wav(str(source_path))
    chunk_ms = int(chunk_length_minutes * 60 * 1000)
    chunks: list[str] = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = source_path.with_name(
            f"{source_path.stem}_chunk_{i}.wav"
        )
        chunk.export(str(chunk_path), format="wav")
        if not chunk_path.is_file():
            raise RuntimeError(f"Failed to create audio chunk: {chunk_path}")
        chunks.append(str(chunk_path))
    return chunks

def process_input(source: str, chunk_length_minutes: float = 10) -> list[str]:
    if source.startswith("http://") or source.startswith("https://"):
        print("Downloading audio from YouTube...")
        wav_path = download_audio_from_youtube(source)
    else:
        print("Converting audio to WAV...")
        wav_path = convert_to_wav(source)


    print("Chunking audio...")

    chunks = chunk_audio(wav_path, chunk_length_minutes)
    print(f"Audio has been chunked into {len(chunks)} parts.")
    return chunks   
    
    
    
