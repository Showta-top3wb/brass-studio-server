from pathlib import Path
import shutil
import tempfile

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Brass Studio 2.0 Phase 2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}
ANALYSIS_SECONDS = 30


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "phase": "2-1",
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    temp_path = None

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="MP3、WAV、M4Aファイルのみ対応しています。",
        )

    try:
        print(f"Upload started: {filename}", flush=True)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        print(f"File saved: {temp_path}", flush=True)

        audio_info = sf.info(temp_path)
        duration = float(audio_info.duration)
        sample_rate = int(audio_info.samplerate)

        print(
            f"Audio info: duration={duration}, sample_rate={sample_rate}",
            flush=True,
        )

        frames_to_read = min(
            int(audio_info.frames),
            sample_rate * ANALYSIS_SECONDS,
        )

        print(
            f"Reading first {ANALYSIS_SECONDS} seconds with soundfile",
            flush=True,
        )

        audio, sample_rate = sf.read(
            temp_path,
            frames=frames_to_read,
            dtype="float32",
            always_2d=True,
        )

        print(f"Audio loaded: shape={audio.shape}", flush=True)

        if audio.size == 0:
            raise ValueError("音声データが空です。")

        audio = np.mean(audio, axis=1)

        peak = float(np.max(np.abs(audio)))

        if peak > 0:
            audio = audio / peak

        print("Starting BPM analysis", flush=True)

        onset_envelope = librosa.onset.onset_strength(
            y=audio,
            sr=sample_rate,
            hop_length=512,
        )

        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=512,
        )

        tempo_values = np.asarray(tempo).reshape(-1)
        bpm = float(tempo_values[0]) if tempo_values.size else 0.0

        if not np.isfinite(bpm) or bpm <= 0:
            bpm = 0.0

        print(f"BPM detected: {bpm}", flush=True)
        print("Upload completed", flush=True)

        return {
            "status": "success",
            "filename": filename,
            "format": extension.lstrip("."),
            "duration": round(duration, 3),
            "bpm": round(bpm, 1),
        }

    except Exception as error:
        print(f"Upload error: {repr(error)}", flush=True)

        raise HTTPException(
            status_code=422,
            detail=f"音声ファイルを解析できませんでした: {error}",
        ) from error

    finally:
        await file.close()

        if temp_path:
            Path(temp_path).unlink(missing_ok=True)