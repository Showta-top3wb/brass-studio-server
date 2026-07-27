from pathlib import Path
import shutil
import tempfile
import subprocess
import sys

import librosa
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Brass Studio 2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "phase": "2-preparation",
    }
@app.get("/demucs-health")
async def demucs_health():
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "demucs",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        if process.returncode != 0:
            raise RuntimeError(
                process.stderr.strip()
                or process.stdout.strip()
                or "Demucsを起動できませんでした。"
            )

        return {
            "status": "ok",
            "demucs": "available",
        }

    except subprocess.TimeoutExpired as error:
        raise HTTPException(
            status_code=504,
            detail="Demucsの起動確認がタイムアウトしました。",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Demucsを起動できませんでした: {error}",
        ) from error

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="MP3、WAV、M4Aファイルのみ対応しています。",
        )

    temp_path = None

    try:
        print(f"Upload started: {filename}", flush=True)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        print(f"File saved: {temp_path}", flush=True)

        try:
            audio_info = sf.info(temp_path)
            duration = float(audio_info.duration)
            print("Duration read with soundfile", flush=True)
        except Exception:
            duration = float(librosa.get_duration(path=temp_path))
            print("Duration read with librosa", flush=True)

        print("Upload completed", flush=True)

        return {
            "status": "success",
            "filename": filename,
            "format": extension.lstrip("."),
            "duration": round(duration, 3),
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