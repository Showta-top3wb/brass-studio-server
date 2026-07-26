from pathlib import Path
import shutil
import tempfile

import librosa
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Brass Studio 2.0 Phase1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}
ALLOWED_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/x-m4a",
    "application/octet-stream",
}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, object]:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="MP3、WAV、M4Aファイルのみアップロードできます。",
        )

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="対応していない音声形式です。",
        )

    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        audio, sample_rate = librosa.load(
            temp_path,
            sr=None,
            mono=True,
        )

        duration = float(
            librosa.get_duration(
                y=audio,
                sr=sample_rate,
            )
        )

        return {
            "status": "success",
            "filename": filename,
            "content_type": file.content_type,
            "format": extension.lstrip("."),
            "duration": round(duration, 3),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"音声ファイルを解析できませんでした: {exc}",
        ) from exc
    finally:
        await file.close()

        if temp_path:
            Path(temp_path).unlink(missing_ok=True)