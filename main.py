from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

APP_NAME = "Brass Studio Analysis API"
MAX_FILE_SIZE = 200 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}

app = FastAPI(title=APP_NAME, version="1.0.0")

configured_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": APP_NAME, "status": "running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_audio(
    audio: Annotated[UploadFile, File(...)],
) -> dict:
    filename = audio.filename or "audio"
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="MP3・WAV・M4Aのみ対応しています",
        )

    temporary_path: str | None = None
    total_size = 0

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temporary_file:
            temporary_path = temporary_file.name

            while True:
                chunk = await audio.read(CHUNK_SIZE)
                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="ファイルは200MB以下にしてください",
                    )

                temporary_file.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="空のファイルは解析できません",
            )

        return {
            "status": "connected",
            "message": "解析サーバーへのアップロードに成功しました",
            "file": {
                "name": filename,
                "extension": extension,
                "sizeBytes": total_size,
            },
            "analysis": {
                "engine": "connection-test",
                "bpm": None,
                "key": None,
                "timeSignature": None,
                "measureCount": None,
                "parts": {},
            },
        }

    finally:
        await audio.close()

        if temporary_path:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
