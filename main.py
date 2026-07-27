from pathlib import Path
import importlib.util
import shutil
import tempfile
import uuid

import librosa
import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from worker import process_audio_job


app = FastAPI(title="Brass Studio 2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}

JOB_INPUT_DIRECTORY = (
    Path(tempfile.gettempdir()) / "brass-studio-job-inputs"
)

JOB_INPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

jobs: dict[str, dict[str, object]] = {}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "phase": "demucs-test",
    }


@app.get("/demucs-health")
async def demucs_health():
    demucs_available = (
        importlib.util.find_spec("demucs") is not None
    )

    torch_available = (
        importlib.util.find_spec("torch") is not None
    )

    torchaudio_available = (
        importlib.util.find_spec("torchaudio") is not None
    )

    if not all(
        [
            demucs_available,
            torch_available,
            torchaudio_available,
        ]
    ):
        raise HTTPException(
            status_code=500,
            detail={
                "demucs": demucs_available,
                "torch": torch_available,
                "torchaudio": torchaudio_available,
            },
        )

    return {
        "status": "ok",
        "demucs": "available",
        "torch": "available",
        "torchaudio": "available",
    }


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
        print(
            f"Upload started: {filename}",
            flush=True,
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:
            temp_path = temp_file.name

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

        try:
            audio_info = sf.info(temp_path)
            duration = float(audio_info.duration)

        except Exception:
            duration = float(
                librosa.get_duration(
                    path=temp_path,
                )
            )

        return {
            "status": "success",
            "filename": filename,
            "format": extension.lstrip("."),
            "duration": round(duration, 3),
        }

    except Exception as error:
        print(
            f"Upload error: {error!r}",
            flush=True,
        )

        raise HTTPException(
            status_code=422,
            detail=(
                "音声ファイルを解析できませんでした: "
                f"{error}"
            ),
        ) from error

    finally:
        await file.close()

        if temp_path:
            Path(temp_path).unlink(
                missing_ok=True,
            )


@app.post("/separate")
async def separate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="MP3、WAV、M4Aファイルのみ対応しています。",
        )

    job_id = uuid.uuid4().hex

    job_directory = (
        JOB_INPUT_DIRECTORY / job_id
    )

    job_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = (
        job_directory / f"input{extension}"
    )

    try:
        with input_path.open("wb") as output_file:
            shutil.copyfileobj(
                file.file,
                output_file,
            )

    except Exception as error:
        shutil.rmtree(
            job_directory,
            ignore_errors=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"音源の保存に失敗しました: {error}",
        ) from error

    finally:
        await file.close()

    jobs[job_id] = {
        "status": "queued",
        "job_id": job_id,
        "filename": filename,
        "error": None,
        "files": {},
    }

    background_tasks.add_task(
        run_separation_job,
        job_id,
        input_path,
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "message": "音源分離を開始しました。",
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="ジョブが見つかりません。",
        )

    return job


@app.get(
    "/jobs/{job_id}/download/{stem_name}"
)
async def download_stem(
    job_id: str,
    stem_name: str,
):
    allowed_stems = {
        "vocals",
        "drums",
        "bass",
        "other",
    }

    if stem_name not in allowed_stems:
        raise HTTPException(
            status_code=400,
            detail="不正なステム名です。",
        )

    job = jobs.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="ジョブが見つかりません。",
        )

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="音源分離が完了していません。",
        )

    files = job.get("files", {})
    file_path_value = files.get(stem_name)

    if not file_path_value:
        raise HTTPException(
            status_code=404,
            detail="分離ファイルが見つかりません。",
        )

    file_path = Path(str(file_path_value))

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="分離ファイルが削除されています。",
        )

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=f"{stem_name}.wav",
    )


def run_separation_job(
    job_id: str,
    input_path: Path,
):
    jobs[job_id]["status"] = "processing"

    print(
        f"Separation job started: {job_id}",
        flush=True,
    )

    try:
        result = process_audio_job(
            input_path=input_path,
            job_id=job_id,
        )

        if result.get("status") == "completed":
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["files"] = result.get(
                "files",
                {},
            )

        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = result.get(
                "error",
                "音源分離に失敗しました。",
            )

    except Exception as error:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(error)

        print(
            f"Separation job failed: {job_id}: {error!r}",
            flush=True,
        )

    finally:
        input_directory = input_path.parent

        shutil.rmtree(
            input_directory,
            ignore_errors=True,
        )

        print(
            f"Separation job finished: {job_id}",
            flush=True,
        )