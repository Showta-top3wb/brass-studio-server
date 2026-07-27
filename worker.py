from pathlib import Path
import shutil
import tempfile
import uuid

from tasks import copy_separated_stems, separate_audio


WORK_DIRECTORY = Path(
    tempfile.gettempdir()
) / "brass-studio-worker"

RESULT_DIRECTORY = Path(
    tempfile.gettempdir()
) / "brass-studio-results"


def process_audio_job(
    input_path: str | Path,
    job_id: str | None = None,
) -> dict[str, object]:
    source_path = Path(input_path).resolve()
    current_job_id = job_id or uuid.uuid4().hex

    job_work_directory = WORK_DIRECTORY / current_job_id
    job_result_directory = RESULT_DIRECTORY / current_job_id

    job_work_directory.mkdir(parents=True, exist_ok=True)
    job_result_directory.mkdir(parents=True, exist_ok=True)

    print(
        f"Worker job started: {current_job_id}",
        flush=True,
    )

    try:
        stems = separate_audio(
            input_path=source_path,
            output_directory=job_work_directory,
        )

        result_files = copy_separated_stems(
            stems=stems,
            destination_directory=job_result_directory,
        )

        print(
            f"Worker job completed: {current_job_id}",
            flush=True,
        )

        return {
            "status": "completed",
            "job_id": current_job_id,
            "files": {
                stem_name: str(file_path)
                for stem_name, file_path in result_files.items()
            },
        }

    except Exception as error:
        print(
            f"Worker job failed: {current_job_id}: {error!r}",
            flush=True,
        )

        return {
            "status": "failed",
            "job_id": current_job_id,
            "error": str(error),
        }

    finally:
        shutil.rmtree(
            job_work_directory,
            ignore_errors=True,
        )