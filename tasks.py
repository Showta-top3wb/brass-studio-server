from pathlib import Path
import shutil
import subprocess
import sys


DEMUCS_MODEL = "htdemucs_ft"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a"}


class AudioSeparationError(RuntimeError):
    pass


def separate_audio(
    input_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    source_path = Path(input_path).resolve()
    destination_path = Path(output_directory).resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"音源が見つかりません: {source_path}")

    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("MP3、WAV、M4Aファイルのみ対応しています。")

    destination_path.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "demucs",
        "--name",
        DEMUCS_MODEL,
        "--out",
        str(destination_path),
        str(source_path),
    ]

    print(f"Demucs started: {' '.join(command)}", flush=True)

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if process.stdout:
        print(process.stdout, flush=True)

    if process.stderr:
        print(process.stderr, flush=True)

    if process.returncode != 0:
        raise AudioSeparationError(
            process.stderr.strip()
            or process.stdout.strip()
            or "Demucsの音源分離に失敗しました。"
        )

    result_directory = (
        destination_path
        / DEMUCS_MODEL
        / source_path.stem
    )

    if not result_directory.exists():
        raise AudioSeparationError(
            "Demucsの出力フォルダが見つかりません。"
        )

    expected_stems = {
        "vocals": result_directory / "vocals.wav",
        "drums": result_directory / "drums.wav",
        "bass": result_directory / "bass.wav",
        "other": result_directory / "other.wav",
    }

    missing_stems = [
        name
        for name, path in expected_stems.items()
        if not path.exists()
    ]

    if missing_stems:
        raise AudioSeparationError(
            "分離ファイルが不足しています: "
            + ", ".join(missing_stems)
        )

    return expected_stems


def copy_separated_stems(
    stems: dict[str, Path],
    destination_directory: str | Path,
) -> dict[str, Path]:
    destination_path = Path(destination_directory).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)

    copied_files: dict[str, Path] = {}

    for stem_name, source_path in stems.items():
        destination_file = destination_path / f"{stem_name}.wav"
        shutil.copy2(source_path, destination_file)
        copied_files[stem_name] = destination_file

    return copied_files