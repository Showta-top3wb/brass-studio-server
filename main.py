from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


APP_NAME = "Brass Studio Analysis API"
APP_VERSION = "1.2.0"

MAX_FILE_SIZE = 200 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
}

TARGET_SAMPLE_RATE = 22050
MAX_ANALYSIS_SECONDS = 300


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
)


configured_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "*",
    ).split(",")
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins,
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=["*"],
)


MAJOR_PROFILE = np.array(
    [
        6.35,
        2.23,
        3.48,
        2.33,
        4.38,
        4.09,
        2.52,
        5.19,
        2.39,
        3.66,
        2.29,
        2.88,
    ],
    dtype=np.float64,
)


MINOR_PROFILE = np.array(
    [
        6.33,
        2.68,
        3.52,
        5.38,
        2.60,
        3.53,
        2.54,
        4.75,
        3.98,
        2.69,
        3.34,
        3.17,
    ],
    dtype=np.float64,
)


KEY_NAMES = [
    "C",
    "C♯",
    "D",
    "E♭",
    "E",
    "F",
    "F♯",
    "G",
    "A♭",
    "A",
    "B♭",
    "B",
]


FIFTHS_BY_ROOT = {
    0: 0,
    1: 7,
    2: 2,
    3: -3,
    4: 4,
    5: -1,
    6: 6,
    7: 1,
    8: -4,
    9: 3,
    10: -2,
    11: 5,
}


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "version": APP_VERSION,
    }


@app.post("/analyze")
async def analyze_audio(
    audio: Annotated[
        UploadFile,
        File(...),
    ],
) -> dict[str, Any]:
    filename = audio.filename or "audio"
    extension = Path(filename).suffix.lower()

    validate_extension(extension)

    temporary_path: str | None = None

    try:
        temporary_path, total_size = (
            await save_upload_to_temporary_file(
                audio=audio,
                extension=extension,
            )
        )

        analysis = analyze_file(
            file_path=temporary_path,
        )

        return {
            "status": "completed",
            "message": "音源解析が完了しました",
            "file": {
                "name": filename,
                "title": Path(filename).stem,
                "extension": extension,
                "sizeBytes": total_size,
            },
            "analysis": analysis,
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            "Analysis error:",
            repr(error),
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "音源を解析できませんでした。"
                "別のMP3・WAV・M4Aで試してください"
            ),
        ) from error

    finally:
        await audio.close()

        if temporary_path:
            delete_temporary_file(
                temporary_path,
            )


def validate_extension(
    extension: str,
) -> None:
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "MP3・WAV・M4Aのみ"
                "対応しています"
            ),
        )


async def save_upload_to_temporary_file(
    audio: UploadFile,
    extension: str,
) -> tuple[str, int]:
    total_size = 0

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension,
    ) as temporary_file:
        temporary_path = temporary_file.name

        while True:
            chunk = await audio.read(
                CHUNK_SIZE,
            )

            if not chunk:
                break

            total_size += len(chunk)

            if total_size > MAX_FILE_SIZE:
                temporary_file.close()

                delete_temporary_file(
                    temporary_path,
                )

                raise HTTPException(
                    status_code=413,
                    detail=(
                        "ファイルは200MB以下に"
                        "してください"
                    ),
                )

            temporary_file.write(chunk)

    if total_size == 0:
        delete_temporary_file(
            temporary_path,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "空のファイルは"
                "解析できません"
            ),
        )

    return temporary_path, total_size


def analyze_file(
    file_path: str,
) -> dict[str, Any]:
    duration = get_duration(
        file_path,
    )

    audio_data, sample_rate = librosa.load(
        file_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
        duration=MAX_ANALYSIS_SECONDS,
    )

    if audio_data.size == 0:
        raise ValueError(
            "Decoded audio is empty",
        )

    audio_data = remove_silence(
        audio_data,
    )

    bpm_result = detect_tempo(
        audio_data=audio_data,
        sample_rate=sample_rate,
    )

    key_result = detect_key(
        audio_data=audio_data,
        sample_rate=sample_rate,
    )

    time_signature_result = (
        detect_time_signature(
            audio_data=audio_data,
            sample_rate=sample_rate,
            bpm=bpm_result["bpm"],
        )
    )

    measure_count = estimate_measure_count(
        duration=duration,
        bpm=bpm_result["bpm"],
        time_signature=(
            time_signature_result[
                "timeSignature"
            ]
        ),
    )

    return {
        "engine": "librosa-basic-v1",
        "durationSeconds": round(
            duration,
            3,
        ),
        "sampleRate": sample_rate,
        "bpm": bpm_result["bpm"],
        "bpmConfidence": (
            bpm_result["confidence"]
        ),
        "key": key_result["key"],
        "keyConfidence": (
            key_result["confidence"]
        ),
        "keyRoot": key_result["root"],
        "keyMode": key_result["mode"],
        "fifths": key_result["fifths"],
        "timeSignature": (
            time_signature_result[
                "timeSignature"
            ]
        ),
        "timeSignatureConfidence": (
            time_signature_result[
                "confidence"
            ]
        ),
        "measureCount": measure_count,
        "analysisSeconds": min(
            round(duration, 3),
            MAX_ANALYSIS_SECONDS,
        ),
        "parts": {},
        "notes": (
            "現在は曲全体の基本解析です。"
            "パート分離と音符解析は次工程です"
        ),
    }


def get_duration(
    file_path: str,
) -> float:
    try:
        info = sf.info(
            file_path,
        )

        if info.duration > 0:
            return float(
                info.duration,
            )

    except Exception:
        pass

    return float(
        librosa.get_duration(
            path=file_path,
        )
    )


def remove_silence(
    audio_data: np.ndarray,
) -> np.ndarray:
    trimmed_audio, _ = librosa.effects.trim(
        audio_data,
        top_db=40,
    )

    if trimmed_audio.size == 0:
        return audio_data

    return trimmed_audio


def detect_tempo(
    audio_data: np.ndarray,
    sample_rate: int,
) -> dict[str, int]:
    onset_envelope = (
        librosa.onset.onset_strength(
            y=audio_data,
            sr=sample_rate,
        )
    )

    tempo_array, beat_frames = (
        librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            units="frames",
        )
    )

    raw_tempo = float(
        np.asarray(
            tempo_array,
        ).reshape(-1)[0]
    )

    bpm = normalize_bpm(
        raw_tempo,
    )

    confidence = calculate_tempo_confidence(
        onset_envelope=onset_envelope,
        beat_frames=np.asarray(
            beat_frames,
        ),
    )

    return {
        "bpm": bpm,
        "confidence": confidence,
    }


def normalize_bpm(
    tempo: float,
) -> int:
    if not np.isfinite(tempo):
        return 120

    while tempo < 60:
        tempo *= 2

    while tempo > 220:
        tempo /= 2

    return int(
        round(tempo),
    )


def calculate_tempo_confidence(
    onset_envelope: np.ndarray,
    beat_frames: np.ndarray,
) -> int:
    if (
        onset_envelope.size == 0
        or beat_frames.size < 4
    ):
        return 20

    valid_frames = beat_frames[
        beat_frames < onset_envelope.size
    ]

    if valid_frames.size == 0:
        return 20

    beat_strength = float(
        np.mean(
            onset_envelope[
                valid_frames
            ]
        )
    )

    overall_strength = float(
        np.mean(
            onset_envelope
        )
    )

    if overall_strength <= 0:
        return 20

    ratio = (
        beat_strength
        / overall_strength
    )

    confidence = int(
        round(
            np.clip(
                25 + ratio * 25,
                20,
                94,
            )
        )
    )

    return confidence


def detect_key(
    audio_data: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    harmonic_audio, _ = librosa.effects.hpss(
        audio_data,
    )

    chroma = librosa.feature.chroma_cqt(
        y=harmonic_audio,
        sr=sample_rate,
        bins_per_octave=36,
    )

    chroma_average = np.mean(
        chroma,
        axis=1,
    )

    chroma_norm = normalize_vector(
        chroma_average,
    )

    best_score = float("-inf")
    second_score = float("-inf")
    best_root = 0
    best_mode = "major"

    for root in range(12):
        major_score = profile_score(
            chroma=chroma_norm,
            profile=MAJOR_PROFILE,
            root=root,
        )

        minor_score = profile_score(
            chroma=chroma_norm,
            profile=MINOR_PROFILE,
            root=root,
        )

        for score, mode in (
            (major_score, "major"),
            (minor_score, "minor"),
        ):
            if score > best_score:
                second_score = best_score
                best_score = score
                best_root = root
                best_mode = mode

            elif score > second_score:
                second_score = score

    confidence = key_confidence(
        best_score=best_score,
        second_score=second_score,
    )

    mode_label = (
        "Major"
        if best_mode == "major"
        else "Minor"
    )

    return {
        "key": (
            f"{KEY_NAMES[best_root]} "
            f"{mode_label}"
        ),
        "root": KEY_NAMES[best_root],
        "mode": best_mode,
        "fifths": FIFTHS_BY_ROOT[
            best_root
        ],
        "confidence": confidence,
    }


def normalize_vector(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    norm = float(
        np.linalg.norm(values)
    )

    if norm <= 0:
        return values

    return values / norm


def profile_score(
    chroma: np.ndarray,
    profile: np.ndarray,
    root: int,
) -> float:
    rotated_profile = np.roll(
        profile,
        root,
    )

    normalized_profile = normalize_vector(
        rotated_profile,
    )

    return float(
        np.dot(
            chroma,
            normalized_profile,
        )
    )


def key_confidence(
    best_score: float,
    second_score: float,
) -> int:
    if not np.isfinite(best_score):
        return 20

    difference = max(
        0.0,
        best_score - second_score,
    )

    confidence = int(
        round(
            np.clip(
                40 + difference * 350,
                35,
                92,
            )
        )
    )

    return confidence


def detect_time_signature(
    audio_data: np.ndarray,
    sample_rate: int,
    bpm: int,
) -> dict[str, Any]:
    onset_envelope = (
        librosa.onset.onset_strength(
            y=audio_data,
            sr=sample_rate,
        )
    )

    _, beat_frames = (
        librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            bpm=bpm,
            units="frames",
        )
    )

    beat_frames = np.asarray(
        beat_frames,
        dtype=int,
    )

    valid_frames = beat_frames[
        beat_frames
        < onset_envelope.size
    ]

    if valid_frames.size < 12:
        return {
            "timeSignature": "4/4",
            "confidence": 35,
        }

    beat_strengths = onset_envelope[
        valid_frames
    ]

    scores = {
        "2/4": meter_score(
            beat_strengths,
            2,
        ),
        "3/4": meter_score(
            beat_strengths,
            3,
        ),
        "4/4": meter_score(
            beat_strengths,
            4,
        ),
    }

    best_signature = max(
        scores,
        key=scores.get,
    )

    ordered_scores = sorted(
        scores.values(),
        reverse=True,
    )

    difference = (
        ordered_scores[0]
        - ordered_scores[1]
    )

    confidence = int(
        round(
            np.clip(
                35 + difference * 40,
                35,
                75,
            )
        )
    )

    return {
        "timeSignature": best_signature,
        "confidence": confidence,
    }


def meter_score(
    beat_strengths: np.ndarray,
    meter: int,
) -> float:
    grouped_strengths: list[float] = []

    for position in range(meter):
        values = beat_strengths[
            position::meter
        ]

        if values.size == 0:
            grouped_strengths.append(
                0.0,
            )
        else:
            grouped_strengths.append(
                float(
                    np.mean(values)
                )
            )

    first_beat = grouped_strengths[0]

    other_beats = (
        grouped_strengths[1:]
    )

    others_average = (
        float(
            np.mean(other_beats)
        )
        if other_beats
        else 0.0
    )

    return (
        first_beat
        - others_average
    )


def estimate_measure_count(
    duration: float,
    bpm: int,
    time_signature: str,
) -> int:
    beats, beat_type = [
        int(value)
        for value
        in time_signature.split("/")
    ]

    quarter_notes_per_measure = (
        beats
        * (4 / beat_type)
    )

    total_quarter_notes = (
        duration
        * bpm
        / 60
    )

    measure_count = round(
        total_quarter_notes
        / quarter_notes_per_measure
    )

    return max(
        1,
        int(measure_count),
    )


def delete_temporary_file(
    file_path: str,
) -> None:
    try:
        os.remove(
            file_path,
        )
    except OSError:
        pass
