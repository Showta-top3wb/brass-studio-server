from __future__ import annotations

import base64
import os
import re
import tempfile
from pathlib import Path
from typing import Annotated
from xml.sax.saxutils import escape

import librosa
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)


APP_NAME = (
    "Brass Studio Analysis API"
)
APP_VERSION = "1.5.0"

MAX_FILE_SIZE = (
    200 * 1024 * 1024
)
CHUNK_SIZE = 1024 * 1024
MAX_ANALYSIS_SECONDS = 600
TARGET_SAMPLE_RATE = 22050

ALLOWED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
}

DEFAULT_PARTS = (
    "trumpet,trombone,"
    "tenor-sax,tuba,"
    "snare-drum,bass-drum"
)

PARTS = {
    "trumpet": {
        "name": "Trumpet",
        "abbr": "Tpt.",
        "clef": "G",
        "line": 2,
        "diatonic": -1,
        "chromatic": -2,
    },
    "trombone": {
        "name": "Trombone",
        "abbr": "Tbn.",
        "clef": "F",
        "line": 4,
        "diatonic": 0,
        "chromatic": 0,
    },
    "tenor-sax": {
        "name": "Tenor Sax",
        "abbr": "T. Sax",
        "clef": "G",
        "line": 2,
        "diatonic": -8,
        "chromatic": -14,
    },
    "tuba": {
        "name": "Tuba",
        "abbr": "Tba.",
        "clef": "F",
        "line": 4,
        "diatonic": 0,
        "chromatic": 0,
    },
    "snare-drum": {
        "name": "Snare Drum",
        "abbr": "S.D.",
        "clef": "percussion",
        "line": 2,
        "diatonic": 0,
        "chromatic": 0,
        "percussion": True,
    },
    "bass-drum": {
        "name": "Bass Drum",
        "abbr": "B.D.",
        "clef": "percussion",
        "line": 2,
        "diatonic": 0,
        "chromatic": 0,
        "percussion": True,
    },
}

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
    ]
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
    ]
)

MAJOR_FIFTHS = {
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

MINOR_FIFTHS = {
    0: -3,
    1: 4,
    2: -1,
    3: 6,
    4: 1,
    5: -4,
    6: 3,
    7: -2,
    8: 5,
    9: 0,
    10: 7,
    11: 2,
}

SUPPORTED_TIME_SIGNATURES = {
    "2/4",
    "3/4",
    "4/4",
    "5/4",
    "6/8",
    "7/8",
    "9/8",
    "12/8",
}


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
)

# 公開APIのため、GitHub Pagesや
# Safariから確実に接続できるようにする。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/ping")
async def ping() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "POST received",
        "version": APP_VERSION,
    }


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def sanitize_filename(
    value: str,
) -> str:
    value = value.strip()

    value = re.sub(
        r'[\\/:*?"<>|]+',
        "_",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    value = value.strip(" ._")

    return (
        value[:120]
        or "Brass Studio Score"
    )


def parse_time_signature(
    value: str,
) -> tuple[str, int, int]:
    if value == "auto":
        return "4/4", 4, 4

    if (
        value
        not in SUPPORTED_TIME_SIGNATURES
    ):
        supported = "・".join(
            sorted(
                SUPPORTED_TIME_SIGNATURES
            )
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "対応していない拍子です。"
                f"対応拍子: {supported}"
            ),
        )

    beats, beat_type = map(
        int,
        value.split("/"),
    )

    return (
        value,
        beats,
        beat_type,
    )


def estimate_key(
    y: np.ndarray,
    sr: int,
) -> dict[str, str | int]:
    chroma = (
        librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
        )
    )

    mean_chroma = np.mean(
        chroma,
        axis=1,
    )

    norm = np.linalg.norm(
        mean_chroma
    )

    if norm < 1e-12:
        return {
            "name": "C Major",
            "mode": "major",
            "fifths": 0,
            "confidence": 35,
        }

    mean_chroma = (
        mean_chroma / norm
    )

    best_score = -1.0
    best_root = 0
    best_mode = "major"
    scores: list[float] = []

    for root in range(12):
        for mode, profile in (
            (
                "major",
                MAJOR_PROFILE,
            ),
            (
                "minor",
                MINOR_PROFILE,
            ),
        ):
            rotated = np.roll(
                profile,
                root,
            )

            rotated = (
                rotated /
                np.linalg.norm(rotated)
            )

            score = float(
                np.dot(
                    mean_chroma,
                    rotated,
                )
            )

            scores.append(score)

            if score > best_score:
                best_score = score
                best_root = root
                best_mode = mode

    scores.sort(reverse=True)

    difference = (
        scores[0] - scores[1]
        if len(scores) > 1
        else 0.0
    )

    confidence = int(
        round(
            clamp(
                45 + difference * 700,
                35,
                92,
            )
        )
    )

    fifths = (
        MAJOR_FIFTHS[best_root]
        if best_mode == "major"
        else MINOR_FIFTHS[best_root]
    )

    mode_name = (
        "Major"
        if best_mode == "major"
        else "Minor"
    )

    return {
        "name": (
            f"{KEY_NAMES[best_root]} "
            f"{mode_name}"
        ),
        "mode": best_mode,
        "fifths": fifths,
        "confidence": confidence,
    }


def estimate_tempo(
    y: np.ndarray,
    sr: int,
    manual_bpm: int | None,
) -> tuple[int, int]:
    if manual_bpm is not None:
        if not 40 <= manual_bpm <= 240:
            raise HTTPException(
                status_code=400,
                detail=(
                    "手動BPMは"
                    "40〜240で指定してください"
                ),
            )

        return manual_bpm, 100

    onset_envelope = (
        librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=512,
        )
    )

    tempo, beat_frames = (
        librosa.beat.beat_track(
            onset_envelope=(
                onset_envelope
            ),
            sr=sr,
            hop_length=512,
        )
    )

    tempo_value = float(
        np.asarray(tempo)
        .reshape(-1)[0]
    )

    if (
        not np.isfinite(tempo_value)
        or tempo_value <= 0
    ):
        return 120, 35

    bpm = int(
        round(
            clamp(
                tempo_value,
                40,
                240,
            )
        )
    )

    beat_count = len(
        np.asarray(beat_frames)
    )

    confidence = int(
        round(
            clamp(
                45 + beat_count * 0.7,
                45,
                88,
            )
        )
    )

    return bpm, confidence


def create_musicxml(
    title: str,
    bpm: int,
    key: dict[str, str | int],
    time_signature: str,
    measures: int,
    parts: list[str],
) -> str:
    beats, beat_type = map(
        int,
        time_signature.split("/"),
    )

    divisions = 4

    measure_duration = int(
        divisions *
        beats *
        (4 / beat_type)
    )

    score_parts: list[str] = []
    xml_parts: list[str] = []

    for index, part_id in enumerate(
        parts,
        start=1,
    ):
        part = PARTS[part_id]

        score_parts.append(
            f'<score-part id="P{index}">'
            f"<part-name>"
            f"{escape(part['name'])}"
            f"</part-name>"
            f"<part-abbreviation>"
            f"{escape(part['abbr'])}"
            f"</part-abbreviation>"
            "</score-part>"
        )

    for index, part_id in enumerate(
        parts,
        start=1,
    ):
        part = PARTS[part_id]
        xml_measures: list[str] = []

        for measure_number in range(
            1,
            measures + 1,
        ):
            attributes = ""

            if measure_number == 1:
                transpose = ""

                if part["chromatic"]:
                    transpose = (
                        "<transpose>"
                        f"<diatonic>"
                        f"{part['diatonic']}"
                        f"</diatonic>"
                        f"<chromatic>"
                        f"{part['chromatic']}"
                        f"</chromatic>"
                        "</transpose>"
                    )

                staff_details = ""

                if part.get(
                    "percussion"
                ):
                    staff_details = (
                        "<staff-details>"
                        "<staff-lines>1"
                        "</staff-lines>"
                        "</staff-details>"
                    )

                attributes = (
                    "<attributes>"
                    f"<divisions>"
                    f"{divisions}"
                    f"</divisions>"
                    "<key>"
                    f"<fifths>"
                    f"{key['fifths']}"
                    f"</fifths>"
                    f"<mode>"
                    f"{key['mode']}"
                    f"</mode>"
                    "</key>"
                    "<time>"
                    f"<beats>{beats}"
                    f"</beats>"
                    f"<beat-type>"
                    f"{beat_type}"
                    f"</beat-type>"
                    "</time>"
                    "<clef>"
                    f"<sign>"
                    f"{part['clef']}"
                    f"</sign>"
                    f"<line>"
                    f"{part['line']}"
                    f"</line>"
                    "</clef>"
                    f"{transpose}"
                    f"{staff_details}"
                    "</attributes>"
                    '<direction placement="above">'
                    "<direction-type>"
                    "<metronome>"
                    "<beat-unit>quarter"
                    "</beat-unit>"
                    f"<per-minute>"
                    f"{bpm}"
                    f"</per-minute>"
                    "</metronome>"
                    "</direction-type>"
                    f'<sound tempo="{bpm}"/>'
                    "</direction>"
                )

            final_barline = ""

            if (
                measure_number
                == measures
            ):
                final_barline = (
                    '<barline location="right">'
                    "<bar-style>"
                    "light-heavy"
                    "</bar-style>"
                    "</barline>"
                )

            xml_measures.append(
                f'<measure number="'
                f'{measure_number}">'
                f"{attributes}"
                "<note>"
                '<rest measure="yes"/>'
                f"<duration>"
                f"{measure_duration}"
                f"</duration>"
                "<voice>1</voice>"
                f"{final_barline}"
                "</measure>"
            )

        xml_parts.append(
            f'<part id="P{index}">'
            f'{"".join(xml_measures)}'
            "</part>"
        )

    return (
        '<?xml version="1.0" '
        'encoding="UTF-8"?>'
        '<score-partwise version="4.0">'
        "<work>"
        f"<work-title>"
        f"{escape(title)}"
        f"</work-title>"
        "</work>"
        f"<movement-title>"
        f"{escape(title)}"
        f"</movement-title>"
        "<identification>"
        "<encoding>"
        "<software>"
        "Brass Studio"
        "</software>"
        "</encoding>"
        "</identification>"
        "<part-list>"
        f'{"".join(score_parts)}'
        "</part-list>"
        f'{"".join(xml_parts)}'
        "</score-partwise>"
    )


@app.post("/analyze")
async def analyze(
    audio: Annotated[
        UploadFile,
        File(...),
    ],
    parts: str = DEFAULT_PARTS,
    time_signature: str = "auto",
    manual_bpm: int | None = None,
    title: str | None = None,
) -> dict:
    original_filename = (
        audio.filename or "audio"
    )

    extension = (
        Path(original_filename)
        .suffix
        .lower()
    )

    if (
        extension
        not in ALLOWED_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "MP3・WAV・M4Aのみ"
                "対応しています"
            ),
        )

    selected_parts = [
        part.strip()
        for part in parts.split(",")
        if part.strip() in PARTS
    ]

    selected_parts = list(
        dict.fromkeys(
            selected_parts
        )
    )

    if not selected_parts:
        raise HTTPException(
            status_code=400,
            detail=(
                "1つ以上のパートを"
                "選択してください"
            ),
        )

    (
        parsed_time_signature,
        beats,
        beat_type,
    ) = parse_time_signature(
        time_signature
    )

    temporary_path: str | None = None
    uploaded_size = 0

    try:
        with (
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension,
            )
        ) as temporary_file:
            temporary_path = (
                temporary_file.name
            )

            while True:
                chunk = await audio.read(
                    CHUNK_SIZE
                )

                if not chunk:
                    break

                uploaded_size += len(
                    chunk
                )

                if (
                    uploaded_size
                    > MAX_FILE_SIZE
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "ファイルは"
                            "200MB以下に"
                            "してください"
                        ),
                    )

                temporary_file.write(
                    chunk
                )

        if uploaded_size == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "音声ファイルが空です"
                ),
            )

        try:
            y, sr = librosa.load(
                temporary_path,
                sr=TARGET_SAMPLE_RATE,
                mono=True,
                duration=(
                    MAX_ANALYSIS_SECONDS
                ),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "音声ファイルを"
                    "読み込めませんでした。"
                    "別のMP3・WAV・M4Aで"
                    "試してください"
                ),
            ) from exc

        if y.size == 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "音声データを"
                    "検出できませんでした"
                ),
            )

        duration = float(
            librosa.get_duration(
                y=y,
                sr=sr,
            )
        )

        if duration < 1.0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "1秒以上の音声ファイルを"
                    "使用してください"
                ),
            )

        peak = float(
            np.max(np.abs(y))
        )

        if peak < 1e-5:
            raise HTTPException(
                status_code=422,
                detail=(
                    "音量が小さすぎるか、"
                    "無音のファイルです"
                ),
            )

        (
            bpm,
            bpm_confidence,
        ) = estimate_tempo(
            y=y,
            sr=sr,
            manual_bpm=manual_bpm,
        )

        key = estimate_key(
            y=y,
            sr=sr,
        )

        beats_per_measure = (
            beats *
            (4 / beat_type)
        )

        total_quarter_beats = (
            duration *
            bpm /
            60
        )

        measure_count = max(
            1,
            int(
                round(
                    total_quarter_beats /
                    beats_per_measure
                )
            ),
        )

        raw_title = (
            title
            or Path(
                original_filename
            ).stem
        )

        score_title = (
            sanitize_filename(
                raw_title
            )
        )

        musicxml = create_musicxml(
            title=score_title,
            bpm=bpm,
            key=key,
            time_signature=(
                parsed_time_signature
            ),
            measures=measure_count,
            parts=selected_parts,
        )

        encoded_musicxml = (
            base64.b64encode(
                musicxml.encode(
                    "utf-8"
                )
            ).decode("ascii")
        )

        return {
            "status": "complete",
            "title": score_title,
            "analysis": {
                "durationSeconds": (
                    round(duration, 2)
                ),
                "bpm": bpm,
                "bpmConfidence": (
                    bpm_confidence
                ),
                "key": key["name"],
                "keyConfidence": (
                    key["confidence"]
                ),
                "timeSignature": (
                    parsed_time_signature
                ),
                "timeSignatureConfidence": (
                    100
                    if (
                        time_signature
                        != "auto"
                    )
                    else 50
                ),
                "measureCount": (
                    measure_count
                ),
            },
            "selectedParts": (
                selected_parts
            ),
            "musicxml": {
                "filename": (
                    f"{score_title}"
                    ".musicxml"
                ),
                "mimeType": (
                    "application/vnd."
                    "recordare."
                    "musicxml+xml"
                ),
                "base64": (
                    encoded_musicxml
                ),
            },
            "notice": (
                f"Ver.{APP_VERSION}は"
                "BPM・Key・拍子・小節数の"
                "解析と、選択パートを含む"
                "MusicXML土台生成に"
                "対応しています。"
                "音符の自動採譜と"
                "パート分離は"
                "まだ含まれていません。"
            ),
        }

    finally:
        await audio.close()

        if temporary_path:
            try:
                os.remove(
                    temporary_path
                )
            except OSError:
                pass
