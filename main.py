from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Annotated
from xml.sax.saxutils import escape

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

MAX_FILE_SIZE = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a'}
PARTS = {
    'trumpet': ('Trumpet', 'Tpt.', 'G', 2, -1, -2, False),
    'trombone': ('Trombone', 'Tbn.', 'F', 4, 0, 0, False),
    'tenor-sax': ('Tenor Sax', 'T. Sax', 'G', 2, -8, -14, False),
    'tuba': ('Tuba', 'Tba.', 'F', 4, 0, 0, False),
    'snare-drum': ('Snare Drum', 'S.D.', 'percussion', 2, 0, 0, True),
    'bass-drum': ('Bass Drum', 'B.D.', 'percussion', 2, 0, 0, True),
}
KEY_NAMES = ['C','C♯','D','E♭','E','F','F♯','G','A♭','A','B♭','B']
MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
FIFTHS = {0:0,1:7,2:2,3:-3,4:4,5:-1,6:6,7:1,8:-4,9:3,10:-2,11:5}
app = FastAPI(title='Brass Studio', version='1.1.0')

@app.get('/health')
async def health():
    return {'status':'ok','version':'1.1.0'}

def estimate_key(y: np.ndarray, sr: int):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    avg = np.mean(chroma, axis=1)
    avg = avg / (np.linalg.norm(avg) + 1e-12)
    best = (-1.0, 0, 'major')
    scores = []
    for root in range(12):
        for mode, profile in [('major', MAJOR), ('minor', MINOR)]:
            p = np.roll(profile, root)
            p = p / np.linalg.norm(p)
            score = float(np.dot(avg, p))
            scores.append(score)
            if score > best[0]: best = (score, root, mode)
    scores.sort(reverse=True)
    confidence = max(35, min(92, round(45 + (scores[0]-scores[1])*700)))
    suffix = 'Major' if best[2] == 'major' else 'Minor'
    return {'name':f'{KEY_NAMES[best[1]]} {suffix}','mode':best[2],'fifths':FIFTHS[best[1]],'confidence':confidence}

def create_musicxml(title: str, bpm: int, key: dict, time_sig: str, measures: int, selected: list[str]):
    beats, beat_type = map(int, time_sig.split('/'))
    divisions = 4
    duration = int(divisions * beats * (4 / beat_type))
    part_list = []
    parts_xml = []
    for i, part_key in enumerate(selected, 1):
        name, abbr, clef, line, diatonic, chromatic, percussion = PARTS[part_key]
        part_list.append(f'<score-part id="P{i}"><part-name>{escape(name)}</part-name><part-abbreviation>{escape(abbr)}</part-abbreviation></score-part>')
        measure_xml = []
        for m in range(1, measures + 1):
            attrs = ''
            if m == 1:
                transpose = f'<transpose><diatonic>{diatonic}</diatonic><chromatic>{chromatic}</chromatic></transpose>' if chromatic else ''
                staff = '<staff-details><staff-lines>1</staff-lines></staff-details>' if percussion else ''
                attrs = f'<attributes><divisions>{divisions}</divisions><key><fifths>{key["fifths"]}</fifths><mode>{key["mode"]}</mode></key><time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time><clef><sign>{clef}</sign><line>{line}</line></clef>{transpose}{staff}</attributes><direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>{bpm}</per-minute></metronome></direction-type><sound tempo="{bpm}"/></direction>'
            final = '<barline location="right"><bar-style>light-heavy</bar-style></barline>' if m == measures else ''
            measure_xml.append(f'<measure number="{m}">{attrs}<note><rest measure="yes"/><duration>{duration}</duration><voice>1</voice><type>whole</type></note>{final}</measure>')
        parts_xml.append(f'<part id="P{i}">{"".join(measure_xml)}</part>')
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0"><work><work-title>{escape(title)}</work-title></work><movement-title>{escape(title)}</movement-title><identification><creator type="arranger">Brass Studio</creator><encoding><software>Brass Studio Ver.1.1</software></encoding></identification><part-list>{''.join(part_list)}</part-list>{''.join(parts_xml)}</score-partwise>'''

@app.post('/analyze')
async def analyze(audio: Annotated[UploadFile, File(...)], parts: str = 'trumpet,trombone,tenor-sax,tuba,snare-drum,bass-drum', time_signature: str = 'auto', manual_bpm: int | None = None):
    filename = audio.filename or 'audio'
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, 'MP3・WAV・M4Aのみ対応しています')
    selected = [p for p in parts.split(',') if p in PARTS]
    if not selected:
        raise HTTPException(400, '1つ以上のパートを選択してください')
    temp_path = None
    size = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            temp_path = f.name
            while chunk := await audio.read(1024*1024):
                size += len(chunk)
                if size > MAX_FILE_SIZE: raise HTTPException(413, 'ファイルは200MB以下にしてください')
                f.write(chunk)
        y, sr = librosa.load(temp_path, sr=22050, mono=True, duration=600)
        if y.size == 0: raise HTTPException(422, '音声データが見つかりませんでした')
        duration = float(librosa.get_duration(y=y, sr=sr))
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        if manual_bpm is None:
            tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr)
            bpm = int(round(float(np.asarray(tempo).reshape(-1)[0])))
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            confidence = 35 if len(beat_times) < 4 else max(35, min(92, round(92 - (np.std(np.diff(beat_times))/(np.mean(np.diff(beat_times))+1e-9))*180)))
        else:
            bpm = max(40, min(240, int(manual_bpm)))
            confidence = 100
        key = estimate_key(y, sr)
        time_sig = '4/4' if time_signature == 'auto' else time_signature
        if time_sig not in {'2/4','3/4','4/4','6/8'}: raise HTTPException(400, '拍子設定が不正です')
        beats, beat_type = map(int, time_sig.split('/'))
        measure_count = max(1, round((duration*bpm/60)/(beats*(4/beat_type))))
        title = Path(filename).stem
        xml = create_musicxml(title, bpm, key, time_sig, measure_count, selected)
        return {
            'status':'complete',
            'file':{'name':filename,'sizeBytes':size,'durationSeconds':round(duration,2),'sampleRate':sr},
            'analysis':{'bpm':bpm,'bpmConfidence':confidence,'key':key['name'],'keyConfidence':key['confidence'],'timeSignature':time_sig,'timeSignatureConfidence':100 if time_signature != 'auto' else 55,'measureCount':measure_count},
            'selectedParts':selected,
            'musicxml':{'filename':f'{title}.musicxml','base64':base64.b64encode(xml.encode()).decode()},
            'notice':'Ver.1.1はBPM・Key・拍子・小節数の解析と、MusicXMLの空スコア生成に対応しています。パート別音符採譜は次段階です。'
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, '音源を解析できませんでした') from exc
    finally:
        await audio.close()
        if temp_path:
            try: os.remove(temp_path)
            except OSError: pass

app.mount('/static', StaticFiles(directory='static'), name='static')
@app.get('/', include_in_schema=False)
async def home():
    return FileResponse('static/index.html')
