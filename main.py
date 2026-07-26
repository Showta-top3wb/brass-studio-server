from __future__ import annotations
import base64, math, os, tempfile
from pathlib import Path
from typing import Annotated
from xml.sax.saxutils import escape
import librosa, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
APP_NAME='Brass Studio Analysis API';MAX_FILE_SIZE=200*1024*1024;CHUNK_SIZE=1024*1024;ALLOWED_EXTENSIONS={'.mp3','.wav','.m4a'}
PARTS={'trumpet':{'name':'Trumpet','abbr':'Tpt.','clef':'G','line':2,'diatonic':-1,'chromatic':-2},'trombone':{'name':'Trombone','abbr':'Tbn.','clef':'F','line':4,'diatonic':0,'chromatic':0},'tenor-sax':{'name':'Tenor Sax','abbr':'T. Sax','clef':'G','line':2,'diatonic':-8,'chromatic':-14},'tuba':{'name':'Tuba','abbr':'Tba.','clef':'F','line':4,'diatonic':0,'chromatic':0},'snare-drum':{'name':'Snare Drum','abbr':'S.D.','clef':'percussion','line':2,'diatonic':0,'chromatic':0,'percussion':True},'bass-drum':{'name':'Bass Drum','abbr':'B.D.','clef':'percussion','line':2,'diatonic':0,'chromatic':0,'percussion':True}}
KEY_NAMES=['C','C♯','D','E♭','E','F','F♯','G','A♭','A','B♭','B'];MAJOR=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]);MINOR=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]);FIFTHS={0:0,1:7,2:2,3:-3,4:4,5:-1,6:6,7:1,8:-4,9:3,10:-2,11:5}
app=FastAPI(title=APP_NAME,version='1.3.0');app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=False,allow_methods=['*'],allow_headers=['*'])
@app.get('/')
async def root():return {'name':APP_NAME,'version':'1.3.0','status':'running'}
@app.get('/health')
async def health():return {'status':'ok','version':'1.3.0'}
def clamp(v,a,b):return max(a,min(b,v))
def estimate_key(y,sr):
 c=librosa.feature.chroma_cqt(y=y,sr=sr);m=np.mean(c,axis=1);m=m/(np.linalg.norm(m)+1e-12);best=(-1,0,'major');scores=[]
 for r in range(12):
  for mode,p in [('major',MAJOR),('minor',MINOR)]:
   q=np.roll(p,r);q=q/np.linalg.norm(q);sc=float(np.dot(m,q));scores.append(sc)
   if sc>best[0]:best=(sc,r,mode)
 scores.sort(reverse=True);conf=int(round(clamp(45+(scores[0]-scores[1])*700,35,92)));root=best[1];mode=best[2];return {'name':f"{KEY_NAMES[root]} {'Major' if mode=='major' else 'Minor'}",'mode':mode,'fifths':FIFTHS[root],'confidence':conf}
def create_musicxml(title,bpm,key,time_sig,measures,parts):
 beats,beat_type=map(int,time_sig.split('/'));divisions=4;duration=int(divisions*beats*(4/beat_type));plist=[];px=[]
 for i,p in enumerate(parts,1):
  d=PARTS[p];plist.append(f'<score-part id="P{i}"><part-name>{escape(d["name"])}</part-name><part-abbreviation>{escape(d["abbr"])}</part-abbreviation></score-part>')
 for i,p in enumerate(parts,1):
  d=PARTS[p];ms=[]
  for n in range(1,measures+1):
   attrs=''
   if n==1:
    tr=f'<transpose><diatonic>{d["diatonic"]}</diatonic><chromatic>{d["chromatic"]}</chromatic></transpose>' if d['chromatic'] else ''
    staff='<staff-details><staff-lines>1</staff-lines></staff-details>' if d.get('percussion') else ''
    attrs=f'<attributes><divisions>{divisions}</divisions><key><fifths>{key["fifths"]}</fifths><mode>{key["mode"]}</mode></key><time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time><clef><sign>{d["clef"]}</sign><line>{d["line"]}</line></clef>{tr}{staff}</attributes><direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>{bpm}</per-minute></metronome></direction-type><sound tempo="{bpm}"/></direction>'
   end='<barline location="right"><bar-style>light-heavy</bar-style></barline>' if n==measures else ''
   ms.append(f'<measure number="{n}">{attrs}<note><rest measure="yes"/><duration>{duration}</duration><voice>1</voice><type>whole</type></note>{end}</measure>')
  px.append(f'<part id="P{i}">{"".join(ms)}</part>')
 return f'<?xml version="1.0" encoding="UTF-8"?><score-partwise version="4.0"><work><work-title>{escape(title)}</work-title></work><movement-title>{escape(title)}</movement-title><part-list>{"".join(plist)}</part-list>{"".join(px)}</score-partwise>'
@app.post('/analyze')
async def analyze(audio:Annotated[UploadFile,File(...)],parts:str='trumpet,trombone,tenor-sax,tuba,snare-drum,bass-drum',time_signature:str='auto',manual_bpm:int|None=None,title:str|None=None):
 fn=audio.filename or 'audio';ext=Path(fn).suffix.lower()
 if ext not in ALLOWED_EXTENSIONS:raise HTTPException(400,'MP3・WAV・M4Aのみ対応しています')
 sel=[p for p in parts.split(',') if p in PARTS]
 if not sel:raise HTTPException(400,'1つ以上のパートを選択してください')
 tmp=None;size=0
 try:
  with tempfile.NamedTemporaryFile(delete=False,suffix=ext) as t:
   tmp=t.name
   while True:
    c=await audio.read(CHUNK_SIZE)
    if not c:break
    size+=len(c)
    if size>MAX_FILE_SIZE:raise HTTPException(413,'ファイルは200MB以下にしてください')
    t.write(c)
  y,sr=librosa.load(tmp,sr=22050,mono=True,duration=600);dur=float(librosa.get_duration(y=y,sr=sr));onset=librosa.onset.onset_strength(y=y,sr=sr,hop_length=512)
  if manual_bpm is not None:bpm=int(clamp(manual_bpm,40,240));conf=100
  else:
   tempo,_=librosa.beat.beat_track(onset_envelope=onset,sr=sr,hop_length=512);bpm=int(round(clamp(float(np.asarray(tempo).reshape(-1)[0]),40,240)));conf=70
  key=estimate_key(y,sr);ts='4/4' if time_signature=='auto' else time_signature;beats,beat_type=map(int,ts.split('/'));measures=max(1,int(round((dur*bpm/60)/(beats*(4/beat_type)))));score_title=(title or Path(fn).stem).strip();xml=create_musicxml(score_title,bpm,key,ts,measures,sel);b64=base64.b64encode(xml.encode()).decode()
  return {'status':'complete','title':score_title,'analysis':{'bpm':bpm,'bpmConfidence':conf,'key':key['name'],'keyConfidence':key['confidence'],'timeSignature':ts,'timeSignatureConfidence':100 if time_signature!='auto' else 50,'measureCount':measures},'selectedParts':sel,'musicxml':{'filename':f'{score_title}.musicxml','base64':b64},'notice':'Ver.1.3.0はBPM・Key・拍子・小節数の解析とMusicXML土台生成に対応しています。'}
 finally:
  await audio.close()
  if tmp:
   try:os.remove(tmp)
   except OSError:pass
