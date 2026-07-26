"use strict";

const API_BASE_URL =
  "https://brass-studio-api.onrender.com";

const MAX_FILE_SIZE =
  200 * 1024 * 1024;

const $ = id =>
  document.getElementById(id);

const partInputs = [
  ...document.querySelectorAll(
    'input[name="part"]'
  )
];

const ui = {
  audioFile: $("audioFile"),
  fileInfo: $("fileInfo"),
  audioPlayer: $("audioPlayer"),
  sourceError: $("sourceError"),
  selectAll: $("selectAll"),
  clearAll: $("clearAll"),
  partsError: $("partsError"),
  songTitle: $("songTitle"),
  tempoMode: $("tempoMode"),
  manualTempoField:
    $("manualTempoField"),
  manualTempo: $("manualTempo"),
  timeSignature: $("timeSignature"),
  analyzeButton: $("analyzeButton"),
  progressCard: $("progressCard"),
  progressLabel: $("progressLabel"),
  resultCard: $("resultCard"),
  resultBpm: $("resultBpm"),
  bpmConfidence: $("bpmConfidence"),
  resultKey: $("resultKey"),
  keyConfidence: $("keyConfidence"),
  resultTime: $("resultTime"),
  timeConfidence: $("timeConfidence"),
  resultMeasures: $("resultMeasures"),
  resultNotice: $("resultNotice"),
  musicXmlButton: $("musicXmlButton"),
  pdfButton: $("pdfButton"),
  resetButton: $("resetButton"),
  toast: $("toast")
};

const state = {
  file: null,
  objectUrl: "",
  result: null,
  running: false
};

ui.audioFile.addEventListener(
  "change",
  handleFileChange
);

ui.selectAll.addEventListener(
  "click",
  () => {
    partInputs.forEach(input => {
      input.checked = true;
    });

    updateControls();
  }
);

ui.clearAll.addEventListener(
  "click",
  () => {
    partInputs.forEach(input => {
      input.checked = false;
    });

    updateControls();
  }
);

partInputs.forEach(input => {
  input.addEventListener(
    "change",
    updateControls
  );
});

ui.tempoMode.addEventListener(
  "change",
  () => {
    ui.manualTempoField.hidden =
      ui.tempoMode.value !== "manual";
  }
);

ui.analyzeButton.addEventListener(
  "click",
  analyze
);

ui.musicXmlButton.addEventListener(
  "click",
  downloadMusicXml
);

ui.pdfButton.addEventListener(
  "click",
  openPdf
);

ui.resetButton.addEventListener(
  "click",
  () => {
    location.reload();
  }
);

updateControls();

function handleFileChange() {
  const file =
    ui.audioFile.files?.[0];

  if (!file) {
    state.file = null;
    updateControls();
    return;
  }

  const extension =
    file.name
      .split(".")
      .pop()
      .toLowerCase();

  const allowed =
    ["mp3", "wav", "m4a"];

  if (
    !allowed.includes(extension) ||
    !file.size ||
    file.size > MAX_FILE_SIZE
  ) {
    state.file = null;

    ui.sourceError.textContent =
      "MP3・WAV・M4A、200MB以下を選択してください";

    updateControls();
    return;
  }

  state.file = file;
  state.result = null;

  ui.sourceError.textContent = "";
  ui.resultCard.hidden = true;

  if (state.objectUrl) {
    URL.revokeObjectURL(
      state.objectUrl
    );
  }

  state.objectUrl =
    URL.createObjectURL(file);

  ui.audioPlayer.src =
    state.objectUrl;

  ui.audioPlayer.hidden = false;

  ui.fileInfo.textContent =
    `${file.name} / ${formatBytes(file.size)}`;

  if (!ui.songTitle.value.trim()) {
    ui.songTitle.value =
      file.name.replace(
        /\.[^.]+$/,
        ""
      );
  }

  updateControls();
}

function selectedParts() {
  return partInputs
    .filter(input => input.checked)
    .map(input => input.value);
}

function updateControls() {
  const hasParts =
    selectedParts().length > 0;

  ui.partsError.textContent =
    hasParts
      ? ""
      : "1つ以上選択してください";

  ui.analyzeButton.disabled =
    state.running ||
    !state.file ||
    !hasParts;
}

async function analyze() {
  if (
    state.running ||
    !state.file
  ) {
    return;
  }

  state.running = true;
  state.result = null;

  ui.resultCard.hidden = true;
  ui.progressCard.hidden = false;
  ui.progressLabel.textContent =
    "API接続確認中";
  ui.sourceError.textContent = "";

  updateControls();

  try {
    await checkApi();

    const result =
      await uploadAndAnalyze();

    renderResult(result);

  } catch (error) {
    console.error(error);

    ui.sourceError.textContent =
      error instanceof Error
        ? error.message
        : "解析に失敗しました";

  } finally {
    state.running = false;
    ui.progressCard.hidden = true;
    updateControls();
  }
}

async function checkApi() {
  let response;

  try {
    response = await fetch(
      `${API_BASE_URL}/ping`,
      {
        method: "POST",
        mode: "cors",
        cache: "no-store"
      }
    );
  } catch {
    throw new Error(
      "APIへ接続できませんでした"
    );
  }

  if (!response.ok) {
    throw new Error(
      `API接続テスト失敗（HTTP ${response.status}）`
    );
  }
}

function uploadAndAnalyze() {
  const formData =
    new FormData();

  formData.append(
    "audio",
    state.file,
    state.file.name
  );

  const query =
    new URLSearchParams({
      parts:
        selectedParts().join(","),
      time_signature:
        ui.timeSignature.value,
      title:
        ui.songTitle.value.trim() ||
        "Untitled"
    });

  if (
    ui.tempoMode.value === "manual"
  ) {
    const bpm =
      Number(ui.manualTempo.value);

    if (
      !Number.isFinite(bpm) ||
      bpm < 40 ||
      bpm > 240
    ) {
      return Promise.reject(
        new Error(
          "手動BPMは40〜240で入力してください"
        )
      );
    }

    query.set(
      "manual_bpm",
      String(Math.round(bpm))
    );
  }

  const url =
    `${API_BASE_URL}/analyze?${query.toString()}`;

  return new Promise(
    (resolve, reject) => {
      const xhr =
        new XMLHttpRequest();

      xhr.open(
        "POST",
        url,
        true
      );

      xhr.responseType = "text";
      xhr.timeout =
        10 * 60 * 1000;

      xhr.upload.onprogress =
        event => {
          if (
            !event.lengthComputable
          ) {
            ui.progressLabel.textContent =
              "アップロード中";
            return;
          }

          const percent =
            Math.round(
              event.loaded /
              event.total *
              100
            );

          ui.progressLabel.textContent =
            `アップロード中 ${percent}%`;
        };

      xhr.upload.onload = () => {
        ui.progressLabel.textContent =
          "音源を解析中";
      };

      xhr.onload = () => {
        const data =
          parseJson(xhr.responseText);

        if (
          xhr.status >= 200 &&
          xhr.status < 300
        ) {
          if (!data) {
            reject(
              new Error(
                "APIの応答を読み取れませんでした"
              )
            );
            return;
          }

          resolve(data);
          return;
        }

        reject(
          new Error(
            data?.detail ||
            `解析に失敗しました（HTTP ${xhr.status}）`
          )
        );
      };

      xhr.onerror = () => {
        reject(
          new Error(
            "音源の送信中に通信が切れました"
          )
        );
      };

      xhr.ontimeout = () => {
        reject(
          new Error(
            "解析が10分でタイムアウトしました"
          )
        );
      };

      xhr.onabort = () => {
        reject(
          new Error(
            "アップロードが中断されました"
          )
        );
      };

      xhr.send(formData);
    }
  );
}

function renderResult(result) {
  if (!result?.analysis) {
    throw new Error(
      "解析結果の形式が不正です"
    );
  }

  state.result = result;

  ui.resultBpm.textContent =
    result.analysis.bpm;

  ui.bpmConfidence.textContent =
    `信頼度 ${result.analysis.bpmConfidence}%`;

  ui.resultKey.textContent =
    result.analysis.key;

  ui.keyConfidence.textContent =
    `信頼度 ${result.analysis.keyConfidence}%`;

  ui.resultTime.textContent =
    result.analysis.timeSignature;

  ui.timeConfidence.textContent =
    `信頼度 ${result.analysis.timeSignatureConfidence}%`;

  ui.resultMeasures.textContent =
    result.analysis.measureCount;

  ui.resultNotice.textContent =
    result.notice || "";

  ui.resultCard.hidden = false;

  showToast(
    "解析が完了しました"
  );
}

function downloadMusicXml() {
  const musicXml =
    state.result?.musicxml;

  if (!musicXml?.base64) {
    ui.sourceError.textContent =
      "MusicXMLがありません";
    return;
  }

  const binary =
    atob(musicXml.base64);

  const bytes =
    new Uint8Array(
      binary.length
    );

  for (
    let index = 0;
    index < binary.length;
    index += 1
  ) {
    bytes[index] =
      binary.charCodeAt(index);
  }

  const blob =
    new Blob(
      [bytes],
      {
        type:
          musicXml.mimeType ||
          "application/vnd.recordare.musicxml+xml"
      }
    );

  const url =
    URL.createObjectURL(blob);

  const link =
    document.createElement("a");

  link.href = url;
  link.download =
    musicXml.filename ||
    "score.musicxml";

  document.body.appendChild(link);
  link.click();
  link.remove();

  setTimeout(
    () => {
      URL.revokeObjectURL(url);
    },
    1000
  );
}

function openPdf() {
  if (!state.result) {
    return;
  }

  const result =
    state.result;

  const popup =
    window.open("", "_blank");

  if (!popup) {
    ui.sourceError.textContent =
      "ポップアップを許可してください";
    return;
  }

  popup.document.write(
    `<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta
  name="viewport"
  content="width=device-width,initial-scale=1"
>
<title>${escapeHtml(result.title)}</title>
<style>
body{
  font-family:
    -apple-system,
    BlinkMacSystemFont,
    "Helvetica Neue",
    sans-serif;
  padding:32px;
  line-height:1.6;
}
button{
  padding:12px 18px;
  margin-bottom:24px;
}
h1{
  margin-bottom:8px;
}
.summary{
  font-size:18px;
}
@media print{
  button{
    display:none;
  }
}
</style>
</head>
<body>
<button onclick="window.print()">
PDFとして保存
</button>

<h1>${escapeHtml(result.title)}</h1>

<p class="summary">
${result.analysis.bpm} BPM /
${escapeHtml(result.analysis.key)} /
${escapeHtml(result.analysis.timeSignature)} /
${result.analysis.measureCount} 小節
</p>

<p>
${escapeHtml(result.notice || "")}
</p>
</body>
</html>`
  );

  popup.document.close();
}

function parseJson(text) {
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function showToast(message) {
  if (!ui.toast) {
    return;
  }

  ui.toast.textContent =
    message;

  ui.toast.hidden =
    false;

  setTimeout(
    () => {
      ui.toast.hidden = true;
    },
    2500
  );
}

function formatBytes(bytes) {
  if (
    bytes < 1024 * 1024
  ) {
    return `${
      (bytes / 1024).toFixed(1)
    } KB`;
  }

  return `${
    (
      bytes /
      1024 /
      1024
    ).toFixed(1)
  } MB`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
