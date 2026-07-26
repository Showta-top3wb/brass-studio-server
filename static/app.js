"use strict";

const API_BASE_URL = "https://brass-studio-api.onrender.com";

const $ = id => document.getElementById(id);

const partInputs = [
  ...document.querySelectorAll('input[name="part"]')
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
  manualTempoField: $("manualTempoField"),
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
  result: null
};

ui.audioFile.onchange = () => {
  const file = ui.audioFile.files?.[0];

  if (!file) {
    return;
  }

  const extension = file.name
    .split(".")
    .pop()
    .toLowerCase();

  if (
    !["mp3", "wav", "m4a"].includes(extension) ||
    !file.size ||
    file.size > 200 * 1024 * 1024
  ) {
    ui.sourceError.textContent =
      "MP3・WAV・M4A、200MB以下を選択してください";

    state.file = null;
    update();
    return;
  }

  state.file = file;
  state.result = null;
  ui.sourceError.textContent = "";

  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
  }

  state.objectUrl = URL.createObjectURL(file);
  ui.audioPlayer.src = state.objectUrl;
  ui.audioPlayer.hidden = false;

  ui.fileInfo.textContent =
    `${file.name} / ${formatBytes(file.size)}`;

  if (!ui.songTitle.value.trim()) {
    ui.songTitle.value =
      file.name.replace(/\.[^.]+$/, "");
  }

  update();
};

ui.selectAll.onclick = () => {
  partInputs.forEach(input => {
    input.checked = true;
  });

  update();
};

ui.clearAll.onclick = () => {
  partInputs.forEach(input => {
    input.checked = false;
  });

  update();
};

partInputs.forEach(input => {
  input.onchange = update;
});

ui.tempoMode.onchange = () => {
  ui.manualTempoField.hidden =
    ui.tempoMode.value !== "manual";
};

ui.analyzeButton.onclick = analyze;
ui.musicXmlButton.onclick = downloadMusicXml;
ui.pdfButton.onclick = openPdf;

ui.resetButton.onclick = () => {
  location.reload();
};

function selected() {
  return partInputs
    .filter(input => input.checked)
    .map(input => input.value);
}

function update() {
  const hasParts = selected().length > 0;

  ui.partsError.textContent =
    hasParts
      ? ""
      : "1つ以上選択してください";

  ui.analyzeButton.disabled =
    !state.file || !hasParts;
}

update();

async function analyze() {
  ui.analyzeButton.disabled = true;
  ui.progressCard.hidden = false;
  ui.progressLabel.textContent =
    "アップロード準備中";

  ui.resultCard.hidden = true;
  ui.sourceError.textContent = "";

  try {
    const formData = new FormData();

    formData.append(
      "audio",
      state.file,
      state.file.name
    );

    const query = new URLSearchParams({
      parts: selected().join(","),
      time_signature: ui.timeSignature.value,
      title:
        ui.songTitle.value.trim() ||
        "Untitled"
    });

    if (ui.tempoMode.value === "manual") {
      query.set(
        "manual_bpm",
        String(
          Number(ui.manualTempo.value) ||
          120
        )
      );
    }

    const result = await uploadAudio(
      `${API_BASE_URL}/analyze?${query.toString()}`,
      formData
    );

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
      result.notice;

    ui.resultCard.hidden = false;

    showToast("解析が完了しました");
      } catch (error) {
    console.error(error);

    ui.sourceError.textContent =
      error?.message ||
      "解析に失敗しました";

  } finally {
    ui.progressCard.hidden = true;
    update();
  }
}

function uploadAudio(url, formData) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.open("POST", url, true);
    xhr.responseType = "json";
    xhr.timeout = 10 * 60 * 1000;

    xhr.upload.onprogress = event => {
      if (!event.lengthComputable) {
        ui.progressLabel.textContent =
          "アップロード中";

        return;
      }

      const percent = Math.round(
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
      let data = xhr.response;

      if (!data && xhr.responseText) {
        try {
          data = JSON.parse(
            xhr.responseText
          );
        } catch {
          data = null;
        }
      }

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
          "APIとの通信に失敗しました"
        )
      );
    };

    xhr.ontimeout = () => {
      reject(
        new Error(
          "解析がタイムアウトしました"
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
  });
}

function downloadMusicXml() {
  const musicXml =
    state.result?.musicxml;

  if (!musicXml) {
    return;
  }

  const binary =
    atob(musicXml.base64);

  const bytes =
    new Uint8Array(binary.length);

  for (
    let index = 0;
    index < binary.length;
    index += 1
  ) {
    bytes[index] =
      binary.charCodeAt(index);
  }

  const url = URL.createObjectURL(
    new Blob(
      [bytes],
      {
        type:
          "application/vnd.recordare.musicxml+xml"
      }
    )
  );

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
    () => URL.revokeObjectURL(url),
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
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escapeHtml(result.title)}</title>
<style>
body{
  font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif;
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
<button onclick="window.print()">PDFとして保存</button>
<h1>${escapeHtml(result.title)}</h1>
<p class="summary">
${result.analysis.bpm} BPM /
${escapeHtml(result.analysis.key)} /
${escapeHtml(result.analysis.timeSignature)} /
${result.analysis.measureCount} 小節
</p>
<p>${escapeHtml(result.notice || "")}</p>
</body>
</html>`
  );

  popup.document.close();
}
function showToast(message) {
  if (!ui.toast) {
    return;
  }

  ui.toast.textContent = message;
  ui.toast.hidden = false;

  setTimeout(() => {
    ui.toast.hidden = true;
  }, 2500);
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(
    bytes /
    1024 /
    1024
  ).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}