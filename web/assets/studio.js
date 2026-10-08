"use strict";
const $ = id => document.getElementById(id);
const state = { source: document.body.dataset.page === "webcam" ? "webcam" : "upload", mode: "letters",
  models: [], file: null, previewURL: null, busy: false, result: null, stream: null,
  session: null, running: false, generation: 0, controller: null, timer: null,
  demonstrations: [], selectedSign: null, videosOnly: false };
let toastTimer;
const samples = [
  { id: "sample-1", file: "sample-1.jpg", title: "Letter Y" },
  { id: "sample-2", file: "sample-2.jpg", title: "Letter X" },
  { id: "sample-3", file: "sample-3.jpg", title: "Letter D" }
];
const currentModel = () => state.models.find(model => model.id === state.mode);
const available = () => Boolean(currentModel()?.available);
function showError(message = "") { $("input-error").textContent = message; $("input-error").hidden = !message; }
function toast(message) { clearTimeout(toastTimer); $("toast").textContent = message; $("toast").hidden = false; toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500); }
async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The request could not be completed. Try again.");
  return data;
}
function setBusy(value) {
  state.busy = value;
  $("translate-button").disabled = value || !state.file || !available();
  $("remove-file").disabled = value;
  $("dropzone").disabled = value || !available();
  $("camera-button").disabled = value || !available();
  $("vocabulary-button").disabled = value || state.running || !available();
  document.querySelectorAll(".mode-button, .sample-button").forEach(button => { button.disabled = value; });
  $("translate-button").querySelector("span").textContent = value ? "Translating…" : "Translate sign";
}
function resetResult() {
  state.result = null;
  $("result-empty").hidden = false; $("result-content").hidden = true; $("result-loading").hidden = true;
  $("result-status").textContent = "READY WHEN YOU ARE"; $("result-status").className = "result-status";
  $("result-time").textContent = "";
  $("copy-result").disabled = true; $("speak-result").disabled = true;
  $("model-details").hidden = true; $("model-details").open = false;
  $("empty-description").textContent = state.source === "webcam" ? "Enable your camera and start a sign when you’re ready." : state.mode === "sentences" ? "Choose a sign clip and select Translate sign to begin." : "Choose a photo and select Translate sign to begin.";
}
function loadingResult() {
  $("result-empty").hidden = true; $("result-content").hidden = true; $("result-loading").hidden = false;
  $("result-status").textContent = "PROCESSING YOUR SIGN"; $("result-status").className = "result-status";
  $("copy-result").disabled = true; $("speak-result").disabled = true; $("model-details").hidden = true;
}
function detailRow(label, value) {
  const row = document.createElement("div"); row.className = "detail-row";
  const key = document.createElement("span"); key.textContent = label;
  const val = document.createElement("span"); val.textContent = value;
  row.append(key, val); return row;
}
function showResult(result, elapsed = null) {
  state.result = result;
  $("result-empty").hidden = true; $("result-loading").hidden = true; $("result-content").hidden = false;
  const matched = Boolean(result.label);
  $("result-label").textContent = result.label || "Let’s try that again.";
  $("result-label").classList.toggle("letter-result", matched && state.mode === "letters");
  $("result-message").textContent = result.message;
  $("result-status").textContent = result.status === "recognized" ? "SIGN RECOGNIZED" : result.status === "uncertain" ? "POSSIBLE MATCH · PLEASE CONFIRM" : "NO CLEAR MATCH";
  $("result-status").className = "result-status " + (result.status === "recognized" ? "success" : "uncertain");
  $("copy-result").disabled = !matched;
  $("speak-result").disabled = !matched || !("speechSynthesis" in window);
  $("result-time").textContent = elapsed ? (elapsed / 1000).toFixed(1) + "s" : "";
  const details = result.details;
  $("model-details").hidden = !details;
  $("details-content").replaceChildren();
  if (details) {
    $("details-content").append(detailRow("Model", details.model), detailRow("Model score", (details.score * 100).toFixed(1) + "%"));
    if (details.frames) $("details-content").append(detailRow("Frames used", details.frames));
    if (details.agreement !== undefined) $("details-content").append(detailRow("Frame agreement", (details.agreement * 100).toFixed(0) + "%"));
    if (details.alternatives) details.alternatives.forEach(item => $("details-content").append(detailRow(item.label, (item.score * 100).toFixed(1) + "%")));
    const note = document.createElement("p");
    note.textContent = details.note || "The model score is evidence for this match, not a guarantee of accuracy.";
    $("details-content").append(note);
  }
  if (state.source === "upload" && result.details?.boxes?.length) drawDetections(result.details.boxes);
}
function clearFile() {
  if (state.previewURL) URL.revokeObjectURL(state.previewURL);
  state.previewURL = null; state.file = null; $("file-input").value = "";
  $("preview-video").pause(); $("preview-video").removeAttribute("src"); $("preview-video").load();
  $("preview-image").removeAttribute("src");
  $("file-preview").hidden = true; $("dropzone").hidden = false; $("detection-overlay").hidden = true;
  $("upload-ready").lastChild.textContent = " Your preview appears before translation";
  $("translate-button").disabled = true;
}
function chooseFile(file) {
  if (state.busy || !file) return;
  showError();
  const extension = file.name.split(".").pop().toLowerCase();
  const isVideo = state.mode === "sentences";
  if (!(isVideo ? ["mp4", "webm", "mov", "avi"] : ["jpg", "jpeg", "png", "webp"]).includes(extension)) {
    showError(isVideo ? "A sentence needs a complete sign video. Choose MP4, WebM, MOV or AVI." : "Choose a JPG, PNG or WebP image. For a sign video, select Sentences."); return;
  }
  if (file.size > 30 * 1024 * 1024) { showError("Choose a file smaller than 30 MB."); return; }
  clearFile(); resetResult();
  state.file = file; state.previewURL = URL.createObjectURL(file);
  $("preview-image").hidden = isVideo; $("preview-video").hidden = !isVideo;
  const media = $(isVideo ? "preview-video" : "preview-image"); media.src = state.previewURL;
  if (!isVideo) media.onerror = () => { clearFile(); showError("This image could not be opened. Choose a readable JPG, PNG or WebP."); };
  else media.onerror = () => { showError("Your browser cannot preview this video. The server can still try to read it when you select Translate sign."); };
  $("file-name").textContent = file.name;
  $("file-size").textContent = file.size < 1024 * 1024 ? Math.max(1, Math.round(file.size / 1024)) + " KB" : (file.size / (1024 * 1024)).toFixed(1) + " MB";
  $("file-preview").hidden = false; $("dropzone").hidden = true;
  $("upload-ready").lastChild.textContent = " Ready. Confirm to translate.";
  $("translate-button").disabled = !available();
}
function drawDetections(boxes) {
  const image = $("preview-image"), canvas = $("detection-overlay");
  if (!image.naturalWidth) return;
  const stage = image.parentElement, width = stage.clientWidth, height = stage.clientHeight;
  const ratio = Math.min(width / image.naturalWidth, height / image.naturalHeight);
  const displayedWidth = image.naturalWidth * ratio, displayedHeight = image.naturalHeight * ratio;
  const offsetX = (width - displayedWidth) / 2, offsetY = (height - displayedHeight) / 2;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr; canvas.height = height * dpr; canvas.hidden = false;
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr); ctx.strokeStyle = "#67dae2"; ctx.lineWidth = 1.5; ctx.font = "14px Manrope, sans-serif";
  boxes.forEach(box => {
    const [x1, y1, x2, y2] = box.bounds;
    const x = offsetX + x1 * displayedWidth, y = offsetY + y1 * displayedHeight;
    ctx.strokeRect(x, y, (x2 - x1) * displayedWidth, (y2 - y1) * displayedHeight);
    ctx.fillStyle = "#67dae2"; ctx.fillRect(x, Math.max(0, y - 26), ctx.measureText(box.label).width + 16, 26);
    ctx.fillStyle = "#111829"; ctx.fillText(box.label, x + 8, Math.max(18, y - 8));
  });
}
function updateModeUI() {
  const model = currentModel(), isVideo = state.mode === "sentences", enabled = available();
  document.querySelectorAll(".mode-button").forEach(button => { const active = button.dataset.mode === state.mode; button.classList.toggle("active", active); button.setAttribute("aria-pressed", active); });
  $("input-format").textContent = state.source === "webcam" ? "CAMERA" : isVideo ? "VIDEO" : "PHOTO";
  $("input-title").textContent = state.source === "webcam" ? "Sign at your own pace" : isVideo ? "Upload a sign clip" : "Upload a photo";
  $("input-description").textContent = isVideo ? "Upload one complete sign. A still photo can’t capture a phrase." : "Upload a clear photo with your whole hand in view.";
  $("format-hint").textContent = isVideo ? "MP4, WebM, MOV or AVI · up to 30 MB · 45 seconds" : "JPG, PNG or WebP · up to 30 MB";
  $("file-input").accept = isVideo ? ".mp4,.webm,.mov,.avi" : ".jpg,.jpeg,.png,.webp";
  $("mode-note").textContent = model?.note || "Connecting to the recognition models…";
  $("samples-section").hidden = isVideo || state.mode === "words";
  $("upload-panel").hidden = state.source !== "upload" || !enabled;
  $("webcam-panel").hidden = state.source !== "webcam" || !enabled;
  $("model-unavailable").hidden = enabled || !state.models.length;
  $("unavailable-description").textContent = model?.reason || "";
  $("vocabulary-button").disabled = !enabled;
  $("camera-button").disabled = !enabled; $("dropzone").disabled = !enabled;
  $("translate-button").disabled = !state.file || !enabled;
  $("webcam-description").textContent = isVideo ? "Sign a complete phrase, then keep your hands still for six seconds." : "Hold a clear sign with your hands still for six seconds.";
}
async function changeMode(mode) {
  if (state.busy || state.mode === mode) return;
  await stopCamera(); clearFile(); state.mode = mode; showError(); resetResult(); updateModeUI();
}
async function translateUpload() {
  if (!state.file || state.busy || !available()) return;
  showError(); loadingResult(); setBusy(true);
  const form = new FormData(); form.append("file", state.file); form.append("mode", state.mode);
  const started = performance.now();
  try { showResult(await request("/api/upload", { method: "POST", body: form }), performance.now() - started); }
  catch (error) { showError(error.message); resetResult(); }
  finally { setBusy(false); }
}
async function enableCamera() {
  if (state.busy || !available()) return;
  showError();
  if (!navigator.mediaDevices?.getUserMedia) { showError("Camera access needs localhost or HTTPS. Open this app through its local server."); return; }
  setBusy(true);
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 }, facingMode: "user" }, audio: false });
    $("camera-video").srcObject = state.stream; $("camera-video").hidden = false;
    await $("camera-video").play();
    state.stream.getVideoTracks()[0].addEventListener("ended", () => { stopCamera(); showError("The camera disconnected. Enable it again to continue."); }, { once: true });
    $("camera-placeholder").hidden = true; $("stop-camera").hidden = false; $("camera-indicator").hidden = false;
    $("camera-state").textContent = "Camera ready"; $("camera-button").querySelector("span").textContent = "Start a sign";
  } catch (error) {
    await stopCamera();
    showError(error.name === "NotAllowedError" ? "Camera access was declined. Allow camera access in your browser, or upload a file." :
      error.name === "NotFoundError" ? "No camera was found. Connect a webcam, or upload a file." : "The camera could not start. Check that another app isn’t using it.");
  } finally { setBusy(false); }
}
async function startSign() {
  if (!state.stream || state.running || state.busy) return;
  showError(); resetResult();
  $("camera-button").disabled = true;
  $("vocabulary-button").disabled = true;
  try {
    const form = new FormData(); form.append("mode", state.mode);
    const session = await request("/api/live/session", { method: "POST", body: form });
    state.session = session.id; state.running = true; state.generation += 1;
    $("capture-progress").hidden = false; $("capture-message").textContent = "Bring your hands into view";
    $("capture-countdown").textContent = ""; $("pause-progress").style.width = "0%";
    $("camera-state").textContent = "Waiting for your hands"; $("camera-button").querySelector("span").textContent = "Signing…";
    $("result-status").textContent = "LISTENING TO YOUR HANDS";
    $("empty-description").textContent = "Take your time. The result appears after your six-second pause.";
    document.querySelectorAll(".mode-button").forEach(button => { button.disabled = true; });
    pumpFrame(state.generation);
  } catch (error) { $("camera-button").disabled = false; $("vocabulary-button").disabled = !available(); showError(error.message); }
}
async function pumpFrame(generation) {
  if (!state.running || generation !== state.generation || !state.stream) return;
  const started = performance.now();
  try {
    const video = $("camera-video");
    if (!video.videoWidth || video.readyState < 2) throw new Error("The camera is still warming up. Please start a sign again.");
    const canvas = document.createElement("canvas");
    const ratio = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * ratio); canvas.height = Math.round(video.videoHeight * ratio);
    // Raw camera orientation matches training; only the visible preview is mirrored.
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.78));
    if (!blob) throw new Error("The camera frame could not be captured. Start a sign again.");
    if (!state.running || generation !== state.generation) return;
    const form = new FormData(); form.append("file", blob, "frame.jpg");
    state.controller = new AbortController();
    const response = await request("/api/live/" + state.session + "/frame", { method: "POST", body: form, signal: state.controller.signal });
    if (!state.running || generation !== state.generation) return;
    const phase = response.phase;
    if (phase === "complete" || phase === "timeout") {
      showResult(response.result);
      $("camera-state").textContent = "Sign complete"; $("capture-message").textContent = "Ready for your next sign";
      $("capture-countdown").textContent = ""; $("pause-progress").style.width = "100%";
      await endSession();
      $("camera-button").querySelector("span").textContent = "Start another sign"; $("camera-button").disabled = false;
      document.querySelectorAll(".mode-button").forEach(button => { button.disabled = false; });
      return;
    }
    if (phase === "holding") {
      $("camera-state").textContent = "Hold still"; $("capture-message").textContent = "Keep your hands still to finish";
      $("capture-countdown").textContent = response.remaining.toFixed(1) + "s";
      $("pause-progress").style.width = ((6 - response.remaining) / 6 * 100) + "%";
    } else {
      $("camera-state").textContent = phase === "signing" ? "Capturing your sign" : "Hands out of view";
      $("capture-message").textContent = phase === "signing" ? "Sign naturally. Pause when you’re finished." : "Keep your hands visible so we can follow your sign";
      $("capture-countdown").textContent = ""; $("pause-progress").style.width = "0%";
    }
  } catch (error) {
    if (error.name === "AbortError" || generation !== state.generation) return;
    await endSession(); $("camera-button").disabled = false; $("camera-button").querySelector("span").textContent = "Start a sign";
    document.querySelectorAll(".mode-button").forEach(button => { button.disabled = false; });
    showError(error.message); resetResult(); return;
  }
  // Await each request. At most one frame is in flight; slow inference never queues video.
  if (state.running && generation === state.generation) state.timer = setTimeout(() => pumpFrame(generation), Math.max(0, 167 - (performance.now() - started)));
}
async function endSession() {
  state.running = false; state.generation += 1; clearTimeout(state.timer); state.controller?.abort(); state.controller = null;
  const id = state.session; state.session = null;
  $("vocabulary-button").disabled = state.busy || !available();
  if (id) await fetch("/api/live/" + id, { method: "DELETE", keepalive: true }).catch(() => {});
}
async function stopCamera() {
  await endSession();
  if (state.stream) state.stream.getTracks().forEach(track => track.stop());
  state.stream = null; $("camera-video").srcObject = null; $("camera-video").hidden = true;
  $("camera-placeholder").hidden = false; $("stop-camera").hidden = true; $("camera-indicator").hidden = true; $("capture-progress").hidden = true;
  $("camera-button").querySelector("span").textContent = "Enable camera"; $("camera-button").disabled = !available();
  document.querySelectorAll(".mode-button").forEach(button => { button.disabled = state.busy; });
}
const normalizeSign = value => value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/đ/gi, "d").toLowerCase().replace(/[.!?]/g, "").trim();
function validDemo(demo) {
  if (!demo || typeof demo.label !== "string" || typeof demo.source_title !== "string") return false;
  try {
    if (new URL(demo.source_url).protocol !== "https:") return false;
    if (demo.watch_url && new URL(demo.watch_url).protocol !== "https:") return false;
  } catch { return false; }
  if (demo.provider === "local") return /^\/assets\/signs\/[a-z0-9/-]+\.mp4$/.test(demo.video_url);
  return demo.provider === "youtube" && /^[A-Za-z0-9_-]{11}$/.test(demo.video_id) &&
    [demo.start, demo.end].every(value => value === undefined || (Number.isInteger(value) && value >= 0)) &&
    (demo.end === undefined || demo.end > (demo.start || 0));
}
const demoForSign = label => state.demonstrations.find(demo => normalizeSign(demo.label) === normalizeSign(label));
function renderVocabulary() {
  const model = currentModel(), query = normalizeSign($("vocabulary-search").value);
  const sentenceMode = state.mode === "sentences";
  const demoCount = (model?.labels || []).filter(label => demoForSign(label)).length;
  $("demo-library-note").hidden = !sentenceMode;
  $("demo-library-note").textContent = "Choose a phrase to see how to sign it. " + demoCount + " of " + (model?.count || 0) + " phrases have a reference video.";
  $("video-filter").hidden = !sentenceMode || !demoCount;
  $("video-filter").textContent = "With videos · " + demoCount;
  $("video-filter").setAttribute("aria-pressed", state.videosOnly);
  const labels = (model?.labels || []).filter(label => normalizeSign(label).includes(query) && (!sentenceMode || !state.videosOnly || demoForSign(label)));
  if (sentenceMode) labels.sort((a, b) => Number(Boolean(demoForSign(b))) - Number(Boolean(demoForSign(a))));
  $("vocabulary-list").replaceChildren();
  labels.forEach(label => {
    if (!sentenceMode) {
      const chip = document.createElement("span"); chip.className = "vocabulary-chip" + (state.mode === "letters" ? " letter-chip" : ""); chip.textContent = label; $("vocabulary-list").append(chip); return;
    }
    const demo = demoForSign(label), button = document.createElement("button");
    button.type = "button"; button.className = "vocabulary-sign" + (demo ? " has-video" : "");
    const title = document.createElement("strong"), hint = document.createElement("span");
    title.textContent = label; hint.textContent = demo ? "Watch video" : "Video unavailable";
    button.append(title, hint); button.setAttribute("aria-label", label + " · " + hint.textContent);
    button.addEventListener("click", () => showSignDemo(label)); $("vocabulary-list").append(button);
  });
  $("vocabulary-list").classList.toggle("sentence-list", sentenceMode);
  if (!labels.length) { const p = document.createElement("p"); p.textContent = "No supported signs match your search."; $("vocabulary-list").append(p); }
}
function clearDemoPlayer() {
  // Removing the player stops audio and releases it when navigating or closing.
  $("demo-player").querySelector("video")?.pause(); $("demo-player").replaceChildren();
}
function showVocabularyList() {
  clearDemoPlayer(); $("demo-detail").hidden = true; $("vocabulary-browser").hidden = false;
  $("vocabulary-dialog").classList.remove("showing-demo");
  renderVocabulary();
  if (state.selectedSign) {
    [...$("vocabulary-list").querySelectorAll("button")].find(button => button.querySelector("strong").textContent === state.selectedSign)?.focus();
  }
}
function showSignDemo(label) {
  if (!currentModel()?.labels.includes(label) || state.mode !== "sentences") return;
  state.selectedSign = label; const demo = demoForSign(label);
  clearDemoPlayer(); $("vocabulary-browser").hidden = true; $("demo-detail").hidden = false;
  $("vocabulary-dialog").classList.add("showing-demo"); $("demo-title").textContent = label;
  $("demo-unavailable").hidden = Boolean(demo); $("demo-player").hidden = !demo;
  $("demo-credit").hidden = !demo; $("demo-open-source").hidden = !demo;
  $("demo-guidance").hidden = !demo; $("practice-sign").hidden = !demo;
  if (demo) {
    if (demo.provider === "youtube" && /^[A-Za-z0-9_-]{11}$/.test(demo.video_id)) {
      const iframe = document.createElement("iframe");
      iframe.src = "https://www.youtube-nocookie.com/embed/" + demo.video_id + "?rel=0&playsinline=1" + (demo.start ? "&start=" + demo.start : "") + (demo.end ? "&end=" + demo.end : "");
      iframe.title = "How to sign: " + label;
      iframe.allow = "encrypted-media; fullscreen; picture-in-picture"; iframe.allowFullscreen = true;
      iframe.referrerPolicy = "strict-origin-when-cross-origin"; $("demo-player").append(iframe);
    } else if (demo.provider === "local" && /^\/assets\/signs\/[a-z0-9/-]+\.mp4$/.test(demo.video_url)) {
      const video = document.createElement("video"); video.src = demo.video_url; video.controls = true; video.playsInline = true; video.preload = "metadata";
      video.setAttribute("aria-label", "How to sign: " + label); $("demo-player").append(video);
    }
    $("demo-source").textContent = demo.source_title; $("demo-source").href = demo.source_url;
    $("demo-signer").textContent = demo.signer ? " · " + demo.signer : "";
    $("demo-open-source").href = demo.watch_url || demo.source_url;
    $("demo-reference-note").textContent = demo.note || "A learning reference. Regional signing and the model’s training examples may differ.";
  }
  $("demo-back").focus(); $("vocabulary-dialog").scrollTop = 0;
}
function openVocabulary() {
  const model = currentModel(); if (!model || state.running || state.busy) return;
  state.selectedSign = null; state.videosOnly = false;
  $("vocabulary-title").textContent = model.title + " · " + model.count + " supported signs";
  $("vocabulary-note").textContent = model.note; $("vocabulary-search").value = "";
  showVocabularyList(); $("vocabulary-dialog").showModal();
}
async function loadSamples() {
  // Image selection prepares a preview only; the Translate button runs inference.
  for (const sample of samples) {
    const button = document.createElement("button"); button.type = "button"; button.className = "sample-button";
    button.setAttribute("aria-label", "Preview dataset " + sample.title.toLowerCase());
    const image = document.createElement("img"); image.src = "/assets/samples/" + sample.file; image.alt = ""; image.loading = "lazy";
    const text = document.createElement("div"), title = document.createElement("strong"), hint = document.createElement("span");
    title.textContent = sample.title; hint.textContent = "Preview sign"; text.append(title, hint); button.append(image, text);
    image.onerror = () => { button.remove(); if (!$("sample-list").children.length) $("samples-section").hidden = true; };
    button.addEventListener("click", async () => {
      if (state.busy) return;
      try { const response = await fetch(image.src); if (!response.ok) throw new Error("The sample image could not load.");
        chooseFile(new File([await response.blob()], sample.file, { type: "image/jpeg" })); }
      catch (error) { showError(error.message); }
    });
    $("sample-list").append(button);
  }
}
document.querySelectorAll(".mode-button").forEach(button => button.addEventListener("click", () => changeMode(button.dataset.mode)));
$("dropzone").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", event => chooseFile(event.target.files[0]));
$("dropzone").addEventListener("dragover", event => { event.preventDefault(); if (!state.busy) $("dropzone").classList.add("drag-over"); });
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("drag-over"));
$("dropzone").addEventListener("drop", event => { event.preventDefault(); $("dropzone").classList.remove("drag-over"); if (available()) chooseFile(event.dataTransfer.files[0]); });
$("remove-file").addEventListener("click", () => { clearFile(); resetResult(); showError(); });
$("translate-button").addEventListener("click", translateUpload);
$("camera-button").addEventListener("click", () => state.stream ? startSign() : enableCamera());
$("stop-camera").addEventListener("click", stopCamera);
$("copy-result").addEventListener("click", async () => { try { await navigator.clipboard.writeText(state.result.label); toast("Translation copied."); } catch { toast("Copy isn’t available here. Select the translation to copy it."); } });
$("speak-result").addEventListener("click", () => {
  if (!state.result?.label) return;
  const voices = speechSynthesis.getVoices(), voice = voices.find(v => v.lang.toLowerCase().startsWith("vi"));
  if (voices.length && !voice) { toast("A Vietnamese voice isn’t installed in this browser."); return; }
  speechSynthesis.cancel(); const utterance = new SpeechSynthesisUtterance(state.result.label);
  utterance.lang = "vi-VN"; utterance.rate = 0.9; if (voice) utterance.voice = voice;
  utterance.onerror = () => toast("The voice could not play. You can still read or copy the text.");
  speechSynthesis.speak(utterance);
});
$("vocabulary-button").addEventListener("click", openVocabulary);
$("close-dialog").addEventListener("click", () => $("vocabulary-dialog").close());
$("vocabulary-dialog").addEventListener("close", clearDemoPlayer);
$("demo-back").addEventListener("click", showVocabularyList);
$("demo-other-signs").addEventListener("click", () => { state.videosOnly = true; $("vocabulary-search").value = ""; showVocabularyList(); $("video-filter").focus(); });
$("video-filter").addEventListener("click", () => { state.videosOnly = !state.videosOnly; renderVocabulary(); });
$("practice-sign").addEventListener("click", () => {
  $("vocabulary-dialog").close();
  toast(state.source === "webcam" ? "Ready to try " + state.selectedSign + (state.stream ? " Start a sign when you’re ready." : " Enable your camera, then start a sign.") : "Ready to try " + state.selectedSign + " Upload your sign clip, then confirm to translate.");
  $(state.source === "webcam" ? "camera-button" : "dropzone").focus();
});
$("vocabulary-dialog").addEventListener("click", event => { if (event.target === $("vocabulary-dialog")) { const rect = event.target.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close(); } });
$("vocabulary-search").addEventListener("input", renderVocabulary);
window.addEventListener("resize", () => { if (state.source === "upload" && state.result?.details?.boxes) drawDetections(state.result.details.boxes); });
document.addEventListener("visibilitychange", () => { if (document.hidden && state.stream) { stopCamera(); toast("Camera paused while this tab was away. Enable it when you’re ready."); } });
window.addEventListener("pagehide", () => { state.stream?.getTracks().forEach(track => track.stop()); if (state.session) fetch("/api/live/" + state.session, { method: "DELETE", keepalive: true }).catch(() => {}); });
async function initialize() {
  resetResult(); setBusy(true);
  try {
    const data = await request("/api/models"); state.models = data.models;
    // A missing reference catalog must not prevent recognition from starting.
    try { const catalog = await request("/assets/signs/sentence-videos.json"); state.demonstrations = Array.isArray(catalog.demonstrations) ? catalog.demonstrations.filter(validDemo) : []; }
    catch { state.demonstrations = []; }
    const active = state.models.filter(model => model.available);
    $("service-status").lastChild.textContent = " " + active.length + " models ready";
    if (!active.length) throw new Error("No recognition models are available. Check the server setup.");
    if (!available()) state.mode = active[0].id;
    $("words-badge").textContent = state.models.find(model => model.id === "words")?.available ? "" : "SETUP";
    $("words-badge").hidden = state.models.find(model => model.id === "words")?.available || false;
    updateModeUI(); if (state.source === "upload") loadSamples();
  } catch (error) { $("service-status").lastChild.textContent = " Models offline"; $("service-status").classList.add("offline"); showError(error.message); }
  finally { setBusy(false); updateModeUI(); }
}
initialize();
