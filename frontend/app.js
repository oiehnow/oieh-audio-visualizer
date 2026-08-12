/* Audio Visualizer frontend — vanilla JS, no build step.
   Preview protocol: client-clocked pull over WS, one frame in flight.
   Settings live in localStorage and are pushed into BOTH state and the DOM
   controls at load, so the preview, the controls, and the export can never
   silently disagree (the WYSIWYG contract, client side). */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const ACCENT = "#3ce6a8";
  const SEEK_DIM = "#3a3a44";
  const JOB_KEY = "av_job";
  const SETTINGS_KEY = "av_settings";

  const DEFAULT_SETTINGS = {
    style: "bars",
    color1: "#3ce6a8",
    color2: "#1e5cff",
    gradient: true,
    bg_mode: "solid",
    bg_color: "#101014",
    bg_image_id: null,
    sensitivity: 1.0,
    width: 1920,
    height: 1080,
    fps: 60,
    quality: "balanced",
    codec: "av1_nvenc",
  };

  const state = {
    audioId: null,
    filename: null,
    duration: 0,
    overview: null,
    playing: false,
    exporting: false,
    jobId: null,
    settings: { ...DEFAULT_SETTINGS },
  };

  // ---- settings persistence ----------------------------------------------
  function loadSettings() {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null"); } catch { /* defaults */ }
    if (!saved || typeof saved !== "object") return;
    for (const k of Object.keys(DEFAULT_SETTINGS)) {
      if (k in saved) state.settings[k] = saved[k];
    }
  }

  function saveSettings() {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings)); } catch { /* quota */ }
  }

  function resetSettings() {
    state.settings = { ...DEFAULT_SETTINGS };
    localStorage.removeItem(SETTINGS_KEY);
    syncControlsFromState();
    applySettings({ sizeChanged: true });
  }

  // ---- elements -----------------------------------------------------------
  const player = $("#player");
  const canvas = $("#preview");
  const cctx = canvas.getContext("2d");
  const stage = $("#stage");
  const previewBox = $("#previewBox");
  const placeholder = $("#placeholder");
  const exportOverlay = $("#exportOverlay");
  const seekbar = $("#seekbar");
  const playBtn = $("#playBtn");
  const renderBtn = $("#renderBtn");
  const cancelBtn = $("#cancelBtn");
  const progressWrap = $("#progressWrap");
  const progressFill = $("#progressFill");
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  const bgFileInput = $("#bgFileInput");

  // ---- helpers ------------------------------------------------------------
  const fmtTime = (s) => {
    if (!isFinite(s) || s < 0) s = 0;
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return `${m}:${String(sec).padStart(2, "0")}`;
  };
  const fmtSize = (b) => b >= 1 << 30 ? `${(b / (1 << 30)).toFixed(2)} GB` : `${(b / (1 << 20)).toFixed(1)} MB`;

  let toastTimer = null;
  function toast(msg, ms = 4000) {
    const el = $("#toast");
    el.textContent = msg;
    el.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), ms);
  }

  // ---- controls <- state sync --------------------------------------------
  // The DOM is written FROM state (never trusted as a source), so browser
  // form restoration or a stale cached page can't desync what the user sees
  // from what the preview and export actually use.
  function syncControlsFromState() {
    const s = state.settings;
    document.querySelectorAll(".style-thumb").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.style === s.style)));

    $("#color1").value = s.color1;
    $("#color2").value = s.color2;
    $("#gradient").checked = s.gradient;
    $("#color2").disabled = !s.gradient;

    const bgRadio = document.querySelector(`input[name="bgmode"][value="${s.bg_mode}"]`);
    if (bgRadio) bgRadio.checked = true;
    const transparent = s.bg_mode === "transparent";
    $("#bgColor").value = s.bg_color;
    $("#bgColor").disabled = s.bg_mode !== "solid";
    $("#codec").disabled = transparent;
    $("#codecRow").classList.toggle("hidden", transparent);
    $("#alphaHint").classList.toggle("hidden", !transparent);
    const hasImage = !!s.bg_image_id && s.bg_mode === "image";
    $("#bgThumbRow").classList.toggle("hidden", !hasImage);
    const thumb = $("#bgThumb");
    const want = s.bg_image_id ? `/api/background/${s.bg_image_id}` : "";
    if (thumb.getAttribute("src") !== want) thumb.src = want;

    const resRadio = document.querySelector(`input[name="res"][value="${s.width}x${s.height}"]`);
    const customRes = !resRadio;
    (resRadio || document.querySelector('input[name="res"][value="custom"]')).checked = true;
    $("#customW").value = s.width;
    $("#customH").value = s.height;
    $("#customW").disabled = $("#customH").disabled = !customRes;

    const fpsRadio = document.querySelector(`input[name="fps"][value="${s.fps}"]`);
    if (fpsRadio) fpsRadio.checked = true;
    $("#quality").value = s.quality;
    $("#codec").value = s.codec;
    $("#sensitivity").value = s.sensitivity;
    $("#sensValue").textContent = `${(+s.sensitivity).toFixed(2)}×`;
  }

  // ---- WebSocket manager --------------------------------------------------
  let ws = null;
  let wsReady = false;
  let backoff = 500;
  let pendingResolve = null;
  let pendingTimer = null;

  function resolvePending(value) {
    if (!pendingResolve) return;
    const r = pendingResolve;
    pendingResolve = null;
    clearTimeout(pendingTimer);
    r(value);
  }

  function connectWS() {
    ws = new WebSocket(`ws://${location.host}/ws/preview`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      backoff = 500;
      wsReady = true;
      ws.send(JSON.stringify({ type: "settings", settings: state.settings }));
      if (state.audioId) ws.send(JSON.stringify({ type: "use_audio", audio_id: state.audioId }));
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        handleControl(JSON.parse(ev.data));
      } else if (pendingResolve) {
        resolvePending(ev.data);
      } else {
        // frame that outlived its 1 s request timeout: still the newest content —
        // drawing it beats leaving a stale frame on the canvas
        drawFrame(ev.data);
      }
    };
    ws.onclose = () => {
      wsReady = false;
      resolvePending(null);
      setTimeout(connectWS, backoff);
      backoff = Math.min(backoff * 2, 4000);
    };
    ws.onerror = () => ws.close();
  }

  function handleControl(msg) {
    if (msg.type === "ready") {
      refreshFrame();
    } else if (msg.type === "export_active") {
      resolvePending(null);
      player.pause();
      showExportOverlay(true);
    } else if (msg.type === "bg_missing") {
      resolvePending(null);
      handleBgMissing();
    } else if (msg.type === "error") {
      resolvePending(null);
      if (typeof msg.message === "string" && msg.message.startsWith("invalid settings")) {
        resetSettings();  // corrupted saved settings: recover instead of looping
        toast("저장된 설정이 유효하지 않아 기본값으로 복원했습니다");
        return;
      }
      toast(msg.message || "오류가 발생했습니다");
    }
  }

  function handleBgMissing() {
    state.settings.bg_mode = "solid";
    state.settings.bg_image_id = null;
    saveSettings();
    syncControlsFromState();
    // send the corrected settings IMMEDIATELY: while playing, previewLoop issues
    // frame requests every ~33 ms and each bg_missing reply would reset the 80 ms
    // applySettings debounce — the settings message would never get through
    if (wsReady) ws.send(JSON.stringify({ type: "settings", settings: state.settings }));
    if (!state.playing && state.audioId) refreshFrame();
    toast("배경 이미지가 만료되었습니다 — 다시 업로드해주세요");
  }

  function requestFrame(t) {
    return new Promise((resolve) => {
      if (!wsReady || !state.audioId || pendingResolve) { resolve(null); return; }
      pendingResolve = resolve;
      pendingTimer = setTimeout(() => resolvePending(null), 1000);
      ws.send(JSON.stringify({ type: "frame", t }));
    });
  }

  async function drawFrame(buf) {
    const bmp = await createImageBitmap(new Blob([buf], { type: "image/jpeg" }));
    if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
      canvas.width = bmp.width;
      canvas.height = bmp.height;
    }
    cctx.drawImage(bmp, 0, 0);
    bmp.close();
    placeholder.classList.add("hidden");
    if (!state.exporting) showExportOverlay(false); // frames only flow when no export runs
  }

  // ---- paused-preview frame pump -----------------------------------------
  // The preview must always converge to (current time, current settings).
  // A dirty flag + retry loop guarantees that: a request that gets dropped
  // (another in flight, export running, reconnect) is re-issued instead of
  // silently leaving a stale frame on the canvas.
  let frameInFlight = false;
  let frameDirty = false;

  async function refreshFrame() {
    frameDirty = true;
    if (frameInFlight || state.playing) return;  // previewLoop drives while playing
    frameInFlight = true;
    while (frameDirty && !state.playing) {
      frameDirty = false;
      const jpeg = await requestFrame(player.currentTime || 0);
      if (jpeg) await drawFrame(jpeg);
      else if (frameDirty) await sleep(150);  // dropped: back off briefly, then retry
    }
    frameInFlight = false;
  }

  let previewLoopRunning = false;
  async function previewLoop() {
    if (previewLoopRunning) return;  // fast pause->play toggles must not stack loops
    previewLoopRunning = true;
    try {
      while (state.playing) {
        const t0 = performance.now();
        const jpeg = await requestFrame(player.currentTime);
        if (jpeg) await drawFrame(jpeg);
        updateTransport();
        const dt = performance.now() - t0;
        if (dt < 1000 / 30) await sleep(1000 / 30 - dt);
      }
    } finally {
      previewLoopRunning = false;
    }
    refreshFrame();  // settle on the exact pause position
  }

  // ---- export overlay (shown while any export blocks the preview) ---------
  let overlayRetry = null;
  function showExportOverlay(on) {
    exportOverlay.classList.toggle("hidden", !on);
    if (on && !state.exporting && !overlayRetry) {
      // export started from another tab: probe until the preview channel frees up
      overlayRetry = setInterval(() => {
        if (exportOverlay.classList.contains("hidden")) { clearInterval(overlayRetry); overlayRetry = null; return; }
        refreshFrame();
      }, 2000);
    }
    if (!on && overlayRetry) { clearInterval(overlayRetry); overlayRetry = null; }
  }

  // ---- preview letterbox --------------------------------------------------
  function fitPreviewBox() {
    const ar = state.settings.width / state.settings.height;
    const sw = stage.clientWidth - 24, sh = stage.clientHeight - 24;
    let w = sw, h = w / ar;
    if (h > sh) { h = sh; w = h * ar; }
    previewBox.style.width = `${Math.max(2, Math.floor(w))}px`;
    previewBox.style.height = `${Math.max(2, Math.floor(h))}px`;
  }

  // ---- settings wiring ----------------------------------------------------
  let settingsTimer = null;
  function applySettings({ sizeChanged = false } = {}) {
    if (sizeChanged) fitPreviewBox();
    saveSettings();
    clearTimeout(settingsTimer);
    settingsTimer = setTimeout(() => {
      if (wsReady) ws.send(JSON.stringify({ type: "settings", settings: state.settings }));
      if (state.audioId) refreshFrame();
    }, 80);
  }

  function initControls() {
    // style picker
    const thumbs = [...document.querySelectorAll(".style-thumb")];
    thumbs.forEach((btn) => btn.addEventListener("click", () => {
      state.settings.style = btn.dataset.style;
      thumbs.forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      applySettings();
    }));

    // colors
    $("#color1").addEventListener("input", (e) => { state.settings.color1 = e.target.value; applySettings(); });
    $("#color2").addEventListener("input", (e) => { state.settings.color2 = e.target.value; applySettings(); });
    $("#gradient").addEventListener("input", (e) => {
      state.settings.gradient = e.target.checked;
      $("#color2").disabled = !e.target.checked;
      applySettings();
    });

    // background
    document.querySelectorAll('input[name="bgmode"]').forEach((r) => r.addEventListener("change", (e) => {
      bgModeSeq += 1;
      if (e.target.value === "image" && !state.settings.bg_image_id) {
        // no image yet: keep the previous mode until an upload succeeds
        syncControlsFromState();
        bgFileInput.click();
        return;
      }
      state.settings.bg_mode = e.target.value;
      syncControlsFromState();
      applySettings();
    }));
    $("#bgColor").addEventListener("input", (e) => { state.settings.bg_color = e.target.value; applySettings(); });
    $("#bgImageBtn").addEventListener("click", () => bgFileInput.click());
    bgFileInput.addEventListener("change", () => {
      if (bgFileInput.files.length) uploadBackground(bgFileInput.files[0]);
      bgFileInput.value = "";
    });

    // resolution
    const customW = $("#customW"), customH = $("#customH");
    const clampEven = (v) => Math.min(4096, Math.max(16, Math.round(v / 2) * 2));
    document.querySelectorAll('input[name="res"]').forEach((r) => r.addEventListener("change", (e) => {
      const custom = e.target.value === "custom";
      customW.disabled = customH.disabled = !custom;
      if (custom) {
        state.settings.width = clampEven(+customW.value || 1920);
        state.settings.height = clampEven(+customH.value || 1080);
      } else {
        const [w, h] = e.target.value.split("x").map(Number);
        state.settings.width = w;
        state.settings.height = h;
        customW.value = w;
        customH.value = h;
      }
      applySettings({ sizeChanged: true });
    }));
    [customW, customH].forEach((inp) => inp.addEventListener("change", () => {
      inp.value = clampEven(+inp.value || 16);
      state.settings.width = +customW.value;
      state.settings.height = +customH.value;
      applySettings({ sizeChanged: true });
    }));

    // fps / quality / codec
    document.querySelectorAll('input[name="fps"]').forEach((r) => r.addEventListener("change", (e) => {
      state.settings.fps = +e.target.value;
      applySettings();
    }));
    $("#quality").addEventListener("change", (e) => { state.settings.quality = e.target.value; applySettings(); });
    $("#codec").addEventListener("change", (e) => { state.settings.codec = e.target.value; applySettings(); });

    // sensitivity
    $("#sensitivity").addEventListener("input", (e) => {
      state.settings.sensitivity = +e.target.value;
      $("#sensValue").textContent = `${(+e.target.value).toFixed(2)}×`;
      applySettings();
    });

    // volume
    $("#volume").addEventListener("input", (e) => { player.volume = +e.target.value; });
  }

  // ---- background image upload -------------------------------------------
  let bgModeSeq = 0;  // bumped on every explicit bg-mode choice; an in-flight
                      // upload must not override a mode the user picked later
  function uploadBackground(file) {
    if (file.size > 64 * 2 ** 20) { toast("이미지가 64 MB를 초과합니다"); return; }
    const seq = bgModeSeq;
    const fd = new FormData();
    fd.append("file", file);
    fetch("/api/background", { method: "POST", body: fd })
      .then(async (res) => {
        if (!res.ok) {
          const d = await res.json().catch(() => null);
          throw new Error(d?.detail || "배경 이미지 업로드에 실패했습니다");
        }
        return res.json();
      })
      .then((d) => {
        state.settings.bg_image_id = d.bg_id;
        if (bgModeSeq === seq) state.settings.bg_mode = "image";
        syncControlsFromState();
        applySettings();
      })
      .catch((e) => toast(e.message || "배경 이미지 업로드에 실패했습니다"));
  }

  // ---- upload -------------------------------------------------------------
  function setDzState(mode) {
    $("#dzIdle").classList.toggle("hidden", mode !== "idle");
    $("#dzBusy").classList.toggle("hidden", mode !== "busy");
    $("#dzLoaded").classList.toggle("hidden", mode !== "loaded");
  }

  function uploadFile(file) {
    setDzState("busy");
    $("#dzStatus").textContent = "업로드 중… 0%";
    const fd = new FormData();
    fd.append("file", file);
    fd.append("settings", JSON.stringify(state.settings));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/audio");
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round((e.loaded / e.total) * 100);
      $("#dzStatus").textContent = pct >= 100 ? "분석 중…" : `업로드 중… ${pct}%`;
    };
    xhr.onload = () => {
      if (xhr.status !== 200) {
        let detail = "업로드에 실패했습니다";
        try { detail = JSON.parse(xhr.responseText).detail || detail; } catch { /* keep default */ }
        toast(detail);
        setDzState(state.audioId ? "loaded" : "idle");
        return;
      }
      const res = JSON.parse(xhr.responseText);
      state.audioId = res.audio_id;
      state.filename = res.filename;
      state.duration = res.duration;
      state.overview = res.overview;
      state.playing = false;
      player.src = `/api/audio/${res.audio_id}/file`;
      $("#dzName").textContent = res.filename;
      $("#dzMeta").textContent = `${fmtTime(res.duration)} · ${(res.sample_rate / 1000).toFixed(1)} kHz`;
      $("#totalTime").textContent = fmtTime(res.duration);
      $("#curTime").textContent = "0:00";
      setDzState("loaded");
      playBtn.disabled = false;
      renderBtn.disabled = state.exporting;
      drawSeekbar();
      if (wsReady) ws.send(JSON.stringify({ type: "use_audio", audio_id: state.audioId }));
      // first frame arrives via the "ready" control message
    };
    xhr.onerror = () => {
      toast("업로드에 실패했습니다 (네트워크 오류)");
      setDzState(state.audioId ? "loaded" : "idle");
    };
    xhr.send(fd);
  }

  function initUpload() {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) uploadFile(fileInput.files[0]);
      fileInput.value = "";
    });
    dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag");
      if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });
    // keep stray drops from navigating away
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => e.preventDefault());
  }

  // ---- transport + seekbar ------------------------------------------------
  function updateTransport() {
    $("#curTime").textContent = fmtTime(player.currentTime);
    drawSeekbar();
  }

  function drawSeekbar() {
    const dpr = devicePixelRatio || 1;
    const w = seekbar.clientWidth, h = seekbar.clientHeight;
    if (!w) return;
    if (seekbar.width !== Math.round(w * dpr)) {
      seekbar.width = Math.round(w * dpr);
      seekbar.height = Math.round(h * dpr);
    }
    const g = seekbar.getContext("2d");
    const W = seekbar.width, H = seekbar.height, mid = H / 2;
    g.clearRect(0, 0, W, H);
    const frac = state.duration ? player.currentTime / state.duration : 0;
    const ov = state.overview;
    const n = ov ? ov.length : 0;
    for (let x = 0; x < W; x++) {
      let lo = -0.04, hi = 0.04;
      if (n) {
        const b = ov[Math.min(n - 1, Math.floor((x / W) * n))];
        lo = b[0]; hi = b[1];
      }
      const y1 = mid - hi * mid * 0.92;
      const y2 = mid - lo * mid * 0.92;
      g.fillStyle = x / W <= frac && n ? ACCENT : SEEK_DIM;
      g.fillRect(x, y1, 1, Math.max(1, y2 - y1));
    }
    if (n) { // playhead
      const pw = Math.max(2, Math.round(1.5 * dpr));
      g.fillStyle = "#ffffff";
      g.fillRect(Math.max(0, Math.min(W - pw, frac * W)), 0, pw, H);
    }
  }

  function initTransport() {
    playBtn.addEventListener("click", () => {
      if (player.paused) player.play(); else player.pause();
    });
    player.addEventListener("play", () => {
      state.playing = true;
      playBtn.textContent = "❚❚";
      previewLoop();
    });
    player.addEventListener("pause", () => { state.playing = false; playBtn.textContent = "▶"; });
    player.addEventListener("ended", () => { state.playing = false; playBtn.textContent = "▶"; updateTransport(); });
    player.addEventListener("seeked", () => { if (!state.playing) refreshFrame(); });
    player.addEventListener("timeupdate", () => { if (!state.playing) updateTransport(); });

    let scrubbing = false;
    const seekTo = (e) => {
      if (!state.duration) return;
      const rect = seekbar.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      player.currentTime = frac * state.duration;
      updateTransport();
      if (!state.playing) refreshFrame();  // live scrub preview (pump coalesces)
    };
    seekbar.addEventListener("pointerdown", (e) => {
      if (!state.audioId) return;
      scrubbing = true;
      seekbar.setPointerCapture(e.pointerId);
      seekTo(e);
    });
    seekbar.addEventListener("pointermove", (e) => { if (scrubbing) seekTo(e); });
    seekbar.addEventListener("pointerup", () => { scrubbing = false; });
  }

  // ---- export -------------------------------------------------------------
  let es = null;

  function setExporting(on) {
    state.exporting = on;
    renderBtn.disabled = on || !state.audioId;
    progressWrap.classList.toggle("hidden", !on);
    showExportOverlay(on && !!state.audioId);
    if (on) player.pause();
    if (!on) {
      progressFill.style.width = "0%";
      $("#progPct").textContent = "0%";
      $("#progFps").textContent = "";
      $("#progEta").textContent = "";
    }
  }

  function renderProgress(snap) {
    progressFill.style.width = `${snap.percent || 0}%`;
    $("#progPct").textContent = `${(snap.percent || 0).toFixed(1)}%`;
    $("#progFps").textContent = snap.render_fps ? `${snap.render_fps.toFixed(0)} fps` : "";
    $("#progEta").textContent = snap.eta_s != null ? `남은 시간 ${fmtTime(snap.eta_s)}` : "";
  }

  function finishExport() {
    if (es) { es.close(); es = null; }
    state.jobId = null;
    localStorage.removeItem(JOB_KEY);
    setExporting(false);
    if (state.audioId) refreshFrame();
  }

  function openEvents(jobId) {
    if (es) es.close();
    es = new EventSource(`/api/export/${jobId}/events`);
    es.addEventListener("progress", (ev) => renderProgress(JSON.parse(ev.data)));
    es.addEventListener("done", (ev) => {
      const d = JSON.parse(ev.data);
      finishExport();
      toast(`렌더 완료: ${d.file.name}`);
      refreshRenders();
    });
    es.addEventListener("cancelled", () => {
      finishExport();
      toast("렌더링이 취소되었습니다");
    });
    es.addEventListener("error", (ev) => {
      // EventSource fires bare "error" events on connection drops (no data) —
      // those trigger auto-reconnect; only server-sent errors carry data.
      if (!ev.data) return;
      const d = JSON.parse(ev.data);
      finishExport();
      toast(`렌더 오류: ${d.message}`);
    });
  }

  async function startExport() {
    if (!state.audioId || state.exporting) return;
    let res;
    try {
      res = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_id: state.audioId, settings: state.settings }),
      });
    } catch {
      toast("서버에 연결할 수 없습니다");
      return;
    }
    if (res.status === 409) { toast("이미 렌더링이 진행 중입니다"); showExportOverlay(true); return; }
    if (!res.ok) {
      const d = await res.json().catch(() => null);
      if (typeof d?.detail === "string" && d.detail.includes("background image expired")) {
        handleBgMissing();  // same recovery as the preview path: back to solid
        return;
      }
      toast(d?.detail || "렌더링을 시작하지 못했습니다");
      return;
    }
    const { job_id } = await res.json();
    state.jobId = job_id;
    localStorage.setItem(JOB_KEY, job_id);
    setExporting(true);
    openEvents(job_id);
  }

  async function cancelExport() {
    if (!state.jobId) return;
    try { await fetch(`/api/export/${state.jobId}`, { method: "DELETE" }); } catch { /* SSE reports it */ }
  }

  async function rehydrateJob() {
    const jobId = localStorage.getItem(JOB_KEY);
    if (!jobId) return;
    try {
      const res = await fetch(`/api/export/${jobId}`);
      if (!res.ok) { localStorage.removeItem(JOB_KEY); return; }
      const snap = await res.json();
      if (snap.state === "pending" || snap.state === "running") {
        state.jobId = jobId;
        setExporting(true);
        renderProgress(snap);
        openEvents(jobId);
      } else {
        localStorage.removeItem(JOB_KEY);
      }
    } catch { /* server unreachable; leave the key for next load */ }
  }

  // ---- renders list -------------------------------------------------------
  async function refreshRenders() {
    let list = [];
    try {
      list = await (await fetch("/api/renders")).json();
    } catch { return; }
    const ul = $("#rendersList");
    ul.textContent = "";
    $("#rendersEmpty").classList.toggle("hidden", list.length > 0);
    for (const f of list) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.className = "r-name";
      a.textContent = f.name;
      a.title = f.name;
      a.href = f.url;
      a.download = f.name;
      const size = document.createElement("span");
      size.className = "r-size";
      size.textContent = fmtSize(f.size);
      const show = document.createElement("button");
      show.className = "btn ghost small";
      show.textContent = "표시";
      show.title = "탐색기에서 파일 표시";
      show.addEventListener("click", () => reveal(f.name));
      li.append(a, size, show);
      ul.append(li);
    }
  }

  function reveal(name) {
    fetch("/api/reveal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).catch(() => toast("탐색기를 열 수 없습니다"));
  }

  // ---- quit ---------------------------------------------------------------
  function initQuit() {
    $("#quitBtn").addEventListener("click", async () => {
      try { await fetch("/api/shutdown", { method: "POST" }); } catch { /* already down */ }
      document.body.innerHTML = '<div class="goodbye">앱이 종료되었습니다 — 이 탭을 닫아도 됩니다.</div>';
    });
  }

  // ---- init ---------------------------------------------------------------
  function init() {
    loadSettings();
    initControls();
    syncControlsFromState();
    initUpload();
    initTransport();
    initQuit();
    renderBtn.addEventListener("click", startExport);
    cancelBtn.addEventListener("click", cancelExport);
    $("#openFolderBtn").addEventListener("click", () => reveal(null));
    window.addEventListener("resize", () => { fitPreviewBox(); drawSeekbar(); });
    fitPreviewBox();
    drawSeekbar();
    connectWS();
    refreshRenders();
    rehydrateJob();
  }

  init();
})();
