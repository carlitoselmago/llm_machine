const API_URL_STORAGE_KEY = "api_url";
const MODEL_STORAGE_KEY = "model";

const state = {
  running: false,
  startedAt: 0,
  stopAt: 0,
  clientStats: [],
  abortControllers: new Set(),
  samples: [],
  renderTimer: null,
  runStopTimer: null
};

function nowMs() {
  return performance.now();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeApiBase(raw) {
  let value = (raw || "").trim();
  if (!value) value = "http://127.0.0.1:8080";
  value = value.replace(/\/+$/, "");
  if (!/\/v1$/i.test(value)) {
    value = `${value}/v1`;
  }
  return value;
}

function percentile(values, q) {
  if (!values.length) return 0;
  if (values.length === 1) return values[0];
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  if (lo === hi) return sorted[lo];
  const ratio = pos - lo;
  return sorted[lo] * (1 - ratio) + sorted[hi] * ratio;
}

function makeClientStat(id) {
  return {
    id,
    inflight: 0,
    total: 0,
    ok: 0,
    fail: 0,
    latencies: [],
    ttfts: [],
    bytes: 0
  };
}

function setUiRunning(running) {
  document.getElementById("startBtn").disabled = running;
  document.getElementById("stopBtn").disabled = !running;
  document.getElementById("reloadModelsBtn").disabled = running;
  document.getElementById("clients").disabled = running;
  document.getElementById("duration").disabled = running;
  document.getElementById("maxTokens").disabled = running;
  document.getElementById("temperature").disabled = running;
  document.getElementById("thinkMs").disabled = running;
  document.getElementById("model").disabled = running;
}

async function loadModels() {
  const select = document.getElementById("model");
  const input = document.getElementById("apiUrl");
  const base = normalizeApiBase(input.value);
  input.value = base.replace(/\/v1$/i, "");
  localStorage.setItem(API_URL_STORAGE_KEY, input.value);
  try {
    const res = await fetch(`${base}/models`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const modelOptions = Array.isArray(data)
      ? data.map((id) => ({ id, label: id }))
      : (data.data || []).map((item) => {
          const id = item?.id || "";
          const hasSize = Number.isFinite(item?.size_gb);
          const sizeText = hasSize ? ` (${item.size_gb.toFixed(2)} GB)` : "";
          return { id, label: item?.display_name || `${id}${sizeText}` };
        });
    const models = modelOptions.map((item) => item.id);
    select.innerHTML = "";
    for (const model of modelOptions) {
      const opt = document.createElement("option");
      opt.value = model.id;
      opt.textContent = model.label || model.id;
      select.appendChild(opt);
    }
    if (!models.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No running models";
      select.appendChild(opt);
    }
    const last = localStorage.getItem(MODEL_STORAGE_KEY);
    if (last && models.includes(last)) {
      select.value = last;
    }
    select.onchange = () => localStorage.setItem(MODEL_STORAGE_KEY, select.value);
  } catch (err) {
    select.innerHTML = "<option>Error loading models</option>";
    setStatus(`Failed to load models: ${err.message || err}`, true);
  }
}

function setStatus(message, isError = false) {
  const el = document.getElementById("statusLine");
  el.textContent = message;
  el.style.color = isError ? "#fca5a5" : "#94a3b8";
}

function resetState(clientCount) {
  state.startedAt = Date.now();
  state.samples = [];
  state.clientStats = Array.from({ length: clientCount }, (_, i) => makeClientStat(i));
  state.abortControllers.clear();
}

function aggregate() {
  let total = 0;
  let ok = 0;
  let fail = 0;
  let inflight = 0;
  let bytes = 0;
  const allLat = [];
  const allTtft = [];
  for (const c of state.clientStats) {
    total += c.total;
    ok += c.ok;
    fail += c.fail;
    inflight += c.inflight;
    bytes += c.bytes;
    allLat.push(...c.latencies);
    allTtft.push(...c.ttfts);
  }
  const elapsedSec = Math.max(0.001, (Date.now() - state.startedAt) / 1000);
  return {
    total,
    ok,
    fail,
    inflight,
    bytes,
    elapsedSec,
    rps: total / elapsedSec,
    avgLat: allLat.length ? allLat.reduce((a, b) => a + b, 0) / allLat.length : 0,
    p95Lat: percentile(allLat, 0.95),
    avgTtft: allTtft.length ? allTtft.reduce((a, b) => a + b, 0) / allTtft.length : 0
  };
}

function updateMetricsUi() {
  const metrics = aggregate();
  const errorPct = metrics.total ? (metrics.fail / metrics.total) * 100 : 0;
  const mib = metrics.bytes / (1024 * 1024);
  const items = [
    ["Elapsed (s)", metrics.elapsedSec.toFixed(1)],
    ["In-Flight", String(metrics.inflight)],
    ["Requests", String(metrics.total)],
    ["RPS", metrics.rps.toFixed(2)],
    ["Success", String(metrics.ok)],
    ["Errors", `${metrics.fail} (${errorPct.toFixed(1)}%)`],
    ["Avg Latency", `${metrics.avgLat.toFixed(1)} ms`],
    ["P95 Latency", `${metrics.p95Lat.toFixed(1)} ms`],
    ["Avg TTFT", `${metrics.avgTtft.toFixed(1)} ms`],
    ["Payload", `${mib.toFixed(2)} MiB`]
  ];
  const wrap = document.getElementById("metrics");
  wrap.innerHTML = items
    .map(([label, value]) => `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`)
    .join("");
}

function updateClientsTable() {
  const tbody = document.getElementById("clientsBody");
  tbody.innerHTML = state.clientStats
    .map((c) => {
      const avgLat = c.latencies.length ? c.latencies.reduce((a, b) => a + b, 0) / c.latencies.length : 0;
      const p95 = percentile(c.latencies, 0.95);
      const avgTtft = c.ttfts.length ? c.ttfts.reduce((a, b) => a + b, 0) / c.ttfts.length : 0;
      return `
      <tr>
        <td>${c.id}</td>
        <td>${c.inflight}</td>
        <td>${c.total}</td>
        <td>${c.ok}</td>
        <td>${c.fail}</td>
        <td>${avgLat.toFixed(1)}</td>
        <td>${p95.toFixed(1)}</td>
        <td>${avgTtft.toFixed(1)}</td>
      </tr>`;
    })
    .join("");
}

function updateErrorBox() {
  const counts = {};
  for (const c of state.clientStats) {
    if (!c.errors) continue;
    for (const [key, val] of Object.entries(c.errors)) {
      counts[key] = (counts[key] || 0) + val;
    }
  }
  const top = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);
  document.getElementById("errors").textContent = top.length
    ? top.map(([k, v]) => `${v}x ${k}`).join("\n")
    : "(none)";
}

function pushSample() {
  const m = aggregate();
  state.samples.push({
    t: (Date.now() - state.startedAt) / 1000,
    rps: m.rps,
    latency: m.avgLat,
    inflight: m.inflight
  });
  if (state.samples.length > 300) state.samples.shift();
}

function drawChart() {
  const canvas = document.getElementById("chart");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0a0f1f";
  ctx.fillRect(0, 0, width, height);
  if (!state.samples.length) return;

  const maxRps = Math.max(1, ...state.samples.map((s) => s.rps));
  const maxLat = Math.max(1, ...state.samples.map((s) => s.latency));
  const maxT = Math.max(1, state.samples[state.samples.length - 1].t);

  ctx.strokeStyle = "#1e293b";
  ctx.beginPath();
  for (let i = 0; i <= 5; i++) {
    const y = (height / 5) * i;
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.stroke();

  const drawLine = (color, selector, maxValue) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    state.samples.forEach((s, idx) => {
      const x = (s.t / maxT) * (width - 10) + 5;
      const y = height - (selector(s) / maxValue) * (height - 10) - 5;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  drawLine("#22d3ee", (s) => s.rps, maxRps);
  drawLine("#f59e0b", (s) => s.latency, maxLat);

  ctx.fillStyle = "#22d3ee";
  ctx.fillText(`RPS max ${maxRps.toFixed(1)}`, 10, 16);
  ctx.fillStyle = "#f59e0b";
  ctx.fillText(`Avg Lat max ${maxLat.toFixed(1)}ms`, 130, 16);
}

function render() {
  if (!state.running) return;
  pushSample();
  updateMetricsUi();
  updateClientsTable();
  updateErrorBox();
  drawChart();
  const remaining = state.stopAt ? Math.max(0, Math.round((state.stopAt - Date.now()) / 1000)) : null;
  const msg = remaining !== null
    ? `Running... ${remaining}s remaining. Check GPU live: nvidia-smi -l 1`
    : "Running... Check GPU live: nvidia-smi -l 1";
  setStatus(msg);
}

function parseSseLine(line) {
  if (!line.startsWith("data:")) return null;
  const raw = line.slice(5).trim();
  if (!raw || raw === "[DONE]") return { done: true };
  try {
    return { data: JSON.parse(raw) };
  } catch {
    return null;
  }
}

async function runClient(clientId, settings) {
  const stats = state.clientStats[clientId];
  while (state.running) {
    const requestStarted = nowMs();
    let firstTokenAt = null;
    let aborted = false;
    stats.inflight += 1;
    const controller = new AbortController();
    state.abortControllers.add(controller);
    try {
      const body = {
        model: settings.model,
        prompt: settings.prompt,
        max_tokens: settings.maxTokens,
        temperature: settings.temperature,
        stream: true
      };
      const res = await fetch(`${settings.apiBase}/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        stats.bytes += value?.byteLength || 0;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          const parsed = parseSseLine(line);
          if (!parsed) continue;
          if (parsed.done) continue;
          const payload = parsed.data;
          if (payload?.error?.message) throw new Error(payload.error.message);
          const token = payload?.choices?.[0]?.text ?? payload?.choices?.[0]?.delta?.content;
          if (token && firstTokenAt === null) {
            firstTokenAt = nowMs();
          }
        }
      }
      stats.ok += 1;
    } catch (err) {
      if (err?.name === "AbortError") {
        aborted = true;
      } else {
        stats.fail += 1;
        stats.errors = stats.errors || {};
        const key = String(err?.message || err).slice(0, 160);
        stats.errors[key] = (stats.errors[key] || 0) + 1;
      }
    } finally {
      const ended = nowMs();
      stats.inflight = Math.max(0, stats.inflight - 1);
      if (!aborted) {
        stats.total += 1;
        stats.latencies.push(ended - requestStarted);
        if (firstTokenAt !== null) {
          stats.ttfts.push(firstTokenAt - requestStarted);
        }
      }
      state.abortControllers.delete(controller);
    }
    if (settings.thinkMs > 0 && state.running) {
      await sleep(settings.thinkMs);
    }
  }
}

function stopRun() {
  state.running = false;
  for (const ctrl of state.abortControllers) ctrl.abort();
  state.abortControllers.clear();
  if (state.renderTimer) clearInterval(state.renderTimer);
  if (state.runStopTimer) clearTimeout(state.runStopTimer);
  state.renderTimer = null;
  state.runStopTimer = null;
  setUiRunning(false);
  updateMetricsUi();
  updateClientsTable();
  updateErrorBox();
  drawChart();
  setStatus("Stopped.");
}

async function startRun() {
  if (state.running) return;
  const apiInput = document.getElementById("apiUrl").value;
  const model = document.getElementById("model").value;
  const clients = parseInt(document.getElementById("clients").value, 10);
  const duration = parseInt(document.getElementById("duration").value, 10);
  const maxTokens = parseInt(document.getElementById("maxTokens").value, 10);
  const temperature = parseFloat(document.getElementById("temperature").value);
  const thinkMs = parseInt(document.getElementById("thinkMs").value, 10);
  const prompt = document.getElementById("prompt").value;
  const apiBase = normalizeApiBase(apiInput);

  if (!model) return setStatus("Select a running model first.", true);
  if (!Number.isFinite(clients) || clients < 1) return setStatus("Clients must be >= 1.", true);
  if (!Number.isFinite(maxTokens) || maxTokens < 1) return setStatus("Max tokens must be >= 1.", true);
  if (!Number.isFinite(temperature) || temperature < 0) return setStatus("Temperature must be >= 0.", true);
  if (!Number.isFinite(thinkMs) || thinkMs < 0) return setStatus("Think time must be >= 0.", true);
  if (!Number.isFinite(duration) || duration < 0) return setStatus("Duration must be >= 0.", true);

  state.running = true;
  state.stopAt = duration > 0 ? Date.now() + duration * 1000 : 0;
  resetState(clients);
  setUiRunning(true);
  setStatus("Starting...");

  for (let i = 0; i < clients; i++) {
    runClient(i, { apiBase, model, prompt, maxTokens, temperature, thinkMs });
  }

  state.renderTimer = setInterval(render, 500);
  if (duration > 0) {
    state.runStopTimer = setTimeout(() => {
      if (state.running) stopRun();
    }, duration * 1000);
  }
}

function init() {
  const api = localStorage.getItem(API_URL_STORAGE_KEY);
  if (api) document.getElementById("apiUrl").value = api;
  document.getElementById("reloadModelsBtn").addEventListener("click", loadModels);
  document.getElementById("startBtn").addEventListener("click", startRun);
  document.getElementById("stopBtn").addEventListener("click", stopRun);
  loadModels();
}

window.addEventListener("beforeunload", () => {
  if (state.running) stopRun();
});

document.addEventListener("DOMContentLoaded", init);
