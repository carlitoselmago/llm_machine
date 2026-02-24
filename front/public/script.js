let currentController = null;
let isGenerating = false;

const API_URL_STORAGE_KEY = "api_url";
const MODEL_STORAGE_KEY = "model";
const DEFAULT_API_URL = "http://127.0.0.1:1234";

function normalizeApiBase(raw) {
  let value = (raw || "").trim();
  if (!value) value = DEFAULT_API_URL;
  value = value.replace(/\/+$/, "");
  if (!/\/v1$/i.test(value)) {
    value = `${value}/v1`;
  }
  return value;
}

function getApiBase() {
  const input = document.getElementById("api_url");
  const normalized = normalizeApiBase(input?.value || "");
  if (input) {
    input.value = input.value.trim();
  }
  return normalized;
}

function persistApiUrl() {
  const input = document.getElementById("api_url");
  if (!input) return;
  const raw = (input.value || "").trim() || DEFAULT_API_URL;
  localStorage.setItem(API_URL_STORAGE_KEY, raw);
}

function setupApiUrlField() {
  const input = document.getElementById("api_url");
  if (!input) return;
  input.value = localStorage.getItem(API_URL_STORAGE_KEY) || DEFAULT_API_URL;
  input.addEventListener("change", persistApiUrl);
  input.addEventListener("blur", persistApiUrl);
}

async function generate() {
  const promptEl = document.getElementById("prompt");
  const button = document.getElementById("generateBtn");

  if (isGenerating && currentController) {
    currentController.abort();
    return;
  }

  isGenerating = true;
  currentController = new AbortController();

  const originalLabel = button.textContent;
  button.textContent = "Cancel";
  button.disabled = false;

  unwrapGenerated(promptEl);

  const payload = {
    prompt: promptEl.innerText,
    model: document.getElementById("model").value,
    temperature: parseFloat(document.getElementById("temperature").value),
    top_p: parseFloat(document.getElementById("top_p").value),
    top_k: parseInt(document.getElementById("top_k").value, 10),
    frequency_penalty: parseFloat(document.getElementById("frequency_penalty").value),
    presence_penalty: parseFloat(document.getElementById("presence_penalty").value),
    max_tokens: parseInt(document.getElementById("max_tokens").value, 10),
    stream: true
  };

  const span = document.createElement("span");
  span.className = "llm-generated";
  promptEl.appendChild(span);
  placeCaretAtEnd(promptEl);
  promptEl.scrollTop = promptEl.scrollHeight;

  try {
    const apiBase = getApiBase();
    const res = await fetch(`${apiBase}/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: currentController.signal
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(errorText || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data:")) continue;

        const data = line.replace("data:", "").trim();
        if (data === "[DONE]") return;

        try {
          const json = JSON.parse(data);
          const token = json.choices?.[0]?.text ?? json.choices?.[0]?.delta?.content;
          if (token) {
            span.textContent += token;
            placeCaretAtEnd(promptEl);
            promptEl.scrollTop = promptEl.scrollHeight;
          }
        } catch {
          // ignore partial JSON chunks
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      console.log("Generation cancelled");
    } else {
      console.error("Streaming failed", err);
    }
  } finally {
    isGenerating = false;
    currentController = null;
    button.textContent = originalLabel;
    button.disabled = false;
  }
}

function placeCaretAtEnd(el) {
  el.focus();
  const range = document.createRange();
  range.selectNodeContents(el);
  range.collapse(false);
  const sel = window.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
}

function unwrapGenerated(promptEl) {
  promptEl.querySelectorAll(".llm-generated").forEach(span => {
    const textNode = document.createTextNode(span.textContent);
    span.replaceWith(textNode);
  });
}

function updateValue(rangeInput, isint = false) {
  const valueSpan = rangeInput.previousElementSibling;
  let val = Number(rangeInput.value).toFixed(1);
  if (isint) {
    val = parseInt(Number(rangeInput.value), 10);
  }
  valueSpan.textContent = val;
}

async function loadModels() {
  const select = document.getElementById("model");

  try {
    const apiBase = getApiBase();
    const res = await fetch(`${apiBase}/models`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const models = Array.isArray(data) ? data : (data.data || []).map(m => m.id);

    select.innerHTML = "";

    models.forEach(model => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      select.appendChild(option);
    });

    if (!models.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No models available";
      select.appendChild(option);
    }

    const lastModel = localStorage.getItem(MODEL_STORAGE_KEY);
    if (lastModel && models.includes(lastModel)) {
      select.value = lastModel;
    }

    select.onchange = () => {
      localStorage.setItem(MODEL_STORAGE_KEY, select.value);
    };
  } catch (err) {
    console.error("Failed to load models", err);
    select.innerHTML = "<option>Error loading models</option>";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setupApiUrlField();
  const apiUrlInput = document.getElementById("api_url");
  apiUrlInput?.addEventListener("change", loadModels);
  apiUrlInput?.addEventListener("blur", loadModels);
  loadModels();
});
