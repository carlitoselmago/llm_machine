let currentController = null;
let isGenerating = false;

const API_URL_STORAGE_KEY = "api_url";
const MODEL_STORAGE_KEY = "model";
const DEFAULT_API_URL = "http://127.0.0.1:1234";
const MAX_OUTPUT_TOKENS = 20000;
const MODEL_CONTEXT_LIMITS_STORAGE_KEY = "model_context_limits";
const DEFAULT_PROMPT_WINDOW_TOKENS = 1024;
const MIN_PROMPT_WINDOW_TOKENS = 64;
const CONTEXT_SAFETY_TOKENS = 32;

function loadModelContextLimits() {
  try {
    const raw = localStorage.getItem(MODEL_CONTEXT_LIMITS_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return {};
    return parsed;
  } catch {
    return {};
  }
}

function saveModelContextLimits(map) {
  localStorage.setItem(MODEL_CONTEXT_LIMITS_STORAGE_KEY, JSON.stringify(map || {}));
}

let modelContextLimits = loadModelContextLimits();

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

function extractErrorMessage(raw, fallback = "Request failed") {
  if (!raw) return fallback;
  try {
    const data = JSON.parse(raw);
    if (typeof data === "string") return data;
    if (data?.error?.message) return data.error.message;
    if (typeof data?.detail === "string") return data.detail;
    if (data?.detail) return JSON.stringify(data.detail);
    if (typeof data?.message === "string") return data.message;
    return raw;
  } catch {
    return raw;
  }
}

function showOutput(message, isError = false) {
  const outputEl = document.getElementById("output");
  if (!outputEl) return;
  outputEl.textContent = message || "";
  outputEl.style.color = isError ? "#fca5a5" : "";
}

function estimatePromptTokens(promptText) {
  const text = (promptText || "").trim();
  if (!text) return 1;
  return Math.max(1, Math.ceil(text.length / 4));
}

function applyRollingWindow(promptText, promptBudgetTokens) {
  const text = promptText || "";
  const safeBudget = Math.max(MIN_PROMPT_WINDOW_TOKENS, parseInt(promptBudgetTokens, 10) || MIN_PROMPT_WINDOW_TOKENS);
  const maxChars = safeBudget * 4;
  if (text.length <= maxChars) {
    return { prompt: text, truncated: false, approxTokens: estimatePromptTokens(text) };
  }
  const start = text.length - maxChars;
  const breakIndex = text.indexOf("\n", start);
  const sliceStart = breakIndex !== -1 && breakIndex < text.length - 32 ? breakIndex + 1 : start;
  const prompt = text.slice(sliceStart);
  return { prompt, truncated: true, approxTokens: estimatePromptTokens(prompt), droppedChars: sliceStart };
}

function parseContextLengthError(message) {
  if (!message) return null;
  const inputMatch = message.match(/passed\s+(\d+)\s+input tokens/i);
  const requestedMatch = message.match(/requested\s+(\d+)\s+output tokens/i);
  const contextMatch =
    message.match(/context length is only\s+(\d+)\s+tokens/i) ||
    message.match(/max_total_tokens=(\d+)/i) ||
    message.match(/max_model_len[^0-9]*(\d+)/i);
  const maxInputMatch = message.match(/maximum input length of\s+(\d+)/i);

  const inputTokens = inputMatch ? parseInt(inputMatch[1], 10) : null;
  const requestedTokens = requestedMatch ? parseInt(requestedMatch[1], 10) : null;
  const contextLen = contextMatch ? parseInt(contextMatch[1], 10) : null;
  const maxInputTokens = maxInputMatch ? parseInt(maxInputMatch[1], 10) : null;

  if (!inputTokens && !requestedTokens && !contextLen && !maxInputTokens) return null;
  return { inputTokens, requestedTokens, contextLen, maxInputTokens };
}

function applyContextLimitFromError(message, model, promptText) {
  if (!message) return null;
  const limitMatch = message.match(/max_total_tokens=(\d+)/i) || message.match(/max_model_len[^0-9]*(\d+)/i);
  if (!limitMatch) return null;
  const limit = parseInt(limitMatch[1], 10);
  if (!Number.isFinite(limit) || limit < 1) return null;
  if (model) {
    modelContextLimits[model] = limit;
    saveModelContextLimits(modelContextLimits);
  }

  const input = document.getElementById("max_tokens");
  if (!input) return limit;
  input.max = String(Math.max(1, limit - CONTEXT_SAFETY_TOKENS));

  const current = parseInt(input.value, 10);
  const safeOutput = Math.max(1, limit - estimatePromptTokens(promptText) - CONTEXT_SAFETY_TOKENS);
  if (!Number.isFinite(current) || current > safeOutput) {
    input.value = String(safeOutput);
  }
  return limit;
}

async function generate() {
  const promptEl = document.getElementById("prompt");
  const button = document.getElementById("generateBtn");
  const model = document.getElementById("model").value;
  if (!model) {
    showOutput("Error: Select a running model in the dropdown.", true);
    return;
  }

  let maxTokens = parseInt(document.getElementById("max_tokens").value, 10);
  if (!Number.isFinite(maxTokens) || maxTokens < 1) {
    showOutput("Error: Max tokens must be a positive number.", true);
    return;
  }
  if (maxTokens > MAX_OUTPUT_TOKENS) {
    maxTokens = MAX_OUTPUT_TOKENS;
    document.getElementById("max_tokens").value = String(MAX_OUTPUT_TOKENS);
    showOutput(`Max tokens limited to ${MAX_OUTPUT_TOKENS}.`, false);
  } else {
    showOutput("", false);
  }

  const promptText = promptEl.innerText;
  const knownLimit = modelContextLimits[model];
  let promptBudgetTokens = DEFAULT_PROMPT_WINDOW_TOKENS;
  if (Number.isFinite(knownLimit) && knownLimit > 0) {
    const maxOutputByLimit = Math.max(1, knownLimit - MIN_PROMPT_WINDOW_TOKENS - CONTEXT_SAFETY_TOKENS);
    if (maxTokens > maxOutputByLimit) {
      maxTokens = maxOutputByLimit;
      document.getElementById("max_tokens").value = String(maxTokens);
      showOutput(`Max tokens auto-adjusted to ${maxTokens} for model context limit ${knownLimit}.`, false);
    }
    promptBudgetTokens = Math.max(MIN_PROMPT_WINDOW_TOKENS, knownLimit - maxTokens - CONTEXT_SAFETY_TOKENS);
  }
  let runtimeMaxTokens = maxTokens;
  let runtimePromptBudgetTokens = promptBudgetTokens;
  let rolling = applyRollingWindow(promptText, runtimePromptBudgetTokens);
  if (rolling.truncated) {
    showOutput(`Rolling window active: using the latest ~${rolling.approxTokens} tokens as prompt context.`, false);
  }

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

  const span = document.createElement("span");
  span.className = "llm-generated";
  promptEl.appendChild(span);
  placeCaretAtEnd(promptEl);
  promptEl.scrollTop = promptEl.scrollHeight;

  try {
    const apiBase = getApiBase();
    let res = null;
    for (let attempt = 1; attempt <= 3; attempt++) {
      rolling = applyRollingWindow(promptText, runtimePromptBudgetTokens);
      if (rolling.truncated) {
        showOutput(`Rolling window active: using the latest ~${rolling.approxTokens} tokens as prompt context.`, false);
      }
      const payload = {
        prompt: rolling.prompt,
        model,
        temperature: parseFloat(document.getElementById("temperature").value),
        top_p: parseFloat(document.getElementById("top_p").value),
        top_k: parseInt(document.getElementById("top_k").value, 10),
        frequency_penalty: parseFloat(document.getElementById("frequency_penalty").value),
        presence_penalty: parseFloat(document.getElementById("presence_penalty").value),
        max_tokens: runtimeMaxTokens,
        stream: true
      };

      res = await fetch(`${apiBase}/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: currentController.signal
      });
      if (res.ok) {
        document.getElementById("max_tokens").value = String(runtimeMaxTokens);
        break;
      }
      const errorText = await res.text();
      const message = extractErrorMessage(errorText, `HTTP ${res.status}`);
      const contextError = parseContextLengthError(message);
      if (!contextError || attempt >= 3) {
        throw new Error(message);
      }

      const detectedLimit = applyContextLimitFromError(message, model, rolling.prompt) || contextError.contextLen;
      if (Number.isFinite(detectedLimit) && detectedLimit > 0) {
        modelContextLimits[model] = detectedLimit;
        saveModelContextLimits(modelContextLimits);
      }

      if (Number.isFinite(contextError.contextLen) && Number.isFinite(contextError.inputTokens)) {
        runtimeMaxTokens = Math.max(1, Math.min(runtimeMaxTokens, contextError.contextLen - contextError.inputTokens - CONTEXT_SAFETY_TOKENS));
      } else if (Number.isFinite(contextError.contextLen)) {
        runtimeMaxTokens = Math.max(1, Math.min(runtimeMaxTokens, contextError.contextLen - MIN_PROMPT_WINDOW_TOKENS - CONTEXT_SAFETY_TOKENS));
      } else {
        runtimeMaxTokens = Math.max(1, Math.floor(runtimeMaxTokens * 0.8));
      }

      if (
        Number.isFinite(contextError.inputTokens) &&
        Number.isFinite(contextError.maxInputTokens) &&
        contextError.inputTokens > contextError.maxInputTokens
      ) {
        const ratio = contextError.maxInputTokens / contextError.inputTokens;
        runtimePromptBudgetTokens = Math.max(
          MIN_PROMPT_WINDOW_TOKENS,
          Math.floor(runtimePromptBudgetTokens * ratio * 0.9),
        );
      }
      if (Number.isFinite(contextError.contextLen)) {
        runtimePromptBudgetTokens = Math.max(
          MIN_PROMPT_WINDOW_TOKENS,
          Math.min(runtimePromptBudgetTokens, contextError.contextLen - runtimeMaxTokens - CONTEXT_SAFETY_TOKENS),
        );
      }
      showOutput(`Context limit reached. Retrying automatically (${attempt + 1}/3) with max tokens ${runtimeMaxTokens}.`, false);
    }

    if (!res || !res.ok) {
      throw new Error("Request failed before stream started");
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

        let json;
        try {
          json = JSON.parse(data);
        } catch {
          continue;
        }
        if (json?.error?.message) {
          throw new Error(json.error.message);
        }
        const token = json.choices?.[0]?.text ?? json.choices?.[0]?.delta?.content;
        if (token) {
          span.textContent += token;
          placeCaretAtEnd(promptEl);
          promptEl.scrollTop = promptEl.scrollHeight;
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      console.log("Generation cancelled");
      showOutput("Generation cancelled.", false);
    } else {
      console.error("Streaming failed", err);
      const message = err.message || String(err);
      const contextLimit = applyContextLimitFromError(message, model, rolling.prompt);
      if (contextLimit) {
        showOutput(`Error: ${message}\nModel context limit is ${contextLimit} tokens (prompt + output). Max tokens was auto-adjusted.`, true);
      } else {
        showOutput(`Error: ${message}`, true);
      }
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
    const modelOptions = Array.isArray(data)
      ? data.map((id) => ({ id, label: id }))
      : (data.data || []).map((item) => {
          const id = item?.id || "";
          const hasSize = Number.isFinite(item?.size_gb);
          const sizeText = hasSize ? ` (${item.size_gb.toFixed(2)} GB)` : "";
          const label = item?.display_name || `${id}${sizeText}`;
          return { id, label };
        });
    const modelIds = modelOptions.map((item) => item.id);

    select.innerHTML = "";

    modelOptions.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.label || model.id;
      select.appendChild(option);
    });

    if (!modelOptions.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No running models available (start one in /admin)";
      select.appendChild(option);
    }

    const lastModel = localStorage.getItem(MODEL_STORAGE_KEY);
    if (lastModel && modelIds.includes(lastModel)) {
      select.value = lastModel;
    }

    select.onchange = () => {
      localStorage.setItem(MODEL_STORAGE_KEY, select.value);
      const knownLimit = modelContextLimits[select.value];
      const maxTokensInput = document.getElementById("max_tokens");
      if (maxTokensInput) {
        if (Number.isFinite(knownLimit) && knownLimit > 0) {
          maxTokensInput.max = String(Math.max(1, knownLimit - CONTEXT_SAFETY_TOKENS));
        } else {
          maxTokensInput.max = String(MAX_OUTPUT_TOKENS);
        }
      }
    };
    select.onchange();
  } catch (err) {
    console.error("Failed to load models", err);
    select.innerHTML = "<option>Error loading models</option>";
    showOutput(`Error loading models: ${err.message || String(err)}`, true);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setupApiUrlField();
  const apiUrlInput = document.getElementById("api_url");
  apiUrlInput?.addEventListener("change", loadModels);
  apiUrlInput?.addEventListener("blur", loadModels);
  loadModels();
});
