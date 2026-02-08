let currentController = null;
let isGenerating = false;

async function generate() {
  const promptEl = document.getElementById("prompt");
  const button = document.getElementById("generateBtn");

  // 🔁 If already generating → cancel
  if (isGenerating && currentController) {
    currentController.abort();
    return;
  }

  // --- start generation ---
  isGenerating = true;
  currentController = new AbortController();

  const originalLabel = button.textContent;
  button.textContent = "Cancel";
  button.disabled = false;

  // turn previous generations into normal text
  unwrapGenerated(promptEl);

  const payload = {
    prompt: promptEl.innerText,
    model: document.getElementById("model").value,

    temperature: parseFloat(document.getElementById("temperature").value),
    top_p: parseFloat(document.getElementById("top_p").value),
    top_k: parseInt(document.getElementById("top_k").value),

    frequency_penalty: parseFloat(document.getElementById("frequency_penalty").value),
    presence_penalty: parseFloat(document.getElementById("presence_penalty").value),

    max_tokens: parseInt(document.getElementById("max_tokens").value)
  };

  // create span immediately
  const span = document.createElement("span");
  span.className = "llm-generated";
  promptEl.appendChild(span);
  placeCaretAtEnd(promptEl);
  promptEl.scrollTop = promptEl.scrollHeight;

  try {
    const res = await fetch("api/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: currentController.signal // 👈 IMPORTANT
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data:")) continue;

        const data = line.replace("data:", "").trim();
        if (data === "[DONE]") return;

        try {
          const json = JSON.parse(data);
          const token = json.choices?.[0]?.text;
          if (token) {
            span.textContent += token;
            placeCaretAtEnd(promptEl);
            promptEl.scrollTop = promptEl.scrollHeight;
          }
        } catch {
          // ignore partial JSON
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
    // 🔄 reset UI state
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
    val = parseInt(Number(rangeInput.value));
  }
  valueSpan.textContent = val;
}


async function loadModels() {
  const select = document.getElementById("model");

  try {
    const res = await fetch("api/models");
    const models = await res.json();

    select.innerHTML = "";

    models.forEach(model => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      select.appendChild(option);
    });

    const lastModel = localStorage.getItem("model");
    if (lastModel && models.includes(lastModel)) {
      select.value = lastModel;
    }

    select.addEventListener("change", () => {
      localStorage.setItem("model", select.value);
    });

  } catch (err) {
    console.error("Failed to load models", err);
    select.innerHTML = "<option>Error loading models</option>";
  }
}

document.addEventListener("DOMContentLoaded", loadModels);
