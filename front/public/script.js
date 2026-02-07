async function generate() {
    const promptEl = document.getElementById("prompt");


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
    //console.log(payload);
    const res = await fetch("/api/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await res.json();
    const text = data.choices?.[0]?.text || "";

    // add new generated span
    const span = document.createElement("span");
    span.className = "llm-generated";
    span.textContent = text;

    promptEl.appendChild(span);
    placeCaretAtEnd(promptEl);
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

function updateValue(rangeInput,isint=false) {
    const valueSpan = rangeInput.nextElementSibling;
    let val= Number(rangeInput.value).toFixed(1);
    if (isint){
        val=parseInt(Number(rangeInput.value));
    } 
    valueSpan.textContent = val;
}
