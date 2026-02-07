import Fastify from "fastify";
import path from "path";
import { fileURLToPath } from "url";

const fastify = Fastify({ logger: true });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/* ========= CONFIG ========= */

// LM Studio / OpenAI-compatible endpoint
// example: http://10.0.0.5:1234/v1/completions
const LLM_API_URL = "http://127.0.0.1:1234/v1/completions";

// Optional API key (LM Studio usually ignores it)
const LLM_API_KEY = "";

/* ========= STATIC FRONTEND ========= */

fastify.register(import("@fastify/static"), {
  root: path.join(__dirname, "public"),
  prefix: "/"
});

/* ========= API ========= */

fastify.post("/api/complete", async (request, reply) => {
  const {
    prompt,
    model,
    temperature = 0.7,
    max_tokens = 200,
    top_p = 1,
    frequency_penalty = 0,
    presence_penalty = 0
  } = request.body;

  if (!prompt || !model) {
    return reply.code(400).send({ error: "Missing prompt or model" });
  }

  const payload = {
    model,
    prompt,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty
  };

  try {
    const res = await fetch(LLM_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(LLM_API_KEY && { Authorization: `Bearer ${LLM_API_KEY}` })
      },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    reply.send(data);
  } catch (err) {
    fastify.log.error(err);
    reply.code(500).send({ error: "LLM backend error" });
  }
});


fastify.post("/api/respond", async (request, reply) => {
  const {
    input,
    model,
    temperature,
    top_p,
    top_k,
    frequency_penalty,
    presence_penalty,
    max_tokens
  } = request.body;

  const payload = {
    model,
    input,
    temperature,
    top_p,
    top_k,
    frequency_penalty,
    presence_penalty,
    max_output_tokens: max_tokens
  };

  const res = await fetch("http://127.0.0.1:1234/v1/responses", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  reply.send(data);
});

/* ========= START ========= */

const PORT = 3000;

fastify.listen({ port: PORT, host: "0.0.0.0" })
  .then(() => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
