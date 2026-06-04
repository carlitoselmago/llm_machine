"use strict";

const Fastify = require("fastify");
const path = require("path");

const fastify = Fastify({ logger: true });

/* ========= CONFIG ========= */

// LM Studio / OpenAI-compatible endpoint
// example: http://10.0.0.5:1234/v1/completions
const LLM_API_URL = "http://127.0.0.1:1234/v1/completions";
// Optional API key (LM Studio usually ignores it)
const LLM_API_KEY = "";

/* ========= STATIC FRONTEND ========= */

fastify.register(require("@fastify/static"), {
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
    top_k= 10,
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
    top_k,
    frequency_penalty,
    presence_penalty,
    stream : true
  };

  try {
    const res = await fetch(LLM_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    reply.raw.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive"
    });

    for await (const chunk of res.body) {
      reply.raw.write(chunk);
    }

    reply.raw.end();
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

fastify.get("/api/models", async (request, reply) => {
  try {
    const res = await fetch("http://127.0.0.1:1234/v1/models");
    const data = await res.json();

    // OpenAI-style: { data: [{ id: "model-name", ... }] }
    const models = (data.data || []).map(m => m.id);

    reply.send(models);
  } catch (err) {
    fastify.log.error(err);
    reply.code(500).send({ error: "Could not fetch models" });
  }
});

/* ========= START ========= */

const PORT = 3000;

async function start() {
  await fastify.listen({ port: PORT, host: "127.0.0.1" });
  console.log(`Server running on http://localhost:${PORT}`);
}

if (require.main === module) {
  start();
}

module.exports = { start, PORT };
