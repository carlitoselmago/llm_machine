# LLM Orchestrator (RunPod, Single Multi-GPU Machine)

FastAPI controller that serves the a chat frontend, manages Hugging Face model downloads, launches one `vllm/vllm-openai` container per model/GPU via Docker SDK, and exposes an OpenAI-compatible `/v1` API.

## Architecture

```text
Browser (/ and /admin)
  -> FastAPI controller (backend/)
    -> Docker SDK (/var/run/docker.sock)
      -> vLLM containers (1 per model, 1 per GPU)
```

## Structure

- Controller container (`backend/Dockerfile`):
  - FastAPI app
  - Static end-user frontend (`front/` build output)
  - Admin UI (`/admin`)
  - Docker orchestration logic
  - OpenAI-compatible proxy (`/v1/*`)
- Dynamic vLLM containers:
  - one per running model
  - pinned to one GPU via Docker SDK `DeviceRequest`

## Prerequisites (RunPod)

- Ubuntu/Debian-based RunPod GPU instance
- NVIDIA drivers installed on host (`nvidia-smi` works)
- Docker daemon running
- `docker.sock` available at `/var/run/docker.sock`

## Quick Start

```bash
chmod +x install.sh
./install.sh
```

Then open:

- End-user chat UI: `http://<server-ip>:8080/`
- Admin UI: `http://<server-ip>:8080/admin`
- OpenAI API base: `http://<server-ip>:8080/v1`

Default admin credentials (override in `docker-compose.yml`):

- Username: `admin`
- Password: `workshop`

## Manual Start

```bash
mkdir -p models
export HF_TOKEN=your_token_if_needed
docker compose up -d --build
```

## Local Dev (Linux, without RunPod)

Use this when you want fast iteration (auto-reload) on a Linux machine, without a RunPod instance.

### Option A: Run the controller in dev mode (recommended)

Prereqs:

- Python 3.11
- Docker Engine running (required only for model lifecycle operations such as start/stop)
- NVIDIA drivers available on the host (`nvidia-smi` works) and NVIDIA Container Toolkit (required only if you want to start vLLM model containers)
- Without Docker/NVIDIA, the backend still starts in degraded mode for local UI/API development

Start the FastAPI dev server (auto-reload):

```bash
chmod +x dev-local.sh
./dev-local.sh
```

Optional env vars:

```bash
HF_TOKEN=your_token_if_needed ./dev-local.sh
ADMIN_USERNAME=admin ADMIN_PASSWORD=workshop ./dev-local.sh
DEFAULT_GPU_MEMORY_UTILIZATION=0.60 DEFAULT_MAX_NUM_SEQS=32 ./dev-local.sh
```

`dev-local.sh` defaults `DEFAULT_GPU_MEMORY_UTILIZATION=0.65` and `DEFAULT_MAX_NUM_SEQS=64` for better stability on desktop GPUs. Lower them further (for example `0.60` and `32`) if vLLM reports CUDA OOM during warmup.

Common gotcha (Docker permissions):

- If you see Docker “permission denied” errors, make sure your user can run `docker ps` without `sudo` (e.g. add your user to the `docker` group and re-login).
- `dev-local.sh` auto-detects rootless Docker at `/run/user/<uid>/docker.sock` and sets `DOCKER_HOST` when `/var/run/docker.sock` is missing.
- If your Docker config uses a missing credential helper (for example `docker-credential-desktop` on Linux), `dev-local.sh` falls back to a local `DOCKER_CONFIG` without credential helpers for public image pulls.

Quick local smoke checks:

```bash
curl -u admin:workshop http://127.0.0.1:8080/api/admin/health
curl http://127.0.0.1:8080/v1/models
```

### Option B: Frontend-only testing (no Docker/GPU)

If you only need to iterate on the browser UI, you can serve `front/public/` and point the **API URL** field to any OpenAI-compatible backend (LM Studio, etc.).

```bash
cd front/public
python3 -m http.server 3000
```

Open `http://127.0.0.1:3000/`, set **API URL** (example: `http://127.0.0.1:1234`), then reload the model list.

## Project Layout

```text
front/    # end-user chat UI
backend/  # FastAPI controller, admin UI, orchestrator, proxy
models/   # downloaded Hugging Face models (bind-mounted)
```

## End-user Frontend (`/`)

The existing `front/` UI now includes an **API URL** field so users can switch backends without code changes.

Examples:

- LM Studio: `http://127.0.0.1:1234`
- This controller: `http://<server-ip>:8080`

Behavior:

- frontend stores the URL in `localStorage`
- frontend appends `/v1` automatically unless already present
- frontend calls:
  - `GET <base>/models`
  - `POST <base>/completions`

## Admin UI (`/admin`)

Features:

- Download Hugging Face models to `./models/<sanitized_model_id>`
- List downloaded/running models
- Start model (auto-assign free GPU)
- Stop model
- Delete downloaded model (only when stopped)
- View GPU allocation and managed container metadata

Admin API routes (Basic auth):

- `GET /api/admin/health`
- `GET /api/admin/gpus`
- `GET /api/admin/models`
- `POST /api/admin/models/download`
- `POST /api/admin/models/{model_id}/start`
- `POST /api/admin/models/{model_id}/stop`
- `DELETE /api/admin/models/{model_id}`
- `GET /api/admin/containers`

## OpenAI-Compatible API (`/v1`)

Supported routes:

- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Notes:

- `/v1/models` lists running models only
- requests fail if target model is downloaded but not running (`409`)
- unknown models return `404`

### Example: List Models

```bash
curl http://<server-ip>:8080/v1/models
```

### Example: Completions

```bash
curl http://<server-ip>:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "your_model_id",
    "prompt": "Hello",
    "max_tokens": 32,
    "stream": false
  }'
```

### Example: Chat Completions

```bash
curl http://<server-ip>:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "your_model_id",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

## Legacy Compatibility Endpoints

Kept for compatibility with older callers:

- `GET /api/models`
- `POST /api/complete`

## GPU Assignment Behavior

- controller detects GPUs via `nvidia-smi`
- one running model container per GPU
- start fails with `409` when no free GPU exists
- GPU is released when model stops
- controller reconstructs running allocations from Docker labels on restart

## Docker Compose

`docker-compose.yml` runs a single controller service and mounts:

- `./models:/models`
- `/var/run/docker.sock:/var/run/docker.sock`

It also requests GPU access using `gpus: all` and includes a `deploy` reservation block for compatibility/documentation.

## Troubleshooting

### Docker connection errors

- Ensure `/var/run/docker.sock` is mounted
- Verify controller container can talk to Docker daemon

### No GPUs detected

- Verify host `nvidia-smi` works
- Install NVIDIA Container Toolkit
- Restart Docker after toolkit configuration

### vLLM start timeout

- Large models can take several minutes to initialize
- Increase `VLLM_STARTUP_TIMEOUT_SECONDS` if needed
- Check container logs:

```bash
docker logs <vllm-container-id>
```

### vLLM container keeps restarting on desktop GPU

- If logs show `Free memory on device ... is less than desired GPU memory utilization`, reduce memory target
- If logs show CUDA OOM during warmup with many dummy requests, also reduce sequence concurrency (`DEFAULT_MAX_NUM_SEQS`)
- Local dev defaults are `DEFAULT_GPU_MEMORY_UTILIZATION=0.65` and `DEFAULT_MAX_NUM_SEQS=64`
- For heavily used desktop GPUs, try lower values (for example `0.60` and `32`)

### Model downloaded but not available in `/v1/models`

- `/v1/models` shows only running models
- GGUF folders are not supported by this vLLM workflow
- Use a Hugging Face Transformers model with `config.json` (or Mistral `params.json`)
- If needed, inspect crash logs:

```bash
docker logs <vllm-container-id>
```

### Hugging Face download issues

- Set `HF_TOKEN` for gated/private models
- Confirm sufficient disk space in `./models`

## Security Note (Workshop Scope)

This project uses simple HTTP Basic auth for admin routes/UI only. It is suitable for short-lived workshop environments and private networks, not direct internet exposure.

Recommended for real production exposure:

- reverse proxy with TLS
- stronger auth (OIDC or managed auth)
- network restrictions / VPN

## Non-Goals

- Kubernetes
- Multi-node orchestration
- Billing
- Autoscaling
- Serverless execution
- Complex scheduling
