# LLM Orchestrator (RunPod Pod-Compatible)

This version is designed for **RunPod Pods** where Docker Compose and Docker-in-Docker are not available.

## Runtime design

- One FastAPI controller process serves:
  - End-user frontend at `/`
  - Admin UI/API at `/admin` and `/api/admin/*`
  - OpenAI-compatible API at `/v1/*`
- Model runtimes are started as local subprocesses:
  - `vllm serve ...`
  - one process per model
  - one GPU per process via `CUDA_VISIBLE_DEVICES`

No runtime Docker SDK / container spawning is required.

## Why this works on RunPod

RunPod Pods do not support Docker Compose or running your own Docker daemon inside the pod. This implementation avoids both.

## Build image for RunPod template

Build from repo root using:

- Dockerfile: `backend/Dockerfile`

The runtime image is based on `vllm/vllm-openai:latest`, then adds:
- controller backend code
- frontend static build artifacts

## RunPod environment variables

Set these in Pod env vars (or defaults are used):

- `HOST=0.0.0.0`
- `PORT=8080`
- `MODELS_DIR=/workspace/models`
- `ADMIN_USERNAME=admin`
- `ADMIN_PASSWORD=workshop`
- `HF_TOKEN=<optional>`
- `DEFAULT_GPU_MEMORY_UTILIZATION=0.65` (optional)
- `DEFAULT_MAX_NUM_SEQS=64` (optional)
- `DEFAULT_MAX_MODEL_LEN=20000` (optional)

## Volumes

Mount persistent volume to keep models across restarts:

- recommended path: `/workspace/models`

## Start command

Image default command is already set:

```bash
/app/runpod-start.sh
```

So normally no override is required.

## Endpoints

- Chat UI: `http://<pod-ip>:8080/`
- Admin UI: `http://<pod-ip>:8080/admin`
- OpenAI API: `http://<pod-ip>:8080/v1`

## Admin workflow

1. Download a HF model (`repo_id`) in `/admin`
2. Start model (auto GPU assignment)
3. Use selected model from frontend or OpenAI API
4. Stop/delete model from admin panel

## OpenAI API examples

List models:

```bash
curl http://<pod-ip>:8080/v1/models
```

Completions:

```bash
curl http://<pod-ip>:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your_model","prompt":"hello","max_tokens":64,"stream":false}'
```

Chat completions:

```bash
curl http://<pod-ip>:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your_model","messages":[{"role":"user","content":"hello"}],"stream":false}'
```

## Local dev

Local dev scripts remain available (`dev-local.bat`, `dev-local.sh`) for UI/backend iteration.

- Windows: `.\dev-local.bat`
- Linux/macOS: `./dev-local.sh`

These scripts install `requirements-dev.txt` (which includes `fastapi` + `uvicorn`) and then start:

`http://localhost:8080`

Starting/stopping models from `/admin` also requires a working `vllm` executable in the local environment.
If `vllm` is missing, UI/API still run, but model runtime actions will fail.

On Windows, the practical options are:
- run model runtimes on RunPod/Linux and use the frontend API URL field
- or run local dev under WSL2/Linux where `vllm` is available

Native Windows is not a supported local runtime target for vLLM. Use Windows only for UI/backend iteration, or use WSL2/Linux/RunPod for actual model startup.

## Limitations

- Single machine only
- No multi-node orchestration
- No auto-scaling
- Basic admin auth only (workshop-grade)
