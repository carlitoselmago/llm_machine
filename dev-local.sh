#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p models

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)'; then
    PYTHON_BIN="python3"
  else
    echo "[dev-local] Python 3.11 is required for local dev (this project targets 3.11)."
    echo "[dev-local] Set PYTHON_BIN=/path/to/python3.11 if needed."
    exit 1
  fi
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "[dev-local] Creating virtual environment..."
  "$PYTHON_BIN" -m venv .venv
fi

if [[ "${SKIP_PIP_INSTALL:-0}" == "1" ]]; then
  echo "[dev-local] Skipping pip install (SKIP_PIP_INSTALL=1)"
else
  echo "[dev-local] Installing/updating Python dependencies..."
  ./.venv/bin/python -m pip install --disable-pip-version-check -r backend/requirements.txt
fi

export PATH="$ROOT_DIR/.venv/bin:$PATH"

if [[ -z "${VLLM_EXECUTABLE:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/vllm" ]]; then
    export VLLM_EXECUTABLE="$ROOT_DIR/.venv/bin/vllm"
  elif command -v vllm >/dev/null 2>&1; then
    export VLLM_EXECUTABLE="$(command -v vllm)"
  else
    echo "[dev-local] vLLM is not installed in this environment."
    echo "[dev-local] Install with: ./.venv/bin/python -m pip install vllm"
  fi
fi

export MODELS_DIR="$ROOT_DIR/models"
export ADMIN_USERNAME="admin"
export ADMIN_PASSWORD="workshop"
export FRONT_STATIC_DIR="$ROOT_DIR/front/public"
export DEFAULT_GPU_MEMORY_UTILIZATION="${DEFAULT_GPU_MEMORY_UTILIZATION:-0.65}"
export DEFAULT_MAX_NUM_SEQS="${DEFAULT_MAX_NUM_SEQS:-64}"

docker_config_dir="${DOCKER_CONFIG:-$HOME/.docker}"
docker_config_file="$docker_config_dir/config.json"
if [[ -f "$docker_config_file" ]]; then
  creds_store="$(sed -n 's/.*"credsStore"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$docker_config_file" | head -n 1)"
  if [[ -n "$creds_store" ]] && ! command -v "docker-credential-$creds_store" >/dev/null 2>&1; then
    if [[ -z "${DOCKER_CONFIG:-}" ]]; then
      local_docker_config="$ROOT_DIR/.docker-dev"
      mkdir -p "$local_docker_config"
      cat > "$local_docker_config/config.json" <<'JSON'
{
  "auths": {}
}
JSON
      export DOCKER_CONFIG="$local_docker_config"
      echo "[dev-local] Missing docker-credential-$creds_store; using local Docker config without credential helpers."
    else
      echo "[dev-local] DOCKER_CONFIG is set and references missing helper docker-credential-$creds_store."
      echo "[dev-local] Update that config or unset DOCKER_CONFIG for local dev fallback."
    fi
  fi
fi

if [[ -z "${DOCKER_HOST:-}" ]]; then
  rootless_sock="/run/user/$(id -u)/docker.sock"
  if [[ ! -S /var/run/docker.sock && -S "$rootless_sock" ]]; then
    export DOCKER_HOST="unix://$rootless_sock"
    echo "[dev-local] Using rootless Docker socket: $DOCKER_HOST"
  fi
fi

if [[ -z "${DOCKER_HOST:-}" && ! -S /var/run/docker.sock ]]; then
  echo "[dev-local] Docker socket not found. Backend will start in degraded mode (model start/stop disabled)."
fi

echo "[dev-local] Starting FastAPI dev server on http://localhost:8080"
cd backend
exec ../.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
