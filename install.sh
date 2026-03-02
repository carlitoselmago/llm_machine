#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[runpod-install] %s\n' "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! need_cmd nvidia-smi; then
  log "nvidia-smi not found. This script must run on a GPU-enabled RunPod pod."
  exit 1
fi

if ! need_cmd python3.11 && ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)' >/dev/null 2>&1; then
  log "Python 3.11 is required. Install Python 3.11 or use a RunPod template with Python 3.11."
  exit 1
fi

PYTHON_BIN="python3.11"
if ! need_cmd "$PYTHON_BIN"; then
  PYTHON_BIN="python3"
fi

if [[ ! -x .venv/bin/python ]]; then
  log "Creating virtual environment..."
  "$PYTHON_BIN" -m venv .venv
fi

log "Installing backend dependencies..."
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r backend/requirements.txt

if [[ -f front/package.json ]]; then
  if ! need_cmd npm; then
    log "npm is required to build front/ assets."
    exit 1
  fi
  log "Building frontend static assets..."
  pushd front >/dev/null
  npm install
  popd >/dev/null
  mkdir -p backend/app/static/front
  cp -r front/public/. backend/app/static/front/
fi

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
mkdir -p "$MODELS_DIR"

export MODELS_DIR
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"
export FRONT_STATIC_DIR="${FRONT_STATIC_DIR:-$ROOT_DIR/backend/app/static/front}"
export ADMIN_USERNAME="${ADMIN_USERNAME:-admin}"
export ADMIN_PASSWORD="${ADMIN_PASSWORD:-workshop}"

log "Starting controller on http://0.0.0.0:${PORT}"
cd backend
exec ../.venv/bin/python -m uvicorn app.main:app --host "$HOST" --port "$PORT"
