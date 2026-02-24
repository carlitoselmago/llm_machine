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

export MODELS_DIR="$ROOT_DIR/models"
export ADMIN_USERNAME="admin"
export ADMIN_PASSWORD="workshop"
export FRONT_STATIC_DIR="$ROOT_DIR/front/public"

echo "[dev-local] Starting FastAPI dev server on http://localhost:8080"
cd backend
exec ../.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
