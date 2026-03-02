#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
mkdir -p "$MODELS_DIR"

export MODELS_DIR
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"
export FRONT_STATIC_DIR="${FRONT_STATIC_DIR:-app/static/front}"

exec uvicorn app.main:app --host "$HOST" --port "$PORT"
