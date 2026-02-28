#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/carlitoselmago/llm_machine"
TARGET_DIR="${1:-llm_machine}"

log() {
  printf '[bootstrap] %s\n' "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

SUDO=""
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  SUDO="sudo"
fi

if ! need_cmd git; then
  if ! need_cmd apt-get; then
    log "git is required and apt-get is not available. Install git manually, then rerun."
    exit 1
  fi
  log "Installing git..."
  $SUDO apt-get update
  $SUDO apt-get install -y git
fi

if [[ -d "$TARGET_DIR/.git" ]]; then
  log "Repository already exists at $TARGET_DIR, pulling latest changes..."
  git -C "$TARGET_DIR" pull --ff-only
elif [[ -e "$TARGET_DIR" ]]; then
  log "Path '$TARGET_DIR' exists but is not a git repo. Remove/rename it and rerun."
  exit 1
else
  log "Cloning project..."
  git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"
chmod +x install.sh

log "Running installer..."
./install.sh

log "Done."
log "Client:  http://$(hostname -I | awk '{print $1}'):8080/"
log "Admin:   http://$(hostname -I | awk '{print $1}'):8080/admin"
log "Stress:  http://$(hostname -I | awk '{print $1}'):8080/stress.html"
log "OpenAI:  http://$(hostname -I | awk '{print $1}'):8080/v1"
