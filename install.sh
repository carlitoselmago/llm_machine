#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[install] %s\n' "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

SUDO=""
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  SUDO="sudo"
fi

if ! need_cmd apt-get; then
  log "This installer targets Ubuntu/Debian (apt-get required)."
  exit 1
fi

if ! need_cmd docker; then
  log "Installing Docker Engine..."
  $SUDO apt-get update
  $SUDO apt-get install -y ca-certificates curl gnupg
  $SUDO install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  $SUDO chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
  $SUDO apt-get update
  $SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
else
  log "Docker already installed."
fi

if ! docker compose version >/dev/null 2>&1; then
  log "Installing Docker Compose plugin..."
  $SUDO apt-get update
  $SUDO apt-get install -y docker-compose-plugin
else
  log "Docker Compose plugin already available."
fi

if ! need_cmd nvidia-smi; then
  log "nvidia-smi is not available on the host. Ensure this is a RunPod GPU instance with NVIDIA drivers installed."
  exit 1
fi

if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  log "Installing NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  $SUDO apt-get update
  $SUDO apt-get install -y nvidia-container-toolkit
  if need_cmd nvidia-ctk; then
    $SUDO nvidia-ctk runtime configure --runtime=docker
  fi
  $SUDO systemctl restart docker || true
else
  log "NVIDIA Container Toolkit already installed."
fi

mkdir -p models

log "Starting services..."
docker compose up -d --build

log "Controller UI: http://$(hostname -I | awk '{print $1}'):8080/"
log "Admin UI:      http://$(hostname -I | awk '{print $1}'):8080/admin"
log "OpenAI API:    http://$(hostname -I | awk '{print $1}'):8080/v1"
