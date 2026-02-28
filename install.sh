#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[install] %s\n' "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

COMPOSE_CMD=()

resolve_compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
    return
  fi
  if need_cmd docker-compose; then
    COMPOSE_CMD=(docker-compose)
    return
  fi
}

has_systemd() {
  [[ -d /run/systemd/system ]] && need_cmd systemctl
}

wait_for_docker() {
  local retries="${1:-20}"
  local sleep_seconds="${2:-1}"
  local i
  for ((i = 1; i <= retries; i++)); do
    if docker info >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_seconds"
  done
  return 1
}

launch_dockerd_manual() {
  local dockerd_log="$1"
  local dockerd_flags="$2"
  if [[ -n "$SUDO" ]]; then
    $SUDO nohup dockerd --host=unix:///var/run/docker.sock $dockerd_flags >"$dockerd_log" 2>&1 &
  else
    nohup dockerd --host=unix:///var/run/docker.sock $dockerd_flags >"$dockerd_log" 2>&1 &
  fi
}

start_docker_without_systemd() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi

  if need_cmd service; then
    log "systemd is unavailable; trying 'service docker start'..."
    if $SUDO service docker start >/dev/null 2>&1; then
      configure_docker_socket
      if wait_for_docker 15 1; then
        log "Docker service is running (service mode)."
        return 0
      fi
    fi
  fi

  if ! need_cmd dockerd; then
    return 1
  fi

  local dockerd_log="${DOCKERD_LOG_PATH:-/tmp/dockerd.log}"
  local dockerd_flags="${DOCKERD_FLAGS:-}"
  local dockerd_retry_flags="${DOCKERD_RETRY_FLAGS:---iptables=false --bridge=none --ip-forward=false --ip-masq=false --storage-driver=vfs}"
  mkdir -p /var/run
  log "systemd is unavailable; trying to start dockerd manually..."
  launch_dockerd_manual "$dockerd_log" "$dockerd_flags"

  export DOCKER_HOST="unix:///var/run/docker.sock"
  export DOCKER_SOCK_PATH="/var/run/docker.sock"
  if wait_for_docker 25 1; then
    log "dockerd is running (manual mode)."
    return 0
  fi

  if grep -qiE 'iptables|NAT chain DOCKER|Permission denied' "$dockerd_log" 2>/dev/null; then
    log "dockerd failed with iptables/bridge permissions; retrying in reduced-network mode..."
    pkill -f 'dockerd --host=unix:///var/run/docker.sock' >/dev/null 2>&1 || true
    sleep 1
    launch_dockerd_manual "$dockerd_log" "$dockerd_retry_flags"
    if wait_for_docker 25 1; then
      log "dockerd is running (reduced-network mode)."
      return 0
    fi
  fi

  log "dockerd failed to start in this environment."
  log "Last dockerd logs:"
  tail -n 80 "$dockerd_log" || true
  return 1
}

configure_docker_socket() {
  if [[ -n "${DOCKER_HOST:-}" ]]; then
    if [[ "${DOCKER_HOST}" == unix://* ]]; then
      export DOCKER_SOCK_PATH="${DOCKER_HOST#unix://}"
    fi
    return
  fi

  local rootless_sock="/run/user/$(id -u)/docker.sock"
  if [[ -S /var/run/docker.sock ]]; then
    export DOCKER_HOST="unix:///var/run/docker.sock"
    export DOCKER_SOCK_PATH="/var/run/docker.sock"
    return
  fi
  if [[ -S "$rootless_sock" ]]; then
    export DOCKER_HOST="unix://$rootless_sock"
    export DOCKER_SOCK_PATH="$rootless_sock"
    log "Using rootless Docker socket: $DOCKER_HOST"
    return
  fi

  if need_cmd docker; then
    local context_host=""
    context_host="$(docker context inspect --format '{{(index .Endpoints "docker").Host}}' 2>/dev/null || true)"
    if [[ "$context_host" == unix://* ]]; then
      local context_sock="${context_host#unix://}"
      if [[ -S "$context_sock" ]]; then
        export DOCKER_HOST="$context_host"
        export DOCKER_SOCK_PATH="$context_sock"
        log "Using Docker context socket: $DOCKER_HOST"
        return
      fi
    fi
  fi
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

resolve_compose_cmd
if [[ ${#COMPOSE_CMD[@]} -eq 0 ]]; then
  log "Installing Docker Compose plugin..."
  $SUDO apt-get update
  $SUDO apt-get install -y docker-compose-plugin
  resolve_compose_cmd
else
  log "Docker Compose is available."
fi
if [[ ${#COMPOSE_CMD[@]} -eq 0 ]]; then
  log "Docker Compose is not available (neither 'docker compose' nor 'docker-compose')."
  exit 1
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
  if has_systemd; then
    $SUDO systemctl restart docker || true
  else
    log "Skipping Docker restart: systemd is not available in this environment."
  fi
else
  log "NVIDIA Container Toolkit already installed."
fi

mkdir -p models

configure_docker_socket
if ! docker info >/dev/null 2>&1; then
  if has_systemd; then
    log "Docker daemon not reachable; trying to start docker service..."
    $SUDO systemctl start docker || true
    configure_docker_socket
  else
    start_docker_without_systemd || true
    configure_docker_socket
  fi
fi

if ! docker info >/dev/null 2>&1; then
  log "Docker daemon is not reachable."
  log "Current DOCKER_HOST: ${DOCKER_HOST:-<unset>}"
  log "If Docker is rootless, set these and rerun:"
  log "  export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock"
  log "  export DOCKER_SOCK_PATH=/run/user/$(id -u)/docker.sock"
  log "If this is a non-privileged RunPod/container, Docker-in-Docker may be blocked."
  log "Use a pod/template with Docker daemon support, or mount a host docker socket."
  exit 1
fi

log "Starting services..."
"${COMPOSE_CMD[@]}" up -d --build

log "Controller UI: http://$(hostname -I | awk '{print $1}'):8080/"
log "Admin UI:      http://$(hostname -I | awk '{print $1}'):8080/admin"
log "OpenAI API:    http://$(hostname -I | awk '{print $1}'):8080/v1"
