from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from typing import Any

import docker
import httpx
from docker.errors import APIError, DockerException, NotFound
from docker.types import DeviceRequest

from .config import Config

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ManagedContainerRecord:
    container_id: str
    name: str
    status: str
    repo_id: str | None
    model_id: str | None
    gpu_id: str | None
    host_port: int | None
    served_model_name: str | None


class DockerManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.client: docker.DockerClient | None = None

    def connect(self) -> None:
        if self.client is not None:
            return
        try:
            self.client = docker.from_env()
            self.client.ping()
        except DockerException as exc:
            raise RuntimeError(f"Failed to connect to Docker daemon: {exc}") from exc

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def is_connected(self) -> bool:
        return self.client is not None

    def pull_image_if_needed(self, image: str | None = None) -> None:
        self._require_client()
        target = image or self.config.vllm_image
        try:
            self.client.images.pull(target)
        except APIError as exc:
            raise RuntimeError(f"Failed to pull image {target}: {exc.explanation}") from exc

    def list_managed_containers(self) -> list[ManagedContainerRecord]:
        self._require_client()
        label_filter = f"{self.config.managed_label_key}=true"
        containers = self.client.containers.list(all=True, filters={"label": label_filter})
        records: list[ManagedContainerRecord] = []
        for container in containers:
            labels = container.labels or {}
            host_port_label = labels.get(self.config.label_host_port)
            records.append(
                ManagedContainerRecord(
                    container_id=container.id,
                    name=container.name,
                    status=container.status,
                    repo_id=labels.get(self.config.label_repo_id),
                    model_id=labels.get(self.config.label_model_id),
                    gpu_id=labels.get(self.config.label_gpu_id),
                    host_port=int(host_port_label) if host_port_label and host_port_label.isdigit() else self._extract_host_port(container.attrs),
                    served_model_name=labels.get(self.config.label_served_model_name),
                )
            )
        return records

    def start_vllm_container(
        self,
        *,
        model_id: str,
        repo_id: str,
        gpu_id: str,
        host_port: int,
        served_model_name: str | None = None,
        trust_remote_code: bool = False,
        dtype: str | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_num_seqs: int | None = None,
    ) -> ManagedContainerRecord:
        self._require_client()
        container_name = f"{self.config.container_name_prefix}-{model_id}-{gpu_id}"
        labels = {
            self.config.managed_label_key: "true",
            self.config.label_repo_id: repo_id,
            self.config.label_model_id: model_id,
            self.config.label_gpu_id: gpu_id,
            self.config.label_host_port: str(host_port),
        }
        if served_model_name:
            labels[self.config.label_served_model_name] = served_model_name

        cmd = [f"/models/{model_id}", "--port", str(self.config.vllm_internal_port)]
        if served_model_name:
            cmd += ["--served-model-name", served_model_name]
        if trust_remote_code:
            cmd.append("--trust-remote-code")
        if dtype:
            cmd += ["--dtype", dtype]
        if max_model_len is not None:
            cmd += ["--max-model-len", str(max_model_len)]
        if gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
        if max_num_seqs is not None:
            cmd += ["--max-num-seqs", str(max_num_seqs)]

        for attempt in (1, 2):
            try:
                container = self.client.containers.run(
                    self.config.vllm_image,
                    command=cmd,
                    name=container_name,
                    detach=True,
                    remove=False,
                    labels=labels,
                    ports={f"{self.config.vllm_internal_port}/tcp": ("0.0.0.0", host_port)},
                    volumes={self.config.models_dir: {"bind": "/models", "mode": "rw"}},
                    device_requests=[DeviceRequest(device_ids=[gpu_id], capabilities=[["gpu"]])],
                    restart_policy={"Name": "unless-stopped"},
                )
                break
            except APIError as exc:
                explanation = str(exc.explanation or exc)
                lowered = explanation.lower()
                cleaned = self._remove_stale_managed_container(container_name, model_id=model_id, gpu_id=gpu_id)
                if attempt == 1 and "already in use by container" in lowered and cleaned:
                    logger.warning("Removed stale container %s and retrying start", container_name)
                    continue
                if 'capabilities: [[gpu]]' in lowered:
                    detail = (
                        "Docker GPU runtime is unavailable. Install/configure NVIDIA Container Toolkit "
                        "and restart Docker."
                    )
                    if cleaned:
                        detail += f" Removed stale container {container_name} from failed start."
                    raise RuntimeError(f"Failed to start vLLM container: {detail}") from exc
                if cleaned:
                    explanation = f"{explanation}. Removed stale container {container_name} from failed start."
                raise RuntimeError(f"Failed to start vLLM container: {explanation}") from exc

        return ManagedContainerRecord(
            container_id=container.id,
            name=container.name,
            status=container.status,
            repo_id=repo_id,
            model_id=model_id,
            gpu_id=gpu_id,
            host_port=host_port,
            served_model_name=served_model_name,
        )

    def wait_for_model_ready(self, host_port: int, *, container_id: str | None = None) -> None:
        deadline = time.time() + self.config.vllm_startup_timeout_seconds
        url = f"http://127.0.0.1:{host_port}/v1/models"
        last_error = "unknown"
        restarting_hits = 0
        with httpx.Client(timeout=10.0) as client:
            while time.time() < deadline:
                if container_id:
                    status = self._get_container_status(container_id)
                    if status in {"exited", "dead"}:
                        reason = self._extract_container_failure_summary(container_id)
                        raise RuntimeError(f"vLLM container exited before readiness: {reason}")
                    if status == "restarting":
                        restarting_hits += 1
                        if restarting_hits >= 3:
                            reason = self._extract_container_failure_summary(container_id)
                            raise RuntimeError(f"vLLM container is restarting repeatedly: {reason}")
                    else:
                        restarting_hits = 0
                try:
                    resp = client.get(url)
                    if resp.status_code < 500:
                        return
                    last_error = f"HTTP {resp.status_code}"
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                time.sleep(self.config.vllm_health_poll_seconds)
        if container_id:
            reason = self._extract_container_failure_summary(container_id)
            raise RuntimeError(f"Timed out waiting for vLLM readiness on port {host_port}: {last_error}. Last container error: {reason}")
        raise RuntimeError(f"Timed out waiting for vLLM readiness on port {host_port}: {last_error}")

    def stop_and_remove_container(self, container_id: str) -> None:
        self._require_client()
        try:
            container = self.client.containers.get(container_id)
        except NotFound:
            return
        try:
            container.stop(timeout=30)
        except APIError:
            logger.warning("Container stop failed for %s; forcing remove", container_id)
        try:
            container.remove(force=True)
        except (APIError, NotFound) as exc:
            raise RuntimeError(f"Failed to remove container {container_id}: {exc}") from exc

    def _remove_stale_managed_container(self, container_name: str, *, model_id: str, gpu_id: str) -> bool:
        self._require_client()
        try:
            container = self.client.containers.get(container_name)
        except NotFound:
            return False
        except APIError:
            return False

        labels = container.labels or {}
        if labels.get(self.config.managed_label_key) != "true":
            return False
        if labels.get(self.config.label_model_id) != model_id:
            return False
        if labels.get(self.config.label_gpu_id) != gpu_id:
            return False

        try:
            container.reload()
            if container.status == "running":
                return False
            container.remove(force=True)
            return True
        except APIError:
            return False

    def _get_container_status(self, container_id: str) -> str | None:
        self._require_client()
        try:
            container = self.client.containers.get(container_id)
            container.reload()
            return container.status
        except (NotFound, APIError):
            return None

    def _extract_container_failure_summary(self, container_id: str) -> str:
        self._require_client()
        try:
            container = self.client.containers.get(container_id)
            raw = container.logs(tail=200)
        except (NotFound, APIError) as exc:
            return f"unable to read container logs: {exc}"

        text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "no logs available"

        priority_patterns = (
            "ValueError:",
            "RuntimeError:",
            "failed to",
            "error:",
        )
        for pattern in priority_patterns:
            for line in reversed(lines):
                if pattern.lower() in line.lower():
                    return line
        return lines[-1]

    def find_free_host_port(self, used_ports: set[int]) -> int:
        for port in range(self.config.host_port_start, self.config.host_port_end + 1):
            if port in used_ports:
                continue
            if self._can_bind(port):
                return port
        raise RuntimeError("No free host port available in configured range")

    def _require_client(self) -> None:
        if self.client is None:
            raise RuntimeError("Docker client is not connected")

    @staticmethod
    def _can_bind(port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    @staticmethod
    def _extract_host_port(attrs: dict[str, Any]) -> int | None:
        ports = attrs.get("NetworkSettings", {}).get("Ports", {})
        for bindings in ports.values():
            if not bindings:
                continue
            host_port = bindings[0].get("HostPort")
            if host_port and str(host_port).isdigit():
                return int(host_port)
        return None
