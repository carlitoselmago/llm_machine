from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass

import httpx

from .config import Config

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ManagedProcessRecord:
    process_id: str
    pid: int
    name: str
    status: str
    repo_id: str | None
    model_id: str | None
    gpu_id: str | None
    host_port: int | None
    served_model_name: str | None
    log_file: str


class ProcessManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._processes: dict[str, tuple[subprocess.Popen[bytes], ManagedProcessRecord]] = {}

    def connect(self) -> None:
        os.makedirs(self.config.process_logs_dir, exist_ok=True)

    def close(self) -> None:
        # Keep model processes running unless explicitly stopped.
        return

    def is_connected(self) -> bool:
        return True

    def list_managed_processes(self) -> list[ManagedProcessRecord]:
        with self._lock:
            records: list[ManagedProcessRecord] = []
            stale: list[str] = []
            for process_id, (proc, rec) in self._processes.items():
                code = proc.poll()
                if code is None:
                    status = "running"
                else:
                    status = f"exited({code})"
                    stale.append(process_id)
                records.append(
                    ManagedProcessRecord(
                        process_id=rec.process_id,
                        pid=rec.pid,
                        name=rec.name,
                        status=status,
                        repo_id=rec.repo_id,
                        model_id=rec.model_id,
                        gpu_id=rec.gpu_id,
                        host_port=rec.host_port,
                        served_model_name=rec.served_model_name,
                        log_file=rec.log_file,
                    )
                )
            for process_id in stale:
                self._processes.pop(process_id, None)
            return records

    def start_vllm_process(
        self,
        *,
        model_id: str,
        repo_id: str,
        gpu_id: str,
        host_port: int,
        model_ref: str,
        tokenizer_ref: str | None = None,
        hf_config_path: str | None = None,
        served_model_name: str | None = None,
        trust_remote_code: bool = False,
        dtype: str | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_num_seqs: int | None = None,
    ) -> ManagedProcessRecord:
        process_id = uuid.uuid4().hex[:12]
        name = f"{self.config.container_name_prefix}-{model_id}-{gpu_id}"
        log_file = os.path.join(self.config.process_logs_dir, f"{name}.log")

        cmd: list[str] = [
            self.config.vllm_executable,
            "serve",
            model_ref,
            "--host",
            "0.0.0.0",
            "--port",
            str(host_port),
        ]
        if served_model_name:
            cmd.extend(["--served-model-name", served_model_name])
        if trust_remote_code:
            cmd.append("--trust-remote-code")
        if dtype:
            cmd.extend(["--dtype", dtype])
        if max_model_len is not None:
            cmd.extend(["--max-model-len", str(max_model_len)])
        if gpu_memory_utilization is not None:
            cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
        if max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(max_num_seqs)])
        if tokenizer_ref:
            cmd.extend(["--tokenizer", tokenizer_ref])
        if hf_config_path:
            cmd.extend(["--hf-config-path", hf_config_path])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        os.makedirs(self.config.process_logs_dir, exist_ok=True)
        log_handle = open(log_file, "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            log_handle.close()
            raise

        rec = ManagedProcessRecord(
            process_id=process_id,
            pid=proc.pid,
            name=name,
            status="running",
            repo_id=repo_id,
            model_id=model_id,
            gpu_id=gpu_id,
            host_port=host_port,
            served_model_name=served_model_name,
            log_file=log_file,
        )
        with self._lock:
            self._processes[process_id] = (proc, rec)
        return rec

    def wait_for_model_ready(self, host_port: int, runtime_id: str | None = None) -> None:
        deadline = time.time() + self.config.vllm_startup_timeout_seconds
        url = f"http://127.0.0.1:{host_port}/v1/models"
        last_error = "unknown"
        with httpx.Client(timeout=10.0) as client:
            while time.time() < deadline:
                if runtime_id is not None:
                    proc = self._process_by_id(runtime_id)
                    if proc is not None and proc.poll() is not None:
                        raise RuntimeError(
                            f"vLLM process exited before readiness. Log tail: {self._log_tail(runtime_id)}"
                        )
                try:
                    resp = client.get(url)
                    if resp.status_code < 500:
                        return
                    last_error = f"HTTP {resp.status_code}"
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                time.sleep(self.config.vllm_health_poll_seconds)
        raise RuntimeError(f"Timed out waiting for vLLM readiness on port {host_port}: {last_error}")

    def stop_process(self, runtime_id: str) -> None:
        with self._lock:
            found = self._processes.get(runtime_id)
        if found is None:
            return
        proc, _ = found

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

        with self._lock:
            self._processes.pop(runtime_id, None)

    def find_free_host_port(self, used_ports: set[int]) -> int:
        for port in range(self.config.host_port_start, self.config.host_port_end + 1):
            if port in used_ports:
                continue
            if self._can_bind(port):
                return port
        raise RuntimeError("No free host port available in configured range")

    def _process_by_id(self, process_id: str) -> subprocess.Popen[bytes] | None:
        with self._lock:
            found = self._processes.get(process_id)
            return found[0] if found else None

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

    def _log_tail(self, process_id: str, max_lines: int = 25) -> str:
        with self._lock:
            found = self._processes.get(process_id)
            if not found:
                return "(no process found)"
            _, rec = found
            log_file = rec.log_file
        if not os.path.isfile(log_file):
            return "(no log file)"
        try:
            with open(log_file, "rb") as f:
                data = f.read().decode("utf-8", errors="replace")
            lines = data.splitlines()
            return " | ".join(lines[-max_lines:])
        except OSError:
            return "(failed to read log file)"
