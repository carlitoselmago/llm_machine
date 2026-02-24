from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass

from .schemas import GpuInfo

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _GpuRecord:
    gpu_id: str
    name: str
    allocated_model_id: str | None = None


class GPUManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._gpus: dict[str, _GpuRecord] = {}

    def refresh(self) -> list[GpuInfo]:
        cmd = ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("nvidia-smi not found. NVIDIA runtime/toolkit not available.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"nvidia-smi failed: {exc.stderr.strip() or exc.stdout.strip()}") from exc

        discovered: dict[str, _GpuRecord] = {}
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",", maxsplit=1)]
            if len(parts) != 2:
                continue
            gpu_id, name = parts
            previous = self._gpus.get(gpu_id)
            discovered[gpu_id] = _GpuRecord(
                gpu_id=gpu_id,
                name=name,
                allocated_model_id=previous.allocated_model_id if previous else None,
            )

        with self._lock:
            self._gpus = discovered
            return self.list_gpus()

    def list_gpus(self) -> list[GpuInfo]:
        with self._lock:
            return [
                GpuInfo(
                    gpu_id=rec.gpu_id,
                    name=rec.name,
                    allocated=rec.allocated_model_id is not None,
                    allocated_model_id=rec.allocated_model_id,
                )
                for rec in sorted(self._gpus.values(), key=lambda r: int(r.gpu_id))
            ]

    def allocate(self, model_id: str) -> str:
        with self._lock:
            for gpu_id in sorted(self._gpus.keys(), key=lambda x: int(x)):
                rec = self._gpus[gpu_id]
                if rec.allocated_model_id is None:
                    rec.allocated_model_id = model_id
                    logger.info("Allocated GPU %s to model %s", gpu_id, model_id)
                    return gpu_id
        raise RuntimeError("No free GPU available")

    def reserve_existing(self, gpu_id: str, model_id: str) -> None:
        with self._lock:
            rec = self._gpus.get(gpu_id)
            if rec is None:
                raise RuntimeError(f"GPU {gpu_id} not detected")
            if rec.allocated_model_id and rec.allocated_model_id != model_id:
                raise RuntimeError(f"GPU {gpu_id} already allocated to {rec.allocated_model_id}")
            rec.allocated_model_id = model_id

    def release(self, gpu_id: str) -> None:
        with self._lock:
            rec = self._gpus.get(gpu_id)
            if rec:
                rec.allocated_model_id = None
