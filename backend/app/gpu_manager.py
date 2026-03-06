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
    free_memory_mb: int | None = None
    total_memory_mb: int | None = None
    allocated_model_id: str | None = None


class GPUManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._gpus: dict[str, _GpuRecord] = {}

    def refresh(self) -> list[GpuInfo]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
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
            parts = [part.strip() for part in line.split(",", maxsplit=3)]
            if len(parts) != 4:
                continue
            gpu_id, name, free_mem_raw, total_mem_raw = parts
            try:
                free_mem = int(free_mem_raw)
            except ValueError:
                free_mem = None
            try:
                total_mem = int(total_mem_raw)
            except ValueError:
                total_mem = None
            previous = self._gpus.get(gpu_id)
            discovered[gpu_id] = _GpuRecord(
                gpu_id=gpu_id,
                name=name,
                free_memory_mb=free_mem,
                total_memory_mb=total_mem,
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

    def allocate(self, model_id: str, min_free_ratio: float | None = None) -> str:
        with self._lock:
            candidates = [rec for rec in self._gpus.values() if rec.allocated_model_id is None]
            if min_free_ratio is not None:
                threshold = max(0.0, min(float(min_free_ratio), 1.0))
                eligible = [
                    rec
                    for rec in candidates
                    if rec.free_memory_mb is None
                    or rec.total_memory_mb is None
                    or rec.total_memory_mb <= 0
                    or (rec.free_memory_mb / rec.total_memory_mb) >= threshold
                ]
                if not eligible:
                    observed = []
                    for rec in sorted(candidates, key=lambda r: int(r.gpu_id)):
                        if rec.free_memory_mb is None or rec.total_memory_mb is None or rec.total_memory_mb <= 0:
                            observed.append(f"{rec.gpu_id}:unknown")
                        else:
                            ratio = rec.free_memory_mb / rec.total_memory_mb
                            observed.append(
                                f"{rec.gpu_id}:{rec.free_memory_mb}/{rec.total_memory_mb}MiB ({ratio:.2f})"
                            )
                    observed_text = ", ".join(observed) if observed else "none"
                    raise RuntimeError(
                        "No free GPU meets required free-memory ratio "
                        f"{threshold:.2f}. Available free ratios: {observed_text}. "
                        "Lower GPU memory utilization or free GPU memory."
                    )
                candidates = eligible
            candidates.sort(
                key=lambda rec: (
                    rec.free_memory_mb if rec.free_memory_mb is not None else -1,
                    -int(rec.gpu_id),
                ),
                reverse=True,
            )
            for rec in candidates:
                rec.allocated_model_id = model_id
                if rec.free_memory_mb is not None and rec.total_memory_mb is not None:
                    logger.info(
                        "Allocated GPU %s to model %s (free %s MiB / total %s MiB)",
                        rec.gpu_id,
                        model_id,
                        rec.free_memory_mb,
                        rec.total_memory_mb,
                    )
                else:
                    logger.info("Allocated GPU %s to model %s", rec.gpu_id, model_id)
                return rec.gpu_id
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
