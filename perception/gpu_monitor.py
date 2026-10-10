"""GPU memory and utilization sampling (PyTorch allocator + NVML device counters)."""

from __future__ import annotations

from typing import Any

import torch

from core.logging_setup import get_logger

log = get_logger(__name__)


class GpuMonitor:
    """Samples this process's CUDA allocator and the whole device (NVML). NVML failures degrade to torch-only."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._nvml: Any = None
        self._handle: Any = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as exc:  # noqa: BLE001 - NVML is optional diagnostics
            log.warning("NVML unavailable; GPU device stats limited to torch", error=str(exc))

    def sample(self) -> dict[str, float]:
        """Current memory (MB), utilization (%) and SM clock (MHz) where available."""
        out: dict[str, float] = {}
        if torch.cuda.is_available():
            out["torch_allocated_mb"] = torch.cuda.memory_allocated(self.device_index) / 1e6
            out["torch_reserved_mb"] = torch.cuda.memory_reserved(self.device_index) / 1e6
            out["torch_peak_mb"] = torch.cuda.max_memory_allocated(self.device_index) / 1e6
        if self._nvml is not None:
            try:
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                out["device_used_mb"] = mem.used / 1e6
                out["device_total_mb"] = mem.total / 1e6
                out["gpu_util_pct"] = float(util.gpu)
                out["sm_clock_mhz"] = float(self._nvml.nvmlDeviceGetClockInfo(self._handle, self._nvml.NVML_CLOCK_SM))
            except Exception as exc:  # noqa: BLE001
                log.warning("NVML sample failed", error=str(exc))
        return out

    def other_gpu_processes(self) -> list[str]:
        """Names of other processes using the GPU (games, other ML jobs) that would skew benchmarks."""
        if self._nvml is None:
            return []
        import os

        names: list[str] = []
        try:
            procs = list(self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle))
            procs += list(self._nvml.nvmlDeviceGetGraphicsRunningProcesses(self._handle))
            for proc in procs:
                if proc.pid == os.getpid():
                    continue
                try:
                    name = self._nvml.nvmlSystemGetProcessName(proc.pid)
                except Exception:  # noqa: BLE001
                    name = f"pid {proc.pid}"
                names.append(str(name).replace("\\", "/").rsplit("/", 1)[-1])
        except Exception as exc:  # noqa: BLE001
            log.warning("NVML process query failed", error=str(exc))
        return sorted(set(names))
