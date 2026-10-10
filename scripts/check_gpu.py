"""Verify that the GPU is actually used (not just available) by PyTorch, YOLO26s and CTranslate2.

Run from the repo root:  .venv/Scripts/python scripts/check_gpu.py
Exits with status 1 if any check fails.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _nvml_used_mb() -> float | None:
    """Return device-wide GPU memory in use (MB) via NVML, or None if NVML is unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e6
        pynvml.nvmlShutdown()
        return used
    except Exception:  # noqa: BLE001 - diagnostic script, any NVML failure just means "unknown"
        return None


def check_torch() -> bool:
    """Run an fp16 matmul on the GPU and confirm the result tensor lives on CUDA."""
    import torch

    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return False
    print(f"device: {torch.cuda.get_device_name(0)}")
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(5):  # warm-up so clocks ramp up
        x @ x
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        y = x @ x
    torch.cuda.synchronize()
    tflops = 50 * 2 * 4096**3 / (time.perf_counter() - start) / 1e12
    ok = y.device.type == "cuda"
    print(f"{'OK' if ok else 'FAIL'}: fp16 matmul ran on {y.device}, ~{tflops:.1f} TFLOPS")
    return ok


def check_yolo() -> bool:
    """Run YOLO26s and confirm the predictor's model weights sit on the GPU in half precision."""
    import torch
    from ultralytics import YOLO

    weights = REPO_ROOT / "models" / "yolo26s.pt"
    model = YOLO(str(weights) if weights.exists() else "yolo26s.pt")
    frame = (np.random.default_rng(0).random((1080, 1920, 3)) * 255).astype(np.uint8)
    for _ in range(3):
        model.predict(frame, device=0, half=True, verbose=False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    n = 30
    result: Any = None
    for _ in range(n):
        result = model.predict(frame, device=0, half=True, verbose=False)[0]  # type: ignore[index]
    torch.cuda.synchronize()
    per_frame_ms = (time.perf_counter() - start) / n * 1000
    # AutoBackend wraps the real network as `.model`; check the actual weights, not a wrapper attribute.
    assert model.predictor is not None
    param = next(model.predictor.model.model.parameters())
    ok = param.device.type == "cuda" and param.dtype == torch.float16
    print(
        f"{'OK' if ok else 'FAIL'}: YOLO26s weights on {param.device} ({param.dtype}); "
        f"{per_frame_ms:.1f} ms/frame end-to-end at 1080p; stage ms {result.speed}"
    )
    return ok


def check_ctranslate2() -> bool:
    """Confirm CTranslate2 (faster-whisper's backend) sees the GPU and supports float16."""
    # Import torch first: it puts its bundled cuBLAS/cuDNN DLLs on the search path that CTranslate2 needs.
    import torch  # noqa: F401, I001
    import ctranslate2

    count = ctranslate2.get_cuda_device_count()
    types = ctranslate2.get_supported_compute_types("cuda") if count else set()
    ok = count > 0 and "float16" in types
    status = "OK" if ok else "FAIL"
    print(f"{status}: CTranslate2 {ctranslate2.__version__}, CUDA devices={count}, types={sorted(types)}")
    return ok


def main() -> int:
    """Run all GPU checks and report device memory use."""
    results = [check_torch(), check_yolo(), check_ctranslate2()]
    used = _nvml_used_mb()
    if used is not None:
        print(f"GPU memory in use (device-wide, NVML): {used:.0f} MB")
    print("ALL OK" if all(results) else "SOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
