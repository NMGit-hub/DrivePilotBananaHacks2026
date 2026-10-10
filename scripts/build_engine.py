"""Build the TensorRT engine for the detector (one-time per machine, ~7-10 min).

    .venv/Scripts/python scripts/build_engine.py            # uses perception.* from config.yaml
    .venv/Scripts/python scripts/build_engine.py --force    # rebuild even if the engine exists

Why a separate environment: ultralytics' FP16 TensorRT export converts the ONNX graph with NVIDIA ModelOpt,
whose ONNX extras pull in onnxruntime-gpu, which collides with the onnxruntime the voice pipeline uses. So the
export runs in `.venv-trtbuild` (created here with uv, gitignored) with the *same* TensorRT version as the
runtime environment (an engine only loads with the TensorRT version that built it), and the runtime `.venv`
only needs `tensorrt` to load the result.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402

BUILD_VENV = REPO_ROOT / ".venv-trtbuild"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"


def pinned(name: str) -> str:
    """The exact `name==version` pin from requirements.txt (so both environments match)."""
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec.lower().startswith(name.lower() + "=="):
            return spec
    raise SystemExit(f"{name} is not pinned in requirements.txt")


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Run a command, streaming its output; exit on failure."""
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {cmd[0]}")


def ensure_build_venv() -> Path:
    """Create the build environment if it doesn't exist yet. Returns its python executable."""
    python = BUILD_VENV / "Scripts" / "python.exe"
    if not python.exists():
        run(["uv", "venv", "--python", "3.11", str(BUILD_VENV)])
    packages = [
        pinned("torch"),
        pinned("torchvision"),
        pinned("ultralytics"),
        pinned("tensorrt-cu12"),
        "onnx",
        "onnxslim",
        "nvidia-modelopt[onnx]>=0.44",
    ]
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--extra-index-url",
            TORCH_INDEX,
            "--index-strategy",
            "unsafe-best-match",
            *packages,
        ]
    )
    return python


def main(argv: list[str] | None = None) -> int:
    """Build the engine described by config.yaml's perception section."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild even if the engine already exists")
    args = parser.parse_args(argv)
    cfg = load_config()
    p = cfg.perception
    weights = cfg.resolve_path(p.weights)
    h, w = p.imgsz
    engine = weights.with_name(f"{weights.stem}_{h}x{w}_{'fp16' if p.half else 'fp32'}.engine")
    if engine.exists() and not args.force:
        print(f"engine already built: {engine}")
        return 0
    python = ensure_build_venv()
    export = (
        "from pathlib import Path; from ultralytics import YOLO; "
        f"f = YOLO({str(weights)!r}).export(format='engine', imgsz=({h}, {w}), "
        f"quantize={16 if p.half else None}, device={p.device}, workspace={p.engine_workspace_gb}, verbose=False); "
        f"Path(f).replace({str(engine)!r}); print('built', {str(engine)!r})"
    )
    env = dict(os.environ, YOLO_AUTOINSTALL="false")  # every dependency is installed explicitly above
    weights.parent.mkdir(parents=True, exist_ok=True)
    run([str(python), "-c", export], env=env)
    print(f"done: {engine} ({engine.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
