"""Build the TensorRT FP16 engines of the sign detector and the value classifier (one-time per machine).

    .venv/Scripts/python scripts/signs/export_engines.py                 # both, from config.yaml `signs:`
    .venv/Scripts/python scripts/signs/export_engines.py --force         # rebuild even if they exist
    .venv/Scripts/python scripts/signs/export_engines.py --setup-only    # just create .venv-trtbuild
    .venv/Scripts/python scripts/signs/export_engines.py --deliver <main checkout>/models/signs

Same recipe as scripts/build_engine.py (read its docstring): ultralytics' FP16 TensorRT export needs NVIDIA
ModelOpt, whose extras would replace the voice pipeline's onnxruntime, so the export runs in a separate
`.venv-trtbuild` with the *same* TensorRT version as `.venv` (an engine only loads with the TensorRT that built
it). Engines land next to the weights as <stem>_<h>x<w>_fp16.engine (perception/signs.engine_path), each with a
sidecar <engine>.json recording the weights' sha256, input size and TensorRT version; an engine is rebuilt when
its weights changed. --deliver copies weights, engines and sidecars to another checkout's models/signs/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from perception.signs import engine_path, engine_staleness, file_sha256, sidecar_path  # noqa: E402

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
    """Create the build environment if needed. Returns its python executable."""
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


def export(python: Path, weights: Path, imgsz: tuple[int, int], half: bool, device: int, workspace: float) -> Path:
    """Export one model to TensorRT in the build venv; returns the engine path."""
    engine = engine_path(weights, imgsz, half)
    h, w = imgsz
    code = (
        "from pathlib import Path; from ultralytics import YOLO; "
        f"f = YOLO({str(weights)!r}).export(format='engine', imgsz=({h}, {w}), "
        f"quantize={16 if half else None}, device={device}, workspace={workspace}, verbose=False); "
        f"Path(f).replace({str(engine)!r}); print('built', {str(engine)!r})"
    )
    run([str(python), "-c", code], env=dict(os.environ, YOLO_AUTOINSTALL="false"))
    return engine


def write_sidecar(engine: Path, weights: Path, imgsz: tuple[int, int], half: bool, version: str) -> None:
    """Record what an engine was built from (perception/signs.py refuses an engine whose weights hash differs)."""
    data = {
        "weights": weights.name,
        "weights_sha256": file_sha256(weights),
        "imgsz": list(imgsz),
        "half": half,
        "tensorrt": version,
        "built": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    sidecar_path(engine).write_text(json.dumps(data, indent=2), encoding="utf-8")


def trt_version(python: Path) -> str:
    """TensorRT version of an environment."""
    out = subprocess.run(
        [str(python), "-c", "import tensorrt; print(tensorrt.__version__)"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    """Build the engines described by config.yaml's signs section."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true")
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--deliver", type=Path, default=None, help="copy weights + engines to this models/signs dir")
    args = p.parse_args(argv)
    cfg = load_config()
    s = cfg.signs
    det_w, cls_w = cfg.resolve_path(s.weights), cfg.resolve_path(s.classifier_weights)
    jobs = [(det_w, s.imgsz), (cls_w, (s.classifier_imgsz, s.classifier_imgsz))]
    if args.deliver is None:
        python = ensure_build_venv()
        if args.setup_only:
            print(f"build venv ready: {python} (TensorRT {trt_version(python)})")
            return 0
        version = trt_version(python)
        for weights, imgsz in jobs:
            engine = engine_path(weights, imgsz, s.half)
            if not weights.exists():
                raise SystemExit(f"weights missing: {weights} (train first: scripts/signs/train.py)")
            stale = engine_staleness(engine, weights) if engine.exists() else "missing"
            if stale is None and sidecar_path(engine).exists() and not args.force:
                print(f"engine already built from these weights: {engine}")
                continue
            export(python, weights, imgsz, s.half, s.device, 4.0)
            write_sidecar(engine, weights, imgsz, s.half, version)
            print(f"done: {engine} ({engine.stat().st_size / 1e6:.1f} MB)")
        return 0
    args.deliver.mkdir(parents=True, exist_ok=True)
    runtime = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if not runtime.exists():
        runtime = Path(sys.executable)
    for weights, imgsz in jobs:
        engine = engine_path(weights, imgsz, s.half)
        if not sidecar_path(engine).exists() or engine_staleness(engine, weights) is not None:
            raise SystemExit(f"{engine.name} is missing, stale or has no sidecar: run export_engines.py first")
        for f in (weights, engine, sidecar_path(engine)):
            shutil.copy2(f, args.deliver / f.name)
            print(f"{f.name}: {f.stat().st_size / 1e6:.2f} MB -> {args.deliver}")
    print(f"TensorRT (runtime env): {trt_version(runtime)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
