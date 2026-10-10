"""Train the "Hey DrivePilot" openWakeWord classifier and export it to ONNX.

Uses openWakeWord's own training code unchanged: ``openwakeword.train.Model`` (fully connected
classifier on 16 x 96 speech-embedding frames, Adam, warmup + cosine LR, hard-example mining,
linearly increasing negative-class weight) and its ``auto_train`` schedule (3 sequences, checkpoint
selection on validation recall / false positives, averaging of the best checkpoints), with
``openwakeword.data.mmap_batch_generator`` building each batch from:

    ACAV100M_sample      openWakeWord's 2000 h of precomputed general negative features (17.3 GB, mmap)
    positive             augmented "hey drive pilot" clips (train speakers only)
    adversarial_negative augmented near-miss / in-car / phonetically similar phrases
    background_negative  augmented noise-only windows (car interior, traffic, music, ...)

Validation during training (checkpoint selection only — the final numbers come from evaluate.py):
    * false positives per hour on ``validation_set_features.npy`` (~11 h DiPCo/SBCSAE/MUSDB features)
    * recall/accuracy on augmented clips from the *validation* speakers (not the test speakers)

Adaptations (no change to the method): the 3 GB false-positive window tensor is built lazily on the
GPU in chunks instead of in host RAM, and batches come from an in-process DataLoader (num_workers=0)
because Windows' spawn-based workers would each copy the generator.

Usage (training venv, from the repo root):

    .venv-wakeword/Scripts/python scripts/wakeword/train_model.py --config scripts/wakeword/training_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml
from common import (
    ACAV_FEATURES,
    MODEL_NAME,
    MODELS_DIR,
    OWW_FEATURES_DIR,
    REPO_ROOT,
    VALIDATION_FEATURES,
    setup_logging,
    write_json,
)

log = logging.getLogger("train_model")


class FalsePositiveWindows:
    """Iterable of (windows, labels) batches over a long (N, 96) feature stream, built on the GPU.

    Equivalent to train.py's ``np.array([X[i:i+16] for i in range(len(X)-16)])`` + DataLoader, without
    materialising ~3 GB of overlapping windows in host memory.
    """

    def __init__(self, features: np.ndarray, n_frames: int, device, chunk: int = 65536) -> None:
        import torch

        self.x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(device)
        self.n_frames = n_frames
        self.n_windows = self.x.shape[0] - n_frames  # same count as train.py's range(0, N - 16)
        self.chunk = chunk

    def __iter__(self):
        import torch

        windows = self.x.unfold(0, self.n_frames, 1).permute(0, 2, 1)  # (N-15, 16, 96) view
        for i in range(0, self.n_windows, self.chunk):
            xb = windows[i : min(i + self.chunk, self.n_windows)].contiguous()
            yield xb, torch.zeros(xb.shape[0], device=xb.device)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).with_name("training_config.yaml")))
    parser.add_argument("--output", default=str(MODELS_DIR / f"{MODEL_NAME}.onnx"),
                        help="ONNX path; training_history.json is written next to it. Keep the file name "
                             f"{MODEL_NAME}.onnx: openwakeword names the score after the file.")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override top-level config values, e.g. layer_size=64 steps=80000")
    args = parser.parse_args()
    setup_logging("train_model")

    import torch
    from openwakeword.data import mmap_batch_generator
    from openwakeword.train import Model

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for item in args.override:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    feat_cfg = json.loads((OWW_FEATURES_DIR / "feature_config.json").read_text(encoding="utf-8"))
    input_shape = tuple(feat_cfg["input_shape"])
    log.info("config: %s", cfg)
    log.info("input shape %s, window %d samples", input_shape, feat_cfg["total_length"])
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    oww = Model(
        n_classes=1,
        input_shape=input_shape,
        model_type=cfg["model_type"],
        layer_dim=cfg["layer_size"],
        seconds_per_example=1280 * input_shape[0] / 16000,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; training is meant to run on the GPU")
    log.info("training device: %s (%s)", oww.device, torch.cuda.get_device_name(0))

    data_files = {
        "ACAV100M_sample": str(ACAV_FEATURES),
        "positive": str(OWW_FEATURES_DIR / "positive_train.npy"),
        "adversarial_negative": str(OWW_FEATURES_DIR / "negative_train.npy"),
        "background_negative": str(OWW_FEATURES_DIR / "background_train.npy"),
    }
    for k, v in data_files.items():
        log.info("%s: %s rows", k, np.load(v, mmap_mode="r").shape)

    # same transforms as openwakeword/train.py
    def f(x, n=input_shape[0]):
        if n > x.shape[1] or n < x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i : i + n, :] for i in range(0, x.shape[0] - n, n)])
        return x

    data_transforms = {key: f for key in data_files}
    label_transforms = {
        key: ((lambda x: [1 for _ in x]) if key == "positive" else (lambda x: [0 for _ in x])) for key in data_files
    }
    batch_generator = mmap_batch_generator(
        data_files,
        n_per_class=cfg["batch_n_per_class"],
        data_transform_funcs=data_transforms,
        label_transform_funcs=label_transforms,
    )

    class IterDataset(torch.utils.data.IterableDataset):
        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            return self.generator

    x_train = torch.utils.data.DataLoader(IterDataset(batch_generator), batch_size=None, num_workers=0)

    fp_features = np.load(VALIDATION_FEATURES)
    log.info(
        "false-positive validation features: %s (%.2f h at 80 ms/frame)",
        fp_features.shape,
        fp_features.shape[0] * 0.08 / 3600,
    )
    x_val_fp = FalsePositiveWindows(fp_features, input_shape[0], oww.device)

    pos_val = np.load(OWW_FEATURES_DIR / "positive_val.npy")
    neg_val = np.load(OWW_FEATURES_DIR / "negative_val.npy")
    labels = np.hstack((np.ones(pos_val.shape[0]), np.zeros(neg_val.shape[0]))).astype(np.float32)
    x_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(np.vstack((pos_val, neg_val))), torch.from_numpy(labels)),
        batch_size=len(labels),
    )

    t0 = time.time()
    best_model = oww.auto_train(
        X_train=x_train,
        X_val=x_val,
        false_positive_val_data=x_val_fp,
        steps=cfg["steps"],
        max_negative_weight=cfg["max_negative_weight"],
        target_fp_per_hour=cfg["target_false_positives_per_hour"],
    )
    log.info("training finished in %.1f min", (time.time() - t0) / 60)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    oww.export_model(model=best_model, model_name=out.stem, output_dir=str(out.parent))
    torch.save(best_model.state_dict(), out.with_suffix(".pt"))

    # Check the ONNX export against PyTorch on real validation features.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    sample = np.vstack((pos_val[:64], neg_val[:64])).astype(np.float32)
    onnx_scores = np.array([sess.run(None, {sess.get_inputs()[0].name: s[None]})[0][0, 0] for s in sample])
    best_model.eval()
    with torch.no_grad():
        torch_scores = best_model.to("cpu")(torch.from_numpy(sample)).numpy()[:, 0]
    max_diff = float(np.abs(onnx_scores - torch_scores).max())
    log.info("ONNX vs PyTorch max abs score difference: %.2e", max_diff)
    if max_diff > 1e-4:
        raise RuntimeError("ONNX export does not match the PyTorch model")

    history = {
        k: [float(np.asarray(v).item()) if np.asarray(v).size == 1 else np.asarray(v).tolist() for v in vals]
        for k, vals in oww.history.items()
    }
    write_json(
        Path(args.output).parent / "training_history.json",
        {
            "config": cfg,
            "feature_config": feat_cfg,
            "input_shape": list(input_shape),
            "train_rows": {k: int(np.load(v, mmap_mode="r").shape[0]) for k, v in data_files.items()},
            "val_rows": {"positive": int(pos_val.shape[0]), "negative": int(neg_val.shape[0])},
            "best_model_scores": [{k: float(np.asarray(v)) for k, v in s.items()} for s in oww.best_model_scores],
            "history": history,
            "onnx_vs_torch_max_diff": max_diff,
            "train_minutes": round((time.time() - t0) / 60, 2),
        },
    )
    log.info("saved %s (%s)", out, out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
