"""Fine-tune the sign detector (YOLO26s) or train the speed-value classifier (YOLO26n-cls) on the laptop GPU.

    .venv/Scripts/python scripts/signs/train.py det --epochs 40 --batch 12      # ~overnight
    .venv/Scripts/python scripts/signs/train.py det --resume                     # continue a stopped run
    .venv/Scripts/python scripts/signs/train.py cls --epochs 25

Runs land in runs/signs/<det|cls>/ and save a checkpoint every epoch, so a run can be stopped (the GPU is shared
with the app) and resumed. When a run finishes, its best weights are copied to models/signs/ under the names
config.yaml's `signs:` section expects. Horizontal flips are off: they would turn "no left turn" into "no right
turn", mirror one-way arrows and digits.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never let ultralytics pip-install into the shared venv

RUNS = REPO_ROOT / "runs" / "signs"
MODELS = REPO_ROOT / "models" / "signs"
BASE_DET = Path("C:/ClaudeProjects/DrivePilot_BananaHacks2026/models/yolo26s.pt")


def train_det(args: argparse.Namespace) -> Path:
    """Fine-tune YOLO26s on data/signs/det; returns the best weights."""
    from ultralytics import YOLO

    run = RUNS / args.name
    last = run / "weights" / "last.pt"
    if args.resume:
        model = YOLO(str(last))
        model.train(resume=True, workers=args.workers)  # fewer workers = less RAM on the shared machine
    else:
        base = Path(args.base) if args.base else (REPO_ROOT / "models" / "yolo26s.pt")
        if not base.exists():
            base = BASE_DET
        model = YOLO(str(base))
        model.train(
            data=str(REPO_ROOT / "data" / "signs" / "det" / "data.yaml"),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            device=0,
            cache=False,  # the shared machine has no RAM to spare for the whole dataset
            project=str(RUNS),
            name=args.name,
            exist_ok=True,
            save_period=1,
            patience=args.patience,
            fliplr=0.0,
            scale=args.scale,
            mosaic=1.0,
            close_mosaic=args.close_mosaic,
            cos_lr=True,
            optimizer=args.optimizer,  # "auto" ignores lr0 (it picks SGD 0.01 for long runs): name one to fine-tune
            lr0=args.lr0,
            warmup_epochs=args.warmup_epochs,
            plots=True,
            seed=0,
            fraction=args.fraction,
            val=args.val,
        )
    best = run / "weights" / "best.pt"
    if args.no_copy:
        return best
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, MODELS / "signs_yolo26s.pt")
    print(f"best detector weights -> {MODELS / 'signs_yolo26s.pt'}")
    return best


def train_cls(args: argparse.Namespace) -> Path:
    """Train the speed-value classifier on data/signs/cls; returns the best weights."""
    from ultralytics import YOLO

    run = RUNS / args.name
    if args.resume:
        model = YOLO(str(run / "weights" / "last.pt"))
        model.train(resume=True, workers=args.workers)  # fewer workers = less RAM on the shared machine
    else:
        model = YOLO(str(MODELS / "yolo26n-cls.pt"))
        model.train(
            data=str(REPO_ROOT / "data" / "signs" / "cls"),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            device=0,
            cache=False,
            project=str(RUNS),
            name=args.name,
            exist_ok=True,
            save_period=1,
            patience=args.patience,
            fliplr=0.0,
            erasing=0.0,
            dropout=args.dropout,  # fewer overconfident readings of blurred numbers
            auto_augment=None,
            scale=0.1,  # RandomResizedCrop keeps >= 90% of the crop: the digits must stay in view
            hsv_h=0.01,
            hsv_s=0.3,
            hsv_v=0.4,
            cos_lr=True,
            seed=0,
        )
    best = run / "weights" / "best.pt"
    if args.no_copy:
        return best
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, MODELS / "speed_value_cls.pt")
    print(f"best classifier weights -> {MODELS / 'speed_value_cls.pt'}")
    return best


def main(argv: list[str] | None = None) -> int:
    """Train one model."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["det", "cls"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--imgsz", type=int, default=None, help="det: 640 (tiles at native scale); cls: 96")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--scale", type=float, default=0.3, help="det: random rescale +-; small keeps signs realistic")
    p.add_argument("--close-mosaic", type=int, default=5)
    p.add_argument("--fraction", type=float, default=1.0, help="det: train on this fraction of the tiles")
    p.add_argument("--base", default=None, help="det: starting weights (default: COCO yolo26s.pt)")
    p.add_argument("--lr0", type=float, default=0.01, help="det: initial learning rate (lower for a fine-tune)")
    p.add_argument("--optimizer", default="auto", help="det: auto (lr chosen by ultralytics, lr0 ignored), AdamW, SGD")
    p.add_argument("--warmup-epochs", type=float, default=3.0, help="det: warmup epochs (0 for a fine-tune)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dropout", type=float, default=0.2, help="cls: dropout before the classifier head")
    p.add_argument("--name", default=None, help="run folder under runs/signs (default: det / cls)")
    p.add_argument("--no-copy", action="store_true", help="don't copy the best weights to models/signs")
    p.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)
    args.name = args.name or args.what
    if args.imgsz is None:
        args.imgsz = 640 if args.what == "det" else 96
    if args.what == "det":
        train_det(args)
    else:
        if args.batch == 12:
            args.batch = 256
        train_cls(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
