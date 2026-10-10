# Sign fine-tune (V2-M6): exact commands

Fine-tuned US traffic-sign detector (YOLO26s, 13 MUTCD kinds) plus a speed-limit value classifier (YOLO26n-cls,
5-85 mph or "none"), used by the app through `perception/signs.py`. Status, numbers and the write-up:
[STATUS.md](STATUS.md). Run everything from the repo root with the main venv (`.venv/Scripts/python`); data-only
helpers that need `pyarrow`/`huggingface_hub` use `.venv-signs` (created with `uv venv --python 3.11 .venv-signs`
and `uv pip install --python .venv-signs/Scripts/python.exe pyarrow pillow numpy opencv-python-headless
huggingface_hub pyyaml`). Ultralytics never auto-installs (`YOLO_AUTOINSTALL=false` is set by the scripts).

Everything under `data/signs/` is local (gitignored). Disk: about 7 GB.

## 1. Sources (downloads, no account needed)

```bash
# BDD100K detection frames with human "traffic sign" boxes (train shards 0-1 and valid), ~2.5 GB
.venv-signs/Scripts/python -c "from huggingface_hub import snapshot_download as s; s('dronefreak/BDD100K', repo_type='dataset', local_dir='data/signs/raw/bdd100k_det', allow_patterns=['data/images/valid/*', 'data/images/train/shard_000/*', 'data/images/train/shard_001/*'], max_workers=16)"
# ctacke/speed-limit-signs: US dashcam speed limits 25-80 mph (MIT), 1.7 GB
.venv-signs/Scripts/python -c "from huggingface_hub import snapshot_download as s; s('ctacke/speed-limit-signs', repo_type='dataset', local_dir='data/signs/raw/ctacke')"
```

Sign templates (public-domain MUTCD renders from Wikimedia Commons) go to `data/signs/templates/`
(`MUTCD_R1-1.png`, `R1-2`, `R2-1`, `R3-1`, `R3-2`, `R3-4`, `R5-1`, `R5-1a`, `R6-1L`, `R6-1R`, `R6-2R`, `R10-11`,
`R10-11a`, `R10-11b`, `S1-1`, `S5-1`, `W11-2`), fetched with
`curl -L -o data/signs/templates/MUTCD_R1-1.png "https://commons.wikimedia.org/wiki/Special:FilePath/MUTCD_R1-1.svg?width=400"`
(one per code). The number font is Overpass (SIL OFL):
`curl -L -o data/signs/fonts/Overpass.ttf "https://github.com/google/fonts/raw/main/ofl/overpass/Overpass%5Bwght%5D.ttf"`.

LISA (if a Roboflow export is placed in `data/signs/raw/lisa_yolo/{train,valid}/{images,labels}` with its
`data.yaml`) is picked up automatically by `prepare_data.py`; its labels are mapped by
`perception/sign_classes.kind_from_label`.

## 2. Labels (Claude, costs money: about $0.83 per 1,000 crops with Opus 5.5)

```bash
# Kind (+ mph when readable) of every BDD sign box; unclear boxes are masked later, never background.
.venv/Scripts/python scripts/signs/label_bdd_signs.py --split train --shards shard_000 --min-h 14 --env-file .env
.venv/Scripts/python scripts/signs/label_bdd_signs.py --split train --shards shard_001 --min-h 14 --out data/signs/bdd_labels/train_s1.jsonl --env-file .env
.venv/Scripts/python scripts/signs/label_bdd_signs.py --split valid --shards shard_000 --limit 9000 --env-file .env
# Which ctacke crops are really readable (the dataset labels every frame of a sequence, even dark/blurred ones)
.venv/Scripts/python scripts/signs/verify_ctacke.py --env-file .env
# Held-out set (BDD valid) with the hand review in make_heldout.py -> eval/sign_labels_bdd.json (tracked)
.venv/Scripts/python scripts/signs/make_heldout.py
# Contact sheets for checking labels by eye
.venv/Scripts/python scripts/signs/review_sheet.py data/signs/bdd_labels/valid.jsonl --kind speed_limit
```

## 3. Datasets

```bash
.venv/Scripts/python scripts/signs/prepare_data.py det    # 640x640 tiles at native scale -> data/signs/det (~2 min)
.venv/Scripts/python scripts/signs/prepare_data.py cls    # 96x96 value crops -> data/signs/cls (~8 min)
```

## 4. Training (shared GPU: batch 8 keeps the detector at ~2.6 GB)

```bash
.venv/Scripts/python scripts/signs/train.py det --epochs 40 --batch 8 --workers 4    # ~4 h; checkpoint every epoch
.venv/Scripts/python scripts/signs/train.py det --resume                             # after a stop
.venv/Scripts/python scripts/signs/train.py cls --epochs 20                          # ~10 min
```

Best weights are copied to `models/signs/signs_yolo26s.pt` and `models/signs/speed_value_cls.pt`.

## 5. Evaluation

```bash
.venv/Scripts/python scripts/signs/eval_values.py --write-metrics      # classifier alone; picks value_conf on validation
.venv/Scripts/python scripts/signs/evaluate.py all --write-metrics     # detector + classifier as the app runs them
.venv/Scripts/python scripts/signs/bench_signs.py                      # per-frame time on the GPU
```

Metrics go to `models/signs/metrics.json` (tracked).

## 6. TensorRT engines and delivery

```bash
.venv/Scripts/python scripts/signs/export_engines.py            # creates .venv-trtbuild on first use, ~10 min per engine
.venv/Scripts/python scripts/signs/export_engines.py --deliver C:/ClaudeProjects/DrivePilot_BananaHacks2026/models/signs
```

## Using it in the app

```python
from core.config import load_config
from perception.signs import SignDetector

cfg = load_config()
signs = SignDetector(cfg.signs, cfg.root)  # loads both engines, verifies CUDA + FP16, refuses the CPU
for d in signs.detect(frame):  # frame: core.types.Frame (1280x720 or 1920x1080)
    print(d.kind, d.value, d.conf, d.box)  # value: mph for speed limits read confidently, else None
```

Call `SignDetector` from one thread (the one that constructed it, or call `warmup()` there first). A value is
reported only when the classifier is at least `signs.value_conf` sure and the sign is at least `signs.value_min_px`
tall (at 1280 px frame width). Announcing a new limit should still wait for two agreeing frames (V2_BUILD_PLAN).
