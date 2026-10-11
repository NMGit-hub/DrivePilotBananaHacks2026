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

Look-alike "distractors" (never labeled: route shields and number plates that must not become speed limits) go to
`data/signs/templates/distractors/`, same Special:FilePath URL with `?width=300`: `I-5`, `I-15`, `I-25`, `I-35`,
`I-40`, `I-45`, `I-55`, `I-65`, `I-70`, `I-80`, `I-90`, `US_20`, `US_25`, `US_30`, `US_40`, `US_50`, `US_60`,
`US_101`, `California_25`, `NY-25A`, `M-25`, `Texas_35`, `East_plate`, `MUTCD_W13-1`, `MUTCD_W13-1P`,
`MUTCD_W13-2`, `MUTCD_W13-3`, `MUTCD_E13-1P`, `MUTCD_R2-3`, `MUTCD_R2-4`, `MUTCD_R7-1`, `MUTCD_R7-8`,
`MUTCD_D3-1` (all `.svg` on Commons, saved as `.png`).

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
# Held-out sets (BDD valid) with the hand reviews in make_heldout.py (tracked):
.venv/Scripts/python scripts/signs/make_heldout.py               # eval/sign_labels_bdd.json (original, seen in development)
.venv/Scripts/python scripts/signs/label_bdd_signs.py --split valid --shards shard_000 shard_001 --min-h 12 --limit 9000 --model claude-opus-5-5 --workers 6 --max-usd 8 --out data/signs/bdd_labels/fresh_valid.jsonl --env-file .env
.venv/Scripts/python scripts/signs/make_heldout.py --set fresh   # eval/sign_labels_bdd_fresh.json (headline; checked blind)
# Contact sheets for checking labels by eye
.venv/Scripts/python scripts/signs/review_sheet.py data/signs/bdd_labels/valid.jsonl --kind speed_limit
```

## 3. Datasets

```bash
.venv/Scripts/python scripts/signs/prepare_data.py det    # 640x640 tiles at native scale -> data/signs/det (~2 min)
.venv/Scripts/python scripts/signs/prepare_data.py cls    # 96x96 value crops -> data/signs/cls (~8 min)
```

Hard negatives (both datasets pick them up when the files exist; rebuild after mining):

```bash
# value classifier: non-speed BDD signs it reads as a number; check the sheet, keep the true negatives in
# data/signs/hardneg_cls_curated.jsonl (one JSON line each, copied from hardneg_cls.jsonl)
.venv/Scripts/python scripts/signs/mine_hard_negatives.py --weights models/signs/speed_value_cls.pt
# detector: BDD frames with none of our labeled signs where it still fires -> data/signs/hardneg_det_raw.txt;
# most such hits are real signs BDD never boxed, so every hit goes on contact sheets and only the frames checked by
# eye are kept (the tile numbers in curate_det_negatives.py) -> data/signs/hardneg_det.txt
.venv/Scripts/python scripts/signs/mine_det_negatives.py --weights runs/signs/det/weights/best.pt
.venv/Scripts/python scripts/signs/dump_det_hits.py --weights runs/signs/det/weights/best.pt
.venv/Scripts/python scripts/signs/review_sheet.py data/signs/hardneg_det_dets.jsonl --kind speed_limit --count 132
.venv/Scripts/python scripts/signs/curate_det_negatives.py
# unboxed real signs in the training frames (masked by prepare_data.py det, never background)
.venv/Scripts/python scripts/signs/mine_unboxed.py --weights runs/signs/det/weights/best.pt
```

Only the training fraction of BDD train is ever trained on (synthetic and classifier backgrounds included);
each build records its training images in `data/signs/used_for_training_{det,cls}.txt` (keep copies of the files
of builds that delivered models: the delivered ones are `*_det_ft_build.txt` and `*_cls_cls5.txt`), and the tune
set (`evaluate.py tune`) leaves out every `used_for_training_*.txt` image and the first build's backgrounds.

## 4. Training (shared GPU: batch 8 keeps the detector at ~2.6 GB; don't export engines while training: RAM)

```bash
.venv/Scripts/python scripts/signs/train.py det --epochs 40 --batch 8 --workers 4    # ~4 h; checkpoint every epoch
.venv/Scripts/python scripts/signs/train.py det --resume                             # after a stop
.venv/Scripts/python scripts/signs/train.py cls --epochs 20 --dropout 0.2            # ~10 min
# fine-tune on the rebuilt dataset with hard negatives, from the first run's best weights (name the optimizer:
# ultralytics' "auto" ignores lr0 and picks its own learning rate)
.venv/Scripts/python scripts/signs/train.py det --name det_ft --base runs/signs/det/weights/best.pt --epochs 12 --optimizer AdamW --lr0 0.0002 --warmup-epochs 0 --close-mosaic 4 --batch 8 --workers 3 --no-copy
```

Best weights are copied to `models/signs/signs_yolo26s.pt` and `models/signs/speed_value_cls.pt` (pass
`--name <run> --no-copy` to keep a run aside; copy its `runs/signs/<run>/weights/best.pt` by hand if it wins).

## 5. Evaluation

```bash
.venv/Scripts/python scripts/signs/eval_values.py --views 1 3 5 7      # classifier alone: threshold per view count
.venv/Scripts/python scripts/signs/evaluate.py tune                    # end to end on the tune set (thresholds)
# after choosing (never before): held-out sets
.venv/Scripts/python scripts/signs/evaluate.py all --write-metrics     # fresh, bdd, bdd_noaudit, public, tune
.venv/Scripts/python scripts/signs/eval_values.py --views 1 3 5 7 --heldout both --write-metrics
.venv/Scripts/python scripts/signs/evaluate.py tune --cached --set signs.value_conf=0.95   # re-score without the GPU
.venv/Scripts/python scripts/signs/bench_signs.py                      # per-frame time on a dev clip
.venv/Scripts/python scripts/signs/bench_frames.py --write-metrics     # per-frame time on frames with speed limits
```

`evaluate.py` runs the detector once per frame (`raw_detections`) and applies the app's rules with the same
`perception.signs.select()` the app uses. Every reported speed value is judged: right, wrong, or unverifiable
(on a sign of unknown kind or an unreadable limit); `value_conf` = the larger of the lowest threshold with no
wrong value on the classifier's BDD validation crops and on the tune set end to end.

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
reported only when all `signs.value_views` crops of the sign (shifted/scaled) read the same number, each at least
`signs.value_conf` sure, the sign is at least `signs.value_min_px` tall (at 1280 px frame width) and doesn't touch
the frame border. Announcing a new limit should still wait for two agreeing frames (V2_BUILD_PLAN).
