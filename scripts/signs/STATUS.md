# V2-M6 sign fine-tune: status

Branch `v2-signs` (worktree `C:\ClaudeProjects\DrivePilot_signs`). Commands: [README.md](README.md).
Last update: 2026-10-10 PENDING.

## What is delivered
- `perception/signs.py`: `SignDetector(cfg.signs, root).detect(frame) -> list[SignDetection]` (the contract in
  SIGN_FINETUNE_PROMPT.md, plus `value_conf` per detection and `last_timings_ms`). Two stages: YOLO26s fine-tuned
  on 13 MUTCD kinds at 736x1280 (the whole frame, signs at full size), then a speed-value classifier (YOLO26n-cls,
  96x96 crop, 5-85 mph or "none") on every speed-limit / school-speed-limit box at least 16 px tall. A value is
  reported only at classifier probability >= `value_conf` (PENDING, chosen on validation and tune data, never on
  the held-out sets) and not for a sign touching the frame border. `raw_detections()` + `select()` split the
  network from the app's rules so the evaluation applies exactly the app's rules. At startup it verifies both
  networks run on CUDA in FP16 (for TensorRT: FP16 bindings or the engine's `images_cast_to_fp16` cast layer, plus
  the export metadata), refuses the CPU, refuses a classifier whose classes aren't 5-85 mph + "none", and does not
  use an engine built from other weights (sidecar JSON with the weights' sha256; it falls back to the .pt on the GPU
  and warns).
- `perception/sign_classes.py`: the taxonomy (class order fixed) and label mapping.
- `config.yaml` `signs:` section and `core/config.py` `SignsConfig` (appended only).
- `eval/sign_labels_bdd_fresh.json` (headline held-out set, checked blind) and `eval/sign_labels_bdd.json`
  (original held-out set, seen during development).
- `models/signs/metrics.json` (tracked): every number below.
- `tests/test_signs.py`: PENDING tests: taxonomy and label mapping, value parsing, tiling/masking, leakage-free
  dataset planning, synthetic pasting, the app's rules (`select`), the detector contract with fakes, GPU/FP16
  verification and stale engines with fake backends, the evaluation's value judging, and `gpu` tests with the
  real engines (a blank frame, and DO NOT ENTER on a dev clip).

## Classes
speed_limit, school_speed_limit, stop, yield, do_not_enter, wrong_way, no_turn_on_red, one_way, no_left_turn,
no_right_turn, no_u_turn, pedestrian_crossing, school_zone. All 13 kept; wrong_way, no_turn_on_red and
school_speed_limit are thin in real data and lean on synthetic examples (see Known issues).

## Data (non-commercial use is fine for the hackathon; recorded per source)
| Source | What | License |
| --- | --- | --- |
| BDD100K detection frames, HF mirror `dronefreak/BDD100K` | 1280x720 US dashcam frames with human "traffic sign" boxes (no sign type) | BDD100K license (research, non-commercial) |
| Claude Opus 5.5 pseudo-labels of those boxes | kind + mph per box: 37k train crops, 9k + 9k held-out crops; "unclear" boxes masked in training | our labels |
| `ctacke/speed-limit-signs` (HF) | 1,045 US dashcam speed-limit boxes, 25-80 mph, 1920x1080 | MIT |
| MUTCD sign renders, Wikimedia Commons | templates for synthetic signs (any speed value) and look-alike distractors (route shields, advisory plates) | public domain (US government works) |
| Overpass font | numbers on synthetic speed-limit signs | SIL OFL |
| Not used: LISA (official download 404s; the Roboflow copy needs a key that wasn't provided in time), the HF Mapillary mirror `ThankGod/mapillary_traffic_sign_dataset_cleaned` (boxes visibly misaligned; category ids undocumented) | | |

Claude labeling spend: about $47.9 in total (Opus 5.5: train shards $31.0, original held-out $7.5, fresh
held-out $7.5, ctacke re-read $1.0, pilots).

## Design choices (and why)
- **One "speed_limit" class + a value classifier, not per-value detector classes.** Real data per value is far
  too thin for per-value boxes (BDD's ~250 training speed limits are ~85% "25"; ctacke covers 25-80 with few
  distinct signs), while a crop classifier learns every value from synthetic MUTCD renders plus the real crops,
  has an explicit reject class ("none": other signs, route shields, unreadable numbers) and a probability that is
  thresholded so a doubtful value is withheld. A reasoned choice, not a measured head-to-head (no time to train a
  second detector).
- **Whole frame at 1280 px wide.** Distant signs are 15-40 px tall in a 720p frame; at the main detector's 384x640
  they would shrink to 8-20 px. Training uses 640x640 tiles cut at native scale (half-tile overlap, signs cut by
  a tile edge masked); inference runs on the full 736x1280 frame (fully convolutional).
- **Unknown signs are masked, never background.** Boxes whose kind the labeler couldn't tell, sign boxes too small
  to label, and real signs BDD's annotators never boxed (found by the first detector, `mine_unboxed.py`: 412 boxes
  in 383 training frames, about two thirds real signs on a viewed sample) are painted over in training tiles, so a
  real stop sign is never taught as "nothing". Masks no longer paint over a labeled neighbour.
- **Look-alikes as negatives.** Route shields, advisory/exit speed plates and NIGHT/MINIMUM plates carry numbers:
  they are pasted unlabeled into training frames and are reject crops for the classifier. Real BDD frames where
  the first detector fired on none of our labeled signs were mined (872 frames), but checking every detection by
  eye showed most were real signs BDD never boxed (nearly all one-way, school-zone, stop and yield hits, and many
  speed limits, e.g. a clear SPEED LIMIT 30); only the 299 frames whose every detection is clearly not one of our
  kinds (yellow warning diamonds, EXIT 40 / RAMP 25 advisory plates, WEST/EAST 25 route shields, text regulatory
  signs, non-US round limits, logos) became hard negatives (`curate_det_negatives.py`).

## Training setup
- Detector, run 1 (`runs/signs/det`): YOLO26s from the COCO weights, 640x640 tiles (14.5k training tiles from
  2,841 BDD frames with our signs, 852 without, 871 ctacke frames, 3,500 synthetic composites), batch 8 (2.6 GB on
  the shared RTX 4050 Laptop), cosine LR, AdamW (ultralytics auto, lr 0.000588), mosaic (off for the last 5
  epochs), scale +-30%, no horizontal flips (they'd turn "no left turn" into "no right turn"); early-stopped at
  epoch 39 of 40 (patience 15), best epoch 24: mAP50 0.854 / mAP50-95 0.575 on the BDD-train validation tiles
  (pseudo-labels). Ultralytics 8.4.175, torch 2.14.1+cu126. It crashed twice from system-RAM shortages on the
  shared machine and was resumed from its per-epoch checkpoints (same schedule).
- Detector, fine-tune (`runs/signs/det_ft`): from run 1's best weights on the rebuilt dataset (leakage fixes,
  masks for unboxed signs, 299 curated hard negatives; 15.2k training tiles), 12 epochs, AdamW lr0 0.0002 (named
  explicitly: ultralytics' "auto" ignores lr0 and would have used run 1's 0.000588), no warmup, mosaic off for the
  last 4 epochs. PENDING result; the winner of run 1 vs fine-tune was picked on the tune set.
- Value classifier PENDING.

## Numbers
PENDING

## For the integrating session (main)
- Merge `v2-signs`; it only appends to `config.yaml` (`signs:`), `core/config.py` (`SignsConfig` + the `signs`
  field at the end of `AppConfig`) and `.gitignore`, and adds `scripts/__init__.py` + `scripts/signs/__init__.py`
  (so mypy maps `scripts.signs.*` consistently). Everything else is new files.
- Files in `models/signs/` (delivered there, gitignored): PENDING. The engines only load with the TensorRT in
  `.venv` (tensorrt-cu12 11.4.0.106); on another machine rebuild them with `scripts/signs/export_engines.py`.
- Run it on its own thread at a few fps on the latest frame (like the lanes thread). Construct `SignDetector` on
  that thread (CUDA state is per thread; the constructor warms up and verifies CUDA/FP16).
- `SignDetection.value` is None whenever the reading isn't certain (a wrong limit is worse than none). For
  "Speed limit is now 35." still require two agreeing frames (V2_BUILD_PLAN): one frame can be a look-alike.
  `value_conf` is there if you want to log it. `school_speed_limit` also carries a value (15-25 mph).
- `kind` is one of `perception.sign_classes.KINDS`; `SignDetection.to_record()` is ready for run logs.

## Known issues / compromises
PENDING

## Best Fine-tune write-up
PENDING
