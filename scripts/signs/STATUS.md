# V2-M6 sign fine-tune: status

Branch `v2-signs` (worktree `C:\ClaudeProjects\DrivePilot_signs`). Commands: [README.md](README.md).
Last update: 2026-10-10 (in progress).

## Delivered so far
- `perception/signs.py`: `SignDetector(cfg.signs, root).detect(frame) -> list[SignDetection]` (contract from
  SIGN_FINETUNE_PROMPT.md, plus `value_conf` per detection and `last_timings_ms`). Two stages: YOLO26s fine-tuned
  on 13 MUTCD kinds at 736x1280 (whole frame, signs at full size), then a speed-value classifier (YOLO26n-cls,
  96x96 crop, 5-85 mph or "none") on every speed-limit / school-speed-limit box at least 16 px tall. A value is
  reported only at classifier probability >= 0.98 (chosen on validation data, not on the held-out set).
  Verifies CUDA + FP16 for both networks at startup and refuses the CPU.
- `perception/sign_classes.py`: the taxonomy (class order fixed) and label mapping.
- `config.yaml` `signs:` section, `core/config.py` `SignsConfig` (appended).
- `eval/sign_labels_bdd.json`: the hand-checked BDD100K held-out set.
- `tests/test_signs.py`: label conversion, class mapping, value parsing, tiling/masking, the detector contract
  with a fake model, and a `gpu` test with the real engines.

## Classes
speed_limit, school_speed_limit, stop, yield, do_not_enter, wrong_way, no_turn_on_red, one_way, no_left_turn,
no_right_turn, no_u_turn, pedestrian_crossing, school_zone (all 13 kept; wrong_way, no_turn_on_red and
school_speed_limit are thin in real data and lean on synthetic examples: see Known issues).

## Data (all non-commercial use is fine for the hackathon; recorded per source)
| Source | What | License |
| --- | --- | --- |
| BDD100K detection frames, HF mirror `dronefreak/BDD100K` | 1280x720 US dashcam frames with human "traffic sign" boxes (no sign type) | BDD100K license (research, non-commercial) |
| Claude Opus 5.5 pseudo-labels of those boxes | kind + mph per box; 37k train crops, 9k held-out crops; "unclear" boxes masked in training | our labels |
| `ctacke/speed-limit-signs` (HF) | 1,045 US dashcam speed-limit boxes, 25-80 mph, 1920x1080 | MIT |
| MUTCD sign renders, Wikimedia Commons | templates for synthetic signs (any speed value) | public domain (US government works) |
| Overpass font | numbers on synthetic speed-limit signs | SIL OFL |
| Not used: LISA (official download now 404s; Roboflow copy needs a key that wasn't provided in time), the HF Mapillary mirror `ThankGod/mapillary_traffic_sign_dataset_cleaned` (boxes visibly misaligned with images; category ids undocumented) | | |

## Design choices (and why)
- **One "speed_limit" class + a value classifier, not per-value detector classes.** Real data per value is
  far too thin for per-value boxes (BDD's ~250 training speed limits are ~85% "25"; ctacke covers 25-80 with
  few distinct signs), while a crop classifier can learn every value from synthetic MUTCD renders plus the real
  crops, has an explicit reject class ("none": other signs, route shields, unreadable numbers) and a probability
  that can be thresholded so a doubtful value is withheld. This was a reasoned choice, not a measured
  head-to-head (no time to train a second detector); the classifier alone reads 75/80 clearly readable held-out
  limits right with no wrong value (see Numbers).
- **Whole frame at 1280 px wide.** Distant signs are 15-40 px tall in a 720p frame; at the main detector's
  384x640 they would shrink to 8-20 px. Training uses 640x640 tiles cut at native scale (half-tile overlap,
  signs cut by a tile edge masked), inference runs on the full 736x1280 frame (fully convolutional). Measured:
  ~3.5 ms TensorRT inference, ~7 ms per frame with pre/post-processing.
- **Unknown signs are masked, not background.** Boxes whose kind the labeler couldn't tell, and sign boxes too
  small to label, are painted over in training tiles so a real stop sign is never taught as "nothing".
- **Look-alikes as negatives.** Route shields, advisory/exit speed plates and NIGHT/MINIMUM speed plates carry
  numbers; they are pasted unlabeled into training frames and are reject crops for the classifier, and the
  real ones the models still confused were mined and added (`mine_hard_negatives.py`, `mine_det_negatives.py`).

## Training setup
- Detector: YOLO26s from the COCO weights, 640x640 tiles (14.5k training tiles from 2,841 BDD frames with our
  signs, 852 BDD frames without, 871 ctacke frames, 3,500 synthetic composites), batch 8 (2.6 GB on the shared
  RTX 4050 Laptop), 40 epochs, cosine LR, mosaic (off for the last 5 epochs), scale +-30%, no horizontal flips
  (they'd turn "no left turn" into "no right turn"), ultralytics 8.4.175 / torch 2.14.1+cu126.
- Value classifier: YOLO26n-cls, 96x96 crops, 18 classes (5-85 mph + "none"), ~33k training crops (1,200
  synthetic per value; real BDD, ctacke and hard-negative crops), dropout 0.2, 20 epochs, no flips/erasing.

## For the integrating session (main)
- Merge `v2-signs`; it only appends to `config.yaml` (`signs:`), `core/config.py` (`SignsConfig` + the `signs`
  field at the end of `AppConfig`) and `.gitignore`, and adds `scripts/__init__.py` + `scripts/signs/__init__.py`
  (so mypy maps `scripts.signs.*` consistently). Everything else is new files.
- Files in `models/signs/` (delivered there, gitignored): `signs_yolo26s.pt`, `signs_yolo26s_736x1280_fp16.engine`,
  `speed_value_cls.pt`, `speed_value_cls_96x96_fp16.engine` (+ tracked `metrics.json`). The engines only load
  with the TensorRT in `.venv` (tensorrt-cu12 11.4.0.106); on another machine rebuild them with
  `scripts/signs/export_engines.py`.
- Run it on its own thread at a few fps on the latest frame (like the lanes thread). Construct `SignDetector` on
  that thread (CUDA state is per thread; the constructor warms up and verifies CUDA/FP16).
- `SignDetection.value` is None whenever the reading isn't certain (a wrong limit is worse than none). For
  "Speed limit is now 35." still require two agreeing frames (V2_BUILD_PLAN): one frame can be a look-alike.
  `value_conf` is there if you want to log it. `school_speed_limit` also carries a value (15-25 mph).
- `kind` is one of `perception.sign_classes.KINDS`; `SignDetection.to_record()` is ready for run logs.

## Numbers
To be filled with measured results (detector training in progress).

## Known issues / compromises
- **Training labels are Claude pseudo-labels, not hand labels.** BDD100K boxes every sign by hand but not its
  type; Claude Opus 5.5 assigned the type (and mph) per crop. Haiku 4.5 was tried first and rejected (many real
  one-way/school/speed signs called "not a sign"). Opus is cautious ("unclear" boxes are masked, never trained
  as background) but not perfect: mining showed it called some TOWN/CITY/AREA SPEED LIMIT variants "other".
  Labeling cost $39 (+ $1 for the ctacke re-read) on the project's Anthropic key.
- **The held-out set is hand-checked for speed limits only.** Every speed-limit box was checked by eye on the
  full-resolution crop and a size-sorted sample of the labeler's "other"/"unclear" boxes was checked for missed
  speed limits; the other kinds keep Opus's labels (spot-checked). A label audit triggered by the model's own
  unmatched speed values added 2 school speed limits that BDD's annotators never boxed or Opus called "other"
  and marked 3 doubtful boxes as ignore (`make_heldout.py` AUDIT); `evaluate.py --no-audit` gives the numbers
  without it.
- **BDD100K is mostly New York and the Bay Area**: its speed limits are mostly 25 mph (NYC). Other values come
  from ctacke (25-80 mph, few distinct signs: its frames are video sequences of the same signs) and synthetic
  MUTCD renders. The value classifier is checked on real 20-50 mph signs only.
- **ctacke's public val split shares sign sequences with its train split** (adjacent frames of the same physical
  signs), so the public-split numbers are optimistic; the BDD held-out set (separate videos) is the honest one.
- **Thin real data:** wrong_way (1 held-out instance), no_turn_on_red (7), school_speed_limit (2-4) rely on
  synthetic examples; their held-out numbers are anecdotal.
- **LISA is not used**: its official download returns 404 and the Roboflow copy needs an API key that wasn't
  provided in time; `prepare_data.py` picks it up automatically if a Roboflow YOLO export is placed in
  `data/signs/raw/lisa_yolo/`.
- **Not validated on our own camera** (the phone's 114° ultrawide): signs appear smaller and later than in BDD's
  narrower view; re-check on our drives.
- A crashed run: the detector run was resumed once from its epoch-6 checkpoint after a RAM shortage (a TensorRT
  export ran alongside it); resumed runs continue the same schedule.

## Best Fine-tune write-up
To be filled.
