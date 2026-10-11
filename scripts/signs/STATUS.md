# V2-M6 sign fine-tune: status

Branch `v2-signs` (worktree `C:\ClaudeProjects\DrivePilot_signs`). Commands: [README.md](README.md).
Last update: 2026-10-10 ~22:00 PDT. Every number below is in `models/signs/metrics.json` (tracked).

## Result in one paragraph
A two-stage US sign reader runs on the RTX 4050 at **6.6-7.1 ms per frame (median)** as TensorRT FP16 engines:
YOLO26s fine-tuned on 13 MUTCD kinds, then a speed-value classifier whose reading must agree over 5 shifted crops.
On a fresh BDD100K held-out slice labeled after development and hand-checked blind (2,675 frames + 500 sign-free
frames), it reads **69 of 77 clearly readable speed limits right (89.6%)** and gives **no value that contradicts a
labeled sign**. It also reported 6 values where the labels have no sign: checked by eye, 5 are correct readings of
real speed-limit signs the labels miss, and 1 is a white "20" panel of doubtful kind (see Numbers). Public split
(ctacke val): speed-limit AP50 0.990, school-speed-limit AP50 1.000.

## What is delivered
- `perception/signs.py`: `SignDetector(cfg.signs, root).detect(frame) -> list[SignDetection]` (the contract in
  SIGN_FINETUNE_PROMPT.md, plus `value_conf` per detection and `last_timings_ms`). Two stages: YOLO26s fine-tuned
  on 13 MUTCD kinds at 736x1280 (the whole frame, signs at full size), then a speed-value classifier (YOLO26n-cls,
  96x96 crop, 5-85 mph or "none") on every speed-limit / school-speed-limit box at least 16 px tall. A value is
  reported only if **all 5 views** of the box (the box, shifted 8% left/right, scaled 0.9 and 1.12) read the same
  number, each with probability >= `value_conf` = 0.9, and the sign doesn't touch the frame border.
  `raw_detections()` + `select()` split the networks from the app's rules, and the evaluation applies the very same
  `select()`. At startup it verifies both networks run on CUDA in FP16 (for TensorRT: FP16 bindings or the engine's
  `images_cast_to_fp16` cast layer, plus the export metadata), refuses the CPU, refuses a classifier whose classes
  aren't 5-85 mph + "none", and won't use an engine built from other weights (sidecar JSON with the weights'
  sha256; it then falls back to the .pt on the GPU and warns).
- `perception/sign_classes.py`: the taxonomy (class order fixed) and label mapping.
- `config.yaml` `signs:` section and `core/config.py` `SignsConfig` (appended only).
- `eval/sign_labels_bdd_fresh.json` (headline held-out set, checked blind) and `eval/sign_labels_bdd.json`
  (original held-out set, seen during development).
- `models/signs/metrics.json` (tracked): every number below, plus the first-pass numbers (single view, value_conf
  0.995) under `first_pass_single_view_value_conf_0.995`.
- `tests/test_signs.py`: 109 tests without hardware (taxonomy and label mapping, value parsing, tiling/masking,
  leakage-free dataset planning, synthetic pasting, label fixes, the app's rules, value views, the detector
  contract with fakes, GPU/FP16 verification and stale engines with fake backends, the evaluation's value judging,
  threshold selection) and 2 `gpu` tests with the real engines (a blank frame; DO NOT ENTER on a dev clip): pass.

## Classes
speed_limit, school_speed_limit, stop, yield, do_not_enter, wrong_way, no_turn_on_red, one_way, no_left_turn,
no_right_turn, no_u_turn, pedestrian_crossing, school_zone. All 13 kept; wrong_way, no_turn_on_red and
no_right_turn have only a handful of real examples and are weak (see Numbers and Known issues).

## Data (non-commercial use is fine for the hackathon; recorded per source)
| Source | What | License |
| --- | --- | --- |
| BDD100K detection frames, HF mirror `dronefreak/BDD100K` | 1280x720 US dashcam frames with human "traffic sign" boxes (no sign type) | BDD100K license (research, non-commercial) |
| Claude Opus 5.5 pseudo-labels of those boxes | kind + mph per box: 37k train crops, 9k + 9k held-out crops; "unclear" boxes masked in training | our labels |
| `ctacke/speed-limit-signs` (HF) | 1,045 US dashcam speed-limit boxes, 25-80 mph, 1920x1080 | MIT |
| MUTCD sign renders, Wikimedia Commons | templates for synthetic signs (any speed value) and look-alike distractors (route shields, advisory plates) | public domain (US government works) |
| Overpass font | numbers on synthetic speed-limit signs | SIL OFL |
| Not used: LISA (official download 404s; the Roboflow copy needs a key that wasn't provided in time), the HF Mapillary mirror `ThankGod/mapillary_traffic_sign_dataset_cleaned` (boxes visibly misaligned; category ids undocumented) | | |

Claude labeling spend: about $47.9 in total (Opus 5.5: train shards $31.0, original held-out $7.5, fresh held-out
$7.5, ctacke re-read $1.0, pilots).

## Design choices (and why)
- **One "speed_limit" class + a value classifier, not per-value detector classes.** Real data per value is far too
  thin for per-value boxes (BDD's ~250 training speed limits are ~85% "25"; ctacke covers 25-80 with few distinct
  signs), while a crop classifier learns every value from synthetic MUTCD renders plus the real crops, has an
  explicit reject class ("none": other signs, route shields, unreadable numbers) and a probability that is
  thresholded so a doubtful value is withheld. A reasoned choice, not a measured head-to-head.
- **A value must agree over 5 views.** A detector box is a little off in a different way every frame. A misread
  number changes with the crop while a clear one doesn't: one confident misread (a SPEED LIMIT 30 read as 20 at p
  0.999) dropped to 0.74 under an 8% shift, while clear readings stay at 1.000 in every view. On the BDD validation
  boxes, 5 views let 0.9 be safe (1 view needed 0.97 for the same right count).
- **Whole frame at 1280 px wide.** Distant signs are 15-40 px tall in a 720p frame; at the main detector's 384x640
  they would shrink to 8-20 px. Training uses 640x640 tiles cut at native scale (half-tile overlap, signs cut by a
  tile edge masked); inference runs on the full 736x1280 frame (fully convolutional).
- **Unknown signs are masked, never background.** Boxes whose kind the labeler couldn't tell, sign boxes too small
  to label, and real signs BDD's annotators never boxed (found by the first detector, `mine_unboxed.py`) are
  painted over in training tiles, so a real stop sign is never taught as "nothing". Masks don't paint over a
  labeled neighbour.
- **Look-alikes as negatives.** Route shields, advisory/exit speed plates and NIGHT/MINIMUM plates carry numbers:
  they are pasted unlabeled into training frames and are reject crops for the classifier. Real BDD frames where the
  first detector fired on none of our labeled signs were mined (872 frames), but checking every detection by eye
  showed most were real signs BDD never boxed (nearly all one-way, school-zone, stop and yield hits, and many speed
  limits, e.g. a clear SPEED LIMIT 30); only the 299 frames whose every detection is clearly not one of our kinds
  (yellow warning diamonds, EXIT 40 / RAMP 25 advisory plates, WEST/EAST 25 route shields, text regulatory signs,
  non-US round limits, logos) became hard negatives (`curate_det_negatives.py`).

## Training setup
- Detector, run 1 (`runs/signs/det`): YOLO26s from the COCO weights, 640x640 tiles (14.5k training tiles from
  2,841 BDD frames with our signs, 852 without, 871 ctacke frames, 3,500 synthetic composites), batch 8 (2.6 GB on
  the shared RTX 4050 Laptop), cosine LR, AdamW (ultralytics auto, lr 0.000588), mosaic (off for the last 5
  epochs), scale +-30%, no horizontal flips (they'd turn "no left turn" into "no right turn"). Early-stopped at
  epoch 39 of 40 (patience 15); best epoch 24: mAP50 0.854 / mAP50-95 0.575 on the BDD-train validation tiles
  (pseudo-labels). Ultralytics 8.4.175, torch 2.14.1+cu126; about 4-7 min per epoch on the shared GPU.
- Detector, fine-tune (`runs/signs/det_ft`, **delivered**): from run 1's best weights on the rebuilt dataset
  (leakage fixes, masks for unboxed signs, 299 curated hard negatives; 15.2k training tiles), 12 epochs, AdamW lr0
  0.0002 (named explicitly: ultralytics' "auto" ignores lr0 and would have used run 1's 0.000588), no warmup,
  mosaic off for the last 4 epochs, ~1.2 h. Picked over run 1 on the tune set (516 BDD-train frames no build ever
  trained on): speed-limit AP50 0.841 vs 0.800, precision 0.862 vs 0.788, mean AP50 over 11 labeled kinds 0.759 vs
  0.760.
- Value classifier (`runs/signs/cls5`, **delivered**): YOLO26n-cls, 96x96 crops, 18 classes (5-85 mph + "none"),
  33.8k training crops (about 1,200 synthetic per value plus real BDD and ctacke speed limits; "none" = 10.3k:
  6,000 real BDD "other" signs, real crops of our other kinds and of unreadable limits, 24 curated look-alikes x4,
  600 synthetic other-kind/look-alike renders), dropout 0.2, 20 epochs, batch 256, no
  flips/erasing, trained only on the clean build (no validation-fraction frame). Top-1 0.978 on its validation set.
- Thresholds (`value_views` 5, `value_conf` 0.9), chosen before the final held-out run with the FP16 engines: the
  lowest threshold with no wrong value on 1,463 real BDD validation boxes (280 labeled boxes of our kinds, 1,180
  "other" signs, 3 curated look-alikes; highest consensus misread 0.851) and with no wrong value end to end on the
  tune set (safe from 0.6), taking the larger. Two validation crops the classifier read at p 1.0 were checked by eye:
  pseudo-label errors (a real SPEED LIMIT 25 called "other", a doubtful "20" panel), fixed in `LABEL_FIXES`.

## Numbers (TensorRT FP16 engines, value_views 5, value_conf 0.9)

### Speed limits: the priority
| Set | Clearly readable limits read right | Values contradicting a labeled sign | Values on no labeled sign (checked by eye) |
| --- | --- | --- | --- |
| **Fresh BDD held-out** (2,675 + 500 sign-free frames) | **69/77 = 89.6%** (+ 1/1 school) ; 4 missed, 4 no value | **0** | 6: 5 correct readings of real limits the labels miss, 1 doubtful "20" panel |
| Original BDD held-out, with audit | 69/78 + 3/4 school = 88.5% | **0** | 5: 3 correct readings of real limits, 2 doubtful (a red-header "45", a SCHOOL-topped "25") |
| Original BDD held-out, `--no-audit` | 69/78 + 2/2 school | **0** | 7: the 5 above + 2 real school / dark limits the audit adds |
| Public (ctacke val, 100 frames) | 80/86 + 5/6 school = 93% | **0** | 2: clear SPEED LIMIT 50 signs ctacke doesn't box (correct) |
| Tune (516 BDD-train frames, pseudo-labels) | 19/29 | **0** | 0 (1 unverifiable: the doubtful "20" panel) |

"Values on no labeled sign" are counted as WRONG by `evaluate.py` (strict); every one was looked at on the
full-resolution frame. The doubtful ones are signs whose kind I can't confirm, read as printed: a white "20" panel
above a disabled-parking sign outside a grocery store (fresh; maybe a parking-lot limit or a time limit), a
rectangular sign with a red header over "45" (not a standard R2-1), and a sign with a SCHOOL header over unreadable
text read as 25. **They may be wrong values; the 0-wrong bar is met only on the labeled signs.** Unverifiable
values (on a sign of unknown kind or an unreadable limit; fresh 8, original 7): every one viewed, all plausible
readings of dark, small or red-header limits (25/40/35/15). Classifier alone at 0.9 (every labeled limit with a
value, cropped at its labeled box): fresh 73/78 readable right, 0 wrong on readable ones; 1 misread on a not-readable
box (a PHOTO ENFORCED plate under a 30 sign, only the bottom of the digits visible, read 25 at 0.994; end to end
the detector doesn't report that box, so no value was given). First pass (single view, value_conf 0.995): fresh
68/77 right, and one real misread on the original set (a clearly readable SPEED LIMIT 30 read as 20 at p 0.999),
which the 5-view rule removes (it was added after that evaluation; its settings were chosen on validation/tune).

### Detection, per kind (precision / recall at conf 0.4, AP50)
| Kind | Fresh: labeled, P, R, AP50 | Original (audit): labeled, P, R, AP50 |
| --- | --- | --- |
| speed_limit | 94, 0.72, 0.88, 0.875 | 90, 0.82, 0.90, 0.907 |
| school_speed_limit | 1, 0.14, 1.00, 1.000 | 4, 0.80, 1.00, 1.000 |
| stop | 56, 0.88, 0.80, 0.789 | 47, 0.73, 0.77, 0.748 |
| yield | 22, 0.71, 0.91, 0.877 | 25, 0.62, 0.80, 0.728 |
| do_not_enter | 40, 0.81, 0.82, 0.811 | 49, 0.62, 0.78, 0.741 |
| wrong_way | 2, 0.00, 0.00, 0.100 | 1, 0.00, 0.00, 0.000 |
| no_turn_on_red | 4, 0.29, 0.50, 0.250 | 7, 0.38, 0.43, 0.528 |
| one_way | 501, 0.79, 0.85, 0.855 | 442, 0.79, 0.85, 0.826 |
| no_left_turn | 57, 0.77, 0.77, 0.750 | 62, 0.80, 0.84, 0.859 |
| no_right_turn | 10, 0.46, 0.50, 0.426 | 8, 0.47, 0.88, 0.573 |
| no_u_turn | 20, 0.60, 0.75, 0.716 | 27, 0.76, 0.81, 0.815 |
| pedestrian_crossing | 35, 0.54, 0.60, 0.562 | 41, 0.72, 0.76, 0.775 |
| school_zone | 172, 0.81, 0.89, 0.837 | 158, 0.80, 0.88, 0.833 |

Public split (ctacke val): speed_limit AP50 **0.990** (P 0.94, R 1.00), school_speed_limit AP50 **1.000** (done-when
bar: >= 0.8). Precision on BDD is understated: the labels come from BDD's boxes, which miss many real signs (the
review of the detector's "false alarms" found mostly unboxed real signs), so part of every "FP" column is right.
Wrong way, no turn on red and no right turn are weak (few real examples); treat them as unreliable.

### Speed (RTX 4050 Laptop, TensorRT 11.4.0.106 FP16, school-zone dev clip, 904 frames, every frame)
| | total median | p95 | max | TensorRT inference median |
| --- | --- | --- | --- | --- |
| Alone (run 1 / run 2) | 7.14 / 6.80 ms | 15.4 / 7.44 ms | 24.1 / 14.1 ms | 3.1 / 3.4 ms |
| With the app's perception (YOLO26s + ByteTrack at 30 fps) | 6.56 ms | 8.31 ms | 15.1 ms | 3.1 ms |
Classifying a speed sign (5 views) adds ~3-4 ms on the frames that have one (classify_total max 3.2-4.2 ms; median
0.1 ms because most frames have none). The first standalone run's p95 came from CPU-side pre/post-processing
spikes; the repeat was clean. Not throttled (charger in, CPU at 115% of base). GPU memory for both engines: ~1.5 GB
(whole device, nothing else running). Both networks verified cuda:0 / FP16 at startup (`_verify_gpu`, gpu tests).

### Test sheet sign rows (dev clips)
The vision model got 2/6 sign rows (PROGRESS.md). The detector: "what do those red signs ahead say?" → do_not_enter
(0.85) ✓; "what did that yellow sign say?" → school_zone (0.91) ✓ (pedestrian_crossing in 1 of 9 frames);
"what did that white sign on the right say?" (LAST LOCAL TRUCK ROUTE) → nothing ✓ (an interim model called it a
speed limit); "what does the overhead sign say?" (too far) → nothing ✓; "can I turn right on red here?" → one_way
(0.82): supports the expected "the cross street is one way going left" (no NO TURN ON RED sign is visible);
"what street is that on the right?" → not a sign-kind question.

## For the integrating session (main)
- Merge `v2-signs`; it only appends to `config.yaml` (`signs:`), `core/config.py` (`SignsConfig` + the `signs`
  field at the end of `AppConfig`) and `.gitignore`, and adds `scripts/__init__.py` + `scripts/signs/__init__.py`
  (so mypy maps `scripts.signs.*` consistently). Everything else is new files.
- Files in `models/signs/` (delivered there, gitignored): see Delivery below. The engines only load with the
  TensorRT in `.venv` (tensorrt-cu12 11.4.0.106); on another machine rebuild them with
  `scripts/signs/export_engines.py`. If the .pt changes, the engine's sidecar no longer matches and the detector
  falls back to the .pt (slower, with a warning) until the engines are rebuilt.
- Run it on its own thread at a few fps on the latest frame (like the lanes thread). Construct `SignDetector` on
  that thread (CUDA state is per thread; the constructor warms up and verifies CUDA/FP16).
- `SignDetection.value` is None whenever the reading isn't certain (a wrong limit is worse than none). For
  "Speed limit is now 35." still require two agreeing frames (V2_BUILD_PLAN): one frame can be a look-alike (see
  the doubtful signs above). `school_speed_limit` also carries a value (15-25 mph).
- `kind` is one of `perception.sign_classes.KINDS`; `SignDetection.to_record()` is ready for run logs.
- The weak kinds (wrong_way, no_turn_on_red, no_right_turn) are worth a higher `class_conf` in the app or not
  announcing them on one detection.

## Delivery
Copied on 2026-10-10 ~21:35 PDT with `scripts/signs/export_engines.py --deliver` (it refuses a missing, stale or
sidecar-less engine) to `C:\ClaudeProjects\DrivePilot_BananaHacks2026\models\signs\` (gitignored there):

| File | Size | What |
| --- | --- | --- |
| `signs_yolo26s.pt` | 20.33 MB | detector weights (fine-tune `det_ft`, sha256 8db6751c2bd3...) |
| `signs_yolo26s_736x1280_fp16.engine` | 215.10 MB | detector TensorRT FP16 engine |
| `signs_yolo26s_736x1280_fp16.engine.json` | 253 B | sidecar: weights sha256, imgsz, TensorRT version |
| `speed_value_cls.pt` | 3.23 MB | value classifier weights (`cls5`, sha256 0081abaf84ae...) |
| `speed_value_cls_96x96_fp16.engine` | 4.73 MB | classifier TensorRT FP16 engine |
| `speed_value_cls_96x96_fp16.engine.json` | 252 B | sidecar |

TensorRT 11.4.0.106 (built in `.venv-trtbuild`, the same version as the app's `.venv`). Smoke-tested from the
main checkout: `SignDetector` loads both engines (no fallback), passes the CUDA/FP16 checks and finds the two DO
NOT ENTER signs on the school-zone dev clip at 25 s. `metrics.json` comes with the branch merge (tracked); the
settings it was measured with (`value_views: 5`, `value_conf: 0.9`) are in the branch's `config.yaml`.

## Known issues / compromises
- **Training labels are Claude pseudo-labels, not hand labels.** BDD100K boxes many signs by hand but not their
  type; Claude Opus 5.5 assigned the type (and mph) per crop. Haiku 4.5 was tried first and rejected (many real
  one-way/school/speed signs called "not a sign"). Opus is cautious ("unclear" boxes are masked, never trained as
  background) but not perfect: it called some CITY/TOWN/AREA SPEED LIMIT variants and dim speed limits "other".
- **BDD100K does not box every sign**, and the held-out labels inherit that: 5 of the fresh set's 6 "values on no
  labeled sign" are real speed limits the labels miss (23-30 px: smaller than the 40 px down to which the labeler's
  "other" boxes were checked). Unboxed signs the first detector found are masked in training (412 hits in 383
  frames, about two thirds real on a viewed sample); unboxed signs it missed still train as background. The
  fine-tune's dataset was built before a review fix and also masked the 47 unboxed speed-kind hits (some of them
  look-alikes that would better have stayed negatives); the code now keeps speed-kind hits as background
  (`data/signs/unboxed_det_v1_nokinds.jsonl` is the list the fine-tune used), and two validation-fraction label
  fixes were added after it was built (they don't touch training frames).
- **Possible wrong values on doubtful signs**: see Numbers (a "20" panel, a red-header "45", a SCHOOL-topped "25").
  The app should keep the two-agreeing-frames rule before announcing a limit.
- **The original held-out set was seen during development**: interim models were evaluated on it, which led to
  the look-alike negatives and a detection-triggered label audit (2 school speed limits added, 3 doubtful boxes
  ignored). So the headline is the fresh slice, labeled after development and hand-checked blind before any model
  ran on it; the original set is reported with and without the audit. The 5-view rule was added after the first
  final evaluation (which had one misread on the original set and none on the fresh set); its settings were chosen
  on validation and tune data only, and the first-pass numbers are kept in metrics.json.
- **Held-out labels are hand-checked for speed limits only.** On both sets every speed-limit box was checked by eye
  on the full-resolution crop and the largest upright "other" boxes were checked for missed speed limits (fresh:
  the 528 largest of 1,235, down to 40 px). The other kinds keep Opus's labels (original set: spot-checked).
- **Earlier builds leaked tune frames into training** (synthetic and classifier backgrounds drawn without the val
  fraction; classifiers cls1-cls4 also trained val-fraction reject crops). Fixed in the current builds; the tune set
  leaves out every frame any build trained on (235 of 751), and the delivered classifier (cls5) was trained on the
  clean build only. Its best epoch was picked on the same validation set the threshold was chosen on (mild bias;
  the tune set and the held-out sets are independent of it).
- **BDD100K is mostly New York and the Bay Area**: its speed limits are mostly 25 mph (NYC). Other values come from
  ctacke (25-80 mph, few distinct signs: video sequences of the same signs) and synthetic MUTCD renders. Real
  speed-value checks cover 20-55 mph only.
- **ctacke's public val split shares sign sequences with its train split** (adjacent frames of the same physical
  signs), so the public-split numbers are optimistic and ctacke is not used to choose thresholds.
- **Weak kinds:** wrong_way (2 fresh instances, none found), no_turn_on_red and no_right_turn have few real
  examples and lean on synthetic ones.
- **LISA is not used**: its official download returns 404 and the Roboflow copy needs an API key that wasn't
  provided in time; `prepare_data.py` picks it up automatically if a Roboflow YOLO export is placed in
  `data/signs/raw/lisa_yolo/`.
- **Not validated on our own camera** (the phone's 114° ultrawide): signs appear smaller and later than in BDD's
  narrower view; re-check on our drives.
- **Process incidents on the shared machine:** run 1 crashed twice from system-RAM shortages and was resumed from
  its per-epoch checkpoints. After it early-stopped, a watchdog script kept "resuming" the finished run, which made
  ultralytics start 7 short COCO8 trainings that wrote `runs/detect/train`, `train-2` ... `train-7` into the MAIN
  checkout and downloaded `C:\ClaudeProjects\datasets\coco8`. Deleting them was blocked by the permission system;
  they are left for the user to remove. Nothing else in the main checkout was touched.

## Best Fine-tune write-up
**What:** a US traffic-sign reader for a dashcam co-pilot: YOLO26s fine-tuned on 13 MUTCD sign kinds plus a
speed-limit value classifier, running as TensorRT FP16 engines at ~7 ms per frame on a laptop RTX 4050.

**Data:** no public dataset had US signs, US dashcam framing and speed values together, so we built one. BDD100K
dashcam frames have human boxes around signs but no sign type: Claude Opus 5.5 labeled 37k sign crops (kind + mph,
"unclear" when unsure; $31). Speed values beyond NYC's ubiquitous 25 come from the ctacke dashcam set (25-80 mph,
MIT) and 3,500 synthetic composite frames of public-domain MUTCD renders with every value 5-85, pasted into real frames
with perspective, blur, lighting and poles, next to unlabeled look-alikes (route shields, EXIT/RAMP advisory
plates). Two findings shaped the training: BDD doesn't box every sign (so a frame without labels is not a frame
without signs: unboxed detections are masked, never taught as background), and most "false alarms" mined from
unlabeled frames were real signs, so the 299 hard-negative frames were curated by eye from 872.

**Training:** 640x640 tiles cut at native resolution (distant signs stay 15-40 px), inference on the whole 1280-wide
frame; 39 epochs from COCO weights, then a 12-epoch fine-tune with hard negatives, picked on a separate tune set.
The classifier (18 classes incl. a reject class) learns every value from synthetic renders plus real crops. A value
is only spoken when 5 shifted crops all agree at >= 0.9: misreads wobble with the crop, real readings don't.

**Results:** on a fresh held-out slice labeled after development and checked blind: 89.6% of clearly readable speed
limits read right, no value contradicting a labeled sign (every other value checked by eye: real unlabeled limits
read right, and one doubtful panel), speed-limit AP50 0.875 (0.990 on the public ctacke split). Before/after on the
co-pilot's test-sheet sign questions: the general vision model answered 2 of 6; the detector answers DO NOT ENTER,
the school crossing sign, the one-way cross street, and correctly finds no sign on a truck-route plate and an
unreadable overhead sign, in ~7 ms instead of a cloud round-trip.
