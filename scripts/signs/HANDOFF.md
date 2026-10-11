# Sign fine-tune (V2-M6): handoff to the next session (2026-10-10, ~17:15 PDT)

You are continuing the V2-M6 sign fine-tune session. Read `SIGN_FINETUNE_PROMPT.md` (task + ground rules),
`CLAUDE.md`, then this file, then `scripts/signs/README.md` and `scripts/signs/STATUS.md` (draft). Work only in
the worktree `C:\ClaudeProjects\DrivePilot_signs` (branch `v2-signs`), never commit to `main`; the only allowed
write in the main checkout is the final delivery to `C:\ClaudeProjects\DrivePilot_BananaHacks2026\models\signs\`.
Python: `C:\ClaudeProjects\DrivePilot_BananaHacks2026\.venv\Scripts\python.exe` (call it PY below), with
`YOLO_AUTOINSTALL=false`. Deadlines: working detector to the other session by **Sun 2026-10-11 11:00 PDT**, project
deadline Sun 20:59 PDT. Check the time with `date` (it is easy to lose track).

## FINAL STATE (2026-10-10 ~21:40 PDT) — the milestone is done; read scripts/signs/STATUS.md
- Delivered to the main checkout's models/signs (det_ft detector + cls5 classifier, engines + sidecars; smoke-tested).
- Final settings: value_views 5, value_conf 0.9 (chosen on BDD validation boxes + tune, before the final held-out run).
- Fresh held-out: 69/77 clearly readable limits right, no value contradicting a labeled sign; flagged values all
  checked by eye (STATUS "Numbers"). Full pytest 914 passed, ruff, mypy clean; branch pushed.
- The last focused review (value views, reporting) is done and fixed (~22:10): "other" boxes are their own value
  category, the two "20" panels were read as SPEED LIMIT 20 in their source videos (data/videos/probe/), view sweeps
  and timing are in metrics.json, STATUS discloses that the view offsets were probed on held-out frames.
  Delivered models and config are unchanged by these fixes. 920 tests pass.
- Left for the user: delete the stray COCO8 runs in the main checkout (`runs/detect/train`, `train-2..7`) and
  `C:\ClaudeProjects\datasets\coco8` (the permission system blocked deleting them).

## Session 3b state (2026-10-10 ~20:10 PDT) — superseded by FINAL STATE above
- Main run `runs/signs/det` early-stopped at epoch 39 (best epoch 24, mAP50 0.854). The old watchdog then kept
  "resuming" the finished run: 7 stray COCO8 runs landed in the MAIN checkout `runs/detect/train`, `train-2..7`
  and `C:\ClaudeProjects\datasets\coco8` (deleting them was denied by the permission classifier: the user was
  asked to remove them). Watchdog killed. Never use `--resume` on a finished run.
- All reviewer fixes 1-16 are implemented and committed (cb11c1f, 8c49c3e), plus a second independent review's
  findings (thresholds chosen on wrong+unverifiable, raw cache tied to model hashes, eval_values runs the FP16
  engine, speed-kind unboxed hits not masked, mining/curation split). 106 non-GPU sign tests, ruff, mypy clean.
- Fresh held-out set built and hand-checked BLIND: `eval/sign_labels_bdd_fresh.json` (2,675 frames + 500
  sign-free; 77 clearly readable speed limits). `make_heldout.py --set fresh`.
- Detector hard negatives: 872 mined -> 299 curated by eye (`curate_det_negatives.py`; most mined hits were real
  unboxed signs). `mine_unboxed.py` masks unboxed non-speed hits in training frames.
- Fine-tune `runs/signs/det_ft` RUNNING (detached, started 19:32, 12 epochs, AdamW lr0 0.0002, ~7.5 min/epoch,
  ends ~21:00). NOTE: its dataset masked ALL unboxed hits incl. speed kinds (built before the review fix); a
  re-mine with kinds is running (`data/signs/mine_unboxed2.log`) to count how many speed-kind masks there were.
- Next: pick det vs det_ft on `evaluate.py tune --tag _det/_ft` (detection metrics; `--set signs.weights=...`);
  `prepare_data.py cls` (clean build with BDD look-alikes in val) -> `train.py cls --name cls5 --epochs 20
  --batch 256 --workers 2 --no-copy`; copy winners to models/signs/{signs_yolo26s,speed_value_cls}.pt (cls5 is
  mandatory: cls3 leaked val-fraction reject crops); export engines (nothing else heavy running); thresholds with
  the engines: value_conf = max(eval_values BDD-validation choice, evaluate tune lowest with 0 wrong+unverifiable);
  final eval all sets + check every unverifiable value by eye; bench; sheet; STATUS (draft written, PENDING
  markers); deliver.

## Session 3 progress (2026-10-10, 17:10-17:45 PDT; ended by the usage limit)
- At 17:12: detector training alive (epoch 17/40, ~2.6 GB GPU, ETA ~20:00), watchdog PID 24908 idle (no
  watchdog.log yet = no restarts needed), fresh_valid.jsonl complete (9,000 lines). Free RAM only ~1.2 GB.
- DONE, committed (74 non-GPU sign tests pass): `perception/signs.py` refactor (fixes 5-8, 13, 15): `RawSign`,
  `SignDetector.raw_detections(frame, classify_min_px)`, module `select(raw, cfg, w, h)` (per-kind conf ->
  `drop_duplicates(cfg.duplicate_iou)` -> value rule incl. `value_edge_px`, both px rules measured at 1280 width),
  `check_value_classes`, `engine_is_fp16` + `engine_layer_names` (the `images_cast_to_fp16` check), sidecar
  `<engine>.json` helpers (`sidecar_path`, `file_sha256`, `engine_staleness`; `_resolve` falls back to the .pt
  when stale), `mem_get_info` logging.
- DONE, partly (committed, not yet exercised): `prepare_data.py` helpers `in_val`, `rel_path`,
  `write_used_for_training`, `original_build_images`, `training_images`, `restore_boxes` (used in load_scaled
  and write_tiles: fix 10), `paste_footprint` + `synth_signs` (fix 9, overlap checked before pasting).
- STILL TO DO from fix list: build_det/build_cls must use `in_val` for synth/cls backgrounds, drop `--val-frac`,
  call `write_used_for_training`, add val-fraction other/hardneg crops to cls/val (fix 3/4);
  `evaluate.py` rewrite on `raw_detections` + `select` with the new wrong-value accounting (fixes 1, 11) and
  `tune_items` excluding `training_images()`; `export_engines.py` must write the sidecar JSON (weights_sha256,
  imgsz, TRT version) and rebuild on hash mismatch; `eval_values.py` (fix 4); mine_det_negatives: skip val-fraction
  frames; make_heldout `--set fresh` + 500 sign-free frames (fix 12); tests listed in fix 16; then plan steps 2-9.
- Note: evaluate.py still calls `detect()` at value_conf 0 (works, but not yet the shared-select design).

## User decisions so far
- Approved: spend ~$7 more on Claude labels for a fresh held-out slice (done/running, see below) and keep training.
- Approved earlier implicitly: Opus 5.5 labeling (~$40 spent before the fresh slice; tell the user the total).
- The user asked for a Roboflow API key in `C:\ClaudeProjects\DrivePilot_signs\.env` (ROBOFLOW_API_KEY) for LISA;
  it never arrived. LISA is not used (official download 404s). `prepare_data.py` would pick up a Roboflow YOLO export
  in `data/signs/raw/lisa_yolo/{train,valid}` automatically.
- Tell the user when long training starts/ends (other session's latency benchmarks are invalid meanwhile).

## Things running right now (check first!)
1. **Detector training** (YOLO26s, `runs/signs/det/`), resumed at 17:04 from epoch 16/40 as a detached process
   (`scripts/signs/train.py det --resume --workers 3`, logs `data/signs/det_train3.log/.err`). Progress:
   `cut -d, -f1,7-9 runs/signs/det/results.csv | tail` (cols: epoch, recall, mAP50, mAP50-95 on the BDD-train val
   tiles; ~0.82-0.84 mAP50 at epochs 10-16). ~6-8 min/epoch → expected end ~20:00-20:45. It crashed twice from
   **system RAM** shortages (15.8 GB machine shared with the other session's app/tests); each time it resumed
   from the last epoch checkpoint (`last.pt`, saved every epoch, `save_period=1`, 80 MB each → delete old
   `epochN.pt` later to save disk).
2. **Watchdog** (detached PowerShell, PID 24908): `C:\Users\neilm\AppData\Local\Temp\claude\C--ClaudeProjects-DrivePilot-BananaHacks2026\a6367ee2-22af-4349-9b8b-82a7444c005c\scratchpad\watchdog.ps1`.
   Every 90 s: if no `python.exe` with "train.py det" in its command line is running and `results.csv` has < 40
   epochs, it relaunches `train.py det --resume --workers 3` (logs `data/signs/det_train_r<n>.log`, notes in
   `data/signs/watchdog.log`). It exits when 40 epochs are recorded. **If you start the fine-tune (also
   "train.py det") while the watchdog still runs, it won't interfere (it only acts when no train.py det runs),
   but kill it (`Stop-Process -Id 24908`) once the main run is finished to be safe.**
3. **Fresh held-out labeling: DONE** (9,000 labels, 0 unparsed, $7.46; total Claude labeling spend now ≈ $47.9).
   Details: Claude Opus 5.5, started ~17:06: the next 9,000 BDD valid sign
   crops (shards 000/001, min height 12) → `data/signs/bdd_labels/fresh_valid.jsonl`. The name deliberately does
   NOT match the `valid*.jsonl` glob used by `read_labels("valid")`, so the original held-out set is unaffected.
   Re-run the same command to finish if it stopped (it resumes; spend is capped by --max-usd):
   `PY scripts/signs/label_bdd_signs.py --split valid --shards shard_000 shard_001 --min-h 12 --limit 9000 --model claude-opus-5-5 --workers 6 --max-usd 8 --out data/signs/bdd_labels/fresh_valid.jsonl --env-file C:/ClaudeProjects/DrivePilot_BananaHacks2026/.env`

## Git state
- Branch `v2-signs`, pushed up to `70595dc`; `186e7b4` committed but **not pushed**.
- **Uncommitted** (all intended): `scripts/signs/train.py` (resume passes `--workers`), `config.yaml` + `core/config.py`
  (new `SignsConfig` fields `value_edge_px: 2.0`, `duplicate_iou: 0.6`, not yet used by `perception/signs.py`),
  `tests/test_signs.py` (GPU test rewritten: blank-frame + engine-suffix check, and a dev-clip test that expects
  `do_not_enter` at 25.0 s of `red_light_rain_school_zone`), `scripts/signs/STATUS.md` (design choices, training
  setup, known issues drafts), this file. Run ruff/mypy/pytest before committing.
- Last full checks: 878 tests passed (full suite), ruff, ruff format, mypy clean (before the uncommitted edits).

## What exists (all in the worktree)
- `perception/sign_classes.py`: KINDS (13, order fixed), VALUE_KINDS, SPEED_VALUES 5..85, `kind_from_label`,
  `parse_speed_value`.
- `perception/signs.py`: `SignDetection(kind, value, conf, box, value_conf)`, `SignDetector(cfg, root)` (`detect`,
  `warmup`, `_verify_gpu`, `_resolve` engine/pt), `crop_square` (must equal `prepare_data.crop_square`; test
  asserts it), `decide_value`, `parse_value_class`, `engine_path` (`<stem>_<h>x<w>_fp16.engine`), `drop_duplicates`
  (module constant DUPLICATE_IOU=0.6 → to move to config, see fixes).
- `scripts/signs/`: label_bdd_signs.py, verify_ctacke.py, synth.py, prepare_data.py, make_heldout.py, train.py,
  evaluate.py (sets: bdd, public, tune; `--no-audit`), eval_values.py, mine_hard_negatives.py,
  mine_det_negatives.py, export_engines.py (`--setup-only`, `--force`, `--deliver <dir>`), bench_signs.py
  (`--with-perception` runs `scripts/bench_pipeline.py` concurrently), sheet_signs.py (test-sheet sign rows),
  review_sheet.py (contact sheets), README.md, STATUS.md.
- `eval/sign_labels_bdd.json` (tracked): original held-out set, 2,623 BDD valid frames (shard_000) built by
  make_heldout.py from `data/signs/bdd_labels/valid.jsonl` + REVIEW + AUDIT. Counts: speed_limit 89 (78 clearly
  readable), school_speed_limit 2 (+2 via audit), one_way 442, school_zone 158, no_left_turn 62, do_not_enter 49,
  stop 47, pedestrian_crossing 41, no_u_turn 27, yield 25, no_right_turn 8, no_turn_on_red 7, wrong_way 1.
- `.venv-trtbuild` (worktree; TensorRT 11.4.0.106, same as `.venv`) and `.venv-signs` (pyarrow, huggingface_hub).
- Models: `models/signs/speed_value_cls.pt` = classifier **cls3** (copied from runs/signs/cls3); base weights
  `models/signs/yolo26n-cls.pt`, `yolo26s-cls.pt`; detector base `C:/ClaudeProjects/DrivePilot_BananaHacks2026/models/yolo26s.pt`.
  Interim engine `data/signs/interim_last_736x1280_fp16.engine` (epoch ~4 weights, timing test only).

## Data (gitignored, `data/signs/`)
- `raw/bdd100k_det/data/images/{train/shard_000..003, valid/shard_000..001}` (HF mirror `dronefreak/BDD100K`;
  `metadata.jsonl` per shard, boxes xywh, class 9 = traffic sign). Train shards 0-1 labeled; 2-3 downloaded but unused.
- `raw/ctacke/` (HF `ctacke/speed-limit-signs`, MIT, 1920x1080, folders mphXX; mph25 = SCHOOL SPEED LIMIT).
- `bdd_labels/train.jsonl` (shard 0, 18,699 crops, $15.57), `train_s1.jsonl` (shard 1, 18,548, $15.43),
  `valid.jsonl` (9,000 crops / 2,623 frames, $7.46), `fresh_valid.jsonl` (running). Pilot files in `pilot/`.
- `ctacke_readable.jsonl` / `.raw.jsonl` (Opus re-read: 844/1045 readable, $0.97).
- `hardneg_cls.jsonl` (64 mined), `hardneg_cls_curated.jsonl` (27 true negatives kept by eye: route shields,
  EXIT/advisory plates, misc). `hardneg_det.txt`: not created yet.
- `templates/` (17 MUTCD PNGs), `templates/distractors/` (33 route shields/plates), `fonts/Overpass.ttf`.
- `det/` (detector dataset as used by the current run: 14,518 train tiles, 425 val; built BEFORE distractors /
  hard negatives / leakage fixes), `cls/` (classifier dataset, latest build = cls4's: reject crops exclude the
  val fraction; files `*_trueNN.png` in `val/none` are unreadable crops of real NN mph signs).

## Measured so far (interim; final numbers must be re-measured)
- Classifier on labeled held-out crops (no detector), 80 clearly readable (78 speed_limit + 2 school):
  cls3: no top-1 mistakes; at 0.9 → 75 right / 0 wrong / 5 withheld (validation picked 0.9).
  cls4 (`runs/signs/cls4/weights/best.pt`, trained without tune-split leakage of reject crops): validation picked
  0.6 but held-out has 1 wrong (30→50 at p 0.766) up to 0.7; at 0.8: 75/0/5; at 0.9: 73/0/7.
  config `value_conf` is currently 0.9.
- Interim end-to-end on the original held-out (epoch-4 detector, cls2 @0.98): speed_limit AP50 0.80, recall 0.85;
  66/78 readable read right; **10 wrong values**, mostly route shields (EAST 25, I-45-like, airport 40) and text
  signs detected as speed limits → led to distractors + hard negatives + the audit.
- Speed (interim engine, GPU idle): detect+classify median 7.2 ms, p95 9.0 ms; TensorRT inference 3.45 ms;
  detector verified cuda:0 fp16. Engine size 214 MB. Under training load timings rise; re-measure at the end.
- Sheet sign rows (interim): finds do_not_enter (0.87) for "red signs ahead", school_zone (0.85), one_way and no
  no_turn_on_red for "right on red?"; false speed_limit (0.67, no value) on LAST LOCAL TRUCK ROUTE sign.
- ENGINE NOTE: ModelOpt AutoCast engines keep FP32 I/O (`TensorRTBackend.fp16` is False) but contain an
  `images_cast_to_fp16` layer (engine inspector layer names) → use that, plus metadata quantize==16, to verify FP16.

## Independent review findings (all accepted) and the planned fixes — NOT yet implemented
1. evaluate.py wrong-value hole: values on ignore regions / on labeled signs touching ignore boxes escape. Fix:
   for every reported value, first match any labeled sign at IoU>=0.5 (any kind): right iff it's a speed kind
   with the same value (speed sign with unknown value → "unverifiable"); else wrong. Unmatched + on ignore →
   "unverifiable" (must be checked by eye, listed in STATUS); unmatched elsewhere → wrong. Update
   `test_eval_ignore_regions_count_neither_way_but_values_are_listed` and make_heldout docstring (it says values
   there count as wrong).
2. Held-out influenced by development (audit + distractors chosen after seeing held-out errors). Fix: headline =
   the **fresh slice** (labeled now, hand-check its speed limits **blind**, before running any model on it) and
   `--no-audit` for the original set; disclose in STATUS.
3. Tune-set leakage: synthetic det backgrounds `neg[n_neg:n_neg+synth]` and cls backgrounds `neg[:400]` ignore the
   val fraction (186 of 751 tune frames were training backgrounds). Fix: filter both by `stable_frac >= VAL_FRAC`;
   remove the `--val-frac` CLI flag (always VAL_FRAC); write `data/signs/used_for_training.txt`; `tune_items`
   must also exclude the ORIGINAL build's backgrounds, reproducible as: `rng = random.Random(0)`; one
   `rng.randrange(2**31)` per BDD positive (bdd_samples order); then `rng.shuffle(neg)` over ALL neg (no hard
   negatives existed then); `n_neg = int(0.3*len(pos))`; synth = `neg[n_neg:n_neg+3500]`; and cls backgrounds =
   unshuffled `bdd_samples("train")[1][:400]`.
4. Threshold validation weak: cls/val has no real look-alikes; 77% of real positives are ctacke val (leaks with
   ctacke train). Fix: add BDD-train val-fraction "other"/not_a_sign crops and val-fraction hard negatives to
   cls/val as "none"; exclude ctacke from threshold selection (report separately); final value_conf = max(lowest
   0-wrong threshold on cls val, lowest 0-wrong end to end on the cleaned tune set).
5. Stale engine silently used. Fix: at export write a sidecar `<engine>.json` with the weights' sha256 (+ imgsz,
   TRT version); `_resolve` warns and falls back to the .pt (GPU PyTorch) on mismatch, or if no sidecar and the
   weights are newer than the engine; export rebuilds when the hash differs.
6. Validate classifier class names == {v5..v85, none} (raise otherwise).
7. FP16 check: add the `images_cast_to_fp16`/"fp16" layer-name check (see ENGINE NOTE); log device memory via
   `torch.cuda.mem_get_info` (memory_allocated misses TensorRT).
8. No value for boxes within `value_edge_px` of the frame border (config field already added).
9. synth `_synth_job`: check overlap BEFORE compositing (currently overlapping signs are pasted unlabeled).
10. `mask_box` enlargement paints neighbouring labeled signs (537/4,076 boxes touched): restore labeled boxes'
    pixels after masking (load_scaled and write_tiles).
11. evaluate.py: raise on missing frames; guard empty timing lists.
12. Add a sample (~500) of sign-free BDD valid frames (no traffic-sign boxes in metadata) to the fresh held-out
    file, to measure false alarms/wrong values on plain driving.
13. Eval/runtime parity with class_conf: planned refactor — `SignDetector.raw_detections(frame, classify_min_px)`
    returns `RawSign(kind, conf, box, top_value, top_prob)` for all boxes at predict conf; module function
    `select(raw, cfg, frame_w, frame_h)` applies per-kind conf → `drop_duplicates(cfg.duplicate_iou)` → value
    rule (VALUE_KINDS, height*1280/w >= value_min_px, not within value_edge_px of the border, top_prob >=
    value_conf); `detect = select(raw_detections(frame, cfg.value_min_px))`. evaluate.py stores raw detections
    (cfg conf 0.05, classify_min_px 1) and calls the same `select` per threshold setting.
14. STATUS says 0.98: fix (now 0.9, may change after #4).
15. DUPLICATE_IOU → `cfg.duplicate_iou` (config field added).
16. Tests to add: `_verify_gpu` both branches with fake backends (CPU param → SignGpuError; engine metadata fp32
    with half → error); `_synth_job` labels; `bdd_samples` masking (unlabeled masked, "other" background);
    `ctacke_samples` with check files; `_real_crops` "none@v"; eval_values sweep + `_true` parsing; the ignore
    hole; split disjointness (no tune frame in any training job); stale-engine fallback; class-name validation.

## Remaining plan (in order)
1. Implement fixes 1, 3-16 + tests (perception/signs.py refactor first). Keep ruff/mypy/pytest clean.
2. Fresh held-out: when `fresh_valid.jsonl` is done, extend make_heldout.py with `--set original|fresh`
   (original reads only `valid.jsonl`; fresh reads `fresh_valid.jsonl` + 500 sign-free frames) →
   `eval/sign_labels_bdd_fresh.json`. Hand-check every fresh speed/school-speed box by eye on full-res crops
   (`review_sheet.py fresh_labels --kind speed_limit`), mark not-clearly-readable keys, BEFORE any model run on it.
   Also check the labeler's misses (size-sorted upright low-saturation "other"/"unclear" crops) as done before.
3. When the main run ends: `PY scripts/signs/mine_det_negatives.py --weights runs/signs/det/weights/best.pt`
   (writes `data/signs/hardneg_det.txt`); rebuild `prepare_data.py det` and `cls` with the fixes; optionally
   retrain the classifier (cls5) on the clean build; fine-tune:
   `PY scripts/signs/train.py det --name det_ft --base runs/signs/det/weights/best.pt --epochs 12 --lr0 0.002 --warmup-epochs 0 --batch 8 --workers 3 --no-copy`
   (detached via Start-Process; >2 h jobs get killed if run as harness background tasks). Pick det vs det_ft on
   the **tune** set (not held-out). Copy the winner to `models/signs/signs_yolo26s.pt` and the classifier to
   `models/signs/speed_value_cls.pt`.
4. Thresholds per fix #4 (`eval_values.py`, `evaluate.py tune`); set `value_conf` in config.yaml + SignsConfig default.
5. Export engines with NOTHING else heavy running (ModelOpt export peaked ~14 GB RAM and crashed training once):
   `PY scripts/signs/export_engines.py --force`.
6. Final evaluation with the engines: `evaluate.py all --write-metrics` (fresh = headline; original `--no-audit`
   and with audit; public = ctacke val, optimistic; tune), `eval_values.py --write-metrics`,
   `bench_signs.py --clip C:/ClaudeProjects/DrivePilot_BananaHacks2026/data/videos/dev/red_light_rain_school_zone__b61456bb-5bfb50ee.mov`
   alone and `--with-perception --set perception.weights=C:/ClaudeProjects/DrivePilot_BananaHacks2026/models/yolo26s.pt`
   (check `Get-Counter '\Processor Information(_Total)\% Processor Performance'` and nvidia-smi for throttling first),
   `sheet_signs.py --videos C:/ClaudeProjects/DrivePilot_BananaHacks2026/data/videos/dev`. Run the GPU tests
   from the main checkout or with the dev clip path available. Check "Done when": public speed-limit mAP50 >= 0.8
   with per-class AP; held-out >= 80% of clearly readable speed limits with the right value and 0 wrong values;
   per-class P/R; timing <= 10 ms; FP16/CUDA verified.
7. STATUS.md: numbers + how measured, compromises (pseudo-labels, held-out influence + audit, NYC/25 mph bias,
   ctacke leakage, thin classes, crashes/resumes, LISA missing, camera not validated, label spend total),
   integration notes (already drafted), Best Fine-tune write-up (data, training setup, results, before/after on
   the sheet's sign rows: vision model 2/6; detector answers DO NOT ENTER, school zone, one way/no NTOR rows).
8. Second independent subagent review of the fixes; fix findings with tests; full pytest, ruff, ruff format,
   mypy; commit; push `origin v2-signs`.
9. Deliver: `PY scripts/signs/export_engines.py --deliver C:/ClaudeProjects/DrivePilot_BananaHacks2026/models/signs`
   (copies `signs_yolo26s.pt`, `signs_yolo26s_736x1280_fp16.engine`, `speed_value_cls.pt`,
   `speed_value_cls_96x96_fp16.engine`; add the sidecar JSONs) and record names, sizes, TensorRT version in
   STATUS.md; make sure `models/signs/metrics.json` is committed in the worktree.

## Gotchas
- Bash heredocs in this environment can corrupt escapes (a `"\t"` became a literal tab): edit code with
  Edit/Write tools; use heredocs only for text without backslashes.
- Harness background tasks are killed after 2 h: launch long jobs with PowerShell `Start-Process ... -WindowStyle Hidden`.
- Shared machine: GPU 6 GB (keep training <= 3.5 GB: batch 8 = 2.6 GB), RAM 15.8 GB (other session's app,
  cache_perception.py and pytest use several GB); keep dataloader workers <= 3; never run engine export during training.
- Old orphan python workers from 10:42 AM (PIDs 31092, 6036, 25404, 34308) are not ours: leave them.
- Running training from stdin (`python -`) breaks Windows dataloader workers: always use script files.
- `read_labels(split)` globs `<split>*.jsonl`; parallel label runs must write separate files.
- ultralytics classify val uses folder names; reject class folder is `none`; value folders `v5`..`v85`.
