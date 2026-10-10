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
To be filled.

## Best Fine-tune write-up
To be filled.
