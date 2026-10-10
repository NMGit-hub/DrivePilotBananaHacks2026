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

## Numbers
To be filled with measured results (detector training in progress).

## Known issues / compromises
To be filled.

## Best Fine-tune write-up
To be filled.
