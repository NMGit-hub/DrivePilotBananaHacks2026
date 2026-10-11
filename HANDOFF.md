# Handoff: DrivePilot, V2 integrated (2026-10-11, ~00:45 PDT)

For the next Claude Code session. Read this first, then CLAUDE.md (standing rules), V2_BUILD_PLAN.md (V2 milestones +
the user's decisions), PROGRESS.md (detailed status, measurements, known issues: kept current), DEMO.md (the demo
plan) and PLAN.md.

**Deadline: Sunday 2026-10-11, 8:59 pm PDT.** Everything below is committed on `main` and pushed to `origin/main`.

## 1. State of V2 (details and numbers in PROGRESS.md "V2")
| Milestone | State |
| --- | --- |
| V2-M0 test inputs | done |
| V2-M1 parking spot | built, **below the bar on BDD100K footage** (1/30 lot spots; the Claude Haiku check rejects every candidate, so nothing false is spoken). Needs the user's own lot footage. |
| V2-M2 overlay page | done (screenshots in `docs/screenshots/`, linked from README) |
| V2-M3 pedestrians | done |
| V2-M4 memory + highlight | done |
| V2-M5 YOLOPv2 lanes/drift | done; drift below its bars on BDD (PROGRESS Known issues); the phone's 114° view (70° center crop) is untested on real phone footage |
| V2-M6 sign detector | done; "School zone ahead." once per zone, "Speed limit is now N." once per new limit (fixed 2026-10-10 23:15) |
| V2-M7 RF-DETR | cut |
| V2-M8 integration | done: bench with lanes + signs passes every target (3 runs); 45-row sheet 33 right / 6 partly / 6 wrong (73 %, final run); README screenshots; DEMO.md; independent review (7 bugs, 3 safety, all fixed) |

## 2. What's left (needs the user, see section 3)
- Own drives: record with the phone camera app + GPSLogger, align with `scripts/align_gpx.py` (README "Record your
  own drive"), then run `scripts/eval_lanes.py` (cache, index on frames labeled by hand, drift) and
  `scripts/eval_parking.py` on them with `--set video.file.hfov_deg=null`. Nothing recorded yet
  (`data/videos/own/` doesn't exist).
- V1's deferred live drive; the submission video (OBS, PLAN.md "Demo video"; DEMO.md "Recording it").

## 3. Things the user should know / decide
- Parking finds almost nothing on BDD100K footage; it needs a recording of a slow drive straight down lot aisles.
- `video.file.hfov_deg: 60` in config.yaml is right for BDD100K replays; use `--set video.file.hfov_deg=null` for
  the phone's own recordings.
- Stray COCO8 runs from the sign session (`runs/detect/train*` here and `C:\ClaudeProjects\datasets\coco8`) can be
  deleted by the user; the sign session's worktree `C:\ClaudeProjects\DrivePilot_signs` (branch `v2-signs`, merged)
  too.
- Rotate the API keys after the hackathon (pasted in chat on 2026-10-09).

## 4. Working notes
- Run everything with `.venv/Scripts/python`; full tests `.venv/Scripts/python -m pytest -q` (~3 min);
  `-m ruff check .`, `-m ruff format --check .`, `-m mypy .` (tests/ is excluded from mypy).
- **Don't edit code with Bash heredocs containing backslashes** (a `\b` became a backspace character in
  `voice/router.py` this session): use Edit/Write, or a patch script written with the Write tool.
- The app builds the lane model, then the sign model, before perception starts (both capture CUDA graphs while
  loading; built in parallel both failed). Nothing reloads a TensorRT engine after start-up.
- Browser-pane screenshots: 800x500 at most; pause the replay (POST `/api/control {"action":"pause"}`) and call
  `draw()` before a screenshot; the files land in the Claude tool-results folder. `.claude/launch.json` has
  `drivepilot-lanes`, `-signs`, `-green`, `-overlays`, `-replay`, `-camera`.
- The test sheet (`eval/run_sheet.py`, ~8 min for 45 rows) varies run to run by a few rows (model + judge): compare
  categories over runs, not single rows.
- Agents: give them the main venv by absolute path; tell long-running ones to save work incrementally; they can't
  write `.md` report files (ask for the report in their final message).
- The Anthropic SDK is 1.13: `temperature` goes in `extra_body` (handled in `answers/vision.py`).
