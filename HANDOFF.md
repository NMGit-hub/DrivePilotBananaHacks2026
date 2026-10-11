# Handoff: DrivePilot, V2 in progress (2026-10-10, ~20:15 PDT)

For the next Claude Code session. Read this first, then CLAUDE.md (standing rules), V2_BUILD_PLAN.md (V2 milestones +
the user's decisions), PROGRESS.md (detailed status, measurements, known issues: kept current) and PLAN.md.

**Deadline: Sunday 2026-10-11, 8:59 pm PDT.** Cut from the bottom of the V2 order if time runs short
(V2_BUILD_PLAN.md "Order and cut line"). Everything below is committed on `main`; pushed to `origin/main` unless it
says otherwise (V2-M5 is pushed at its milestone end, after the review fixes).

## 1. State of V2 (details and numbers in PROGRESS.md "V2")
| Milestone | State |
| --- | --- |
| V2-M0 test inputs | done (simulated GPS drives, BDD GPX, clip finder) |
| V2-M1 parking spot | built, **measured below the bar on real footage**: 1/30 clear spots on the 28 usable BDD lot clips, 0.7 false/10 min; crawl clips (both halves labeled now) 0/8 found; the model check (Claude Haiku) rejects all spot events, so nothing false is spoken but nothing is found either. Needs our own lot footage (straight aisle drives with the phone). |
| V2-M2 overlay page | built + checked in the browser (now also our lane, the drift line red + edge pulse) |
| V2-M3 pedestrians | done; **now gated by YOLOPv2's drivable area** beside the feet (crawl 12 -> 9 warnings, lot 6 -> 5, dev 0; only false/borderline ones removed) |
| V2-M4 memory + highlight | built; sheet 22/3/4 of 29 (was 23/3/3), highlights right on the 7 object rows |
| V2-M5 YOLOPv2 lanes/drift | **done** (app lanes thread, page incl. the green target lane, lane guidance, vision context, pedestrians); lane index 57/58 answered; drift: 1 borderline alert in 5.7 min of real highway driving, virtual drifts 13/18 on time, lane changes 12/12 quiet (below the bars: PROGRESS Known issues); review: 6 bugs fixed. **Not done: the phone's 114° ultrawide (PLAN: undistort or a center crop) for lanes.** |
| V2-M6 sign detector | the other session (`C:\ClaudeProjects\DrivePilot_signs`, branch `v2-signs`) finished the 40-epoch detector and started a 12-epoch fine-tune (`det_ft`) at ~20:00 (GPU busy until ~21:30); due Sunday 11:00; **the app side is ready and wired** (sign alerts now muted in quiet mode) |
| V2-M7 RF-DETR | not started (cut first) |
| V2-M8 integration | started: the test sheet runs lane detection like the app (`eval/run_sheet.py`, `--no-lanes`); bench, full sheet, screenshots, README still to do |

## 2. Next steps (in order)
1. ~~V2-M5 review findings~~ **done**: 6 bugs fixed with regression tests (PROGRESS "V2-M5"), pushed. The sheet now
   has 45 rows (16 V2 rows: lanes, memory, signs, parking, road/safety) and runs lane detection and the parking
   watch like the app.
2. **V2-M6 sign detector merge** when `scripts/signs/STATUS.md` in the signs worktree says ready. Merge `v2-signs`
   (it only appends `SignsConfig` + `signs:`, adds files, `scripts/__init__.py`, `scripts/signs/__init__.py`; expect a
   small conflict at the end of `AppConfig`). In `app._start_signs` replace `getattr(self.cfg, "signs", None)` with
   `self.cfg.signs`; add the sign detector to `eval/run_sheet.py` the way the lane model was added (lock-step at
   `sign_alerts.fps`, `SignTracker` context for the vision model); run the app on a dev clip ("signs: on"), the sheet's
   sign rows, fps with it on.
3. **V2-M8 integration**: `scripts/bench_app.py --silent` with everything on (only when no training uses the GPU:
   `nvidia-smi --query-compute-apps=pid --format=csv`), the full test sheet (`eval/run_sheet.py --previous
   runs/20261010-142635_sheet/results.csv`), screenshots of the overlay page for the README (`.claude/launch.json` has
   `drivepilot-lanes`: a highway clip; note the browser pane only composites canvas frames when visible: pause the
   replay and call `draw()` before a screenshot), README + PROGRESS, push. Then a short demo plan for the user.

## 3. Things the user should know / decide (batch into one message)
- Parking finds almost nothing on BDD100K footage; it needs a recording of a slow drive straight down lot aisles
  with the phone (the user said recording will take time). The NTNU parking dataset (AAAI 2019, "Early Detection of
  Vacant Parking Spaces Using Dashcam Videos") is only available by email request to Mei-Chen Yeh
  (myeh@csie.ntnu.edu.tw): the user's call.
- `video.file.hfov_deg: 60` in config.yaml is right for BDD100K replays; set it to null for the user's own phone
  recordings (the phone's 114°).
- Remind the user to rotate the API keys after the hackathon (pasted in chat on 2026-10-09).

## 4. Working notes
- Run everything with `.venv/Scripts/python`; full tests `.venv/Scripts/python -m pytest -q` (2-6 min, slower while
  the GPU trains); `-m ruff check .`, `-m ruff format --check .`, `-m mypy .` (tests/ is excluded from mypy).
- **Don't edit code with Bash heredocs containing backslashes** (they mangle `\b`, `\n`): use Edit/Write, or a
  patch script written to the scratchpad (write files with `newline="\n"`).
- Perception caches: `scripts/cache_perception.py` writes `data/cache/perception/<clip stem>.pkl` (tracks + ego
  motion per frame); parking, pedestrian and lane evals replay them in seconds. Cached: the 10 dev clips, 40 lot
  clips, 32 crawl clips.
- Agents in worktrees have no `.venv`, `.env` or data: give them the main venv by absolute path and data paths;
  tell long-running ones to save their work incrementally (two stalled this session). Merge their branches yourself
  after review. The merged pedestrian worktree `.claude/worktrees/agent-a646a9a46d39b56d8` is still locked by a dead
  agent process: `git worktree remove -f -f <path>` then `git branch -d worktree-agent-a646a9a46d39b56d8`.
- The Anthropic SDK is 1.13: `temperature` goes in `extra_body` (handled in `answers/vision.py`).
- Piper synthesis is random: voice benchmark rows vary run to run; compare runs, not rows.
- `.claude/launch.json` has `drivepilot-overlays` (a dev clip with GPX, no voice/audio) for the preview tool.
- Per CLAUDE.md: commit after every working piece, push at milestone ends, an independent review per milestone,
  keep PROGRESS.md updated, batch questions for the user into one message.
