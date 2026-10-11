# Handoff: DrivePilot, V2 in progress (2026-10-10, ~16:30 PDT)

For the next Claude Code session. Read this first, then CLAUDE.md (standing rules), V2_BUILD_PLAN.md (V2 milestones +
the user's decisions), PROGRESS.md (detailed status, measurements, known issues: kept current) and PLAN.md.

**Deadline: Sunday 2026-10-11, 8:59 pm PDT.** Cut from the bottom of the V2 order if time runs short
(V2_BUILD_PLAN.md "Order and cut line"). Everything below is committed and pushed to `origin/main` unless it says
otherwise.

## 1. State of V2 (details and numbers in PROGRESS.md "V2")
| Milestone | State |
| --- | --- |
| V2-M0 test inputs | done (simulated GPS drives, BDD GPX, clip finder) |
| V2-M1 parking spot | built, **measured below the bar on real footage**: 1/30 clear spots on the 28 usable BDD lot clips, 0.7 false/10 min; the model check (Claude Haiku) then rejects all 5 spot events incl. the 1 hit, so nothing false is spoken but nothing is found either. A Haiku periodic scan was measured instead (47 % at 26.6 false/10 min): not shipped. Needs our own lot footage (straight aisle drives with the phone). |
| V2-M2 overlay page | built + checked in the browser (boxes on cars, badges, YOU badge, light labels, pedestrians amber/red, nav banner with lanes, speed, speed-limit badge, roundabout diagram, highlight pulse) |
| V2-M3 pedestrians | merged from an agent worktree; 0 alerts on the 10 dev clips; on 32 extra crawl clips 12 alerts (of 8 looked at: 3 right, 1 borderline, 4 false: glare, a dark interior, a reflection, a sidewalk walker) |
| V2-M4 memory + highlight | built; sheet 22/3/4 of 29 (was 23/3/3), highlights right on the 7 object rows |
| V2-M5 YOLOPv2 lanes/drift | **part 1 committed on an agent branch, not merged** (section 2.1) |
| V2-M6 sign detector | training in another session (`C:\ClaudeProjects\DrivePilot_signs`, branch `v2-signs`, due Sunday 11:00); **the app side is ready and wired** (`state/signs.py`, `perception/sign_worker.py`, `app._start_signs`) |
| V2-M7 RF-DETR | not started (cut first) |
| V2-M8 integration | not started |

Also this session: an independent review of M1/M2/M4 (11 bugs, all but one fixed with 41 regression tests in
`tests/test_v2_review.py`; the unfixed one, single-gap rows in simulation, is an xfail with the reason), a replay
seek fix, README for V2. 1092 tests pass; ruff, ruff format, mypy clean.

## 1b. Session of 2026-10-10 evening (stopped at the usage limit)
- **V2-M5 lanes branch merged into main** (`d644cc0`; the agent's pending `state/lanes.py` change committed on its
  branch first). Not wired into the app yet. **No lane tests exist yet** (the agent never wrote any): still to do.
- Lane index re-measured from main: **56/57 answered frames right, 57/65 answered** (86 % of all labeled frames,
  bar 85 %); lane count 19/26 (it undercounts lanes on the right: use `lanes_left`/index for guidance, not `lanes_right`).
- **Drift eval added** (`scripts/eval_lanes.py drift [--frames]`, frames in `runs/lane_drift/`; `LaneTracker.lane_changes`
  is now filled). Mask caches copied to `runs/lanes_cache/`. **Result: 10 drift alerts in 5.7 min of highway
  driving (bar: 0).** Frames looked at: lines sit on the real lane lines; most alerts are steady driving. Cause: the
  alert uses the absolute offset, which includes the dashcam's mount position (per-clip median offset -0.53..+0.22 m)
  plus ±0.3-0.5 m wander (p5/p95). 5c5dc708 at 14-18 s may be a real mild drift right (the dash line comes close);
  40f7f545 at 29-31 s moves left as a car passes on the right. Planned fix: learn the habitual lane position (running
  median at speed, arm after ~10 s) and measure drift relative to it; require gap ≤ ~0.1 m (near the line); re-run.
  Also: 490ba810 / 7576a8b6 show 13 / 7 spurious "lane changes" (unwrap flipping on forks and at night): debounce.
- Review notes so far: `LaneWorker` duplicates `SignWorker` (merge into one latest-frame worker); `update()` holds the
  lock for the whole ~17-22 ms measurement (other threads' `snapshot()` would block: measure outside the lock);
  tracker CPU ~17-22 ms/frame (half in `_search`).
- Then: virtual-drift clip (`warp_lateral` exists; no `virtual` subcommand yet), integration (section 2.1), review.
- **Crawl labels half B**: a labeling subagent was started writing `eval/parking_labels_crawl_b.yaml` clip by clip
  (uncommitted). Check how many clips it finished, validate, commit, then score (section 2.2).

## 2. Next steps (in order)
1. **Finish and merge V2-M5 lanes.** The agent stalled during the drift evaluation (no final report). Its worktree:
   `.claude/worktrees/agent-af32ec16370458bc6`, branch `worktree-agent-af32ec16370458bc6`, commit `75e6b3d`
   ("YOLOPv2 lanes on the GPU, the lane tracker, hand-labeled lane index": `perception/lanes.py` LaneWorker with the
   official TorchScript in FP16, `state/lanes.py` LaneTracker, `eval/lane_labels.yaml` 65 hand-labeled frames on 11
   highway clips, `scripts/eval_lanes.py`, `lanes:` config) plus an **uncommitted change to `state/lanes.py`** in
   that worktree. Do: look at that diff (`git -C .claude/worktrees/agent-af32ec16370458bc6 diff`), run its tests and
   `scripts/eval_lanes.py` with the main venv from the worktree root, finish the drift evaluation (0 drift alerts on
   the BDD highway clips' straight driving / lane changes; a "virtual drift" clip), review the diff strictly (or
   with a review subagent), merge into main, then integrate: app (world thread offers frames to the LaneWorker;
   drift alert as a SAFETY utterance, pre-rendered), overlay (our lane polygon, the drift line red), lane index into
   lane guidance instead of the model's count when confident, pedestrians' `PathZone` from the drivable area (should
   remove the sidewalk false alarm). Highway clips are in `data/videos/highway/` with GPX (fetched by the agent).
   The weights `models/lanes/yolopv2.pt` (156,380,200 bytes, user-approved download) are in place.
2. **Crawl-clip labels, half B.** `eval/parking_labels_crawl_a.yaml` exists (16 clips, 7 usable, only 2 clear spots,
   both the stall we park in; committed). The labeler for the other 16 clips stalled without writing
   `eval/parking_labels_crawl_b.yaml`. Re-run a labeling subagent on: 5b75efcd-38b94e2a 6c616c13-dbd2fbe1
   6e391737-c6346f6c 6e692d74-de71f4ca 776ea20e-f4bdf433 7960b4af-fe2ade34 817d2b98-9dba766a 8dbb49b4-5ac68c54
   92b74188-873c9063 a4a9768c-0641c4b6 a4c6b745-eaa4f49d a4fd7534-866c5cb7 ad64ebe9-40de2259 bb1a255a-99414370
   bcaf73c1-9433521c c5b64f27-53710b20 (videos `data/videos/parking_crawl/`; same schema and method as half A; tell
   it to write the YAML incrementally, clip by clip, so a stall doesn't lose the work). Then score:
   `scripts/eval_parking.py --labels eval/parking_labels_crawl_a.yaml` (and b), and
   `scripts/eval_parking_check.py --labels ...` (live model checks). Half A so far: 0/2 found, 3 false spot events
   in 4.1 min (the model check rejected all). Record in PROGRESS.md. Low priority vs lanes and signs: the geometry
   needs straight aisle drives, which these clips mostly aren't.
3. **V2-M6 sign detector merge** when `scripts/signs/STATUS.md` in the signs worktree says ready. Merge `v2-signs`
   (it only appends `SignsConfig` + `signs:` and adds files; expect a small conflict at the end of `AppConfig`).
   Then in `app._start_signs` replace `getattr(self.cfg, "signs", None)` with `self.cfg.signs`, run the app on a dev
   clip and check "signs: on" in the status line, the sheet's sign rows, the fps with it on.
4. **V2-M8 integration**: `scripts/bench_app.py` with everything on (only when no training uses the GPU: check
   `nvidia-smi`; the sign session's training made all GPU timings this session pessimistic), the full test sheet
   (`eval/run_sheet.py --previous runs/20261010-142635_sheet/results.csv`), screenshots of the overlay page for the
   README, PROGRESS.md, push. Then a short demo plan for the user (replays of dev clips through the V2 page; the live
   drive and our own footage when the user can record).

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
