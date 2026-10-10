# Handoff: DrivePilot, V2 in progress (2026-10-10, ~16:15 PDT)

For the next Claude Code session. Read this first, then CLAUDE.md (standing rules), V2_BUILD_PLAN.md (the V2
milestones + the user's decisions), PROGRESS.md (detailed status, measurements, known issues), PLAN.md (product
plan) and BUILD_PLAN.md (V1 milestones, done).

**Deadline: Sunday 2026-10-11, 8:59 pm PDT** (the user's answer; not Monday). Cut from the bottom of the V2 order
if time runs short (V2_BUILD_PLAN.md "Order and cut line").

## 1. What happened in the last session
- V1 polish: the machine was not throttled; the early-wake voice bug was fixed, independently reviewed (2 risks + 2
  nits, all fixed) and measured (woke-but-heard-nothing rows 4 → 0; real voices 14/14). Pushed.
- **5A done:** `data/bdd100k/bdd100k_info.zip` (the user downloaded it; BDD100K's new home is
  https://dl.cv.ethz.ch/bdd100k/data/). `scripts/bdd_info_to_gpx.py` + `nav/bdd_info.py` → `data/gps/<id>.gpx` for
  every dev clip; GPX replay aligns on a `video_start` waypoint (checked against hand-labeled stops: positions within
  −0.26/+0.19 s). The vision model now gets our speed whenever there's a GPS fix.
- **V2-M0 done** (merged from an agent worktree, `481ffbe`): `nav/sim_drive.py`, `scripts/route_to_gpx.py`
  (simulated drives along saved Mapbox routes), `eval/nav_check.py` + `scripts/sim_nav_check.py` (11/11 drives
  pass: lane reminders within −4..+1 % of 1 mi / 0.25 mi, roundabouts, arrival, reroute 4.6-5.7 s after a missed
  turn), and **a looping clip's clock now runs on across passes** (a long GPX plays to its end; trade-off: a BDD
  clip looped with its own GPX gets GPS only on the first pass). `scripts/find_bdd_clips.py` finds clips by scene
  (426 "parking lot") and GPS motion; 40 parking-lot clips + GPX are in `data/videos/parking_lot/`, `data/gps/`.
- **V2-M1 part 1 committed (`1cce9f1`): find me a parking spot**, built and unit-tested, **NOT yet run on real
  footage**:
  - `state/parking.py` (`ParkingTracker`, inside `WorldState`; events `parking_candidate|spot|taken|pulling_out|
    lot|parked`, payload `(ParkingEvent, Frame)`): parked cars from box bottoms (pinhole, horizon from car boxes),
    our speed = the median approach rate of the cars beside the aisle (they're parked), rows + stall pitch measured
    from them (lengths scale with the pitch, so a wrong FOV changes no decision), a space between two parked
    neighbours (≥ 1.7 pitches) whose aisle-side face is visible and empty, confirmed only once we can see
    `min_visible_depth_m` (1.0 m) into the middle of the stall past the nearer neighbour (the user's "small car
    hidden behind a bigger one" false positive), with the neighbour predicted from our motion once it leaves a narrow
    camera's frame. Street mode (parallel parking) shares the logic with swapped dimensions: untested.
  - `answers/parking.py` (`ParkingWatch`): the user's rules (V2_BUILD_PLAN.md "Decisions"), a vision-model check per
    space (started at the "candidate" event), the lot offer with a 5 s **plain yes/no reply window** (CLAUDE.md
    exception; `VoicePipeline.expect_reply`, opened after the offer finished playing), phrasing "Open spot on your
    left, two cars ahead, just after the white SUV."
  - Router `RouteKind.PARKING` (find / find_street / stop), `parse_yes_no`; assistant + app wiring.
  - Tests: `tests/test_parking.py` (12 simulated-lot scenes), `tests/test_parking_watch.py` (27),
    reply-window tests in `tests/test_voice.py`. 903 tests pass; ruff, format, mypy clean.
- **Parking labels:** `eval/parking_labels.yaml`, 40 clips (28 usable, 35 clear + 12 ambiguous open spots, 66
  look-alikes), labeled by two subagents; the header lists the best positive/negative clips. Not spot-checked yet.
- **The sign fine-tune (V2-M6) runs in a separate Claude session** started by the user from
  `SIGN_FINETUNE_PROMPT.md`: worktree `C:\ClaudeProjects\DrivePilot_signs`, branch `v2-signs` (pushed to origin).
  Contract: `perception/signs.py` (`SignDetector.detect(frame) -> list[SignDetection]`), weights/engine copied to
  `models/signs/`, status in `scripts/signs/STATUS.md` on that branch, due Sunday 11:00 am PDT. This session's job:
  merge it and wire alerts / memory / overlays / prompts. While it trains, GPU timings here aren't valid.
- PLAN.md was deleted from the working tree by something other than this session (not the user); it was restored
  from git. If it disappears again, say so.

## 2. This session (2026-10-10, 13:30-15:00 PDT): done
- **V2-M1 parking measured on real footage** (`scripts/cache_perception.py`, `scripts/eval_parking.py`): after fixes
  (scale clamp, cut-off boxes/hood, GPS speed, lot hold, strict pull-outs) **recall 1/30 clear spots, 0.7 false /
  10 min**: below the bar on recall, precise. Labels spot-checked (6 frames: real). A Claude Haiku periodic scan was
  measured by a subagent (`eval/parking_scan.py`): 47 % recall at 26.6 false / 10 min: not shipped. Details and
  causes in PROGRESS.md (V2-M1, Known issues). Needs our own lot footage to tune. No `ALERT@` parking sheet rows
  (they would fail).
- **V2-M2 overlay page** (part 1) and **V2-M4 memory + highlight** built, tested, checked live (PROGRESS.md).
- **V2-M6 integration part 1** (`state/signs.py`, `perception/sign_worker.py`, `sign_alerts:` config): sightings,
  "Speed limit is now 35." after two agreeing frames, school zone, context lines; app wiring waits for the merge.
- `config.yaml` has `parking:`, `memory:`, `sign_alerts` defaults, `ui:` overlay keys, `vision.mark_objects`.

## 3. In flight / next steps (in order) (updated 2026-10-10 ~16:15 PDT)
Done since: the independent review of M1/M2/M4 (11 bugs fixed with tests, pushed: `06d7325`); **V2-M3 pedestrians
merged** (`e3b9976`, pushed; its worktree `.claude/worktrees/agent-a646a9a46d39b56d8` is merged and can be removed
with `git worktree remove -f -f ...` once its agent process is gone); the sign thread wired into the app (`b9c1ea9`,
switches on once `v2-signs` brings `signs:` + models); `video.file.hfov_deg: 60` for BDD100K replays (set it to
null for our own phone drives); 32 crawling lot clips fetched to `data/videos/parking_crawl` (+ GPX, perception
cached) at the user's request ("find videos of driving through a parking lot").
1. **Crawl-clip parking labels**: two labeling subagents write `eval/parking_labels_crawl_a.yaml` (done: 7/16
   usable, only 2 clear spots, both the stall we park in) and `..._b.yaml`; then score with
   `scripts/eval_parking.py --labels <file>` (it reads caches by clip id) and record the result in PROGRESS.md.
2. **V2-M5 YOLOPv2 lanes/drift**: the user OK'd the download; `models/lanes/yolopv2.pt` is in place (156,380,200
   bytes). A subagent builds `perception/lanes.py` (LaneWorker), `state/lanes.py` (LaneTracker: lane count/index,
   drift events, overlay polylines), `scripts/eval_lanes.py`, tests, in a worktree; it fetches 10 BDD highway clips.
   Then integrate: the app (world thread offers frames; drift alert = SAFETY? PLAN says safety alerts interrupt),
   overlay (our lane, the drift line red), lane index into `answers/lanes.py` / nav lane guidance instead of the
   model's count when confident, pedestrians' PathZone from the drivable area.
3. **V2-M6 sign detector**: merge `v2-signs` when `scripts/signs/STATUS.md` (in `C:\ClaudeProjects\DrivePilot_signs`)
   says ready (due Sunday 11:00). The app side is ready (`state/signs.py`, `perception/sign_worker.py`, app wiring):
   after the merge, check the `getattr(cfg, "signs")` bridge in `app._start_signs` becomes a direct access, run the
   sheet's sign rows, and measure fps with it on.
4. V2-M8 integration: `scripts/bench_app.py` with everything on (when the GPU is free of training), the full sheet
   again (last: 22/3/4 of 29 before the pedestrian merge), screenshots for the README, PROGRESS, push.

## 4. Working notes
- Run everything with `.venv/Scripts/python`; tests `.venv/Scripts/python -m pytest` (~3 min); `ruff check .`,
  `ruff format --check .`, `mypy .`.
- **Don't edit code with Bash heredocs** (they mangle `\b`, `\n`, `\r`; a long heredoc also failed to parse): use
  Edit/Write, or write a patch script to the scratchpad and run it.
- The Anthropic SDK is 1.13: `temperature` goes in `extra_body` (see the claude-api skill before touching
  `answers/vision.py`).
- Piper synthesis is random: `scripts/voice_eval.py` rows vary run to run; compare runs, not rows.
- Agents in worktrees: their checkout has no `.venv`, `.env` or data: give them the main venv by absolute path and
  data paths explicitly; merge their branch yourself after review. `.claude/worktrees/agent-af5c10b4b87b88fdf` is
  merged and can be removed (`git worktree remove`).
- Servers: `.claude/launch.json` has `drivepilot-camera` and `drivepilot-replay` for the preview tool.
- Per CLAUDE.md: commit after every working piece, push at the end of each milestone, independent review per
  milestone, keep PROGRESS.md updated, batch questions for the user into one message.
- Remind the user to rotate the API keys after the hackathon (they were pasted in chat on 2026-10-09).
