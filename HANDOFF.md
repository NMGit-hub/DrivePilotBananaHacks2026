# Handoff: DrivePilot, V2 in progress (2026-10-10, ~13:45 PDT)

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

## 2. Next steps (in order)
1. **Run the parking feature on the labeled clips** (the big unknown: real YOLO boxes at 384x640, frame-edge cars,
   the horizon/speed estimates). Write `scripts/eval_parking.py`: for each usable clip in `eval/parking_labels.yaml`,
   run perception + `WorldState` in lock-step (`--set video.file.hfov_deg=60`; BDD dashcams ~60°, the app default is
   the phone's 114°), arm the watch at t=0 (or only use the tracker's `spot` events), and score: recall of clear open
   spots (a spot event on the right side between ~t_visible−1 s and t_pass), false alarms (spot events matching no
   labeled spot; ambiguous ones neither count nor fail; any hit on a look-alike is a false alarm), lot events vs
   `in_lot`, parked events vs `parked_at`, pulling-out events. Plan targets: ≥ 70 % recall, ≤ 1 false spot per 10 min
   of lot driving (V2_BUILD_PLAN.md V2-M1). Then tune (`min_visible_depth_m`, `gap_ratio`, `confirm_s`, tolerances)
   and look at failures frame by frame (`scripts/contact_sheet.py`). With the vision check on (Claude Haiku), also
   measure what it rejects/accepts.
2. Spot-check ~6 labeled spots on full-resolution frames before trusting the numbers.
3. Add the `parking:` section to `config.yaml` (values currently only as defaults in `core/config.py`: CLAUDE.md
   wants them in config.yaml), and parking rows with `ALERT@<t>` to the test sheet.
4. V2-M1 strict self-review + independent subagent review, fixes, PROGRESS.md, push.
5. Then V2-M2 overlays → M3 pedestrians → M4 memory/highlight → M5 YOLOPv2 lanes/drift (V2_BUILD_PLAN.md), the sign
   detector merge when `v2-signs` is ready, V2-M8 integration.

## 3. Known gaps of the parking feature (also in PROGRESS.md)
- Never run on real footage yet (step 1). Street mode untested. No overlay yet (V2-M2 draws the spot).
- On a 60° camera a spot is confirmed only ~6 m ahead (the stall must be seen 1 m deep past its neighbour); the
  phone's 114° ultrawide sees deeper earlier. The model check starts earlier to keep the alert timely.
- The lot offer needs the vision model (no model, no offer). The 5-minute timeout runs on media time.

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
