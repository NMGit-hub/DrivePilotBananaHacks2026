# Handoff: DrivePilot, V2 nearly integrated (2026-10-10, ~21:50 PDT)

For the next Claude Code session. Read this first, then CLAUDE.md (standing rules), V2_BUILD_PLAN.md (V2 milestones +
the user's decisions), PROGRESS.md (detailed status, measurements, known issues: kept current) and PLAN.md.

**Deadline: Sunday 2026-10-11, 8:59 pm PDT.** Everything below is committed on `main` and pushed to `origin/main`
(last commit: this handoff). 1252 tests pass; ruff, ruff format, mypy clean.

## 1. State of V2 (details and numbers in PROGRESS.md "V2")
| Milestone | State |
| --- | --- |
| V2-M0 test inputs | done |
| V2-M1 parking spot | built, **below the bar on BDD100K footage** (1/30 lot spots, 0/8 on the crawl clips; the Claude Haiku check rejects every candidate, so nothing false is spoken). Needs the user's own lot footage. Both crawl label halves are in `eval/parking_labels_crawl_{a,b}.yaml`. |
| V2-M2 overlay page | done; now also our lane (lines with a dark halo), the red drift line + edge pulse, the green target lane ("move 1 lane right") |
| V2-M3 pedestrians | done; gated by YOLOPv2's drivable area **beside** the feet, never within 0.9 m of the path's middle (review safety fix); crawl 12 -> 9 warnings, lot 6 -> 5, dev 0 |
| V2-M4 memory + highlight | done |
| V2-M5 YOLOPv2 lanes/drift | **done + reviewed (6 bugs fixed) + pushed.** Lane index 57/58 answered (88 % of labeled frames right); drift: 1 borderline alert in 5.7 min of real highway driving, virtual drifts 13/18 on time (5 late), lane changes 12/12 quiet: below the plan's bars (PROGRESS Known issues). Wide cameras: the lane model sees the center cropped to 70° (untested on phone footage: none exists). |
| V2-M6 sign detector | **merged** (`2b89b2b`; the other session's branch `v2-signs`, models in `models/signs/`, gitignored). App: `signs=on`, 3.8 fps, perception 30 fps with lanes + signs on. **Bug found, not fixed: section 2.1.** |
| V2-M7 RF-DETR | not started (cut) |
| V2-M8 integration | partly: bench with lanes on passed every target (`runs/20261010-212804_bench_app`); the test sheet has 45 rows and runs lanes, signs and the parking watch like the app; **the full sheet run, a bench with signs on, README screenshots are still to do** |

## 2. Next steps (in order)
1. **Sign alerts repeat** (found in the last live check, `runs/signs_live_check.txt` and the newest
   `runs/*red_light_rain_school_zone*` run folder): on the school-zone dev clip "School zone ahead." was spoken 4
   times: twice in the same frame (t=0.25: two school-zone signs) and again at 2.35 s and 12.19 s. `state/signs.py`
   (`SignTracker.update`, `SignAlertsConfig.repeat_s`) is meant to say it once per zone: dedupe per frame and per
   zone (a school zone once until `repeat_s` has passed without one), with a regression test, then re-run the clip
   (`python -m app --video data/videos/dev/red_light_rain_school_zone__b61456bb-5bfb50ee.mov --no-voice --no-audio
   --no-ui --exit-at-end --set video.file.hfov_deg=60`; sign events are `"kind": "sign"` in `events.jsonl`). Also
   check speed-limit alerts on `ego_holdout_long_stop` (a SPEED LIMIT 30 sign at 8.5-10.2 s).
2. **V2-M8, the rest:**
   - `scripts/bench_app.py --silent` again with signs on (GPU must be free: `nvidia-smi --query-compute-apps=pid
     --format=csv`; the sign session is finished, so it should be).
   - The full test sheet, 45 rows: `.venv/Scripts/python eval/run_sheet.py --previous
     runs/20261010-142635_sheet/results.csv` (takes a while; lanes + signs + parking watch run in lock-step; the
     judge is Claude Haiku). Compare with 22/3/4 of 29 before; look at the 16 new rows (lanes, memory, signs,
     parking, road/safety; the notes column says what's in the frame). Record in PROGRESS.md.
   - README screenshots of the V2 page (`.claude/launch.json`: `drivepilot-lanes` is a highway clip,
     `drivepilot-overlays` a dev clip). **The browser pane only composites canvas frames while visible**: pause the
     replay (POST `/api/control {"action":"pause"}`) and call `draw()` in the page before a screenshot; screenshot
     files land in the Claude tool-results folder: copy them into e.g. `docs/screenshots/` and link them from
     README.md.
   - PROGRESS.md + README, push. Then a short demo plan for the user (replays through the V2 page: `drivepilot-lanes`
     for lanes / target lane, the school-zone clip for signs, the SF red-light clip for "tell me when it's green").
3. Only if time remains: the sign session asked the user to delete stray COCO8 runs (`runs/detect/train*` in this
   checkout and `C:\ClaudeProjects\datasets\coco8`): tell the user, don't delete them yourself.

## 3. Things the user should know / decide (batch into one message)
- Parking finds almost nothing on BDD100K footage; it needs a recording of a slow drive straight down lot aisles
  with the phone. The NTNU parking dataset is only available by email request to Mei-Chen Yeh
  (myeh@csie.ntnu.edu.tw): the user's call.
- Drift warnings and lanes are tuned on 60° BDD dashcams only; the phone's 114° view goes through a 70° center crop
  that has never seen real phone footage. A recorded highway drive (with GPS) would let us check both.
- `video.file.hfov_deg: 60` in config.yaml is right for BDD100K replays; set it to null for the user's own phone
  recordings (the phone's 114°).
- Stray COCO8 runs from the sign session (see 2.3) can be deleted by the user.
- Remind the user to rotate the API keys after the hackathon (pasted in chat on 2026-10-09).

## 4. Working notes
- Run everything with `.venv/Scripts/python`; full tests `.venv/Scripts/python -m pytest -q` (~3 min);
  `-m ruff check .`, `-m ruff format --check .`, `-m mypy .` (tests/ is excluded from mypy).
- **Don't edit code with Bash heredocs containing backslashes or quotes that need escaping**: use Edit/Write, or a
  patch script written to the scratchpad and run with the venv (write files with `newline="\n"`).
- The app gets its GPU models injected: `build_app` passes `lane_model_factory(cfg, status)` and
  `sign_model_factory(cfg, status)`; a `DrivePilotApp` built without them (unit tests) runs with lanes and signs off.
- Evals that replay caches: perception `data/cache/perception/<clip stem>.pkl`; lane masks `runs/lanes_cache/*.npz`
  (highway, dev, crawl and 4 lot clips; `scripts/eval_lanes.py cache --clip ...` for more). `eval_lanes.py index /
  drift / virtual`, `eval_pedestrians.py --road`, `eval_parking.py`, `eval_parking_check.py`.
- The V2-M5 review's reproducers were in the previous session's scratchpad, not in the repo; its findings are
  regression tests now (`tests/test_lanes.py` "review findings", `tests/test_pedestrians.py` drivable area).
- Agents: give them the main venv by absolute path; tell long-running ones to save work incrementally (that worked:
  the crawl labeler survived a usage-limit stop and was resumed with SendMessage). Subagents can't write `.md`
  report files (the harness blocks it): ask for the report in their final message.
- Leftover worktrees: `.claude/worktrees/agent-a646a9a46d39b56d8` (pedestrians, merged) and
  `agent-af32ec16370458bc6` (lanes, merged): `git worktree remove -f -f <path>`, then `git branch -d <branch>`.
- The Anthropic SDK is 1.13: `temperature` goes in `extra_body` (handled in `answers/vision.py`).
- Piper synthesis is random: voice benchmark rows vary run to run; compare runs, not rows.
- Per CLAUDE.md: commit after every working piece, push at milestone ends, an independent review per milestone,
  keep PROGRESS.md updated, batch questions for the user into one message.
