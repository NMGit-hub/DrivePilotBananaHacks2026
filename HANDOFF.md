# Handoff: DrivePilot, end of the V1 build (2026-10-10)

For the next Claude Code session. Read this first, then CLAUDE.md (standing rules), PROGRESS.md (detailed status,
measurements, known issues), PLAN.md (product plan) and BUILD_PLAN.md (V1 milestones).

## 1. Decisions from the user (2026-10-10)

- **No live drive for now.** The user can't drive or record a drive at the moment. PLAN.md's V1 "done" test (the
  same build working once on a live drive) is **deferred**, not dropped: do it when the user can.
- **Go ahead with V2 without the live drive.** The user explicitly approved starting V2 (CLAUDE.md's scope line now
  says so). Everything tagged V2 in PLAN.md is in scope, plus the new parking feature below.
- **New V2 feature: "find me a parking spot".** This is the feature PLAN.md line 41 was waiting for ("DO NOT
  START UNTIL I TELL YOU ABOUT THE NEW PARKING FEATURE"). In the user's words: *ask DrivePilot to help find a
  parking spot (e.g. in a busy lot), and it tells you when it has found one.* See section 4 for a starting design;
  agree the details with the user before building (it's a genuine product decision).
- The user needs a way to get recorded drives with GPS tracks **without leaving the house**: section 5.

## 2. Where V1 stands

All V1 milestones M0-M8 are built, reviewed (an independent subagent review per milestone, every finding fixed
with a regression test) and pushed to `origin/main`. 786 tests pass; ruff and mypy are clean.

| Area | State |
| --- | --- |
| Video in | `video.source`: phone camera over USB (scrcpy, OnePlus 12 ultrawide, hardware decoding) or a file at real time; checked live today at 30 fps |
| Perception | YOLO26s TensorRT FP16 + ByteTrack + optical-flow ego motion on the GPU |
| World state | 4-way-stop queue, traffic lights + "tell me when it's green", lead car, pedestrians |
| Answers | regex router; local right-of-way answers (< 1 ms); Claude Haiku vision answers with safety nets (never vouches for traffic, fixed school-bus rule); navigation (Mapbox) |
| Voice | "Hey DrivePilot" (openWakeWord, threshold 0.15), Silero VAD end of question, faster-whisper small.en on CUDA, Piper speech, alert clips |
| App | `python -m app --video <clip>` / `--camera`; page at http://localhost:8000 (video + Q/A text, space/arrows/Q); phone GPS page at /gps; run folders in `runs/` |
| Test sheet | `eval/run_sheet.py` on `eval/test_sheet.csv` (29 rows): 23 right, 3 partly, 3 wrong (sign reading) |
| Latency | all PLAN targets met at full speed (local 0.83 s, model 1.71 s, alerts 41 ms, 26.8 fps): `scripts/bench_app.py --silent` |

**Not done (needs the user):** the live drive; more real voices for the wake word (one speaker so far, 14/14
heard once woken); recorded drives with GPX, so the sheet has navigation, 4-way-stop (with other cars) and
roundabout rows.

## 3. Open problems to fix first

> Update 2026-10-10 (next session): items 1 and 2 are done. Item 1 is fixed, reviewed and measured, and the
> machine was not throttled (see PROGRESS.md, "V1 polish"). Item 3 is still open; the sign fine-tune (V2) targets it.

1. **Wake word fires early, then the question ends at the pause after "Hey DrivePilot".** Seen on synthetic
   voices (`scripts/voice_eval.py --limit 40`: e.g. "Stop watching the light" woke, transcript empty) after the
   threshold went from 0.25 to 0.15. A lower threshold fires earlier inside the phrase, so the end of "...pilot"
   lasts longer than the 0.4 s `voice.endpoint.wake_tail_s` window; it then counts as the question, and the pause
   after it ends the question with only the wake phrase heard. Suggested fix (more robust than tuning the
   window): in `voice/pipeline.py`, when the final transcript is empty after `strip_wake_fragment` (only the wake
   phrase was said so far), don't drop it: restart the endpointer and keep listening for the question (up to
   `no_speech_timeout_s`). Add a test, then check with `scripts/voice_eval.py --limit 40` (target: no "W ... None"
   rows except the unintelligible Reza voice) and `scripts/wakeword/check_real.py --evaluate data/wakeword/real`
   (must stay 14/14).
2. **CPU throttling.** At 10:20 the CPU was stuck at ~0.75 GHz under load (31% of rated speed) although on the
   charger; perception fell to 7 fps and answers to 3-5 s. Earlier the GPU was throttled the same way until the
   user set Best performance with the charger in. Before any timing measurement, check with
   `Get-Counter '\Processor Information(_Total)\% Processor Performance'` (should go well above 100% under load)
   and `nvidia-smi` (P0, > 2 GHz under load); if throttled, ask the user (charger, heat, restart).
3. Sign reading by Claude Haiku is the weakest sheet category (2/6); lane counting by the model is unreliable.

## 4. The parking feature: a starting design (confirm with the user)

**What the user said:** ask DrivePilot to help find a parking spot (e.g. in a busy lot), and it tells you when it
has found one.

**A design that fits the current architecture (it mirrors "tell me when it's green"):**
- Voice: "Hey DrivePilot, help me find a parking spot" / "find me a spot" / "look for parking" starts a
  **parking watch**; "stop looking" / "never mind" cancels it; it also ends when we park (stopped for a while
  with the spot ahead) or after a timeout. Router: a new WATCH intent (`voice/router.py`), answered locally
  ("Okay, I'll look for an open spot.").
- Seeing a spot (perception + world state, per frame, local and fast):
  - **Lot spaces:** a gap between two parked cars (side-by-side boxes in a row, roughly equal size and spacing)
    with no car in it, on the left or right, within ~2-4 car lengths; parked cars are the tracker's "stopped"
    vehicles that don't move while we drive past (`state/vehicle_motion.py` already compensates for our motion).
  - **A car leaving:** a parked car that starts moving out (reverse lights, motion out of its row) = "a spot is
    opening up on your left".
  - **Street parking:** a long gap along the curb between parked cars.
  - Candidate spots are confirmed over several frames (no flicker), then the vision model can double-check one
    frame for things YOLO can't see: painted lines, handicap/EV/no-parking signs, a hydrant or driveway
    ("Is this an open, legal parking space?", structured JSON, like `answers/lanes.py`).
- Alert: once per spot, e.g. "Open spot on your left, two cars ahead." / "A car is pulling out on your right."
  (Utterance with priority REQUESTED, like the light alert; navigation-style phrasing).
- V2 screen: the spot outlined on the overlay page.
- **Open questions for the user:** lots only, or street parking too? Should it check signs/legality, or just
  "open"? How far ahead/aside should it look? Keep looking after the first spot, or stop? Any preference for
  sides (driver's side first)? What counts as "found" when you're already moving into it?
- Testing without driving: synthetic scenarios in tests (pinhole-camera fixtures like
  `tests/test_intersection_safety.py`), dev clips with parked cars (`school_bus_stop_arm_parking`,
  `stopped_behind_school_bus`), and public parking-lot dashcam footage (section 5); add sheet rows with
  `ALERT@<t>` expectations.

## 5. Recorded drives and GPS tracks without leaving the house

The app plays any video file with an optional GPX track: `python -m app --video <file> --gpx <track.gpx>` (GPX time
at video t=0 can be shifted with `--set video.file.gpx_offset_s=<seconds>`). Three ways to get footage + GPS from
home, best first:

### A. Real dashcam clips that come with GPS: BDD100K (recommended)
> Update 2026-10-10: **done by the user.** BDD100K has moved: the data is now at https://dl.cv.ethz.ch/bdd100k/data/
> (not bdd-data.berkeley.edu as in the steps below). `data/bdd100k/bdd100k_info.zip` is 4.09 GB (not ~11 GB):
> 80,000 JSON files under `bdd100k/info/100k/{train,val}/<clip id>.json` (70,000 train, 10,000 val; no test
> split), 47.7 GB uncompressed, so never extract it all. All 10 dev clips are in `val` (CRC-checked). Steps 6-8
> (the "next Claude session" part) are not done yet: they wait for the user's go.

BDD100K (Berkeley DeepDrive) has 100,000 40-second US dashcam clips, each with a GPS/IMU "info" file (1 fps
location, speed and heading). The dev clips already in `data/videos/dev/` come from it, but from a mirror without the
info files.
**Step by step for the user (about 20 minutes plus the download):**
1. Make sure there's ~25 GB free on C: (the GPS download is a single ~11 GB zip; leave room to work).
2. In Chrome, open **https://bdd-data.berkeley.edu/** and click **Download** (or **Register / Sign up**). Create a
   free account with your email, confirm the email, and log in. You'll be asked to accept the BDD100K license
   (research / non-commercial use): accept it.
3. On the download page, find the **Info** item: "Info" / "GPS/IMU" / `bdd100k_info.zip`, about **11 GB**. Download
   only that one. You do **not** need the big video zips: Claude can fetch just the clips it needs.
   - If the official site doesn't load or the download fails, the same files are mirrored by HyperAI:
     https://hyper.ai/en/datasets/11926 (look for `bdd100k_info.zip`; it may also need a free account).
4. In File Explorer, create the folder `C:\ClaudeProjects\DrivePilot_BananaHacks2026\data\bdd100k\` and move the
   downloaded `bdd100k_info.zip` into it. **Don't unzip it** (it holds ~100,000 small files; Claude will pull out
   only the clips it needs).
5. Tell Claude: "bdd100k_info.zip is in data/bdd100k/". That's all; the rest is Claude's job (below).

**Then the next Claude session does:**
6. Open the zip with Python's `zipfile` (no full extraction) and look at one real JSON first, e.g. the entry for
   clip `b97b9ee8-612c36ef` (`data/videos/dev/red_light_turns_green_sf__b97b9ee8-612c36ef.mov`), to learn the
   actual field names (expected: a list of locations with latitude, longitude, speed, course and a timestamp in
   ms; verify, don't assume) and the split folders (train/val). Our dev clips are from the `val` split.
7. Write `scripts/bdd_info_to_gpx.py` (zip + clip id -> GPX whose first point is the video's start, 1 fix/s; tests
   with a small fixture), convert the dev clips, and run e.g.
   `.venv/Scripts/python -m app --video data/videos/dev/red_light_turns_green_sf__b97b9ee8-612c36ef.mov --gpx data/gps/b97b9ee8-612c36ef.gpx`.
8. For new scenes (busy parking lots for the parking feature, 4-way stops with other cars, roundabouts), pick clip
   ids from the info/label files or by scanning, and download single videos by HTTP range requests from the mirror
   `scripts/fetch_dev_clips.py` already uses (`linxxx3/bdd100k_videos` on Hugging Face), then convert their GPS.
Limits: 40 s per clip (navigation questions over a whole route need option B), US cities (mostly the Bay Area and
New York), 720p, non-commercial license (fine for the hackathon, not for the submission video, which should use
the team's own footage).

### B. A realistic GPS drive along a real route, with any video (for navigation)
Navigation only needs GPS; the video can be anything. This tests "I'm going to ...", lane reminders at 1 mile and
0.25 mile, roundabout exits, missed turns and rerouting end to end in the app, at home:
1. Fetch a real route: `.venv/Scripts/python scripts/fetch_route.py --from 47.6158,-122.2037 --to "Redmond Town Center"`
   (saves to `data/routes/`; six are already saved, incl. a Carmel roundabout route: `carmel_library.json`).
2. Next Claude session: write `scripts/route_to_gpx.py`: drive along the route line at realistic speeds (from
   Mapbox's step durations; slow down for turns, stop for a few seconds at some intersections, 1 fix per second,
   a few metres of GPS noise), optionally with a deliberate wrong turn to test rerouting.
3. Run the app with a long dev clip on loop as the "camera" and the GPX:
   `.venv/Scripts/python -m app --video <clip> --gpx data/gps/redmond.gpx --set video.file.loop=true`,
   then say or type "I'm going to Redmond Town Center" at the start. The guidance alerts should arrive at the
   right distances (tests/test_nav*.py already check this offline with a synthetic route).

### C. Public long dashcam drives with GPS (for longer real footage)
- **comma2k19** (comma.ai): 33 hours of California highway driving (280), 1-minute segments with GNSS; on Hugging
  Face / Academic Torrents; ~100 GB in total, download a few segments. Good for highway lane questions.
- **KITTI raw** (Karlsruhe, Germany): short city/highway sequences with GPS (OXTS); German signs and roads.
Both need a converter to GPX like option A. Prefer A for US scenes.

What doesn't work from home: recording with the phone itself (the GPS doesn't move and the camera sees a room);
pointing the phone at a screen playing a dashcam video tests the live camera path, but not GPS.

## 6. Working notes for the next session

- Run everything with `.venv/Scripts/python`; tests: `.venv/Scripts/python -m pytest` (~2-3 min); lint
  `ruff check .`, `ruff format --check .`, `mypy .`.
- **Don't edit code with Bash heredocs**: in this Windows environment they turn `\b`, `\n`, `\r` into control
  characters (it corrupted README.md and several patches). Use Edit/Write, or write a patch script to the
  scratchpad and run it.
- The Anthropic SDK is 1.13: `messages.stream()` has no `temperature` argument (it goes in `extra_body`); see the
  claude-api skill before touching `answers/vision.py`.
- Servers: `.claude/launch.json` has `drivepilot-camera` and `drivepilot-replay` for the preview tool. The phone
  needs `adb reverse tcp:8000 tcp:8000` for the GPS page; adb is in the scrcpy winget package folder
  (`%LOCALAPPDATA%\Microsoft\WinGet\Packages\Genymobile.scrcpy_*\scrcpy-win64-v4.1\adb.exe`).
- Run folders under `runs/` hold logs, events and summaries of every app run, benchmark and sheet run.
- Per CLAUDE.md: commit after every working piece, push at the end of each milestone, independent review per
  milestone, keep PROGRESS.md updated, and batch questions for the user into one message.
- The API keys were pasted in chat on 2026-10-09: remind the user to rotate them after the hackathon.
