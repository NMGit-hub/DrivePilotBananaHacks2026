# V2 build plan

Everything tagged V2 in PLAN.md, plus the new "find me a parking spot" feature, on top of V1 without changing V1's
behavior. Approved by the user on 2026-10-10 without V1's live drive (deferred, not dropped: HANDOFF.md). Each
milestone ends like V1's: its "done when" test, a strict self-review plus an independent subagent review, fixes
with regression tests, a full test run, a commit and a push.

**No live drive is possible for now**, so every feature gets a test that runs at home. The test inputs:
- **Synthetic scenes** in pytest: pinhole-camera fixtures with box jitter and dropouts, like
  `tests/test_intersection_safety.py` (parking rows, pedestrians stepping out, lane geometry).
- **BDD100K clips with their GPS** (HANDOFF.md 5A): `bdd100k_info.zip` gives each clip a 1 Hz GPX (speed, heading).
  Clips by scene: the public scene labels (`dronefreak/BDD100K-Scenario-Classification` on Hugging Face, no login)
  tag **426 clips "parking lot"** (377 train, 49 val); single videos come by range request from the mirror
  `scripts/fetch_dev_clips.py` already uses.
- **Simulated GPS along real Mapbox routes** (HANDOFF.md 5B): `scripts/route_to_gpx.py` drives a saved route at
  realistic speeds (stops, slowdowns for turns, GPS noise, an optional wrong turn); any clip on loop plays as the
  camera.
- **The test sheet** (`eval/run_sheet.py`) with `ALERT@<t>` rows for spoken alerts, and new rows per feature.
- Later, when the user can drive: our own recorded drives replace BDD100K for tuning and the demo.

**Runtime shape (unchanged, plus):** a lanes thread (YOLOPv2, 10-15 fps on the latest frame), an optional sign
detector (a few fps), and a per-frame overlay packet. GPU budget to keep: perception ≥ 25 fps and every PLAN.md
latency target with all of it running (`scripts/bench_app.py`).

## V2 feature coverage

| Feature | You say / it says | Built in | Tested without a live drive |
| --- | --- | --- | --- |
| **Find me a parking spot** (new) | "help me find a parking spot" → "Okay, I'll look for an open spot." … "Open spot on your left, two cars ahead." / "A car is pulling out on your right." | V2-M1 | Synthetic lot scenes; hand-labeled BDD100K parking-lot clips (recall, false alerts per minute, alert time); sheet `ALERT@` rows |
| Visualization (all overlays) | (on screen) car badges and timers, signal labels, nav banner, roundabout diagram, lane fill, drift line, sign cards, pedestrian outlines, parking spot outline | V2-M2 (+ each feature's milestone) | Packet unit tests; browser check on dev clips (overlay pixels on the right objects, frame and overlay from the same message, page fps); screenshots |
| Pedestrian and cyclist alerts | "Pedestrian stepping into the road on your right." | V2-M3 | Synthetic crossing / sidewalk / standing tracks; dev clips with crossings; 0 false alerts on all clips' sidewalk traffic |
| Memory: what was that | "what was the sign we just passed?" → answer + thumbnail | V2-M4 | Router table; sheet rows 3-15 s after passing a sign on dev clips |
| Answer highlight | (on screen) the car/sign the answer means pulses 2 s | V2-M4 | Unit tests (tag never spoken); dev frames with labeled targets: the right object highlighted |
| YOLOPv2 lanes (lane count/position per frame) | "which lane for 520 east?" uses it instead of the model's count | V2-M5 | Hand-labeled lane index on highway/multi-lane frames; navigation lane answers with simulated GPS |
| Lane drift warning | "Careful, you're drifting right." | V2-M5 | Synthetic lane tracks; BDD highway clips with GPS speed (lane changes and straight driving must stay quiet); a "virtual drift" clip (a straight clip warped with a slow sideways shift) |
| Fine-tuned US sign detector (Best Fine-tune) | "Speed limit is now 35.", school zone alerts; sign answers read better | V2-M6 (**needs a go/no-go**) | Held-out mAP; hand-labeled speed-limit signs on BDD clips (no wrong limit ever announced); the sheet's sign rows (2/6 today) |
| RF-DETR-S vs YOLO26s (PLAN tech stack) | (none) | V2-M7, stretch | Sheet + benchmark side by side |

---

## V2-M0: test inputs without a drive (size: S)
- `scripts/route_to_gpx.py` (HANDOFF 5B): saved Mapbox route → GPX at 1 fix/s, speeds from the step durations,
  slowdowns before turns, stops at some intersections, a few metres of noise, `--wrong-turn <step>`.
- `scripts/bdd_info_to_gpx.py` (HANDOFF 5A, when the zip is in `data/bdd100k/`): read one real entry first, learn
  the fields, then clip id → GPX starting at the video's first frame; GPX for every dev clip.
- `scripts/find_bdd_clips.py`: clip ids by scene ("parking lot", "highway", …) from the public scene labels; the
  fetcher learns the train split; fetch ~25 parking-lot clips and ~10 highway clips.
- The sheet gets a `gpx` column, so navigation rows run with GPS.

**Done when:** pytest covers both converters on small fixtures; the app replays a simulated drive to Redmond Town
Center (`--video <clip> --gpx … loop`) and an automated check confirms the lane reminders (~1 mi, ~0.25 mi), turn
and arrival alerts fire within 10% of their distances and a wrong turn reroutes; dev clips play with their real
GPS; the parking-lot and highway clips are on disk with their GPX.

## V2-M1: find me a parking spot (size: L; built after the user answers the questions below)
- **Voice:** a PARKING watch in `voice/router.py` ("help me find a parking spot", "find me a spot", "look for
  parking"; "stop looking" / "never mind" cancel), answered locally, like "tell me when it's green". It ends when
  we park (stopped in a spot), on cancel, or after a timeout.
- **Where the cars are:** each car's ground contact point (box bottom) through a pinhole model (camera height and
  field of view from config, horizon from the vanishing point) gives its distance ahead and to the side.
  `state/vehicle_motion.py` can't be used while we drive (it cancels our motion only for the background, not for
  cars right beside us), so parked cars are found differently: in a lot most cars are parked, so the parked cars
  themselves are the landmarks; our own motion is their common displacement (cross-checked with GPS speed), and
  a car that moves against them is moving.
- **A spot:** two parked neighbors in the same row with a gap of at least one stall between them (~2.3 m between
  their sides for lot stalls; ~6 m along a curb, if street parking is in), nothing detected in it (car, person,
  cart), within range, seen for ≥ 0.7 s while we approach (so a missed detection can't fake one).
  **A car pulling out:** a car parked ≥ 3 s starts moving out of its row.
- **Model check** of each candidate (one structured-JSON call like `answers/lanes.py`, the spot outlined on the
  frame): open or not, and any restriction (handicap, EV, reserved, compact, fire lane, driveway, hydrant,
  no-parking sign). What gets skipped depends on the user's answers. Then the alert, once per spot.
- **Alternative to measure, not assume:** a periodic model scan (every ~2 s while the watch is on) instead of, or
  besides, the geometry. The variant that scores best on the labeled clips ships.
- **Overlay** (with V2-M2): the spot outlined in green.

**Done when:** synthetic lot scenes pass: one alert per open spot, on the right side and count; none for missed
detections, row ends, an aisle crossing, a person in a gap or a moving car; "pulling out" fires once. On the
hand-labeled BDD100K parking-lot clips: **≥ 70% of clearly open, unrestricted spots announced before we reach
them, ≤ 1 false "open spot" per 10 minutes of lot driving** (targets to confirm once labeled), alerts within 2 s of
the spot becoming visible; 0 parking alerts without a watch on any clip; new sheet rows with `ALERT@` pass.
**Caveat:** BDD100K's dashcam lens is narrower than our 114° ultrawide, so side spots appear later there; our own
lot footage retunes it later.

## V2-M2: the overlay page (size: M)
- The backend sends each frame as JPEG **and** its overlay packet in one WebSocket message (frame and overlay can't
  drift apart), replacing the V1 MJPEG stream; the page draws the video and a second canvas on top. Lane overlays
  are smoothed between their 10-15 fps updates. PLAN.md's layout (top banner, corner badges, side panel, bottom
  status) and color meanings (green / red / amber / white).
- Overlays from existing state first: 4-way badges (1st, 2nd), timers since stopped and the YOU badge turning green;
  signal state labels; nav banner with the next step; speed; roundabout mini-diagram with the exit; pedestrians and
  cyclists in amber. Each later milestone adds its own (parking spot, lanes, drift line, sign cards, highlight,
  memory thumbnail). V1's plain page stays available (`ui.overlays: false`).

**Done when:** pytest covers the packet builder (scaling, ids, colors by meaning); in the browser on dev clips the
overlay pixels sit on the tracked objects (checked at known boxes), the page draws ≥ 20 fps, frame-to-screen
latency is measured, and perception fps doesn't drop (`scripts/bench_pipeline.py`); screenshots of each overlay.

## V2-M3: pedestrian and cyclist alerts (size: S-M)
- Person and bicycle are already detected and tracked. A path zone ahead (our lane out to a speed-dependent
  distance: from the camera geometry at first, YOLOPv2's drivable area once V2-M5 lands). Alert when a person or
  cyclist enters it, or is about to (from their motion), while we move or start moving; once per track; safety
  priority (interrupts answers; kept in quiet mode). People crossing in front while we wait at a red light are
  normal: no alert unless we start moving with someone in the path.

**Done when:** synthetic tracks: crossing → one alert; sidewalk walking, standing at the curb, crossing while we
wait → none; dev clips: the crossings in `red_light_construction_pedestrian` and the SF stop-sign clip (person with
a dog) alert only if we move while they're in the path, and 0 false alerts on all dev clips; alert < 0.3 s from the
detection frame.

## V2-M4: memory and answer highlight (size: M)
- **Memory:** frames at 1 fps for 60 s plus a crop of every sign seen (stop signs and lights now; every sign once
  the V2-M6 detector exists). A MEMORY intent ("what was that sign?", "what did we just pass?") sends the recent
  frames and crops to the vision model; the page shows a thumbnail of the frame it answered from.
- **Highlight:** local answers already know their cars' track ids. For model answers, the latest frame goes with
  numbered boxes and the model ends with a hidden tag (`[ref 12]`) that is stripped before speech; the page pulses
  that box for 2 s.

**Done when:** router table rows pass; sheet rows asked 3-15 s after passing a sign on dev clips are judged right;
the tag is never spoken (tests); on dev frames with labeled targets the right object is highlighted in ≥ 80%; model
answer latency is unchanged.

## V2-M5: YOLOPv2 lanes and drift warning (size: L)
- YOLOPv2 (MIT; `yolopv2.pt`, 640x640; drivable area + lane lines) on the GPU at 10-15 fps on its own thread: FP16
  TorchScript first, TensorRT if that is too slow. Undistortion for the phone's ultrawide (a one-time checkerboard
  calibration, `scripts/calibrate_camera.py`) or the center-crop fallback; BDD100K needs neither.
- Lane lines → top-down view (homography from the vanishing point and camera height) → our lane's boundaries,
  lane width check, our offset from the lane center. **Lane count and our lane index per frame**, used by M6's lane
  guidance instead of the vision model's count (the model stays the fallback when lanes are unclear).
- **Drift:** slowly closing in on a line (not the fast sideways move of a lane change, which we can't tell from a
  signal), above 25 mph by GPS speed; once per drift. Overlays: our lane, the line turning red, the edge pulse,
  the green target lane.

**Done when:** lanes ≥ 10 fps with perception still ≥ 25 fps; lane index right on ≥ 85% of hand-labeled frames
(highway and multi-lane dev/BDD clips); synthetic lane tracks: drift → one alert, lane change and straight → none;
BDD highway clips with GPS: 0 drift alerts on straight driving and lane changes; the virtual-drift clip alerts
within 1 s of the expected time.

## V2-M6: fine-tuned US sign detector (size: L; only with the user's go-ahead)
- Data: the LISA US sign dataset (47 classes incl. speed limits and school signs; academic license) and/or the
  Mapillary Traffic Sign Dataset's US classes (needs a free Mapillary account). Fine-tune YOLO26s on the laptop GPU
  (hours; run overnight, not during benchmarks), export to TensorRT, run a few fps.
- "Speed limit is now 35." once per new limit (two agreeing frames, high confidence: a wrong limit is worse than
  none); school-zone alert; sign crops for memory; sign cards and the speed-limit badge; the detector's reading
  added to the vision prompt for sign questions.

**Done when:** held-out mAP50 ≥ 0.8 on speed-limit classes; on hand-labeled BDD clips ≥ 80% of clearly readable
speed limits announced and 0 wrong values; the sheet's sign rows improve on 2/6.

## V2-M7: RF-DETR-S vs YOLO26s (stretch; cut first)
Detection swapped by config; the test sheet and the benchmark run with each; keep the better one.

## V2-M8: integration, sheet, demo readiness (size: M)
Everything on at once within the GPU budget (`scripts/bench_app.py` with V2 on: every latency target still met);
the sheet grows to 40-50 rows (parking, pedestrian, memory and navigation rows with simulated GPS); README and
PROGRESS updated; OBS-ready page. Then V1's deferred live drive and our own footage when the user can drive.

---

## Order and cut line
Proposed order: **V2-M0 → V2-M1 (parking) → V2-M2 (overlays) → V2-M3 (pedestrians) → V2-M4 (memory, highlight)
→ V2-M5 (lanes, drift) → V2-M6 (signs, if approved) → V2-M7 → V2-M8.** The parking feature goes first because it
is the one new feature the user asked for, and the overlay page next because every on-screen item depends on it.
If time runs short, cut from the bottom (PLAN.md's fallback rule): RF-DETR, then the sign fine-tune, then drift;
a V2 feature only goes in the demo if it works.

## Open questions (asked 2026-10-10, before building)
Parking (HANDOFF.md section 4 + this plan): lots only or street parking too; legality checks; range; after the
first spot; side preference; when it counts as found / the watch ends; phrasing. Also: the sign fine-tune go/no-go
(PLAN.md open decision) and its data access; the submission deadline time on Monday (sets the cut line).
