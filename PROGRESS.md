# PROGRESS

Live status of the V1 build. Updated at every milestone. Newest information first within each section.

## Done

### Showcase v2 (2026-10-11 evening, the user's notes on the first reel)
`runs/showcase/v2/DrivePilot_showcase_v2.mp4` (2:29.9; `build_reel.sh` there rebuilds it from the scenes, the
`render_v2*.sh` scripts re-render them): 3 narrated intro slides (`scripts/showcase_intro.py`: what it is, phone +
laptop, here the dashcam; what it does with frames of the scenes), then 7 scenes, titles drawn in, slow stretches cut
(`scripts/showcase_reel.py video@a-b,c-d`):
1. red light + "tell me when it's green" (kept);
2. **lane guidance from navigation**: a route set before the scene (SR 520 E -> the NE 40th St exit), "Am I in the
   right lane?" -> "use the right lane. You're in lane 2 from the left, so move one lane right", the target lane in
   green, the car moves over, "In 500 feet, take the exit";
3. rain (kept);
4. **exit signs tied to navigation**: "Exit 10B, Factoria Blvd, ½ mile" read off the gantry, then the route's "In
   half a mile, take exit 10B toward Factoria Boulevard" with the lane arrows;
5. **roundabout** (two clips joined, the route set on SR 520 before the exit: on the 92nd Ave NE lid every GPS fix
   snaps onto the freeway below): "At the roundabout, take the 4th exit onto 92nd Avenue Northeast", the exit diagram,
   a person crossing outlined in amber (shown, not spoken: we're stopped, by design), "Which exit is mine?" -> "At the
   roundabout now, take the 4th exit...", "Not this one." as we pass an exit;
6. **parking**: the app's own offer ("Looks like a parking lot. Want help finding a spot?"), a plain "yes" (no wake
   word, as the app takes it), then "Can I park here?" at the empty stall -> "an ordinary parking space with no
   wheelchair symbol, blue paint, or posted signs... It appears fine, but check any signs posted nearby";
7. **night, the car ahead keeps braking** (the user's request: W Lake Sammamish Pkwy, 2026-10-10 23:02, found by an
   agent, 3.2 km from home): "Why does the car ahead keep braking?" during its hardest stop -> "I can see a car ahead
   with its brake lights lit... There's no clear hazard visible on the road ahead."
No snow (none in the footage). Every scene is a real run; for several, more than one take was rendered and the best
used (answers vary). Fixed on the way: a lane answer is no longer repeated (and cut into) by the next lane reminder for
the same maneuver; diagonal / U-turn lane arrows were empty boxes in the banner (Segoe UI has no such glyphs: Segoe UI
Symbol now). Not fixed: the parking-spot finder (0/8 on this camera; an agent traced it to several stages at once:
turn gating, rows that rarely form here, empty rows, the hood cutting boxes; no single principled fix, reverted); GPS on
the roundabout lid reroutes after the exit (the scene ends before).

### The user's own dashcam footage: showcase videos, evals, fixes (2026-10-11 afternoon)
The user's Mercedes dashcam clips (D:\Dashcam, HANDOFF.md section 2) are the data for the evals and the submission.
- **Camera:** horizontal FOV **96°** (median of 18 turns, IQR 92-103°; `scripts/dashcam_calibrate.py --turns 20`,
  which now skips a turn when > 10 % of its frames have no usable homography: 2 of 20 skipped). The camera is
  pitched up (horizon ~70 % down the frame): new `lanes.view_center_y` puts the lane model's 70° crop low (0.65);
  centered, the crop was mostly sky and the horizon fell past `horizon_max` (no lanes at all). The masked text bar
  ends the hood's box at 96 % of the frame, so `perception.hood_bottom` (0.97, dashcam 0.93) makes the hood rule
  see it (a box was drawn on our own hood). `scripts/eval_lanes.py cache --hfov` now feeds the lane model the app's
  crop (it fed the whole frame).
  For these clips: `--set video.file.hfov_deg=96 --set lanes.view_center_y=0.65 --set perception.hood_bottom=0.93`
  (the showcase renderer applies them to `dm_` clips by itself).
- **Clips:** 52 picked by an agent from contact sheets (privacy: none within 2.2 km of home, no garages/driveways,
  no school pick-ups; the burned-in GPS bar cropped from everything shown) into `data/videos/own/` with their GPX
  (`scripts/dashcam_gpx.py gpx`): 19 highway (day, rain, night), 13 parking lots, 11 city with lights, showcase
  moments (`data/videos/own/selection.csv`, gitignored).
- **Showcase renderer** (`scripts/render_showcase.py`, `ui/render.py`, `ui/timeline.py`, built by an agent of the
  previous session, reviewed twice here and fixed): plays a clip in lock-step through the app's real pieces
  (perception, world, lanes, signs, watches, memory, navigation, the Assistant with Claude), draws the V2 page's
  overlays, HUD and caption cards into each frame, and speaks every line with Piper (the driver's "Hey Drive
  Pilot, ..." in a second voice) placed like `voice/speaker.py` would play it. Review fixes: an alert that makes
  answers stale also drops answers still on their way (a stale "wait" right after "Light's green"), thumbnails
  cropped above the bar, the bar's row scaled to the frame size, guidance waits only for a new route's answer, the
  parking watch's pacing re-based after a question, the source stopped once at a time. `scripts/showcase_reel.py`
  joins videos with title cards. **Rendered** (`runs/showcase/final/`, gitignored; frames of each looked at):
  `DrivePilot_showcase.mp4` (3 min 29 s, 7 scenes): red light + "tell me when it's green" (the alert at 33.04 s,
  the light changed at 33.0 s by eye), highway lanes + "which lane am I in?" (second, right), rain + "what did that
  yellow sign say?" (the 35 mph curve sign), exit signs read off a gantry, a roundabout's YIELD / ONE WAY /
  crossing signs, a parking stall + "can I park here?", night highway + "how many lanes?". Model answers vary from
  run to run: for the roundabout, parking and highway scenes 2 takes were rendered and the better one used (every
  take is a real run; the first roundabout take called YIELD "too blurry", the first lane take said "third lane"
  before the yellow-edge fix; the rain scene was rendered again for the thumbnail fix, same answer). Seen and not used: at a 6-head intersection (`dm_20260926_185554`) "Light's green"
  never came because the right-turn heads stayed red (the safety rule working as designed); at night
  (`dm_20260929_195410`) a blooming green LED read YELLOW (Known issues); the parking finder found no spot.
- **Measured on the user's footage:**

  | What | Result | (BDD100K, before) |
  | --- | --- | --- |
  | Drift warning, 19 highway clips, 37.2 min >= 25 mph | 8 alerts: 5 real (2 clear, 3 borderline), **3 false = 4.8 false / h** (a ramp narrowing, an exit-lane split at night, a merge) | 1 borderline in 5.7 min |
  | Lane changes detected (5 clips checked) | 13 of 15 real, none missed (2 false: a lane added / split on the right) | |
  | Lane index, 256 hand-labeled frames of 18 clips | **188/221 right (85 %)**, answers 86 %; was 121/200 (60 %) | 52/57 (91 %), was 57/58 |
  | Parking spots, 13 hand-labeled lots (12 usable, 8 clear targets) | **0/8 found**, 3 false spots, 7 false "parked" | recall 1/30, 0.7 false / 10 min (bar: >= 70 %) |
  | Parking answers (25 parking rows of the 3 test sheets) | 20 right / 4 partly / 1 wrong; was 15 / 10 / 0 | |

  Drift and lane changes judged by an agent from time-slice images (`runs/lane_drift_own_verdicts.txt`); labels
  `eval/lane_labels_own.yaml`, `eval/parking_labels_own.yaml` (agents, not spot-checked by a second pass).
- **Lane index fix (the yellow left edge):** 78 of the 79 misses were one lane too high: the wide paved left shoulder
  (yellow edge line, shoulder, barrier) passed as a lane by width and drivable area. The lane model now marks which
  line pixels are yellow paint (`perception.lanes.yellow_lines`), each line gets its yellow share, and counting lanes
  to the left stops at the nearest yellow line (US: yellow separates opposing traffic or marks a one-way roadway's
  left edge), only while another line of the frame is white (orange street lights make every line yellow). Own
  footage 60 % -> 85 %; BDD100K 98 % -> 91 %: 4 of its 5 misses are one NYC clip where the tracker places our lane
  one to the left (its yellow edge reads as our own line; the old count got it right by counting the shoulder).
  Both sets together: 69 % -> 86 %.
- **Parking answers:** "Can I park here?" at an ordinary stall said "I can't see any markings that decide this...".
  The prompt now says an ordinary stall or curb with no restriction in view is normally fine: say so first, then to
  check posted signs. On the parking rows (sets used before, so not blind): +5 right, 1 new wrong (a painted NO
  PARKING box it didn't read, now called an ordinary stall; before it hedged).
- **Independent review** of the afternoon's commits (4 bugs, all fixed with tests; the yellow-edge logic, crop and hood
  plumbing checked correct): (1) the reel drifted from its audio when a clip's frame rate differed from the first
  one's: now refused, and each clip's audio taken contiguously by the writer's own sample counts (±1 sample at
  NTSC rates); (2) the safety nets undid a plain yes to common parking questions ("is it okay to park here?" counted
  as a go question; "You're fine here." as permission) and said the traffic disclaimer instead: a parking question
  is not a go question unless it is about moving off ("pull out of this parking spot", "back out": those stay
  strict, and "can I back out / back up / reverse" is now a go question, which it wasn't); "you're fine/good" is
  allowed in a parking answer ("good to go", "go ahead" still caught); (3) answer thumbnails from memory frames
  (downscaled) kept the masked bar's rows: cropped by share now; (4) the lane-mask cache records its crop, and the
  replays refuse masks that don't match `--hfov` (a forgotten `cache --hfov` was silently wrong); `virtual` refuses
  a cropped camera; `eval_pedestrians.py` passes the crop to the drivable area.
- Tests: all pass (new: the yellow edge, the hood threshold, the low lane crop, the reel, the renderer's stale
  answers, the parking nets, the cache round trip); ruff, mypy clean.

### Wake modes, unrequested alerts, destination box (2026-10-11, user request; branch `voice-alerts-destination`)
- **Wake modes** (`voice.wake_mode`, `--wake`, Settings on the page while running): `hey_drivepilot` (default,
  unchanged), `hey`, `none`. In the two open modes a second Silero VAD watches the idle microphone; 0.15 s of
  speech starts a question (0.4 s pre-roll, no wake-phrase tail), and Whisper's transcript decides whether it was
  meant for us. `router.strip_hey`: starts with "hey"; a transcript that is only a lone "hey" (or Whisper's "Say.",
  "Hi." for it) opens the listening window like the wake word, with the blip; an empty transcript (noise) opens
  nothing. `router.meant_for_us`: commands, or anything phrased as a question/request; in "hey" mode after the
  "hey" (and after a lone "hey"), so "Hey, I'm so hungry" isn't answered; with no wake word a destination must be an
  instruction ("take me to ...", not "we're going to grandma's") and bare "never mind" / "what?" are ignored (they
  would cancel the driver's watch / repeat). DrivePilot's own voice never starts a question: ignored while the
  speaker plays or has something queued and for `open_mic.echo_tail_s` (0.6 s) after, counting the microphone's
  backlog; a failing check counts as talking. The "Hey DrivePilot" model keeps running in every mode ("Hey
  DrivePilot, stop"). **Verified:** pipeline tests with fake VAD/STT (`tests/test_wake_modes.py`, incl. noise then
  talk, live and batch) and `scripts/wake_modes_check.py` with the real models (Silero, the wake-word model,
  Whisper on the GPU) on the six Piper test voices, clean and in car noise at 10 dB: **197/204 as expected**. Every
  one-breath "Hey, ..." question heard (incl. "Hey there, ..."), talk without "hey" and "Hey, I'm so hungry"
  ignored; no wake word: questions heard, "I love this song" / "we're almost there" / "we're going to grandma's
  house later" / "never mind" ignored. Misses: "Hey." + a pause + the question in car noise, 3/6 (Whisper hears
  nothing for the lone word in noise, and noise must not open a window; clean 6/6); one voice (reza_ibrahim)
  saying "Hey Drive Pilot" (3, a known synthetic-voice limit of the wake word). **2 min of car noise alone in
  "none" mode: 0 speech starts, 0 answers.**
- **Said without asking** (`alerts.auto`, each switchable in config and under Settings on the page):
  "Light's green. Check before you go." (INFO; PLAN.md: never a bare "go"; "tell me when it's green" keeps its
  own sentence),
  "The car ahead is moving.", turn-by-turn ("In half a mile, take exit 14 onto SR 520 East toward Redmond.", at
  `nav.turn_prompts_m` 800 m / 150 m; not roundabouts, which have their own; the first maneuver isn't repeated after
  the destination answer named it), "You're in the wrong lane for your turn. Move one lane left." (lane detection +
  the route's lanes, the same match the overlay's target lane uses; after 2 s in a wrong lane within 800 m; once
  more within 250 m; "That's the right lane." after 2 s in a good lane; keyed by route, so a reroute's maneuvers are
  new), "You're over the limit. It's 30 here." (GPS speed over the sign detector's limit by 5 mph for 4 s; never a
  school limit; not one read more than 5 min ago or before a turn onto another road: heading 60° away for 3 s),
  "Stop sign ahead." (3 frames in a row). Quiet mode mutes the INFO ones (turn-by-turn and wrong lane are guidance
  the driver asked for by setting a destination). The last turn prompt is said down to 20 m (a maneuver right after
  another, or a GPS gap). All fixed texts are pre-rendered.
  **Found live on the SF red-light replay and fixed:** (1) "Stop sign ahead." for the grey *back* of the cross
  street's sign: now the box must be about square and its middle redder than green (median R - G >= 8; faces on
  earlier runs' frames, washed-out pink to maroon at dawn, measured +15..+77, backs -16..-5). (2) "The car ahead is
  moving." while we waited first at a red light, for a car that crossed in front of us and drove off up the street:
  now the car must itself have been stopped in front of us for 2 s and the motion tracker must see it moving (a
  steady, gentle departure is announced ~3 s after it starts), and nothing is said at a stop where our light read
  red or yellow during the stop or in the 15 s before it (on the approach: a truck may hide it once we stop; a frame
  where ego motion can't tell keeps that). ("Light's green." is the alert there; a car pulling away on red may be
  turning right, and the alert must never sound like "go".) After the fixes the same replay says "School zone
  ahead." and "Light's green." at 36.56 s (the light's change) and nothing else; switching "Light's green" off on
  the page and seeking back: the light change is logged, nothing is said.
  **Independent review** (13 findings, all checked and fixed with tests): the car-ahead red-light gaps above, a
  noise blip opening a hidden listening window in "hey" mode (and the race behind it, found again on real voices),
  overheard destinations / "never mind" in "none" mode, the bare "Light's green.", the right-lane confirmation
  without a hold, wrong-lane state across reroutes, a stale limit after a turn, close maneuvers without a prompt,
  single-frame stop signs and a hue check for warm light, the echo guard ignoring the microphone's backlog, "hey
  there", hotkeys after clicking a setting, and ranking half-typed names ("costc" -> Costco before Cost Plus).
- **Destination box** on the V2 page: "Where to?" with suggestions as you type (`/api/places`: Mapbox Search Box +
  Geocoding v6 with autocomplete, merged, best match first by the same scoring as spoken destinations), Enter or a
  click navigates (`/api/destination`; a picked suggestion's coordinates are used as is, a typed one is geocoded
  like "I'm going to ..."), the answer is spoken and shown, a route card shows "To Ferry Building · 26 min · 3.4 mi"
  with **End route**, the last 5 picks are chips (browser storage), **D** focuses it, `--destination PLACE` sets
  one at start once there's a GPS fix. **Verified in the browser** on the SF replay with its GPX: typed "ferry
  build" -> Ferry Building first of the suggestions -> "Okay, Ferry Building: 26 minutes, 3.4 miles. First, turn
  right onto 15th Street now." and the route card.
- Tests: `tests/test_wake_modes.py`, `tests/test_auto_alerts.py` (watches, engine, turn-by-turn on the fixture
  route, the page's endpoints, the app's wiring, Mapbox search on a mock transport, the world raising both events);
  four alert-engine tests now switch the unrequested green off (they test the watch). Full suite: 1476 passed.

### V2 (2026-10-10/11): V2-M0..M6 and M8 done (M1 parking and M5 drift below their accuracy bars on public footage; M7 RF-DETR cut)
- **The user's own dashcam footage (2026-10-11, HANDOFF.md section 2; results in the section above):** 950 Mercedes dashcam clips
  (30.2 h, D:\Dashcam). Their burned-in GPS/speed line is read into GPX (`nav/dashcam_overlay.py`,
  `scripts/dashcam_gpx.py`): 97.7 % of 1 Hz samples read on 837 clips, the time field agrees with the file name on
  99.6 %; 4.3 h of highway, 9.2 h city, 1.9 h lot-speed crawl, 111 night clips. Horizontal FOV 96° from 18 turns
  (`scripts/dashcam_calibrate.py`).
- **Vision answers improved without training (2026-10-11, the user's go-ahead):** a second ("reading") model, the
  place from GPS, and parking guidance in the prompt. (1) `vision.reading`: Claude Haiku 5.5 (thinking off, effort
  low) answers sign, parking and scene-description questions; Claude Haiku 4.5 keeps lights, lanes, memory and go
  questions (`answers/scene.py: uses_reading_model`). (2) `nav/location.py`: "Location (from GPS): Bayside,
  Queens, New York City, New York." from Mapbox reverse geocoding, looked up in the background (never delays an
  answer), plus a prompt line that local rules follow it (with NYC's no right on red). (3) Parking guidance in the
  system prompt (stall markings, curb paint, hydrants... decide parking; STOP / DO NOT ENTER / ONE WAY don't) and
  answers capped at ~30 words. Also: the test sheet now replays each clip's GPX (speed, location, the world's GPS
  speed, as in the app) and judges with a fixed model (`--judge-model`, Haiku 4.5 at temperature 0, as before).
  **Measured** (`eval/run_sheet.py`, judge fixed; a **held-out sheet** `eval/test_sheet_heldout.csv`: 27 new
  questions, 12 parking + 7 signs + 2 memory + 6 lights/road/safety, labeled on full-resolution crops by an agent
  that never saw an answer):

  | Answers by | Tuning sheet (45) right/partly/wrong | Held-out (27) right/partly/wrong |
  | --- | --- | --- |
  | before (Haiku 4.5, old prompt, no location) | 33 / 6 / 6 | 9 / 11 / 7 (33 %) |
  | Haiku 4.5 + location + prompt | 34 / 8 / 3 | 11 / 10 / 6 (41 %) |
  | Haiku 5.5 (thinking off) + location + prompt | 36 / 4 / 5 | 15 / 7 / 5 (56 %) |
  | Haiku 5.5, adaptive thinking (effort low) | 26 / 5 / 14 (slower, often cut off) | (not run) |
  | **split: Haiku 4.5 + Haiku 5.5 for reading (shipped)** | **36 / 6 / 3** | **18 / 5 / 4 (67 %)** |

  Why the split: Haiku 5.5 reads far more (the night exit sign, "2 HOUR", DO NOT ENTER, "31st", the NYPD traffic
  stop) but invents text on unreadable message boards, over-counts lanes, misnamed passed signs, and once called a
  red light yellow; Haiku 4.5 was right on those. The held-out set was looked at to choose the split, so it isn't
  blind any more. **Final blind check** (`eval/test_sheet_heldout2.csv`: 18 more questions labeled by the same agent,
  never looked at before this one run; the morning's code `0ab318b` run from a separate worktree, same judge):
  **before 8 / 7 / 3, shipped 10 / 5 / 3** (`runs/20261011-105719_sheet_baseline0ab318b`, `runs/20261011-105738_sheet`):
  better on parking (2 -> 3 right, 2 partly -> right, 1 right -> wrong: it hedged on a hatched call-box island) and
  "what's going on" (0 -> 1 of 2), the same on signs (2/4), lights (3/3), memory and safety. So the honest size of
  the gain is about +2 of 18 on unseen questions, smaller than on the set used to choose it. **Bench** with the
  reading model (`runs/20261011-110450_bench_app`, `runs/20261011-110607_bench_app`): every target met; model answers 1.84 / 2.11 s and
  1.55 / 1.66 s (median / p90, target 2.5; before 1.3-1.5 s): Haiku 5.5's first sentences are longer. Answer times unchanged (model rows ~1.0-1.3
  s median to the first sentence; reading questions 1.1-1.3 s). Checked in the real app: both models start, the
  location resolves ("Bayside, Queens, New York City, New York"), a sign question goes to Haiku 5.5 (30 mph, 1.17 s),
  a light question to Haiku 4.5 (red, 0.74 s).
- **V2-M8 done (2026-10-11 ~00:40).** Independent review of everything since the last push (7 bugs + 3 risks + 7
  nits, each with a reproducer; all fixed with regression tests except 2 cosmetic nits): **safety**: the new "Check
  the intersection before you go." was added to any model answer containing "green" ("It's about to turn green",
  "the left-turn arrow just turned green", "the cross street's light...", "it was green when we went through",
  "the green sign says...") -> now only a plain present statement about our light, while stopped, never against
  the tracker's red/yellow, never on memory answers; the new "to clear" exemption let "you have time to clear the
  lane" / "room to clear the turn" through the strict net -> only a waiting clause is exempt; "can I pull forward a
  little?" (in DEMO.md) wasn't a go question and "you can pull forward" wasn't permission -> both caught. Also: lane
  position vs guidance routing (11 phrasings each way), the sign model skipped (not raced) if the lane model is
  still loading after the timeout, school zones end 2 min after their last confirmed school sign (one-frame
  look-alikes kept a zone alive forever; a second school 3 min later was silent; school speed-limit signs count),
  an inverted consistency check in the alignment, an unknown `--tz` traceback, a misleading DST warning, a partial
  perception cache that could land over a full one, duplicate sign close-ups. Not fixed (cosmetic): the worker rate
  overestimates by N/(N-1) on short runs; a GPX 1.1 speed-source label. **Final checks after the fixes:** 1366
  tests pass, ruff + mypy clean; `scripts/bench_app.py --silent` with everything on passes every target
  (`runs/20261011-002333_bench_app`: local 0.56 / 0.62 s, model 1.48 / 1.48 s, alerts 12 ms, perception 29.3 fps,
  lanes 10.6 fps, signs 3.8 fps, 6/6); **test sheet 33 right, 6 partly, 6 wrong of 45 (73 %)**
  (`runs/20261011-002438_sheet`; lights 7/7, right of way 2/2, controls 1/1, road 6/7, safety 5/6, lanes 4/5,
  memory 2/3 + 1 partly, signs 4/7, parking 2/7 + 2 partly; row 45 now right: the pedestrian warning is kept).
  Still wrong: 10 (a street name it can't read), 11 (NYC's no right on red: no location context), 18 (lane count 2
  instead of 3, flaky), 25 (names the real DO NOT ENTER sign on the left, which the sheet didn't note, then
  concludes "no parking": the sheet now says so, partly right from the next run), 41 and 43 (parking judgments).
- **`scripts/align_gpx.py` for our own drives** (2026-10-11, by an agent, reviewed and refactored here;
  `nav/align.py`, 44 tests): phone video (`VID_YYYYMMDD_HHMMSS`, Pacific time, DST-aware; `tzdata` added for
  Windows) + GPSLogger GPX (1.0, or 1.1 with speed in extensions, or no speed) -> a copy of the GPX with the
  `video_start` waypoint. The start comes from the file name (default) and the MP4 `creation_time` via PyAV (start
  or end of the recording decided from name + duration; a warning when they disagree by > 2 s), then is refined by
  the first clean pull-away from a stop: ego motion "stopped -> moving" vs the GPS speed rising, checked against
  the same stop's arrival ("high") and the stopped/moving agreement over the overlap; only "high" matches are
  applied by default, else the timestamp start is kept and the summary says so. **Validation**
  (`scripts/eval_align.py`, BDD100K clips with a known `video_start` moved off by -9, -7, ..., +8 s, cached ego
  motion): 6 of 70 clips had a usable stop and pull-away, and the start came back within 0.32 s on 52 of 54 tries
  (median 0.22 s; the 2 others -1.15 s: 3f225bea at +6/+8 s, whose first stop is ambiguous there, so its second
  is used); the other 64 clips correctly report "no refinement"; "medium" matches (parked at the start) were off by
  up to 13 s, so they're opt-in. Re-checked here end to end on the GPU (592aaebe, start 2 s late): -0.23 s.
  `eval.perception_cache.build_cache` takes `max_s` now (the tool caches only the first `--search-s` seconds).
- **Test sheet, all 45 rows** (`eval/run_sheet.py`, lanes + signs + parking watch in lock-step, judge Claude Haiku,
  `runs/20261010-233000_sheet`): **32 right, 8 partly, 5 wrong** (71 % right). The 29 original rows: 21 / 6 / 2
  (before: 22 / 3 / 4: rows flip both ways run to run; the model and the judge vary). The 16 new V2 rows: 11 / 2 / 3:
  lanes 4/5 (+ row 30 after the fix below), memory 2/3, signs 36 right (30 mph), road/safety 38-39 right/partly,
  parking 40-42 right, 43 wrong (deflected to the stop sign), 44 partly (can't read the time limit). **Fixed from
  it** (re-run of the changed rows, `runs/20261010-234026_sheet` and the next one): "which lane am I in?" was routed
  to lane *guidance* and said "There's no destination set" without a route (now the camera answers it: row 30 wrong
  -> right, "the middle lane"); a hidden `[ref N]` tag left "green,." / "30 mph,." (punctuation cleaned up); a
  model answer saying our light is green now ends with "Check the intersection before you go." like the local
  light answers (row 8 partly -> right); sign questions also get close-ups of the signs the detectors saw in the
  last 5 s (`vision.sign_closeups`, labeled with the detector's guess); memory questions about signs send their
  frames at the kept 960 px instead of 640 (row 35, a night exit sign: the model had invented "Washington St / San
  Bruno Exwy / 170 St"; now "Cross Bronx Expwy" and "178 St" are right, the rest still wrong). Not fixed: row 14
  (two DO NOT ENTER signs 28 px tall in the rain: the detector sees them, the close-up is too blurry, and the model
  rightly won't name a sign it can't read: the detector's kind is 62-81 % precise per box on BDD, too unsure to
  speak), row 10 (a street name it can't read), row 25 (invented a DO NOT ENTER sign for "can I park here?").
- **Test sheet again after those fixes** (`runs/20261010-235356_sheet`): **31 right, 8 partly, 6 wrong** of 45:
  the same level, but different rows. Fixed rows held (8 lights partly -> right: lights 7/7; 30 lanes wrong ->
  right). Rows that changed by model variance: 11 (can't read the sign: partly -> wrong), 18 (lane count 3 -> 2: the
  known flaky count), 41 (hedged on the handicap spot), 45 (see next). **Found from it:** for "can I pull out now?"
  the strict safety net took "...so you need to wait until they clear and check both ways yourself" for "it's
  clear" and replaced that whole pedestrian warning with the generic sentence: now "until/till ... clear" and
  "to clear" are exempt (anything else with clear/safe still is: tests). Per category over both runs: lights
  6-7/7, right of way 2/2, controls 1/1, road 5/7 (+2 partly), safety 4-5/6, lanes 4/5, memory 2/3, signs 4/7,
  parking 2-3/7 (the weakest: the model hedges on spots and once invented a sign).
- **README screenshots** of the V2 page (`docs/screenshots/`, taken in the browser pane on 4 replays: highway
  lanes + "which lane am I in?", the speed-limit alert and badge, a pedestrian pulsing at a red light, "tell me when
  it's green"). Found while taking them: the two answer bugs above. `DEMO.md`: a 7-replay demo plan.
- **Full app with everything on (lanes + signs), `scripts/bench_app.py --silent`, GPU free** (2026-10-10 23:27):
  **every target met, twice** (`runs/20261010-232722_bench_app`, `runs/20261010-232825_bench_app`): local answers
  0.58 / 0.62 s and 0.57 / 0.73 s (median / p90, target 1.0), model answers 1.32 / 1.37 s and 1.55 / 1.59 s (2.5),
  alerts 12 ms and 10 ms max (0.3 s), perception 29.3 fps (25.2), lane detection 10.6 fps (10), sign detection
  3.8 fps (new check: >= 75 % of `sign_alerts.fps`), 6/6 heard and routed right. **The first run found a startup
  bug**: the lane and sign models were built side by side on their threads, and both capture CUDA graphs while
  loading (ultralytics' TensorRT loader captures one per engine load, 17 for the sign model, in global capture
  mode): each broke the other's capture and **neither model started** (the bench still passed its V1 checks; the
  21:44 live check had only been lucky with the timing). Now `start()` builds them one at a time before perception
  runs (engines are never reloaded after start-up: counted, 0 loads in 30 detections); regression test. Also the
  per-model rate counted from when the model was ready, so a model built first and waiting for the next one read
  slow (9.7 fps for 10.6): now from its first frame (test). One earlier run with the fix had a local p90 of 1.01 s:
  one answer's first sentence took Piper 0.72 s instead of ~0.25 s (CPU contention, not seen in the two re-runs).
- **Sign alerts once per zone / per new limit** (2026-10-10 23:15, HANDOFF 2.1). Cause, from the detector's own
  detections on the school-zone dev clip (dumped at the app's 4 fps): the street has a school sign on each side
  (two sightings in the first frame), the right one's AHEAD plaque is boxed as a second "school zone" at 1.6 s and
  2.4 s and the sign itself jumps once (a new sighting), and at 12.2-12.5 s a worker's hi-vis vest reads as a school
  sign (0.72). Each sighting was announced on its own. Now a confirmed school-zone sign starts a zone only after
  `sign_alerts.repeat_s` (300 s) without any school-zone sign, and every one seen meanwhile extends it (a one-frame
  look-alike never starts one). Re-run through the app (`runs/20261010-231342_red_light_rain_school_zone_*`):
  **"School zone ahead." once, at 0.25 s** (was 4 times). Speed limits were already once per new value; the SPEED
  LIMIT 30 sign on ego_holdout_long_stop is read from 8.74 s and **"Speed limit is now 30." said once at 8.65 s**
  in the app (`runs/20261010-231438_ego_holdout_long_stop_*`). 3 regression tests (the two zone tests fail on the
  old code). `sign_alerts:` is now spelled out in config.yaml.
- **V2-M6 sign detector merged** (2026-10-10 ~21:40; the sign session's branch `v2-signs`, its STATUS in
  `scripts/signs/STATUS.md`: fresh BDD held-out 69/77 clearly readable speed limits right, 0 values contradicting a
  label, 6.6-7.1 ms a frame as TensorRT FP16). The app injects it like the lane model (`sign_model_factory`); the
  test sheet runs it in lock-step (`--no-signs`). Live check on the school-zone dev clip: `signs=on lanes=on`,
  signs 3.8 fps, lanes 10.4 fps, perception 30 fps, no errors; **but "School zone ahead." was said 4 times** (twice
  in one frame, then at 2.4 s and 12.2 s): not yet fixed (HANDOFF 2.1).
- **V2-M5 YOLOPv2 lanes and drift warning** (2026-10-10 evening; part 1 by an agent, finished, reviewed and integrated
  here). `perception/lanes.py`: the official YOLOPv2 TorchScript in FP16 on its own CUDA stream, the forward pass
  replayed as a CUDA graph (captured thread-locally: in the default global mode the detector thread's GPU work broke
  the capture in the app and lost one of its frames; the app now also waits for the sign/lane models before
  perception starts), 12 fps on the newest frame (`perception/frame_worker.py`, shared with the sign thread).
  `state/lanes.py`: lane lines as parallel curves in the top-down view, horizon from the lines' vanishing point,
  metric scale from our lane's width; lane count and our lane counted against the drivable area (None when not
  clear). **Lane index on 65 hand-labeled highway frames (11 BDD clips): right on 57 of 58 answered (98 %),
  answered 58/65 (88 % of all frames right: bar 85 %); lane count 17/25 answered** (it undercounts lanes on the
  right, so lane guidance needs the count to match the route's, else asks the vision model; the vision prompt
  gets only our lane from the left). Found and
  fixed by the new tests (`tests/test_lanes.py`, rendered masks) and the clips: the heading/curvature search was
  biased by histogram bin phase (lines 0.15 m off; now scored on smoothed histograms with a wider fine search);
  the vanishing-row vote was won by a curved line's own segments (0.67 instead of 0.47 on a 1250 m curve; now both
  sides' lines must meet) and a curvy start locked in a wrong horizon (an outlier gate; now a running median).
  **Drift** (`scripts/eval_lanes.py drift`, the 11 highway clips with their GPS, 5.7 min above 25 mph): measured
  from the absolute offset it gave **10 false alerts**: the camera's mount + the driver's habit put the measured
  offset at -0.5..+0.2 m. Now drift is measured from our usual place (median offset over 2 min at speed, after 8 s
  of it), must be steady (toward the line in 80 % of 2.5 s) and not a lane change starting (a 0.6 s speed check:
  the 1.5 s fit lags a quick move), and our lane must keep its shape (its width changing less than our shift: a
  lane splitting off moves one line, a drift moves us between both): **1 alert left at the chosen 0.2 m gap**, a
  borderline real slow move (night, easing 0.8 m right while a car passes close on the left). **Virtual drift** (`eval_lanes.py virtual`: a straight clip warped into a slow
  sideways slide, the lane model run on the warped frames; 3 clips, both sides, at the 0.2 m gap): **full lane
  changes (0.8 and 1.2 m/s) 12/12 quiet; slow drifts (0.12-0.3 m/s) 13/18 alerted within 1 s of the expected time,
  3 late by 1.2-3.3 s, 2 effectively missed** (552991d2 leftward, 6 s and 21 s late: the car started 0.4 m right of
  its usual place, so the slide had to reach ~1.1 m, where the warped frames distort). So the plan's bars (0 alerts
  on straight driving / lane changes; within 1 s on the virtual clip) are not fully met: see Known issues.
  **Independent review (V2-M5): 6 bugs, each with a reproducer, all fixed with regression tests:** (1, safety) the
  pedestrian road gate read the mask under someone's feet, where YOLOPv2 leaves a hole around a person and our
  hood line is never road: people crossing a few metres ahead read as "off the road" (34/34 and 39/50 frames on two
  dev clips; a person dead ahead never warned about when pulling away): now the road beside their feet counts, and
  nobody within 0.9 m of the path's middle is gated; (2) a first wide lane (a city street, a lot) locked out normal
  lanes for the session (now: two seconds of rejected plausible widths restart the width history; the scale
  history is 50 s, a camera property); (3) drift noise grew with every lane change (k x width; now a base fixed at
  each crossing); (4) lane guidance said "I can't tell" instead of asking the vision model when lane detection's
  count disagreed with the route's; (5) the old lane won the vote for ~0.35 s after a lane change; (6) the previous
  segment's road and lane survived a seek/loop. Also: the drift shows on the page for 4 s, config cross-checks, the
  pedestrian eval replays the app's ~0.1 s mask lag.
  On the V2 page: our lane (lines with a dark halo: they sit on the road's own paint), the other lines dashed
  gray, the drift line red and that screen edge pulsing (checked in the browser). Lane detection's count/lane goes
  to the vision prompt and to lane guidance (`answers/lanes.py: CameraLaneReader`). Drift warnings are SAFETY
  (spoken in quiet mode, pre-rendered). **Full app with lanes on** (`scripts/bench_app.py --silent`,
  `runs/20261010-212804_bench_app`, GPU free): every target met: local answers 0.55 s median / 0.61 s p90, model
  answers 1.33 / 1.37 s, alert 8 ms, perception 29.5 fps, lane detection 10.6 fps (V2 bar >= 10).
- **Pedestrian warnings use YOLOPv2's drivable area** (V2-M5): someone at least 0.9 m from the path's middle counts
  as in our path only if the ground beside their feet is road (a sidewalk isn't; stepping off it still warns as
  "entering"; masks <= 0.3 s old). `scripts/eval_pedestrians.py --road` on cached masks: **crawl clips 12 -> 9
  warnings** (removed: a man inside a store seen through glass at night, a glare blob, a dashboard reflection),
  **lot clips 6 -> 5** (a worker beside a truck at the clip's start), **dev clips 0 -> 0**. (A first version that
  read the mask under the feet removed more, 12 -> 8 and 6 -> 3, but also hid people crossing right in front of
  us: the review's safety finding.)
- **Crawl clips, labels half B** (`eval/parking_labels_crawl_b.yaml`, a labeling subagent, saved clip by clip: 10/16
  usable, 9 clear + 5 ambiguous open spots, 35 look-alikes): parking tracker **0/6 clear unrestricted spots found,
  2 false spot events in 5.2 min**, lot offer in 8/10 lot clips, 4 false pull-outs; the model check rejected all 5
  spot events of both crawl halves (all 5 false). Same picture as half A.
- **Sign alerts were spoken in quiet mode** (INFO alerts bypassed the alert engine's quiet check): fixed, tested.
- **V2-M1 parking on real footage** (`scripts/cache_perception.py` caches tracks + ego motion per clip once;
  `scripts/eval_parking.py` replays `ParkingTracker` with GPS speed and scores it against `eval/parking_labels.yaml`,
  `--sweep` for tuning). First run: 0 spot events. Fixed: the self-measured scale ran to its 2.5x clamp in sparse lots
  (gaps taken for the pitch; now only spacings near the prior count, scale kept to 0.75-1.35), our hood and cut-off
  boxes gave positions (now none at the bottom edge, at the side only while the nearest corner is in view), our speed
  comes from GPS when there's a fix (`WorldState(speed_source=...)`, wired in the app), "looks like a lot" holds 5 s
  (it flapped), pull-outs must move >= 0.8 m from a row toward our path with our own turn subtracted. **Result on
  the 28 usable clips (30 clear unrestricted spots, 13.5 min of lot driving): recall 1/30, 0.7 false spots per
  10 min, lot offer in 15/25 lot clips (7 outside lot windows, which the model check gates), parked 4/5 (1 false),
  pull-outs 1 false.** The plan's bar (>= 70 %, <= 1 false / 10 min) is not met on recall. Why (measured): most
  labeled spots pass at 6-13 m/s (13-30 mph, well above a parking-search crawl), and the tracker almost never sees
  parked cars in a row on the spot's side (median 0: YOLO tracks churn on half-hidden cars at the frame edges; the
  per-clip camera scale differs 0.4-2.9x from GPS); loosening the thresholds (`--sweep`) adds false spots, not
  recall. Labels spot-checked on 6 full-resolution frames: the spots are real. **Alternative measured** (V2_BUILD_PLAN
  asked for it), a periodic Claude Haiku scan (`eval/parking_scan.py`, `scripts/eval_parking_scan.py`, by a
  subagent, ~2,400 calls, ~$7): best 47 % recall at 26.6 false / 10 min, handicap stalls announced as open 3 times;
  Sonnet 4.6: 20 % at 1.9. Not shipped. The geometry + per-candidate model check stays: rare but precise.
  **Model check measured** (`scripts/eval_parking_check.py`, the stall now outlined by its ground footprint): of the
  5 spot events on the 40 lot clips + crawl half A, Claude Haiku rejected all 5 (4 false, and the 1 hit, whose stall
  is at the frame edge by the time it's confirmed on a 60° camera): nothing false is spoken, nothing found either.
  **32 more lot clips at parking-search speed** (`data/videos/parking_crawl`, user's request, labels half A in
  `eval/parking_labels_crawl_a.yaml`: 7/16 usable, 2 clear spots, both the stall we park in): 0/2 found, 3 false spot
  events in 4.1 min. Even crawling, the clips mostly cross lots at an angle instead of driving down an aisle, which
  is the only case the geometry handles.
- **V2-M2 overlay page** (`ui/overlay.py`, `ui/static/v2.html`, `ui.overlays`): each frame and its overlay packet go
  in one binary WebSocket message (built on the world thread right after that frame's update, so they can't drift);
  4-way badges with timers (next car amber, YOU badge green on our turn), people amber, the light's state, parked
  cars gray, open spots green while the parking watch is on, the hood filtered out; HUD with the nav banner and
  Mapbox lanes, GPS speed and a roundabout diagram. Checked in the browser (SF red-light clip): sampled canvas
  pixels at every box edge carry the stroke color; page 21-23 fps, ~85 ms from capture to screen while the GPU was
  shared with the sign fine-tune; perception 29.6 fps. V1 page at /v1.
- **V2-M4 memory and answer highlight:** `state/memory.py` keeps a frame a second for 60 s and the best close-up of
  each sign seen (stop signs now; the V2-M6 detector adds its kinds with their reading); a MEMORY route ("what was
  that sign?", "what did we just pass?", "...back there") sends 5 frames over the last 15 s plus up to 4 sign
  close-ups, numbered; an answer's hidden "[image k]" shows that frame as a thumbnail on the page. Model answers
  get the tracked objects numbered on the latest frame (`vision.mark_objects`); the hidden "[ref N]" is kept only
  if it fits what the answer names (class, color, side; a tag on a far-off object yields to a much bigger one that
  fits; quoted sign text names nothing), else the one object matching the description, else nothing; right-of-way
  answers pulse the car that goes next. Tags and stray mark numbers are stripped before speech (tests). **Measured
  live (Claude Haiku):** full sheet 22 right / 3 partly / 4 wrong of 29 (before: 23 / 3 / 3; the changed rows are
  the lane count and a distant sign, both flaky before), model answer median ~1.0 s (unchanged); highlights on the
  7 object rows all right after two fixes (3 pulse the right object: person crossing, bus, school bus; 4 correctly
  none: sign, building, construction, road); first try had 1 of 3 right (a person tagged for "the bus"). The memory
  row ("what did that yellow sign say?") stays partly right (reads "SCHOOL ... AHEAD"). 986 tests pass.
- **Independent review of V2-M1 fixes, M2, M4 (2026-10-10): 11 bugs + 7 risks, each with a reproducer; fixed with
  regression tests** (`tests/test_v2_review.py`, 35 of its 41 tests fail on the reviewed code): **safety**: a hidden
  tag between two sentences glued a bare "Go." / "Merge now." onto the previous sentence past the start-anchored
  permission net (now every piece of a sentence is checked, which also catches "Sure thing. Go.", and the splitter
  splits before "["); a 4-way deadlock or a refusal while moving still pulsed one car and painted it amber as "goes
  next"; a sentence replaced by the safety net still pulsed the car it named; sign text was stripped ("marked 35",
  "[EXIT ONLY]") while mark/image numbers were spoken ("In Image 2, ...", "Car #1", "(ref: 1)", a tag cut off at the
  end); a memory answer naming a sign close-up showed no thumbnail, and thumbnails landed under the question; our
  hood could be mark 1 and a big car beside us overrode the model's right tag; highlights didn't show while a replay
  was paused (and pulsed forever once paused: the overlay now carries `pulse_until`); open-ended spot labels matched
  any event to the clip's end (capped at 8 s); memory routes for present-tense questions ("if I go past...", "did
  you see the light turn green?"); a cut-off car's samples survived a turn; GPS speed now wins over a possibly wrong
  "stopped"; the page could draw an older frame decoded late, and counted fps on decodes; sign close-up updates are
  atomic. **Not fixed (measured):** a row where most spacings are one-stall gaps measures the gap as its pitch in
  simulation (xfail test): three fixes (tighter window, lower quartile, harmonic voting) each added 1-3 false spots
  on the real lot clips, so the original estimator stays. 1035 tests pass.
- **5A (BDD100K GPS):** `nav/bdd_info.py` + `scripts/bdd_info_to_gpx.py` read `data/bdd100k/bdd100k_info.zip` without
  unpacking (fields checked on real files and a 3,000-file survey: -1 = unknown speed/course, bad accuracy, repeated
  timestamps, gaps up to 23 s), write GPX 1.0 with a `video_start` waypoint; `GpxReplay` aligns on it, never
  interpolates across a gap, interpolates speed. All 10 dev clips converted. Alignment checked against the
  hand-labeled stops: GPS positions stop within −0.26 / +0.19 s of the labels (the phone's speed field lags 0.5-1 s).
  Found and fixed: the vision model only got our speed while navigating ("how fast am I going?" → "I can't see your
  speedometer"); now "You're going 17 miles per hour" (GPX: 17.0 mph).
- **V2-M0 (test inputs without a drive), via an agent worktree:** simulated drives along the six saved Mapbox routes
  (`nav/sim_drive.py`, `scripts/route_to_gpx.py`: speeds from step durations, braking for maneuvers by type, stops,
  correlated GPS noise, missed turns) checked through the real `NavHandler` (`scripts/sim_nav_check.py`): 11/11 drives
  pass (lane reminders −4..+1 % of 1 mi / 0.25 mi, roundabout callouts, arrival, reroute 4.6-5.7 s after a missed
  turn; 220/220 over 20 seeds at 3-4 m noise; 216/220 at 5 m, failures kept). Looping clips: the clock runs on
  across passes (a long GPX plays to its end; checked live for 110 s). `scripts/find_bdd_clips.py`: clips by scene
  (public scene labels, 426 "parking lot") and GPS motion; 40 parking-lot clips + GPX fetched.
- **V2-M1 part 1 (find me a parking spot):** `state/parking.py`, `answers/parking.py`, router PARKING intents,
  plain yes/no reply window (`VoicePipeline.expect_reply`), wiring; design and the user's rules in their docstrings
  and HANDOFF.md section 1. Verified so far only in simulation: 12 simulated-lot scenes (one open stall found once
  with the right side and distance, two adjacent stalls, full rows with jitter, dropouts, row ends, a small car
  pulled in behind an SUV, a person in a stall, a car in the aisle, a wrong field of view, a car pulling out, lot
  vs fast street, parked after a turn), 27 watch/assistant tests, 3 reply-window tests. 903 tests pass.
- **Parking labels** for evaluation: `eval/parking_labels.yaml` (40 clips, 28 usable; 35 clear + 12 ambiguous open
  spots; 66 look-alikes), by two labeling subagents; not yet spot-checked.
- **V2-M3 (pedestrian and cyclist alerts), via an agent worktree:** `state/pedestrians.py` (owned by `WorldState`,
  event `pedestrian_in_path`, snapshot `pedestrians`), spoken by `AlertEngine` from 10 fixed pre-rendered sentences
  ("Pedestrian stepping into the road on your right.", "Cyclist ahead on your left.", ...), SAFETY priority (kept in
  quiet mode; the speaker now also puts it ahead of queued alerts and cuts off a playing non-safety alert, which is
  said again after it), once per track, a group's repeats (same kind and side within 3 s) once; config
  `pedestrians:` + `alerts.safety_gap_s`.
  Design in the module docstring: distance from ground contact and box height (log-combined); x measured from the
  focus of expansion (one lot clip's dashcam was aimed ~6° off the car's axis: "straight ahead" was the sidewalk);
  path = 1.5 m either side, 6 m + 2 s at our speed (≤ 35 m), bent by the homography turn rate less a curvature
  uncertainty and narrowed far out by it; "about to enter" from the lateral speed in excess of a noise floor
  (within 15 m); only while moving (ego motion + GPS ≥ 0.5 m/s); cyclist riding ahead with us at our speed: not
  warned (decision, tested). Verified: 54 synthetic tests (pinhole street sim with jitter, dropouts, ID switches,
  a 150 m curve, a misread turn rate, a 6°-off camera, a horizon 25-35 px off, a tree, the windshield mount, legs
  hidden behind a car, a motorcyclist, a parked bicycle; a mutation check confirmed the tests catch each rule
  being weakened) + 1 speaker test; replayed through the real WorldState + AlertEngine on cached perception
  (`scripts/eval_pedestrians.py`, frames drawn and looked at): **0 alerts on all 10 dev clips** (the
  construction-clip crossing and the SF person with the dog happen while we are stopped or 16-20 m ahead of a 9 m
  path at walking pace: silent, as specified); on 29 parking-lot clips 6 alerts: 2 right (a man stepping out
  between vans into our path; at night, people stepping into the crosswalk just ahead as we roll up to a stop sign),
  4 borderline (someone at an aisle edge; a worker beside a truck in the first 0.1 s; at a gas station, a man by a
  car we turn towards and someone at the pump island beside the lane we turn into), 0 clearly false (the first
  version had 9, all false or borderline). Cost on the world thread: 0.08 ms mean, 0.3 ms p99 per frame;
  detection frame → first audio sample handed to the output ~1 ms (real Piper clips, fake device). Independent
  review: 8 findings + 5 suspicions; fixed: a 3 s gap that silenced a second, unrelated hazard; "too tall" vetoing
  real people when the horizon is off; a parked bicycle turning "cyclist" after one overlap; the far path narrowing
  to ±0.5 m (now ≥ ±1 m) and predictions only within 15 m (now 2 s at our speed); relinks across kinds; a mount test
  that didn't test the mount rule; no warnings when ego motion can't see but GPS says we drive; a cut-off alert
  replayed after it had finished. The test sheet now ignores safety warnings for `ALERT@` rows and runs its BDD
  clips at 60°.

### V1 polish (2026-10-10): throttle check, the early-wake voice bug fixed and reviewed
- **Throttle check** (HANDOFF.md section 3 item 2): not throttled. On the charger; under a 4-thread load the busy
  cores ran at ~135% of rated speed (~3.9 GHz); all 20 threads busy: 89% (2.66 GHz, the package power limit,
  not the 31% stuck state seen earlier); GPU fp16 matmul 22.9 TFLOPS at P0. The Windows power mode overlay is
  not the "Best performance" one, though: set it before timing runs and drives.
- **Wake word fired early, question ended at the pause after "Hey DrivePilot"** (HANDOFF.md section 3 item 1):
  fixed in `voice/pipeline.py`. Audio is now held from the end of every question until its transcript is known;
  an empty transcript (only the wake phrase was heard) resumes listening on that audio, so the question after
  the pause is still heard; the no-speech timeout still counts from the wake word. A new wake word while a
  transcript is pending takes over (and doesn't switch off the page's listening light). Resumed audio has no
  wake phrase at its start, so only a whole repeated wake phrase is stripped there ("is it my turn,
  DrivePilot?" keeps its words); a repeated "Hey DrivePilot" at the start of any transcript is now stripped.
  The benchmarks (`voice_eval.py`, `check_real.py`, `run_sheet.py --voice`) feed silence until the pipeline is
  idle, so a question still being waited for can't spill into the next clip. 10 new test cases (the pipeline
  ones fail on the old code).
- **Independent review of the fix: no bugs, 2 risks + 2 nits, each confirmed with a reproducer, all fixed** with
  tests that fail on the reviewed code: a repeated "Hey DrivePilot" during resumed listening was swallowed and
  the window still ran from the first wake (question lost); now a wake word during a question restarts the
  listening window (not the question: a wake on a trailing "...DrivePilot?" would lose it, and `on_wake` would
  make the app count it as superseded). An end found while replaying held audio was dated at the replay (app
  latency under-reported when Whisper is slow); held chunks keep their arrival time. The batch test now forces
  the `flush()` path. Whisper's spellings of a repeated wake phrase ("It's a driver's pilot", "Hey Dr. Pilot")
  are stripped from resumed questions. Also fixed (pre-existing, found by the review): a late question that a
  newer wake word superseded switched the page's listening light off during the new question.
- **Measured** (final, `runs/20261010-114439_voice`; before the fix: `runs/20261010-110459_voice`):
  `scripts/voice_eval.py --limit 40`: "woke, heard nothing" rows **4 → 0** (before: 3 clear voices + Reza;
  every remaining miss is the Reza voice not waking); heard and routed right of all rows 82.5% / 85% (clean / car
  noise) before, **95% / 92.5%** after; wake on intelligible clips 100% / 96.9%; 0 false wakes in 10 min; local
  answer audio 0.49 s / 0.52 s median. Real voices (`check_real.py --evaluate data/wakeword/real`): 14/20 wake,
  **14/14 heard and routed right** (unchanged). Checked in the browser: the listening light stays on through a
  superseded question and goes off after a normal one. 807 tests pass; ruff + mypy clean.
  Note: Piper synthesis is random (same sentence, different audio each run), so per-row wake results on the
  borderline Reza voice, and rarely another voice, change from run to run; the wake score itself depends only
  on the audio (checked: in-stream = fresh model on the same audio).

### M9 (in progress, 2026-10-10): hardware bring-up with the team
- **GPU throttling fixed** (the team plugged in the charger and set Best performance): fp16 matmul 22.4 TFLOPS
  (was 2.1). Re-measured the full pipeline (`scripts/bench_app.py --silent`,
  `runs/20261010-100132_bench_app`): **every latency target met**: local answers 0.83 s median / 0.94 s
  p90 (target 1.0), model answers 1.71 / 1.83 s (2.5), "Light's green" 41 ms from the detection frame to sound
  (0.3 s), perception 26.8 fps while Whisper shares the GPU (25.2), 6/6 spoken questions heard, routed and
  played.
- **Live phone camera through the app** (`python -m app --camera`, OnePlus 12 over USB): ultrawide camera 3,
  1920x1080, hardware (D3D11VA) decoding, 30 fps processed, page live, microphone + speaker + Claude ready, no
  errors. Not yet tried by the team: questions by voice live, the phone GPS page.
- **First real-voice session** (`scripts/wakeword/check_real.py`, one speaker, 20 takes, 6 with a fumbled
  wake phrase): at threshold 0.25, 12/20 woke and only 8 of those were heard right. Found and fixed: the chunk
  the wake word fired in was missing from the question's audio (an 80 ms hole that garbled the first word),
  and the 0.4 s pre-roll cut the wake phrase in half. Now (threshold 0.15, 1.5 s pre-roll): 14/20 wake (the
  misses are the 5 fumbled takes and 1 clear one) and **14/14 are heard and routed right**. End of question
  after 0.5 s of silence (was 0.6): none of the takes was cut off. A pause after the wake phrase no longer
  ends the question (speech in the first 0.4 s after the wake may be just "...pilot"; then 1.2 s of
  silence is needed), and "Hey DrivePilot, stop" is still heard. Final on the real takes: 14/20 wake,
  **14/14 heard and routed right**. Synthetic voices (`scripts/voice_eval.py --limit 40`): all 5 clear
  voices wake 100%, 0 false wakes in 10 min of speech + car noise; misses are the known-unintelligible
  Reza voice.

### M8: test sheet, scoring (2026-10-10)
- `eval/test_sheet.csv`: 29 questions at verified moments of the 9 dev clips (each expected answer checked on
  full-resolution frames; three of my first expectations were wrong and were corrected): right of way 2,
  lights 7 (incl. "tell me when it's green" with its alert time), signs 6, road 6, parking 2, lanes 1,
  safety 4, controls 1. `eval/run_sheet.py` plays each clip frame by frame, asks through the same Assistant
  as the app, has Claude Haiku judge each answer (verdict + reason; broken JSON tolerated), keeps a `manual`
  column that wins over the judge, and writes results.csv / report.md / report.json (`--voice`: questions
  spoken through the wake word and Whisper).
- **Score: 23 right, 3 partly, 3 wrong of 29** (79% right; 84% counting partly as half), typed questions,
  `runs/20261010-085118_sheet` (+ row 15 re-judged in `runs/*_sheet` after the judge fix). Lights 7/7, road
  6/6, right of way 2/2, lanes 1/1, controls 1/1, safety 3/4 + 1 partly (didn't mention the person with the
  dog), parking 1/2 + 1 partly (hedged on the meter), signs 2/6 + 1 partly. The 3 wrong are sign reading:
  "31 St" not read, DO NOT ENTER read as "STOP", and New York's no-right-on-red rule not known (the model
  doesn't know which city it is in).
- The first run (16/29) found real safety bugs, fixed with tests: behind a stopped school bus, "should I go
  around them?" got "check both ways before going around it", "can I pass the school bus?" got the generic
  gap refusal, and the bus with its STOP arm out was said to be stopped "for pedestrians" (now: a fixed local
  rule for passing a school bus, a prompt rule, and a net that replaces a model answer mentioning a school
  bus when the question is about passing); a "no turn on red" sign and overhead-board text were invented
  (prompt: never name a sign you can't read; vision temperature 0, sent in the request body since the
  1.x SDK dropped the argument).
- Answer times (typed, until the first sentence is ready and synthesized): median 1.29 s overall, local
  ~0.4 s, vision model ~1.4-1.6 s. Spoken end-to-end times: `scripts/bench_app.py` (below, M7).
- Not covered by the sheet yet: navigation (dev clips have no GPX), a 4-way stop with other cars, roundabouts,
  lane guidance with route data: they need our own recorded drives (M9). Navigation is covered by 70 tests.

### M4 + M7 independent review (2026-10-10): 15 findings, all confirmed, all fixed with tests
- Speech: a slow one-piece answer (navigation, lane reading, destination) was still spoken after "stop", the
  wake word, a newer question or "Light's green" (now dropped if overtaken); the wake word and "repeat that"
  threw away a queued or playing alert for good (now they stop answers only); a stop or alert landing while a
  sound was being prepared was lost (start checked under the output's lock); alerts waited for an answer's
  synthesis (synthesis moved to its own thread; pre-rendered alerts ready at once: 550 ms -> < 100 ms);
  route-changing navigation alerts didn't drop stale answers; "repeat that" repeated only the last sentence.
- Navigation questions were refused while the camera was out (they only need GPS).
- Latency bookkeeping could file a latency under the wrong question (now matched by utterance id; every
  utterance, dropped ones too, gets a record).
- Page server: reachable through DNS rebinding and from other sites' pages over the WebSocket (now: allowed
  host names only, same-origin POSTs and WebSocket); messages could reach the page out of order; page
  messages and summary.json are redacted.
- A lost camera ended the app after 100 restarts (now retried forever with backoff); at the end of a replay
  the clock kept running (now the last frame is held and seeking back works).
- bench_app could mismatch questions and silently leave out answers that never played; a flaky end-to-end test
  and two tests that couldn't fail were fixed.

### M7: app integration, V1 screen, run logging (2026-10-10)
- `python -m app` runs everything in one process for the camera or a file (`--video`, `--camera`, `--gpx`,
  `--no-voice`, `--no-audio`, `--ask`, `--exit-at-end`); components that can't start (no mic, no key, no audio
  device) are reported on the page and the rest runs. A live camera that drops out is restarted; meanwhile
  questions get "I've lost the camera feed". The vision model gets a tiny startup request (checks the key,
  opens the connection).
- `ui/`: the V1 page (MJPEG video + the conversation as text, status line, hotkeys space / arrows / Q, history
  kept across reloads) and the phone GPS page (once a second, screen kept on); JSON-only POSTs.
- Run folder: events, frames (1/s + every frame sent to the vision model), detections, perf, `summary.json`.
  `core/latency.py` measures end of question → first sound and detection frame → alert sound.
- **Full-pipeline benchmark** (re-run after the review fixes, `runs/20261010-090257_bench_app`: same
  numbers, 6/6 answers played, alert 94 ms) (`scripts/bench_app.py --silent`, SF red-light clip, 6 synthesized spoken
  questions over car noise, fed in real time): 6/6 heard and routed right; "Light's green" alert **87 ms**
  from the detection frame to sound (target < 300 ms); local answers **2.3 s** (target 1.0) and model answers
  **3.1 s** (target 2.5); perception **15 fps** while Whisper shares the GPU (target 25.2). All three misses
  come from the GPU throttling (Known issues): Whisper takes 1.5-1.85 s per question instead of ~0.2 s, and
  YOLO slows while it runs. Breakdown for a local answer: end of question detected 0.6 s after speech,
  Whisper (started at 0.25 s of silence) 1.5-1.85 s, speech synthesis ~0.25 s. Re-measure once the GPU runs
  at full speed.
- The benchmark found two bugs (fixed, with tests): "Light's green" interrupted "Not yet, the light's red" and
  the old answer then resumed; an answer to a slowly transcribed question was spoken over the next question.
- Checked in the browser: video, a typed question answered by the vision model, pause / seek / resume.

### M6: navigation (2026-10-10; independent review done, all findings fixed)
- `nav/mapbox.py`: "take me to ..." via Mapbox Search Box + Geocoding v6 candidates, picked by spoken-word
  match vs distance (`pick_place`; plain geocoding alone sent "Space Needle" to Arizona); Directions v5
  driving-traffic with steps, banners, lanes, `roundabout_exits=true` and our heading. `FixtureClient` for tests.
- `nav/route.py` (steps, lanes at the maneuver, exit numbers, roundabout exits passed), `nav/progress.py`
  (snapping within reach of the last fix, off route > 50 m for 5 s or wrong way, joining from a parking lot,
  arrival incl. turning into the lot), `nav/handler.py` (route/lane answers, lane reminders at ~1 mi and
  ~0.25 mi, roundabout "not this one", background reroutes with backoff, navigation ends on arrival),
  `nav/phrasing.py` (US units, passenger-style lanes), `nav/gps.py` (GPX replay, phone fixes).
- `scripts/fetch_route.py`: real routes printed and saved under `data/routes/` (gitignored: Mapbox terms).
  Checked by eye on real routes: lanes on I-405/SR 520 ramps, exit refs, Carmel roundabouts (exit counts).
- **Independent review found 11 issues, all confirmed with offline probes; all fixed**, 26 regression tests
  (`tests/test_nav_review.py`): "did I miss my turn?" said "No, you're on the route" just after missing it;
  lane answers could name the opposite lane (skipped the imminent maneuver, read "right" in "the right lane"
  as a road name, said "any lane" when Mapbox had no lane data); "how far to my turn?" gave the destination;
  the arrive step was spoken as "You have arrived ... in 800 feet"; reroute storms (starting in a parking lot,
  turning into the destination's lot, after arrival, offline failures every 20 s); one noisy fix at a
  cloverleaf crossing jumped progress ahead for good; reroute races (spoke after cancel, landed after a new
  destination, stale "rerouted"); stale GPS answered as current; parse errors / NaN headings escaped;
  1st-exit roundabouts; "take me home", "stop navigating". Probes re-run: all fixed.
- Tests: 70 navigation tests (fixture route + synthetic GPS tracks), 3 runs clean.

### M4 (part 1): voice pipeline (2026-10-10; review with M7)
- `voice/`: mic (16 kHz), wake word (openWakeWord, 0.25), streaming Silero VAD v6 endpointing (0.6 s silence,
  speculative transcription at 0.25 s), faster-whisper small.en float16 on CUDA (cuBLAS from PyTorch's CUDA 12
  build), Piper TTS with pre-rendered alert clips and a chime, `AudioSpeaker` with priorities (alerts interrupt
  an answer, which resumes), "stop" silences everything.
- **Synthetic spoken-question benchmark** (`scripts/voice_eval.py`, 245 questions from both router tables,
  6 held-out Piper voices, clean + car noise at 10 dB SNR, 31 min; `runs/20261010-064344_voice/`):
  wake rate 94.6% clean / 95.6% car noise on clips where the wake phrase is intelligible (5 of 6 voices
  97.6-100%; the 6th, en_US-reza_ibrahim, 52-62%: its "Hey DrivePilot" comes out as "Hi-Lac" / "a drive
  highlight" even to Whisper); **0 false wakes** in 10 min of speech + car noise; WER 7-8%; routing of what
  was heard 94.1%, now **95.9%** (97.5% without that voice) after router fixes re-scored on the saved
  transcripts. Local answer audio: median **2.0 s** vs the 1.0 s target: Whisper takes 1.7 s because the GPU
  is throttled (see Known issues); endpointing itself is 0.5 s.

### M5 review (2026-10-10)
- Independent review found 5 bugs (3 safety) and several risks; all fixed with tests
  (`tests/test_vision_safety.py`): a per-sentence net replaces any sentence that vouches for traffic ("Looks
  clear, go ahead", "Merge now.") with "I can't judge moving traffic, so check both ways yourself before you
  go."; the prompt never implies it's safe; tracker context gives right-of-way verdicts; deadlines and "stop"
  close the HTTP stream within 0.5 s; up to 3 light close-ups; sign questions reach further back. Live re-run
  (Claude Haiku): 9/10, first sentence median 1.0 s.

### M5 (part 1): vision-model answers (2026-10-10)
- `answers/vision.py`: one `VisionModel` interface (streaming text + JSON) with Claude Haiku (default),
  Gemini, Ollama and Mock providers, picked by `vision_model`; failures become `VisionError`.
- `answers/scene.py`: frames (latest full-size, earlier smaller), a traffic-light close-up for light questions,
  day/time, our motion, tracker state; first-token/total deadlines with spoken fallbacks; bare-"go" net.
  `answers/sentences.py`: speech starts on the first sentence. Assistant streams and lets "stop" cut it off.
- Tools: `scripts/ask_model.py` + `eval/model_questions.yaml` (live questions on dev clips).

### Wake word "Hey DrivePilot" (2026-10-10, trained in the background since step 2)
- `models/wakeword/hey_drivepilot.onnx` (openWakeWord, 205 KB, tracked) + `scripts/wakeword/` (data, training,
  evaluation; README with exact commands). Recommended: threshold 0.25, debounce 2 s, no `patience`.

### M3: router, local answers, alert engine (2026-10-10)
- **Independent review found safety bugs; all fixed with tests** (`tests/test_answers_safety.py` + a second
  router table of the reviewer's 149 adversarial phrasings): 47/53 split-second phrasings had reached the
  vision model while moving (now matched by shape: question frame + maneuver); "is it safe to pull out?" got
  "Yes…" (no answer starts with "yes" any more); "should I wait?" got the "can I go?" answer (own intent);
  "can I go?" at a yellow while moving got a judgment (refused); watch phrasings over-matched ("tell me where
  to go" armed a light alert); a requested "you're up" could be eaten by the cooldown, survive "never mind",
  be contradicted by a race, or fire at an unrelated light minutes later. All confirmed fixed by re-running
  the reviewer's scenarios.
- `voice/router.py`: ordered keyword/regex router: voice controls, "I'm going to <place>", "tell me when it's
  green" / "when I can go" watches, split-second questions (refused while moving), right of way (local),
  navigation, lanes, everything else → vision model. 273-case tables incl. the tricky cases ("can I go
  right on red?" is a sign question; "did I miss my turn?" is navigation). ~50 µs per question.
- `answers/local.py`: PLAN.md phrasing ("The white car on your left, then the gray truck across from you. You're
  third."), ties ("You and the car on your right stopped together. The car on the right goes first."), 4-way
  deadlock, caveats (car out of view, narrow view, pedestrians, lost/maybe-parked cars), light answers. Never a
  bare "go": every permissive answer says to check first (asserted in tests).
- `answers/alerts.py`: once-per-event alerts with a cooldown; "You're up, check and go." (spoken on its own) and
  "Light's green, check the intersection and go." (only when asked); watches expire on timeout, seek-back and
  driving away; quiet mode keeps only requested/safety alerts.
- `answers/assistant.py`: question → route → handler → speaker; quiet/normal/repeat/stop/never mind; navigation
  (M6) and vision model (M5) behind small interfaces. `scripts/ask_replay.py`: typed questions on a replay.

### M2: world state + 4-way-stop queue + traffic lights (2026-10-10)
- `state/vehicle_motion.py`: per-car stopped/moving in the world (frame-to-frame homographies cancel our own
  motion; one vectorized remap), stop dated from positions, noise floors for box jitter, "arrived while we
  watched" only from what we saw while standing still.
- `state/approach.py`: left/ahead/right zones, the car in front of us ("lead", seen from behind), front car per
  approach. `state/stop_queue.py`: arrival order, ties to the right (2-, 3-way), 4-way deadlock, departures
  (sustained driving, not a creep), lost-track re-linking, parked/lost cars dropped *without* an alert, copies
  in snapshots. `state/traffic_light.py`: lit-lamp classifier + per-light tracker, anchored while stopped,
  "turned green" only for our own light. `state/colors.py`: white balance + exposure, chroma voting.
- `state/world.py`: at the stop line (not behind another car), pedestrians ahead, `drove_off` / `segment_reset`
  events, frame buffer. `perception/pipeline.py: AsyncListener` runs the world update off the perception thread.
- Tools: `scripts/world_timeline.py`, `scripts/eval_traffic_lights.py` (+ `eval/light_labels.yaml`).
- **Independent review found 7 bugs (6 safety) + 3 risks**, each confirmed with a reproducer; all fixed and
  re-checked with the reviewer's own scripts, plus `tests/test_intersection_safety.py` (pinhole-camera
  scenarios with box jitter): a creeping car no longer "departs"; our own creep doesn't either; stopping
  behind another car no longer starts the queue; jitter no longer makes an already-waiting car look newly
  arrived (was 37/40 runs wrong, now 0/40); a cross-street light turning green is not "our" green; a car lost
  from view never triggers "you're up"; snapshots are immutable; world update 2.5x cheaper.

### M1: video capture, detection/tracking, ego motion, performance (2026-10-10)
- **Video sources** (`capture/`): one `VideoSource` interface. `FileSource` paces on the video's timestamps
  (real time; decode-bound playback slows down instead of starving), lock-step mode for evals, pause/seek/loop,
  timestamp jumps start a new segment. `ScrcpyCameraSource` talks to scrcpy-server 4.1 directly over adb (no
  window capture), hardware H.264 decoding (D3D11VA → CUDA → software), stall watchdog, fast stop.
- **Detection** (`perception/detector.py`): YOLO26s as a **TensorRT FP16 engine** (built once by
  `scripts/build_engine.py`; eager PyTorch was launch-bound on Windows: 22 ms vs 2.7 ms inference) + ultralytics
  BYTETracker driven directly; cross-class vehicle de-duplication; ids unique across seeks; refuses the CPU.
- **Ego motion** (`perception/ego_motion.py`): LK optical flow with sky/hood ROI and object masks; long-baseline
  (0.6 s) "combined" measure (zoom rate × 50 vs horizontal flow) + hysteresis; runs in parallel with detection.
- **Worker** (`perception/pipeline.py`): source → detect ∥ ego → listeners; survives transient errors; `stale()`.
- **Tools:** `scripts/bench_pipeline.py`, `scripts/eval_ego_motion.py` (+ `eval/ego_stop_labels.yaml`),
  `scripts/fetch_dev_clips.py`, `scripts/build_engine.py`, `scripts/check_gpu.py`.
- Independent review: 3 bugs + 7 risks + 2 nits, all fixed with regression tests.

### M0: core foundations (2026-10-10)
- `core/config.py`: `config.yaml` → strict, immutable pydantic settings. Unknown keys and uncoerced types fail
  with the dotted field path; YAML 1.2 scalar rules; dotted overrides (`--set video.source=camera`); secrets
  from env/.env as `SecretStr` (real env wins over .env); `gps_source` resolution; non-UTF-8 files → clear error.
- `core/logging_setup.py`: structured logger (any field name), console + JSON lines, secret redaction
  everywhere (re-setup keeps it), numpy/NaN-safe JSON.
- `core/clock.py`: `WallClock` (camera) and thread-safe pausable/seekable `MediaClock` (file playback), both on
  `time.perf_counter` (Windows `time.monotonic` ticks every 15.6 ms).
- `core/runlog.py`: per-run review folder (config, events, detections, perf, frames). Producers never block;
  frames drop when the writer is behind; buffered writes flushed when idle; clean drain on close/atexit.
- `core/types.py`: read-only, identity-compared `Frame` with a `segment` counter for seeks.
- `app.py --print-config` (keys masked). Independent review found 12 issues; all fixed, with tests.

### Step 2: V1 build plan approved (2026-10-09)
- BUILD_PLAN.md: milestones M0–M9 with "done when" tests, plus a V1 feature coverage table (all features, not
  just the 4-way stop; "tell me when the light is green" added at the user's request).
- Decisions: wake word "Hey DrivePilot" (custom openWakeWord model trained locally; training runs in the
  background, scripts in `scripts/wakeword/`); default vision model Claude Haiku (`claude-haiku-4-5`);
  push to GitHub after every milestone; git identity NMGit-hub confirmed.

### Step 1: project setup (2026-10-09)
- CLAUDE.md with the standing rules; .gitignore; .env.example; package folders per PLAN.md "Code layout"
  (+ `core/`, `tests/`, `scripts/`); `config.yaml` skeleton; README with setup steps.
- Python 3.11.15 venv in `.venv` (uv-managed CPython, not the Microsoft Store build).
- `requirements.txt` (pinned top-level) + `requirements-lock.txt` (full freeze; `.venv` matches it exactly).

## Verified (and how)
- **Answers on real replays** (`scripts/ask_replay.py`), re-run after the M3 fixes. SF stop-sign clip: "can I
  go?" while still rolling → "I can't judge that safely while we're moving."; at the stop → "Nobody else is
  waiting at the stop, so it's your turn. Check the intersection before you go."; "is it safe to pull out?" →
  "By arrival order it's your turn, but I can't judge moving traffic. Check both ways before you go. Watch for
  the pedestrian ahead." (a pedestrian with a dog crosses); "can I merge now?" while moving → refused. SF
  red-light clip: "can I go?" → "Not yet, the light's red. I'll tell you when it turns green."; "tell me when
  it's green" → alert at 36.56 s (hand label 36.4 s). Local answers take < 1 ms.
- **Vision model on dev clips** (`scripts/ask_model.py --suite eval/model_questions.yaml`, Claude Haiku, live):
  9/10 right (truck-route sign read exactly, overhead board honestly "too far to read", TD Bank, construction,
  pedestrian crossing, red light, school-zone signs "can't read the text"); first sentence median 0.97 s,
  max 1.6 s (target < 2.5 s to first audio). Miss: lane count (2 instead of 3) on one of two runs.
- **Wake word** (`scripts/wakeword/evaluate.py`, held-out synthetic voices, `models/wakeword/metrics.json`):
  recall 97.1% clean / 97.5% in unseen car noise (10 dB); 0 false accepts in 10.7 h of LibriSpeech.
- **World timelines on real clips** (`scripts/world_timeline.py`): SF stop: arrival 21.54 s (ego label soft stop
  21.6 s), nobody waiting (checked on frames: the minivan on the left was crossing), left at 24.9 s. Suburban
  clip: rolls through its stop sign (no queue), later stops behind a car at a red light (no stop-sign queue).
- **Traffic lights** (`scripts/eval_traffic_lights.py`, 4 labeled clips): 262/280 sampled moments right (94%),
  17 "unsure", 1 wrong (at the change itself); red→green caught at 36.56 s (label 36.4 s); 0 false
  "turned green" events.
- **Car colors**: checked by eye on 41 vehicle crops from 5 dev clips (before: nearly all "blue" at dawn).
- **World update cost**: 4-9 ms p50 per frame on real clips (5 cars avg), measured with the CPU loaded by
  wake-word training; runs on its own thread (`AsyncListener`).
- **Tests:** 372 pass (incl. 2 GPU); ruff + mypy clean.
- **M1 performance** (`scripts/bench_pipeline.py`, dev clip c4130097, 720p30, other GPU users present: Epic
  launcher overlay, wake-word training): real-time **29.6 fps** processed of 30.2 (target ≥ 25.2); p95
  processing (detect+track ∥ ego) **27.5 ms** (< 40); p95 decode-to-result latency **32 ms** (< 70);
  lock-step throughput 47 fps; GPU: TensorRT engine ~90 MB, device total < 1 GB. Inference 5–7 ms in the
  pipeline (2.7 ms in isolation). Detector start-up verifies TensorRT backend on cuda:0 in fp16.
- **Ego stopped/moving** (`scripts/eval_ego_motion.py`): hand labels from x-t images (magnified at
  transitions). Dev clips: 6/7 within ±0.5 s (worst 0.43 s); the rain clip is +0.64 s and briefly split.
  **Blind held-out (first look, 3 clips): 1/3 strict pass**; boundary errors +0.07, +0.30, +0.53, +0.75 s
  (mean 0.41 s) plus a 1.5 s "stop" during a slow parking crawl. So: ~±0.5 s, usually late. M2's queue is built
  to tolerate this (creeping/brief blips don't reset arrival; ties use a 1 s window).
- **Live phone camera** (2026-10-09, OnePlus 12 via USB): `ScrcpyCameraSource` delivered 1920x1080 at ~30 fps
  from cameras 0/2/3/5. **Camera 3 = physical ultrawide** (same view as camera 0 at zoom 0.6; chosen because it
  can't switch lenses by itself). A recorded 6 s stream decodes at 12.6 ms/frame with D3D11VA (software was
  ~42 ms: too slow for 30 fps). Hardware decoding not yet re-tested live (phone unplugged).
- **Tests:** 128 (126 non-GPU + 2 GPU) pass; full suite run 3x clean after fixing a recorder test that could
  starve; ruff + mypy clean.
- **API keys work** (live calls, 2026-10-09): Anthropic `claude-haiku-4-5` answered in 1.06 s; Mapbox public token
  works for Geocoding v6 and Directions v5 `driving-traffic` (lane data on 116 intersections of a 12.7 km route).
- **GPU actually used:** `scripts/check_gpu.py` (fp16 matmul ~21 TFLOPS on cuda:0; CTranslate2 sees the GPU).

## Next
- **After the submission (2026-10-11):** the parking tracker on the 96° dashcam (0/8 on the user's lots); drift on
  road-shape changes (ramps, exit splits: 4.8 false / h); the night green-LED misread; the lane tracker placing our
  lane one over on the NYC clip; blur plates in published videos if wanted. Rotate the API keys after the hackathon.
- **V2 next steps:** our own drives (`scripts/align_gpx.py`, README "Record your own drive") to check lanes and drift on the phone's 114° view and to tune parking; then V1's deferred live drive. Earlier: **V2 build plan** (`V2_BUILD_PLAN.md`, 2026-10-10): milestones V2-M0..M8 with "done when" tests and how each
  feature is tested without a live drive. The user answered the parking questions, set the deadline (**Sunday
  2026-10-11, 8:59 pm PDT**) and moved the sign fine-tune to a separate session (`SIGN_FINETUNE_PROMPT.md`).
  The user said go on 2026-10-10.
- **Before calling V1 done (needs the team):** check the CPU throttling (below) and re-run
  `scripts/bench_app.py --silent` (all targets were met at 10:01; at 10:20 the CPU was stuck at ~0.75 GHz);
  ask a few questions live by voice with the phone camera (`python -m app --camera`) and try the phone GPS page;
  record more voices with `scripts/wakeword/check_real.py`, ideally in the car; recorded drives with GPX
  (4-way stops with other cars, a roundabout, the 405/520 interchange, parking signs) so the test sheet gets
  navigation / 4-way / roundabout rows; then the live drive (PLAN.md's V1 definition).
- **2026-10-10: the user approved V2 without the live drive** (deferred), plus a new V2 feature, "find me a
  parking spot". HANDOFF.md has the decisions, a starting design, and how to get recorded drives + GPS at home.
- M9 (needs the team): fix the GPU throttling and re-measure latency; live phone camera + GPS page;
  real-voice wake-word check (`scripts/wakeword/check_real.py`); recorded drives with GPX to add
  navigation, 4-way-stop and roundabout rows to the test sheet; then the live drive.

## Test data
- Source: BDD100K `videos/100k/val` clips (720p, 30 fps, ~40 s, US cities), pulled by HTTP range requests
  from a byte-for-byte Hugging Face mirror of the official zip (`linxxx3/bdd100k_videos`); see
  `eval/dev_clips.yaml` and `scripts/fetch_dev_clips.py`. Development/tuning only. Replace with our own drives.
- No dev clip shows several cars waiting at a 4-way stop: the queue logic is verified with synthetic scenarios
  until we record one.
- BDD100K's GPS/IMU "info" files are not in that mirror, so dev clips have **no GPX** yet. **2026-10-10: the user
  downloaded `data/bdd100k/bdd100k_info.zip`** (from BDD100K's new home, https://dl.cv.ethz.ch/bdd100k/data/;
  4.09 GB, 80,000 clips' info: 70,000 train + 10,000 val; all 10 dev clips present, CRC-checked). Converting it to
  GPX (HANDOFF.md 5A steps 6-8) waits for the user's go.

## Known issues / compromises
- **On the user's dashcam footage (2026-10-11):** (1) **parking spots 0/8** found on 12 lots (3 false spots, 7 false
  "parked" at long stops at lights and lot exits): the tracker rarely fires on this 96° camera, not investigated
  further (deadline). (2) **Drift: 4.8 false alerts / h** at highway speed, all from the road changing shape (a ramp
  line narrowing in, an exit-only lane split at night moving the left line one lane over, a merge); 2 borderline
  alerts came from "our usual place" carried over from the previous lane after a lane change. (3) A green LED at
  night, blooming, read **YELLOW** (`dm_20260929_195410`, 25-27 s): no "Light's green" there. (4) The lane index is
  wrong where the tracker places our lane one to the left (one BDD100K NYC clip, 4 frames, since the yellow-edge
  rule), and lanes to our right are still undercounted (the count stays out of the prompt). (5) The yellow-edge rule
  trusts the HSV color: faded yellow below saturation 70, or a yellow line in deep shade, falls back to the old count.
- **Showcase videos:** model answers differ from run to run, and for 3 scenes the better of 2 real takes was used
  (PROGRESS "own dashcam footage"). The renderer differs from the app in small ways: "stop" / "never mind" stop the
  answer but not a playing alert; the wake word cuts an answer where the question starts (the app: ~0.7 s later, at
  the wake word); an answer card shows its whole sentence even when an alert cut its audio; frames go to x264 as
  untagged BT.601 (colors shift slightly in BT.709 players); a jump in the clip's timestamps doesn't reset lanes and
  memory as the app does. Other cars' license plates are readable in the videos (not blurred).
- **The new parking prompt** calls a stall "ordinary" when it misses painted words on it (a NO PARKING box on a
  held-out sheet row; before, it hedged), and `scripts/eval_lanes.py --frames` draws lines without the crop's
  mapping on a cropped camera (the saved frames' lines are offset; the numbers are right).
- **Open-mic wake modes (`hey`, `none`) are tested on synthetic voices only**, not real voices in the car. Without
  the full wake word, passengers' questions are answered too ("none" mode answers any question; "hey" mode any
  that starts with "hey"), and every utterance in the car is transcribed (Whisper on the GPU, ~0.1-0.3 s each).
  While DrivePilot talks (and 0.6 s after) speech is ignored in these modes, so stopping an answer needs "Hey
  DrivePilot, stop". In "hey" mode, say the question in one breath in a noisy car ("Hey, is it my turn?"): a lone
  "Hey." and a pause works clean but only 3/6 synthetic voices in car noise (Whisper hears nothing for one short
  word, and noise must not open a window).
- **Unrequested alerts are checked on public replays, not our own drives:** "The car ahead is moving." has no
  positive case on our clips (the SF one is at a red light, where it is now suppressed by design); the wrong-lane
  warning needs lane detection's count to match the route's lanes (untested on the phone's 114° view, like lanes in
  general); speeding uses the last limit the sign detector read (forgotten after 5 min or a turn, but a road whose
  limit changes without a sign isn't noticed); "Stop sign ahead." needs a visible red face, so a sign in deep shadow
  at night may not count, and backs in warm evening light are rejected by hue only in theory (no golden-hour
  footage).
- **The reading model (Claude Haiku 5.5) sometimes names a sign it can't read** ("looks like a round red-and-white
  stop sign, but it's too blurry to read" where there was only a small street-name sign; "likely a stop sign" for a
  street blade; text invented on an unreadable LED message board), although it says it can't read it. Not tuned
  further: both question sets used for tuning are spent, and the blind set showed the honest gain is small. Model
  answers also got ~0.3 s slower (still within the 2.5 s target).
- **`align_gpx.py` is untested on a real OnePlus + GPSLogger pair** (none exists yet): whether OnePlus writes
  `creation_time` at the start or the end is detected, not known; GPSLogger's speed field placement is handled in
  every form seen in GPX files; the 0.25 s pull-away bias and the thresholds come from BDD100K's iPhone data. A drive
  needs a full stop + pull-away in its first `--search-s` (180 s) for the refinement, else the timestamp start
  (±1 s) is used. Speed from positions (no speed in the GPX) is less precise (one clip -1.65 s).
- **The sign detector takes a hi-vis vest for a school sign** (school-zone dev clip, 12.2-12.5 s, conf 0.72, two
  frames): inside a zone that's quiet now, but on its own it would say "School zone ahead." once. Not tuned away
  (one case; raising `signs.class_conf.school_zone` would cost real distant signs, which score 0.66-0.9).
- **Drift warning (V2-M5) below its bars on BDD100K footage:** 1 alert in 5.7 min of real highway driving, a real
  slow move toward a line but not a drift a driver would call one (bar: 0); on the virtual-drift clips 13/18 slow
  drifts on time, 3 late, 2 effectively missed (bar: within 1 s). Learning "our usual place in the lane" needs 8 s at speed
  and absorbs part of a long drift in a 40 s clip (the late ones); a noisy measured position can suppress a real
  drift as a "lane change starting" (the missed ones: the safe side). Tuned on 60° BDD dashcams only; recheck on the
  phone's 114° view (PLAN: undistort or center crop) and our own drives. Drift warnings need GPS speed >= 25 mph.
- **Lanes on the phone's 114° ultrawide are untested:** the lane model gets the frame's center cropped to 70°
  (`lanes.max_view_deg`, V2_BUILD_PLAN's center-crop fallback; no checkerboard calibration): checked only
  geometrically (tests) because no phone footage exists yet. Record a drive and re-run `scripts/eval_lanes.py`.
- **Lane count from lane detection is weaker than our lane's index** (17/24 vs 55/56 on labeled frames): it misses
  lanes on the right; lane guidance only uses it when it matches the route's count.
- **"Find me a parking spot" finds few spots on real footage: 1 of 30 labeled clear spots on BDD100K lots** (0.7
  false per 10 min; plan bar >= 70 % recall). Precise but rarely fires; a Claude Haiku scan was measured instead and
  is far worse on false alarms (see Done, V2-M1). Causes: fast through-aisle driving in the clips, parked rows not
  formed from churning edge tracks, unknown BDD camera scales. Our own lot footage with the phone (114°, known
  mount) is needed to tune it; street mode is untested; the lot offer needs the vision model. On a ~60° camera a
  spot is confirmed only ~6 m ahead (the hidden-small-car rule).
- **Pedestrian warnings on the 32 extra crawl clips (`data/videos/parking_crawl`, checked after the merge): 12**, of
  which 8 were drawn and looked at: 3 look right (people walking in the lot aisle ahead), 1 borderline (someone at a
  curb's edge), 4 false (two on unusable footage: sun glare and a dark interior shot; a windshield reflection; a man
  walking along a sidewalk beside parked cars). Dev clips stay at 0. False warnings are the safe mistake but they
  interrupt. **With the drivable area (lanes on): 8**; removed were the dark-interior one (a man inside a store seen
  through its glass doors), the glare blob, a hand after the phone was taken off its mount, and a person between
  parked cars at an aisle edge. Still there: the dashboard reflection on 2f4afe73 (it lies over the road).
- **Pedestrian alerts (V2-M3): the path is a geometric corridor from where we're heading, gated by YOLOPv2's
  drivable area when the lanes model runs** (V2-M5: removed 3 false + 4 borderline warnings on the lot and crawl
  clips); with lanes off, people at the edge of a narrow aisle or by a pump island can be called "in our path". Turns are read from the frame homography (~1°/s off on straight roads, so gentle curves are
  only partly bent). Further limits: no prediction while turning hard (≥ 10°/s) or for boxes cut off at a frame edge;
  boxes cut off at a side edge never alert (on a 60° camera that hides someone within ~2.6 m of the bumper's
  corners; the phone's 114° lens sees them); the first ~1.5 s after a reset use the image centre as our heading
  and the prior horizon; a new track beyond 9 m waits ~0.3 s for its motion before it can be "in the path"; a
  bicycle not ridden for 2 frames is treated as parked; a "person" over 1.3× too tall for where it stands (with
  0.8° of horizon slack) is ignored beyond 8 m (trees, poles; also someone whose legs are hidden behind a car).
  Without GPS the zone length assumes 8 m/s and turns bend as if at 8 m/s (half the bend at 4 m/s). With live
  1 Hz phone GPS, the speed can read ~0 for up to 1 s after we pull away, which holds back the "someone in our path
  as we start moving" warning by up to that second (GPS slower than 0.5 m/s overrides ego motion's "moving", to
  stay silent when ego motion flickers at a red light; the GPX replay interpolates, so the evals don't show it).
  "Stop" drops a warning queued that instant (the person counts as warned). **The field of view must be right**:
  BDD100K clips in the app need `--set video.file.hfov_deg=60` (at the 114° default, people 16-20 m away read as
  6-7 m and two false warnings fire on the dev clips; the evals and the sheet set it). Tuned on BDD100K footage
  only (narrow 60° dashcams); the dev clips have no real case of someone in our path while we move, so true
  positives on real footage come from the lot clips (2) and the synthetic tests.
- **A looping clip played with its own GPX gets GPS only on its first pass** (the clock runs on across passes so a
  long simulated drive plays to its end).
- PLAN.md was deleted from the working tree by an unknown process on 2026-10-10 (not the user, not this session);
  restored from git.
- **After the wake word, the microphone listens for up to 4 s** (`voice.endpoint.no_speech_timeout_s`, counted from
  the latest wake word), now also after an empty transcript (noise, a cough, a hallucination): a passenger
  talking in that window can be taken as the question. The price of not losing a question asked after a pause.
- **The synthetic voice benchmark varies from run to run** (`scripts/voice_eval.py`): Piper's synthesis is random,
  so the same question is different audio each run; on 40 questions a run can lose or gain a wake on the
  borderline Reza voice and, rarely, on a clear voice (a bad synthesis). Compare several runs, not single rows.
- **CPU throttling seen at 10:20 on 2026-10-10** (after a fast run at 10:01): on the charger, the CPU ran at ~31%
  of its rated speed under load (~0.75 GHz), so perception fell to 7-8 fps and answers to 3-5 s. Likely heat or
  the power source (a lower-wattage charger, e.g. in a car); check before measuring or driving.
- **Sign reading by the vision model (Claude Haiku)** is the weakest category on the sheet (2/6): small or
  distant text is misread (DO NOT ENTER as "STOP") or not read; the model doesn't know the city's rules (no
  location context in V1). It now says when it can't read a sign instead of inventing text.
- `vision.temperature: 0` works with Claude Haiku 4.5 (sent in the request body); Claude models newer than the
  4.6 line reject it: set it to null when switching (the startup check shows the failure on the page).
- **The laptop must be plugged into its charger and in Best performance mode**: otherwise the GPU throttles
  to ~1/10 of its speed (seen 2026-10-10: 210 MHz, "SW power cap" + "SW thermal slowdown"), and Whisper takes
  1.7 s per question instead of ~0.25 s: answers then take 2-3 s.
- **Navigation edge cases left:** a 15 m overshoot of a turn is still "on the route" when asked (GPS noise is
  ~5-10 m; at 30 m it reroutes); arrival by turning into a lot within 150 m of the end is assumed (a missed
  final turn that close is called "arrived"); `fork` steps count as exits for "which exit?".
- **Wake word trained on synthetic speech only** (Piper voices). One real speaker so far (20 takes at a desk):
  14/20 woke at threshold 0.15, the misses being mostly fumbled takes; more voices, recorded in the car, are
  still needed before the demo (threshold 0.10 is a safe fallback: also 0 false accepts on 21 h). Held-out car noise is one car; no in-car conversation/radio in the false-
  accept tests. The training data uses non-commercial licenses (fine for the hackathon, not for a product).
  `data/wakeword/` (34 GB) and `.venv-wakeword/` (5.7 GB) can be deleted; only the 205 KB model is needed.
- **Lane counting by the vision model is unreliable** (said 2 lanes instead of 3 on one of two runs). M6 lane
  guidance will ask for a structured count with a confidence and fall back to the route's lane data alone.
- **No real multi-car 4-way stop on video yet.** BUILD_PLAN M2's "queue log matches my hand-labeled timeline
  on the dev 4-way-stop clip" can't be checked until we record one: the queue is verified with synthetic and
  pinhole-camera scenarios (with jitter) and on real single-car stops. Record 4-way stops with 1-3 other cars.
- **"You're up" comes ~1 s after the other car starts moving** (PLAN target for alerts is 0.3 s from
  detection): a departure is only confirmed after 1 s of sustained driving, because a car creeping up to its
  line looks the same for the first second. Deliberate: a false "you're up" is the dangerous mistake.
- **Cars already waiting when we arrive are assumed to be ahead of us** (we can't see when they arrived).
  Wrongly yielding is the safe mistake; answers say "already waiting when you got here".
- **A car ahead that's lost from view (8 s) or hasn't moved for 20 s is dropped without an alert**; "can I go?"
  then says who it lost track of instead of "yes".
- **Ego-motion accuracy** generalizes to ~±0.5 s (held-out 1/3 strict at ±0.5 s; see Verified). Rain/wiper
  footage is worse (rain is out of scope per PLAN.md). Re-check on our own drives with our mount.
- **Small, distant traffic lights** can be missed by YOLO at the 384x640 input after they change color; while
  stopped, lights are anchored (read at a remembered position) to cover this. Moving vehicles' lights rely on
  detection.
- `scripts/build_engine.py` uses a separate `.venv-trtbuild`: ultralytics' FP16 TensorRT export needs NVIDIA
  ModelOpt, whose extras install onnxruntime-gpu, which collides with the voice pipeline's onnxruntime. Ultralytics'
  auto-installer is disabled in our code (`YOLO_AUTOINSTALL=false`) after it altered the environment once.
- Windows "Balanced" power mode limits CPU boost; use **Best performance** and stay plugged in for drives.
- The API keys were shared in chat; consider rotating them after the hackathon.
