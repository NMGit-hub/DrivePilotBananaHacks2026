# PROGRESS

Live status of the V1 build. Updated at every milestone. Newest information first within each section.

## Done

### V2 start (2026-10-10): throttle check, the early-wake voice bug fixed
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
  idle, so a question still being waited for can't spill into the next clip. 10 new test cases (the pipeline ones fail on the old code).
- **Measured:** `scripts/voice_eval.py --limit 40` (`runs/20261010-111819_voice`, before:
  `runs/20261010-110459_voice`): the "woke, heard nothing" rows went from 4 (3 clear voices + Reza) to 1 (Reza
  only); heard and routed right of all rows 82.5% / 85% (clean / car noise) before, 87.5-90% / 90-92.5% after
  (two runs); 0 false wakes in 10 min; local answer audio 0.49 s / 0.52 s median. Real voices
  (`check_real.py --evaluate data/wakeword/real`): 14/20 wake, **14/14 heard and routed right** (unchanged).
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
- **V2 build plan proposed** (`V2_BUILD_PLAN.md`, 2026-10-10): milestones V2-M0..M8 with "done when" tests and how
  each feature is tested without a live drive; waiting for the user's approval and answers (parking design, sign
  fine-tune go/no-go, Monday's deadline) before building.
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
- BDD100K's GPS/IMU "info" files are not in that mirror, so dev clips have **no GPX**. The GPX replay path is
  tested with synthetic GPX files in tests until real drives are recorded.

## Known issues / compromises
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
