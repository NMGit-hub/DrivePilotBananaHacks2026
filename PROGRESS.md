# PROGRESS

Live status of the V1 build. Updated at every milestone. Newest information first within each section.

## Done

### M3: router, local answers, alert engine (2026-10-10; independent review pending)
- `voice/router.py`: ordered keyword/regex router: voice controls, "I'm going to <place>", "tell me when it's
  green" / "when I can go" watches, split-second questions (refused while moving), right of way (local),
  navigation, lanes, everything else → vision model. 123-phrasing table test incl. the tricky cases ("can I go
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
- **Answers on real replays** (`scripts/ask_replay.py`): SF stop-sign clip: "can I go?" while moving → "We're
  moving..."; at the stop → "Yes, nobody else is waiting. Check the intersection before you go." plus "Watch
  for the pedestrian ahead" (a pedestrian with a dog crosses); "can I merge now?" while moving → refused. SF
  red-light clip: "can I go?" → "Not yet, the light's red. I'll tell you when it turns green."; "tell me when
  it's green" → alert at 36.56 s (hand label 36.4 s). Local answers take < 1 ms.
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
- M3 independent review; M5 (vision model: Claude Haiku default, Gemini/Ollama/mock) in progress; then M4
  (voice audio), M6 (navigation), M7 (app + screen), M8 (test sheet).
- Background: "Hey DrivePilot" wake-word training (separate agent, `.venv-wakeword`).

## Test data
- Source: BDD100K `videos/100k/val` clips (720p, 30 fps, ~40 s, US cities), pulled by HTTP range requests
  from a byte-for-byte Hugging Face mirror of the official zip (`linxxx3/bdd100k_videos`); see
  `eval/dev_clips.yaml` and `scripts/fetch_dev_clips.py`. Development/tuning only. Replace with our own drives.
- No dev clip shows several cars waiting at a 4-way stop: the queue logic is verified with synthetic scenarios
  until we record one.
- BDD100K's GPS/IMU "info" files are not in that mirror, so dev clips have **no GPX**. The GPX replay path is
  tested with synthetic GPX files in tests until real drives are recorded.

## Known issues / compromises
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
