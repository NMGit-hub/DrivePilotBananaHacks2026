# V1 build plan

The V1 scope from PLAN.md, split into milestones. Each milestone ends with its "done when" test, a strict
self-review plus an independent review, fixes, a full test run and a commit. Nothing from V2 is built.

**Runtime shape (one process).** Capture thread → perception thread (YOLO26s + ByteTrack + ego motion on the
GPU) → world state → alert engine. In parallel: mic thread → wake word → end-of-speech detection → faster-whisper →
router → handler (local / navigation / vision model) → speaker thread (priority queue, interrupt and resume).
FastAPI serves the V1 page (video + question/answer text) and the phone GPS page. In file mode, all timing
logic runs on the video's own clock, so pause/seek work and tests are deterministic.

**Latency targets (from PLAN.md):** local answers < 1.0 s from the end of the question to the first spoken
word; model answers < 2.5 s; alerts < 0.3 s from detection to sound. Perception keeps up with real time.

**Decisions (2026-10-09):** wake word "Hey DrivePilot" (custom openWakeWord model trained locally); vision model
default Claude Haiku (`claude-haiku-4-5`), with Gemini kept as a drop-in; push to GitHub after every milestone.

## V1 feature coverage

The 4-way stop is one feature among many. Every V1 feature, where it is built, and how it is tested:

| Feature (PLAN.md) | You say / it says | Built in | Tested by |
| --- | --- | --- | --- |
| 4-way stop: who came first, can I go | "who got here first?", "can I go?", alert "You're up, check and go." | M2 (queue), M3 (answers, alert) | Synthetic scenarios + labeled dev clip |
| Lane guidance: am I in the right lane | "which lane for 520 east?", "am I in the right lane?", alert "In half a mile, move one lane left" | M5 (model counts lanes, reads overhead signs) + M6 (Mapbox lanes, lane math, reminders at 1 mi / 0.25 mi) | Lane-math unit tests; fixture routes; model on dev frames |
| Roundabout exits | "which exit?", alert "Not this one, next exit." | M6 (exit counting from GPS), M5 (landmark description) | Fixture roundabout route + synthetic GPX |
| Traffic lights | "is the left arrow green?" | M5 (light crop + full frame to the model) | Model on dev frames with lights |
| **Tell me when the light is green** (added) | "tell me when it's green" → alert "Light's green, check the intersection and go." | M2 (per-frame light-state tracker on YOLO light crops: color + red→green transition, debounced), M3 (watch request + once-per-event alert) | Unit tests on synthetic crops/sequences; dev clips with real light changes |
| Signs: what did that say, speed limit, turn rules | "can I turn right on red here?", "what's the speed limit?" | M5 (time and day in every prompt) | Model on dev frames |
| Parking | "can I park here?", "what does the red curb mean?" | M5 | Model on dev frames |
| Road awareness | "is the road closed ahead?", "why is that car stopped?" | M5 (frames + tracker state) | Model on dev frames |
| Navigation by voice | "I'm going to Redmond Town Center" | M3 (router), M6 (geocoding + directions), M4 (voice) | Fixture + live Mapbox calls |
| Route questions | "is my exit this one or the next?", "did I miss my turn?", "what's after this light?" | M6 (route progress, reroute) | Fixture route + synthetic GPX |
| Voice controls | quiet mode, normal mode, repeat that, stop | M3 (logic), M4 (audio) | Router + speaker tests |
| Safety rules | never a bare "go"; says what it can't see; refuses split-second questions at speed | M3 (local phrasing + refusals), M5 (model prompt) | Router table + answer checks |
| Replay mode | real-time file playback + GPX, pause/skip, typed questions | M1 (pacing/seek), M6 (GPX sync), M7 (hotkeys) | FileSource tests, end-to-end run |
| Test sheet | accuracy and answer-time scores | M8 | Runner on the starter sheet |
| Screen | video + question and answer text | M7 | End-to-end run |

---

## M0: Core foundations
Config loading (`config.yaml` → validated, typed settings; keys from `.env`, never logged), structured
logging (console + JSON lines per run), run folders, a clock abstraction (video clock in file mode, wall
clock in camera mode), shared data types.

**Done when:** pytest covers config loading (defaults, overrides, invalid values with clear errors, env
vars, masked secrets); `python -m app --print-config` prints the resolved config with keys masked.

## M1: Video capture + detection/tracking + performance baseline
- `VideoSource` interface. `FileSource`: real-time pacing, drops frames when behind, pause/seek.
  `CameraSource`: talks to scrcpy-server directly (adb + scrcpy 4.1 protocol, H.264 decoded with PyAV). No
  window capture, low lag. Protocol parsing is unit-tested on a recorded stream; the real phone test is in M9.
- YOLO26s + ByteTrack via ultralytics on CUDA in half precision. Classes: car, truck, bus, motorcycle, stop sign,
  traffic light.
- Ego motion: optical flow on a downscaled frame with vehicle boxes masked out; "we're stopped" after 0.5 s.
- Headless runner logs FPS, GPU memory (torch + NVML) and per-stage latency (decode, detect+track, motion).

**Done when:** a headless real-time run of a dev clip processes ≥ 25 of 30 fps with p95 perception processing
time < 40 ms and p95 decode-to-result latency < 70 ms, and the model is verified on `cuda` in fp16; ego-stopped intervals match hand-labeled intervals on
3 dev clips (±0.5 s); FileSource pacing/seek tests pass.

## M2: World state + 4-way stop queue
- Per-car "stopped" with ego-motion compensation (so cars already waiting when we arrive count), position
  left / ahead / right, and a filter that keeps parked cars out of the queue.
- At-a-stop context: we stopped within N s of seeing a stop sign.
- Queue sorted by stop time. Stops within 1 s are a tie, and the car on the right goes first (a yield-to-the-right
  ordering that also handles 3-way ties; a 4-way tie is reported as such). Departures happen when a car moves out
  across the intersection. Brief track loss is not a departure: a lost car is re-linked by position.
- "Your turn" = every car ahead of us has departed → the event fires once. Caveats when part of the
  intersection is out of frame or a car is clipped by the frame edge.
- Traffic-light state tracker: classify each detected light's lit lamp (red / yellow / green / unknown) from
  its crop every frame, pick the lights that face us, and debounce red→green transitions (feeds "tell me when
  it's green").

**Done when:** pytest scenarios pass: arrival order, ties to the right (2- and 3-way), 4-way deadlock, departures,
your-turn fires exactly once, parked car excluded, occlusion recovery. On the dev 4-way-stop clip the queue log
matches my hand-labeled timeline. The light tracker reports the right state on labeled dev frames and catches the
red→green change on dev clips that have one.

## M3: Router, local answers, alert engine (text level)
- Router: voice controls (quiet / normal / repeat / stop / "I'm going to …"), right-of-way → local,
  "tell me when it's green" → light watch, route and lane → navigation, split-second questions at speed
  ("can I merge now?") → safe refusal, everything else → vision model. Tricky cases are handled explicitly (e.g. "can I go right on red?" is a sign question, not
  right-of-way).
- Local right-of-way phrasing per PLAN.md, safety rules built in (never a bare "go"; say what it can't see).
- Alert engine: once-per-event rules, priorities (safety interrupts others), quiet mode, repeat-last.

**Done when:** a pytest router table of ≥ 60 phrasings routes correctly; the alert rules fire once per event;
typed questions on a dev-clip replay get correct local answers.

## M4: Voice pipeline (audio)
- Mic → openWakeWord "Hey DrivePilot" (CPU; custom model trained locally in parallel from the start, with
  the stock "hey jarvis" model as a stand-in until it's ready) → end of speech after ~0.6 s of silence (Silero VAD) → faster-whisper on the GPU,
  with speculative decoding during the trailing silence to save time → router.
- Piper TTS streamed sentence by sentence; an interruptible player; a chime plus pre-rendered alert clips;
  safety alerts interrupt and the answer resumes afterwards.
- Mic-free test harness: spoken questions are generated with TTS (wake word + question, optionally mixed with
  road noise) and fed through wake word → STT → router.

**Done when:** on the synthetic set the wake word triggers ≥ 95% with 0 false wakes on 10 min of non-wake
speech/noise; ≥ 95% route correctly; measured end of question → first audio for local answers is < 1.0 s
(median and p90); alerts reach the speaker in < 0.3 s; whisper is verified on the GPU.

## M5: Vision-model answers
- One `VisionModel` interface (streaming text + structured JSON) with providers Claude Haiku (default,
  `claude-haiku-4-5`), Gemini (`gemini-3.8-flash`, low thinking), Ollama and Mock, selected by `vision_model`.
- Frame buffer (2–3 recent frames, plus a traffic-light crop when relevant); prompt with tracker state, nav
  step, time/day and speed; left/right from the driver's view; 1–2 sentences; never a bare command.
  Answers stream into TTS; timeouts give a spoken fallback.

**Done when:** pytest with a mocked provider covers the interface contract, prompt contents, streaming →
sentences, timeouts and switching providers by config. With the Claude key (verified working): sign / parking /
light / road / lane-count questions on dev frames are answered correctly, and median question end → first
audio is < 2.5 s.

## M6: Navigation
- Mapbox geocoding + directions client, plus a fixture-backed mock with the same interface. Route model:
  steps, lanes and roundabout exits.
- GPS: GPX replay synced to the video; live from a phone web page over `adb reverse`.
- Route progress: snap GPS to the route, distance to the next maneuver, off route > 50 m for 5 s → reroute
  and say so.
- Lane guidance (route lanes + the model's lane count/position + overhead sign reading), reminders at 1 mi and
  0.25 mi, roundabout exit counting with a "not this one" alert, route questions, destination by voice.

**Done when:** pytest covers snapping, progress, off-route/reroute, lane math and roundabout counting on
fixture routes + synthetic GPX; a replayed GPX along a fixture route fires the right alerts at the right
distances. With the token, real routes are fetched and saved as fixtures.

## M7: App integration + V1 screen + run logging
- `app.py` wires everything for both sources. The V1 page shows the video plus the question and answer as
  text (no overlays). Replay hotkeys: space = pause, ←/→ = skip 5 s, Q = type a question.
- Every run writes frames (1 fps + every frame sent to the model), detections, questions, answers, alerts
  and timings to `runs/<timestamp>/`.

**Done when:** the full app plays a dev clip end to end with the page live; synthesized spoken questions
get spoken answers; hotkeys work; the run folder is complete; a full-pipeline headless benchmark meets the
latency targets.

## M8: Test sheet, scoring, performance pass
- `eval/`: a test sheet in PLAN.md's format; the runner seeks to each row's time (with warm-up so the tracker
  has history), asks the question via text or the full voice path, has an AI judge mark it, keeps a manual
  review column, and reports accuracy per category plus median answer time.

**Done when:** the runner scores a starter sheet built from labeled dev-clip moments and writes the report;
every latency target is met, or the miss and its cause are written in PROGRESS.md.

## M9: Hardware bring-up and live drive (needs you)
scrcpy camera (is the ultrawide listed?), the phone GPS page, wake word and STT tuning on real in-car audio,
recorded drives with GPX, and the live drive.

**Done when (= PLAN.md's V1 definition):** on a recorded drive, the 4-way stop, lane, roundabout and sign
questions are each answered correctly by voice, and the same build works once on a live drive.
