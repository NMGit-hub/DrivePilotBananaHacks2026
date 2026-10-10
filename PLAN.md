# Dashcam Voice Co-Pilot — Banana Hacks 2026 Plan

Oct 9, 2026 · Bono

## Overview

A phone on the dash streams video to a laptop that watches the road, answers spoken questions in about 1–2 seconds, speaks up on its own when something matters, and highlights what it is talking about on screen.

**The pitch:** drivers get confused at the same moments every day — a 4-way stop where nobody knows who's next, an interchange with five lanes, a roundabout, a parking sign with three time windows. You ask out loud and get a spoken answer, with the car, lane or sign highlighted.

**Hackathon fit:** Banana Hacks 2026 runs online Oct 9–12, 2026. Judges score technical execution, innovation, use of the image AI theme, impact and presentation.

| Criterion | How this project scores |
| --- | --- |
| Technical execution | Local real-time tracking and lane detection on the GPU plus a vision-language model, working live in a car |
| Innovation | Hybrid design: instant answers from tracker state, model reasoning only where it's needed |
| Image AI theme | Everything runs on camera frames: detection, tracking, lane segmentation, sign reading |
| Impact | Fewer right-of-way mistakes, missed exits and parking tickets; helpful for new and nervous drivers |
| Presentation | Live-drive video with overlays and voice, plus an accuracy score from the test sheet |

**Name:** not chosen yet. This doc uses [wake word] as a placeholder.

## V1 and V2

V1 is the complete working co-pilot — phone camera, laptop, voice in and voice out, answering the core driving questions; V2 adds visualization, fine-tunes and extra features on top of it without changing V1.

**V1 — the base app (must work first)**

- Phone camera to laptop, live and from recorded drives (replay mode)
- Wake word, speech-to-text, spoken answers
- YOLO26s + ByteTrack car tracking and stop detection
- 4-way stop: who came first, can I go, and the spoken "you're up" alert
- Navigation: destination by voice, Mapbox route, phone GPS
- Which lane should I be in (route data + the model reading the frame and overhead signs)
- Roundabout exits and route questions (is it this exit, did I miss my turn, what's next)
- Any scene question through the vision model (Gemini Flash for now, swappable): signs, turn rules, traffic lights, parking, construction
- Test sheet with accuracy and answer-time scores
- Screen: plain camera video with your question and the answer as text, no overlays

**V2 — extras on top of V1**
**IMPORTANT** - DO NOT START UNTIL I TELL YOU ABOUT THE NEW PARKING FEATURE TO ADD HERE IN THE CLAUDE PROMPT (FOR V2 ONLY. CONINUTE WITH V1)

> Update 2026-10-10: the user described it (V2): ask DrivePilot to help find a parking spot (e.g. in a busy lot), and it
> tells you when it has found one. V2 is approved without V1's live drive. See HANDOFF.md.

- Visualization: all overlays (car badges, green target lane, red drift line, roundabout diagram, sign cards, curb highlights)
- YOLOPv2 lane detection, replacing the model's lane count with a per-frame one
- Lane drift warning
- Pedestrian and cyclist alerts
- Fine-tuned US sign detector: automatic speed-limit and school-zone alerts (Best Fine-tune entry)
- Memory: "what was that sign" with a thumbnail
- Answer highlight: the box the answer refers to pulses on screen

**V1 is done when:** on a recorded drive, the 4-way stop, lane, roundabout and sign questions are each answered correctly by voice, and the same build works once on a live drive. Only then start V2.

## Hardware and setup

Everything runs on the phone and laptop you already have, plus a phone mount and a way to power the laptop in the car. Questions are wake word only.

| Part | What it is | Role |
| --- | --- | --- |
| Laptop | Surface Laptop Studio 2, i7-13700H, RTX 4050 6 GB | Runs every model, the UI and the voice |
| Camera | OnePlus 12, ultrawide 114° (main 85°) | Dashcam feed and GPS |
| Link | USB-C cable, phone to laptop | Low-lag video, charges the phone |
| Mic | Laptop's dual far-field mics | Wake word and questions |
| Power | USB-C PD car charger (100 W) or an inverter | GPU work drains the laptop battery fast |

**Camera bridge**

1. Install scrcpy on the laptop and enable USB debugging on the phone.
2. Run `scrcpy --list-cameras` and check whether the ultrawide is listed.
3. If it is: `scrcpy --video-source=camera --camera-id=<ultrawide id> --camera-size=1920x1080`, and capture it in Python as a virtual camera or window.
4. If it is not: use DroidCam on the main lens. Cross traffic must then be a little further ahead to stay in frame.

**Mounting:** phone high and centered on the windshield, ultrawide lens unobstructed, level with the horizon. Keep the same spot for recorded and live drives so test footage matches the real thing.

**One-time calibration:** film a printed checkerboard with the ultrawide and run OpenCV camera calibration. That gives the numbers to undistort frames for the lane model.

## Architecture

Everything runs on the laptop except sign reading, general questions and routing, which call the cloud only when a local answer isn't possible.

```
 Phone (OnePlus 12)                      Microphone
 ultrawide video + GPS over USB          laptop mics, wake word only
          |                                       |
          v                                       v
 Perception (local GPU)                  Speech in (local)
 YOLO26s + ByteTrack tracking            wake word
 YOLOPv2 lanes, 10-15 fps                faster-whisper on GPU
 stop and motion detection                        |
          |                                       v
          v          tracker state
 World state  ------------------------>  Router
 stop queue, lanes, signs,               local / route / model  ----+
 people, route position                           |                 |
          |                                       | if not local    | local answer
          v                                       v                 |
 Alert engine                            Cloud (only when needed)   |
 drift, your turn, people                Gemini Flash (swappable) + frames
          |          \                   Mapbox route + lane data   |
          v           \  alerts                   |                 |
 Browser overlay       \                          v                 |
 V2: boxes, lanes,      +--------------> Speaker (local TTS)  <-----+
 badges                                  answers and alerts
```

The vision path (left) builds a live world state; the voice path (right) asks it first. Right-of-way questions skip the cloud and go straight to the speaker.

**Tech stack**

| Layer | Choice |
| --- | --- |
| Language and server | Python 3.11, FastAPI with WebSockets |
| Camera capture | scrcpy (or DroidCam) into OpenCV |
| Detection and tracking | `ultralytics` YOLO26s + ByteTrack, CUDA, half precision; RF-DETR-S tested against it in V2 on the test sheet |
| Lanes and drivable area (V2) | YOLOPv2 |
| Motion (are you stopped) | OpenCV optical flow on a downscaled frame |
| Wake word | openWakeWord (or Porcupine) |
| Speech-to-text | faster-whisper, small or distil model, GPU |
| Vision-language model | Swappable, set by `vision_model` in `config.yaml`. Default: Gemini Flash (free tier). Drop-ins: Claude Haiku (Anthropic API) or a local model through Ollama |
| Navigation | Mapbox Geocoding + Directions |
| Text-to-speech | Windows built-in voice or Piper; pre-recorded alert clips |
| Front end | One HTML page with two canvases |
| Recording | OBS Studio |

**Code layout for Claude Code**

- `capture/` — camera and video-file sources behind one interface; GPS live and GPX replay
- `perception/` — detector, tracker, lanes, motion, undistortion
- `state/` — world state: stop queue, lanes, signs, people, route position, frame buffer
- `voice/` — wake word, speech-to-text, router, text-to-speech, alert player
- `answers/` — local handlers, navigation handler, model handler with prompts, behind one interface so the provider is a config switch
- `nav/` — Mapbox client, route tracking, rerouting
- `ui/` — the overlay page
- `eval/` — test sheet runner and scoring
- `config.yaml` — source, paths, thresholds, `vision_model`, API keys from environment variables

## Replay mode and testing

Build against recorded drives first; the live camera and a video file feed the exact same pipeline, switched by one config setting.

**Recording your own drives**

- Same mount position as live. Ultrawide, 1080p, 30 fps.
- Run a GPS logger at the same time (GPSLogger on Android, GPX export, 1-second interval). Note the clock time when recording starts so video and GPS line up.
- Target footage: 4-way stops (with 1, 2 and 3 other cars), at least one roundabout, a multi-lane interchange (405 / 520), street parking with signs, a school zone, construction, a pedestrian crossing, and a gentle drift toward a lane line on an empty road.
- Daytime first. Add a dusk or rain clip later to show limits honestly.

**Public footage for tuning only:** BDD100K has dashcam clips with GPS and lane labels. Use it to tune the tracker and lane model. The submission video uses only your own footage.

**Replay mode**

- `config.yaml`: `source: camera` or `source: file`, plus `video_path` and `gpx_path`.
- Plays at real-time speed, replays GPS in sync, and accepts spoken or typed questions at any moment.
- Hotkeys: space = pause, arrow keys = skip 5 s, Q = type a question.
- Every run logs frames, detections, questions, answers and timings to a folder for review.

**Test sheet (accuracy score)**

| clip | time (s) | question | expected answer | type |
| --- | --- | --- | --- | --- |
| stop_01.mp4 | 42 | who came first? | white SUV on the left | right of way |
| i405_03.mp4 | 118 | which lane for 520 east? | two left lanes | lanes |
| park_02.mp4 | 15 | can I park here? | yes, 2 hours until 6pm | parking |

- A script runs every row, asks the question at that moment, and marks each answer right or wrong (an AI judge plus your review).
- Target 30–50 rows by Sunday. Run it after every change; the pitch shows the final score per category.
- Also time each answer (question end → first spoken word) and report the median.

## Voice system

Voice is the main interface: you ask, it answers, and it speaks up on its own for safety; the screen only backs up what it says.

**Pipeline**

1. **Wake word** — openWakeWord or Porcupine, running locally on the CPU, trained on your chosen name ([wake word]). A 2–3 syllable name with distinct sounds misfires least.
2. **Speech-to-text** — faster-whisper (small or distil model) on the GPU. Stops listening after ~0.6 s of silence.
3. **Router** — sends the question to the right handler (see Features): instant local answer, or a model call.
4. **Text-to-speech** — Windows built-in voice or Piper. Answers stream sentence by sentence, so speaking starts before the full answer is done.

**Two kinds of speech**

- **Answers** — only when you ask. Example: "[wake word], when can I go?" → "Two cars stopped before you. The white SUV on the left goes first, then the pickup across. You're third."
- **Alerts** — spoken on their own: drifting, your turn, pedestrian, lane change needed, roundabout exit, speed limit change.

**Rules for alerts**

- Common alerts are pre-recorded audio files, so they play instantly with no speech generation.
- A short chime plays before every alert.
- Each alert fires once per event: once per drift, once per new speed limit. It repeats only if the event happens again or keeps going.
- Priority: safety alerts (drift, pedestrian) interrupt anything; an interrupted answer resumes after.

**Voice controls**

| You say | It does |
| --- | --- |
| "[wake word], quiet mode" | Only safety alerts until turned off |
| "[wake word], normal mode" | All alerts back on |
| "[wake word], repeat that" | Replays the last message |
| "[wake word], I'm going to [place]" | Sets the destination for navigation |
| "[wake word], stop" | Stops speaking now |

**Latency targets:** instant local answers under 1 s from the end of your question; model answers under 2.5 s; alerts under 0.3 s from detection.

## Features

Eleven features, ordered by build priority; the first three carry the demo and the rest add breadth. Each heading is tagged V1 or V2. Each has three parts: how it works, what you hear, and what you see — every "On screen" part is V2; V1 shows only video and the question and answer as text.

**How questions are routed:** a fast keyword router sends right-of-way questions ("first", "go", "my turn") to the local tracker for an instant answer. Navigation questions go to the route data. Everything else goes to the vision model (Gemini Flash by default) with 2–3 recent frames, the tracker state, the current navigation step, time of day and speed. That model is told to answer in 1–2 sentences, use left/right from the driver's view, and never give a bare command.

### 1. 4-way stop: who came first, when can I go (V1)

**How it works**

- YOLO26s detects cars, trucks, buses and motorcycles every frame; ByteTrack gives each one a stable ID.
- YOLO's built-in "stop sign" class tells the app you're at a stop.
- You're stopped when the background barely moves (optical flow on a small copy of the frame) for 0.5 s.
- Another car is stopped when its box barely moves for 0.5 s. Its stop time and position (left, ahead, right) are logged.
- A car that moves out across the intersection is marked departed.
- The queue is sorted by stop time. If two cars stopped within 1 s of each other, the one on the right goes first.
- Your turn = every car ahead of you in the queue has departed.
- If part of the intersection is out of frame, it says so instead of guessing.

**You say / it says**

- "[wake word], who got here first?" → "The white SUV on the left, then the pickup across from you. You're third."
- "[wake word], can I go?" → "Not yet, the pickup is next. I'll tell you." Then, on its own: "You're up — check and go."
- Tie: "You and the car on your right stopped together. The car on the right goes first."

**On screen:** a box on each car with a badge (1st, 2nd), a timer since it stopped, and your own badge (YOU: 3rd) that turns green on your turn.

### 2. Lane drift warning (V2)

**How it works**

- YOLOPv2 finds lane lines and the drivable area, run at 10–15 fps on the undistorted frame (or its center crop).
- A one-time perspective transform turns the lane lines into a top-down view, so the app can measure how far you are from the lane center.
- Drift = slowly moving toward a line and getting close to crossing it. A fast sideways move is treated as an intentional lane change, since the app can't see your turn signal.
- Only active above about 25 mph (GPS speed) to avoid false alarms in parking lots and at slow city speeds.

**You say / it says:** on its own, "Careful, you're drifting right."

**On screen:** your lane is outlined; the line you're drifting toward turns red, and the screen edge on that side pulses.

### 3. Lane guidance: which lane should I be in (V1)

**How it works**

- The Mapbox route gives a list of lanes for each upcoming junction, left to right, with which ones are valid for your route (see Navigation).
- V1: the model counts the lanes in the frame and says which one you're in, counted from the left. V2: YOLOPv2 lane lines do this every frame, faster.
- The difference becomes "move one lane left" or "stay in this lane."
- If not every lane is visible, it falls back to the route data alone.
- For overhead signs, the model reads the sign and matches it to your route ("I-405 N, two left lanes").
- Reminders at about 1 mile and 0.25 mile before the junction.

**You say / it says**

- "[wake word], which lane for 520 east?" → "The two left lanes. You're in the middle lane, move one left."
- On its own: "In half a mile, move one lane left for 520 East."

**On screen:** the lane you need filled green with an arrow, other lanes dimmed. Fallback: a large "← 1 lane" arrow.

### 4. Roundabout exits (V1)

**How it works**

- Mapbox marks each roundabout step with the exit number to take.
- GPS position along the route tells the app which exits you have passed.
- The model can describe the right exit by a landmark it sees ("just past the gas station").

**You say / it says**

- "[wake word], which exit?" → "Second exit, toward Redmond Way."
- On its own, passing exit 1: "Not this one, next exit."

**On screen:** a small roundabout diagram in the corner with your exit highlighted, plus a marker on the real exit once it's visible.

### 5. Signs: what did that say, speed limit, turn rules (V1; auto alerts V2)

**How it works**

- On a question, the model reads the sign from the latest frames.
- V2: fine-tune a YOLO sign detector on a US sign dataset (Mapillary Traffic Sign Dataset or LISA) so speed limits and school zones are caught on their own. This doubles as a Best Fine-tune entry.
- The current time and day go in every prompt, so time-limited rules get the right answer.

**You say / it says**

- "[wake word], can I turn right on red here?" → "No — that sign says no turn on red, 7am to 7pm, and it's 5:40pm."
- "[wake word], what's the speed limit?" → "35."
- "[wake word], can I use the express lane?" → "It's a toll lane right now; you'd need a Good To Go pass."
- On its own (V2): "Speed limit is now 35."

**On screen:** a box around the sign and a readout card with its meaning; the speed limit stays as a badge in the corner.

### 6. Traffic lights (V1)

**How it works**

- YOLO's built-in "traffic light" class finds the signal; the app crops it.
- On a question, the crop and the full frame go to the model, which tells a left arrow from a full green.
- Glare and distance make this unreliable, so answers include what it can't see.

**You say / it says:** "[wake word], is the left arrow green?" → "Yes, green arrow for the left lane. Oncoming traffic still has a red."

**On screen:** the signal boxed, with its state as a label.

### 7. Parking: can I park here (V1)

**How it works**

- The model reads parking signs, curb paint, hydrants and spot markings in the latest frames.
- Current time and day go in the prompt, so it can work out time windows ("2 hr 8am–6pm, except Sunday").

**You say / it says**

- "[wake word], can I park here?" → "Yes, 2-hour parking until 6pm. It's 4:15, so you have until 6."
- "[wake word], what does the red curb mean?" → "No stopping or parking at any time."
- "[wake word], is this spot EV-only?" → "Yes, EV charging only while charging."

**On screen:** the curb highlighted green (allowed) or red (not allowed), with the rule as a short label.

### 8. Pedestrians and cyclists (V2)

**How it works**

- YOLO's "person" and "bicycle" classes, tracked like cars.
- An alert fires when one enters your path zone (the drivable area ahead, from YOLOPv2) while you're moving or about to move.

**You say / it says:** on its own, "Pedestrian stepping into the crosswalk on your right." Asked: "[wake word], is that person crossing?" → "They're waiting at the curb, facing the street."

**On screen:** pedestrians and cyclists outlined in amber; red when inside your path.

### 9. Road awareness (V1)

**How it works:** model questions over recent frames, with tracker state added.

**You say / it says**

- "[wake word], is the road closed ahead?" → "Right lane is closed by cones, about 200 feet ahead. Left lane is open."
- "[wake word], what does that construction sign want?" → "Merge left; flagger ahead."
- "[wake word], why is that car stopped?" → "Hazards are on and it's pulled half onto the shoulder — looks broken down."

**On screen:** the thing it's describing is boxed and labeled.

### 10. Memory: what was that (V2)

**How it works**

- A rolling buffer keeps one frame per second for the last 60 s, plus a crop of every sign the detector saw.
- "What was that" questions go to the model with those frames and crops.

**You say / it says:** "[wake word], what was the sign we just passed?" → "Speed limit 30, school zone when flashing — it wasn't flashing."

**On screen:** a thumbnail of the frame it answered from pops up in the side panel.

### 11. Route questions (V1)

**How it works:** answered straight from the Mapbox route and GPS, no camera needed (see Navigation).

**You say / it says**

- "[wake word], is my exit this one or the next?" → "The next one, in 1.2 miles."
- "[wake word], did I miss my turn?" → "Yes. Take the next right; I've rerouted."
- "[wake word], what's after this light?" → "Turn left on 148th Avenue in a quarter mile."

**On screen:** the top banner shows the next step and distance.

## Visualization (V2)

V1 shows plain video plus the question and answer as text. In V2, a browser page on the laptop draws every overlay on top of the live video, so viewers see what the co-pilot sees while it talks.

**How it's drawn**

- The Python backend sends each frame (JPEG) plus a small JSON packet over a WebSocket: boxes and IDs, badges, lane polygons, alerts, the nav step, the last question and answer.
- The page draws the video on a canvas and the overlays on a second canvas above it.
- Lane overlays update at 10–15 fps and are smoothed between updates, so they don't flicker.
- The same page works in replay and live mode; OBS records it for the submission.

**Screen layout**

| Area | Shows |
| --- | --- |
| Top banner | Next navigation step and distance ("In 0.5 mi, 2nd exit") |
| Main video (~75% width) | Camera feed with all overlays |
| Top-right corner | Speed limit badge, current speed, roundabout mini-diagram when relevant |
| Side panel | Your question as text, the answer as text, memory thumbnails |
| Bottom strip | Status: listening, thinking, quiet mode, FPS, latency of the last answer |

**Color meanings** (keep consistent everywhere)

| Color | Means |
| --- | --- |
| Green | Where you should be, your turn, allowed |
| Red | Danger: drift line, person in your path, not allowed |
| Amber | Watch this: pedestrians, cyclists, the car that goes next |
| White / gray | Tracked but nothing to do |

**Overlay list:** 4-way stop badges and timers · target lane fill and arrow · drift line in red and edge pulse · roundabout diagram and exit marker · sign box and readout card · signal state label · curb highlight with rule · pedestrian and cyclist outlines · road-awareness box and label · memory thumbnail · nav banner.

**Answer highlight:** when a model answer mentions something on screen ("the white SUV on the left"), it returns which tracked ID or region it meant, and the page pulses that box for 2 seconds.

## Navigation

Mapbox Directions supplies the route, lane-by-lane guidance and roundabout exit numbers; the phone supplies GPS; the camera confirms what the route says.

**Getting the route**

1. You say "[wake word], I'm going to [place]."
2. Mapbox Geocoding turns the place into coordinates.
3. Mapbox Directions (driving-traffic profile, with steps, banner instructions and lane data turned on) returns the route.
4. The app keeps the step list: each maneuver, its distance, its lanes (left to right, with which are valid for your route) and, for roundabouts, the exit number.

**GPS from the phone**

- Live: a tiny web page on the phone sends its location to the laptop once per second. Run `adb reverse tcp:8000 tcp:8000` over the same USB cable so the phone reaches the laptop as `localhost`, which browsers allow location access for.
- Fallback: an Android app that shares GPS over the network (NMEA stream).
- Replay: the GPX file recorded with the clip plays back in sync.

**Tracking progress**

- Each GPS point is snapped to the route line to get distance to the next maneuver.
- More than about 50 m off the route for 5 s → reroute and say so.
- Speed comes from GPS and gates features (drift warnings above 25 mph).

**Combining route and camera**

- Lanes: route says which lanes are valid; lane detection says which lane you're in; the difference is the instruction.
- Overhead signs: the model reads them and checks they match the route, which covers places where Mapbox lane data is missing.
- Roundabouts: GPS counts exits passed; the camera marks the exit to take once it's in view.

**Cost:** Mapbox's free tier covers far more requests than a weekend of testing needs. Check current limits when you sign up.

## Safety and scope limits

The co-pilot advises and the driver decides; that line is built into every answer and is part of the pitch.

**Built-in rules**

- Never a bare "go." Always what it sees plus "check before you go."
- Says what it can't see: "I can't see the right side of the intersection."
- No answers that need a split-second decision at speed, such as "can I merge now?" at highway speed. It replies that it can't judge that safely.
- Everything while driving is voice-only. The laptop is set up before leaving and not touched while moving.

**For the live drive**

- A passenger runs the laptop and OBS if possible.
- Quiet route, daytime, good weather, chosen in advance.
- Phone mount and cables secured so nothing falls or blocks the view.

**Out of scope this weekend:** night and rain accuracy, side and rear cameras, turn-signal input, controlling the car in any way.

## Build schedule and fallback

V1 is built Friday night and Saturday and must work end to end before any V2 work starts; V2 fills Sunday, and Monday is for the live drive and submission.

**Friday night (once home) — V1 foundation**

- [ ] Python environment with CUDA PyTorch; confirm the RTX 4050 is used
- [ ] scrcpy camera test: is the ultrawide listed?
- [ ] Record 2–3 short test drives with GPS logging (4-way stop, a busy street)
- [ ] Replay mode: a clip plays with YOLO26s + ByteTrack running

**Saturday — finish V1**

- [ ] Ego-stopped and car-stopped detection, the stop queue
- [ ] Wake word, faster-whisper, keyword router, spoken answers
- [ ] "Who came first / can I go" answered locally; "you're up" alert
- [ ] Vision model handler (Gemini Flash, swappable): signs, parking, lights, road questions
- [ ] Mapbox route, phone GPS, lane guidance, roundabout exits, route questions
- [ ] Simple screen: video plus question and answer text
- [ ] Start the test sheet
- [ ] Evening: one live drive with V1 — V1 is done

**Sunday — V2**

- [ ] Overlay page: car badges, target lane, roundabout diagram, sign cards
- [ ] YOLOPv2 lanes, undistortion, drift warning
- [ ] Pedestrian and cyclist alerts, memory
- [ ] Sign detector fine-tune, if time allows
- [ ] Full set of test drives; test sheet to 30–50 rows and run it

**Monday — ship**

- [ ] Final live drive, recorded with OBS
- [ ] Edit the demo video and write the submission
- [ ] Submit with time to spare

**Fallback if behind:** V1 alone is a complete submission. Cut V2 items from the bottom of the Sunday list first; keep a V2 feature in the video only if it works.

## Demo video and submission

The video is a live drive recorded with OBS, about 3 minutes, built around the 4-way stop moment.

**Shot list**

1. **Hook (15 s):** a 4-way stop, three cars, everyone hesitating. "[wake word], who got here first?" — answer plus badges on screen.
2. **Problem (20 s):** one line each on confusing stops, lanes, roundabouts and parking signs.
3. **Live drive (90 s):** "your turn" alert, a lane guidance prompt with the green lane, a roundabout exit, a parking sign question, and a drift warning if V2 is done.
4. **How it works (30 s):** the architecture diagram; local tracking for instant answers, the vision model for reasoning.
5. **Results (20 s):** test-sheet accuracy per category and median answer time.
6. **Close (15 s):** safety framing (advises, never decides) and what's next.

**Recording tips**

- OBS captures the browser page with the system audio, so the spoken answers are in the video.
- Add a small picture-in-picture of a second phone or camera filming the windshield, so viewers believe it's live.
- Take several runs of the route and keep the best take of each moment.

**Submission checklist**

- [ ] Video uploaded
- [ ] Repo with README, setup steps and a short sample clip for judges to run replay mode
- [ ] Write-up: problem, how it works, results, limits, safety
- [ ] Category entries chosen (main prize; Best Fine-tune if the sign detector is done; People's Choice)

## Risks and open decisions

The biggest risks are the camera's view of cross traffic and lane detection through the ultrawide lens; both have fallbacks.

| Risk | Effect | Fallback |
| --- | --- | --- |
| Ultrawide not available to scrcpy | Side cars at 4-way stops out of frame | DroidCam on the main lens; stop a little further back; say when it can't see |
| Lane model confused by ultrawide distortion | Wrong lane count, false drift alerts | Undistort, or feed the center crop only |
| Road noise beats the wake word | Missed questions | Tune the wake-word sensitivity on recorded in-car audio; windows up, music off |
| GPU too slow running everything at once | Laggy overlays | Lanes at lower fps; YOLO26n instead of 26s; smaller input size |
| Mapbox has no lane data for a junction | No lane guidance there | Read the overhead sign with the model |
| Model answer too slow on a bad connection | Answers over 3 s | Phone hotspot with good signal; local answers still work offline |
| Laptop battery dies mid-drive | Lost take | 100 W USB-C car charger or inverter |

**Open decisions**

- [ ] Name for the co-pilot (the wake word)
- [ ] Ultrawide via scrcpy or main lens via DroidCam (decided by tonight's test)
- [ ] Submission deadline time on Monday (check the event schedule)
- [ ] Whether to attempt the sign-detector fine-tune for Best Fine-tune
