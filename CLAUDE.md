# CLAUDE.md: standing rules for this project

DrivePilot is a dashcam voice co-pilot for Banana Hacks 2026. PLAN.md is the full product plan; PROGRESS.md
is the live status. Read both before starting work.

## Scope
- **V2 is approved (user, 2026-10-10)**, without waiting for V1's live drive (deferred until the user can
  drive). V2 = everything tagged V2 in PLAN.md (overlays, YOLOPv2 lanes, drift warning, pedestrian and cyclist
  alerts, sign fine-tune, memory, answer highlight) plus the new "find me a parking spot" feature. See
  HANDOFF.md for the decisions, the parking feature and open problems.
- V1 is the whole co-pilot, not just the 4-way stop: lane guidance ("am I in the right lane?"), roundabout exits,
  traffic lights (incl. "tell me when the light is green", added by the user), signs, parking, road questions,
  navigation by voice and route questions. BUILD_PLAN.md has the feature → milestone → test table.
- V1 screen = plain camera video + the question and answer as text. No overlays.

## Platform
- Windows 11, Surface Laptop Studio 2, i7-13700H, RTX 4050 Laptop GPU (6 GB, compute 8.9), Python 3.11.
- Use CUDA, and verify the GPU is *actually used* (check the device of tensors and models, and GPU memory and
  utilization while running), not just that `torch.cuda.is_available()` is true.
- The venv lives in `.venv` (created with `uv`, Python 3.11). Run things with `.venv/Scripts/python`.

## Architecture rules
- **Video input is one config setting:** `video.source: camera` (phone via scrcpy) or `video.source: file`
  (recorded drive + optional GPX). Same pipeline for both; file input plays at real-time speed.
- **Detection:** YOLO26s + ByteTrack through the `ultralytics` library.
- **Vision model:** behind one interface (`answers/vision/`); `vision_model` in `config.yaml` picks the
  provider. **Claude Haiku (`claude-haiku-4-5`) is the default** (user switched from Gemini on 2026-10-09);
  Gemini Flash, Ollama and a mock stay available as drop-ins and never leak into callers.
- **Voice input is wake word only:** "Hey DrivePilot" (custom openWakeWord model trained locally, in
  `scripts/wakeword/`). No push-to-talk buttons or extra devices. (Typed questions exist only as
  the replay/dev tool described in PLAN.md "Replay mode".) One exception (user, 2026-10-10): right after
  DrivePilot itself asks a yes/no question (the V2 parking-lot offer), it listens a few seconds for a plain
  "yes" / "no" without the wake word.
- **Secrets** live in `.env` (gitignored). `.env.example` documents them. Never commit keys, and never print
  them in logs.
- Config values go in `config.yaml`, not hard-coded numbers.

## Code quality bar
- Type hints everywhere; docstrings on public functions and classes; structured logging (no bare `print`
  outside CLI entry points).
- pytest tests for all logic testable without hardware. Run `.venv/Scripts/python -m pytest` before each commit.
- Verify by running things. Never claim something works because it should. Run it and check the output.
- Hard problems get solved properly: no stubs, faked results, or quietly lowered bars. Any compromise goes in
  PROGRESS.md under "Known issues / compromises" and is reported to the user.
- At the end of each milestone: strict self-review (plus an independent subagent review), fix findings,
  re-run tests, then commit.

## Workflow
- **Commit after every piece that works**, with clear messages. Work on `main` (tracks `origin/main`).
  **Push to GitHub (`origin main`) at the end of every milestone.** Git identity: NMGit-hub (confirmed).
- **Keep PROGRESS.md updated:** what's done, what's verified and how, what's next, known issues.
- Only stop for things that need the user: physical actions (phone, scrcpy, mic, driving), missing API keys
  (build against a mock until provided), or genuine decisions. Batch all asks into one message with exact steps.

## Layout
- `capture/`: camera and video-file sources behind one interface; GPS live and GPX replay
- `perception/`: detector + tracker, motion (ego stopped), undistortion
- `state/`: world state: stop queue, signs, route position, frame buffer
- `voice/`: wake word, speech-to-text, router, text-to-speech, alert player
- `answers/`: local handlers, navigation handler, vision-model handler + prompts, provider interface
- `nav/`: Mapbox client, route tracking, rerouting
- `ui/`: the V1 page (video + question/answer text)
- `eval/`: test-sheet runner and scoring
- `core/`: shared config loading, logging, clock, and data types
- `tests/`: pytest suite (no hardware needed)
- `config.yaml`: all settings; API keys come from environment variables (`.env`)
