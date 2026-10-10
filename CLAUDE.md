# CLAUDE.md: standing rules for this project

DrivePilot is a dashcam voice co-pilot for Banana Hacks 2026. PLAN.md is the full product plan; PROGRESS.md
is the live status. Read both before starting work.

## Scope
- **Build V1 only.** Everything tagged V2 in PLAN.md (overlays, YOLOPv2 lanes, drift warning, pedestrian and
  cyclist alerts, sign fine-tune, memory, answer highlight) is off limits until the user explicitly says so.
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
  provider. Gemini Flash is the default. Other providers are drop-ins and never leak into callers.
- **Voice input is wake word only.** No push-to-talk buttons or extra devices. (Typed questions exist only as
  the replay/dev tool described in PLAN.md "Replay mode".)
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
  Do not push unless the user has said to.
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
