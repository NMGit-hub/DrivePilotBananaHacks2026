# PROGRESS

Live status of the V1 build. Updated at every milestone. Newest information first within each section.

## Done

### M0: core foundations (2026-10-10)
- `core/config.py`: `config.yaml` → strict, immutable pydantic settings. Unknown keys and uncoerced types fail
  with the dotted field path; YAML 1.2 scalar rules; dotted overrides (`--set video.source=camera`); secrets
  from env/.env as `SecretStr` (real env wins over .env); `gps_source` resolution; non-UTF-8 files → clear error.
- `core/logging_setup.py`: structured logger (any field name), console + JSON lines, secret redaction
  everywhere (re-setup keeps it), numpy/NaN-safe JSON.
- `core/clock.py`: `WallClock` (camera) and thread-safe pausable/seekable `MediaClock` (file playback).
- `core/runlog.py`: per-run review folder (config, events, detections, perf, frames). Text never blocks or drops;
  frames drop when the writer is behind; clean drain on close/atexit.
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
- `requirements.txt` (pinned top-level) + `requirements-lock.txt` (full freeze).
- Dev footage: BDD100K validation clips in `data/videos/` (gitignored). See "Test data" below.

## Verified (and how)
- **M0:** 64 pytest tests pass (config, clocks, logging, run folder incl. stalled-writer and concurrent-close
  cases, Frame, CLI); ruff and mypy clean. `python -m app --print-config` shows keys only as set/missing.
- **API keys work** (live calls, 2026-10-09): Anthropic `claude-haiku-4-5` answered a text prompt in 1.06 s;
  the Mapbox public token works for Geocoding v6 and Directions v5 `driving-traffic` (a 12.7 km Redmond →
  Seattle route came back with lane data on 116 intersections). Keys are in `.env` (gitignored).
- **Phone camera list** (user ran `scrcpy --list-cameras`, OnePlus CPH2583, Android 16): ids 0 and 5 are logical
  back cameras with zoom 0.6–10 (0.6 = ultrawide FOV); id 3 (4000x3000) is probably the physical ultrawide.
  Not yet verified with a captured frame (the phone was unplugged afterwards).
- **GPU actually used:** `scripts/check_gpu.py`
  - fp16 matmul on `cuda:0` at ~21 TFLOPS
  - YOLO26s predictor weights on `cuda:0` in float16; ~29 ms/frame end-to-end at 1080p, measured while
    another GPU job was running, so this is a rough number (a proper benchmark is in milestone M1)
  - CTranslate2 4.8.2 sees 1 CUDA device with float16 support
- **requirements.txt installs cleanly** into a fresh Python 3.11 venv with the README command, and all key
  imports work with `torch.cuda.is_available() == True`.

## Next
- M1: video sources (file + scrcpy camera), YOLO26s + ByteTrack, ego motion, headless benchmark.
- In the background: "Hey DrivePilot" wake-word training.

## Test data
- Source: BDD100K `videos/100k/val` clips (720p, 30 fps, ~40 s, US cities), pulled by HTTP range requests
  from a byte-for-byte Hugging Face mirror of the official zip (`linxxx3/bdd100k_videos`). Only individual
  clips are downloaded, not the 1.97 TB archive. Development/tuning only. Replace with our own drives.
- BDD100K's GPS/IMU "info" files are not in that mirror, so dev clips have **no GPX**. The GPX replay path is
  tested with synthetic GPX files in tests until real drives are recorded.

## Known issues / compromises
- scrcpy 4.1 (with adb) installed via winget; live camera capture not yet tested (needs the phone plugged in).
- The API keys were shared in chat; consider rotating them after the hackathon.
