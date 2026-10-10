# PROGRESS

Live status of the V1 build. Updated at every milestone. Newest information first within each section.

## Done

### Step 1: project setup (2026-10-09)
- CLAUDE.md with the standing rules; .gitignore; .env.example; package folders per PLAN.md "Code layout"
  (+ `core/`, `tests/`, `scripts/`); `config.yaml` skeleton; README with setup steps.
- Python 3.11.15 venv in `.venv` (uv-managed CPython, not the Microsoft Store build).
- `requirements.txt` (pinned top-level) + `requirements-lock.txt` (full freeze).
- Dev footage: BDD100K validation clips in `data/videos/` (gitignored). See "Test data" below.

## Verified (and how)
- **GPU actually used:** `scripts/check_gpu.py`
  - fp16 matmul on `cuda:0` at ~21 TFLOPS
  - YOLO26s predictor weights on `cuda:0` in float16; ~29 ms/frame end-to-end at 1080p, measured while
    another GPU job was running, so this is a rough number (a proper benchmark is in milestone M1)
  - CTranslate2 4.8.2 sees 1 CUDA device with float16 support
- **requirements.txt installs cleanly** into a fresh Python 3.11 venv with the README command, and all key
  imports work with `torch.cuda.is_available() == True`.

## Next
- Write the V1 milestone plan and get the user's OK (PLAN step 2).

## Test data
- Source: BDD100K `videos/100k/val` clips (720p, 30 fps, ~40 s, US cities), pulled by HTTP range requests
  from a byte-for-byte Hugging Face mirror of the official zip (`linxxx3/bdd100k_videos`). Only individual
  clips are downloaded, not the 1.97 TB archive. Development/tuning only. Replace with our own drives.
- BDD100K's GPS/IMU "info" files are not in that mirror, so dev clips have **no GPX**. The GPX replay path is
  tested with synthetic GPX files in tests until real drives are recorded.

## Known issues / compromises
- Git identity was not configured on this machine. A repo-local identity (`NMGit-hub`, matching the
  existing GitHub commit) is set; it needs the user's confirmation.
- `scrcpy`/`adb` are not installed yet (needed only for live camera mode).
