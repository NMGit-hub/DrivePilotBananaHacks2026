# DrivePilot: dashcam voice co-pilot (Banana Hacks 2026)

A phone on the dash streams video to a laptop. The laptop watches the road, answers spoken questions in
about 1–2 seconds, and speaks up on its own when something matters: who goes next at a 4-way stop, which lane
to be in, which roundabout exit, what a sign means, whether you can park here.

- **Local and instant:** YOLO26s + ByteTrack on the GPU track every car. Right-of-way questions are answered
  from tracker state with no cloud call.
- **Model only when needed:** sign, parking, traffic-light and road questions go to a vision-language model
  (Gemini Flash by default, swappable in `config.yaml`) with the latest frames.
- **Navigation:** Mapbox route + phone GPS for lane guidance, roundabout exits and route questions.

See [PLAN.md](PLAN.md) for the full design and [PROGRESS.md](PROGRESS.md) for current status.
This repository is building **V1** (the base app); V2 extras come later.

## Requirements

- Windows 11 with an NVIDIA GPU and a recent driver (developed on an RTX 4050 Laptop GPU, 6 GB)
- [uv](https://docs.astral.sh/uv/) (installs Python 3.11 for you) and git
- For live camera mode: an Android phone with USB debugging, plus [scrcpy](https://github.com/Genymobile/scrcpy)
  (it bundles `adb`)
- API keys: Gemini (vision model) and Mapbox (navigation). Without keys the app still runs. Local answers
  work, and the cloud features report that they are not configured.

## Setup

```powershell
git clone https://github.com/NMGit-hub/DrivePilotBananaHacks2026.git
cd DrivePilotBananaHacks2026

# 1. Python 3.11 virtual environment
uv python install 3.11
uv venv --python 3.11 .venv

# 2. Dependencies (CUDA 12.6 PyTorch comes from the PyTorch index listed in requirements.txt)
uv pip install --python .venv\Scripts\python.exe -r requirements.txt --index-strategy unsafe-best-match

# 3. Check that the GPU is actually used (PyTorch, YOLO26s and CTranslate2)
.venv\Scripts\python scripts\check_gpu.py

# 4. API keys
copy .env.example .env
#    then edit .env and fill in GEMINI_API_KEY and MAPBOX_ACCESS_TOKEN
```

Without uv: create a Python 3.11 venv any way you like, then run
`python -m pip install -r requirements.txt` (pip reads the extra PyTorch index from the file).
`requirements-lock.txt` pins every transitive package of the verified environment.

Model weights (YOLO26s, the speech-to-text model, voices) download into `models/` on first run. That folder
is gitignored.

## Configuration

Everything lives in [config.yaml](config.yaml). The most important settings:

| Setting | Meaning |
| --- | --- |
| `video.source` | `camera` (phone via scrcpy) or `file` (a recorded drive). Same pipeline for both. |
| `video.video_path`, `video.gpx_path` | The recorded drive and its optional GPX track (file mode). File mode plays at real-time speed. |
| `vision_model` | Which vision-language provider answers scene questions: `gemini` (default), `claude`, `ollama` or `mock`. |

## Running

Run instructions are added here as each milestone lands (see PROGRESS.md).

## Tests

```powershell
.venv\Scripts\python -m pytest
```

The tests need no camera, microphone or API keys.

## Test footage

Your own drives go in `data/videos/` (gitignored). For development the repo uses a few public
[BDD100K](https://bdd-data.berkeley.edu/) validation clips. Those are not committed and are only for tuning.
The submission video uses only our own footage.
