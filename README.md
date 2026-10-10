# DrivePilot: dashcam voice co-pilot (Banana Hacks 2026)

A phone on the dash streams video to a laptop. The laptop watches the road, answers spoken questions in
about 1–2 seconds, and speaks up on its own when something matters: who goes next at a 4-way stop, which lane
to be in, which roundabout exit, what a sign means, whether you can park here.

- **Local and instant:** YOLO26s + ByteTrack on the GPU track every car. Right-of-way questions are answered
  from tracker state with no cloud call.
- **Model only when needed:** sign, parking, traffic-light, lane and road questions go to a vision-language
  model (Claude Haiku by default; Gemini Flash or a local Ollama model via `config.yaml`) with the latest frames.
- **Navigation:** Mapbox route + phone GPS for lane guidance, roundabout exits and route questions.

See [PLAN.md](PLAN.md) for the full design and [PROGRESS.md](PROGRESS.md) for current status.
This repository is building **V1** (the base app); V2 extras come later.

## Requirements

- Windows 11 with an NVIDIA GPU and a recent driver (developed on an RTX 4050 Laptop GPU, 6 GB)
- [uv](https://docs.astral.sh/uv/) (installs Python 3.11 for you) and git
- For live camera mode: an Android phone with USB debugging, plus [scrcpy](https://github.com/Genymobile/scrcpy)
  (it bundles `adb`)
- API keys: Anthropic (vision model) and Mapbox (navigation). Without keys the app still runs. Local answers
  work, and the cloud features say that they are not available.

## Setup

```powershell
git clone https://github.com/NMGit-hub/DrivePilotBananaHacks2026.git
cd DrivePilotBananaHacks2026

# 1. Python 3.11 virtual environment
uv python install 3.11
uv venv --python 3.11 .venv

# 2. Dependencies (CUDA 12.6 PyTorch comes from the PyTorch index listed in requirements.txt)
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --index-strategy unsafe-best-match

# 3. Check that the GPU is actually used (PyTorch, YOLO26s and CTranslate2)
.venv/Scripts/python scripts/check_gpu.py

# 4. Build the TensorRT engine for the detector (one-time per machine, ~10 min; uses its own build venv)
.venv/Scripts/python scripts/build_engine.py

# 5. API keys
copy .env.example .env
#    then edit .env and fill in ANTHROPIC_API_KEY and MAPBOX_ACCESS_TOKEN
```

Without uv: create a Python 3.11 venv any way you like, then run
`python -m pip install -r requirements.txt` (pip reads the extra PyTorch index from the file).
`requirements-lock.txt` pins every transitive package of the verified environment.

Model weights (YOLO26s, the speech-to-text model) download into `models/` on first run. The Piper voice
(`en_US-lessac-medium.onnx` and its `.onnx.json`, from the `rhasspy/piper-voices` repository on Hugging Face)
goes in `models/piper/` (see `voice.tts.voice` in config.yaml). The wake-word model is in the repo.

## Running

```powershell
# Replay a recorded drive (optionally with its GPX track), with voice and the screen:
.venv/Scripts/python -m app --video data/videos/dev/red_light_turns_green_sf__b97b9ee8-612c36ef.mov

# Live: the phone camera over USB (scrcpy), GPS from the phone's browser
adb reverse tcp:8000 tcp:8000
.venv/Scripts/python -m app --camera
```

- The screen is at <http://localhost:8000/>: the video, plus your question and the answer as text.
  Hotkeys: **space** pause, **←/→** skip 5 s, **Q** type a question.
- Say **"Hey DrivePilot"**, then the question ("who got here first?", "which lane for 520 east?",
  "I'm going to Redmond Town Center", "tell me when it's green", "quiet mode", "repeat that", "stop").
- On the phone, open <http://localhost:8000/gps> and tap Start: it sends the location once a second.
- Options: `--no-voice` (typed questions only), `--no-audio` (answers on the screen only), `--gpx track.gpx`,
  `--ask "22.5=who got here first?"` (a typed question at a video time), `--exit-at-end`,
  `--set key=value` (any config value), `--print-config`.
- Every run writes a review folder under `runs/`: log, questions, answers, alerts, world events, GPS,
  detections, frames (one a second plus every frame sent to the vision model) and `summary.json` with
  latencies.

For drives: plug the laptop into its own charger and set Windows to **Best performance** power mode.

## Tools

```powershell
.venv/Scripts/python scripts/fetch_dev_clips.py        # download the public dev clips (BDD100K)
.venv/Scripts/python scripts/bench_app.py --silent     # whole app + spoken questions: latencies vs targets
.venv/Scripts/python scripts/bench_pipeline.py         # perception FPS / latency / GPU memory vs targets
.venv/Scripts/python scripts/voice_eval.py             # wake word + Whisper + routing on synthetic voices
.venv/Scripts/python scripts/eval_ego_motion.py        # "are we stopped?" vs hand labels
.venv/Scripts/python scripts/eval_traffic_lights.py    # traffic-light reading vs hand labels
.venv/Scripts/python scripts/ask_model.py --suite eval/model_questions.yaml   # vision model on dev frames
.venv/Scripts/python scripts/fetch_route.py --from 47.6158,-122.2037 --to "Redmond Town Center"
```

## Configuration

Everything lives in [config.yaml](config.yaml); API keys come from `.env`. The most important settings:

| Setting | Meaning |
| --- | --- |
| `video.source` | `camera` (phone via scrcpy) or `file` (a recorded drive). Same pipeline for both. |
| `video.file.path`, `video.file.gpx_path` | The recorded drive and its optional GPX track (file mode plays at real-time speed). |
| `vision_model` | Which vision-language provider answers scene questions: `claude` (default, Claude Haiku), `gemini`, `ollama` or `mock`. |
| `nav.home` | The address for "take me home". |
| `voice.wakeword.threshold` | Wake-word sensitivity (0.25; lower hears more, with more false wakes). |

## Tests

```powershell
.venv/Scripts/python -m pytest
```

The tests need no camera, microphone or API keys (a few use the GPU and are marked `gpu`).

## Test footage

Your own drives go in `data/videos/` (gitignored). For development the repo uses a few public
[BDD100K](https://bdd-data.berkeley.edu/) validation clips. Those are not committed and are only for tuning.
The submission video uses only our own footage.
