# Prompt for the sign fine-tune session (V2-M6)

You're working on DrivePilot (Banana Hacks 2026), a dashcam voice co-pilot, **in parallel with another Claude
session**. Your job is one V2 milestone: **V2-M6, a fine-tuned US traffic-sign detector** (the hackathon's "Best
Fine-tune" entry). The other session builds the rest of V2 on `main` and will integrate your detector into the app.

Read first: `CLAUDE.md` (the standing rules apply to you too), `V2_BUILD_PLAN.md` (V2-M6 and "Decisions"),
`HANDOFF.md` section 6 (working notes for this machine), `PROGRESS.md` "Known issues" (sign reading by the vision
model is the weakest test-sheet category: 2/6), `perception/detector.py` (how YOLO26s runs as a TensorRT FP16
engine) and `scripts/build_engine.py` (why engine export needs a separate `.venv-trtbuild`).

## Ground rules for working next to the other session
1. **Your own branch and worktree.** Work on branch `v2-signs` in a separate git worktree. If the app didn't
   already start you in one, run `git worktree add ..\DrivePilot_signs -b v2-signs` from
   `C:\ClaudeProjects\DrivePilot_BananaHacks2026` and work only there. Never commit to `main`; never edit files in
   the main checkout (one exception, rule 7). Commit after each working piece and push your branch
   (`git push -u origin v2-signs`). The other session merges it.
2. **Python:** use the main venv by absolute path,
   `C:\ClaudeProjects\DrivePilot_BananaHacks2026\.venv\Scripts\python.exe` (ultralytics 8.4.175, torch
   2.14.1+cu126). Don't install or upgrade anything in it (an auto-install once broke the voice stack's
   onnxruntime). If you need extra packages, make your own venv with uv (e.g. `.venv-signs` in your worktree). Set
   `YOLO_AUTOINSTALL=false` for every ultralytics run. Base weights:
   `C:\ClaudeProjects\DrivePilot_BananaHacks2026\models\yolo26s.pt`.
3. **Shared GPU (RTX 4050 Laptop, 6 GB) and CPU.** The other session runs the app, tests and benchmarks at the
   same time (about 2 GB of GPU memory). Keep training under ~3.5 GB (choose the batch size and check with
   `nvidia-smi`), dataloader `workers` ≤ 4, no RAM caching of the whole dataset, and save every epoch so a run can
   be stopped and resumed (`resume=True`). Tell the user when a long training run starts and ends: latency
   benchmarks in the other session aren't valid while it runs. Before trusting timings, check the machine isn't
   throttled (HANDOFF.md section 3 item 2: the charger in, Windows power mode "Best performance").
4. **Disk:** C: has ~80 GB free and the other session needs room too. Keep datasets + runs under ~30 GB, delete
   archives after extracting, and keep data under your worktree's `data/` (gitignored).
5. **Shared files are append-only for you:** a new top-level `signs:` section at the END of `config.yaml`; a
   `SignsConfig` class and a `signs: SignsConfig` field appended at the end of `AppConfig` in `core/config.py`;
   a `.gitignore` exception so `models/signs/metrics.json` is tracked. Don't edit other existing files (`app.py`,
   `state/`, `answers/`, `ui/`, `PROGRESS.md`, ...): the other session owns them. Your status goes in
   `scripts/signs/STATUS.md`, kept current; the other session folds it into PROGRESS.md.
6. Don't write code with Bash heredocs (here they turn `\b`, `\n`, `\r` into control characters): use the
   Write/Edit tools, or write scripts to a scratch folder and run them.
7. **Delivery:** copy the final weights and engine into `C:\ClaudeProjects\DrivePilot_BananaHacks2026\models\signs\`
   (gitignored there; the only write allowed in the main checkout) and record file names, sizes and the TensorRT
   version in STATUS.md.

## What to build
- **Classes (US, MUTCD):** speed limit with its value (15-70 mph), school speed limit / school zone, stop, yield,
  do not enter, wrong way, no turn on red, one way, no left / no right / no U-turn, pedestrian crossing. Drop a
  class if the data is too thin, and say so. **Reading the speed-limit value right is the priority:** per-value
  classes, or one "speed limit" class plus a second-stage classifier on the crop, whichever measures better.
- **Data:** check what is actually downloadable before committing to a plan. Candidates: the LISA US traffic sign
  dataset (cvrr.ucsd.edu, academic license), the US classes of the Mapillary Traffic Sign Dataset (needs the
  user's free Mapillary account), anything else you find (check licenses: non-commercial is fine for the
  hackathon; record each one). If the user has to do something (accounts, licenses, downloads), ask once, in one
  message with exact steps, and keep working on whatever doesn't depend on it.
- **Validation in our own domain:** the app is tested on BDD100K dashcam clips (US cities, 720p, 30 fps). The dev
  clips are in `C:\ClaudeProjects\DrivePilot_BananaHacks2026\data\videos\dev\` (read-only for you;
  `eval/dev_clips.yaml` describes them; `scripts/fetch_dev_clips.py` fetches more BDD100K clips by id by range
  request). Hand-label a small held-out set of real US signs from such clips, checked on full-resolution frames:
  that number matters more than the public test split.
- **Training:** fine-tune YOLO26s on the laptop GPU. Choose the image size for small, distant signs (measure 640
  against larger sizes or crops of the frame); keep the runs in your worktree. Export a TensorRT FP16 engine the
  way `scripts/build_engine.py` does (separate `.venv-trtbuild`, the same TensorRT version as `.venv`), from your
  own script.
- **The module the app will use (integration contract):** `perception/signs.py` with

      @dataclass(frozen=True, slots=True)
      class SignDetection:
          kind: str               # "speed_limit", "school_speed_limit", "stop", "yield", "do_not_enter", ...
          value: int | None       # mph for speed limits, else None
          conf: float
          box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels

      class SignDetector:
          def __init__(self, cfg: SignsConfig, root: Path) -> None: ...  # loads the engine; refuses the CPU
          def detect(self, frame: Frame) -> list[SignDetection]: ...     # one core.types.Frame (1280x720 or 1920x1080)

  It must verify it really runs on CUDA in FP16 (CLAUDE.md), measure its per-frame time on this GPU while the app
  runs (target ≤ 10 ms; it will run at a few fps), and never report a speed-limit value it isn't confident about:
  a wrong limit is worse than none.
- **Not yours** (the other session does them, against the contract above): the alerts ("Speed limit is now 35."),
  memory, overlays, changes to the vision-model prompt, wiring into the app.
- **Scripts and docs:** `scripts/signs/` (prepare data, train, evaluate, export) with a README of exact commands;
  `models/signs/metrics.json` (tracked); `tests/test_signs.py` (label conversion, class mapping, value parsing,
  the detector's contract with a fake, plus a `gpu`-marked test with the real engine).

## Done when
- Held-out mAP50 ≥ 0.8 on the speed-limit classes of the public split, with per-class AP reported.
- On the hand-labeled BDD100K set: ≥ 80% of clearly readable speed-limit signs detected with the right value and
  **0 wrong values** at the confidence threshold the app will use; precision/recall per class for the rest.
- Per-frame time measured on this GPU; the engine verified on CUDA in FP16.
- pytest, ruff and mypy clean (`.venv/Scripts/python -m pytest`, `ruff check .`, `ruff format --check .`,
  `mypy .`); a strict self-review plus an independent subagent review, findings fixed with tests; branch pushed.
- `scripts/signs/STATUS.md`: what was done, the numbers and how they were measured, any compromise, and a short
  write-up for the Best Fine-tune entry (data, training setup, results, before/after on sign questions).

## Timeline
The project deadline is **Sunday 2026-10-11, 8:59 pm PDT**. The other session needs a working detector by
**Sunday 11:00 am PDT** to integrate it. Plan for that: data ready and training started Saturday afternoon or
evening, training overnight, evaluation and export Sunday morning. If the full plan won't fit, deliver the best
honest version by 11:00 am and say what's missing. No stubs, no faked numbers (CLAUDE.md).
