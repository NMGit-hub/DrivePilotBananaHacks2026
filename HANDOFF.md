# Handoff: DrivePilot, the user's own dashcam footage (2026-10-11, ~12:45 PDT)

For the next Claude Code session. Read this first, then CLAUDE.md (standing rules), PROGRESS.md (status, measurements,
known issues: kept current), DEMO.md, V2_BUILD_PLAN.md and PLAN.md.

**Deadline: Sunday 2026-10-11, 8:59 pm PDT** (about 8 hours from this handoff). `main` = `origin/main` (last commit
`669fd08` + this handoff). Another session ("voice-alerts-destination": wake-word modes, unrequested alerts, a
destination box) merges into main too: `git pull` before every commit.

## 1. Where things stand
- V2 is done (PROGRESS.md "V2"): overlays, pedestrians, memory, YOLOPv2 lanes/drift, the sign detector, parking
  (below its bar on public footage), the 45-row test sheet, README screenshots, DEMO.md, align_gpx.py.
- Vision answers improved today (PROGRESS "Vision answers improved"): a second model (Claude Haiku 5.5) for
  sign/parking/scene questions, the place from GPS, parking guidance; blind check 8 -> 10 right of 18.
- **New today: the user's own car dashcam footage is the data for the evals AND the hackathon submission.**

## 2. The dashcam footage (D:\Dashcam, external drive)
- Mercedes MBUX dashcam: 950 clips `dm_YYYYMMDD_HHMMSS.mp4` (local Pacific start time), 2 min each, 1920x1280,
  25 fps, **every frame a keyframe** (seek with PyAV, not OpenCV: 50 ms vs 560 ms a frame), no audio; 30.2 h over
  23 days (14 Sep - 11 Oct), Seattle Eastside (I-405, Bellevue, Redmond...). 4 files are damaged (won't open).
- A burned-in text bar in rows 1235-1279: date, time, **speed (0.1 mph, ~2 Hz)**, **lat/lon in whole arc-seconds**
  (~31 m N-S, 21 m E-W; once a second). The bar shows coordinates near the user's home: **crop it out of anything
  shown publicly**. Ego motion's ROI already excludes it (roi_bottom 0.85); the hood is visible just above it.
- `nav/dashcam_overlay.py` reads it (glyph templates in `nav/dashcam_glyphs.npz`, learned from the footage; signs
  from N/S/E/W; malformed lines skipped) and smooths positions (RTS). `scripts/dashcam_gpx.py`:
  `scan D:/Dashcam --jobs 10` (done: `data/cache/dashcam/scan.csv` + per-clip `data/cache/dashcam/<clip>.json`,
  1 Hz) and `gpx <clip stems>` (writes `data/gps/<clip>.gpx` with `video_start`; not run yet for any clip).
  **Scan result: 837 clips with GPS, 97.7 % of samples read, time field matches the file name on 99.6 %**
  (409 off: not yet looked at). 4.3 h highway (>= 45 mph), 9.2 h city, 1.9 h crawl (1-10 mph), 9.2 h stopped;
  117 clips with >= 90 s highway, 54 with >= 30 s crawl, 111 night (median top-half brightness < 60).
- **Camera:** looks rectified (straight lines stay straight). **Horizontal FOV ~97 deg** from 6 turns
  (`scripts/dashcam_calibrate.py D:/Dashcam --turns 6`: median f 849 px; 4 of 6 turns at 92-97 deg, 2 outliers at
  117 deg, probably ego motion losing track mid-turn). Use `--set video.file.hfov_deg=97` for these clips (the lane
  model then gets the center cropped to 70 deg). Camera height unknown (config prior 1.3 m; the lane tracker sets
  its own scale from lane width).

## 3. Next steps, in order (estimates)
1. **(20 min) Firm up the FOV**: add a coverage check to `image_swing_px` (skip turns where the homography was
   missing on > 10 % of frames), run `--turns 20`, record the median in PROGRESS and in a config comment.
2. **(45 min) Pick and copy clips** from `scan.csv` into `data/videos/own/` (C: has ~60 GB free; 280 MB a clip) and
   write their GPX (`dashcam_gpx.py gpx <stems>`): ~15 highway (day, night, rain: 2026-10-09/10 had rain), ~10
   parking lots (crawl_s >= 30, **not** at the user's homes: clips starting at 0 mph near 47°35'47"N 122°7'7"W, or in
   the garage of house 17449, are home), ~10 intersections with traffic lights, a few night/rain/tunnel. Look at a
   contact sheet of candidates before copying.
3. **(30-45 min) Showcase videos (the user's priority for the submission)**: a background agent of the previous
   session was building `scripts/render_showcase.py` + `ui/render.py` + `ui/timeline.py` (**untracked in the
   working tree, not reviewed or committed**): the V2 overlays drawn onto a clip in lock-step, captions for
   questions/answers/alerts, Piper audio, the dashcam bar cropped. Check its state (read the files, run
   `python scripts/render_showcase.py --help`, its tests), finish it if needed, then render examples with
   `--set video.file.hfov_deg=97`: **rain, night, highway lanes, intersection with lights (+ "tell me when it's
   green"), parking lot (+ "can I park here?"), a sign question** into `runs/showcase/`. Look at frames of each before
   calling it done. Review, test, commit.
4. **(1.5-2 h) Evals on the user's footage**:
   - Drift: `scripts/eval_lanes.py cache --clip data/videos/own/<clip>.mp4 --set video.file.hfov_deg=97` for the
     highway clips, then `eval_lanes.py drift --clip ... --hfov 97` (needs `data/gps/<clip>.gpx`): false alerts per
     hour of real highway driving (BDD gave 1 borderline in 5.7 min).
   - Lane index: hand labels in a new `eval/lane_labels_own.yaml` (same format as `eval/lane_labels.yaml`; an agent
     can label frames from `eval_lanes.py frames`), then `eval_lanes.py index --labels eval/lane_labels_own.yaml
     --hfov 97`.
   - Parking: `scripts/cache_perception.py --clip ... --set video.file.hfov_deg=97`, labels like
     `eval/parking_labels*.yaml` (an agent, saving clip by clip), `scripts/eval_parking.py` (check its hfov option).
   - Optionally pedestrians (`eval_pedestrians.py`) and a few test-sheet rows on own clips.
5. **(30 min) PROGRESS + README** (the dashcam workflow, the numbers), push. Final message to the user: what was
   measured on their footage, where the showcase videos are.

## 4. Working notes
- `.venv/Scripts/python` for everything; full tests `-m pytest -q` (~3-4 min), `-m ruff check .`, `-m ruff format
  --check .`, `-m mypy .`.
- **Never write code through Bash heredocs with backslashes or `$`** (a `\b` became a backspace; `$d` expanded in a
  path): use Edit/Write or a patch script written with the Write tool. For Windows links use PowerShell.
- Privacy: the footage shows the user's home, house number, plates and GPS: never commit `data/`, crop the overlay
  bar in published videos, avoid home clips in the showcase.
- Agents: main venv by absolute path; save work incrementally; they can't write `.md` reports (ask in the final
  message). Agents' work lands uncommitted: review, test, commit it yourself.
- Remind the user to rotate the API keys after the hackathon; the stray COCO8 runs and the `DrivePilot_signs`
  worktree can be deleted by the user.
