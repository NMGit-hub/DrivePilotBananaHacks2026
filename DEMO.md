# Demo plan (V2 page, replays)

Seven short replays, ~6 minutes in all. Every moment below was checked on 2026-10-10/11 (test sheet
`runs/*_sheet`, README screenshots). These are public BDD100K clips: fine for a live demo of the software; the
submission video should use our own footage (README "Test footage").

**Before:** laptop on its charger, Windows power mode **Best performance**, nothing else on the GPU, `.env` keys in
place. Open <http://localhost:8000/> full screen. Say **"Hey DrivePilot, ..."** and the question, or press **Q** and
type it (safer if the room is loud). **Space** pauses, **←/→** skip 5 s. One replay at a time (port 8000); stop
each with Ctrl+C.

(`hfov_deg=60`: BDD100K dashcams are ~60°; our phone's own footage uses `null`. `start_s`: skips the dull start.)

## 1. "Tell me when it's green" (SF red light, ~45 s)
```powershell
.venv/Scripts/python -m app --set video.file.hfov_deg=60 --video data/videos/dev/red_light_turns_green_sf__b97b9ee8-612c36ef.mov --gpx data/gps/b97b9ee8-612c36ef.gpx --set video.file.start_s=15
```
- ~19.6 s: **"School zone ahead."** on its own (a real school crossing sign at the corner; said once).
- From ~22 s we wait at the red light. Ask **"can I go?"** → "Not yet, the light's red. I'll tell you when it turns
  green." (local, instant).
- Ask **"tell me when it's green"** → at 36.6 s **"Light's green, check the intersection and go."** (the light label
  on the video turns GREEN).

## 2. Speed limit, memory and a pedestrian (NYC, ~35 s)
```powershell
.venv/Scripts/python -m app --set video.file.hfov_deg=60 --video data/videos/dev/ego_holdout_long_stop__b5aa77fa-b1916843.mov --gpx data/gps/b5aa77fa-b1916843.gpx --set video.file.start_s=5
```
- 8.7 s: **"Speed limit is now 30."** and the SPEED LIMIT 30 badge (the fine-tuned sign detector).
- ~15 s (sign out of view): **"what was the speed limit on that sign we just passed?"** → "30 mph", with the frame it
  answered from as a thumbnail.
- ~30.8 s (stopped at a red light, a woman crosses in front): **"is someone crossing in front of us?"** → yes; her box
  pulses. Then **"can I pull forward a little?"** → wait for her.

## 3. Lanes on a highway (~30 s)
```powershell
.venv/Scripts/python -m app --set video.file.hfov_deg=60 --video data/videos/highway/7dc2914d-a20a969a.mov --gpx data/gps/7dc2914d-a20a969a.gpx --set video.file.start_s=10
```
- Our lane's lines from YOLOPv2 and the GPS speed (~60 mph) on the video.
- **"which lane am I in?"** → the leftmost lane, the barrier on the left; ~27 s **"how many lanes are to my left?"** →
  none.

## 4. Who goes first at a stop sign (SF dawn, ~20 s)
```powershell
.venv/Scripts/python -m app --set video.file.hfov_deg=60 --video data/videos/dev/stop_sign_cross_traffic_sf_dawn__be986afd-012d020c.mov --gpx data/gps/be986afd-012d020c.gpx --set video.file.start_s=15
```
- ~20 s, still rolling: **"can I go?"** → "I can't judge that safely while we're moving."
- ~22.6 s, stopped: **"who got here first?"** → "Nobody else is waiting. You're first. Check before you go." (from the
  tracker, ~0.1 s). **"is it safe to pull out?"** → never a "yes": "check both ways".

## 5. School bus (Queens, ~15 s)
```powershell
.venv/Scripts/python -m app --set video.file.hfov_deg=60 --video data/videos/dev/school_bus_stop_arm_parking__c33700f7-08262a44.mov --gpx data/gps/c33700f7-08262a44.gpx --set video.file.start_s=12
```
- ~18 s: **"why is that bus stopped?"** → loading children, wait. **"can I pass the school bus?"** → "No. Never pass a
  school bus that's stopped with its red lights flashing or its stop arm out..." (a fixed rule, not the model).

## 6. Parking: a handicap spot (lot, ~10 s)
```powershell
.venv/Scripts/python -m app --set video.file.hfov_deg=60 --video data/videos/parking_lot/4025360f-4416d623.mov --gpx data/gps/4025360f-4416d623.gpx
```
- ~6.5 s: **"is that empty spot in front of us a handicap spot?"** → yes, the wheelchair symbol; **"can I park in the
  empty spot straight ahead?"** → only with a permit.
- "help me find a parking spot" works ("Okay, I'll look for an open spot.") but rarely finds a spot on these
  clips: don't promise a spot in the demo (PROGRESS "Known issues").

## 7. Signs in the rain (NYC school zone, ~25 s)
```powershell
.venv/Scripts/python -m app --set video.file.hfov_deg=60 --video data/videos/dev/red_light_rain_school_zone__b61456bb-5bfb50ee.mov --gpx data/gps/b61456bb-5bfb50ee.gpx
```
- 0.25 s: **"School zone ahead."** once (two signs, one per side of the street).
- ~3 s: **"what did that yellow sign say?"** → children walking, "AHEAD". ~9 s **"what's going on on the right?"** →
  construction.

## If something goes wrong
- No answer from the model: the page's status line says "vision off": check the internet and `.env`.
- The page shows "lanes off" or "signs off": a model didn't start; restart the replay (the log says why).
- Slow answers (3+ s): the charger is unplugged or the power mode isn't Best performance.
