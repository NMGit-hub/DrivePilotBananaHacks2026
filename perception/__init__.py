"""Local GPU perception: YOLO26s + ByteTrack detection and tracking, ego-motion (are we stopped), undistortion."""

import os

# ultralytics silently pip-installs missing packages by default (it once added onnxruntime-gpu next to our
# onnxruntime during an export). Dependencies are pinned in requirements.txt instead: turn that off before
# ultralytics is imported anywhere.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
