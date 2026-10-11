"""Load and validate `config.yaml` plus secrets from the environment / `.env`.

Usage:
    cfg = load_config()                                   # repo config.yaml + .env
    cfg = load_config(overrides=["video.source=camera"])  # dotted-key overrides (CLI style)

Every section is a strict, immutable pydantic model: unknown keys are errors, scalars are not coerced (`fps: true`
or `realtime: "no"` fail), and YAML is read with YAML 1.2 scalar rules (no `yes/no` booleans, no base-60 numbers
like `1:30`). A typo in config.yaml fails loudly at startup instead of silently falling back to a default.
Secrets never live in config.yaml: they are read from environment variables (optionally populated from `.env`)
into `SecretStr` fields that mask themselves in reprs, logs and dumps.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import yaml
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    Strict,
    ValidationError,
    field_validator,
    model_validator,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed or fails validation."""


# Paths come from YAML as strings, so they are the one type allowed to be converted.
ConfigPath = Annotated[Path, Strict(False)]


class _Strict(BaseModel):
    """Base for all config sections: unknown keys rejected, no type coercion, immutable after loading."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _Yaml12Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 core-schema scalars: only true/false are booleans, no octal-by-leading-zero,
    no base-60 numbers."""


_BOOL, _INT, _FLOAT = "tag:yaml.org,2002:bool", "tag:yaml.org,2002:int", "tag:yaml.org,2002:float"
_Yaml12Loader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag not in (_BOOL, _INT, _FLOAT)]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12Loader.add_implicit_resolver(_BOOL, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))
_Yaml12Loader.add_implicit_resolver(
    _INT, re.compile(r"^(?:[-+]?(?:0|[1-9][0-9]*)|0x[0-9a-fA-F]+|0o[0-7]+)$"), list("-+0123456789")
)
_Yaml12Loader.add_implicit_resolver(
    _FLOAT,
    re.compile(
        r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


def _yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_Yaml12Loader)


# --------------------------------------------------------------------------------------------- video


class FileVideoConfig(_Strict):
    """Recorded-drive input (`video.source: file`)."""

    path: ConfigPath = Path("data/videos/dev_clip.mov")
    gpx_path: ConfigPath | None = None
    gpx_offset_s: float = Field(0.0, description="GPX time at video t=0 minus the first GPX point time.")
    realtime: bool = Field(True, description="Play at real-time speed, dropping frames if processing lags.")
    start_s: float = Field(0.0, ge=0.0, description="Start playback this many seconds into the clip.")
    loop: bool = False


class CameraVideoConfig(_Strict):
    """Live phone camera through scrcpy-server (`video.source: camera`)."""

    scrcpy_dir: ConfigPath | None = Field(
        None, description="Folder with adb.exe and scrcpy-server; auto-detected if null."
    )
    camera_id: int = Field(3, ge=0, description="OnePlus 12: 3 = the physical ultrawide (verified 2026-10-09).")
    camera_zoom: float = Field(1.0, gt=0.0, description="Zoom; on a logical camera (id 0) 0.6 selects the ultrawide.")
    width: int = Field(1920, gt=0)
    height: int = Field(1080, gt=0)
    fps: int = Field(30, gt=0, le=60)
    bitrate_mbps: float = Field(8.0, gt=0.0)
    stall_timeout_s: float = Field(3.0, gt=0.0, description="No video bytes for this long = the camera stalled.")
    decoder: Literal["auto", "d3d11va", "cuda", "software"] = Field(
        "auto", description="H.264 decoding: auto tries hardware (D3D11VA, then CUDA) before software."
    )


class VideoConfig(_Strict):
    """Video input. `source` is the single switch between live camera and recorded file."""

    source: Literal["camera", "file"] = "file"
    file: FileVideoConfig = FileVideoConfig()
    camera: CameraVideoConfig = CameraVideoConfig()


# ----------------------------------------------------------------------------------------------- gps


class GpsConfig(_Strict):
    """GPS input: GPX replay in file mode, the phone web page in camera mode."""

    source: Literal["auto", "gpx", "phone", "none"] = "auto"
    stale_s: float = Field(3.0, gt=0.0, description="A phone fix older than this is ignored.")


class NavConfig(_Strict):
    """Navigation: Mapbox routing, progress, rerouting and spoken guidance (nav/)."""

    profile: Literal["driving-traffic", "driving"] = "driving-traffic"
    country: str | None = Field("us", description="Limit destination search to a country (ISO code) or null.")
    timeout_s: float = Field(5.0, gt=0.0)
    off_route_m: float = Field(50.0, gt=0.0, description="PLAN.md: more than ~50 m from the route...")
    off_route_s: float = Field(5.0, ge=0.0, description="...for 5 s means we left it: reroute.")
    reroute_cooldown_s: float = Field(20.0, ge=0.0)
    snap_window_m: float = Field(500.0, gt=0.0, description="Snap near where we were (routes can loop back).")
    max_backtrack_m: float = Field(50.0, ge=0.0, description="A bigger jump backwards along the route is noise.")
    arrival_m: float = Field(30.0, gt=0.0, description="Arrived when this close to the end of the route line...")
    arrival_radius_m: float = Field(
        60.0, gt=0.0, description="...or this close (straight line) to it (parking lots, the far side of the road)."
    )
    near_end_m: float = Field(
        150.0, ge=0.0, description="Leaving the route this close to the end means we turned in: arrived, no reroute."
    )
    join_slack_m: float = Field(
        100.0, ge=0.0, description="Starting off the road (a parking lot): off route only this much further away."
    )
    home: str | None = Field(None, description='Address for "take me home" (null: ask for the address).')
    lane_reminders_m: Annotated[tuple[float, ...], Strict(False)] = Field(  # YAML gives a list
        (1609.0, 402.0), description="PLAN.md: lane reminders about 1 mile and 0.25 mile before a junction."
    )


# --------------------------------------------------------------------------------------- perception

DETECTION_CLASSES = ("car", "truck", "bus", "motorcycle", "stop sign", "traffic light", "person", "bicycle")


class TrackerConfig(_Strict):
    """ByteTrack settings (ultralytics BYTETracker)."""

    track_high_thresh: float = Field(0.25, ge=0.0, le=1.0, description="First-stage association score.")
    track_low_thresh: float = Field(0.1, ge=0.0, le=1.0, description="Second-stage (low-score) association.")
    new_track_thresh: float = Field(0.25, ge=0.0, le=1.0, description="Min score to start a new track.")
    track_buffer_s: float = Field(1.5, gt=0.0, description="Keep lost tracks alive this long (occlusions).")
    match_thresh: float = Field(0.8, gt=0.0, le=1.0)
    fuse_score: bool = True


class PerceptionConfig(_Strict):
    """YOLO26s detector on the GPU plus ByteTrack."""

    weights: ConfigPath = Path("models/yolo26s.pt")
    engine: Literal["tensorrt", "pytorch"] = Field(
        "tensorrt", description="tensorrt: ~10x faster on Windows (built once from the weights); pytorch: eager."
    )
    engine_workspace_gb: float = Field(4.0, gt=0.0, description="TensorRT builder workspace.")
    device: int = Field(0, ge=0, description="CUDA device index. The pipeline refuses to run on the CPU.")
    half: bool = Field(True, description="FP16 inference.")
    imgsz: Annotated[tuple[int, int], Strict(False)] = Field(
        (384, 640), description="Inference size (height, width), multiples of 32; 384x640 fits 16:9 frames."
    )
    conf: float = Field(0.25, gt=0.0, lt=1.0)
    iou: float = Field(0.7, gt=0.0, lt=1.0)
    classes: Annotated[tuple[str, ...], Strict(False)] = DETECTION_CLASSES  # YAML gives a list
    vehicle_dedup_iou: float = Field(
        0.7, gt=0.0, le=1.0, description="Merge overlapping vehicle boxes of different classes (car vs truck)."
    )
    tracker: TrackerConfig = TrackerConfig()

    @field_validator("imgsz")
    @classmethod
    def _multiple_of_32(cls, value: tuple[int, int]) -> tuple[int, int]:
        if any(v % 32 or not 128 <= v <= 1920 for v in value):
            raise ValueError("height and width must be multiples of 32 between 128 and 1920")
        return value

    @field_validator("classes")
    @classmethod
    def _known_classes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(value) - set(DETECTION_CLASSES))
        if unknown:
            raise ValueError(f"unsupported classes {unknown}; choose from {list(DETECTION_CLASSES)}")
        return value


class EgoMotionConfig(_Strict):
    """'Are we stopped?' from optical flow on a small grayscale copy of the frame."""

    work_width: int = Field(320, ge=160, le=960, description="Downscaled width for optical flow.")
    roi_top: float = Field(0.30, ge=0.0, lt=1.0, description="Ignore the sky above this fraction of height.")
    roi_bottom: float = Field(0.85, gt=0.0, le=1.0, description="Ignore the hood/dashboard below this fraction.")
    box_margin: float = Field(0.15, ge=0.0, le=1.0, description="Grow object boxes by this fraction when masking.")
    max_points: int = Field(300, ge=20)
    min_points: int = Field(20, ge=4, description="Below this, motion is 'unknown' and the state is held.")
    fb_max_error_px: float = Field(1.0, gt=0.0, description="Forward-backward LK consistency limit.")
    min_dt_s: float = Field(1 / 60, gt=0.0, description="Floor on the frame time step used for per-second rates.")
    metric: Literal["combined", "zoom", "horizontal", "magnitude"] = Field(
        "combined", description="Motion measure for the stopped state (see perception/ego_motion.py)."
    )
    zoom_weight: float = Field(50.0, gt=0.0, description="Zoom-rate scale in the combined measure.")
    baseline_s: float = Field(
        0.6, gt=0.0, le=2.0, description="Long-baseline motion: compare with the frame this long ago (lower noise)."
    )
    stopped_below: float = Field(1.5, gt=0.0, description="Smoothed measure below this = still.")
    moving_above: float = Field(3.0, gt=0.0, description="Smoothed measure above this = moving (hysteresis).")
    stopped_after_s: float = Field(0.5, ge=0.0, description="Still this long before we call it stopped.")
    smoothing_s: float = Field(0.2, ge=0.0, description="Rolling-median window that removes one-frame glitches.")

    @model_validator(mode="after")
    def _ordered(self) -> EgoMotionConfig:
        if self.roi_bottom <= self.roi_top:
            raise ValueError("roi_bottom must be greater than roi_top")
        if self.moving_above < self.stopped_below:
            raise ValueError("moving_above must be >= stopped_below")
        return self


# ---------------------------------------------------------------------------------------- world state


class VehicleMotionConfig(_Strict):
    """Is each tracked vehicle stopped? (state/vehicle_motion.py)"""

    window_s: float = Field(0.6, gt=0.0, description="Speed is fitted over this much track history.")
    min_samples: int = Field(10, ge=3, description="Samples needed (with min_span_frac) before judging speed.")
    min_span_frac: float = Field(0.8, gt=0.0, le=1.0, description="History must span this share of window_s.")
    stopped_below: float = Field(0.12, gt=0.0, description="Box heights per second below which a car is still.")
    moving_above: float = Field(0.35, gt=0.0, description="Box heights per second above which it moves.")
    noise_px_per_s: float = Field(
        10.0, ge=0.0, description="Speed floor in px/s: box jitter on small (distant) cars is not motion."
    )
    moving_noise_factor: float = Field(2.5, ge=1.0, description="Moving needs this multiple of the noise floor.")
    stopped_after_s: float = Field(0.5, ge=0.0, description="PLAN.md: a car is stopped after 0.5 s of stillness.")
    still_radius: float = Field(
        0.1, gt=0.0, description="Box heights: a car within this of where it rests has stopped (dates the stop)."
    )
    still_radius_px: float = Field(4.0, ge=0.0, description="Pixel floor for still_radius (small boxes jitter).")
    max_shift_px: float = Field(200.0, gt=0.0, description="Homographies moving more per frame are rejected.")
    max_scale_change: float = Field(0.15, gt=0.0, description="...or scaling more than this per frame.")
    max_track_age_s: float = Field(3.0, gt=0.0)


class IntersectionConfig(_Strict):
    """Where cars wait relative to us, and when we are at a stop sign (state/approach.py, state/world.py)."""

    side_x: float = Field(0.30, gt=0.0, lt=0.5, description="Left/right zones: outer fraction of the width.")
    min_side_aspect: float = Field(1.3, gt=0.0, description="Cross-street cars are seen side-on (w/h).")
    max_ahead_aspect: float = Field(2.2, gt=0.0)
    lead_center_band: float = Field(0.25, gt=0.0, lt=0.5)
    lead_min_height: float = Field(0.10, gt=0.0, lt=1.0, description="Lead car: box height fraction (< ~9 m).")
    max_lead_aspect: float = Field(1.9, gt=0.0, description="Lead car is seen from behind (not side-on).")
    narrow_fov_deg: float = Field(100.0, gt=0.0, description="Below this the sides of an intersection may be cut off.")
    color_refresh_s: float = Field(1.0, gt=0.0, description="Re-name a vehicle's color this often.")
    frame_buffer_s: float = Field(3.0, gt=0.0, description="Recent frames kept for vision-model questions.")
    frame_buffer_spacing_s: float = Field(0.25, gt=0.0)
    min_height: float = Field(0.03, gt=0.0, lt=1.0, description="Smaller boxes are beyond the intersection.")
    min_bottom_y: float = Field(0.30, ge=0.0, lt=1.0)
    max_bottom_y: float = Field(0.98, gt=0.0, le=1.0)
    stop_sign_min_conf: float = Field(0.45, gt=0.0, lt=1.0)
    stop_sign_memory_s: float = Field(
        10.0, gt=0.0, description="We are 'at a stop' if a stop sign was seen this recently."
    )
    camera_hfov_deg: float = Field(114.0, gt=0.0, lt=180.0, description="Below ~100° the sides may be out of view.")
    drive_off_s: float = Field(
        5.0, gt=0.0, description="Moving this long after a stop = we drove away (requests about it expire)."
    )
    people_center_band: float = Field(0.3, gt=0.0, le=0.5, description="Pedestrians this close to center are ahead.")
    people_min_height: float = Field(0.06, gt=0.0, lt=1.0, description="...and at least this tall (frame fraction).")


class StopQueueConfig(_Strict):
    """The 4-way-stop queue (state/stop_queue.py)."""

    tie_window_s: float = Field(1.0, ge=0.0, description="PLAN.md: stops within 1 s are a tie (right goes first).")
    depart_moving_s: float = Field(1.0, gt=0.0, description="A departing car keeps moving this long...")
    depart_moved: float = Field(1.5, gt=0.0, description="...and moves this many box heights (a creep is less).")
    depart_lost_moving_s: float = Field(0.5, gt=0.0, description="Lost from view while moving (this long)...")
    depart_lost_moved: float = Field(1.0, gt=0.0, description="...and this many box heights from its stop = gone.")
    undepart_s: float = Field(3.0, ge=0.0, description="'Departed' but at rest again on its approach: reinstated.")
    relink_iou: float = Field(0.3, gt=0.0, le=1.0)
    lost_timeout_s: float = Field(8.0, gt=0.0)
    parked_after_s: float = Field(
        20.0, gt=0.0, description="A car ahead that hasn't moved this long after we arrived is probably parked."
    )
    ego_depart_s: float = Field(1.5, gt=0.0, description="Moving this long = we left (creeping is not).")


class TrafficLightConfig(_Strict):
    """Traffic-light reading (state/traffic_light.py)."""

    min_height_frac: float = Field(0.012, gt=0.0)
    center_band: float = Field(0.40, gt=0.0, le=0.5)
    max_center_y: float = Field(0.65, gt=0.0, le=1.0)
    v_min: int = Field(150, ge=0, le=255)
    lamp_contrast: int = Field(60, ge=0, le=255, description="A lit lamp is this much brighter than its housing…")
    lamp_v_cap: int = Field(245, ge=0, le=255, description="…but never needs to exceed this (lamps clip at 255).")
    min_lit_frac: float = Field(0.02, ge=0.0, lt=1.0)
    max_lamp_frac: float = Field(0.40, gt=0.0, le=1.0)
    s_min_halo: int = Field(70, ge=0, le=255)
    v_halo_min: int = Field(90, ge=0, le=255)
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    upright_ratio: float = Field(1.6, gt=1.0, description="Boxes this much taller than wide are upright housings.")
    position_penalty: float = Field(0.3, ge=0.0, le=1.0, description="Confidence factor if the lamp is misplaced.")
    match_iou: float = Field(0.3, gt=0.0, le=1.0, description="Detections overlapping a known light this much match.")
    vote_share: float = Field(0.6, gt=0.5, le=1.0, description="Majority needed for a light's stable color.")
    rival_share: float = Field(
        0.6, gt=0.0, le=1.0, description="A light this prominent (vs ours) that disagrees makes us unsure."
    )
    stable_s: float = Field(0.5, gt=0.0, description="Reported state = majority over this window.")
    green_confirm_frames: int = Field(5, ge=1, description="Consecutive green frames that confirm the change.")
    red_memory_s: float = Field(5.0, gt=0.0, description="A green counts as a change if red was seen this recently.")
    anchor_pad: float = Field(0.15, ge=0.0, le=1.0, description="Pad light boxes so the whole housing is read.")
    anchor_weight: float = Field(0.2, gt=0.0, le=1.0, description="Vote weight of a remembered (undetected) light.")


# ------------------------------------------------------------------------------------------- answers


class AlertsConfig(_Strict):
    """Spoken alerts (answers/alerts.py)."""

    cooldown_s: float = Field(15.0, ge=0.0, description="The same alert kind is never repeated within this time.")
    watch_timeout_s: float = Field(
        300.0, gt=0.0, description="'Tell me when it's green' requests expire after this long."
    )
    approach_timeout_s: float = Field(
        45.0, gt=0.0, description="A request made while driving expires if we don't stop within this time."
    )
    seek_back_tolerance_s: float = Field(1.0, ge=0.0, description="Video jumping back more than this drops requests.")


# ------------------------------------------------------------------------------------------- vision


class ClaudeVisionConfig(_Strict):
    """Anthropic Claude provider settings."""

    model: str = "claude-haiku-4-5"
    max_tokens: int = Field(300, gt=0)
    max_retries: int = Field(0, ge=0, description="A retry can't arrive within the answer budget anyway.")


class GeminiVisionConfig(_Strict):
    """Google Gemini provider settings."""

    model: str = "gemini-3.8-flash"
    thinking_level: Literal["low", "medium", "high"] = "low"
    max_tokens: int = Field(300, gt=0)


class OllamaVisionConfig(_Strict):
    """Local Ollama provider settings."""

    model: str = "qwen2.5vl:7b"
    host: str = "http://localhost:11434"
    timeout_s: float = Field(15.0, gt=0.0)


class VisionConfig(_Strict):
    """Per-provider settings for the vision-language model (the provider itself is `vision_model`)."""

    frames: int = Field(3, ge=1, le=6, description="Recent frames sent with each question (latest last).")
    frame_spacing_s: float = Field(0.7, gt=0.0, description="Time between the frames sent.")
    sign_frame_spacing_s: float = Field(1.2, gt=0.0, description="...for questions about a sign just passed.")
    latest_width: int = Field(1280, ge=320, description="The latest frame is sent at up to this width...")
    older_width: int = Field(640, ge=160, description="...earlier frames smaller (motion, things just passed).")
    jpeg_quality: int = Field(80, ge=30, le=100)
    first_token_timeout_s: float = Field(3.0, gt=0.0, description="No text by then: spoken fallback.")
    total_timeout_s: float = Field(8.0, gt=0.0, description="The whole answer must arrive by then.")
    time_override: str | None = Field(
        None, description="Fixed day and time for prompts, e.g. 'Saturday 17:40' (replays / test sheet)."
    )
    temperature: float | None = Field(
        0.0, ge=0.0, le=1.0, description="0 = most literal; null for Claude models that reject it (after 4.6)."
    )
    claude: ClaudeVisionConfig = ClaudeVisionConfig()
    gemini: GeminiVisionConfig = GeminiVisionConfig()
    ollama: OllamaVisionConfig = OllamaVisionConfig()

    @model_validator(mode="after")
    def _deadlines(self) -> VisionConfig:
        if self.first_token_timeout_s > self.total_timeout_s:
            raise ValueError("first_token_timeout_s must be <= total_timeout_s")
        return self


# --------------------------------------------------------------------------------------------- voice


class WakeWordConfig(_Strict):
    """openWakeWord "Hey DrivePilot" (voice/wakeword.py)."""

    model: ConfigPath = Path("models/wakeword/hey_drivepilot.onnx")
    threshold: float = Field(
        0.15, gt=0.0, lt=1.0, description="0 false wakes/h down to 0.10 (metrics.json); real voices need <= 0.2."
    )
    debounce_s: float = Field(2.0, ge=0.0, description="Ignore further triggers for this long after one.")


class EndpointConfig(_Strict):
    """When the question ends: Silero VAD after the wake word (voice/endpoint.py)."""

    speech_threshold: float = Field(0.5, gt=0.0, lt=1.0, description="VAD probability that counts as speech.")
    end_silence_s: float = Field(
        0.5, gt=0.0, description="PLAN.md: ~0.6 s; 0.5 keeps local answers under 1 s (real takes not cut off)."
    )
    speculative_silence_s: float = Field(
        0.25, ge=0.0, description="Start transcribing after this much silence (kept if the silence lasts)."
    )
    no_speech_timeout_s: float = Field(4.0, gt=0.0, description="Give up if nothing is said after the wake word.")
    wake_tail_s: float = Field(
        0.4,
        ge=0.0,
        description="Speech this soon after the wake fires may be just the end of the wake phrase (unless "
        "it runs on without a pause).",
    )
    wake_pause_s: float = Field(
        1.2, gt=0.0, description="...so after only that, the question ends after this much silence, not end_silence_s."
    )
    max_utterance_s: float = Field(10.0, gt=0.0)
    pre_roll_s: float = Field(
        1.5,
        ge=0.0,
        description="Audio kept from before the wake word fired: the whole phrase, so Whisper "
        'hears it intact (a cut "...lot, tell me" was misheard on real voices).',
    )


class SttConfig(_Strict):
    """faster-whisper on the GPU (voice/stt.py)."""

    model: str = Field("small.en", description="faster-whisper model name (downloaded once) or a local path.")
    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: str = "float16"
    beam_size: int = Field(1, ge=1, le=10)
    max_new_tokens: int = Field(64, ge=8, le=440, description="Questions are short; bounds the decode time.")
    initial_prompt: str = Field(
        "Hey DrivePilot. A driver asks about the road: stop signs, traffic lights, lanes, exits, merging, "
        "parking, turns.",
        description="Biases spelling and vocabulary.",
    )


class TtsConfig(_Strict):
    """Piper text-to-speech and the alert clips (voice/tts.py, voice/speaker.py)."""

    voice: ConfigPath = Path("models/piper/en_US-lessac-medium.onnx")
    length_scale: float = Field(0.95, gt=0.3, lt=3.0, description="< 1 speaks faster.")
    clips_dir: ConfigPath = Path("models/piper/clips")
    chime_volume: float = Field(0.25, ge=0.0, le=1.0)
    listen_chime: bool = Field(True, description="A short tone when the wake word is heard.")


class VoiceConfig(_Strict):
    """Microphone in, speech out."""

    input_device: str | int | None = Field(None, description="sounddevice name/index; null = system default.")
    output_device: str | int | None = None
    output_volume: float = Field(1.0, gt=0.0, le=1.0, description="Speech and alert volume (1 = full).")
    wakeword: WakeWordConfig = WakeWordConfig()
    endpoint: EndpointConfig = EndpointConfig()
    stt: SttConfig = SttConfig()
    tts: TtsConfig = TtsConfig()


# ------------------------------------------------------------------------------------- ui / logging


class UiConfig(_Strict):
    """The local web page (video + question/answer text) and the phone GPS endpoint."""

    host: str = "127.0.0.1"
    port: int = Field(8000, gt=0, lt=65536)


class LoggingConfig(_Strict):
    """Console/file logging and the per-run review folder."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    runs_dir: ConfigPath = Path("runs")
    frame_log_interval_s: float = Field(1.0, gt=0.0, description="Save one frame per this many seconds.")


# --------------------------------------------------------------------------------------------- signs


class SignsConfig(_Strict):
    """Fine-tuned US sign detector + speed-value classifier (perception/signs.py, V2-M6)."""

    weights: ConfigPath = Field(Path("models/signs/signs_yolo26s.pt"), description="Detector weights (.pt).")
    classifier_weights: ConfigPath = Field(
        Path("models/signs/speed_value_cls.pt"), description="Speed-limit value classifier weights (.pt)."
    )
    engine: Literal["tensorrt", "pytorch"] = Field(
        "tensorrt", description="tensorrt: engines built by scripts/signs/export_engines.py; pytorch: eager."
    )
    device: int = Field(0, ge=0, description="CUDA device index. Refuses to run on the CPU.")
    half: bool = Field(True, description="FP16 inference.")
    imgsz: Annotated[tuple[int, int], Strict(False)] = Field(
        (736, 1280), description="Detector input (height, width): 16:9 frames at 1280 wide, signs at full size."
    )
    classifier_imgsz: int = Field(96, ge=32, le=256, description="Square crop size of the value classifier.")
    conf: float = Field(0.4, gt=0.0, lt=1.0, description="Minimum detection confidence (all kinds).")
    class_conf: dict[str, float] = Field(
        default_factory=dict, description="Per-kind minimum confidence overriding `conf`, e.g. {stop: 0.5}."
    )
    iou: float = Field(0.6, gt=0.0, lt=1.0)
    value_conf: float = Field(
        0.9, gt=0.0, le=1.0, description="Minimum classifier probability (in every view) to report a speed-limit value."
    )
    value_min_px: float = Field(
        16.0, gt=0.0, description="Sign boxes shorter than this (pixels at 1280 px frame width) never get a value."
    )
    value_edge_px: float = Field(
        2.0, ge=0.0, description="No value for a sign this close to the frame border (its number may be cut off)."
    )
    duplicate_iou: float = Field(
        0.6, gt=0.0, le=1.0, description="One sign reported as two kinds above this overlap: keep the surer one."
    )
    value_views: int = Field(
        5, ge=1, le=7, description="Shifted/scaled crops of a speed sign that must all read the same value."
    )

    @field_validator("imgsz")
    @classmethod
    def _multiple_of_32(cls, value: tuple[int, int]) -> tuple[int, int]:
        if any(v % 32 or not 128 <= v <= 1920 for v in value):
            raise ValueError("height and width must be multiples of 32 between 128 and 1920")
        return value

    @field_validator("class_conf")
    @classmethod
    def _conf_range(cls, value: dict[str, float]) -> dict[str, float]:
        # Kind names are checked against the taxonomy by perception/signs.py (core doesn't import perception).
        if any(not 0.0 < v < 1.0 for v in value.values()):
            raise ValueError("class_conf values must be between 0 and 1")
        return value


# --------------------------------------------------------------------------------------------- root


class Secrets(BaseModel):
    """API keys, read from environment variables. Values are `SecretStr` so they never print."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    anthropic_api_key: SecretStr | None = None
    mapbox_access_token: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    ENV_NAMES: ClassVar[dict[str, str]] = {
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "mapbox_access_token": "MAPBOX_ACCESS_TOKEN",
        "gemini_api_key": "GEMINI_API_KEY",
    }

    def configured(self) -> dict[str, bool]:
        """Map each secret name to whether it is set (never exposes values)."""
        return {name: getattr(self, name) is not None for name in self.ENV_NAMES}

    def values(self) -> list[str]:
        """Return the raw secret values that are set (for log redaction only)."""
        return [s.get_secret_value() for s in (getattr(self, n) for n in self.ENV_NAMES) if s is not None]


class AppConfig(_Strict):
    """The whole validated configuration."""

    video: VideoConfig = VideoConfig()
    gps: GpsConfig = GpsConfig()
    nav: NavConfig = NavConfig()
    perception: PerceptionConfig = PerceptionConfig()
    ego_motion: EgoMotionConfig = EgoMotionConfig()
    vehicle_motion: VehicleMotionConfig = VehicleMotionConfig()
    intersection: IntersectionConfig = IntersectionConfig()
    stop_queue: StopQueueConfig = StopQueueConfig()
    traffic_light: TrafficLightConfig = TrafficLightConfig()
    alerts: AlertsConfig = AlertsConfig()
    vision_model: Literal["claude", "gemini", "ollama", "mock"] = "claude"
    vision: VisionConfig = VisionConfig()
    voice: VoiceConfig = VoiceConfig()
    ui: UiConfig = UiConfig()
    logging: LoggingConfig = LoggingConfig()
    signs: SignsConfig = SignsConfig()

    # Not read from YAML: filled in by load_config.
    root: ConfigPath = Field(default=REPO_ROOT, exclude=True)
    secrets: Secrets = Field(default_factory=Secrets, exclude=True)

    @field_validator("root")
    @classmethod
    def _root_is_absolute(cls, value: Path) -> Path:
        return value.resolve()

    @model_validator(mode="after")
    def _check_cross_field(self) -> AppConfig:
        if self.gps.source == "gpx" and self.video.file.gpx_path is None:
            raise ValueError("gps.source is 'gpx' but video.file.gpx_path is not set")
        return self

    @property
    def gps_source(self) -> Literal["gpx", "phone", "none"]:
        """The effective GPS source: 'auto' means GPX replay in file mode (if a track is set), the phone otherwise."""
        if self.gps.source != "auto":
            return self.gps.source
        if self.video.source == "file":
            return "gpx" if self.video.file.gpx_path is not None else "none"
        return "phone"

    def resolve_path(self, path: Path) -> Path:
        """Resolve a config path relative to the repo root (absolute paths pass through)."""
        return path if path.is_absolute() else (self.root / path).resolve()

    def redacted_dump(self) -> dict[str, Any]:
        """Return the config as plain data with secrets reduced to set/unset flags (safe to log or print)."""
        data = self.model_dump(mode="json")
        data["secrets"] = {name: ("set" if ok else "missing") for name, ok in self.secrets.configured().items()}
        return data


# ------------------------------------------------------------------------------------------ loading


def _parse_override(item: str) -> tuple[list[str], Any]:
    """Parse "a.b.c=value" into (["a","b","c"], value), with the value parsed as YAML (so 3 → int, null → None)."""
    if "=" not in item:
        raise ConfigError(f"override {item!r} must look like section.key=value")
    key, raw = item.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ConfigError(f"override {item!r} has an empty key")
    try:
        value = _yaml_load(raw) if raw.strip() else ""
    except yaml.YAMLError as exc:
        raise ConfigError(f"override {item!r}: value is not valid YAML ({exc})") from exc
    return parts, value


def _apply_override(data: dict[str, Any], parts: list[str], value: Any) -> None:
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        if not isinstance(child, dict):
            raise ConfigError(f"override path {'.'.join(parts)!r}: {part!r} is not a section")
        node = child
    node[parts[-1]] = value


def load_secrets(env_file: Path | None = DEFAULT_ENV_PATH, environ: dict[str, str] | None = None) -> Secrets:
    """Read secrets from the process environment, falling back to `env_file` (real env vars win).

    The `.env` file is read without modifying `os.environ`, so nothing leaks into child processes.
    """
    env = dict(os.environ if environ is None else environ)
    file_values: dict[str, str | None] = {}
    if env_file is not None and Path(env_file).is_file():
        try:
            file_values = dotenv_values(env_file, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigError(f"cannot read {env_file} (save it as UTF-8): {exc}") from exc
    found: dict[str, SecretStr] = {}
    for field_name, env_name in Secrets.ENV_NAMES.items():
        raw = env.get(env_name) or file_values.get(env_name) or ""
        if raw.strip():
            found[field_name] = SecretStr(raw.strip())
    return Secrets(**found)


def load_config(
    path: Path | str | None = None,
    overrides: list[str] | dict[str, Any] | None = None,
    env_file: Path | str | None = DEFAULT_ENV_PATH,
    environ: dict[str, str] | None = None,
) -> AppConfig:
    """Load, override and validate the configuration.

    Args:
        path: YAML file; defaults to the repo's config.yaml. Relative paths inside it resolve against its folder.
        overrides: dotted-key overrides, either ["video.source=camera", ...] or {"video.source": "camera"}.
        env_file: `.env` file for secrets (None = environment only).
        environ: environment mapping to read secrets from (defaults to os.environ; for tests).

    Raises:
        ConfigError: the file is missing/unparseable or any value fails validation. The message lists every
            offending field with its dotted path.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        data = _yaml_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"cannot read {config_path} (save it as UTF-8): {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping, got {type(data).__name__}")

    items = overrides.items() if isinstance(overrides, dict) else [_parse_override(o) for o in overrides or []]
    for key, value in items:
        parts = key.split(".") if isinstance(key, str) else key
        _apply_override(data, parts, value)

    for reserved in ("root", "secrets"):
        if reserved in data:
            raise ConfigError(f"{config_path}: '{reserved}' cannot be set in the config file")

    try:
        cfg = AppConfig(
            **data,
            root=config_path.resolve().parent,
            secrets=load_secrets(Path(env_file) if env_file else None, environ),
        )
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors())
        raise ConfigError(f"{config_path}: invalid configuration: {problems}") from None
    return cfg
