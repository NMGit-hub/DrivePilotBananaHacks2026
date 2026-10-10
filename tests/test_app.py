"""M7: the app wiring, the V1 page's server and hub, latency bookkeeping, the lane reader."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from answers.assistant import CAMERA_LOST
from answers.lanes import VisionLaneReader, parse_lane_reading
from answers.speech import TextSpeaker, Utterance
from answers.vision import MockVision, VisionError
from app import DrivePilotApp, apply_cli, build_parser, parse_ask
from capture.base import VideoSource
from capture.file_source import FileSource
from core.clock import WallClock
from core.config import load_config
from core.latency import LatencyBook
from core.runlog import RunRecorder
from core.types import Frame
from nav.gps import GpxReplay, PhoneGps
from nav.mapbox import FixtureClient, Place
from perception.detector import DetectionResult, Track
from tests.nav_fixture import START, build
from ui.hub import UiHub
from ui.server import create_server
from voice.pipeline import HeardQuestion
from voice.speaker import SpeechRecord

RESPONSE, LINE = build()
PLACE = Place("Redmond Town Center", "7525 166th Ave NE, Redmond, WA", LINE.points[-1], "poi")


def wait_for(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout
    while not predicate() and time.perf_counter() < deadline:
        time.sleep(0.01)
    return bool(predicate())


def frame(t: float = 0.0, width: int = 320) -> Frame:
    image = np.full((width * 9 // 16, width, 3), 90, np.uint8)
    return Frame(image=image, t=t, index=0, captured_at=time.perf_counter())


# --------------------------------------------------------------------------------------------- latency


def test_latency_of_an_answer_and_an_alert() -> None:
    book = LatencyBook()
    with book.question(7, origin=100.0):
        book.on_say("answer", "First sentence.")
        book.on_say("answer", "Second sentence.")  # only the first one is timed
    with book.detection(origin=200.0):
        book.on_say("alert", "You're up, check and go.", "your_turn")
    assert book.on_played("answer", "Second sentence.", 101.0) is None
    measured = book.on_played("answer", "First sentence.", 100.8)
    assert measured is not None and measured.seconds == pytest.approx(0.8) and book.answers == {7: pytest.approx(0.8)}
    alert = book.on_played("alert", "You're up, check and go.", 200.25)
    assert alert is not None and alert.key == "your_turn" and alert.seconds == pytest.approx(0.25)
    book.on_say("answer", "Not inside a question.")  # e.g. "repeat that" from elsewhere: not timed
    assert book.on_played("answer", "Not inside a question.", 300.0) is None


def test_latency_of_a_dropped_utterance_is_not_counted() -> None:
    book = LatencyBook()
    with book.question(1, origin=10.0):
        book.on_say("answer", "Stopped before it played.")
    assert book.on_played("answer", "Stopped before it played.", None) is None and book.answers == {}


# ------------------------------------------------------------------------------------------ lane reader


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"lanes": 3, "our_lane": 2, "confidence": "high"}, (3, 2)),
        ({"lanes": 3, "our_lane": 2, "confidence": "medium"}, None),  # unreliable counting: only confident ones
        ({"lanes": 3, "our_lane": 4, "confidence": "high"}, None),
        ({"lanes": True, "our_lane": 1, "confidence": "high"}, None),
        ({"lanes": "3", "our_lane": 1, "confidence": "high"}, None),
        ({"lanes": 12, "our_lane": 1, "confidence": "high"}, None),
        ({}, None),
    ],
)
def test_lane_reading_is_used_only_when_sure(data: dict[str, Any], expected: tuple[int, int] | None) -> None:
    assert parse_lane_reading(data) == expected


def test_lane_reader_asks_about_the_latest_frame() -> None:
    cfg = load_config().vision
    model = MockVision('{"lanes": 4, "our_lane": 3, "confidence": "high"}')
    saved: list[Frame] = []
    latest = frame(12.0)
    assert VisionLaneReader(model, lambda: latest, cfg, on_frame=saved.append)() == (4, 3)
    assert saved == [latest] and len(model.requests[0].images) == 1
    failing = MockVision(error=VisionError("mock: timeout"))
    assert VisionLaneReader(failing, lambda: latest, cfg)() is None
    assert VisionLaneReader(model, lambda: None, cfg)() is None


# ------------------------------------------------------------------------------------------- hub + server


class Actions:
    def __init__(self) -> None:
        self.asked: list[str] = []
        self.controls: list[Any] = []
        self.fixes: list[tuple[float, float]] = []

    def ask_typed(self, text: str) -> None:
        self.asked.append(text)

    def toggle_pause(self) -> bool | None:
        self.controls.append("pause")
        return True

    def seek(self, delta_s: float) -> bool:
        self.controls.append(delta_s)
        return True

    def gps_fix(self, lat: float, lon: float, speed: Any, heading: Any, accuracy: Any) -> bool:
        self.fixes.append((lat, lon))
        return True


def test_server_pages_and_api() -> None:
    hub, actions = UiHub(), Actions()
    with TestClient(create_server(hub, actions)) as client:
        assert "DrivePilot" in client.get("/").text and "watchPosition" in client.get("/gps").text
        assert client.post("/api/ask", json={"text": "  who got here first?  "}).json() == {"ok": True}
        assert client.post("/api/ask", data={"text": "hi"}).status_code == 415  # a plain form from another site
        assert client.post("/api/ask", json={"text": ""}).status_code == 422
        assert client.post("/api/control", json={"action": "pause"}).json() == {"ok": True, "paused": True}
        assert client.post("/api/control", json={"action": "seek", "delta_s": -5}).json() == {"ok": True}
        assert client.post("/api/control", json={"action": "reboot"}).status_code == 422
        assert client.post("/api/gps", json={"lat": 47.61, "lon": -122.2, "heading": 90.0}).json() == {"ok": True}
        assert client.post("/api/gps", json={"lat": 147.0, "lon": 0.0}).status_code == 422
    assert actions.asked == ["who got here first?"] and actions.controls == ["pause", -5.0]
    assert actions.fixes == [(47.61, -122.2)]


def test_websocket_gets_history_then_live_messages() -> None:
    hub = UiHub()
    hub.publish("question", text="who got here first?", via="typed", t=1.0)
    hub.set_status(t=1.0, live=False)
    with TestClient(create_server(hub, Actions())) as client, client.websocket_connect("/ws") as socket:
        hello = socket.receive_json()
        assert hello["type"] == "hello" and hello["session"] == hub.session
        assert [m["text"] for m in hello["history"]] == ["who got here first?"] and hello["status"]["live"] is False
        threading.Thread(target=hub.publish, args=("answer",), kwargs={"text": "You're first."}).start()
        message = socket.receive_json()
        assert message["type"] == "answer" and message["text"] == "You're first." and message["id"] == 2


def test_video_stream_sends_jpeg_frames() -> None:
    hub = UiHub(width=160).start()
    hub.offer_frame(frame(width=640))
    assert wait_for(lambda: hub.latest_jpeg()[1] is not None)
    threading.Timer(0.5, hub.stop).start()  # ends the (otherwise endless) stream: TestClient reads it to the end
    with TestClient(create_server(hub, Actions())) as client:
        response = client.get("/video.mjpg")
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert response.content.startswith(b"--frame\r\nContent-Type: image/jpeg")
    jpeg = hub.latest_jpeg()[1]
    assert jpeg is not None
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[1] == 160  # scaled for the page


def test_hub_keeps_newest_frame_only_and_bounded_history() -> None:
    hub = UiHub(fps=1000.0, history=3)
    for i in range(5):
        hub.publish("note", text=str(i))
    assert [m["text"] for m in hub.history()] == ["2", "3", "4"]
    hub.start()
    hub.offer_frame(frame(1.0))
    hub.offer_frame(frame(2.0))
    assert wait_for(lambda: hub.encoded >= 1)
    hub.stop()
    assert hub.encoded <= 2


# ------------------------------------------------------------------------------------------- app wiring


class FakeDetector:
    def reset(self) -> None:
        pass

    def __call__(self, f: Frame) -> DetectionResult:
        return DetectionResult([Track(1, "car", 0.9, (10.0, 10.0, 50.0, 40.0))], {"detect_track_total": 1.0})


class InstantSpeaker(TextSpeaker):
    """Records like TextSpeaker and reports each utterance as played 10 ms later (like AudioSpeaker)."""

    def __init__(self) -> None:
        super().__init__()
        self.on_record: Any = None
        self.cues = 0

    def say(self, utterance: Utterance) -> None:
        super().say(utterance)
        if self.on_record is not None:
            now = time.perf_counter()
            self.on_record(SpeechRecord(utterance.kind, utterance.text, now, now + 0.01))

    def listening(self) -> None:
        self.cues += 1


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("app") / "drive.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    rng = np.random.default_rng(3)
    base = rng.integers(0, 255, (180, 640, 3), np.uint8)
    for i in range(90):  # 3 s, the scene sliding by
        writer.write(np.ascontiguousarray(base[:, i : i + 320]))
    writer.release()
    return path


def test_app_runs_a_replay_end_to_end(tmp_path: Path, clip: Path) -> None:
    cfg = load_config()
    speaker = InstantSpeaker()
    vision = MockVision("That sign says Main Street.")
    gps = GpxReplay.from_points([0.0, 3.0], [START, LINE.point_at(45.0)])
    with RunRecorder(tmp_path, "test", {"x": 1}) as recorder:
        hub = UiHub()
        app = DrivePilotApp(
            cfg,
            lambda: FileSource(clip, realtime=True),
            FakeDetector(),  # type: ignore[arg-type]
            speaker,
            vision=vision,
            routing=FixtureClient({"redmond town center": PLACE}, {"redmond town center": RESPONSE}),
            gps=gps,
            recorder=recorder,
            hub=hub,
            scripted=[(0.3, "I'm going to Redmond Town Center"), (0.8, "what does that sign say?")],
            exit_at_end=True,
        )
        speaker.on_record = app.on_speech_record

        def speak_later() -> None:  # a spoken question, as the voice pipeline hands it over
            wait_for(lambda: len(app.questions) >= 2)
            app.on_wake()
            now = time.perf_counter()
            app.on_heard(HeardQuestion("what's my next turn?", 1.0, 2.0, 2.6, now, now + 0.2, 200.0, True, "silence"))

        talker = threading.Thread(target=speak_later)
        talker.start()
        assert app.run() == 0
        talker.join()
        run_dir = recorder.dir

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    routes = {q["text"]: (q["route"], q["answer"]) for q in summary["questions"]}
    assert routes["I'm going to Redmond Town Center"][1].startswith("Okay, Redmond Town Center")
    assert routes["what does that sign say?"] == ("model/scene", "That sign says Main Street.")
    assert routes["what's my next turn?"][0] == "nav/next_step"
    assert routes["what's my next turn?"][1].startswith("Turn left onto 148th Avenue Northeast")
    voice = next(q for q in summary["questions"] if q["via"] == "voice")
    assert voice["first_audio_s"] == pytest.approx(0.6 + 0.2, abs=0.2)  # from the end of speech, not of endpointing
    assert summary["first_audio_s"]["nav"]["n"] == 1 and summary["perception"]["frames"] > 30
    assert speaker.cues == 1  # the wake word blipped and stopped speech

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = {e["kind"] for e in events}
    assert {"start", "question", "answer", "said", "gps", "world", "latency"} <= kinds
    assert any("model_t" in p.name for p in (run_dir / "frames").iterdir())  # the frames the model saw
    assert (run_dir / "detections.jsonl").stat().st_size > 0
    assert vision.requests and "Navigation: next" in vision.requests[0].text  # route context for the model
    messages = [(m["type"], m.get("text")) for m in hub.history()]
    assert ("question", "what does that sign say?") in messages
    assert ("answer", "That sign says Main Street.") in messages
    assert ("listening", None) in messages


class FlakyCamera(VideoSource):
    """A live source: the first one fails after a few frames (unplugged), later ones keep going."""

    instances = 0

    def __init__(self) -> None:
        super().__init__(WallClock(), "fake-camera")
        FlakyCamera.instances += 1
        self.fails = FlakyCamera.instances == 1

    @property
    def is_live(self) -> bool:
        return True

    @property
    def fps(self) -> float:
        return 30.0

    def _open(self) -> None:
        pass

    def _run(self) -> None:
        image = np.full((90, 160, 3), 80, np.uint8)
        for i in range(10**6):
            if self._stop.is_set():
                return
            if self.fails and i == 10:
                raise RuntimeError("camera unplugged")
            self._publish(Frame(image=image, t=self.clock.now(), index=i, captured_at=time.perf_counter()))
            time.sleep(1 / 30)

    def _close(self) -> None:
        pass


def test_a_lost_camera_is_restarted_and_said_while_it_is_out() -> None:
    FlakyCamera.instances = 0
    speaker = TextSpeaker()
    app = DrivePilotApp(load_config(), FlakyCamera, FakeDetector(), speaker)  # type: ignore[arg-type]
    result: list[int] = []
    runner = threading.Thread(target=lambda: result.append(app.run()))
    runner.start()
    try:
        assert wait_for(lambda: app.camera_restarts == 1 and app.worker is not None and app.worker.processed > 5)
    finally:
        app._stop.set()
        runner.join(10.0)
    assert result == [0] and FlakyCamera.instances == 2
    time.sleep(STALE_WAIT)  # the camera is stopped now: the snapshot is stale
    assert app.assistant.ask("who got here first?").text == CAMERA_LOST


STALE_WAIT = 1.6


def test_phone_gps_page_feeds_phone_gps_only(clip: Path) -> None:
    cfg = load_config()
    phone = PhoneGps()
    app = DrivePilotApp(cfg, lambda: FileSource(clip), FakeDetector(), TextSpeaker(), gps=phone)  # type: ignore[arg-type]
    assert app.gps_fix_from_phone(47.6, -122.2, 10.0, 90.0, 5.0) and phone.fix_at(0.0) is not None
    replay = DrivePilotApp(cfg, lambda: FileSource(clip), FakeDetector(), TextSpeaker())  # type: ignore[arg-type]
    assert replay.gps_fix_from_phone(47.6, -122.2, None, None, None) is False


def test_cli_options() -> None:
    assert parse_ask("22.5=who got here first?") == (22.5, "who got here first?")
    args = build_parser().parse_args(["--video", "clip.mov", "--gpx", "track.gpx"])
    cfg = apply_cli(load_config(), args)
    assert cfg.video.source == "file" and cfg.video.file.path.name == "clip.mov" and cfg.video.file.path.is_absolute()
    assert cfg.video.file.gpx_path is not None and cfg.gps_source == "gpx"
    assert apply_cli(load_config(), build_parser().parse_args(["--camera"])).video.source == "camera"
