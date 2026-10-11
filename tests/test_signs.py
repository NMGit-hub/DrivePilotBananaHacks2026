"""Sign detector (V2-M6): taxonomy and label mapping, value parsing, dataset conversion, the detector's contract."""

from __future__ import annotations

import json
import os
import random
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
import torch

from core.config import SignsConfig, load_config
from core.types import Frame
from perception import signs as signs_mod
from perception.sign_classes import KIND_INDEX, KINDS, SPEED_VALUES, VALUE_KINDS, kind_from_label, parse_speed_value
from perception.signs import (
    RawSign,
    SignDetection,
    SignDetector,
    SignGpuError,
    check_value_classes,
    crop_square,
    decide_value,
    engine_is_fp16,
    engine_path,
    engine_staleness,
    file_sha256,
    parse_value_class,
    select,
    sidecar_path,
    touches_border,
)
from scripts.signs import label_bdd_signs as labeler
from scripts.signs import prepare_data as prep

# ------------------------------------------------------------------------------------------ taxonomy


def test_kinds_are_stable_and_value_kinds_known() -> None:
    # Trained weights depend on this exact order: changing it silently relabels every class.
    assert KINDS == (
        "speed_limit",
        "school_speed_limit",
        "stop",
        "yield",
        "do_not_enter",
        "wrong_way",
        "no_turn_on_red",
        "one_way",
        "no_left_turn",
        "no_right_turn",
        "no_u_turn",
        "pedestrian_crossing",
        "school_zone",
    )
    assert set(KINDS) >= VALUE_KINDS
    assert all(KIND_INDEX[k] == i for i, k in enumerate(KINDS))
    assert SPEED_VALUES[0] == 5 and SPEED_VALUES[-1] == 85 and all(v % 5 == 0 for v in SPEED_VALUES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("35", 35),
        (35, 35),
        ("mph45", 45),
        ("speedLimit25", 25),
        ("SPEED LIMIT 55 MPH", 55),
        ("schoolSpeedLimit25", 25),
        ("37", None),  # not a posted limit
        ("120", None),
        ("0", None),
        ("25 35", None),  # two numbers: ambiguous
        ("SPEED LIMIT", None),
        ("", None),
        (None, None),
        (True, None),  # bools are ints in Python: never a speed
        (90, None),
    ],
)
def test_parse_speed_value(text: Any, expected: int | None) -> None:
    assert parse_speed_value(text) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        # LISA names
        ("speedLimit35", ("speed_limit", 35)),
        ("speedLimit15", ("speed_limit", 15)),
        ("schoolSpeedLimit25", ("school_speed_limit", 25)),
        ("stop", ("stop", None)),
        ("yield", ("yield", None)),
        ("doNotEnter", ("do_not_enter", None)),
        ("noLeftTurn", ("no_left_turn", None)),
        ("noRightTurn", ("no_right_turn", None)),
        ("pedestrianCrossing", ("pedestrian_crossing", None)),
        ("school", ("school_zone", None)),
        # look-alikes that must never become speed limits
        ("rampSpeedAdvisory35", None),
        ("truckSpeedLimit55", None),
        ("zoneAhead25", None),
        ("speedLimitUrdbl", ("speed_limit", None)),
        ("stopAhead", None),
        ("yieldAhead", None),
        ("signalAhead", None),
        # ctacke folders and other spellings
        ("mph45", ("speed_limit", 45)),
        ("Speed Limit 30", ("speed_limit", 30)),
        ("speed-limit-65", ("speed_limit", 65)),
        ("WRONG WAY", ("wrong_way", None)),
        ("No Turn On Red", ("no_turn_on_red", None)),
        ("one_way", ("one_way", None)),
        ("no_u_turn", ("no_u_turn", None)),
        ("do_not_turn_l", ("no_left_turn", None)),
        ("street name", None),
    ],
)
def test_kind_from_label(label: str, expected: tuple[str, int | None] | None) -> None:
    assert kind_from_label(label) == expected


def test_speed_limit_with_implausible_number_keeps_kind_but_no_value() -> None:
    assert kind_from_label("speedLimit37") == ("speed_limit", None)


# ------------------------------------------------------------------------------------- labeler reply


def test_parse_reply_reads_every_cell_in_order() -> None:
    reply = 'Here: {"labels": [{"i": 2, "kind": "stop", "value": null}, {"i": 1, "kind": "speed_limit", "value": 35}]}'
    assert labeler.parse_reply(reply, 2) == [("speed_limit", 35), ("stop", None)]


def test_parse_reply_rejects_incomplete_or_malformed() -> None:
    assert labeler.parse_reply('{"labels": [{"i": 1, "kind": "stop", "value": null}]}', 2) is None
    assert labeler.parse_reply("no json here", 1) is None
    assert labeler.parse_reply('{"labels": [{"kind": "stop"}]}', 1) is None


def test_parse_reply_sanitizes_kind_and_value() -> None:
    reply = json.dumps(
        {
            "labels": [
                {"i": 1, "kind": "banana", "value": None},  # unknown kind -> unclear (masked, never background)
                {"i": 2, "kind": "speed_limit", "value": 37},  # implausible value -> None
                {"i": 3, "kind": "stop", "value": 35},  # value on a non-speed kind is dropped
                {"i": 4, "kind": "school_speed_limit", "value": "20"},
            ]
        }
    )
    assert labeler.parse_reply(reply, 4) == [
        ("unclear", None),
        ("speed_limit", None),
        ("stop", None),
        ("school_speed_limit", 20),
    ]


# -------------------------------------------------------------------------------- dataset conversion


def test_tiles_cover_frames_with_half_tile_overlap() -> None:
    for seed in range(20):
        tiles = prep.tiles_for(1280, 720, random.Random(seed))
        assert sorted(x for x, _ in tiles) == [0, 320, 640]
        assert all(0 <= y <= 720 - prep.TILE for _, y in tiles)
    xs = sorted(x for x, _ in prep.tiles_for(1024, 522, random.Random(0)))  # a LISA-sized frame
    assert xs[0] == 0 and xs[-1] + prep.TILE == 1024 and all(b - a <= prep.TILE // 2 for a, b in pairwise(xs))
    assert prep.tiles_for(640, 480, random.Random(0)) == [(0, 0)]


def _read_tile_labels(out: Path) -> dict[str, list[list[float]]]:
    res = {}
    for txt in (out / "labels" / "train").glob("*.txt"):
        res[txt.stem] = [[float(v) for v in line.split()] for line in txt.read_text().splitlines() if line]
    return res


def _det_dirs(tmp_path: Path) -> Path:
    for sub in ("images", "labels"):
        (tmp_path / sub / "train").mkdir(parents=True)
    return tmp_path


def test_write_tiles_labels_and_masks(tmp_path: Path) -> None:
    out = _det_dirs(tmp_path)
    img = np.full((720, 1280, 3), 80, np.uint8)
    img[100:140, 200:232] = 255  # the "speed limit" (white)
    img[300:330, 1000:1030] = 250  # an unclear sign: must be painted over
    boxes = [
        prep.Box((200.0, 100.0, 232.0, 140.0), "speed_limit", 35),
        prep.Box((1000.0, 300.0, 1030.0, 330.0), None),
    ]
    sample = prep.Sample(Path("x/shard_000/abc.jpg"), boxes, "bdd")
    offsets = prep.tiles_for(1280, 720, random.Random(0))  # write_tiles draws the same offsets from the same seed
    counts = prep.write_tiles(sample, img, out, "train", random.Random(0), keep_empty=1.0)
    assert counts["speed_limit"] == 1
    labels = _read_tile_labels(out)
    (x0, y0) = offsets[0]
    left = next(v for k, v in labels.items() if k.endswith("_0"))
    assert len(left) == 1
    cls, cx, cy, w, h = left[0]
    assert cls == KIND_INDEX["speed_limit"]
    assert abs(cx * 640 - (216 - x0)) < 1e-2 and abs(cy * 640 - (120 - y0)) < 1e-2
    assert abs(w * 640 - 32) < 1e-3 and abs(h * 640 - 40) < 1e-3
    import cv2

    (x2, y2) = offsets[2]  # the right tile (x = 640) holds the unclear sign
    tile = cv2.imread(str(next((out / "images" / "train").glob("*_2.jpg"))))
    region = tile[300 - y2 : 330 - y2, 1000 - x2 : 1030 - x2]
    assert region.size and region.max() < 200  # the unclear sign is gone (masked with its surroundings' color)
    assert not labels[next(k for k in labels if k.endswith("_2"))]  # and it is not a label either


def test_box_cut_by_a_tile_edge_is_masked_not_labeled(tmp_path: Path) -> None:
    out = _det_dirs(tmp_path)
    img = np.full((720, 1280, 3), 80, np.uint8)
    # A sign at x=600..700 straddles the left tile's edge (640) and the right tile's (640): whole only in the middle.
    sample = prep.Sample(Path("x/s/abc.jpg"), [prep.Box((600.0, 200.0, 700.0, 300.0), "stop")], "bdd")
    prep.write_tiles(sample, img, out, "train", random.Random(3), keep_empty=1.0)
    labels = _read_tile_labels(out)
    assert [len(labels[k]) for k in sorted(labels)] == [0, 1, 0]
    ((_cls, _cx, _cy, w, h),) = labels[sorted(labels)[1]]
    assert abs(w * 640 - 100) < 1e-3 and abs(h * 640 - 100) < 1e-3  # the whole box, never a clipped part


def test_padded_tile_rescales_labels() -> None:
    assert prep._rescale_line("2 0.5 0.5 0.1 0.2", 0.5, 0.75) == "2 0.250000 0.375000 0.050000 0.150000"


def test_crop_matches_training_crop_and_pads() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    box = (100.3, 50.7, 130.2, 88.9)
    a = crop_square(img, box, 96)
    b = prep.crop_square(img, box)
    assert a is not None and b is not None and a.shape == (96, 96, 3)
    assert np.array_equal(a, b)  # runtime and training crops must be identical
    assert crop_square(img, (5.0, 5.0, 6.0, 6.0), 96) is None  # degenerate
    edge = crop_square(img, (1270.0, 700.0, 1290.0, 730.0), 96)  # partly outside: clipped, still a crop
    assert edge is not None and edge.shape == (96, 96, 3)


def test_value_class_names_round_trip() -> None:
    for v in SPEED_VALUES:
        assert parse_value_class(prep.value_class(v)) == v
    assert prep.value_class(None) == "none"
    assert parse_value_class("none") is None
    assert parse_value_class("v37") is None and parse_value_class("vx") is None and parse_value_class("35") is None


def test_stable_split_is_deterministic() -> None:
    assert prep.stable_frac("abc") == prep.stable_frac("abc")
    fracs = [prep.stable_frac(f"img{i}") for i in range(2000)]
    assert 0.04 < sum(f < 0.06 for f in fracs) / len(fracs) < 0.08


def test_mask_box_paints_the_box() -> None:
    img = np.full((100, 100, 3), 50, np.uint8)
    img[40:60, 40:60] = 255
    prep.mask_box(img, (40.0, 40.0, 60.0, 60.0))
    assert img[40:60, 40:60].max() == 50


# ----------------------------------------------------------------------------------- value decision


def test_decide_value_needs_confidence_and_a_speed_class() -> None:
    names = {0: "none", 1: "v25", 2: "v35"}
    assert decide_value(np.array([0.02, 0.03, 0.95]), names, 0.9) == (35, pytest.approx(0.95))
    assert decide_value(np.array([0.1, 0.05, 0.85]), names, 0.9)[0] is None  # not confident enough
    assert decide_value(np.array([0.97, 0.02, 0.01]), names, 0.9)[0] is None  # "none" wins: no value


# ------------------------------------------------------------------------------ detector contract


class _FakeBoxes:
    def __init__(self, rows: list[tuple[float, float, float, float, float, int]]) -> None:
        arr = np.array(rows, dtype=np.float32).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = arr[:, :4], arr[:, 4], arr[:, 5]

    def cpu(self) -> _FakeBoxes:
        return self

    def numpy(self) -> _FakeBoxes:
        return self


class _FakeYOLO:
    """Stands in for ultralytics.YOLO: detector boxes and classifier probabilities come from class attributes."""

    det_rows: ClassVar[list[tuple[float, float, float, float, float, int]]] = []
    probs: ClassVar[list[float]] = []  # for classes 0-2 ("none", "v25", "v35"); the other values get 0
    cls_names: ClassVar[dict[int, str]] = {
        0: "none",
        1: "v25",
        2: "v35",
        **{i + 3: f"v{v}" for i, v in enumerate(v for v in SPEED_VALUES if v not in (25, 35))},
    }
    det_names: ClassVar[dict[int, str]] = dict(enumerate(KINDS))
    view_probs: ClassVar[list[list[float]] | None] = None  # per-crop probabilities (views); None: `probs` for all
    crop_counts: ClassVar[list[int]] = []  # crops per classifier call
    cls_calls = 0

    def __init__(self, path: str, task: str) -> None:
        self.task = task
        self.names = self.det_names if task == "detect" else self.cls_names
        self.predictor = SimpleNamespace(model=None)

    def predict(self, image: np.ndarray | list[np.ndarray], **kwargs: Any) -> list[Any]:
        if self.task == "detect":
            return [SimpleNamespace(boxes=_FakeBoxes(self.det_rows), speed={"inference": 1.0})]
        type(self).cls_calls += 1
        crops = image if isinstance(image, list) else [image]
        type(self).crop_counts.append(len(crops))
        assert all(c.shape == (96, 96, 3) for c in crops)
        per_crop = self.view_probs or [self.probs] * len(crops)
        out = []
        for i in range(len(crops)):
            p = per_crop[i % len(per_crop)]
            out.append(
                SimpleNamespace(probs=SimpleNamespace(data=torch.tensor(p + [0.0] * (len(self.cls_names) - len(p)))))
            )
        return out


@pytest.fixture
def fake_detector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setattr(signs_mod, "YOLO", _FakeYOLO)
    monkeypatch.setattr(signs_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(signs_mod.torch.cuda, "mem_get_info", lambda *_: (0, 1))
    monkeypatch.setattr(SignDetector, "_verify_gpu", lambda self, what, model: ("cuda:0", "fp16"))
    _FakeYOLO.cls_calls = 0
    _FakeYOLO.det_names = dict(enumerate(KINDS))
    _FakeYOLO.view_probs = None
    _FakeYOLO.crop_counts = []

    def make(**overrides: Any) -> SignDetector:
        cfg = SignsConfig(engine="pytorch", **overrides)
        return SignDetector(cfg, tmp_path)

    return make


def _frame(w: int = 1280, h: int = 720) -> Frame:
    return Frame(np.zeros((h, w, 3), np.uint8), t=0.0, index=0, captured_at=0.0)


def test_detector_reads_a_confident_speed_value(fake_detector: Any) -> None:
    _FakeYOLO.det_rows = [(600, 200, 630, 240, 0.9, KIND_INDEX["speed_limit"]), (100, 100, 140, 140, 0.8, 2)]
    _FakeYOLO.probs = [0.0005, 0.0005, 0.999]
    dets = fake_detector().detect(_frame())
    assert [d.kind for d in dets] == ["speed_limit", "stop"]
    assert dets[0].value == 35 and dets[0].value_conf == pytest.approx(0.999)
    assert dets[1].value is None and dets[1].value_conf is None
    assert dets[0].box == (600.0, 200.0, 630.0, 240.0)
    assert isinstance(dets[0], SignDetection) and dets[0].to_record()["value"] == 35


def test_detector_withholds_an_unsure_value(fake_detector: Any) -> None:
    _FakeYOLO.det_rows = [(600, 200, 630, 240, 0.9, KIND_INDEX["speed_limit"])]
    _FakeYOLO.probs = [0.1, 0.45, 0.45]
    (det,) = fake_detector().detect(_frame())
    assert det.kind == "speed_limit" and det.value is None  # a wrong limit is worse than none


def test_small_sign_gets_no_value_attempt(fake_detector: Any) -> None:
    _FakeYOLO.det_rows = [(600, 200, 610, 212, 0.9, KIND_INDEX["speed_limit"])]  # 12 px < value_min_px
    _FakeYOLO.probs = [0.0, 0.0, 1.0]
    (det,) = fake_detector().detect(_frame())
    assert det.value is None and det.value_conf is None and _FakeYOLO.cls_calls == 3  # warmup calls only


def test_value_min_px_is_measured_at_1280_width(fake_detector: Any) -> None:
    # 1080p: a 20 px box is 13.3 px at 1280 wide -> below 16, no value attempt.
    _FakeYOLO.det_rows = [(900, 300, 915, 320, 0.9, KIND_INDEX["speed_limit"])]
    _FakeYOLO.probs = [0.0, 0.0, 1.0]
    (det,) = fake_detector().detect(_frame(1920, 1080))
    assert det.value is None
    _FakeYOLO.det_rows = [(900, 300, 920, 330, 0.9, KIND_INDEX["speed_limit"])]  # 30 px -> 20 px
    (det,) = fake_detector().detect(_frame(1920, 1080))
    assert det.value == 35


def test_school_speed_limit_also_gets_a_value(fake_detector: Any) -> None:
    _FakeYOLO.det_rows = [(600, 200, 625, 250, 0.7, KIND_INDEX["school_speed_limit"])]
    _FakeYOLO.probs = [0.0, 0.999, 0.001]
    (det,) = fake_detector().detect(_frame())
    assert det.kind == "school_speed_limit" and det.value == 25


def test_per_kind_confidence_threshold(fake_detector: Any) -> None:
    _FakeYOLO.det_rows = [(0, 0, 30, 30, 0.45, KIND_INDEX["stop"]), (50, 0, 80, 30, 0.45, KIND_INDEX["yield"])]
    _FakeYOLO.probs = [1.0, 0.0, 0.0]
    dets = fake_detector(conf=0.4, class_conf={"stop": 0.5}).detect(_frame())
    assert [d.kind for d in dets] == ["yield"]


def test_one_sign_reported_as_two_kinds_keeps_the_surer_one(fake_detector: Any) -> None:
    _FakeYOLO.det_rows = [
        (600, 200, 630, 240, 0.6, KIND_INDEX["school_speed_limit"]),
        (601, 201, 631, 241, 0.8, KIND_INDEX["speed_limit"]),
        (700, 200, 730, 240, 0.7, KIND_INDEX["stop"]),  # a different sign nearby stays
    ]
    _FakeYOLO.probs = [0.0, 0.0, 1.0]
    dets = fake_detector().detect(_frame())
    assert [d.kind for d in dets] == ["speed_limit", "stop"]


def test_drop_duplicates_keeps_stacked_signs() -> None:
    from perception.signs import drop_duplicates

    a = SignDetection("one_way", None, 0.9, (100.0, 100.0, 160.0, 120.0))
    b = SignDetection("stop", None, 0.8, (110.0, 122.0, 150.0, 162.0))  # mounted right below: no overlap
    c = SignDetection("yield", None, 0.5, (100.0, 101.0, 160.0, 121.0))  # same box as a, less sure
    assert drop_duplicates([c, b, a], 0.6) == [a, b]


def test_detections_sorted_and_timed(fake_detector: Any) -> None:
    _FakeYOLO.det_rows = [(0, 0, 30, 30, 0.5, 2), (50, 0, 80, 30, 0.9, 3)]
    det = fake_detector()
    out = det.detect(_frame())
    assert [d.conf for d in out] == sorted((d.conf for d in out), reverse=True)
    assert {"inference", "detect_total", "classify_total", "total"} <= set(det.last_timings_ms)


def test_unknown_class_conf_kind_rejected(fake_detector: Any) -> None:
    with pytest.raises(ValueError, match="unknown kinds"):
        fake_detector(class_conf={"speed_limits": 0.5})


def test_detector_refuses_mismatched_classes(fake_detector: Any) -> None:
    _FakeYOLO.det_names = {0: "car", 1: "truck"}
    with pytest.raises(ValueError, match="taxonomy"):
        fake_detector()


def test_detector_refuses_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(signs_mod.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SignGpuError):
        SignDetector(SignsConfig(), tmp_path)


def test_engine_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(signs_mod, "YOLO", _FakeYOLO)
    monkeypatch.setattr(signs_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(signs_mod.torch.cuda, "mem_get_info", lambda *_: (0, 1))
    monkeypatch.setattr(SignDetector, "_verify_gpu", lambda self, what, model: ("cuda:0", "fp16"))
    _FakeYOLO.det_names = dict(enumerate(KINDS))
    cfg = SignsConfig(weights=tmp_path / "det.pt", classifier_weights=tmp_path / "cls.pt")
    with pytest.raises(FileNotFoundError):  # neither engine nor weights
        SignDetector(cfg, tmp_path)
    (tmp_path / "det.pt").write_bytes(b"")
    (tmp_path / "cls.pt").write_bytes(b"")
    d = SignDetector(cfg, tmp_path)  # weights only: GPU PyTorch fallback
    assert d.det_path.suffix == ".pt" and d._fallback == {"detector": True, "classifier": True}
    det_engine = engine_path(tmp_path / "det.pt", (736, 1280), True)
    cls_engine = engine_path(tmp_path / "cls.pt", (96, 96), True)
    assert det_engine.name == "det_736x1280_fp16.engine" and cls_engine.name == "cls_96x96_fp16.engine"
    det_engine.write_bytes(b"")
    cls_engine.write_bytes(b"")
    d = SignDetector(cfg, tmp_path)
    assert d.det_path == det_engine and d.cls_path == cls_engine and not any(d._fallback.values())


def test_config_section_loads_and_validates() -> None:
    cfg = load_config()
    assert cfg.signs.imgsz == (736, 1280) and 0 < cfg.signs.value_conf <= 1
    with pytest.raises(Exception, match="multiples of 32"):
        SignsConfig(imgsz=(720, 1280))
    with pytest.raises(Exception, match="between 0 and 1"):
        SignsConfig(class_conf={"stop": 1.5})


# ------------------------------------------------------------------------------------ real engines

_CFG = load_config()
_DET_ENGINE = engine_path(_CFG.resolve_path(_CFG.signs.weights), _CFG.signs.imgsz, _CFG.signs.half)


# The dev clips live in the main checkout (data/videos/dev, gitignored); SIGNS_DEV_CLIPS points elsewhere.
_SCHOOL_CLIP = (
    Path(os.environ.get("SIGNS_DEV_CLIPS", str(_CFG.root / "data" / "videos" / "dev")))
    / "red_light_rain_school_zone__b61456bb-5bfb50ee.mov"
)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA GPU")
@pytest.mark.skipif(not _DET_ENGINE.exists(), reason="sign engines not built (scripts/signs/export_engines.py)")
def test_real_engines_run_on_cuda_fp16() -> None:
    det = SignDetector(_CFG.signs, _CFG.root)
    for name, model in (("detector", det.det), ("classifier", det.cls)):
        device, precision = det._verify_gpu(name, model)  # raises if not on the GPU as configured
        assert device.startswith("cuda") and precision == "fp16"
    assert det.det_path.suffix == ".engine" and det.cls_path.suffix == ".engine"
    # A blank frame: no signs, and fast (the measured numbers are in scripts/signs/STATUS.md).
    times = []
    for _ in range(20):
        assert det.detect(_frame()) == []
        times.append(det.last_timings_ms["total"])
    assert float(np.median(times)) < 30.0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA GPU")
@pytest.mark.skipif(not _DET_ENGINE.exists(), reason="sign engines not built (scripts/signs/export_engines.py)")
@pytest.mark.skipif(not _SCHOOL_CLIP.exists(), reason="dev clip missing (scripts/fetch_dev_clips.py)")
def test_real_engines_read_the_do_not_enter_signs_on_a_dev_clip() -> None:
    import cv2

    det = SignDetector(_CFG.signs, _CFG.root)
    cap = cv2.VideoCapture(str(_SCHOOL_CLIP))
    cap.set(cv2.CAP_PROP_POS_MSEC, 25_000)  # the sheet's "what do those red signs ahead say?" (DO NOT ENTER x2)
    ok, img = cap.read()
    cap.release()
    assert ok
    out = det.detect(Frame(img, 25.0, 0, 0.0))
    assert any(d.kind == "do_not_enter" for d in out), out
    assert all(d.value is None for d in out if d.kind not in ("speed_limit", "school_speed_limit"))


# ------------------------------------------------------------------------- the app's rules: select()


def test_select_applies_per_kind_conf_then_duplicates_then_the_value_rule() -> None:
    cfg = SignsConfig(conf=0.4, class_conf={"stop": 0.6}, value_conf=0.9, value_min_px=16.0, duplicate_iou=0.6)
    raw = [
        RawSign("stop", 0.5, (10.0, 10.0, 40.0, 40.0)),  # below stop's own 0.6: dropped
        RawSign("speed_limit", 0.8, (100.0, 100.0, 130.0, 140.0), 35, 0.95),  # value reported
        RawSign("school_speed_limit", 0.7, (101.0, 101.0, 131.0, 141.0), 25, 0.99),  # same sign, less sure
        RawSign("speed_limit", 0.6, (300.0, 100.0, 330.0, 140.0), 45, 0.85),  # unsure: no value
        RawSign("speed_limit", 0.55, (500.0, 100.0, 510.0, 112.0), 25, 0.99),  # 12 px: not attempted
        RawSign("speed_limit", 0.52, (700.0, 100.0, 730.0, 140.0), None, 0.99),  # reject class wins: no value
        RawSign("yield", 0.5, (900.0, 100.0, 940.0, 140.0)),
    ]
    out = select(raw, cfg, 1280, 720)
    assert [(d.kind, d.value) for d in out] == [
        ("speed_limit", 35),
        ("speed_limit", None),
        ("speed_limit", None),
        ("speed_limit", None),
        ("yield", None),
    ]
    assert [d.value_conf for d in out] == [pytest.approx(0.95), pytest.approx(0.85), None, pytest.approx(0.99), None]


def test_select_withholds_values_at_the_frame_border() -> None:
    cfg = SignsConfig(value_conf=0.9, value_edge_px=2.0)

    def value(box: tuple[float, float, float, float], w: int = 1280, h: int = 720) -> int | None:
        return select([RawSign("speed_limit", 0.9, box, 35, 0.99)], cfg, w, h)[0].value

    assert value((1200.0, 5.0, 1230.0, 45.0)) == 35
    assert value((100.0, 0.5, 130.0, 40.0)) is None  # top edge: the number may be cut off
    assert value((1250.0, 300.0, 1279.0, 340.0)) is None  # right edge
    assert value((100.0, 680.0, 130.0, 719.0)) is None  # bottom edge
    assert value((100.0, 2.5, 130.0, 60.0), 1920, 1080) is None  # the margin is measured at 1280: 3 px at 1920
    assert touches_border((0.0, 10.0, 20.0, 30.0), 1280, 720, 0.0) is False  # edge_px 0 turns the rule off


def test_classifier_classes_must_be_exactly_the_value_set() -> None:
    good = dict(enumerate(["none", *(f"v{v}" for v in SPEED_VALUES)]))
    check_value_classes(good)
    with pytest.raises(ValueError, match="don't match"):
        check_value_classes({0: "none", 1: "v25", 2: "v35"})
    with pytest.raises(ValueError, match="v90"):
        check_value_classes({**good, 99: "v90"})


def test_detector_refuses_a_classifier_with_other_classes(fake_detector: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FakeYOLO, "cls_names", {0: "none", 1: "v25", 2: "v35"})
    with pytest.raises(ValueError, match="value classifier"):
        fake_detector()


def test_raw_detections_classify_down_to_the_given_height(fake_detector: Any) -> None:
    _FakeYOLO.det_rows = [
        (600, 200, 610, 212, 0.9, KIND_INDEX["speed_limit"]),  # 12 px tall
        (100, 100, 140, 140, 0.9, KIND_INDEX["stop"]),
    ]
    _FakeYOLO.probs = [0.0, 0.0, 1.0]
    det = fake_detector()
    raw = det.raw_detections(_frame(), classify_min_px=1.0)  # the evaluation's setting
    assert raw[0].top_value == 35 and raw[0].top_prob == pytest.approx(1.0)
    assert raw[1].top_value is None and raw[1].top_prob is None  # never classified: not a speed kind
    out = det.detect(_frame())  # the app: below value_min_px, so no value (and no classifier call)
    assert out[0].kind == "speed_limit" and out[0].value is None and out[0].value_conf is None


# ----------------------------------------------------------------------- GPU / FP16 / stale engines


class TensorRTBackend:
    """Stands in for ultralytics' TensorRT backend (_verify_gpu checks the class name)."""

    def __init__(self, quantize: int | None, fp16: bool, layers: list[str]) -> None:
        self.metadata = {"args": {"quantize": quantize}}
        self.fp16 = fp16
        self.model = layers  # engine_layer_names is patched to return this


def _verify(backend: Any, fallback: bool = False, **cfg: Any) -> tuple[str, str]:
    det = SignDetector.__new__(SignDetector)
    det.cfg = SignsConfig(**cfg)
    det._fallback = {"detector": fallback}
    return det._verify_gpu("detector", SimpleNamespace(predictor=SimpleNamespace(model=backend)))  # type: ignore[arg-type]


def test_verify_gpu_rejects_weights_on_the_cpu() -> None:
    with pytest.raises(SignGpuError, match="cpu"):
        _verify(SimpleNamespace(model=torch.nn.Linear(2, 2)), engine="pytorch")  # CPU, FP32
    with pytest.raises(SignGpuError):
        _verify(SimpleNamespace(model=torch.nn.Linear(2, 2)), fallback=True)  # the .pt fallback is checked too


def test_verify_gpu_checks_the_engine_really_computes_in_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signs_mod, "engine_layer_names", lambda engine: engine)

    def trt(quantize: int | None, fp16: bool, layers: list[str], device: str = "cuda:0") -> Any:
        return SimpleNamespace(backend=TensorRTBackend(quantize, fp16, layers), device=torch.device(device))

    # ModelOpt AutoCast engine: FP32 input binding, the input cast to FP16 inside the network.
    assert _verify(trt(16, False, ["images_cast_to_fp16", "/model.0/conv/Conv"])) == ("cuda:0", "fp16")
    assert _verify(trt(16, True, []))[1] == "fp16"  # FP16 bindings
    with pytest.raises(SignGpuError, match="fp32"):
        _verify(trt(None, False, ["conv"]))  # an FP32 engine while half is configured
    with pytest.raises(SignGpuError):
        _verify(trt(16, False, ["conv"]))  # metadata says 16, but nothing in the engine is FP16
    with pytest.raises(SignGpuError):
        _verify(trt(16, True, [], device="cpu"))
    assert _verify(trt(None, False, ["conv"]), half=False)[1] == "fp32"  # FP32 when that's configured
    assert engine_is_fp16(False, 16, ["images_cast_to_fp16"]) and not engine_is_fp16(True, None, [])


def test_stale_engine_detection(tmp_path: Path) -> None:
    import os

    w = tmp_path / "det.pt"
    w.write_bytes(b"new weights")
    e = engine_path(w, (736, 1280), True)
    e.write_bytes(b"engine")
    os.utime(w, (e.stat().st_mtime - 10, e.stat().st_mtime - 10))
    assert engine_staleness(e, w) is None  # no sidecar, engine newer than the weights
    sidecar_path(e).write_text(json.dumps({"weights_sha256": file_sha256(w)}))
    assert engine_staleness(e, w) is None
    sidecar_path(e).write_text(json.dumps({"weights_sha256": "0" * 64}))
    assert "other weights" in (engine_staleness(e, w) or "")
    sidecar_path(e).unlink()
    os.utime(w, (e.stat().st_mtime + 10, e.stat().st_mtime + 10))
    assert "newer" in (engine_staleness(e, w) or "")
    w.unlink()
    assert engine_staleness(e, w) is None  # engine delivered without weights: nothing to compare


def test_detector_falls_back_to_the_weights_when_the_engine_is_stale(fake_detector: Any, tmp_path: Path) -> None:
    for name, size in (("det", (736, 1280)), ("cls", (96, 96))):
        (tmp_path / f"{name}.pt").write_bytes(name.encode())
        engine = engine_path(tmp_path / f"{name}.pt", size, True)
        engine.write_bytes(b"engine")
        sidecar_path(engine).write_text(json.dumps({"weights_sha256": file_sha256(tmp_path / f"{name}.pt")}))
    cfg = SignsConfig(weights=tmp_path / "det.pt", classifier_weights=tmp_path / "cls.pt")
    d = SignDetector(cfg, tmp_path)
    assert d.det_path.suffix == ".engine" and not any(d._fallback.values())
    (tmp_path / "det.pt").write_bytes(b"retrained")  # new weights, old engine
    d = SignDetector(cfg, tmp_path)
    assert d.det_path.suffix == ".pt" and d._fallback == {"detector": True, "classifier": False}


# ---------------------------------------------------------------------------------- evaluation logic

from scripts.signs import evaluate as ev  # noqa: E402


def _cfg(**kw: Any) -> SignsConfig:
    return SignsConfig(**{"conf": 0.4, "value_conf": 0.9, "value_min_px": 16.0, **kw})


def _res(
    gts: list[ev.Gt],
    raw: list[RawSign],
    ignore: list[tuple[float, float, float, float]] | None = None,
    w: int = 1280,
    h: int = 720,
) -> ev.ImageResult:
    return ev.ImageResult("img", gts, ignore or [], raw, w, h)


def _sl(box: tuple[float, float, float, float], value: int | None, p: float = 0.99, conf: float = 0.9) -> RawSign:
    return RawSign("speed_limit", conf, box, value, p)


def test_eval_counts_right_wrong_and_withheld_values() -> None:
    sl = ev.Gt((100, 100, 130, 140), "speed_limit", 35, True)
    stop = ev.Gt((300, 100, 340, 140), "stop", None, None)
    raw = [
        _sl((101, 101, 131, 141), 35, 0.95),  # right value
        _sl((299, 99, 341, 141), 45, 0.99, conf=0.8),  # speed limit on a stop sign: FP + wrong value
        _sl((600, 100, 630, 140), 25, 0.5, conf=0.7),  # FP, value withheld (p < 0.9): not wrong
    ]
    s = ev.summarize([_res([sl, stop], raw)], _cfg())
    assert s["speed_values"] == {"speed_limit:right": 1}
    assert [w["said"] for w in s["wrong_values"]] == [45]
    assert s["reported_values"] == {"right": 1, "wrong": 1}
    assert s["per_kind"]["speed_limit"]["tp"] == 1 and s["per_kind"]["speed_limit"]["fp"] == 2
    assert s["per_kind"]["stop"]["fn"] == 1


def test_eval_values_near_ignore_regions_are_still_judged() -> None:
    # A value on a labeled sign of another kind that also lies on an ignore region is WRONG (this was a hole).
    stop = ev.Gt((300, 100, 340, 140), "stop", None, None)
    s = ev.summarize([_res([stop], [_sl((300, 100, 340, 140), 25)], ignore=[(299, 99, 341, 141)])], _cfg())
    assert len(s["wrong_values"]) == 1 and s["per_kind"]["stop"]["fn"] == 1
    # A wrong number on a labeled limit touching an ignore box: wrong.
    sl = ev.Gt((100, 100, 130, 140), "speed_limit", 35, True)
    s = ev.summarize([_res([sl], [_sl((100, 100, 130, 140), 25)], ignore=[(95, 95, 135, 145)])], _cfg())
    assert [w["said"] for w in s["wrong_values"]] == [25] and s["speed_values"] == {"speed_limit:wrong": 1}
    # Only a value on an ignore region with no labeled sign under it is unverifiable (listed for a look).
    s = ev.summarize([_res([], [_sl((100, 100, 130, 140), 25)], ignore=[(98, 98, 132, 142)])], _cfg())
    assert s["per_kind"] == {} and s["wrong_values"] == [] and [u["said"] for u in s["unverifiable_values"]] == [25]
    # A labeled speed limit whose number nobody could read: unverifiable too.
    unknown = ev.Gt((100, 100, 130, 140), "speed_limit", None, False)
    s = ev.summarize([_res([unknown], [_sl((100, 100, 130, 140), 25)])], _cfg())
    assert s["reported_values"] == {"unverifiable": 1}
    # A value far from every label and ignore region (e.g. on a sign-free frame): wrong.
    s = ev.summarize([_res([], [_sl((700, 100, 730, 140), 25)])], _cfg())
    assert len(s["wrong_values"]) == 1 and s["per_kind"]["speed_limit"]["fp"] == 1


def test_eval_small_or_unsure_values_are_not_reported() -> None:
    sl = ev.Gt((100, 100, 112, 112), "speed_limit", 35, True)  # 12 px tall
    s = ev.summarize([_res([sl], [_sl((100, 100, 112, 112), 30)])], _cfg())
    assert s["speed_values"] == {"speed_limit:no_value": 1} and not s["wrong_values"]
    # at 1080p the rule is measured at 1280 px width: 24 px tall at 1920 = 16 px -> value reported
    sl2 = ev.Gt((100, 100, 118, 124), "speed_limit", 35, True)
    r = _res([sl2], [_sl((100, 100, 118, 124), 35)], w=1920, h=1080)
    assert ev.summarize([r], _cfg())["speed_values"] == {"speed_limit:right": 1}


def test_eval_matching_is_by_kind_and_iou() -> None:
    g = ev.Gt((100, 100, 140, 140), "stop", None, None)
    wrong_kind = RawSign("yield", 0.9, (100.0, 100.0, 140.0, 140.0))
    far = RawSign("stop", 0.9, (150.0, 150.0, 190.0, 190.0))
    s = ev.summarize([_res([g], [wrong_kind, far])], _cfg())
    assert s["per_kind"]["stop"] == {**s["per_kind"]["stop"], "tp": 0, "fp": 1, "fn": 1}
    assert s["per_kind"]["yield"]["fp"] == 1


def test_eval_average_precision() -> None:
    gts = [ev.Gt((0, 0, 10, 10), "stop", None, None), ev.Gt((50, 0, 60, 10), "stop", None, None)]
    raw = [
        RawSign("stop", 0.9, (0.0, 0.0, 10.0, 10.0)),  # TP
        RawSign("stop", 0.8, (100.0, 0.0, 110.0, 10.0)),  # FP
        RawSign("stop", 0.7, (50.0, 0.0, 60.0, 10.0)),  # TP
    ]
    ap, n = ev.average_precision([_res(gts, raw)], "stop", _cfg())
    assert n == 2 and ap == pytest.approx(0.5 + 0.5 * (2 / 3))
    assert np.isnan(ev.average_precision([_res(gts, raw)], "yield", _cfg())[0])


def test_eval_value_sweep_counts_fall_with_the_threshold() -> None:
    sl = ev.Gt((100, 100, 130, 140), "speed_limit", 35, True)
    rows = ev.value_sweep([_res([sl], [_sl((100, 100, 130, 140), 45, 0.92)])], _cfg())
    by_t = {r["value_conf"]: r for r in rows}
    assert by_t[0.9]["wrong_total"] == 1 and by_t[0.93]["wrong_total"] == 0 and by_t[0.93]["readable"] == 1


def test_eval_raw_cache_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "RAW_CACHE", tmp_path)
    g = ev.Gt((0, 0, 10, 10), "stop", None, None)
    r = _res([g], [_sl((1.0, 2.0, 3.0, 4.0), 35, 0.9, conf=0.5)])
    fp = {"detector_sha256": "a", "classifier_sha256": "b", "imgsz": [736, 1280]}
    ev.save_raw("x", [r], {"frames": 1}, fp)
    (back,), timing = ev.load_raw("x", [("img", Path("img.jpg"), [g], [])], fp)
    assert back.raw == r.raw and back.gts == [g] and timing == {"frames": 1}
    with pytest.raises(KeyError):
        ev.load_raw("x", [("other", Path("o.jpg"), [], [])], fp)
    with pytest.raises(ValueError, match="other models"):  # detections of another detector are never reused
        ev.load_raw("x", [("img", Path("img.jpg"), [g], [])], {**fp, "detector_sha256": "c"})


def test_eval_model_fingerprint_follows_the_weights(tmp_path: Path) -> None:
    (tmp_path / "d.pt").write_bytes(b"one")
    (tmp_path / "c.pt").write_bytes(b"cls")
    cfg = SignsConfig(weights=tmp_path / "d.pt", classifier_weights=tmp_path / "c.pt")
    a = ev.model_fingerprint(cfg)
    (tmp_path / "d.pt").write_bytes(b"two")
    assert ev.model_fingerprint(cfg) != a and ev.model_fingerprint(cfg)["classifier_sha256"] == a["classifier_sha256"]


def test_eval_unverifiable_needs_a_real_overlap_and_counts_for_thresholds() -> None:
    # A big box whose centre falls in a tiny unlabeled sign box is not excused: the value is wrong.
    s = ev.summarize([_res([], [_sl((100, 100, 160, 180), 25)], ignore=[(128, 138, 134, 146)])], _cfg())
    assert len(s["wrong_values"]) == 1 and not s["unverifiable_values"]
    # Unverifiable values count in the sweep column used to choose thresholds.
    unknown = ev.Gt((100, 100, 130, 140), "speed_limit", None, False)
    school = ev.Gt((300, 100, 330, 140), "school_speed_limit", 20, True)
    raw = [_sl((100, 100, 130, 140), 25, 0.95), RawSign("school_speed_limit", 0.9, (300, 100, 330, 140), 20, 0.99)]
    by_t = {r["value_conf"]: r for r in ev.value_sweep([_res([unknown, school], raw)], _cfg())}
    assert by_t[0.9]["wrong_total"] == 0 and by_t[0.9]["wrong_or_unverifiable"] == 1
    assert by_t[0.97]["wrong_or_unverifiable"] == 0
    assert by_t[0.9]["readable"] == 1 and by_t[0.9]["right_on_readable"] == 1  # school limits count on both sides


def test_eval_refuses_missing_frames(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Det:
        def __init__(self, cfg: SignsConfig, root: Path) -> None:
            self.det_path = self.cls_path = Path("x")

    monkeypatch.setattr(signs_mod, "SignDetector", _Det)
    with pytest.raises(FileNotFoundError):
        ev.run_detector([("a", tmp_path / "missing.jpg", [], [])], SignsConfig())


def test_tune_set_is_the_val_fraction_minus_anything_trained(monkeypatch: pytest.MonkeyPatch) -> None:
    imgs = [Path(f"/x/img{i}.jpg") for i in range(300)]
    val = [p for p in imgs if prep.in_val(p)]
    assert len(val) >= 3
    boxes = [prep.Box((0.0, 0.0, 10.0, 10.0), "stop"), prep.Box((20.0, 0.0, 30.0, 10.0), None)]
    samples = [prep.Sample(p, boxes, "bdd") for p in imgs]
    monkeypatch.setattr(prep, "bdd_samples", lambda split, fixes=True: (samples, []))
    monkeypatch.setattr(prep, "training_images", lambda: {prep.rel_path(val[0])})
    items = ev.tune_items()
    assert [i[0] for i in items] == [p.name for p in val[1:]]
    _name, _path, gts, ignore = items[0]
    assert [g.kind for g in gts] == ["stop"] and ignore == [(20.0, 0.0, 30.0, 10.0)]


def test_heldout_items_read_signs_ignore_and_audit(tmp_path: Path) -> None:
    data = {
        "images": [
            {
                "image": "valid/shard_000/a.jpg",
                "signs": [{"box": [0, 0, 10, 20], "kind": "speed_limit", "value": 25, "readable": True}],
                "ignore": [[30, 0, 40, 20]],
                "audit": {
                    "signs": [{"box": [50, 0, 60, 20], "kind": "stop", "value": None, "readable": None}],
                    "ignore": [],
                },
            },
            {"image": "valid/shard_001/b.jpg", "signs": [], "ignore": [], "sign_free": True},
        ]
    }
    path = tmp_path / "h.json"
    path.write_text(json.dumps(data))
    with_audit = ev.heldout_items(path)
    assert [g.kind for g in with_audit[0][2]] == ["speed_limit", "stop"] and with_audit[0][3] == [(30, 0, 40, 20)]
    assert [g.kind for g in ev.heldout_items(path, audit=False)[0][2]] == ["speed_limit"]
    assert with_audit[1][2] == [] and with_audit[1][3] == []


def test_eval_values_threshold_rules(tmp_path: Path) -> None:
    from scripts.signs import eval_values as evv

    rows = evv.sweep([(35, 0.95), (25, 0.8), (None, 0.99), (40, 0.7)], [35, None, None, 45])
    at = {r["threshold"]: r for r in rows}
    assert at[0.7] == {"threshold": 0.7, "right": 1, "wrong": 2, "withheld": 1}
    assert at[0.85] == {"threshold": 0.85, "right": 1, "wrong": 0, "withheld": 3}
    assert evv.lowest_safe(rows) == 0.85
    unsafe = evv.sweep([(40, 0.999)], [45])  # wrong even at the highest threshold: no safe threshold at all
    assert evv.lowest_safe(unsafe) is None


# --------------------------------------------------------------------- dataset building (leakage etc.)


def test_plan_det_jobs_never_trains_on_the_val_fraction() -> None:
    imgs = [Path(f"/x/train/shard_000/img{i}.jpg") for i in range(400)]
    pos = [prep.Sample(p, [prep.Box((0.0, 0.0, 10.0, 10.0), "stop")], "bdd") for p in imgs[:100]]
    neg = [prep.Sample(p, [], "bdd") for p in imgs[100:]]
    hard = next(s for s in neg if not prep.in_val(s.image))
    jobs, synth = prep.plan_det_jobs(pos, neg, {prep.rel_path(hard.image)}, [], [], [], 0.3, 200)
    trained = [j[0].image for j in jobs if j[1] == "train"] + [s[0].image for s in synth]
    assert trained and not any(prep.in_val(p) for p in trained)
    assert all(j[1] == "val" for j in jobs if prep.in_val(j[0].image))
    assert len(synth) == 200 and all(s[1] == "train" for s in synth)
    assert any(j[0] is hard and j[3] == 1.0 for j in jobs)  # hard negatives keep all their tiles
    assert not {s[0].image for s in synth} & {j[0].image for j in jobs}  # a background is used once


def test_cls_backgrounds_come_from_the_training_fraction() -> None:
    neg = [prep.Sample(Path(f"/x/n{i}.jpg"), [], "bdd") for i in range(600)]
    bgs = prep.cls_backgrounds(neg)
    assert len(bgs) == prep.CLS_BACKGROUNDS and not any(prep.in_val(s.image) for s in bgs)


def test_synthetic_signs_never_land_on_existing_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    from scripts.signs import synth

    calls: list[tuple[int, int, int, int]] = []
    real = synth.composite

    def spy(bg: np.ndarray, sign: np.ndarray, alpha: np.ndarray, x: int, y: int, rng: Any, pole: bool) -> Any:
        calls.append((x, y, alpha.shape[1], alpha.shape[0]))
        return real(bg, sign, alpha, x, y, rng, pole)

    monkeypatch.setattr(synth, "composite", spy)

    class _Renderer:
        def any(self, kind: str, value: int | None, rng: Any) -> Image.Image:
            return Image.new("RGBA", (60, 75), (255, 255, 255, 255))

        def distractor(self, rng: Any) -> Image.Image:
            return Image.new("RGBA", (60, 60), (0, 128, 0, 255))

    existing = [prep.Box((600.0, 0.0, 1280.0, 720.0), "stop"), prep.Box((0.0, 0.0, 300.0, 400.0), None)]
    pasted = 0
    for seed in range(40):
        calls.clear()
        _img, boxes = prep.synth_signs(
            np.full((720, 1280, 3), 90, np.uint8), existing, _Renderer(), np.random.default_rng(seed)
        )
        for x, y, w, h in calls:
            assert all(prep._iou(prep.paste_footprint(x, y, w, h, 720), b.xyxy) == 0 for b in existing)
        assert boxes[:2] == existing and all(b.kind is not None for b in boxes[2:])  # every pasted sign is labeled
        pasted += len(boxes) - 2
    assert pasted > 0


def test_masking_never_paints_a_labeled_neighbour(tmp_path: Path) -> None:
    import cv2

    img = np.full((720, 1280, 3), 80, np.uint8)
    img[100:140, 200:232] = 255  # labeled speed limit
    img[100:140, 236:268] = 250  # unclear sign right next to it: its mask margin reaches the labeled one
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), img)
    boxes = [prep.Box((200.0, 100.0, 232.0, 140.0), "speed_limit", 35), prep.Box((236.0, 100.0, 268.0, 140.0), None)]
    loaded = prep.load_scaled(prep.Sample(path, boxes, "bdd"))
    assert loaded is not None
    out, _ = loaded
    assert out[100:140, 200:232].min() == 255  # the labeled sign is intact
    assert out[100:140, 240:264].max() < 200  # the unclear one is masked


def test_unboxed_detections_become_masks_only_where_nothing_is_labeled() -> None:
    s = prep.Sample(Path("/x/a.jpg"), [prep.Box((100.0, 100.0, 130.0, 140.0), "stop")], "bdd")
    unboxed = {prep.rel_path(s.image): [(101.0, 101.0, 131.0, 141.0), (500.0, 100.0, 530.0, 140.0)]}
    out = prep.with_unboxed_masks(s, unboxed)
    assert [b.kind for b in out.boxes] == ["stop", None] and out.boxes[1].xyxy == (500.0, 100.0, 530.0, 140.0)
    assert prep.with_unboxed_masks(s, {}) is s


def test_bdd_samples_mask_unknown_and_unlabeled_signs(monkeypatch: pytest.MonkeyPatch) -> None:
    img = "train/shard_000/a.jpg"
    recs = [
        {"key": "train/a/0", "image": img, "box": [0, 0, 10, 20], "kind": "stop", "value": None},
        {"key": "train/a/1", "image": img, "box": [20, 0, 30, 20], "kind": "unclear", "value": None},
        {"key": "train/a/2", "image": img, "box": [40, 0, 50, 20], "kind": "other", "value": None},
        {"key": "train/b/0", "image": "train/shard_000/b.jpg", "box": [0, 0, 10, 20], "kind": "other", "value": None},
    ]
    meta = {
        img: {"0": (0, 0, 10, 20), "1": (20, 0, 30, 20), "2": (40, 0, 50, 20), "3": (60, 0, 66, 8)},
        "train/shard_000/b.jpg": {"0": (0, 0, 10, 20)},
    }
    monkeypatch.setattr(prep, "read_labels", lambda split, fixes=True: recs)
    monkeypatch.setattr(prep, "dev_clip_ids", lambda: set())
    monkeypatch.setattr(prep, "bdd_sign_boxes", lambda split: meta)
    pos, neg = prep.bdd_samples("train")
    assert [s.image.name for s in pos] == ["a.jpg"] and [s.image.name for s in neg] == ["b.jpg"]
    # unclear and too-small-to-label boxes are masked; "other" signs stay background
    assert [(b.xyxy, b.kind) for b in pos[0].boxes] == [
        ((0, 0, 10, 20), "stop"),
        ((20, 0, 30, 20), None),
        ((60, 0, 66, 8), None),
    ]
    assert neg[0].boxes == []


def test_ctacke_samples_apply_the_reread_checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for folder in ("mph35", "mph25"):
        (tmp_path / "labels" / "train" / folder).mkdir(parents=True)
        (tmp_path / "images" / "train" / folder).mkdir(parents=True)
        (tmp_path / "images" / "train" / folder / "f.png").write_bytes(b"")
    lines = "0 0.5 0.5 0.01 0.02\n0 0.6 0.5 0.01 0.02\n0 0.7 0.5 0.01 0.02\n"
    (tmp_path / "labels" / "train" / "mph35" / "f.txt").write_text(lines)
    (tmp_path / "labels" / "train" / "mph25" / "f.txt").write_text("0 0.5 0.5 0.01 0.02\n")
    check = tmp_path / "check.jsonl"
    check.write_text(
        json.dumps({"key": "ctacke/train/mph35/f/0", "readable": True})
        + "\n"
        + json.dumps({"key": "ctacke/train/mph35/f/1", "readable": False})
        + "\n"
    )
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"key": "ctacke/train/mph35/f/2", "kind": "unclear"}) + "\n")
    monkeypatch.setattr(prep, "CTACKE", tmp_path)
    monkeypatch.setattr(prep, "CTACKE_CHECK", check)
    monkeypatch.setattr(prep, "CTACKE_CHECK_RAW", raw)
    by_folder = {s.image.parent.name: s for s in prep.ctacke_samples("train")}
    got = [(b.kind, b.value, b.cls_label) for b in by_folder["mph35"].boxes]
    assert got == [("speed_limit", 35, None), ("speed_limit", 35, "none"), (None, 35, "none")]
    assert [(b.kind, b.value) for b in by_folder["mph25"].boxes] == [("school_speed_limit", 25)]


def test_real_crops_keep_the_true_value_of_unreadable_limits(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "f.png"
    cv2.imwrite(str(path), np.full((200, 300, 3), 120, np.uint8))
    boxes = [
        prep.Box((50.0, 50.0, 90.0, 100.0), "speed_limit", 30, cls_label="none"),  # unreadable real 30
        prep.Box((150.0, 50.0, 190.0, 100.0), "speed_limit", 45),
        prep.Box((200.0, 50.0, 240.0, 100.0), "stop"),
        prep.Box((10.0, 10.0, 14.0, 14.0), "speed_limit", 25),  # too small
    ]
    labels = [label for label, _crop in prep._real_crops([prep.Sample(path, boxes, "bdd")], 2, 0, 14)]
    assert labels == ["none@30", "v45", "v45", "none"]


def test_heldout_build_applies_the_review(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.signs import make_heldout as mh

    img = "valid/shard_000/a.jpg"
    meta = {
        img: {
            "0": (0, 0, 10, 20),
            "1": (20, 0, 30, 20),
            "2": (40, 0, 50, 20),
            "3": (60, 0, 70, 20),
            "9": (90, 0, 96, 8),
        }
    }
    monkeypatch.setattr(mh, "bdd_sign_boxes", lambda split: meta)
    monkeypatch.setattr(mh, "dev_clip_ids", lambda: set())
    recs = [
        {"key": "valid/a/0", "image": img, "box": [0, 0, 10, 20], "kind": "speed_limit", "value": 35},
        {"key": "valid/a/1", "image": img, "box": [20, 0, 30, 20], "kind": "speed_limit", "value": 25},
        {"key": "valid/a/2", "image": img, "box": [40, 0, 50, 20], "kind": "other", "value": None},
        {"key": "valid/a/3", "image": img, "box": [60, 0, 70, 20], "kind": "stop", "value": None},
    ]
    review = {"not_clearly_readable": ["valid/a/1"], "ignore": ["valid/a/3"]}
    (im,), counts = mh.build_images(recs, review, {"valid/a/2": ("speed_limit", 40, True)}, audit=False)
    assert [(s["kind"], s["value"], s["readable"]) for s in im["signs"]] == [
        ("speed_limit", 35, True),
        ("speed_limit", 25, False),
        ("speed_limit", 40, True),
    ]
    assert im["ignore"] == [[60, 0, 70, 20], [90, 0, 96, 8]] and "audit" not in im and counts["readable_speed"] == 2


def test_unboxed_speed_kind_detections_are_not_masked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "unboxed.jsonl"
    rec = {"image": "a.jpg", "boxes": [[0, 0, 10, 20], [30, 0, 40, 20]], "kinds": ["speed_limit", "one_way"]}
    path.write_text(json.dumps(rec) + "\n")
    monkeypatch.setattr(prep, "UNBOXED", path)
    assert prep.read_unboxed() == {"a.jpg": [(30, 0, 40, 20)]}  # a look-alike "speed limit" stays a negative
    path.write_text(json.dumps({"image": "a.jpg", "boxes": [[0, 0, 10, 20]]}) + "\n")  # old file without kinds
    with pytest.raises(ValueError, match="kinds"):
        prep.read_unboxed()


def test_original_build_backgrounds_are_reproduced(monkeypatch: pytest.MonkeyPatch) -> None:
    pos = [prep.Sample(Path(f"/x/p{i}.jpg"), [prep.Box((0.0, 0.0, 9.0, 9.0), "stop")], "bdd") for i in range(20)]
    neg = [prep.Sample(Path(f"/x/n{i}.jpg"), [], "bdd") for i in range(5000)]
    monkeypatch.setattr(prep, "bdd_samples", lambda split, fixes=True: (pos, neg))
    # The first build, as it was written: one seed per positive, shuffle every negative, int(0.3 * pos) real
    # negatives, the next 3,500 synthetic backgrounds; the classifier builds took neg[:400] unshuffled.
    rng = random.Random(0)
    for _ in pos:
        rng.randrange(2**31)
    shuffled = list(neg)
    rng.shuffle(shuffled)
    expected = {prep.rel_path(s.image) for s in shuffled[6:3506] + neg[:400]}
    assert prep.original_build_images() == expected
    assert not {prep.rel_path(s.image) for s in shuffled[:6]} - {prep.rel_path(s.image) for s in neg[:400]} & expected


def test_bdd_other_records_split_by_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    recs = [
        {"key": f"k{i}", "image": f"train/shard_000/img{i}.jpg", "box": [0, 0, 10, 20], "kind": "other", "value": None}
        for i in range(300)
    ]
    recs.append(
        {"key": "s", "image": "train/shard_000/small.jpg", "box": [0, 0, 10, 10], "kind": "other", "value": None}
    )
    monkeypatch.setattr(prep, "read_labels", lambda split, fixes=True: recs)
    monkeypatch.setattr(prep, "dev_clip_ids", lambda: set())
    val = prep.bdd_other_records("train", 10**6, 0, val=True)
    train = prep.bdd_other_records("train", 10**6, 0, val=False)
    assert val and all(prep.in_val(Path(r["image"])) for r in val)
    assert not any(prep.in_val(Path(r["image"])) for r in train)
    assert len(val) + len(train) == 300  # the 10 px box is too small for either


def test_engine_layer_names_reads_the_inspector_json() -> None:
    class _Inspector:
        def get_engine_information(self, fmt: Any) -> str:
            return json.dumps({"Layers": ["images_cast_to_fp16", {"Name": "/model.0/conv"}], "Bindings": ["images"]})

    class _Engine:
        def create_engine_inspector(self) -> _Inspector:
            return _Inspector()

    pytest.importorskip("tensorrt")
    assert signs_mod.engine_layer_names(_Engine()) == ["images_cast_to_fp16", "/model.0/conv"]


def test_malformed_sidecar_means_stale(tmp_path: Path) -> None:
    w = tmp_path / "w.pt"
    w.write_bytes(b"w")
    e = engine_path(w, (96, 96), True)
    e.write_bytes(b"e")
    sidecar_path(e).write_text("[1, 2]")  # valid JSON, not an object
    assert engine_staleness(e, w) is not None
    sidecar_path(e).write_text("{not json")
    assert "unreadable" in (engine_staleness(e, w) or "")


def test_label_fixes_apply_only_when_asked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = next(iter(prep.LABEL_FIXES))
    rec = {"key": key, "image": "train/shard_000/x.jpg", "box": [0, 0, 10, 20], "kind": "other", "value": None}
    (tmp_path / "train.jsonl").write_text(json.dumps(rec) + "\n")
    monkeypatch.setattr(prep, "LABELS", tmp_path)
    (fixed,) = prep.read_labels("train")
    assert (fixed["kind"], fixed["value"]) == prep.LABEL_FIXES[key] and fixed["labeler_kind"] == "other"
    assert prep.read_labels("train", fixes=False)[0]["kind"] == "other"  # the first build saw the original label
    assert all(prep.in_val(Path(k.split("/")[1])) for k in prep.LABEL_FIXES)  # never a training frame


def test_value_views_must_all_agree() -> None:
    from perception.signs import VALUE_VIEWS, consensus, read_value, view_boxes

    boxes = view_boxes((100.0, 100.0, 130.0, 140.0), 5)
    assert boxes[0] == (100.0, 100.0, 130.0, 140.0) and len(boxes) == 5 and len(VALUE_VIEWS) == 7
    assert boxes[1] == pytest.approx((97.6, 100.0, 127.6, 140.0))  # shifted left by 8% of the width
    assert consensus([(35, 0.99), (35, 0.95), (35, 0.97)]) == (35, 0.95)  # the weakest view decides
    assert consensus([(35, 0.99), (30, 0.99)]) == (None, 0.99)  # disagreement: no value
    assert consensus([(35, 0.99), (None, 0.6)]) == (None, 0.6)  # one view sees no number: no value
    img = np.zeros((720, 1280, 3), np.uint8)
    seen: list[int] = []

    def classify(crops: list[np.ndarray]) -> list[tuple[int | None, float]]:
        seen.append(len(crops))
        return [(25, 0.99), (25, 0.98), (20, 0.99)][: len(crops)]

    assert read_value(classify, img, (100.0, 100.0, 130.0, 140.0), 96, 3) == (None, 0.98) and seen == [3]
    assert read_value(classify, img, (100.0, 100.0, 130.0, 140.0), 96, 2) == (25, 0.98)
    assert read_value(classify, img, (5.0, 5.0, 6.0, 6.0), 96, 3) is None  # degenerate crop: not read


def test_detector_reads_values_over_views(fake_detector: Any) -> None:
    _FakeYOLO.det_rows = [(600, 200, 630, 240, 0.9, KIND_INDEX["speed_limit"])]
    _FakeYOLO.probs = [0.0005, 0.0005, 0.999]
    det = fake_detector(value_views=5)
    calls = _FakeYOLO.cls_calls
    (d,) = det.detect(_frame())
    assert d.value == 35 and _FakeYOLO.cls_calls == calls + 1  # one classifier call ...
    assert _FakeYOLO.crop_counts[-1] == 5  # ... with the 5 views
    det3 = fake_detector(value_views=3)
    det3.detect(_frame())
    assert _FakeYOLO.crop_counts[-1] == 3


def test_detector_withholds_a_value_when_the_views_disagree(fake_detector: Any) -> None:
    _FakeYOLO.det_rows = [(600, 200, 630, 240, 0.9, KIND_INDEX["speed_limit"])]
    sure_35, sure_25, unsure_35 = [0.0, 0.0, 0.999], [0.0, 0.999, 0.0], [0.04, 0.0, 0.96]
    _FakeYOLO.view_probs = [sure_35, sure_35, sure_25, sure_35, sure_35]  # one view reads 25
    (d,) = fake_detector(value_views=5, value_conf=0.9).detect(_frame())
    assert d.value is None and d.value_conf == pytest.approx(0.999, abs=1e-3)
    _FakeYOLO.view_probs = [sure_35, sure_35, unsure_35, sure_35, sure_35]  # all 35, the weakest at 0.96
    (d,) = fake_detector(value_views=5, value_conf=0.97).detect(_frame())
    assert d.value is None and d.value_conf == pytest.approx(0.96, abs=1e-3)  # the weakest view decides
    (d,) = fake_detector(value_views=5, value_conf=0.95).detect(_frame())
    assert d.value == 35


def test_scaled_views_are_clipped_at_the_frame_edge() -> None:
    from perception.signs import read_value, view_boxes

    near_edge = (1250.0, 300.0, 1278.0, 340.0)
    boxes = view_boxes(near_edge, 5)
    assert max(b[2] for b in boxes) > 1280  # the shifted and scaled views reach past the frame...
    img = np.zeros((720, 1280, 3), np.uint8)
    got: list[int] = []

    def classify(crops: list[np.ndarray]) -> list[tuple[int | None, float]]:
        got.extend(c.shape[0] for c in crops)
        return [(25, 0.99)] * len(crops)

    assert read_value(classify, img, near_edge, 96, 5) == (25, 0.99) and got == [96] * 5  # ...and is clipped


def test_eval_values_on_other_boxes_are_their_own_category() -> None:
    other = ev.Gt((100, 100, 130, 140), ev.OTHER, None, None)  # a box the pseudo-labeler called "other"
    stop = ev.Gt((300, 100, 340, 140), "stop", None, None)
    raw = [_sl((101, 101, 131, 141), 25), _sl((300, 100, 340, 140), 35, conf=0.8), _sl((700, 100, 730, 140), 25)]
    s = ev.summarize([_res([other, stop], raw)], _cfg())
    assert s["reported_values"] == {"on_other": 1, "wrong": 1, "on_nothing": 1}
    assert s["values_contradicting_a_label"] == 1  # only the value on the labeled stop sign
    assert sorted(w["on"] for w in s["wrong_values"]) == ["on_nothing", "on_other", "wrong"]  # all strictly wrong
    assert "other" not in s["per_kind"] and s["per_kind"]["stop"]["fn"] == 1  # "other" boxes are never missed


def test_heldout_items_load_other_boxes_and_the_audit_drops(tmp_path: Path) -> None:
    data = {
        "images": [
            {
                "image": "valid/shard_000/a.jpg",
                "signs": [],
                "ignore": [],
                "negatives": [[0, 0, 10, 20], [30, 0, 40, 20]],
                "audit": {
                    "signs": [{"box": [30, 0, 40, 20], "kind": "speed_limit", "value": 20, "readable": True}],
                    "ignore": [],
                    "drop_negatives": [[30, 0, 40, 20]],
                },
            }
        ]
    }
    path = tmp_path / "h.json"
    path.write_text(json.dumps(data))
    (audited,) = ev.heldout_items(path)
    assert [(g.kind, g.box) for g in audited[2]] == [("speed_limit", (30, 0, 40, 20)), (ev.OTHER, (0, 0, 10, 20))]
    (plain,) = ev.heldout_items(path, audit=False)
    assert [g.kind for g in plain[2]] == [ev.OTHER, ev.OTHER]


def test_eval_values_heldout_uses_the_app_rules(tmp_path: Path) -> None:
    from scripts.signs import eval_values as evv

    signs = [
        {"box": [100, 100, 130, 140], "kind": "speed_limit", "value": 25, "readable": True},  # read
        {"box": [200, 100, 210, 112], "kind": "speed_limit", "value": 25, "readable": False},  # 12 px: too small
        {"box": [300, 0.5, 330, 40], "kind": "speed_limit", "value": 30, "readable": False},  # at the frame border
        {"box": [400, 100, 430, 140], "kind": "speed_limit", "value": None, "readable": False},  # no value
        {"box": [500, 100, 530, 140], "kind": "stop", "value": None, "readable": None},
    ]
    path = tmp_path / "h.json"
    path.write_text(json.dumps({"images": [{"image": "valid/shard_000/a.jpg", "signs": signs}]}))
    items, readable = evv.heldout_items(path, SignsConfig())
    assert [it[1] for it in items] == [(100, 100, 130, 140)] and readable == [True]


def test_training_images_read_every_recorded_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "used_for_training_det.txt").write_text("a.jpg\n")
    (tmp_path / "used_for_training_det_ft_build.txt").write_text("b.jpg\n")  # an earlier build, kept
    (tmp_path / "used_for_training_cls_cls5.txt").write_text("c.jpg\n")
    monkeypatch.setattr(prep, "DATA", tmp_path)
    monkeypatch.setattr(prep, "original_build_images", lambda: {"o.jpg"})
    assert prep.training_images() == {"a.jpg", "b.jpg", "c.jpg", "o.jpg"}


def test_hard_negative_curation_blocks() -> None:
    from scripts.signs import curate_det_negatives as cur

    assert cur.rest_excluded(200) and not cur.rest_excluded(214)  # one-way block: only checked non-signs kept
    assert cur.rest_excluded(92) and not cur.rest_excluded(84)  # NO TURN ON RED dropped, NO TURNS text sign kept
