"""Sign detector (V2-M6): taxonomy and label mapping, value parsing, dataset conversion, the detector's contract."""

from __future__ import annotations

import json
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
    SignDetection,
    SignDetector,
    SignGpuError,
    crop_square,
    decide_value,
    engine_path,
    parse_value_class,
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
    probs: ClassVar[list[float]] = []
    cls_names: ClassVar[dict[int, str]] = {0: "none", 1: "v25", 2: "v35"}
    det_names: ClassVar[dict[int, str]] = dict(enumerate(KINDS))
    cls_calls = 0

    def __init__(self, path: str, task: str) -> None:
        self.task = task
        self.names = self.det_names if task == "detect" else self.cls_names
        self.predictor = SimpleNamespace(model=None)

    def predict(self, image: np.ndarray, **kwargs: Any) -> list[Any]:
        if self.task == "detect":
            return [SimpleNamespace(boxes=_FakeBoxes(self.det_rows), speed={"inference": 1.0})]
        type(self).cls_calls += 1
        assert image.shape == (96, 96, 3)
        return [SimpleNamespace(probs=SimpleNamespace(data=torch.tensor(self.probs)))]


@pytest.fixture
def fake_detector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setattr(signs_mod, "YOLO", _FakeYOLO)
    monkeypatch.setattr(signs_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(signs_mod.torch.cuda, "memory_allocated", lambda *_: 0)
    monkeypatch.setattr(SignDetector, "_verify_gpu", lambda self, what, model: ("cuda:0", "fp16"))
    _FakeYOLO.cls_calls = 0
    _FakeYOLO.det_names = dict(enumerate(KINDS))

    def make(**overrides: Any) -> SignDetector:
        cfg = SignsConfig(engine="pytorch", **overrides)
        return SignDetector(cfg, tmp_path)

    return make


def _frame(w: int = 1280, h: int = 720) -> Frame:
    return Frame(np.zeros((h, w, 3), np.uint8), t=0.0, index=0, captured_at=0.0)


def test_detector_reads_a_confident_speed_value(fake_detector: Any) -> None:
    _FakeYOLO.det_rows = [(600, 200, 630, 240, 0.9, KIND_INDEX["speed_limit"]), (100, 100, 140, 140, 0.8, 2)]
    _FakeYOLO.probs = [0.005, 0.005, 0.99]
    dets = fake_detector().detect(_frame())
    assert [d.kind for d in dets] == ["speed_limit", "stop"]
    assert dets[0].value == 35 and dets[0].value_conf == pytest.approx(0.99)
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
    _FakeYOLO.probs = [0.0, 0.99, 0.01]
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
    assert drop_duplicates([c, b, a]) == [a, b]


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
    monkeypatch.setattr(signs_mod.torch.cuda, "memory_allocated", lambda *_: 0)
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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA GPU")
@pytest.mark.skipif(not _DET_ENGINE.exists(), reason="sign engines not built (scripts/signs/export_engines.py)")
def test_real_engines_run_on_cuda_fp16_and_find_a_sign() -> None:
    det = SignDetector(_CFG.signs, _CFG.root)
    for name, model in (("detector", det.det), ("classifier", det.cls)):
        device, precision = det._verify_gpu(name, model)  # raises if not on the GPU as configured
        assert device.startswith("cuda") and precision == "fp16"
    # A blank frame: no signs, and fast.
    assert det.detect(_frame()) == []
    # A drawn stop sign (red octagon with white STOP) on a road-gray frame is found.
    import cv2

    img = np.full((720, 1280, 3), (110, 115, 120), np.uint8)
    cx, cy, r = 900, 250, 40
    pts = np.array(
        [[cx + r * np.cos(np.pi / 8 + k * np.pi / 4), cy + r * np.sin(np.pi / 8 + k * np.pi / 4)] for k in range(8)],
        np.int32,
    )
    cv2.fillPoly(img, [pts], (40, 30, 200))
    cv2.polylines(img, [pts], True, (255, 255, 255), 3)
    cv2.putText(img, "STOP", (cx - 33, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.rectangle(img, (cx - 3, cy + r), (cx + 3, cy + 200), (90, 90, 90), -1)
    out = det.detect(Frame(img, 0.0, 0, 0.0))
    assert any(d.kind == "stop" for d in out), out
    times = []
    for _ in range(20):
        det.detect(_frame())
        times.append(det.last_timings_ms["total"])
    assert float(np.median(times)) < 30.0  # generous: the measured number is in scripts/signs/STATUS.md


# ---------------------------------------------------------------------------------- evaluation logic

from scripts.signs import evaluate as ev  # noqa: E402


def _cfg(**kw: Any) -> SignsConfig:
    return SignsConfig(**{"conf": 0.4, "value_conf": 0.9, "value_min_px": 16.0, **kw})


def _res(gts: list[ev.Gt], dets: list[ev.Det], ignore: list[tuple[float, float, float, float]] | None = None) -> Any:
    r = ev.ImageResult("img", gts, ignore or [])
    r.dets = dets
    return r


def test_eval_counts_right_wrong_and_withheld_values() -> None:
    sl = ev.Gt((100, 100, 130, 140), "speed_limit", 35, True)
    stop = ev.Gt((300, 100, 340, 140), "stop", None, None)
    dets = [
        ev.Det((101, 101, 131, 141), "speed_limit", 0.9, 35, 0.95),  # right value
        ev.Det((299, 99, 341, 141), "speed_limit", 0.8, 45, 0.99),  # speed limit on a stop sign: FP + wrong value
        ev.Det((600, 100, 630, 140), "speed_limit", 0.7, 25, 0.5),  # FP, value withheld (p < 0.9): not wrong
    ]
    s = ev.summarize([_res([sl, stop], dets)], _cfg())
    assert s["speed_values"] == {"speed_limit:right": 1}
    assert [w["said"] for w in s["wrong_values"]] == [45]
    assert s["per_kind"]["speed_limit"]["tp"] == 1 and s["per_kind"]["speed_limit"]["fp"] == 2
    assert s["per_kind"]["stop"]["fn"] == 1


def test_eval_ignore_regions_count_neither_way_but_values_are_listed() -> None:
    dets = [ev.Det((100, 100, 130, 140), "speed_limit", 0.9, 25, 0.99)]
    s = ev.summarize([_res([], dets, ignore=[(98, 98, 132, 142)])], _cfg())
    assert s["per_kind"] == {} and s["wrong_values"] == []
    assert [u["said"] for u in s["unverifiable_values"]] == [25]


def test_eval_small_or_unsure_values_are_not_reported() -> None:
    sl = ev.Gt((100, 100, 112, 112), "speed_limit", 35, True)  # 12 px tall
    dets = [ev.Det((100, 100, 112, 112), "speed_limit", 0.9, 30, 0.99)]
    s = ev.summarize([_res([sl], dets)], _cfg())
    assert s["speed_values"] == {"speed_limit:no_value": 1} and not s["wrong_values"]
    # at 1080p the same rule is measured at 1280 px width: 24 px tall at 1920 = 16 px -> value reported
    sl2 = ev.Gt((100, 100, 118, 124), "speed_limit", 35, True)
    r = _res([sl2], [ev.Det((100, 100, 118, 124), "speed_limit", 0.9, 35, 0.99)])
    r.scale = 1.5
    assert ev.summarize([r], _cfg())["speed_values"] == {"speed_limit:right": 1}


def test_eval_matching_is_by_kind_and_iou() -> None:
    g = ev.Gt((100, 100, 140, 140), "stop", None, None)
    wrong_kind = ev.Det((100, 100, 140, 140), "yield", 0.9, None, None)
    far = ev.Det((150, 150, 190, 190), "stop", 0.9, None, None)
    s = ev.summarize([_res([g], [wrong_kind, far])], _cfg())
    assert s["per_kind"]["stop"] == {**s["per_kind"]["stop"], "tp": 0, "fp": 1, "fn": 1}
    assert s["per_kind"]["yield"]["fp"] == 1


def test_eval_average_precision() -> None:
    gts = [ev.Gt((0, 0, 10, 10), "stop", None, None), ev.Gt((50, 0, 60, 10), "stop", None, None)]
    dets = [
        ev.Det((0, 0, 10, 10), "stop", 0.9, None, None),  # TP
        ev.Det((100, 0, 110, 10), "stop", 0.8, None, None),  # FP
        ev.Det((50, 0, 60, 10), "stop", 0.7, None, None),  # TP
    ]
    ap, n = ev.average_precision([_res(gts, dets)], "stop")
    assert n == 2 and ap == pytest.approx(0.5 + 0.5 * (2 / 3))
    assert np.isnan(ev.average_precision([_res(gts, dets)], "yield")[0])
