"""Frame semantics and the app entry point."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import app
from core.types import Frame

FAKE_KEY = "sk-ant-should-never-print-123456"


def make_frame(**kw: object) -> Frame:
    defaults: dict[str, object] = {"image": np.zeros((4, 6, 3), np.uint8), "t": 1.0, "index": 3, "captured_at": 0.0}
    defaults.update(kw)
    return Frame(**defaults)  # type: ignore[arg-type]


def test_frame_equality_is_identity_and_hashable() -> None:
    a, b = make_frame(), make_frame()
    assert a == a and a != b
    assert a in [b, a]
    assert len({a, b}) == 2


def test_frame_image_is_read_only() -> None:
    frame = make_frame()
    with pytest.raises(ValueError):
        frame.image[0, 0, 0] = 255
    copy = frame.image.copy()
    copy[0, 0, 0] = 255  # copies are writable
    assert frame.width == 6 and frame.height == 4 and frame.segment == 0


def test_print_config_masks_keys(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    assert app.main(["--print-config"]) == 0
    out = capsys.readouterr()
    assert FAKE_KEY not in out.out and FAKE_KEY not in out.err
    data = json.loads(out.out)
    assert data["secrets"]["anthropic_api_key"] == "set"


def test_print_config_applies_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    assert app.main(["--print-config", "--set", "video.source=camera", "--set", "ui.port=9100"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["video"]["source"] == "camera" and data["ui"]["port"] == 9100


def test_config_error_exits_with_2(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert app.main(["--set", "video.source=webcam"]) == 2
    assert "video.source" in capsys.readouterr().err
    assert app.main(["--config", str(tmp_path / "missing.yaml")]) == 2
