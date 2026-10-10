"""Config loading: defaults, overrides, validation errors, secrets from env/.env, redaction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import AppConfig, ConfigError, load_config, load_secrets

FAKE_KEY = "sk-ant-test-0123456789abcdef"
FAKE_TOKEN = "pk.test-token-abcdefghijklmnop"


def write(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_repo_config_loads_and_validates() -> None:
    cfg = load_config(env_file=None, environ={})
    assert cfg.video.source in ("file", "camera")
    assert cfg.vision_model == "claude"
    assert cfg.vision.claude.model == "claude-haiku-4-5"


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, ""), env_file=None, environ={})
    assert cfg == AppConfig(root=tmp_path)
    assert cfg.video.source == "file"
    assert cfg.video.camera.camera_id == 3 and cfg.video.camera.camera_zoom == pytest.approx(1.0)


def test_partial_section_keeps_other_defaults(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, "video:\n  camera:\n    camera_id: 3\n"), env_file=None, environ={})
    assert cfg.video.camera.camera_id == 3
    assert cfg.video.camera.fps == 30
    assert cfg.video.source == "file"


def test_list_overrides_parse_yaml_values(tmp_path: Path) -> None:
    cfg = load_config(
        write(tmp_path, "video:\n  source: file\n"),
        overrides=["video.source=camera", "video.camera.fps=24", "video.file.gpx_path=null", "ui.port=9001"],
        env_file=None,
        environ={},
    )
    assert cfg.video.source == "camera"
    assert cfg.video.camera.fps == 24
    assert cfg.video.file.gpx_path is None
    assert cfg.ui.port == 9001


def test_dict_overrides(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, ""), overrides={"vision_model": "mock"}, env_file=None, environ={})
    assert cfg.vision_model == "mock"


@pytest.mark.parametrize(
    ("yaml_text", "expected_fragment"),
    [
        ("video:\n  source: webcam\n", "video.source"),
        ("video:\n  camera:\n    fps: 0\n", "video.camera.fps"),
        ("video:\n  camera:\n    fps: 120\n", "video.camera.fps"),
        ("vision_model: gpt\n", "vision_model"),
        ("ui:\n  port: 70000\n", "ui.port"),
        ("logging:\n  level: LOUD\n", "logging.level"),
        ("video:\n  sourse: file\n", "video.sourse"),  # typo: unknown keys are rejected
        ("unknown_section: 1\n", "unknown_section"),
    ],
)
def test_invalid_values_raise_with_field_path(tmp_path: Path, yaml_text: str, expected_fragment: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, yaml_text), env_file=None, environ={})
    assert expected_fragment in str(excinfo.value)


def test_cross_field_validation(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="gpx_path"):
        load_config(write(tmp_path, "gps:\n  source: gpx\n"), env_file=None, environ={})


def test_missing_file_and_bad_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", env_file=None, environ={})
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write(tmp_path, "video: [unclosed\n"), env_file=None, environ={})
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path, "- a\n- b\n"), env_file=None, environ={})


@pytest.mark.parametrize("bad", ["video.source", "=camera", "video..=x"])
def test_malformed_overrides(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, ""), overrides=[bad], env_file=None, environ={})


def test_override_into_non_section_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a section"):
        load_config(
            write(tmp_path, "vision_model: claude\n"), overrides=["vision_model.x=1"], env_file=None, environ={}
        )


def test_secrets_cannot_come_from_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="secrets"):
        load_config(write(tmp_path, "secrets:\n  anthropic_api_key: x\n"), env_file=None, environ={})


def test_secrets_from_env_file(tmp_path: Path) -> None:
    env = write(tmp_path, f"ANTHROPIC_API_KEY={FAKE_KEY}\nMAPBOX_ACCESS_TOKEN=\n", name=".env")
    cfg = load_config(write(tmp_path, ""), env_file=env, environ={})
    assert cfg.secrets.anthropic_api_key is not None
    assert cfg.secrets.anthropic_api_key.get_secret_value() == FAKE_KEY
    assert cfg.secrets.mapbox_access_token is None  # empty value counts as missing
    assert cfg.secrets.configured() == {
        "anthropic_api_key": True,
        "mapbox_access_token": False,
        "gemini_api_key": False,
    }


def test_real_environment_wins_over_env_file(tmp_path: Path) -> None:
    env = write(tmp_path, "MAPBOX_ACCESS_TOKEN=from-file-123456\n", name=".env")
    secrets = load_secrets(env, environ={"MAPBOX_ACCESS_TOKEN": FAKE_TOKEN})
    assert secrets.mapbox_access_token is not None
    assert secrets.mapbox_access_token.get_secret_value() == FAKE_TOKEN


def test_secrets_never_appear_in_repr_or_dump(tmp_path: Path) -> None:
    cfg = load_config(
        write(tmp_path, ""), env_file=None, environ={"ANTHROPIC_API_KEY": FAKE_KEY, "MAPBOX_ACCESS_TOKEN": FAKE_TOKEN}
    )
    dumped = json.dumps(cfg.redacted_dump())
    for text in (repr(cfg), str(cfg), repr(cfg.secrets), dumped, json.dumps(cfg.model_dump(mode="json"))):
        assert FAKE_KEY not in text
        assert FAKE_TOKEN not in text
    assert cfg.redacted_dump()["secrets"]["anthropic_api_key"] == "set"
    assert cfg.redacted_dump()["secrets"]["gemini_api_key"] == "missing"
    assert set(cfg.secrets.values()) == {FAKE_KEY, FAKE_TOKEN}


def test_paths_resolve_relative_to_config_folder(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, "video:\n  file:\n    path: clips/a.mp4\n"), env_file=None, environ={})
    assert cfg.resolve_path(cfg.video.file.path) == (tmp_path / "clips" / "a.mp4").resolve()
    absolute = (tmp_path / "x.mp4").resolve()
    assert cfg.resolve_path(absolute) == absolute


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (["video.source=file", "video.file.gpx_path=a.gpx"], "gpx"),
        (["video.source=file"], "none"),
        (["video.source=camera"], "phone"),
        (["video.source=camera", "gps.source=none"], "none"),
        (["video.source=file", "gps.source=phone"], "phone"),
    ],
)
def test_gps_source_resolution(tmp_path: Path, overrides: list[str], expected: str) -> None:
    cfg = load_config(write(tmp_path, ""), overrides=overrides, env_file=None, environ={})
    assert cfg.gps_source == expected


def test_gpx_source_without_track_fails_in_camera_mode_too(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="gpx_path"):
        load_config(write(tmp_path, ""), overrides=["video.source=camera", "gps.source=gpx"], env_file=None, environ={})


def test_config_is_immutable() -> None:
    cfg = AppConfig()
    with pytest.raises(ValueError):
        cfg.video.source = "camera"  # type: ignore[misc]
    with pytest.raises(ValueError):
        cfg.gps.source = "gpx"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("yaml_text", "field"),
    [
        ("video:\n  camera:\n    fps: true\n", "video.camera.fps"),  # bool is not an int
        ("video:\n  file:\n    realtime: 'no'\n", "video.file.realtime"),  # string is not a bool
        ("video:\n  file:\n    realtime: no\n", "video.file.realtime"),  # YAML 1.2: bare no is a string
        ("video:\n  file:\n    start_s: 1:30\n", "video.file.start_s"),  # no base-60 numbers
        ("video:\n  camera:\n    camera_id: 1.0\n", "video.camera.camera_id"),  # float is not an int
        ("ui:\n  port: '8000'\n", "ui.port"),  # quoted number stays a string
    ],
)
def test_scalars_are_not_coerced(tmp_path: Path, yaml_text: str, field: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, yaml_text), env_file=None, environ={})
    assert field in str(excinfo.value)


def test_int_is_accepted_for_float_fields(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, "video:\n  file:\n    start_s: 5\n"), env_file=None, environ={})
    assert cfg.video.file.start_s == 5.0


def test_non_utf8_files_raise_config_error(tmp_path: Path) -> None:
    utf16 = tmp_path / "config.yaml"
    utf16.write_text("video:\n  source: file\n", encoding="utf-16")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(utf16, env_file=None, environ={})
    env16 = tmp_path / ".env"
    env16.write_text("ANTHROPIC_API_KEY=abc\n", encoding="utf-16")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(write(tmp_path, "", name="ok.yaml"), env_file=env16, environ={})


def test_directory_as_config_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path, env_file=None, environ={})
