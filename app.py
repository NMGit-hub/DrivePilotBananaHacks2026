"""DrivePilot entry point.

    python -m app --print-config              # show the resolved config (keys masked)
    python -m app --set video.source=camera   # dotted-key overrides, repeatable

The full pipeline (capture → perception → voice → answers → UI) is wired in milestone M7.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.config import ConfigError, load_config
from core.logging_setup import get_logger, setup_logging

log = get_logger("app")


def build_parser() -> argparse.ArgumentParser:
    """Command-line options shared by all run modes."""
    parser = argparse.ArgumentParser(prog="python -m app", description="DrivePilot dashcam voice co-pilot")
    parser.add_argument("--config", type=Path, default=None, help="config file (default: repo config.yaml)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set video.source=camera (repeatable)",
    )
    parser.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load the config and run the requested mode. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, overrides=args.overrides)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg.logging.level, secrets=cfg.secrets.values())

    if args.print_config:
        print(json.dumps(cfg.redacted_dump(), indent=2))
        return 0

    log.error("the full pipeline is not wired yet (milestone M7); use --print-config")
    return 1


if __name__ == "__main__":
    sys.exit(main())
