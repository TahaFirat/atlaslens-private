from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from atlaslens_api.gazetteer.management import (
    ForwardGazetteerManager,
    GazetteerInfo,
    GazetteerManager,
)


def default_cache_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "AtlasLens" / "gazetteer"
    return Path.home() / ".atlaslens" / "gazetteer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas gazetteer")
    parser.add_argument("--cache-root", type=Path, default=default_cache_root())
    parser.add_argument("--json", action="store_true", dest="json_output")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install")
    commands.add_parser("verify")
    commands.add_parser("info")
    remove = commands.add_parser("remove")
    remove.add_argument("--yes", action="store_true")
    commands.add_parser("install-forward")
    commands.add_parser("verify-forward")
    commands.add_parser("info-forward")
    remove_forward = commands.add_parser("remove-forward")
    remove_forward.add_argument("--yes", action="store_true")
    return parser


def _render(info: GazetteerInfo, *, json_output: bool) -> None:
    if json_output:
        print(info.model_dump_json())
        return
    print(f"status: {info.status}")
    print(f"dataset: {info.dataset}")
    print(f"places: {info.place_count}")
    if info.alias_count:
        print(f"aliases: {info.alias_count}")
    if info.schema_version:
        print(f"schema: {info.schema_version}")
    if info.license:
        print(f"license: {info.license}")
    if info.reason_code:
        print(f"reason: {info.reason_code}")


def main(
    argv: Sequence[str] | None = None,
    *,
    manager_factory: Callable[[Path], GazetteerManager] = GazetteerManager,
    forward_manager_factory: Callable[[Path], ForwardGazetteerManager] = (
        ForwardGazetteerManager
    ),
) -> int:
    args = build_parser().parse_args(argv)
    try:
        manager = manager_factory(args.cache_root)
        forward_manager = forward_manager_factory(args.cache_root)
        if args.command == "install":
            result = manager.install()
        elif args.command == "verify":
            result = manager.verify()
        elif args.command == "info":
            result = manager.info()
        elif args.command == "remove":
            result = manager.remove(confirmed=args.yes)
        elif args.command == "install-forward":
            result = forward_manager.install()
        elif args.command == "verify-forward":
            result = forward_manager.verify()
        elif args.command == "info-forward":
            result = forward_manager.info()
        elif args.command == "remove-forward":
            result = forward_manager.remove(confirmed=args.yes)
        else:  # pragma: no cover
            raise ValueError("unknown gazetteer command")
    except OSError:
        message = "gazetteer filesystem or network operation failed"
        if args.json_output:
            print(json.dumps({"status": "error", "message": message}))
        else:
            print(f"gazetteer error: {message}", file=sys.stderr)
        return 2
    except ValueError as exc:
        if args.json_output:
            print(json.dumps({"status": "error", "message": str(exc)}))
        else:
            print(f"gazetteer error: {exc}", file=sys.stderr)
        return 2
    _render(result, json_output=args.json_output)
    return 0 if result.status in {"ready", "not_installed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
