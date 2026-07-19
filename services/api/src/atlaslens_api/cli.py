from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence

Command = Callable[[Sequence[str] | None], int]
_GROUPS = (
    "embeddings",
    "retrieval",
    "dataset",
    "models",
    "gazetteer",
    "benchmark",
    "acceptance",
    "diagnostics",
    "nvidia-vision",
    "private-demo",
)


def _print_help() -> None:
    print("usage: atlas <group> [options]")
    print("")
    print("groups:")
    for group in _GROUPS:
        print(f"  {group}")


def _command(group: str) -> tuple[Command, bool]:
    if group == "embeddings":
        from atlaslens_api.retrieval.cli import main

        return main, True
    if group == "retrieval":
        from atlaslens_api.retrieval.cli import main

        return main, True
    if group == "dataset":
        from atlaslens_api.dataset_acquisition.cli import main

        return main, False
    if group == "models":
        from atlaslens_api.model_management.cli import main

        return main, False
    if group == "gazetteer":
        from atlaslens_api.gazetteer.cli import main

        return main, False
    if group == "benchmark":
        from atlaslens_api.evaluation.cli import main

        return main, False
    if group == "acceptance":
        from atlaslens_api.acceptance.cli import main

        return main, False
    if group == "diagnostics":
        from atlaslens_api.phase6b.diagnostics import main

        return main, False
    if group == "nvidia-vision":
        from atlaslens_api.nvidia_vision.cli import main

        return main, False
    if group == "private-demo":
        from atlaslens_api.demo_case_cli import main

        return main, False
    raise KeyError(group)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if not arguments:
        print(
            json.dumps(
                {
                    "status": "error",
                    "code": "command_required",
                    "groups": list(_GROUPS),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        command, preserve_group = _command(arguments[0])
    except (ImportError, KeyError):
        print(
            json.dumps({"status": "error", "code": "unknown_command"}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    return command(arguments if preserve_group else arguments[1:])


if __name__ == "__main__":
    raise SystemExit(main())
