from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atlas diagnostics")
    parser.add_argument("target", choices=("phase6b",))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[5]
    script = project_root / "scripts" / "verify_phase6b_models.py"
    command = [sys.executable, str(script), "--project-root", str(project_root)]
    if args.json:
        command.append("--json")
    # Fixed interpreter plus a repository-owned script; no argument becomes an executable.
    return subprocess.run(command, check=False).returncode  # noqa: S603
