"""Convenience entry point for frozen CarryWaterActive post-hoc evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_semtraj2d import main


def _inject_default_private_dir() -> None:
    """Make the task-specific wrapper useful without weakening train-time isolation."""

    if any(
        argument == "--private-dir" or argument.startswith("--private-dir=")
        for argument in sys.argv[1:]
    ):
        return
    private_dir = (
        PROJECT_ROOT
        / "data"
        / "CarryWaterActive"
        / "private"
        / "carrywater_active"
    )
    sys.argv.extend(["--private-dir", str(private_dir)])


if __name__ == "__main__":
    _inject_default_private_dir()
    raise SystemExit(main())
