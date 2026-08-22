"""Convenience entry point for the CarryWaterActive smoke experiment."""

from __future__ import annotations

from pathlib import Path
import sys

from run_obstacle_avoid import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        default = Path(__file__).resolve().parent / "configs" / "carrywater_active_smoke.yaml"
        sys.argv[1:1] = ["--config", str(default)]
    raise SystemExit(main())
