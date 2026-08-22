"""Convenience entry point for SemTraj2D; all options are handled by the shared runner."""

from __future__ import annotations

from pathlib import Path
import sys

from run_obstacle_avoid import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        default = Path(__file__).resolve().parent / "configs" / "semtraj2d_disk_upper_smoke.yaml"
        sys.argv[1:1] = ["--config", str(default)]
    raise SystemExit(main())
