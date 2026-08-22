"""Run compact closed-loop versus frozen-semantic ablations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "obstacle_avoid_smoke.yaml"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "ablation"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    variants = {
        "semantic_closed_loop": [],
        "frozen_one_shot_bank": ["--freeze-revisions"],
    }
    summary: dict[str, object] = {}
    for name, flags in variants.items():
        output = output_root / name
        command = [
            sys.executable,
            str(ROOT / "run_obstacle_avoid.py"),
            "--config",
            str(Path(args.config).resolve()),
            "--backend",
            "evidence_policy",
            "--output",
            str(output),
            "--overwrite",
            *flags,
        ]
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        summary[name] = json.loads((output / "result.json").read_text(encoding="utf-8"))
    (output_root / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    names = list(summary)
    ious = [float(summary[name]["final_metrics"]["iou"]) for name in names]
    queries = [int(summary[name]["oracle_queries"]) for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(names, ious, color=("tab:blue", "tab:gray"))
    axes[0].set(ylabel="evaluation-only boundary IoU", ylim=(0.0, 1.0))
    axes[1].bar(names, queries, color=("tab:blue", "tab:gray"))
    axes[1].set(ylabel="trajectory Oracle queries")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_root / "ablation.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
