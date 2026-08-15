"""Evaluation-only metrics and plots; never feed these values to the LLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from .data import FeatureLibrary
from .learner import ConstraintEnsemble
from .oracle import CircularEvaluationOracle
from .types import HypothesisEvidence, QueryRecord, Trajectory, VIOLATION_LABEL


@dataclass(frozen=True)
class BoundaryMetrics:
    iou: float
    false_safe_rate: float
    false_unsafe_rate: float
    accuracy: float
    predicted_unsafe_fraction: float
    heldout_expert_safe_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_boundary(
    ensemble: ConstraintEnsemble,
    library: FeatureLibrary,
    oracle: CircularEvaluationOracle,
    heldout_experts: list[Trajectory],
    workspace_x: tuple[float, float],
    workspace_y: tuple[float, float],
    resolution: int,
    device: torch.device,
) -> tuple[BoundaryMetrics, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    xs = np.linspace(*workspace_x, resolution)
    ys = np.linspace(*workspace_y, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
    features = library.grid_features(points, ensemble.compiled.variables)
    with torch.no_grad():
        scores = ensemble.mean_state_score(
            torch.as_tensor(features, dtype=torch.float32, device=device)
        ).cpu().numpy()
    predicted = scores > 0.0
    truth = oracle.state_violation_mask(points)
    intersection = int(np.sum(predicted & truth))
    union = int(np.sum(predicted | truth))
    false_safe = int(np.sum((~predicted) & truth))
    false_unsafe = int(np.sum(predicted & (~truth)))
    expert_safe: list[bool] = []
    for expert in heldout_experts:
        values = library.torch_features(
            torch.as_tensor(expert.states, dtype=torch.float32, device=device),
            ensemble.compiled.variables,
        )
        expert_safe.append(ensemble.predict_features(values) == 0)
    return (
        BoundaryMetrics(
            iou=intersection / max(union, 1),
            false_safe_rate=false_safe / max(int(np.sum(truth)), 1),
            false_unsafe_rate=false_unsafe / max(int(np.sum(~truth)), 1),
            accuracy=float(np.mean(predicted == truth)),
            predicted_unsafe_fraction=float(np.mean(predicted)),
            heldout_expert_safe_rate=float(np.mean(expert_safe)),
        ),
        (grid_x, grid_y, scores.reshape(grid_x.shape)),
    )


def plot_boundary(
    path: str | Path,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    experts: list[Trajectory],
    queries: list[QueryRecord],
    oracle: CircularEvaluationOracle,
    title: str,
) -> None:
    grid_x, grid_y, scores = grid
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.contourf(grid_x, grid_y, scores, levels=30, cmap="coolwarm", alpha=0.45)
    if float(scores.min()) <= 0.0 <= float(scores.max()):
        axis.contour(grid_x, grid_y, scores, levels=[0.0], colors="black", linewidths=2.0)
    for index, expert in enumerate(experts):
        axis.plot(
            expert.states[:, 0],
            expert.states[:, 1],
            color="tab:green",
            alpha=0.45,
            label="held-out expert" if index == 0 else None,
        )
    safe_labeled = False
    violation_labeled = False
    for query in queries:
        is_violation = query.label == VIOLATION_LABEL
        label = None
        if is_violation and not violation_labeled:
            label, violation_labeled = "violation query", True
        if not is_violation and not safe_labeled:
            label, safe_labeled = "safe query", True
        axis.plot(
            query.trajectory.states[:, 0],
            query.trajectory.states[:, 1],
            color="tab:red" if is_violation else "tab:blue",
            alpha=0.22,
            label=label,
        )
    center, radius = oracle.evaluation_geometry()
    axis.add_patch(
        Circle(center, radius, fill=False, linestyle="--", color="magenta", linewidth=2.0, label="evaluation truth")
    )
    axis.set(xlabel="x position [m]", ylabel="y position [m]", title=title)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_semantic_trace(
    path: str | Path,
    evidence_history: list[dict[str, object]],
    bank_audit: list[dict[str, object]],
) -> None:
    figure, (score_axis, bank_axis) = plt.subplots(1, 2, figsize=(13, 4.8))
    ids = sorted(
        {
            str(item["hypothesis_id"])
            for report in evidence_history
            for item in report.get("hypotheses", [])
        }
    )
    rounds = [int(report["outer_round"]) for report in evidence_history]
    for hypothesis_id in ids:
        values: list[float] = []
        for report in evidence_history:
            by_id = {str(item["hypothesis_id"]): item for item in report.get("hypotheses", [])}
            values.append(float(by_id[hypothesis_id]["selection_score"]) if hypothesis_id in by_id else np.nan)
        score_axis.plot(rounds, values, marker="o", label=hypothesis_id)
    score_axis.set(xlabel="outer semantic round", ylabel="leakage-safe selection score", title="Hypothesis evidence")
    score_axis.grid(alpha=0.25)
    score_axis.legend(fontsize=7)

    event_y = {"add": 1.0, "retain_and_query": 0.5, "propose_intervention": 0.0, "retire": -0.5}
    for index, event in enumerate(bank_audit):
        event_name = str(event.get("event", "other"))
        y = event_y.get(event_name, -1.0)
        outer_round = int(event.get("outer_round", 0))
        bank_axis.scatter(outer_round, y, s=45)
        bank_axis.annotate(
            str(event.get("hypothesis_id", "")),
            (outer_round, y),
            xytext=(3, 3 + 8 * (index % 2)),
            textcoords="offset points",
            fontsize=6,
            rotation=20,
        )
    bank_axis.set_yticks(list(event_y.values()))
    bank_axis.set_yticklabels(list(event_y))
    bank_axis.set(xlabel="outer semantic round", title="LLM/semantic actions")
    bank_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
