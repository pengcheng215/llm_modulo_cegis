"""Evaluation-only metrics and plots; never feed these values to the LLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from .data import FeatureLibrary
from .learner import ConstraintEnsemble
from .oracle import CircularEvaluationOracle, TrajectoryEvaluationOracle
from .types import HypothesisEvidence, QueryRecord, SAFE_LABEL, Trajectory, VIOLATION_LABEL


@dataclass(frozen=True)
class BoundaryMetrics:
    iou: float
    false_safe_rate: float
    false_unsafe_rate: float
    accuracy: float
    predicted_unsafe_fraction: float
    heldout_expert_safe_rate: float
    trajectory_accuracy: float = float("nan")
    trajectory_balanced_accuracy: float = float("nan")
    trajectory_safe_accuracy: float = float("nan")
    trajectory_violation_recall: float = float("nan")
    trajectory_auroc: float = float("nan")
    trajectory_auprc: float = float("nan")
    trajectory_worst_group_balanced_accuracy: float = float("nan")
    trajectory_worst_pair_target_balanced_accuracy: float = float("nan")
    trajectory_exact_pair_accuracy: float = float("nan")
    trajectory_pair_ranking_accuracy: float = float("nan")
    trajectory_minimum_clause_recall: float = float("nan")
    evaluation_trajectory_count: int = 0
    trajectory_group_balanced_accuracy: dict[str, float] = field(default_factory=dict)
    trajectory_group_safe_accuracy: dict[str, float] = field(default_factory=dict)
    trajectory_group_violation_recall: dict[str, float] = field(default_factory=dict)
    trajectory_pair_target_balanced_accuracy: dict[str, float] = field(default_factory=dict)
    trajectory_pair_target_safe_accuracy: dict[str, float] = field(default_factory=dict)
    trajectory_pair_target_violation_recall: dict[str, float] = field(default_factory=dict)
    trajectory_clause_recall: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_boundary(
    ensemble: ConstraintEnsemble,
    library: FeatureLibrary,
    oracle: TrajectoryEvaluationOracle,
    heldout_experts: list[Trajectory],
    workspace_x: tuple[float, float],
    workspace_y: tuple[float, float],
    resolution: int,
    device: torch.device,
) -> tuple[BoundaryMetrics, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    raw_state_dimension = int(getattr(library, "raw_state_dimension", 2))
    is_planar = bool(getattr(library, "is_planar", raw_state_dimension == 2))
    if is_planar:
        xs = np.linspace(*workspace_x, resolution)
        ys = np.linspace(*workspace_y, resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
        features = library.grid_features(points, ensemble.compiled.variables)
        with torch.no_grad():
            scores = ensemble.mean_state_score(
                torch.as_tensor(features, dtype=torch.float32, device=device)
            ).cpu().numpy()
        decision_threshold = float(ensemble.decision_threshold.item())
        decision_values = scores - decision_threshold
        predicted = decision_values > 0.0
        supports_state_grid = bool(getattr(oracle, "supports_state_grid", True))
        if supports_state_grid:
            truth = oracle.state_violation_mask(points)
            intersection = int(np.sum(predicted & truth))
            union = int(np.sum(predicted | truth))
            false_safe = int(np.sum((~predicted) & truth))
            false_unsafe = int(np.sum(predicted & (~truth)))
            iou = intersection / max(union, 1)
            false_safe_rate = false_safe / max(int(np.sum(truth)), 1)
            false_unsafe_rate = false_unsafe / max(int(np.sum(~truth)), 1)
            accuracy = float(np.mean(predicted == truth))
            predicted_unsafe_fraction = float(np.mean(predicted))
        else:
            iou = float("nan")
            false_safe_rate = float("nan")
            false_unsafe_rate = float("nan")
            accuracy = float("nan")
            predicted_unsafe_fraction = float("nan")
    else:
        # A 12-D observation model has no faithful x/y state grid.  In
        # particular, filling the remaining channels with arbitrary constants
        # would turn a trajectory constraint into a misleading planar slice.
        # Return an explicit empty plotting payload while retaining all private
        # trajectory-bank metrics below.
        grid_x = np.empty((0, 0), dtype=np.float32)
        grid_y = np.empty((0, 0), dtype=np.float32)
        decision_values = np.empty((0, 0), dtype=np.float32)
        iou = float("nan")
        false_safe_rate = float("nan")
        false_unsafe_rate = float("nan")
        accuracy = float("nan")
        predicted_unsafe_fraction = float("nan")
    expert_safe: list[bool] = []
    for expert in heldout_experts:
        values = library.torch_features(
            torch.as_tensor(expert.states, dtype=torch.float32, device=device),
            ensemble.compiled.variables,
        )
        expert_safe.append(ensemble.predict_features(values) == 0)
    trajectory_metrics = _evaluate_private_trajectories(ensemble, library, oracle, device)
    return (
        BoundaryMetrics(
            iou=iou,
            false_safe_rate=false_safe_rate,
            false_unsafe_rate=false_unsafe_rate,
            accuracy=accuracy,
            predicted_unsafe_fraction=predicted_unsafe_fraction,
            heldout_expert_safe_rate=float(np.mean(expert_safe)),
            **trajectory_metrics,
        ),
        (grid_x, grid_y, decision_values.reshape(grid_x.shape)),
    )


def _evaluate_private_trajectories(
    ensemble: ConstraintEnsemble,
    library: FeatureLibrary,
    oracle: TrajectoryEvaluationOracle,
    device: torch.device,
) -> dict[str, object]:
    bank = oracle.evaluation_trajectories()
    if bank is None:
        return {}
    observations, labels, groups, _ = bank
    predictions: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for states in observations:
            features = library.torch_features(
                torch.as_tensor(states, dtype=torch.float32, device=device),
                ensemble.compiled.variables,
            )
            score = float(ensemble.mean_hard_trajectory_score(features).cpu().item())
            scores.append(score)
            predictions.append(int(score > float(ensemble.decision_threshold.item())))
    label_array = np.asarray(labels, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    safe = label_array == 0
    violation = label_array == 1
    safe_accuracy = float(np.mean(prediction_array[safe] == 0)) if np.any(safe) else float("nan")
    violation_recall = float(np.mean(prediction_array[violation] == 1)) if np.any(violation) else float("nan")
    balanced = _balanced_accuracy(label_array, prediction_array)
    group_values: dict[str, float] = {}
    group_safe_accuracy: dict[str, float] = {}
    group_violation_recall: dict[str, float] = {}
    for group in sorted(set(map(str, groups.tolist()))):
        selected = np.asarray(groups).astype(str) == group
        group_values[group] = _balanced_accuracy(label_array[selected], prediction_array[selected])
        group_safe = selected & safe
        group_violation = selected & violation
        group_safe_accuracy[group] = (
            float(np.mean(prediction_array[group_safe] == 0))
            if np.any(group_safe)
            else float("nan")
        )
        group_violation_recall[group] = (
            float(np.mean(prediction_array[group_violation] == 1))
            if np.any(group_violation)
            else float("nan")
        )
    finite_groups = [value for value in group_values.values() if np.isfinite(value)]
    pair_target_balanced: dict[str, float] = {}
    pair_target_safe: dict[str, float] = {}
    pair_target_violation: dict[str, float] = {}
    exact_pair_accuracy = float("nan")
    pair_ranking_accuracy = float("nan")
    clause_recall: dict[str, float] = {}
    metadata_loader = getattr(oracle, "evaluation_metadata", None)
    metadata = metadata_loader() if callable(metadata_loader) else None
    if metadata and {"pair_ids", "pair_roles", "pair_targets"}.issubset(metadata):
        pair_ids = np.asarray(metadata["pair_ids"]).astype(str)
        pair_roles = np.asarray(metadata["pair_roles"]).astype(str)
        pair_targets = np.asarray(metadata["pair_targets"]).astype(str)
        for target in sorted(set(pair_targets.tolist())):
            selected = pair_targets == target
            pair_target_balanced[target] = _balanced_accuracy(
                label_array[selected], prediction_array[selected]
            )
            target_safe = selected & safe
            target_violation = selected & violation
            pair_target_safe[target] = float(
                np.mean(prediction_array[target_safe] == 0)
            )
            pair_target_violation[target] = float(
                np.mean(prediction_array[target_violation] == 1)
            )
        exact: list[bool] = []
        ranked: list[bool] = []
        for pair_id in sorted(set(pair_ids.tolist())):
            selected = np.flatnonzero(pair_ids == pair_id)
            if len(selected) != 2:
                raise ValueError(f"private pair {pair_id!r} does not contain two rows")
            safe_rows = selected[pair_roles[selected] == "safe"]
            violation_rows = selected[pair_roles[selected] == "violation"]
            if len(safe_rows) != 1 or len(violation_rows) != 1:
                raise ValueError(f"private pair {pair_id!r} lacks one safe and one violation row")
            safe_index = int(safe_rows[0])
            violation_index = int(violation_rows[0])
            exact.append(
                prediction_array[safe_index] == SAFE_LABEL
                and prediction_array[violation_index] == VIOLATION_LABEL
            )
            ranked.append(score_array[violation_index] > score_array[safe_index])
        exact_pair_accuracy = float(np.mean(exact))
        pair_ranking_accuracy = float(np.mean(ranked))
    if metadata and {"clause_ids", "clause_labels"}.issubset(metadata):
        clause_ids = np.asarray(metadata["clause_ids"]).astype(str)
        clause_labels = np.asarray(metadata["clause_labels"], dtype=np.int64)
        for index, clause_id in enumerate(clause_ids.tolist()):
            selected = clause_labels[:, index] == VIOLATION_LABEL
            clause_recall[str(clause_id)] = (
                float(np.mean(prediction_array[selected] == VIOLATION_LABEL))
                if np.any(selected)
                else float("nan")
            )
    finite_pair_targets = [
        value for value in pair_target_balanced.values() if np.isfinite(value)
    ]
    finite_clause_recalls = [value for value in clause_recall.values() if np.isfinite(value)]
    return {
        "trajectory_accuracy": float(np.mean(label_array == prediction_array)),
        "trajectory_balanced_accuracy": balanced,
        "trajectory_safe_accuracy": safe_accuracy,
        "trajectory_violation_recall": violation_recall,
        "trajectory_auroc": _binary_auroc(label_array, score_array),
        "trajectory_auprc": _binary_average_precision(label_array, score_array),
        "trajectory_worst_group_balanced_accuracy": min(finite_groups) if finite_groups else float("nan"),
        "trajectory_worst_pair_target_balanced_accuracy": (
            min(finite_pair_targets) if finite_pair_targets else float("nan")
        ),
        "trajectory_exact_pair_accuracy": exact_pair_accuracy,
        "trajectory_pair_ranking_accuracy": pair_ranking_accuracy,
        "trajectory_minimum_clause_recall": (
            min(finite_clause_recalls) if finite_clause_recalls else float("nan")
        ),
        "evaluation_trajectory_count": int(len(label_array)),
        "trajectory_group_balanced_accuracy": group_values,
        "trajectory_group_safe_accuracy": group_safe_accuracy,
        "trajectory_group_violation_recall": group_violation_recall,
        "trajectory_pair_target_balanced_accuracy": pair_target_balanced,
        "trajectory_pair_target_safe_accuracy": pair_target_safe,
        "trajectory_pair_target_violation_recall": pair_target_violation,
        "trajectory_clause_recall": clause_recall,
    }


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    recalls: list[float] = []
    for label in (0, 1):
        selected = labels == label
        if np.any(selected):
            recalls.append(float(np.mean(predictions[selected] == label)))
    return float(np.mean(recalls)) if len(recalls) == 2 else float("nan")


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(negative))
    if not n_positive or not n_negative:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def _binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_positive = int(np.sum(labels == 1))
    if not n_positive:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order] == 1
    ordered_scores = scores[order]
    true_positives = np.cumsum(ordered_labels)
    false_positives = np.cumsum(~ordered_labels)
    # Precision-recall operating points exist only after a complete tied-score
    # group has entered the predicted-positive set.  Evaluating examples inside
    # a tie one by one makes AP depend on the archive's arbitrary row order.
    distinct_ends = np.flatnonzero(
        np.r_[ordered_scores[:-1] != ordered_scores[1:], True]
    )
    tp = true_positives[distinct_ends].astype(np.float64)
    fp = false_positives[distinct_ends].astype(np.float64)
    recall = tp / n_positive
    precision = tp / (tp + fp)
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def plot_boundary(
    path: str | Path,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    experts: list[Trajectory],
    queries: list[QueryRecord],
    oracle: TrajectoryEvaluationOracle,
    title: str,
) -> None:
    grid_x, grid_y, scores = grid
    figure, axis = plt.subplots(figsize=(9, 6))
    if not grid_x.size or not grid_y.size or not scores.size:
        axis.text(
            0.5,
            0.5,
            "No faithful planar state-grid view\n(trajectory-only evaluation)",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
        )
        axis.set_title(title)
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return
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
    geometry = oracle.evaluation_geometry()
    if isinstance(geometry, tuple):
        center, radius = geometry
        axis.add_patch(
            Circle(center, radius, fill=False, linestyle="--", color="magenta", linewidth=2.0, label="evaluation truth")
        )
    elif isinstance(geometry, dict):
        truth_labeled = False
        xs = np.asarray([float(grid_x.min()), float(grid_x.max())])
        for clause in geometry.get("clauses", []):
            kind = str(clause.get("kind"))
            label = "evaluation truth" if not truth_labeled else None
            if kind == "circle_exclusion":
                axis.add_patch(
                    Circle(
                        clause["center"],
                        float(clause["radius"]),
                        fill=False,
                        linestyle="--",
                        color="magenta",
                        linewidth=2.0,
                        label=label,
                    )
                )
                truth_labeled = True
            elif kind == "equality_band" and clause.get("variable") == "y_position":
                center = float(clause["center"])
                width = float(clause["half_width"])
                axis.axhline(center - width, linestyle="--", color="magenta", linewidth=1.5, label=label)
                axis.axhline(center + width, linestyle="--", color="magenta", linewidth=1.5)
                truth_labeled = True
            elif kind == "linear_halfspace":
                normal = np.asarray(clause["normal"], dtype=float)
                if abs(normal[1]) > 1.0e-9:
                    ys = (float(clause["offset"]) - normal[0] * xs) / normal[1]
                    axis.plot(xs, ys, linestyle="--", color="magenta", linewidth=1.5, label=label)
                    truth_labeled = True
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
