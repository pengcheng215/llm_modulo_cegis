"""Hypothesis-conditioned neural constraint models and MIL training."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import FeatureLibrary
from .hypotheses import CompiledClause, CompiledHypothesis
from .types import QueryRecord, SAFE_LABEL, Trajectory, VIOLATION_LABEL


def normalized_smooth_max(values: torch.Tensor, beta: float, dim: int) -> torch.Tensor:
    if beta <= 0.0:
        raise ValueError("smooth-max beta must be positive")
    count = values.shape[dim]
    return (torch.logsumexp(beta * values, dim=dim) - math.log(count)) / beta


def _mlp(input_dim: int, hidden_dims: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, int(width)), nn.Tanh()))
        previous = int(width)
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def _constraint_network(input_dim: int, hidden_dims: Sequence[int], model_family: str) -> nn.Module:
    if model_family == "linear":
        return nn.Linear(input_dim, 1)
    if model_family == "mlp":
        return _mlp(input_dim, hidden_dims)
    raise ValueError(f"unsupported compiled model family: {model_family}")


class ConstraintClauseHead(nn.Module):
    """Numeric head for one typed atomic clause."""

    def __init__(self, compiled: CompiledClause, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.compiled = compiled
        input_dim = len(compiled.clause.variables)
        low = torch.as_tensor(compiled.input_low, dtype=torch.float32)
        high = torch.as_tensor(compiled.input_high, dtype=torch.float32)
        if torch.any(high <= low):
            raise ValueError("invalid feature bounds")
        self.register_buffer("input_low", low)
        self.register_buffer("input_high", high)
        self.threshold: nn.Parameter | None = None
        self.center: nn.Parameter | None = None
        self.log_half_width: nn.Parameter | None = None
        self.log_scale: nn.Parameter | None = None
        relation = compiled.clause.relation
        if relation in {"upper_bound", "lower_bound"}:
            self.threshold = nn.Parameter(torch.zeros(1))
            self.log_scale = nn.Parameter(torch.zeros(1))
            self.joint_network = None
            self.independent_networks = nn.ModuleList()
        elif relation == "equality_band":
            self.center = nn.Parameter(torch.zeros(1))
            self.log_half_width = nn.Parameter(torch.tensor([-0.75]))
            self.log_scale = nn.Parameter(torch.zeros(1))
            self.joint_network = None
            self.independent_networks = nn.ModuleList()
        elif compiled.clause.coupling == "joint":
            self.joint_network: nn.Module | None = _constraint_network(
                input_dim, hidden_dims, compiled.clause.model_family
            )
            self.independent_networks = nn.ModuleList()
        else:
            self.joint_network = None
            self.independent_networks = nn.ModuleList(
                [_constraint_network(1, hidden_dims, compiled.clause.model_family) for _ in range(input_dim)]
            )

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        return 2.0 * (features - self.input_low) / (self.input_high - self.input_low) - 1.0

    def state_score(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize(features)
        relation = self.compiled.clause.relation
        if self.log_scale is not None:
            scalar = normalized[..., 0]
            scale = F.softplus(self.log_scale) + 1.0e-4
            if relation == "upper_bound":
                assert self.threshold is not None
                return scale * (scalar - self.threshold)
            if relation == "lower_bound":
                assert self.threshold is not None
                return scale * (self.threshold - scalar)
            assert self.center is not None and self.log_half_width is not None
            half_width = F.softplus(self.log_half_width) + 1.0e-3
            return scale * (torch.abs(scalar - self.center) - half_width)
        if self.joint_network is not None:
            return self.joint_network(normalized).squeeze(-1)
        component_scores = [
            network(normalized[..., index : index + 1]).squeeze(-1)
            for index, network in enumerate(self.independent_networks)
        ]
        return torch.max(torch.stack(component_scores, dim=-1), dim=-1).values

    def trajectory_score(self, features: torch.Tensor, beta: float, *, hard: bool = False) -> torch.Tensor:
        scores = self.state_score(features)
        operator = self.compiled.clause.temporal_operator
        if operator == "max":
            return torch.max(scores, dim=-1).values if hard else normalized_smooth_max(scores, beta=beta, dim=-1)
        if operator == "mean":
            return torch.mean(scores, dim=-1)
        if operator == "last":
            return scores[..., -1]
        raise RuntimeError(f"uncompiled temporal operator: {operator}")


class HypothesisConstraintNet(nn.Module):
    """State logit for one atomic or OR-of-violations composite hypothesis."""

    def __init__(self, compiled: CompiledHypothesis, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.compiled = compiled
        low = torch.as_tensor(compiled.input_low, dtype=torch.float32)
        high = torch.as_tensor(compiled.input_high, dtype=torch.float32)
        if torch.any(high <= low):
            raise ValueError("invalid feature bounds")
        self.register_buffer("input_low", low)
        self.register_buffer("input_high", high)
        self.clause_heads = nn.ModuleList(
            [ConstraintClauseHead(clause, hidden_dims) for clause in compiled.clauses]
        )

    @property
    def joint_network(self) -> nn.Module | None:
        """Backward-compatible view for tests and atomic model inspection."""
        return self.clause_heads[0].joint_network if len(self.clause_heads) == 1 else None

    @property
    def independent_networks(self) -> nn.ModuleList:
        return self.clause_heads[0].independent_networks

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        return 2.0 * (features - self.input_low) / (self.input_high - self.input_low) - 1.0

    def _clause_features(self, features: torch.Tensor, clause: CompiledClause) -> torch.Tensor:
        return features[..., list(clause.variable_indices)]

    def state_score(self, features: torch.Tensor) -> torch.Tensor:
        clause_scores = [
            head.state_score(self._clause_features(features, head.compiled))
            for head in self.clause_heads
        ]
        return torch.max(torch.stack(clause_scores, dim=-1), dim=-1).values

    def clause_state_scores(self, features: torch.Tensor) -> torch.Tensor:
        """Per-state score for each atomic clause, before composite max."""

        return torch.stack(
            [
                head.state_score(self._clause_features(features, head.compiled))
                for head in self.clause_heads
            ],
            dim=-1,
        )

    def trajectory_score(self, feature_trajectories: torch.Tensor, beta: float) -> torch.Tensor:
        if feature_trajectories.ndim == 2:
            feature_trajectories = feature_trajectories.unsqueeze(0)
        clause_scores = [
            head.trajectory_score(self._clause_features(feature_trajectories, head.compiled), beta)
            for head in self.clause_heads
        ]
        return torch.max(torch.stack(clause_scores, dim=-1), dim=-1).values

    def clause_trajectory_scores(self, feature_trajectories: torch.Tensor, beta: float) -> torch.Tensor:
        if feature_trajectories.ndim == 2:
            feature_trajectories = feature_trajectories.unsqueeze(0)
        return torch.stack(
            [
                head.trajectory_score(self._clause_features(feature_trajectories, head.compiled), beta)
                for head in self.clause_heads
            ],
            dim=-1,
        )

    def trajectory_score_with_clause_masks(
        self,
        feature_trajectories: torch.Tensor,
        clause_masks: torch.Tensor,
        beta: float,
        *,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score bags while restricting causal ``max`` witnesses.

        ``clause_masks`` has shape ``[batch, clauses, horizon]``.  It is used
        only to remove states that cannot explain a label change relative to a
        known-safe anchor.  Mean clauses retain their full statistic once any
        selected feature changed; last clauses are usable only when the final
        selected feature changed.  The returned validity mask identifies bags
        for which at least one clause has a causally available witness.
        """

        if feature_trajectories.ndim == 2:
            feature_trajectories = feature_trajectories.unsqueeze(0)
        if not hard and beta <= 0.0:
            raise ValueError("smooth-max beta must be positive")
        expected = (
            feature_trajectories.shape[0],
            len(self.clause_heads),
            feature_trajectories.shape[1],
        )
        if tuple(clause_masks.shape) != expected:
            raise ValueError(
                f"clause_masks must have shape {expected}, got {tuple(clause_masks.shape)}"
            )
        clause_scores: list[torch.Tensor] = []
        clause_validity: list[torch.Tensor] = []
        for clause_index, head in enumerate(self.clause_heads):
            features = self._clause_features(feature_trajectories, head.compiled)
            mask = clause_masks[:, clause_index, :].to(dtype=torch.bool)
            operator = head.compiled.clause.temporal_operator
            if operator == "max":
                state_scores = head.state_score(features)
                valid = torch.any(mask, dim=-1)
                floor = torch.finfo(state_scores.dtype).min
                # Avoid an all-``-inf`` logsumexp for a clause that is
                # representation-invariant while another composite clause is
                # valid.  The placeholder is removed by ``where`` below and
                # therefore contributes no gradient or score.
                effective_mask = mask.clone()
                effective_mask[~valid, 0] = True
                masked = state_scores.masked_fill(~effective_mask, floor)
                if hard:
                    score = torch.max(masked, dim=-1).values
                else:
                    count = (
                        torch.sum(effective_mask, dim=-1)
                        .clamp_min(1)
                        .to(state_scores.dtype)
                    )
                    score = (
                        torch.logsumexp(beta * masked, dim=-1)
                        - torch.log(count)
                    ) / beta
                score = torch.where(valid, score, torch.full_like(score, floor))
            elif operator == "mean":
                valid = torch.any(mask, dim=-1)
                score = head.trajectory_score(features, beta)
            elif operator == "last":
                valid = mask[:, -1]
                score = head.trajectory_score(features, beta)
            else:  # pragma: no cover - compilation rejects this first
                raise RuntimeError(f"uncompiled temporal operator: {operator}")
            floor = torch.finfo(score.dtype).min
            clause_scores.append(
                torch.where(valid, score, torch.full_like(score, floor))
            )
            clause_validity.append(valid)
        valid_matrix = torch.stack(clause_validity, dim=-1)
        scores = torch.max(torch.stack(clause_scores, dim=-1), dim=-1).values
        return scores, torch.any(valid_matrix, dim=-1)

    def hard_trajectory_score(self, feature_trajectory: torch.Tensor) -> torch.Tensor:
        clause_scores = [
            head.trajectory_score(self._clause_features(feature_trajectory, head.compiled), beta=20.0, hard=True)
            for head in self.clause_heads
        ]
        return torch.max(torch.stack(clause_scores, dim=-1), dim=-1).values

    def hard_clause_trajectory_scores(self, feature_trajectory: torch.Tensor) -> torch.Tensor:
        """Exact inference-time score of every atomic clause."""

        return torch.stack(
            [
                head.trajectory_score(
                    self._clause_features(feature_trajectory, head.compiled),
                    beta=20.0,
                    hard=True,
                )
                for head in self.clause_heads
            ],
            dim=-1,
        )


class ConstraintEnsemble(nn.Module):
    def __init__(self, members: Sequence[HypothesisConstraintNet]) -> None:
        super().__init__()
        if not members:
            raise ValueError("ensemble cannot be empty")
        self.members = nn.ModuleList(members)
        self.register_buffer("decision_threshold", torch.zeros((), dtype=torch.float32))

    @property
    def compiled(self) -> CompiledHypothesis:
        return self.members[0].compiled

    def member_trajectory_scores(self, features: torch.Tensor, beta: float) -> torch.Tensor:
        return torch.stack([member.trajectory_score(features, beta) for member in self.members], dim=0)

    def mean_trajectory_score(self, features: torch.Tensor, beta: float) -> torch.Tensor:
        return torch.mean(self.member_trajectory_scores(features, beta), dim=0)

    def mean_clause_trajectory_scores(self, features: torch.Tensor, beta: float) -> torch.Tensor:
        values = torch.stack(
            [member.clause_trajectory_scores(features, beta) for member in self.members],
            dim=0,
        )
        return torch.mean(values, dim=0)

    def trajectory_uncertainty(self, features: torch.Tensor, beta: float) -> torch.Tensor:
        values = self.member_trajectory_scores(features, beta)
        return values.var(dim=0, unbiased=False) if len(self.members) > 1 else torch.zeros_like(values[0])

    def member_hard_trajectory_scores(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([member.hard_trajectory_score(features) for member in self.members], dim=0)

    def mean_hard_trajectory_score(self, features: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.member_hard_trajectory_scores(features), dim=0)

    def mean_hard_clause_trajectory_scores(self, features: torch.Tensor) -> torch.Tensor:
        values = torch.stack(
            [member.hard_clause_trajectory_scores(features) for member in self.members],
            dim=0,
        )
        return torch.mean(values, dim=0)

    def hard_trajectory_uncertainty(self, features: torch.Tensor) -> torch.Tensor:
        values = self.member_hard_trajectory_scores(features)
        return values.var(dim=0, unbiased=False) if len(self.members) > 1 else torch.zeros_like(values[0])

    def mean_state_score(self, features: torch.Tensor) -> torch.Tensor:
        values = torch.stack(
            [member.state_score(features) for member in self.members], dim=0
        )
        return torch.mean(values, dim=0)

    def mean_clause_state_scores(self, features: torch.Tensor) -> torch.Tensor:
        values = torch.stack(
            [member.clause_state_scores(features) for member in self.members],
            dim=0,
        )
        return torch.mean(values, dim=0)

    def predict_features(self, features: torch.Tensor) -> int:
        with torch.no_grad():
            score = self.mean_hard_trajectory_score(features)
        return int(float(score.item()) > float(self.decision_threshold.item()))

    def set_decision_threshold(self, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("decision threshold must be finite")
        self.decision_threshold.fill_(float(value))


def choose_decision_threshold(
    calibration_scores: Sequence[float],
    calibration_labels: Sequence[int],
    expert_scores: Sequence[float],
    *,
    minimum_expert_safe_rate: float = 0.95,
) -> dict[str, object]:
    """Select a trajectory threshold without using private geometry.

    Safe means ``score <= threshold``.  Candidate thresholds are evaluated on
    held-out whole-trajectory membership labels, subject to a known-safe expert
    consistency constraint. Ties prefer higher violation recall and then the
    larger threshold, which preserves more unseen safe trajectories without
    changing calibration classification.
    """

    scores = np.asarray(calibration_scores, dtype=np.float64)
    labels = np.asarray(calibration_labels, dtype=np.int64)
    experts = np.asarray(expert_scores, dtype=np.float64)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("calibration scores and labels must be aligned vectors")
    if len(scores) == 0 or not np.all(np.isfinite(scores)) or not np.all(np.isfinite(experts)):
        raise ValueError("calibration and expert scores must be non-empty and finite")
    if not 0.0 <= minimum_expert_safe_rate <= 1.0:
        raise ValueError("minimum_expert_safe_rate must be in [0,1]")
    unique = np.unique(np.concatenate((scores, experts, np.asarray([0.0]))))
    epsilon = max(1.0e-6, float(np.ptp(unique)) * 1.0e-6)
    candidates = [float(unique[0] - epsilon)]
    candidates.extend(float(0.5 * (left + right)) for left, right in zip(unique[:-1], unique[1:]))
    candidates.append(float(unique[-1] + epsilon))
    safe_mask = labels == SAFE_LABEL
    violation_mask = labels == VIOLATION_LABEL
    rows: list[dict[str, object]] = []
    for threshold in candidates:
        predictions = (scores > threshold).astype(np.int64)
        safe_accuracy = float(np.mean(predictions[safe_mask] == SAFE_LABEL)) if np.any(safe_mask) else 0.5
        violation_recall = (
            float(np.mean(predictions[violation_mask] == VIOLATION_LABEL)) if np.any(violation_mask) else 0.5
        )
        expert_safe_rate = float(np.mean(experts <= threshold)) if len(experts) else 1.0
        rows.append(
            {
                "threshold": threshold,
                "safe_accuracy": safe_accuracy,
                "violation_recall": violation_recall,
                "balanced_accuracy": 0.5 * (safe_accuracy + violation_recall),
                "expert_safe_rate": expert_safe_rate,
                "eligible": expert_safe_rate >= minimum_expert_safe_rate,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    pool = eligible or rows
    selected = max(
        pool,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["violation_recall"]),
            float(row["threshold"]),
        ),
    )
    return {
        "selected_threshold": float(selected["threshold"]),
        "selected_metrics": selected,
        "expert_constraint_satisfied": bool(eligible),
        "candidate_count": len(rows),
    }


def describe_ensemble_parameters(ensemble: ConstraintEnsemble) -> dict[str, object]:
    """Return human-readable numeric boundaries for diagnostic artifacts.

    Constraint heads train in normalized feature coordinates.  Reporting only
    the raw tensors made a learned ``threshold=-0.35`` easy to misread, because
    for ``y_position`` that value means roughly ``y=-1.4 m``.  This conversion
    is evaluation-neutral and is never included in the semantic prompt.
    """

    clause_descriptions: list[dict[str, object]] = []
    for clause_index, compiled_clause in enumerate(ensemble.compiled.clauses):
        clause = compiled_clause.clause
        low = np.asarray(compiled_clause.input_low, dtype=np.float64)
        high = np.asarray(compiled_clause.input_high, dtype=np.float64)
        span = high - low
        member_descriptions: list[dict[str, object]] = []
        for member in ensemble.members:
            head = member.clause_heads[clause_index]
            if head.threshold is not None:
                normalized = float(head.threshold.detach().cpu().item())
                raw = float(low[0] + 0.5 * (normalized + 1.0) * span[0])
                member_description: dict[str, object] = {
                    "kind": "scalar_bound",
                    "threshold_normalized": normalized,
                    "threshold_raw": raw,
                    "scale": float(F.softplus(head.log_scale).detach().cpu().item()),
                }
            elif head.center is not None and head.log_half_width is not None:
                normalized_center = float(head.center.detach().cpu().item())
                normalized_half_width = float(F.softplus(head.log_half_width).detach().cpu().item() + 1.0e-3)
                raw_center = float(low[0] + 0.5 * (normalized_center + 1.0) * span[0])
                raw_half_width = float(0.5 * normalized_half_width * span[0])
                member_description = {
                    "kind": "equality_band",
                    "center_raw": raw_center,
                    "half_width_raw": raw_half_width,
                    "interval_raw": [raw_center - raw_half_width, raw_center + raw_half_width],
                }
            elif isinstance(head.joint_network, nn.Linear):
                normalized_weight = head.joint_network.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
                normalized_bias = float(head.joint_network.bias.detach().cpu().item())
                raw_weight = 2.0 * normalized_weight / span
                raw_bias = normalized_bias - float(np.sum(normalized_weight * (high + low) / span))
                member_description = {
                    "kind": "affine_boundary",
                    "raw_weights": {
                        variable: float(value) for variable, value in zip(clause.variables, raw_weight.tolist())
                    },
                    "raw_bias": raw_bias,
                }
            else:
                member_description = {
                    "kind": "mlp" if head.joint_network is not None else "independent_networks",
                    "parameter_count": int(sum(parameter.numel() for parameter in head.parameters())),
                }
            member_descriptions.append(member_description)
        clause_descriptions.append(
            {
                "clause_id": clause.clause_id,
                "variables": list(clause.variables),
                "relation": clause.relation,
                "coupling": clause.coupling,
                "model_family": clause.model_family,
                "members": member_descriptions,
            }
        )
    return {
        "hypothesis_id": ensemble.compiled.hypothesis.hypothesis_id,
        "parameter_count": int(sum(parameter.numel() for parameter in ensemble.parameters())),
        "decision_threshold": float(ensemble.decision_threshold.detach().cpu().item()),
        "clauses": clause_descriptions,
    }


def build_ensemble(
    compiled: CompiledHypothesis,
    *,
    hidden_dims: Sequence[int],
    ensemble_size: int,
    seed: int,
    device: torch.device,
) -> ConstraintEnsemble:
    members: list[HypothesisConstraintNet] = []
    for index in range(ensemble_size):
        torch.manual_seed(seed + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + index)
        members.append(HypothesisConstraintNet(compiled, hidden_dims))
    return ConstraintEnsemble(members).to(device)


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 160
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-5
    margin: float = 0.2
    smoothmax_beta: float = 20.0
    expert_weight: float = 2.0
    query_weight: float = 1.0
    violation_weight: float = 3.0
    safe_state_weight: float = 1.0
    latent_witness_weight: float = 1.5
    latent_witness_fraction: float = 0.10
    latent_witness_mode: str = "source_model"
    latent_witness_minimum_anchor_feature_deformation: float = 1.0e-5
    hard_trajectory_alignment_weight: float = 0.0
    # For a violation generated from a known-safe anchor by the same source
    # hypothesis, forbid max-MIL from assigning credit to states whose selected
    # features did not change.  This is a causal representation constraint,
    # not an Oracle-derived point label.
    violation_pooling_mode: str = "source_anchor_changed_states"
    violation_pooling_change_tolerance: float = 1.0e-6
    output_regularization: float = 1.0e-4
    bootstrap_queries: bool = False
    # Optional coverage correction for small CEGIS buffers.  Classic bagging
    # is performed first, then every omitted record is appended once.  This
    # retains stochastic member weights without discarding paid Oracle labels.
    bootstrap_ensure_full_coverage: bool = False
    background_safe_weight: float = 0.0
    background_sample_count: int = 256


@dataclass(frozen=True)
class TrainingSummary:
    mean_final_loss: float
    member_final_losses: tuple[float, ...]
    available_query_count: int = 0
    available_safe_query_count: int = 0
    available_violation_query_count: int = 0
    member_query_draw_counts: tuple[int, ...] = ()
    member_unique_query_counts: tuple[int, ...] = ()
    member_unique_safe_query_counts: tuple[int, ...] = ()
    member_unique_violation_query_counts: tuple[int, ...] = ()
    member_latent_witness_counts: tuple[int, ...] = ()
    member_source_anchor_masked_violation_counts: tuple[int, ...] = ()
    member_unique_source_anchor_masked_violation_counts: tuple[int, ...] = ()
    member_source_anchor_unresolved_violation_counts: tuple[int, ...] = ()
    member_representation_invariant_violation_counts: tuple[int, ...] = ()
    bootstrap_queries: bool = False
    bootstrap_ensure_full_coverage: bool = False

    def query_coverage_dict(self) -> dict[str, object]:
        return {
            "bootstrap_queries": self.bootstrap_queries,
            "bootstrap_ensure_full_coverage": self.bootstrap_ensure_full_coverage,
            "available_query_count": self.available_query_count,
            "available_safe_query_count": self.available_safe_query_count,
            "available_violation_query_count": self.available_violation_query_count,
            "member_query_draw_counts": list(self.member_query_draw_counts),
            "member_unique_query_counts": list(self.member_unique_query_counts),
            "member_unique_safe_query_counts": list(self.member_unique_safe_query_counts),
            "member_unique_violation_query_counts": list(
                self.member_unique_violation_query_counts
            ),
            "member_latent_witness_counts": list(self.member_latent_witness_counts),
            "member_source_anchor_masked_violation_counts": list(
                self.member_source_anchor_masked_violation_counts
            ),
            "member_unique_source_anchor_masked_violation_counts": list(
                self.member_unique_source_anchor_masked_violation_counts
            ),
            "member_source_anchor_unresolved_violation_counts": list(
                self.member_source_anchor_unresolved_violation_counts
            ),
            "member_representation_invariant_violation_counts": list(
                self.member_representation_invariant_violation_counts
            ),
        }


def _feature_batch(
    trajectories: list[Trajectory],
    variables: tuple[str, ...],
    library: FeatureLibrary,
    device: torch.device,
) -> torch.Tensor:
    if not trajectories:
        raise ValueError("cannot stack empty trajectories")
    horizons = {len(item.states) for item in trajectories}
    if len(horizons) != 1:
        raise ValueError("all trajectories must have the same horizon")
    array = np.stack([library.numpy_features(item.states, variables) for item in trajectories])
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _safe_loss(scores: torch.Tensor, margin: float) -> torch.Tensor:
    return F.softplus(scores + margin).mean()


def _violation_loss(scores: torch.Tensor, margin: float) -> torch.Tensor:
    return F.softplus(margin - scores).mean()


def _latent_witnesses(
    model: HypothesisConstraintNet,
    violation_batch: torch.Tensor,
    safe_reference: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("latent_witness_fraction must be in (0,1]")
    batch, horizon, dimension = violation_batch.shape
    count = max(1, int(round(horizon * fraction)))
    with torch.no_grad():
        safe = model.normalize(safe_reference.reshape(-1, dimension))
        violation = model.normalize(violation_batch.reshape(-1, dimension))
        novelty = torch.cdist(violation, safe).min(dim=1).values.reshape(batch, horizon)
        indices = torch.topk(novelty, k=count, dim=1).indices
    return torch.gather(violation_batch, 1, indices.unsqueeze(-1).expand(-1, -1, dimension))


def _source_model_witness_feature(
    record: QueryRecord,
    model: HypothesisConstraintNet,
    experts_by_id: dict[str, Trajectory],
    variables: tuple[str, ...],
    library: FeatureLibrary,
    device: torch.device,
    minimum_anchor_feature_deformation: float,
    *,
    legacy_false_safe_only: bool = False,
) -> torch.Tensor | None:
    """Resolve a causal witness without teaching unchanged safe anchor states as unsafe.

    Generated queries preserve a known-safe expert anchor.  A source model's
    argmax is useful only for that same hypothesis, and only if its selected
    feature representation actually changed from the anchor.  An unchanged
    endpoint is a common multiple-instance shortcut, so it falls back to the
    most-deformed selected-feature state.
    """

    if record.label != VIOLATION_LABEL:
        return None
    witness_kind = record.trajectory.metadata.get("source_witness_kind")
    shared_intervention = witness_kind == "intervention_max_deformation"
    source_model_witness = witness_kind in {
        "model_argmax_before_oracle",
        "target_clause_argmax_before_oracle",
    }
    matching_source = (
        source_model_witness
        and record.source_hypothesis_id
        == model.compiled.hypothesis.hypothesis_id
    )
    if legacy_false_safe_only:
        matching_source = matching_source and (
            record.trajectory.metadata.get("source") == "model_false_safe"
        )
    if not shared_intervention and not matching_source:
        return None
    raw_index = record.trajectory.metadata.get("source_witness_index")
    if raw_index is None:
        return None
    index = int(raw_index)
    if not 0 <= index < len(record.trajectory.states):
        return None
    candidate_features = library.torch_features(
        torch.as_tensor(record.trajectory.states, dtype=torch.float32, device=device),
        variables,
    )
    expert_id = str(record.trajectory.metadata.get("expert_id", ""))
    anchor = experts_by_id.get(expert_id)
    if (
        not legacy_false_safe_only
        and anchor is not None
        and anchor.states.shape == record.trajectory.states.shape
    ):
        anchor_features = library.torch_features(
            torch.as_tensor(anchor.states, dtype=torch.float32, device=device),
            variables,
        )
        lows, highs = library.bounds(variables)
        scale = torch.as_tensor(
            np.maximum(
                np.asarray(highs, dtype=np.float32)
                - np.asarray(lows, dtype=np.float32),
                1.0e-6,
            ),
            dtype=torch.float32,
            device=device,
        )
        deformation = torch.linalg.vector_norm(
            (candidate_features - anchor_features) / scale,
            dim=-1,
        )
        tolerance = max(float(minimum_anchor_feature_deformation), 0.0)
        if float(deformation[index].detach().cpu().item()) <= tolerance:
            index = int(torch.argmax(deformation).detach().cpu().item())
        if float(deformation[index].detach().cpu().item()) <= tolerance:
            return None
    return candidate_features[index]


def _source_anchor_clause_masks(
    records: Sequence[QueryRecord],
    model: HypothesisConstraintNet,
    experts_by_id: dict[str, Trajectory],
    variables: tuple[str, ...],
    library: FeatureLibrary,
    device: torch.device,
    change_tolerance: float,
) -> tuple[torch.Tensor, int, int, int, int]:
    """Build clause-aware changed-state masks for source-model violations.

    A trajectory generated from a known-safe expert may differ in only part of
    the horizon.  When that query is an Oracle violation for the hypothesis
    that generated it, an unchanged selected-feature state is causally
    incapable of explaining the changed trajectory label.  Non-source records
    and records without a resolvable anchor retain the ordinary all-state MIL
    pool so this policy never invents a pairing.
    """

    if not np.isfinite(change_tolerance) or change_tolerance < 0.0:
        raise ValueError(
            "violation_pooling_change_tolerance must be finite and nonnegative"
        )
    if not records:
        raise ValueError("cannot build source-anchor masks for an empty record set")
    horizon = len(records[0].trajectory.states)
    masks = torch.ones(
        (len(records), len(model.clause_heads), horizon),
        dtype=torch.bool,
        device=device,
    )
    masked_count = 0
    unique_masked_records: set[int] = set()
    unresolved_count = 0
    invariant_count = 0
    model_id = model.compiled.hypothesis.hypothesis_id
    for record_index, record in enumerate(records):
        if record.label != VIOLATION_LABEL:
            continue
        if record.source_hypothesis_id != model_id:
            continue
        expert_id = str(record.trajectory.metadata.get("expert_id", ""))
        anchor = experts_by_id.get(expert_id)
        if (
            anchor is None
            or anchor.states.shape != record.trajectory.states.shape
            or len(record.trajectory.states) != horizon
        ):
            unresolved_count += 1
            continue
        candidate_features = library.torch_features(
            torch.as_tensor(
                record.trajectory.states,
                dtype=torch.float32,
                device=device,
            ),
            variables,
        )
        anchor_features = library.torch_features(
            torch.as_tensor(anchor.states, dtype=torch.float32, device=device),
            variables,
        )
        masks[record_index].fill_(False)
        any_representation_change = False
        for clause_index, head in enumerate(model.clause_heads):
            candidate_clause = model._clause_features(
                candidate_features, head.compiled
            )
            anchor_clause = model._clause_features(anchor_features, head.compiled)
            deformation = torch.linalg.vector_norm(
                head.normalize(candidate_clause) - head.normalize(anchor_clause),
                dim=-1,
            )
            changed = deformation > float(change_tolerance)
            masks[record_index, clause_index] = changed
            operator = head.compiled.clause.temporal_operator
            if operator == "last":
                any_representation_change = any_representation_change or bool(
                    changed[-1].detach().cpu().item()
                )
            else:
                any_representation_change = any_representation_change or bool(
                    torch.any(changed).detach().cpu().item()
                )
        if not any_representation_change:
            invariant_count += 1
            # A missing causal support is useful diagnostic evidence but is
            # not enough to discard a paid trajectory-level Oracle label.
            # Preserve the legacy bag loss in this unresolved case.
            masks[record_index].fill_(True)
        else:
            masked_count += 1
            unique_masked_records.add(id(record))
    return (
        masks,
        masked_count,
        len(unique_masked_records),
        unresolved_count,
        invariant_count,
    )


def fit_ensemble(
    ensemble: ConstraintEnsemble,
    experts: list[Trajectory],
    query_records: list[QueryRecord],
    library: FeatureLibrary,
    config: TrainerConfig,
    *,
    seed: int,
    device: torch.device,
) -> TrainingSummary:
    if config.violation_pooling_mode not in {
        "all_states",
        "source_anchor_changed_states",
    }:
        raise ValueError(
            "violation_pooling_mode must be all_states or "
            "source_anchor_changed_states"
        )
    if (
        not np.isfinite(config.violation_pooling_change_tolerance)
        or config.violation_pooling_change_tolerance < 0.0
    ):
        raise ValueError(
            "violation_pooling_change_tolerance must be finite and nonnegative"
        )
    variables = ensemble.compiled.variables
    expert_batch = _feature_batch(experts, variables, library, device)
    final_losses: list[float] = []
    member_query_draw_counts: list[int] = []
    member_unique_query_counts: list[int] = []
    member_unique_safe_query_counts: list[int] = []
    member_unique_violation_query_counts: list[int] = []
    member_latent_witness_counts: list[int] = []
    member_source_anchor_masked_violation_counts: list[int] = []
    member_unique_source_anchor_masked_violation_counts: list[int] = []
    member_source_anchor_unresolved_violation_counts: list[int] = []
    member_representation_invariant_violation_counts: list[int] = []
    expert_id_pairs = [
        (str(expert.metadata.get("trajectory_id")), expert)
        for expert in experts
        if expert.metadata.get("trajectory_id") is not None
    ]
    if config.violation_pooling_mode == "source_anchor_changed_states":
        expert_ids = [item[0] for item in expert_id_pairs]
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError(
                "source-anchor violation pooling requires unique expert trajectory_id values"
            )
    experts_by_id = dict(expert_id_pairs)
    for member_index, model in enumerate(ensemble.members):
        generator = np.random.default_rng(seed + 1009 * member_index)
        if config.bootstrap_queries and query_records:
            selected_records: list[QueryRecord] = []
            for label in (SAFE_LABEL, VIOLATION_LABEL):
                pool = [record for record in query_records if record.label == label]
                if pool:
                    indices = generator.integers(0, len(pool), size=len(pool))
                    selected_records.extend(pool[int(index)] for index in indices)
                    if config.bootstrap_ensure_full_coverage:
                        selected_ids = {id(record) for record in selected_records}
                        selected_records.extend(
                            record for record in pool if id(record) not in selected_ids
                        )
        else:
            selected_records = list(query_records)
        unique_records = {id(record): record for record in selected_records}
        member_query_draw_counts.append(len(selected_records))
        member_unique_query_counts.append(len(unique_records))
        member_unique_safe_query_counts.append(
            sum(record.label == SAFE_LABEL for record in unique_records.values())
        )
        member_unique_violation_query_counts.append(
            sum(record.label == VIOLATION_LABEL for record in unique_records.values())
        )
        safe_trajectories = [record.trajectory for record in selected_records if record.label == SAFE_LABEL]
        violation_records = [
            record for record in selected_records if record.label == VIOLATION_LABEL
        ]
        violation_trajectories = [record.trajectory for record in violation_records]
        safe_batch = _feature_batch(safe_trajectories, variables, library, device) if safe_trajectories else None
        violation_batch = (
            _feature_batch(violation_trajectories, variables, library, device)
            if violation_trajectories
            else None
        )
        safe_reference = torch.cat([expert_batch, safe_batch], dim=0) if safe_batch is not None else expert_batch
        violation_clause_masks: torch.Tensor | None = None
        masked_violation_count = 0
        unique_masked_violation_count = 0
        unresolved_violation_count = 0
        invariant_violation_count = 0
        if (
            violation_batch is not None
            and config.violation_pooling_mode == "source_anchor_changed_states"
        ):
            (
                violation_clause_masks,
                masked_violation_count,
                unique_masked_violation_count,
                unresolved_violation_count,
                invariant_violation_count,
            ) = _source_anchor_clause_masks(
                violation_records,
                model,
                experts_by_id,
                variables,
                library,
                device,
                config.violation_pooling_change_tolerance,
            )
        member_source_anchor_masked_violation_counts.append(
            masked_violation_count
        )
        member_unique_source_anchor_masked_violation_counts.append(
            unique_masked_violation_count
        )
        member_source_anchor_unresolved_violation_counts.append(
            unresolved_violation_count
        )
        member_representation_invariant_violation_counts.append(
            invariant_violation_count
        )
        witnesses: torch.Tensor | None = None
        if violation_batch is not None and config.latent_witness_weight > 0.0:
            if config.latent_witness_mode == "novelty":
                witnesses = _latent_witnesses(
                    model, violation_batch, safe_reference, config.latent_witness_fraction
                )
            elif config.latent_witness_mode == "source_model_all_interventions":
                located: list[torch.Tensor] = []
                for record in selected_records:
                    feature = _source_model_witness_feature(
                        record,
                        model,
                        experts_by_id,
                        variables,
                        library,
                        device,
                        config.latent_witness_minimum_anchor_feature_deformation,
                    )
                    if feature is not None:
                        located.append(feature)
                if located:
                    witnesses = torch.stack(located, dim=0)
            elif config.latent_witness_mode in {"source_model", "source_model_legacy"}:
                located = []
                for record in selected_records:
                    feature = _source_model_witness_feature(
                        record,
                        model,
                        experts_by_id,
                        variables,
                        library,
                        device,
                        config.latent_witness_minimum_anchor_feature_deformation,
                        legacy_false_safe_only=True,
                    )
                    if feature is not None:
                        located.append(feature)
                if located:
                    witnesses = torch.stack(located, dim=0)
            elif config.latent_witness_mode != "none":
                raise ValueError(
                    "latent_witness_mode must be source_model, "
                    "source_model_all_interventions, novelty, or none"
                )
        member_latent_witness_counts.append(
            0 if witnesses is None else int(witnesses.shape[0])
        )
        background_batch: torch.Tensor | None = None
        if config.background_safe_weight > 0.0 and config.background_sample_count > 0:
            lows, highs = library.bounds(variables)
            background = generator.uniform(
                np.asarray(lows, dtype=np.float32),
                np.asarray(highs, dtype=np.float32),
                size=(config.background_sample_count, len(variables)),
            ).astype(np.float32)
            background_batch = torch.as_tensor(background, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        final_loss = float("nan")
        model.train()
        for _ in range(config.epochs):
            optimizer.zero_grad()
            expert_state_scores = model.state_score(expert_batch)
            expert_scores = model.trajectory_score(expert_batch, config.smoothmax_beta)
            loss = config.expert_weight * _safe_loss(expert_scores, config.margin)
            loss = loss + config.safe_state_weight * _safe_loss(expert_state_scores, config.margin)
            if safe_batch is not None:
                loss = loss + config.query_weight * _safe_loss(
                    model.trajectory_score(safe_batch, config.smoothmax_beta), config.margin
                )
                loss = loss + config.safe_state_weight * _safe_loss(model.state_score(safe_batch), config.margin)
            if violation_batch is not None:
                if violation_clause_masks is None:
                    violation_scores = model.trajectory_score(
                        violation_batch, config.smoothmax_beta
                    )
                else:
                    violation_scores, _ = model.trajectory_score_with_clause_masks(
                        violation_batch,
                        violation_clause_masks,
                        config.smoothmax_beta,
                    )
                loss = loss + config.violation_weight * _violation_loss(
                    violation_scores, config.margin
                )
            if config.hard_trajectory_alignment_weight > 0.0:
                hard_weight = float(config.hard_trajectory_alignment_weight)
                loss = loss + hard_weight * config.expert_weight * _safe_loss(
                    model.hard_trajectory_score(expert_batch), config.margin
                )
                if safe_batch is not None:
                    loss = loss + hard_weight * config.query_weight * _safe_loss(
                        model.hard_trajectory_score(safe_batch), config.margin
                    )
                if violation_batch is not None:
                    if violation_clause_masks is None:
                        hard_violation_scores = model.hard_trajectory_score(
                            violation_batch
                        )
                    else:
                        (
                            hard_violation_scores,
                            _,
                        ) = model.trajectory_score_with_clause_masks(
                            violation_batch,
                            violation_clause_masks,
                            config.smoothmax_beta,
                            hard=True,
                        )
                    loss = loss + hard_weight * config.violation_weight * _violation_loss(
                        hard_violation_scores, config.margin
                    )
            if witnesses is not None:
                loss = loss + config.latent_witness_weight * _violation_loss(model.state_score(witnesses), config.margin)
            if background_batch is not None:
                # Optional sparse-forbidden-set prior. It is disabled by
                # default because a global bound need not occupy little volume.
                loss = loss + config.background_safe_weight * _safe_loss(
                    model.state_score(background_batch), config.margin
                )
            loss = loss + config.output_regularization * torch.mean(expert_state_scores.square())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
        model.eval()
        final_losses.append(final_loss)
    return TrainingSummary(
        float(np.mean(final_losses)),
        tuple(final_losses),
        available_query_count=len(query_records),
        available_safe_query_count=sum(
            record.label == SAFE_LABEL for record in query_records
        ),
        available_violation_query_count=sum(
            record.label == VIOLATION_LABEL for record in query_records
        ),
        member_query_draw_counts=tuple(member_query_draw_counts),
        member_unique_query_counts=tuple(member_unique_query_counts),
        member_unique_safe_query_counts=tuple(member_unique_safe_query_counts),
        member_unique_violation_query_counts=tuple(
            member_unique_violation_query_counts
        ),
        member_latent_witness_counts=tuple(member_latent_witness_counts),
        member_source_anchor_masked_violation_counts=tuple(
            member_source_anchor_masked_violation_counts
        ),
        member_unique_source_anchor_masked_violation_counts=tuple(
            member_unique_source_anchor_masked_violation_counts
        ),
        member_source_anchor_unresolved_violation_counts=tuple(
            member_source_anchor_unresolved_violation_counts
        ),
        member_representation_invariant_violation_counts=tuple(
            member_representation_invariant_violation_counts
        ),
        bootstrap_queries=bool(config.bootstrap_queries),
        bootstrap_ensure_full_coverage=bool(
            config.bootstrap_ensure_full_coverage
        ),
    )


class LearnerRegistry:
    """Owns one independently trained neural ensemble per active hypothesis."""

    def __init__(
        self,
        library: FeatureLibrary,
        *,
        hidden_dims: Sequence[int],
        ensemble_size: int,
        seed: int,
        device: torch.device,
    ) -> None:
        self.library = library
        self.hidden_dims = tuple(map(int, hidden_dims))
        self.ensemble_size = int(ensemble_size)
        self.seed = int(seed)
        self.device = device
        self.models: dict[str, ConstraintEnsemble] = {}

    @staticmethod
    def _structural_seed_offset(compiled: CompiledHypothesis) -> int:
        fingerprint = repr(compiled.hypothesis.signature()).encode("utf-8")
        digest = hashlib.sha256(fingerprint).digest()
        return int.from_bytes(digest[:8], "big") % 2_000_000_000

    def ensure(self, compiled: CompiledHypothesis) -> ConstraintEnsemble:
        key = compiled.hypothesis.hypothesis_id
        if key not in self.models:
            stable_offset = self._structural_seed_offset(compiled)
            self.models[key] = build_ensemble(
                compiled,
                hidden_dims=self.hidden_dims,
                ensemble_size=self.ensemble_size,
                seed=self.seed + stable_offset,
                device=self.device,
            )
        return self.models[key]

    def reinitialize(self, hypothesis_id: str, *, seed_offset: int = 0) -> ConstraintEnsemble:
        if hypothesis_id not in self.models:
            raise KeyError(hypothesis_id)
        compiled = self.models[hypothesis_id].compiled
        stable_offset = self._structural_seed_offset(compiled)
        ensemble = build_ensemble(
            compiled,
            hidden_dims=self.hidden_dims,
            ensemble_size=self.ensemble_size,
            seed=self.seed + stable_offset + int(seed_offset),
            device=self.device,
        )
        self.models[hypothesis_id] = ensemble
        return ensemble

    def predict(self, hypothesis_id: str, trajectory: Trajectory) -> tuple[int, float, float]:
        ensemble = self.models[hypothesis_id]
        features = self.library.torch_features(
            torch.as_tensor(trajectory.states, dtype=torch.float32, device=self.device),
            ensemble.compiled.variables,
        )
        with torch.no_grad():
            # The hypothesis semantics are existential for max aggregation:
            # one violating state makes the trajectory violating.  Smooth-max
            # remains a differentiable training surrogate only; using it for
            # evidence or acquisition can dilute a short violation over a long
            # trajectory and disagree with final evaluation.
            score = float(ensemble.mean_hard_trajectory_score(features).item())
            uncertainty = float(ensemble.hard_trajectory_uncertainty(features).item())
        threshold = float(ensemble.decision_threshold.item())
        return int(score > threshold), score, uncertainty

    def hard_clause_score(
        self,
        hypothesis_id: str,
        trajectory: Trajectory,
        clause_id: str,
    ) -> float:
        """Return the calibrated inference-time score for one named clause."""

        ensemble = self.models[hypothesis_id]
        clause_ids = [clause.clause.clause_id for clause in ensemble.compiled.clauses]
        if clause_id not in clause_ids:
            raise KeyError(f"unknown clause_id {clause_id!r} for {hypothesis_id!r}")
        features = self.library.torch_features(
            torch.as_tensor(trajectory.states, dtype=torch.float32, device=self.device),
            ensemble.compiled.variables,
        )
        with torch.no_grad():
            scores = ensemble.mean_hard_clause_trajectory_scores(features)
        return float(scores[..., clause_ids.index(clause_id)].item())
