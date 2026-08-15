"""Hypothesis-conditioned neural constraint models and MIL training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import FeatureLibrary
from .hypotheses import CompiledHypothesis
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


class HypothesisConstraintNet(nn.Module):
    """State logit conditioned on one compiled semantic structure."""

    def __init__(self, compiled: CompiledHypothesis, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.compiled = compiled
        input_dim = len(compiled.variables)
        low = torch.as_tensor(compiled.input_low, dtype=torch.float32)
        high = torch.as_tensor(compiled.input_high, dtype=torch.float32)
        if torch.any(high <= low):
            raise ValueError("invalid feature bounds")
        self.register_buffer("input_low", low)
        self.register_buffer("input_high", high)
        self.threshold: nn.Parameter | None = None
        self.log_scale: nn.Parameter | None = None
        if compiled.hypothesis.relation in {"upper_bound", "lower_bound"}:
            self.threshold = nn.Parameter(torch.zeros(1))
            self.log_scale = nn.Parameter(torch.zeros(1))
            self.joint_network = None
            self.independent_networks = nn.ModuleList()
        elif compiled.hypothesis.coupling == "joint":
            self.joint_network: nn.Module | None = _constraint_network(
                input_dim, hidden_dims, compiled.hypothesis.model_family
            )
            self.independent_networks = nn.ModuleList()
        else:
            self.joint_network = None
            self.independent_networks = nn.ModuleList(
                [_constraint_network(1, hidden_dims, compiled.hypothesis.model_family) for _ in range(input_dim)]
            )

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        return 2.0 * (features - self.input_low) / (self.input_high - self.input_low) - 1.0

    def state_score(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize(features)
        if self.threshold is not None and self.log_scale is not None:
            scalar = normalized[..., 0]
            scale = F.softplus(self.log_scale) + 1.0e-4
            if self.compiled.hypothesis.relation == "upper_bound":
                return scale * (scalar - self.threshold)
            return scale * (self.threshold - scalar)
        if self.joint_network is not None:
            return self.joint_network(normalized).squeeze(-1)
        component_scores = [
            network(normalized[..., index : index + 1]).squeeze(-1)
            for index, network in enumerate(self.independent_networks)
        ]
        return torch.max(torch.stack(component_scores, dim=-1), dim=-1).values

    def trajectory_score(self, feature_trajectories: torch.Tensor, beta: float) -> torch.Tensor:
        if feature_trajectories.ndim == 2:
            feature_trajectories = feature_trajectories.unsqueeze(0)
        scores = self.state_score(feature_trajectories)
        operator = self.compiled.hypothesis.temporal_operator
        if operator == "max":
            return normalized_smooth_max(scores, beta=beta, dim=-1)
        if operator == "mean":
            return torch.mean(scores, dim=-1)
        if operator == "last":
            return scores[..., -1]
        raise RuntimeError(f"uncompiled temporal operator: {operator}")

    def hard_trajectory_score(self, feature_trajectory: torch.Tensor) -> torch.Tensor:
        scores = self.state_score(feature_trajectory)
        operator = self.compiled.hypothesis.temporal_operator
        if operator == "max":
            return torch.max(scores, dim=-1).values
        if operator == "mean":
            return torch.mean(scores, dim=-1)
        return scores[..., -1]


class ConstraintEnsemble(nn.Module):
    def __init__(self, members: Sequence[HypothesisConstraintNet]) -> None:
        super().__init__()
        if not members:
            raise ValueError("ensemble cannot be empty")
        self.members = nn.ModuleList(members)

    @property
    def compiled(self) -> CompiledHypothesis:
        return self.members[0].compiled

    def member_trajectory_scores(self, features: torch.Tensor, beta: float) -> torch.Tensor:
        return torch.stack([member.trajectory_score(features, beta) for member in self.members], dim=0)

    def mean_trajectory_score(self, features: torch.Tensor, beta: float) -> torch.Tensor:
        return self.member_trajectory_scores(features, beta).mean(dim=0)

    def trajectory_uncertainty(self, features: torch.Tensor, beta: float) -> torch.Tensor:
        values = self.member_trajectory_scores(features, beta)
        return values.var(dim=0, unbiased=False) if len(self.members) > 1 else torch.zeros_like(values[0])

    def mean_state_score(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([member.state_score(features) for member in self.members], dim=0).mean(dim=0)

    def predict_features(self, features: torch.Tensor) -> int:
        with torch.no_grad():
            member_scores = torch.stack([member.hard_trajectory_score(features) for member in self.members])
        return int(float(member_scores.mean().item()) > 0.0)


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
    output_regularization: float = 1.0e-4
    bootstrap_queries: bool = True


@dataclass(frozen=True)
class TrainingSummary:
    mean_final_loss: float
    member_final_losses: tuple[float, ...]


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
    variables = ensemble.compiled.variables
    expert_batch = _feature_batch(experts, variables, library, device)
    final_losses: list[float] = []
    for member_index, model in enumerate(ensemble.members):
        generator = np.random.default_rng(seed + 1009 * member_index)
        if config.bootstrap_queries and query_records:
            selected_records: list[QueryRecord] = []
            for label in (SAFE_LABEL, VIOLATION_LABEL):
                pool = [record for record in query_records if record.label == label]
                if pool:
                    indices = generator.integers(0, len(pool), size=len(pool))
                    selected_records.extend(pool[int(index)] for index in indices)
        else:
            selected_records = list(query_records)
        safe_trajectories = [record.trajectory for record in selected_records if record.label == SAFE_LABEL]
        violation_trajectories = [record.trajectory for record in selected_records if record.label == VIOLATION_LABEL]
        safe_batch = _feature_batch(safe_trajectories, variables, library, device) if safe_trajectories else None
        violation_batch = (
            _feature_batch(violation_trajectories, variables, library, device)
            if violation_trajectories
            else None
        )
        safe_reference = torch.cat([expert_batch, safe_batch], dim=0) if safe_batch is not None else expert_batch
        witnesses = (
            _latent_witnesses(model, violation_batch, safe_reference, config.latent_witness_fraction)
            if violation_batch is not None and config.latent_witness_weight > 0.0
            else None
        )
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
                loss = loss + config.violation_weight * _violation_loss(
                    model.trajectory_score(violation_batch, config.smoothmax_beta), config.margin
                )
            if witnesses is not None:
                loss = loss + config.latent_witness_weight * _violation_loss(model.state_score(witnesses), config.margin)
            loss = loss + config.output_regularization * torch.mean(expert_state_scores.square())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
        model.eval()
        final_losses.append(final_loss)
    return TrainingSummary(float(np.mean(final_losses)), tuple(final_losses))


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

    def ensure(self, compiled: CompiledHypothesis) -> ConstraintEnsemble:
        key = compiled.hypothesis.hypothesis_id
        if key not in self.models:
            stable_offset = sum((index + 1) * ord(character) for index, character in enumerate(key))
            self.models[key] = build_ensemble(
                compiled,
                hidden_dims=self.hidden_dims,
                ensemble_size=self.ensemble_size,
                seed=self.seed + stable_offset,
                device=self.device,
            )
        return self.models[key]

    def predict(self, hypothesis_id: str, trajectory: Trajectory) -> tuple[int, float, float]:
        ensemble = self.models[hypothesis_id]
        features = self.library.torch_features(
            torch.as_tensor(trajectory.states, dtype=torch.float32, device=self.device),
            ensemble.compiled.variables,
        )
        with torch.no_grad():
            score = float(ensemble.mean_trajectory_score(features.unsqueeze(0), beta=20.0).item())
            uncertainty = float(ensemble.trajectory_uncertainty(features.unsqueeze(0), beta=20.0).item())
        return int(score > 0.0), score, uncertainty
