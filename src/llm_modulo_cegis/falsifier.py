"""Hypothesis-specific trajectory synthesis without hidden-truth access."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .data import FeatureLibrary
from .learner import ConstraintEnsemble
from .types import InterventionSpec, Trajectory


@dataclass(frozen=True)
class FalsifierConfig:
    steps: int = 180
    learning_rate: float = 0.025
    smoothmax_beta: float = 20.0
    expert_weight: float = 0.3
    smoothness_weight: float = 0.2
    length_weight: float = 1.0
    boundary_weight: float = 8.0
    uncertainty_weight: float = 0.5
    step_penalty_weight: float = 30.0
    workspace_penalty_weight: float = 30.0
    epsilon: float = 0.05
    max_step: float = 0.35


@dataclass
class FalsifierResult:
    trajectory: Trajectory
    mode: str
    hypothesis_id: str
    initial_loss: float
    final_loss: float
    final_score: float
    final_uncertainty: float
    valid: bool
    validation_reason: str


def displacement_actions(states: np.ndarray) -> np.ndarray:
    actions = np.zeros_like(states, dtype=np.float32)
    actions[:-1] = np.diff(states, axis=0)
    return actions


def generate_warmup_candidate(
    expert: Trajectory,
    index: int,
    rng: np.random.Generator,
    workspace_x: tuple[float, float],
    workspace_y: tuple[float, float],
) -> Trajectory:
    """Task-agnostic interpolation and local deformation."""
    states = expert.states.astype(np.float64)
    line = np.linspace(states[0], states[-1], len(states))
    cycle = index % 10
    alpha = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.25, 0.40, 0.65, 0.90)[cycle]
    candidate = (1.0 - alpha) * states + alpha * line
    phase = np.linspace(0.0, 1.0, len(states))
    window = np.sin(np.pi * phase) ** 2
    sign = 1.0 if (index // 10) % 2 == 0 else -1.0
    candidate[:, 1] += sign * rng.uniform(0.0, 0.15) * window
    candidate[0] = states[0]
    candidate[-1] = states[-1]
    candidate[:, 0] = np.clip(candidate[:, 0], *workspace_x)
    candidate[:, 1] = np.clip(candidate[:, 1], *workspace_y)
    candidate = candidate.astype(np.float32)
    return Trajectory(
        candidate,
        displacement_actions(candidate),
        metadata={"source": "warmup", "alpha": alpha, "expert_id": expert.metadata.get("trajectory_id")},
    )


class HypothesisFalsifier:
    """Optimize raw waypoints through a hypothesis's differentiable feature map."""

    def __init__(
        self,
        library: FeatureLibrary,
        config: FalsifierConfig,
        workspace_x: tuple[float, float],
        workspace_y: tuple[float, float],
        device: torch.device,
    ) -> None:
        self.library = library
        self.config = config
        self.workspace_x = workspace_x
        self.workspace_y = workspace_y
        self.device = device

    def generate(
        self,
        ensemble: ConstraintEnsemble,
        expert: Trajectory,
        intervention: InterventionSpec,
        *,
        initialization_mix: float,
    ) -> FalsifierResult:
        mode = self._numeric_mode(intervention.kind)
        expert_tensor = torch.as_tensor(expert.states, dtype=torch.float32, device=self.device)
        initial = self._initial_path(expert_tensor, intervention, initialization_mix)
        interior = torch.nn.Parameter(initial[1:-1].clone())
        optimizer = torch.optim.Adam([interior], lr=self.config.learning_rate)
        previous = [parameter.requires_grad for parameter in ensemble.parameters()]
        for parameter in ensemble.parameters():
            parameter.requires_grad_(False)
        ensemble.eval()
        initial_loss = float("nan")
        final_loss = float("nan")
        try:
            for step in range(self.config.steps):
                optimizer.zero_grad()
                path = torch.cat((expert_tensor[:1], interior, expert_tensor[-1:]), dim=0)
                loss, _, _ = self._objective(path, expert_tensor, ensemble, mode)
                if step == 0:
                    initial_loss = float(loss.detach().cpu().item())
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    interior[:, 0].clamp_(*self.workspace_x)
                    interior[:, 1].clamp_(*self.workspace_y)
                final_loss = float(loss.detach().cpu().item())
        finally:
            for parameter, required in zip(ensemble.parameters(), previous):
                parameter.requires_grad_(required)
        with torch.no_grad():
            path = torch.cat((expert_tensor[:1], interior, expert_tensor[-1:]), dim=0)
            _, score, uncertainty = self._objective(path, expert_tensor, ensemble, mode)
        states = path.detach().cpu().numpy().astype(np.float32)
        hypothesis_id = ensemble.compiled.hypothesis.hypothesis_id
        trajectory = Trajectory(
            states,
            displacement_actions(states),
            metadata={
                "source": intervention.kind,
                "source_hypothesis_id": hypothesis_id,
                "intervention_variable": intervention.variable,
                "expert_id": expert.metadata.get("trajectory_id"),
            },
        )
        valid, reason = self.validate(trajectory, expert)
        return FalsifierResult(
            trajectory=trajectory,
            mode=intervention.kind,
            hypothesis_id=hypothesis_id,
            initial_loss=initial_loss,
            final_loss=final_loss,
            final_score=float(score.item()),
            final_uncertainty=float(uncertainty.item()),
            valid=valid,
            validation_reason=reason,
        )

    def _initial_path(
        self,
        expert: torch.Tensor,
        intervention: InterventionSpec,
        mix: float,
    ) -> torch.Tensor:
        time = torch.linspace(0.0, 1.0, len(expert), device=expert.device)[:, None]
        straight = expert[0][None, :] * (1.0 - time) + expert[-1][None, :] * time
        initial = (1.0 - mix) * expert + mix * straight
        if intervention.kind != "local_feature_stress":
            return initial
        window = torch.sin(torch.pi * time).square().squeeze(-1)
        variable = intervention.variable
        if variable in {"x_position", "x_velocity"}:
            initial[:, 0] = initial[:, 0] + 0.12 * window
        elif variable in {"y_position", "y_velocity"}:
            initial[:, 1] = initial[:, 1] + 0.12 * window
        elif variable == "speed":
            zigzag = torch.sin(torch.linspace(0.0, 10.0 * torch.pi, len(expert), device=expert.device))
            initial[:, 1] = initial[:, 1] + 0.04 * window * zigzag
        initial[0] = expert[0]
        initial[-1] = expert[-1]
        return initial

    @staticmethod
    def _numeric_mode(kind: str) -> str:
        if kind in {"model_false_safe", "shortcut"}:
            return "false_safe"
        if kind == "model_false_unsafe":
            return "false_unsafe"
        return "boundary"

    def _objective(
        self,
        path: torch.Tensor,
        expert: torch.Tensor,
        ensemble: ConstraintEnsemble,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.library.torch_features(path, ensemble.compiled.variables).unsqueeze(0)
        score = ensemble.mean_trajectory_score(features, self.config.smoothmax_beta).squeeze(0)
        uncertainty = ensemble.trajectory_uncertainty(features, self.config.smoothmax_beta).squeeze(0)
        first = path[1:] - path[:-1]
        second = path[2:] - 2.0 * path[1:-1] + path[:-2]
        length = torch.linalg.vector_norm(first, dim=-1).sum()
        smoothness = torch.mean(torch.sum(second.square(), dim=-1))
        expert_distance = torch.mean(torch.sum((path - expert).square(), dim=-1))
        step_penalty = torch.mean(F.relu(torch.linalg.vector_norm(first, dim=-1) - self.config.max_step).square())
        workspace_penalty = torch.mean(
            F.relu(self.workspace_x[0] - path[:, 0]).square()
            + F.relu(path[:, 0] - self.workspace_x[1]).square()
            + F.relu(self.workspace_y[0] - path[:, 1]).square()
            + F.relu(path[:, 1] - self.workspace_y[1]).square()
        )
        if mode == "false_safe":
            semantic = F.relu(score + self.config.epsilon).square()
            length_weight = self.config.length_weight
            expert_weight = self.config.expert_weight
        elif mode == "false_unsafe":
            semantic = F.relu(self.config.epsilon - score).square()
            length_weight = 0.0
            expert_weight = 5.0 * self.config.expert_weight
        else:
            semantic = score.square()
            length_weight = 0.25 * self.config.length_weight
            expert_weight = self.config.expert_weight
        loss = (
            expert_weight * expert_distance
            + self.config.smoothness_weight * smoothness
            + length_weight * length
            + self.config.boundary_weight * semantic
            - self.config.uncertainty_weight * uncertainty
            + self.config.step_penalty_weight * step_penalty
            + self.config.workspace_penalty_weight * workspace_penalty
        )
        return loss, score, uncertainty

    def validate(self, trajectory: Trajectory, expert: Trajectory) -> tuple[bool, str]:
        states = trajectory.states
        if not np.all(np.isfinite(states)):
            return False, "non_finite"
        if not np.allclose(states[0], expert.states[0], atol=1.0e-5):
            return False, "start_changed"
        if not np.allclose(states[-1], expert.states[-1], atol=1.0e-5):
            return False, "goal_changed"
        if np.any(states[:, 0] < self.workspace_x[0]) or np.any(states[:, 0] > self.workspace_x[1]):
            return False, "x_workspace"
        if np.any(states[:, 1] < self.workspace_y[0]) or np.any(states[:, 1] > self.workspace_y[1]):
            return False, "y_workspace"
        if float(np.max(np.linalg.norm(np.diff(states, axis=0), axis=1))) > 1.2 * self.config.max_step:
            return False, "step_limit"
        return True, "valid"
