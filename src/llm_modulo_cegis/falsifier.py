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
    feature_stress_weight: float = 0.15
    invariance_weight: float = 0.05
    epsilon: float = 0.05
    max_step: float = 0.35
    false_unsafe_trust_radius: float = 0.32
    false_unsafe_use_hard_margin: bool = True
    false_unsafe_anchor_margin: float = 0.02
    false_unsafe_hard_margin: float = 0.05
    false_unsafe_smooth_margin_weight: float = 0.25
    false_unsafe_gradient_tolerance: float = 1.0e-8
    false_unsafe_radius_ladder: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        radius = float(self.false_unsafe_trust_radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("false_unsafe_trust_radius must be finite and strictly positive")


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
    """Task-agnostic paired deformation relative to the demonstrated detour."""

    # Warmup evidence should not depend on whether an expert happened to pass
    # above or below an obstacle.  Each pair uses the same expert and scale:
    # one member contracts its demonstrated detour toward the endpoint chord,
    # while the other continues the demonstrated detour away from that chord.
    # Keeping this bank deterministic prevents numeric model seeds from
    # accidentally changing the warmup class balance.
    del rng  # retained in the public API for backward compatibility
    states = expert.states.astype(np.float64)
    line = np.linspace(states[0], states[-1], len(states))
    deformation_basis = states - line
    warmup_basis = "demonstrated_detour"
    if float(np.max(np.linalg.norm(deformation_basis, axis=1))) <= 1.0e-6:
        # A perfectly straight demonstration has no detour direction to scale.
        # Use a deterministic, task-agnostic normal to its endpoint chord so
        # different warmup scales remain distinct without reading hidden truth.
        chord = states[-1] - states[0]
        chord_length = float(np.linalg.norm(chord))
        if chord_length > 1.0e-8:
            normal = np.asarray((-chord[1], chord[0]), dtype=np.float64) / chord_length
        else:
            normal = np.asarray((0.0, 1.0), dtype=np.float64)
        workspace_scale = min(
            float(workspace_x[1] - workspace_x[0]),
            float(workspace_y[1] - workspace_y[0]),
        )
        amplitude = max(0.10 * chord_length, 0.02 * workspace_scale, 1.0e-3)
        profile = np.sin(np.pi * np.linspace(0.0, 1.0, len(states)))[:, None]
        deformation_basis = amplitude * profile * normal[None, :]
        warmup_basis = "chord_normal_fallback"
    pair_index = index // 2
    # The default warmup cap is 50 queries (25 pairs).  Give every pair in
    # that range a distinct scale instead of cycling after ten pairs: an
    # exact repeat would spend another Oracle query without adding evidence
    # and could put duplicates in different evidence splits.
    scales = (
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.15,
        0.25,
        0.40,
        0.65,
        0.90,
        0.03,
        0.05,
        0.07,
        0.09,
        0.12,
        0.18,
        0.30,
        0.50,
        0.75,
        0.95,
        0.025,
        0.045,
        0.075,
        0.20,
        0.60,
    )
    if pair_index < len(scales):
        alpha = scales[pair_index]
    else:
        # Deterministic low-discrepancy continuation for non-default larger
        # caps.  Numeric model seeds still cannot alter the warmup bank.
        golden_ratio_conjugate = 0.6180339887498949
        alpha = 0.015 + 0.97 * ((pair_index * golden_ratio_conjugate) % 1.0)
    direction = -1.0 if index % 2 == 0 else 1.0
    if warmup_basis == "demonstrated_detour":
        direction_name = "toward_chord" if direction < 0.0 else "continue_detour"
    else:
        direction_name = (
            "negative_chord_normal" if direction < 0.0 else "positive_chord_normal"
        )
    candidate = states + direction * alpha * deformation_basis
    candidate[0] = states[0]
    candidate[-1] = states[-1]
    candidate[:, 0] = np.clip(candidate[:, 0], *workspace_x)
    candidate[:, 1] = np.clip(candidate[:, 1], *workspace_y)
    candidate = candidate.astype(np.float32)
    intervention_index = int(np.argmax(np.linalg.norm(candidate - states, axis=1)))
    return Trajectory(
        candidate,
        displacement_actions(candidate),
        metadata={
            "source": "warmup",
            "alpha": alpha,
            "warmup_scale_index": pair_index,
            "warmup_pair_index": pair_index,
            "warmup_direction": direction_name,
            "warmup_basis": warmup_basis,
            "signed_demo_relative_scale": direction * alpha,
            "expert_id": expert.metadata.get("trajectory_id"),
            "source_witness_index": intervention_index,
            "source_witness_kind": "intervention_max_deformation",
        },
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
        restart_index: int = 0,
        trust_radius: float | None = None,
    ) -> FalsifierResult:
        mode = self._numeric_mode(intervention.kind)
        if mode == "false_unsafe":
            effective_trust_radius = float(
                self.config.false_unsafe_trust_radius
                if trust_radius is None
                else trust_radius
            )
            if not np.isfinite(effective_trust_radius) or effective_trust_radius <= 0.0:
                raise ValueError("false-unsafe trust radius must be finite and strictly positive")
        else:
            effective_trust_radius = 0.0
        expert_tensor = torch.as_tensor(expert.states, dtype=torch.float32, device=self.device)
        initial = self._initial_path(expert_tensor, intervention, initialization_mix, restart_index)
        if mode == "false_unsafe" and effective_trust_radius > 0.0:
            initial = self._project_to_trust_region(
                initial,
                expert_tensor,
                effective_trust_radius,
            )
        interior = torch.nn.Parameter(initial[1:-1].clone())
        optimizer = torch.optim.Adam([interior], lr=self.config.learning_rate)
        previous = [parameter.requires_grad for parameter in ensemble.parameters()]
        for parameter in ensemble.parameters():
            parameter.requires_grad_(False)
        ensemble.eval()
        initial_loss = float("nan")
        final_loss = float("nan")
        initial_hard_score_gradient_norm: float | None = None
        initial_smooth_score_gradient_norm: float | None = None
        best_achieved_path: torch.Tensor | None = None
        best_achieved_deviation = float("inf")
        hard_margin_checkpoint_step: int | None = None
        rejected_hard_margin_checkpoints: dict[str, int] = {}

        def checkpoint_if_achieved(candidate_path: torch.Tensor, step: int) -> None:
            nonlocal best_achieved_path, best_achieved_deviation, hard_margin_checkpoint_step
            if mode != "false_unsafe" or not self.config.false_unsafe_use_hard_margin:
                return
            with torch.no_grad():
                checkpoint_features = self.library.torch_features(
                    candidate_path,
                    ensemble.compiled.variables,
                )
                target_score = float(
                    self._hard_crossing_score(
                        ensemble,
                        checkpoint_features,
                        intervention,
                    )
                    .detach()
                    .cpu()
                    .item()
                )
                full_score = float(
                    ensemble.mean_hard_trajectory_score(checkpoint_features)
                    .detach()
                    .cpu()
                    .item()
                )
                required_score = float(ensemble.decision_threshold.item()) + max(
                    0.0,
                    float(self.config.false_unsafe_hard_margin),
                )
                if min(full_score, target_score) < required_score:
                    return
                candidate_states = (
                    candidate_path.detach().cpu().numpy().astype(np.float32)
                )
                checkpoint_valid, checkpoint_reason = self.validate(
                    Trajectory(
                        candidate_states,
                        displacement_actions(candidate_states),
                        expert.dt,
                    ),
                    expert,
                )
                if not checkpoint_valid:
                    rejected_hard_margin_checkpoints[checkpoint_reason] = (
                        rejected_hard_margin_checkpoints.get(checkpoint_reason, 0) + 1
                    )
                    return
                deviation = float(
                    torch.max(
                        torch.linalg.vector_norm(candidate_path - expert_tensor, dim=-1)
                    )
                    .detach()
                    .cpu()
                    .item()
                )
                if deviation < best_achieved_deviation:
                    best_achieved_path = candidate_path.detach().clone()
                    best_achieved_deviation = deviation
                    hard_margin_checkpoint_step = step
        try:
            for step in range(self.config.steps):
                optimizer.zero_grad()
                path = torch.cat((expert_tensor[:1], interior, expert_tensor[-1:]), dim=0)
                loss, optimization_score, _ = self._objective(
                    path,
                    expert_tensor,
                    ensemble,
                    mode,
                    intervention,
                )
                if step == 0:
                    initial_loss = float(loss.detach().cpu().item())
                    if mode == "false_unsafe" and self.config.false_unsafe_use_hard_margin:
                        score_gradient = torch.autograd.grad(
                            optimization_score,
                            interior,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        initial_hard_score_gradient_norm = (
                            0.0
                            if score_gradient is None
                            else float(torch.linalg.vector_norm(score_gradient).detach().cpu().item())
                        )
                        smooth_features = self.library.torch_features(
                            path,
                            ensemble.compiled.variables,
                        ).unsqueeze(0)
                        smooth_score = self._smooth_crossing_score(
                            ensemble,
                            smooth_features,
                            intervention,
                        )
                        smooth_gradient = torch.autograd.grad(
                            smooth_score,
                            interior,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        initial_smooth_score_gradient_norm = (
                            0.0
                            if smooth_gradient is None
                            else float(
                                torch.linalg.vector_norm(smooth_gradient)
                                .detach()
                                .cpu()
                                .item()
                            )
                        )
                    checkpoint_if_achieved(path, step=0)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    interior[:, 0].clamp_(*self.workspace_x)
                    interior[:, 1].clamp_(*self.workspace_y)
                    if mode == "false_unsafe" and effective_trust_radius > 0.0:
                        interior.copy_(
                            self._project_to_trust_region(
                                interior,
                                expert_tensor[1:-1],
                                effective_trust_radius,
                            )
                        )
                    checkpoint_if_achieved(
                        torch.cat((expert_tensor[:1], interior, expert_tensor[-1:]), dim=0),
                        step=step + 1,
                    )
        finally:
            for parameter, required in zip(ensemble.parameters(), previous):
                parameter.requires_grad_(required)
        with torch.no_grad():
            path = (
                best_achieved_path
                if best_achieved_path is not None
                else torch.cat((expert_tensor[:1], interior, expert_tensor[-1:]), dim=0)
            )
            final_loss_tensor, score, uncertainty = self._objective(
                path,
                expert_tensor,
                ensemble,
                mode,
                intervention,
            )
            final_loss = float(final_loss_tensor.detach().cpu().item())
            path_features = self.library.torch_features(path, ensemble.compiled.variables)
            target_clause_index = self._target_clause_index(ensemble, intervention)
            witness_scores = (
                ensemble.mean_state_score(path_features)
                if target_clause_index is None
                else ensemble.mean_clause_state_scores(path_features)[..., target_clause_index]
            )
            source_witness_index = int(
                torch.argmax(witness_scores).item()
            )
            candidate_hard_score = float(
                ensemble.mean_hard_trajectory_score(path_features).detach().cpu().item()
            )
            candidate_target_hard_score = float(
                self._hard_crossing_score(
                    ensemble,
                    path_features,
                    intervention,
                )
                .detach()
                .cpu()
                .item()
            )
            candidate_smooth_score = float(
                ensemble.mean_trajectory_score(
                    path_features.unsqueeze(0),
                    self.config.smoothmax_beta,
                )
                .detach()
                .cpu()
                .item()
            )
            candidate_target_smooth_score = float(
                self._smooth_crossing_score(
                    ensemble,
                    path_features.unsqueeze(0),
                    intervention,
                )
                .detach()
                .cpu()
                .item()
            )
            expert_features = self.library.torch_features(
                expert_tensor,
                ensemble.compiled.variables,
            )
            expert_hard_score = float(
                ensemble.mean_hard_trajectory_score(expert_features).detach().cpu().item()
            )
            decision_threshold = float(ensemble.decision_threshold.detach().cpu().item())
        states = path.detach().cpu().numpy().astype(np.float32)
        hypothesis_id = ensemble.compiled.hypothesis.hypothesis_id
        maximum_deviation = float(np.max(np.linalg.norm(states - expert.states, axis=1)))
        metadata: dict[str, object] = {
            "source": intervention.kind,
            "source_hypothesis_id": hypothesis_id,
            "intervention_variable": intervention.variable,
            "expert_id": expert.metadata.get("trajectory_id"),
            "source_witness_index": source_witness_index,
            "source_witness_kind": (
                "target_clause_argmax_before_oracle"
                if target_clause_index is not None
                else "model_argmax_before_oracle"
            ),
            "trust_radius": effective_trust_radius if mode == "false_unsafe" else None,
            "max_expert_deviation": maximum_deviation,
        }
        if mode == "false_unsafe":
            hard_margin = max(0.0, float(self.config.false_unsafe_hard_margin))
            hard_margin_target = decision_threshold + hard_margin
            optimization_hard_margin_achieved = bool(
                candidate_hard_score >= hard_margin_target
                and candidate_target_hard_score >= hard_margin_target
            )
            metadata.update(
                {
                    "false_unsafe_hard_margin_enabled": bool(
                        self.config.false_unsafe_use_hard_margin
                    ),
                    "false_unsafe_hard_margin": hard_margin,
                    "source_decision_threshold": decision_threshold,
                    "source_expert_hard_score": expert_hard_score,
                    "source_expert_prediction": int(expert_hard_score > decision_threshold),
                    "source_candidate_hard_score": candidate_hard_score,
                    "source_candidate_target_hard_score": candidate_target_hard_score,
                    "source_candidate_smooth_score": candidate_smooth_score,
                    "source_candidate_target_smooth_score": candidate_target_smooth_score,
                    "optimization_target_clause_id": intervention.clause_id,
                    "optimization_hard_margin_target": hard_margin_target,
                    "optimization_hard_margin_achieved": optimization_hard_margin_achieved,
                    "optimization_hard_margin_deficit": max(
                        0.0,
                        hard_margin_target
                        - min(candidate_hard_score, candidate_target_hard_score),
                    ),
                    "hard_margin_checkpoint_step": hard_margin_checkpoint_step,
                    "rejected_hard_margin_checkpoints": dict(
                        rejected_hard_margin_checkpoints
                    ),
                    "initial_hard_score_gradient_norm": initial_hard_score_gradient_norm,
                    "initial_smooth_score_gradient_norm": initial_smooth_score_gradient_norm,
                    "falsifier_reachability_status": (
                        "hard_checkpoint_disabled"
                        if not self.config.false_unsafe_use_hard_margin
                        else (
                            "hard_margin_checkpoint"
                            if hard_margin_checkpoint_step is not None
                            else (
                                "zero_initial_crossing_gradient"
                                if initial_hard_score_gradient_norm is not None
                                and initial_smooth_score_gradient_norm is not None
                                and initial_hard_score_gradient_norm
                                <= max(0.0, float(self.config.false_unsafe_gradient_tolerance))
                                and initial_smooth_score_gradient_norm
                                <= max(0.0, float(self.config.false_unsafe_gradient_tolerance))
                                else "hard_margin_not_reached"
                            )
                        )
                    ),
                }
            )
        trajectory = Trajectory(
            states,
            displacement_actions(states),
            metadata=metadata,
        )
        valid, reason = self.validate(trajectory, expert)
        if mode == "false_unsafe" and effective_trust_radius > 0.0:
            if maximum_deviation > 1.01 * effective_trust_radius:
                valid, reason = False, "false_unsafe_trust_region"
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

    def false_unsafe_radii(self) -> tuple[float, ...]:
        """Return a sanitized ascending radius ladder capped by the trust region."""

        maximum = float(self.config.false_unsafe_trust_radius)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("false_unsafe_trust_radius must be finite and strictly positive")
        configured = tuple(float(value) for value in self.config.false_unsafe_radius_ladder)
        radii = sorted(
            {
                min(value, maximum)
                for value in configured
                if np.isfinite(value) and value > 0.0
            }
        )
        if not radii or not np.isclose(radii[-1], maximum):
            radii.append(maximum)
        return tuple(radii)

    def _initial_path(
        self,
        expert: torch.Tensor,
        intervention: InterventionSpec,
        mix: float,
        restart_index: int,
    ) -> torch.Tensor:
        time = torch.linspace(0.0, 1.0, len(expert), device=expert.device)[:, None]
        straight = expert[0][None, :] * (1.0 - time) + expert[-1][None, :] * time
        initial = (1.0 - mix) * expert + mix * straight
        # Smooth, task-agnostic basis restarts cover directions and frequencies.
        # The differentiable feature/model objective below decides which
        # deformation is useful; no feature name is mapped to a hand-coded axis.
        phase = time.squeeze(-1)
        frequency = 1 + restart_index % 3
        angle = (restart_index * 2.399963229728653) % (2.0 * torch.pi)
        direction = torch.stack((torch.cos(torch.as_tensor(angle)), torch.sin(torch.as_tensor(angle)))).to(
            device=expert.device, dtype=expert.dtype
        )
        basis = torch.sin(torch.pi * phase) * torch.sin(frequency * torch.pi * phase)
        amplitude = (0.03 + 0.04 * mix) * (1.5 if intervention.kind == "local_feature_stress" else 1.0)
        initial = initial + amplitude * basis[:, None] * direction[None, :]
        initial[0] = expert[0]
        initial[-1] = expert[-1]
        return initial

    @staticmethod
    def _project_to_trust_region(
        candidate: torch.Tensor,
        reference: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        if radius <= 0.0:
            return candidate
        delta = candidate - reference
        norms = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(float(radius) / torch.clamp(norms, min=1.0e-12), max=1.0)
        return reference + scale * delta

    @staticmethod
    def _numeric_mode(kind: str) -> str:
        if kind in {"model_false_safe", "shortcut"}:
            return "false_safe"
        if kind == "model_false_unsafe":
            return "false_unsafe"
        if kind == "local_feature_stress":
            return "feature_stress"
        return "boundary"

    @staticmethod
    def _target_clause_index(
        ensemble: ConstraintEnsemble,
        intervention: InterventionSpec,
    ) -> int | None:
        if intervention.clause_id is None:
            return None
        clause_ids = [clause.clause.clause_id for clause in ensemble.compiled.clauses]
        if intervention.clause_id not in clause_ids:
            raise ValueError(
                f"unknown clause_id {intervention.clause_id!r} for "
                f"{ensemble.compiled.hypothesis.hypothesis_id!r}"
            )
        return clause_ids.index(intervention.clause_id)

    def _hard_crossing_score(
        self,
        ensemble: ConstraintEnsemble,
        features: torch.Tensor,
        intervention: InterventionSpec,
    ) -> torch.Tensor:
        clause_index = self._target_clause_index(ensemble, intervention)
        if clause_index is None:
            return ensemble.mean_hard_trajectory_score(features).squeeze()
        scores = ensemble.mean_hard_clause_trajectory_scores(features)
        return scores[..., clause_index].squeeze()

    def _smooth_crossing_score(
        self,
        ensemble: ConstraintEnsemble,
        features: torch.Tensor,
        intervention: InterventionSpec,
    ) -> torch.Tensor:
        clause_index = self._target_clause_index(ensemble, intervention)
        if clause_index is None:
            return ensemble.mean_trajectory_score(
                features,
                self.config.smoothmax_beta,
            ).squeeze()
        scores = ensemble.mean_clause_trajectory_scores(
            features,
            self.config.smoothmax_beta,
        )
        return scores[..., clause_index].squeeze()

    def _objective(
        self,
        path: torch.Tensor,
        expert: torch.Tensor,
        ensemble: ConstraintEnsemble,
        mode: str,
        intervention: InterventionSpec,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.library.torch_features(path, ensemble.compiled.variables).unsqueeze(0)
        smooth_score = ensemble.mean_trajectory_score(
            features,
            self.config.smoothmax_beta,
        ).squeeze()
        score = smooth_score
        if intervention.clause_id is not None and mode != "false_unsafe":
            clause_ids = [clause.clause.clause_id for clause in ensemble.compiled.clauses]
            if intervention.clause_id in clause_ids:
                clause_scores = ensemble.mean_clause_trajectory_scores(features, self.config.smoothmax_beta)
                score = clause_scores[0, clause_ids.index(intervention.clause_id)]
        uncertainty = ensemble.trajectory_uncertainty(
            features,
            self.config.smoothmax_beta,
        ).squeeze()
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
            if self.config.false_unsafe_use_hard_margin:
                hard_score = self._hard_crossing_score(
                    ensemble,
                    features,
                    intervention,
                )
                target_smooth_score = self._smooth_crossing_score(
                    ensemble,
                    features,
                    intervention,
                )
                hard_uncertainty = ensemble.hard_trajectory_uncertainty(features).squeeze()
                threshold = ensemble.decision_threshold.to(dtype=path.dtype, device=path.device)
                target = threshold + max(0.0, float(self.config.false_unsafe_hard_margin))
                semantic = F.relu(target - hard_score).square()
                smooth_weight = max(
                    0.0,
                    float(self.config.false_unsafe_smooth_margin_weight),
                )
                if smooth_weight > 0.0:
                    semantic = semantic + smooth_weight * F.relu(
                        target - target_smooth_score
                    ).square()
                score = hard_score
                uncertainty = hard_uncertainty
            else:
                semantic = F.relu(self.config.epsilon - score).square()
            length_weight = 0.0
            expert_weight = 5.0 * self.config.expert_weight
        else:
            semantic = score.square()
            length_weight = 0.25 * self.config.length_weight
            expert_weight = self.config.expert_weight
        feature_stress = torch.zeros((), dtype=path.dtype, device=path.device)
        invariance = torch.zeros((), dtype=path.dtype, device=path.device)
        if mode == "feature_stress" and intervention.variable is not None:
            candidate_feature = self.library.torch_features(path, (intervention.variable,))[..., 0]
            expert_feature = self.library.torch_features(expert, (intervention.variable,))[..., 0]
            low, high = self.library.bounds((intervention.variable,))
            scale = max(high[0] - low[0], 1.0e-6)
            feature_stress = torch.mean(((candidate_feature - expert_feature) / scale).square()).clamp(max=4.0)
        if intervention.variable is not None:
            preserved = tuple(
                name
                for name in self.library.names
                if name not in {intervention.variable, "progress"}
            )
            if preserved:
                candidate_preserved = self.library.torch_features(path, preserved)
                expert_preserved = self.library.torch_features(expert, preserved)
                low, high = self.library.bounds(preserved)
                scales = torch.as_tensor(
                    np.maximum(np.asarray(high) - np.asarray(low), 1.0e-6),
                    dtype=path.dtype,
                    device=path.device,
                )
                invariance = torch.mean(((candidate_preserved - expert_preserved) / scales).square())
        uncertainty_term = (
            self.config.uncertainty_weight * uncertainty
            if mode == "false_unsafe" and self.config.false_unsafe_use_hard_margin
            else -self.config.uncertainty_weight * uncertainty
        )
        loss = (
            expert_weight * expert_distance
            + self.config.smoothness_weight * smoothness
            + length_weight * length
            + self.config.boundary_weight * semantic
            + uncertainty_term
            + self.config.step_penalty_weight * step_penalty
            + self.config.workspace_penalty_weight * workspace_penalty
            - self.config.feature_stress_weight * feature_stress
            + self.config.invariance_weight * invariance
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
