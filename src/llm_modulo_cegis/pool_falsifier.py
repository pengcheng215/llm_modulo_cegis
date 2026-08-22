"""Oracle-blind falsification from a public pool of valid global rollouts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .data import FeatureLibrary
from .falsifier import FalsifierConfig, FalsifierResult
from .learner import ConstraintEnsemble
from .types import InterventionSpec, Trajectory


class PoolHypothesisFalsifier:
    """Select model-informative rollouts without perturbing expert observations.

    Every pool member is constructed independently in control space and passes
    a public dynamics validator before learning starts.  Labels are absent from
    the pool and are obtained only if shared acquisition spends an Oracle query.
    """

    uses_validated_global_rollout_pool = True

    def __init__(
        self,
        library: FeatureLibrary,
        config: FalsifierConfig,
        candidates: list[Trajectory],
        device: torch.device,
        *,
        validator: Callable[[Trajectory], object] | None = None,
    ) -> None:
        if not candidates:
            raise ValueError("pool falsifier requires at least one candidate")
        shapes = {tuple(item.states.shape) for item in candidates}
        if len(shapes) != 1:
            raise ValueError("pool candidates must share one fixed observation shape")
        self.library = library
        self.config = config
        self.candidates = [item.copy() for item in candidates]
        self.device = device
        self.validator = validator
        self._warmup_indices: set[int] = set()

    @staticmethod
    def candidate_rank_offset(
        *,
        outer_round: int,
        pool_slot: int,
        pool_size: int,
        restarts: int,
    ) -> int:
        """Map a proposal slot to a fresh ranked-pool window.

        The continuous falsifier interprets restarts as independent optimizer
        initializations.  For a frozen public pool they instead address ranked
        candidates.  Advancing the window by round prevents a large requested
        proposal slate from repeatedly returning rank zero.
        """

        if outer_round < 1:
            raise ValueError("outer_round must be at least one")
        if pool_slot < 0:
            raise ValueError("pool_slot must be non-negative")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if restarts <= 0:
            raise ValueError("restarts must be positive")
        return ((outer_round - 1) * pool_size + pool_slot) * restarts

    def warmup_candidate(
        self,
        expert: Trajectory,
        index: int,
        rng: np.random.Generator,
    ) -> Trajectory:
        del rng
        if index >= len(self.candidates):
            raise RuntimeError("public candidate pool exhausted during warmup")
        self._warmup_indices.add(int(index))
        candidate = self.candidates[index].copy()
        pair_id = str(candidate.metadata.get("candidate_pair_id", f"pair_{index // 2:04d}"))
        member = int(candidate.metadata.get("candidate_pair_member", index % 2))
        difference = candidate.states - expert.states
        witness = int(np.argmax(np.linalg.norm(difference, axis=1)))
        candidate.metadata.update(
            {
                "source": "warmup_global_pool",
                "warmup_pair_index": index // 2,
                "warmup_scale_index": index // 2,
                "warmup_direction": f"unlabeled_member_{member}",
                "warmup_basis": "independent_control_space_rollout_pair",
                "alpha": float(index // 2 + 1),
                "expert_id": f"global_pool::{pair_id}",
                "source_witness_index": witness,
                "source_witness_kind": "maximum_observation_difference_before_oracle",
                "pool_candidate_id": candidate.metadata.get("trajectory_id"),
            }
        )
        return candidate

    def false_unsafe_radii(self) -> tuple[float, ...]:
        maximum = float(self.config.false_unsafe_trust_radius)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("false_unsafe_trust_radius must be finite and positive")
        return (maximum,)

    @staticmethod
    def _target_clause_index(
        ensemble: ConstraintEnsemble,
        intervention: InterventionSpec,
    ) -> int | None:
        if intervention.clause_id is None:
            return None
        clause_ids = [item.clause.clause_id for item in ensemble.compiled.clauses]
        if intervention.clause_id not in clause_ids:
            raise ValueError(
                f"unknown clause_id {intervention.clause_id!r} for "
                f"{ensemble.compiled.hypothesis.hypothesis_id!r}"
            )
        return clause_ids.index(intervention.clause_id)

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
        del initialization_mix
        states = torch.as_tensor(
            np.stack([item.states for item in self.candidates]),
            dtype=torch.float32,
            device=self.device,
        )
        features = self.library.torch_features(states, ensemble.compiled.variables)
        expert_features = self.library.torch_features(
            torch.as_tensor(expert.states, dtype=torch.float32, device=self.device),
            ensemble.compiled.variables,
        )
        clause_index = self._target_clause_index(ensemble, intervention)
        with torch.no_grad():
            hard_scores = ensemble.mean_hard_trajectory_score(features)
            smooth_scores = ensemble.mean_trajectory_score(
                features,
                self.config.smoothmax_beta,
            )
            uncertainties = ensemble.trajectory_uncertainty(
                features,
                self.config.smoothmax_beta,
            )
            if clause_index is None:
                target_scores = hard_scores
            else:
                target_scores = ensemble.mean_hard_clause_trajectory_scores(features)[
                    :, clause_index
                ]
            low, high = self.library.bounds(ensemble.compiled.variables)
            scale = torch.as_tensor(
                np.maximum(np.asarray(high) - np.asarray(low), 1.0e-6),
                dtype=features.dtype,
                device=features.device,
            )
            feature_distance = torch.mean(
                ((features - expert_features.unsqueeze(0)) / scale).square(),
                dim=(1, 2),
            )
        hard = hard_scores.detach().cpu().numpy()
        target = target_scores.detach().cpu().numpy()
        smooth = smooth_scores.detach().cpu().numpy()
        uncertainty = uncertainties.detach().cpu().numpy()
        distance = feature_distance.detach().cpu().numpy()
        threshold = float(ensemble.decision_threshold.item())
        mode = intervention.kind
        if mode == "model_false_unsafe":
            required = threshold + max(0.0, float(self.config.false_unsafe_hard_margin))
            eligible = np.flatnonzero((hard >= required) & (target >= required))
            objective = distance + 0.25 * np.abs(target - required) - 0.10 * uncertainty
        elif mode in {"model_false_safe", "shortcut"}:
            eligible = np.flatnonzero(hard <= threshold)
            distance_weight = 0.35 if mode == "shortcut" else 0.15
            objective = (
                np.abs(hard - threshold)
                - 0.20 * uncertainty
                - distance_weight * np.sqrt(np.maximum(distance, 0.0))
            )
        elif mode == "local_feature_stress" and intervention.variable is not None:
            candidate_feature = self.library.torch_features(
                states,
                (intervention.variable,),
            )[..., 0]
            expert_feature = self.library.torch_features(
                torch.as_tensor(expert.states, dtype=torch.float32, device=self.device),
                (intervention.variable,),
            )[..., 0]
            stress = torch.mean(
                (candidate_feature - expert_feature.unsqueeze(0)).square(),
                dim=1,
            ).detach().cpu().numpy()
            eligible = np.arange(len(self.candidates))
            objective = -stress + 0.10 * np.abs(hard - threshold)
        else:
            eligible = np.arange(len(self.candidates))
            objective = np.abs(hard - threshold) - 0.25 * uncertainty
        if self._warmup_indices:
            eligible = np.asarray(
                [index for index in eligible if int(index) not in self._warmup_indices],
                dtype=np.int64,
            )
        if len(eligible) == 0:
            eligible = np.asarray(
                [
                    index
                    for index in range(len(self.candidates))
                    if index not in self._warmup_indices
                ],
                dtype=np.int64,
            )
        if len(eligible) == 0:
            raise RuntimeError("public candidate pool was exhausted by warmup")
        ordered = eligible[np.argsort(objective[eligible], kind="mergesort")]
        requested_rank = int(restart_index)
        selected_rank = requested_rank % len(ordered)
        selected_index = int(ordered[selected_rank])
        selected = self.candidates[selected_index].copy()
        selected_features = features[selected_index]
        with torch.no_grad():
            if clause_index is None:
                witness_scores = ensemble.mean_state_score(selected_features)
            else:
                witness_scores = ensemble.mean_clause_state_scores(selected_features)[
                    :, clause_index
                ]
        witness_index = int(torch.argmax(witness_scores).item())
        expert_hard_score = float(
            ensemble.mean_hard_trajectory_score(expert_features).detach().cpu().item()
        )
        hard_margin_target = threshold + max(
            0.0,
            float(self.config.false_unsafe_hard_margin),
        )
        hard_margin_achieved = bool(
            hard[selected_index] >= hard_margin_target
            and target[selected_index] >= hard_margin_target
        )
        selected.metadata.update(
            {
                "source": intervention.kind,
                "source_hypothesis_id": ensemble.compiled.hypothesis.hypothesis_id,
                "intervention_variable": intervention.variable,
                "optimization_target_clause_id": intervention.clause_id,
                "expert_id": expert.metadata.get("trajectory_id"),
                "pool_candidate_id": selected.metadata.get("trajectory_id"),
                "pool_selection_requested_rank": requested_rank,
                "pool_selection_rank": selected_rank,
                "source_witness_index": witness_index,
                "source_witness_kind": "pool_model_argmax_before_oracle",
                "source_expert_hard_score": expert_hard_score,
                "source_expert_prediction": int(expert_hard_score > threshold),
                "source_decision_threshold": threshold,
                "source_candidate_hard_score": float(hard[selected_index]),
                "source_candidate_target_hard_score": float(target[selected_index]),
                "source_candidate_smooth_score": float(smooth[selected_index]),
                "source_candidate_target_smooth_score": float(smooth[selected_index]),
                "optimization_hard_margin_target": hard_margin_target,
                "optimization_hard_margin_achieved": hard_margin_achieved,
                "hard_margin_checkpoint_step": 0 if hard_margin_achieved else None,
                "rejected_hard_margin_checkpoints": {},
                "initial_hard_score_gradient_norm": None,
                "initial_smooth_score_gradient_norm": None,
                "falsifier_reachability_status": (
                    "public_pool_hard_margin_candidate"
                    if hard_margin_achieved
                    else "public_pool_no_hard_margin_candidate"
                ),
                # The planar trust radius has units of metres and is not a
                # valid distance bound for a globally generated 12-D rollout.
                # Keep the requested rung only as an audit field; acquisition
                # uses the normalized public-feature distance below.
                "trust_radius": None,
                "requested_planar_radius_ignored": trust_radius,
                "normalized_expert_deformation": float(
                    np.sqrt(max(float(distance[selected_index]), 0.0))
                ),
                "max_expert_deviation": float(
                    np.max(np.linalg.norm(selected.states - expert.states, axis=1))
                ),
            }
        )
        valid, reason = self.validate(selected, expert)
        return FalsifierResult(
            trajectory=selected,
            mode=mode,
            hypothesis_id=ensemble.compiled.hypothesis.hypothesis_id,
            initial_loss=float(objective[selected_index]),
            final_loss=float(objective[selected_index]),
            final_score=float(smooth[selected_index]),
            final_uncertainty=float(uncertainty[selected_index]),
            valid=valid,
            validation_reason=reason,
        )

    def validate(self, trajectory: Trajectory, expert: Trajectory) -> tuple[bool, str]:
        del expert
        if self.validator is None:
            return True, "valid"
        result = self.validator(trajectory)
        valid = bool(getattr(result, "valid", False))
        reason = str(getattr(result, "reason", "invalid"))
        return valid, reason


__all__ = ["PoolHypothesisFalsifier"]
