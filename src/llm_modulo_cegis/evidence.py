"""Convert numeric learner behavior into leakage-safe semantic evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import FeatureLibrary
from .learner import ConstraintEnsemble, LearnerRegistry
from .types import HypothesisEvidence, QueryRecord, SAFE_LABEL, Trajectory, VIOLATION_LABEL


@dataclass(frozen=True)
class EvidenceConfig:
    minimum_per_label: int = 3
    complexity_penalty: float = 0.015
    uncertainty_penalty: float = 0.02


class EvidenceCompiler:
    """The only numeric-to-language bridge used by the LLM."""

    def __init__(
        self,
        library: FeatureLibrary,
        config: EvidenceConfig,
        device: torch.device,
    ) -> None:
        self.library = library
        self.config = config
        self.device = device

    def compile(
        self,
        registry: LearnerRegistry,
        active_ids: list[str],
        experts: list[Trajectory],
        records: list[QueryRecord],
    ) -> list[HypothesisEvidence]:
        return [self._one(registry.models[hypothesis_id], experts, records) for hypothesis_id in active_ids]

    def _one(
        self,
        ensemble: ConstraintEnsemble,
        experts: list[Trajectory],
        records: list[QueryRecord],
    ) -> HypothesisEvidence:
        hypothesis_id = ensemble.compiled.hypothesis.hypothesis_id
        labels: list[int] = []
        predictions: list[int] = []
        margins: list[float] = []
        uncertainties: list[float] = []
        for record in records:
            features = self.library.torch_features(
                torch.as_tensor(record.trajectory.states, dtype=torch.float32, device=self.device),
                ensemble.compiled.variables,
            ).unsqueeze(0)
            with torch.no_grad():
                score = float(ensemble.mean_trajectory_score(features, beta=20.0).item())
                uncertainty = float(ensemble.trajectory_uncertainty(features, beta=20.0).item())
            labels.append(record.label)
            predictions.append(int(score > 0.0))
            margins.append(abs(score))
            uncertainties.append(uncertainty)
        labels_array = np.asarray(labels, dtype=np.int64)
        predictions_array = np.asarray(predictions, dtype=np.int64)
        safe_mask = labels_array == SAFE_LABEL
        violation_mask = labels_array == VIOLATION_LABEL
        safe_accuracy = float(np.mean(predictions_array[safe_mask] == SAFE_LABEL)) if np.any(safe_mask) else 0.0
        violation_recall = (
            float(np.mean(predictions_array[violation_mask] == VIOLATION_LABEL)) if np.any(violation_mask) else 0.0
        )
        balanced_accuracy = 0.5 * (safe_accuracy + violation_recall)
        false_safe = int(np.sum(violation_mask & (predictions_array == SAFE_LABEL)))
        false_unsafe = int(np.sum(safe_mask & (predictions_array == VIOLATION_LABEL)))
        expert_safe_predictions: list[bool] = []
        for expert in experts:
            features = self.library.torch_features(
                torch.as_tensor(expert.states, dtype=torch.float32, device=self.device),
                ensemble.compiled.variables,
            )
            expert_safe_predictions.append(ensemble.predict_features(features) == SAFE_LABEL)
        expert_safe_rate = float(np.mean(expert_safe_predictions))
        sourced = [record for record in records if record.source_hypothesis_id == hypothesis_id]
        intervention_yield = float(np.mean([record.label == VIOLATION_LABEL for record in sourced])) if sourced else 0.0
        counterexample_rate = float(np.mean(labels_array != predictions_array)) if len(records) else 1.0
        complexity = len(ensemble.compiled.variables) + int(ensemble.compiled.hypothesis.coupling == "joint")
        mean_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.0
        selection_score = (
            0.42 * balanced_accuracy
            + 0.18 * expert_safe_rate
            + 0.15 * violation_recall
            + 0.10 * safe_accuracy
            + 0.15 * intervention_yield
            - self.config.complexity_penalty * complexity
            - self.config.uncertainty_penalty * mean_uncertainty
        )
        sufficient = int(np.sum(safe_mask)) >= self.config.minimum_per_label and int(np.sum(violation_mask)) >= self.config.minimum_per_label
        return HypothesisEvidence(
            hypothesis_id=hypothesis_id,
            balanced_accuracy=balanced_accuracy,
            safe_accuracy=safe_accuracy,
            violation_recall=violation_recall,
            expert_safe_rate=expert_safe_rate,
            counterexample_rate=counterexample_rate,
            false_safe_count=false_safe,
            false_unsafe_count=false_unsafe,
            mean_abs_margin=float(np.mean(margins)) if margins else 0.0,
            mean_uncertainty=mean_uncertainty,
            intervention_violation_yield=intervention_yield,
            intervention_count=len(sourced),
            complexity=complexity,
            selection_score=float(selection_score),
            evidence_sufficient=sufficient,
        )


def evidence_report(
    outer_round: int,
    evidence: list[HypothesisEvidence],
    label_counts: dict[str, int],
) -> dict[str, object]:
    ordered = sorted(evidence, key=lambda item: item.selection_score, reverse=True)
    return {
        "outer_round": outer_round,
        "trajectory_label_counts": label_counts,
        "ranking": [item.hypothesis_id for item in ordered],
        "hypotheses": [item.to_dict() for item in ordered],
        "important_note": (
            "All metrics come from trajectory-level labels and model behavior. "
            "No obstacle center, radius, state label, or evaluation IoU is included."
        ),
    }
