"""Convert numeric learner behavior into leakage-safe semantic evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import torch

from .data import FeatureLibrary
from .learner import ConstraintEnsemble, LearnerRegistry
from .types import HypothesisEvidence, QueryRecord, SAFE_LABEL, Trajectory, VIOLATION_LABEL


@dataclass(frozen=True)
class EvidenceConfig:
    minimum_per_label: int = 3
    complexity_penalty: float = 0.015
    capacity_penalty: float = 0.012
    uncertainty_penalty: float = 0.02
    minimum_class_accuracy: float = 0.20
    degenerate_predictor_penalty: float = 0.20
    champion_minimum_safe_accuracy: float = 0.60
    champion_minimum_violation_recall: float = 0.60
    champion_minimum_expert_safe_rate: float = 0.90
    champion_minimum_fit_expert_safe_rate: float = 0.90
    prequential_window_rounds: int = 0
    nested_minimum_balanced_accuracy_gain: float = 0.08
    progress_proxy_minimum_balanced_accuracy_gain: float = 0.08
    dynamics_proxy_minimum_balanced_accuracy_gain: float = 0.20
    intervention_yield_selection_weight: float = 0.03
    representation_collision_tolerance: float = 1.0e-5
    representation_collision_penalty: float = 1.0
    linear_max_support_gate_enforced: bool = True
    linear_max_support_tolerance: float = 1.0e-5
    linear_max_minimum_contradictory_anchors: int = 2


def _cross_2d(origin: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    return float(
        (left[0] - origin[0]) * (right[1] - origin[1])
        - (left[1] - origin[1]) * (right[0] - origin[0])
    )


def _convex_hull_2d(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Return a CCW hull after tolerance-aware point deduplication."""

    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]
    unique: list[np.ndarray] = []
    for point in ordered:
        if not unique or all(
            float(np.linalg.norm(point - previous)) > tolerance
            for previous in unique
        ):
            unique.append(point)
    if len(unique) <= 2:
        return np.asarray(unique, dtype=np.float64)

    def half(sequence: list[np.ndarray]) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for point in sequence:
            while len(result) >= 2 and _cross_2d(result[-2], result[-1], point) <= 0.0:
                result.pop()
            result.append(point)
        return result

    lower = half(unique)
    upper = half(list(reversed(unique)))
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _points_inside_convex_hull(
    anchor_points: np.ndarray,
    candidate_points: np.ndarray,
    tolerance: float,
) -> bool:
    """Check finite 1D/2D sampled-point containment without SciPy.

    This is deliberately a sufficient test.  Unsupported dimensions return
    ``False`` rather than turning an approximate optimizer into a hard
    structural rejection.
    """

    anchor = np.asarray(anchor_points, dtype=np.float64)
    candidate = np.asarray(candidate_points, dtype=np.float64)
    if (
        anchor.ndim != 2
        or candidate.ndim != 2
        or anchor.shape[1] != candidate.shape[1]
        or anchor.shape[1] not in (1, 2)
        or len(anchor) == 0
        or len(candidate) == 0
        or not np.all(np.isfinite(anchor))
        or not np.all(np.isfinite(candidate))
    ):
        return False
    if anchor.shape[1] == 1:
        # A hard structural certificate must never expand the hull: affine
        # weights are unbounded, so an arbitrarily small outward displacement
        # can still be separated.  ``tolerance`` is used only to deduplicate
        # anchor points while constructing a conservative (possibly smaller)
        # hull; numerical uncertainty therefore produces a false negative,
        # never a false structural rejection.
        lower = float(np.min(anchor[:, 0]))
        upper = float(np.max(anchor[:, 0]))
        return bool(np.all(candidate[:, 0] >= lower) and np.all(candidate[:, 0] <= upper))

    hull = _convex_hull_2d(anchor, tolerance)
    if len(hull) == 0:
        return False
    if len(hull) == 1:
        return bool(np.all(candidate == hull[0]))
    if len(hull) == 2:
        edge = hull[1] - hull[0]
        length = float(np.linalg.norm(edge))
        if length <= 0.0:
            return bool(np.all(candidate == hull[0]))
        offsets = candidate - hull[0]
        along = offsets @ edge / length
        perpendicular = np.abs(edge[0] * offsets[:, 1] - edge[1] * offsets[:, 0]) / length
        return bool(
            np.all(along >= 0.0)
            and np.all(along <= length)
            and np.all(perpendicular == 0.0)
        )

    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 0.0:
            continue
        offsets = candidate - start
        signed_distance = (
            edge[0] * offsets[:, 1] - edge[1] * offsets[:, 0]
        ) / length
        if np.any(signed_distance < 0.0):
            return False
    return True


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
        if (
            not np.isfinite(config.linear_max_support_tolerance)
            or config.linear_max_support_tolerance < 0.0
        ):
            raise ValueError("linear_max_support_tolerance must be finite and nonnegative")
        if not isinstance(config.linear_max_support_gate_enforced, (bool, np.bool_)):
            raise ValueError("linear_max_support_gate_enforced must be boolean")
        if (
            isinstance(config.linear_max_minimum_contradictory_anchors, (bool, np.bool_))
            or not isinstance(config.linear_max_minimum_contradictory_anchors, Integral)
            or config.linear_max_minimum_contradictory_anchors < 1
        ):
            raise ValueError(
                "linear_max_minimum_contradictory_anchors must be a positive integer"
            )

    def compile(
        self,
        registry: LearnerRegistry,
        active_ids: list[str],
        experts: list[Trajectory],
        records: list[QueryRecord],
        fit_experts: list[Trajectory] | None = None,
    ) -> list[HypothesisEvidence]:
        evidence = [
            self._one(registry.models[hypothesis_id], experts, records, fit_experts or [])
            for hypothesis_id in active_ids
        ]
        known_safe_experts = [*experts, *(fit_experts or [])]
        evidence = self._apply_representation_collision_gates(
            registry,
            evidence,
            records,
            known_safe_experts,
        )
        evidence = self._apply_linear_max_support_order_gates(
            registry,
            evidence,
            records,
            known_safe_experts,
        )
        return self._apply_nested_minimality(registry, evidence)

    def _apply_linear_max_support_order_gates(
        self,
        registry: LearnerRegistry,
        evidence: list[HypothesisEvidence],
        records: list[QueryRecord],
        known_safe_experts: list[Trajectory],
    ) -> list[HypothesisEvidence]:
        """Reject an affine-max structure contradicted by convex support order.

        For an affine state score ``g``, if every candidate feature point lies
        in the convex hull of a known-safe anchor, then
        ``max_candidate g <= max_anchor g`` for every possible parameter
        vector.  The same threshold therefore cannot accept the anchor and
        reject an Oracle-violating candidate.  Requiring two distinct anchors
        protects the hard gate against one mislabeled trajectory while keeping
        the statement parameter- and private-geometry-independent.
        """

        supported_ids = {
            item.hypothesis_id
            for item in evidence
            if len(registry.models[item.hypothesis_id].compiled.hypothesis.atomic_clauses()) == 1
            and (
                lambda clause: clause.coupling == "joint"
                and clause.relation == "forbidden_region"
                and clause.model_family == "linear"
                and clause.temporal_operator == "max"
                and len(clause.variables) in (1, 2)
            )(
                registry.models[item.hypothesis_id]
                .compiled.hypothesis.atomic_clauses()[0]
            )
        }
        if not supported_ids:
            return evidence

        experts_by_id: dict[str, Trajectory] = {}
        for expert in known_safe_experts:
            trajectory_id = expert.metadata.get("trajectory_id")
            if trajectory_id is None:
                continue
            key = str(trajectory_id)
            if key in experts_by_id:
                raise ValueError(
                    "linear-max support gate requires unique expert trajectory_id values"
                )
            experts_by_id[key] = expert

        tolerance = float(self.config.linear_max_support_tolerance)
        minimum_anchors = int(
            self.config.linear_max_minimum_contradictory_anchors
        )
        updated: list[HypothesisEvidence] = []
        for item in evidence:
            if item.hypothesis_id not in supported_ids:
                updated.append(item)
                continue
            clause = (
                registry.models[item.hypothesis_id]
                .compiled.hypothesis.atomic_clauses()[0]
            )
            low, high = self.library.bounds(clause.variables)
            low_array = np.asarray(low, dtype=np.float64)
            scale = np.maximum(
                np.asarray(high, dtype=np.float64) - low_array,
                1.0e-12,
            )
            pair_count = 0
            contradiction_count = 0
            unresolved_count = 0
            contradictory_anchors: set[str] = set()
            for record in records:
                if record.label != VIOLATION_LABEL or record.source == "final_calibration":
                    continue
                expert_id = record.trajectory.metadata.get("expert_id")
                anchor = (
                    experts_by_id.get(str(expert_id))
                    if expert_id is not None
                    else None
                )
                if anchor is None:
                    unresolved_count += 1
                    continue
                pair_count += 1
                anchor_features = self.library.numpy_features(
                    anchor.states, clause.variables
                ).astype(np.float64)
                candidate_features = self.library.numpy_features(
                    record.trajectory.states, clause.variables
                ).astype(np.float64)
                anchor_normalized = 2.0 * (anchor_features - low_array) / scale - 1.0
                candidate_normalized = (
                    2.0 * (candidate_features - low_array) / scale - 1.0
                )
                if _points_inside_convex_hull(
                    anchor_normalized,
                    candidate_normalized,
                    tolerance,
                ):
                    contradiction_count += 1
                    contradictory_anchors.add(str(expert_id))

            distinct_anchor_count = len(contradictory_anchors)
            gate_triggered = distinct_anchor_count >= minimum_anchors
            gate_applied = bool(
                self.config.linear_max_support_gate_enforced and gate_triggered
            )
            reasons = list(item.ineligibility_reasons)
            champion_eligible = item.champion_eligible
            if gate_applied:
                reasons.append("linear_max_support_order_contradicted_by_oracle")
                champion_eligible = False
            updated.append(
                replace(
                    item,
                    champion_eligible=champion_eligible,
                    ineligibility_reasons=tuple(dict.fromkeys(reasons)),
                    linear_max_support_pair_count=pair_count,
                    linear_max_support_contradiction_count=contradiction_count,
                    linear_max_support_distinct_anchor_count=distinct_anchor_count,
                    linear_max_support_unresolved_pair_count=unresolved_count,
                    linear_max_support_gate_triggered=gate_triggered,
                    linear_max_support_gate_applied=gate_applied,
                )
            )
        return updated

    def _apply_representation_collision_gates(
        self,
        registry: LearnerRegistry,
        evidence: list[HypothesisEvidence],
        records: list[QueryRecord],
        known_safe_experts: list[Trajectory],
    ) -> list[HypothesisEvidence]:
        """Reject structures that map opposite labels to one representation.

        For a pure ``last`` hypothesis, the selected terminal feature vector is
        a sufficient statistic: no learner using that structure can distinguish
        two trajectories with the same vector.  A known-safe expert and an
        Oracle-violating endpoint-preserving counterpart therefore constitute a
        logical structure counterexample, independent of neural fit quality.
        """

        tolerance = max(float(self.config.representation_collision_tolerance), 1.0e-12)
        updated: list[HypothesisEvidence] = []
        usable_records = [record for record in records if record.source != "final_calibration"]
        for item in evidence:
            hypothesis = registry.models[item.hypothesis_id].compiled.hypothesis
            if not all(clause.temporal_operator == "last" for clause in hypothesis.atomic_clauses()):
                updated.append(item)
                continue
            low, high = self.library.bounds(hypothesis.variables)
            low_array = np.asarray(low, dtype=np.float64)
            scale = np.maximum(np.asarray(high, dtype=np.float64) - low_array, 1.0e-12)
            samples: list[tuple[np.ndarray, int]] = []
            for expert in known_safe_experts:
                terminal = self.library.numpy_features(expert.states, hypothesis.variables)[-1]
                samples.append(((terminal.astype(np.float64) - low_array) / scale, SAFE_LABEL))
            for record in usable_records:
                terminal = self.library.numpy_features(record.trajectory.states, hypothesis.variables)[-1]
                samples.append(((terminal.astype(np.float64) - low_array) / scale, record.label))

            groups: list[dict[str, object]] = []
            for representation, label in samples:
                group = next(
                    (
                        candidate
                        for candidate in groups
                        if float(np.max(np.abs(representation - candidate["representative"]))) <= tolerance
                    ),
                    None,
                )
                if group is None:
                    groups.append({"representative": representation, "labels": {int(label)}})
                else:
                    group["labels"].add(int(label))
            contradictory = sum(
                SAFE_LABEL in group["labels"] and VIOLATION_LABEL in group["labels"]
                for group in groups
            )
            if contradictory <= 0:
                updated.append(
                    replace(
                        item,
                        representation_group_count=len(groups),
                        contradictory_representation_group_count=0,
                    )
                )
                continue
            reason = "terminal_invariance_contradicted_by_oracle"
            reasons = tuple(dict.fromkeys([*item.ineligibility_reasons, reason]))
            penalty = float(self.config.representation_collision_penalty)
            updated.append(
                replace(
                    item,
                    selection_score=item.selection_score - penalty,
                    query_priority=item.query_priority - penalty,
                    champion_eligible=False,
                    ineligibility_reasons=reasons,
                    representation_group_count=len(groups),
                    contradictory_representation_group_count=int(contradictory),
                )
            )
        return updated

    def _apply_nested_minimality(
        self,
        registry: LearnerRegistry,
        evidence: list[HypothesisEvidence],
    ) -> list[HypothesisEvidence]:
        """Stop a strict feature superset from winning without material evidence.

        An ``(x, y, progress)`` MLP contains the ``(x, y)`` MLP as a special
        case.  Treating the former as a distinct explanation without demanding
        an out-of-sample gain makes the hypothesis search collapse toward the
        largest feature set.  This gate is deliberately structural and never
        inspects the private evaluation geometry.
        """

        by_id = {item.hypothesis_id: item for item in evidence}
        feature_groups = {spec.name: spec.group for spec in self.library.specs}
        updated: list[HypothesisEvidence] = []
        for item in evidence:
            hypothesis = registry.models[item.hypothesis_id].compiled.hypothesis
            if len(hypothesis.atomic_clauses()) != 1:
                updated.append(item)
                continue
            variables = set(hypothesis.variables)
            dominated_by: list[str] = []
            progress_proxy_for: list[str] = []
            dynamics_proxy_for: list[str] = []
            for simpler_id, simpler_item in by_id.items():
                if (
                    simpler_id == item.hypothesis_id
                    or not simpler_item.evidence_sufficient
                    or not simpler_item.champion_eligible
                ):
                    continue
                simpler = registry.models[simpler_id].compiled.hypothesis
                if len(simpler.atomic_clauses()) != 1:
                    continue
                structurally_nested = (
                    set(simpler.variables) < variables
                    and simpler.coupling == hypothesis.coupling
                    and simpler.relation == hypothesis.relation
                    and simpler.temporal_operator == hypothesis.temporal_operator
                    and simpler.model_family == hypothesis.model_family
                )
                material_gain = (
                    item.balanced_accuracy - simpler_item.balanced_accuracy
                    >= self.config.nested_minimum_balanced_accuracy_gain
                )
                if structurally_nested and not material_gain:
                    dominated_by.append(simpler_id)
                hypothesis_progress = {
                    variable for variable in variables if feature_groups.get(variable) == "time"
                }
                simpler_progress = {
                    variable for variable in simpler.variables if feature_groups.get(variable) == "time"
                }
                physical_overlap = (variables - hypothesis_progress) & set(simpler.variables)
                progress_proxy = (
                    bool(hypothesis_progress)
                    and not simpler_progress
                    and bool(physical_overlap)
                    and simpler.coupling == hypothesis.coupling
                    and simpler.relation == hypothesis.relation
                    and simpler.temporal_operator == hypothesis.temporal_operator
                    and simpler.model_family == hypothesis.model_family
                )
                progress_material_gain = (
                    item.balanced_accuracy - simpler_item.balanced_accuracy
                    >= self.config.progress_proxy_minimum_balanced_accuracy_gain
                )
                if progress_proxy and not progress_material_gain:
                    progress_proxy_for.append(simpler_id)
                hypothesis_groups = {feature_groups.get(variable) for variable in variables}
                simpler_groups = {
                    feature_groups.get(variable) for variable in simpler.variables
                }
                dynamics_proxy = (
                    hypothesis_groups == {"velocity"}
                    and simpler_groups == {"position"}
                    and simpler.coupling == hypothesis.coupling
                    and simpler.relation == hypothesis.relation
                    and simpler.temporal_operator == hypothesis.temporal_operator
                    and simpler.model_family == hypothesis.model_family
                )
                dynamics_material_gain = (
                    item.balanced_accuracy - simpler_item.balanced_accuracy
                    >= self.config.dynamics_proxy_minimum_balanced_accuracy_gain
                )
                if dynamics_proxy and not dynamics_material_gain:
                    dynamics_proxy_for.append(simpler_id)
            if not dominated_by and not progress_proxy_for and not dynamics_proxy_for:
                updated.append(item)
                continue
            new_reasons: list[str] = list(item.ineligibility_reasons)
            if dominated_by:
                new_reasons.append("nested_without_material_evidence_gain")
            if progress_proxy_for:
                new_reasons.append("task_progress_proxy_without_material_evidence_gain")
            if dynamics_proxy_for:
                new_reasons.append("derived_dynamics_proxy_without_material_evidence_gain")
            reasons = tuple(dict.fromkeys(new_reasons))
            updated.append(
                replace(
                    item,
                    champion_eligible=False,
                    ineligibility_reasons=reasons,
                )
            )
        return updated

    def _one(
        self,
        ensemble: ConstraintEnsemble,
        experts: list[Trajectory],
        records: list[QueryRecord],
        fit_experts: list[Trajectory],
    ) -> HypothesisEvidence:
        hypothesis_id = ensemble.compiled.hypothesis.hypothesis_id
        # Use only predictions made before the Oracle label was revealed, plus
        # the explicitly held-out warmup audit split whose labels never enter
        # gradient fitting.  The previous implementation re-scored training
        # queries after fitting, systematically favoring high-capacity MLPs.
        labels: list[int] = []
        predictions: list[int] = []
        margins: list[float] = []
        uncertainties: list[float] = []
        latest_outer_round = max((record.outer_round for record in records), default=0)
        if self.config.prequential_window_rounds > 0:
            earliest_outer_round = max(1, latest_outer_round - self.config.prequential_window_rounds + 1)
        else:
            earliest_outer_round = 0
        for record in records:
            if (
                self.config.prequential_window_rounds > 0
                and record.source != "warmup_validation"
                and record.outer_round < earliest_outer_round
            ):
                continue
            if hypothesis_id not in record.predictions_before_query:
                continue
            labels.append(record.label)
            predictions.append(int(record.predictions_before_query[hypothesis_id]))
            margins.append(abs(float(record.scores_before_query.get(hypothesis_id, 0.0))))
            uncertainties.append(float(record.uncertainties_before_query.get(hypothesis_id, 0.0)))
        expert_safe_predictions: list[bool] = []
        for expert in experts:
            features = self.library.torch_features(
                torch.as_tensor(expert.states, dtype=torch.float32, device=self.device),
                ensemble.compiled.variables,
            ).unsqueeze(0)
            with torch.no_grad():
                score = float(ensemble.mean_hard_trajectory_score(features).item())
                uncertainty = float(ensemble.hard_trajectory_uncertainty(features).item())
            prediction = int(score > float(ensemble.decision_threshold.item()))
            expert_safe_predictions.append(prediction == SAFE_LABEL)
            labels.append(SAFE_LABEL)
            predictions.append(prediction)
            margins.append(abs(score))
            uncertainties.append(uncertainty)
        labels_array = np.asarray(labels, dtype=np.int64)
        predictions_array = np.asarray(predictions, dtype=np.int64)
        safe_mask = labels_array == SAFE_LABEL
        violation_mask = labels_array == VIOLATION_LABEL
        safe_accuracy = float(np.mean(predictions_array[safe_mask] == SAFE_LABEL)) if np.any(safe_mask) else 0.5
        violation_recall = (
            float(np.mean(predictions_array[violation_mask] == VIOLATION_LABEL)) if np.any(violation_mask) else 0.5
        )
        balanced_accuracy = 0.5 * (safe_accuracy + violation_recall)
        false_safe = int(np.sum(violation_mask & (predictions_array == SAFE_LABEL)))
        false_unsafe = int(np.sum(safe_mask & (predictions_array == VIOLATION_LABEL)))
        expert_safe_rate = float(np.mean(expert_safe_predictions)) if expert_safe_predictions else safe_accuracy
        fit_expert_predictions: list[bool] = []
        for expert in fit_experts:
            features = self.library.torch_features(
                torch.as_tensor(expert.states, dtype=torch.float32, device=self.device),
                ensemble.compiled.variables,
            )
            with torch.no_grad():
                fit_expert_predictions.append(ensemble.predict_features(features) == SAFE_LABEL)
        fit_expert_safe_rate = (
            float(np.mean(fit_expert_predictions)) if fit_expert_predictions else expert_safe_rate
        )
        sourced = [record for record in records if record.source_hypothesis_id == hypothesis_id]
        intervention_yield = float(np.mean([record.label == VIOLATION_LABEL for record in sourced])) if sourced else 0.0
        counterexample_rate = float(np.mean(labels_array != predictions_array)) if len(labels) else 1.0
        clauses = ensemble.compiled.hypothesis.atomic_clauses()
        complexity = sum(
            1
            + len(clause.variables)
            + int(clause.coupling == "joint" and len(clause.variables) > 1)
            + 2 * int(clause.model_family == "mlp")
            for clause in clauses
        )
        complexity += 2 * (len(clauses) - 1)
        parameter_count = sum(parameter.numel() for parameter in ensemble.parameters())
        capacity_cost = float(np.log1p(parameter_count))
        mean_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.0
        selection_score = (
            0.47 * balanced_accuracy
            + 0.18 * expert_safe_rate
            + 0.12 * violation_recall
            + 0.08 * safe_accuracy
            + self.config.intervention_yield_selection_weight * intervention_yield
            - self.config.complexity_penalty * complexity
            - self.config.capacity_penalty * capacity_cost
            - self.config.uncertainty_penalty * mean_uncertainty
        )
        if (
            safe_accuracy < self.config.minimum_class_accuracy
            or violation_recall < self.config.minimum_class_accuracy
        ):
            selection_score -= self.config.degenerate_predictor_penalty
        sufficient = int(np.sum(safe_mask)) >= self.config.minimum_per_label and int(np.sum(violation_mask)) >= self.config.minimum_per_label
        ineligibility_reasons: list[str] = []
        if not sufficient:
            ineligibility_reasons.append("insufficient_out_of_sample_labels")
        if safe_accuracy < self.config.champion_minimum_safe_accuracy:
            ineligibility_reasons.append("safe_accuracy_below_gate")
        if violation_recall < self.config.champion_minimum_violation_recall:
            ineligibility_reasons.append("violation_recall_below_gate")
        if expert_safe_rate < self.config.champion_minimum_expert_safe_rate:
            ineligibility_reasons.append("structure_audit_expert_safe_rate_below_gate")
        if fit_expert_safe_rate < self.config.champion_minimum_fit_expert_safe_rate:
            ineligibility_reasons.append("fit_expert_safe_rate_below_gate")
        champion_eligible = not ineligibility_reasons
        query_priority = (
            selection_score
            + 0.10 * counterexample_rate
            + 0.05 * min(mean_uncertainty, 2.0)
            + (0.05 if not sufficient else 0.0)
        )
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
            parameter_count=parameter_count,
            prequential_count=len(labels),
            query_priority=float(query_priority),
            fit_expert_safe_rate=fit_expert_safe_rate,
            champion_eligible=champion_eligible,
            ineligibility_reasons=tuple(ineligibility_reasons),
        )


def evidence_report(
    outer_round: int,
    evidence: list[HypothesisEvidence],
    label_counts: dict[str, int],
    records: list[QueryRecord] | None = None,
) -> dict[str, object]:
    ordered = sorted(evidence, key=lambda item: item.selection_score, reverse=True)
    qualified = [item for item in ordered if item.champion_eligible]
    pair_complementarity: list[dict[str, object]] = []
    disagreement_rates: list[float] = []
    if records:
        evidence_by_id = {item.hypothesis_id: item for item in evidence}
        ids = sorted(evidence_by_id)
        for record in records:
            available = [record.predictions_before_query[item] for item in ids if item in record.predictions_before_query]
            if len(available) >= 2:
                disagreement_rates.append(float(np.mean(available) * (1.0 - np.mean(available)) * 4.0))
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                if any(
                    min(evidence_by_id[item].safe_accuracy, evidence_by_id[item].violation_recall) <= 0.0
                    for item in (left, right)
                ):
                    # Do not manufacture a composite from complementary
                    # all-safe/all-unsafe degenerate classifiers.
                    continue
                labels: list[int] = []
                combined: list[int] = []
                for record in records:
                    if left in record.predictions_before_query and right in record.predictions_before_query:
                        labels.append(record.label)
                        combined.append(
                            max(record.predictions_before_query[left], record.predictions_before_query[right])
                        )
                if not labels:
                    continue
                label_array = np.asarray(labels)
                prediction_array = np.asarray(combined)
                safe = label_array == SAFE_LABEL
                violation = label_array == VIOLATION_LABEL
                safe_accuracy = float(np.mean(prediction_array[safe] == SAFE_LABEL)) if np.any(safe) else 0.5
                recall = float(np.mean(prediction_array[violation] == VIOLATION_LABEL)) if np.any(violation) else 0.5
                combined_balanced = 0.5 * (safe_accuracy + recall)
                baseline = max(
                    evidence_by_id[left].balanced_accuracy,
                    evidence_by_id[right].balanced_accuracy,
                )
                pair_complementarity.append(
                    {
                        "hypothesis_ids": [left, right],
                        "combined_balanced_accuracy": combined_balanced,
                        "gain_over_best_single": combined_balanced - baseline,
                        "prequential_count": len(labels),
                    }
                )
        pair_complementarity.sort(key=lambda item: float(item["gain_over_best_single"]), reverse=True)
    return {
        "outer_round": outer_round,
        "trajectory_label_counts": label_counts,
        "ranking": [item.hypothesis_id for item in ordered],
        "qualified_ranking": [item.hypothesis_id for item in qualified],
        "selection_status": "qualified" if qualified else "inconclusive",
        "hypotheses": [item.to_dict() for item in ordered],
        "cross_hypothesis_mean_disagreement": float(np.mean(disagreement_rates)) if disagreement_rates else 0.0,
        "pair_complementarity": pair_complementarity[:10],
        "important_note": (
            "Accuracy metrics are out-of-sample: pre-query predictions plus a label-stratified "
            "warmup audit split excluded from fitting. "
            "When configured, selection uses a rolling prequential window so obsolete early model versions "
            "do not permanently dominate the current structure; the full history remains in query diagnostics. "
            "All metrics come from trajectory-level labels and model behavior. "
            "No obstacle center, radius, state label, or evaluation IoU is included. "
            "A hypothesis with champion_eligible=false may be queried but must not be declared final or used to prune competitors."
        ),
    }
