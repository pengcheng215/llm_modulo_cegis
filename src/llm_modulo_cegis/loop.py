"""Bi-level semantic--numeric CEGIS controller."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .data import FeatureLibrary
from .evidence import EvidenceCompiler, evidence_report
from .evaluation import BoundaryMetrics, evaluate_boundary, plot_boundary, plot_semantic_trace
from .falsifier import (
    FalsifierResult,
    HypothesisFalsifier,
    displacement_actions,
    generate_warmup_candidate,
)
from .hypotheses import ConstraintHypothesis, HypothesisBank, RevisionAction, compile_hypothesis
from .learner import (
    LearnerRegistry,
    TrainerConfig,
    TrainingSummary,
    choose_decision_threshold,
    describe_ensemble_parameters,
    fit_ensemble,
)
from .oracle import TrajectoryEvaluationOracle, TrajectoryMembershipOracle
from .semantic import SemanticReasoner
from .types import InterventionSpec, QueryBuffer, QueryRecord, SAFE_LABEL, Trajectory, VIOLATION_LABEL


def _json_safe(value: object) -> object:
    """Replace non-finite diagnostics with JSON ``null`` recursively."""

    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_text(value: object) -> str:
    return json.dumps(
        _json_safe(value),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


@dataclass(frozen=True)
class LoopConfig:
    warmup_queries: int = 20
    max_warmup_queries: int = 50
    minimum_label_count: int = 4
    minimum_safe_query_count: int = 2
    minimum_violation_query_count: int = 4
    warmup_validation_safe_count: int = 1
    warmup_validation_violation_count: int = 3
    final_calibration_safe_count: int = 0
    final_calibration_violation_count: int = 0
    expert_structure_validation: int = 2
    outer_rounds: int = 3
    queries_per_hypothesis: int = 1
    oracle_query_budget_per_round: int = 4
    require_full_round_budget: bool = False
    candidate_pool_per_hypothesis: int = 3
    query_hypothesis_beam: int = 3
    acquisition_disagreement_weight: float = 0.35
    acquisition_boundary_weight: float = 0.25
    acquisition_uncertainty_weight: float = 0.15
    acquisition_novelty_weight: float = 0.15
    acquisition_potential_weight: float = 0.10
    candidate_deduplication_rms: float = 0.03
    candidate_history_deduplication_rms: float = 1.0e-4
    minimum_safe_label_fraction: float = 0.35
    minimum_violation_label_fraction: float = 0.35
    reserve_label_seeking_queries: bool = True
    maximum_label_balance_queries_per_round: int = 2
    calibrate_decision_threshold_during_cegis: bool = False
    decision_threshold_minimum_per_label: int = 2
    decision_threshold_minimum_fit_expert_safe_rate: float = 0.95
    safe_query_boundary_bisection_steps: int = 12
    safe_query_boundary_scan_points: int = 32
    safe_query_boundary_margin: float = 0.02
    falsifier_restarts: int = 2
    grid_resolution: int = 100
    freeze_revisions: bool = False
    maximum_active_hypotheses: int = 6
    retain_best_qualified_checkpoint: bool = False
    finalize_qualified_champion: bool = False
    finalization_epochs: int = 160
    finalization_minimum_expert_safe_rate: float = 0.95
    finalization_minimum_calibration_safe_accuracy: float = 0.50
    finalization_minimum_calibration_violation_recall: float = 0.50
    finalization_minimum_calibration_balanced_accuracy: float = 0.60
    finalization_allowed_calibration_drop: float = 0.02
    finalization_minimum_calibration_per_label: int = 2
    finalization_minimum_selection_per_label: int = 3
    finalization_scratch_restarts: int = 3
    finalization_finetune_incumbent: bool = True
    finalization_finetune_epochs: int = 80
    finalization_finetune_learning_rate_scale: float = 0.25
    finalization_finetune_background_weights: tuple[float, ...] = (0.0,)
    finalization_unsafe_volume_probe_count: int = 0
    finalization_disable_latent_witness: bool = False


@dataclass(frozen=True)
class RunResult:
    champion_hypothesis_id: str
    champion_hypothesis: dict[str, object]
    final_metrics: BoundaryMetrics
    oracle_queries: int
    label_counts: dict[str, int]
    active_hypotheses: tuple[str, ...]
    retired_hypotheses: tuple[str, ...]
    llm_interactions: int
    llm_fallbacks: int
    llm_augmentations: int
    llm_accepted_initial_hypotheses: int
    selection_status: str = "qualified"
    champion_eligible: bool = True
    champion_ineligibility_reasons: tuple[str, ...] = ()
    finalization_applied: bool = False
    decision_threshold: float = 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["final_metrics"] = self.final_metrics.to_dict()
        return payload


@dataclass
class AcquisitionCandidate:
    source: ConstraintHypothesis
    intervention: InterventionSpec
    result: FalsifierResult
    predictions: dict[str, int]
    scores: dict[str, float]
    uncertainties: dict[str, float]
    acquisition_score: float
    acquisition_components: dict[str, float]


class SemanticNumericCEGIS:
    """Outer semantic synthesis around independent inner neural CEGIS learners."""

    def __init__(
        self,
        *,
        task_description: str,
        feature_library: FeatureLibrary,
        reasoner: SemanticReasoner,
        registry: LearnerRegistry,
        trainer_config: TrainerConfig,
        evidence_compiler: EvidenceCompiler,
        falsifier: HypothesisFalsifier,
        oracle: TrajectoryMembershipOracle,
        evaluation_oracle: TrajectoryEvaluationOracle,
        experts: list[Trajectory],
        heldout_experts: list[Trajectory],
        workspace_x: tuple[float, float],
        workspace_y: tuple[float, float],
        loop_config: LoopConfig,
        output_dir: str | Path,
        seed: int,
        device: torch.device,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if (
            float(loop_config.safe_query_boundary_margin)
            > float(falsifier.config.false_unsafe_hard_margin) + 1.0e-12
        ):
            raise ValueError(
                "safe_query_boundary_margin cannot exceed the false-unsafe "
                "generation hard margin"
            )
        if loop_config.finalize_qualified_champion:
            calibration_capacity = min(
                int(loop_config.final_calibration_safe_count),
                int(loop_config.final_calibration_violation_count),
            )
            selection_capacity = min(
                int(loop_config.warmup_validation_safe_count),
                int(loop_config.warmup_validation_violation_count),
            )
            if (
                int(loop_config.finalization_minimum_calibration_per_label)
                > calibration_capacity
            ):
                raise ValueError(
                    "finalization_minimum_calibration_per_label exceeds the "
                    "reserved final-calibration examples for at least one label"
                )
            if (
                int(loop_config.finalization_minimum_selection_per_label)
                > selection_capacity
            ):
                raise ValueError(
                    "finalization_minimum_selection_per_label exceeds the "
                    "reserved warmup-validation examples for at least one label"
                )
        if loop_config.decision_threshold_minimum_per_label < 1:
            raise ValueError("decision_threshold_minimum_per_label must be positive")
        if not 0.0 <= loop_config.decision_threshold_minimum_fit_expert_safe_rate <= 1.0:
            raise ValueError(
                "decision_threshold_minimum_fit_expert_safe_rate must be in [0,1]"
            )
        self.task_description = task_description
        self.library = feature_library
        self.reasoner = reasoner
        self.registry = registry
        self.trainer_config = trainer_config
        self.evidence_compiler = evidence_compiler
        self.falsifier = falsifier
        self.oracle = oracle
        self.evaluation_oracle = evaluation_oracle
        audit_count = min(max(0, loop_config.expert_structure_validation), max(0, len(experts) - 1))
        self.structure_audit_experts = experts[-audit_count:] if audit_count else []
        self.experts = experts[:-audit_count] if audit_count else experts
        self.heldout_experts = heldout_experts
        self.workspace_x = workspace_x
        self.workspace_y = workspace_y
        self.config = loop_config
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.device = device
        self.progress = progress or (lambda _: None)
        self.buffer = QueryBuffer()
        self.rng = np.random.default_rng(seed)
        self.evidence_history: list[dict[str, object]] = []
        self.evaluation_history: list[dict[str, object]] = []
        self.query_diagnostics: list[dict[str, object]] = []
        self.all_hypothesis_evaluation: dict[str, object] = {}
        self.finalization_diagnostics: dict[str, object] = {"applied": False}
        self.threshold_calibration_history: list[dict[str, object]] = []
        self.latest_threshold_calibration: dict[str, dict[str, object]] = {}
        self.qualified_checkpoint_history: list[dict[str, object]] = []
        self.stage_diagnostics: list[dict[str, object]] = [
            {
                "stage": "expert_data",
                "fit_experts": [self._trajectory_summary(item) for item in self.experts],
                "structure_audit_experts": [
                    self._trajectory_summary(item) for item in self.structure_audit_experts
                ],
                "heldout_evaluation_experts": [
                    self._trajectory_summary(item) for item in self.heldout_experts
                ],
                "important_note": (
                    "All three groups contain known-safe demonstrations. Held-out evaluation "
                    "predictions below are diagnostic only and never enter training, selection, or LLM prompts."
                ),
            }
        ]
        self.pending_interventions: dict[str, list[InterventionSpec]] = {}
        self.query_priorities: dict[str, float] = {}

    def run(self) -> RunResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        initial = self.reasoner.propose_initial(self.task_description, self.library)
        if len(initial) > self.config.maximum_active_hypotheses:
            raise ValueError("initial reasoner output exceeds maximum_active_hypotheses")
        bank = HypothesisBank.from_hypotheses(initial, self.library)
        self._collect_warmup()
        champion_id = ""
        champion_metrics: BoundaryMetrics | None = None
        champion_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        final_selection_status = "inconclusive"
        final_champion_eligible = False
        final_ineligibility_reasons: tuple[str, ...] = ("no_numeric_evidence",)
        best_qualified_checkpoint: dict[str, object] | None = None

        for outer_round in range(1, self.config.outer_rounds + 1):
            active = bank.active()
            summaries = self._fit_active(active, outer_round)
            self._score_warmup_validation(active)
            self._record_model_stage(outer_round, "pre_query_fit", active, summaries)
            self._collect_hypothesis_queries(active, outer_round)
            summaries = self._fit_active(active, outer_round)
            evidence = self.evidence_compiler.compile(
                self.registry,
                [item.hypothesis_id for item in active],
                self.structure_audit_experts,
                self.buffer.records,
                self.experts,
            )
            report = evidence_report(outer_round, evidence, self.buffer.label_counts(), self.buffer.records)
            self.evidence_history.append(report)
            self._record_model_stage(outer_round, "post_query_refit", active, summaries, evidence)
            self.query_priorities = {item.hypothesis_id: item.query_priority for item in evidence}
            ordered = sorted(evidence, key=lambda item: item.selection_score, reverse=True)
            qualified = [item for item in ordered if item.champion_eligible]
            numeric_champion_evidence = qualified[0] if qualified else ordered[0]
            numeric_champion_id = numeric_champion_evidence.hypothesis_id
            selection_status = "qualified" if qualified else "inconclusive"
            actions = []
            # Revision actions define the search space and interventions for the
            # *next* numeric round.  Querying the semantic backend after the
            # terminal round wastes an LLM call and leaves recorded add/replace
            # actions unapplied (and therefore misleading in the audit log).
            has_next_round = outer_round < self.config.outer_rounds
            if has_next_round and not self.config.freeze_revisions:
                actions = self.reasoner.revise(
                    self.task_description,
                    self.library,
                    bank,
                    report,
                    evidence,
                    outer_round,
                )
            if not qualified and actions:
                # Defense in depth: even a custom semantic backend may not
                # retire or replace candidates when no model clears the
                # numeric qualification gates.  Keep at most one targeted
                # intervention and a provisional query target.
                interventions = [
                    action
                    for action in actions
                    if action.action == "propose_intervention"
                    and action.target_hypothesis_id == numeric_champion_id
                ][:1]
                actions = [
                    RevisionAction(
                        "retain_and_query",
                        numeric_champion_id,
                        "No hypothesis clears the champion gates; retain only as a provisional query target.",
                    ),
                    *interventions,
                ]
            semantic_retained = [
                action.target_hypothesis_id
                for action in actions
                if action.action == "retain_and_query"
                and action.target_hypothesis_id in self.registry.models
                and any(
                    item.hypothesis_id == action.target_hypothesis_id and item.champion_eligible
                    for item in evidence
                )
            ]
            champion_id = semantic_retained[0] if semantic_retained else numeric_champion_id
            champion_evidence = next(item for item in evidence if item.hypothesis_id == champion_id)
            final_selection_status = selection_status
            final_champion_eligible = champion_evidence.champion_eligible
            final_ineligibility_reasons = champion_evidence.ineligibility_reasons
            if (
                self.config.retain_best_qualified_checkpoint
                and champion_evidence.champion_eligible
            ):
                checkpoint_key = (
                    float(champion_evidence.selection_score),
                    int(outer_round),
                )
                incumbent_key = (
                    tuple(best_qualified_checkpoint["selection_key"])
                    if best_qualified_checkpoint is not None
                    else (-float("inf"), -1)
                )
                selected_as_best = bool(checkpoint_key > incumbent_key)
                checkpoint_row = {
                    "outer_round": int(outer_round),
                    "hypothesis_id": champion_id,
                    "selection_key": list(checkpoint_key),
                    "selection_evidence": champion_evidence.to_dict(),
                    "decision_threshold": float(
                        self.registry.models[champion_id].decision_threshold.item()
                    ),
                    "selected_as_best_so_far": selected_as_best,
                }
                self.qualified_checkpoint_history.append(checkpoint_row)
                if selected_as_best:
                    best_qualified_checkpoint = {
                        **checkpoint_row,
                        "model_state": deepcopy(
                            self.registry.models[champion_id].state_dict()
                        ),
                    }
            champion = self.registry.models[champion_id]
            champion_metrics, champion_grid = evaluate_boundary(
                champion,
                self.library,
                self.evaluation_oracle,
                self.heldout_experts,
                self.workspace_x,
                self.workspace_y,
                self.config.grid_resolution,
                self.device,
            )
            self.evaluation_history.append(
                {
                    "outer_round": outer_round,
                    "champion_hypothesis_id": champion_id,
                    "selection_status": selection_status,
                    "champion_eligible": champion_evidence.champion_eligible,
                    "champion_ineligibility_reasons": list(champion_evidence.ineligibility_reasons),
                    "training_losses": {key: value.mean_final_loss for key, value in summaries.items()},
                    "evaluation_only_metrics": champion_metrics.to_dict(),
                }
            )
            self.progress(
                f"outer_round={outer_round} champion={champion_id} "
                f"status={selection_status} "
                f"score={champion_evidence.selection_score:.3f} labels={self.buffer.label_counts()} "
                f"evaluation_iou={champion_metrics.iou:.3f}"
            )
            if has_next_round and not self.config.freeze_revisions:
                interventions = bank.apply_actions(
                    actions,
                    self.library,
                    outer_round=outer_round,
                    minimum_active=1,
                )
                self.pending_interventions.clear()
                for intervention in interventions:
                    self.pending_interventions.setdefault(intervention.target_hypothesis_id, []).append(intervention)
                if len(bank.active()) > self.config.maximum_active_hypotheses:
                    raise RuntimeError("semantic revision exceeded maximum_active_hypotheses")

        assert champion_metrics is not None and champion_grid is not None
        if (
            self.config.retain_best_qualified_checkpoint
            and best_qualified_checkpoint is not None
        ):
            champion_id = str(best_qualified_checkpoint["hypothesis_id"])
            self.registry.models[champion_id].load_state_dict(
                best_qualified_checkpoint["model_state"],
                strict=True,
            )
            final_selection_status = "qualified"
            final_champion_eligible = True
            final_ineligibility_reasons = ()
            champion_metrics, champion_grid = evaluate_boundary(
                self.registry.models[champion_id],
                self.library,
                self.evaluation_oracle,
                self.heldout_experts,
                self.workspace_x,
                self.workspace_y,
                self.config.grid_resolution,
                self.device,
            )
            restored_round = int(best_qualified_checkpoint["outer_round"])
            self.stage_diagnostics.append(
                {
                    "stage": "best_qualified_checkpoint_restore",
                    "restored_outer_round": restored_round,
                    "champion_hypothesis_id": champion_id,
                    "selection_key": best_qualified_checkpoint["selection_key"],
                    "selection_evidence": best_qualified_checkpoint[
                        "selection_evidence"
                    ],
                    "decision_threshold": float(
                        self.registry.models[champion_id].decision_threshold.item()
                    ),
                    "reason": (
                        "retain the strongest public-qualified numeric checkpoint "
                        "instead of allowing a later stochastic refit to regress"
                    ),
                }
            )
            self.evaluation_history.append(
                {
                    "stage": "best_qualified_checkpoint_restore",
                    "restored_outer_round": restored_round,
                    "champion_hypothesis_id": champion_id,
                    "selection_status": "qualified",
                    "champion_eligible": True,
                    "evaluation_only_metrics": champion_metrics.to_dict(),
                }
            )
            self.progress(
                "restored best qualified checkpoint "
                f"round={restored_round} champion={champion_id} "
                f"score={float(best_qualified_checkpoint['selection_key'][0]):.3f}"
            )
        finalization_applied = False
        if final_champion_eligible and self.config.finalize_qualified_champion:
            before_finalization = champion_metrics
            finalization_applied = self._finalize_champion(champion_id)
            champion_metrics, champion_grid = evaluate_boundary(
                self.registry.models[champion_id],
                self.library,
                self.evaluation_oracle,
                self.heldout_experts,
                self.workspace_x,
                self.workspace_y,
                self.config.grid_resolution,
                self.device,
            )
            self.finalization_diagnostics["evaluation_only_before"] = before_finalization.to_dict()
            self.finalization_diagnostics["evaluation_only_after"] = champion_metrics.to_dict()
            self.evaluation_history.append(
                {
                    "stage": "finalization",
                    "champion_hypothesis_id": champion_id,
                    "evaluation_only_metrics": champion_metrics.to_dict(),
                }
            )
        for hypothesis in bank.active():
            metrics, _ = evaluate_boundary(
                self.registry.models[hypothesis.hypothesis_id],
                self.library,
                self.evaluation_oracle,
                self.heldout_experts,
                self.workspace_x,
                self.workspace_y,
                self.config.grid_resolution,
                self.device,
            )
            self.all_hypothesis_evaluation[hypothesis.hypothesis_id] = metrics.to_dict()
        self._save_artifacts(bank, champion_id)
        # A static x-y contour is meaningful only for the planar adapter.
        # CarryWaterActive is evaluated on complete 12-D trajectories; drawing
        # a zero-velocity x-y slice would be both misleading and unsupported by
        # its feature library.
        if self.library.is_planar:
            plot_boundary(
                self.output_dir / "learned_boundary.png",
                champion_grid,
                self.heldout_experts,
                self.buffer.records,
                self.evaluation_oracle,
                f"Semantic--Numeric CEGIS champion: {champion_id}",
            )
        plot_semantic_trace(
            self.output_dir / "semantic_trace.png",
            self.evidence_history,
            bank.audit_log,
        )
        champion_hypothesis = bank.get(champion_id)
        active_ids = tuple(item.hypothesis_id for item in bank.active())
        retired_ids = tuple(key for key, entry in bank.entries.items() if entry.status == "retired")
        interactions = self.reasoner.interactions
        fallback_count = sum(bool(item.get("used_fallback", False)) for item in interactions)
        augmentation_count = sum(
            bool(item.get("used_augmentation", False)) or bool(item.get("policy_augmented", False))
            for item in interactions
        )
        accepted_initial = sum(
            int(item.get("accepted_llm_count", 0) or 0)
            for item in interactions
            if item.get("phase") == "initial"
        )
        result = RunResult(
            champion_id,
            champion_hypothesis.to_dict(),
            champion_metrics,
            self.oracle.query_count,
            self.buffer.label_counts(),
            active_ids,
            retired_ids,
            len(interactions),
            fallback_count,
            augmentation_count,
            accepted_initial,
            final_selection_status,
            final_champion_eligible,
            final_ineligibility_reasons,
            finalization_applied,
            float(self.registry.models[champion_id].decision_threshold.item()),
        )
        (self.output_dir / "result.json").write_text(
            _json_text(result.to_dict()),
            encoding="utf-8",
        )
        return result

    def _collect_warmup(self) -> None:
        candidate_index = 0
        safe_target = max(
            self.config.minimum_label_count,
            self.config.minimum_safe_query_count,
            self.config.warmup_validation_safe_count + self.config.final_calibration_safe_count + 1,
        )
        violation_target = max(
            self.config.minimum_label_count,
            self.config.minimum_violation_query_count,
            self.config.warmup_validation_violation_count + self.config.final_calibration_violation_count + 1,
        )
        count_fields = {
            "warmup_queries": self.config.warmup_queries,
            "max_warmup_queries": self.config.max_warmup_queries,
            "minimum_label_count": self.config.minimum_label_count,
            "minimum_safe_query_count": self.config.minimum_safe_query_count,
            "minimum_violation_query_count": self.config.minimum_violation_query_count,
            "warmup_validation_safe_count": self.config.warmup_validation_safe_count,
            "warmup_validation_violation_count": self.config.warmup_validation_violation_count,
            "final_calibration_safe_count": self.config.final_calibration_safe_count,
            "final_calibration_violation_count": self.config.final_calibration_violation_count,
        }
        negative = [name for name, value in count_fields.items() if int(value) < 0]
        if negative:
            raise ValueError("warmup counts must be nonnegative: " + ", ".join(negative))
        if self.config.warmup_queries > self.config.max_warmup_queries:
            raise ValueError("warmup_queries cannot exceed max_warmup_queries")
        if safe_target + violation_target > self.config.max_warmup_queries:
            raise ValueError(
                "warmup label targets cannot fit within max_warmup_queries: "
                f"safe={safe_target}, violation={violation_target}, "
                f"max={self.config.max_warmup_queries}"
            )
        if not self.experts:
            raise ValueError("warmup requires at least one fit expert")
        split_requirements = {
            "validation_safe": self.config.warmup_validation_safe_count,
            "validation_violation": self.config.warmup_validation_violation_count,
            "calibration_safe": self.config.final_calibration_safe_count,
            "calibration_violation": self.config.final_calibration_violation_count,
        }
        seen_candidates = {
            (record.trajectory.states.shape, record.trajectory.states.tobytes())
            for record in self.buffer.records
        }
        duplicate_candidates: list[dict[str, object]] = []
        candidate_attempt_limit = max(
            self.config.max_warmup_queries + 2,
            4 * self.config.max_warmup_queries,
        )
        while (
            len(self.buffer) < self.config.max_warmup_queries
            and candidate_index < candidate_attempt_limit
        ):
            counts = self.buffer.label_counts()
            if (
                len(self.buffer) >= self.config.warmup_queries
                and counts["safe"] >= safe_target
                and counts["violation"] >= violation_target
                and candidate_index % 2 == 0
            ):
                provisional_groups, provisional_errors = self._warmup_pair_groups(
                    self.buffer.records
                )
                if not provisional_errors:
                    try:
                        self._choose_warmup_group_roles(
                            provisional_groups,
                            split_requirements,
                        )
                        break
                    except ValueError:
                        # Label counts alone do not guarantee that correlated
                        # pair families can populate both holdouts.  Use the
                        # remaining budget to obtain another complete pair.
                        pass
            # Consecutive candidates are a controlled pair on the same expert:
            # one contracts the demonstrated detour and one continues it.
            expert = self.experts[(candidate_index // 2) % len(self.experts)]
            pool_warmup = getattr(
                getattr(self, "falsifier", None),
                "warmup_candidate",
                None,
            )
            candidate = (
                pool_warmup(expert, candidate_index, self.rng)
                if callable(pool_warmup)
                else generate_warmup_candidate(
                    expert,
                    candidate_index,
                    self.rng,
                    self.workspace_x,
                    self.workspace_y,
                )
            )
            candidate_index += 1
            fingerprint = (candidate.states.shape, candidate.states.tobytes())
            if fingerprint in seen_candidates:
                duplicate_candidates.append(
                    {
                        "candidate_index": candidate_index - 1,
                        "metadata": candidate.metadata,
                    }
                )
                continue
            seen_candidates.add(fingerprint)
            self.buffer.add(QueryRecord(candidate, self.oracle.query(candidate), "warmup", 0))
        counts = self.buffer.label_counts()
        if counts["safe"] < safe_target or counts["violation"] < violation_target:
            failure = {
                "actual_counts": counts,
                "target_counts": {
                    "safe": safe_target,
                    "violation": violation_target,
                },
                "queries": len(self.buffer),
                "max_warmup_queries": self.config.max_warmup_queries,
                "candidate_attempts": candidate_index,
                "candidate_attempt_limit": candidate_attempt_limit,
                "duplicate_candidates": duplicate_candidates,
                "records": [
                    {
                        "label": record.label,
                        "trajectory": self._trajectory_summary(record.trajectory),
                        "metadata": record.trajectory.metadata,
                    }
                    for record in self.buffer.records
                ],
            }
            (self.output_dir / "warmup_failure_diagnostics.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                "warmup failed label coverage: "
                f"actual={counts}, target={{'safe': {safe_target}, "
                f"'violation': {violation_target}}}"
            )
        groups, group_errors = self._warmup_pair_groups(self.buffer.records)
        try:
            if group_errors:
                raise ValueError("; ".join(group_errors))
            group_roles = self._choose_warmup_group_roles(groups, split_requirements)
        except ValueError as exc:
            failure = {
                "reason": str(exc),
                "requirements": split_requirements,
                "label_counts": counts,
                "candidate_attempts": candidate_index,
                "candidate_attempt_limit": candidate_attempt_limit,
                "duplicate_candidates": duplicate_candidates,
                "groups": [
                    {
                        key: value
                        for key, value in group.items()
                        if key != "records"
                    }
                    for group in groups
                ],
            }
            (self.output_dir / "warmup_split_failure_diagnostics.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"warmup pair-family split failed: {exc}") from exc

        source_by_role = {
            "train": ("warmup", "warmup_training"),
            "validation": ("warmup_validation", "heldout_structure_selection"),
            "calibration": ("final_calibration", "final_threshold_calibration"),
        }
        for group, role in zip(groups, group_roles):
            source, evidence_role = source_by_role[role]
            for record in group["records"]:
                record.source = source
                record.trajectory.metadata["evidence_role"] = evidence_role
        self.progress(f"warmup queries={len(self.buffer)} labels={self.buffer.label_counts()}")
        self.stage_diagnostics.append(
            {
                "stage": "warmup_oracle_queries",
                "label_counts": self.buffer.label_counts(),
                "pair_family_roles": {
                    str(group["family_id"]): role
                    for group, role in zip(groups, group_roles)
                },
                "candidate_attempts": candidate_index,
                "duplicate_candidate_count": len(duplicate_candidates),
                "queries": [
                    {
                        "label": record.label,
                        "evidence_role": record.source,
                        "trajectory": self._trajectory_summary(record.trajectory),
                    }
                    for record in self.buffer.records
                ],
            }
        )

    @staticmethod
    def _warmup_pair_groups(
        records: list[QueryRecord],
    ) -> tuple[list[dict[str, object]], list[str]]:
        """Group correlated paired probes so no family crosses evidence roles."""

        groups_by_id: dict[str, dict[str, object]] = {}
        ordered_ids: list[str] = []
        for record_index, record in enumerate(records):
            raw_pair = record.trajectory.metadata.get("warmup_pair_index")
            if isinstance(raw_pair, (int, np.integer)):
                family_id = f"pair:{int(raw_pair)}"
                pair_index: int | None = int(raw_pair)
            else:
                family_id = f"record:{record_index}"
                pair_index = None
            if family_id not in groups_by_id:
                groups_by_id[family_id] = {
                    "family_id": family_id,
                    "pair_index": pair_index,
                    "records": [],
                }
                ordered_ids.append(family_id)
            group_records = groups_by_id[family_id]["records"]
            assert isinstance(group_records, list)
            group_records.append(record)

        groups: list[dict[str, object]] = []
        errors: list[str] = []
        for order, family_id in enumerate(ordered_ids):
            group = groups_by_id[family_id]
            group_records = group["records"]
            assert isinstance(group_records, list)
            safe_count = sum(record.label == SAFE_LABEL for record in group_records)
            violation_count = sum(record.label == VIOLATION_LABEL for record in group_records)
            expert_ids = {
                record.trajectory.metadata.get("expert_id") for record in group_records
            }
            alphas = {record.trajectory.metadata.get("alpha") for record in group_records}
            directions = [
                record.trajectory.metadata.get("warmup_direction") for record in group_records
            ]
            if group["pair_index"] is not None:
                if len(group_records) != 2:
                    errors.append(f"{family_id} must contain exactly two records")
                if len(expert_ids) != 1 or len(alphas) != 1:
                    errors.append(f"{family_id} does not share one expert and scale")
                if len(set(directions)) != len(directions):
                    errors.append(f"{family_id} repeats a warmup direction")
            group.update(
                {
                    "order": order,
                    "record_count": len(group_records),
                    "safe_count": safe_count,
                    "violation_count": violation_count,
                    "labels": [int(record.label) for record in group_records],
                    "expert_ids": sorted(str(value) for value in expert_ids),
                    "alphas": sorted(float(value) for value in alphas if value is not None),
                    "directions": directions,
                }
            )
            groups.append(group)
        return groups, errors

    @staticmethod
    def _choose_warmup_group_roles(
        groups: list[dict[str, object]],
        requirements: dict[str, int],
    ) -> tuple[str, ...]:
        """Find a deterministic minimum-size group-stratified holdout split."""

        validation_safe = int(requirements["validation_safe"])
        validation_violation = int(requirements["validation_violation"])
        calibration_safe = int(requirements["calibration_safe"])
        calibration_violation = int(requirements["calibration_violation"])
        total_safe = sum(int(group["safe_count"]) for group in groups)
        total_violation = sum(int(group["violation_count"]) for group in groups)

        # State: capped validation/calibration counts plus the actual number of
        # held-out labels.  The latter guarantees at least one training record
        # of each label remains.  Values retain the most tail-preferring stable
        # assignment for equivalent states.
        initial_state = (0, 0, 0, 0, 0, 0)
        states: dict[tuple[int, int, int, int, int, int], tuple[tuple[int, ...], int, int]] = {
            initial_state: ((), 0, 0)
        }
        for group_index, group in enumerate(groups):
            safe_count = int(group["safe_count"])
            violation_count = int(group["violation_count"])
            bit = 1 << group_index
            next_states: dict[
                tuple[int, int, int, int, int, int],
                tuple[tuple[int, ...], int, int],
            ] = {}
            for state, (roles, calibration_mask, validation_mask) in states.items():
                vs, vv, cs, cv, held_safe, held_violation = state
                for role_code in (0, 1, 2):  # train, validation, calibration
                    new_vs, new_vv, new_cs, new_cv = vs, vv, cs, cv
                    new_held_safe, new_held_violation = held_safe, held_violation
                    new_calibration_mask = calibration_mask
                    new_validation_mask = validation_mask
                    if role_code != 0:
                        new_held_safe += safe_count
                        new_held_violation += violation_count
                        if (
                            new_held_safe > total_safe - 1
                            or new_held_violation > total_violation - 1
                        ):
                            continue
                    if role_code == 1:
                        new_vs = min(validation_safe, vs + safe_count)
                        new_vv = min(validation_violation, vv + violation_count)
                        new_validation_mask |= bit
                    elif role_code == 2:
                        new_cs = min(calibration_safe, cs + safe_count)
                        new_cv = min(calibration_violation, cv + violation_count)
                        new_calibration_mask |= bit
                    new_state = (
                        new_vs,
                        new_vv,
                        new_cs,
                        new_cv,
                        new_held_safe,
                        new_held_violation,
                    )
                    candidate = (
                        roles + (role_code,),
                        new_calibration_mask,
                        new_validation_mask,
                    )
                    incumbent = next_states.get(new_state)
                    if incumbent is None or (
                        candidate[1], candidate[2], tuple(reversed(candidate[0]))
                    ) > (
                        incumbent[1], incumbent[2], tuple(reversed(incumbent[0]))
                    ):
                        next_states[new_state] = candidate
            states = next_states

        feasible: list[
            tuple[
                tuple[int, int, int, int, int, int],
                tuple[tuple[int, ...], int, int],
            ]
        ] = []
        required_caps = (
            validation_safe,
            validation_violation,
            calibration_safe,
            calibration_violation,
        )
        for state, assignment in states.items():
            if (
                state[:4] == required_caps
                and total_safe - state[4] >= 1
                and total_violation - state[5] >= 1
            ):
                feasible.append((state, assignment))
        if not feasible:
            raise ValueError(
                "no whole-family partition satisfies both holdouts while "
                "leaving at least one training record per label"
            )
        _, (role_codes, _, _) = min(
            feasible,
            key=lambda item: (
                item[0][4] + item[0][5],
                -item[1][1],
                -item[1][2],
                tuple(-value for value in reversed(item[1][0])),
            ),
        )
        role_names = ("train", "validation", "calibration")
        return tuple(role_names[code] for code in role_codes)

    def _score_warmup_validation(self, active: list[ConstraintHypothesis]) -> None:
        """Freeze each hypothesis' first prediction on the selection holdout."""

        for record in self.buffer.records:
            if record.source != "warmup_validation":
                continue
            for hypothesis in active:
                hypothesis_id = hypothesis.hypothesis_id
                if hypothesis_id in record.predictions_before_query:
                    continue
                prediction, score, uncertainty = self.registry.predict(hypothesis_id, record.trajectory)
                record.predictions_before_query[hypothesis_id] = prediction
                record.scores_before_query[hypothesis_id] = score
                record.uncertainties_before_query[hypothesis_id] = uncertainty

    def _fit_active(
        self,
        active: list[ConstraintHypothesis],
        outer_round: int,
    ) -> dict[str, TrainingSummary]:
        summaries: dict[str, TrainingSummary] = {}
        for index, hypothesis in enumerate(active):
            ensemble = self.registry.ensure(compile_hypothesis(hypothesis, self.library))
            summaries[hypothesis.hypothesis_id] = fit_ensemble(
                ensemble,
                self.experts,
                [
                    record
                    for record in self.buffer.records
                    if record.source not in {"warmup_validation", "final_calibration"}
                ],
                self.library,
                self.trainer_config,
                seed=self.seed + outer_round * 7919 + index * 101,
                device=self.device,
            )
        self._calibrate_active_decision_thresholds(active, outer_round)
        return summaries

    def _calibrate_active_decision_thresholds(
        self,
        active: list[ConstraintHypothesis],
        outer_round: int,
    ) -> None:
        """Fit trajectory decision thresholds without consuming holdout labels.

        Neural scores can rank safe and violating trajectories correctly while
        their arbitrary zero point drifts under class imbalance.  Threshold is
        therefore treated as another numeric parameter, selected on the
        disjoint ``final_calibration`` warmup family and constrained using only
        fit experts.  Neither gradient-training records nor
        ``warmup_validation`` labels enter this choice.
        """

        if not self.config.calibrate_decision_threshold_during_cegis:
            return
        records = [
            record for record in self.buffer.records if record.source == "final_calibration"
        ]
        labels = [int(record.label) for record in records]
        label_counts = {
            "safe": sum(label == SAFE_LABEL for label in labels),
            "violation": sum(label == VIOLATION_LABEL for label in labels),
        }
        minimum = int(self.config.decision_threshold_minimum_per_label)
        fit_pass_index = sum(
            int(item.get("outer_round", -1)) == int(outer_round)
            for item in self.threshold_calibration_history
        )
        rows: list[dict[str, object]] = []
        for hypothesis in active:
            hypothesis_id = hypothesis.hypothesis_id
            ensemble = self.registry.models[hypothesis_id]
            before = float(ensemble.decision_threshold.item())
            row: dict[str, object] = {
                "hypothesis_id": hypothesis_id,
                "calibration_query_count": len(records),
                "calibration_label_counts": dict(label_counts),
                "calibration_pair_ids": sorted(
                    {
                        str(record.trajectory.metadata.get("warmup_pair_index"))
                        for record in records
                    }
                ),
                "minimum_per_label": minimum,
                "calibration_source": "final_calibration",
                "excluded_sources": ["warmup", "warmup_validation", "active_queries"],
                "threshold_before": before,
                "threshold_after": before,
                "applied": False,
            }
            if label_counts["safe"] < minimum or label_counts["violation"] < minimum:
                row["reason"] = "insufficient_disjoint_calibration_labels"
                rows.append(row)
                self.latest_threshold_calibration[hypothesis_id] = row
                failure = {
                    "outer_round": int(outer_round),
                    "hypothesis_id": hypothesis_id,
                    "required_per_label": minimum,
                    "actual_label_counts": dict(label_counts),
                    "calibration_source": "final_calibration",
                    "policy": "fail_enabled_calibration_without_both_labels",
                }
                failure_path = self.output_dir / (
                    f"threshold_calibration_failure_round_{int(outer_round)}_"
                    f"{hypothesis_id}.json"
                )
                failure_path.write_text(_json_text(failure), encoding="utf-8")
                raise RuntimeError(
                    "decision-threshold calibration lacks a disjoint safe/violation "
                    f"split for {hypothesis_id!r}; see {failure_path}"
                )
            else:
                scores = [
                    self._raw_hard_score_for_ensemble(ensemble, record.trajectory)
                    for record in records
                ]
                expert_scores = [
                    self._raw_hard_score_for_ensemble(ensemble, expert)
                    for expert in self.experts
                ]
                calibration = choose_decision_threshold(
                    scores,
                    labels,
                    expert_scores,
                    minimum_expert_safe_rate=(
                        self.config.decision_threshold_minimum_fit_expert_safe_rate
                    ),
                )
                threshold = float(calibration["selected_threshold"])
                ensemble.set_decision_threshold(threshold)
                row.update(
                    {
                        "threshold_after": threshold,
                        "applied": True,
                        "reason": "disjoint_final_calibration",
                        "selected_metrics": calibration["selected_metrics"],
                        "fit_expert_constraint_satisfied": calibration[
                            "expert_constraint_satisfied"
                        ],
                        "candidate_threshold_count": calibration["candidate_count"],
                        "score_min": float(np.min(scores)),
                        "score_max": float(np.max(scores)),
                        "fit_expert_score_min": float(np.min(expert_scores)),
                        "fit_expert_score_max": float(np.max(expert_scores)),
                    }
                )
            rows.append(row)
            self.latest_threshold_calibration[hypothesis_id] = row
        self.threshold_calibration_history.append(
            {
                "outer_round": int(outer_round),
                "fit_pass_index_within_round": int(fit_pass_index),
                "models": rows,
            }
        )

    def _collect_hypothesis_queries(
        self,
        active: list[ConstraintHypothesis],
        outer_round: int,
    ) -> None:
        sources = self._query_sources(active)
        pool_size = max(1, self.config.candidate_pool_per_hypothesis, self.config.queries_per_hypothesis)
        candidates: list[AcquisitionCandidate] = []
        for hypothesis_index, hypothesis in enumerate(sources):
            for query_index in range(pool_size):
                intervention = self._intervention_for(hypothesis, outer_round, query_index)
                expert_index = (outer_round + hypothesis_index + query_index) % len(self.experts)
                expert = self._expert_anchor_for_intervention(
                    hypothesis,
                    intervention,
                    expert_index,
                )
                restart_offset = 0
                pool_rank_offset = getattr(
                    self.falsifier,
                    "candidate_rank_offset",
                    None,
                )
                if callable(pool_rank_offset):
                    restart_offset = int(
                        pool_rank_offset(
                            outer_round=outer_round,
                            pool_slot=query_index,
                            pool_size=pool_size,
                            restarts=max(1, int(self.config.falsifier_restarts)),
                        )
                    )
                result = self._synthesize_best(
                    hypothesis,
                    expert,
                    intervention,
                    outer_round,
                    restart_offset=restart_offset,
                )
                if intervention.kind == "model_false_unsafe":
                    result = self._refine_false_unsafe_to_nearest_boundary(
                        result,
                        expert,
                        active,
                    )
                predictions: dict[str, int] = {}
                scores: dict[str, float] = {}
                uncertainties: dict[str, float] = {}
                for other in active:
                    prediction, score, uncertainty = self.registry.predict(other.hypothesis_id, result.trajectory)
                    predictions[other.hypothesis_id] = prediction
                    scores[other.hypothesis_id] = score
                    uncertainties[other.hypothesis_id] = uncertainty
                acquisition_score, components = self._acquisition_score(
                    hypothesis.hypothesis_id,
                    result.trajectory,
                    predictions,
                    scores,
                    uncertainties,
                )
                candidates.append(
                    AcquisitionCandidate(
                        hypothesis,
                        intervention,
                        result,
                        predictions,
                        scores,
                        uncertainties,
                        acquisition_score,
                        components,
                    )
                )

        candidates.sort(key=lambda item: item.acquisition_score, reverse=True)
        candidate_ranks = {id(candidate): rank for rank, candidate in enumerate(candidates)}
        initial_label_acquisition = {
            id(candidate): self._label_acquisition_components(candidate)
            for candidate in candidates
        }
        selected: list[AcquisitionCandidate] = []
        selected_reasons: dict[int, str] = {}
        selected_sequence: dict[int, int] = {}
        selected_label_acquisition: dict[int, dict[str, object]] = {}
        query_labels: dict[int, int] = {}
        budget = self.config.oracle_query_budget_per_round
        if budget <= 0:
            budget = max(1, len(sources) * max(1, self.config.queries_per_hypothesis))

        # Freeze every model prediction before the first Oracle call, then spend
        # the batch sequentially.  The only quantity that changes between slots
        # is the observed label balance and the empirical intervention yield.
        # This preserves pre-query evidence while allowing a failed safe-seeking
        # probe to be followed by another one in the same round.
        for query_sequence_index in range(budget):
            counts_before = self._trainable_label_counts()
            remaining_budget = budget - query_sequence_index
            label_balance_queries_used = sum(
                reason.startswith("adaptive_label_balance_")
                for reason in selected_reasons.values()
            )
            candidate, reason, label_acquisition = self._choose_next_acquisition(
                candidates,
                selected,
                counts_before,
                remaining_budget,
                label_balance_queries_used,
                query_labels,
            )
            if candidate is None:
                break
            label = self.oracle.query(candidate.result.trajectory)
            selected.append(candidate)
            selected_reasons[id(candidate)] = reason
            selected_sequence[id(candidate)] = query_sequence_index
            selected_label_acquisition[id(candidate)] = label_acquisition
            query_labels[id(candidate)] = label
            self.buffer.add(
                QueryRecord(
                    candidate.result.trajectory,
                    label,
                    candidate.intervention.kind,
                    outer_round,
                    candidate.source.hypothesis_id,
                    candidate.predictions,
                    candidate.scores,
                    candidate.uncertainties,
                )
            )
            self._consume_pending(candidate.intervention)
            bounds = self._trajectory_summary(candidate.result.trajectory)
            self.progress(
                f"query round={outer_round} sequence={query_sequence_index} "
                f"rank={candidate_ranks[id(candidate)]} "
                f"source={candidate.source.hypothesis_id} kind={candidate.intervention.kind} "
                f"reason={reason} label={'safe' if label == SAFE_LABEL else 'violation'} "
                f"estimated_p_safe={float(label_acquisition['estimated_safe_probability']):.3f} "
                f"y=[{bounds['y_min']:.3f},{bounds['y_max']:.3f}] "
                f"prequery_predictions={candidate.predictions}"
            )

        selected_ids = {id(item) for item in selected}
        for candidate_rank, candidate in enumerate(candidates):
            queried = id(candidate) in selected_ids
            label = query_labels.get(id(candidate))
            self.query_diagnostics.append(
                {
                    "outer_round": outer_round,
                    "candidate_rank": candidate_rank,
                    "query_sequence_index": selected_sequence.get(id(candidate)),
                    "queried": queried,
                    "selection_reason": selected_reasons.get(id(candidate), "not_selected"),
                    "source_hypothesis_id": candidate.source.hypothesis_id,
                    "source_expert_id": candidate.result.trajectory.metadata.get("expert_id"),
                    "intervention": candidate.intervention.to_dict(),
                    "oracle_label": label,
                    "trajectory": self._trajectory_summary(candidate.result.trajectory),
                    "predictions_before_query": candidate.predictions,
                    "scores_before_query": candidate.scores,
                    "uncertainties_before_query": candidate.uncertainties,
                    "acquisition_score": candidate.acquisition_score,
                    "acquisition_components": candidate.acquisition_components,
                    "label_acquisition_initial": initial_label_acquisition[id(candidate)],
                    "label_acquisition_at_selection": selected_label_acquisition.get(id(candidate)),
                    "falsifier": {
                        "initial_loss": candidate.result.initial_loss,
                        "final_loss": candidate.result.final_loss,
                        "final_score": candidate.result.final_score,
                        "final_uncertainty": candidate.result.final_uncertainty,
                        "validation_reason": candidate.result.validation_reason,
                        "false_unsafe_trust_radius": candidate.result.trajectory.metadata.get(
                            "trust_radius"
                        ),
                        "max_expert_deviation": candidate.result.trajectory.metadata.get(
                            "max_expert_deviation"
                        ),
                        "source_witness_index": candidate.result.trajectory.metadata.get(
                            "source_witness_index"
                        ),
                        "boundary_refined": candidate.result.trajectory.metadata.get(
                            "boundary_refined", False
                        ),
                        "boundary_refinement_alpha": candidate.result.trajectory.metadata.get(
                            "boundary_refinement_alpha"
                        ),
                        "boundary_trigger_hypothesis_id": candidate.result.trajectory.metadata.get(
                            "boundary_trigger_hypothesis_id"
                        ),
                        "boundary_trigger_clause_id": candidate.result.trajectory.metadata.get(
                            "boundary_trigger_clause_id"
                        ),
                        "boundary_trigger_score": candidate.result.trajectory.metadata.get(
                            "boundary_trigger_score"
                        ),
                        "boundary_trigger_full_score": candidate.result.trajectory.metadata.get(
                            "boundary_trigger_full_score"
                        ),
                        "boundary_trigger_target_score": candidate.result.trajectory.metadata.get(
                            "boundary_trigger_target_score"
                        ),
                        "boundary_refinement_status": candidate.result.trajectory.metadata.get(
                            "boundary_refinement_status"
                        ),
                        "safe_query_causal_rejector_ids": candidate.result.trajectory.metadata.get(
                            "safe_query_causal_rejector_ids", []
                        ),
                        "source_anchor_hard_score": candidate.result.trajectory.metadata.get(
                            "source_anchor_hard_score"
                        ),
                        "source_anchor_margin_satisfied": candidate.result.trajectory.metadata.get(
                            "source_anchor_margin_satisfied"
                        ),
                        "decision_threshold": candidate.result.trajectory.metadata.get(
                            "decision_threshold"
                        ),
                        "optimization_target_clause_id": candidate.result.trajectory.metadata.get(
                            "optimization_target_clause_id"
                        ),
                        "generation_hard_margin_target": candidate.result.trajectory.metadata.get(
                            "generation_hard_margin_target"
                        ),
                        "generation_full_hard_score": candidate.result.trajectory.metadata.get(
                            "generation_full_hard_score"
                        ),
                        "generation_target_hard_score": candidate.result.trajectory.metadata.get(
                            "generation_target_hard_score"
                        ),
                        "generation_achieved_hard_margin": candidate.result.trajectory.metadata.get(
                            "generation_achieved_hard_margin"
                        ),
                        "generation_hard_margin_satisfied": candidate.result.trajectory.metadata.get(
                            "generation_hard_margin_satisfied"
                        ),
                        "query_target_clause_id": candidate.result.trajectory.metadata.get(
                            "query_target_clause_id"
                        ),
                        "query_hard_margin_target": candidate.result.trajectory.metadata.get(
                            "query_hard_margin_target"
                        ),
                        "query_source_full_hard_score": candidate.result.trajectory.metadata.get(
                            "query_source_full_hard_score"
                        ),
                        "query_source_target_hard_score": candidate.result.trajectory.metadata.get(
                            "query_source_target_hard_score"
                        ),
                        "query_achieved_hard_margin": candidate.result.trajectory.metadata.get(
                            "query_achieved_hard_margin"
                        ),
                        "query_hard_margin_satisfied": candidate.result.trajectory.metadata.get(
                            "query_hard_margin_satisfied"
                        ),
                        "hard_margin_checkpoint_step": candidate.result.trajectory.metadata.get(
                            "hard_margin_checkpoint_step"
                        ),
                        "rejected_hard_margin_checkpoints": candidate.result.trajectory.metadata.get(
                            "rejected_hard_margin_checkpoints", {}
                        ),
                        "initial_hard_score_gradient_norm": candidate.result.trajectory.metadata.get(
                            "initial_hard_score_gradient_norm"
                        ),
                        "initial_smooth_score_gradient_norm": candidate.result.trajectory.metadata.get(
                            "initial_smooth_score_gradient_norm"
                        ),
                        "falsifier_reachability_status": candidate.result.trajectory.metadata.get(
                            "falsifier_reachability_status"
                        ),
                        "radius_ladder_attempted": candidate.result.trajectory.metadata.get(
                            "radius_ladder_attempted", []
                        ),
                        "selected_trust_radius": candidate.result.trajectory.metadata.get(
                            "selected_trust_radius"
                        ),
                        "radius_ladder_status": candidate.result.trajectory.metadata.get(
                            "radius_ladder_status"
                        ),
                        "radius_ladder_attempts": candidate.result.trajectory.metadata.get(
                            "radius_ladder_attempts", []
                        ),
                        "pre_refinement_max_expert_deviation": candidate.result.trajectory.metadata.get(
                            "pre_refinement_max_expert_deviation"
                        ),
                    },
                }
            )
        final_counts = self._trainable_label_counts()
        final_deficits = self._label_deficits(final_counts, sum(final_counts.values()))
        self.progress(
            f"outer_round={outer_round} candidate_pool={len(candidates)} "
            f"oracle_selected={len(selected)} trainable_labels={final_counts} "
            f"remaining_label_deficits={final_deficits} "
            f"query_sources={[item.hypothesis_id for item in sources]}"
        )
        if self.config.require_full_round_budget and len(selected) != budget:
            selected_ids = {id(item) for item in selected}
            failure = {
                "outer_round": int(outer_round),
                "requested_queries": int(budget),
                "selected_queries": len(selected),
                "candidate_count": len(candidates),
                "queryable_candidate_count": sum(
                    self._candidate_is_queryable(item) for item in candidates
                ),
                "history_duplicate_candidate_count": sum(
                    self._duplicates_query_history(item) for item in candidates
                ),
                "unselected_candidate_count": sum(
                    id(item) not in selected_ids for item in candidates
                ),
                "query_sources": [item.hypothesis_id for item in sources],
                "selected_pool_candidate_ids": [
                    item.result.trajectory.metadata.get("pool_candidate_id")
                    for item in selected
                ],
                "policy": "fail_the_run_do_not_drop_the_seed",
            }
            diagnostic_path = (
                self.output_dir
                / f"query_budget_underfill_round_{int(outer_round)}.json"
            )
            diagnostic_path.write_text(_json_text(failure), encoding="utf-8")
            raise RuntimeError(
                "outer-loop Oracle query budget underfilled: "
                f"round={outer_round}, selected={len(selected)}, requested={budget}; "
                f"see {diagnostic_path}"
            )

    def _expert_anchor_for_intervention(
        self,
        hypothesis: ConstraintHypothesis,
        intervention: InterventionSpec,
        default_index: int,
    ) -> Trajectory:
        """Choose a model-accepted expert anchor for a false-unsafe search.

        Demonstrations are known safe, but a partially trained hypothesis may
        already reject some of them.  Such an anchor cannot define a causal
        model-safe-to-model-unsafe crossing, so prefer the accepted expert that
        lies closest to the calibrated boundary while retaining a small safe
        margin.  Other intervention kinds keep the round-robin schedule.
        """

        fallback = self.experts[default_index % len(self.experts)]
        if intervention.kind != "model_false_unsafe":
            return fallback
        ensemble = self.registry.models.get(hypothesis.hypothesis_id)
        if ensemble is None:
            return fallback
        threshold = float(ensemble.decision_threshold.item())
        anchor_margin = max(
            0.0,
            float(self.falsifier.config.false_unsafe_anchor_margin),
        )
        accepted: list[tuple[float, int, Trajectory]] = []
        for index, expert in enumerate(self.experts):
            _, score, _ = self.registry.predict(hypothesis.hypothesis_id, expert)
            if score <= threshold - anchor_margin:
                cyclic_distance = (index - default_index) % len(self.experts)
                accepted.append((threshold - score, cyclic_distance, expert))
        if not accepted:
            return fallback
        # Minimum safe-side distance is the easiest genuine crossing; the
        # cyclic tie-break retains deterministic round-to-round diversity.
        return min(accepted, key=lambda item: (item[0], item[1]))[2]

    def _trainable_label_counts(self) -> dict[str, int]:
        """Count labels that actually enter gradient fitting.

        Calibration and selection holdouts are valuable audit data, but treating
        them as training coverage previously hid a severe safe-label shortage.
        """

        records = [
            record
            for record in self.buffer.records
            if record.source not in {"warmup_validation", "final_calibration"}
        ]
        return {
            "safe": sum(record.label == SAFE_LABEL for record in records),
            "violation": sum(record.label == VIOLATION_LABEL for record in records),
        }

    def _label_deficits(
        self,
        counts: dict[str, int],
        projected_total: int,
    ) -> dict[str, int]:
        safe_target = int(np.ceil(self.config.minimum_safe_label_fraction * projected_total - 1.0e-12))
        violation_target = int(
            np.ceil(self.config.minimum_violation_label_fraction * projected_total - 1.0e-12)
        )
        return {
            "safe": max(0, safe_target - int(counts.get("safe", 0))),
            "violation": max(0, violation_target - int(counts.get("violation", 0))),
        }

    def _choose_next_acquisition(
        self,
        candidates: list[AcquisitionCandidate],
        selected: list[AcquisitionCandidate],
        label_counts: dict[str, int],
        remaining_budget: int,
        label_balance_queries_used: int = 0,
        selected_labels: dict[int, int] | None = None,
    ) -> tuple[AcquisitionCandidate | None, str, dict[str, object]]:
        """Choose one candidate using only frozen predictions and past labels."""

        selected_ids = {id(candidate) for candidate in selected}
        available = [
            candidate
            for candidate in candidates
            if id(candidate) not in selected_ids
            and not self._duplicates_selected(candidate, selected)
            and not self._duplicates_query_history(candidate)
            and self._candidate_is_queryable(candidate)
        ]
        if not available:
            return None, "candidate_pool_exhausted", {}

        projected_total = sum(label_counts.values()) + max(0, int(remaining_budget))
        deficits = self._label_deficits(label_counts, projected_total)
        target_label: int | None = None
        balance_cap = max(0, int(self.config.maximum_label_balance_queries_per_round))
        if self.config.reserve_label_seeking_queries and label_balance_queries_used < balance_cap:
            safe_pressure = deficits["safe"] / max(
                1,
                int(np.ceil(self.config.minimum_safe_label_fraction * projected_total - 1.0e-12)),
            )
            violation_pressure = deficits["violation"] / max(
                1,
                int(np.ceil(self.config.minimum_violation_label_fraction * projected_total - 1.0e-12)),
            )
            if deficits["safe"] > 0 or deficits["violation"] > 0:
                target_label = SAFE_LABEL if safe_pressure >= violation_pressure else VIOLATION_LABEL

        scored = [(candidate, self._label_acquisition_components(candidate)) for candidate in available]
        selected_labels = selected_labels or {}
        confirmed_safe_signatures: set[tuple[str, ...]] = set()
        for selected_candidate in selected:
            if selected_labels.get(id(selected_candidate)) != SAFE_LABEL:
                continue
            selected_components = self._label_acquisition_components(
                selected_candidate
            )
            signature = tuple(
                map(str, selected_components["prediction_rejection_signature"])
            )
            if selected_components["safe_query_eligible"] and signature:
                confirmed_safe_signatures.add(signature)
        reason = "global_acquisition"
        if target_label == SAFE_LABEL:
            safe_pool = [
                item
                for item in scored
                if item[0].intervention.kind in {"model_false_unsafe", "boundary_uncertainty"}
                and bool(item[1]["safe_query_eligible"])
                and tuple(item[1]["prediction_rejection_signature"])
                not in confirmed_safe_signatures
            ]
            if safe_pool:
                candidate, components = max(
                    safe_pool,
                    key=lambda item: float(item[1]["safe_label_utility"]),
                )
                reason = "adaptive_label_balance_safe"
            else:
                candidate, components = max(scored, key=lambda item: item[0].acquisition_score)
        elif target_label == VIOLATION_LABEL:
            violation_pool = [
                item
                for item in scored
                if item[0].intervention.kind
                in {"model_false_safe", "shortcut", "boundary_uncertainty", "local_feature_stress"}
            ]
            if violation_pool:
                candidate, components = max(
                    violation_pool,
                    key=lambda item: float(item[1]["violation_label_utility"]),
                )
                reason = "adaptive_label_balance_violation"
            else:
                candidate, components = max(scored, key=lambda item: item[0].acquisition_score)
        else:
            candidate, components = max(scored, key=lambda item: item[0].acquisition_score)

        selection_components: dict[str, object] = dict(components)
        selection_components.update(
            {
                "trainable_label_counts_before_query": dict(label_counts),
                "remaining_budget_including_query": int(remaining_budget),
                "label_balance_queries_used_before_query": int(label_balance_queries_used),
                "maximum_label_balance_queries_per_round": balance_cap,
                "projected_trainable_total": int(projected_total),
                "label_deficits_before_query": deficits,
                "target_label": (
                    "safe" if target_label == SAFE_LABEL else "violation" if target_label == VIOLATION_LABEL else None
                ),
            }
        )
        return candidate, reason, selection_components

    def _label_acquisition_components(self, candidate: AcquisitionCandidate) -> dict[str, object]:
        predictions = list(candidate.predictions.values())
        safe_vote_fraction = (
            float(np.mean([prediction == SAFE_LABEL for prediction in predictions]))
            if predictions
            else 0.5
        )
        unsafe_voters = sorted(
            hypothesis_id
            for hypothesis_id, prediction in candidate.predictions.items()
            if prediction == VIOLATION_LABEL
        )
        kind = candidate.intervention.kind
        prior_alpha, prior_beta = {
            "model_false_unsafe": (2.0, 1.0),
            "boundary_uncertainty": (1.0, 1.0),
            "local_feature_stress": (1.0, 1.0),
            "model_false_safe": (1.0, 2.0),
            "shortcut": (1.0, 2.0),
        }.get(kind, (1.0, 1.0))
        historical = [record for record in self.buffer.records if record.source == kind]
        historical_safe = sum(record.label == SAFE_LABEL for record in historical)
        historical_violation = sum(record.label == VIOLATION_LABEL for record in historical)
        posterior_safe_yield = (prior_alpha + historical_safe) / (
            prior_alpha + prior_beta + historical_safe + historical_violation
        )

        metadata = candidate.result.trajectory.metadata
        deformation = metadata.get("max_expert_deviation")
        trust_radius = metadata.get("trust_radius")
        normalized_pool_deformation = metadata.get("normalized_expert_deformation")
        if normalized_pool_deformation is not None:
            normalized_deformation = float(max(float(normalized_pool_deformation), 0.0))
            expert_proximity = float(np.exp(-normalized_deformation / 0.15))
        elif deformation is not None and trust_radius is not None and float(trust_radius) > 0.0:
            normalized_deformation = float(np.clip(float(deformation) / float(trust_radius), 0.0, 1.0))
            expert_proximity = 1.0 - normalized_deformation
        else:
            normalized_deformation = None
            expert_proximity = 0.25

        if kind in {"model_false_unsafe", "boundary_uncertainty"}:
            estimated_safe_probability = (
                0.35 * posterior_safe_yield
                + 0.25 * safe_vote_fraction
                + 0.40 * expert_proximity
            )
        else:
            estimated_safe_probability = (
                0.55 * posterior_safe_yield
                + 0.35 * safe_vote_fraction
                + 0.10 * expert_proximity
            )
        estimated_safe_probability = float(np.clip(estimated_safe_probability, 0.0, 1.0))
        global_information = float(np.clip(candidate.acquisition_score, 0.0, 1.0))
        violation_information = 0.55 + 0.25 * global_information + 0.20 * float(safe_vote_fraction > 0.0)
        source_prediction = candidate.predictions.get(candidate.source.hypothesis_id)
        source_rejects = source_prediction == VIOLATION_LABEL
        causal_rejectors = sorted(
            map(str, candidate.result.trajectory.metadata.get("safe_query_causal_rejector_ids", []))
        )
        causal_signature = self._causal_rejection_signature(
            candidate,
            causal_rejectors,
        )
        source_anchor_margin_satisfied = bool(metadata.get("source_anchor_margin_satisfied", False))
        generation_hard_margin_satisfied = bool(
            metadata.get("generation_hard_margin_satisfied", False)
        )
        query_hard_margin_satisfied = bool(
            metadata.get("query_hard_margin_satisfied", False)
        )
        target_clause_resolved = bool(
            len(candidate.source.atomic_clauses()) <= 1
            or metadata.get("query_target_clause_id")
        )
        source_is_causal_rejector = candidate.source.hypothesis_id in causal_rejectors
        if (
            causal_rejectors
            and source_anchor_margin_satisfied
            and generation_hard_margin_satisfied
            and query_hard_margin_satisfied
            and target_clause_resolved
            and source_is_causal_rejector
        ):
            safe_tier = "causal_model_boundary_counterexample"
        elif causal_rejectors:
            safe_tier = "crossing_without_source_hard_margin"
        else:
            safe_tier = "noncausal_coverage_only"
        safe_query_eligible = bool(
            causal_rejectors
            and source_anchor_margin_satisfied
            and generation_hard_margin_satisfied
            and query_hard_margin_satisfied
            and target_clause_resolved
            and source_is_causal_rejector
        )
        causal_fraction = len(causal_rejectors) / max(len(predictions), 1)
        safe_information = (
            0.25 * global_information
            + 0.50 * causal_fraction
            + 0.25 * float(generation_hard_margin_satisfied)
        )
        return {
            "estimated_safe_probability": estimated_safe_probability,
            "estimated_violation_probability": 1.0 - estimated_safe_probability,
            "safe_vote_fraction": safe_vote_fraction,
            "unsafe_voter_hypothesis_ids": unsafe_voters,
            "prediction_rejection_signature": (
                causal_signature if safe_query_eligible else unsafe_voters
            ),
            "source_rejects_after_projection": source_rejects,
            "safe_query_causal_rejector_ids": causal_rejectors,
            "safe_query_eligible": safe_query_eligible,
            "source_anchor_margin_satisfied": source_anchor_margin_satisfied,
            "source_generation_hard_margin_satisfied": generation_hard_margin_satisfied,
            "source_query_hard_margin_satisfied": query_hard_margin_satisfied,
            "source_target_clause_resolved": target_clause_resolved,
            "source_is_causal_rejector_at_query": source_is_causal_rejector,
            "posterior_safe_yield_for_intervention": float(posterior_safe_yield),
            "historical_intervention_safe_count": historical_safe,
            "historical_intervention_violation_count": historical_violation,
            "normalized_expert_deformation": normalized_deformation,
            "expert_proximity": expert_proximity,
            "safe_query_tier": safe_tier,
            "safe_label_utility": estimated_safe_probability * safe_information,
            "violation_label_utility": (1.0 - estimated_safe_probability) * violation_information,
        }

    @staticmethod
    def _causal_rejection_signature(
        candidate: AcquisitionCandidate,
        causal_rejectors: list[str],
    ) -> list[str]:
        """Encode the causal source clause without including unrelated voters."""

        source_id = candidate.source.hypothesis_id
        source_clause_id = candidate.result.trajectory.metadata.get(
            "query_target_clause_id"
        )
        signature: list[str] = []
        for hypothesis_id in sorted(causal_rejectors):
            if hypothesis_id == source_id and source_clause_id:
                signature.append(f"{hypothesis_id}::{source_clause_id}")
            else:
                signature.append(hypothesis_id)
        return signature

    @staticmethod
    def _candidate_is_queryable(candidate: AcquisitionCandidate) -> bool:
        """Do not spend Oracle budget on a failed false-unsafe synthesis."""

        if candidate.intervention.kind != "model_false_unsafe":
            return True
        metadata = candidate.result.trajectory.metadata
        return bool(
            metadata.get("source_anchor_margin_satisfied", False)
            and metadata.get("generation_hard_margin_satisfied", False)
            and metadata.get("query_hard_margin_satisfied", False)
            and (
                len(candidate.source.atomic_clauses()) <= 1
                or metadata.get("query_target_clause_id")
            )
            and candidate.source.hypothesis_id
            in metadata.get("safe_query_causal_rejector_ids", [])
        )

    def _duplicates_selected(
        self,
        candidate: AcquisitionCandidate,
        selected: list[AcquisitionCandidate],
    ) -> bool:
        return any(
            self._trajectory_rms(candidate.result.trajectory, item.result.trajectory)
            < self.config.candidate_deduplication_rms
            for item in selected
        )

    def _duplicates_query_history(self, candidate: AcquisitionCandidate) -> bool:
        threshold = float(self.config.candidate_history_deduplication_rms)
        if threshold <= 0.0:
            return False
        return any(
            self._trajectory_rms(candidate.result.trajectory, record.trajectory) < threshold
            for record in self.buffer.records
        )

    def _refine_false_unsafe_to_nearest_boundary(
        self,
        result: FalsifierResult,
        expert: Trajectory,
        active: list[ConstraintHypothesis],
    ) -> FalsifierResult:
        """Shrink a safe-anchor probe to the first frozen-model boundary.

        The numeric falsifier may push all the way to its trust-region shell.
        That is useful for finding violations, but poor for discovering model
        false-unsafes.  Because the expert endpoint is known safe, a bisection
        along the homotopy from expert to optimized endpoint finds the minimum
        deformation that any active model rejects, without consulting Oracle.
        """

        if bool(
            getattr(
                self.falsifier,
                "uses_validated_global_rollout_pool",
                False,
            )
        ):
            return self._certify_pool_false_unsafe_endpoint(result, expert, active)
        steps = max(0, int(self.config.safe_query_boundary_bisection_steps))
        if steps <= 0:
            return result
        library = getattr(self, "library", None)
        if library is not None and not library.is_planar:
            result.trajectory.metadata["boundary_refinement_status"] = (
                "disabled_for_nonplanar_adapter"
            )
            return result
        expert_states = np.asarray(expert.states, dtype=np.float32)
        endpoint_states = np.asarray(result.trajectory.states, dtype=np.float32)
        if expert_states.shape != endpoint_states.shape or np.allclose(expert_states, endpoint_states):
            return result

        def trajectory_at(alpha: float) -> Trajectory:
            states = (expert_states + float(alpha) * (endpoint_states - expert_states)).astype(np.float32)
            return Trajectory(states, displacement_actions(states), expert.dt)

        endpoint_predictions: dict[str, int] = {}
        expert_predictions: dict[str, int] = {}
        expert_crossing_scores: dict[str, float] = {}
        endpoint_crossing_scores: dict[str, float] = {}
        crossings: list[tuple[float, str, str | None, float, float]] = []
        endpoint = trajectory_at(1.0)
        scan_points = max(2, int(self.config.safe_query_boundary_scan_points))
        boundary_margin = max(0.0, float(self.config.safe_query_boundary_margin))
        source_target_clause_id_raw = result.trajectory.metadata.get(
            "optimization_target_clause_id"
        )
        source_target_clause_id = (
            str(source_target_clause_id_raw)
            if source_target_clause_id_raw not in {None, ""}
            else None
        )
        source_hypothesis = next(
            (
                hypothesis
                for hypothesis in active
                if hypothesis.hypothesis_id == result.hypothesis_id
            ),
            None,
        )
        if (
            source_hypothesis is not None
            and len(source_hypothesis.atomic_clauses()) > 1
            and source_target_clause_id is None
        ):
            result.trajectory.metadata.update(
                {
                    "safe_query_causal_rejector_ids": [],
                    "query_target_clause_id": None,
                    "query_hard_margin_satisfied": False,
                    "boundary_refinement_status": "missing_target_clause_for_composite",
                }
            )
            return result

        def decision_threshold(hypothesis_id: str) -> float:
            models = getattr(self.registry, "models", {})
            ensemble = models.get(hypothesis_id) if hasattr(models, "get") else None
            return (
                float(ensemble.decision_threshold.item())
                if ensemble is not None
                else 0.0
            )

        def crossing_clause_id(hypothesis_id: str) -> str | None:
            return (
                source_target_clause_id
                if hypothesis_id == result.hypothesis_id
                else None
            )

        def observe_crossing(
            hypothesis_id: str,
            trajectory: Trajectory,
        ) -> tuple[int, float, float, float]:
            prediction, full_score, uncertainty = self.registry.predict(
                hypothesis_id,
                trajectory,
            )
            clause_id = crossing_clause_id(hypothesis_id)
            crossing_score = float(full_score)
            if clause_id is not None:
                clause_scorer = getattr(self.registry, "hard_clause_score", None)
                if not callable(clause_scorer):
                    raise RuntimeError(
                        "clause-targeted boundary refinement requires "
                        "registry.hard_clause_score"
                    )
                crossing_score = float(
                    clause_scorer(hypothesis_id, trajectory, clause_id)
                )
            return prediction, float(full_score), crossing_score, float(uncertainty)

        for hypothesis in active:
            hypothesis_id = hypothesis.hypothesis_id
            expert_prediction, expert_full_score, expert_crossing_score, _ = observe_crossing(
                hypothesis_id,
                expert,
            )
            endpoint_prediction, _, endpoint_crossing_score, _ = observe_crossing(
                hypothesis_id,
                endpoint,
            )
            expert_predictions[hypothesis_id] = expert_prediction
            endpoint_predictions[hypothesis_id] = endpoint_prediction
            expert_crossing_scores[hypothesis_id] = expert_crossing_score
            endpoint_crossing_scores[hypothesis_id] = endpoint_crossing_score
            if expert_prediction != SAFE_LABEL:
                continue
            target_score = decision_threshold(hypothesis_id) + boundary_margin
            if expert_full_score >= target_score or expert_crossing_score >= target_score:
                continue
            low = 0.0
            low_meets_target = False
            bracket: tuple[float, float] | None = None
            for scan_index in range(1, scan_points + 1):
                high = scan_index / scan_points
                high_prediction, _, high_crossing_score, _ = observe_crossing(
                    hypothesis_id,
                    trajectory_at(high),
                )
                high_meets_target = bool(
                    high_prediction == VIOLATION_LABEL
                    and high_crossing_score >= target_score
                )
                if not low_meets_target and high_meets_target:
                    bracket = (low, high)
                    break
                low = high
                low_meets_target = high_meets_target
            if bracket is None:
                continue
            low, high = bracket
            coarse_high = high
            for _ in range(steps):
                middle = 0.5 * (low + high)
                middle_prediction, _, middle_crossing_score, _ = observe_crossing(
                    hypothesis_id,
                    trajectory_at(middle),
                )
                if (
                    middle_crossing_score >= target_score
                    and middle_prediction == VIOLATION_LABEL
                ):
                    high = middle
                else:
                    low = middle
            crossings.append(
                (
                    high,
                    hypothesis_id,
                    crossing_clause_id(hypothesis_id),
                    target_score,
                    coarse_high,
                )
            )

        endpoint_crossing_ids = sorted(
            hypothesis_id for _, hypothesis_id, _, _, _ in crossings
        )
        source_crossings = [
            crossing
            for crossing in crossings
            if crossing[1] == result.hypothesis_id
        ]
        refinement_status = (
            "crossing_found"
            if source_crossings
            else (
                "no_source_safe_to_target_clause_crossing"
                if crossings
                else "no_safe_to_unsafe_crossing"
            )
        )
        result.trajectory.metadata.update(
            {
                "expert_predictions_before_boundary_refinement": expert_predictions,
                "endpoint_predictions_before_boundary_refinement": endpoint_predictions,
                "expert_crossing_scores_before_boundary_refinement": expert_crossing_scores,
                "endpoint_crossing_scores_before_boundary_refinement": endpoint_crossing_scores,
                "safe_query_endpoint_crossing_ids": endpoint_crossing_ids,
                "safe_query_causal_rejector_ids": [],
                "boundary_scan_points": scan_points,
                "boundary_refinement_margin": boundary_margin,
                "query_target_clause_id": source_target_clause_id,
                "query_hard_margin_target": decision_threshold(result.hypothesis_id)
                + boundary_margin,
                "query_hard_margin_satisfied": False,
                "boundary_refinement_status": refinement_status,
            }
        )
        if not source_crossings:
            return result
        (
            boundary_alpha,
            trigger_hypothesis_id,
            trigger_clause_id,
            trigger_target_score,
            coarse_high,
        ) = min(
            source_crossings,
            key=lambda item: item[0],
        )
        refined_alpha = boundary_alpha
        refined = trajectory_at(refined_alpha)
        (
            trigger_prediction,
            trigger_full_score,
            trigger_crossing_score,
            _,
        ) = observe_crossing(
            trigger_hypothesis_id,
            refined,
        )
        if (
            trigger_prediction != VIOLATION_LABEL
            or trigger_crossing_score < trigger_target_score
        ):
            # The coarse endpoint is known to attain the requested margin and
            # is a safe fallback for rare float32 interpolation roundoff.
            refined_alpha = coarse_high
            refined = trajectory_at(refined_alpha)
            (
                trigger_prediction,
                trigger_full_score,
                trigger_crossing_score,
                _,
            ) = observe_crossing(
                trigger_hypothesis_id,
                refined,
            )
            if (
                trigger_prediction != VIOLATION_LABEL
                or trigger_crossing_score < trigger_target_score
            ):
                result.trajectory.metadata["boundary_refinement_status"] = "roundoff_lost_rejection"
                return result
        valid, validation_reason = self.falsifier.validate(refined, expert)
        if not valid:
            result.trajectory.metadata["boundary_refinement_status"] = f"refined_candidate_invalid:{validation_reason}"
            return result

        original_maximum_deviation = float(
            np.max(np.linalg.norm(endpoint_states - expert_states, axis=1))
        )
        maximum_deviation = float(
            np.max(np.linalg.norm(refined.states - expert_states, axis=1))
        )
        causal_rejectors: list[str] = []
        causal_rejector_scores: dict[str, float] = {}
        causal_rejector_margins: dict[str, float] = {}
        source_full_score = float("nan")
        source_target_score = float("nan")
        source_uncertainty = float("nan")
        for hypothesis in active:
            hypothesis_id = hypothesis.hypothesis_id
            expert_prediction = expert_predictions[hypothesis_id]
            (
                refined_prediction,
                refined_full_score,
                refined_crossing_score,
                refined_uncertainty,
            ) = observe_crossing(
                hypothesis_id,
                refined,
            )
            threshold = decision_threshold(hypothesis_id)
            target_score = threshold + boundary_margin
            if (
                expert_prediction == SAFE_LABEL
                and refined_prediction == VIOLATION_LABEL
                and refined_crossing_score >= target_score
            ):
                causal_rejectors.append(hypothesis_id)
                causal_rejector_scores[hypothesis_id] = refined_crossing_score
                causal_rejector_margins[hypothesis_id] = (
                    min(refined_full_score, refined_crossing_score) - threshold
                )
            if hypothesis_id == result.hypothesis_id:
                source_full_score = refined_full_score
                source_target_score = refined_crossing_score
                source_uncertainty = refined_uncertainty
        source_threshold = decision_threshold(result.hypothesis_id)
        source_achieved_margin = (
            min(source_full_score, source_target_score) - source_threshold
        )
        source_margin_satisfied = bool(
            result.hypothesis_id in causal_rejectors
            and source_achieved_margin >= boundary_margin
        )
        metadata = dict(result.trajectory.metadata)
        metadata.update(
            {
                "boundary_refined": True,
                "boundary_refinement_alpha": float(refined_alpha),
                "boundary_trigger_hypothesis_id": trigger_hypothesis_id,
                "boundary_trigger_clause_id": trigger_clause_id,
                "boundary_trigger_score": float(trigger_crossing_score),
                "boundary_trigger_full_score": float(trigger_full_score),
                "boundary_trigger_target_score": float(trigger_target_score),
                "boundary_refinement_status": "refined_to_nearest_model_boundary",
                "pre_refinement_max_expert_deviation": original_maximum_deviation,
                "max_expert_deviation": maximum_deviation,
                "safe_query_causal_rejector_ids": causal_rejectors,
                "query_causal_rejector_scores": causal_rejector_scores,
                "query_causal_rejector_margins": causal_rejector_margins,
                "query_source_full_hard_score": source_full_score,
                "query_source_target_hard_score": source_target_score,
                "query_achieved_hard_margin": source_achieved_margin,
                "query_hard_margin_satisfied": source_margin_satisfied,
            }
        )
        refined.metadata.update(metadata)
        return replace(
            result,
            trajectory=refined,
            final_score=float(source_full_score),
            final_uncertainty=float(source_uncertainty),
            valid=True,
            validation_reason=f"{validation_reason};nearest_model_boundary",
        )

    def _certify_pool_false_unsafe_endpoint(
        self,
        result: FalsifierResult,
        expert: Trajectory,
        active: list[ConstraintHypothesis],
    ) -> FalsifierResult:
        """Certify an unchanged, dynamics-valid public rollout at query time.

        A global public-pool rollout cannot be interpolated with an unrelated
        expert without breaking the task dynamics.  Unlike a generic heuristic
        endpoint, however, the queried trajectory is exactly the rollout whose
        generation hard score was recorded.  Re-evaluate that same rollout with
        every frozen model and attach the query-side certificate expected by
        acquisition; no Oracle label is used here.
        """

        metadata = result.trajectory.metadata
        target_clause_raw = metadata.get("optimization_target_clause_id")
        target_clause_id = (
            str(target_clause_raw)
            if target_clause_raw not in {None, ""}
            else None
        )
        source_hypothesis = next(
            (
                hypothesis
                for hypothesis in active
                if hypothesis.hypothesis_id == result.hypothesis_id
            ),
            None,
        )
        if (
            source_hypothesis is None
            or (
                len(source_hypothesis.atomic_clauses()) > 1
                and target_clause_id is None
            )
        ):
            metadata.update(
                {
                    "safe_query_causal_rejector_ids": [],
                    "query_target_clause_id": target_clause_id,
                    "query_hard_margin_satisfied": False,
                    "boundary_refinement_status": (
                        "public_pool_missing_source_or_target_clause"
                    ),
                }
            )
            return result

        margin = max(0.0, float(self.config.safe_query_boundary_margin))
        causal_rejectors: list[str] = []
        expert_predictions: dict[str, int] = {}
        endpoint_predictions: dict[str, int] = {}
        expert_crossing_scores: dict[str, float] = {}
        endpoint_crossing_scores: dict[str, float] = {}
        endpoint_full_scores: dict[str, float] = {}
        for hypothesis in active:
            hypothesis_id = hypothesis.hypothesis_id
            threshold = float(
                self.registry.models[hypothesis_id].decision_threshold.item()
            )
            expert_prediction, expert_full_score, _ = self.registry.predict(
                hypothesis_id,
                expert,
            )
            endpoint_prediction, endpoint_full_score, _ = self.registry.predict(
                hypothesis_id,
                result.trajectory,
            )
            crossing_clause_id = (
                target_clause_id
                if hypothesis_id == result.hypothesis_id
                else None
            )
            if crossing_clause_id is None:
                expert_crossing_score = float(expert_full_score)
                endpoint_crossing_score = float(endpoint_full_score)
            else:
                expert_crossing_score = float(
                    self.registry.hard_clause_score(
                        hypothesis_id,
                        expert,
                        crossing_clause_id,
                    )
                )
                endpoint_crossing_score = float(
                    self.registry.hard_clause_score(
                        hypothesis_id,
                        result.trajectory,
                        crossing_clause_id,
                    )
                )
            target = threshold + margin
            if (
                expert_prediction == SAFE_LABEL
                and float(expert_full_score) < target
                and expert_crossing_score < target
                and endpoint_prediction == VIOLATION_LABEL
                and float(endpoint_full_score) >= target
                and endpoint_crossing_score >= target
            ):
                causal_rejectors.append(hypothesis_id)
            expert_predictions[hypothesis_id] = int(expert_prediction)
            endpoint_predictions[hypothesis_id] = int(endpoint_prediction)
            expert_crossing_scores[hypothesis_id] = expert_crossing_score
            endpoint_crossing_scores[hypothesis_id] = endpoint_crossing_score
            endpoint_full_scores[hypothesis_id] = float(endpoint_full_score)

        source_threshold = float(
            self.registry.models[result.hypothesis_id].decision_threshold.item()
        )
        source_full_score = endpoint_full_scores[result.hypothesis_id]
        source_target_score = endpoint_crossing_scores[result.hypothesis_id]
        hard_margin_satisfied = bool(
            result.hypothesis_id in causal_rejectors
            and source_full_score >= source_threshold + margin
            and source_target_score >= source_threshold + margin
        )
        metadata.update(
            {
                "expert_predictions_before_boundary_refinement": expert_predictions,
                "endpoint_predictions_before_boundary_refinement": endpoint_predictions,
                "expert_crossing_scores_before_boundary_refinement": expert_crossing_scores,
                "endpoint_crossing_scores_before_boundary_refinement": endpoint_crossing_scores,
                "safe_query_endpoint_crossing_ids": sorted(causal_rejectors),
                "safe_query_causal_rejector_ids": sorted(causal_rejectors),
                "boundary_refined": False,
                "boundary_refinement_alpha": 1.0,
                "boundary_refinement_status": "public_pool_endpoint_certified",
                "query_target_clause_id": target_clause_id,
                "query_hard_margin_target": source_threshold + margin,
                "query_source_full_hard_score": source_full_score,
                "query_source_target_hard_score": source_target_score,
                "query_achieved_hard_margin": float(
                    min(source_full_score, source_target_score) - source_threshold
                ),
                "query_hard_margin_satisfied": hard_margin_satisfied,
            }
        )
        return result

    def _query_sources(self, active: list[ConstraintHypothesis]) -> list[ConstraintHypothesis]:
        if not self.query_priorities:
            return active
        maximum = max(self.query_priorities.values(), default=0.0)

        def priority(hypothesis: ConstraintHypothesis) -> float:
            # Newly synthesized hypotheses receive an exploration bonus because
            # no prequential evidence can exist for them yet.
            return self.query_priorities.get(hypothesis.hypothesis_id, maximum + 0.05)

        ordered = sorted(active, key=priority, reverse=True)
        selected = ordered[: max(1, min(self.config.query_hypothesis_beam, len(ordered)))]
        selected_ids = {item.hypothesis_id for item in selected}
        for target_id in self.pending_interventions:
            if target_id not in selected_ids:
                target = next((item for item in active if item.hypothesis_id == target_id), None)
                if target is not None:
                    selected.append(target)
                    selected_ids.add(target_id)
        return selected

    def _acquisition_score(
        self,
        source_hypothesis_id: str,
        trajectory: Trajectory,
        predictions: dict[str, int],
        scores: dict[str, float],
        uncertainties: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        votes = np.asarray(list(predictions.values()), dtype=np.float64)
        vote_mean = float(np.mean(votes)) if len(votes) else 0.5
        disagreement = 4.0 * vote_mean * (1.0 - vote_mean)
        calibrated_margins = [
            abs(
                float(value)
                - float(self.registry.models[hypothesis_id].decision_threshold.item())
            )
            for hypothesis_id, value in scores.items()
        ]
        boundary = float(np.exp(-min(calibrated_margins, default=0.0)))
        uncertainty = float(np.tanh(np.mean(list(uncertainties.values())))) if uncertainties else 0.0
        novelty = self._novelty(trajectory)
        potential = float(np.clip(self.query_priorities.get(source_hypothesis_id, 0.5), 0.0, 1.0))
        components = {
            "cross_hypothesis_disagreement": disagreement,
            "nearest_boundary": boundary,
            "ensemble_uncertainty": uncertainty,
            "trajectory_novelty": novelty,
            "source_potential": potential,
        }
        score = (
            self.config.acquisition_disagreement_weight * disagreement
            + self.config.acquisition_boundary_weight * boundary
            + self.config.acquisition_uncertainty_weight * uncertainty
            + self.config.acquisition_novelty_weight * novelty
            + self.config.acquisition_potential_weight * potential
        )
        return float(score), components

    def _novelty(self, trajectory: Trajectory) -> float:
        if not self.buffer.records:
            return 1.0
        nearest = min(self._trajectory_rms(trajectory, record.trajectory) for record in self.buffer.records)
        library = getattr(self, "library", None)
        if library is not None and not library.is_planar:
            # `_trajectory_rms` is already dimensionless here because every
            # learner-visible feature is divided by its public range.
            return float(np.clip(nearest / 0.15, 0.0, 1.0))
        workspace_scale = float(np.hypot(self.workspace_x[1] - self.workspace_x[0], self.workspace_y[1] - self.workspace_y[0]))
        return float(np.clip(nearest / max(0.1 * workspace_scale, 1.0e-6), 0.0, 1.0))

    def _trajectory_rms(self, left: Trajectory, right: Trajectory) -> float:
        if left.states.shape != right.states.shape:
            return float("inf")
        library = getattr(self, "library", None)
        if library is None or library.is_planar:
            values = left.states - right.states
        else:
            variables = library.names
            left_features = library.numpy_features(left.states, variables)
            right_features = library.numpy_features(right.states, variables)
            low, high = library.bounds(variables)
            scale = np.maximum(np.asarray(high) - np.asarray(low), 1.0e-6)
            values = (left_features - right_features) / scale
        return float(np.sqrt(np.mean(np.sum(values**2, axis=-1))))

    def _consume_pending(self, intervention: InterventionSpec) -> None:
        pending = self.pending_interventions.get(intervention.target_hypothesis_id, [])
        for index, item in enumerate(pending):
            if item == intervention:
                pending.pop(index)
                break

    @staticmethod
    def _trajectory_summary(trajectory: Trajectory) -> dict[str, object]:
        states = trajectory.states
        position_dimension = 3 if states.shape[1] == 12 else 2
        steps = np.linalg.norm(np.diff(states[:, :position_dimension], axis=0), axis=1)
        payload: dict[str, object] = {
            "trajectory_id": trajectory.metadata.get("trajectory_id"),
            "source": trajectory.metadata.get("source"),
            "observation_dimension": int(states.shape[1]),
            "action_dimension": (
                None if trajectory.actions is None else int(trajectory.actions.shape[1])
            ),
            "start": [float(value) for value in states[0]],
            "goal": [float(value) for value in states[-1]],
            "midpoint": [float(value) for value in states[len(states) // 2]],
            "x_min": float(np.min(states[:, 0])),
            "x_max": float(np.max(states[:, 0])),
            "y_min": float(np.min(states[:, 1])),
            "y_max": float(np.max(states[:, 1])),
            "path_length": float(np.sum(steps)),
            "max_step": float(np.max(steps)) if len(steps) else 0.0,
        }
        if states.shape[1] == 12:
            payload.update(
                {
                    "z_min": float(np.min(states[:, 2])),
                    "z_max": float(np.max(states[:, 2])),
                    "target_dz_abs_max": float(np.max(np.abs(states[:, 5]))),
                    "speed_max": float(np.max(np.linalg.norm(states[:, 6:9], axis=1))),
                    "tilt_from_vertical_max": float(
                        np.max(
                            np.arccos(
                                np.clip(
                                    np.cos(states[:, 9]) * np.cos(states[:, 10]),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    ),
                }
            )
        return payload

    def _safe_prediction_summary(
        self,
        hypothesis_id: str,
        trajectories: list[Trajectory],
    ) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for trajectory in trajectories:
            prediction, score, uncertainty = self.registry.predict(hypothesis_id, trajectory)
            row = self._trajectory_summary(trajectory)
            row.update(
                {
                    "predicted_label": prediction,
                    "predicted_safe": prediction == SAFE_LABEL,
                    "score": score,
                    "uncertainty": uncertainty,
                }
            )
            rows.append(row)
        return {
            "count": len(rows),
            "safe_rate": float(np.mean([row["predicted_safe"] for row in rows])) if rows else None,
            "trajectories": rows,
        }

    def _raw_hard_score_for_ensemble(self, ensemble: object, trajectory: Trajectory) -> float:
        features = self.library.torch_features(
            torch.as_tensor(trajectory.states, dtype=torch.float32, device=self.device),
            ensemble.compiled.variables,
        )
        with torch.no_grad():
            return float(ensemble.mean_hard_trajectory_score(features).item())

    def _finalize_champion(self, champion_id: str) -> bool:
        """Train several candidates and atomically commit one on disjoint evidence.

        Gradient fitting excludes both splits. ``final_calibration`` chooses a
        threshold for each candidate; ``warmup_validation`` then decides which
        already-calibrated candidate may replace the incumbent.  This avoids
        selecting a random restart merely because it overfits the four examples
        used to choose its threshold.
        """

        calibration_records = [
            record for record in self.buffer.records if record.source == "final_calibration"
        ]
        selection_records = [
            record for record in self.buffer.records if record.source == "warmup_validation"
        ]
        final_training_records = [
            record
            for record in self.buffer.records
            if record.source not in {"warmup_validation", "final_calibration"}
        ]
        all_train_experts = [*self.experts, *self.structure_audit_experts]
        incumbent = self.registry.models[champion_id]
        calibration_labels = [record.label for record in calibration_records]
        selection_labels = [record.label for record in selection_records]
        incumbent_calibration_scores = [
            self._raw_hard_score_for_ensemble(incumbent, record.trajectory)
            for record in calibration_records
        ]
        incumbent_selection_scores = [
            self._raw_hard_score_for_ensemble(incumbent, record.trajectory)
            for record in selection_records
        ]
        incumbent_threshold = float(incumbent.decision_threshold.item())
        incumbent_calibration = self._score_threshold(
            incumbent_calibration_scores, calibration_labels, incumbent_threshold
        )
        incumbent_selection = self._score_threshold(
            incumbent_selection_scores, selection_labels, incumbent_threshold
        )
        final_trainer_config = replace(
            self.trainer_config,
            epochs=self.config.finalization_epochs,
            bootstrap_queries=False,
            latent_witness_weight=(
                0.0
                if self.config.finalization_disable_latent_witness
                else self.trainer_config.latent_witness_weight
            ),
        )
        label_counts = {
            "safe": sum(label == SAFE_LABEL for label in calibration_labels),
            "violation": sum(label == VIOLATION_LABEL for label in calibration_labels),
        }
        selection_label_counts = {
            "safe": sum(label == SAFE_LABEL for label in selection_labels),
            "violation": sum(label == VIOLATION_LABEL for label in selection_labels),
        }
        sample_rejection_reasons: list[str] = []
        if min(label_counts.values(), default=0) < self.config.finalization_minimum_calibration_per_label:
            sample_rejection_reasons.append("insufficient_calibration_examples_per_label")
        if (
            min(selection_label_counts.values(), default=0)
            < self.config.finalization_minimum_selection_per_label
        ):
            sample_rejection_reasons.append("insufficient_selection_holdout_examples_per_label")
        if sample_rejection_reasons:
            self.registry.models[champion_id] = incumbent
            self.finalization_diagnostics = {
                "attempted": True,
                "applied": False,
                "rejection_reasons": sample_rejection_reasons,
                "champion_hypothesis_id": champion_id,
                "structure_frozen": True,
                "all_train_expert_count": len(all_train_experts),
                "gradient_training_query_count": len(final_training_records),
                "calibration_query_count": len(calibration_records),
                "calibration_label_counts": label_counts,
                "selection_holdout_query_count": len(selection_records),
                "selection_holdout_label_counts": selection_label_counts,
                "calibration_queries_excluded_from_gradient_training": True,
                "selection_holdout_excluded_from_gradient_and_threshold_calibration": True,
                "candidate_count": 0,
                "selected_candidate": None,
                "incumbent_calibration_scores": incumbent_calibration_scores,
                "incumbent_calibration": incumbent_calibration,
                "incumbent_selection_scores": incumbent_selection_scores,
                "incumbent_selection": incumbent_selection,
                "decision_threshold": incumbent_threshold,
            }
            self.progress(
                f"finalization champion={champion_id} committed=False "
                f"calibration_labels={label_counts} selection_labels={selection_label_counts} "
                f"reasons={sample_rejection_reasons}"
            )
            return False

        candidates: list[tuple[str, object, TrainingSummary | None]] = [
            ("incumbent_calibrated", deepcopy(incumbent), None)
        ]
        if self.config.finalization_finetune_incumbent:
            for background_index, background_weight in enumerate(
                self.config.finalization_finetune_background_weights
            ):
                finetuned = deepcopy(incumbent)
                finetuned.set_decision_threshold(0.0)
                finetune_config = replace(
                    final_trainer_config,
                    epochs=self.config.finalization_finetune_epochs,
                    learning_rate=(
                        self.trainer_config.learning_rate
                        * self.config.finalization_finetune_learning_rate_scale
                    ),
                    background_safe_weight=float(background_weight),
                )
                finetune_summary = fit_ensemble(
                    finetuned,
                    all_train_experts,
                    final_training_records,
                    self.library,
                    finetune_config,
                    seed=self.seed + 88001 + 701 * background_index,
                    device=self.device,
                )
                suffix = str(float(background_weight)).replace(".", "p")
                candidates.append(
                    (f"incumbent_finetune_bg_{suffix}", finetuned, finetune_summary)
                )
        for restart in range(max(0, self.config.finalization_scratch_restarts)):
            scratch = self.registry.reinitialize(
                champion_id,
                seed_offset=104729 + 10007 * restart,
            )
            scratch_summary = fit_ensemble(
                scratch,
                all_train_experts,
                final_training_records,
                self.library,
                final_trainer_config,
                seed=self.seed + 99991 + 1009 * restart,
                device=self.device,
            )
            candidates.append((f"scratch_restart_{restart}", scratch, scratch_summary))

        candidate_rows: list[dict[str, object]] = []
        candidate_models: dict[str, object] = {}
        for candidate_name, ensemble, summary in candidates:
            row = self._evaluate_finalization_candidate(
                candidate_name,
                ensemble,
                summary,
                calibration_records,
                selection_records,
                all_train_experts,
                incumbent_calibration,
                incumbent_selection,
            )
            candidate_rows.append(row)
            candidate_models[candidate_name] = ensemble
        eligible_rows = [row for row in candidate_rows if bool(row["eligible"])]
        selected_row = self._choose_finalization_candidate(eligible_rows)
        committed = selected_row is not None
        if committed:
            selected_name = str(selected_row["candidate_name"])
            self.registry.models[champion_id] = candidate_models[selected_name]
            rejection_reasons: list[str] = []
        else:
            selected_name = None
            self.registry.models[champion_id] = incumbent
            rejection_reasons = ["no_candidate_passed_commit_gates"]
        self.finalization_diagnostics = {
            "attempted": True,
            "applied": committed,
            "rejection_reasons": rejection_reasons,
            "champion_hypothesis_id": champion_id,
            "structure_frozen": True,
            "all_train_expert_count": len(all_train_experts),
            "gradient_training_query_count": len(final_training_records),
            "calibration_query_count": len(calibration_records),
            "calibration_label_counts": label_counts,
            "selection_holdout_query_count": len(selection_records),
            "selection_holdout_label_counts": selection_label_counts,
            "calibration_queries_excluded_from_gradient_training": True,
            "selection_holdout_excluded_from_gradient_and_threshold_calibration": True,
            "candidate_count": len(candidate_rows),
            "selected_candidate": selected_name,
            "candidates": candidate_rows,
            "incumbent_calibration_scores": incumbent_calibration_scores,
            "incumbent_calibration": incumbent_calibration,
            "incumbent_selection_scores": incumbent_selection_scores,
            "incumbent_selection": incumbent_selection,
            "decision_threshold": float(self.registry.models[champion_id].decision_threshold.item()),
        }
        self.progress(
            f"finalization champion={champion_id} committed={committed} experts={len(all_train_experts)} "
            f"calibration_labels={label_counts} selection_labels={selection_label_counts} "
            f"selected={selected_name} "
            f"threshold={float(self.registry.models[champion_id].decision_threshold.item()):.4f} "
            f"reasons={rejection_reasons}"
        )
        return committed

    def _evaluate_finalization_candidate(
        self,
        candidate_name: str,
        ensemble: object,
        summary: TrainingSummary | None,
        calibration_records: list[QueryRecord],
        selection_records: list[QueryRecord],
        all_train_experts: list[Trajectory],
        incumbent_calibration: dict[str, float],
        incumbent_selection: dict[str, float],
    ) -> dict[str, object]:
        calibration_scores = [
            self._raw_hard_score_for_ensemble(ensemble, record.trajectory)
            for record in calibration_records
        ]
        calibration_labels = [record.label for record in calibration_records]
        expert_scores = [
            self._raw_hard_score_for_ensemble(ensemble, expert) for expert in all_train_experts
        ]
        has_calibration_labels = (
            SAFE_LABEL in calibration_labels and VIOLATION_LABEL in calibration_labels
        )
        if has_calibration_labels:
            calibration = choose_decision_threshold(
                calibration_scores,
                calibration_labels,
                expert_scores,
                minimum_expert_safe_rate=self.config.finalization_minimum_expert_safe_rate,
            )
            ensemble.set_decision_threshold(float(calibration["selected_threshold"]))
        else:
            calibration = {
                "selected_threshold": 0.0,
                "selected_metrics": None,
                "expert_constraint_satisfied": False,
                "candidate_count": 0,
            }
        threshold = float(ensemble.decision_threshold.item())
        selection_scores = [
            self._raw_hard_score_for_ensemble(ensemble, record.trajectory)
            for record in selection_records
        ]
        selection_labels = [record.label for record in selection_records]
        selection_metrics = self._score_threshold(selection_scores, selection_labels, threshold)
        unsafe_probe_fraction = self._unsafe_probe_fraction(ensemble)
        selected_metrics = calibration.get("selected_metrics")
        rejection_reasons: list[str] = []
        if not isinstance(selected_metrics, dict):
            rejection_reasons.append("missing_both_calibration_labels")
        else:
            self._append_finalization_metric_rejections(
                rejection_reasons, "calibration", selected_metrics
            )
            if (
                float(selected_metrics["balanced_accuracy"])
                + self.config.finalization_allowed_calibration_drop
                < float(incumbent_calibration["balanced_accuracy"])
            ):
                rejection_reasons.append("calibration_worse_than_incumbent")
        if SAFE_LABEL not in selection_labels or VIOLATION_LABEL not in selection_labels:
            rejection_reasons.append("missing_both_selection_holdout_labels")
        else:
            self._append_finalization_metric_rejections(
                rejection_reasons, "selection_holdout", selection_metrics
            )
            if (
                float(selection_metrics["balanced_accuracy"])
                + self.config.finalization_allowed_calibration_drop
                < float(incumbent_selection["balanced_accuracy"])
            ):
                rejection_reasons.append("selection_holdout_worse_than_incumbent")
        if not bool(calibration.get("expert_constraint_satisfied", False)):
            rejection_reasons.append("train_expert_safety_constraint_unsatisfied")
        return {
            "candidate_name": candidate_name,
            "eligible": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
            "training_loss": None if summary is None else summary.mean_final_loss,
            "decision_threshold": threshold,
            "calibration_scores": calibration_scores,
            "calibration": calibration,
            "selection_holdout_scores": selection_scores,
            "selection_holdout": selection_metrics,
            "expert_scores": expert_scores,
            "unsafe_probe_fraction": unsafe_probe_fraction,
        }

    def _unsafe_probe_fraction(self, ensemble: object) -> float | None:
        count = int(self.config.finalization_unsafe_volume_probe_count)
        if count <= 0:
            return None
        lows, highs = self.library.bounds(ensemble.compiled.variables)
        generator = np.random.default_rng(self.seed + 314159)
        probe = generator.uniform(
            np.asarray(lows, dtype=np.float32),
            np.asarray(highs, dtype=np.float32),
            size=(count, len(lows)),
        ).astype(np.float32)
        features = torch.as_tensor(probe, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            scores = ensemble.mean_state_score(features)
        threshold = float(ensemble.decision_threshold.item())
        return float(torch.mean((scores > threshold).float()).item())

    def _append_finalization_metric_rejections(
        self,
        reasons: list[str],
        prefix: str,
        metrics: dict[str, object],
    ) -> None:
        if float(metrics["safe_accuracy"]) < self.config.finalization_minimum_calibration_safe_accuracy:
            reasons.append(f"{prefix}_safe_accuracy_below_gate")
        if float(metrics["violation_recall"]) < self.config.finalization_minimum_calibration_violation_recall:
            reasons.append(f"{prefix}_violation_recall_below_gate")
        if float(metrics["balanced_accuracy"]) < self.config.finalization_minimum_calibration_balanced_accuracy:
            reasons.append(f"{prefix}_balanced_accuracy_below_gate")

    @staticmethod
    def _choose_finalization_candidate(
        candidates: list[dict[str, object]],
    ) -> dict[str, object] | None:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda row: (
                float(row["selection_holdout"]["balanced_accuracy"]),
                float(row["selection_holdout"]["violation_recall"]),
                float(row["calibration"]["selected_metrics"]["balanced_accuracy"]),
                float(row["calibration"]["selected_metrics"]["violation_recall"]),
                -float(row.get("unsafe_probe_fraction") or 0.0),
                (
                    2
                    if str(row["candidate_name"]).startswith("incumbent_finetune")
                    else 1
                    if row["candidate_name"] == "incumbent_calibrated"
                    else 0
                ),
            ),
        )

    @classmethod
    def _score_threshold(
        cls,
        scores: list[float],
        labels: list[int],
        threshold: float,
    ) -> dict[str, float]:
        predictions = [int(score > threshold) for score in scores]
        return cls._binary_calibration_metrics(predictions, labels)

    @staticmethod
    def _binary_calibration_metrics(predictions: list[int], labels: list[int]) -> dict[str, float]:
        safe = [prediction == SAFE_LABEL for prediction, label in zip(predictions, labels) if label == SAFE_LABEL]
        violation = [
            prediction == VIOLATION_LABEL
            for prediction, label in zip(predictions, labels)
            if label == VIOLATION_LABEL
        ]
        safe_accuracy = float(np.mean(safe)) if safe else 0.0
        violation_recall = float(np.mean(violation)) if violation else 0.0
        return {
            "safe_accuracy": safe_accuracy,
            "violation_recall": violation_recall,
            "balanced_accuracy": 0.5 * (safe_accuracy + violation_recall),
        }

    def _record_model_stage(
        self,
        outer_round: int,
        stage: str,
        active: list[ConstraintHypothesis],
        summaries: dict[str, TrainingSummary],
        evidence: list[object] | None = None,
    ) -> None:
        evidence_by_id = {
            item.hypothesis_id: item.to_dict() for item in (evidence or [])
        }
        models: list[dict[str, object]] = []
        for hypothesis in active:
            hypothesis_id = hypothesis.hypothesis_id
            model_row = {
                "hypothesis_id": hypothesis_id,
                "structure": hypothesis.to_dict(),
                "training_loss": summaries[hypothesis_id].mean_final_loss,
                "training_query_coverage": summaries[
                    hypothesis_id
                ].query_coverage_dict(),
                "learned_parameters": describe_ensemble_parameters(self.registry.models[hypothesis_id]),
                "decision_threshold_calibration": self.latest_threshold_calibration.get(
                    hypothesis_id
                ),
                "fit_expert_predictions_postfit_diagnostic": self._safe_prediction_summary(
                    hypothesis_id, self.experts
                ),
                "structure_audit_predictions": self._safe_prediction_summary(
                    hypothesis_id, self.structure_audit_experts
                ),
                "heldout_predictions_evaluation_only": self._safe_prediction_summary(
                    hypothesis_id, self.heldout_experts
                ),
                "selection_evidence": evidence_by_id.get(hypothesis_id),
            }
            models.append(model_row)
            parameters = model_row["learned_parameters"]
            clauses = parameters["clauses"]
            boundary = "non-scalar"
            if clauses and clauses[0]["members"] and "threshold_raw" in clauses[0]["members"][0]:
                thresholds = [member["threshold_raw"] for member in clauses[0]["members"]]
                boundary = f"thresholds={['%.3f' % value for value in thresholds]}"
            evidence_row = evidence_by_id.get(hypothesis_id)
            score_text = "n/a" if evidence_row is None else f"{evidence_row['selection_score']:.3f}"
            eligibility_text = "n/a" if evidence_row is None else str(evidence_row["champion_eligible"])
            self.progress(
                f"diagnostic round={outer_round} stage={stage} id={hypothesis_id} "
                f"boundary={boundary} fit_safe={model_row['fit_expert_predictions_postfit_diagnostic']['safe_rate']:.3f} "
                f"audit_safe={model_row['structure_audit_predictions']['safe_rate']:.3f} "
                f"heldout_safe(eval-only)={model_row['heldout_predictions_evaluation_only']['safe_rate']:.3f} "
                f"selection_score={score_text} eligible={eligibility_text}"
            )
        self.stage_diagnostics.append(
            {
                "outer_round": outer_round,
                "stage": stage,
                "current_oracle_label_counts": self.buffer.label_counts(),
                "models": models,
            }
        )

    def _intervention_for(
        self,
        hypothesis: ConstraintHypothesis,
        outer_round: int,
        query_index: int,
    ) -> InterventionSpec:
        pending = self.pending_interventions.get(hypothesis.hypothesis_id, [])
        if query_index < len(pending):
            resolved = self._resolve_false_unsafe_clause(
                hypothesis,
                pending[query_index],
                max(0, outer_round - 1) + query_index,
            )
            pending[query_index] = resolved
            return resolved
        # Every source gets both sides of the decision boundary early in its
        # pool.  Previously a single pending intervention was duplicated across
        # all pool slots, eliminating the likely-safe counterprobe.
        kinds = ("model_false_unsafe", "shortcut", "model_false_safe", "boundary_uncertainty", "local_feature_stress")
        schedule_index = query_index - len(pending)
        kind = kinds[schedule_index % len(kinds)]
        clauses = hypothesis.atomic_clauses()
        false_unsafe_occurrence = schedule_index // len(kinds)
        clause_cycle_index = (
            max(0, outer_round - 1) + false_unsafe_occurrence
            if kind == "model_false_unsafe"
            else query_index
        )
        clause = clauses[clause_cycle_index % len(clauses)]
        return self._resolve_false_unsafe_clause(
            hypothesis,
            InterventionSpec(
                hypothesis.hypothesis_id,
                kind,
                variable=clause.variables[clause_cycle_index % len(clause.variables)],
                clause_id=clause.clause_id if len(clauses) > 1 else None,
                rationale="Default hypothesis-specific falsification schedule.",
            ),
            clause_cycle_index,
        )

    @staticmethod
    def _resolve_false_unsafe_clause(
        hypothesis: ConstraintHypothesis,
        intervention: InterventionSpec,
        query_index: int,
    ) -> InterventionSpec:
        """Make the target clause explicit before composite numeric synthesis."""

        clauses = hypothesis.atomic_clauses()
        if intervention.kind != "model_false_unsafe" or len(clauses) <= 1:
            return intervention
        clause_by_id = {clause.clause_id: clause for clause in clauses}
        if intervention.clause_id in clause_by_id:
            clause = clause_by_id[intervention.clause_id]
            if intervention.variable in clause.variables:
                return intervention
            variable = clause.variables[query_index % len(clause.variables)]
            rationale = intervention.rationale.strip()
            repair_note = (
                f"Aligned intervention variable to target clause {clause.clause_id}."
            )
            return replace(
                intervention,
                variable=variable,
                rationale=f"{rationale} {repair_note}".strip(),
            )
        variable_matches = [
            clause
            for clause in clauses
            if intervention.variable is not None
            and intervention.variable in clause.variables
        ]
        clause = (
            variable_matches[0]
            if len(variable_matches) == 1
            else clauses[query_index % len(clauses)]
        )
        variable = (
            intervention.variable
            if intervention.variable in clause.variables
            else clause.variables[query_index % len(clause.variables)]
        )
        rationale = intervention.rationale.strip()
        repair_note = f"Resolved composite false-unsafe target to clause {clause.clause_id}."
        return replace(
            intervention,
            variable=variable,
            clause_id=clause.clause_id,
            rationale=f"{rationale} {repair_note}".strip(),
        )

    def _synthesize_best(
        self,
        hypothesis: ConstraintHypothesis,
        expert: Trajectory,
        intervention: InterventionSpec,
        outer_round: int,
        *,
        restart_offset: int = 0,
    ) -> FalsifierResult:
        if (
            intervention.kind == "model_false_unsafe"
            and len(hypothesis.atomic_clauses()) > 1
            and intervention.clause_id is None
        ):
            raise ValueError(
                "composite model_false_unsafe intervention requires an explicit clause_id"
            )
        ensemble = self.registry.models[hypothesis.hypothesis_id]
        if intervention.kind == "shortcut":
            mixes = np.linspace(0.55, 0.90, self.config.falsifier_restarts)
        elif intervention.kind == "model_false_unsafe":
            mixes = np.linspace(0.0, 0.08, self.config.falsifier_restarts)
        else:
            mixes = np.linspace(0.10, 0.60, self.config.falsifier_restarts)
        candidates: list[FalsifierResult] = []
        radius_attempts: list[dict[str, object]] = []
        attempted_radii: list[float] = []
        ladder_status: str | None = None
        anchor_prediction, anchor_score, _ = self.registry.predict(
            hypothesis.hypothesis_id,
            expert,
        )
        decision_threshold = float(ensemble.decision_threshold.item())
        anchor_margin = max(
            0.0,
            float(self.falsifier.config.false_unsafe_anchor_margin),
        )
        hard_margin = max(
            0.0,
            float(self.falsifier.config.false_unsafe_hard_margin),
        )
        hard_margin_target = decision_threshold + hard_margin
        anchor_margin_satisfied = bool(
            anchor_prediction == SAFE_LABEL
            and anchor_score <= decision_threshold - anchor_margin
        )

        if intervention.kind == "model_false_unsafe":
            radii = self.falsifier.false_unsafe_radii()
            if not anchor_margin_satisfied:
                # There is no causal source-model crossing to seek from this
                # expert.  Run one rung for auditable diagnostics, then stop.
                radii = radii[:1]
                ladder_status = "no_model_safe_anchor"
            for radius in radii:
                attempted_radii.append(float(radius))
                rung: list[FalsifierResult] = []
                for restart_index, mix in enumerate(mixes):
                    effective_restart_index = int(restart_offset) + restart_index
                    result = self.falsifier.generate(
                        ensemble,
                        expert,
                        intervention,
                        initialization_mix=float(mix),
                        restart_index=effective_restart_index,
                        trust_radius=float(radius),
                    )
                    prediction, hard_score, uncertainty = self.registry.predict(
                        hypothesis.hypothesis_id,
                        result.trajectory,
                    )
                    target_hard_score = float(
                        result.trajectory.metadata.get(
                            "source_candidate_target_hard_score",
                            hard_score,
                        )
                    )
                    achieved_generation_margin = float(
                        min(hard_score, target_hard_score) - decision_threshold
                    )
                    generation_hard_margin_satisfied = bool(
                        anchor_margin_satisfied
                        and hard_score >= hard_margin_target
                        and target_hard_score >= hard_margin_target
                    )
                    reachability_status = result.trajectory.metadata.get(
                        "falsifier_reachability_status"
                    )
                    if not self.falsifier.config.false_unsafe_use_hard_margin:
                        reachability_status = (
                            "legacy_objective_endpoint_certified"
                            if generation_hard_margin_satisfied
                            else "hard_checkpoint_disabled"
                        )
                    result.trajectory.metadata.update(
                        {
                            "source_anchor_prediction": int(anchor_prediction),
                            "source_anchor_hard_score": float(anchor_score),
                            "source_anchor_margin": anchor_margin,
                            "source_anchor_margin_satisfied": anchor_margin_satisfied,
                            "decision_threshold": decision_threshold,
                            "generation_hard_margin_target": hard_margin_target,
                            "generation_full_hard_score": float(hard_score),
                            "generation_target_hard_score": target_hard_score,
                            "generation_achieved_hard_margin": achieved_generation_margin,
                            "generation_hard_margin_satisfied": generation_hard_margin_satisfied,
                            "falsifier_reachability_status": reachability_status,
                        }
                    )
                    radius_attempts.append(
                        {
                            "radius": float(radius),
                            "restart_index": effective_restart_index,
                            "local_restart_index": restart_index,
                            "valid": bool(result.valid),
                            "prediction": int(prediction),
                            "hard_score": float(hard_score),
                            "target_hard_score": target_hard_score,
                            "decision_threshold": decision_threshold,
                            "achieved_generation_hard_margin": achieved_generation_margin,
                            "generation_hard_margin_satisfied": generation_hard_margin_satisfied,
                            "uncertainty": float(uncertainty),
                            "max_expert_deviation": result.trajectory.metadata.get(
                                "max_expert_deviation"
                            ),
                            "reachability_status": result.trajectory.metadata.get(
                                "falsifier_reachability_status"
                            ),
                            "rejected_hard_margin_checkpoints": result.trajectory.metadata.get(
                                "rejected_hard_margin_checkpoints", {}
                            ),
                        }
                    )
                    rung.append(result)
                candidates.extend(rung)
                if ladder_status == "no_model_safe_anchor":
                    break
                valid_rung = [item for item in rung if item.valid]
                if any(
                    bool(
                        item.trajectory.metadata.get(
                            "generation_hard_margin_satisfied",
                            False,
                        )
                    )
                    for item in valid_rung
                ):
                    ladder_status = "first_hard_margin_crossing"
                    break
            if ladder_status is None:
                ladder_status = "exhausted_without_hard_margin"
        else:
            candidates = [
                self.falsifier.generate(
                    ensemble,
                    expert,
                    intervention,
                    initialization_mix=float(mix),
                    restart_index=int(restart_offset) + restart_index,
                )
                for restart_index, mix in enumerate(mixes)
            ]
        valid = [item for item in candidates if item.valid]
        if not valid:
            pool_warmup = getattr(
                getattr(self, "falsifier", None),
                "warmup_candidate",
                None,
            )
            fallback_index = self.config.warmup_queries + outer_round
            fallback = (
                pool_warmup(expert, fallback_index, self.rng)
                if callable(pool_warmup)
                else generate_warmup_candidate(
                    expert,
                    fallback_index,
                    self.rng,
                    self.workspace_x,
                    self.workspace_y,
                )
            )
            fallback.metadata.update(
                {"source": f"{intervention.kind}_fallback", "source_hypothesis_id": hypothesis.hypothesis_id}
            )
            prediction, score, uncertainty = self.registry.predict(hypothesis.hypothesis_id, fallback)
            if intervention.kind == "model_false_unsafe":
                fallback_target_score = float(score)
                if intervention.clause_id is not None:
                    fallback_target_score = float(
                        self.registry.hard_clause_score(
                            hypothesis.hypothesis_id,
                            fallback,
                            intervention.clause_id,
                        )
                    )
                fallback.metadata.update(
                    {
                        "optimization_target_clause_id": intervention.clause_id,
                        "source_anchor_prediction": int(anchor_prediction),
                        "source_anchor_hard_score": float(anchor_score),
                        "source_anchor_margin": anchor_margin,
                        "source_anchor_margin_satisfied": anchor_margin_satisfied,
                        "decision_threshold": decision_threshold,
                        "generation_hard_margin_target": hard_margin_target,
                        "generation_full_hard_score": float(score),
                        "generation_target_hard_score": fallback_target_score,
                        "generation_achieved_hard_margin": float(
                            min(score, fallback_target_score) - decision_threshold
                        ),
                        "generation_hard_margin_satisfied": False,
                        "radius_ladder_attempted": attempted_radii,
                        "radius_ladder_attempts": radius_attempts,
                        "selected_trust_radius": None,
                        "radius_ladder_status": (
                            f"{ladder_status}:no_valid_candidate_fallback"
                        ),
                    }
                )
            return FalsifierResult(
                fallback,
                intervention.kind,
                hypothesis.hypothesis_id,
                float("nan"),
                float("nan"),
                score,
                uncertainty,
                True,
                f"fallback_prediction_{prediction}",
            )

        def key(item: FalsifierResult) -> tuple[float, ...]:
            prediction, _, _ = self.registry.predict(hypothesis.hypothesis_id, item.trajectory)
            if intervention.kind in {"model_false_safe", "shortcut"}:
                desired = prediction == SAFE_LABEL
            elif intervention.kind == "model_false_unsafe":
                generation_hard_margin_satisfied = bool(
                    item.trajectory.metadata.get(
                        "generation_hard_margin_satisfied",
                        False,
                    )
                )
                causal_crossing = bool(
                    anchor_margin_satisfied
                    and prediction == VIOLATION_LABEL
                    and float(
                        item.trajectory.metadata.get(
                            "generation_target_hard_score",
                            -float("inf"),
                        )
                    )
                    > decision_threshold
                )
                achieved_margin = float(
                    item.trajectory.metadata.get(
                        "generation_achieved_hard_margin",
                        -float("inf"),
                    )
                )
                deformation = float(
                    item.trajectory.metadata.get("max_expert_deviation", float("inf"))
                )
                uncertainty = float(item.final_uncertainty)
                if generation_hard_margin_satisfied:
                    tier = 0.0
                    secondary = deformation
                elif causal_crossing:
                    tier = 1.0
                    secondary = deformation
                else:
                    tier = 2.0
                    secondary = -achieved_margin
                return (tier, secondary, uncertainty, float(item.final_loss))
            else:
                desired = True
            merit = abs(item.final_score) - 0.1 * item.final_uncertainty
            return (0 if desired else 1, merit if "boundary" in intervention.kind else item.final_loss)

        selected = min(valid, key=key)
        if intervention.kind == "model_false_unsafe":
            selected.trajectory.metadata.update(
                {
                    "radius_ladder_attempted": attempted_radii,
                    "radius_ladder_attempts": radius_attempts,
                    "selected_trust_radius": selected.trajectory.metadata.get("trust_radius"),
                    "radius_ladder_status": ladder_status,
                }
            )
        return selected

    def _save_artifacts(self, bank: HypothesisBank, champion_id: str) -> None:
        (self.output_dir / "hypothesis_bank.json").write_text(
            _json_text(bank.to_dict()),
            encoding="utf-8",
        )
        (self.output_dir / "semantic_interactions.json").write_text(
            _json_text(self.reasoner.interactions),
            encoding="utf-8",
        )
        (self.output_dir / "evidence_history.json").write_text(
            _json_text(self.evidence_history),
            encoding="utf-8",
        )
        (self.output_dir / "evaluation_history.json").write_text(
            _json_text(self.evaluation_history),
            encoding="utf-8",
        )
        (self.output_dir / "query_diagnostics.json").write_text(
            _json_text(self.query_diagnostics),
            encoding="utf-8",
        )
        (self.output_dir / "stage_diagnostics.json").write_text(
            _json_text(self.stage_diagnostics),
            encoding="utf-8",
        )
        (self.output_dir / "all_hypothesis_evaluation.json").write_text(
            _json_text(self.all_hypothesis_evaluation),
            encoding="utf-8",
        )
        (self.output_dir / "finalization_diagnostics.json").write_text(
            _json_text(self.finalization_diagnostics),
            encoding="utf-8",
        )
        (self.output_dir / "threshold_calibration_history.json").write_text(
            _json_text(self.threshold_calibration_history),
            encoding="utf-8",
        )
        (self.output_dir / "qualified_checkpoint_history.json").write_text(
            _json_text(self.qualified_checkpoint_history),
            encoding="utf-8",
        )
        records = self.buffer.records
        np.savez_compressed(
            self.output_dir / "oracle_queries.npz",
            observations=np.stack([item.trajectory.states for item in records]).astype(np.float32),
            actions=np.stack([item.trajectory.actions for item in records]).astype(np.float32),
            labels=np.asarray([item.label for item in records], dtype=np.int64),
            outer_rounds=np.asarray([item.outer_round for item in records], dtype=np.int64),
        )
        (self.output_dir / "oracle_query_log.json").write_text(
            _json_text([item.audit_dict() for item in records]),
            encoding="utf-8",
        )
        torch.save(
            {
                "champion_hypothesis_id": champion_id,
                "models": {key: value.state_dict() for key, value in self.registry.models.items()},
                "compiled_hypotheses": {
                    key: value.compiled.hypothesis.to_dict() for key, value in self.registry.models.items()
                },
                "label_convention": {"safe": SAFE_LABEL, "violation": VIOLATION_LABEL},
            },
            self.output_dir / "constraint_models.pt",
        )
