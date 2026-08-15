"""Bi-level semantic--numeric CEGIS controller."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .data import FeatureLibrary
from .evidence import EvidenceCompiler, evidence_report
from .evaluation import BoundaryMetrics, evaluate_boundary, plot_boundary, plot_semantic_trace
from .falsifier import FalsifierResult, HypothesisFalsifier, generate_warmup_candidate
from .hypotheses import ConstraintHypothesis, HypothesisBank, compile_hypothesis
from .learner import LearnerRegistry, TrainerConfig, TrainingSummary, fit_ensemble
from .oracle import CircularEvaluationOracle, TrajectoryMembershipOracle
from .semantic import SemanticReasoner
from .types import InterventionSpec, QueryBuffer, QueryRecord, SAFE_LABEL, Trajectory, VIOLATION_LABEL


@dataclass(frozen=True)
class LoopConfig:
    warmup_queries: int = 20
    max_warmup_queries: int = 50
    minimum_label_count: int = 4
    outer_rounds: int = 3
    queries_per_hypothesis: int = 1
    falsifier_restarts: int = 2
    grid_resolution: int = 100
    freeze_revisions: bool = False
    maximum_active_hypotheses: int = 6


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

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["final_metrics"] = self.final_metrics.to_dict()
        return payload


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
        evaluation_oracle: CircularEvaluationOracle,
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
        self.task_description = task_description
        self.library = feature_library
        self.reasoner = reasoner
        self.registry = registry
        self.trainer_config = trainer_config
        self.evidence_compiler = evidence_compiler
        self.falsifier = falsifier
        self.oracle = oracle
        self.evaluation_oracle = evaluation_oracle
        self.experts = experts
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
        self.pending_interventions: dict[str, list[InterventionSpec]] = {}

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

        for outer_round in range(1, self.config.outer_rounds + 1):
            active = bank.active()
            summaries = self._fit_active(active, outer_round)
            self._collect_hypothesis_queries(active, outer_round)
            summaries = self._fit_active(active, outer_round)
            evidence = self.evidence_compiler.compile(
                self.registry,
                [item.hypothesis_id for item in active],
                self.experts,
                self.buffer.records,
            )
            report = evidence_report(outer_round, evidence, self.buffer.label_counts())
            self.evidence_history.append(report)
            ordered = sorted(evidence, key=lambda item: item.selection_score, reverse=True)
            numeric_champion_id = ordered[0].hypothesis_id
            actions = []
            if not self.config.freeze_revisions:
                actions = self.reasoner.revise(
                    self.task_description,
                    self.library,
                    bank,
                    report,
                    evidence,
                    outer_round,
                )
            semantic_retained = [
                action.target_hypothesis_id
                for action in actions
                if action.action == "retain_and_query" and action.target_hypothesis_id in self.registry.models
            ]
            champion_id = semantic_retained[0] if semantic_retained else numeric_champion_id
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
                    "training_losses": {key: value.mean_final_loss for key, value in summaries.items()},
                    "evaluation_only_metrics": champion_metrics.to_dict(),
                }
            )
            self.progress(
                f"outer_round={outer_round} champion={champion_id} "
                f"score={ordered[0].selection_score:.3f} labels={self.buffer.label_counts()} "
                f"evaluation_iou={champion_metrics.iou:.3f}"
            )
            if outer_round < self.config.outer_rounds and not self.config.freeze_revisions:
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
        self._save_artifacts(bank, champion_id)
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
        )
        (self.output_dir / "result.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result

    def _collect_warmup(self) -> None:
        index = 0
        while index < self.config.max_warmup_queries:
            counts = self.buffer.label_counts()
            if index >= self.config.warmup_queries and min(counts.values()) >= self.config.minimum_label_count:
                break
            expert = self.experts[index % len(self.experts)]
            candidate = generate_warmup_candidate(
                expert,
                index,
                self.rng,
                self.workspace_x,
                self.workspace_y,
            )
            self.buffer.add(QueryRecord(candidate, self.oracle.query(candidate), "warmup", 0))
            index += 1
        if min(self.buffer.label_counts().values()) < self.config.minimum_label_count:
            raise RuntimeError("warmup failed to collect both trajectory labels")
        self.progress(f"warmup queries={len(self.buffer)} labels={self.buffer.label_counts()}")

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
                self.buffer.records,
                self.library,
                self.trainer_config,
                seed=self.seed + outer_round * 7919 + index * 101,
                device=self.device,
            )
        return summaries

    def _collect_hypothesis_queries(
        self,
        active: list[ConstraintHypothesis],
        outer_round: int,
    ) -> None:
        for hypothesis_index, hypothesis in enumerate(active):
            for query_index in range(self.config.queries_per_hypothesis):
                intervention = self._intervention_for(hypothesis, outer_round, query_index)
                expert_index = (outer_round + hypothesis_index + query_index) % len(self.experts)
                expert = self.experts[expert_index]
                result = self._synthesize_best(hypothesis, expert, intervention, outer_round)
                predictions: dict[str, int] = {}
                scores: dict[str, float] = {}
                for other in active:
                    prediction, score, _ = self.registry.predict(other.hypothesis_id, result.trajectory)
                    predictions[other.hypothesis_id] = prediction
                    scores[other.hypothesis_id] = score
                label = self.oracle.query(result.trajectory)
                self.buffer.add(
                    QueryRecord(
                        result.trajectory,
                        label,
                        intervention.kind,
                        outer_round,
                        hypothesis.hypothesis_id,
                        predictions,
                        scores,
                    )
                )
                self.query_diagnostics.append(
                    {
                        "outer_round": outer_round,
                        "source_hypothesis_id": hypothesis.hypothesis_id,
                        "intervention": intervention.to_dict(),
                        "oracle_label": label,
                        "predictions_before_query": predictions,
                        "falsifier": {
                            "initial_loss": result.initial_loss,
                            "final_loss": result.final_loss,
                            "final_score": result.final_score,
                            "final_uncertainty": result.final_uncertainty,
                            "validation_reason": result.validation_reason,
                        },
                    }
                )

    def _intervention_for(
        self,
        hypothesis: ConstraintHypothesis,
        outer_round: int,
        query_index: int,
    ) -> InterventionSpec:
        pending = self.pending_interventions.get(hypothesis.hypothesis_id, [])
        if pending:
            return pending.pop(0)
        kinds = ("shortcut", "local_feature_stress", "model_false_safe", "model_false_unsafe", "boundary_uncertainty")
        kind = kinds[(outer_round + query_index - 1) % len(kinds)]
        return InterventionSpec(
            hypothesis.hypothesis_id,
            kind,
            variable=hypothesis.variables[query_index % len(hypothesis.variables)],
            rationale="Default hypothesis-specific falsification schedule.",
        )

    def _synthesize_best(
        self,
        hypothesis: ConstraintHypothesis,
        expert: Trajectory,
        intervention: InterventionSpec,
        outer_round: int,
    ) -> FalsifierResult:
        ensemble = self.registry.models[hypothesis.hypothesis_id]
        if intervention.kind == "shortcut":
            mixes = np.linspace(0.55, 0.90, self.config.falsifier_restarts)
        elif intervention.kind == "model_false_unsafe":
            mixes = np.linspace(0.0, 0.08, self.config.falsifier_restarts)
        else:
            mixes = np.linspace(0.10, 0.60, self.config.falsifier_restarts)
        candidates = [
            self.falsifier.generate(ensemble, expert, intervention, initialization_mix=float(mix))
            for mix in mixes
        ]
        valid = [item for item in candidates if item.valid]
        if not valid:
            fallback = generate_warmup_candidate(
                expert,
                self.config.warmup_queries + outer_round,
                self.rng,
                self.workspace_x,
                self.workspace_y,
            )
            fallback.metadata.update(
                {"source": f"{intervention.kind}_fallback", "source_hypothesis_id": hypothesis.hypothesis_id}
            )
            prediction, score, uncertainty = self.registry.predict(hypothesis.hypothesis_id, fallback)
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

        def key(item: FalsifierResult) -> tuple[int, float]:
            prediction, _, _ = self.registry.predict(hypothesis.hypothesis_id, item.trajectory)
            if intervention.kind in {"model_false_safe", "shortcut"}:
                desired = prediction == SAFE_LABEL
            elif intervention.kind == "model_false_unsafe":
                desired = prediction == VIOLATION_LABEL
            else:
                desired = True
            merit = abs(item.final_score) - 0.1 * item.final_uncertainty
            return (0 if desired else 1, merit if "boundary" in intervention.kind else item.final_loss)

        return min(valid, key=key)

    def _save_artifacts(self, bank: HypothesisBank, champion_id: str) -> None:
        (self.output_dir / "hypothesis_bank.json").write_text(
            json.dumps(bank.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "semantic_interactions.json").write_text(
            json.dumps(self.reasoner.interactions, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "evidence_history.json").write_text(
            json.dumps(self.evidence_history, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "evaluation_history.json").write_text(
            json.dumps(self.evaluation_history, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "query_diagnostics.json").write_text(
            json.dumps(self.query_diagnostics, indent=2, ensure_ascii=False) + "\n",
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
            json.dumps([item.audit_dict() for item in records], indent=2, ensure_ascii=False) + "\n",
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
