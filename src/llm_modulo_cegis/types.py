"""Shared data contracts for the semantic and numeric loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


SAFE_LABEL = 0
VIOLATION_LABEL = 1


@dataclass
class Trajectory:
    """A fixed-horizon trajectory visible to the learner.

    ``states`` may be a compact planar waypoint representation or a richer
    observation vector.  ``actions`` are optional and may be state-aligned
    (``T`` rows) or transition-aligned (``T-1`` rows); the numeric learner
    deliberately consumes only learner-visible state features.
    """

    states: np.ndarray
    actions: np.ndarray | None = None
    dt: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float32)
        if self.states.ndim != 2 or self.states.shape[0] < 2 or self.states.shape[1] < 2:
            raise ValueError("states must have shape [T, D], T >= 2 and D >= 2")
        if not np.all(np.isfinite(self.states)):
            raise ValueError("states contain non-finite values")
        if self.actions is not None:
            self.actions = np.asarray(self.actions, dtype=np.float32)
            if (
                self.actions.ndim != 2
                or self.actions.shape[0] not in {self.states.shape[0], self.states.shape[0] - 1}
                or self.actions.shape[1] < 1
            ):
                raise ValueError("actions must have shape [T,A] or [T-1,A], A >= 1")
            if not np.all(np.isfinite(self.actions)):
                raise ValueError("actions contain non-finite values")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")

    def copy(self) -> "Trajectory":
        return Trajectory(
            self.states.copy(),
            None if self.actions is None else self.actions.copy(),
            self.dt,
            dict(self.metadata),
        )


@dataclass
class QueryRecord:
    """One trajectory-level Oracle observation plus model predictions."""

    trajectory: Trajectory
    label: int
    source: str
    outer_round: int
    source_hypothesis_id: str | None = None
    predictions_before_query: dict[str, int] = field(default_factory=dict)
    scores_before_query: dict[str, float] = field(default_factory=dict)
    uncertainties_before_query: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in (SAFE_LABEL, VIOLATION_LABEL):
            raise ValueError("label must be 0=safe or 1=violation")

    def audit_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source": self.source,
            "outer_round": self.outer_round,
            "source_hypothesis_id": self.source_hypothesis_id,
            "predictions_before_query": self.predictions_before_query,
            "scores_before_query": self.scores_before_query,
            "uncertainties_before_query": self.uncertainties_before_query,
            "trajectory_metadata": self.trajectory.metadata,
        }


class QueryBuffer:
    """Append-only shared evidence available to every hypothesis learner."""

    def __init__(self) -> None:
        self._records: list[QueryRecord] = []

    def add(self, record: QueryRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> list[QueryRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def label_counts(self) -> dict[str, int]:
        return {
            "safe": sum(record.label == SAFE_LABEL for record in self._records),
            "violation": sum(record.label == VIOLATION_LABEL for record in self._records),
        }


@dataclass(frozen=True)
class HypothesisEvidence:
    """Auditable numeric evidence sent to the semantic reasoner."""

    hypothesis_id: str
    balanced_accuracy: float
    safe_accuracy: float
    violation_recall: float
    expert_safe_rate: float
    counterexample_rate: float
    false_safe_count: int
    false_unsafe_count: int
    mean_abs_margin: float
    mean_uncertainty: float
    intervention_violation_yield: float
    intervention_count: int
    complexity: int
    selection_score: float
    evidence_sufficient: bool
    parameter_count: int = 0
    prequential_count: int = 0
    query_priority: float = 0.0
    fit_expert_safe_rate: float = 1.0
    champion_eligible: bool = True
    ineligibility_reasons: tuple[str, ...] = ()
    representation_group_count: int = 0
    contradictory_representation_group_count: int = 0
    linear_max_support_pair_count: int = 0
    linear_max_support_contradiction_count: int = 0
    linear_max_support_distinct_anchor_count: int = 0
    linear_max_support_unresolved_pair_count: int = 0
    linear_max_support_gate_triggered: bool = False
    linear_max_support_gate_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InterventionSpec:
    """Qualitative query request; the numeric falsifier executes it."""

    target_hypothesis_id: str
    kind: str
    variable: str | None = None
    clause_id: str | None = None
    preserve_endpoints: bool = True
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
