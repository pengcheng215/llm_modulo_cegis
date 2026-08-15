"""Capability-limited trajectory membership Oracle for the benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .types import SAFE_LABEL, VIOLATION_LABEL, Trajectory


class TrajectoryMembershipOracle(Protocol):
    @property
    def query_count(self) -> int: ...

    def query(self, trajectory: Trajectory) -> int: ...


class _MembershipView:
    def __init__(self, oracle: "CircularEvaluationOracle") -> None:
        self.__oracle = oracle

    @property
    def query_count(self) -> int:
        return self.__oracle.query_count

    def query(self, trajectory: Trajectory) -> int:
        return self.__oracle.query(trajectory)


class CircularEvaluationOracle:
    """Private benchmark fixture. Learners receive only ``membership_view``."""

    def __init__(self, center: tuple[float, float], safety_radius: float) -> None:
        self.__center = np.asarray(center, dtype=np.float64)
        self.__radius = float(safety_radius)
        self.query_count = 0

    @classmethod
    def from_private_file(cls, path: str | Path) -> "CircularEvaluationOracle":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(map(float, payload["obstacle_center"])), float(payload["safety_radius"]))

    def membership_view(self) -> TrajectoryMembershipOracle:
        return _MembershipView(self)

    def query(self, trajectory: Trajectory) -> int:
        self.query_count += 1
        return VIOLATION_LABEL if self._polyline_min_distance(trajectory.states) < self.__radius else SAFE_LABEL

    def state_violation_mask(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return np.linalg.norm(points - self.__center, axis=-1) < self.__radius

    def evaluation_geometry(self) -> tuple[np.ndarray, float]:
        return self.__center.copy(), self.__radius

    def _polyline_min_distance(self, states: np.ndarray) -> float:
        points = np.asarray(states, dtype=np.float64)
        starts = points[:-1] - self.__center
        deltas = np.diff(points, axis=0)
        squared = np.sum(deltas * deltas, axis=1)
        fractions = np.zeros_like(squared)
        valid = squared > 0.0
        fractions[valid] = -np.sum(starts[valid] * deltas[valid], axis=1) / squared[valid]
        fractions = np.clip(fractions, 0.0, 1.0)
        closest = starts + fractions[:, None] * deltas
        return float(np.min(np.linalg.norm(closest, axis=1)))
