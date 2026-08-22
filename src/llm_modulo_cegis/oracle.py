"""Capability-limited trajectory membership Oracle for the benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .types import SAFE_LABEL, VIOLATION_LABEL, Trajectory


class TrajectoryMembershipOracle(Protocol):
    @property
    def query_count(self) -> int: ...

    def query(self, trajectory: Trajectory) -> int: ...


class TrajectoryEvaluationOracle(TrajectoryMembershipOracle, Protocol):
    @property
    def supports_state_grid(self) -> bool: ...

    def state_violation_mask(self, points: np.ndarray) -> np.ndarray: ...

    def evaluation_trajectories(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None: ...

    def evaluation_geometry(self) -> object: ...


class _MembershipView:
    def __init__(self, oracle: TrajectoryMembershipOracle) -> None:
        self.__oracle = oracle

    @property
    def query_count(self) -> int:
        return self.__oracle.query_count

    def query(self, trajectory: Trajectory) -> int:
        return self.__oracle.query(trajectory)


class DeferredEvaluationOracle:
    """Evaluation facade used while a private test bank is deliberately unmounted.

    It lets the controller retain one code path for diagnostics and plotting,
    while making every ground-truth evaluation capability unavailable.  The
    membership Oracle is supplied separately.
    """

    query_count = 0

    @property
    def supports_state_grid(self) -> bool:
        return False

    def query(self, trajectory: Trajectory) -> int:
        del trajectory
        raise RuntimeError("membership queries must use the separate Oracle capability")

    def state_violation_mask(self, points: np.ndarray) -> np.ndarray:
        del points
        raise RuntimeError("private state truth is deferred until post-hoc evaluation")

    def evaluation_trajectories(self) -> None:
        return None

    def evaluation_geometry(self) -> None:
        return None


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

    @property
    def supports_state_grid(self) -> bool:
        return True

    def evaluation_trajectories(self) -> None:
        return None

    def query(self, trajectory: Trajectory) -> int:
        self.query_count += 1
        return VIOLATION_LABEL if self._polyline_min_distance(trajectory.states) < self.__radius else SAFE_LABEL

    def state_violation_mask(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return np.linalg.norm(points - self.__center, axis=-1) < self.__radius

    def evaluation_geometry(self) -> tuple[np.ndarray, float]:
        return self.__center.copy(), self.__radius

    def _polyline_min_distance(self, states: np.ndarray) -> float:
        return _polyline_min_distance(states, self.__center)


class RuleEvaluationOracle:
    """Private compositional benchmark fixture.

    The controller receives only :meth:`membership_view`.  Analytic clauses,
    clause labels, evaluation strata, and expected structure remain available
    solely to post-hoc evaluation code.
    """

    _STATE_GRID_KINDS = {"circle_exclusion", "linear_halfspace", "equality_band"}
    _RAW_FEATURE_INDICES = {
        "x_position": 0,
        "y_position": 1,
        "z_position": 2,
        "reference_dx": 3,
        "reference_dy": 4,
        "reference_dz": 5,
        "target_dx": 3,
        "target_dy": 4,
        "target_dz": 5,
        "ref_dx": 3,
        "ref_dy": 4,
        "ref_dz": 5,
        "vx": 6,
        "vy": 7,
        "vz": 8,
        "x_velocity": 6,
        "y_velocity": 7,
        "z_velocity": 8,
        "roll": 9,
        "pitch": 10,
        "yaw": 11,
    }

    def __init__(
        self,
        payload: dict[str, Any],
        evaluation_archive: str | Path | None = None,
    ) -> None:
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("unsupported private oracle schema_version")
        if payload.get("composition", "any_violation") != "any_violation":
            raise ValueError("only any_violation private composition is supported")
        if payload.get("label_convention", {"safe": 0, "violation": 1}) != {
            "safe": SAFE_LABEL,
            "violation": VIOLATION_LABEL,
        }:
            raise ValueError("private label_convention must be safe=0, violation=1")
        clauses = payload.get("clauses")
        if not isinstance(clauses, list) or not clauses:
            raise ValueError("private oracle requires at least one clause")
        clause_ids = [str(item.get("clause_id", "")) for item in clauses if isinstance(item, dict)]
        if len(clause_ids) != len(clauses) or any(not value for value in clause_ids):
            raise ValueError("every private clause requires a clause_id")
        if len(clause_ids) != len(set(clause_ids)):
            raise ValueError("private clause_id values must be unique")
        observation_dimension = payload.get("observation_dimension")
        if observation_dimension is not None:
            observation_dimension = int(observation_dimension)
            if observation_dimension < 2:
                raise ValueError("private observation_dimension must be at least two")
        horizon = payload.get("horizon")
        if horizon is not None:
            horizon = int(horizon)
            if horizon < 2:
                raise ValueError("private horizon must be at least two")
        dt = float(payload.get("dt", 1.0))
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("private dt must be finite and positive")
        self.__payload = payload
        self.__clauses = tuple(dict(item) for item in clauses)
        self.__observation_dimension = observation_dimension
        self.__horizon = horizon
        self.__dt = dt
        self.__evaluation_archive = None if evaluation_archive is None else Path(evaluation_archive)
        self.__evaluation_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.__evaluation_metadata_cache: dict[str, np.ndarray] | None = None
        self.query_count = 0

    @classmethod
    def from_private_files(
        cls,
        oracle_path: str | Path,
        evaluation_archive: str | Path | None = None,
    ) -> "RuleEvaluationOracle":
        payload = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
        return cls(payload, evaluation_archive)

    def membership_view(self) -> TrajectoryMembershipOracle:
        return _MembershipView(self)

    @property
    def supports_state_grid(self) -> bool:
        if self.__observation_dimension not in {None, 2}:
            return False
        return all(self._clause_supports_planar_state_grid(clause) for clause in self.__clauses)

    def query(self, trajectory: Trajectory) -> int:
        label = self.label(trajectory)
        self.query_count += 1
        return label

    def label(self, trajectory: Trajectory) -> int:
        severities = self.clause_severities(trajectory)
        return VIOLATION_LABEL if any(value > 0.0 for value in severities.values()) else SAFE_LABEL

    def clause_severities(self, trajectory: Trajectory) -> dict[str, float]:
        self._validate_trajectory_contract(trajectory)
        return {
            str(clause["clause_id"]): self._clause_severity(trajectory, clause)
            for clause in self.__clauses
        }

    def clause_labels(self, trajectory: Trajectory) -> dict[str, int]:
        return {
            key: VIOLATION_LABEL if value > 0.0 else SAFE_LABEL
            for key, value in self.clause_severities(trajectory).items()
        }

    def state_violation_mask(self, points: np.ndarray) -> np.ndarray:
        if not self.supports_state_grid:
            raise NotImplementedError("this trajectory constraint has no faithful static state-grid target")
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape [N,2]")
        masks = [self._state_clause_mask(points, clause) for clause in self.__clauses]
        return np.logical_or.reduce(masks)

    def evaluation_geometry(self) -> dict[str, Any]:
        return {"composition": "any_violation", "clauses": [dict(item) for item in self.__clauses]}

    def expected_structure(self) -> dict[str, Any] | None:
        value = self.__payload.get("expected_structure")
        return dict(value) if isinstance(value, dict) else None

    def evaluation_trajectories(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if self.__evaluation_archive is None:
            return None
        if self.__evaluation_cache is None:
            with np.load(self.__evaluation_archive, allow_pickle=False) as archive:
                observations = np.asarray(archive["observations"], dtype=np.float32)
                labels = np.asarray(archive["labels"], dtype=np.int64)
                groups = np.asarray(archive["groups"]).astype(str)
                trajectory_ids = np.asarray(archive["trajectory_ids"]).astype(str)
                if observations.ndim != 3 or observations.shape[1] < 2 or observations.shape[2] < 2:
                    raise ValueError("private evaluation observations must have shape [N,T,D], T,D >= 2")
                if (
                    self.__observation_dimension is not None
                    and observations.shape[2] != self.__observation_dimension
                ):
                    raise ValueError(
                        "private evaluation observation dimension does not match analytic Oracle"
                    )
                if self.__horizon is not None and observations.shape[1] != self.__horizon:
                    raise ValueError(
                        "private evaluation horizon does not match analytic Oracle"
                    )
                if labels.shape != (len(observations),) or groups.shape != labels.shape or trajectory_ids.shape != labels.shape:
                    raise ValueError("private evaluation arrays are misaligned")
                if np.any((labels != SAFE_LABEL) & (labels != VIOLATION_LABEL)):
                    raise ValueError("private evaluation labels must be binary")
                recomputed = np.asarray(
                    [self.label(Trajectory(states, dt=self.__dt)) for states in observations],
                    dtype=np.int64,
                )
                if not np.array_equal(labels, recomputed):
                    raise ValueError("private evaluation labels do not match analytic Oracle")
                self.__evaluation_cache = observations, labels, groups, trajectory_ids
        return tuple(value.copy() for value in self.__evaluation_cache)  # type: ignore[return-value]

    def evaluation_metadata(self) -> dict[str, np.ndarray] | None:
        """Return optional private strata only to the evaluation capability.

        This method is intentionally absent from :class:`_MembershipView`.
        It lets the frozen post-hoc evaluator reconstruct balanced matched-pair
        groups without changing the four-array legacy interface.
        """

        if self.__evaluation_archive is None:
            return None
        if self.__evaluation_metadata_cache is None:
            bank = self.evaluation_trajectories()
            assert bank is not None
            count = len(bank[1])
            optional = {
                "pair_ids",
                "pair_roles",
                "pair_targets",
                "clause_ids",
                "clause_labels",
            }
            values: dict[str, np.ndarray] = {}
            with np.load(self.__evaluation_archive, allow_pickle=False) as archive:
                for name in sorted(optional & set(archive.files)):
                    value = np.asarray(archive[name])
                    if name != "clause_ids" and (value.ndim < 1 or len(value) != count):
                        raise ValueError(f"private evaluation metadata {name} is misaligned")
                    values[name] = value.copy()
            if "clause_labels" in values:
                labels = values["clause_labels"]
                clause_ids = values.get("clause_ids")
                if labels.ndim != 2 or clause_ids is None or labels.shape[1] != len(clause_ids):
                    raise ValueError("private clause metadata is misaligned")
            pair_fields = {"pair_ids", "pair_roles", "pair_targets"}
            present_pair_fields = pair_fields & set(values)
            if present_pair_fields and present_pair_fields != pair_fields:
                raise ValueError("private matched-pair metadata is incomplete")
            self.__evaluation_metadata_cache = values
        return {key: value.copy() for key, value in self.__evaluation_metadata_cache.items()}

    @staticmethod
    def _coordinate_index(name: object) -> int:
        value = str(name)
        if value == "x_position":
            return 0
        if value == "y_position":
            return 1
        raise ValueError(f"unsupported coordinate: {value}")

    @classmethod
    def _raw_feature_index(cls, name: object) -> int:
        value = str(name)
        if value not in cls._RAW_FEATURE_INDICES:
            raise ValueError(f"unsupported raw observation feature: {value}")
        return cls._RAW_FEATURE_INDICES[value]

    @classmethod
    def _clause_supports_planar_state_grid(cls, clause: dict[str, Any]) -> bool:
        kind = str(clause.get("kind"))
        if kind not in cls._STATE_GRID_KINDS:
            return False
        if kind == "circle_exclusion":
            return np.asarray(clause.get("center", ())).shape == (2,)
        if kind == "linear_halfspace":
            return np.asarray(clause.get("normal", ())).shape == (2,)
        feature = clause.get("variable", clause.get("feature"))
        return str(feature) in {"x_position", "y_position"}

    @staticmethod
    def _require_state_width(states: np.ndarray, minimum: int, kind: str) -> None:
        if states.ndim != 2 or states.shape[1] < minimum:
            raise ValueError(
                f"{kind} requires trajectory observations with at least {minimum} columns"
            )

    def _validate_trajectory_contract(self, trajectory: Trajectory) -> None:
        states = np.asarray(trajectory.states)
        if (
            self.__observation_dimension is not None
            and states.shape[1] != self.__observation_dimension
        ):
            raise ValueError(
                "trajectory observation dimension does not match analytic Oracle"
            )
        if self.__horizon is not None and len(states) != self.__horizon:
            raise ValueError("trajectory horizon does not match analytic Oracle")
        if not np.isclose(float(trajectory.dt), self.__dt, rtol=0.0, atol=1.0e-9):
            raise ValueError("trajectory dt does not match analytic Oracle")

    def _clause_severity(self, trajectory: Trajectory, clause: dict[str, Any]) -> float:
        states = np.asarray(trajectory.states, dtype=np.float64)
        kind = str(clause.get("kind"))
        if kind == "circle_exclusion":
            center = np.asarray(clause["center"], dtype=np.float64)
            if center.shape != (states.shape[1],):
                raise ValueError("circle center dimension must match trajectory observations")
            return float(clause["radius"]) - _polyline_min_distance(states, center)
        if kind == "linear_halfspace":
            normal = np.asarray(clause["normal"], dtype=np.float64)
            if normal.shape != (states.shape[1],):
                raise ValueError("halfspace normal dimension must match trajectory observations")
            return float(np.max(states @ normal - float(clause["offset"])))
        if kind == "equality_band":
            feature = clause.get("variable", clause.get("feature"))
            index = self._raw_feature_index(feature)
            self._require_state_width(states, index + 1, kind)
            return float(np.max(np.abs(states[:, index] - float(clause["center"]))) - float(clause["half_width"]))
        if kind == "relative_height_band":
            feature = clause.get("feature", clause.get("variable", "reference_dz"))
            index = self._raw_feature_index(feature)
            self._require_state_width(states, index + 1, kind)
            return float(
                np.max(np.abs(states[:, index] - float(clause.get("center", 0.0))))
                - float(clause["half_width"])
            )
        if kind == "speed_upper_bound":
            speed = np.linalg.norm(np.diff(states, axis=0), axis=1) / float(trajectory.dt)
            return float(np.max(speed) - float(clause["threshold"]))
        if kind in {"observed_speed_upper_bound", "l2_upper_bound"}:
            raw_features = clause.get("features", ("vx", "vy", "vz"))
            if not isinstance(raw_features, (list, tuple)) or not raw_features:
                raise ValueError(f"{kind} requires a non-empty features list")
            indices = tuple(self._raw_feature_index(name) for name in raw_features)
            self._require_state_width(states, max(indices) + 1, kind)
            observed_speed = np.linalg.norm(states[:, indices], axis=1)
            return float(np.max(observed_speed) - float(clause["threshold"]))
        if kind in {"tilt_from_vertical_upper_bound", "upright_tilt_upper_bound"}:
            raw_features = clause.get("features", ("roll", "pitch"))
            if tuple(map(str, raw_features)) != ("roll", "pitch"):
                raise ValueError(f"{kind} features must be ['roll', 'pitch']")
            self._require_state_width(states, 11, kind)
            cosine = np.cos(states[:, 9]) * np.cos(states[:, 10])
            tilt = np.arccos(np.clip(cosine, -1.0, 1.0))
            return float(np.max(tilt) - float(clause["threshold"]))
        if kind == "checkpoint_visit":
            center = np.asarray(clause["center"], dtype=np.float64)
            if center.shape != (states.shape[1],):
                raise ValueError("checkpoint center dimension must match trajectory observations")
            return _polyline_min_distance(states, center) - float(clause["radius"])
        raise ValueError(f"unsupported private clause kind: {kind}")

    def _state_clause_mask(self, points: np.ndarray, clause: dict[str, Any]) -> np.ndarray:
        kind = str(clause.get("kind"))
        if kind == "circle_exclusion":
            center = np.asarray(clause["center"], dtype=np.float64)
            return np.linalg.norm(points - center, axis=1) < float(clause["radius"])
        if kind == "linear_halfspace":
            normal = np.asarray(clause["normal"], dtype=np.float64)
            return points @ normal > float(clause["offset"])
        if kind == "equality_band":
            index = self._coordinate_index(clause["variable"])
            return np.abs(points[:, index] - float(clause["center"])) > float(clause["half_width"])
        raise NotImplementedError(kind)


def _polyline_min_distance(states: np.ndarray, center: np.ndarray) -> float:
    points = np.asarray(states, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    starts = points[:-1] - center
    deltas = np.diff(points, axis=0)
    squared = np.sum(deltas * deltas, axis=1)
    fractions = np.zeros_like(squared)
    valid = squared > 0.0
    fractions[valid] = -np.sum(starts[valid] * deltas[valid], axis=1) / squared[valid]
    fractions = np.clip(fractions, 0.0, 1.0)
    closest = starts + fractions[:, None] * deltas
    return float(np.min(np.linalg.norm(closest, axis=1)))
