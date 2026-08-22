"""Public trajectory data contracts and differentiable feature libraries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .types import Trajectory


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    unit: str
    low: float
    high: float
    group: str

    def prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "group": self.group,
        }


FEATURE_SPECS = (
    FeatureSpec("x_position", "horizontal planar position", "m", 0.0, 10.0, "position"),
    FeatureSpec("y_position", "vertical planar position", "m", -4.0, 4.0, "position"),
    FeatureSpec("x_velocity", "finite-difference horizontal velocity", "m/step", -0.5, 0.5, "velocity"),
    FeatureSpec("y_velocity", "finite-difference vertical velocity", "m/step", -0.5, 0.5, "velocity"),
    FeatureSpec("speed", "planar finite-difference speed magnitude", "m/step", 0.0, 0.75, "velocity"),
    FeatureSpec("progress", "normalized trajectory time", "ratio", 0.0, 1.0, "time"),
)


SEMTRAJ_FEATURE_SPECS = (
    FeatureSpec("x_position", "horizontal planar position", "m", 0.0, 10.0, "position"),
    FeatureSpec("y_position", "vertical planar position", "m", -4.0, 4.0, "position"),
    FeatureSpec("x_velocity", "finite-difference horizontal velocity", "m/step", -0.5, 0.5, "velocity"),
    FeatureSpec("y_velocity", "finite-difference vertical velocity", "m/step", -0.5, 0.5, "velocity"),
    FeatureSpec("speed", "planar finite-difference speed magnitude", "m/step", 0.0, 0.5, "velocity"),
    FeatureSpec("x_acceleration", "finite-difference horizontal acceleration", "m/step^2", -1.0, 1.0, "acceleration"),
    FeatureSpec("y_acceleration", "finite-difference vertical acceleration", "m/step^2", -1.0, 1.0, "acceleration"),
    FeatureSpec("acceleration_norm", "finite-difference acceleration magnitude", "m/step^2", 0.0, 1.0, "acceleration"),
    FeatureSpec("heading_sin", "sine of the instantaneous motion heading", "ratio", -1.0, 1.0, "direction"),
    FeatureSpec("heading_cos", "cosine of the instantaneous motion heading", "ratio", -1.0, 1.0, "direction"),
    FeatureSpec("path_length_so_far", "cumulative distance travelled from the start", "m", 0.0, 20.0, "path_history"),
    FeatureSpec("progress", "normalized trajectory time", "ratio", 0.0, 1.0, "time"),
)


CARRYWATER_ACTIVE_FEATURE_SPECS = (
    FeatureSpec("x_position", "end-effector world x position", "m", -3.0, 3.0, "position"),
    FeatureSpec("y_position", "end-effector world y position", "m", -3.0, 3.0, "position"),
    FeatureSpec("z_position", "end-effector world z position", "m", 0.1, 1.2, "position"),
    FeatureSpec("target_dx", "target x minus current x", "m", -5.0, 5.0, "relative_position"),
    FeatureSpec("target_dy", "target y minus current y", "m", -5.0, 5.0, "relative_position"),
    FeatureSpec("target_dz", "requested carrying height minus current z", "m", -0.4, 0.4, "relative_position"),
    FeatureSpec("x_velocity", "observed horizontal x velocity", "m/s", -1.5, 1.5, "velocity"),
    FeatureSpec("y_velocity", "observed horizontal y velocity", "m/s", -1.5, 1.5, "velocity"),
    FeatureSpec("z_velocity", "observed vertical velocity", "m/s", -1.5, 1.5, "velocity"),
    FeatureSpec("speed", "three-dimensional translational speed magnitude", "m/s", 0.0, 2.0, "velocity"),
    FeatureSpec("roll", "signed cup roll angle", "rad", -0.8, 0.8, "orientation"),
    FeatureSpec("pitch", "signed cup pitch angle", "rad", -0.8, 0.8, "orientation"),
    FeatureSpec("yaw", "signed cup yaw angle", "rad", -3.2, 3.2, "orientation"),
    FeatureSpec("abs_roll", "absolute cup roll angle", "rad", 0.0, 0.8, "orientation"),
    FeatureSpec("abs_pitch", "absolute cup pitch angle", "rad", 0.0, 0.8, "orientation"),
    FeatureSpec(
        "tilt_from_vertical",
        "cup tilt angle from vertical derived from roll and pitch",
        "rad",
        0.0,
        1.0,
        "orientation",
    ),
    FeatureSpec("tilt_linf", "maximum absolute roll or pitch", "rad", 0.0, 0.8, "orientation"),
    FeatureSpec("progress", "normalized trajectory time", "ratio", 0.0, 1.0, "time"),
)


@dataclass(frozen=True)
class TaskSpec:
    """Learner-visible task contract with no hidden constraint parameters."""

    schema_version: int
    suite_name: str
    task_instance_id: str
    task_family: str
    task_description: str
    horizon: int
    workspace_x: tuple[float, float]
    workspace_y: tuple[float, float]
    max_step: float
    start_goal_policy: str
    feature_library_version: str
    feature_specs: tuple[FeatureSpec, ...]
    raw_state_dimension: int = 2
    action_dimension: int = 2
    action_horizon: str = "state_aligned"
    trajectory_adapter: str = "planar_waypoint_v1"
    dt: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSpec":
        required = {
            "schema_version",
            "suite_name",
            "task_instance_id",
            "task_family",
            "task_description",
            "horizon",
            "workspace",
            "max_step",
            "start_goal_policy",
            "feature_library_version",
            "feature_schema",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"task_spec is missing fields: {missing}")
        optional = {
            "suite_version",
            "trajectory_label_convention",
            "learner_information_contract",
            "raw_state_dimension",
            "action_dimension",
            "action_horizon",
            "trajectory_adapter",
            "dt",
        }
        unknown = sorted(set(payload) - required - optional)
        if unknown:
            raise ValueError(f"task_spec contains unknown or private fields: {unknown}")
        if payload.get("trajectory_label_convention", {"safe": 0, "violation": 1}) != {
            "safe": 0,
            "violation": 1,
        }:
            raise ValueError("task_spec trajectory_label_convention must be safe=0, violation=1")
        workspace = payload["workspace"]
        if not isinstance(workspace, dict):
            raise ValueError("task_spec.workspace must be an object")
        if set(workspace) != {"x", "y"}:
            raise ValueError("task_spec.workspace must contain exactly x and y")
        feature_rows = payload["feature_schema"]
        if not isinstance(feature_rows, list) or not feature_rows:
            raise ValueError("task_spec.feature_schema must be a non-empty list")
        required_feature_fields = {"name", "description", "unit", "low", "high", "group"}
        optional_feature_fields = {"causal_status"}
        specs: list[FeatureSpec] = []
        for index, row in enumerate(feature_rows):
            if not isinstance(row, dict):
                raise ValueError(f"feature_schema[{index}] must be an object")
            missing_feature = sorted(required_feature_fields - set(row))
            unknown_feature = sorted(set(row) - required_feature_fields - optional_feature_fields)
            if missing_feature or unknown_feature:
                raise ValueError(
                    f"feature_schema[{index}] schema mismatch: "
                    f"missing={missing_feature}, unknown={unknown_feature}"
                )
            spec = FeatureSpec(
                name=str(row["name"]),
                description=str(row["description"]),
                unit=str(row["unit"]),
                low=float(row["low"]),
                high=float(row["high"]),
                group=str(row["group"]),
            )
            if not spec.name or not spec.description or not spec.unit or not spec.group:
                raise ValueError(f"feature_schema[{index}] contains an empty string")
            if not np.isfinite(spec.low) or not np.isfinite(spec.high) or spec.low >= spec.high:
                raise ValueError(f"feature_schema[{index}] bounds must be finite and increasing")
            specs.append(spec)
        result = cls(
            schema_version=int(payload["schema_version"]),
            suite_name=str(payload["suite_name"]),
            task_instance_id=str(payload["task_instance_id"]),
            task_family=str(payload["task_family"]),
            task_description=str(payload["task_description"]),
            horizon=int(payload["horizon"]),
            workspace_x=tuple(map(float, workspace["x"])),
            workspace_y=tuple(map(float, workspace["y"])),
            max_step=float(payload["max_step"]),
            start_goal_policy=str(payload["start_goal_policy"]),
            feature_library_version=str(payload["feature_library_version"]),
            feature_specs=tuple(specs),
            raw_state_dimension=int(payload.get("raw_state_dimension", 2)),
            action_dimension=int(payload.get("action_dimension", 2)),
            action_horizon=str(payload.get("action_horizon", "state_aligned")),
            trajectory_adapter=str(payload.get("trajectory_adapter", "planar_waypoint_v1")),
            dt=float(payload.get("dt", 1.0)),
        )
        if result.schema_version not in {1, 2}:
            raise ValueError(f"unsupported task_spec schema_version: {result.schema_version}")
        if result.schema_version == 2:
            generic_required = {
                "raw_state_dimension",
                "action_dimension",
                "action_horizon",
                "trajectory_adapter",
                "dt",
            }
            missing_generic = sorted(generic_required - set(payload))
            if missing_generic:
                raise ValueError(f"schema_version=2 task_spec is missing fields: {missing_generic}")
        if result.horizon < 2 or result.max_step <= 0.0:
            raise ValueError("task_spec horizon and max_step must be positive")
        if len(result.workspace_x) != 2 or len(result.workspace_y) != 2:
            raise ValueError("workspace bounds must each contain two values")
        if not all(np.isfinite(value) for value in (*result.workspace_x, *result.workspace_y, result.max_step)):
            raise ValueError("workspace and max_step must be finite")
        if result.workspace_x[0] >= result.workspace_x[1] or result.workspace_y[0] >= result.workspace_y[1]:
            raise ValueError("workspace bounds must be increasing")
        if not result.task_instance_id or not result.task_family or not result.task_description:
            raise ValueError("task identifiers and description cannot be empty")
        if result.raw_state_dimension < 2 or result.action_dimension < 1:
            raise ValueError("raw_state_dimension must be >=2 and action_dimension must be >=1")
        if result.action_horizon not in {"state_aligned", "transition"}:
            raise ValueError("action_horizon must be state_aligned or transition")
        if not result.trajectory_adapter or not np.isfinite(result.dt) or result.dt <= 0.0:
            raise ValueError("trajectory_adapter must be non-empty and dt must be finite and positive")
        return result


def load_task_spec(dataset_dir: str | Path) -> TaskSpec:
    path = Path(dataset_dir) / "task_spec.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return TaskSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


class FeatureLibrary:
    """Compile semantic variables to NumPy and differentiable Torch features."""

    def __init__(
        self,
        specs: tuple[FeatureSpec, ...] = FEATURE_SPECS,
        *,
        feature_library_version: str = "planar_v1",
        raw_state_dimension: int = 2,
    ) -> None:
        self.specs = specs
        self.feature_library_version = str(feature_library_version)
        self.raw_state_dimension = int(raw_state_dimension)
        self._by_name = {spec.name: spec for spec in specs}
        if len(self._by_name) != len(specs):
            raise ValueError("feature names must be unique")
        supported = {
            spec.name
            for spec in (*SEMTRAJ_FEATURE_SPECS, *CARRYWATER_ACTIVE_FEATURE_SPECS)
        }
        unknown = set(self._by_name) - supported
        if unknown:
            raise ValueError(f"features have no differentiable implementation: {sorted(unknown)}")

    @classmethod
    def from_task_spec(cls, task_spec: TaskSpec) -> "FeatureLibrary":
        return cls(
            task_spec.feature_specs,
            feature_library_version=task_spec.feature_library_version,
            raw_state_dimension=task_spec.raw_state_dimension,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def schema_for_prompt(self) -> list[dict[str, Any]]:
        return [spec.prompt_dict() for spec in self.specs]

    def bounds(self, variables: tuple[str, ...]) -> tuple[list[float], list[float]]:
        self.validate_variables(variables)
        return (
            [self._by_name[name].low for name in variables],
            [self._by_name[name].high for name in variables],
        )

    def validate_variables(self, variables: tuple[str, ...]) -> None:
        unknown = set(variables) - set(self._by_name)
        if unknown:
            raise ValueError(f"unknown variables: {sorted(unknown)}")

    @property
    def is_planar(self) -> bool:
        return self.raw_state_dimension == 2

    @property
    def is_carrywater_active(self) -> bool:
        return (
            self.raw_state_dimension == 12
            and self.feature_library_version.startswith("carrywater_active")
        )

    def numpy_features(self, states: np.ndarray, variables: tuple[str, ...]) -> np.ndarray:
        self.validate_variables(variables)
        values = self._numpy_all(np.asarray(states, dtype=np.float32))
        return np.column_stack([values[name] for name in variables]).astype(np.float32)

    def torch_features(self, states: torch.Tensor, variables: tuple[str, ...]) -> torch.Tensor:
        self.validate_variables(variables)
        if states.ndim == 2:
            states = states.unsqueeze(0)
            squeeze = True
        elif states.ndim == 3:
            squeeze = False
        else:
            raise ValueError("states must have shape [T,D] or [B,T,D]")
        if states.shape[-1] != self.raw_state_dimension:
            raise ValueError(
                f"raw state dimension mismatch: expected {self.raw_state_dimension}, "
                f"got {states.shape[-1]}"
            )
        if self.is_carrywater_active:
            horizon = states.shape[1]
            progress = torch.linspace(
                0.0,
                1.0,
                horizon,
                dtype=states.dtype,
                device=states.device,
            )[None, :].expand(states.shape[0], -1)
            roll = states[..., 9]
            pitch = states[..., 10]
            velocity = states[..., 6:9]
            vertical_cosine = torch.cos(roll) * torch.cos(pitch)
            vertical_sine = torch.linalg.vector_norm(
                torch.stack(
                    (
                        torch.sin(pitch),
                        torch.sin(roll) * torch.cos(pitch),
                    ),
                    dim=-1,
                ),
                dim=-1,
            )
            all_values = {
                "x_position": states[..., 0],
                "y_position": states[..., 1],
                "z_position": states[..., 2],
                "target_dx": states[..., 3],
                "target_dy": states[..., 4],
                "target_dz": states[..., 5],
                "x_velocity": velocity[..., 0],
                "y_velocity": velocity[..., 1],
                "z_velocity": velocity[..., 2],
                "speed": torch.linalg.vector_norm(velocity, dim=-1),
                "roll": roll,
                "pitch": pitch,
                "yaw": states[..., 11],
                "abs_roll": torch.abs(roll),
                "abs_pitch": torch.abs(pitch),
                "tilt_from_vertical": torch.atan2(vertical_sine, vertical_cosine),
                "tilt_linf": torch.amax(torch.abs(states[..., 9:11]), dim=-1),
                "progress": progress,
            }
            output = torch.stack([all_values[name] for name in variables], dim=-1)
            return output.squeeze(0) if squeeze else output
        if not self.is_planar:
            raise ValueError(
                f"unsupported feature library {self.feature_library_version!r} "
                f"for raw state dimension {self.raw_state_dimension}"
            )
        velocity = torch.zeros_like(states)
        velocity[:, :-1] = states[:, 1:] - states[:, :-1]
        velocity[:, -1] = velocity[:, -2]
        acceleration = torch.zeros_like(states)
        acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        acceleration[:, 0] = acceleration[:, 1]
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        acceleration_norm = torch.linalg.vector_norm(acceleration, dim=-1)
        safe_speed = torch.clamp(speed, min=1.0e-8)
        heading_cos = torch.where(speed > 1.0e-8, velocity[..., 0] / safe_speed, torch.zeros_like(speed))
        heading_sin = torch.where(speed > 1.0e-8, velocity[..., 1] / safe_speed, torch.zeros_like(speed))
        path_length_so_far = torch.cumsum(speed, dim=1) - speed
        horizon = states.shape[1]
        progress = torch.linspace(0.0, 1.0, horizon, dtype=states.dtype, device=states.device)
        progress = progress[None, :].expand(states.shape[0], -1)
        all_values = {
            "x_position": states[..., 0],
            "y_position": states[..., 1],
            "x_velocity": velocity[..., 0],
            "y_velocity": velocity[..., 1],
            "speed": speed,
            "x_acceleration": acceleration[..., 0],
            "y_acceleration": acceleration[..., 1],
            "acceleration_norm": acceleration_norm,
            "heading_sin": heading_sin,
            "heading_cos": heading_cos,
            "path_length_so_far": path_length_so_far,
            "progress": progress,
        }
        output = torch.stack([all_values[name] for name in variables], dim=-1)
        return output.squeeze(0) if squeeze else output

    def grid_features(
        self,
        points: np.ndarray,
        variables: tuple[str, ...],
        *,
        nominal_progress: float = 0.5,
    ) -> np.ndarray:
        """State features for evaluation; dynamic features use zero velocity."""
        self.validate_variables(variables)
        if not self.is_planar:
            raise NotImplementedError(
                "static x-y grid features are defined only for planar tasks"
            )
        points = np.asarray(points, dtype=np.float32)
        values = {
            "x_position": points[:, 0],
            "y_position": points[:, 1],
            "x_velocity": np.zeros(len(points), dtype=np.float32),
            "y_velocity": np.zeros(len(points), dtype=np.float32),
            "speed": np.zeros(len(points), dtype=np.float32),
            "x_acceleration": np.zeros(len(points), dtype=np.float32),
            "y_acceleration": np.zeros(len(points), dtype=np.float32),
            "acceleration_norm": np.zeros(len(points), dtype=np.float32),
            "heading_sin": np.zeros(len(points), dtype=np.float32),
            "heading_cos": np.zeros(len(points), dtype=np.float32),
            "path_length_so_far": np.zeros(len(points), dtype=np.float32),
            "progress": np.full(len(points), nominal_progress, dtype=np.float32),
        }
        return np.column_stack([values[name] for name in variables]).astype(np.float32)

    def _numpy_all(self, states: np.ndarray) -> dict[str, np.ndarray]:
        if states.ndim != 2 or states.shape[1] != self.raw_state_dimension:
            raise ValueError(
                f"states must have shape [T,{self.raw_state_dimension}]"
            )
        if self.is_carrywater_active:
            velocity = states[:, 6:9]
            roll = states[:, 9]
            pitch = states[:, 10]
            vertical_cosine = np.cos(roll) * np.cos(pitch)
            vertical_sine = np.linalg.norm(
                np.column_stack(
                    (
                        np.sin(pitch),
                        np.sin(roll) * np.cos(pitch),
                    )
                ),
                axis=1,
            )
            return {
                "x_position": states[:, 0],
                "y_position": states[:, 1],
                "z_position": states[:, 2],
                "target_dx": states[:, 3],
                "target_dy": states[:, 4],
                "target_dz": states[:, 5],
                "x_velocity": velocity[:, 0],
                "y_velocity": velocity[:, 1],
                "z_velocity": velocity[:, 2],
                "speed": np.linalg.norm(velocity, axis=1),
                "roll": roll,
                "pitch": pitch,
                "yaw": states[:, 11],
                "abs_roll": np.abs(roll),
                "abs_pitch": np.abs(pitch),
                "tilt_from_vertical": np.arctan2(vertical_sine, vertical_cosine),
                "tilt_linf": np.max(np.abs(states[:, 9:11]), axis=1),
                "progress": np.linspace(0.0, 1.0, len(states), dtype=np.float32),
            }
        if not self.is_planar:
            raise ValueError(
                f"unsupported feature library {self.feature_library_version!r}"
            )
        velocity = np.zeros_like(states)
        velocity[:-1] = np.diff(states, axis=0)
        velocity[-1] = velocity[-2]
        acceleration = np.zeros_like(states)
        acceleration[1:] = np.diff(velocity, axis=0)
        acceleration[0] = acceleration[1]
        speed = np.linalg.norm(velocity, axis=1)
        safe_speed = np.maximum(speed, 1.0e-8)
        moving = speed > 1.0e-8
        heading_cos = np.zeros_like(speed)
        heading_sin = np.zeros_like(speed)
        heading_cos[moving] = velocity[moving, 0] / safe_speed[moving]
        heading_sin[moving] = velocity[moving, 1] / safe_speed[moving]
        path_length_so_far = np.cumsum(speed) - speed
        return {
            "x_position": states[:, 0],
            "y_position": states[:, 1],
            "x_velocity": velocity[:, 0],
            "y_velocity": velocity[:, 1],
            "speed": speed,
            "x_acceleration": acceleration[:, 0],
            "y_acceleration": acceleration[:, 1],
            "acceleration_norm": np.linalg.norm(acceleration, axis=1),
            "heading_sin": heading_sin,
            "heading_cos": heading_cos,
            "path_length_so_far": path_length_so_far,
            "progress": np.linspace(0.0, 1.0, len(states), dtype=np.float32),
        }


def load_expert_dataset(dataset_dir: str | Path, split: str) -> list[Trajectory]:
    """Read a public split without opening the private evaluation directory."""
    root = Path(dataset_dir)
    task_spec = load_task_spec(root) if (root / "task_spec.json").is_file() else None
    raw_state_dimension = 2 if task_spec is None else task_spec.raw_state_dimension
    action_dimension = 2 if task_spec is None else task_spec.action_dimension
    action_horizon = "state_aligned" if task_spec is None else task_spec.action_horizon
    dt = 1.0 if task_spec is None else task_spec.dt
    archive_path = root / "expert_trajectories.npz"
    splits_path = root / "splits.json"
    if not archive_path.is_file() or not splits_path.is_file():
        raise FileNotFoundError(f"invalid expert dataset directory: {root}")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    split_sets = {name: set(map(str, values)) for name, values in splits.items()}
    names = sorted(split_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(f"dataset splits overlap: {left}/{right}: {sorted(overlap)}")
    if split == "all":
        selected: set[str] | None = None
    elif split in splits:
        selected = set(map(str, splits[split]))
    else:
        raise ValueError(f"unknown split: {split}")
    result: list[Trajectory] = []
    with np.load(archive_path, allow_pickle=False) as archive:
        ids = [str(value) for value in archive["trajectory_ids"].tolist()]
        if len(ids) != len(set(ids)):
            raise ValueError("trajectory_ids must be unique")
        union = set().union(*split_sets.values()) if split_sets else set()
        if union != set(ids):
            raise ValueError("dataset splits must cover every trajectory_id exactly once")
        observations = archive["observations"]
        actions = archive["actions"]
        if observations.ndim != 3 or observations.shape[-1] != raw_state_dimension:
            raise ValueError(
                "observations must have shape "
                f"[N,T,{raw_state_dimension}]"
            )
        expected_action_steps = observations.shape[1] - int(action_horizon == "transition")
        if (
            actions.ndim != 3
            or actions.shape[0] != observations.shape[0]
            or actions.shape[1] != expected_action_steps
            or actions.shape[2] != action_dimension
        ):
            raise ValueError(
                "actions must have shape "
                f"[N,{expected_action_steps},{action_dimension}]"
            )
        if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(actions)):
            raise ValueError("expert archive contains non-finite values")
        if "labels" in archive and np.any(np.asarray(archive["labels"]) != 0):
            raise ValueError("expert archive must contain only known-safe trajectories")
        if "lengths" in archive and np.any(np.asarray(archive["lengths"]) != observations.shape[1]):
            raise ValueError("expert archive lengths must equal the fixed horizon")
        if task_spec is None or task_spec.trajectory_adapter == "planar_waypoint_v1":
            expected_actions = np.zeros_like(observations, dtype=np.float32)
            expected_actions[:, :-1] = np.diff(observations, axis=1)
            if not np.allclose(actions, expected_actions, atol=1.0e-6):
                raise ValueError("expert actions do not match displacement convention")
        for index, trajectory_id in enumerate(ids):
            if selected is not None and trajectory_id not in selected:
                continue
            result.append(
                Trajectory(
                    states=archive["observations"][index],
                    actions=archive["actions"][index],
                    dt=dt,
                    metadata={
                        "trajectory_id": trajectory_id,
                        "source": "expert",
                        "split": split,
                        "trajectory_adapter": (
                            "planar_waypoint_v1"
                            if task_spec is None
                            else task_spec.trajectory_adapter
                        ),
                    },
                )
            )
    if not result:
        raise ValueError(f"split {split!r} is empty")
    return result


def load_candidate_pool(dataset_dir: str | Path) -> list[Trajectory]:
    """Load public, unlabeled, dynamically valid membership-query candidates."""

    root = Path(dataset_dir)
    task_spec = load_task_spec(root)
    archive_path = root / "candidate_trajectories.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        allowed_fields = {
            "observations",
            "actions",
            "trajectory_ids",
            "pair_ids",
            "pair_members",
            "reference_xyz",
            "phase_ids",
            "lengths",
        }
        unexpected = sorted(set(archive.files) - allowed_fields)
        if unexpected:
            raise ValueError(
                "public candidate pool contains unregistered or potentially "
                f"private fields: {unexpected}"
            )
        required_fields = {
            "observations",
            "actions",
            "trajectory_ids",
            "pair_ids",
            "pair_members",
        }
        missing = sorted(required_fields - set(archive.files))
        if missing:
            raise ValueError(f"candidate archive is missing fields: {missing}")
        observations = np.asarray(archive["observations"], dtype=np.float32)
        actions = np.asarray(archive["actions"], dtype=np.float32)
        trajectory_ids = np.asarray(archive["trajectory_ids"]).astype(str)
        pair_ids = np.asarray(archive["pair_ids"]).astype(str)
        pair_members = np.asarray(archive["pair_members"], dtype=np.int64)
        expected_action_steps = task_spec.horizon - int(task_spec.action_horizon == "transition")
        if observations.shape != (
            len(trajectory_ids),
            task_spec.horizon,
            task_spec.raw_state_dimension,
        ):
            raise ValueError("candidate observations do not match TaskSpec")
        if actions.shape != (
            len(trajectory_ids),
            expected_action_steps,
            task_spec.action_dimension,
        ):
            raise ValueError("candidate actions do not match TaskSpec")
        if pair_ids.shape != trajectory_ids.shape or pair_members.shape != trajectory_ids.shape:
            raise ValueError("candidate pair metadata is misaligned")
        if len(trajectory_ids) % 2 != 0:
            raise ValueError("candidate pool must contain complete two-member pairs")
        for start in range(0, len(trajectory_ids), 2):
            if pair_ids[start] != pair_ids[start + 1]:
                raise ValueError("candidate pair rows must be adjacent")
            if set(map(int, pair_members[start : start + 2])) != {0, 1}:
                raise ValueError("every candidate pair must contain members 0 and 1")
        unique_pair_ids, pair_counts = np.unique(pair_ids, return_counts=True)
        if len(unique_pair_ids) * 2 != len(pair_ids) or np.any(pair_counts != 2):
            raise ValueError("every candidate pair_id must occur exactly twice")
        if len(set(trajectory_ids.tolist())) != len(trajectory_ids):
            raise ValueError("candidate trajectory_ids must be unique")
        if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(actions)):
            raise ValueError("candidate archive contains non-finite values")
        result = [
            Trajectory(
                states=observations[index],
                actions=actions[index],
                dt=task_spec.dt,
                metadata={
                    "trajectory_id": str(trajectory_id),
                    "source": "public_unlabeled_candidate_pool",
                    "candidate_pair_id": str(pair_ids[index]),
                    "candidate_pair_member": int(pair_members[index]),
                    "trajectory_adapter": task_spec.trajectory_adapter,
                },
            )
            for index, trajectory_id in enumerate(trajectory_ids)
        ]
    if not result:
        raise ValueError("candidate pool is empty")
    return result


def load_public_workspace(dataset_dir: str | Path) -> tuple[tuple[float, float], tuple[float, float]]:
    task_spec_path = Path(dataset_dir) / "task_spec.json"
    if task_spec_path.is_file():
        spec = load_task_spec(dataset_dir)
        return spec.workspace_x, spec.workspace_y
    manifest = Path(dataset_dir) / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return (0.0, 10.0), (-4.0, 4.0)
