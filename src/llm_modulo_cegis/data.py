"""Public ObstacleAvoid data and a differentiable feature library."""

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


class FeatureLibrary:
    """Compile semantic variables to NumPy and differentiable Torch features."""

    def __init__(self, specs: tuple[FeatureSpec, ...] = FEATURE_SPECS) -> None:
        self.specs = specs
        self._by_name = {spec.name: spec for spec in specs}
        if len(self._by_name) != len(specs):
            raise ValueError("feature names must be unique")

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
            raise ValueError("states must have shape [T,2] or [B,T,2]")
        if states.shape[-1] != 2:
            raise ValueError("raw states must be planar")
        velocity = torch.zeros_like(states)
        velocity[:, :-1] = states[:, 1:] - states[:, :-1]
        velocity[:, -1] = velocity[:, -2]
        horizon = states.shape[1]
        progress = torch.linspace(0.0, 1.0, horizon, dtype=states.dtype, device=states.device)
        progress = progress[None, :].expand(states.shape[0], -1)
        all_values = {
            "x_position": states[..., 0],
            "y_position": states[..., 1],
            "x_velocity": velocity[..., 0],
            "y_velocity": velocity[..., 1],
            "speed": torch.linalg.vector_norm(velocity, dim=-1),
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
        points = np.asarray(points, dtype=np.float32)
        values = {
            "x_position": points[:, 0],
            "y_position": points[:, 1],
            "x_velocity": np.zeros(len(points), dtype=np.float32),
            "y_velocity": np.zeros(len(points), dtype=np.float32),
            "speed": np.zeros(len(points), dtype=np.float32),
            "progress": np.full(len(points), nominal_progress, dtype=np.float32),
        }
        return np.column_stack([values[name] for name in variables]).astype(np.float32)

    @staticmethod
    def _numpy_all(states: np.ndarray) -> dict[str, np.ndarray]:
        if states.ndim != 2 or states.shape[1] != 2:
            raise ValueError("states must have shape [T,2]")
        velocity = np.zeros_like(states)
        velocity[:-1] = np.diff(states, axis=0)
        velocity[-1] = velocity[-2]
        return {
            "x_position": states[:, 0],
            "y_position": states[:, 1],
            "x_velocity": velocity[:, 0],
            "y_velocity": velocity[:, 1],
            "speed": np.linalg.norm(velocity, axis=1),
            "progress": np.linspace(0.0, 1.0, len(states), dtype=np.float32),
        }


def load_expert_dataset(dataset_dir: str | Path, split: str) -> list[Trajectory]:
    """Read a public split without opening the private evaluation directory."""
    root = Path(dataset_dir)
    archive_path = root / "expert_trajectories.npz"
    splits_path = root / "splits.json"
    if not archive_path.is_file() or not splits_path.is_file():
        raise FileNotFoundError(f"invalid ObstacleAvoid dataset directory: {root}")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if split == "all":
        selected: set[str] | None = None
    elif split in splits:
        selected = set(map(str, splits[split]))
    else:
        raise ValueError(f"unknown split: {split}")
    result: list[Trajectory] = []
    with np.load(archive_path, allow_pickle=False) as archive:
        ids = [str(value) for value in archive["trajectory_ids"].tolist()]
        for index, trajectory_id in enumerate(ids):
            if selected is not None and trajectory_id not in selected:
                continue
            result.append(
                Trajectory(
                    states=archive["observations"][index],
                    actions=archive["actions"][index],
                    metadata={"trajectory_id": trajectory_id, "source": "expert", "split": split},
                )
            )
    if not result:
        raise ValueError(f"split {split!r} is empty")
    return result


def load_public_workspace(dataset_dir: str | Path) -> tuple[tuple[float, float], tuple[float, float]]:
    manifest = Path(dataset_dir) / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return (0.0, 10.0), (-4.0, 4.0)
