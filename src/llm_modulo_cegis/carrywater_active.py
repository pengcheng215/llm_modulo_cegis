"""Public dynamics and validity checks for CarryWaterActive trajectories.

The safety rule is deliberately absent from this module.  It validates only
public feasibility: observation consistency, deterministic dynamics, action
bounds, and broad task-domain bounds.  Invalid rollouts must be rejected before
they can consume a membership-Oracle query.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Trajectory


OBSERVATION_DIM = 12
ACTION_DIM = 6
VELOCITY_DECAY = 0.98
MAX_ABS_ACCELERATION = 2.0
MAX_ABS_ANGULAR_RATE = 1.5
POSITION_LOW = np.asarray((-1.5, -1.5, 0.2), dtype=np.float64)
POSITION_HIGH = np.asarray((1.5, 1.5, 1.2), dtype=np.float64)
REFERENCE_LOW = np.asarray((-1.6, -1.6, 0.2), dtype=np.float64)
REFERENCE_HIGH = np.asarray((1.6, 1.6, 1.2), dtype=np.float64)
MAX_ABS_DYNAMIC_VELOCITY = 1.2
MAX_ABS_ORIENTATION = np.asarray((np.pi, np.pi, np.pi), dtype=np.float64)


@dataclass(frozen=True)
class CarryWaterValidity:
    valid: bool
    reason: str
    maximum_residual: float


def wrap_angles(values: np.ndarray) -> np.ndarray:
    """Map Euler-angle channels to the public ``[-pi, pi)`` convention."""

    array = np.asarray(values, dtype=np.float64)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def reconstruct_observations(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    initial_orientation: np.ndarray,
    actions: np.ndarray,
    reference_xyz: np.ndarray,
    *,
    dt: float,
) -> np.ndarray:
    """Roll out the registered public point-cup dynamics."""

    actions = np.asarray(actions, dtype=np.float64)
    reference_xyz = np.asarray(reference_xyz, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError("actions must have shape [T-1,6]")
    horizon = actions.shape[0] + 1
    if reference_xyz.shape != (horizon, 3):
        raise ValueError("reference_xyz must have shape [T,3]")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    position = np.empty((horizon, 3), dtype=np.float64)
    velocity = np.empty((horizon, 3), dtype=np.float64)
    orientation = np.empty((horizon, 3), dtype=np.float64)
    position[0] = np.asarray(initial_position, dtype=np.float64)
    velocity[0] = np.asarray(initial_velocity, dtype=np.float64)
    orientation[0] = np.asarray(initial_orientation, dtype=np.float64)
    for index in range(horizon - 1):
        velocity[index + 1] = (
            VELOCITY_DECAY * velocity[index] + dt * actions[index, :3]
        )
        position[index + 1] = position[index] + dt * velocity[index + 1]
        orientation[index + 1] = wrap_angles(
            orientation[index] + dt * actions[index, 3:]
        )
    observations = np.column_stack(
        (
            position,
            reference_xyz - position,
            velocity,
            orientation,
        )
    )
    return observations.astype(np.float32)


def inverse_dynamics_actions(
    position: np.ndarray,
    orientation: np.ndarray,
    *,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct actions that exactly realize smooth position/orientation paths."""

    position = np.asarray(position, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3 or position.shape[0] < 2:
        raise ValueError("position must have shape [T,3]")
    if orientation.shape != position.shape:
        raise ValueError("orientation must have shape [T,3]")
    velocity = np.empty_like(position)
    velocity[1:] = np.diff(position, axis=0) / float(dt)
    velocity[0] = velocity[1]
    acceleration = (velocity[1:] - VELOCITY_DECAY * velocity[:-1]) / float(dt)
    angular_rate = wrap_angles(np.diff(orientation, axis=0)) / float(dt)
    actions = np.column_stack((acceleration, angular_rate))
    return velocity.astype(np.float32), actions.astype(np.float32)


def validate_trajectory(
    trajectory: Trajectory,
    *,
    atol: float = 2.5e-4,
) -> CarryWaterValidity:
    """Check public feasibility without evaluating any hidden safety clause."""

    states = np.asarray(trajectory.states, dtype=np.float64)
    actions = None if trajectory.actions is None else np.asarray(trajectory.actions, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != OBSERVATION_DIM:
        return CarryWaterValidity(False, "observation_shape", float("inf"))
    if actions is None or actions.shape != (len(states) - 1, ACTION_DIM):
        return CarryWaterValidity(False, "action_shape", float("inf"))
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        return CarryWaterValidity(False, "non_finite", float("inf"))
    if np.any(np.abs(actions[:, :3]) > MAX_ABS_ACCELERATION + atol):
        return CarryWaterValidity(False, "acceleration_bound", float("inf"))
    if np.any(np.abs(actions[:, 3:]) > MAX_ABS_ANGULAR_RATE + atol):
        return CarryWaterValidity(False, "angular_rate_bound", float("inf"))
    position = states[:, :3]
    reference = position + states[:, 3:6]
    velocity = states[:, 6:9]
    orientation = states[:, 9:12]
    if np.any(position < POSITION_LOW - atol) or np.any(position > POSITION_HIGH + atol):
        return CarryWaterValidity(False, "position_domain", float("inf"))
    if np.any(reference < REFERENCE_LOW - atol) or np.any(reference > REFERENCE_HIGH + atol):
        return CarryWaterValidity(False, "reference_domain", float("inf"))
    if np.any(np.abs(velocity) > MAX_ABS_DYNAMIC_VELOCITY + atol):
        return CarryWaterValidity(False, "velocity_domain", float("inf"))
    if np.any(np.abs(orientation) > MAX_ABS_ORIENTATION + atol):
        return CarryWaterValidity(False, "orientation_domain", float("inf"))
    reconstructed = reconstruct_observations(
        position[0],
        velocity[0],
        orientation[0],
        actions,
        reference,
        dt=trajectory.dt,
    ).astype(np.float64)
    residual = float(np.max(np.abs(reconstructed - states)))
    if residual > atol:
        return CarryWaterValidity(False, "dynamics_residual", residual)
    return CarryWaterValidity(True, "valid", residual)


__all__ = [
    "ACTION_DIM",
    "MAX_ABS_ACCELERATION",
    "MAX_ABS_ANGULAR_RATE",
    "OBSERVATION_DIM",
    "VELOCITY_DECAY",
    "CarryWaterValidity",
    "inverse_dynamics_actions",
    "reconstruct_observations",
    "validate_trajectory",
    "wrap_angles",
]
