"""Generate the CarryWaterActive active constraint-identification benchmark.

The learner-visible bundle contains a public task/dynamics contract, known-safe
expert trajectories, and an unlabeled candidate pool.  The private bundle
contains the analytic three-clause membership rule and a balanced matched-pair
evaluation bank.  Every observation in every archive is produced by the same
public dynamics rollout and is revalidated before export.

The generator deliberately creates representation collisions that distinguish
reference-relative height from world height, three-dimensional speed from any
single velocity component, and physical cup tilt from yaw.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np


SUITE_NAME = "CarryWaterActive"
SUITE_VERSION = "1.0.0"
SCHEMA_VERSION = 1
TASK_SPEC_SCHEMA_VERSION = 2
TASK_ID = "carrywater_active"

HORIZON = 120
DT = 0.05
OBSERVATION_DIMENSION = 12
ACTION_DIMENSION = 6

VELOCITY_DECAY = 0.98
LINEAR_ACCELERATION_LIMIT = 2.0
ANGULAR_RATE_LIMIT = 1.5
VELOCITY_COMPONENT_LIMIT = 1.2

WORKSPACE_LOW = np.asarray([-1.50, -1.50, 0.20], dtype=np.float64)
WORKSPACE_HIGH = np.asarray([1.50, 1.50, 1.20], dtype=np.float64)
REFERENCE_LOW = np.asarray([-1.60, -1.60, 0.20], dtype=np.float64)
REFERENCE_HIGH = np.asarray([1.60, 1.60, 1.20], dtype=np.float64)

SAFE_LABEL = 0
VIOLATION_LABEL = 1

PHASE_NAMES = ("lift", "transport", "place")
CLAUSE_IDS = ("height", "speed", "tilt")

TASK_DESCRIPTION = (
    "Carry an open cup through lift, transport, and placement while following "
    "the requested height reference. Motion should remain gentle regardless of "
    "travel direction, and the cup should remain upright; rotation about the "
    "vertical axis is allowed. Infer the qualitative constraint structure and "
    "learn its unknown numerical boundaries."
)


OBSERVATION_SCHEMA = (
    ("world_x", "cup world-frame x position", "m", -1.50, 1.50, "position"),
    ("world_y", "cup world-frame y position", "m", -1.50, 1.50, "position"),
    ("world_z", "cup world-frame z position", "m", 0.20, 1.20, "position"),
    (
        "target_dx",
        "current public task-reference x minus cup x",
        "m",
        -3.10,
        3.10,
        "reference_relative",
    ),
    (
        "target_dy",
        "current public task-reference y minus cup y",
        "m",
        -3.10,
        3.10,
        "reference_relative",
    ),
    (
        "target_dz",
        "current public task-reference z minus cup z",
        "m",
        -1.00,
        1.00,
        "reference_relative",
    ),
    ("vx", "cup world-frame x velocity", "m/s", -1.20, 1.20, "velocity"),
    ("vy", "cup world-frame y velocity", "m/s", -1.20, 1.20, "velocity"),
    ("vz", "cup world-frame z velocity", "m/s", -1.20, 1.20, "velocity"),
    ("roll", "cup roll angle", "rad", -math.pi, math.pi, "orientation"),
    ("pitch", "cup pitch angle", "rad", -math.pi, math.pi, "orientation"),
    ("yaw", "cup yaw angle", "rad", -math.pi, math.pi, "orientation"),
)

ACTION_SCHEMA = (
    ("ax", "world-frame x acceleration command", "m/s^2", -2.0, 2.0),
    ("ay", "world-frame y acceleration command", "m/s^2", -2.0, 2.0),
    ("az", "world-frame z acceleration command", "m/s^2", -2.0, 2.0),
    ("omega_roll", "roll angular-rate command", "rad/s", -1.5, 1.5),
    ("omega_pitch", "pitch angular-rate command", "rad/s", -1.5, 1.5),
    ("omega_yaw", "yaw angular-rate command", "rad/s", -1.5, 1.5),
)

# This is the exact learner-facing feature vocabulary implemented by
# data.CARRYWATER_ACTIVE_FEATURE_SPECS.  Raw state names live in the separate
# dynamics contract; hypotheses are written against these semantic names.
FEATURE_SCHEMA = (
    ("x_position", "end-effector world x position", "m", -3.0, 3.0, "position"),
    ("y_position", "end-effector world y position", "m", -3.0, 3.0, "position"),
    ("z_position", "end-effector world z position", "m", 0.1, 1.2, "position"),
    ("target_dx", "target x minus current x", "m", -5.0, 5.0, "relative_position"),
    ("target_dy", "target y minus current y", "m", -5.0, 5.0, "relative_position"),
    ("target_dz", "requested carrying height minus current z", "m", -0.4, 0.4, "relative_position"),
    ("x_velocity", "observed horizontal x velocity", "m/s", -1.5, 1.5, "velocity"),
    ("y_velocity", "observed horizontal y velocity", "m/s", -1.5, 1.5, "velocity"),
    ("z_velocity", "observed vertical velocity", "m/s", -1.5, 1.5, "velocity"),
    ("speed", "three-dimensional translational speed magnitude", "m/s", 0.0, 2.0, "velocity"),
    ("roll", "signed cup roll angle", "rad", -0.8, 0.8, "orientation"),
    ("pitch", "signed cup pitch angle", "rad", -0.8, 0.8, "orientation"),
    ("yaw", "signed cup yaw angle", "rad", -3.2, 3.2, "orientation"),
    ("abs_roll", "absolute cup roll angle", "rad", 0.0, 0.8, "orientation"),
    ("abs_pitch", "absolute cup pitch angle", "rad", 0.0, 0.8, "orientation"),
    (
        "tilt_from_vertical",
        "cup tilt angle from vertical derived from roll and pitch",
        "rad",
        0.0,
        1.0,
        "orientation",
    ),
    ("tilt_linf", "maximum absolute roll or pitch", "rad", 0.0, 0.8, "orientation"),
    ("progress", "normalized trajectory time", "ratio", 0.0, 1.0, "time"),
)


@dataclass(frozen=True)
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    reference_xyz: np.ndarray
    phase_ids: np.ndarray


def _wrap_angle(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_array_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _public_dynamics_rollout(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    initial_orientation: np.ndarray,
    actions: np.ndarray,
    reference_xyz: np.ndarray,
    phase_ids: np.ndarray,
) -> Episode:
    """Roll out the one public deterministic dynamics model."""

    actions = np.asarray(actions, dtype=np.float64)
    reference_xyz = np.asarray(reference_xyz, dtype=np.float64)
    phase_ids = np.asarray(phase_ids, dtype=np.int8)
    if actions.shape != (HORIZON - 1, ACTION_DIMENSION):
        raise ValueError(f"actions must have shape {(HORIZON - 1, ACTION_DIMENSION)}")
    if reference_xyz.shape != (HORIZON, 3):
        raise ValueError(f"reference_xyz must have shape {(HORIZON, 3)}")
    if phase_ids.shape != (HORIZON,):
        raise ValueError(f"phase_ids must have shape {(HORIZON,)}")
    if not np.all(np.isin(phase_ids, np.arange(len(PHASE_NAMES), dtype=np.int8))):
        raise ValueError("phase_ids contain an unknown public phase")
    if np.any(np.abs(actions[:, :3]) > LINEAR_ACCELERATION_LIMIT + 1.0e-10):
        raise ValueError("linear acceleration action exceeds the public bound")
    if np.any(np.abs(actions[:, 3:]) > ANGULAR_RATE_LIMIT + 1.0e-10):
        raise ValueError("angular-rate action exceeds the public bound")
    if np.any(reference_xyz < REFERENCE_LOW - 1.0e-10) or np.any(
        reference_xyz > REFERENCE_HIGH + 1.0e-10
    ):
        raise ValueError("reference trajectory leaves the public reference domain")

    position = np.asarray(initial_position, dtype=np.float64).copy()
    velocity = np.asarray(initial_velocity, dtype=np.float64).copy()
    orientation = _wrap_angle(np.asarray(initial_orientation, dtype=np.float64))
    observations = np.empty((HORIZON, OBSERVATION_DIMENSION), dtype=np.float64)

    def emit(index: int) -> None:
        observations[index] = np.concatenate(
            (position, reference_xyz[index] - position, velocity, orientation)
        )

    emit(0)
    for index, action in enumerate(actions):
        velocity = np.clip(
            VELOCITY_DECAY * velocity + DT * action[:3],
            -VELOCITY_COMPONENT_LIMIT,
            VELOCITY_COMPONENT_LIMIT,
        )
        position = position + DT * velocity
        orientation = _wrap_angle(orientation + DT * action[3:])
        emit(index + 1)

    if np.any(observations[:, :3] < WORKSPACE_LOW - 1.0e-8) or np.any(
        observations[:, :3] > WORKSPACE_HIGH + 1.0e-8
    ):
        raise ValueError("public rollout left the workspace")
    return Episode(
        observations=observations.astype(np.float32),
        actions=actions.astype(np.float32),
        reference_xyz=reference_xyz.astype(np.float32),
        phase_ids=phase_ids.copy(),
    )


def _validate_episode(episode: Episode, *, atol: float = 2.0e-5) -> None:
    observations = np.asarray(episode.observations, dtype=np.float64)
    if observations.shape != (HORIZON, OBSERVATION_DIMENSION):
        raise ValueError("observations have the wrong fixed shape")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations contain non-finite values")
    expected_reference = observations[:, :3] + observations[:, 3:6]
    if not np.allclose(expected_reference, episode.reference_xyz, atol=atol, rtol=0.0):
        raise ValueError("reference-relative observation channels disagree with sidecar context")
    replay = _public_dynamics_rollout(
        observations[0, :3],
        observations[0, 6:9],
        observations[0, 9:12],
        episode.actions,
        episode.reference_xyz,
        episode.phase_ids,
    )
    if not np.allclose(replay.observations, observations, atol=atol, rtol=0.0):
        maximum = float(np.max(np.abs(replay.observations - observations)))
        raise ValueError(f"episode does not replay under public dynamics: max error={maximum}")


def _private_rule(private_rng: np.random.Generator) -> dict[str, object]:
    height_half_width = float(private_rng.uniform(0.032, 0.042))
    speed_threshold = float(private_rng.uniform(0.42, 0.52))
    tilt_threshold = float(private_rng.uniform(0.14, 0.19))
    expected_clauses = [
        {
            "clause_id": "height",
            "variables": ["target_dz"],
            "coupling": "joint",
            "relation": "equality_band",
            "temporal_operator": "max",
            "model_family": "linear",
        },
        {
            "clause_id": "speed",
            "variables": ["speed"],
            "coupling": "joint",
            "relation": "upper_bound",
            "temporal_operator": "max",
            "model_family": "linear",
        },
        {
            "clause_id": "tilt",
            "variables": ["tilt_from_vertical"],
            "coupling": "joint",
            "relation": "upper_bound",
            "temporal_operator": "max",
            "model_family": "linear",
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_instance_id": TASK_ID,
        "composition": "any_violation",
        "observation_dimension": OBSERVATION_DIMENSION,
        "action_dimension": ACTION_DIMENSION,
        "horizon": HORIZON,
        "dt": DT,
        "label_convention": {"safe": SAFE_LABEL, "violation": VIOLATION_LABEL},
        "warning": "Private analytic membership rule; never expose to the learner or semantic reasoner.",
        "clauses": [
            {
                "clause_id": "height",
                "kind": "equality_band",
                "feature": "reference_dz",
                "center": 0.0,
                "half_width": height_half_width,
            },
            {
                "clause_id": "speed",
                "kind": "l2_upper_bound",
                "features": ["vx", "vy", "vz"],
                "threshold": speed_threshold,
            },
            {
                "clause_id": "tilt",
                "kind": "upright_tilt_upper_bound",
                "features": ["roll", "pitch"],
                "formula": "acos(cos(roll)*cos(pitch))",
                "threshold": tilt_threshold,
            },
        ],
        "expected_structure": {
            "representable": True,
            "composition": "any_violation",
            "clauses": expected_clauses,
        },
    }


def _rule_values(rule: dict[str, object]) -> tuple[float, float, float]:
    clauses = rule["clauses"]
    assert isinstance(clauses, list) and len(clauses) == 3
    return (
        float(clauses[0]["half_width"]),
        float(clauses[1]["threshold"]),
        float(clauses[2]["threshold"]),
    )


def _tilt_from_vertical(observations: np.ndarray) -> np.ndarray:
    roll = np.asarray(observations, dtype=np.float64)[..., 9]
    pitch = np.asarray(observations, dtype=np.float64)[..., 10]
    cosine = np.clip(np.cos(roll) * np.cos(pitch), -1.0, 1.0)
    return np.arccos(cosine)


def _clause_state_values(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float64)
    height = np.abs(observations[:, 5])
    speed = np.linalg.norm(observations[:, 6:9], axis=1)
    tilt = _tilt_from_vertical(observations)
    return np.column_stack((height, speed, tilt))


def _normalized_severities(episode: Episode, rule: dict[str, object]) -> np.ndarray:
    thresholds = np.asarray(_rule_values(rule), dtype=np.float64)
    return np.max(_clause_state_values(episode.observations) / thresholds[None, :] - 1.0, axis=0)


def _clause_labels(episode: Episode, rule: dict[str, object]) -> np.ndarray:
    return (_normalized_severities(episode, rule) > 0.0).astype(np.int8)


def _state_clause_masks(episode: Episode, rule: dict[str, object]) -> np.ndarray:
    thresholds = np.asarray(_rule_values(rule), dtype=np.float64)
    return (_clause_state_values(episode.observations) > thresholds[None, :]).astype(np.int8)


def _trajectory_label(episode: Episode, rule: dict[str, object]) -> int:
    return VIOLATION_LABEL if bool(np.any(_clause_labels(episode, rule))) else SAFE_LABEL


def _phase_layout(variant: int) -> np.ndarray:
    lift = 23 + (variant % 7)
    place = 23 + ((variant * 3 + 2) % 7)
    transport = HORIZON - lift - place
    if transport < 55:
        raise AssertionError("transport phase unexpectedly short")
    return np.concatenate(
        (
            np.zeros(lift, dtype=np.int8),
            np.ones(transport, dtype=np.int8),
            np.full(place, 2, dtype=np.int8),
        )
    )


def _phase_bump(phase_ids: np.ndarray, phase: int) -> np.ndarray:
    indices = np.flatnonzero(phase_ids == phase)
    if len(indices) < 5:
        raise ValueError("phase has insufficient states")
    local = np.linspace(0.0, np.pi, len(indices), dtype=np.float64)
    bump = np.zeros(HORIZON, dtype=np.float64)
    bump[indices] = np.sin(local) ** 2
    return bump


def _whole_trajectory_wave(phase: float = 0.0) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, HORIZON, dtype=np.float64)
    envelope = np.sin(np.pi * progress) ** 2
    return envelope * np.sin(2.0 * np.pi * progress + phase)


def _position_from_velocity(initial_position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    position = np.empty((HORIZON, 3), dtype=np.float64)
    position[0] = np.asarray(initial_position, dtype=np.float64)
    for index in range(HORIZON - 1):
        position[index + 1] = position[index] + DT * velocity[index + 1]
    return position


def _actions_from_profiles(velocity: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    actions = np.empty((HORIZON - 1, ACTION_DIMENSION), dtype=np.float64)
    actions[:, :3] = (velocity[1:] - VELOCITY_DECAY * velocity[:-1]) / DT
    actions[:, 3:] = _wrap_angle(orientation[1:] - orientation[:-1]) / DT
    return actions


def _build_episode_from_profiles(
    initial_position: np.ndarray,
    velocity: np.ndarray,
    orientation: np.ndarray,
    reference_error_z: np.ndarray,
    phase_ids: np.ndarray,
    reference_xy_offsets: np.ndarray | None = None,
) -> Episode:
    velocity = np.asarray(velocity, dtype=np.float64)
    orientation = _wrap_angle(np.asarray(orientation, dtype=np.float64))
    reference_error_z = np.asarray(reference_error_z, dtype=np.float64)
    if velocity.shape != (HORIZON, 3) or orientation.shape != (HORIZON, 3):
        raise ValueError("state profiles have wrong shape")
    if reference_error_z.shape != (HORIZON,):
        raise ValueError("reference_error_z has wrong shape")
    position = _position_from_velocity(initial_position, velocity)
    if reference_xy_offsets is None:
        progress = np.linspace(0.0, 1.0, HORIZON, dtype=np.float64)
        reference_xy_offsets = np.column_stack(
            (
                0.025 * np.sin(2.0 * np.pi * progress) * np.sin(np.pi * progress) ** 2,
                0.025 * np.cos(2.0 * np.pi * progress) * np.sin(np.pi * progress) ** 2,
            )
        )
    reference_xyz = position.copy()
    reference_xyz[:, :2] += np.asarray(reference_xy_offsets, dtype=np.float64)
    reference_xyz[:, 2] += reference_error_z
    actions = _actions_from_profiles(velocity, orientation)
    episode = _public_dynamics_rollout(
        position[0],
        velocity[0],
        orientation[0],
        actions,
        reference_xyz,
        phase_ids,
    )
    _validate_episode(episode)
    return episode


def _motion_profiles(
    *,
    direction_bin: int,
    height_bin: int,
    style: int,
    phase_variant: int,
    speed_peak: float,
    height_error_amplitude: float,
    tilt_amplitude: float,
    tilt_azimuth: float,
    yaw_offset: float,
    yaw_rate: float,
    rng: np.random.Generator,
) -> Episode:
    phase_ids = _phase_layout(phase_variant)
    direction_angle = 2.0 * np.pi * (direction_bin % 8) / 8.0
    direction = np.asarray((np.cos(direction_angle), np.sin(direction_angle)))
    perpendicular = np.asarray((-direction[1], direction[0]))

    velocity = np.zeros((HORIZON, 3), dtype=np.float64)
    lift_indices = np.flatnonzero(phase_ids == 0)
    transport_indices = np.flatnonzero(phase_ids == 1)
    place_indices = np.flatnonzero(phase_ids == 2)

    carry_heights = np.asarray((0.46, 0.58, 0.72, 0.86), dtype=np.float64)
    carry_height = float(carry_heights[height_bin % len(carry_heights)] + rng.uniform(-0.012, 0.012))
    lift_delta = float(rng.uniform(0.10, 0.16))
    place_delta = float(rng.uniform(0.07, 0.14))
    initial_z = carry_height - lift_delta

    lift_weights = np.sin(np.linspace(0.0, np.pi, len(lift_indices))) ** 2
    place_weights = np.sin(np.linspace(0.0, np.pi, len(place_indices))) ** 2
    velocity[lift_indices, 2] = lift_delta * lift_weights / (DT * max(np.sum(lift_weights), 1.0e-12))
    velocity[place_indices, 2] = -place_delta * place_weights / (
        DT * max(np.sum(place_weights), 1.0e-12)
    )

    local = np.linspace(0.0, 1.0, len(transport_indices), dtype=np.float64)
    forward = np.sin(np.pi * local) ** 2
    transverse = 0.16 * (1.0 if style % 2 == 0 else -1.0) * np.sin(2.0 * np.pi * local) * forward
    xy = forward[:, None] * direction[None, :] + transverse[:, None] * perpendicular[None, :]
    xy_norm = np.linalg.norm(xy, axis=1)
    xy *= float(speed_peak) / max(float(np.max(xy_norm)), 1.0e-12)
    velocity[transport_indices, :2] = xy

    max_speed = float(np.max(np.linalg.norm(velocity, axis=1)))
    if max_speed > speed_peak * 1.001:
        velocity[:, 2] *= float(speed_peak) / max_speed

    displacement = DT * np.sum(velocity[1:, :2], axis=0)
    initial_xy = -0.5 * displacement + rng.uniform(-0.08, 0.08, size=2)
    initial_position = np.asarray((initial_xy[0], initial_xy[1], initial_z), dtype=np.float64)

    height_wave = _whole_trajectory_wave(float(rng.uniform(-np.pi, np.pi)))
    peak = max(float(np.max(np.abs(height_wave))), 1.0e-12)
    reference_error_z = float(height_error_amplitude) * height_wave / peak

    orientation = np.zeros((HORIZON, 3), dtype=np.float64)
    tilt_wave = np.sin(np.pi * np.linspace(0.0, 1.0, HORIZON, dtype=np.float64)) ** 2
    orientation[:, 0] = float(tilt_amplitude) * np.cos(tilt_azimuth) * tilt_wave
    orientation[:, 1] = float(tilt_amplitude) * np.sin(tilt_azimuth) * tilt_wave
    orientation[:, 2] = _wrap_angle(
        yaw_offset + yaw_rate * DT * np.arange(HORIZON, dtype=np.float64)
    )

    return _build_episode_from_profiles(
        initial_position,
        velocity,
        orientation,
        reference_error_z,
        phase_ids,
    )


def _generate_experts(
    rule: dict[str, object],
    public_rng: np.random.Generator,
    count: int = 64,
) -> tuple[list[Episode], list[dict[str, int | str]]]:
    if count != 64:
        raise ValueError("CarryWaterActive v1 freezes exactly 64 experts")
    # Conservative design limits are below every possible private boundary.
    # Therefore fixed public_seed => byte-identical experts for every private seed.
    height_limit, speed_limit, tilt_limit = 0.032, 0.42, 0.14
    combinations = [(direction, height, style) for direction in range(8) for height in range(4) for style in range(2)]
    order = public_rng.permutation(len(combinations))
    near_kinds = ("height", "speed", "tilt", "interior")
    episodes: list[Episode] = []
    metadata: list[dict[str, int | str]] = []
    for output_index, combination_index in enumerate(order):
        direction, height, style = combinations[int(combination_index)]
        near_kind = near_kinds[output_index % len(near_kinds)]
        speed_fraction = public_rng.uniform(0.86, 0.94) if near_kind == "speed" else public_rng.uniform(0.42, 0.64)
        height_fraction = public_rng.uniform(0.86, 0.94) if near_kind == "height" else public_rng.uniform(0.22, 0.52)
        tilt_fraction = public_rng.uniform(0.86, 0.94) if near_kind == "tilt" else public_rng.uniform(0.22, 0.52)
        episode = _motion_profiles(
            direction_bin=direction,
            height_bin=height,
            style=style,
            phase_variant=output_index,
            speed_peak=float(speed_fraction * speed_limit),
            height_error_amplitude=float(height_fraction * height_limit),
            tilt_amplitude=float(tilt_fraction * tilt_limit),
            tilt_azimuth=2.0 * np.pi * ((output_index * 5) % 16) / 16.0,
            yaw_offset=float(public_rng.uniform(-np.pi, np.pi)),
            yaw_rate=float(public_rng.uniform(-0.65, 0.65)),
            rng=public_rng,
        )
        if _trajectory_label(episode, rule) != SAFE_LABEL:
            raise RuntimeError("expert generation produced an unsafe trajectory")
        episodes.append(episode)
        metadata.append(
            {
                "direction_bin": direction,
                "height_bin": height,
                "style": style,
                "near_kind": near_kind,
            }
        )
    return episodes, metadata


def _split_expert_ids(
    ids: np.ndarray,
    metadata: list[dict[str, int | str]],
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Create 40/12/12 splits while retaining every direction and height."""

    for _ in range(5000):
        validation: list[int] = []
        test: list[int] = []
        train: list[int] = []
        for direction in range(8):
            members = [index for index, row in enumerate(metadata) if row["direction_bin"] == direction]
            members = list(np.asarray(members)[rng.permutation(len(members))])
            validation_count = 2 if direction < 4 else 1
            test_count = 1 if direction < 4 else 2
            validation.extend(map(int, members[:validation_count]))
            test.extend(map(int, members[validation_count : validation_count + test_count]))
            train.extend(map(int, members[validation_count + test_count :]))
        if (len(train), len(validation), len(test)) != (40, 12, 12):
            raise AssertionError("expert split sizes are incorrect")
        valid = True
        for selected in (train, validation, test):
            directions = {int(metadata[index]["direction_bin"]) for index in selected}
            heights = {int(metadata[index]["height_bin"]) for index in selected}
            if directions != set(range(8)) or heights != set(range(4)):
                valid = False
        if valid:
            return {
                "train": [str(ids[index]) for index in sorted(train)],
                "validation": [str(ids[index]) for index in sorted(validation)],
                "test": [str(ids[index]) for index in sorted(test)],
            }
    raise RuntimeError("failed to create coverage-preserving expert splits")


def _episode_arrays(episodes: list[Episode], ids: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "observations": np.stack([item.observations for item in episodes]).astype(np.float32),
        "actions": np.stack([item.actions for item in episodes]).astype(np.float32),
        "reference_xyz": np.stack([item.reference_xyz for item in episodes]).astype(np.float32),
        "phase_ids": np.stack([item.phase_ids for item in episodes]).astype(np.int8),
        "lengths": np.full(len(episodes), HORIZON, dtype=np.int32),
        "trajectory_ids": np.asarray(ids),
    }


def _replace_profiles(
    base: Episode,
    velocity: np.ndarray,
    orientation: np.ndarray,
    reference_error_z: np.ndarray,
) -> Episode:
    initial_position = np.asarray(base.observations[0, :3], dtype=np.float64)
    reference_offsets = np.asarray(base.reference_xyz[:, :2] - base.observations[:, :2], dtype=np.float64)
    return _build_episode_from_profiles(
        initial_position,
        velocity,
        orientation,
        reference_error_z,
        base.phase_ids,
        reference_offsets,
    )


def _scale_speed_addition(
    base_velocity: np.ndarray,
    phase_bump: np.ndarray,
    add_axis: int,
    target_speed: float,
    sign: float,
) -> np.ndarray:
    base_velocity = np.asarray(base_velocity, dtype=np.float64)
    direction = np.zeros(3, dtype=np.float64)
    direction[add_axis] = float(sign)

    def value(amplitude: float) -> float:
        candidate = base_velocity + amplitude * phase_bump[:, None] * direction[None, :]
        return float(np.max(np.linalg.norm(candidate, axis=1)))

    low, high = 0.0, 0.10
    while value(high) < target_speed and high < 1.50:
        high *= 1.7
    if value(high) < target_speed:
        raise RuntimeError("could not reach targeted speed severity")
    for _ in range(70):
        midpoint = 0.5 * (low + high)
        if value(midpoint) < target_speed:
            low = midpoint
        else:
            high = midpoint
    return base_velocity + high * phase_bump[:, None] * direction[None, :]


def _tilt_orientation(
    yaw: np.ndarray,
    bump: np.ndarray,
    azimuth: float,
    target_tilt: float,
) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)

    def maximum(amplitude: float) -> float:
        roll = amplitude * np.cos(azimuth) * bump
        pitch = amplitude * np.sin(azimuth) * bump
        cosine = np.clip(np.cos(roll) * np.cos(pitch), -1.0, 1.0)
        return float(np.max(np.arccos(cosine)))

    low, high = 0.0, max(0.05, target_tilt * 1.2)
    while maximum(high) < target_tilt and high < 1.0:
        high *= 1.5
    if maximum(high) < target_tilt:
        raise RuntimeError("could not reach targeted tilt severity")
    for _ in range(70):
        midpoint = 0.5 * (low + high)
        if maximum(midpoint) < target_tilt:
            low = midpoint
        else:
            high = midpoint
    orientation = np.zeros((HORIZON, 3), dtype=np.float64)
    orientation[:, 0] = high * np.cos(azimuth) * bump
    orientation[:, 1] = high * np.sin(azimuth) * bump
    orientation[:, 2] = yaw
    return orientation


def _make_evaluation_pair(
    target: str,
    index: int,
    rule: dict[str, object],
    rng: np.random.Generator,
) -> tuple[Episode, Episode, dict[str, object]]:
    height_limit, speed_limit, tilt_limit = _rule_values(rule)
    direction_bin = index % 8
    height_bin = (index // 8) % 4
    target_phase = index % 3
    style = (index // 32) % 2
    near_boundary = index < 64
    normalized_margin = float(
        rng.uniform(0.02, 0.08) if near_boundary else rng.uniform(0.12, 0.22)
    )
    base = _motion_profiles(
        direction_bin=direction_bin,
        height_bin=height_bin,
        style=style,
        phase_variant=index + 17 * (1 + ("height_only", "speed_only", "tilt_only", "multi_clause").index(target)),
        speed_peak=0.48 * speed_limit,
        height_error_amplitude=0.34 * height_limit,
        tilt_amplitude=0.34 * tilt_limit,
        tilt_azimuth=2.0 * np.pi * ((index * 3) % 16) / 16.0,
        yaw_offset=float(rng.uniform(-np.pi, np.pi)),
        yaw_rate=float(rng.uniform(-0.70, 0.70)),
        rng=rng,
    )
    if _trajectory_label(base, rule) != SAFE_LABEL:
        raise RuntimeError("private safe counterfactual base is unsafe")

    target_pattern = {
        "height_only": (1, 0, 0),
        "speed_only": (0, 1, 0),
        "tilt_only": (0, 0, 1),
    }.get(target)
    if target == "multi_clause":
        target_pattern = (
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        )[index // 32]
    assert target_pattern is not None

    safe_velocity = np.asarray(base.observations[:, 6:9], dtype=np.float64).copy()
    safe_orientation = np.asarray(base.observations[:, 9:12], dtype=np.float64).copy()
    safe_error = np.asarray(base.observations[:, 5], dtype=np.float64).copy()
    velocity = safe_velocity.copy()
    orientation = safe_orientation.copy()
    unsafe_error = safe_error.copy()
    bump = _phase_bump(base.phase_ids, target_phase)
    held_velocity_axis = index % 3
    # Keep one rotating proxy component byte-identical, but inject the norm
    # violation through a horizontal component.  Avoiding vertical injection
    # prevents the intervention from moving the public height reference outside
    # its declared domain while still covering held vx/vy/vz proxies.
    added_velocity_axis = 1 if held_velocity_axis == 0 else 0
    added_velocity_sign = -1.0 if index % 2 else 1.0

    if target_pattern[0]:
        sign = -1.0 if index % 2 else 1.0
        unsafe_error = sign * height_limit * (1.0 + normalized_margin) * bump
    if target_pattern[1]:
        velocity = _scale_speed_addition(
            velocity,
            bump,
            added_velocity_axis,
            speed_limit * (1.0 + normalized_margin),
            added_velocity_sign,
        )
    if target_pattern[2]:
        azimuth = 2.0 * np.pi * (index % 8) / 8.0
        orientation = _tilt_orientation(
            np.asarray(base.observations[:, 11], dtype=np.float64),
            bump,
            azimuth,
            tilt_limit * (1.0 + normalized_margin),
        )

    # Rebuild both members once from the exact same float32 base context.  This
    # makes proxy collisions byte-exact without ever editing observations.
    safe = _replace_profiles(base, safe_velocity, safe_orientation, safe_error)
    violation = _replace_profiles(base, velocity, orientation, unsafe_error)
    actual_pattern = tuple(map(int, _clause_labels(violation, rule)))
    if actual_pattern != tuple(target_pattern):
        raise RuntimeError(
            f"{target}[{index}] expected clause pattern {target_pattern}, got {actual_pattern}"
        )

    if target == "height_only" and not np.array_equal(
        safe.observations[:, 2], violation.observations[:, 2]
    ):
        raise RuntimeError("height-only pair lost its exact world-z representation collision")
    if target == "speed_only" and not np.array_equal(
        safe.observations[:, 6 + held_velocity_axis],
        violation.observations[:, 6 + held_velocity_axis],
    ):
        raise RuntimeError("speed-only pair lost its exact component representation collision")
    if target == "tilt_only" and not np.array_equal(
        safe.observations[:, 11], violation.observations[:, 11]
    ):
        raise RuntimeError("tilt-only pair lost its exact yaw representation collision")

    return safe, violation, {
        "direction_bin": direction_bin,
        "height_bin": height_bin,
        "target_phase": target_phase,
        "near_boundary": int(near_boundary),
        "held_velocity_axis": held_velocity_axis if target == "speed_only" else -1,
        "target_pattern": target_pattern,
    }


def _generate_private_evaluation(
    rule: dict[str, object],
    private_rng: np.random.Generator,
    pairs_per_target: int = 128,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if pairs_per_target != 128:
        raise ValueError("CarryWaterActive v1 freezes 128 pairs per target")
    target_groups = ("height_only", "speed_only", "tilt_only", "multi_clause")
    episodes: list[Episode] = []
    labels: list[int] = []
    groups: list[str] = []
    pair_ids: list[str] = []
    pair_roles: list[str] = []
    pair_targets: list[str] = []
    direction_bins: list[int] = []
    height_bins: list[int] = []
    target_phases: list[int] = []
    near_boundary_values: list[int] = []
    held_velocity_axes: list[int] = []
    clause_labels: list[np.ndarray] = []
    state_masks: list[np.ndarray] = []
    severities: list[np.ndarray] = []
    first_violation: list[np.ndarray] = []

    for target in target_groups:
        for index in range(pairs_per_target):
            try:
                safe, violation, metadata = _make_evaluation_pair(
                    target, index, rule, private_rng
                )
            except Exception as error:
                raise RuntimeError(
                    f"private matched-pair generation failed for {target}[{index}]"
                ) from error
            pair_id = f"{target}_{index:04d}"
            for role, group, label, episode in (
                ("safe", "safe_counterfactual", SAFE_LABEL, safe),
                ("violation", target, VIOLATION_LABEL, violation),
            ):
                _validate_episode(episode)
                computed_label = _trajectory_label(episode, rule)
                if computed_label != label:
                    raise RuntimeError("matched pair label does not match analytic Oracle")
                masks = _state_clause_masks(episode, rule)
                first = np.full(3, -1, dtype=np.int16)
                for clause_index in range(3):
                    hits = np.flatnonzero(masks[:, clause_index])
                    if len(hits):
                        first[clause_index] = int(hits[0])
                episodes.append(episode)
                labels.append(label)
                groups.append(group)
                pair_ids.append(pair_id)
                pair_roles.append(role)
                pair_targets.append(target)
                direction_bins.append(int(metadata["direction_bin"]))
                height_bins.append(int(metadata["height_bin"]))
                target_phases.append(int(metadata["target_phase"]))
                near_boundary_values.append(int(metadata["near_boundary"]))
                held_velocity_axes.append(int(metadata["held_velocity_axis"]))
                clause_labels.append(_clause_labels(episode, rule))
                state_masks.append(masks)
                severities.append(_normalized_severities(episode, rule))
                first_violation.append(first)

    order = private_rng.permutation(len(episodes))
    ids = np.asarray([f"private_eval_{index:05d}" for index in range(len(episodes))], dtype="U32")
    base_arrays = _episode_arrays(episodes, ids)
    arrays: dict[str, np.ndarray] = {
        **base_arrays,
        "labels": np.asarray(labels, dtype=np.int8),
        "groups": np.asarray(groups, dtype="U24"),
        "pair_ids": np.asarray(pair_ids, dtype="U40"),
        "pair_roles": np.asarray(pair_roles, dtype="U12"),
        "pair_targets": np.asarray(pair_targets, dtype="U24"),
        "clause_ids": np.asarray(CLAUSE_IDS, dtype="U16"),
        "clause_labels": np.stack(clause_labels).astype(np.int8),
        "state_clause_masks": np.stack(state_masks).astype(np.int8),
        "normalized_severities": np.stack(severities).astype(np.float32),
        "first_violation_timesteps": np.stack(first_violation).astype(np.int16),
        "direction_bins": np.asarray(direction_bins, dtype=np.int8),
        "height_bins": np.asarray(height_bins, dtype=np.int8),
        "target_phases": np.asarray(target_phases, dtype=np.int8),
        "near_boundary": np.asarray(near_boundary_values, dtype=np.int8),
        "held_velocity_axes": np.asarray(held_velocity_axes, dtype=np.int8),
    }
    per_trajectory_keys = [
        key
        for key, value in arrays.items()
        if value.ndim >= 1 and len(value) == len(episodes) and key != "clause_ids"
    ]
    for key in per_trajectory_keys:
        arrays[key] = arrays[key][order]

    labels_array = arrays["labels"]
    groups_array = arrays["groups"].astype(str)
    clause_array = arrays["clause_labels"]
    statistics = {
        "trajectory_count": int(len(labels_array)),
        "safe_count": int(np.sum(labels_array == SAFE_LABEL)),
        "violation_count": int(np.sum(labels_array == VIOLATION_LABEL)),
        "group_counts": {
            group: int(np.sum(groups_array == group)) for group in sorted(set(groups_array))
        },
        "clause_violation_counts": {
            clause_id: int(np.sum(clause_array[:, index] == 1))
            for index, clause_id in enumerate(CLAUSE_IDS)
        },
    }
    if statistics["safe_count"] != 512 or statistics["violation_count"] != 512:
        raise AssertionError("private evaluation bank is not balanced")
    return arrays, statistics


def _make_public_candidate(index: int, rng: np.random.Generator) -> Episode:
    # Four broad acquisition regimes are generated before a public-seed-only
    # shuffle.  Their eventual row/pair positions therefore encode no Oracle
    # answer, while the pool still spans interior and each semantic direction.
    regime = index % 4
    height_error = (0.008, 0.074, 0.012, 0.012)[regime]
    speed_peak = (0.20, 0.24, 0.66, 0.24)[regime]
    tilt_amplitude = (0.04, 0.06, 0.06, 0.27)[regime]
    return _motion_profiles(
        direction_bin=int(rng.integers(0, 8)),
        height_bin=int(rng.integers(0, 4)),
        style=int(rng.integers(0, 2)),
        phase_variant=index + 1000,
        speed_peak=float(speed_peak * rng.uniform(0.94, 1.06)),
        height_error_amplitude=float(height_error * rng.uniform(0.94, 1.06)),
        tilt_amplitude=float(tilt_amplitude * rng.uniform(0.94, 1.06)),
        tilt_azimuth=float(rng.uniform(-np.pi, np.pi)),
        yaw_offset=float(rng.uniform(-np.pi, np.pi)),
        yaw_rate=float(rng.uniform(-0.95, 0.95)),
        rng=rng,
    )


def _generate_public_candidate_pool(
    rule: dict[str, object],
    public_rng: np.random.Generator,
    order_rng: np.random.Generator,
    count: int = 512,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if count != 512:
        raise ValueError("CarryWaterActive v1 freezes 512 public candidates")
    episodes: list[Episode] = []
    for index in range(count):
        episode = _make_public_candidate(index, public_rng)
        _validate_episode(episode)
        episodes.append(episode)

    # This order depends only on the public seed, never on hidden labels/rules.
    order = order_rng.permutation(count)
    episodes = [episodes[int(index)] for index in order]
    hidden_labels = np.asarray(
        [_trajectory_label(episode, rule) for episode in episodes], dtype=np.int8
    )
    prefix = hidden_labels[:12]
    if int(np.sum(prefix == SAFE_LABEL)) < 3 or int(np.sum(prefix == VIOLATION_LABEL)) < 3:
        raise RuntimeError("fixed public ordering lacks a mixed 12-query warmup prefix")
    ids = np.asarray([f"candidate_{index:05d}" for index in range(count)], dtype="U32")
    arrays = {
        **_episode_arrays(episodes, ids),
        "pair_ids": np.asarray(
            [f"acq_pair_{index // 2:04d}" for index in range(count)], dtype="U32"
        ),
        "pair_members": np.asarray([index % 2 for index in range(count)], dtype=np.int8),
    }
    forbidden_keys = {
        "labels",
        "label",
        "groups",
        "clause_labels",
        "state_clause_masks",
        "pair_roles",
        "pair_targets",
        "normalized_severities",
    }
    leaked = forbidden_keys & set(arrays)
    if leaked:
        raise RuntimeError(f"public candidate archive leaked answer fields: {sorted(leaked)}")
    statistics = {
        "candidate_count": count,
        "evaluation_only_safe_count": int(np.sum(hidden_labels == SAFE_LABEL)),
        "evaluation_only_violation_count": int(np.sum(hidden_labels == VIOLATION_LABEL)),
        "evaluation_only_first_12_safe": int(np.sum(hidden_labels[:12] == SAFE_LABEL)),
        "evaluation_only_first_12_violation": int(np.sum(hidden_labels[:12] == VIOLATION_LABEL)),
    }
    return arrays, statistics


def _task_spec() -> dict[str, object]:
    feature_schema = [
        {
            "name": name,
            "description": description,
            "unit": unit,
            "low": low,
            "high": high,
            "group": group,
            "causal_status": "not_disclosed",
        }
        for name, description, unit, low, high, group in FEATURE_SCHEMA
    ]
    return {
        "schema_version": TASK_SPEC_SCHEMA_VERSION,
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "task_instance_id": TASK_ID,
        "task_family": "carry_water_active",
        "task_description": TASK_DESCRIPTION,
        "horizon": HORIZON,
        "workspace": {
            "x": [float(WORKSPACE_LOW[0]), float(WORKSPACE_HIGH[0])],
            "y": [float(WORKSPACE_LOW[1]), float(WORKSPACE_HIGH[1])],
        },
        "max_step": float(DT * np.sqrt(3.0) * VELOCITY_COMPONENT_LIMIT),
        "start_goal_policy": "varied direction, target height, and phase duration; endpoints remain inside the public workspace",
        "feature_library_version": "carrywater_active_v1",
        "feature_schema": feature_schema,
        "raw_state_dimension": OBSERVATION_DIMENSION,
        "action_dimension": ACTION_DIMENSION,
        "action_horizon": "transition",
        "trajectory_adapter": "carrywater_active_v1",
        "dt": DT,
        "trajectory_label_convention": {"safe": SAFE_LABEL, "violation": VIOLATION_LABEL},
        "learner_information_contract": {
            "dynamics_file": "dynamics_spec.json",
            "known_safe_archive": "expert_trajectories.npz",
            "unlabeled_membership_candidates": "candidate_trajectories.npz",
            "reference_sidecar": "reference_xyz",
            "phase_sidecar": "phase_ids",
        },
    }


def _dynamics_spec() -> dict[str, object]:
    observation_schema = [
        {
            "index": index,
            "name": name,
            "description": description,
            "unit": unit,
            "low": low,
            "high": high,
            "group": group,
        }
        for index, (name, description, unit, low, high, group) in enumerate(OBSERVATION_SCHEMA)
    ]
    action_schema = [
        {
            "index": index,
            "name": name,
            "description": description,
            "unit": unit,
            "low": low,
            "high": high,
        }
        for index, (name, description, unit, low, high) in enumerate(ACTION_SCHEMA)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_instance_id": TASK_ID,
        "horizon": HORIZON,
        "dt": DT,
        "observation_dimension": OBSERVATION_DIMENSION,
        "action_dimension": ACTION_DIMENSION,
        "observation_schema": observation_schema,
        "action_schema": action_schema,
        "derived_feature_schema": [
            {
                "name": "speed",
                "description": "three-dimensional translational speed magnitude",
                "unit": "m/s",
                "formula": "sqrt(vx^2 + vy^2 + vz^2)",
                "inputs": ["vx", "vy", "vz"],
                "low": 0.0,
                "high": float(np.sqrt(3.0) * VELOCITY_COMPONENT_LIMIT),
                "causal_status": "not_disclosed",
            },
            {
                "name": "tilt_from_vertical",
                "description": "angle between the cup vertical axis and world vertical, invariant to yaw",
                "unit": "rad",
                "formula": "acos(cos(roll) * cos(pitch))",
                "inputs": ["roll", "pitch"],
                "low": 0.0,
                "high": math.pi,
                "causal_status": "not_disclosed",
            },
        ],
        "phase_context": {
            "sidecar": "phase_ids",
            "mapping": {str(index): name for index, name in enumerate(PHASE_NAMES)},
            "note": "phase is public task context and an evaluation stratum, not an extra observation dimension",
        },
        "reference_context": {
            "sidecar": "reference_xyz",
            "consistency": "observation[3:6] == reference_xyz - observation[0:3]",
        },
        "public_dynamics": {
            "timing": "observation[t] -- action[t] --> observation[t+1]",
            "velocity_update": "v_next = clip(velocity_decay * v + dt * acceleration, componentwise)",
            "position_update": "p_next = p + dt * v_next",
            "orientation_update": "angles_next = wrap(angles + dt * angular_rate)",
            "velocity_decay": VELOCITY_DECAY,
            "velocity_component_limit": VELOCITY_COMPONENT_LIMIT,
            "workspace_low": WORKSPACE_LOW.tolist(),
            "workspace_high": WORKSPACE_HIGH.tolist(),
            "reference_low": REFERENCE_LOW.tolist(),
            "reference_high": REFERENCE_HIGH.tolist(),
        },
        "membership_input_contract": "one dynamically valid full episode with public reference and phase context",
        "invalid_episode_policy": "reject before a membership query is counted",
    }


def _write_readme(path: Path) -> None:
    path.write_text(
        "# CarryWaterActive\n\n"
        "CarryWaterActive is a synthetic active constraint-identification benchmark. "
        "The learner receives only `public/carrywater_active/`, including 64 known-safe experts and a "
        "512-trajectory unlabeled candidate pool. `splits.json` uses "
        "`train/validation/test = 40/12/12`; `test` is reserved as the "
        "structure-audit expert split and is not a private safety test.\n\n"
        "Every observation is replayable under the public dynamics in "
        "`public/carrywater_active/dynamics_spec.json`. Actions have shape `[T-1, 6]`, so there is no "
        "fabricated terminal action. The sibling `private/carrywater_active/` directory backs the "
        "capability-limited membership Oracle and post-hoc evaluator and must not be "
        "mounted as learner input. The candidate archive deliberately contains no "
        "`labels`, clause annotations, thresholds, or private seed.\n"
        ,
        encoding="utf-8",
    )


def _assert_no_public_answer_leak(public_dir: Path) -> None:
    forbidden_json_keys = {
        "private_seed",
        "clauses",
        "expected_structure",
        "half_width",
        "threshold",
        "clause_labels",
        "normalized_severities",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            overlap = forbidden_json_keys & set(value)
            if overlap:
                raise RuntimeError(f"public JSON leaked private fields: {sorted(overlap)}")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in public_dir.glob("*.json"):
        visit(json.loads(path.read_text(encoding="utf-8")))
    with np.load(public_dir / "candidate_trajectories.npz", allow_pickle=False) as archive:
        forbidden_arrays = {
            "labels",
            "groups",
            "clause_ids",
            "clause_labels",
            "state_clause_masks",
            "pair_roles",
            "pair_targets",
            "normalized_severities",
        }
        overlap = forbidden_arrays & set(archive.files)
        if overlap:
            raise RuntimeError(f"public candidate archive leaked answer arrays: {sorted(overlap)}")


def generate_dataset(
    output_dir: Path,
    *,
    public_seed: int,
    private_seed: int,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset directory: {output_dir}")
    public_dir = output_dir / "public" / TASK_ID
    private_dir = output_dir / "private" / TASK_ID
    public_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)

    public_rng = np.random.default_rng(public_seed)
    public_candidate_order_rng = np.random.default_rng(public_seed + 2)
    private_rng = np.random.default_rng(private_seed)
    rule = _private_rule(private_rng)

    experts, expert_metadata = _generate_experts(rule, public_rng)
    expert_ids = np.asarray([f"expert_{index:04d}" for index in range(len(experts))], dtype="U24")
    expert_arrays = {
        **_episode_arrays(experts, expert_ids),
        "labels": np.full(len(experts), SAFE_LABEL, dtype=np.int8),
    }
    splits = _split_expert_ids(expert_ids, expert_metadata, public_rng)
    candidate_arrays, candidate_statistics = _generate_public_candidate_pool(
        rule, public_rng, public_candidate_order_rng
    )
    evaluation_arrays, evaluation_statistics = _generate_private_evaluation(rule, private_rng)

    _json_write(public_dir / "task_spec.json", _task_spec())
    _json_write(public_dir / "dynamics_spec.json", _dynamics_spec())
    _json_write(public_dir / "splits.json", splits)
    _deterministic_npz(public_dir / "expert_trajectories.npz", expert_arrays)
    _deterministic_npz(public_dir / "candidate_trajectories.npz", candidate_arrays)
    public_files = (
        "task_spec.json",
        "dynamics_spec.json",
        "splits.json",
        "expert_trajectories.npz",
        "candidate_trajectories.npz",
    )
    public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": SUITE_NAME,
        "dataset_version": SUITE_VERSION,
        "task_instance_id": TASK_ID,
        "public_seed": int(public_seed),
        "n_expert_trajectories": len(experts),
        "n_unlabeled_candidate_trajectories": int(len(candidate_arrays["observations"])),
        "horizon": HORIZON,
        "observation_shape": list(expert_arrays["observations"].shape),
        "action_shape": list(expert_arrays["actions"].shape),
        "all_expert_trajectories_are": "known_safe",
        "candidate_pool_is": "unlabeled",
        "candidate_archive_has_labels": False,
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "learner_visible_files": [*public_files, "manifest.json"],
        "expert_array_sha256": _canonical_array_sha256(expert_arrays),
        "candidate_array_sha256": _canonical_array_sha256(candidate_arrays),
        "integrity": {name: _file_sha256(public_dir / name) for name in public_files},
        "privacy_note": "No private seed, numerical safety boundary, Oracle label, or clause annotation for candidates is exported.",
    }
    _json_write(public_dir / "manifest.json", public_manifest)

    _json_write(private_dir / "oracle.json", rule)
    _json_write(private_dir / "expected_structure.json", rule["expected_structure"])
    _deterministic_npz(private_dir / "evaluation_trajectories.npz", evaluation_arrays)
    private_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_instance_id": TASK_ID,
        "warning": "Private Oracle/evaluation bundle; never mount as learner input.",
        **evaluation_statistics,
        "evaluation_array_sha256": _canonical_array_sha256(evaluation_arrays),
        "integrity": {
            "oracle.json": _file_sha256(private_dir / "oracle.json"),
            "expected_structure.json": _file_sha256(private_dir / "expected_structure.json"),
            "evaluation_trajectories.npz": _file_sha256(
                private_dir / "evaluation_trajectories.npz"
            ),
        },
    }
    _json_write(private_dir / "manifest.json", private_manifest)
    _write_readme(output_dir / "README.md")

    suite_manifest = {
        "schema_version": SCHEMA_VERSION,
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "task_instance_id": TASK_ID,
        "public_root": f"public/{TASK_ID}/",
        "private_root": f"private/{TASK_ID}/",
        "public_manifest_sha256": _file_sha256(public_dir / "manifest.json"),
        "information_contract": {
            "learner_root": f"public/{TASK_ID}/",
            "membership_oracle_backing_file": f"private/{TASK_ID}/oracle.json",
            "posthoc_evaluator_root": f"private/{TASK_ID}/",
        },
        "generator_runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    _json_write(output_dir / "suite_manifest.json", suite_manifest)
    _assert_no_public_answer_leak(public_dir)

    return {
        "output_dir": str(output_dir),
        "experts": len(experts),
        "expert_splits": public_manifest["split_sizes"],
        "public_candidates": candidate_statistics,
        "private_evaluation": evaluation_statistics,
        "public_candidate_archive_keys": sorted(candidate_arrays),
        "public_answer_leak_check": "passed",
        "dynamics_replay_check": "passed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "data" / SUITE_NAME),
        help="New, non-existing dataset directory.",
    )
    parser.add_argument("--public-seed", type=int, default=20260821)
    parser.add_argument(
        "--private-seed",
        type=int,
        required=True,
        help="Generation-only private seed; it is never written to an artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = generate_dataset(
        Path(args.output),
        public_seed=int(args.public_seed),
        private_seed=int(args.private_seed),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
