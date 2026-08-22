"""Generate the SemTraj2D causal constraint-identification benchmark.

The public bundle contains only a TaskSpec and known-safe demonstrations.  The
private bundle contains the analytic membership rule and a globally generated,
stratified trajectory test bank.  Evaluation trajectories are sampled from
cubic Bezier control points and never reuse expert interior waypoints.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
import secrets
from pathlib import Path
import sys
import zipfile

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from llm_modulo_cegis.data import SEMTRAJ_FEATURE_SPECS
from llm_modulo_cegis.oracle import RuleEvaluationOracle
from llm_modulo_cegis.types import SAFE_LABEL, Trajectory, VIOLATION_LABEL


SUITE_NAME = "SemTraj2D"
SUITE_VERSION = "1.1.0"
SCHEMA_VERSION = 1
HORIZON = 100
WORKSPACE_X = (0.0, 10.0)
WORKSPACE_Y = (-4.0, 4.0)
MAX_STEP = 0.35


TASK_DESCRIPTIONS = {
    "disk_clean": (
        "A planar point robot moves from a start on the left to a goal on the right. "
        "A persistent localized keep-out region must never be entered. Route choice outside "
        "that region is unconstrained. Infer the relevant observed variables and the unknown "
        "numerical boundary."
    ),
    "disk_upper_proxy": (
        "A planar point robot moves from a start on the left to a goal on the right. "
        "A persistent localized keep-out region must never be entered. Route choice outside "
        "that region is unconstrained. Infer the relevant observed variables and the unknown "
        "numerical boundary."
    ),
    "diagonal_halfspace": (
        "A planar point robot must remain on the permitted side of an unknown diagonal "
        "workspace boundary throughout the motion. Position coordinates are observable; "
        "speed and timing are incidental. Infer the coupled positional rule and its boundary."
    ),
    "lane_band": (
        "A planar point robot must remain inside an unknown horizontal operating lane for "
        "the complete trajectory. Motion timing and speed variations are not task rules. "
        "Infer the simplest feature-specific constraint and its numerical band."
    ),
    "speed_limit": (
        "A planar point robot may take different geometric routes, but it must respect an "
        "unknown maximum translational speed at every step. Absolute position and normalized "
        "time are not restricted. Infer the dynamic constraint and its numerical limit."
    ),
    "disk_and_speed": (
        "A planar point robot must both avoid a persistent localized keep-out region and "
        "respect an unknown maximum translational speed. A trajectory is invalid when either "
        "requirement is violated. Infer both heterogeneous clauses and their boundaries."
    ),
    "lane_and_speed": (
        "A planar point robot must remain inside an unknown horizontal operating lane and "
        "also respect an unknown maximum translational speed throughout the motion. A "
        "trajectory is invalid when either requirement is violated. Infer the feature-specific "
        "equality and inequality clauses and both numerical boundaries."
    ),
    "eventually_visit_open_set": (
        "A planar point robot must visit a required checkpoint at least once before reaching "
        "the goal. Merely avoiding forbidden states is insufficient. If the available "
        "hypothesis language cannot express this eventual-visit requirement, report that the "
        "result is inconclusive rather than inventing a proxy constraint."
    ),
}


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
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _actions(observations: np.ndarray) -> np.ndarray:
    result = np.zeros_like(observations, dtype=np.float32)
    result[..., :-1, :] = np.diff(observations, axis=-2)
    return result


def _bezier(
    start: np.ndarray,
    control_1: np.ndarray,
    control_2: np.ndarray,
    goal: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
    time_warp_sigma: float = 0.0,
    time_warp_focus: float | None = None,
    focused_increment: float | None = None,
) -> np.ndarray:
    if time_warp_focus is not None:
        if not 0.0 <= time_warp_focus <= 1.0:
            raise ValueError("time_warp_focus must be in [0,1]")
        increment = float(
            focused_increment if focused_increment is not None else rng.uniform(0.018, 0.030)
        )
        if not 0.0 < increment < 1.0:
            raise ValueError("focused_increment must be in (0,1)")
        increments = np.full(horizon - 1, (1.0 - increment) / (horizon - 2), dtype=np.float64)
        focus_index = min(horizon - 2, int(round(time_warp_focus * (horizon - 2))))
        increments[focus_index] = increment
        time = np.concatenate(([0.0], np.cumsum(increments)))
        time[-1] = 1.0
    elif time_warp_sigma <= 0.0:
        time = np.linspace(0.0, 1.0, horizon, dtype=np.float64)
    else:
        increments = np.exp(rng.normal(0.0, time_warp_sigma, size=horizon - 1))
        increments /= np.sum(increments)
        time = np.concatenate(([0.0], np.cumsum(increments)))
        time[-1] = 1.0
    one_minus = 1.0 - time
    curve = (
        (one_minus**3)[:, None] * start
        + (3.0 * one_minus**2 * time)[:, None] * control_1
        + (3.0 * one_minus * time**2)[:, None] * control_2
        + (time**3)[:, None] * goal
    )
    curve[0] = start
    curve[-1] = goal
    return curve.astype(np.float32)


def _valid_trajectory(states: np.ndarray) -> bool:
    if states.shape != (HORIZON, 2) or not np.all(np.isfinite(states)):
        return False
    if np.any(states[:, 0] < WORKSPACE_X[0]) or np.any(states[:, 0] > WORKSPACE_X[1]):
        return False
    if np.any(states[:, 1] < WORKSPACE_Y[0]) or np.any(states[:, 1] > WORKSPACE_Y[1]):
        return False
    return float(np.max(np.linalg.norm(np.diff(states, axis=0), axis=1))) <= MAX_STEP


def _endpoint_pair(task_key: str, rng: np.random.Generator, *, ood: bool = False) -> tuple[np.ndarray, np.ndarray]:
    start_x = rng.uniform(0.15, 0.95) if ood else rng.uniform(0.35, 0.75)
    goal_x = rng.uniform(9.05, 9.85) if ood else rng.uniform(9.25, 9.65)
    if task_key == "diagonal_halfspace":
        start_y, goal_y = rng.uniform(-1.8, -0.7, size=2)
    elif task_key in {"lane_band", "lane_and_speed"}:
        start_y, goal_y = rng.uniform(-0.45, 0.45, size=2)
    else:
        limit = 1.1 if ood else 0.55
        start_y, goal_y = rng.uniform(-limit, limit, size=2)
    return np.asarray((start_x, start_y)), np.asarray((goal_x, goal_y))


def _speed_controls(
    start: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Create paths whose largest velocity component is not always horizontal."""

    if variant == "vertical":
        sign = float(rng.choice((-1.0, 1.0)))
        control_1 = np.asarray(
            (start[0] + rng.uniform(0.08, 0.40), sign * rng.uniform(3.0, 3.75))
        )
        control_2 = np.asarray(
            (goal[0] - rng.uniform(0.08, 0.40), -sign * rng.uniform(3.0, 3.75))
        )
        return control_1, control_2
    if variant == "horizontal":
        control_1 = np.asarray((rng.uniform(2.7, 3.7), rng.uniform(-0.35, 0.35)))
        control_2 = np.asarray((rng.uniform(6.3, 7.3), rng.uniform(-0.35, 0.35)))
        return control_1, control_2
    raise ValueError(f"unknown speed-control variant: {variant}")


def _derive_seed(root_seed: int, label: str) -> int:
    if root_seed < 0:
        raise ValueError("root seed must be non-negative")
    width = max(16, (root_seed.bit_length() + 7) // 8)
    material = root_seed.to_bytes(width, "big") + b"\0" + label.encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big")


def _derive_private_seed(private_seed: int, label: str) -> int:
    return _derive_seed(private_seed, f"private:{label}")


def _private_rule(task_key: str, instance_seed: int) -> dict[str, object]:
    rng = np.random.default_rng(instance_seed)
    center = [float(5.0 + rng.uniform(-0.18, 0.18)), float(rng.uniform(-0.18, 0.18))]
    radius = float(rng.uniform(1.25, 1.45))
    speed_threshold = float(rng.uniform(0.165, 0.185))
    common: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "composition": "any_violation",
        "label_convention": {"safe": SAFE_LABEL, "violation": VIOLATION_LABEL},
        "warning": "Private analytic membership rule. Never expose to the semantic reasoner or learner.",
    }
    if task_key in {"disk_clean", "disk_upper_proxy"}:
        clauses = [{"clause_id": "spatial", "kind": "circle_exclusion", "center": center, "radius": radius}]
        expected = _expected_atomic(("x_position", "y_position"), "joint", "forbidden_region", "max", "mlp")
    elif task_key == "diagonal_halfspace":
        normal = [float(rng.uniform(-0.38, -0.26)), 1.0]
        offset = float(rng.uniform(0.10, 0.35))
        clauses = [{"clause_id": "diagonal", "kind": "linear_halfspace", "normal": normal, "offset": offset}]
        expected = _expected_atomic(("x_position", "y_position"), "joint", "forbidden_region", "max", "linear")
    elif task_key == "lane_band":
        lane_center = float(rng.uniform(-0.12, 0.12))
        half_width = float(rng.uniform(0.72, 0.88))
        clauses = [{"clause_id": "lane", "kind": "equality_band", "variable": "y_position", "center": lane_center, "half_width": half_width}]
        expected = _expected_atomic(("y_position",), "joint", "equality_band", "max", "linear")
    elif task_key == "speed_limit":
        clauses = [{"clause_id": "speed", "kind": "speed_upper_bound", "threshold": speed_threshold}]
        expected = _expected_atomic(("speed",), "joint", "upper_bound", "max", "linear")
    elif task_key == "disk_and_speed":
        clauses = [
            {"clause_id": "spatial", "kind": "circle_exclusion", "center": center, "radius": radius},
            {"clause_id": "speed", "kind": "speed_upper_bound", "threshold": speed_threshold},
        ]
        expected = {
            "representable": True,
            "composition": "any_violation",
            "clauses": [
                _expected_clause(("x_position", "y_position"), "joint", "forbidden_region", "max", "mlp"),
                _expected_clause(("speed",), "joint", "upper_bound", "max", "linear"),
            ],
        }
    elif task_key == "lane_and_speed":
        lane_center = float(rng.uniform(-0.12, 0.12))
        half_width = float(rng.uniform(0.72, 0.88))
        clauses = [
            {
                "clause_id": "lane",
                "kind": "equality_band",
                "variable": "y_position",
                "center": lane_center,
                "half_width": half_width,
            },
            {"clause_id": "speed", "kind": "speed_upper_bound", "threshold": speed_threshold},
        ]
        expected = {
            "representable": True,
            "composition": "any_violation",
            "clauses": [
                _expected_clause(("y_position",), "joint", "equality_band", "max", "linear"),
                _expected_clause(("speed",), "joint", "upper_bound", "max", "linear"),
            ],
        }
    elif task_key == "eventually_visit_open_set":
        clauses = [{"clause_id": "checkpoint", "kind": "checkpoint_visit", "center": center, "radius": float(rng.uniform(0.45, 0.60))}]
        expected = {
            "representable": False,
            "reason": "eventual existential visit is outside max/mean/last violation aggregation",
            "composition": "eventually",
            "clauses": [],
        }
    else:
        raise KeyError(task_key)
    return {**common, "clauses": clauses, "expected_structure": expected}


def _expected_clause(
    variables: tuple[str, ...],
    coupling: str,
    relation: str,
    temporal_operator: str,
    model_family: str,
) -> dict[str, object]:
    return {
        "variables": list(variables),
        "coupling": coupling,
        "relation": relation,
        "temporal_operator": temporal_operator,
        "model_family": model_family,
    }


def _expected_atomic(
    variables: tuple[str, ...],
    coupling: str,
    relation: str,
    temporal_operator: str,
    model_family: str,
) -> dict[str, object]:
    return {
        "representable": True,
        "composition": "any_violation",
        "clauses": [_expected_clause(variables, coupling, relation, temporal_operator, model_family)],
    }


def _expert_candidate(task_key: str, rule: dict[str, object], index: int, rng: np.random.Generator) -> np.ndarray:
    start, goal = _endpoint_pair(task_key, rng)
    if task_key in {"disk_clean", "disk_upper_proxy", "disk_and_speed"}:
        side = 1.0 if task_key == "disk_upper_proxy" else (-1.0 if index % 2 == 0 else 1.0)
        if task_key != "disk_upper_proxy" and rng.random() < 0.25:
            side *= -1.0
        amplitude = rng.uniform(2.05, 2.65)
        control_1 = np.asarray((rng.uniform(2.8, 3.8), side * amplitude))
        control_2 = np.asarray((rng.uniform(6.2, 7.2), side * amplitude))
        return _bezier(start, control_1, control_2, goal, HORIZON, rng)
    if task_key == "diagonal_halfspace":
        control_1 = np.asarray((rng.uniform(2.5, 3.8), rng.uniform(-2.3, -1.0)))
        control_2 = np.asarray((rng.uniform(6.2, 7.5), rng.uniform(-2.6, -1.1)))
        return _bezier(start, control_1, control_2, goal, HORIZON, rng)
    if task_key in {"lane_band", "lane_and_speed"}:
        lane = rule["clauses"][0]
        center = float(lane["center"])
        width = 0.72 * float(lane["half_width"])
        start[1] = center + rng.uniform(-0.35 * width, 0.35 * width)
        goal[1] = center + rng.uniform(-0.35 * width, 0.35 * width)
        control_1 = np.asarray((rng.uniform(2.4, 3.8), center + rng.uniform(-width, width)))
        control_2 = np.asarray((rng.uniform(6.2, 7.6), center + rng.uniform(-width, width)))
        return _bezier(start, control_1, control_2, goal, HORIZON, rng)
    if task_key == "speed_limit":
        control_1, control_2 = _speed_controls(
            start,
            goal,
            rng,
            "vertical" if index % 2 == 0 else "horizontal",
        )
        return _bezier(start, control_1, control_2, goal, HORIZON, rng)
    if task_key == "eventually_visit_open_set":
        checkpoint = rule["clauses"][0]
        center = np.asarray(checkpoint["center"], dtype=float)
        control_1 = np.asarray((rng.uniform(2.5, 3.7), center[1] + rng.uniform(-0.25, 0.25)))
        control_2 = np.asarray((rng.uniform(6.3, 7.5), center[1] + rng.uniform(-0.25, 0.25)))
        return _bezier(start, control_1, control_2, goal, HORIZON, rng)
    raise KeyError(task_key)


def _generate_experts(
    task_key: str,
    rule: dict[str, object],
    seed: int,
    count: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    oracle = RuleEvaluationOracle(rule)
    experts: list[np.ndarray] = []
    attempts = 0
    while len(experts) < count and attempts < 10000:
        attempts += 1
        states = _expert_candidate(task_key, rule, len(experts), rng)
        if _valid_trajectory(states) and oracle.label(Trajectory(states)) == SAFE_LABEL:
            experts.append(states)
    if len(experts) != count:
        raise RuntimeError(f"{task_key}: generated only {len(experts)}/{count} safe experts")
    return np.stack(experts).astype(np.float32)


def _generate_paired_disk_experts(
    rule: dict[str, object],
    seed: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate matched clean/proxy demonstrations differing only in route side."""

    rng = np.random.default_rng(seed)
    oracle = RuleEvaluationOracle(rule)
    clean: list[np.ndarray] = []
    proxy: list[np.ndarray] = []
    attempts = 0
    while len(clean) < count and attempts < 20000:
        attempts += 1
        index = len(clean)
        start, goal = _endpoint_pair("disk_clean", rng)
        c1x, c2x = rng.uniform(2.8, 3.8), rng.uniform(6.2, 7.2)
        amplitude_1, amplitude_2 = rng.uniform(2.10, 2.70, size=2)
        clean_side = -1.0 if index % 2 == 0 else 1.0
        clean_states = _bezier(
            start,
            np.asarray((c1x, clean_side * amplitude_1)),
            np.asarray((c2x, clean_side * amplitude_2)),
            goal,
            HORIZON,
            rng,
        )
        proxy_states = _bezier(
            start,
            np.asarray((c1x, amplitude_1)),
            np.asarray((c2x, amplitude_2)),
            goal,
            HORIZON,
            rng,
        )
        if not _valid_trajectory(clean_states) or not _valid_trajectory(proxy_states):
            continue
        if oracle.label(Trajectory(clean_states)) != SAFE_LABEL:
            continue
        if oracle.label(Trajectory(proxy_states)) != SAFE_LABEL:
            continue
        clean.append(clean_states)
        proxy.append(proxy_states)
    if len(clean) != count:
        raise RuntimeError(f"paired disk experts: generated only {len(clean)}/{count}")
    return np.stack(clean).astype(np.float32), np.stack(proxy).astype(np.float32)


def _sample_probe(
    task_key: str,
    rule: dict[str, object],
    group: str,
    desired_label: int,
    rng: np.random.Generator,
    variant: str | None = None,
) -> np.ndarray:
    ood = group == "ood"
    start, goal = _endpoint_pair(task_key, rng, ood=ood)
    c1x, c2x = rng.uniform(2.0, 4.2), rng.uniform(5.8, 8.0)
    sigma = 0.0
    time_warp_focus = None
    focused_increment = None
    if task_key in {"disk_clean", "disk_upper_proxy"}:
        if desired_label == SAFE_LABEL:
            side = -1.0 if group == "ood" else rng.choice((-1.0, 1.0))
            amplitude = rng.uniform(2.0, 3.0)
        else:
            side = rng.choice((-1.0, 1.0))
            amplitude = rng.uniform(0.0, 1.25)
        if group == "boundary":
            amplitude = rng.uniform(1.55, 2.15)
        c1y = c2y = side * amplitude
    elif task_key == "diagonal_halfspace":
        if desired_label == SAFE_LABEL:
            c1y, c2y = rng.uniform(-2.8, -0.8, size=2)
        else:
            c1y, c2y = rng.uniform(2.0, 3.7, size=2)
        if group == "boundary":
            c1y, c2y = rng.uniform(0.6, 3.0, size=2)
    elif task_key == "lane_band":
        lane = rule["clauses"][0]
        center = float(lane["center"])
        width = float(lane["half_width"])
        start[1] = center + rng.uniform(-0.45 * width, 0.45 * width)
        goal[1] = center + rng.uniform(-0.45 * width, 0.45 * width)
        if desired_label == SAFE_LABEL:
            c1y, c2y = center + rng.uniform(-0.95 * width, 0.95 * width, size=2)
        else:
            sign = rng.choice((-1.0, 1.0))
            c1y, c2y = center + sign * rng.uniform(1.3 * width, 2.8 * width, size=2)
        if group == "boundary":
            sign = rng.choice((-1.0, 1.0))
            c1y = c2y = center + sign * rng.uniform(1.05 * width, 1.65 * width)
    elif task_key == "speed_limit":
        speed_variant = variant or str(rng.choice(("vertical", "horizontal")))
        control_1, control_2 = _speed_controls(
            start,
            goal,
            rng,
            speed_variant,
        )
        c1x, c1y = map(float, control_1)
        c2x, c2y = map(float, control_2)
        if desired_label == VIOLATION_LABEL:
            time_warp_focus = 0.0 if speed_variant == "vertical" else 0.5
            focused_increment = (
                rng.uniform(0.014, 0.026) if group == "boundary" else rng.uniform(0.020, 0.034)
            )
        elif group == "boundary":
            time_warp_focus = 0.0 if speed_variant == "vertical" else 0.5
            focused_increment = rng.uniform(0.012, 0.021)
    elif task_key == "disk_and_speed":
        target = group
        spatial_violation = desired_label == VIOLATION_LABEL and target in {"spatial_only", "multi_clause"}
        speed_violation = desired_label == VIOLATION_LABEL and target in {"speed_only", "multi_clause"}
        if target not in {"spatial_only", "speed_only", "multi_clause"} and desired_label == VIOLATION_LABEL:
            spatial_violation = bool(rng.integers(0, 2))
            speed_violation = not spatial_violation or bool(rng.integers(0, 2))
        amplitude = rng.uniform(0.0, 1.2) if spatial_violation else rng.uniform(2.1, 2.9)
        side = rng.choice((-1.0, 1.0))
        c1y = c2y = side * amplitude
        sigma = rng.uniform(0.85, 1.35) if speed_violation else rng.uniform(0.0, 0.18)
        if group == "boundary":
            c1y = c2y = side * rng.uniform(1.55, 2.2)
            sigma = rng.uniform(0.2, 0.75)
    elif task_key == "lane_and_speed":
        lane = rule["clauses"][0]
        center = float(lane["center"])
        width = float(lane["half_width"])
        target = group
        lane_violation = desired_label == VIOLATION_LABEL and target in {
            "lane_only",
            "multi_clause",
        }
        speed_violation = desired_label == VIOLATION_LABEL and target in {
            "speed_only",
            "multi_clause",
        }
        if target not in {"lane_only", "speed_only", "multi_clause"} and desired_label == VIOLATION_LABEL:
            lane_violation = bool(rng.integers(0, 2))
            speed_violation = not lane_violation or bool(rng.integers(0, 2))
        start[1] = center + rng.uniform(-0.35 * width, 0.35 * width)
        goal[1] = center + rng.uniform(-0.35 * width, 0.35 * width)
        if lane_violation:
            sign = float(rng.choice((-1.0, 1.0)))
            c1y = c2y = center + sign * rng.uniform(1.35 * width, 2.6 * width)
        else:
            c1y, c2y = center + rng.uniform(-0.70 * width, 0.70 * width, size=2)
        if speed_violation:
            time_warp_focus = 0.5
            focused_increment = rng.uniform(0.020, 0.034)
        if group == "boundary":
            if desired_label == SAFE_LABEL:
                sign = float(rng.choice((-1.0, 1.0)))
                c1y = c2y = center + sign * rng.uniform(1.24 * width, 1.33 * width)
                time_warp_focus = None
                focused_increment = None
            else:
                sign = float(rng.choice((-1.0, 1.0)))
                c1y = c2y = center + sign * rng.uniform(1.34 * width, 1.52 * width)
                time_warp_focus = None
                focused_increment = None
    elif task_key == "eventually_visit_open_set":
        checkpoint = rule["clauses"][0]
        center_y = float(checkpoint["center"][1])
        if desired_label == SAFE_LABEL:
            c1y, c2y = center_y + rng.uniform(-0.25, 0.25, size=2)
        else:
            side = rng.choice((-1.0, 1.0))
            c1y = c2y = center_y + side * rng.uniform(1.2, 3.0)
        if group == "boundary":
            side = rng.choice((-1.0, 1.0))
            c1y = c2y = center_y + side * rng.uniform(0.45, 1.1)
    else:
        raise KeyError(task_key)
    return _bezier(
        start,
        np.asarray((c1x, c1y)),
        np.asarray((c2x, c2y)),
        goal,
        HORIZON,
        rng,
        time_warp_sigma=sigma,
        time_warp_focus=time_warp_focus,
        focused_increment=focused_increment,
    )


def _target_clause_pattern(task_key: str, group: str, desired_label: int) -> tuple[int, ...] | None:
    if task_key not in {"disk_and_speed", "lane_and_speed"}:
        return None
    if desired_label == SAFE_LABEL:
        return (0, 0)
    if group == "spatial_only":
        return (1, 0)
    if group == "lane_only":
        return (1, 0)
    if group == "speed_only":
        return (0, 1)
    if group == "multi_clause":
        return (1, 1)
    return None


def _counterfactual_pair(
    task_key: str,
    rule: dict[str, object],
    rng: np.random.Generator,
    variant: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a safe/violation pair with identical endpoints and a controlled change."""

    oracle = RuleEvaluationOracle(rule)
    for attempt in range(20000):
        start, goal = _endpoint_pair(task_key, rng, ood=task_key == "disk_upper_proxy")
        c1x, c2x = rng.uniform(2.7, 3.7), rng.uniform(6.3, 7.3)
        if task_key in {"disk_clean", "disk_upper_proxy"}:
            safe_side = -1.0 if task_key == "disk_upper_proxy" else rng.choice((-1.0, 1.0))
            safe = _bezier(
                start,
                np.asarray((c1x, safe_side * rng.uniform(2.25, 2.75))),
                np.asarray((c2x, safe_side * rng.uniform(2.25, 2.75))),
                goal,
                HORIZON,
                rng,
            )
            violation = _bezier(
                start,
                np.asarray((c1x, rng.uniform(-0.25, 0.25))),
                np.asarray((c2x, rng.uniform(-0.25, 0.25))),
                goal,
                HORIZON,
                rng,
            )
        elif task_key == "diagonal_halfspace":
            safe = _bezier(
                start,
                np.asarray((c1x, rng.uniform(-2.4, -1.2))),
                np.asarray((c2x, rng.uniform(-2.6, -1.3))),
                goal,
                HORIZON,
                rng,
            )
            violation = _bezier(
                start,
                np.asarray((c1x, rng.uniform(2.8, 3.6))),
                np.asarray((c2x, rng.uniform(3.0, 3.8))),
                goal,
                HORIZON,
                rng,
            )
        elif task_key == "lane_band":
            lane = rule["clauses"][0]
            center = float(lane["center"])
            width = float(lane["half_width"])
            start[1] = center + rng.uniform(-0.3 * width, 0.3 * width)
            goal[1] = center + rng.uniform(-0.3 * width, 0.3 * width)
            safe = _bezier(
                start,
                np.asarray((c1x, center + rng.uniform(-0.4 * width, 0.4 * width))),
                np.asarray((c2x, center + rng.uniform(-0.4 * width, 0.4 * width))),
                goal,
                HORIZON,
                rng,
            )
            sign = rng.choice((-1.0, 1.0))
            violation = _bezier(
                start,
                np.asarray((c1x, center + sign * 2.2 * width)),
                np.asarray((c2x, center + sign * 2.2 * width)),
                goal,
                HORIZON,
                rng,
            )
        elif task_key == "speed_limit":
            speed_variant = variant or str(rng.choice(("vertical", "horizontal")))
            control_1, control_2 = _speed_controls(start, goal, rng, speed_variant)
            safe = _bezier(start, control_1, control_2, goal, HORIZON, rng)
            violation = _bezier(
                start,
                control_1,
                control_2,
                goal,
                HORIZON,
                rng,
                time_warp_focus=0.0 if speed_variant == "vertical" else 0.5,
                focused_increment=rng.uniform(0.020, 0.034),
            )
        elif task_key == "disk_and_speed":
            side = rng.choice((-1.0, 1.0))
            control_1 = np.asarray((c1x, side * rng.uniform(2.3, 2.7)))
            control_2 = np.asarray((c2x, side * rng.uniform(2.3, 2.7)))
            safe = _bezier(start, control_1, control_2, goal, HORIZON, rng)
            if attempt % 2 == 0:
                violation = _bezier(
                    start,
                    np.asarray((c1x, rng.uniform(-0.2, 0.2))),
                    np.asarray((c2x, rng.uniform(-0.2, 0.2))),
                    goal,
                    HORIZON,
                    rng,
                )
            else:
                violation = _bezier(
                    start,
                    control_1,
                    control_2,
                    goal,
                    HORIZON,
                    rng,
                    time_warp_sigma=rng.uniform(0.9, 1.3),
                )
        elif task_key == "lane_and_speed":
            lane = rule["clauses"][0]
            center = float(lane["center"])
            width = float(lane["half_width"])
            start[1] = center + rng.uniform(-0.25 * width, 0.25 * width)
            goal[1] = center + rng.uniform(-0.25 * width, 0.25 * width)
            control_1 = np.asarray((c1x, center + rng.uniform(-0.40 * width, 0.40 * width)))
            control_2 = np.asarray((c2x, center + rng.uniform(-0.40 * width, 0.40 * width)))
            safe = _bezier(start, control_1, control_2, goal, HORIZON, rng)
            if attempt % 2 == 0:
                sign = float(rng.choice((-1.0, 1.0)))
                violation = _bezier(
                    start,
                    np.asarray((c1x, center + sign * 2.0 * width)),
                    np.asarray((c2x, center + sign * 2.0 * width)),
                    goal,
                    HORIZON,
                    rng,
                )
            else:
                violation = _bezier(
                    start,
                    control_1,
                    control_2,
                    goal,
                    HORIZON,
                    rng,
                    time_warp_focus=0.5,
                    focused_increment=rng.uniform(0.020, 0.034),
                )
        elif task_key == "eventually_visit_open_set":
            checkpoint = rule["clauses"][0]
            center_y = float(checkpoint["center"][1])
            safe = _bezier(
                start,
                np.asarray((c1x, center_y + rng.uniform(-0.15, 0.15))),
                np.asarray((c2x, center_y + rng.uniform(-0.15, 0.15))),
                goal,
                HORIZON,
                rng,
            )
            side = rng.choice((-1.0, 1.0))
            violation = _bezier(
                start,
                np.asarray((c1x, center_y + side * rng.uniform(1.6, 2.4))),
                np.asarray((c2x, center_y + side * rng.uniform(1.6, 2.4))),
                goal,
                HORIZON,
                rng,
            )
        else:
            raise KeyError(task_key)
        component_condition = True
        if task_key == "speed_limit" and variant is not None:
            delta = np.diff(violation, axis=0)
            speed_threshold = float(rule["clauses"][0]["threshold"])
            if variant == "vertical":
                component_condition = float(np.max(np.abs(delta[:, 0]))) <= speed_threshold
            elif variant == "horizontal":
                component_condition = float(np.max(np.abs(delta[:, 1]))) <= speed_threshold
        if (
            _valid_trajectory(safe)
            and _valid_trajectory(violation)
            and oracle.label(Trajectory(safe)) == SAFE_LABEL
            and oracle.label(Trajectory(violation)) == VIOLATION_LABEL
            and component_condition
        ):
            return safe, violation
    raise RuntimeError(f"{task_key}: failed to generate matched counterfactual pair")


def _generate_evaluation_bank(
    task_key: str,
    rule: dict[str, object],
    seed: int,
    per_group_label_count: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    oracle = RuleEvaluationOracle(rule)
    groups = ["id", "boundary", "counterfactual", "ood"]
    if task_key == "disk_and_speed":
        groups.extend(["spatial_only", "speed_only", "multi_clause"])
    if task_key == "lane_and_speed":
        groups.extend(["lane_only", "speed_only", "multi_clause"])
    observations: list[np.ndarray] = []
    labels: list[int] = []
    group_values: list[str] = []
    pair_ids: list[str] = []
    pair_roles: list[str] = []
    clause_rows: list[list[int]] = []
    clause_ids = [str(item["clause_id"]) for item in rule["clauses"]]
    for group in groups:
        if group == "counterfactual":
            for pair_index in range(per_group_label_count):
                variant = None
                if task_key == "speed_limit":
                    variant = "vertical" if pair_index % 2 == 0 else "horizontal"
                safe_states, violation_states = _counterfactual_pair(
                    task_key,
                    rule,
                    rng,
                    variant=variant,
                )
                for role, states, label in (
                    ("safe", safe_states, SAFE_LABEL),
                    ("violation", violation_states, VIOLATION_LABEL),
                ):
                    severities = oracle.clause_severities(Trajectory(states))
                    observations.append(states)
                    labels.append(label)
                    group_values.append(group)
                    pair_ids.append(f"cf_{pair_index:04d}")
                    pair_roles.append(role)
                    clause_rows.append([int(severities[name] > 0.0) for name in clause_ids])
            continue
        for desired_label in (SAFE_LABEL, VIOLATION_LABEL):
            accepted = 0
            attempts = 0
            while accepted < per_group_label_count and attempts < 200000:
                attempts += 1
                variant = None
                if task_key == "speed_limit":
                    variant = "vertical" if accepted % 2 == 0 else "horizontal"
                states = _sample_probe(
                    task_key,
                    rule,
                    group,
                    desired_label,
                    rng,
                    variant=variant,
                )
                if not _valid_trajectory(states):
                    continue
                trajectory = Trajectory(states)
                severities = oracle.clause_severities(trajectory)
                label = VIOLATION_LABEL if any(value > 0.0 for value in severities.values()) else SAFE_LABEL
                if label != desired_label:
                    continue
                if task_key == "speed_limit" and desired_label == VIOLATION_LABEL:
                    delta = np.diff(states, axis=0)
                    speed_threshold = float(rule["clauses"][0]["threshold"])
                    if variant == "vertical" and float(np.max(np.abs(delta[:, 0]))) > speed_threshold:
                        continue
                    if variant == "horizontal" and float(np.max(np.abs(delta[:, 1]))) > speed_threshold:
                        continue
                if group == "boundary" and abs(max(severities.values())) > 0.045:
                    continue
                pattern = tuple(int(severities[name] > 0.0) for name in clause_ids)
                target = _target_clause_pattern(task_key, group, desired_label)
                if target is not None and pattern != target:
                    continue
                observations.append(states)
                labels.append(label)
                group_values.append(group)
                pair_ids.append("")
                pair_roles.append("")
                clause_rows.append(list(pattern))
                accepted += 1
            if accepted != per_group_label_count:
                raise RuntimeError(
                    f"{task_key}/{group}/label={desired_label}: accepted {accepted}/{per_group_label_count}"
                )
    order = rng.permutation(len(observations))
    return {
        "observations": np.stack(observations).astype(np.float32)[order],
        "labels": np.asarray(labels, dtype=np.int8)[order],
        "groups": np.asarray(group_values, dtype="U24")[order],
        "trajectory_ids": np.asarray([f"evaluation_{index:05d}" for index in range(len(observations))], dtype="U32")[order],
        "pair_ids": np.asarray(pair_ids, dtype="U24")[order],
        "pair_roles": np.asarray(pair_roles, dtype="U12")[order],
        "clause_ids": np.asarray(clause_ids, dtype="U24"),
        "clause_labels": np.asarray(clause_rows, dtype=np.int8)[order],
    }


def _task_spec(task_key: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "task_instance_id": task_key,
        "task_family": task_key,
        "task_description": TASK_DESCRIPTIONS[task_key],
        "horizon": HORIZON,
        "workspace": {"x": list(WORKSPACE_X), "y": list(WORKSPACE_Y)},
        "max_step": MAX_STEP,
        "start_goal_policy": "start in the left strip and goal in the right strip; exact endpoints vary by trajectory",
        "feature_library_version": "semtraj2d-12d-v1",
        "feature_schema": [
            {
                **asdict(spec),
                "causal_status": "not_disclosed",
            }
            for spec in SEMTRAJ_FEATURE_SPECS
        ],
        "trajectory_label_convention": {"safe": SAFE_LABEL, "violation": VIOLATION_LABEL},
        "learner_information_contract": [
            "task_spec.json",
            "expert_trajectories.npz",
            "splits.json",
            "whole-trajectory membership Oracle",
        ],
    }


def _split_ids(ids: np.ndarray, seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n_train = int(round(0.60 * len(ids)))
    n_validation = int(round(0.20 * len(ids)))
    return {
        "train": [str(ids[index]) for index in order[:n_train]],
        "validation": [str(ids[index]) for index in order[n_train : n_train + n_validation]],
        "test": [str(ids[index]) for index in order[n_train + n_validation :]],
    }


def _export_task(
    root: Path,
    task_key: str,
    rule: dict[str, object],
    experts: np.ndarray,
    evaluation: dict[str, np.ndarray],
    public_seed: int,
    split_seed: int,
) -> dict[str, object]:
    public_dir = root / "public" / task_key
    private_dir = root / "private" / task_key
    public_dir.mkdir(parents=True, exist_ok=False)
    private_dir.mkdir(parents=True, exist_ok=False)
    ids = np.asarray([f"expert_{index:04d}" for index in range(len(experts))], dtype="U24")
    expert_arrays = {
        "observations": experts.astype(np.float32),
        "actions": _actions(experts).astype(np.float32),
        "lengths": np.full(len(experts), HORIZON, dtype=np.int32),
        "labels": np.full(len(experts), SAFE_LABEL, dtype=np.int8),
        "trajectory_ids": ids,
    }
    _deterministic_npz(public_dir / "expert_trajectories.npz", expert_arrays)
    _json_write(public_dir / "splits.json", _split_ids(ids, split_seed))
    _json_write(public_dir / "task_spec.json", _task_spec(task_key))
    public_files = ["expert_trajectories.npz", "splits.json", "task_spec.json"]
    public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": f"{SUITE_NAME}-{task_key}",
        "dataset_version": SUITE_VERSION,
        "task_instance_id": task_key,
        "public_generation_seed": public_seed,
        "n_expert_trajectories": len(experts),
        "horizon": HORIZON,
        "observation_shape": list(experts.shape),
        "action_shape": list(experts.shape),
        "all_exported_trajectories_are": "expert_safe",
        "learner_visible_files": public_files,
        "public_array_sha256": _canonical_array_sha256(expert_arrays),
        "integrity": {name: _file_sha256(public_dir / name) for name in public_files},
        "privacy_note": "No private path, clause, geometry, or numerical constraint parameter is listed here.",
    }
    _json_write(public_dir / "manifest.json", public_manifest)

    membership_rule = {key: value for key, value in rule.items() if key != "expected_structure"}
    _json_write(private_dir / "oracle.json", membership_rule)
    _json_write(private_dir / "expected_structure.json", rule["expected_structure"])
    _deterministic_npz(private_dir / "evaluation_trajectories.npz", evaluation)
    labels = np.asarray(evaluation["labels"], dtype=int)
    groups = np.asarray(evaluation["groups"]).astype(str)
    clause_labels = np.asarray(evaluation["clause_labels"], dtype=int)
    private_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_instance_id": task_key,
        "warning": "Private evaluation bundle; mount only after the learned model and champion are frozen.",
        "n_evaluation_trajectories": int(len(labels)),
        "class_counts": {"safe": int(np.sum(labels == 0)), "violation": int(np.sum(labels == 1))},
        "group_class_counts": {
            group: {
                "safe": int(np.sum((groups == group) & (labels == 0))),
                "violation": int(np.sum((groups == group) & (labels == 1))),
            }
            for group in sorted(set(groups))
        },
        "clause_violation_counts": {
            str(clause_id): int(np.sum(clause_labels[:, index] == 1))
            for index, clause_id in enumerate(evaluation["clause_ids"].astype(str))
        },
        "evaluation_array_sha256": _canonical_array_sha256(evaluation),
        "integrity": {
            "oracle.json": _file_sha256(private_dir / "oracle.json"),
            "expected_structure.json": _file_sha256(private_dir / "expected_structure.json"),
            "evaluation_trajectories.npz": _file_sha256(private_dir / "evaluation_trajectories.npz"),
        },
    }
    _json_write(private_dir / "manifest.json", private_manifest)
    return {
        "task_instance_id": task_key,
        "public_dir": f"public/{task_key}",
        "public_manifest_sha256": _file_sha256(public_dir / "manifest.json"),
        "expert_array_sha256": public_manifest["public_array_sha256"],
        "evaluation_array_sha256": private_manifest["evaluation_array_sha256"],
        "n_experts": len(experts),
        "n_evaluation": len(labels),
        "evaluation_class_counts": dict(Counter(map(int, labels))),
    }


def generate_suite(
    output_dir: Path,
    *,
    public_seed: int,
    private_seed: int,
    expert_count: int,
    per_group_label_count: int,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing suite directory: {output_dir}")
    if expert_count < 15:
        raise ValueError("expert_count must be at least 15")
    if per_group_label_count < 8:
        raise ValueError("per_group_label_count must be at least 8")
    output_dir.mkdir(parents=True)
    task_rows: list[dict[str, object]] = []
    shared_disk_rule = _private_rule(
        "disk_clean",
        _derive_private_seed(private_seed, "shared_disk_rule"),
    )
    shared_disk_evaluation = _generate_evaluation_bank(
        "disk_clean",
        shared_disk_rule,
        _derive_private_seed(private_seed, "shared_disk_evaluation"),
        per_group_label_count,
    )
    paired_clean_experts, paired_proxy_experts = _generate_paired_disk_experts(
        shared_disk_rule,
        _derive_seed(public_seed, "public:paired_disk:experts"),
        expert_count,
    )
    for index, task_key in enumerate(TASK_DESCRIPTIONS):
        if task_key in {"disk_clean", "disk_upper_proxy"}:
            rule = json.loads(json.dumps(shared_disk_rule))
            evaluation = {name: value.copy() for name, value in shared_disk_evaluation.items()}
        else:
            rule = _private_rule(
                task_key,
                _derive_private_seed(private_seed, f"{task_key}:rule"),
            )
            evaluation = _generate_evaluation_bank(
                task_key,
                rule,
                _derive_private_seed(private_seed, f"{task_key}:evaluation"),
                per_group_label_count,
            )
        if task_key == "disk_clean":
            experts = paired_clean_experts.copy()
        elif task_key == "disk_upper_proxy":
            experts = paired_proxy_experts.copy()
        else:
            experts = _generate_experts(
                task_key,
                rule,
                _derive_seed(public_seed, f"public:{task_key}:experts"),
                expert_count,
            )
        split_label = (
            "public:paired_disk:splits"
            if task_key in {"disk_clean", "disk_upper_proxy"}
            else f"public:{task_key}:splits"
        )
        task_rows.append(
            _export_task(
                output_dir,
                task_key,
                rule,
                experts,
                evaluation,
                public_seed,
                _derive_seed(public_seed, split_label),
            )
        )
    clean = next(row for row in task_rows if row["task_instance_id"] == "disk_clean")
    confounded = next(row for row in task_rows if row["task_instance_id"] == "disk_upper_proxy")
    if clean["evaluation_array_sha256"] != confounded["evaluation_array_sha256"]:
        raise RuntimeError("paired disk tasks must have byte-identical logical evaluation arrays")
    if TASK_DESCRIPTIONS["disk_clean"] != TASK_DESCRIPTIONS["disk_upper_proxy"]:
        raise RuntimeError("paired disk tasks must expose byte-identical task descriptions")
    suite_manifest = {
        "schema_version": SCHEMA_VERSION,
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "public_generation_seed": public_seed,
        "expert_count_per_task": expert_count,
        "per_group_label_count": per_group_label_count,
        "task_count": len(task_rows),
        "tasks": task_rows,
        "paired_causal_test": {
            "clean": "disk_clean",
            "confounded": "disk_upper_proxy",
            "same_private_rule_and_evaluation": True,
            "only_public_expert_route_regime_changes": True,
        },
        "information_contract": {
            "learner_root": "public/",
            "posthoc_evaluator_root": "private/",
            "evaluation_trajectories_use_expert_interior_waypoints": False,
        },
    }
    _json_write(output_dir / "suite_manifest.json", suite_manifest)
    _json_write(
        output_dir / "public" / "suite_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "suite_name": SUITE_NAME,
            "suite_version": SUITE_VERSION,
            "public_generation_seed": public_seed,
            "task_count": len(task_rows),
            "tasks": [
                {
                    "task_instance_id": row["task_instance_id"],
                    "public_dir": str(row["public_dir"]).split("/", 1)[-1],
                    "public_manifest_sha256": row["public_manifest_sha256"],
                    "expert_array_sha256": row["expert_array_sha256"],
                    "n_experts": row["n_experts"],
                }
                for row in task_rows
            ],
        },
    )
    _json_write(
        output_dir / "private" / "suite_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "suite_name": SUITE_NAME,
            "suite_version": SUITE_VERSION,
            "task_count": len(task_rows),
            "tasks": [
                {
                    "task_instance_id": row["task_instance_id"],
                    "evaluation_array_sha256": row["evaluation_array_sha256"],
                    "n_evaluation": row["n_evaluation"],
                    "evaluation_class_counts": row["evaluation_class_counts"],
                }
                for row in task_rows
            ],
        },
    )
    return suite_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(PACKAGE_ROOT / "data" / SUITE_NAME),
        help="New, non-existing suite directory.",
    )
    parser.add_argument("--public-seed", type=int, default=20260821)
    parser.add_argument(
        "--private-seed-file",
        help=(
            "Optional private file containing a hexadecimal 128-bit-or-longer seed. "
            "If omitted, a cryptographically random seed is used and never exported."
        ),
    )
    parser.add_argument("--expert-count", type=int, default=30)
    parser.add_argument("--per-group-label-count", type=int, default=48)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.private_seed_file:
        private_seed_text = Path(args.private_seed_file).read_text(encoding="utf-8").strip()
        if len(private_seed_text) < 32:
            raise ValueError("private seed must contain at least 128 bits (32 hexadecimal digits)")
        private_seed = int(private_seed_text, 16)
    else:
        private_seed = secrets.randbits(256)
    manifest = generate_suite(
        Path(args.output),
        public_seed=int(args.public_seed),
        private_seed=private_seed,
        expert_count=int(args.expert_count),
        per_group_label_count=int(args.per_group_label_count),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
