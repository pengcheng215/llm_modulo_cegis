"""Run and audit the preregistered CarryWaterActive Q48 two-arm experiment.

The experiment separates numeric fitting from structure selection:

* ``correct_composite_only`` supplies the correct qualitative structure;
* ``full_bank`` supplies that structure together with atomics and proxies.

Private evaluation is always a separate post-hoc subprocess after the training
artifacts have been sealed.  This runner never uses a private metric to select,
revise, or retrain a model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

import numpy as np
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from llm_modulo_cegis.carrywater_active import validate_trajectory
from llm_modulo_cegis.data import load_candidate_pool, load_task_spec
from llm_modulo_cegis.types import SAFE_LABEL, Trajectory, VIOLATION_LABEL


DEFAULT_PLAN = PACKAGE_ROOT / "configs" / "carrywater_active_q48_multiseed_plan.yaml"
DEFAULT_FORMAL_OUTPUT = PACKAGE_ROOT / "outputs" / "carrywater_active_q48_5seed"
DEFAULT_PILOT_OUTPUT = PACKAGE_ROOT / "outputs" / "carrywater_active_q48_pilot_seed7"
ARMS = ("correct_composite_only", "full_bank")
FIXED_SEEDS = (7, 19, 37, 73, 109)
CORRECT_HYPOTHESIS_ID = "h_carrywater_composite"
ALLOWED_CONFIG_DIFFERENCES = {
    "output_dir",
    "semantic_reasoner.hypothesis_bank",
    "semantic.beam_width",
    "semantic.max_initial_hypotheses",
    "loop.maximum_active_hypotheses",
    "loop.query_hypothesis_beam",
    "loop.candidate_pool_per_hypothesis",
}
PRIVATE_SUCCESS_THRESHOLDS = {
    "trajectory_balanced_accuracy": 0.70,
    "trajectory_exact_pair_accuracy": 0.50,
    "trajectory_worst_pair_target_balanced_accuracy": 0.60,
    "trajectory_minimum_clause_recall": 0.60,
    "heldout_expert_safe_rate": 0.90,
}
CORE_SEALED_ARTIFACTS = {
    "constraint_models.pt",
    "result.json",
    "hypothesis_bank.json",
    "oracle_queries.npz",
    "oracle_query_log.json",
}
Q48_AUDIT_SEALED_ARTIFACTS = {
    "evidence_history.json",
    "evaluation_history.json",
    "query_diagnostics.json",
    "threshold_calibration_history.json",
    "qualified_checkpoint_history.json",
    "stage_diagnostics.json",
}


class AuditError(RuntimeError):
    """Raised when a completed run violates the preregistered contract."""


def repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_hash(*arrays: np.ndarray) -> str:
    """Hash arrays with explicit dtype and shape, independent of host byte order."""

    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
        header = json.dumps(
            {"shape": list(canonical.shape), "dtype": canonical.dtype.str},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_directory(output_root: Path, arm: str, seed: int) -> Path:
    return output_root / arm / f"seed_{seed}"


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(child, path))
    return result


def config_differences(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    return {
        key
        for key in set(left_flat) | set(right_flat)
        if left_flat.get(key) != right_flat.get(key)
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_plan(plan: dict[str, Any], plan_path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    """Validate plan, configs, and the one intended treatment difference."""

    _require(int(plan.get("schema_version", 0)) == 1, "plan schema_version must be 1")
    seeds = tuple(int(seed) for seed in plan.get("seeds", ()))
    _require(seeds == FIXED_SEEDS, f"seeds must be exactly {list(FIXED_SEEDS)}")
    arms = plan.get("arms")
    _require(isinstance(arms, dict) and tuple(arms) == ARMS, f"arms must be {ARMS}")

    budget = plan.get("oracle_budget", {})
    expected_budget = {
        "expected_total_queries": 48,
        "warmup_queries": 12,
        "warmup_max_queries": 12,
        "outer_rounds": 3,
        "queries_per_outer_round": 12,
    }
    for key, expected in expected_budget.items():
        _require(int(budget.get(key, -1)) == expected, f"oracle_budget.{key} must be {expected}")
    _require(bool(budget.get("require_full_round_budget")), "query underfill must fail the run")
    expected_rounds = budget.get("expected_round_distribution", {})
    _require(
        expected_rounds
        == {"round_0_warmup": 12, "round_1": 12, "round_2": 12, "round_3": 12},
        "round distribution must be 12/12/12/12",
    )

    controls = plan.get("controls", {})
    exact_controls = {
        "device": "cpu",
        "semantic_backend": "frozen_bank",
        "trainer_epochs": 80,
        "ensemble_size": 2,
        "violation_pooling_mode": "all_states",
        "falsifier_restarts": 1,
        "safe_query_boundary_bisection_steps": 0,
        "candidate_proposals_per_round": 49,
    }
    for key, expected in exact_controls.items():
        _require(controls.get(key) == expected, f"controls.{key} must be {expected!r}")
    for key in (
        "freeze_revisions",
        "public_candidate_pool_is_identical_across_arms_and_seeds",
        "warmup_trajectory_hash_must_match_all_runs",
    ):
        _require(bool(controls.get(key)), f"controls.{key} must be true")
    _require(not bool(controls.get("bootstrap_queries")), "bootstrap_queries must be false")
    _require(int(controls.get("llm_interactions_per_run", -1)) == 0, "LLM calls must be zero")
    _require(
        bool(controls.get("retain_best_public_qualified_checkpoint")),
        "best public-qualified checkpoint retention must be enabled",
    )
    _require(
        controls.get("decision_threshold_calibration_labels")
        == "disjoint_final_calibration_warmup_family",
        "decision-threshold calibration must use the disjoint final_calibration split",
    )
    _require(
        controls.get("decision_threshold_selection_holdout") == "warmup_validation",
        "threshold selection holdout must remain warmup_validation",
    )

    protocol = plan.get("evaluation_protocol", {})
    _require(not bool(protocol.get("inline_private_evaluation")), "private evaluation must be deferred")
    _require(
        protocol.get("private_evaluation_timing")
        == "posthoc_only_after_training_artifacts_are_frozen",
        "private evaluation timing is not post-hoc",
    )

    configs: dict[str, dict[str, Any]] = {}
    banks: dict[str, dict[str, Any]] = {}
    paths: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        arm_plan = arms[arm]
        config_path = repository_path(str(arm_plan["config"]))
        bank_path = repository_path(str(arm_plan["frozen_hypothesis_bank"]))
        _require(config_path.is_file(), f"missing config: {config_path}")
        _require(bank_path.is_file(), f"missing bank: {bank_path}")
        config = load_yaml(config_path)
        bank = load_json(bank_path)
        configs[arm], banks[arm] = config, bank
        paths[arm] = {"config": str(config_path), "bank": str(bank_path)}

        _require(config.get("device") == "cpu", f"{arm}: device must be cpu")
        _require(config.get("semantic_reasoner", {}).get("backend") == "frozen_bank", f"{arm}: frozen bank required")
        configured_bank = repository_path(config["semantic_reasoner"]["hypothesis_bank"])
        _require(configured_bank == bank_path, f"{arm}: config/plan bank paths differ")
        _require(not bool(config.get("data", {}).get("inline_private_evaluation")), f"{arm}: inline private evaluation forbidden")
        _require("private_dir" not in config.get("data", {}), f"{arm}: private_dir must not be training input")
        _require(bool(config.get("data", {}).get("membership_oracle_path")), f"{arm}: membership oracle path required")
        _require(int(config.get("trainer", {}).get("epochs", -1)) == 80, f"{arm}: epochs must be 80")
        _require(int(config.get("model", {}).get("ensemble_size", -1)) == 2, f"{arm}: ensemble_size must be 2")
        _require(not bool(config.get("trainer", {}).get("bootstrap_queries")), f"{arm}: bootstrap must be off")
        _require(config.get("trainer", {}).get("violation_pooling_mode") == "all_states", f"{arm}: all_states required")
        loop = config.get("loop", {})
        checks = {
            "warmup_queries": 12,
            "max_warmup_queries": 12,
            "outer_rounds": 3,
            "oracle_query_budget_per_round": 12,
            "falsifier_restarts": 1,
            "safe_query_boundary_bisection_steps": 0,
            "queries_per_hypothesis": 0,
        }
        for key, expected in checks.items():
            _require(int(loop.get(key, -1)) == expected, f"{arm}: loop.{key} must be {expected}")
        _require(bool(loop.get("freeze_revisions")), f"{arm}: revisions must be frozen")
        _require(
            bool(loop.get("retain_best_qualified_checkpoint")),
            f"{arm}: best qualified checkpoint retention must be enabled",
        )
        _require(bool(loop.get("require_full_round_budget")), f"{arm}: strict budget flag required")

        active = int(arm_plan["active_hypotheses"])
        pool = int(arm_plan["candidate_pool_per_hypothesis"])
        beam = int(arm_plan["query_hypothesis_beam"])
        proposals = int(arm_plan["candidate_proposals_per_round"])
        _require(pool * beam == proposals == 49, f"{arm}: candidate proposal count must be 49")
        _require(int(loop.get("candidate_pool_per_hypothesis", -1)) == pool, f"{arm}: pool size mismatch")
        _require(int(loop.get("query_hypothesis_beam", -1)) == beam, f"{arm}: query beam mismatch")
        _require(int(loop.get("maximum_active_hypotheses", -1)) == active, f"{arm}: active count mismatch")
        _require(int(config.get("semantic", {}).get("max_initial_hypotheses", -1)) == active, f"{arm}: initial count mismatch")
        _require(int(config.get("semantic", {}).get("beam_width", -1)) == active, f"{arm}: semantic beam mismatch")

    differences = config_differences(configs[ARMS[0]], configs[ARMS[1]])
    unexpected = differences - ALLOWED_CONFIG_DIFFERENCES
    _require(not unexpected, f"unregistered arm config differences: {sorted(unexpected)}")

    correct_entries = banks["correct_composite_only"].get("entries", {})
    full_entries = banks["full_bank"].get("entries", {})
    _require(set(correct_entries) == {CORRECT_HYPOTHESIS_ID}, "correct-only bank must contain exactly one composite")
    _require(len(full_entries) == 7, "full bank must contain seven hypotheses")
    _require(CORRECT_HYPOTHESIS_ID in full_entries, "full bank is missing the correct composite")
    correct_payload = correct_entries[CORRECT_HYPOTHESIS_ID].get("hypothesis")
    full_payload = full_entries[CORRECT_HYPOTHESIS_ID].get("hypothesis")
    _require(correct_payload == full_payload, "correct composite differs between arms")
    _require(len(correct_payload.get("clauses", ())) == 3, "correct composite must contain three clauses")

    correct_loop = configs["correct_composite_only"]["loop"]
    full_loop = configs["full_bank"]["loop"]
    warmup_role_counts = {
        "warmup_validation": {
            "safe": int(correct_loop.get("warmup_validation_safe_count", 0)),
            "violation": int(correct_loop.get("warmup_validation_violation_count", 0)),
        },
        "final_calibration": {
            "safe": int(correct_loop.get("final_calibration_safe_count", 0)),
            "violation": int(correct_loop.get("final_calibration_violation_count", 0)),
        },
    }
    for role, counts in warmup_role_counts.items():
        for label_name, count in counts.items():
            other = int(
                full_loop.get(
                    f"{role}_{label_name}_count",
                    0,
                )
            )
            _require(
                count == other,
                f"warmup role count differs across arms: {role}.{label_name}",
            )
            _require(count >= 1, f"{role}.{label_name} must reserve at least one query")
    registered_warmup_labels = controls.get("warmup_label_counts", {"safe": 3, "violation": 9})
    for label_name in ("safe", "violation"):
        reserved = sum(
            counts[label_name] for counts in warmup_role_counts.values()
        )
        available = int(registered_warmup_labels[label_name])
        _require(
            reserved <= available,
            f"reserved {label_name} warmup roles ({reserved}) exceed public prefix supply ({available})",
        )
    reserved_total = sum(sum(counts.values()) for counts in warmup_role_counts.values())
    _require(
        reserved_total < int(budget["warmup_queries"]),
        "warmup evidence splits must leave at least one ordinary training query",
    )
    registered_calibration_counts = controls.get(
        "decision_threshold_calibration_label_counts",
        warmup_role_counts["final_calibration"],
    )
    _require(
        warmup_role_counts["final_calibration"] == registered_calibration_counts,
        "plan calibration label counts disagree with both arm configs",
    )

    return {
        "valid": True,
        "plan": str(plan_path.resolve()),
        "plan_sha256": file_sha256(plan_path),
        "seeds": list(seeds),
        "arms": list(ARMS),
        "config_paths": paths,
        "config_differences": sorted(differences),
        "allowed_config_differences": sorted(ALLOWED_CONFIG_DIFFERENCES),
        "correct_composite_shared_exactly": True,
        "queries_per_run": 48,
        "candidate_proposals_per_round": 49,
        "query_budget_matched": True,
        "compute_budget_matched": False,
        "compute_note": (
            "Proposal calls are matched, but full_bank scores more active models per proposal; "
            "runtime and planned cross-model scoring slots are reported separately."
        ),
        "warmup_evidence_split": {
            "training": "warmup",
            "structure_selection": "warmup_validation",
            "decision_threshold": "final_calibration",
            "role_label_counts": warmup_role_counts,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--output-root")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--pilot", action="store_true", help="Run/audit only the first fixed seed (7).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summarize-only", action="store_true", help="Audit existing runs; launch nothing.")
    mode.add_argument("--validate-only", action="store_true", help="Validate plan/config/bank fairness only.")
    return parser.parse_args()


def build_training_command(config_path: Path, seed: int, output: Path) -> list[str]:
    return [
        sys.executable,
        str(PACKAGE_ROOT / "run_carrywater_active.py"),
        "--config",
        str(config_path.resolve()),
        "--seed",
        str(seed),
        "--output",
        str(output.resolve()),
    ]


def build_posthoc_command(output: Path, *, diagnostic_all: bool) -> list[str]:
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "tools" / "evaluate_carrywater_active.py"),
        "--run-dir",
        str(output.resolve()),
    ]
    if diagnostic_all:
        command.extend(
            [
                "--diagnostic-all-hypotheses",
                "--output",
                str((output / "posthoc_all_hypotheses_diagnostic.json").resolve()),
            ]
        )
    return command


def _run_stage(
    command: list[str],
    *,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    started_at = utc_now()
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration = time.perf_counter() - start
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return {
        "command": command,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "duration_seconds": duration,
        "returncode": int(completed.returncode),
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
    }


def run_one(plan: dict[str, Any], output_root: Path, arm: str, seed: int) -> dict[str, Any]:
    output = run_directory(output_root, arm, seed)
    if output.exists():
        raise FileExistsError(f"refusing to reuse run directory: {output}")
    config_path = repository_path(plan["arms"][arm]["config"])
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": str(seed),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "arm": arm,
        "seed": seed,
        "started_at_utc": utc_now(),
        "stages": {},
    }
    stages = (
        ("training", build_training_command(config_path, seed, output)),
        ("official_posthoc", build_posthoc_command(output, diagnostic_all=False)),
        ("diagnostic_all_posthoc", build_posthoc_command(output, diagnostic_all=True)),
    )
    try:
        for name, command in stages:
            stage = _run_stage(
                command,
                environment=environment,
                stdout_path=output / f"{name}_stdout.log",
                stderr_path=output / f"{name}_stderr.log",
            )
            metadata["stages"][name] = stage
            if stage["returncode"] != 0:
                stderr = (output / stage["stderr"]).read_text(encoding="utf-8")
                tail = "\n".join(stderr.splitlines()[-20:])
                raise RuntimeError(f"{arm} seed={seed} stage={name} failed\n{tail}")
    finally:
        output.mkdir(parents=True, exist_ok=True)
        metadata["completed_at_utc"] = utc_now()
        metadata["total_duration_seconds"] = sum(
            float(stage["duration_seconds"])
            for stage in metadata["stages"].values()
        )
        (output / "q48_run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return metadata


def launch_runs(
    plan: dict[str, Any],
    plan_path: Path,
    output_root: Path,
    seeds: Iterable[int],
    max_workers: int,
) -> None:
    if max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}; choose a new directory")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_plan_resolved.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_root / "experiment_plan_source.json").write_text(
        json.dumps(
            {"path": str(plan_path.resolve()), "sha256": file_sha256(plan_path)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    tasks = [(arm, int(seed)) for seed in seeds for arm in ARMS]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_one, plan, output_root, arm, seed): (arm, seed)
            for arm, seed in tasks
        }
        for future in as_completed(future_map):
            arm, seed = future_map[future]
            try:
                metadata = future.result()
                print(
                    f"completed arm={arm} seed={seed} "
                    f"seconds={metadata['total_duration_seconds']:.1f}",
                    flush=True,
                )
            except Exception as exc:  # let already-started work finish and remain auditable
                message = f"{arm}/seed_{seed}: {exc}"
                failures.append(message)
                print(f"failed {message}", flush=True)
    if failures:
        raise RuntimeError("one or more Q48 runs failed:\n" + "\n".join(failures))


def _metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None or not math.isfinite(float(value)):
        raise AuditError(f"required private metric {key!r} is missing or non-finite")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise AuditError(f"private metric {key!r} is outside [0,1]: {result}")
    return result


def _close(left: float, right: float, tolerance: float = 1.0e-10) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _history_hypotheses(round_row: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    """Read the canonical evidence rows, with a historical read-only alias."""

    if "hypotheses" in round_row:
        rows = round_row["hypotheses"]
    elif "evidence" in round_row:
        rows = round_row["evidence"]
    else:
        raise AuditError(
            f"evidence-history row has neither 'hypotheses' nor legacy 'evidence': {output}"
        )
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise AuditError(f"public evidence rows are not a list of mappings in {output}")
    return rows


def _checkpoint_selection_key(row: dict[str, Any]) -> tuple[float, int]:
    raw = row.get("selection_key")
    if not isinstance(raw, list) or len(raw) != 2:
        raise AuditError("qualified checkpoint selection_key must be [score, outer_round]")
    score = float(raw[0])
    outer_round = int(raw[1])
    if not math.isfinite(score) or outer_round <= 0:
        raise AuditError("qualified checkpoint selection_key is invalid")
    if int(row.get("outer_round", -1)) != outer_round:
        raise AuditError("qualified checkpoint key/outer_round mismatch")
    return score, outer_round


def _extract_final_public_evidence(
    output: Path,
    champion_id: str,
    *,
    selection_status: str,
    result_decision_threshold: float,
    retain_best_qualified_checkpoint: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve evidence for the model actually frozen in ``result.json``.

    With checkpoint retention enabled, the final CEGIS refit may be weaker than
    an earlier public-qualified checkpoint.  In that case ``result.json`` and
    the checkpoint file contain the restored model, so using only the final
    history row would audit the wrong numeric model.
    """

    history = load_json(output / "evidence_history.json")
    if not isinstance(history, list) or not history:
        raise AuditError(f"empty evidence history in {output}")
    final_round = history[-1]
    if not isinstance(final_round, dict):
        raise AuditError(f"invalid final evidence-history row in {output}")
    final_rows = _history_hypotheses(final_round, output)
    final_matches = [row for row in final_rows if row.get("hypothesis_id") == champion_id]

    history_path = output / "qualified_checkpoint_history.json"
    stage_path = output / "stage_diagnostics.json"
    checkpoint_history = load_json(history_path) if history_path.is_file() else []
    stage_diagnostics = load_json(stage_path) if stage_path.is_file() else []
    if not isinstance(checkpoint_history, list) or not isinstance(stage_diagnostics, list):
        raise AuditError("checkpoint history and stage diagnostics must be lists")
    restore_stages = [
        row
        for row in stage_diagnostics
        if isinstance(row, dict) and row.get("stage") == "best_qualified_checkpoint_restore"
    ]

    if not retain_best_qualified_checkpoint:
        if restore_stages:
            raise AuditError("checkpoint restore stage exists while retention is disabled")
        if len(final_matches) != 1:
            raise AuditError(
                f"final public evidence does not contain one champion row: {champion_id}"
            )
        return final_matches[0], {
            "source": "final_evidence_history_round",
            "outer_round": int(final_round.get("outer_round", len(history))),
            "checkpoint_history_count": len(checkpoint_history),
            "restored": False,
            "restored_from_earlier_round": False,
        }

    if not history_path.is_file() or not stage_path.is_file():
        raise AuditError(
            "checkpoint retention requires qualified_checkpoint_history.json and stage_diagnostics.json"
        )
    if selection_status != "qualified":
        if checkpoint_history or restore_stages:
            raise AuditError(
                "an inconclusive retained-checkpoint run cannot contain qualified checkpoints or a restore stage"
            )
        if len(final_matches) != 1:
            raise AuditError(
                f"inconclusive final evidence does not contain provisional champion {champion_id}"
            )
        return final_matches[0], {
            "source": "final_evidence_history_round",
            "outer_round": int(final_round.get("outer_round", len(history))),
            "checkpoint_history_count": 0,
            "restored": False,
            "restored_from_earlier_round": False,
        }

    if not checkpoint_history:
        raise AuditError("qualified retained-checkpoint run has empty checkpoint history")
    round_rows: dict[int, dict[str, Any]] = {}
    for fallback_round, round_row in enumerate(history, start=1):
        if not isinstance(round_row, dict):
            raise AuditError("evidence history contains a non-mapping row")
        outer_round = int(round_row.get("outer_round", fallback_round))
        if outer_round in round_rows:
            raise AuditError(f"duplicate evidence-history outer_round={outer_round}")
        round_rows[outer_round] = round_row

    incumbent_key = (-float("inf"), -1)
    validated: list[tuple[tuple[float, int], dict[str, Any]]] = []
    previous_round = 0
    for checkpoint in checkpoint_history:
        if not isinstance(checkpoint, dict):
            raise AuditError("qualified checkpoint history contains a non-mapping row")
        key = _checkpoint_selection_key(checkpoint)
        if key[1] <= previous_round:
            raise AuditError("qualified checkpoint history is not in increasing round order")
        previous_round = key[1]
        selected_as_best = key > incumbent_key
        if bool(checkpoint.get("selected_as_best_so_far")) != selected_as_best:
            raise AuditError("qualified checkpoint selected_as_best_so_far flag is inconsistent")
        if selected_as_best:
            incumbent_key = key
        checkpoint_id = str(checkpoint.get("hypothesis_id"))
        evidence = checkpoint.get("selection_evidence")
        if not isinstance(evidence, dict):
            raise AuditError("qualified checkpoint is missing selection_evidence")
        if evidence.get("hypothesis_id") != checkpoint_id or not bool(
            evidence.get("champion_eligible")
        ):
            raise AuditError("qualified checkpoint evidence is not eligible or has the wrong id")
        if not _close(float(evidence.get("selection_score")), key[0]):
            raise AuditError("qualified checkpoint evidence score disagrees with selection_key")
        checkpoint_round = round_rows.get(key[1])
        if checkpoint_round is None:
            raise AuditError("qualified checkpoint refers to a missing evidence-history round")
        if checkpoint_round.get("selection_status") != "qualified":
            raise AuditError("qualified checkpoint refers to a non-qualified public round")
        matching_round_evidence = [
            row
            for row in _history_hypotheses(checkpoint_round, output)
            if row.get("hypothesis_id") == checkpoint_id
        ]
        if len(matching_round_evidence) != 1 or matching_round_evidence[0] != evidence:
            raise AuditError("qualified checkpoint evidence differs from evidence_history")
        threshold = float(checkpoint.get("decision_threshold"))
        if not math.isfinite(threshold):
            raise AuditError("qualified checkpoint decision threshold is non-finite")
        validated.append((key, checkpoint))

    best_key, best = max(validated, key=lambda item: item[0])
    if not bool(best.get("selected_as_best_so_far")):
        raise AuditError("maximum qualified checkpoint was not marked selected")
    if str(best.get("hypothesis_id")) != champion_id:
        raise AuditError("restored best checkpoint champion disagrees with result")
    if not _close(float(best.get("decision_threshold")), result_decision_threshold, 1.0e-7):
        raise AuditError("restored checkpoint threshold disagrees with result")
    if len(restore_stages) != 1:
        raise AuditError("qualified retained-checkpoint run must contain exactly one restore stage")
    restore = restore_stages[0]
    expected_stage_fields = {
        "restored_outer_round": int(best["outer_round"]),
        "champion_hypothesis_id": champion_id,
        "selection_key": best["selection_key"],
        "selection_evidence": best["selection_evidence"],
    }
    for field, expected in expected_stage_fields.items():
        if restore.get(field) != expected:
            raise AuditError(f"checkpoint restore stage disagrees on {field}")
    if not _close(float(restore.get("decision_threshold")), result_decision_threshold, 1.0e-7):
        raise AuditError("checkpoint restore stage threshold disagrees with result")
    final_outer_round = int(final_round.get("outer_round", len(history)))
    return dict(best["selection_evidence"]), {
        "source": "best_qualified_checkpoint_restore",
        "outer_round": int(best["outer_round"]),
        "selection_key": list(best_key),
        "decision_threshold": float(best["decision_threshold"]),
        "checkpoint_history_count": len(checkpoint_history),
        "restored": True,
        "restored_from_earlier_round": int(best["outer_round"]) != final_outer_round,
        "last_round_selection_status": final_round.get("selection_status"),
    }


def _public_input_hashes(manifest: dict[str, Any], dataset_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    inputs = manifest.get("inputs", {})
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_file():
            continue
        matches = [value for key, value in inputs.items() if key.replace("\\", "/").endswith(f"/{path.name}")]
        if len(matches) != 1:
            raise AuditError(f"implementation manifest does not uniquely record public input {path.name}")
        actual = file_sha256(path)
        if matches[0] != actual:
            raise AuditError(f"public input hash mismatch for {path.name}")
        result[path.name] = actual
    return result


def _audit_runner_metadata(output: Path, arm: str, seed: int) -> dict[str, Any]:
    metadata = load_json(output / "q48_run_metadata.json")
    if metadata.get("arm") != arm or int(metadata.get("seed", -1)) != seed:
        raise AuditError("runner metadata arm/seed mismatch")
    expected_stages = {"training", "official_posthoc", "diagnostic_all_posthoc"}
    if set(metadata.get("stages", {})) != expected_stages:
        raise AuditError("runner metadata does not contain all three stages")
    for stage_name, stage in metadata["stages"].items():
        if int(stage.get("returncode", -1)) != 0:
            raise AuditError(f"stage {stage_name} did not exit successfully")
        for stream in ("stdout", "stderr"):
            if not (output / str(stage.get(stream))).is_file():
                raise AuditError(f"missing captured {stage_name} {stream} log")
    return metadata


def audit_run(
    plan: dict[str, Any],
    output_root: Path,
    arm: str,
    seed: int,
    *,
    require_runner_metadata: bool = True,
) -> dict[str, Any]:
    """Strictly audit one frozen training run and both post-hoc evaluations."""

    output = run_directory(output_root, arm, seed)
    required = (
        "result.json",
        "resolved_config.yaml",
        "freeze_manifest.json",
        "implementation_manifest.json",
        "semantic_interactions.json",
        "evidence_history.json",
        "oracle_queries.npz",
        "oracle_query_log.json",
        "posthoc_evaluation.json",
        "posthoc_all_hypotheses_diagnostic.json",
        "evaluation_history.json",
        "query_diagnostics.json",
        "threshold_calibration_history.json",
        "qualified_checkpoint_history.json",
        "stage_diagnostics.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise AuditError(f"{arm}/seed_{seed} is missing artifacts: {missing}")

    result = load_json(output / "result.json")
    config = load_yaml(output / "resolved_config.yaml")
    freeze = load_json(output / "freeze_manifest.json")
    implementation = load_json(output / "implementation_manifest.json")
    interactions = load_json(output / "semantic_interactions.json")
    official = load_json(output / "posthoc_evaluation.json")
    diagnostic = load_json(output / "posthoc_all_hypotheses_diagnostic.json")
    metadata = (
        _audit_runner_metadata(output, arm, seed)
        if require_runner_metadata
        else {"total_duration_seconds": None, "stages": {}}
    )

    expected_queries = int(plan["oracle_budget"]["expected_total_queries"])
    if int(result.get("oracle_queries", -1)) != expected_queries:
        raise AuditError(f"result.oracle_queries is not {expected_queries}")
    if result.get("llm_interactions") != 0 or result.get("llm_fallbacks") != 0 or result.get("llm_augmentations") != 0:
        raise AuditError("frozen-bank run unexpectedly used or repaired LLM output")
    if interactions != []:
        raise AuditError("semantic_interactions.json must be empty for a frozen bank")

    status = str(result.get("selection_status"))
    eligible = bool(result.get("champion_eligible"))
    if status not in {"qualified", "inconclusive"} or (status == "qualified") != eligible:
        raise AuditError("selection_status and champion_eligible are inconsistent")
    champion_id = str(result.get("champion_hypothesis_id"))
    if arm == "correct_composite_only" and champion_id != CORRECT_HYPOTHESIS_ID:
        raise AuditError("correct-only arm selected a hypothesis outside its supplied composite")

    if not bool(freeze.get("training_complete")):
        raise AuditError("training artifact is not sealed complete")
    if bool(freeze.get("private_evaluation_loaded_before_freeze", True)):
        raise AuditError("private evaluation was loaded before freeze")
    if freeze.get("private_evaluation_mode") != "deferred_posthoc":
        raise AuditError("freeze manifest does not record deferred post-hoc evaluation")
    if freeze.get("champion_hypothesis_id") != champion_id or freeze.get("selection_status") != status:
        raise AuditError("freeze manifest disagrees with result")
    sealed_artifacts = freeze.get("training_artifact_sha256", {})
    if not CORE_SEALED_ARTIFACTS.issubset(sealed_artifacts):
        raise AuditError(
            "freeze manifest does not seal all five required core training artifacts"
        )
    missing_q48_audit_seals = Q48_AUDIT_SEALED_ARTIFACTS - set(sealed_artifacts)
    if missing_q48_audit_seals:
        raise AuditError(
            "Q48 freeze manifest does not seal all six audit artifacts: "
            f"{sorted(missing_q48_audit_seals)}"
        )
    for name, expected_hash in sealed_artifacts.items():
        path = output / name
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise AuditError(f"sealed artifact hash mismatch: {name}")

    if int(config.get("seed", -1)) != seed:
        raise AuditError("resolved numeric seed mismatch")
    if repository_path(config.get("output_dir")) != output.resolve():
        raise AuditError("resolved output directory mismatch")
    if config.get("semantic_reasoner", {}).get("backend") != "frozen_bank":
        raise AuditError("resolved semantic backend is not frozen_bank")
    if bool(config.get("data", {}).get("inline_private_evaluation")):
        raise AuditError("resolved config enabled inline private evaluation")
    loop = config.get("loop", {})
    if not bool(loop.get("require_full_round_budget")):
        raise AuditError("resolved config did not enforce full per-round query budgets")
    retain_best_checkpoint = bool(loop.get("retain_best_qualified_checkpoint", False))
    if retain_best_checkpoint:
        checkpoint_artifacts = (
            output / "qualified_checkpoint_history.json",
            output / "stage_diagnostics.json",
        )
        missing_checkpoint_artifacts = [
            path.name for path in checkpoint_artifacts if not path.is_file()
        ]
        if missing_checkpoint_artifacts:
            raise AuditError(
                "retained-checkpoint run is missing artifacts: "
                f"{missing_checkpoint_artifacts}"
            )

    runtime = implementation.get("runtime", {})
    runtime_environment = runtime.get("environment", {})
    expected_environment = {"PYTHONHASHSEED": str(seed), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for key, expected in expected_environment.items():
        if runtime_environment.get(key) != expected:
            raise AuditError(f"runtime environment {key} was not pinned to {expected}")
    if int(runtime.get("seed", -1)) != seed or runtime.get("private_evaluation_mode") != "deferred_posthoc":
        raise AuditError("implementation runtime seed/private mode mismatch")
    if implementation.get("evaluation_only_inputs"):
        raise AuditError("training implementation manifest contains evaluation-only private files")
    for section in ("inputs", "evaluation_only_inputs"):
        forbidden = [
            key for key in implementation.get(section, {})
            if "/private/" in key.replace("\\", "/") and not key.endswith("/oracle.json")
        ]
        if forbidden:
            raise AuditError(f"training manifest exposed private evaluation files: {forbidden}")

    query_log = load_json(output / "oracle_query_log.json")
    if not isinstance(query_log, list):
        raise AuditError("oracle_query_log.json must be a list")
    with np.load(output / "oracle_queries.npz", allow_pickle=False) as archive:
        required_arrays = {"observations", "actions", "labels", "outer_rounds"}
        if not required_arrays.issubset(archive.files):
            raise AuditError(f"oracle archive missing arrays: {sorted(required_arrays - set(archive.files))}")
        observations = np.asarray(archive["observations"])
        actions = np.asarray(archive["actions"])
        labels = np.asarray(archive["labels"])
        outer_rounds = np.asarray(archive["outer_rounds"])
    lengths = {len(query_log), len(observations), len(actions), len(labels), len(outer_rounds)}
    if lengths != {expected_queries}:
        raise AuditError(f"result/log/archive query counts are not all {expected_queries}: {sorted(lengths)}")
    expected_distribution = {0: 12, 1: 12, 2: 12, 3: 12}
    distribution = dict(sorted(Counter(map(int, outer_rounds)).items()))
    if distribution != expected_distribution:
        raise AuditError(f"oracle round distribution mismatch: {distribution}")
    if not np.array_equal(outer_rounds[:12], np.zeros(12, dtype=outer_rounds.dtype)):
        raise AuditError("warmup is not the first 12-query prefix")

    dataset_dir = repository_path(config["data"]["dataset_dir"])
    task_spec = load_task_spec(dataset_dir)
    candidates = load_candidate_pool(dataset_dir)
    candidate_by_id = {str(item.metadata.get("trajectory_id")): item for item in candidates}
    expected_warmup_ids = [str(item.metadata.get("trajectory_id")) for item in candidates[:12]]
    trajectory_hashes: list[str] = []
    pool_ids: list[str] = []
    for index, entry in enumerate(query_log):
        if int(entry.get("label", -1)) != int(labels[index]):
            raise AuditError(f"query {index}: log/archive label mismatch")
        if int(entry.get("outer_round", -1)) != int(outer_rounds[index]):
            raise AuditError(f"query {index}: log/archive round mismatch")
        metadata_row = entry.get("trajectory_metadata", {})
        pool_id = str(metadata_row.get("pool_candidate_id", ""))
        if pool_id not in candidate_by_id:
            raise AuditError(f"query {index}: unregistered public pool candidate {pool_id!r}")
        candidate = candidate_by_id[pool_id]
        if not np.array_equal(observations[index], candidate.states) or not np.array_equal(actions[index], candidate.actions):
            raise AuditError(f"query {index}: archive rollout differs from registered public candidate")
        trajectory = Trajectory(observations[index], actions[index], dt=float(task_spec.dt))
        validity = validate_trajectory(trajectory)
        if not validity.valid:
            raise AuditError(f"query {index}: public dynamics invalid ({validity.reason})")
        pool_ids.append(pool_id)
        trajectory_hashes.append(canonical_array_hash(observations[index], actions[index]))
    if len(set(trajectory_hashes)) != expected_queries:
        raise AuditError("oracle queries do not contain 48 unique trajectories")
    if pool_ids[:12] != expected_warmup_ids:
        raise AuditError("warmup does not use the registered first 12 public candidates")
    if set(pool_ids[12:]) & set(expected_warmup_ids):
        raise AuditError("active queries reused a warmup public rollout")
    warmup_labels = labels[:12]
    if Counter(map(int, warmup_labels)) != Counter({SAFE_LABEL: 3, VIOLATION_LABEL: 9}):
        raise AuditError("warmup label counts are not safe=3, violation=9")
    warmup_sources = [str(entry.get("source", "")) for entry in query_log[:12]]
    warmup_evidence_roles = [
        str(entry.get("trajectory_metadata", {}).get("evidence_role", ""))
        for entry in query_log[:12]
    ]
    allowed_role_pairs = {
        "warmup": "warmup_training",
        "warmup_validation": "heldout_structure_selection",
        "final_calibration": "final_threshold_calibration",
    }
    for index, (source, evidence_role) in enumerate(
        zip(warmup_sources, warmup_evidence_roles)
    ):
        if source not in allowed_role_pairs:
            raise AuditError(f"warmup query {index}: unexpected evidence source {source!r}")
        if evidence_role != allowed_role_pairs[source]:
            raise AuditError(
                f"warmup query {index}: source/evidence-role mismatch "
                f"({source!r}, {evidence_role!r})"
            )
    configured_role_counts = {
        ("warmup_validation", SAFE_LABEL): int(
            loop.get("warmup_validation_safe_count", 0)
        ),
        ("warmup_validation", VIOLATION_LABEL): int(
            loop.get("warmup_validation_violation_count", 0)
        ),
        ("final_calibration", SAFE_LABEL): int(
            loop.get("final_calibration_safe_count", 0)
        ),
        ("final_calibration", VIOLATION_LABEL): int(
            loop.get("final_calibration_violation_count", 0)
        ),
    }
    observed_role_label_counts = Counter(
        (source, int(label))
        for source, label in zip(warmup_sources, warmup_labels)
    )
    for key, expected_count in configured_role_counts.items():
        if observed_role_label_counts[key] != expected_count:
            raise AuditError(
                "warmup evidence-role count disagrees with resolved config: "
                f"source={key[0]} label={key[1]} "
                f"observed={observed_role_label_counts[key]} expected={expected_count}"
            )

    public_hashes = _public_input_hashes(implementation, dataset_dir)
    official_flag = official.get("diagnostic_all_hypotheses_enabled")
    diagnostic_flag = diagnostic.get("diagnostic_all_hypotheses_enabled")
    if official_flag is not False or diagnostic_flag is not True:
        raise AuditError("official/diagnostic post-hoc modes are not isolated")
    for payload_name, payload in (("official", official), ("diagnostic", diagnostic)):
        if payload.get("evaluation_protocol") != "frozen_checkpoint_posthoc_private":
            raise AuditError(f"{payload_name} evaluation protocol mismatch")
        if payload.get("champion_hypothesis_id") != champion_id:
            raise AuditError(f"{payload_name} post-hoc champion changed after freeze")
        if payload.get("selection_status_frozen_before_private_evaluation") != status:
            raise AuditError(f"{payload_name} post-hoc selection status mismatch")
    if official.get("frozen_artifact_sha256") != diagnostic.get("frozen_artifact_sha256"):
        raise AuditError("official and diagnostic evaluations used different frozen artifacts")
    posthoc_frozen = official.get("frozen_artifact_sha256", {})
    for name in ("constraint_models.pt", "result.json"):
        if posthoc_frozen.get(name) != sealed_artifacts.get(name):
            raise AuditError(f"post-hoc evaluation did not use sealed {name}")
    if official.get("champion_metrics") != diagnostic.get("champion_metrics"):
        raise AuditError("official and diagnostic champion metrics differ")
    all_diagnostic = diagnostic.get("all_hypothesis_metrics_diagnostic_only", {})
    if CORRECT_HYPOTHESIS_ID not in all_diagnostic:
        raise AuditError("diagnostic-all output omitted the correct composite")

    structure = official.get("structure_metrics", {})
    qualified = status == "qualified"
    exact = bool(structure.get("exact_structure_recovery"))
    qualified_exact = bool(structure.get("qualified_exact_structure_recovery"))
    erroneous = bool(structure.get("erroneous_qualified_champion"))
    inconclusive = not qualified
    if qualified_exact != (qualified and exact) or erroneous != (qualified and not exact):
        raise AuditError("private structure status fields are internally inconsistent")
    if sum(map(int, (qualified_exact, erroneous, inconclusive))) != 1:
        raise AuditError("qualified-exact/erroneous/inconclusive status partition is invalid")

    private = official.get("champion_metrics", {})
    private_values = {key: _metric(private, key) for key in PRIVATE_SUCCESS_THRESHOLDS}
    safe_accuracy = _metric(private, "trajectory_safe_accuracy")
    violation_recall = _metric(private, "trajectory_violation_recall")
    balanced_accuracy = private_values["trajectory_balanced_accuracy"]
    if not _close(balanced_accuracy, 0.5 * (safe_accuracy + violation_recall)):
        raise AuditError("private balanced accuracy identity failed")
    pair_values = {str(key): float(value) for key, value in private.get("trajectory_pair_target_balanced_accuracy", {}).items()}
    if not pair_values or not _close(private_values["trajectory_worst_pair_target_balanced_accuracy"], min(pair_values.values())):
        raise AuditError("worst pair-target balanced accuracy identity failed")
    clause_values = {str(key): float(value) for key, value in private.get("trajectory_clause_recall", {}).items()}
    if not clause_values or not _close(private_values["trajectory_minimum_clause_recall"], min(clause_values.values())):
        raise AuditError("minimum clause recall identity failed")
    exact_pair = private_values["trajectory_exact_pair_accuracy"]
    if exact_pair > safe_accuracy + 1.0e-10 or exact_pair > violation_recall + 1.0e-10:
        raise AuditError("exact-pair accuracy cannot exceed either marginal class accuracy")
    private_success = all(private_values[key] >= threshold for key, threshold in PRIVATE_SUCCESS_THRESHOLDS.items())

    result_decision_threshold = float(result.get("decision_threshold"))
    if not math.isfinite(result_decision_threshold):
        raise AuditError("result decision threshold is missing or non-finite")
    public, public_evidence_provenance = _extract_final_public_evidence(
        output,
        champion_id,
        selection_status=status,
        result_decision_threshold=result_decision_threshold,
        retain_best_qualified_checkpoint=retain_best_checkpoint,
    )
    public_safe = float(public["safe_accuracy"])
    public_violation = float(public["violation_recall"])
    public_balanced = float(public["balanced_accuracy"])
    if not _close(public_balanced, 0.5 * (public_safe + public_violation)):
        raise AuditError("public balanced accuracy identity failed")
    if bool(public.get("champion_eligible")) != eligible:
        raise AuditError("final public evidence eligibility disagrees with result")

    diagnostic_correct = all_diagnostic[CORRECT_HYPOTHESIS_ID]
    labels_int = labels.astype(np.int64)
    active_labels = labels_int[outer_rounds > 0]
    planned_proposals = int(plan["controls"]["candidate_proposals_per_round"])
    active_hypotheses = int(plan["arms"][arm]["active_hypotheses"])
    return {
        "arm": arm,
        "seed": seed,
        "run_dir": str(output.resolve()),
        "oracle_queries": expected_queries,
        "round_0_queries": distribution[0],
        "round_1_queries": distribution[1],
        "round_2_queries": distribution[2],
        "round_3_queries": distribution[3],
        "unique_trajectory_count": len(set(trajectory_hashes)),
        "warmup_sequence_hash": hashlib.sha256("\n".join(trajectory_hashes[:12]).encode("ascii")).hexdigest(),
        "warmup_label_sequence": "".join(map(str, map(int, labels_int[:12]))),
        "warmup_source_sequence": warmup_sources,
        "warmup_evidence_role_sequence": warmup_evidence_roles,
        "warmup_role_label_counts": {
            f"{source}:{'safe' if label == SAFE_LABEL else 'violation'}": int(count)
            for (source, label), count in sorted(observed_role_label_counts.items())
        },
        "warmup_configured_role_label_counts": {
            f"{source}:{'safe' if label == SAFE_LABEL else 'violation'}": int(count)
            for (source, label), count in sorted(configured_role_counts.items())
        },
        "warmup_pool_ids": pool_ids[:12],
        "all_pool_ids": pool_ids,
        "total_safe_yield": float(np.mean(labels_int == SAFE_LABEL)),
        "active_safe_yield": float(np.mean(active_labels == SAFE_LABEL)),
        "champion_hypothesis_id": champion_id,
        "selection_status": status,
        "champion_eligible": eligible,
        "qualified": qualified,
        "exact_structure": exact,
        "qualified_exact": qualified_exact,
        "erroneous_qualified": erroneous,
        "inconclusive": inconclusive,
        "correct_composite_champion": champion_id == CORRECT_HYPOTHESIS_ID,
        "public_balanced_accuracy": public_balanced,
        "public_safe_accuracy": public_safe,
        "public_violation_recall": public_violation,
        "public_expert_safe_rate": float(public["expert_safe_rate"]),
        "public_fit_expert_safe_rate": float(public["fit_expert_safe_rate"]),
        "public_evidence_provenance": public_evidence_provenance,
        "qualified_checkpoint_restored": bool(
            public_evidence_provenance["restored"]
        ),
        "qualified_checkpoint_restored_from_earlier_round": bool(
            public_evidence_provenance["restored_from_earlier_round"]
        ),
        "qualified_checkpoint_artifacts_sealed": True,
        "all_q48_audit_artifacts_sealed": True,
        "private_trajectory_balanced_accuracy": balanced_accuracy,
        "private_trajectory_safe_accuracy": safe_accuracy,
        "private_trajectory_violation_recall": violation_recall,
        "private_trajectory_exact_pair_accuracy": exact_pair,
        "private_trajectory_pair_ranking_accuracy": _metric(private, "trajectory_pair_ranking_accuracy"),
        "private_trajectory_worst_pair_target_balanced_accuracy": private_values["trajectory_worst_pair_target_balanced_accuracy"],
        "private_trajectory_minimum_clause_recall": private_values["trajectory_minimum_clause_recall"],
        "private_heldout_expert_safe_rate": private_values["heldout_expert_safe_rate"],
        "private_success": private_success,
        "private_clause_recall": clause_values,
        "private_pair_target_balanced_accuracy": pair_values,
        "diagnostic_correct_composite_balanced_accuracy": float(diagnostic_correct["trajectory_balanced_accuracy"]),
        "diagnostic_correct_composite_exact_pair_accuracy": float(diagnostic_correct["trajectory_exact_pair_accuracy"]),
        "diagnostic_correct_composite_worst_pair_target_balanced_accuracy": float(diagnostic_correct["trajectory_worst_pair_target_balanced_accuracy"]),
        "diagnostic_correct_composite_minimum_clause_recall": float(diagnostic_correct["trajectory_minimum_clause_recall"]),
        "public_input_sha256": public_hashes,
        "implementation_sha256": implementation.get("files", {}),
        "total_duration_seconds": metadata.get("total_duration_seconds"),
        "planned_candidate_proposals": planned_proposals * int(plan["oracle_budget"]["outer_rounds"]),
        "planned_cross_hypothesis_scoring_slots": planned_proposals * active_hypotheses * int(plan["oracle_budget"]["outer_rounds"]),
    }


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "sample_sd": None, "median": None, "min": None, "max": None}
    return {
        "n": len(finite),
        "mean": statistics.mean(finite),
        "sample_sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def diagnose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered fault-localization decision tree."""

    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    counts = {arm: len(items) for arm, items in by_arm.items()}
    if not counts["correct_composite_only"] or counts["correct_composite_only"] != counts["full_bank"]:
        raise AuditError("diagnosis requires equal, nonzero seed counts in both arms")
    n = counts["correct_composite_only"]
    stable_target = max(1, math.ceil(0.8 * n))
    correct_qualified_exact = sum(bool(row["qualified_exact"]) for row in by_arm["correct_composite_only"])
    correct_private = sum(bool(row["private_success"]) for row in by_arm["correct_composite_only"])
    full_qualified_exact = sum(bool(row["qualified_exact"]) for row in by_arm["full_bank"])
    full_private = sum(bool(row["private_success"]) for row in by_arm["full_bank"])
    full_erroneous = sum(bool(row["erroneous_qualified"]) for row in by_arm["full_bank"])

    if correct_qualified_exact < stable_target:
        code = "numeric_fitting_or_acquisition_instability"
        explanation = "即使预先给出正确三子句结构，也不能稳定得到 qualified champion；问题首先在数值拟合、阈值校准或查询采集。"
    elif correct_private < stable_target:
        code = "public_gate_private_generalization_misalignment"
        explanation = "正确结构能通过公开资格门，但私有反事实指标不稳定；公开门槛与真正泛化质量没有对齐。"
    elif full_erroneous > 0:
        code = "structure_ranking_or_proxy_rejection_failure"
        explanation = "正确结构可以拟合，但 full bank 曾把错误 proxy 作为 qualified champion；问题在结构排序或 proxy 排除。"
    elif full_qualified_exact < stable_target:
        code = "conservative_selection_or_insufficient_competitive_evidence"
        explanation = "正确结构单独可学，但竞争时经常保持 inconclusive；查询采集或资格门提供的区分证据不足。"
    elif full_private < stable_target:
        code = "competitive_numeric_generalization_instability"
        explanation = "双臂结构选择均稳定，但 full bank 的冻结冠军私有泛化仍不稳定；竞争训练影响了数值校准。"
    else:
        code = "ready_for_matched_baselines"
        explanation = "两臂均达到至少 80% 固定种子稳定率且无错误 qualified champion；下一步可比较 PUCL、全特征 MLP 与随机查询。"
    return {
        "code": code,
        "explanation": explanation,
        "seed_count_per_arm": n,
        "stability_target_count": stable_target,
        "pilot_is_preliminary": n < 5,
        "counts": {
            "correct_only_qualified_exact": correct_qualified_exact,
            "correct_only_private_success": correct_private,
            "full_bank_qualified_exact": full_qualified_exact,
            "full_bank_private_success": full_private,
            "full_bank_erroneous_qualified": full_erroneous,
        },
    }


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    metric_names = (
        "public_balanced_accuracy",
        "public_safe_accuracy",
        "public_violation_recall",
        "public_expert_safe_rate",
        "public_fit_expert_safe_rate",
        "total_safe_yield",
        "active_safe_yield",
        "private_trajectory_balanced_accuracy",
        "private_trajectory_safe_accuracy",
        "private_trajectory_violation_recall",
        "private_trajectory_exact_pair_accuracy",
        "private_trajectory_pair_ranking_accuracy",
        "private_trajectory_worst_pair_target_balanced_accuracy",
        "private_trajectory_minimum_clause_recall",
        "private_heldout_expert_safe_rate",
        "diagnostic_correct_composite_balanced_accuracy",
        "diagnostic_correct_composite_exact_pair_accuracy",
        "total_duration_seconds",
    )
    return {
        "n": len(selected),
        "qualified_count": sum(bool(row["qualified"]) for row in selected),
        "exact_structure_count": sum(bool(row["exact_structure"]) for row in selected),
        "qualified_exact_count": sum(bool(row["qualified_exact"]) for row in selected),
        "erroneous_qualified_count": sum(bool(row["erroneous_qualified"]) for row in selected),
        "inconclusive_count": sum(bool(row["inconclusive"]) for row in selected),
        "correct_composite_champion_count": sum(bool(row["correct_composite_champion"]) for row in selected),
        "private_success_count": sum(bool(row["private_success"]) for row in selected),
        "qualified_checkpoint_restore_count": sum(
            bool(row["qualified_checkpoint_restored"]) for row in selected
        ),
        "earlier_qualified_checkpoint_restore_count": sum(
            bool(row["qualified_checkpoint_restored_from_earlier_round"])
            for row in selected
        ),
        "planned_candidate_proposals_per_run": selected[0]["planned_candidate_proposals"] if selected else None,
        "planned_cross_hypothesis_scoring_slots_per_run": selected[0]["planned_cross_hypothesis_scoring_slots"] if selected else None,
        "metrics": {name: numeric_summary(row[name] for row in selected) for name in metric_names},
    }


def _assert_cross_run_fairness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    warmup_hashes = {row["warmup_sequence_hash"] for row in rows}
    warmup_labels = {row["warmup_label_sequence"] for row in rows}
    warmup_sources = {
        tuple(row["warmup_source_sequence"])
        for row in rows
    }
    warmup_evidence_roles = {
        tuple(row["warmup_evidence_role_sequence"])
        for row in rows
    }
    warmup_role_counts = {
        json.dumps(row["warmup_role_label_counts"], sort_keys=True)
        for row in rows
    }
    warmup_configured_role_counts = {
        json.dumps(row["warmup_configured_role_label_counts"], sort_keys=True)
        for row in rows
    }
    warmup_ids = {tuple(row["warmup_pool_ids"]) for row in rows}
    public_hashes = {json.dumps(row["public_input_sha256"], sort_keys=True) for row in rows}
    implementations = {json.dumps(row["implementation_sha256"], sort_keys=True) for row in rows}
    if (
        len(warmup_hashes) != 1
        or len(warmup_labels) != 1
        or len(warmup_sources) != 1
        or len(warmup_evidence_roles) != 1
        or len(warmup_role_counts) != 1
        or len(warmup_configured_role_counts) != 1
        or len(warmup_ids) != 1
    ):
        raise AuditError(
            "warmup trajectory/label/evidence-role slate differs across arms or seeds"
        )
    if len(public_hashes) != 1:
        raise AuditError("public input hashes differ across arms or seeds")
    if len(implementations) != 1:
        raise AuditError("implementation hashes differ across arms or seeds")
    return {
        "warmup_sequence_hash": next(iter(warmup_hashes)),
        "warmup_label_sequence": next(iter(warmup_labels)),
        "warmup_source_sequence": list(next(iter(warmup_sources))),
        "warmup_evidence_role_sequence": list(next(iter(warmup_evidence_roles))),
        "warmup_role_label_counts": json.loads(next(iter(warmup_role_counts))),
        "warmup_configured_role_label_counts": json.loads(
            next(iter(warmup_configured_role_counts))
        ),
        "warmup_pool_ids": list(next(iter(warmup_ids))),
        "public_input_sha256": json.loads(next(iter(public_hashes))),
        "implementation_identical_across_runs": True,
        "query_budget_identical": True,
        "proposal_calls_identical": True,
        "compute_budget_identical": False,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_summary(
    output_root: Path,
    plan: dict[str, Any],
    plan_validation: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    pilot: bool,
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (int(row["seed"]), ARMS.index(row["arm"])))
    fairness = _assert_cross_run_fairness(rows)
    diagnosis = diagnose(rows)
    payload = {
        "schema_version": 1,
        "experiment_id": plan.get("experiment_id"),
        "created_at_utc": utc_now(),
        "pilot": pilot,
        "posthoc_only_private_metrics": True,
        "private_success_thresholds": PRIVATE_SUCCESS_THRESHOLDS,
        "plan_validation": plan_validation,
        "cross_run_fairness": fairness,
        "runs": rows,
        "arm_summaries": {arm: _arm_summary(rows, arm) for arm in ARMS},
        "diagnosis": diagnosis,
        "statistical_limit": (
            "The fixed five-seed sweep is an engineering stability audit, not a claim of statistical significance."
        ),
        "compute_caveat": (
            "Both arms use Q=48 and 49 proposal calls per round. full_bank evaluates more models per proposal, "
            "so elapsed time and cross-hypothesis scoring slots are not compute-matched."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "q48_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    scalar_rows = [
        {
            key: _csv_value(value)
            for key, value in row.items()
            if key not in {"all_pool_ids", "public_input_sha256", "implementation_sha256"}
        }
        for row in rows
    ]
    fieldnames = sorted({key for row in scalar_rows for key in row})
    with (output_root / "q48_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)

    lines = [
        "# CarryWaterActive Q48 双臂固定种子审计",
        "",
        f"- 模式：{'pilot（仅 seed 7）' if pilot else '正式 5 seeds'}",
        f"- 诊断：`{diagnosis['code']}`",
        f"- 解释：{diagnosis['explanation']}",
        "- 所有私有指标均在训练 artifact 冻结后，由独立 post-hoc 进程计算。",
        "- 两臂查询预算与候选 proposal 次数相同，但 full-bank 的跨模型评分计算量更高。",
        "",
        "| arm | qualified | exact | qualified-exact | erroneous-qualified | inconclusive | earlier restore | private success | private BAcc mean | exact-pair mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        summary = payload["arm_summaries"][arm]
        metrics = summary["metrics"]
        lines.append(
            "| {arm} | {qualified}/{n} | {exact}/{n} | {qe}/{n} | {wrong}/{n} | {inc}/{n} | {restore}/{n} | {private}/{n} | {bacc:.4f} | {pair:.4f} |".format(
                arm=arm,
                n=summary["n"],
                qualified=summary["qualified_count"],
                exact=summary["exact_structure_count"],
                qe=summary["qualified_exact_count"],
                wrong=summary["erroneous_qualified_count"],
                inc=summary["inconclusive_count"],
                restore=summary["earlier_qualified_checkpoint_restore_count"],
                private=summary["private_success_count"],
                bacc=float(metrics["private_trajectory_balanced_accuracy"]["mean"]),
                pair=float(metrics["private_trajectory_exact_pair_accuracy"]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## 完整性结论",
            "",
            "每个 run 均已验证：result/log/npz 都为 48 条、round 分布 12/12/12/12、轨迹唯一且来自公开 pool、动力学有效、LLM 调用为 0、私有评估未在 freeze 前加载、official post-hoc 未启用 diagnostic-all。",
            "",
            "> 五个固定数值种子只用于工程稳定性审计，不能替代统计显著性实验。",
            "",
        ]
    )
    (output_root / "q48_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def summarize(
    plan: dict[str, Any],
    plan_validation: dict[str, Any],
    output_root: Path,
    seeds: Iterable[int],
    *,
    pilot: bool,
) -> dict[str, Any]:
    rows = [
        audit_run(plan, output_root, arm, int(seed))
        for seed in seeds
        for arm in ARMS
    ]
    return write_summary(output_root, plan, plan_validation, rows, pilot=pilot)


def main() -> int:
    args = parse_args()
    plan_path = repository_path(args.plan)
    plan = load_yaml(plan_path)
    plan_validation = validate_plan(plan, plan_path)
    print(json.dumps(plan_validation, indent=2, ensure_ascii=False), flush=True)
    if args.validate_only:
        return 0

    seeds = FIXED_SEEDS[:1] if args.pilot else FIXED_SEEDS
    if args.output_root:
        output_root = repository_path(args.output_root)
    else:
        output_root = DEFAULT_PILOT_OUTPUT if args.pilot else DEFAULT_FORMAL_OUTPUT
    if not args.summarize_only:
        launch_runs(plan, plan_path, output_root, seeds, int(args.max_workers))
    summary = summarize(
        plan,
        plan_validation,
        output_root,
        seeds,
        pilot=bool(args.pilot),
    )
    print(json.dumps(summary["diagnosis"], indent=2, ensure_ascii=False), flush=True)
    print(f"summary={output_root / 'q48_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
