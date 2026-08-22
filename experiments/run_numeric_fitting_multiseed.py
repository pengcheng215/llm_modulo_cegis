"""Run the preregistered query-bootstrap versus full-buffer numeric-fitting sweep.

The paired arms deliberately share the frozen semantic bank, falsifier, safe
acquisition policy, warmup slate, Oracle budget, and numeric seeds.  The sole
configured treatment is ``trainer.bootstrap_queries``.  Private-geometry IoU
metrics are retained as diagnostics and never enter the selection rule.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from run_falsifier_multiseed import (
    extract_warmup_artifact,
    numeric_summary,
    sha256,
    shared_inputs,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_PLAN = PACKAGE_ROOT / "configs" / "numeric_fitting_multiseed_plan.yaml"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs" / "numeric_fitting_multiseed_5seed"
ARMS = ("bootstrap_queries", "full_buffer")
HOLDOUT_SOURCES = {"warmup_validation", "final_calibration"}


def repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Summarize existing completed run directories without launching runs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the plan and print all commands without launching runs.",
    )
    return parser.parse_args()


def run_directory(output_root: Path, arm: str, seed: int) -> Path:
    return output_root / arm / f"seed_{seed}"


def validate_plan(plan: dict[str, Any]) -> None:
    if int(plan.get("schema_version", -1)) != 1:
        raise ValueError("numeric-fitting plan schema_version must be 1")
    seeds = [int(value) for value in plan.get("seeds", [])]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("the paired experiment requires five distinct numeric seeds")
    if set(plan.get("arms", {})) != set(ARMS):
        raise ValueError(f"plan arms must be exactly {list(ARMS)}")
    expected = {"bootstrap_queries": True, "full_buffer": False}
    for arm, value in expected.items():
        actual = plan["arms"][arm].get("trainer_bootstrap_queries")
        if not isinstance(actual, bool) or actual is not value:
            raise ValueError(f"{arm}.trainer_bootstrap_queries must be {value}")
    controls = plan.get("controls", {})
    if str(controls.get("device")) != "cpu":
        raise ValueError("the preregistered sweep must use CPU")
    if str(controls.get("violation_pooling_mode")) != "all_states":
        raise ValueError("the historical bootstrap sweep must use all-state pooling")
    if str(controls.get("semantic_backend")) != "frozen_bank":
        raise ValueError("the preregistered sweep must use a frozen semantic bank")
    if not bool(controls.get("freeze_revisions")):
        raise ValueError("semantic revisions must remain frozen")
    if float(controls.get("false_unsafe_trust_radius", 0.0)) != 0.32:
        raise ValueError("both arms must use hard single-0.32")
    if list(controls.get("false_unsafe_radius_ladder", [])):
        raise ValueError("the radius ladder must be empty")
    if str(controls.get("falsifier_objective")) != "hard_margin":
        raise ValueError("both arms must use the hard-margin falsifier")
    rule = plan.get("decision_rule", {})
    if rule.get("preferred_if_supported") != "full_buffer":
        raise ValueError("decision-rule preferred arm must be full_buffer")
    if rule.get("fallback") != "bootstrap_queries":
        raise ValueError("decision-rule fallback must be bootstrap_queries")
    base_config = repository_path(str(plan["base_config"]))
    frozen_bank = repository_path(str(plan["frozen_hypothesis_bank"]))
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    if not frozen_bank.is_file():
        raise FileNotFoundError(frozen_bank)


def build_command(
    plan: dict[str, Any], arm: str, seed: int, output: Path
) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "run_obstacle_avoid.py"),
        "--config",
        str(repository_path(str(plan["base_config"])).resolve()),
        "--initial-hypothesis-bank",
        str(repository_path(str(plan["frozen_hypothesis_bank"])).resolve()),
        "--freeze-revisions",
        "--false-unsafe-single-radius",
        str(float(plan["controls"]["false_unsafe_trust_radius"])),
        # Preserve the historical fitting ablation: it compares query
        # resampling only, before source-anchor pooling became the default.
        "--violation-pooling-mode",
        str(plan["controls"]["violation_pooling_mode"]),
        "--audit-only-linear-max-support-gate",
        "--seed",
        str(seed),
        "--output",
        str(output.resolve()),
    ]
    command.append(
        "--bootstrap-queries"
        if bool(plan["arms"][arm]["trainer_bootstrap_queries"])
        else "--no-bootstrap-queries"
    )
    return command


def validate_commands(plan: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in (int(value) for value in plan["seeds"]):
        for arm in ARMS:
            command = build_command(plan, arm, seed, run_directory(output_root, arm, seed))
            bootstrap_flags = [
                flag
                for flag in command
                if flag in {"--bootstrap-queries", "--no-bootstrap-queries"}
            ]
            if len(bootstrap_flags) != 1:
                raise AssertionError(f"{arm} seed={seed}: expected one bootstrap flag")
            expected = (
                "--bootstrap-queries" if arm == "bootstrap_queries" else "--no-bootstrap-queries"
            )
            if bootstrap_flags[0] != expected:
                raise AssertionError(f"{arm} seed={seed}: wrong bootstrap flag")
            rows.append({"arm": arm, "seed": seed, "command": command})
    return rows


def run_one(
    plan: dict[str, Any], output_root: Path, arm: str, seed: int
) -> dict[str, Any]:
    output = run_directory(output_root, arm, seed)
    if output.exists():
        raise FileExistsError(f"refusing to reuse run directory: {output}")
    command = build_command(plan, arm, seed, output)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": str(seed),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
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
    elapsed = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "run_stderr.log").write_text(completed.stderr, encoding="utf-8")
    metadata = {
        "arm": arm,
        "seed": seed,
        "command": command,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "duration_seconds_parallel_context_only": elapsed,
        "returncode": completed.returncode,
    }
    (output / "sweep_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-20:])
        raise RuntimeError(f"{arm} seed={seed} failed ({completed.returncode})\n{tail}")
    return metadata


def launch_runs(plan: dict[str, Any], output_root: Path, max_workers: int) -> None:
    if max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_plan_resolved.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    tasks = [
        (arm, int(seed)) for seed in plan["seeds"] for arm in ARMS
    ]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one, plan, output_root, arm, seed): (arm, seed)
            for arm, seed in tasks
        }
        for future in as_completed(futures):
            arm, seed = futures[future]
            try:
                metadata = future.result()
                print(
                    f"completed arm={arm} seed={seed} "
                    f"seconds={metadata['duration_seconds_parallel_context_only']:.1f}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(f"{arm}/seed_{seed}: {exc}")
                print(f"failed arm={arm} seed={seed}: {exc}", flush=True)
    if failures:
        raise RuntimeError("one or more sweep runs failed:\n" + "\n".join(failures))


def _label_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "safe": sum(int(row["label"]) == 0 for row in entries),
        "violation": sum(int(row["label"]) == 1 for row in entries),
    }


def _bootstrap_selection(
    pool: list[int], generator: np.random.Generator, enabled: bool
) -> set[int]:
    if not pool:
        return set()
    if not enabled:
        return set(pool)
    sampled = generator.integers(0, len(pool), size=len(pool))
    return {pool[int(index)] for index in sampled}


def extract_training_coverage(
    output: Path,
    config: dict[str, Any],
    spatial_hypothesis_id: str,
) -> dict[str, Any]:
    """Reconstruct query-buffer coverage for every spatial fit call.

    ``fit_ensemble`` creates a fresh deterministic NumPy generator for each
    member and call.  Replaying those integer draws exposes actual unique query
    coverage without reading model weights or private geometry.
    """

    query_log = load_json(output / "oracle_query_log.json")
    stages = load_json(output / "stage_diagnostics.json")
    if not isinstance(query_log, list) or not isinstance(stages, list):
        raise ValueError("query/stage diagnostics must be lists")
    trainable_indices = [
        index
        for index, row in enumerate(query_log)
        if str(row.get("source")) not in HOLDOUT_SOURCES
    ]
    trainable = [query_log[index] for index in trainable_indices]
    final_label_counts = _label_counts(trainable)
    source_counts = dict(sorted(Counter(str(row.get("source")) for row in trainable).items()))
    bootstrap = bool(config.get("trainer", {}).get("bootstrap_queries"))
    ensemble_size = int(config.get("model", {}).get("ensemble_size", 0))
    seed = int(config.get("seed", 0))
    selected_across_calls = [set() for _ in range(ensemble_size)]
    call_rows: list[dict[str, Any]] = []

    for stage in stages:
        stage_name = str(stage.get("stage", ""))
        if stage_name not in {"pre_query_fit", "post_query_refit"}:
            continue
        outer_round = int(stage.get("outer_round", -1))
        model_ids = [str(row.get("hypothesis_id")) for row in stage.get("models", [])]
        if spatial_hypothesis_id not in model_ids:
            continue
        hypothesis_index = model_ids.index(spatial_hypothesis_id)
        available: list[int] = []
        for index in trainable_indices:
            record_round = int(query_log[index].get("outer_round", -1))
            if record_round == 0:
                available.append(index)
            elif stage_name == "pre_query_fit" and record_round < outer_round:
                available.append(index)
            elif stage_name == "post_query_refit" and record_round <= outer_round:
                available.append(index)
        pools = {
            label: [index for index in available if int(query_log[index]["label"]) == label]
            for label in (0, 1)
        }
        fit_seed = seed + outer_round * 7919 + hypothesis_index * 101
        member_rows: list[dict[str, Any]] = []
        for member_index in range(ensemble_size):
            generator = np.random.default_rng(fit_seed + 1009 * member_index)
            selected_safe = _bootstrap_selection(pools[0], generator, bootstrap)
            selected_violation = _bootstrap_selection(pools[1], generator, bootstrap)
            selected = selected_safe | selected_violation
            selected_across_calls[member_index].update(selected)
            member_rows.append(
                {
                    "member_index": member_index,
                    "unique_count": len(selected),
                    "available_count": len(available),
                    "unique_fraction": len(selected) / len(available) if available else None,
                    "safe_unique_count": len(selected_safe),
                    "safe_available_count": len(pools[0]),
                    "safe_unique_fraction": (
                        len(selected_safe) / len(pools[0]) if pools[0] else None
                    ),
                    "violation_unique_count": len(selected_violation),
                    "violation_available_count": len(pools[1]),
                    "violation_unique_fraction": (
                        len(selected_violation) / len(pools[1]) if pools[1] else None
                    ),
                }
            )
        call_rows.append(
            {
                "outer_round": outer_round,
                "stage": stage_name,
                "hypothesis_index": hypothesis_index,
                "available_query_count": len(available),
                "available_label_counts": {
                    "safe": len(pools[0]),
                    "violation": len(pools[1]),
                },
                "members": member_rows,
            }
        )

    final_fit = call_rows[-1] if call_rows else None
    final_members = final_fit["members"] if final_fit else []
    cumulative_fractions = [
        len(selected) / len(trainable_indices) if trainable_indices else None
        for selected in selected_across_calls
    ]
    final_fractions = [
        float(row["unique_fraction"])
        for row in final_members
        if row["unique_fraction"] is not None
    ]
    safe_fractions = [
        float(row["safe_unique_fraction"])
        for row in final_members
        if row["safe_unique_fraction"] is not None
    ]
    violation_fractions = [
        float(row["violation_unique_fraction"])
        for row in final_members
        if row["violation_unique_fraction"] is not None
    ]
    return {
        "trainer_bootstrap_queries": bootstrap,
        "trainable_query_count": len(trainable),
        "trainable_safe_count": final_label_counts["safe"],
        "trainable_violation_count": final_label_counts["violation"],
        "trainable_source_counts": source_counts,
        "spatial_fit_call_count": len(call_rows),
        "spatial_final_fit_member_unique_fraction_mean": (
            statistics.mean(final_fractions) if final_fractions else None
        ),
        "spatial_final_fit_member_unique_fraction_min": (
            min(final_fractions) if final_fractions else None
        ),
        "spatial_final_fit_safe_unique_fraction_mean": (
            statistics.mean(safe_fractions) if safe_fractions else None
        ),
        "spatial_final_fit_violation_unique_fraction_mean": (
            statistics.mean(violation_fractions) if violation_fractions else None
        ),
        "spatial_cumulative_member_unique_fraction_mean": (
            statistics.mean(value for value in cumulative_fractions if value is not None)
            if any(value is not None for value in cumulative_fractions)
            else None
        ),
        "spatial_cumulative_member_unique_fraction_min": (
            min(value for value in cumulative_fractions if value is not None)
            if any(value is not None for value in cumulative_fractions)
            else None
        ),
        "spatial_fit_coverage_by_call": call_rows,
    }


def _last_spatial_evidence(
    history: list[dict[str, Any]], spatial_hypothesis_id: str
) -> tuple[dict[str, Any], int]:
    if not history:
        raise ValueError("evidence_history.json is empty")
    last = history[-1]
    matches = [
        row
        for row in last.get("hypotheses", [])
        if row.get("hypothesis_id") == spatial_hypothesis_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one final evidence row for {spatial_hypothesis_id}, got {len(matches)}"
        )
    return matches[0], int(last.get("outer_round", -1))


def extract_run(output: Path, arm: str, seed: int, plan: dict[str, Any]) -> dict[str, Any]:
    result = load_json(output / "result.json")
    config = load_yaml(output / "resolved_config.yaml")
    manifest = load_json(output / "implementation_manifest.json")
    frozen_bank = load_json(output / "frozen_bank_source.json")
    evidence_history = load_json(output / "evidence_history.json")
    all_evaluation = load_json(output / "all_hypothesis_evaluation.json")
    finalization = load_json(output / "finalization_diagnostics.json")
    spatial_id = str(plan["spatial_hypothesis_id"])
    spatial, evidence_round = _last_spatial_evidence(evidence_history, spatial_id)
    spatial_evaluation = all_evaluation.get(spatial_id)
    if not isinstance(spatial_evaluation, dict):
        raise ValueError(f"missing evaluation-only metrics for {spatial_id}")
    coverage = extract_training_coverage(output, config, spatial_id)
    warmup = extract_warmup_artifact(output)
    champion = str(result["champion_hypothesis_id"])
    status = str(result.get("selection_status", "inconclusive"))
    champion_eligible = bool(result.get("champion_eligible", False))
    return {
        "arm": arm,
        "seed": seed,
        "output": str(output.resolve()),
        "champion": champion,
        "selection_status": status,
        "champion_eligible": champion_eligible,
        "champion_ineligibility_reasons": result.get(
            "champion_ineligibility_reasons", []
        ),
        "qualified_spatial_champion": (
            champion == spatial_id and status == "qualified" and champion_eligible
        ),
        "spatial_evidence_outer_round": evidence_round,
        "spatial_evidence_eligible": bool(spatial.get("champion_eligible", False)),
        "spatial_evidence_ineligibility_reasons": spatial.get("ineligibility_reasons", []),
        "spatial_balanced_accuracy": float(spatial["balanced_accuracy"]),
        "spatial_safe_accuracy": float(spatial["safe_accuracy"]),
        "spatial_violation_recall": float(spatial["violation_recall"]),
        "spatial_expert_safe_rate": float(spatial["expert_safe_rate"]),
        "spatial_fit_expert_safe_rate": float(spatial["fit_expert_safe_rate"]),
        "spatial_selection_score": float(spatial["selection_score"]),
        "spatial_prequential_count": int(spatial["prequential_count"]),
        "spatial_mean_uncertainty": float(spatial["mean_uncertainty"]),
        "oracle_queries": int(result["oracle_queries"]),
        "llm_interactions": int(result["llm_interactions"]),
        "finalization_attempted": bool(finalization.get("attempted", False)),
        "finalization_applied": bool(finalization.get("applied", False)),
        "finalization_selected_candidate": finalization.get("selected_candidate"),
        "decision_threshold": float(result.get("decision_threshold", 0.0)),
        "champion_iou_evaluation_only": float(result["final_metrics"]["iou"]),
        "champion_false_safe_rate_evaluation_only": float(
            result["final_metrics"]["false_safe_rate"]
        ),
        "champion_false_unsafe_rate_evaluation_only": float(
            result["final_metrics"]["false_unsafe_rate"]
        ),
        "spatial_iou_evaluation_only": float(spatial_evaluation["iou"]),
        "spatial_false_safe_rate_evaluation_only": float(
            spatial_evaluation["false_safe_rate"]
        ),
        "spatial_false_unsafe_rate_evaluation_only": float(
            spatial_evaluation["false_unsafe_rate"]
        ),
        **coverage,
        **warmup,
        "config": config,
        "manifest": manifest,
        "frozen_bank": frozen_bank,
    }


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.pop("output_dir", None)
    normalized.get("trainer", {}).pop("bootstrap_queries", None)
    return normalized


def validate_artifacts(rows: list[dict[str, Any]], plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_count = 2 * len(plan["seeds"])
    if len(rows) != expected_count:
        errors.append(f"expected {expected_count} run rows, got {len(rows)}")
        return errors
    controls = plan["controls"]
    expected_oracle = int(controls["oracle_queries_per_run"])
    expected_llm = int(controls["llm_interactions_per_run"])
    expected_trainable = int(controls["trainable_query_records_per_run"])
    expected_warmup = int(controls["warmup_queries"])
    expected_warmup_labels = controls["benchmark_warmup_labels"]
    baseline_files = rows[0]["manifest"].get("files")
    baseline_inputs = shared_inputs(rows[0]["manifest"])
    baseline_bank = rows[0]["frozen_bank"].get("sha256")
    baseline_warmup_hashes = rows[0]["_warmup_trajectory_hashes"]
    baseline_warmup_roles = rows[0]["warmup_role_signature"]
    baseline_warmup_labels = rows[0]["warmup_label_sequence"]

    expected_pairs = {
        (arm, int(seed)) for seed in plan["seeds"] for arm in ARMS
    }
    actual_pairs = {(str(row["arm"]), int(row["seed"])) for row in rows}
    if actual_pairs != expected_pairs:
        errors.append("run arm/seed set differs from the preregistered paired design")

    safe_controls = controls["safe_acquisition"]
    for row in rows:
        prefix = f"{row['arm']} seed {row['seed']}"
        config = row["config"]
        if row["_warmup_artifact_errors"]:
            errors.extend(f"{prefix}: {item}" for item in row["_warmup_artifact_errors"])
        if int(row["oracle_queries"]) != expected_oracle:
            errors.append(f"{prefix}: expected {expected_oracle} Oracle queries")
        if int(row["llm_interactions"]) != expected_llm:
            errors.append(f"{prefix}: expected {expected_llm} LLM interactions")
        if int(row["trainable_query_count"]) != expected_trainable:
            errors.append(f"{prefix}: trainable query count mismatch")
        if int(row["warmup_query_count"]) != expected_warmup:
            errors.append(f"{prefix}: warmup count mismatch")
        if int(row["warmup_safe_count"]) != int(expected_warmup_labels["safe"]):
            errors.append(f"{prefix}: warmup safe-label count mismatch")
        if int(row["warmup_violation_count"]) != int(expected_warmup_labels["violation"]):
            errors.append(f"{prefix}: warmup violation-label count mismatch")
        if row["_warmup_trajectory_hashes"] != baseline_warmup_hashes:
            errors.append(f"{prefix}: warmup trajectory slate differs")
        if row["warmup_role_signature"] != baseline_warmup_roles:
            errors.append(f"{prefix}: warmup role split differs")
        if row["warmup_label_sequence"] != baseline_warmup_labels:
            errors.append(f"{prefix}: warmup label sequence differs")
        role_counts = row["warmup_role_label_counts"]
        for role, per_label in (
            ("warmup_validation", int(controls["warmup_validation_per_label"])),
            ("final_calibration", int(controls["final_calibration_per_label"])),
        ):
            if any(int(role_counts[role][label]) < per_label for label in ("safe", "violation")):
                errors.append(f"{prefix}: {role} quota mismatch")
        actual_bootstrap = bool(config.get("trainer", {}).get("bootstrap_queries"))
        expected_bootstrap = bool(plan["arms"][row["arm"]]["trainer_bootstrap_queries"])
        if actual_bootstrap is not expected_bootstrap:
            errors.append(f"{prefix}: trainer.bootstrap_queries mismatch")
        falsifier = config.get("falsifier", {})
        if not bool(falsifier.get("false_unsafe_use_hard_margin")):
            errors.append(f"{prefix}: hard-margin objective disabled")
        if float(falsifier.get("false_unsafe_trust_radius", 0.0)) != float(
            controls["false_unsafe_trust_radius"]
        ):
            errors.append(f"{prefix}: false-unsafe trust radius mismatch")
        if [float(value) for value in falsifier.get("false_unsafe_radius_ladder", [])] != [
            float(value) for value in controls["false_unsafe_radius_ladder"]
        ]:
            errors.append(f"{prefix}: false-unsafe radius ladder mismatch")
        loop = config.get("loop", {})
        for key, expected in safe_controls.items():
            actual = loop.get(key)
            if isinstance(expected, float):
                matches = float(actual) == float(expected)
            else:
                matches = actual == expected
            if not matches:
                errors.append(f"{prefix}: safe-acquisition control {key} mismatch")
        if str(config.get("device")) != str(controls["device"]):
            errors.append(f"{prefix}: device mismatch")
        if row["manifest"].get("files") != baseline_files:
            errors.append(f"{prefix}: code fingerprint mismatch")
        if shared_inputs(row["manifest"]) != baseline_inputs:
            errors.append(f"{prefix}: shared input fingerprint mismatch")
        if row["frozen_bank"].get("sha256") != baseline_bank:
            errors.append(f"{prefix}: frozen bank hash mismatch")
        runtime = row["manifest"].get("runtime", {})
        environment = runtime.get("environment", {})
        if int(runtime.get("seed", -1)) != int(row["seed"]):
            errors.append(f"{prefix}: manifest seed mismatch")
        if str(environment.get("PYTHONHASHSEED")) != str(row["seed"]):
            errors.append(f"{prefix}: PYTHONHASHSEED mismatch")
        if int(runtime.get("torch_num_threads", 0)) != int(controls["torch_num_threads"]):
            errors.append(f"{prefix}: torch thread count mismatch")
        if int(row["spatial_fit_call_count"]) != 2 * int(loop.get("outer_rounds", 0)):
            errors.append(f"{prefix}: spatial fit-call count mismatch")
        fractions = [
            row["spatial_final_fit_member_unique_fraction_mean"],
            row["spatial_final_fit_member_unique_fraction_min"],
            row["spatial_cumulative_member_unique_fraction_mean"],
            row["spatial_cumulative_member_unique_fraction_min"],
        ]
        if any(value is None or not 0.0 <= float(value) <= 1.0 for value in fractions):
            errors.append(f"{prefix}: invalid reconstructed coverage fraction")
        if row["arm"] == "full_buffer" and any(abs(float(value) - 1.0) > 1.0e-12 for value in fractions):
            errors.append(f"{prefix}: full-buffer arm did not cover every query")

    for seed in (int(value) for value in plan["seeds"]):
        bootstrap = next(
            row for row in rows if row["arm"] == "bootstrap_queries" and row["seed"] == seed
        )
        full = next(
            row for row in rows if row["arm"] == "full_buffer" and row["seed"] == seed
        )
        if normalized_config(bootstrap["config"]) != normalized_config(full["config"]):
            errors.append(
                f"seed {seed}: paired configs differ beyond output_dir/bootstrap_queries"
            )
    return errors


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"config", "manifest", "frozen_bank"}
        and not key.startswith("_")
        and key != "spatial_fit_coverage_by_call"
    }


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    champions = Counter(str(row["champion"]) for row in rows)
    return {
        "runs": len(rows),
        "qualified_spatial_champion_seed_count": sum(
            bool(row["qualified_spatial_champion"]) for row in rows
        ),
        "spatial_evidence_eligible_seed_count": sum(
            bool(row["spatial_evidence_eligible"]) for row in rows
        ),
        "champion_counts": dict(sorted(champions.items())),
        "spatial_balanced_accuracy": numeric_summary(
            [float(row["spatial_balanced_accuracy"]) for row in rows]
        ),
        "spatial_safe_accuracy": numeric_summary(
            [float(row["spatial_safe_accuracy"]) for row in rows]
        ),
        "spatial_violation_recall": numeric_summary(
            [float(row["spatial_violation_recall"]) for row in rows]
        ),
        "spatial_expert_safe_rate": numeric_summary(
            [float(row["spatial_expert_safe_rate"]) for row in rows]
        ),
        "spatial_fit_expert_safe_rate": numeric_summary(
            [float(row["spatial_fit_expert_safe_rate"]) for row in rows]
        ),
        "spatial_final_fit_member_unique_fraction": numeric_summary(
            [float(row["spatial_final_fit_member_unique_fraction_mean"]) for row in rows]
        ),
        "spatial_cumulative_member_unique_fraction": numeric_summary(
            [float(row["spatial_cumulative_member_unique_fraction_mean"]) for row in rows]
        ),
        "oracle_queries": numeric_summary([float(row["oracle_queries"]) for row in rows]),
        "champion_iou_evaluation_only": numeric_summary(
            [float(row["champion_iou_evaluation_only"]) for row in rows]
        ),
        "spatial_iou_evaluation_only": numeric_summary(
            [float(row["spatial_iou_evaluation_only"]) for row in rows]
        ),
    }


def paired_results(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    public_metrics = (
        "spatial_balanced_accuracy",
        "spatial_safe_accuracy",
        "spatial_violation_recall",
        "spatial_expert_safe_rate",
        "spatial_fit_expert_safe_rate",
    )
    for seed in seeds:
        bootstrap = next(
            row for row in rows if row["arm"] == "bootstrap_queries" and row["seed"] == seed
        )
        full = next(
            row for row in rows if row["arm"] == "full_buffer" and row["seed"] == seed
        )
        pair: dict[str, Any] = {
            "seed": seed,
            "bootstrap_champion": bootstrap["champion"],
            "full_buffer_champion": full["champion"],
            "bootstrap_status": bootstrap["selection_status"],
            "full_buffer_status": full["selection_status"],
            "bootstrap_qualified_spatial_champion": bootstrap[
                "qualified_spatial_champion"
            ],
            "full_buffer_qualified_spatial_champion": full[
                "qualified_spatial_champion"
            ],
            "delta_qualified_spatial_champion": int(
                bool(full["qualified_spatial_champion"])
            )
            - int(bool(bootstrap["qualified_spatial_champion"])),
            "bootstrap_spatial_evidence_eligible": bootstrap["spatial_evidence_eligible"],
            "full_buffer_spatial_evidence_eligible": full["spatial_evidence_eligible"],
            "bootstrap_final_fit_coverage": bootstrap[
                "spatial_final_fit_member_unique_fraction_mean"
            ],
            "full_buffer_final_fit_coverage": full[
                "spatial_final_fit_member_unique_fraction_mean"
            ],
            "bootstrap_oracle_queries": bootstrap["oracle_queries"],
            "full_buffer_oracle_queries": full["oracle_queries"],
            "bootstrap_spatial_iou_evaluation_only": bootstrap[
                "spatial_iou_evaluation_only"
            ],
            "full_buffer_spatial_iou_evaluation_only": full[
                "spatial_iou_evaluation_only"
            ],
            "delta_spatial_iou_evaluation_only": full["spatial_iou_evaluation_only"]
            - bootstrap["spatial_iou_evaluation_only"],
        }
        for metric in public_metrics:
            pair[f"bootstrap_{metric}"] = bootstrap[metric]
            pair[f"full_buffer_{metric}"] = full[metric]
            pair[f"delta_{metric}"] = float(full[metric]) - float(bootstrap[metric])
        pairs.append(pair)
    return pairs


def apply_decision_rule(
    pairs: list[dict[str, Any]],
    arm_summaries: dict[str, dict[str, Any]],
    rule: dict[str, Any],
) -> dict[str, Any]:
    bootstrap_count = int(
        arm_summaries["bootstrap_queries"]["qualified_spatial_champion_seed_count"]
    )
    full_count = int(
        arm_summaries["full_buffer"]["qualified_spatial_champion_seed_count"]
    )
    strict_gain = full_count > bootstrap_count
    floors = {key: float(value) for key, value in rule["public_safety_floors"].items()}
    delta_mins = {
        key: float(value)
        for key, value in rule["obvious_public_safety_degradation"][
            "paired_delta_min"
        ].items()
    }
    floor_regressions: list[dict[str, Any]] = []
    large_drops: list[dict[str, Any]] = []
    for pair in pairs:
        for metric, floor in floors.items():
            before = float(pair[f"bootstrap_{metric}"])
            after = float(pair[f"full_buffer_{metric}"])
            if before >= floor and after < floor:
                floor_regressions.append(
                    {"seed": pair["seed"], "metric": metric, "before": before, "after": after}
                )
        for metric, minimum_delta in delta_mins.items():
            delta = float(pair[f"delta_{metric}"])
            if delta < minimum_delta:
                large_drops.append(
                    {"seed": pair["seed"], "metric": metric, "delta": delta}
                )
    use_floor_regressions = bool(
        rule["obvious_public_safety_degradation"].get(
            "any_pass_to_fail_safety_floor_regression_is_degradation", True
        )
    )
    obvious_degradation = bool(large_drops) or (
        use_floor_regressions and bool(floor_regressions)
    )
    select_full = strict_gain and not obvious_degradation
    return {
        "bootstrap_qualified_spatial_champion_seed_count": bootstrap_count,
        "full_buffer_qualified_spatial_champion_seed_count": full_count,
        "strict_qualified_spatial_champion_seed_gain": strict_gain,
        "public_safety_floor_regressions": floor_regressions,
        "public_safety_large_paired_drops": large_drops,
        "obvious_public_safety_degradation": obvious_degradation,
        "private_geometry_used_for_decision": False,
        "selected_default": (
            str(rule["preferred_if_supported"])
            if select_full
            else str(rule["fallback"])
        ),
        "reason": (
            "full_buffer strictly increased qualified spatial champions without a "
            "public-evidence safety degradation"
            if select_full
            else (
                "retain bootstrap_queries because full_buffer did not strictly increase "
                "qualified spatial champions"
                if not strict_gain
                else "retain bootstrap_queries because a public-evidence safety guard failed"
            )
        ),
    }


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def write_outputs(
    output_root: Path,
    plan_path: Path,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    validation_errors: list[str],
) -> dict[str, Any]:
    public_rows = [public_row(row) for row in rows]
    arm_summaries = {
        arm: summarize_arm([row for row in rows if row["arm"] == arm])
        for arm in ARMS
    }
    pairs = paired_results(rows, [int(value) for value in plan["seeds"]])
    decision = apply_decision_rule(pairs, arm_summaries, plan["decision_rule"])
    if validation_errors:
        decision["selected_default"] = None
        decision["reason"] = "artifact validation failed; no method selection is valid"
    summary = {
        "experiment_id": plan["experiment_id"],
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256(plan_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "artifact_validation_passed": not validation_errors,
        "artifact_validation_errors": validation_errors,
        "private_geometry_is_diagnostic_only": True,
        "rows": public_rows,
        "arm_summaries": arm_summaries,
        "paired_results": pairs,
        "decision": decision,
    }
    (output_root / "numeric_fitting_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if public_rows:
        fieldnames = list(public_rows[0])
        with (output_root / "numeric_fitting_per_run_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(public_rows)

    report = [
        "# Numeric fitting：bootstrap queries vs full buffer（5-seed 配对实验）",
        "",
        f"计划：`{plan_path.name}`（SHA256 `{sha256(plan_path)}`）。",
        "",
        "IoU、false-safe 和 false-unsafe 均为 private-geometry 诊断，不参与选择。",
        "",
        "## 每 seed 结果",
        "",
        "| seed | bootstrap champion/status | full champion/status | spatial qualified B/F | spatial public S/V B→F | audit/fit-safe B→F | final query coverage B→F | spatial IoU B→F* |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        report.append(
            "| {seed} | {bc}/{bs} | {fc}/{fs} | {bq}/{fq} | "
            "{bsa}/{bvr}→{fsa}/{fvr} | {bea}/{bfe}→{fea}/{ffe} | "
            "{bco}→{fco} | {biou}→{fiou} |".format(
                seed=pair["seed"],
                bc=pair["bootstrap_champion"],
                bs=pair["bootstrap_status"],
                fc=pair["full_buffer_champion"],
                fs=pair["full_buffer_status"],
                bq=int(bool(pair["bootstrap_qualified_spatial_champion"])),
                fq=int(bool(pair["full_buffer_qualified_spatial_champion"])),
                bsa=_format(pair["bootstrap_spatial_safe_accuracy"]),
                bvr=_format(pair["bootstrap_spatial_violation_recall"]),
                fsa=_format(pair["full_buffer_spatial_safe_accuracy"]),
                fvr=_format(pair["full_buffer_spatial_violation_recall"]),
                bea=_format(pair["bootstrap_spatial_expert_safe_rate"]),
                bfe=_format(pair["bootstrap_spatial_fit_expert_safe_rate"]),
                fea=_format(pair["full_buffer_spatial_expert_safe_rate"]),
                ffe=_format(pair["full_buffer_spatial_fit_expert_safe_rate"]),
                bco=_format(pair["bootstrap_final_fit_coverage"]),
                fco=_format(pair["full_buffer_final_fit_coverage"]),
                biou=_format(pair["bootstrap_spatial_iou_evaluation_only"]),
                fiou=_format(pair["full_buffer_spatial_iou_evaluation_only"]),
            )
        )
    report.extend(
        [
            "",
            "\* evaluation-only；不进入 decision rule。",
            "",
            "## 汇总与预注册决策",
            "",
            "| arm | qualified spatial champions | spatial eligible | median spatial BA | median violation recall | mean final-fit coverage | mean spatial IoU* |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        item = arm_summaries[arm]
        report.append(
            "| {arm} | {q} | {e} | {ba} | {vr} | {coverage} | {iou} |".format(
                arm=arm,
                q=item["qualified_spatial_champion_seed_count"],
                e=item["spatial_evidence_eligible_seed_count"],
                ba=_format(item["spatial_balanced_accuracy"]["median"]),
                vr=_format(item["spatial_violation_recall"]["median"]),
                coverage=_format(
                    item["spatial_final_fit_member_unique_fraction"]["mean"]
                ),
                iou=_format(item["spatial_iou_evaluation_only"]["mean"]),
            )
        )
    report.extend(
        [
            "",
            f"- artifact validation: `{'PASS' if not validation_errors else 'FAIL'}`",
            f"- strict qualified-spatial gain: `{decision['strict_qualified_spatial_champion_seed_gain']}`",
            f"- obvious public safety degradation: `{decision['obvious_public_safety_degradation']}`",
            f"- private geometry used for decision: `{decision['private_geometry_used_for_decision']}`",
            f"- **selected default: `{decision['selected_default']}`**",
            "",
            str(decision["reason"]),
            "",
            "五个 seeds 仅支持工程稳定性判断，不构成统计显著性证明。",
        ]
    )
    if validation_errors:
        report.extend(("", "## Artifact validation errors", ""))
        report.extend(f"- {error}" for error in validation_errors)
    (output_root / "NUMERIC_FITTING_MULTISEED_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def summarize(plan_path: Path, plan: dict[str, Any], output_root: Path) -> dict[str, Any]:
    rows = [
        extract_run(run_directory(output_root, arm, int(seed)), arm, int(seed), plan)
        for seed in plan["seeds"]
        for arm in ARMS
    ]
    validation_errors = validate_artifacts(rows, plan)
    resolved_plan = output_root / "experiment_plan_resolved.yaml"
    if not resolved_plan.exists():
        validation_errors.append("missing experiment_plan_resolved.yaml")
    elif load_yaml(resolved_plan) != plan:
        validation_errors.append("current plan differs from the plan frozen at launch")
    summary = write_outputs(output_root, plan_path, plan, rows, validation_errors)
    if validation_errors:
        raise RuntimeError("artifact validation failed:\n" + "\n".join(validation_errors))
    return summary


def main() -> int:
    args = parse_args()
    plan_path = repository_path(args.plan).resolve()
    output_root = repository_path(args.output_root).resolve()
    plan = load_yaml(plan_path)
    validate_plan(plan)
    commands = validate_commands(plan, output_root)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "plan": str(plan_path),
                    "plan_sha256": sha256(plan_path),
                    "runner_sha256": sha256(Path(__file__).resolve()),
                    "run_count": len(commands),
                    "commands": commands,
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    if not args.summarize_only:
        launch_runs(plan, output_root, args.max_workers)
    summary = summarize(plan_path, plan, output_root)
    print(json.dumps(summary["decision"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
