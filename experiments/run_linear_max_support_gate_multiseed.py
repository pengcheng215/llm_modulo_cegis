"""Run the preregistered audit-only versus enforced support-gate sweep.

Both arms compute the same public convex-support certificate for the affine
linear-max hypothesis.  The sole treatment is whether a triggered certificate
is allowed to remove that hypothesis from champion eligibility.  Private
geometry is retained only as an evaluation diagnostic and is never passed to
the adoption rule.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import run_numeric_fitting_multiseed as numeric_runner
import run_violation_pooling_multiseed as pooling_runner
from run_falsifier_multiseed import canonical_array_hash, sha256, shared_inputs


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_PLAN = PACKAGE_ROOT / "configs" / "linear_max_support_gate_multiseed_plan.yaml"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs" / "linear_max_support_gate_multiseed_5seed_v1"
ARMS = ("audit_only", "enforced")
BASELINE_ARM = "audit_only"
CANDIDATE_ARM = "enforced"
EXPECTED_SEEDS = (1007, 1019, 1037, 1073, 1109)
GATE_FLAGS = {
    BASELINE_ARM: "--audit-only-linear-max-support-gate",
    CANDIDATE_ARM: "--enforce-linear-max-support-gate",
}
EFFECTIVE_SUPPORT_TOLERANCE_DEFAULT = 1.0e-5
EFFECTIVE_MINIMUM_ANCHORS_DEFAULT = 2

# This exact projection, rather than a blacklist, is the only per-run payload
# accepted by ``apply_decision_rule``.  In particular it contains no IoU,
# false-safe/false-unsafe grid metric, obstacle geometry, or model weights.
DECISION_INPUT_ALLOWLIST = (
    "arm",
    "seed",
    "champion",
    "selection_status",
    "champion_eligible",
    "qualified_spatial_champion",
    "spatial_evidence_eligible",
    "champion_safe_accuracy",
    "champion_violation_recall",
    "champion_expert_safe_rate",
    "champion_fit_expert_safe_rate",
    "affine_linear_max_support_pair_count",
    "affine_linear_max_support_contradiction_count",
    "affine_linear_max_support_distinct_anchor_count",
    "affine_linear_max_support_unresolved_pair_count",
    "affine_linear_max_support_gate_triggered",
    "affine_linear_max_support_gate_applied",
)


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
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Summarize an existing complete 10-run artifact tree without launching runs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the plan and all 10 commands without creating output directories.",
    )
    return parser.parse_args()


def run_directory(output_root: Path, arm: str, seed: int) -> Path:
    return output_root / arm / f"seed_{seed}"


def _effective_evidence_value(
    evidence: dict[str, Any], key: str, default: float | int | bool
) -> float | int | bool:
    value = evidence.get(key, default)
    return value


def validate_plan(plan: dict[str, Any]) -> None:
    if int(plan.get("schema_version", -1)) != 1:
        raise ValueError("linear-max support-gate plan schema_version must be 1")
    seeds = tuple(int(value) for value in plan.get("seeds", []))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"seeds must be exactly {list(EXPECTED_SEEDS)}")
    if set(plan.get("arms", {})) != set(ARMS):
        raise ValueError(f"plan arms must be exactly {list(ARMS)}")
    for arm, expected in ((BASELINE_ARM, False), (CANDIDATE_ARM, True)):
        configured = plan["arms"][arm]
        actual = configured.get("linear_max_support_gate_enforced")
        if not isinstance(actual, bool) or actual is not expected:
            raise ValueError(
                f"{arm}.linear_max_support_gate_enforced must be {expected}"
            )
        if str(configured.get("cli_flag")) != GATE_FLAGS[arm]:
            raise ValueError(f"{arm}.cli_flag must be {GATE_FLAGS[arm]}")

    controls = plan.get("controls", {})
    if str(controls.get("device")) != "cpu":
        raise ValueError("the preregistered sweep must use CPU")
    if str(controls.get("semantic_backend")) != "frozen_bank":
        raise ValueError("the preregistered sweep must use a frozen semantic bank")
    if not bool(controls.get("freeze_revisions")):
        raise ValueError("semantic revisions must remain frozen")
    if bool(controls.get("trainer_bootstrap_queries")):
        raise ValueError("both arms must use the complete trainable query buffer")
    if str(controls.get("violation_pooling_mode")) != "source_anchor_changed_states":
        raise ValueError("both arms must use source-anchor changed-state pooling")
    if float(controls.get("violation_pooling_change_tolerance", -1.0)) != 1.0e-6:
        raise ValueError("violation-pooling change tolerance must be 1e-6")
    if str(controls.get("falsifier_objective")) != "hard_margin":
        raise ValueError("both arms must use the hard-margin falsifier")
    if float(controls.get("false_unsafe_trust_radius", 0.0)) != 0.32:
        raise ValueError("both arms must use hard single-0.32")
    if list(controls.get("false_unsafe_radius_ladder", [])):
        raise ValueError("the radius ladder must be empty")
    if float(controls.get("linear_max_support_tolerance", -1.0)) != 1.0e-5:
        raise ValueError("linear-max support tolerance must be 1e-5")
    if int(controls.get("linear_max_minimum_contradictory_anchors", -1)) != 2:
        raise ValueError("linear-max support gate must require two anchors")
    if int(controls.get("oracle_queries_per_run", -1)) != 26:
        raise ValueError("each run must use exactly 26 Oracle queries")
    if int(controls.get("llm_interactions_per_run", -1)) != 0:
        raise ValueError("the frozen-bank experiment must have zero LLM interactions")
    if int(controls.get("trainable_query_records_per_run", -1)) != 18:
        raise ValueError("each run must expose 18 trainable query records")
    if not bool(controls.get("paired_full_oracle_slate_must_match")):
        raise ValueError("paired full Oracle slates must be required to match")

    rule = plan.get("decision_rule", {})
    if str(rule.get("candidate")) != CANDIDATE_ARM:
        raise ValueError(f"decision candidate must be {CANDIDATE_ARM}")
    if str(rule.get("baseline")) != BASELINE_ARM:
        raise ValueError(f"decision baseline must be {BASELINE_ARM}")
    if str(rule.get("fallback")) != BASELINE_ARM:
        raise ValueError(f"decision fallback must be {BASELINE_ARM}")
    if int(rule.get("minimum_candidate_spatial_evidence_eligible_seeds", -1)) != 5:
        raise ValueError("candidate must have spatial evidence eligible in 5/5 seeds")
    if int(rule.get("minimum_candidate_qualified_spatial_champions", -1)) != 4:
        raise ValueError("candidate must have at least four qualified spatial champions")
    required_true = (
        "require_candidate_qualified_count_at_least_baseline",
        "require_gate_triggered_in_all_runs",
        "require_candidate_gate_applied_in_all_runs",
        "require_baseline_gate_not_applied_in_all_runs",
        "reject_candidate_certified_affine_champion",
        "paired_rescue_requires_qualified_spatial_champion",
        "require_all_candidate_final_champions_qualified",
        "require_finite_unit_interval_public_rates",
        "reject_on_any_paired_threshold_crossing_regression",
    )
    for key in required_true:
        if not bool(rule.get(key)):
            raise ValueError(f"decision_rule.{key} must be true")
    if int(rule.get("minimum_distinct_contradictory_anchors", -1)) != 2:
        raise ValueError("decision rule must require two distinct contradictory anchors")
    expected_floors = {
        "champion_safe_accuracy": 0.60,
        "champion_violation_recall": 0.55,
        "champion_expert_safe_rate": 0.90,
        "champion_fit_expert_safe_rate": 0.90,
    }
    actual_floors = {
        str(key): float(value)
        for key, value in rule.get("public_champion_floors", {}).items()
    }
    if actual_floors != expected_floors:
        raise ValueError(f"public champion floors must be {expected_floors}")
    allowlist = tuple(str(value) for value in rule.get("decision_input_allowlist", []))
    if allowlist != DECISION_INPUT_ALLOWLIST:
        raise ValueError("decision input allowlist differs from the runner's exact projection")
    private_markers = ("iou", "false_safe", "false_unsafe", "ground_truth", "geometry")
    if any(any(marker in field.lower() for marker in private_markers) for field in allowlist):
        raise ValueError("decision input allowlist contains a private diagnostic")

    base_config = repository_path(str(plan["base_config"]))
    frozen_bank = repository_path(str(plan["frozen_hypothesis_bank"]))
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    if not frozen_bank.is_file():
        raise FileNotFoundError(frozen_bank)
    if str(plan.get("affine_hypothesis_id")) != "h_affine_spatial":
        raise ValueError("the preregistered affine hypothesis must be h_affine_spatial")
    base = load_yaml(base_config)
    trainer = base.get("trainer", {})
    evidence = base.get("evidence", {})
    if bool(trainer.get("bootstrap_queries")):
        raise ValueError("base config unexpectedly enables query bootstrap")
    if str(trainer.get("violation_pooling_mode")) != str(
        controls["violation_pooling_mode"]
    ):
        raise ValueError("base config violation-pooling mode differs from the plan")
    if float(trainer.get("violation_pooling_change_tolerance", -1.0)) != float(
        controls["violation_pooling_change_tolerance"]
    ):
        raise ValueError("base config violation-pooling tolerance differs from the plan")
    effective_tolerance = float(
        _effective_evidence_value(
            evidence,
            "linear_max_support_tolerance",
            EFFECTIVE_SUPPORT_TOLERANCE_DEFAULT,
        )
    )
    effective_minimum = int(
        _effective_evidence_value(
            evidence,
            "linear_max_minimum_contradictory_anchors",
            EFFECTIVE_MINIMUM_ANCHORS_DEFAULT,
        )
    )
    if effective_tolerance != float(controls["linear_max_support_tolerance"]):
        raise ValueError("effective base support tolerance differs from the plan")
    if effective_minimum != int(controls["linear_max_minimum_contradictory_anchors"]):
        raise ValueError("effective base minimum-anchor count differs from the plan")


def build_command(
    plan: dict[str, Any], arm: str, seed: int, output: Path
) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    return [
        sys.executable,
        str(PACKAGE_ROOT / "run_obstacle_avoid.py"),
        "--config",
        str(repository_path(str(plan["base_config"])).resolve()),
        "--initial-hypothesis-bank",
        str(repository_path(str(plan["frozen_hypothesis_bank"])).resolve()),
        "--freeze-revisions",
        "--false-unsafe-single-radius",
        str(float(plan["controls"]["false_unsafe_trust_radius"])),
        "--no-bootstrap-queries",
        "--violation-pooling-mode",
        str(plan["controls"]["violation_pooling_mode"]),
        GATE_FLAGS[arm],
        "--seed",
        str(seed),
        "--output",
        str(output.resolve()),
    ]


def _command_control_signature(command: list[str]) -> tuple[str, ...]:
    ignored_with_value = {"--seed", "--output"}
    ignored_flags = set(GATE_FLAGS.values())
    signature: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in ignored_flags:
            index += 1
            continue
        if token in ignored_with_value:
            index += 2
            continue
        signature.append(token)
        index += 1
    return tuple(signature)


def validate_commands(plan: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in (int(value) for value in plan["seeds"]):
        commands: list[list[str]] = []
        for arm in ARMS:
            command = build_command(plan, arm, seed, run_directory(output_root, arm, seed))
            gate_flags = [token for token in command if token in set(GATE_FLAGS.values())]
            if gate_flags != [GATE_FLAGS[arm]]:
                raise AssertionError(f"{arm} seed={seed}: wrong or ambiguous gate flag")
            if command.count("--no-bootstrap-queries") != 1:
                raise AssertionError(f"{arm} seed={seed}: full-buffer flag missing")
            if command.count("--violation-pooling-mode") != 1:
                raise AssertionError(f"{arm} seed={seed}: pooling flag missing")
            pooling_index = command.index("--violation-pooling-mode")
            if command[pooling_index + 1] != "source_anchor_changed_states":
                raise AssertionError(f"{arm} seed={seed}: wrong pooling mode")
            rows.append({"arm": arm, "seed": seed, "command": command})
            commands.append(command)
        if len({_command_control_signature(command) for command in commands}) != 1:
            raise AssertionError(
                f"seed={seed}: commands differ beyond seed/output/gate treatment"
            )
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
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-20:])
        raise RuntimeError(f"{arm} seed={seed} failed ({completed.returncode})\n{tail}")
    return metadata


def launch_runs(plan: dict[str, Any], output_root: Path, max_workers: int) -> None:
    if max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    # A formal comparison gets a new root, not merely new leaf directories.
    if output_root.exists():
        raise FileExistsError(f"formal output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "experiment_plan_resolved.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    tasks = [(arm, int(seed)) for seed in plan["seeds"] for arm in ARMS]
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


def _last_evidence_row(
    history: list[dict[str, Any]], hypothesis_id: str
) -> tuple[dict[str, Any], int]:
    if not history:
        raise ValueError("evidence_history.json is empty")
    final = history[-1]
    matches = [
        item
        for item in final.get("hypotheses", [])
        if str(item.get("hypothesis_id")) == hypothesis_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one final evidence row for {hypothesis_id}, got {len(matches)}"
        )
    return matches[0], int(final.get("outer_round", -1))


def extract_full_oracle_slate_hash(output: Path) -> dict[str, Any]:
    """Hash the complete ordered public query slate, not only warmup.

    The hash covers observations, actions, Oracle trajectory labels, and outer
    rounds in archive order.  Array headers (shape/dtype) are part of each
    component hash, and component boundaries are length-prefixed.
    """

    with np.load(output / "oracle_queries.npz", allow_pickle=False) as archive:
        required = ("observations", "actions", "labels", "outer_rounds")
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(f"oracle archive missing arrays: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in required}
    lengths = {int(value.shape[0]) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(
            "oracle archive arrays have unequal leading dimensions: "
            + ", ".join(f"{key}={value.shape[0]}" for key, value in arrays.items())
        )
    digest = hashlib.sha256()
    component_hashes: dict[str, str] = {}
    for key in required:
        component = canonical_array_hash(arrays[key])
        component_hashes[key] = component
        encoded_key = key.encode("ascii")
        digest.update(len(encoded_key).to_bytes(8, "little"))
        digest.update(encoded_key)
        digest.update(bytes.fromhex(component))
    return {
        "full_oracle_slate_sha256": digest.hexdigest(),
        "full_oracle_slate_query_count": next(iter(lengths)),
        "_full_oracle_slate_component_hashes": component_hashes,
    }


def extract_run(
    output: Path, arm: str, seed: int, plan: dict[str, Any]
) -> dict[str, Any]:
    row = pooling_runner.extract_run(output, arm, seed, plan)
    history = load_json(output / "evidence_history.json")
    query_log = load_json(output / "oracle_query_log.json")
    if not isinstance(history, list):
        raise ValueError("evidence_history.json must be a list")
    if not isinstance(query_log, list):
        raise ValueError("oracle_query_log.json must be a list")
    affine_id = str(plan["affine_hypothesis_id"])
    affine, affine_round = _last_evidence_row(history, affine_id)
    champion, champion_round = _last_evidence_row(history, str(row["champion"]))
    required_gate_fields = (
        "linear_max_support_pair_count",
        "linear_max_support_contradiction_count",
        "linear_max_support_distinct_anchor_count",
        "linear_max_support_unresolved_pair_count",
        "linear_max_support_gate_triggered",
        "linear_max_support_gate_applied",
    )
    missing = [key for key in required_gate_fields if key not in affine]
    if missing:
        raise ValueError(f"affine final evidence lacks support-gate fields: {missing}")
    row.update(
        {
            "qualified_final_champion": bool(
                row["selection_status"] == "qualified" and row["champion_eligible"]
            ),
            "champion_evidence_outer_round": champion_round,
            "champion_safe_accuracy": float(champion["safe_accuracy"]),
            "champion_violation_recall": float(champion["violation_recall"]),
            "champion_expert_safe_rate": float(champion["expert_safe_rate"]),
            "champion_fit_expert_safe_rate": float(champion["fit_expert_safe_rate"]),
            "champion_balanced_accuracy": float(champion["balanced_accuracy"]),
            "champion_selection_score": float(champion["selection_score"]),
            "affine_evidence_outer_round": affine_round,
            "affine_evidence_eligible": bool(affine["champion_eligible"]),
            "affine_ineligibility_reasons": list(affine["ineligibility_reasons"]),
            "affine_safe_accuracy": float(affine["safe_accuracy"]),
            "affine_violation_recall": float(affine["violation_recall"]),
            "affine_expert_safe_rate": float(affine["expert_safe_rate"]),
            "affine_fit_expert_safe_rate": float(affine["fit_expert_safe_rate"]),
            "affine_linear_max_support_pair_count": int(
                affine["linear_max_support_pair_count"]
            ),
            "affine_linear_max_support_contradiction_count": int(
                affine["linear_max_support_contradiction_count"]
            ),
            "affine_linear_max_support_distinct_anchor_count": int(
                affine["linear_max_support_distinct_anchor_count"]
            ),
            "affine_linear_max_support_unresolved_pair_count": int(
                affine["linear_max_support_unresolved_pair_count"]
            ),
            "affine_linear_max_support_gate_triggered": bool(
                affine["linear_max_support_gate_triggered"]
            ),
            "affine_linear_max_support_gate_applied": bool(
                affine["linear_max_support_gate_applied"]
            ),
            "certified_affine_champion": bool(
                str(row["champion"]) == affine_id
                and affine["linear_max_support_gate_triggered"]
            ),
            "oracle_query_log_count": len(query_log),
            "sweep_metadata": load_json(output / "sweep_run_metadata.json"),
            **extract_full_oracle_slate_hash(output),
        }
    )
    return row


def _normalized_treatment_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.pop("output_dir", None)
    normalized.get("evidence", {}).pop("linear_max_support_gate_enforced", None)
    return normalized


def _has_exact_gate_flag(arguments: list[Any], arm: str) -> bool:
    flags = [str(value) for value in arguments if str(value) in set(GATE_FLAGS.values())]
    return flags == [GATE_FLAGS[arm]]


def validate_artifacts(rows: list[dict[str, Any]], plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_count = len(ARMS) * len(plan["seeds"])
    if len(rows) != expected_count:
        return [f"expected {expected_count} run rows, got {len(rows)}"]
    expected_pairs = {
        (arm, int(seed)) for seed in plan["seeds"] for arm in ARMS
    }
    actual_pairs = {(str(row["arm"]), int(row["seed"])) for row in rows}
    if actual_pairs != expected_pairs:
        errors.append("run arm/seed set differs from the preregistered paired design")

    controls = plan["controls"]
    expected_warmup_labels = controls["benchmark_warmup_labels"]
    safe_controls = controls["safe_acquisition"]
    baseline_files = rows[0]["manifest"].get("files")
    baseline_inputs = shared_inputs(rows[0]["manifest"])
    baseline_bank = rows[0]["frozen_bank"].get("sha256")
    baseline_warmup_hashes = rows[0]["_warmup_trajectory_hashes"]
    baseline_warmup_roles = rows[0]["warmup_role_signature"]
    baseline_warmup_labels = rows[0]["warmup_label_sequence"]

    for row in rows:
        arm = str(row["arm"])
        prefix = f"{arm} seed {row['seed']}"
        config = row["config"]
        trainer = config.get("trainer", {})
        evidence = config.get("evidence", {})
        falsifier = config.get("falsifier", {})
        loop = config.get("loop", {})
        expected_enforced = bool(
            plan["arms"][arm]["linear_max_support_gate_enforced"]
        )
        if row.get("_warmup_artifact_errors"):
            errors.extend(f"{prefix}: {item}" for item in row["_warmup_artifact_errors"])
        if int(config.get("seed", -1)) != int(row["seed"]):
            errors.append(f"{prefix}: resolved seed mismatch")
        if int(row["oracle_queries"]) != int(controls["oracle_queries_per_run"]):
            errors.append(f"{prefix}: Oracle query count mismatch")
        if int(row["full_oracle_slate_query_count"]) != int(
            controls["oracle_queries_per_run"]
        ):
            errors.append(f"{prefix}: Oracle archive query count mismatch")
        if int(row["full_oracle_slate_query_count"]) != int(row["oracle_queries"]):
            errors.append(f"{prefix}: result/archive Oracle query counts disagree")
        if int(row["oracle_query_log_count"]) != int(
            controls["oracle_queries_per_run"]
        ):
            errors.append(f"{prefix}: Oracle query-log length mismatch")
        if int(row["full_oracle_slate_query_count"]) != int(
            controls["oracle_queries_per_run"]
        ):
            errors.append(f"{prefix}: Oracle archive length mismatch")
        if not (
            int(row["oracle_queries"])
            == int(row["oracle_query_log_count"])
            == int(row["full_oracle_slate_query_count"])
        ):
            errors.append(
                f"{prefix}: result/query-log/archive Oracle counts are not identical"
            )
        if int(row["llm_interactions"]) != int(controls["llm_interactions_per_run"]):
            errors.append(f"{prefix}: LLM interaction count mismatch")
        if int(row["trainable_query_count"]) != int(
            controls["trainable_query_records_per_run"]
        ):
            errors.append(f"{prefix}: trainable query count mismatch")
        if int(row["warmup_query_count"]) != int(controls["warmup_queries"]):
            errors.append(f"{prefix}: warmup query count mismatch")
        if int(row["warmup_safe_count"]) != int(expected_warmup_labels["safe"]):
            errors.append(f"{prefix}: warmup safe-label count mismatch")
        if int(row["warmup_violation_count"]) != int(
            expected_warmup_labels["violation"]
        ):
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
            if any(
                int(role_counts[role][label]) < per_label
                for label in ("safe", "violation")
            ):
                errors.append(f"{prefix}: {role} quota mismatch")

        for metric in (
            "champion_safe_accuracy",
            "champion_violation_recall",
            "champion_expert_safe_rate",
            "champion_fit_expert_safe_rate",
        ):
            value = float(row[metric])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                errors.append(f"{prefix}: {metric} is not a finite unit-interval rate")

        if bool(trainer.get("bootstrap_queries")):
            errors.append(f"{prefix}: query bootstrap unexpectedly enabled")
        if bool(trainer.get("bootstrap_ensure_full_coverage", False)):
            errors.append(f"{prefix}: coverage-preserving bootstrap unexpectedly enabled")
        if str(trainer.get("violation_pooling_mode")) != str(
            controls["violation_pooling_mode"]
        ):
            errors.append(f"{prefix}: violation-pooling mode mismatch")
        if float(trainer.get("violation_pooling_change_tolerance", -1.0)) != float(
            controls["violation_pooling_change_tolerance"]
        ):
            errors.append(f"{prefix}: violation-pooling tolerance mismatch")
        actual_enforced = bool(evidence.get("linear_max_support_gate_enforced", True))
        if actual_enforced is not expected_enforced:
            errors.append(f"{prefix}: support-gate treatment mismatch")
        effective_tolerance = float(
            evidence.get(
                "linear_max_support_tolerance",
                EFFECTIVE_SUPPORT_TOLERANCE_DEFAULT,
            )
        )
        effective_minimum = int(
            evidence.get(
                "linear_max_minimum_contradictory_anchors",
                EFFECTIVE_MINIMUM_ANCHORS_DEFAULT,
            )
        )
        if effective_tolerance != float(controls["linear_max_support_tolerance"]):
            errors.append(f"{prefix}: effective support tolerance mismatch")
        if effective_minimum != int(
            controls["linear_max_minimum_contradictory_anchors"]
        ):
            errors.append(f"{prefix}: effective minimum-anchor count mismatch")
        pair_count = int(row["affine_linear_max_support_pair_count"])
        contradiction_count = int(
            row["affine_linear_max_support_contradiction_count"]
        )
        distinct_count = int(row["affine_linear_max_support_distinct_anchor_count"])
        unresolved_count = int(row["affine_linear_max_support_unresolved_pair_count"])
        if min(pair_count, contradiction_count, distinct_count, unresolved_count) < 0:
            errors.append(f"{prefix}: negative support diagnostic count")
        if contradiction_count > pair_count or distinct_count > contradiction_count:
            errors.append(f"{prefix}: inconsistent support diagnostic counts")
        expected_triggered = distinct_count >= effective_minimum
        if bool(row["affine_linear_max_support_gate_triggered"]) is not expected_triggered:
            errors.append(f"{prefix}: trigger disagrees with distinct-anchor count")
        expected_applied = expected_enforced and expected_triggered
        if bool(row["affine_linear_max_support_gate_applied"]) is not expected_applied:
            errors.append(f"{prefix}: applied flag disagrees with treatment and trigger")

        if not bool(falsifier.get("false_unsafe_use_hard_margin")):
            errors.append(f"{prefix}: hard-margin falsifier disabled")
        if float(falsifier.get("false_unsafe_trust_radius", 0.0)) != float(
            controls["false_unsafe_trust_radius"]
        ):
            errors.append(f"{prefix}: false-unsafe trust radius mismatch")
        if [float(value) for value in falsifier.get("false_unsafe_radius_ladder", [])]:
            errors.append(f"{prefix}: false-unsafe radius ladder is not empty")
        for key, expected in safe_controls.items():
            actual = loop.get(key)
            matches = float(actual) == float(expected) if isinstance(expected, float) else actual == expected
            if not matches:
                errors.append(f"{prefix}: safe-acquisition control {key} mismatch")
        if str(config.get("device")) != str(controls["device"]):
            errors.append(f"{prefix}: device mismatch")
        if str(config.get("semantic_reasoner", {}).get("backend")) != "frozen_bank":
            errors.append(f"{prefix}: semantic backend is not frozen_bank")
        if not bool(loop.get("freeze_revisions")):
            errors.append(f"{prefix}: semantic revisions are not frozen")

        if row["manifest"].get("files") != baseline_files:
            errors.append(f"{prefix}: code fingerprint differs")
        if shared_inputs(row["manifest"]) != baseline_inputs:
            errors.append(f"{prefix}: shared input fingerprint differs")
        if row["frozen_bank"].get("sha256") != baseline_bank:
            errors.append(f"{prefix}: frozen-bank hash differs")
        runtime = row["manifest"].get("runtime", {})
        environment = runtime.get("environment", {})
        if int(runtime.get("seed", -1)) != int(row["seed"]):
            errors.append(f"{prefix}: manifest seed mismatch")
        if str(environment.get("PYTHONHASHSEED")) != str(row["seed"]):
            errors.append(f"{prefix}: PYTHONHASHSEED mismatch")
        if int(runtime.get("torch_num_threads", 0)) != int(controls["torch_num_threads"]):
            errors.append(f"{prefix}: torch thread count mismatch")
        for variable, control_key in (
            ("OMP_NUM_THREADS", "omp_num_threads"),
            ("MKL_NUM_THREADS", "mkl_num_threads"),
        ):
            if str(environment.get(variable)) != str(controls[control_key]):
                errors.append(f"{prefix}: {variable} mismatch")
        argv = list(runtime.get("argv", []))
        if not _has_exact_gate_flag(argv, arm):
            errors.append(f"{prefix}: manifest argv does not isolate the gate treatment")
        if "--no-bootstrap-queries" not in argv:
            errors.append(f"{prefix}: manifest argv lacks full-buffer flag")
        metadata = row["sweep_metadata"]
        if (
            str(metadata.get("arm")) != arm
            or int(metadata.get("seed", -1)) != int(row["seed"])
            or int(metadata.get("returncode", -1)) != 0
        ):
            errors.append(f"{prefix}: invalid sweep metadata")
        if not _has_exact_gate_flag(list(metadata.get("command", [])), arm):
            errors.append(f"{prefix}: sweep command does not isolate the gate treatment")

        expected_fit_calls = 2 * int(loop.get("outer_rounds", 0))
        if int(row["spatial_fit_call_count"]) != expected_fit_calls:
            errors.append(f"{prefix}: spatial fit-call count mismatch")
        fractions = (
            row["spatial_final_fit_member_unique_fraction_mean"],
            row["spatial_final_fit_member_unique_fraction_min"],
            row["spatial_cumulative_member_unique_fraction_mean"],
            row["spatial_cumulative_member_unique_fraction_min"],
        )
        if any(value is None or abs(float(value) - 1.0) > 1.0e-12 for value in fractions):
            errors.append(f"{prefix}: full query-buffer coverage not observed")
        if int(row["spatial_pooling_fit_call_count"]) != int(
            row["spatial_fit_call_count"]
        ):
            errors.append(f"{prefix}: source-mask diagnostics missed a spatial fit")
        if int(row["spatial_source_anchor_masked_violation_total"]) <= 0:
            errors.append(f"{prefix}: source-anchor mask was never active")
        if int(row["spatial_unique_source_anchor_masked_violation_total"]) <= 0:
            errors.append(f"{prefix}: source-anchor mask has no unique record")
        if row["spatial_final_source_anchor_masked_violation_counts"] != row[
            "spatial_final_unique_source_anchor_masked_violation_counts"
        ]:
            errors.append(f"{prefix}: final full-buffer mask counts disagree")

    for seed in (int(value) for value in plan["seeds"]):
        per_arm = {
            str(row["arm"]): row for row in rows if int(row["seed"]) == seed
        }
        normalized = {
            json.dumps(
                _normalized_treatment_config(per_arm[arm]["config"]),
                sort_keys=True,
            )
            for arm in ARMS
        }
        if len(normalized) != 1:
            errors.append(
                f"seed {seed}: paired configs differ beyond output/gate enforcement"
            )
        for key in (
            "oracle_queries",
            "llm_interactions",
            "trainable_query_count",
            "warmup_query_count",
            "warmup_safe_count",
            "warmup_violation_count",
            "spatial_fit_call_count",
        ):
            if per_arm[BASELINE_ARM][key] != per_arm[CANDIDATE_ARM][key]:
                errors.append(f"seed {seed}: paired artifact field {key} differs")
        for key in (
            "_warmup_trajectory_hashes",
            "warmup_role_signature",
            "warmup_label_sequence",
            "warmup_role_label_counts",
        ):
            if per_arm[BASELINE_ARM][key] != per_arm[CANDIDATE_ARM][key]:
                errors.append(f"seed {seed}: paired warmup field {key} differs")
        for key in (
            "affine_linear_max_support_pair_count",
            "affine_linear_max_support_contradiction_count",
            "affine_linear_max_support_distinct_anchor_count",
            "affine_linear_max_support_unresolved_pair_count",
            "affine_linear_max_support_gate_triggered",
        ):
            if per_arm[BASELINE_ARM][key] != per_arm[CANDIDATE_ARM][key]:
                errors.append(f"seed {seed}: paired certificate field {key} differs")
        if bool(controls["paired_full_oracle_slate_must_match"]):
            if per_arm[BASELINE_ARM]["full_oracle_slate_sha256"] != per_arm[
                CANDIDATE_ARM
            ]["full_oracle_slate_sha256"]:
                errors.append(
                    f"seed {seed}: paired full Oracle observation/action/label/round "
                    "slates differ"
                )
            if per_arm[BASELINE_ARM][
                "_full_oracle_slate_component_hashes"
            ] != per_arm[CANDIDATE_ARM]["_full_oracle_slate_component_hashes"]:
                errors.append(f"seed {seed}: paired full Oracle component hashes differ")
    return errors


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "config",
        "manifest",
        "frozen_bank",
        "sweep_metadata",
        "spatial_fit_coverage_by_call",
        "spatial_pooling_activity_by_call",
    }
    return {
        key: value
        for key, value in row.items()
        if key not in excluded and not key.startswith("_")
    }


def decision_input(row: dict[str, Any]) -> dict[str, Any]:
    projected = {field: row[field] for field in DECISION_INPUT_ALLOWLIST}
    if tuple(projected) != DECISION_INPUT_ALLOWLIST:
        raise AssertionError("decision projection order/fields changed")
    return projected


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = numeric_runner.summarize_arm(rows)
    summary.update(
        {
            "qualified_final_champion_seed_count": sum(
                bool(row["qualified_final_champion"]) for row in rows
            ),
            "certified_affine_champion_seed_count": sum(
                bool(row["certified_affine_champion"]) for row in rows
            ),
            "affine_gate_triggered_seed_count": sum(
                bool(row["affine_linear_max_support_gate_triggered"])
                for row in rows
            ),
            "affine_gate_applied_seed_count": sum(
                bool(row["affine_linear_max_support_gate_applied"])
                for row in rows
            ),
            "affine_distinct_anchor_count": numeric_runner.numeric_summary(
                [
                    float(row["affine_linear_max_support_distinct_anchor_count"])
                    for row in rows
                ]
            ),
            "champion_counts": dict(
                sorted(Counter(str(row["champion"]) for row in rows).items())
            ),
            "champion_safe_accuracy": numeric_runner.numeric_summary(
                [float(row["champion_safe_accuracy"]) for row in rows]
            ),
            "champion_violation_recall": numeric_runner.numeric_summary(
                [float(row["champion_violation_recall"]) for row in rows]
            ),
            "champion_expert_safe_rate": numeric_runner.numeric_summary(
                [float(row["champion_expert_safe_rate"]) for row in rows]
            ),
            "champion_fit_expert_safe_rate": numeric_runner.numeric_summary(
                [float(row["champion_fit_expert_safe_rate"]) for row in rows]
            ),
        }
    )
    return summary


def paired_results(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    public_metrics = (
        "champion_safe_accuracy",
        "champion_violation_recall",
        "champion_expert_safe_rate",
        "champion_fit_expert_safe_rate",
    )
    pairs: list[dict[str, Any]] = []
    for seed in seeds:
        per_arm = {
            str(row["arm"]): row for row in rows if int(row["seed"]) == seed
        }
        baseline = per_arm[BASELINE_ARM]
        candidate = per_arm[CANDIDATE_ARM]
        pair: dict[str, Any] = {
            "seed": seed,
            "audit_only_champion": baseline["champion"],
            "enforced_champion": candidate["champion"],
            "audit_only_status": baseline["selection_status"],
            "enforced_status": candidate["selection_status"],
            "audit_only_qualified_spatial_champion": baseline[
                "qualified_spatial_champion"
            ],
            "enforced_qualified_spatial_champion": candidate[
                "qualified_spatial_champion"
            ],
            "audit_only_spatial_evidence_eligible": baseline[
                "spatial_evidence_eligible"
            ],
            "enforced_spatial_evidence_eligible": candidate[
                "spatial_evidence_eligible"
            ],
            "audit_only_affine_distinct_anchors": baseline[
                "affine_linear_max_support_distinct_anchor_count"
            ],
            "enforced_affine_distinct_anchors": candidate[
                "affine_linear_max_support_distinct_anchor_count"
            ],
            "audit_only_affine_gate_triggered": baseline[
                "affine_linear_max_support_gate_triggered"
            ],
            "enforced_affine_gate_triggered": candidate[
                "affine_linear_max_support_gate_triggered"
            ],
            "audit_only_affine_gate_applied": baseline[
                "affine_linear_max_support_gate_applied"
            ],
            "enforced_affine_gate_applied": candidate[
                "affine_linear_max_support_gate_applied"
            ],
            "audit_only_certified_affine_champion": baseline[
                "certified_affine_champion"
            ],
            "enforced_certified_affine_champion": candidate[
                "certified_affine_champion"
            ],
            "paired_affine_rescue": bool(
                baseline["certified_affine_champion"]
                and candidate["qualified_spatial_champion"]
            ),
            "audit_only_champion_iou_evaluation_only": baseline[
                "champion_iou_evaluation_only"
            ],
            "enforced_champion_iou_evaluation_only": candidate[
                "champion_iou_evaluation_only"
            ],
            "audit_only_spatial_iou_evaluation_only": baseline[
                "spatial_iou_evaluation_only"
            ],
            "enforced_spatial_iou_evaluation_only": candidate[
                "spatial_iou_evaluation_only"
            ],
        }
        for metric in public_metrics:
            pair[f"audit_only_{metric}"] = baseline[metric]
            pair[f"enforced_{metric}"] = candidate[metric]
            pair[f"delta_{metric}"] = float(candidate[metric]) - float(
                baseline[metric]
            )
        pairs.append(pair)
    return pairs


def apply_decision_rule(
    inputs: list[dict[str, Any]],
    rule: dict[str, Any],
    artifact_validation_passed: bool,
) -> dict[str, Any]:
    if len(inputs) != len(ARMS) * len(EXPECTED_SEEDS):
        raise ValueError("decision rule requires exactly 10 projected inputs")
    if any(tuple(row) != DECISION_INPUT_ALLOWLIST for row in inputs):
        raise ValueError("decision rule received a field outside its public allowlist")
    per_arm = {
        arm: [row for row in inputs if str(row["arm"]) == arm]
        for arm in ARMS
    }
    if any(len(rows) != len(EXPECTED_SEEDS) for rows in per_arm.values()):
        raise ValueError("decision rule requires five rows per arm")
    baseline_by_seed = {
        int(row["seed"]): row for row in per_arm[BASELINE_ARM]
    }
    candidate_by_seed = {
        int(row["seed"]): row for row in per_arm[CANDIDATE_ARM]
    }
    if set(baseline_by_seed) != set(EXPECTED_SEEDS) or set(candidate_by_seed) != set(
        EXPECTED_SEEDS
    ):
        raise ValueError("decision input seeds differ from the preregistered seeds")

    baseline_count = sum(
        bool(row["qualified_spatial_champion"]) for row in per_arm[BASELINE_ARM]
    )
    candidate_count = sum(
        bool(row["qualified_spatial_champion"]) for row in per_arm[CANDIDATE_ARM]
    )
    candidate_spatial_eligible_count = sum(
        bool(row["spatial_evidence_eligible"]) for row in per_arm[CANDIDATE_ARM]
    )
    eligible_target = int(rule["minimum_candidate_spatial_evidence_eligible_seeds"])
    qualified_target = int(rule["minimum_candidate_qualified_spatial_champions"])
    spatial_eligibility_met = candidate_spatial_eligible_count >= eligible_target
    qualified_minimum_met = candidate_count >= qualified_target
    qualified_not_worse = candidate_count >= baseline_count

    minimum_anchors = int(rule["minimum_distinct_contradictory_anchors"])
    candidate_gate_failures: list[dict[str, Any]] = []
    baseline_gate_failures: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        candidate = candidate_by_seed[seed]
        baseline = baseline_by_seed[seed]
        if not (
            bool(candidate["affine_linear_max_support_gate_triggered"])
            and int(candidate["affine_linear_max_support_distinct_anchor_count"])
            >= minimum_anchors
            and bool(candidate["affine_linear_max_support_gate_applied"])
        ):
            candidate_gate_failures.append(
                {
                    "seed": seed,
                    "triggered": candidate[
                        "affine_linear_max_support_gate_triggered"
                    ],
                    "distinct_anchors": candidate[
                        "affine_linear_max_support_distinct_anchor_count"
                    ],
                    "applied": candidate["affine_linear_max_support_gate_applied"],
                }
            )
        if not (
            bool(baseline["affine_linear_max_support_gate_triggered"])
            and int(baseline["affine_linear_max_support_distinct_anchor_count"])
            >= minimum_anchors
            and not bool(baseline["affine_linear_max_support_gate_applied"])
        ):
            baseline_gate_failures.append(
                {
                    "seed": seed,
                    "triggered": baseline[
                        "affine_linear_max_support_gate_triggered"
                    ],
                    "distinct_anchors": baseline[
                        "affine_linear_max_support_distinct_anchor_count"
                    ],
                    "applied": baseline["affine_linear_max_support_gate_applied"],
                }
            )
    gate_behavior_met = not candidate_gate_failures and not baseline_gate_failures

    # The preregistered plan validator pins this structural control id.
    affine_id = "h_affine_spatial"
    candidate_certified_affine_seeds = [
        int(row["seed"])
        for row in per_arm[CANDIDATE_ARM]
        if str(row["champion"]) == affine_id
        and bool(row["affine_linear_max_support_gate_triggered"])
    ]
    baseline_certified_affine_seeds = [
        int(row["seed"])
        for row in per_arm[BASELINE_ARM]
        if str(row["champion"]) == affine_id
        and bool(row["affine_linear_max_support_gate_triggered"])
    ]
    failed_rescue_seeds = [
        seed
        for seed in baseline_certified_affine_seeds
        if not bool(candidate_by_seed[seed]["qualified_spatial_champion"])
    ]
    no_candidate_certified_affine = not candidate_certified_affine_seeds
    paired_rescue_met = not failed_rescue_seeds

    unqualified_candidate_seeds = [
        int(row["seed"])
        for row in per_arm[CANDIDATE_ARM]
        if not (
            str(row["selection_status"]) == "qualified"
            and bool(row["champion_eligible"])
        )
    ]
    all_candidate_champions_qualified = not unqualified_candidate_seeds
    floors = {
        str(key): float(value)
        for key, value in rule["public_champion_floors"].items()
    }
    candidate_floor_failures: list[dict[str, Any]] = []
    paired_threshold_crossing_regressions: list[dict[str, Any]] = []
    invalid_public_rate_inputs: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        baseline = baseline_by_seed[seed]
        candidate = candidate_by_seed[seed]
        for metric, floor in floors.items():
            before = float(baseline[metric])
            after = float(candidate[metric])
            before_valid = math.isfinite(before) and 0.0 <= before <= 1.0
            after_valid = math.isfinite(after) and 0.0 <= after <= 1.0
            if not before_valid:
                invalid_public_rate_inputs.append(
                    {"arm": BASELINE_ARM, "seed": seed, "metric": metric, "value": before}
                )
            if not after_valid:
                invalid_public_rate_inputs.append(
                    {"arm": CANDIDATE_ARM, "seed": seed, "metric": metric, "value": after}
                )
            if not after_valid or after < floor:
                candidate_floor_failures.append(
                    {"seed": seed, "metric": metric, "floor": floor, "value": after}
                )
            if before_valid and before >= floor and (not after_valid or after < floor):
                paired_threshold_crossing_regressions.append(
                    {
                        "seed": seed,
                        "metric": metric,
                        "floor": floor,
                        "before": before,
                        "after": after,
                    }
                )
    public_floors_met = not candidate_floor_failures
    public_rate_inputs_valid = not invalid_public_rate_inputs
    no_threshold_crossing_regression = not paired_threshold_crossing_regressions

    conditions = {
        "all_10_artifacts_and_treatment_isolation_passed": bool(
            artifact_validation_passed
        ),
        "candidate_spatial_evidence_eligible_5_of_5": spatial_eligibility_met,
        "candidate_at_least_4_qualified_spatial_champions": qualified_minimum_met,
        "candidate_qualified_spatial_count_at_least_baseline": qualified_not_worse,
        "both_arms_have_required_trigger_apply_behavior": gate_behavior_met,
        "candidate_never_selects_certified_affine": no_candidate_certified_affine,
        "all_baseline_certified_affine_selections_paired_rescued": paired_rescue_met,
        "all_candidate_final_champions_qualified": all_candidate_champions_qualified,
        "all_candidate_champions_meet_public_floors": public_floors_met,
        "all_public_rate_inputs_are_finite_unit_interval": public_rate_inputs_valid,
        "no_paired_public_threshold_crossing_regression": no_threshold_crossing_regression,
    }
    selected = all(conditions.values())
    failed = [name for name, passed in conditions.items() if not passed]
    strict_gain = candidate_count > baseline_count
    if not artifact_validation_passed:
        interpretation = "no method selection is valid because artifact validation failed"
    elif selected and strict_gain:
        interpretation = (
            "adopt the sound guard; qualified-spatial count is higher in this engineering sweep"
        )
    elif selected:
        interpretation = (
            "adopt the sound guard at tied qualified-spatial count; do not claim a performance improvement"
        )
    else:
        interpretation = "retain audit-only because one or more preregistered guards failed"
    return {
        "decision_input_allowlist": list(DECISION_INPUT_ALLOWLIST),
        "private_geometry_used_for_decision": False,
        "artifact_validation_passed": bool(artifact_validation_passed),
        "audit_only_qualified_spatial_champion_seed_count": baseline_count,
        "enforced_qualified_spatial_champion_seed_count": candidate_count,
        "enforced_spatial_evidence_eligible_seed_count": candidate_spatial_eligible_count,
        "candidate_strict_qualified_spatial_count_gain": strict_gain,
        "candidate_gate_failures": candidate_gate_failures,
        "baseline_gate_failures": baseline_gate_failures,
        "candidate_certified_affine_champion_seeds": candidate_certified_affine_seeds,
        "baseline_certified_affine_champion_seeds": baseline_certified_affine_seeds,
        "failed_paired_rescue_seeds": failed_rescue_seeds,
        "unqualified_candidate_champion_seeds": unqualified_candidate_seeds,
        "candidate_public_floor_failures": candidate_floor_failures,
        "invalid_public_rate_inputs": invalid_public_rate_inputs,
        "paired_public_threshold_crossing_regressions": paired_threshold_crossing_regressions,
        "conditions": conditions,
        "candidate_selected": selected,
        "selected_default": (
            CANDIDATE_ARM
            if selected
            else (str(rule["fallback"]) if artifact_validation_passed else None)
        ),
        "performance_improvement_claim_supported": bool(selected and strict_gain),
        "interpretation": interpretation,
        "reason": "all preregistered guards passed" if selected else "; ".join(failed),
    }


def write_outputs(
    output_root: Path,
    plan_path: Path,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    validation_errors: list[str],
) -> dict[str, Any]:
    public_rows = [public_row(row) for row in rows]
    decision_inputs = [decision_input(row) for row in rows]
    arm_summaries = {
        arm: summarize_arm([row for row in rows if row["arm"] == arm])
        for arm in ARMS
    }
    pairs = paired_results(rows, [int(value) for value in plan["seeds"]])
    decision = apply_decision_rule(
        decision_inputs,
        plan["decision_rule"],
        artifact_validation_passed=not validation_errors,
    )
    summary = {
        "experiment_id": plan["experiment_id"],
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256(plan_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "artifact_validation_passed": not validation_errors,
        "artifact_validation_errors": validation_errors,
        "treatment_isolation": (
            "resolved configs differ only in output_dir and "
            "evidence.linear_max_support_gate_enforced"
        ),
        "decision_input_allowlist": list(DECISION_INPUT_ALLOWLIST),
        "decision_inputs_public": decision_inputs,
        "private_geometry_is_diagnostic_only": True,
        "rows": public_rows,
        "arm_summaries": arm_summaries,
        "paired_results": pairs,
        "decision": decision,
    }
    (output_root / "linear_max_support_gate_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if public_rows:
        fieldnames = list(public_rows[0])
        with (output_root / "linear_max_support_gate_per_run_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(public_rows)

    report = [
        "# Linear-max support gate: audit-only vs enforced (paired five-seed)",
        "",
        f"Plan: `{plan_path.name}` (SHA256 `{sha256(plan_path)}`).",
        "",
        "The adoption rule receives only the explicit public-field allowlist. "
        "Private IoU is an evaluation-only diagnostic.",
        "",
        "| arm | spatial eligible | qualified spatial champion | qualified final champion | affine trigger | affine applied | certified-affine champion | median champion S/V | mean spatial IoU* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = arm_summaries[arm]
        report.append(
            "| {arm} | {eligible}/5 | {spatial}/5 | {qualified}/5 | "
            "{trigger}/5 | {applied}/5 | {affine}/5 | {safe:.3f}/{violation:.3f} | "
            "{iou:.3f} |".format(
                arm=arm,
                eligible=item["spatial_evidence_eligible_seed_count"],
                spatial=item["qualified_spatial_champion_seed_count"],
                qualified=item["qualified_final_champion_seed_count"],
                trigger=item["affine_gate_triggered_seed_count"],
                applied=item["affine_gate_applied_seed_count"],
                affine=item["certified_affine_champion_seed_count"],
                safe=float(item["champion_safe_accuracy"]["median"]),
                violation=float(item["champion_violation_recall"]["median"]),
                iou=float(item["spatial_iou_evaluation_only"]["mean"]),
            )
        )
    report.extend(
        [
            "",
            "\\* evaluation-only private geometry.",
            "",
            "## Preregistered decision",
            "",
            f"- Artifact/treatment validation: `{'PASS' if not validation_errors else 'FAIL'}`",
        ]
    )
    report.extend(
        f"- {name}: `{passed}`"
        for name, passed in decision["conditions"].items()
    )
    report.extend(
        [
            f"- Private geometry used for decision: `{decision['private_geometry_used_for_decision']}`",
            f"- Selected default: `{decision['selected_default']}`",
            f"- Performance-improvement claim supported: `{decision['performance_improvement_claim_supported']}`",
            "",
            str(decision["interpretation"]),
            "",
            "Five seeds assess engineering stability; they are not a claim of statistical significance.",
        ]
    )
    if validation_errors:
        report.extend(("", "## Artifact validation errors", ""))
        report.extend(f"- {error}" for error in validation_errors)
    (output_root / "LINEAR_MAX_SUPPORT_GATE_MULTISEED_REPORT.md").write_text(
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
                    "fresh_output_root": str(output_root),
                    "treatment_matrix": {
                        arm: {
                            "evidence.linear_max_support_gate_enforced": bool(
                                plan["arms"][arm][
                                    "linear_max_support_gate_enforced"
                                ]
                            ),
                            "cli_flag": GATE_FLAGS[arm],
                            "trainer.bootstrap_queries": False,
                            "trainer.violation_pooling_mode": plan["controls"][
                                "violation_pooling_mode"
                            ],
                        }
                        for arm in ARMS
                    },
                    "decision_input_allowlist": list(DECISION_INPUT_ALLOWLIST),
                    "private_geometry_used_for_decision": False,
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
