"""Run the preregistered three-arm violation-pooling numeric sweep.

The treatment matrix separates query-buffer coverage from trajectory-level
violation credit assignment:

* classic_all_states: classic query bootstrap and ordinary all-state MIL;
* full_all_states: complete query buffer and ordinary all-state MIL;
* full_source_anchor_mask: complete buffer and source-anchor changed-state MIL.

All selection decisions use public trajectory membership evidence.  Private
geometry metrics are emitted only as post-selection diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

import run_numeric_fitting_multiseed as numeric_runner
from run_falsifier_multiseed import sha256, shared_inputs


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_PLAN = PACKAGE_ROOT / "configs" / "violation_pooling_multiseed_plan.yaml"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs" / "violation_pooling_multiseed_5seed"
ARMS = (
    "classic_all_states",
    "full_all_states",
    "full_source_anchor_mask",
)
BASELINE_ARMS = ("classic_all_states", "full_all_states")
CANDIDATE_ARM = "full_source_anchor_mask"
FIT_STAGES = {"pre_query_fit", "post_query_refit"}


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
        help="Validate the plan and all 15 commands without launching runs.",
    )
    return parser.parse_args()


def run_directory(output_root: Path, arm: str, seed: int) -> Path:
    return output_root / arm / f"seed_{seed}"


def validate_plan(plan: dict[str, Any]) -> None:
    if int(plan.get("schema_version", -1)) != 1:
        raise ValueError("violation-pooling plan schema_version must be 1")
    seeds = [int(value) for value in plan.get("seeds", [])]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("the paired experiment requires five distinct numeric seeds")
    if set(plan.get("arms", {})) != set(ARMS):
        raise ValueError(f"plan arms must be exactly {list(ARMS)}")
    expected = {
        "classic_all_states": (True, "all_states"),
        "full_all_states": (False, "all_states"),
        "full_source_anchor_mask": (False, "source_anchor_changed_states"),
    }
    for arm, (bootstrap, pooling) in expected.items():
        configured = plan["arms"][arm]
        if configured.get("trainer_bootstrap_queries") is not bootstrap:
            raise ValueError(
                f"{arm}.trainer_bootstrap_queries must be {bootstrap}"
            )
        if str(configured.get("violation_pooling_mode")) != pooling:
            raise ValueError(f"{arm}.violation_pooling_mode must be {pooling}")

    controls = plan.get("controls", {})
    if str(controls.get("device")) != "cpu":
        raise ValueError("the preregistered sweep must use CPU")
    if str(controls.get("semantic_backend")) != "frozen_bank":
        raise ValueError("the preregistered sweep must use a frozen semantic bank")
    if not bool(controls.get("freeze_revisions")):
        raise ValueError("semantic revisions must remain frozen")
    if str(controls.get("falsifier_objective")) != "hard_margin":
        raise ValueError("all arms must use the hard-margin falsifier")
    if float(controls.get("false_unsafe_trust_radius", 0.0)) != 0.32:
        raise ValueError("all arms must use hard single-0.32")
    if list(controls.get("false_unsafe_radius_ladder", [])):
        raise ValueError("the radius ladder must be empty")
    if float(controls.get("violation_pooling_change_tolerance", -1.0)) != 1.0e-6:
        raise ValueError("source-anchor change tolerance must be fixed at 1e-6")

    rule = plan.get("decision_rule", {})
    if str(rule.get("candidate")) != CANDIDATE_ARM:
        raise ValueError(f"decision candidate must be {CANDIDATE_ARM}")
    if tuple(rule.get("baselines", [])) != BASELINE_ARMS:
        raise ValueError(f"decision baselines must be {list(BASELINE_ARMS)}")
    if str(rule.get("fallback")) != "classic_all_states":
        raise ValueError("decision fallback must be classic_all_states")
    if int(rule.get("minimum_candidate_qualified_spatial_champions", -1)) != 4:
        raise ValueError("candidate minimum must be four qualified seeds")
    if not bool(rule.get("require_candidate_strictly_exceeds_each_baseline")):
        raise ValueError("candidate must strictly exceed each baseline")
    if not bool(rule.get("reject_on_any_public_safety_floor_regression")):
        raise ValueError("any paired public safety-floor regression must reject")

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
    treatment = plan["arms"][arm]
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
        "--violation-pooling-mode",
        str(treatment["violation_pooling_mode"]),
        # Preserve the historical pooling experiment: the later structural
        # support gate is audited but cannot change eligibility.
        "--audit-only-linear-max-support-gate",
        "--seed",
        str(seed),
        "--output",
        str(output.resolve()),
    ]
    command.append(
        "--bootstrap-queries"
        if bool(treatment["trainer_bootstrap_queries"])
        else "--no-bootstrap-queries"
    )
    return command


def _command_control_signature(command: list[str]) -> tuple[str, ...]:
    """Remove seed/output and the two preregistered treatment switches."""

    ignored_with_value = {
        "--seed",
        "--output",
        "--violation-pooling-mode",
    }
    ignored_flags = {"--bootstrap-queries", "--no-bootstrap-queries"}
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
        seed_commands: list[list[str]] = []
        for arm in ARMS:
            command = build_command(
                plan, arm, seed, run_directory(output_root, arm, seed)
            )
            bootstrap_flags = [
                token
                for token in command
                if token in {"--bootstrap-queries", "--no-bootstrap-queries"}
            ]
            if len(bootstrap_flags) != 1:
                raise AssertionError(f"{arm} seed={seed}: expected one bootstrap flag")
            expected_bootstrap = (
                "--bootstrap-queries"
                if bool(plan["arms"][arm]["trainer_bootstrap_queries"])
                else "--no-bootstrap-queries"
            )
            if bootstrap_flags[0] != expected_bootstrap:
                raise AssertionError(f"{arm} seed={seed}: wrong bootstrap flag")
            pooling_indices = [
                index
                for index, token in enumerate(command)
                if token == "--violation-pooling-mode"
            ]
            if len(pooling_indices) != 1:
                raise AssertionError(f"{arm} seed={seed}: expected one pooling flag")
            actual_pooling = command[pooling_indices[0] + 1]
            if actual_pooling != str(plan["arms"][arm]["violation_pooling_mode"]):
                raise AssertionError(f"{arm} seed={seed}: wrong pooling mode")
            rows.append({"arm": arm, "seed": seed, "command": command})
            seed_commands.append(command)
        signatures = {_command_control_signature(command) for command in seed_commands}
        if len(signatures) != 1:
            raise AssertionError(
                f"seed={seed}: commands differ beyond output/seed/treatment switches"
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
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
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


def extract_pooling_activity(output: Path, spatial_id: str) -> dict[str, Any]:
    stages = load_json(output / "stage_diagnostics.json")
    if not isinstance(stages, list):
        raise ValueError("stage_diagnostics.json must be a list")
    calls: list[dict[str, Any]] = []
    for stage in stages:
        if str(stage.get("stage", "")) not in FIT_STAGES:
            continue
        matches = [
            row
            for row in stage.get("models", [])
            if str(row.get("hypothesis_id")) == spatial_id
        ]
        if len(matches) != 1:
            continue
        coverage = matches[0].get("training_query_coverage", {})
        calls.append(
            {
                "outer_round": int(stage.get("outer_round", -1)),
                "stage": str(stage.get("stage")),
                "member_source_anchor_masked_violation_counts": list(
                    coverage.get(
                        "member_source_anchor_masked_violation_counts", []
                    )
                ),
                "member_unique_source_anchor_masked_violation_counts": list(
                    coverage.get(
                        "member_unique_source_anchor_masked_violation_counts", []
                    )
                ),
                "member_source_anchor_unresolved_violation_counts": list(
                    coverage.get(
                        "member_source_anchor_unresolved_violation_counts", []
                    )
                ),
                "member_representation_invariant_violation_counts": list(
                    coverage.get(
                        "member_representation_invariant_violation_counts", []
                    )
                ),
            }
        )

    def total(key: str) -> int:
        return sum(int(value) for call in calls for value in call[key])

    final = calls[-1] if calls else None
    return {
        "spatial_pooling_fit_call_count": len(calls),
        "spatial_source_anchor_masked_violation_total": total(
            "member_source_anchor_masked_violation_counts"
        ),
        "spatial_unique_source_anchor_masked_violation_total": total(
            "member_unique_source_anchor_masked_violation_counts"
        ),
        "spatial_source_anchor_unresolved_violation_total": total(
            "member_source_anchor_unresolved_violation_counts"
        ),
        "spatial_representation_invariant_violation_total": total(
            "member_representation_invariant_violation_counts"
        ),
        "spatial_final_source_anchor_masked_violation_counts": (
            final["member_source_anchor_masked_violation_counts"] if final else []
        ),
        "spatial_final_unique_source_anchor_masked_violation_counts": (
            final["member_unique_source_anchor_masked_violation_counts"]
            if final
            else []
        ),
        "spatial_pooling_activity_by_call": calls,
    }


def extract_run(output: Path, arm: str, seed: int, plan: dict[str, Any]) -> dict[str, Any]:
    row = numeric_runner.extract_run(output, arm, seed, plan)
    row.update(extract_pooling_activity(output, str(plan["spatial_hypothesis_id"])))
    return row


def _baseline_validation_rows_and_plan(
    rows: list[dict[str, Any]], plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aliases = {
        "classic_all_states": "bootstrap_queries",
        "full_all_states": "full_buffer",
    }
    baseline_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["arm"] not in aliases:
            continue
        copied = copy.copy(row)
        copied["arm"] = aliases[str(row["arm"])]
        baseline_rows.append(copied)
    baseline_plan = copy.deepcopy(plan)
    baseline_plan["arms"] = {
        "bootstrap_queries": {"trainer_bootstrap_queries": True},
        "full_buffer": {"trainer_bootstrap_queries": False},
    }
    return baseline_rows, baseline_plan


def _normalized_treatment_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.pop("output_dir", None)
    trainer = normalized.get("trainer", {})
    trainer.pop("bootstrap_queries", None)
    trainer.pop("violation_pooling_mode", None)
    return normalized


def _validate_candidate_row(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    plan: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    prefix = f"{candidate['arm']} seed {candidate['seed']}"
    controls = plan["controls"]
    config = candidate["config"]
    trainer = config.get("trainer", {})
    if candidate.get("_warmup_artifact_errors"):
        errors.extend(
            f"{prefix}: {item}"
            for item in candidate["_warmup_artifact_errors"]
        )
    if bool(trainer.get("bootstrap_queries")):
        errors.append(f"{prefix}: candidate unexpectedly bootstraps queries")
    if str(trainer.get("violation_pooling_mode")) != "source_anchor_changed_states":
        errors.append(f"{prefix}: candidate pooling mode mismatch")
    if float(trainer.get("violation_pooling_change_tolerance", -1.0)) != float(
        controls["violation_pooling_change_tolerance"]
    ):
        errors.append(f"{prefix}: source-anchor change tolerance mismatch")
    if _normalized_treatment_config(config) != _normalized_treatment_config(
        reference["config"]
    ):
        errors.append(
            f"{prefix}: config differs from full_all_states beyond output/pooling mode"
        )

    scalar_equal = (
        "oracle_queries",
        "llm_interactions",
        "trainable_query_count",
        "warmup_query_count",
        "warmup_safe_count",
        "warmup_violation_count",
        "spatial_fit_call_count",
    )
    for key in scalar_equal:
        if candidate[key] != reference[key]:
            errors.append(f"{prefix}: paired artifact field {key} differs")
    sequence_equal = (
        "_warmup_trajectory_hashes",
        "warmup_role_signature",
        "warmup_label_sequence",
    )
    for key in sequence_equal:
        if candidate[key] != reference[key]:
            errors.append(f"{prefix}: paired warmup field {key} differs")
    if candidate["warmup_role_label_counts"] != reference["warmup_role_label_counts"]:
        errors.append(f"{prefix}: paired warmup role-label counts differ")
    if candidate["manifest"].get("files") != reference["manifest"].get("files"):
        errors.append(f"{prefix}: code fingerprint differs from paired baseline")
    if shared_inputs(candidate["manifest"]) != shared_inputs(reference["manifest"]):
        errors.append(f"{prefix}: shared input fingerprint differs")
    if candidate["frozen_bank"].get("sha256") != reference["frozen_bank"].get(
        "sha256"
    ):
        errors.append(f"{prefix}: frozen-bank hash differs")

    runtime = candidate["manifest"].get("runtime", {})
    environment = runtime.get("environment", {})
    if int(runtime.get("seed", -1)) != int(candidate["seed"]):
        errors.append(f"{prefix}: manifest seed mismatch")
    if str(environment.get("PYTHONHASHSEED")) != str(candidate["seed"]):
        errors.append(f"{prefix}: PYTHONHASHSEED mismatch")
    if int(runtime.get("torch_num_threads", 0)) != int(controls["torch_num_threads"]):
        errors.append(f"{prefix}: torch thread count mismatch")
    fractions = (
        candidate["spatial_final_fit_member_unique_fraction_mean"],
        candidate["spatial_final_fit_member_unique_fraction_min"],
        candidate["spatial_cumulative_member_unique_fraction_mean"],
        candidate["spatial_cumulative_member_unique_fraction_min"],
    )
    if any(value is None or abs(float(value) - 1.0) > 1.0e-12 for value in fractions):
        errors.append(f"{prefix}: candidate did not use full query-buffer coverage")
    if int(candidate["spatial_pooling_fit_call_count"]) != int(
        candidate["spatial_fit_call_count"]
    ):
        errors.append(f"{prefix}: pooling diagnostics missed a spatial fit call")
    if int(candidate["spatial_source_anchor_masked_violation_total"]) <= 0:
        errors.append(f"{prefix}: source-anchor treatment was never active")
    if int(candidate["spatial_unique_source_anchor_masked_violation_total"]) <= 0:
        errors.append(f"{prefix}: source-anchor treatment has no unique record")
    if candidate["spatial_final_source_anchor_masked_violation_counts"] != candidate[
        "spatial_final_unique_source_anchor_masked_violation_counts"
    ]:
        errors.append(
            f"{prefix}: full-buffer mask draw and unique-record counts differ"
        )
    return errors


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
        errors.append("run arm/seed set differs from the preregistered design")

    baseline_rows, baseline_plan = _baseline_validation_rows_and_plan(rows, plan)
    errors.extend(numeric_runner.validate_artifacts(baseline_rows, baseline_plan))

    for row in rows:
        configured = plan["arms"][str(row["arm"])]
        trainer = row["config"].get("trainer", {})
        prefix = f"{row['arm']} seed {row['seed']}"
        if bool(trainer.get("bootstrap_queries")) is not bool(
            configured["trainer_bootstrap_queries"]
        ):
            errors.append(f"{prefix}: bootstrap mismatch")
        if str(trainer.get("violation_pooling_mode")) != str(
            configured["violation_pooling_mode"]
        ):
            errors.append(f"{prefix}: pooling mode mismatch")
        if float(trainer.get("violation_pooling_change_tolerance", -1.0)) != float(
            plan["controls"]["violation_pooling_change_tolerance"]
        ):
            errors.append(f"{prefix}: pooling change tolerance mismatch")
        environment = row["manifest"].get("runtime", {}).get("environment", {})
        for variable, control_key in (
            ("OMP_NUM_THREADS", "omp_num_threads"),
            ("MKL_NUM_THREADS", "mkl_num_threads"),
        ):
            if str(environment.get(variable)) != str(plan["controls"][control_key]):
                errors.append(f"{prefix}: {variable} mismatch")
        if row["arm"] in BASELINE_ARMS and int(
            row["spatial_source_anchor_masked_violation_total"]
        ) != 0:
            errors.append(
                f"{prefix}: baseline reports active source mask"
            )
        if row["arm"] in BASELINE_ARMS and int(
            row["spatial_unique_source_anchor_masked_violation_total"]
        ) != 0:
            errors.append(
                f"{prefix}: baseline reports unique masked records"
            )

    for seed in (int(value) for value in plan["seeds"]):
        per_arm = {
            str(row["arm"]): row for row in rows if int(row["seed"]) == seed
        }
        candidate = per_arm[CANDIDATE_ARM]
        reference = per_arm["full_all_states"]
        errors.extend(_validate_candidate_row(candidate, reference, plan))
        normalized = {
            json.dumps(
                _normalized_treatment_config(per_arm[arm]["config"]),
                sort_keys=True,
            )
            for arm in ARMS
        }
        if len(normalized) != 1:
            errors.append(
                f"seed {seed}: arms differ beyond output/bootstrap/pooling treatments"
            )
    return errors


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "config",
        "manifest",
        "frozen_bank",
        "spatial_fit_coverage_by_call",
        "spatial_pooling_activity_by_call",
    }
    return {
        key: value
        for key, value in row.items()
        if key not in excluded and not key.startswith("_")
    }


def paired_results(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    public_metrics = (
        "spatial_balanced_accuracy",
        "spatial_safe_accuracy",
        "spatial_violation_recall",
        "spatial_expert_safe_rate",
        "spatial_fit_expert_safe_rate",
    )
    pairs: list[dict[str, Any]] = []
    for seed in seeds:
        per_arm = {
            str(row["arm"]): row for row in rows if int(row["seed"]) == seed
        }
        pair: dict[str, Any] = {"seed": seed}
        for arm in ARMS:
            row = per_arm[arm]
            pair[f"{arm}_champion"] = row["champion"]
            pair[f"{arm}_status"] = row["selection_status"]
            pair[f"{arm}_qualified_spatial_champion"] = row[
                "qualified_spatial_champion"
            ]
            pair[f"{arm}_spatial_evidence_eligible"] = row[
                "spatial_evidence_eligible"
            ]
            pair[f"{arm}_oracle_queries"] = row["oracle_queries"]
            pair[f"{arm}_final_fit_coverage"] = row[
                "spatial_final_fit_member_unique_fraction_mean"
            ]
            pair[f"{arm}_spatial_iou_evaluation_only"] = row[
                "spatial_iou_evaluation_only"
            ]
            for metric in public_metrics:
                pair[f"{arm}_{metric}"] = row[metric]
        for baseline in BASELINE_ARMS:
            pair[
                f"candidate_minus_{baseline}_qualified_spatial_champion"
            ] = int(bool(per_arm[CANDIDATE_ARM]["qualified_spatial_champion"])) - int(
                bool(per_arm[baseline]["qualified_spatial_champion"])
            )
            for metric in public_metrics:
                pair[f"candidate_minus_{baseline}_{metric}"] = float(
                    per_arm[CANDIDATE_ARM][metric]
                ) - float(per_arm[baseline][metric])
            pair[f"candidate_minus_{baseline}_spatial_iou_evaluation_only"] = (
                float(per_arm[CANDIDATE_ARM]["spatial_iou_evaluation_only"])
                - float(per_arm[baseline]["spatial_iou_evaluation_only"])
            )
        pairs.append(pair)
    return pairs


def apply_decision_rule(
    pairs: list[dict[str, Any]],
    arm_summaries: dict[str, dict[str, Any]],
    rule: dict[str, Any],
) -> dict[str, Any]:
    counts = {
        arm: int(summary["qualified_spatial_champion_seed_count"])
        for arm, summary in arm_summaries.items()
    }
    minimum = int(rule["minimum_candidate_qualified_spatial_champions"])
    candidate_count = counts[CANDIDATE_ARM]
    minimum_met = candidate_count >= minimum
    strictly_exceeds = all(
        candidate_count > counts[baseline] for baseline in BASELINE_ARMS
    )
    floors = {
        key: float(value) for key, value in rule["public_safety_floors"].items()
    }
    regressions: list[dict[str, Any]] = []
    for pair in pairs:
        for baseline in BASELINE_ARMS:
            for metric, floor in floors.items():
                before = float(pair[f"{baseline}_{metric}"])
                after = float(pair[f"{CANDIDATE_ARM}_{metric}"])
                if before >= floor and after < floor:
                    regressions.append(
                        {
                            "seed": pair["seed"],
                            "baseline": baseline,
                            "metric": metric,
                            "floor": floor,
                            "before": before,
                            "after": after,
                        }
                    )
    no_regression = not regressions
    selected = minimum_met and strictly_exceeds and no_regression
    failed: list[str] = []
    if not minimum_met:
        failed.append(f"candidate qualified count is below {minimum}/5")
    if not strictly_exceeds:
        failed.append("candidate does not strictly exceed both baselines")
    if not no_regression:
        failed.append("candidate has a paired public safety-floor regression")
    return {
        "qualified_spatial_champion_seed_counts": counts,
        "candidate_minimum_qualified_count_met": minimum_met,
        "candidate_strictly_exceeds_each_baseline": strictly_exceeds,
        "public_safety_floor_regressions": regressions,
        "no_public_safety_floor_regression": no_regression,
        "private_geometry_used_for_decision": False,
        "selected_default": CANDIDATE_ARM if selected else str(rule["fallback"]),
        "candidate_selected": selected,
        "reason": (
            "candidate passed the preregistered qualified-count and public-safety guards"
            if selected
            else "; ".join(failed)
        ),
    }


def write_outputs(
    output_root: Path,
    plan_path: Path,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    validation_errors: list[str],
) -> dict[str, Any]:
    public_rows = [public_row(row) for row in rows]
    arm_summaries = {
        arm: numeric_runner.summarize_arm(
            [row for row in rows if row["arm"] == arm]
        )
        for arm in ARMS
    }
    pairs = paired_results(rows, [int(value) for value in plan["seeds"]])
    decision = apply_decision_rule(pairs, arm_summaries, plan["decision_rule"])
    if validation_errors:
        decision["selected_default"] = None
        decision["candidate_selected"] = False
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
    (output_root / "violation_pooling_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if public_rows:
        fieldnames = list(public_rows[0])
        with (output_root / "violation_pooling_per_run_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(public_rows)

    report = [
        "# Violation pooling: strict three-arm, five-seed comparison",
        "",
        f"Plan: `{plan_path.name}` (SHA256 `{sha256(plan_path)}`).",
        "",
        "Private IoU/false-safe/false-unsafe are diagnostics only and never enter the decision rule.",
        "",
        "| arm | qualified spatial champions | spatial eligible | median public S | median public V | mean fit coverage | mean spatial IoU* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = arm_summaries[arm]
        report.append(
            "| {arm} | {qualified}/5 | {eligible}/5 | {safe:.3f} | "
            "{violation:.3f} | {coverage:.3f} | {iou:.3f} |".format(
                arm=arm,
                qualified=item["qualified_spatial_champion_seed_count"],
                eligible=item["spatial_evidence_eligible_seed_count"],
                safe=float(item["spatial_safe_accuracy"]["median"]),
                violation=float(item["spatial_violation_recall"]["median"]),
                coverage=float(
                    item["spatial_final_fit_member_unique_fraction"]["mean"]
                ),
                iou=float(item["spatial_iou_evaluation_only"]["mean"]),
            )
        )
    report.extend(
        [
            "",
            "\* evaluation-only private geometry.",
            "",
            f"- Artifact validation: `{'PASS' if not validation_errors else 'FAIL'}`",
            f"- Candidate at least 4/5: `{decision['candidate_minimum_qualified_count_met']}`",
            f"- Candidate strictly exceeds both baselines: `{decision['candidate_strictly_exceeds_each_baseline']}`",
            f"- No public safety-floor regression: `{decision['no_public_safety_floor_regression']}`",
            f"- Private geometry used for decision: `{decision['private_geometry_used_for_decision']}`",
            f"- Selected default: `{decision['selected_default']}`",
            "",
            str(decision["reason"]),
            "",
            "Five seeds assess engineering stability; they are not a claim of statistical significance.",
        ]
    )
    if validation_errors:
        report.extend(("", "## Artifact validation errors", ""))
        report.extend(f"- {error}" for error in validation_errors)
    (output_root / "VIOLATION_POOLING_MULTISEED_REPORT.md").write_text(
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
                    "treatment_matrix": {
                        arm: {
                            "trainer.bootstrap_queries": bool(
                                plan["arms"][arm]["trainer_bootstrap_queries"]
                            ),
                            "trainer.violation_pooling_mode": str(
                                plan["arms"][arm]["violation_pooling_mode"]
                            ),
                        }
                        for arm in ARMS
                    },
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
