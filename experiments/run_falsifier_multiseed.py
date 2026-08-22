"""Run and summarize the preregistered hard-single versus hard-ladder sweep."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_PLAN = PACKAGE_ROOT / "configs" / "falsifier_multiseed_plan.yaml"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs" / "falsifier_multiseed_5seed"


def repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Read completed run directories without launching subprocesses.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_directory(output_root: Path, arm: str, seed: int) -> Path:
    return output_root / arm / f"seed_{seed}"


def validate_plan(plan: dict[str, Any]) -> None:
    seeds = [int(seed) for seed in plan.get("seeds", [])]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("the preregistered experiment requires five distinct seeds")
    arms = plan.get("arms", {})
    if set(arms) != {"hard_single_032", "hard_ladder"}:
        raise ValueError("plan must contain hard_single_032 and hard_ladder arms")
    single = arms["hard_single_032"]
    ladder = arms["hard_ladder"]
    if list(single.get("false_unsafe_radius_ladder", [])):
        raise ValueError("hard_single_032 must have an empty radius ladder")
    if float(single.get("false_unsafe_trust_radius", 0.0)) != 0.32:
        raise ValueError("hard_single_032 radius must be 0.32")
    if [float(value) for value in ladder.get("false_unsafe_radius_ladder", [])] != [
        0.04,
        0.08,
        0.16,
        0.32,
    ]:
        raise ValueError("hard_ladder must use 0.04/0.08/0.16/0.32")
    if not bool(single.get("false_unsafe_use_hard_margin")) or not bool(
        ladder.get("false_unsafe_use_hard_margin")
    ):
        raise ValueError("both arms must use the hard-margin objective")
    if str(plan.get("controls", {}).get("violation_pooling_mode")) != "all_states":
        raise ValueError("the historical falsifier sweep must use all-state pooling")


def build_command(
    plan: dict[str, Any],
    arm: str,
    seed: int,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "run_obstacle_avoid.py"),
        "--config",
        str(repository_path(str(plan["base_config"])).resolve()),
        "--initial-hypothesis-bank",
        str(repository_path(str(plan["frozen_hypothesis_bank"])).resolve()),
        "--freeze-revisions",
        # Keep the published falsifier comparison isolated from the later
        # numeric-pooling change.
        "--violation-pooling-mode",
        str(plan["controls"]["violation_pooling_mode"]),
        "--audit-only-linear-max-support-gate",
        "--seed",
        str(seed),
        "--output",
        str(output.resolve()),
    ]
    arm_config = plan["arms"][arm]
    if arm == "hard_single_032":
        command.extend(
            (
                "--false-unsafe-single-radius",
                str(float(arm_config["false_unsafe_trust_radius"])),
            )
        )
    elif arm == "hard_ladder":
        command.append("--false-unsafe-radius-ladder")
        command.extend(str(float(value)) for value in arm_config["false_unsafe_radius_ladder"])
    else:
        raise ValueError(f"unsupported arm: {arm}")
    return command


def run_one(
    plan: dict[str, Any],
    output_root: Path,
    arm: str,
    seed: int,
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
    duration = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "run_stderr.log").write_text(completed.stderr, encoding="utf-8")
    metadata = {
        "arm": arm,
        "seed": seed,
        "command": command,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "duration_seconds_parallel_context_only": duration,
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


def launch_runs(
    plan: dict[str, Any],
    output_root: Path,
    max_workers: int,
) -> None:
    if max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output root is not empty: {output_root}; use a new directory"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_plan_resolved.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    tasks = [
        (arm, int(seed))
        for seed in plan["seeds"]
        for arm in ("hard_single_032", "hard_ladder")
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
            except Exception as exc:  # keep all already-started runs auditable
                failures.append(f"{arm}/seed_{seed}: {exc}")
                print(f"failed arm={arm} seed={seed}: {exc}", flush=True)
    if failures:
        raise RuntimeError("one or more sweep runs failed:\n" + "\n".join(failures))


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values]
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


def canonical_array_hash(*arrays: np.ndarray) -> str:
    """Hash numeric arrays with explicit shape, dtype, order, and byte order."""

    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        little_endian_dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(little_endian_dtype, copy=False))
        header = json.dumps(
            {"shape": list(canonical.shape), "dtype": canonical.dtype.str},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def extract_warmup_artifact(output: Path) -> dict[str, Any]:
    """Audit the ordered warmup slate independently of later active queries."""

    errors: list[str] = []
    query_log = load_json(output / "oracle_query_log.json")
    if not isinstance(query_log, list):
        raise ValueError(f"expected a list in {output / 'oracle_query_log.json'}")
    with np.load(output / "oracle_queries.npz", allow_pickle=False) as archive:
        required = {"observations", "actions", "labels", "outer_rounds"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"oracle archive missing arrays: {sorted(missing)}")
        observations = np.asarray(archive["observations"])
        actions = np.asarray(archive["actions"])
        labels = np.asarray(archive["labels"])
        outer_rounds = np.asarray(archive["outer_rounds"])

    lengths = {
        len(query_log),
        int(observations.shape[0]),
        int(actions.shape[0]),
        int(labels.shape[0]),
        int(outer_rounds.shape[0]),
    }
    if len(lengths) != 1:
        errors.append(
            "oracle query log/archive lengths differ: "
            f"log={len(query_log)}, observations={observations.shape[0]}, "
            f"actions={actions.shape[0]}, labels={labels.shape[0]}, "
            f"outer_rounds={outer_rounds.shape[0]}"
        )
    aligned_count = min(lengths) if lengths else 0
    for index in range(aligned_count):
        entry = query_log[index]
        if int(entry.get("label", -1)) != int(labels[index]):
            errors.append(f"query {index}: log/archive label mismatch")
        if int(entry.get("outer_round", -1)) != int(outer_rounds[index]):
            errors.append(f"query {index}: log/archive outer-round mismatch")

    archive_indices = [int(value) for value in np.flatnonzero(outer_rounds == 0)]
    log_indices = []
    for index, entry in enumerate(query_log):
        metadata = entry.get("trajectory_metadata", {})
        if (
            int(entry.get("outer_round", -1)) == 0
            and metadata.get("source") == "warmup"
            and "warmup_pair_index" in metadata
        ):
            log_indices.append(index)
    if archive_indices != log_indices:
        errors.append(
            f"warmup indices disagree: archive={archive_indices}, log={log_indices}"
        )
    warmup_indices = [index for index in archive_indices if index < aligned_count]
    if warmup_indices != list(range(len(warmup_indices))):
        errors.append(f"warmup queries are not a contiguous prefix: {warmup_indices}")

    trajectory_hashes = [
        canonical_array_hash(observations[index], actions[index])
        for index in warmup_indices
    ]
    sequence_digest = hashlib.sha256()
    for item_hash in trajectory_hashes:
        sequence_digest.update(item_hash.encode("ascii"))
        sequence_digest.update(b"\n")
    label_sequence = "".join(str(int(labels[index])) for index in warmup_indices)

    roles_by_pair: dict[int, set[str]] = {}
    directions_by_pair: dict[int, list[str]] = {}
    expert_scale_by_pair: dict[int, set[tuple[str, float]]] = {}
    role_label_counts = {
        "warmup": {"safe": 0, "violation": 0},
        "warmup_validation": {"safe": 0, "violation": 0},
        "final_calibration": {"safe": 0, "violation": 0},
    }
    for index in warmup_indices:
        entry = query_log[index]
        metadata = entry.get("trajectory_metadata", {})
        pair_index = int(metadata["warmup_pair_index"])
        source = str(entry.get("source"))
        roles_by_pair.setdefault(pair_index, set()).add(source)
        if source not in role_label_counts:
            errors.append(f"query {index}: unexpected warmup evidence role {source!r}")
        else:
            label_name = "safe" if int(labels[index]) == 0 else "violation"
            role_label_counts[source][label_name] += 1
        directions_by_pair.setdefault(pair_index, []).append(
            str(metadata.get("warmup_direction"))
        )
        expert_scale_by_pair.setdefault(pair_index, set()).add(
            (str(metadata.get("expert_id")), float(metadata.get("alpha")))
        )
    for pair_index in sorted(roles_by_pair):
        directions = directions_by_pair[pair_index]
        if len(directions) != 2 or set(directions) != {
            "toward_chord",
            "continue_detour",
        }:
            errors.append(
                f"warmup pair {pair_index}: expected exactly two complementary directions"
            )
        if len(roles_by_pair[pair_index]) != 1:
            errors.append(
                f"warmup pair {pair_index}: correlated family crosses evidence roles "
                f"{sorted(roles_by_pair[pair_index])}"
            )
        if len(expert_scale_by_pair[pair_index]) != 1:
            errors.append(
                f"warmup pair {pair_index}: members do not share expert and scale"
            )
    role_signature = ",".join(
        f"{pair_index}:{next(iter(roles_by_pair[pair_index]))}"
        for pair_index in sorted(roles_by_pair)
        if len(roles_by_pair[pair_index]) == 1
    )
    return {
        "warmup_query_count": len(warmup_indices),
        "warmup_trajectory_sequence_sha256": sequence_digest.hexdigest(),
        "warmup_label_sequence": label_sequence,
        "warmup_safe_count": sum(int(labels[index]) == 0 for index in warmup_indices),
        "warmup_violation_count": sum(int(labels[index]) == 1 for index in warmup_indices),
        "warmup_role_signature": role_signature,
        "warmup_role_label_counts": role_label_counts,
        "_warmup_trajectory_hashes": trajectory_hashes,
        "_warmup_artifact_errors": errors,
    }


def extract_run(output: Path, arm: str, seed: int) -> dict[str, Any]:
    diagnostics = load_json(output / "query_diagnostics.json")
    result = load_json(output / "result.json")
    config = load_yaml(output / "resolved_config.yaml")
    manifest = load_json(output / "implementation_manifest.json")
    frozen_bank = load_json(output / "frozen_bank_source.json")
    false_unsafe = [
        item
        for item in diagnostics
        if item.get("intervention", {}).get("kind") == "model_false_unsafe"
    ]
    queried = [item for item in false_unsafe if bool(item.get("queried"))]
    unsafe_query_errors = []
    for item in queried:
        metadata = item.get("falsifier", {})
        label_metadata = item.get("label_acquisition_initial", {})
        if not (
            metadata.get("generation_hard_margin_satisfied") is True
            and metadata.get("query_hard_margin_satisfied") is True
            and label_metadata.get("safe_query_eligible") is True
        ):
            unsafe_query_errors.append(
                {
                    "round": item.get("outer_round"),
                    "source": item.get("source_hypothesis_id"),
                }
            )
    deformations = [
        float(item["falsifier"]["max_expert_deviation"])
        for item in queried
    ]
    safe_deformations = [
        float(item["falsifier"]["max_expert_deviation"])
        for item in queried
        if int(item["oracle_label"]) == 0
    ]
    safe_hits = sum(int(item["oracle_label"]) == 0 for item in queried)
    optimizer_launches = sum(
        len(item.get("falsifier", {}).get("radius_ladder_attempts") or [])
        for item in false_unsafe
    )
    steps = int(config["falsifier"]["steps"])
    warmup_artifact = extract_warmup_artifact(output)
    return {
        "arm": arm,
        "seed": seed,
        "output": str(output.resolve()),
        "candidate_count": len(false_unsafe),
        "generation_crossing_count": sum(
            item.get("falsifier", {}).get("generation_hard_margin_satisfied") is True
            for item in false_unsafe
        ),
        "query_certified_count": sum(
            item.get("falsifier", {}).get("query_hard_margin_satisfied") is True
            for item in false_unsafe
        ),
        "safe_query_eligible_count": sum(
            item.get("label_acquisition_initial", {}).get("safe_query_eligible") is True
            for item in false_unsafe
        ),
        "false_unsafe_queries": len(queried),
        "oracle_safe_false_unsafe_count": safe_hits,
        "oracle_safe_budget_yield": safe_hits / int(result["oracle_queries"]),
        "conditional_oracle_safe_yield": safe_hits / len(queried) if queried else None,
        "query_deformation_median": statistics.median(deformations) if deformations else None,
        "query_deformation_mean": statistics.mean(deformations) if deformations else None,
        "query_deformation_max": max(deformations) if deformations else None,
        "safe_query_deformation_median": (
            statistics.median(safe_deformations) if safe_deformations else None
        ),
        "false_unsafe_optimizer_launches": optimizer_launches,
        "false_unsafe_gradient_updates": optimizer_launches * steps,
        "iou": float(result["final_metrics"]["iou"]),
        "false_safe_rate": float(result["final_metrics"]["false_safe_rate"]),
        "false_unsafe_rate": float(result["final_metrics"]["false_unsafe_rate"]),
        "accuracy": float(result["final_metrics"]["accuracy"]),
        "champion": result["champion_hypothesis_id"],
        "oracle_queries": int(result["oracle_queries"]),
        "llm_interactions": int(result["llm_interactions"]),
        "unsafe_query_certificate_errors": unsafe_query_errors,
        "config": config,
        "manifest": manifest,
        "frozen_bank": frozen_bank,
        **warmup_artifact,
    }


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.pop("output_dir", None)
    normalized.get("falsifier", {}).pop("false_unsafe_radius_ladder", None)
    return normalized


def shared_inputs(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        key: value
        for key, value in manifest.get("inputs", {}).items()
        if not key.endswith("/resolved_config.yaml")
    }


def validate_artifacts(rows: list[dict[str, Any]], plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_oracle = int(plan["controls"]["oracle_queries_per_run"])
    expected_warmup = int(plan["controls"]["warmup_queries"])
    expected_warmup_labels = plan["controls"].get("benchmark_warmup_labels", {})
    baseline_files = rows[0]["manifest"].get("files")
    baseline_inputs = shared_inputs(rows[0]["manifest"])
    baseline_bank = rows[0]["frozen_bank"].get("sha256")
    baseline_warmup_hashes = rows[0]["_warmup_trajectory_hashes"]
    baseline_warmup_roles = rows[0]["warmup_role_signature"]
    baseline_warmup_labels = rows[0]["warmup_label_sequence"]
    for row in rows:
        if int(row["config"].get("seed")) != int(row["seed"]):
            errors.append(f"{row['arm']} seed {row['seed']}: resolved seed mismatch")
        if row["oracle_queries"] != expected_oracle:
            errors.append(f"{row['arm']} seed {row['seed']}: Oracle budget mismatch")
        if row["llm_interactions"] != 0:
            errors.append(f"{row['arm']} seed {row['seed']}: unexpected LLM interaction")
        if row["unsafe_query_certificate_errors"]:
            errors.append(f"{row['arm']} seed {row['seed']}: uncertified query")
        if row["_warmup_artifact_errors"]:
            errors.extend(
                f"{row['arm']} seed {row['seed']}: {error}"
                for error in row["_warmup_artifact_errors"]
            )
        if int(row["warmup_query_count"]) != expected_warmup:
            errors.append(f"{row['arm']} seed {row['seed']}: warmup query count mismatch")
        if (
            int(row["warmup_safe_count"])
            != int(expected_warmup_labels.get("safe", row["warmup_safe_count"]))
            or int(row["warmup_violation_count"])
            != int(expected_warmup_labels.get("violation", row["warmup_violation_count"]))
        ):
            errors.append(f"{row['arm']} seed {row['seed']}: warmup label balance mismatch")
        if row["_warmup_trajectory_hashes"] != baseline_warmup_hashes:
            differing = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(row["_warmup_trajectory_hashes"], baseline_warmup_hashes)
                    )
                    if left != right
                ),
                min(
                    len(row["_warmup_trajectory_hashes"]),
                    len(baseline_warmup_hashes),
                ),
            )
            errors.append(
                f"{row['arm']} seed {row['seed']}: warmup trajectory mismatch at index {differing}"
            )
        if row["warmup_role_signature"] != baseline_warmup_roles:
            errors.append(f"{row['arm']} seed {row['seed']}: warmup role split mismatch")
        if row["warmup_label_sequence"] != baseline_warmup_labels:
            errors.append(f"{row['arm']} seed {row['seed']}: warmup label sequence mismatch")
        role_counts = row["warmup_role_label_counts"]
        loop_config = row["config"].get("loop", {})
        role_requirements = {
            "warmup_validation": {
                "safe": int(loop_config.get("warmup_validation_safe_count", 0)),
                "violation": int(loop_config.get("warmup_validation_violation_count", 0)),
            },
            "final_calibration": {
                "safe": int(loop_config.get("final_calibration_safe_count", 0)),
                "violation": int(loop_config.get("final_calibration_violation_count", 0)),
            },
        }
        for role, required_counts in role_requirements.items():
            for label_name, required_count in required_counts.items():
                if int(role_counts[role][label_name]) < required_count:
                    errors.append(
                        f"{row['arm']} seed {row['seed']}: {role} {label_name} quota mismatch"
                    )
        if (
            int(role_counts["warmup"]["safe"]) < 1
            or int(role_counts["warmup"]["violation"]) < 1
        ):
            errors.append(
                f"{row['arm']} seed {row['seed']}: warmup training split lacks a label"
            )
        if row["manifest"].get("files") != baseline_files:
            errors.append(f"{row['arm']} seed {row['seed']}: code fingerprint mismatch")
        if shared_inputs(row["manifest"]) != baseline_inputs:
            errors.append(f"{row['arm']} seed {row['seed']}: shared input mismatch")
        if row["frozen_bank"].get("sha256") != baseline_bank:
            errors.append(f"{row['arm']} seed {row['seed']}: frozen bank mismatch")
        runtime = row["manifest"].get("runtime", {})
        environment = runtime.get("environment", {})
        if str(environment.get("PYTHONHASHSEED")) != str(row["seed"]):
            errors.append(f"{row['arm']} seed {row['seed']}: PYTHONHASHSEED mismatch")
        if int(runtime.get("torch_num_threads", 0)) != 1:
            errors.append(f"{row['arm']} seed {row['seed']}: torch threads not fixed to 1")
        expected_arm = plan["arms"][row["arm"]]
        actual_falsifier = row["config"].get("falsifier", {})
        if bool(actual_falsifier.get("false_unsafe_use_hard_margin")) != bool(
            expected_arm["false_unsafe_use_hard_margin"]
        ):
            errors.append(f"{row['arm']} seed {row['seed']}: hard objective mismatch")
        if float(actual_falsifier.get("false_unsafe_trust_radius", 0.0)) != float(
            expected_arm["false_unsafe_trust_radius"]
        ):
            errors.append(f"{row['arm']} seed {row['seed']}: trust radius mismatch")
        if [
            float(value)
            for value in actual_falsifier.get("false_unsafe_radius_ladder", [])
        ] != [
            float(value)
            for value in expected_arm["false_unsafe_radius_ladder"]
        ]:
            errors.append(f"{row['arm']} seed {row['seed']}: radius ladder mismatch")
    for seed in (int(value) for value in plan["seeds"]):
        single = next(row for row in rows if row["arm"] == "hard_single_032" and row["seed"] == seed)
        ladder = next(row for row in rows if row["arm"] == "hard_ladder" and row["seed"] == seed)
        if normalized_config(single["config"]) != normalized_config(ladder["config"]):
            errors.append(f"seed {seed}: paired configs differ beyond output/ladder")
    return errors


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"config", "manifest", "frozen_bank", "unsafe_query_certificate_errors"}
        and not key.startswith("_")
    }


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    queried = sum(int(row["false_unsafe_queries"]) for row in rows)
    safe = sum(int(row["oracle_safe_false_unsafe_count"]) for row in rows)
    return {
        "runs": len(rows),
        "total_false_unsafe_queries": queried,
        "total_oracle_safe_false_unsafe": safe,
        "pooled_conditional_oracle_safe_yield": safe / queried if queried else None,
        "pooled_oracle_budget_yield": safe / sum(int(row["oracle_queries"]) for row in rows),
        "safe_hits_per_seed": numeric_summary(
            [float(row["oracle_safe_false_unsafe_count"]) for row in rows]
        ),
        "query_deformation_median_per_seed": numeric_summary(
            [
                float(row["query_deformation_median"])
                for row in rows
                if row["query_deformation_median"] is not None
            ]
        ),
        "generation_crossings_per_seed": numeric_summary(
            [float(row["generation_crossing_count"]) for row in rows]
        ),
        "optimizer_launches_per_seed": numeric_summary(
            [float(row["false_unsafe_optimizer_launches"]) for row in rows]
        ),
        "optimizer_launches_total": sum(
            int(row["false_unsafe_optimizer_launches"]) for row in rows
        ),
        "gradient_updates_total": sum(
            int(row["false_unsafe_gradient_updates"]) for row in rows
        ),
        "iou": numeric_summary([float(row["iou"]) for row in rows]),
        "false_safe_rate": numeric_summary(
            [float(row["false_safe_rate"]) for row in rows]
        ),
        "false_unsafe_rate": numeric_summary(
            [float(row["false_unsafe_rate"]) for row in rows]
        ),
    }


def paired_results(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    pairs = []
    for seed in seeds:
        single = next(row for row in rows if row["arm"] == "hard_single_032" and row["seed"] == seed)
        ladder = next(row for row in rows if row["arm"] == "hard_ladder" and row["seed"] == seed)
        single_deformation = single["query_deformation_median"]
        ladder_deformation = ladder["query_deformation_median"]
        deformation_reduction = None
        if single_deformation not in {None, 0.0} and ladder_deformation is not None:
            deformation_reduction = (
                float(single_deformation) - float(ladder_deformation)
            ) / float(single_deformation)
        pairs.append(
            {
                "seed": seed,
                "single_safe_hits": single["oracle_safe_false_unsafe_count"],
                "ladder_safe_hits": ladder["oracle_safe_false_unsafe_count"],
                "delta_safe_hits": (
                    ladder["oracle_safe_false_unsafe_count"]
                    - single["oracle_safe_false_unsafe_count"]
                ),
                "single_conditional_yield": single["conditional_oracle_safe_yield"],
                "ladder_conditional_yield": ladder["conditional_oracle_safe_yield"],
                "single_deformation_median": single_deformation,
                "ladder_deformation_median": ladder_deformation,
                "deformation_reduction_fraction": deformation_reduction,
                "single_optimizer_launches": single["false_unsafe_optimizer_launches"],
                "ladder_optimizer_launches": ladder["false_unsafe_optimizer_launches"],
                "optimizer_launch_ratio": (
                    ladder["false_unsafe_optimizer_launches"]
                    / single["false_unsafe_optimizer_launches"]
                ),
                "single_iou": single["iou"],
                "ladder_iou": ladder["iou"],
                "delta_iou": ladder["iou"] - single["iou"],
                "single_false_safe_rate": single["false_safe_rate"],
                "ladder_false_safe_rate": ladder["false_safe_rate"],
            }
        )
    return pairs


def apply_decision_rule(
    pairs: list[dict[str, Any]],
    arm_summaries: dict[str, dict[str, Any]],
    rule: dict[str, Any],
) -> dict[str, Any]:
    iou_deltas = [float(pair["delta_iou"]) for pair in pairs]
    reductions = [
        float(pair["deformation_reduction_fraction"])
        for pair in pairs
        if pair["deformation_reduction_fraction"] is not None
    ]
    safe_deltas = [int(pair["delta_safe_hits"]) for pair in pairs]
    single_safe = int(arm_summaries["hard_single_032"]["total_oracle_safe_false_unsafe"])
    ladder_safe = int(arm_summaries["hard_ladder"]["total_oracle_safe_false_unsafe"])
    iou_rule = rule["iou_guard"]
    yield_rule = rule["yield_stable"]
    deformation_rule = rule["deformation_stable"]
    yield_noninferior_rule = rule["yield_noninferior"]
    deformation_noninferior_rule = rule["deformation_noninferior"]
    iou_guard = (
        statistics.median(iou_deltas)
        >= float(iou_rule["median_paired_delta_min"])
        and min(iou_deltas) >= float(iou_rule["worst_seed_delta_min"])
    )
    yield_stable = (
        sum(delta > 0 for delta in safe_deltas)
        >= int(yield_rule["strict_seed_wins_min"])
        and (
            not bool(yield_rule["pooled_safe_hits_must_be_strictly_higher"])
            or ladder_safe > single_safe
        )
    )
    reduction_threshold = float(deformation_rule["seed_reduction_fraction_min"])
    deformation_stable = (
        len(reductions) >= int(deformation_rule["qualifying_seed_count_min"])
        and sum(reduction >= reduction_threshold for reduction in reductions)
        >= int(deformation_rule["qualifying_seed_count_min"])
        and statistics.median(reductions)
        >= float(deformation_rule["paired_median_reduction_min"])
    )
    yield_noninferior = (
        (
            not bool(yield_noninferior_rule["pooled_safe_hits_must_not_decrease"])
            or ladder_safe >= single_safe
        )
        and sum(delta < 0 for delta in safe_deltas)
        <= int(yield_noninferior_rule["worse_seed_count_max"])
    )
    worsening_threshold = float(
        deformation_noninferior_rule["worsening_fraction_threshold"]
    )
    deformation_noninferior = (
        len(reductions) >= len(pairs) - int(
            deformation_noninferior_rule["worse_seed_count_max"]
        )
        and sum(reduction < -worsening_threshold for reduction in reductions)
        <= int(deformation_noninferior_rule["worse_seed_count_max"])
    )
    select_ladder = iou_guard and (
        (yield_stable and deformation_noninferior)
        or (deformation_stable and yield_noninferior)
    )
    return {
        "iou_guard": iou_guard,
        "yield_stable": yield_stable,
        "deformation_stable": deformation_stable,
        "yield_noninferior": yield_noninferior,
        "deformation_noninferior": deformation_noninferior,
        "safe_hit_seed_wins": sum(delta > 0 for delta in safe_deltas),
        "safe_hit_seed_losses": sum(delta < 0 for delta in safe_deltas),
        "deformation_reduction_seed_wins_at_least_10pct": sum(
            reduction >= reduction_threshold for reduction in reductions
        ),
        "paired_median_deformation_reduction": (
            statistics.median(reductions) if reductions else None
        ),
        "paired_median_delta_iou": statistics.median(iou_deltas),
        "worst_delta_iou": min(iou_deltas),
        "selected_default": "hard_ladder" if select_ladder else str(rule["default"]),
        "reason": (
            "ladder satisfied the preregistered stability and IoU guards"
            if select_ladder
            else "ladder did not satisfy the preregistered stability rule; use the cheaper single radius"
        ),
    }


def format_number(value: Any, digits: int = 4) -> str:
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
        for arm in ("hard_single_032", "hard_ladder")
    }
    pairs = paired_results(rows, [int(seed) for seed in plan["seeds"]])
    decision = apply_decision_rule(pairs, arm_summaries, plan["decision_rule"])
    if validation_errors:
        decision["selected_default"] = None
        decision["reason"] = (
            "artifact validation failed; no method selection is valid until the "
            "experiment is rerun or repaired"
        )
    summary = {
        "experiment_id": plan["experiment_id"],
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256(plan_path),
        "artifact_validation_passed": not validation_errors,
        "artifact_validation_errors": validation_errors,
        "rows": public_rows,
        "arm_summaries": arm_summaries,
        "paired_results": pairs,
        "decision": decision,
    }
    (output_root / "multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if public_rows:
        with (output_root / "per_run_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
            writer.writeheader()
            writer.writerows(public_rows)
    report = [
        "# Hard single-0.32 vs hard ladder：5-seed 配对实验",
        "",
        f"计划：`{plan_path.name}`（SHA256 `{sha256(plan_path)}`）",
        "",
        "## 原始配对结果",
        "",
        "| seed | single safe/query | ladder safe/query | single median deformation | ladder median deformation | reduction | single/ladder launches | ΔIoU |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        single_row = next(
            row for row in rows if row["arm"] == "hard_single_032" and row["seed"] == pair["seed"]
        )
        ladder_row = next(
            row for row in rows if row["arm"] == "hard_ladder" and row["seed"] == pair["seed"]
        )
        report.append(
            "| {seed} | {ss}/{sq} | {ls}/{lq} | {sd} | {ld} | {red} | {so}/{lo} | {diou} |".format(
                seed=pair["seed"],
                ss=pair["single_safe_hits"],
                sq=single_row["false_unsafe_queries"],
                ls=pair["ladder_safe_hits"],
                lq=ladder_row["false_unsafe_queries"],
                sd=format_number(pair["single_deformation_median"], 5),
                ld=format_number(pair["ladder_deformation_median"], 5),
                red=(
                    "NA"
                    if pair["deformation_reduction_fraction"] is None
                    else f"{100.0 * pair['deformation_reduction_fraction']:.1f}%"
                ),
                so=pair["single_optimizer_launches"],
                lo=pair["ladder_optimizer_launches"],
                diou=f"{pair['delta_iou']:+.4f}",
            )
        )
    report.extend(
        [
            "",
            "## 汇总",
            "",
            "| arm | pooled Oracle-safe/query | pooled safe budget yield | mean IoU ± SD | median per-seed deformation | total optimizer launches |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ("hard_single_032", "hard_ladder"):
        arm_summary = arm_summaries[arm]
        report.append(
            "| {arm} | {safe}/{queries} | {budget} | {mean_iou} ± {sd_iou} | {deformation} | {launches} |".format(
                arm=arm,
                safe=arm_summary["total_oracle_safe_false_unsafe"],
                queries=arm_summary["total_false_unsafe_queries"],
                budget=format_number(arm_summary["pooled_oracle_budget_yield"], 4),
                mean_iou=format_number(arm_summary["iou"]["mean"], 4),
                sd_iou=format_number(arm_summary["iou"]["sample_sd"], 4),
                deformation=format_number(
                    arm_summary["query_deformation_median_per_seed"]["median"], 5
                ),
                launches=arm_summary["optimizer_launches_total"],
            )
        )
    report.extend(
        [
            "",
            "## 预注册决策",
            "",
            f"- artifact validation: `{'PASS' if not validation_errors else 'FAIL'}`",
            f"- IoU guard: `{decision['iou_guard']}`",
            f"- stable safe-yield gain: `{decision['yield_stable']}`",
            f"- stable deformation reduction: `{decision['deformation_stable']}`",
            f"- safe-yield noninferiority: `{decision['yield_noninferior']}`",
            f"- deformation noninferiority: `{decision['deformation_noninferior']}`",
            f"- **selected default: `{decision['selected_default']}`**",
            "",
            decision["reason"],
            "",
            "五个 seeds 只支持工程稳定性判断，不构成统计显著性证明。optimizer launches 是计算代理；并行运行时间不作为方法速度比较。",
        ]
    )
    if validation_errors:
        report.extend(("", "## Artifact validation errors", ""))
        report.extend(f"- {error}" for error in validation_errors)
    (output_root / "MULTISEED_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def summarize(plan_path: Path, plan: dict[str, Any], output_root: Path) -> dict[str, Any]:
    rows = []
    for seed in (int(value) for value in plan["seeds"]):
        for arm in ("hard_single_032", "hard_ladder"):
            output = run_directory(output_root, arm, seed)
            rows.append(extract_run(output, arm, seed))
    validation_errors = validate_artifacts(rows, plan)
    resolved_plan_path = output_root / "experiment_plan_resolved.yaml"
    if not resolved_plan_path.exists():
        validation_errors.append("missing experiment_plan_resolved.yaml")
    elif load_yaml(resolved_plan_path) != plan:
        validation_errors.append("current plan differs from the plan frozen at sweep launch")
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
    if not args.summarize_only:
        launch_runs(plan, output_root, args.max_workers)
    summary = summarize(plan_path, plan, output_root)
    print(json.dumps(summary["decision"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
