"""Evaluate any frozen public-TaskSpec checkpoint in a private-data process.

The historical filename is retained for compatibility.  Planar tasks receive
state-grid metrics; higher-dimensional tasks such as CarryWaterActive receive
only faithful whole-trajectory metrics.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import torch
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from llm_modulo_cegis.data import (
    FeatureLibrary,
    load_expert_dataset,
    load_public_workspace,
    load_task_spec,
)
from llm_modulo_cegis.evaluation import evaluate_boundary, plot_boundary
from llm_modulo_cegis.hypotheses import compile_hypothesis, hypothesis_from_dict
from llm_modulo_cegis.learner import LearnerRegistry
from llm_modulo_cegis.oracle import RuleEvaluationOracle
from llm_modulo_cegis.structure_evaluation import evaluate_structure


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Frozen training output directory.")
    parser.add_argument("--private-dir", help="Override private evaluator directory.")
    parser.add_argument("--output", help="Defaults to <run-dir>/posthoc_evaluation.json.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid-resolution", type=int)
    parser.add_argument(
        "--diagnostic-all-hypotheses",
        action="store_true",
        help="Evaluate non-champions too; never use this mode for official model selection.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = _repository_path(args.run_dir)
    checkpoint_path = run_dir / "constraint_models.pt"
    result_path = run_dir / "result.json"
    config_path = run_dir / "resolved_config.yaml"
    freeze_path = run_dir / "freeze_manifest.json"
    for required in (checkpoint_path, result_path, config_path, freeze_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    # Freeze and hash the learned artifact before resolving or opening any
    # private evaluation file.
    frozen_hashes_before = {
        "constraint_models.pt": _file_sha256(checkpoint_path),
        "result.json": _file_sha256(result_path),
    }
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    freeze_manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not bool(freeze_manifest.get("training_complete", False)):
        raise ValueError("training artifact is not sealed complete")
    if bool(freeze_manifest.get("private_evaluation_loaded_before_freeze", True)):
        raise ValueError("official post-hoc evaluation requires a deferred training run")
    sealed_hashes = freeze_manifest.get("training_artifact_sha256", {})
    for name, actual_hash in frozen_hashes_before.items():
        if sealed_hashes.get(name) != actual_hash:
            raise ValueError(f"frozen artifact hash mismatch for {name}")
    dataset_dir = _repository_path(config["data"]["dataset_dir"])
    task_spec = load_task_spec(dataset_dir)
    library = FeatureLibrary.from_task_spec(task_spec)
    device = torch.device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    registry = LearnerRegistry(
        library,
        hidden_dims=config["model"]["hidden_dims"],
        ensemble_size=int(config["model"]["ensemble_size"]),
        seed=int(config.get("seed", 0)),
        device=device,
    )
    compiled_payloads = checkpoint.get("compiled_hypotheses", {})
    model_states = checkpoint.get("models", {})
    if set(compiled_payloads) != set(model_states):
        raise ValueError("checkpoint hypothesis metadata and state dictionaries disagree")
    for hypothesis_id, raw in compiled_payloads.items():
        hypothesis = hypothesis_from_dict(raw)
        if hypothesis.hypothesis_id != hypothesis_id:
            raise ValueError("checkpoint hypothesis id mismatch")
        ensemble = registry.ensure(compile_hypothesis(hypothesis, library))
        ensemble.load_state_dict(model_states[hypothesis_id], strict=True)
        ensemble.eval()
        for parameter in ensemble.parameters():
            parameter.requires_grad_(False)

    champion_id = str(checkpoint["champion_hypothesis_id"])
    if champion_id != str(result["champion_hypothesis_id"]):
        raise ValueError("result champion does not match frozen checkpoint")
    if champion_id not in registry.models:
        raise ValueError("frozen champion model is missing")

    private_value = args.private_dir or config.get("data", {}).get("private_dir")
    if not private_value:
        raise ValueError("private evaluator directory must be supplied after model freeze")
    private_dir = _repository_path(str(private_value))
    private_oracle_path = private_dir / "oracle.json"
    private_bank_path = private_dir / "evaluation_trajectories.npz"
    private_manifest_path = private_dir / "manifest.json"
    expected_structure_path = private_dir / "expected_structure.json"
    oracle = RuleEvaluationOracle.from_private_files(private_oracle_path, private_bank_path)
    expected_structure = json.loads(expected_structure_path.read_text(encoding="utf-8"))
    heldout = load_expert_dataset(dataset_dir, "validation") + load_expert_dataset(dataset_dir, "test")
    workspace_x, workspace_y = load_public_workspace(dataset_dir)
    resolution = int(
        args.grid_resolution
        if args.grid_resolution is not None
        else config.get("loop", {}).get("grid_resolution", 100)
    )

    evaluated_ids = (
        list(registry.models)
        if args.diagnostic_all_hypotheses
        else [champion_id]
    )
    all_metrics: dict[str, dict[str, object]] = {}
    champion_grid = None
    for hypothesis_id in evaluated_ids:
        ensemble = registry.models[hypothesis_id]
        metrics, grid = evaluate_boundary(
            ensemble,
            library,
            oracle,
            heldout,
            workspace_x,
            workspace_y,
            resolution,
            device,
        )
        all_metrics[hypothesis_id] = metrics.to_dict()
        if hypothesis_id == champion_id:
            champion_grid = grid
    assert champion_grid is not None

    frozen_hashes_after = {
        "constraint_models.pt": _file_sha256(checkpoint_path),
        "result.json": _file_sha256(result_path),
    }
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("frozen training artifact changed during private evaluation")

    output_path = (
        _repository_path(args.output)
        if args.output
        else run_dir / "posthoc_evaluation.json"
    )
    payload = {
        "schema_version": 1,
        "evaluation_protocol": "frozen_checkpoint_posthoc_private",
        "task_instance_id": task_spec.task_instance_id,
        "champion_hypothesis_id": champion_id,
        "selection_status_frozen_before_private_evaluation": result.get("selection_status"),
        "frozen_artifact_sha256": frozen_hashes_before,
        "private_input_sha256": {
            "oracle.json": _file_sha256(private_oracle_path),
            "evaluation_trajectories.npz": _file_sha256(private_bank_path),
            "expected_structure.json": _file_sha256(expected_structure_path),
            "manifest.json": _file_sha256(private_manifest_path),
        },
        "champion_metrics": all_metrics[champion_id],
        "structure_metrics": evaluate_structure(
            result.get("champion_hypothesis"),
            expected_structure,
            selection_status=str(result.get("selection_status", "inconclusive")),
        ),
        "diagnostic_all_hypotheses_enabled": bool(args.diagnostic_all_hypotheses),
    }
    if args.diagnostic_all_hypotheses:
        payload["all_hypothesis_metrics_diagnostic_only"] = all_metrics
    payload = _json_safe(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_boundary(
        output_path.with_name("posthoc_learned_boundary.png"),
        champion_grid,
        heldout,
        [],
        oracle,
        f"Frozen {task_spec.suite_name} champion: {champion_id}",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
