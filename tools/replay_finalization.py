"""Replay final-model selection from a completed run without GPT or Oracle calls."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_modulo_cegis.data import FeatureLibrary, load_expert_dataset, load_public_workspace
from llm_modulo_cegis.evaluation import evaluate_boundary
from llm_modulo_cegis.hypotheses import compile_hypothesis, hypothesis_from_dict
from llm_modulo_cegis.learner import LearnerRegistry, TrainerConfig
from llm_modulo_cegis.loop import LoopConfig, SemanticNumericCEGIS
from llm_modulo_cegis.oracle import CircularEvaluationOracle
from llm_modulo_cegis.types import QueryBuffer, QueryRecord, Trajectory


def repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Completed output directory to replay.")
    parser.add_argument("--output", help="JSON report path; defaults inside run_dir.")
    parser.add_argument("--save-model", help="Optional path for the selected replay model state.")
    parser.add_argument("--background-weights", nargs="+", type=float)
    parser.add_argument("--unsafe-probes", type=int)
    parser.add_argument("--scratch-restarts", type=int)
    parser.add_argument("--disable-latent-witness", action="store_true")
    parser.add_argument("--champion-id")
    return parser.parse_args()


def load_records(run_dir: Path) -> list[QueryRecord]:
    arrays = np.load(run_dir / "oracle_queries.npz")
    audit = json.loads((run_dir / "oracle_query_log.json").read_text(encoding="utf-8"))
    if len(audit) != len(arrays["labels"]):
        raise ValueError("oracle NPZ and JSON log lengths differ")
    records: list[QueryRecord] = []
    for index, row in enumerate(audit):
        trajectory = Trajectory(
            arrays["observations"][index],
            arrays["actions"][index],
            metadata=dict(row.get("trajectory_metadata", {})),
        )
        records.append(
            QueryRecord(
                trajectory=trajectory,
                label=int(row["label"]),
                source=str(row["source"]),
                outer_round=int(row["outer_round"]),
                source_hypothesis_id=row.get("source_hypothesis_id"),
                predictions_before_query=dict(row.get("predictions_before_query", {})),
                scores_before_query=dict(row.get("scores_before_query", {})),
                uncertainties_before_query=dict(row.get("uncertainties_before_query", {})),
            )
        )
    return records


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    seed = int(config.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(config.get("device", "cpu")))
    library = FeatureLibrary()
    dataset_dir = repository_path(config["data"]["dataset_dir"])
    experts = load_expert_dataset(dataset_dir, "train")
    heldout = load_expert_dataset(dataset_dir, "validation") + load_expert_dataset(dataset_dir, "test")
    workspace_x, workspace_y = load_public_workspace(dataset_dir)
    loop_config = LoopConfig(**config.get("loop", {}))
    if args.background_weights is not None:
        loop_config = replace(
            loop_config,
            finalization_finetune_background_weights=tuple(args.background_weights),
        )
    if args.unsafe_probes is not None:
        loop_config = replace(
            loop_config,
            finalization_unsafe_volume_probe_count=args.unsafe_probes,
        )
    if args.scratch_restarts is not None:
        loop_config = replace(
            loop_config,
            finalization_scratch_restarts=args.scratch_restarts,
        )
    if args.disable_latent_witness:
        loop_config = replace(loop_config, finalization_disable_latent_witness=True)
    audit_count = min(loop_config.expert_structure_validation, max(0, len(experts) - 1))

    checkpoint = torch.load(run_dir / "constraint_models.pt", map_location=device)
    champion_id = str(args.champion_id or checkpoint["champion_hypothesis_id"])
    if champion_id not in checkpoint["compiled_hypotheses"]:
        raise KeyError(f"unknown checkpoint hypothesis: {champion_id}")
    hypothesis = hypothesis_from_dict(checkpoint["compiled_hypotheses"][champion_id])
    registry = LearnerRegistry(
        library,
        hidden_dims=config["model"]["hidden_dims"],
        ensemble_size=int(config["model"]["ensemble_size"]),
        seed=seed,
        device=device,
    )
    incumbent = registry.ensure(compile_hypothesis(hypothesis, library))
    incumbent.load_state_dict(checkpoint["models"][champion_id])

    runner = object.__new__(SemanticNumericCEGIS)
    runner.library = library
    runner.registry = registry
    runner.trainer_config = TrainerConfig(**config.get("trainer", {}))
    runner.config = loop_config
    runner.experts = experts[:-audit_count] if audit_count else experts
    runner.structure_audit_experts = experts[-audit_count:] if audit_count else []
    runner.buffer = QueryBuffer()
    for record in load_records(run_dir):
        runner.buffer.add(record)
    runner.seed = seed
    runner.device = device
    runner.progress = lambda message: print(message, flush=True)
    runner.finalization_diagnostics = {"applied": False}

    evaluation_oracle = CircularEvaluationOracle.from_private_file(
        dataset_dir / "private_evaluation" / "ground_truth.json"
    )
    before, _ = evaluate_boundary(
        incumbent, library, evaluation_oracle, heldout, workspace_x, workspace_y,
        loop_config.grid_resolution, device,
    )
    committed = runner._finalize_champion(champion_id)
    after, _ = evaluate_boundary(
        registry.models[champion_id], library, evaluation_oracle, heldout,
        workspace_x, workspace_y, loop_config.grid_resolution, device,
    )
    payload = {
        "source_run": str(run_dir),
        "champion_hypothesis_id": champion_id,
        "committed": committed,
        "evaluation_only_before": before.to_dict(),
        "evaluation_only_after": after.to_dict(),
        "finalization": runner.finalization_diagnostics,
    }
    output = Path(args.output).resolve() if args.output else run_dir / "finalization_replay.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.save_model:
        model_path = Path(args.save_model).resolve()
        torch.save(
            {
                "champion_hypothesis_id": champion_id,
                "hypothesis": hypothesis.to_dict(),
                "state_dict": registry.models[champion_id].state_dict(),
                "source_run": str(run_dir),
                "committed": committed,
            },
            model_path,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
