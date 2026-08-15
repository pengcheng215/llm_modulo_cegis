"""Run the complete semantic--numeric CEGIS architecture on ObstacleAvoid."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from llm_modulo_cegis.data import FeatureLibrary, load_expert_dataset, load_public_workspace
from llm_modulo_cegis.evidence import EvidenceCompiler, EvidenceConfig
from llm_modulo_cegis.falsifier import FalsifierConfig, HypothesisFalsifier
from llm_modulo_cegis.learner import LearnerRegistry, TrainerConfig
from llm_modulo_cegis.loop import LoopConfig, SemanticNumericCEGIS
from llm_modulo_cegis.oracle import CircularEvaluationOracle
from llm_modulo_cegis.semantic import (
    EvidencePolicyReasoner,
    LocalQwenSemanticReasoner,
    SemanticConfig,
)


def repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def build_reasoner(config: dict[str, object], semantic_config: SemanticConfig):
    backend = str(config.get("backend", "local_qwen"))
    if backend == "evidence_policy":
        return backend, EvidencePolicyReasoner(semantic_config)
    if backend != "local_qwen":
        raise ValueError(f"unknown semantic backend: {backend}")
    model_value = str(config["model_name_or_path"])
    model_path = repository_path(model_value)
    resolved_model = str(model_path) if model_path.exists() else model_value
    return backend, LocalQwenSemanticReasoner(
        resolved_model,
        semantic_config,
        max_new_tokens=int(config.get("max_new_tokens", 1200)),
        local_files_only=bool(config.get("local_files_only", True)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PACKAGE_ROOT / "configs" / "obstacle_avoid_qwen.yaml"),
    )
    parser.add_argument("--output")
    parser.add_argument("--backend", choices=("local_qwen", "evidence_policy"))
    parser.add_argument("--device")
    parser.add_argument("--outer-rounds", type=int)
    parser.add_argument("--trainer-epochs", type=int)
    parser.add_argument("--freeze-revisions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.output:
        config["output_dir"] = args.output
    if args.backend:
        config["semantic_reasoner"]["backend"] = args.backend
    if args.device:
        config["device"] = args.device
    if args.outer_rounds is not None:
        config["loop"]["outer_rounds"] = args.outer_rounds
    if args.trainer_epochs is not None:
        config["trainer"]["epochs"] = args.trainer_epochs
    if args.freeze_revisions:
        config["loop"]["freeze_revisions"] = True

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = select_device(str(config.get("device", "auto")))
    dataset_dir = repository_path(config["data"]["dataset_dir"])
    output_dir = repository_path(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    experts = load_expert_dataset(dataset_dir, "train")
    heldout = load_expert_dataset(dataset_dir, "validation") + load_expert_dataset(dataset_dir, "test")
    workspace_x, workspace_y = load_public_workspace(dataset_dir)
    library = FeatureLibrary()
    semantic_config = SemanticConfig(**config.get("semantic", {}))
    backend, reasoner = build_reasoner(config["semantic_reasoner"], semantic_config)
    registry = LearnerRegistry(
        library,
        hidden_dims=config["model"]["hidden_dims"],
        ensemble_size=int(config["model"]["ensemble_size"]),
        seed=seed,
        device=device,
    )
    trainer_config = TrainerConfig(**config.get("trainer", {}))
    evidence_compiler = EvidenceCompiler(library, EvidenceConfig(**config.get("evidence", {})), device)
    falsifier = HypothesisFalsifier(
        library,
        FalsifierConfig(**config.get("falsifier", {})),
        workspace_x,
        workspace_y,
        device,
    )
    evaluation_oracle = CircularEvaluationOracle.from_private_file(
        dataset_dir / "private_evaluation" / "ground_truth.json"
    )
    print(
        f"semantic_backend={backend} device={device} train_experts={len(experts)} heldout_experts={len(heldout)}",
        flush=True,
    )
    runner = SemanticNumericCEGIS(
        task_description=str(config["task_description"]),
        feature_library=library,
        reasoner=reasoner,
        registry=registry,
        trainer_config=trainer_config,
        evidence_compiler=evidence_compiler,
        falsifier=falsifier,
        oracle=evaluation_oracle.membership_view(),
        evaluation_oracle=evaluation_oracle,
        experts=experts,
        heldout_experts=heldout,
        workspace_x=workspace_x,
        workspace_y=workspace_y,
        loop_config=LoopConfig(**config.get("loop", {})),
        output_dir=output_dir,
        seed=seed,
        device=device,
        progress=lambda value: print(value, flush=True),
    )
    result = runner.run()
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
