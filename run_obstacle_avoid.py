"""Run semantic--numeric CEGIS on legacy Obstacle2D or a public TaskSpec bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from llm_modulo_cegis.data import (
    FeatureLibrary,
    load_candidate_pool,
    load_expert_dataset,
    load_public_workspace,
    load_task_spec,
)
from llm_modulo_cegis.evidence import EvidenceCompiler, EvidenceConfig
from llm_modulo_cegis.falsifier import FalsifierConfig, HypothesisFalsifier
from llm_modulo_cegis.pool_falsifier import PoolHypothesisFalsifier
from llm_modulo_cegis.learner import LearnerRegistry, TrainerConfig
from llm_modulo_cegis.loop import LoopConfig, SemanticNumericCEGIS
from llm_modulo_cegis.oracle import (
    CircularEvaluationOracle,
    DeferredEvaluationOracle,
    RuleEvaluationOracle,
)
from llm_modulo_cegis.semantic import (
    EvidencePolicyReasoner,
    FrozenBankSemanticReasoner,
    LocalQwenSemanticReasoner,
    OpenAISemanticReasoner,
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(path: Path) -> str:
    """Return a stable repository-relative path when possible."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_reasoner(config: dict[str, object], semantic_config: SemanticConfig):
    backend = str(config.get("backend", "local_qwen"))
    if backend == "frozen_bank":
        return backend, FrozenBankSemanticReasoner(repository_path(str(config["hypothesis_bank"])))
    if backend == "evidence_policy":
        return backend, EvidencePolicyReasoner(semantic_config)
    if backend == "openai":
        return backend, OpenAISemanticReasoner(
            str(config.get("model", "gpt-5.6")),
            semantic_config,
            reasoning_effort=str(config.get("reasoning_effort", "medium")),
            max_output_tokens=int(config.get("max_output_tokens", 2400)),
            env_file=config.get("env_file"),
        )
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
    parser.add_argument("--backend", choices=("local_qwen", "openai", "evidence_policy", "frozen_bank"))
    parser.add_argument("--device")
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the top-level numeric seed and record it in resolved_config.yaml.",
    )
    parser.add_argument("--outer-rounds", type=int)
    parser.add_argument("--trainer-epochs", type=int)
    parser.add_argument(
        "--latent-witness-mode",
        choices=(
            "source_model",
            "source_model_legacy",
            "source_model_all_interventions",
            "novelty",
            "none",
        ),
        help="Override the trainer's latent violation-witness policy.",
    )
    parser.add_argument(
        "--hard-trajectory-alignment-weight",
        type=float,
        help="Add inference-time hard-max losses with this nonnegative weight.",
    )
    parser.add_argument(
        "--violation-pooling-mode",
        choices=("all_states", "source_anchor_changed_states"),
        help=(
            "Choose ordinary trajectory MIL or restrict source-model violation "
            "witnesses to selected-feature states changed from their safe anchor."
        ),
    )
    hull_gate_group = parser.add_mutually_exclusive_group()
    hull_gate_group.add_argument(
        "--enforce-linear-max-support-gate",
        dest="linear_max_support_gate_enforced",
        action="store_true",
        help=(
            "Make a repeated safe-anchor convex-support contradiction block "
            "linear max hypotheses from champion selection."
        ),
    )
    hull_gate_group.add_argument(
        "--audit-only-linear-max-support-gate",
        dest="linear_max_support_gate_enforced",
        action="store_false",
        help="Compute linear-max support diagnostics without enforcing the gate.",
    )
    parser.set_defaults(linear_max_support_gate_enforced=None)
    bootstrap_group = parser.add_mutually_exclusive_group()
    bootstrap_group.add_argument(
        "--bootstrap-queries",
        dest="bootstrap_queries",
        action="store_true",
        help="Opt into classic per-member query bootstrap (ablation only).",
    )
    bootstrap_group.add_argument(
        "--no-bootstrap-queries",
        dest="bootstrap_queries",
        action="store_false",
        help="Fit every ensemble member on the complete query buffer.",
    )
    parser.set_defaults(bootstrap_queries=None)
    coverage_group = parser.add_mutually_exclusive_group()
    coverage_group.add_argument(
        "--coverage-preserving-bootstrap",
        dest="bootstrap_ensure_full_coverage",
        action="store_true",
        help=(
            "Bootstrap each label pool, then append every omitted query once so "
            "each member retains full Oracle-evidence coverage."
        ),
    )
    coverage_group.add_argument(
        "--classic-bootstrap-coverage",
        dest="bootstrap_ensure_full_coverage",
        action="store_false",
        help="Use classic bootstrap without repairing omitted query records.",
    )
    parser.set_defaults(bootstrap_ensure_full_coverage=None)
    parser.add_argument("--freeze-revisions", action="store_true")
    parser.add_argument(
        "--initial-hypothesis-bank",
        help="Replay round-0 hypotheses from a hypothesis_bank.json artifact and freeze revisions.",
    )
    parser.add_argument(
        "--global-acquisition-only",
        action="store_true",
        help=(
            "Disable adaptive label-balance selection. The first pre-update "
            "candidate pool is shared; later pools may diverge after different labels."
        ),
    )
    parser.add_argument(
        "--legacy-false-unsafe-falsifier",
        action="store_true",
        help=(
            "Use the former smooth objective and one 0.08 radius while retaining "
            "the common anchor, hard-certificate, refinement, and acquisition policy."
        ),
    )
    parser.add_argument(
        "--legacy-false-unsafe-objective",
        action="store_true",
        help="Use the former smooth fixed-epsilon false-unsafe objective but keep the radius ladder.",
    )
    parser.add_argument(
        "--single-radius-false-unsafe",
        action="store_true",
        help="Use a single 0.08 false-unsafe trust radius while keeping the configured objective.",
    )
    parser.add_argument(
        "--false-unsafe-single-radius",
        type=float,
        metavar="RADIUS",
        help=(
            "Use one explicit positive false-unsafe radius. This is useful for "
            "separating ladder effects from a larger maximum trust region."
        ),
    )
    parser.add_argument(
        "--false-unsafe-radius-ladder",
        type=float,
        nargs="+",
        metavar="RADIUS",
        help=(
            "Use an explicit increasing false-unsafe radius ladder. The final "
            "radius is also recorded as the maximum trust radius."
        ),
    )
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="Query only the semantic backend; do not load demonstrations, private evaluation data, or the Oracle.",
    )
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
    if args.seed is not None:
        config["seed"] = args.seed
    if args.outer_rounds is not None:
        config["loop"]["outer_rounds"] = args.outer_rounds
    if args.trainer_epochs is not None:
        config["trainer"]["epochs"] = args.trainer_epochs
    if args.latent_witness_mode is not None:
        config["trainer"]["latent_witness_mode"] = args.latent_witness_mode
    if args.hard_trajectory_alignment_weight is not None:
        if args.hard_trajectory_alignment_weight < 0.0:
            raise ValueError("--hard-trajectory-alignment-weight must be nonnegative")
        config["trainer"]["hard_trajectory_alignment_weight"] = float(
            args.hard_trajectory_alignment_weight
        )
    if args.violation_pooling_mode is not None:
        config["trainer"]["violation_pooling_mode"] = str(
            args.violation_pooling_mode
        )
    if args.linear_max_support_gate_enforced is not None:
        config["evidence"]["linear_max_support_gate_enforced"] = bool(
            args.linear_max_support_gate_enforced
        )
    if args.bootstrap_queries is not None:
        config["trainer"]["bootstrap_queries"] = bool(args.bootstrap_queries)
    if args.bootstrap_ensure_full_coverage is not None:
        config["trainer"]["bootstrap_ensure_full_coverage"] = bool(
            args.bootstrap_ensure_full_coverage
        )
    if bool(config["trainer"].get("bootstrap_ensure_full_coverage", False)):
        config["trainer"]["bootstrap_queries"] = True
    if args.freeze_revisions:
        config["loop"]["freeze_revisions"] = True
    if args.initial_hypothesis_bank:
        config["semantic_reasoner"] = {
            "backend": "frozen_bank",
            "hypothesis_bank": str(repository_path(args.initial_hypothesis_bank).resolve()),
        }
        config["loop"]["freeze_revisions"] = True
    if args.global_acquisition_only:
        config["loop"]["reserve_label_seeking_queries"] = False
    if args.legacy_false_unsafe_objective:
        config["falsifier"]["false_unsafe_use_hard_margin"] = False
    if args.single_radius_false_unsafe and (
        args.false_unsafe_single_radius is not None or args.false_unsafe_radius_ladder
    ):
        raise ValueError(
            "--single-radius-false-unsafe cannot be combined with an explicit radius mode"
        )
    if args.single_radius_false_unsafe:
        config["falsifier"]["false_unsafe_trust_radius"] = 0.08
        config["falsifier"]["false_unsafe_radius_ladder"] = []
    if args.false_unsafe_single_radius is not None and args.false_unsafe_radius_ladder:
        raise ValueError(
            "--false-unsafe-single-radius and --false-unsafe-radius-ladder are mutually exclusive"
        )
    if args.false_unsafe_single_radius is not None:
        if args.false_unsafe_single_radius <= 0.0:
            raise ValueError("--false-unsafe-single-radius must be strictly positive")
        config["falsifier"]["false_unsafe_trust_radius"] = float(
            args.false_unsafe_single_radius
        )
        config["falsifier"]["false_unsafe_radius_ladder"] = []
    if args.false_unsafe_radius_ladder:
        radii = [float(value) for value in args.false_unsafe_radius_ladder]
        if any(not np.isfinite(value) or value <= 0.0 for value in radii):
            raise ValueError("--false-unsafe-radius-ladder values must be finite and positive")
        if any(right <= left for left, right in zip(radii, radii[1:])):
            raise ValueError("--false-unsafe-radius-ladder values must be strictly increasing")
        config["falsifier"]["false_unsafe_trust_radius"] = radii[-1]
        config["falsifier"]["false_unsafe_radius_ladder"] = radii
    if args.legacy_false_unsafe_falsifier:
        config["falsifier"]["false_unsafe_use_hard_margin"] = False
        config["falsifier"]["false_unsafe_trust_radius"] = 0.08
        config["falsifier"]["false_unsafe_radius_ladder"] = []

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = select_device(str(config.get("device", "auto")))
    dataset_dir = repository_path(config["data"]["dataset_dir"])
    output_dir = repository_path(config["output_dir"])
    task_spec = load_task_spec(dataset_dir) if (dataset_dir / "task_spec.json").is_file() else None
    if task_spec is not None:
        configured_description = str(config.get("task_description", "")).strip()
        if configured_description and configured_description != task_spec.task_description.strip():
            raise ValueError(
                "config task_description does not match the learner-visible task_spec.json"
            )
        config["task_description"] = task_spec.task_description
        # ``max_step`` is the feasibility rule for the planar waypoint
        # adapter.  Higher-dimensional adapters validate their own public
        # dynamics instead of interpreting an acceleration command as a
        # waypoint displacement.
        if task_spec.trajectory_adapter == "planar_waypoint_v1":
            configured_max_step = float(
                config.get("falsifier", {}).get("max_step", task_spec.max_step)
            )
            if abs(configured_max_step - task_spec.max_step) > 1.0e-12:
                raise ValueError("falsifier.max_step must match the public TaskSpec")
            config.setdefault("falsifier", {})["max_step"] = task_spec.max_step
    private_dir_value = config.get("data", {}).get("private_dir")
    membership_oracle_value = config.get("data", {}).get("membership_oracle_path")
    private_dir = repository_path(str(private_dir_value)) if private_dir_value else None
    inline_private_evaluation = bool(
        config.get("data", {}).get("inline_private_evaluation", False)
    )
    private_oracle_path: Path | None = None
    private_evaluation_path: Path | None = None
    if task_spec is not None and not args.semantic_only:
        if membership_oracle_value:
            private_oracle_path = repository_path(str(membership_oracle_value))
        elif private_dir is not None:
            private_oracle_path = private_dir / "oracle.json"
        else:
            raise ValueError(
                "TaskSpec training requires data.membership_oracle_path; "
                "data.private_dir is supported only as a legacy diagnostic shortcut"
            )
        if private_dir is not None:
            private_evaluation_path = private_dir / "evaluation_trajectories.npz"
        if not private_oracle_path.is_file():
            raise FileNotFoundError(f"invalid private benchmark bundle: {private_dir}")
        if inline_private_evaluation and (
            private_evaluation_path is None or not private_evaluation_path.is_file()
        ):
            raise FileNotFoundError(f"missing private evaluation archive: {private_evaluation_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    implementation_files = [
        PACKAGE_ROOT / "run_obstacle_avoid.py",
        PACKAGE_ROOT / "experiments" / "run_falsifier_multiseed.py",
        PACKAGE_ROOT / "pyproject.toml",
        PACKAGE_ROOT / "requirements.txt",
        *sorted((PACKAGE_ROOT / "src" / "llm_modulo_cegis").glob("*.py")),
    ]
    input_files = [config_path, resolved_config_path]
    capability_input_files: list[Path] = []
    evaluation_only_input_files: list[Path] = []
    semantic_reasoner_config = config.get("semantic_reasoner", {})
    if (
        isinstance(semantic_reasoner_config, dict)
        and semantic_reasoner_config.get("backend") == "frozen_bank"
        and semantic_reasoner_config.get("hypothesis_bank")
    ):
        input_files.append(repository_path(str(semantic_reasoner_config["hypothesis_bank"])))
    if args.semantic_only and task_spec is not None:
        input_files.append(dataset_dir / "task_spec.json")
    if not args.semantic_only:
        input_files.extend(sorted(path for path in dataset_dir.rglob("*") if path.is_file()))
        if private_oracle_path is not None:
            capability_input_files.append(private_oracle_path)
        if inline_private_evaluation and private_evaluation_path is not None:
            evaluation_only_input_files.append(private_evaluation_path)
            private_manifest = private_dir / "manifest.json" if private_dir is not None else None
            if private_manifest is not None and private_manifest.is_file():
                evaluation_only_input_files.append(private_manifest)
    (output_dir / "implementation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "files": {
                    manifest_path(path): file_sha256(path)
                    for path in implementation_files
                },
                "inputs": {
                    manifest_path(path): file_sha256(path)
                    for path in input_files
                },
                "capability_backing_inputs": {
                    manifest_path(path): file_sha256(path)
                    for path in capability_input_files
                },
                "evaluation_only_inputs": {
                    manifest_path(path): file_sha256(path)
                    for path in evaluation_only_input_files
                },
                "runtime": {
                    "seed": seed,
                    "device": str(device),
                    "python_version": platform.python_version(),
                    "python_executable": str(Path(sys.executable).resolve()),
                    "platform": platform.platform(),
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                    "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "environment": {
                        name: os.environ.get(name)
                        for name in (
                            "PYTHONHASHSEED",
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "CUBLAS_WORKSPACE_CONFIG",
                        )
                    },
                    "packages": {
                        name: installed_version(name)
                        for name in (
                            "numpy",
                            "torch",
                            "matplotlib",
                            "PyYAML",
                            "openai",
                        )
                    },
                    "argv": sys.argv,
                    "private_evaluation_mode": (
                        "inline_diagnostic" if inline_private_evaluation else "deferred_posthoc"
                    ),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    library = FeatureLibrary.from_task_spec(task_spec) if task_spec is not None else FeatureLibrary()
    semantic_config = SemanticConfig(**config.get("semantic", {}))
    backend, reasoner = build_reasoner(config["semantic_reasoner"], semantic_config)
    if isinstance(reasoner, FrozenBankSemanticReasoner):
        (output_dir / "frozen_bank_source.json").write_text(
            json.dumps(reasoner.source_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.semantic_only:
        hypotheses = reasoner.propose_initial(str(config["task_description"]), library)
        interactions = reasoner.interactions
        payload = {
            "semantic_backend": backend,
            "hypotheses": [item.to_dict() for item in hypotheses],
            "llm_fallbacks": sum(bool(item.get("used_fallback", False)) for item in interactions),
            "llm_augmentations": sum(bool(item.get("used_augmentation", False)) for item in interactions),
            "accepted_llm_hypotheses": sum(int(item.get("accepted_llm_count", 0) or 0) for item in interactions),
            "interactions": interactions,
            "oracle_queries": 0,
        }
        (output_dir / "semantic_initial_smoke.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return 0

    experts = load_expert_dataset(dataset_dir, "train")
    heldout = load_expert_dataset(dataset_dir, "validation") + load_expert_dataset(dataset_dir, "test")
    workspace_x, workspace_y = load_public_workspace(dataset_dir)
    registry = LearnerRegistry(
        library,
        hidden_dims=config["model"]["hidden_dims"],
        ensemble_size=int(config["model"]["ensemble_size"]),
        seed=seed,
        device=device,
    )
    trainer_config = TrainerConfig(**config.get("trainer", {}))
    evidence_compiler = EvidenceCompiler(library, EvidenceConfig(**config.get("evidence", {})), device)
    falsifier_config = FalsifierConfig(**config.get("falsifier", {}))
    if task_spec is not None and task_spec.trajectory_adapter == "carrywater_active_v1":
        from llm_modulo_cegis.carrywater_active import validate_trajectory

        candidates = load_candidate_pool(dataset_dir)
        invalid_public = []
        for item in (*experts, *heldout, *candidates):
            validity = validate_trajectory(item)
            if not validity.valid:
                invalid_public.append((item.metadata.get("trajectory_id"), validity))
        if invalid_public:
            details = [
                f"{trajectory_id}:{result.reason}"
                for trajectory_id, result in invalid_public[:5]
            ]
            raise ValueError(
                "CarryWaterActive contains publicly invalid trajectories: "
                + ", ".join(details)
            )
        falsifier = PoolHypothesisFalsifier(
            library,
            falsifier_config,
            candidates,
            device,
            validator=validate_trajectory,
        )
        print(
            f"trajectory_adapter=carrywater_active_v1 public_candidates={len(candidates)} "
            "candidate_source=independent_control_space_rollouts",
            flush=True,
        )
    else:
        falsifier = HypothesisFalsifier(
            library,
            falsifier_config,
            workspace_x,
            workspace_y,
            device,
        )
    if task_spec is not None:
        assert private_oracle_path is not None
        membership_oracle_source = RuleEvaluationOracle.from_private_files(private_oracle_path)
        if inline_private_evaluation:
            assert private_evaluation_path is not None
            evaluation_oracle = RuleEvaluationOracle.from_private_files(
                private_oracle_path,
                private_evaluation_path,
            )
        else:
            evaluation_oracle = DeferredEvaluationOracle()
    else:
        evaluation_oracle = CircularEvaluationOracle.from_private_file(
            dataset_dir / "private_evaluation" / "ground_truth.json"
        )
        membership_oracle_source = evaluation_oracle
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
        oracle=membership_oracle_source.membership_view(),
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
    training_artifacts = {
        name: file_sha256(output_dir / name)
        for name in (
            "constraint_models.pt",
            "result.json",
            "hypothesis_bank.json",
            "oracle_queries.npz",
            "oracle_query_log.json",
            "evidence_history.json",
            "evaluation_history.json",
            "query_diagnostics.json",
            "threshold_calibration_history.json",
            "qualified_checkpoint_history.json",
            "stage_diagnostics.json",
        )
    }
    freeze_manifest = {
        "schema_version": 1,
        "training_complete": True,
        "champion_hypothesis_id": result.champion_hypothesis_id,
        "selection_status": result.selection_status,
        "private_evaluation_mode": (
            "inline_diagnostic" if inline_private_evaluation else "deferred_posthoc"
        ),
        "private_evaluation_loaded_before_freeze": inline_private_evaluation,
        "training_artifact_sha256": training_artifacts,
        "implementation_manifest_sha256": file_sha256(
            output_dir / "implementation_manifest.json"
        ),
    }
    (output_dir / "freeze_manifest.json").write_text(
        json.dumps(freeze_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # ``loop.py`` serializes non-finite evaluation placeholders as JSON null.
    # Reuse that strict artifact for console output instead of emitting the
    # non-standard JavaScript tokens NaN/Infinity from the in-memory object.
    print((output_dir / "result.json").read_text(encoding="utf-8").rstrip(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
