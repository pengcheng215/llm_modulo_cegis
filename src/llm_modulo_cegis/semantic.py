"""LLM and deterministic semantic reasoners for outer-loop synthesis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .data import FeatureLibrary
from .hypotheses import (
    ConstraintHypothesis,
    HypothesisBank,
    RevisionAction,
    extract_json_array_objects,
    hypothesis_from_dict,
    revision_action_from_dict,
    validate_hypothesis,
    validate_revision_action,
)
from .types import HypothesisEvidence, InterventionSpec


class SemanticReasoner(Protocol):
    interactions: list[dict[str, Any]]

    def propose_initial(
        self,
        task_description: str,
        feature_library: FeatureLibrary,
    ) -> list[ConstraintHypothesis]: ...

    def revise(
        self,
        task_description: str,
        feature_library: FeatureLibrary,
        bank: HypothesisBank,
        evidence_report: dict[str, object],
        evidence: list[HypothesisEvidence],
        outer_round: int,
    ) -> list[RevisionAction]: ...


@dataclass(frozen=True)
class SemanticConfig:
    beam_width: int = 3
    prune_per_round: int = 2
    max_initial_hypotheses: int = 6
    allow_fallback: bool = True


def canonical_initial_hypotheses() -> list[ConstraintHypothesis]:
    """A deliberately mixed bank used by tests and non-LLM ablations."""
    return [
        ConstraintHypothesis(
            "h_x_position",
            "horizontal forbidden band",
            ("x_position",),
            "joint",
            "forbidden_region",
            "max",
            "mlp",
            "stress horizontal position while preserving the other observed behavior",
            "Tests whether horizontal position alone explains trajectory safety.",
        ),
        ConstraintHypothesis(
            "h_y_position",
            "vertical forbidden band",
            ("y_position",),
            "joint",
            "forbidden_region",
            "max",
            "mlp",
            "stress vertical position while preserving endpoints",
            "Tests whether vertical position alone explains trajectory safety.",
        ),
        ConstraintHypothesis(
            "h_planar_joint",
            "joint planar forbidden region",
            ("x_position", "y_position"),
            "joint",
            "forbidden_region",
            "max",
            "mlp",
            "shorten the demonstrated detour in the coupled planar space",
            "Obstacle avoidance is plausibly a coupled relation between planar coordinates.",
        ),
        ConstraintHypothesis(
            "h_planar_independent",
            "independent planar forbidden bands",
            ("x_position", "y_position"),
            "independent",
            "forbidden_region",
            "max",
            "mlp",
            "stress each planar coordinate separately",
            "Competes with the joint hypothesis to test whether coupling is necessary.",
        ),
        ConstraintHypothesis(
            "h_speed",
            "maximum planar speed",
            ("speed",),
            "joint",
            "upper_bound",
            "max",
            "mlp",
            "increase local speed while preserving the geometric path",
            "Tests a dynamic rather than geometric explanation.",
        ),
    ]


class EvidencePolicyReasoner:
    """Deterministic evidence policy used for tests and the no-LLM ablation."""

    def __init__(self, config: SemanticConfig) -> None:
        self.config = config
        self.interactions: list[dict[str, Any]] = []

    def propose_initial(
        self,
        task_description: str,
        feature_library: FeatureLibrary,
    ) -> list[ConstraintHypothesis]:
        hypotheses = canonical_initial_hypotheses()
        for hypothesis in hypotheses:
            validate_hypothesis(hypothesis, feature_library)
        self.interactions.append(
            {
                "phase": "initial",
                "backend": "evidence_policy",
                "task_description": task_description,
                "parsed": [item.to_dict() for item in hypotheses],
            }
        )
        return hypotheses

    def revise(
        self,
        task_description: str,
        feature_library: FeatureLibrary,
        bank: HypothesisBank,
        evidence_report: dict[str, object],
        evidence: list[HypothesisEvidence],
        outer_round: int,
    ) -> list[RevisionAction]:
        active_ids = {item.hypothesis_id for item in bank.active()}
        ordered = sorted(
            (item for item in evidence if item.hypothesis_id in active_ids),
            key=lambda item: item.selection_score,
            reverse=True,
        )
        if not ordered:
            raise RuntimeError("semantic revision received no active evidence")
        champion = ordered[0]
        # A semantic reasoner should not collapse to the best training fit when
        # an almost-as-well-supported coupled spatial explanation matches an
        # obstacle task. This is a task-description prior, never hidden geometry.
        lowered_task = task_description.lower()
        if "obstacle" in lowered_task or "detour" in lowered_task:
            coupled_spatial = [
                item
                for item in ordered
                if bank.get(item.hypothesis_id).coupling == "joint"
                and {"x_position", "y_position"}.issubset(bank.get(item.hypothesis_id).variables)
            ]
            if coupled_spatial and coupled_spatial[0].selection_score >= champion.selection_score - 0.20:
                champion = coupled_spatial[0]
        actions = [
            RevisionAction(
                "retain_and_query",
                champion.hypothesis_id,
                "Highest current leakage-safe evidence score; retain for another falsification round.",
            )
        ]
        kind = "model_false_safe" if champion.false_safe_count >= champion.false_unsafe_count else "boundary_uncertainty"
        actions.append(
            RevisionAction(
                "propose_intervention",
                champion.hypothesis_id,
                "Target the dominant remaining prediction error of the leading hypothesis.",
                intervention=InterventionSpec(
                    champion.hypothesis_id,
                    kind,
                    variable=bank.get(champion.hypothesis_id).variables[0],
                    rationale="Evidence-directed next query.",
                ),
            )
        )
        removable = max(0, len(ordered) - self.config.beam_width)
        prune_count = min(removable, self.config.prune_per_round)
        prune_candidates = sorted(
            (item for item in ordered if item.hypothesis_id != champion.hypothesis_id),
            key=lambda item: item.selection_score,
        )
        for item in prune_candidates[:prune_count]:
            if item.evidence_sufficient:
                actions.append(
                    RevisionAction(
                        "retire_hypothesis",
                        item.hypothesis_id,
                        "Lower evidence score after both safe and violation trajectory observations.",
                    )
                )
        self.interactions.append(
            {
                "phase": "revision",
                "backend": "evidence_policy",
                "outer_round": outer_round,
                "input_evidence": evidence_report,
                "parsed": [action.to_dict() for action in actions],
            }
        )
        return actions


class LocalQwenSemanticReasoner:
    """Qwen proposes a bank, then revises it from verifier evidence every round."""

    def __init__(
        self,
        model_name_or_path: str,
        config: SemanticConfig,
        *,
        max_new_tokens: int = 1200,
        local_files_only: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("local Qwen backend requires torch and transformers") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        kwargs: dict[str, Any] = {"local_files_only": local_files_only, "trust_remote_code": False}
        if torch.cuda.is_available():
            kwargs["dtype"] = torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if torch.cuda.is_available():
            self._model.to("cuda")
        self._model.eval()
        self.config = config
        self.max_new_tokens = int(max_new_tokens)
        self.interactions: list[dict[str, Any]] = []
        self._fallback = EvidencePolicyReasoner(config)

    def _generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "Return exactly one JSON object with no markdown or commentary."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = outputs[0, inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def propose_initial(
        self,
        task_description: str,
        feature_library: FeatureLibrary,
    ) -> list[ConstraintHypothesis]:
        prompt = build_initial_prompt(task_description, feature_library, self.config.max_initial_hypotheses)
        raw = self._generate(prompt)
        error: str | None = None
        used_fallback = False
        candidate_errors: list[str] = []
        try:
            values = extract_json_array_objects(raw, "hypotheses")
            hypotheses = []
            signatures: set[tuple[Any, ...]] = set()
            for index, value in enumerate(values[: self.config.max_initial_hypotheses]):
                try:
                    hypothesis = hypothesis_from_dict(value)
                    validate_hypothesis(hypothesis, feature_library)
                    if hypothesis.signature() in signatures:
                        raise ValueError("duplicate hypothesis structure")
                    hypotheses.append(hypothesis)
                    signatures.add(hypothesis.signature())
                except Exception as exc:
                    candidate_errors.append(f"candidate[{index}]: {type(exc).__name__}: {exc}")
            if len(hypotheses) < 2:
                used_fallback = True
                for fallback in canonical_initial_hypotheses():
                    if fallback.signature() in signatures:
                        continue
                    hypotheses.append(fallback)
                    signatures.add(fallback.signature())
                    if len(hypotheses) >= self.config.max_initial_hypotheses:
                        break
            if len(hypotheses) < 2:
                raise ValueError("fewer than two valid or safely augmented hypotheses")
            if candidate_errors:
                error = "partial rejection: " + " | ".join(candidate_errors)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not self.config.allow_fallback:
                raise
            used_fallback = True
            hypotheses = canonical_initial_hypotheses()[: self.config.max_initial_hypotheses]
        self.interactions.append(
            {
                "phase": "initial",
                "backend": "local_qwen",
                "prompt": prompt,
                "raw_output": raw,
                "parse_error": error,
                "candidate_errors": candidate_errors,
                "used_fallback": used_fallback,
                "parsed": [item.to_dict() for item in hypotheses],
            }
        )
        return hypotheses

    def revise(
        self,
        task_description: str,
        feature_library: FeatureLibrary,
        bank: HypothesisBank,
        evidence_report: dict[str, object],
        evidence: list[HypothesisEvidence],
        outer_round: int,
    ) -> list[RevisionAction]:
        prompt = build_revision_prompt(task_description, feature_library, bank, evidence_report)
        raw = self._generate(prompt)
        error: str | None = None
        used_fallback = False
        action_errors: list[str] = []
        try:
            raw_actions = extract_json_array_objects(raw, "actions")
            actions = []
            for index, raw_action in enumerate(raw_actions):
                try:
                    action = revision_action_from_dict(raw_action)
                    validate_revision_action(action, bank, feature_library)
                    actions.append(action)
                except Exception as exc:
                    action_errors.append(f"action[{index}]: {type(exc).__name__}: {exc}")
            retained = [action for action in actions if action.action == "retain_and_query"]
            if len(retained) != 1:
                raise ValueError("revision must contain exactly one retain_and_query champion action")
            champion_id = retained[0].target_hypothesis_id
            destructive = {
                "retire_hypothesis",
                "change_variables",
                "change_coupling",
                "change_temporal_operator",
                "change_model_family",
                "split_hypothesis",
            }
            if any(action.action in destructive and action.target_hypothesis_id == champion_id for action in actions):
                raise ValueError("revision cannot retain and retire/replace the same champion")
            if action_errors:
                error = "partial rejection: " + " | ".join(action_errors)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not self.config.allow_fallback:
                raise
            used_fallback = True
            actions = self._fallback.revise(
                task_description,
                feature_library,
                bank,
                evidence_report,
                evidence,
                outer_round,
            )
        self.interactions.append(
            {
                "phase": "revision",
                "backend": "local_qwen",
                "outer_round": outer_round,
                "prompt": prompt,
                "raw_output": raw,
                "parse_error": error,
                "action_errors": action_errors,
                "used_fallback": used_fallback,
                "input_evidence": evidence_report,
                "parsed": [action.to_dict() for action in actions],
            }
        )
        return actions


def build_initial_prompt(task_description: str, library: FeatureLibrary, maximum: int) -> str:
    schema = json.dumps(library.schema_for_prompt(), ensure_ascii=False, indent=2)
    return f"""You are the semantic synthesizer in an inverse constraint learning system.

Task description:
{task_description.strip()}

Available learner-visible variables, including deterministic derived features:
{schema}

Propose a diverse bank of two to {maximum} competing qualitative hypotheses. Include plausible alternatives so trajectory evidence can falsify them. Keep every name, risk_direction, and rationale concise (at most twelve words). Never output coordinates, centers, radii, thresholds, boundary points, or ground-truth claims. The neural learner estimates all numerical boundaries.

Allowed values:
- coupling: joint | independent
- relation: forbidden_region | upper_bound | lower_bound
- temporal_operator: max | mean | last
- model_family: mlp | linear

Return exactly:
{{"hypotheses":[{{"hypothesis_id":"h_identifier","name":"description","variables":["available_name"],"coupling":"joint","relation":"forbidden_region","temporal_operator":"max","model_family":"mlp","risk_direction":"qualitative intervention","rationale":"why this structure is plausible"}}]}}"""


def build_revision_prompt(
    task_description: str,
    library: FeatureLibrary,
    bank: HypothesisBank,
    report: dict[str, object],
) -> str:
    active = [item.to_dict() for item in bank.active()]
    return f"""You are the outer semantic loop of an LLM-Modulo CEGIS system. An external neural learner and trajectory-membership Oracle have tested your hypotheses. Revise the search space; do not guess the numerical boundary.

Task:
{task_description.strip()}

Available variables:
{json.dumps(library.schema_for_prompt(), ensure_ascii=False)}

Active hypotheses:
{json.dumps(active, ensure_ascii=False, indent=2)}

Verifier evidence (trajectory-level only; no hidden geometry):
{json.dumps(report, ensure_ascii=False, indent=2)}

Allowed actions:
- retain_and_query
- retire_hypothesis
- change_variables, change_coupling, change_temporal_operator, change_model_family: include one complete replacement hypothesis with a new id
- split_hypothesis: include two or more complete replacements
- add_hypothesis: include one complete replacement
- propose_intervention: intervention.kind is model_false_safe | model_false_unsafe | boundary_uncertainty | shortcut | local_feature_stress

For an intervention use {{"target_hypothesis_id":"existing id","kind":"allowed kind","variable":"available variable or null","preserve_endpoints":true,"rationale":"reason"}}.
Return one to three actions total. If uncertain, return only retain_and_query and one propose_intervention. For every action provide action, target_hypothesis_id, rationale, replacement or null, replacements as a list, and intervention or null. Include exactly one retain_and_query action naming the current semantic champion. Return exactly {{"actions":[...]}}. Never use evaluation IoU, obstacle geometry, numerical thresholds, or state-level labels."""
