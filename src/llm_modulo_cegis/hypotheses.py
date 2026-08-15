"""Typed hypothesis IR, compiler, bank, and revision actions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .data import FeatureLibrary
from .types import InterventionSpec


ALLOWED_COUPLINGS = {"joint", "independent"}
ALLOWED_RELATIONS = {"forbidden_region", "upper_bound", "lower_bound"}
ALLOWED_TEMPORAL_OPERATORS = {"max", "mean", "last"}
ALLOWED_MODEL_FAMILIES = {"mlp", "linear"}
ALLOWED_ACTIONS = {
    "retain_and_query",
    "retire_hypothesis",
    "change_variables",
    "change_coupling",
    "change_temporal_operator",
    "change_model_family",
    "split_hypothesis",
    "add_hypothesis",
    "propose_intervention",
}
ALLOWED_INTERVENTIONS = {
    "model_false_safe",
    "model_false_unsafe",
    "boundary_uncertainty",
    "shortcut",
    "local_feature_stress",
}


@dataclass(frozen=True)
class ConstraintHypothesis:
    """Qualitative constraint structure; it contains no numerical boundary."""

    hypothesis_id: str
    name: str
    variables: tuple[str, ...]
    coupling: str
    relation: str
    temporal_operator: str
    model_family: str
    risk_direction: str
    rationale: str
    parent_id: str | None = None
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variables"] = list(self.variables)
        return payload

    def signature(self) -> tuple[Any, ...]:
        return (
            self.variables,
            self.coupling,
            self.relation,
            self.temporal_operator,
            self.model_family,
        )


@dataclass(frozen=True)
class CompiledHypothesis:
    hypothesis: ConstraintHypothesis
    input_low: tuple[float, ...]
    input_high: tuple[float, ...]

    @property
    def variables(self) -> tuple[str, ...]:
        return self.hypothesis.variables


@dataclass(frozen=True)
class RevisionAction:
    action: str
    target_hypothesis_id: str | None
    rationale: str
    replacement: ConstraintHypothesis | None = None
    replacements: tuple[ConstraintHypothesis, ...] = ()
    intervention: InterventionSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target_hypothesis_id": self.target_hypothesis_id,
            "rationale": self.rationale,
            "replacement": None if self.replacement is None else self.replacement.to_dict(),
            "replacements": [item.to_dict() for item in self.replacements],
            "intervention": None if self.intervention is None else self.intervention.to_dict(),
        }


@dataclass
class BankEntry:
    hypothesis: ConstraintHypothesis
    status: str = "active"
    retired_round: int | None = None
    retired_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.to_dict(),
            "status": self.status,
            "retired_round": self.retired_round,
            "retired_reason": self.retired_reason,
        }


@dataclass
class HypothesisBank:
    """Versioned population rather than a single irreversible guess."""

    entries: dict[str, BankEntry] = field(default_factory=dict)
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_hypotheses(
        cls,
        hypotheses: list[ConstraintHypothesis],
        feature_library: FeatureLibrary,
    ) -> "HypothesisBank":
        bank = cls()
        for hypothesis in hypotheses:
            compile_hypothesis(hypothesis, feature_library)
            bank.add(hypothesis, outer_round=0, reason="initial proposal")
        if not bank.active():
            raise ValueError("initial hypothesis bank cannot be empty")
        return bank

    def active(self) -> list[ConstraintHypothesis]:
        return [entry.hypothesis for entry in self.entries.values() if entry.status == "active"]

    def get(self, hypothesis_id: str) -> ConstraintHypothesis:
        if hypothesis_id not in self.entries:
            raise KeyError(hypothesis_id)
        return self.entries[hypothesis_id].hypothesis

    def add(self, hypothesis: ConstraintHypothesis, *, outer_round: int, reason: str) -> None:
        if hypothesis.hypothesis_id in self.entries:
            raise ValueError(f"duplicate hypothesis id: {hypothesis.hypothesis_id}")
        active_signatures = {item.signature() for item in self.active()}
        if hypothesis.signature() in active_signatures:
            raise ValueError(f"duplicate active hypothesis structure: {hypothesis.hypothesis_id}")
        self.entries[hypothesis.hypothesis_id] = BankEntry(hypothesis=hypothesis)
        self.audit_log.append(
            {"outer_round": outer_round, "event": "add", "hypothesis_id": hypothesis.hypothesis_id, "reason": reason}
        )

    def retire(self, hypothesis_id: str, *, outer_round: int, reason: str) -> None:
        entry = self.entries.get(hypothesis_id)
        if entry is None:
            raise ValueError(f"cannot retire unknown hypothesis: {hypothesis_id}")
        if entry.status != "active":
            return
        entry.status = "retired"
        entry.retired_round = outer_round
        entry.retired_reason = reason
        self.audit_log.append(
            {"outer_round": outer_round, "event": "retire", "hypothesis_id": hypothesis_id, "reason": reason}
        )

    def apply_actions(
        self,
        actions: list[RevisionAction],
        feature_library: FeatureLibrary,
        *,
        outer_round: int,
        minimum_active: int = 1,
    ) -> list[InterventionSpec]:
        """Validate every LLM action before mutating the bank."""
        interventions: list[InterventionSpec] = []
        for action in actions:
            validate_revision_action(action, self, feature_library)
            if action.action == "retain_and_query":
                self.audit_log.append(
                    {
                        "outer_round": outer_round,
                        "event": action.action,
                        "hypothesis_id": action.target_hypothesis_id,
                        "reason": action.rationale,
                    }
                )
            elif action.action == "retire_hypothesis":
                if len(self.active()) > minimum_active:
                    self.retire(action.target_hypothesis_id or "", outer_round=outer_round, reason=action.rationale)
                else:
                    self.audit_log.append(
                        {
                            "outer_round": outer_round,
                            "event": "retire_rejected_minimum_active",
                            "hypothesis_id": action.target_hypothesis_id,
                            "reason": action.rationale,
                        }
                    )
            elif action.action in {
                "change_variables",
                "change_coupling",
                "change_temporal_operator",
                "change_model_family",
            }:
                assert action.replacement is not None
                self.retire(action.target_hypothesis_id or "", outer_round=outer_round, reason=action.rationale)
                self.add(action.replacement, outer_round=outer_round, reason=action.action)
            elif action.action == "split_hypothesis":
                self.retire(action.target_hypothesis_id or "", outer_round=outer_round, reason=action.rationale)
                for replacement in action.replacements:
                    self.add(replacement, outer_round=outer_round, reason="split_hypothesis")
            elif action.action == "add_hypothesis":
                assert action.replacement is not None
                self.add(action.replacement, outer_round=outer_round, reason=action.rationale)
            elif action.action == "propose_intervention":
                assert action.intervention is not None
                interventions.append(action.intervention)
                self.audit_log.append(
                    {
                        "outer_round": outer_round,
                        "event": action.action,
                        "hypothesis_id": action.target_hypothesis_id,
                        "intervention": action.intervention.to_dict(),
                        "reason": action.rationale,
                    }
                )
        if not self.active():
            raise RuntimeError("revision actions retired the complete hypothesis bank")
        return interventions

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": {key: value.to_dict() for key, value in self.entries.items()},
            "audit_log": self.audit_log,
        }


def validate_hypothesis(hypothesis: ConstraintHypothesis, feature_library: FeatureLibrary) -> None:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,63}", hypothesis.hypothesis_id):
        raise ValueError("hypothesis_id must be a short identifier")
    if not hypothesis.name.strip() or not hypothesis.variables:
        raise ValueError("hypothesis needs a name and at least one variable")
    if len(set(hypothesis.variables)) != len(hypothesis.variables):
        raise ValueError("hypothesis variables must be unique")
    feature_library.validate_variables(hypothesis.variables)
    if hypothesis.coupling not in ALLOWED_COUPLINGS:
        raise ValueError(f"unsupported coupling: {hypothesis.coupling}")
    if hypothesis.relation not in ALLOWED_RELATIONS:
        raise ValueError(f"unsupported relation: {hypothesis.relation}")
    if hypothesis.temporal_operator not in ALLOWED_TEMPORAL_OPERATORS:
        raise ValueError(f"unsupported temporal operator: {hypothesis.temporal_operator}")
    if hypothesis.model_family not in ALLOWED_MODEL_FAMILIES:
        raise ValueError(f"unsupported model family: {hypothesis.model_family}")
    if hypothesis.coupling == "independent" and len(hypothesis.variables) < 2:
        raise ValueError("independent coupling requires at least two variables")
    if hypothesis.relation in {"upper_bound", "lower_bound"}:
        if len(hypothesis.variables) != 1 or hypothesis.coupling != "joint":
            raise ValueError("upper_bound/lower_bound require one scalar variable with joint coupling")
    if not hypothesis.risk_direction.strip() or not hypothesis.rationale.strip():
        raise ValueError("risk_direction and rationale cannot be empty")


def compile_hypothesis(
    hypothesis: ConstraintHypothesis,
    feature_library: FeatureLibrary,
) -> CompiledHypothesis:
    validate_hypothesis(hypothesis, feature_library)
    low, high = feature_library.bounds(hypothesis.variables)
    return CompiledHypothesis(hypothesis, tuple(low), tuple(high))


def validate_revision_action(
    action: RevisionAction,
    bank: HypothesisBank,
    feature_library: FeatureLibrary,
) -> None:
    if action.action not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported revision action: {action.action}")
    target_required = action.action not in {"add_hypothesis"}
    if target_required and action.target_hypothesis_id not in bank.entries:
        raise ValueError(f"action targets unknown hypothesis: {action.target_hypothesis_id}")
    replacement_actions = {
        "change_variables",
        "change_coupling",
        "change_temporal_operator",
        "change_model_family",
        "add_hypothesis",
    }
    if action.action in replacement_actions:
        if action.replacement is None:
            raise ValueError(f"{action.action} requires replacement")
        compile_hypothesis(action.replacement, feature_library)
    if action.action == "split_hypothesis":
        if len(action.replacements) < 2:
            raise ValueError("split_hypothesis requires at least two replacements")
        for replacement in action.replacements:
            compile_hypothesis(replacement, feature_library)
    if action.action == "propose_intervention":
        if action.intervention is None:
            raise ValueError("propose_intervention requires an intervention")
        if action.intervention.kind not in ALLOWED_INTERVENTIONS:
            raise ValueError(f"unsupported intervention: {action.intervention.kind}")
        if action.intervention.target_hypothesis_id not in bank.entries:
            raise ValueError("intervention targets an unknown hypothesis")
        if action.intervention.variable is not None:
            feature_library.validate_variables((action.intervention.variable,))


def hypothesis_from_dict(raw: dict[str, Any]) -> ConstraintHypothesis:
    required = {
        "hypothesis_id",
        "name",
        "variables",
        "coupling",
        "relation",
        "temporal_operator",
        "model_family",
        "risk_direction",
        "rationale",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"hypothesis missing fields: {sorted(missing)}")
    variables = raw["variables"]
    if not isinstance(variables, list):
        raise ValueError("variables must be a JSON list")
    return ConstraintHypothesis(
        hypothesis_id=str(raw["hypothesis_id"]),
        name=str(raw["name"]),
        variables=tuple(map(str, variables)),
        coupling=str(raw["coupling"]),
        relation=str(raw["relation"]),
        temporal_operator=str(raw["temporal_operator"]),
        model_family=str(raw["model_family"]),
        risk_direction=str(raw["risk_direction"]),
        rationale=str(raw["rationale"]),
        parent_id=None if raw.get("parent_id") is None else str(raw["parent_id"]),
        generation=int(raw.get("generation", 0)),
    )


def revision_action_from_dict(raw: dict[str, Any]) -> RevisionAction:
    replacement = raw.get("replacement")
    replacements = raw.get("replacements", [])
    intervention_raw = raw.get("intervention")
    intervention = None
    if intervention_raw is not None:
        intervention = InterventionSpec(
            target_hypothesis_id=str(intervention_raw["target_hypothesis_id"]),
            kind=str(intervention_raw["kind"]),
            variable=None if intervention_raw.get("variable") is None else str(intervention_raw["variable"]),
            preserve_endpoints=bool(intervention_raw.get("preserve_endpoints", True)),
            rationale=str(intervention_raw.get("rationale", "")),
        )
    return RevisionAction(
        action=str(raw["action"]),
        target_hypothesis_id=None if raw.get("target_hypothesis_id") is None else str(raw["target_hypothesis_id"]),
        rationale=str(raw.get("rationale", "")),
        replacement=None if replacement is None else hypothesis_from_dict(replacement),
        replacements=tuple(hypothesis_from_dict(item) for item in replacements),
        intervention=intervention,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one outer JSON object while rejecting non-object payloads."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("LLM output did not contain a valid JSON object")


def extract_json_array_objects(text: str, key: str) -> list[dict[str, Any]]:
    """Recover complete object items from a named JSON array.

    Generation may stop after a complete item but before the enclosing array or
    object is closed. We still validate each recovered item with the trusted IR
    compiler; this function never makes an incomplete object executable.
    """
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
    if match is None:
        raise KeyError(key)
    items: list[dict[str, Any]] = []
    item_start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index in range(match.end(), len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character == "{":
            if depth == 0:
                item_start = index
            depth += 1
        elif character == "}" and depth > 0:
            depth -= 1
            if depth == 0 and item_start is not None:
                value = json.loads(text[item_start : index + 1])
                if isinstance(value, dict):
                    items.append(value)
                item_start = None
        elif character == "]" and depth == 0:
            break
    if not items:
        raise ValueError(f"LLM output contained no complete objects in {key!r}")
    return items
