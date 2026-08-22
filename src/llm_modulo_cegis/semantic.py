"""LLM and deterministic semantic reasoners for outer-loop synthesis."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .data import FeatureLibrary
from .hypotheses import (
    ConstraintHypothesis,
    ConstraintClause,
    HypothesisBank,
    RevisionAction,
    extract_json_array_objects,
    extract_json_object,
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


class FrozenBankSemanticReasoner:
    """Replay an audited initial hypothesis bank without another LLM call.

    This is deliberately an initial-bank replay, not a replay of semantic
    actions.  It supports controlled numeric/acquisition ablations where both
    arms must start from byte-identical GPT hypotheses.
    """

    def __init__(self, artifact_path: str | Path) -> None:
        self.artifact_path = Path(artifact_path).resolve()
        raw_bytes = self.artifact_path.read_bytes()
        self._payload = json.loads(raw_bytes.decode("utf-8"))
        self._sha256 = hashlib.sha256(raw_bytes).hexdigest()
        self.interactions: list[dict[str, Any]] = []
        self._selected_ids = self._initial_ids(self._payload)

    @staticmethod
    def _initial_ids(payload: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        for row in payload.get("audit_log", []):
            if (
                isinstance(row, dict)
                and int(row.get("outer_round", -1)) == 0
                and row.get("event") == "add"
            ):
                hypothesis_id = str(row.get("hypothesis_id", ""))
                if hypothesis_id and hypothesis_id not in ids:
                    ids.append(hypothesis_id)
        if not ids:
            raise ValueError("frozen hypothesis bank has no round-0 add events")
        return ids

    @property
    def source_manifest(self) -> dict[str, object]:
        return {
            "artifact_path": str(self.artifact_path),
            "sha256": self._sha256,
            "initial_hypothesis_ids": list(self._selected_ids),
            "llm_calls": 0,
            "revisions_frozen": True,
        }

    def propose_initial(
        self,
        task_description: str,
        feature_library: FeatureLibrary,
    ) -> list[ConstraintHypothesis]:
        del task_description
        entries = self._payload.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("frozen hypothesis bank is missing entries")
        hypotheses: list[ConstraintHypothesis] = []
        for hypothesis_id in self._selected_ids:
            entry = entries.get(hypothesis_id)
            if not isinstance(entry, dict) or not isinstance(entry.get("hypothesis"), dict):
                raise ValueError(f"frozen hypothesis bank is missing entry: {hypothesis_id}")
            hypothesis = hypothesis_from_dict(entry["hypothesis"])
            if hypothesis.hypothesis_id != hypothesis_id:
                raise ValueError(f"frozen hypothesis id mismatch: {hypothesis_id}")
            validate_hypothesis(hypothesis, feature_library)
            hypotheses.append(hypothesis)
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
        del task_description, feature_library, bank, evidence_report, evidence, outer_round
        raise RuntimeError("FrozenBankSemanticReasoner requires loop.freeze_revisions=true")


@dataclass(frozen=True)
class SemanticConfig:
    beam_width: int = 3
    prune_per_round: int = 2
    max_initial_hypotheses: int = 6
    allow_fallback: bool = True
    fallback_on_backend_error: bool = False
    simplicity_tolerance: float = 0.04
    minimum_composition_gain: float = 0.05
    max_composite_clauses: int = 3
    minimum_class_accuracy: float = 0.20


def conservative_revision_actions(
    bank: HypothesisBank,
    evidence: list[HypothesisEvidence],
) -> list[RevisionAction]:
    """Query without pruning when semantic output or numeric evidence is unsafe.

    A parser failure must not turn an unreliable early ranking into irreversible
    retirement.  Likewise, if no candidate clears the champion qualification
    gates, the best-scoring model is only a provisional experiment target.
    """

    active_ids = {item.hypothesis_id for item in bank.active()}
    ordered = sorted(
        (item for item in evidence if item.hypothesis_id in active_ids),
        key=lambda item: (item.champion_eligible, item.query_priority, item.selection_score),
        reverse=True,
    )
    if not ordered:
        raise RuntimeError("conservative revision received no active evidence")
    target = ordered[0]
    kind = "model_false_safe" if target.false_safe_count >= target.false_unsafe_count else "model_false_unsafe"
    return [
        RevisionAction(
            "retain_and_query",
            target.hypothesis_id,
            "Provisional query target only; no hypothesis pruning is permitted after fallback or failed gates.",
        ),
        RevisionAction(
            "propose_intervention",
            target.hypothesis_id,
            "Collect a discriminating counterexample before any structural retirement.",
            intervention=InterventionSpec(
                target.hypothesis_id,
                kind,
                variable=None,
                rationale="Conservative fallback query; preserve the current hypothesis bank.",
            ),
        ),
    ]


def canonical_initial_hypotheses(
    feature_library: FeatureLibrary | None = None,
) -> list[ConstraintHypothesis]:
    """Return a task-compatible structural bank with no numeric thresholds.

    The old fallback was silently planar.  That made a parser failure on a
    richer task appear to recover, only to fail later when the compiler saw
    unavailable x/y hypotheses.  Fallback is now selected solely from the
    public feature schema.
    """

    names = set() if feature_library is None else set(feature_library.names)
    if {"target_dz", "speed", "tilt_from_vertical"}.issubset(names):
        height = ConstraintClause(
            "height_band",
            ("target_dz",),
            "joint",
            "equality_band",
            "max",
            "linear",
            "increase the absolute reference-relative height error",
            "Tests a reference-relative carrying-height band.",
        )
        speed = ConstraintClause(
            "speed_limit",
            ("speed",),
            "joint",
            "upper_bound",
            "max",
            "linear",
            "increase direction-invariant translational speed",
            "Tests a gentle-motion limit independent of travel direction.",
        )
        tilt = ConstraintClause(
            "tilt_limit",
            ("tilt_from_vertical",),
            "joint",
            "upper_bound",
            "max",
            "linear",
            "increase cup tilt while allowing yaw rotation",
            "Tests uprightness as tilt from vertical rather than yaw.",
        )

        def atomic(identifier: str, name: str, clause: ConstraintClause) -> ConstraintHypothesis:
            return ConstraintHypothesis(
                identifier,
                name,
                clause.variables,
                clause.coupling,
                clause.relation,
                clause.temporal_operator,
                clause.model_family,
                clause.risk_direction,
                clause.rationale,
            )

        return [
            atomic("h_target_dz_band", "reference-relative height band", height),
            atomic("h_speed_3d", "three-dimensional speed limit", speed),
            atomic("h_tilt_vertical", "tilt-from-vertical limit", tilt),
            ConstraintHypothesis(
                "h_carrywater_composite",
                "height, speed, and uprightness constraints",
                ("target_dz", "speed", "tilt_from_vertical"),
                height.coupling,
                height.relation,
                height.temporal_operator,
                height.model_family,
                "stress one physical clause while preserving the other two",
                "Tests the simultaneous three-clause interpretation of carrying water.",
                clauses=(height, speed, tilt),
                composition="any_violation",
            ),
        ]

    # Default planar bank, retained byte-for-byte in structure for regression
    # compatibility and for calls made without a feature library.
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
            "linear",
            "increase local speed while preserving the geometric path",
            "Tests a dynamic rather than geometric explanation.",
        ),
    ]


def _identifier(value: object, fallback: str, prefix: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or fallback)).strip("_-")
    if not text or not text[0].isalpha():
        text = f"{prefix}_{text or fallback}"
    return text[:64]


def _variables(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def normalize_hypothesis_payload(raw: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Repair surface-format mistakes without inventing numeric semantics.

    Small local models commonly omit prose fields, use singular ``variable``,
    or return equality aliases.  These repairs preserve the proposed structure;
    the trusted compiler still rejects unknown variables and illegal relations.
    """

    value = dict(raw)
    relation_aliases = {
        "equality": "equality_band",
        "equal": "equality_band",
        "eq": "equality_band",
        "upper": "upper_bound",
        "lower": "lower_bound",
        "region": "forbidden_region",
    }
    raw_clauses = value.get("clauses", value.get("constraints", []))
    normalized_clauses: list[dict[str, Any]] = []
    if isinstance(raw_clauses, list):
        for clause_index, raw_clause in enumerate(raw_clauses):
            if not isinstance(raw_clause, dict):
                continue
            clause = dict(raw_clause)
            relation = str(clause.get("relation", "forbidden_region")).lower()
            relation = relation_aliases.get(relation, relation)
            variables = _variables(clause.get("variables", clause.get("variable", clause.get("features", []))))
            coupling = str(clause.get("coupling", "joint")).lower()
            model_family = str(
                clause.get("model_family", "linear" if relation != "forbidden_region" else "mlp")
            ).lower()
            if len(variables) == 1:
                coupling = "joint"
            if len(variables) > 1 and relation in {"upper_bound", "lower_bound"} and model_family == "linear":
                relation = "forbidden_region"
            name = str(clause.get("name", f"clause {clause_index + 1}"))
            normalized_clauses.append(
                {
                    "clause_id": _identifier(
                        clause.get("clause_id", clause.get("id")), f"c{clause_index + 1}", "c"
                    ),
                    "variables": variables,
                    "coupling": coupling,
                    "relation": relation,
                    "temporal_operator": str(clause.get("temporal_operator", clause.get("temporal", "max"))).lower(),
                    "model_family": model_family,
                    "risk_direction": str(clause.get("risk_direction", clause.get("intervention", name))),
                    "rationale": str(clause.get("rationale", name)),
                }
            )
    variables = _variables(value.get("variables", value.get("variable", value.get("features", []))))
    if normalized_clauses:
        variables = list(dict.fromkeys(variable for clause in normalized_clauses for variable in clause["variables"]))
        primary = normalized_clauses[0]
    else:
        relation = str(value.get("relation", "forbidden_region")).lower()
        relation = relation_aliases.get(relation, relation)
        coupling = str(value.get("coupling", "joint")).lower()
        model_family = str(
            value.get("model_family", "linear" if relation != "forbidden_region" else "mlp")
        ).lower()
        if len(variables) == 1:
            coupling = "joint"
        if len(variables) > 1 and relation in {"upper_bound", "lower_bound"} and model_family == "linear":
            relation = "forbidden_region"
        primary = {
            "coupling": coupling,
            "relation": relation,
            "temporal_operator": str(value.get("temporal_operator", value.get("temporal", "max"))).lower(),
            "model_family": model_family,
        }
    name = str(value.get("name", value.get("description", f"LLM hypothesis {index + 1}")))
    return {
        "hypothesis_id": _identifier(value.get("hypothesis_id", value.get("id")), f"h_llm_{index + 1}", "h"),
        "name": name,
        "variables": variables,
        "coupling": primary["coupling"],
        "relation": primary["relation"],
        "temporal_operator": primary["temporal_operator"],
        "model_family": primary["model_family"],
        "risk_direction": str(value.get("risk_direction", value.get("intervention", name))),
        "rationale": str(value.get("rationale", name)),
        "parent_id": value.get("parent_id"),
        "generation": int(value.get("generation", 0)),
        "clauses": normalized_clauses,
        "composition": str(value.get("composition", "any_violation")).lower(),
    }


def hypothesis_repair_notes(raw: dict[str, Any], normalized: dict[str, Any]) -> list[str]:
    """Describe semantic-preserving canonicalizations for the audit log."""

    notes: list[str] = []
    raw_variables = _variables(raw.get("variables", raw.get("variable", raw.get("features", []))))
    raw_coupling = str(raw.get("coupling", "joint")).lower()
    raw_relation = str(raw.get("relation", "forbidden_region")).lower()
    if len(raw_variables) == 1 and raw_coupling == "independent" and normalized["coupling"] == "joint":
        notes.append("canonicalized scalar independent coupling to joint")
    if (
        len(raw_variables) > 1
        and raw_relation in {"upper_bound", "lower_bound", "upper", "lower"}
        and normalized["relation"] == "forbidden_region"
    ):
        notes.append("canonicalized multivariate linear inequality to linear forbidden_region")
    return notes


def _extract_named_items(raw: str, key: str) -> list[dict[str, Any]]:
    try:
        outer = extract_json_object(raw)
        values = outer.get(key)
        if isinstance(values, list):
            items = [item for item in values if isinstance(item, dict)]
            if items:
                return items
    except Exception:
        pass
    return extract_json_array_objects(raw, key)


def normalize_revision_action_payload(raw: dict[str, Any], index: int = 0) -> dict[str, Any]:
    value = dict(raw)
    action = str(value.get("action", value.get("type", ""))).lower()
    replacement = value.get("replacement", value.get("hypothesis"))
    replacements = value.get("replacements", [])
    intervention = value.get("intervention")
    if isinstance(replacement, dict):
        replacement = normalize_hypothesis_payload(replacement, index)
    if isinstance(replacements, list):
        replacements = [
            normalize_hypothesis_payload(item, index * 10 + replacement_index)
            for replacement_index, item in enumerate(replacements)
            if isinstance(item, dict)
        ]
    else:
        replacements = []
    if isinstance(intervention, dict):
        intervention = {
            "target_hypothesis_id": str(
                intervention.get("target_hypothesis_id", value.get("target_hypothesis_id", ""))
            ),
            "kind": str(intervention.get("kind", "boundary_uncertainty")).lower(),
            "variable": intervention.get("variable"),
            "clause_id": intervention.get("clause_id"),
            "preserve_endpoints": bool(intervention.get("preserve_endpoints", True)),
            "rationale": str(intervention.get("rationale", value.get("rationale", "evidence-directed query"))),
        }
    return {
        "action": action,
        "target_hypothesis_id": value.get("target_hypothesis_id", value.get("target")),
        "rationale": str(value.get("rationale", "evidence-directed revision")),
        "replacement": replacement,
        "replacements": replacements,
        "intervention": intervention,
    }


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
        hypotheses = canonical_initial_hypotheses(feature_library)
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
        viable = [item for item in ordered if item.champion_eligible]
        if not viable:
            actions = conservative_revision_actions(bank, evidence)
            self.interactions.append(
                {
                    "phase": "revision",
                    "backend": "evidence_policy",
                    "outer_round": outer_round,
                    "input_evidence": evidence_report,
                    "conservative_query_only": True,
                    "parsed": [action.to_dict() for action in actions],
                }
            )
            return actions
        champion_pool = viable
        best_score = champion_pool[0].selection_score
        statistically_tied = [
            item
            for item in champion_pool
            if item.selection_score >= best_score - self.config.simplicity_tolerance
        ]
        # Within the evidence resolution, prefer the smaller explanation.  A
        # joint all-feature MLP must earn a measurable prequential advantage.
        champion = min(
            statistically_tied,
            key=lambda item: (item.complexity, item.parameter_count, -item.selection_score),
        )
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

        active_signatures = {item.signature() for item in bank.active()}
        pair_rows = evidence_report.get("pair_complementarity", [])
        if isinstance(pair_rows, list):
            for row in pair_rows:
                if not isinstance(row, dict):
                    continue
                pair = row.get("hypothesis_ids")
                gain = float(row.get("gain_over_best_single", 0.0))
                if not isinstance(pair, list) or len(pair) != 2 or gain < self.config.minimum_composition_gain:
                    continue
                if any(str(item) not in active_ids for item in pair):
                    continue
                composite = compose_hypotheses(
                    bank.get(str(pair[0])),
                    bank.get(str(pair[1])),
                    outer_round,
                    self.config.max_composite_clauses,
                )
                if composite is not None and composite.signature() not in active_signatures:
                    actions.append(
                        RevisionAction(
                            "compose_hypotheses",
                            champion.hypothesis_id,
                            "The two predicates fix complementary pre-query errors.",
                            replacement=composite,
                        )
                    )
                break

        # A high counterexample rate is evidence for a structural edit, not
        # merely another query.  This keeps the fallback outer loop genuinely
        # revision-capable even when a local LLM cannot emit valid JSON.
        champion_hypothesis = bank.get(champion.hypothesis_id)
        if (
            not any(action.action in {"compose_hypotheses", "change_coupling", "change_model_family"} for action in actions)
            and champion.evidence_sufficient
            and champion.counterexample_rate > 0.25
            and len(champion_hypothesis.atomic_clauses()) == 1
        ):
            if champion_hypothesis.coupling == "independent":
                revised = _replace_atomic_structure(
                    champion_hypothesis,
                    hypothesis_id=f"{champion.hypothesis_id}_joint_r{outer_round}",
                    coupling="joint",
                    outer_round=outer_round,
                )
                action_name = "change_coupling"
            elif champion_hypothesis.model_family == "mlp" and champion.false_unsafe_count > champion.false_safe_count:
                revised = _replace_atomic_structure(
                    champion_hypothesis,
                    hypothesis_id=f"{champion.hypothesis_id}_linear_r{outer_round}",
                    model_family="linear",
                    outer_round=outer_round,
                )
                action_name = "change_model_family"
            else:
                revised = None
                action_name = ""
            if revised is not None and revised.signature() not in active_signatures:
                actions.append(
                    RevisionAction(
                        action_name,
                        champion.hypothesis_id,
                        "Prequential counterexamples reject the current structural assumption.",
                        replacement=revised,
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


def _replace_atomic_structure(
    source: ConstraintHypothesis,
    *,
    hypothesis_id: str,
    outer_round: int,
    coupling: str | None = None,
    model_family: str | None = None,
) -> ConstraintHypothesis:
    return ConstraintHypothesis(
        hypothesis_id=hypothesis_id,
        name=f"revised {source.name}",
        variables=source.variables,
        coupling=coupling or source.coupling,
        relation=source.relation,
        temporal_operator=source.temporal_operator,
        model_family=model_family or source.model_family,
        risk_direction=source.risk_direction,
        rationale="Counterexample evidence requested a lower-error structure.",
        parent_id=source.hypothesis_id,
        generation=max(source.generation + 1, outer_round),
    )


def compose_hypotheses(
    left: ConstraintHypothesis,
    right: ConstraintHypothesis,
    outer_round: int,
    maximum_clauses: int,
) -> ConstraintHypothesis | None:
    left_relations = {clause.relation for clause in left.atomic_clauses()}
    right_relations = {clause.relation for clause in right.atomic_clauses()}
    if set(left.variables) == set(right.variables) and left_relations == right_relations:
        # These are alternative parameterizations of one predicate, not
        # evidence for two simultaneous constraints.
        return None
    clauses: list[ConstraintClause] = []
    signatures: set[tuple[Any, ...]] = set()
    for source in (left, right):
        for clause in source.atomic_clauses():
            if clause.signature() in signatures:
                continue
            signatures.add(clause.signature())
            clauses.append(
                ConstraintClause(
                    clause_id=f"c{len(clauses) + 1}",
                    variables=clause.variables,
                    coupling=clause.coupling,
                    relation=clause.relation,
                    temporal_operator=clause.temporal_operator,
                    model_family=clause.model_family,
                    risk_direction=clause.risk_direction,
                    rationale=clause.rationale,
                )
            )
    if len(clauses) < 2 or len(clauses) > maximum_clauses:
        return None
    variables = tuple(dict.fromkeys(variable for clause in clauses for variable in clause.variables))
    primary = clauses[0]
    return ConstraintHypothesis(
        hypothesis_id=f"h_composite_r{outer_round}_{left.hypothesis_id[:12]}_{right.hypothesis_id[:12]}",
        name=f"{left.name} OR {right.name} violations",
        variables=variables,
        coupling=primary.coupling,
        relation=primary.relation,
        temporal_operator=primary.temporal_operator,
        model_family=primary.model_family,
        risk_direction="stress either constituent constraint while preserving the other",
        rationale="Complementary prequential errors support multiple simultaneous constraints.",
        parent_id=left.hypothesis_id,
        generation=max(left.generation, right.generation) + 1,
        clauses=tuple(clauses),
        composition="any_violation",
    )


def augment_stagnant_revision(
    actions: list[RevisionAction],
    *,
    fallback: EvidencePolicyReasoner,
    task_description: str,
    feature_library: FeatureLibrary,
    bank: HypothesisBank,
    evidence_report: dict[str, object],
    evidence: list[HypothesisEvidence],
    outer_round: int,
) -> tuple[list[RevisionAction], bool]:
    """Guarantee evidence-driven structural progress when errors are mature."""

    structural = {
        "retire_hypothesis",
        "change_variables",
        "change_coupling",
        "change_temporal_operator",
        "change_model_family",
        "split_hypothesis",
        "compose_hypotheses",
        "add_hypothesis",
    }
    if not any(item.champion_eligible for item in evidence):
        return actions, False
    if any(action.action in structural for action in actions):
        return actions, False
    if not any(item.evidence_sufficient and item.counterexample_rate > 0.25 for item in evidence):
        return actions, False
    policy_actions = fallback.revise(
        task_description,
        feature_library,
        bank,
        evidence_report,
        evidence,
        outer_round,
    )
    additions = [action for action in policy_actions if action.action in structural]
    if not additions:
        return actions, False
    return actions + additions[:1], True


def enforce_champion_admissibility(
    actions: list[RevisionAction],
    evidence: list[HypothesisEvidence],
    config: SemanticConfig,
) -> tuple[list[RevisionAction], bool]:
    """Reject a semantic champion that collapses to one trajectory class."""

    viable = [item for item in evidence if item.champion_eligible]
    retained = next((action for action in actions if action.action == "retain_and_query"), None)
    if not viable:
        ordered = sorted(evidence, key=lambda item: (item.query_priority, item.selection_score), reverse=True)
        if not ordered:
            return actions, False
        target = ordered[0]
        retained_intervention = next(
            (
                action
                for action in actions
                if action.action == "propose_intervention"
                and action.target_hypothesis_id == target.hypothesis_id
            ),
            None,
        )
        conservative = [
            RevisionAction(
                "retain_and_query",
                target.hypothesis_id,
                "No hypothesis clears the champion gates; use only as a provisional query target.",
            )
        ]
        if retained_intervention is not None:
            conservative.append(retained_intervention)
        else:
            kind = "model_false_safe" if target.false_safe_count >= target.false_unsafe_count else "model_false_unsafe"
            conservative.append(
                RevisionAction(
                    "propose_intervention",
                    target.hypothesis_id,
                    "Collect more evidence before any structural retirement.",
                    intervention=InterventionSpec(
                        target.hypothesis_id,
                        kind,
                        variable=None,
                        rationale="No hypothesis currently clears the champion gates.",
                    ),
                )
            )
        return conservative, True
    if retained is None:
        return actions, False
    current = next((item for item in evidence if item.hypothesis_id == retained.target_hypothesis_id), None)
    if current is not None and current in viable:
        return actions, False
    best_score = max(item.selection_score for item in viable)
    tied = [item for item in viable if item.selection_score >= best_score - config.simplicity_tolerance]
    replacement = min(tied, key=lambda item: (item.complexity, item.parameter_count, -item.selection_score))
    revised = [action for action in actions if action.action != "retain_and_query"]
    revised.insert(
        0,
        RevisionAction(
            "retain_and_query",
            replacement.hypothesis_id,
            "Verifier replaced a class-degenerate semantic champion.",
        ),
    )
    return revised, True


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
        used_augmentation = False
        accepted_llm_count = 0
        candidate_errors: list[str] = []
        candidate_repairs: list[dict[str, Any]] = []
        try:
            values = _extract_named_items(raw, "hypotheses")
            hypotheses = []
            signatures: set[tuple[Any, ...]] = set()
            for index, value in enumerate(values[: self.config.max_initial_hypotheses]):
                try:
                    normalized = normalize_hypothesis_payload(value, index)
                    notes = hypothesis_repair_notes(value, normalized)
                    if notes:
                        candidate_repairs.append({"candidate_index": index, "repairs": notes})
                    hypothesis = hypothesis_from_dict(normalized)
                    validate_hypothesis(hypothesis, feature_library)
                    if hypothesis.signature() in signatures:
                        raise ValueError("duplicate hypothesis structure")
                    hypotheses.append(hypothesis)
                    signatures.add(hypothesis.signature())
                except Exception as exc:
                    candidate_errors.append(f"candidate[{index}]: {type(exc).__name__}: {exc}")
            accepted_llm_count = len(hypotheses)
            if len(hypotheses) < 2:
                used_augmentation = True
                for fallback in canonical_initial_hypotheses(feature_library):
                    if fallback.signature() in signatures:
                        continue
                    hypotheses.append(fallback)
                    signatures.add(fallback.signature())
                    if len(hypotheses) >= 2:
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
            hypotheses = canonical_initial_hypotheses(feature_library)[
                : self.config.max_initial_hypotheses
            ]
        self.interactions.append(
            {
                "phase": "initial",
                "backend": "local_qwen",
                "prompt": prompt,
                "raw_output": raw,
                "parse_error": error,
                "candidate_errors": candidate_errors,
                "candidate_repairs": candidate_repairs,
                "used_fallback": used_fallback,
                "used_augmentation": used_augmentation,
                "fallback_kind": "full" if used_fallback else ("bank_augmentation" if used_augmentation else "none"),
                "accepted_llm_count": accepted_llm_count,
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
        policy_augmented = False
        champion_repaired = False
        action_errors: list[str] = []
        try:
            raw_actions = _extract_named_items(raw, "actions")
            actions = []
            for index, raw_action in enumerate(raw_actions):
                try:
                    action = revision_action_from_dict(normalize_revision_action_payload(raw_action, index))
                    validate_revision_action(action, bank, feature_library)
                    actions.append(action)
                except Exception as exc:
                    action_errors.append(f"action[{index}]: {type(exc).__name__}: {exc}")
            retained = [action for action in actions if action.action == "retain_and_query"]
            if not retained:
                ranking = evidence_report.get("ranking", [])
                if not isinstance(ranking, list) or not ranking:
                    raise ValueError("revision omitted champion and evidence has no ranking")
                actions.insert(
                    0,
                    RevisionAction(
                        "retain_and_query",
                        str(ranking[0]),
                        "Compiler-added current champion; LLM actions otherwise retained.",
                    ),
                )
                error = "surface repair: compiler added the omitted champion action"
            elif len(retained) > 1:
                ranking = evidence_report.get("ranking", [])
                preferred = str(ranking[0]) if isinstance(ranking, list) and ranking else retained[0].target_hypothesis_id
                chosen = next(
                    (action for action in retained if action.target_hypothesis_id == preferred),
                    retained[0],
                )
                actions = [action for action in actions if action.action != "retain_and_query"]
                actions.insert(0, chosen)
                error = "surface repair: compiler kept one evidence-ranked champion action"
            if action_errors:
                partial = "partial rejection: " + " | ".join(action_errors)
                error = partial if error is None else error + " | " + partial
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not self.config.allow_fallback:
                raise
            used_fallback = True
            actions = conservative_revision_actions(bank, evidence)
        if not used_fallback:
            actions, champion_repaired = enforce_champion_admissibility(actions, evidence, self.config)
            actions, policy_augmented = augment_stagnant_revision(
                actions,
                fallback=self._fallback,
                task_description=task_description,
                feature_library=feature_library,
                bank=bank,
                evidence_report=evidence_report,
                evidence=evidence,
                outer_round=outer_round,
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
                "policy_augmented": policy_augmented,
                "champion_repaired": champion_repaired,
                "input_evidence": evidence_report,
                "parsed": [action.to_dict() for action in actions],
            }
        )
        return actions


def _clause_json_schema() -> dict[str, Any]:
    properties = {
        "clause_id": {"type": "string"},
        "variables": {"type": "array", "items": {"type": "string"}},
        "coupling": {"type": "string", "enum": ["joint", "independent"]},
        "relation": {
            "type": "string",
            "enum": ["forbidden_region", "upper_bound", "lower_bound", "equality_band"],
        },
        "temporal_operator": {"type": "string", "enum": ["max", "mean", "last"]},
        "model_family": {"type": "string", "enum": ["mlp", "linear"]},
        "risk_direction": {"type": "string"},
        "rationale": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _hypothesis_json_schema() -> dict[str, Any]:
    properties = {
        "hypothesis_id": {"type": "string"},
        "name": {"type": "string"},
        "variables": {"type": "array", "items": {"type": "string"}},
        "coupling": {"type": "string", "enum": ["joint", "independent"]},
        "relation": {
            "type": "string",
            "enum": ["forbidden_region", "upper_bound", "lower_bound", "equality_band"],
        },
        "temporal_operator": {"type": "string", "enum": ["max", "mean", "last"]},
        "model_family": {"type": "string", "enum": ["mlp", "linear"]},
        "risk_direction": {"type": "string"},
        "rationale": {"type": "string"},
        "parent_id": {"type": ["string", "null"]},
        "generation": {"type": "integer"},
        "clauses": {"type": "array", "items": _clause_json_schema()},
        "composition": {"type": "string", "enum": ["any_violation"]},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _initial_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "hypotheses": {"type": "array", "items": _hypothesis_json_schema()},
        },
        "required": ["hypotheses"],
        "additionalProperties": False,
    }


def _revision_json_schema() -> dict[str, Any]:
    hypothesis = _hypothesis_json_schema()
    intervention_properties = {
        "target_hypothesis_id": {"type": "string"},
        "kind": {
            "type": "string",
            "enum": [
                "model_false_safe",
                "model_false_unsafe",
                "boundary_uncertainty",
                "shortcut",
                "local_feature_stress",
            ],
        },
        "variable": {"type": ["string", "null"]},
        "clause_id": {"type": ["string", "null"]},
        "preserve_endpoints": {"type": "boolean"},
        "rationale": {"type": "string"},
    }
    intervention = {
        "type": "object",
        "properties": intervention_properties,
        "required": list(intervention_properties),
        "additionalProperties": False,
    }
    action_properties = {
        "action": {"type": "string", "enum": sorted(
            {
                "retain_and_query",
                "retire_hypothesis",
                "change_variables",
                "change_coupling",
                "change_temporal_operator",
                "change_model_family",
                "split_hypothesis",
                "compose_hypotheses",
                "add_hypothesis",
                "propose_intervention",
            }
        )},
        "target_hypothesis_id": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
        "replacement": {"anyOf": [hypothesis, {"type": "null"}]},
        "replacements": {"type": "array", "items": hypothesis},
        "intervention": {"anyOf": [intervention, {"type": "null"}]},
    }
    action = {
        "type": "object",
        "properties": action_properties,
        "required": list(action_properties),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"actions": {"type": "array", "items": action}},
        "required": ["actions"],
        "additionalProperties": False,
    }


def load_project_openai_env(env_file: str | os.PathLike[str] | None = None) -> bool:
    """Load only OPENAI_API_KEY from the repository api.env, without logging it."""

    if os.environ.get("OPENAI_API_KEY"):
        return True
    path = Path(env_file) if env_file is not None else Path(__file__).resolve().parents[3] / "api.env"
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "OPENAI_API_KEY":
            secret = value.strip().strip("\"'")
            if secret:
                os.environ.setdefault("OPENAI_API_KEY", secret)
                return True
    return False


class OpenAISemanticReasoner:
    """GPT semantic reasoner using strict Responses API Structured Outputs."""

    def __init__(
        self,
        model: str,
        config: SemanticConfig,
        *,
        reasoning_effort: str = "medium",
        max_output_tokens: int = 2400,
        env_file: str | os.PathLike[str] | None = None,
        client: Any | None = None,
    ) -> None:
        if not str(model).strip():
            raise ValueError("OpenAI model cannot be empty")
        self.model = str(model).strip()
        self.config = config
        self.reasoning_effort = str(reasoning_effort)
        self.max_output_tokens = int(max_output_tokens)
        self.env_file = env_file
        self._client = client
        self._fallback = EvidencePolicyReasoner(config)
        self.interactions: list[dict[str, Any]] = []

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not load_project_openai_env(self.env_file):
            raise RuntimeError("OPENAI_API_KEY was not found in the environment or project api.env")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("the openai package is required for the openai backend") from exc
        self._client = OpenAI()
        return self._client

    def _generate(self, prompt: str, schema: dict[str, Any], schema_name: str) -> tuple[str, dict[str, Any]]:
        response = self._get_client().responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You synthesize qualitative robot safety constraints. "
                        "Follow the supplied JSON Schema and never invent numerical boundaries."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            reasoning={"effort": self.reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            max_output_tokens=self.max_output_tokens,
            store=False,
        )
        raw = getattr(response, "output_text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("OpenAI response contained no structured output text")
        usage = getattr(response, "usage", None)
        receipt = {
            "response_id": None if getattr(response, "id", None) is None else str(response.id),
            "response_model": None if getattr(response, "model", None) is None else str(response.model),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "store": False,
        }
        return raw, receipt

    def propose_initial(
        self,
        task_description: str,
        feature_library: FeatureLibrary,
    ) -> list[ConstraintHypothesis]:
        prompt = build_initial_prompt(task_description, feature_library, self.config.max_initial_hypotheses)
        error: str | None = None
        receipt: dict[str, Any] = {}
        used_fallback = False
        used_augmentation = False
        candidate_errors: list[str] = []
        candidate_repairs: list[dict[str, Any]] = []
        accepted_llm_count = 0
        raw = ""
        backend_completed = False
        try:
            raw, receipt = self._generate(prompt, _initial_json_schema(), "constraint_hypothesis_bank")
            backend_completed = True
            payload = json.loads(raw)
            hypotheses: list[ConstraintHypothesis] = []
            signatures: set[tuple[Any, ...]] = set()
            for index, value in enumerate(payload.get("hypotheses", [])[: self.config.max_initial_hypotheses]):
                try:
                    normalized = normalize_hypothesis_payload(value, index)
                    notes = hypothesis_repair_notes(value, normalized)
                    if notes:
                        candidate_repairs.append({"candidate_index": index, "repairs": notes})
                    hypothesis = hypothesis_from_dict(normalized)
                    validate_hypothesis(hypothesis, feature_library)
                    if hypothesis.signature() in signatures:
                        raise ValueError("duplicate hypothesis structure")
                    hypotheses.append(hypothesis)
                    signatures.add(hypothesis.signature())
                except Exception as exc:
                    candidate_errors.append(f"candidate[{index}]: {type(exc).__name__}: {exc}")
            accepted_llm_count = len(hypotheses)
            if not hypotheses:
                raise ValueError("GPT returned no compilable hypotheses")
            if len(hypotheses) < 2:
                if not self.config.allow_fallback:
                    raise ValueError("GPT returned fewer than two compilable hypotheses")
                used_augmentation = True
                for fallback in canonical_initial_hypotheses(feature_library):
                    if fallback.signature() in signatures:
                        continue
                    hypotheses.append(fallback)
                    signatures.add(fallback.signature())
                    if len(hypotheses) >= 2:
                        break
            if candidate_errors:
                error = "partial rejection: " + " | ".join(candidate_errors)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if (not backend_completed and not self.config.fallback_on_backend_error) or not self.config.allow_fallback:
                raise
            used_fallback = True
            hypotheses = canonical_initial_hypotheses(feature_library)[
                : self.config.max_initial_hypotheses
            ]
        self.interactions.append(
            {
                "phase": "initial",
                "backend": "openai",
                "model": self.model,
                "prompt": prompt,
                "raw_output": raw,
                "parse_error": error,
                "candidate_errors": candidate_errors,
                "candidate_repairs": candidate_repairs,
                "used_fallback": used_fallback,
                "used_augmentation": used_augmentation,
                "fallback_kind": "full" if used_fallback else ("bank_augmentation" if used_augmentation else "none"),
                "accepted_llm_count": accepted_llm_count,
                "receipt": receipt,
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
        error: str | None = None
        receipt: dict[str, Any] = {}
        used_fallback = False
        policy_augmented = False
        champion_repaired = False
        action_errors: list[str] = []
        raw = ""
        backend_completed = False
        try:
            raw, receipt = self._generate(prompt, _revision_json_schema(), "constraint_revision_actions")
            backend_completed = True
            payload = json.loads(raw)
            actions: list[RevisionAction] = []
            for index, value in enumerate(payload.get("actions", [])):
                try:
                    action = revision_action_from_dict(normalize_revision_action_payload(value, index))
                    validate_revision_action(action, bank, feature_library)
                    actions.append(action)
                except Exception as exc:
                    action_errors.append(f"action[{index}]: {type(exc).__name__}: {exc}")
            retained = [action for action in actions if action.action == "retain_and_query"]
            if len(retained) != 1:
                raise ValueError("GPT revision must name exactly one current champion")
            if action_errors:
                error = "partial rejection: " + " | ".join(action_errors)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if (not backend_completed and not self.config.fallback_on_backend_error) or not self.config.allow_fallback:
                raise
            used_fallback = True
            actions = conservative_revision_actions(bank, evidence)
        if not used_fallback:
            actions, champion_repaired = enforce_champion_admissibility(actions, evidence, self.config)
            actions, policy_augmented = augment_stagnant_revision(
                actions,
                fallback=self._fallback,
                task_description=task_description,
                feature_library=feature_library,
                bank=bank,
                evidence_report=evidence_report,
                evidence=evidence,
                outer_round=outer_round,
            )
        self.interactions.append(
            {
                "phase": "revision",
                "backend": "openai",
                "model": self.model,
                "outer_round": outer_round,
                "prompt": prompt,
                "raw_output": raw,
                "parse_error": error,
                "action_errors": action_errors,
                "used_fallback": used_fallback,
                "policy_augmented": policy_augmented,
                "champion_repaired": champion_repaired,
                "receipt": receipt,
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

Propose a diverse bank of two to {maximum} competing qualitative hypotheses. Give simple scalar bounds, equality bands, linear predicates, and small feature subsets genuine candidates; do not make an all-feature joint MLP the default. A joint MLP is justified only when the task text implies a non-separable interaction. Hypotheses should be structurally non-nested when possible so evidence can distinguish them. Keep prose concise. Never output coordinates, centers, radii, thresholds, boundary points, or ground-truth claims. The neural learner estimates all numerical boundaries.

A hypothesis can contain multiple simultaneous atomic constraints. Safe means every clause is satisfied; trajectory violation means any clause is violated. For an atomic hypothesis return clauses as an empty list. For a composite hypothesis, clauses contains two or more complete atomic clauses, composition is any_violation, and top-level variables is exactly the union of clause variables. This represents cases such as feature A equality_band together with feature B upper_bound.

Allowed values:
- coupling: joint | independent
- relation: forbidden_region | upper_bound | lower_bound | equality_band
- temporal_operator: max | mean | last
- model_family: mlp | linear

Temporal semantics are about violation scores, not raw feature extrema:
- max means ANY time step may witness a violation (for lower_bound it enforces the feature floor at every step);
- mean means the trajectory-average violation score is constrained;
- last evaluates only the final time step.
A requirement such as "the trajectory must eventually reach a high peak" is not supported. Never encode it as lower_bound + max.

Compiler rules:
- one-variable clauses always use coupling=joint;
- coupling=independent requires at least two variables;
- upper_bound, lower_bound, and equality_band are scalar relations with exactly one variable and model_family=linear;
- a multivariate affine half-space uses relation=forbidden_region, coupling=joint, model_family=linear.

For GPT Structured Outputs, also populate parent_id=null, generation=0, clauses=[], and composition="any_violation". Local-model parsing can repair those four surface fields if omitted.
Return exactly:
{{"hypotheses":[{{"hypothesis_id":"h_identifier","name":"description","variables":["available_name"],"coupling":"joint","relation":"upper_bound","temporal_operator":"max","model_family":"linear","risk_direction":"increase the feature","rationale":"why this simple structure is plausible","parent_id":null,"generation":0,"clauses":[],"composition":"any_violation"}}]}}"""


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

Compiler rules: scalar clauses use coupling=joint; independent requires two or more variables; upper_bound/lower_bound/equality_band require exactly one variable and model_family=linear; a multivariate affine inequality is forbidden_region + joint + linear.

Temporal semantics: max means any time step with a positive violation score violates the trajectory; it does not mean the maximum raw feature value. In particular lower_bound + max is an all-time feature floor. mean constrains average violation score, and last examines only the final time step. An eventual/peak-reaching requirement is unsupported.

Allowed actions:
- retain_and_query
- retire_hypothesis
- change_variables, change_coupling, change_temporal_operator, change_model_family: include one complete replacement hypothesis with a new id
- split_hypothesis: include two or more complete replacements
- compose_hypotheses: include one composite replacement with multiple clauses when predicates fix complementary errors
- add_hypothesis: include one complete replacement
- propose_intervention: intervention.kind is model_false_safe | model_false_unsafe | boundary_uncertainty | shortcut | local_feature_stress

Only hypotheses with champion_eligible=true may be named as a semantic champion. If qualified_ranking is empty, retain one provisional query target, propose an intervention, and do not retire, replace, split, compose, or add hypotheses. Prefer a simple eligible hypothesis whenever its prequential score is within the reported evidence resolution of a larger joint MLP. A high counterexample rate should trigger a structural change, split, or composition only after at least one hypothesis clears the gates. Pair complementarity is evidence for multiple simultaneous constraints.

For an intervention use {{"target_hypothesis_id":"existing id","kind":"allowed kind","variable":"available variable or null","clause_id":"target clause or null","preserve_endpoints":true,"rationale":"reason"}}.
Return one to four actions total. For every action provide action, target_hypothesis_id, rationale, replacement or null, replacements as a list, and intervention or null. Include exactly one retain_and_query action naming the current-round semantic champion. The same hypothesis may be replaced for the next round. Return exactly {{"actions":[...]}}. Never use evaluation IoU, obstacle geometry, numerical thresholds, or state-level labels."""
