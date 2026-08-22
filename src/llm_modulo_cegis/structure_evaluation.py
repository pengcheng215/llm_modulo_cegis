"""Private, permutation-invariant scoring of typed constraint structures."""

from __future__ import annotations

from itertools import permutations
from typing import Any


STRUCTURE_FIELDS = (
    "variables",
    "coupling",
    "relation",
    "temporal_operator",
    "model_family",
)


def _clauses(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("clauses")
    if isinstance(raw, list) and raw:
        return [dict(item) for item in raw]
    if all(field in payload for field in STRUCTURE_FIELDS):
        return [dict(payload)]
    return []


def _field_matches(expected: dict[str, Any], predicted: dict[str, Any]) -> dict[str, bool]:
    expected_variables = frozenset(map(str, expected.get("variables", [])))
    predicted_variables = frozenset(map(str, predicted.get("variables", [])))
    unary = len(expected_variables) == len(predicted_variables) == 1
    return {
        "variables": expected_variables == predicted_variables,
        "coupling": unary or str(expected.get("coupling")) == str(predicted.get("coupling")),
        "relation": str(expected.get("relation")) == str(predicted.get("relation")),
        "temporal_operator": str(expected.get("temporal_operator"))
        == str(predicted.get("temporal_operator")),
        "model_family": str(expected.get("model_family")) == str(predicted.get("model_family")),
    }


def evaluate_structure(
    predicted_hypothesis: dict[str, Any] | None,
    expected_structure: dict[str, Any],
    *,
    selection_status: str,
) -> dict[str, Any]:
    """Score one already-frozen champion; this function never selects a model."""

    representable = bool(expected_structure.get("representable", True))
    qualified = selection_status == "qualified"
    if not representable:
        return {
            "representable": False,
            "correct_abstention": not qualified,
            "erroneous_qualified_champion": qualified,
            "exact_structure_recovery": None,
            "qualified_exact_structure_recovery": None,
            "qualified_before_private_evaluation": qualified,
            "expected_clause_count": 0,
            "predicted_clause_count": len(_clauses(predicted_hypothesis or {})),
            "component_accuracy": {},
            "matched_clauses": [],
        }

    expected = _clauses(expected_structure)
    predicted = _clauses(predicted_hypothesis or {})
    composition_match = str(expected_structure.get("composition", "any_violation")) == str(
        (predicted_hypothesis or {}).get("composition", "any_violation")
    )
    pair_rows: dict[tuple[int, int], dict[str, bool]] = {
        (left, right): _field_matches(expected[left], predicted[right])
        for left in range(len(expected))
        for right in range(len(predicted))
    }
    best_pairs: list[tuple[int, int]] = []
    best_score = -1
    if expected and predicted:
        if len(expected) <= len(predicted):
            for selected in permutations(range(len(predicted)), len(expected)):
                pairs = list(enumerate(selected))
                score = sum(sum(pair_rows[pair].values()) for pair in pairs)
                if score > best_score:
                    best_score, best_pairs = score, pairs
        else:
            for selected in permutations(range(len(expected)), len(predicted)):
                pairs = [(expected_index, predicted_index) for predicted_index, expected_index in enumerate(selected)]
                score = sum(sum(pair_rows[pair].values()) for pair in pairs)
                if score > best_score:
                    best_score, best_pairs = score, pairs

    matches_by_field = {field: 0 for field in STRUCTURE_FIELDS}
    matched_rows: list[dict[str, Any]] = []
    for expected_index, predicted_index in best_pairs:
        fields = pair_rows[(expected_index, predicted_index)]
        for field, matched in fields.items():
            matches_by_field[field] += int(matched)
        matched_rows.append(
            {
                "expected_clause_index": expected_index,
                "predicted_clause_index": predicted_index,
                "field_matches": fields,
            }
        )
    denominator = max(len(expected), len(predicted), 1)
    component_accuracy = {
        field: matches_by_field[field] / denominator for field in STRUCTURE_FIELDS
    }
    exact = bool(
        composition_match
        and len(expected) == len(predicted)
        and len(best_pairs) == len(expected)
        and all(all(pair_rows[pair].values()) for pair in best_pairs)
    )
    return {
        "representable": True,
        "correct_abstention": False,
        "erroneous_qualified_champion": bool(qualified and not exact),
        "exact_structure_recovery": exact,
        "qualified_exact_structure_recovery": bool(qualified and exact),
        "qualified_before_private_evaluation": qualified,
        "composition_match": composition_match,
        "expected_clause_count": len(expected),
        "predicted_clause_count": len(predicted),
        "component_accuracy": component_accuracy,
        "matched_clauses": matched_rows,
    }
