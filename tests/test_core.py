from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_modulo_cegis.data import FeatureLibrary
from llm_modulo_cegis.hypotheses import (
    ConstraintHypothesis,
    HypothesisBank,
    RevisionAction,
    compile_hypothesis,
    extract_json_array_objects,
)
from llm_modulo_cegis.learner import LearnerRegistry, TrainerConfig, fit_ensemble
from llm_modulo_cegis.oracle import CircularEvaluationOracle
from llm_modulo_cegis.semantic import EvidencePolicyReasoner, SemanticConfig, canonical_initial_hypotheses
from llm_modulo_cegis.types import HypothesisEvidence, QueryRecord, SAFE_LABEL, Trajectory, VIOLATION_LABEL


def straight(y: float, *, source: str = "toy") -> Trajectory:
    x = np.linspace(0.0, 2.0, 24, dtype=np.float32)
    states = np.column_stack((x, np.full_like(x, y)))
    actions = np.zeros_like(states)
    actions[:-1] = np.diff(states, axis=0)
    return Trajectory(states, actions, metadata={"source": source})


class FeatureTests(unittest.TestCase):
    def test_derived_features_are_differentiable(self) -> None:
        library = FeatureLibrary()
        states = torch.tensor(straight(0.3).states, requires_grad=True)
        features = library.torch_features(states, ("x_position", "y_position", "speed", "progress"))
        self.assertEqual(tuple(features.shape), (24, 4))
        features[:, 2].sum().backward()
        self.assertIsNotNone(states.grad)
        self.assertTrue(torch.isfinite(states.grad).all())


class HypothesisBankTests(unittest.TestCase):
    def test_complete_items_recovered_from_truncated_json_array(self) -> None:
        text = '{"actions":[{"action":"retain_and_query","target_hypothesis_id":"h1"},{"action":"broken"'
        recovered = extract_json_array_objects(text, "actions")
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["target_hypothesis_id"], "h1")

    def test_compiler_accepts_competing_structures(self) -> None:
        library = FeatureLibrary()
        compiled = [compile_hypothesis(item, library) for item in canonical_initial_hypotheses()]
        self.assertEqual(len(compiled), 5)
        self.assertIn(("speed",), [item.variables for item in compiled])
        self.assertIn("independent", [item.hypothesis.coupling for item in compiled])

    def test_scalar_bound_rejects_multivariate_relation(self) -> None:
        library = FeatureLibrary()
        invalid = ConstraintHypothesis(
            "h_bad_bound",
            "invalid multivariate bound",
            ("x_position", "y_position"),
            "joint",
            "upper_bound",
            "max",
            "mlp",
            "increase the variables",
            "Used to verify the compiler rejects an ambiguous relation.",
        )
        with self.assertRaises(ValueError):
            compile_hypothesis(invalid, library)

    def test_model_family_changes_the_compiled_network(self) -> None:
        library = FeatureLibrary()
        linear = ConstraintHypothesis(
            "h_linear",
            "linear planar boundary",
            ("x_position", "y_position"),
            "joint",
            "forbidden_region",
            "max",
            "linear",
            "stress the planar path",
            "A linear control hypothesis used as a capacity comparison.",
        )
        registry = LearnerRegistry(
            library, hidden_dims=(12,), ensemble_size=1, seed=1, device=torch.device("cpu")
        )
        ensemble = registry.ensure(compile_hypothesis(linear, library))
        self.assertIsInstance(ensemble.members[0].joint_network, torch.nn.Linear)

    def test_revision_replaces_a_hypothesis(self) -> None:
        library = FeatureLibrary()
        initial = canonical_initial_hypotheses()[:2]
        bank = HypothesisBank.from_hypotheses(initial, library)
        replacement = ConstraintHypothesis(
            "h_xy_revised",
            "coupled planar region",
            ("x_position", "y_position"),
            "joint",
            "forbidden_region",
            "max",
            "mlp",
            "stress the coupled planar path",
            "The one-dimensional hypothesis left unexplained violations.",
            parent_id="h_x_position",
            generation=1,
        )
        bank.apply_actions(
            [RevisionAction("change_variables", "h_x_position", "Add the missing coordinate.", replacement)],
            library,
            outer_round=1,
        )
        active = {item.hypothesis_id for item in bank.active()}
        self.assertIn("h_xy_revised", active)
        self.assertEqual(bank.entries["h_x_position"].status, "retired")


class LearnerTests(unittest.TestCase):
    def test_trajectory_membership_training(self) -> None:
        library = FeatureLibrary()
        hypothesis = canonical_initial_hypotheses()[1]
        registry = LearnerRegistry(library, hidden_dims=(12,), ensemble_size=1, seed=3, device=torch.device("cpu"))
        ensemble = registry.ensure(compile_hypothesis(hypothesis, library))
        experts = [straight(1.0), straight(1.3)]
        records = [
            QueryRecord(straight(1.1), SAFE_LABEL, "toy", 0),
            QueryRecord(straight(0.0), VIOLATION_LABEL, "toy", 0),
            QueryRecord(straight(0.1), VIOLATION_LABEL, "toy", 0),
        ]
        summary = fit_ensemble(
            ensemble,
            experts,
            records,
            library,
            TrainerConfig(epochs=100, learning_rate=0.01, bootstrap_queries=False),
            seed=4,
            device=torch.device("cpu"),
        )
        self.assertTrue(np.isfinite(summary.mean_final_loss))
        self.assertEqual(registry.predict(hypothesis.hypothesis_id, straight(0.0))[0], VIOLATION_LABEL)
        self.assertEqual(registry.predict(hypothesis.hypothesis_id, straight(1.2))[0], SAFE_LABEL)


class SemanticAndOracleTests(unittest.TestCase):
    def test_evidence_policy_uses_scores_and_prunes(self) -> None:
        library = FeatureLibrary()
        policy = EvidencePolicyReasoner(SemanticConfig(beam_width=3, prune_per_round=2))
        hypotheses = policy.propose_initial("avoid an unknown obstacle", library)
        bank = HypothesisBank.from_hypotheses(hypotheses, library)
        evidence = []
        for index, hypothesis in enumerate(hypotheses):
            score = 0.9 - 0.1 * index
            evidence.append(
                HypothesisEvidence(
                    hypothesis.hypothesis_id,
                    score,
                    score,
                    score,
                    1.0,
                    1.0 - score,
                    1,
                    1,
                    0.2,
                    0.01,
                    score,
                    4,
                    len(hypothesis.variables),
                    score,
                    True,
                )
            )
        actions = policy.revise("avoid", library, bank, {"hypotheses": []}, evidence, 1)
        bank.apply_actions(actions, library, outer_round=1)
        self.assertEqual(len(bank.active()), 3)
        self.assertTrue(any(action.action == "propose_intervention" for action in actions))

    def test_membership_view_hides_geometry(self) -> None:
        oracle = CircularEvaluationOracle((1.0, 0.0), 0.3)
        view = oracle.membership_view()
        self.assertFalse(hasattr(view, "evaluation_geometry"))
        self.assertEqual(view.query(straight(0.0)), VIOLATION_LABEL)
        self.assertEqual(view.query(straight(1.0)), SAFE_LABEL)


if __name__ == "__main__":
    unittest.main()
