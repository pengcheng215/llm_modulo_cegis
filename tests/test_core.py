from __future__ import annotations

import sys
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_modulo_cegis.data import FeatureLibrary
from llm_modulo_cegis.hypotheses import (
    ConstraintClause,
    ConstraintHypothesis,
    HypothesisBank,
    RevisionAction,
    compile_hypothesis,
    extract_json_array_objects,
    hypothesis_from_dict,
)
from llm_modulo_cegis.evidence import (
    EvidenceCompiler,
    EvidenceConfig,
    _points_inside_convex_hull,
)
from llm_modulo_cegis.learner import (
    LearnerRegistry,
    TrainerConfig,
    _source_anchor_clause_masks,
    choose_decision_threshold,
    describe_ensemble_parameters,
    fit_ensemble,
)
from llm_modulo_cegis.falsifier import (
    FalsifierResult,
    FalsifierConfig,
    HypothesisFalsifier,
    generate_warmup_candidate,
)
from llm_modulo_cegis.loop import AcquisitionCandidate, LoopConfig, SemanticNumericCEGIS
from llm_modulo_cegis.oracle import CircularEvaluationOracle
from llm_modulo_cegis.semantic import (
    EvidencePolicyReasoner,
    FrozenBankSemanticReasoner,
    LocalQwenSemanticReasoner,
    OpenAISemanticReasoner,
    SemanticConfig,
    build_initial_prompt,
    canonical_initial_hypotheses,
    normalize_hypothesis_payload,
)
from llm_modulo_cegis.types import (
    HypothesisEvidence,
    InterventionSpec,
    QueryBuffer,
    QueryRecord,
    SAFE_LABEL,
    Trajectory,
    VIOLATION_LABEL,
)


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

    def test_composite_mixed_equality_and_inequality_compiles(self) -> None:
        library = FeatureLibrary()
        hypothesis = ConstraintHypothesis(
            "h_mixed",
            "orientation-like equality and speed bound",
            ("y_position", "speed"),
            "joint",
            "equality_band",
            "max",
            "linear",
            "stress either clause",
            "Two heterogeneous constraints may hold simultaneously.",
            clauses=(
                ConstraintClause(
                    "c_equal",
                    ("y_position",),
                    "joint",
                    "equality_band",
                    "max",
                    "linear",
                    "move away from the demonstrated value",
                    "The feature should remain near a learned reference.",
                ),
                ConstraintClause(
                    "c_upper",
                    ("speed",),
                    "joint",
                    "upper_bound",
                    "max",
                    "linear",
                    "increase speed",
                    "The dynamic feature may have a one-sided limit.",
                ),
            ),
        )
        compiled = compile_hypothesis(hypothesis, library)
        self.assertEqual(len(compiled.clauses), 2)
        registry = LearnerRegistry(
            library, hidden_dims=(8,), ensemble_size=1, seed=2, device=torch.device("cpu")
        )
        member = registry.ensure(compiled).members[0]
        self.assertEqual(len(member.clause_heads), 2)
        values = library.torch_features(torch.tensor(straight(0.0).states), compiled.variables)
        self.assertEqual(member.hard_trajectory_score(values).ndim, 0)

    def test_surface_repair_preserves_weak_llm_structure(self) -> None:
        library = FeatureLibrary()
        repaired = normalize_hypothesis_payload(
            {"id": "candidate-1", "name": "speed limit", "variable": "speed", "relation": "upper"}
        )
        compiled = compile_hypothesis(hypothesis_from_dict(repaired), library)
        self.assertEqual(compiled.hypothesis.relation, "upper_bound")
        self.assertEqual(compiled.hypothesis.model_family, "linear")

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
    def test_source_anchor_mask_excludes_unchanged_endpoint_shortcuts(self) -> None:
        library = FeatureLibrary()
        hypothesis = ConstraintHypothesis(
            "h_source_y",
            "source upper bound",
            ("y_position",),
            "joint",
            "upper_bound",
            "max",
            "linear",
            "increase y",
            "Exercise causal source-anchor MIL masking.",
        )
        registry = LearnerRegistry(
            library,
            hidden_dims=(8,),
            ensemble_size=1,
            seed=3,
            device=torch.device("cpu"),
        )
        model = registry.ensure(
            compile_hypothesis(hypothesis, library)
        ).members[0]
        with torch.no_grad():
            model.clause_heads[0].threshold.zero_()
            model.clause_heads[0].log_scale.zero_()
        x = np.linspace(0.0, 2.0, 6, dtype=np.float32)
        anchor_states = np.column_stack((x, np.ones_like(x)))
        candidate_states = anchor_states.copy()
        candidate_states[1:-1, 1] = -1.0
        anchor = Trajectory(
            anchor_states,
            metadata={"trajectory_id": "safe_anchor"},
        )
        candidate = Trajectory(
            candidate_states,
            metadata={"expert_id": "safe_anchor"},
        )
        record = QueryRecord(
            candidate,
            VIOLATION_LABEL,
            "shortcut",
            1,
            source_hypothesis_id=hypothesis.hypothesis_id,
        )
        masks, masked, unique_masked, unresolved, invariant = _source_anchor_clause_masks(
            [record],
            model,
            {"safe_anchor": anchor},
            model.compiled.variables,
            library,
            torch.device("cpu"),
            1.0e-6,
        )
        self.assertEqual(
            (masked, unique_masked, unresolved, invariant), (1, 1, 0, 0)
        )
        self.assertFalse(bool(masks[0, 0, 0]))
        self.assertFalse(bool(masks[0, 0, -1]))
        self.assertTrue(bool(torch.all(masks[0, 0, 1:-1])))
        features = library.torch_features(
            torch.as_tensor(candidate.states), model.compiled.variables
        ).unsqueeze(0)
        ordinary = model.trajectory_score(features, beta=20.0)
        causal, valid = model.trajectory_score_with_clause_masks(
            features, masks, beta=20.0
        )
        self.assertTrue(bool(valid.item()))
        self.assertGreater(float(ordinary.item()), 0.0)
        self.assertLess(float(causal.item()), 0.0)

        summary = fit_ensemble(
            registry.models[hypothesis.hypothesis_id],
            [anchor],
            [record],
            library,
            TrainerConfig(
                epochs=1,
                bootstrap_queries=False,
                latent_witness_weight=0.0,
                violation_pooling_mode="source_anchor_changed_states",
            ),
            seed=9,
            device=torch.device("cpu"),
        )
        self.assertEqual(
            summary.member_source_anchor_masked_violation_counts, (1,)
        )
        self.assertEqual(
            summary.member_unique_source_anchor_masked_violation_counts, (1,)
        )
        self.assertEqual(
            summary.member_source_anchor_unresolved_violation_counts, (0,)
        )

        # The tolerance is defined in the head's [-1, 1] normalized feature
        # coordinates.  With the public y span of 8, a raw 5e-6 change is a
        # normalized 1.25e-6 change and must therefore survive a 1e-6 mask.
        near_anchor_states = anchor_states.copy()
        near_anchor_states[2, 1] += 5.0e-6
        near_record = QueryRecord(
            Trajectory(
                near_anchor_states,
                metadata={"expert_id": "safe_anchor"},
            ),
            VIOLATION_LABEL,
            "shortcut",
            1,
            source_hypothesis_id=hypothesis.hypothesis_id,
        )
        near_masks, _, _, _, _ = _source_anchor_clause_masks(
            [near_record],
            model,
            {"safe_anchor": anchor},
            model.compiled.variables,
            library,
            torch.device("cpu"),
            1.0e-6,
        )
        self.assertTrue(bool(near_masks[0, 0, 2]))

    def test_source_anchor_mask_does_not_invent_a_pair_for_other_models(self) -> None:
        library = FeatureLibrary()
        hypothesis = canonical_initial_hypotheses()[0]
        registry = LearnerRegistry(
            library,
            hidden_dims=(8,),
            ensemble_size=1,
            seed=5,
            device=torch.device("cpu"),
        )
        model = registry.ensure(compile_hypothesis(hypothesis, library)).members[0]
        anchor = straight(1.0)
        anchor.metadata["trajectory_id"] = "anchor"
        candidate = straight(0.0)
        candidate.metadata["expert_id"] = "anchor"
        record = QueryRecord(
            candidate,
            VIOLATION_LABEL,
            "shortcut",
            1,
            source_hypothesis_id="h_different_source",
        )
        masks, masked, unique_masked, unresolved, invariant = _source_anchor_clause_masks(
            [record],
            model,
            {"anchor": anchor},
            model.compiled.variables,
            library,
            torch.device("cpu"),
            1.0e-6,
        )
        self.assertEqual(
            (masked, unique_masked, unresolved, invariant), (0, 0, 0, 0)
        )
        self.assertTrue(bool(torch.all(masks)))

        unresolved_record = QueryRecord(
            candidate,
            VIOLATION_LABEL,
            "shortcut",
            1,
            source_hypothesis_id=hypothesis.hypothesis_id,
        )
        unresolved_record.trajectory.metadata["expert_id"] = "missing_anchor"
        unresolved_masks, masked, unique_masked, unresolved, invariant = (
            _source_anchor_clause_masks(
                [unresolved_record],
                model,
                {"anchor": anchor},
                model.compiled.variables,
                library,
                torch.device("cpu"),
                1.0e-6,
            )
        )
        self.assertEqual(
            (masked, unique_masked, unresolved, invariant), (0, 0, 1, 0)
        )
        self.assertTrue(bool(torch.all(unresolved_masks)))

    def test_source_anchor_pooling_rejects_ambiguous_or_invalid_controls(self) -> None:
        library = FeatureLibrary()
        hypothesis = canonical_initial_hypotheses()[0]
        registry = LearnerRegistry(
            library,
            hidden_dims=(8,),
            ensemble_size=1,
            seed=8,
            device=torch.device("cpu"),
        )
        ensemble = registry.ensure(compile_hypothesis(hypothesis, library))
        left = straight(1.0)
        right = straight(1.2)
        left.metadata["trajectory_id"] = "duplicate"
        right.metadata["trajectory_id"] = "duplicate"
        config = TrainerConfig(
            epochs=1,
            bootstrap_queries=False,
            latent_witness_weight=0.0,
            violation_pooling_mode="source_anchor_changed_states",
        )
        with self.assertRaisesRegex(ValueError, "unique expert trajectory_id"):
            fit_ensemble(
                ensemble,
                [left, right],
                [],
                library,
                config,
                seed=2,
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            fit_ensemble(
                ensemble,
                [left],
                [],
                library,
                TrainerConfig(
                    epochs=1,
                    violation_pooling_mode="source_anchor_changed_states",
                    violation_pooling_change_tolerance=float("nan"),
                ),
                seed=2,
                device=torch.device("cpu"),
            )

    def test_composite_source_anchor_mask_blocks_nuisance_clause_gradients(self) -> None:
        library = FeatureLibrary()
        hypothesis = ConstraintHypothesis(
            "h_composite_mask",
            "two coordinate limits",
            ("x_position", "y_position"),
            "joint",
            "upper_bound",
            "max",
            "linear",
            "increase either coordinate",
            "Exercise clause-aware causal masks.",
            clauses=(
                ConstraintClause(
                    "c_x",
                    ("x_position",),
                    "joint",
                    "upper_bound",
                    "max",
                    "linear",
                    "increase x",
                    "x clause",
                ),
                ConstraintClause(
                    "c_y",
                    ("y_position",),
                    "joint",
                    "upper_bound",
                    "max",
                    "linear",
                    "increase y",
                    "y clause",
                ),
            ),
        )
        registry = LearnerRegistry(
            library,
            hidden_dims=(8,),
            ensemble_size=1,
            seed=12,
            device=torch.device("cpu"),
        )
        model = registry.ensure(
            compile_hypothesis(hypothesis, library)
        ).members[0]
        with torch.no_grad():
            for head in model.clause_heads:
                head.threshold.zero_()
                head.log_scale.zero_()
        anchor = straight(0.0)
        anchor.metadata["trajectory_id"] = "composite_anchor"
        candidate = anchor.copy()
        candidate.states[len(candidate.states) // 2, 1] = 1.0
        candidate.metadata["expert_id"] = "composite_anchor"
        record = QueryRecord(
            candidate,
            VIOLATION_LABEL,
            "shortcut",
            1,
            source_hypothesis_id=hypothesis.hypothesis_id,
        )
        masks, masked, unique_masked, unresolved, invariant = (
            _source_anchor_clause_masks(
                [record],
                model,
                {"composite_anchor": anchor},
                model.compiled.variables,
                library,
                torch.device("cpu"),
                1.0e-6,
            )
        )
        self.assertEqual(
            (masked, unique_masked, unresolved, invariant), (1, 1, 0, 0)
        )
        self.assertFalse(bool(torch.any(masks[0, 0])))
        self.assertEqual(int(torch.sum(masks[0, 1]).item()), 1)
        features = library.torch_features(
            torch.as_tensor(candidate.states), model.compiled.variables
        ).unsqueeze(0)
        smooth, valid = model.trajectory_score_with_clause_masks(
            features, masks, beta=20.0
        )
        hard, hard_valid = model.trajectory_score_with_clause_masks(
            features, masks, beta=20.0, hard=True
        )
        self.assertTrue(bool(valid.item()))
        self.assertTrue(bool(hard_valid.item()))
        self.assertTrue(bool(torch.isfinite(smooth).all()))
        self.assertTrue(bool(torch.isfinite(hard).all()))
        torch.nn.functional.softplus(0.2 - smooth).mean().backward()
        x_head, y_head = model.clause_heads
        self.assertEqual(float(x_head.threshold.grad.item()), 0.0)
        self.assertEqual(float(x_head.log_scale.grad.item()), 0.0)
        self.assertTrue(bool(torch.isfinite(y_head.threshold.grad).all()))
        self.assertTrue(bool(torch.isfinite(y_head.log_scale.grad).all()))
        with self.assertRaisesRegex(ValueError, "smooth-max beta must be positive"):
            model.trajectory_score_with_clause_masks(features, masks, beta=0.0)

    def test_full_query_buffer_mode_is_fit_seed_independent(self) -> None:
        library = FeatureLibrary()
        hypothesis = canonical_initial_hypotheses()[0]
        registry = LearnerRegistry(
            library,
            hidden_dims=(8,),
            ensemble_size=2,
            seed=17,
            device=torch.device("cpu"),
        )
        initial = registry.ensure(compile_hypothesis(hypothesis, library))
        left = deepcopy(initial)
        right = deepcopy(initial)
        experts = [straight(1.0), straight(1.3)]
        records = [
            QueryRecord(straight(1.1), SAFE_LABEL, "toy", 0),
            QueryRecord(straight(0.0), VIOLATION_LABEL, "toy", 0),
            QueryRecord(straight(0.2), VIOLATION_LABEL, "toy", 0),
        ]
        config = TrainerConfig(
            epochs=2,
            learning_rate=0.001,
            latent_witness_weight=0.0,
            bootstrap_queries=False,
        )
        left_summary = fit_ensemble(
            left,
            experts,
            records,
            library,
            config,
            seed=3,
            device=torch.device("cpu"),
        )
        right_summary = fit_ensemble(
            right,
            experts,
            records,
            library,
            config,
            seed=999,
            device=torch.device("cpu"),
        )
        self.assertEqual(left_summary.available_query_count, 3)
        self.assertEqual(left_summary.member_query_draw_counts, (3, 3))
        self.assertEqual(left_summary.member_unique_query_counts, (3, 3))
        self.assertEqual(left_summary.member_unique_safe_query_counts, (1, 1))
        self.assertEqual(left_summary.member_unique_violation_query_counts, (2, 2))
        self.assertEqual(
            left_summary.query_coverage_dict(),
            right_summary.query_coverage_dict(),
        )
        for key, value in left.state_dict().items():
            self.assertTrue(torch.equal(value, right.state_dict()[key]), key)

    def test_coverage_preserving_bootstrap_never_omits_a_query(self) -> None:
        defaults = TrainerConfig()
        self.assertFalse(defaults.bootstrap_queries)
        self.assertFalse(defaults.bootstrap_ensure_full_coverage)
        self.assertEqual(
            defaults.violation_pooling_mode, "source_anchor_changed_states"
        )
        library = FeatureLibrary()
        hypothesis = canonical_initial_hypotheses()[0]
        registry = LearnerRegistry(
            library,
            hidden_dims=(8,),
            ensemble_size=2,
            seed=23,
            device=torch.device("cpu"),
        )
        ensemble = registry.ensure(compile_hypothesis(hypothesis, library))
        records = [
            QueryRecord(straight(1.0 + 0.1 * index), SAFE_LABEL, "toy", 0)
            for index in range(4)
        ] + [
            QueryRecord(straight(-0.1 * index), VIOLATION_LABEL, "toy", 0)
            for index in range(6)
        ]
        summary = fit_ensemble(
            ensemble,
            [straight(1.6)],
            records,
            library,
            TrainerConfig(
                epochs=1,
                latent_witness_weight=0.0,
                bootstrap_queries=True,
                bootstrap_ensure_full_coverage=True,
            ),
            seed=5,
            device=torch.device("cpu"),
        )
        self.assertTrue(summary.bootstrap_queries)
        self.assertTrue(summary.bootstrap_ensure_full_coverage)
        self.assertEqual(summary.member_unique_query_counts, (10, 10))
        self.assertEqual(summary.member_unique_safe_query_counts, (4, 4))
        self.assertEqual(summary.member_unique_violation_query_counts, (6, 6))
        self.assertTrue(all(count >= 10 for count in summary.member_query_draw_counts))

    def test_equivalent_structures_initialize_independently_of_llm_generated_id(self) -> None:
        library = FeatureLibrary()
        left = ConstraintHypothesis(
            "h_first_name", "first", ("x_position", "y_position"),
            "joint", "forbidden_region", "max", "mlp", "enter", "same structure"
        )
        right = ConstraintHypothesis(
            "h_completely_different_id", "second", ("x_position", "y_position"),
            "joint", "forbidden_region", "max", "mlp", "enter", "same structure"
        )
        registry = LearnerRegistry(
            library, hidden_dims=(8,), ensemble_size=1, seed=13, device=torch.device("cpu")
        )
        left_model = registry.ensure(compile_hypothesis(left, library))
        right_model = registry.ensure(compile_hypothesis(right, library))
        for key, value in left_model.state_dict().items():
            if key == "decision_threshold":
                continue
            self.assertTrue(torch.equal(value, right_model.state_dict()[key]), key)

    def test_diagnostics_convert_normalized_scalar_threshold_to_raw_units(self) -> None:
        library = FeatureLibrary()
        hypothesis = ConstraintHypothesis(
            "h_y_lower",
            "y lower bound",
            ("y_position",),
            "joint",
            "lower_bound",
            "max",
            "linear",
            "decrease y",
            "diagnostic conversion test",
        )
        registry = LearnerRegistry(library, hidden_dims=(8,), ensemble_size=1, seed=1, device=torch.device("cpu"))
        ensemble = registry.ensure(compile_hypothesis(hypothesis, library))
        with torch.no_grad():
            ensemble.members[0].clause_heads[0].threshold.fill_(-0.35)
        diagnostic = describe_ensemble_parameters(ensemble)
        member = diagnostic["clauses"][0]["members"][0]
        self.assertAlmostEqual(member["threshold_raw"], -1.4, places=5)

    def test_inference_uses_hard_max_for_a_single_violating_state(self) -> None:
        library = FeatureLibrary()
        hypothesis = ConstraintHypothesis(
            "h_y_floor",
            "y floor",
            ("y_position",),
            "joint",
            "lower_bound",
            "max",
            "linear",
            "decrease y",
            "hard max regression test",
        )
        registry = LearnerRegistry(library, hidden_dims=(8,), ensemble_size=1, seed=2, device=torch.device("cpu"))
        ensemble = registry.ensure(compile_hypothesis(hypothesis, library))
        with torch.no_grad():
            ensemble.members[0].clause_heads[0].threshold.zero_()
        trajectory = straight(1.0)
        trajectory.states[len(trajectory.states) // 2, 1] = -0.1
        features = library.torch_features(torch.as_tensor(trajectory.states), ("y_position",))
        smooth_score = float(ensemble.mean_trajectory_score(features.unsqueeze(0), beta=20.0).item())
        self.assertLess(smooth_score, 0.0)
        prediction, hard_score, _ = registry.predict(hypothesis.hypothesis_id, trajectory)
        self.assertGreater(hard_score, 0.0)
        self.assertEqual(prediction, VIOLATION_LABEL)
        ensemble.set_decision_threshold(hard_score + 0.01)
        self.assertEqual(registry.predict(hypothesis.hypothesis_id, trajectory)[0], SAFE_LABEL)

    def test_threshold_calibration_respects_experts_and_separates_labels(self) -> None:
        result = choose_decision_threshold(
            calibration_scores=[-0.2, 0.1, 0.3, 0.5],
            calibration_labels=[SAFE_LABEL, SAFE_LABEL, VIOLATION_LABEL, VIOLATION_LABEL],
            expert_scores=[-0.5, 0.15],
            minimum_expert_safe_rate=1.0,
        )
        threshold = result["selected_threshold"]
        self.assertGreaterEqual(threshold, 0.15)
        self.assertLess(threshold, 0.3)
        self.assertEqual(result["selected_metrics"]["balanced_accuracy"], 1.0)
        self.assertTrue(result["expert_constraint_satisfied"])

    def test_cegis_threshold_calibration_uses_only_disjoint_calibration_labels(self) -> None:
        hypothesis = ConstraintHypothesis(
            "h_threshold_fit",
            "threshold fit",
            ("y_position",),
            "joint",
            "lower_bound",
            "max",
            "linear",
            "decrease y",
            "threshold calibration test",
        )

        class YScoreLibrary:
            @staticmethod
            def torch_features(
                states: torch.Tensor,
                variables: tuple[str, ...],
            ) -> torch.Tensor:
                del variables
                return states[..., 1:2]

        class YScoreEnsemble:
            def __init__(self) -> None:
                self.compiled = SimpleNamespace(variables=("y_position",))
                self.decision_threshold = torch.tensor(0.0)

            @staticmethod
            def mean_hard_trajectory_score(features: torch.Tensor) -> torch.Tensor:
                return torch.max(features[..., 0])

            def set_decision_threshold(self, value: float) -> None:
                self.decision_threshold.fill_(float(value))

        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            calibrate_decision_threshold_during_cegis=True,
            decision_threshold_minimum_per_label=1,
            decision_threshold_minimum_fit_expert_safe_rate=1.0,
        )
        controller.buffer = QueryBuffer()
        # Gradient-training and selection-holdout extremes must not influence
        # the separately calibrated threshold.
        for value, label in ((100.0, SAFE_LABEL), (-100.0, VIOLATION_LABEL)):
            controller.buffer.add(QueryRecord(straight(value), label, "warmup", 0))
        controller.buffer.add(
            QueryRecord(straight(100.0), SAFE_LABEL, "warmup_validation", 0)
        )
        controller.buffer.add(
            QueryRecord(straight(-0.2), SAFE_LABEL, "final_calibration", 0)
        )
        controller.buffer.add(
            QueryRecord(straight(0.3), VIOLATION_LABEL, "final_calibration", 0)
        )
        ensemble = YScoreEnsemble()
        controller.registry = SimpleNamespace(models={hypothesis.hypothesis_id: ensemble})
        controller.library = YScoreLibrary()
        controller.device = torch.device("cpu")
        controller.experts = [straight(-0.5)]
        controller.threshold_calibration_history = []
        controller.latest_threshold_calibration = {}

        controller._calibrate_active_decision_thresholds([hypothesis], outer_round=1)

        threshold = float(ensemble.decision_threshold.item())
        self.assertGreater(threshold, -0.2)
        self.assertLess(threshold, 0.3)
        diagnostic = controller.latest_threshold_calibration[hypothesis.hypothesis_id]
        self.assertEqual(diagnostic["calibration_query_count"], 2)
        self.assertEqual(
            diagnostic["calibration_label_counts"],
            {"safe": 1, "violation": 1},
        )
        self.assertEqual(diagnostic["selected_metrics"]["balanced_accuracy"], 1.0)

    def test_warmup_validation_keeps_first_preselection_prediction(self) -> None:
        hypothesis = canonical_initial_hypotheses()[0]
        record = QueryRecord(straight(0.0), SAFE_LABEL, "warmup_validation", 0)
        record.predictions_before_query[hypothesis.hypothesis_id] = VIOLATION_LABEL
        record.scores_before_query[hypothesis.hypothesis_id] = 99.0
        record.uncertainties_before_query[hypothesis.hypothesis_id] = 99.0
        controller = object.__new__(SemanticNumericCEGIS)
        controller.buffer = QueryBuffer()
        controller.buffer.add(record)
        controller.registry = SimpleNamespace(
            predict=lambda hypothesis_id, trajectory: (SAFE_LABEL, -0.25, 0.125)
        )

        controller._score_warmup_validation([hypothesis])

        self.assertEqual(
            record.predictions_before_query[hypothesis.hypothesis_id],
            VIOLATION_LABEL,
        )
        self.assertEqual(record.scores_before_query[hypothesis.hypothesis_id], 99.0)
        self.assertEqual(
            record.uncertainties_before_query[hypothesis.hypothesis_id],
            99.0,
        )

    def test_finalization_candidate_choice_uses_disjoint_selection_holdout(self) -> None:
        def row(name: str, holdout: float, calibration: float) -> dict[str, object]:
            return {
                "candidate_name": name,
                "selection_holdout": {
                    "balanced_accuracy": holdout,
                    "violation_recall": holdout,
                },
                "calibration": {
                    "selected_metrics": {
                        "balanced_accuracy": calibration,
                        "violation_recall": calibration,
                    }
                },
            }

        calibration_favorite = row("scratch_restart_0", 0.75, 1.0)
        holdout_favorite = row("incumbent_finetune", 1.0, 0.75)
        selected = SemanticNumericCEGIS._choose_finalization_candidate(
            [calibration_favorite, holdout_favorite]
        )
        self.assertEqual(selected["candidate_name"], "incumbent_finetune")

        tied_incumbent = row("incumbent_calibrated", 1.0, 0.75)
        tied_finetune = row("incumbent_finetune", 1.0, 0.75)
        selected_tie = SemanticNumericCEGIS._choose_finalization_candidate(
            [tied_incumbent, tied_finetune]
        )
        self.assertEqual(selected_tie["candidate_name"], "incumbent_finetune")

    def test_false_unsafe_trust_region_is_a_hard_pointwise_projection(self) -> None:
        reference = torch.zeros((4, 2), dtype=torch.float32)
        candidate = torch.tensor([[0.3, 0.4], [0.0, 0.0], [-0.2, 0.0], [0.06, 0.08]])
        projected = HypothesisFalsifier._project_to_trust_region(candidate, reference, 0.1)
        distances = torch.linalg.vector_norm(projected - reference, dim=-1)
        self.assertTrue(torch.all(distances <= 0.100001))
        self.assertTrue(torch.allclose(projected[1], candidate[1]))
        self.assertTrue(torch.allclose(projected[3], candidate[3]))

    def test_false_unsafe_rejects_nonpositive_trust_radius(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            FalsifierConfig(false_unsafe_trust_radius=0.0)

    def test_false_unsafe_default_is_preregistered_single_radius(self) -> None:
        config = FalsifierConfig()
        self.assertEqual(config.false_unsafe_trust_radius, 0.32)
        self.assertEqual(config.false_unsafe_radius_ladder, ())

    def test_false_unsafe_objective_uses_calibrated_hard_margin(self) -> None:
        class FakeYMaxEnsemble(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.compiled = SimpleNamespace(
                    variables=("y_position",),
                    clauses=(),
                    hypothesis=SimpleNamespace(hypothesis_id="h_y_max"),
                )
                self.register_buffer("decision_threshold", torch.tensor(0.20))

            @staticmethod
            def mean_hard_trajectory_score(features: torch.Tensor) -> torch.Tensor:
                return torch.amax(features[..., 0], dim=-1)

            @staticmethod
            def mean_trajectory_score(features: torch.Tensor, beta: float) -> torch.Tensor:
                del beta
                # Deliberately above the target: using this smooth score would
                # incorrectly make the hard-margin loss zero.
                return torch.amax(features[..., 0], dim=-1) + 1.0

            @classmethod
            def trajectory_uncertainty(cls, features: torch.Tensor, beta: float) -> torch.Tensor:
                del beta
                return torch.zeros_like(cls.mean_hard_trajectory_score(features))

            @classmethod
            def hard_trajectory_uncertainty(cls, features: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(cls.mean_hard_trajectory_score(features))

        library = FeatureLibrary()
        falsifier = HypothesisFalsifier(
            library,
            FalsifierConfig(
                boundary_weight=1.0,
                expert_weight=0.0,
                smoothness_weight=0.0,
                length_weight=0.0,
                uncertainty_weight=0.0,
                step_penalty_weight=0.0,
                workspace_penalty_weight=0.0,
                invariance_weight=0.0,
                false_unsafe_hard_margin=0.05,
                false_unsafe_smooth_margin_weight=0.0,
            ),
            (0.0, 10.0),
            (-4.0, 4.0),
            torch.device("cpu"),
        )
        expert = torch.as_tensor(straight(0.0).states, dtype=torch.float32)
        states = expert.detach().clone()
        states[len(states) // 2, 1] = 0.10
        path = states.detach().clone().requires_grad_(True)
        loss, score, _ = falsifier._objective(
            path,
            expert,
            FakeYMaxEnsemble(),
            "false_unsafe",
            InterventionSpec("h_y_max", "model_false_unsafe"),
        )
        self.assertAlmostEqual(float(score.item()), 0.10, places=5)
        self.assertAlmostEqual(float(loss.item()), 0.0225, places=5)
        loss.backward()
        self.assertLess(float(path.grad[len(path) // 2, 1].item()), 0.0)

        above_target = expert.detach().clone()
        above_target[len(above_target) // 2, 1] = 0.26
        achieved_loss, _, _ = falsifier._objective(
            above_target,
            expert,
            FakeYMaxEnsemble(),
            "false_unsafe",
            InterventionSpec("h_y_max", "model_false_unsafe"),
        )
        self.assertAlmostEqual(float(achieved_loss.item()), 0.0, places=6)

    def test_false_unsafe_objective_targets_named_composite_clause(self) -> None:
        class FakeCompositeEnsemble(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                clauses = tuple(
                    SimpleNamespace(clause=SimpleNamespace(clause_id=clause_id))
                    for clause_id in ("c_target", "c_nuisance")
                )
                self.compiled = SimpleNamespace(
                    variables=("y_position",),
                    clauses=clauses,
                    hypothesis=SimpleNamespace(hypothesis_id="h_composite"),
                )
                self.register_buffer("decision_threshold", torch.tensor(0.20))

            @staticmethod
            def mean_hard_clause_trajectory_scores(features: torch.Tensor) -> torch.Tensor:
                target = torch.amax(features[..., 0], dim=-1)
                nuisance = torch.ones_like(target)
                return torch.stack((target, nuisance), dim=-1)

            @classmethod
            def mean_hard_trajectory_score(cls, features: torch.Tensor) -> torch.Tensor:
                return torch.amax(cls.mean_hard_clause_trajectory_scores(features), dim=-1)

            @classmethod
            def mean_clause_trajectory_scores(
                cls,
                features: torch.Tensor,
                beta: float,
            ) -> torch.Tensor:
                del beta
                return cls.mean_hard_clause_trajectory_scores(features)

            @classmethod
            def mean_trajectory_score(cls, features: torch.Tensor, beta: float) -> torch.Tensor:
                del beta
                return cls.mean_hard_trajectory_score(features)

            @classmethod
            def trajectory_uncertainty(cls, features: torch.Tensor, beta: float) -> torch.Tensor:
                del beta
                return torch.zeros_like(cls.mean_hard_trajectory_score(features))

            @classmethod
            def hard_trajectory_uncertainty(cls, features: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(cls.mean_hard_trajectory_score(features))

        falsifier = HypothesisFalsifier(
            FeatureLibrary(),
            FalsifierConfig(
                boundary_weight=1.0,
                expert_weight=0.0,
                smoothness_weight=0.0,
                length_weight=0.0,
                uncertainty_weight=0.0,
                step_penalty_weight=0.0,
                workspace_penalty_weight=0.0,
                invariance_weight=0.0,
                false_unsafe_hard_margin=0.05,
                false_unsafe_smooth_margin_weight=0.0,
            ),
            (0.0, 10.0),
            (-4.0, 4.0),
            torch.device("cpu"),
        )
        expert = torch.as_tensor(straight(0.0).states, dtype=torch.float32)
        path = expert.detach().clone()
        path[len(path) // 2, 1] = 0.10
        loss, targeted_score, _ = falsifier._objective(
            path,
            expert,
            FakeCompositeEnsemble(),
            "false_unsafe",
            InterventionSpec(
                "h_composite",
                "model_false_unsafe",
                clause_id="c_target",
            ),
        )
        self.assertAlmostEqual(float(targeted_score.item()), 0.10, places=5)
        self.assertAlmostEqual(float(loss.item()), 0.0225, places=5)
        with self.assertRaisesRegex(ValueError, "unknown clause_id"):
            falsifier._objective(
                path,
                expert,
                FakeCompositeEnsemble(),
                "false_unsafe",
                InterventionSpec(
                    "h_composite",
                    "model_false_unsafe",
                    clause_id="c_missing",
                ),
            )

    def test_initial_hard_margin_crossing_is_checkpointed_before_optimizer_step(self) -> None:
        class FakeYMaxEnsemble(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.compiled = SimpleNamespace(
                    variables=("y_position",),
                    clauses=(),
                    hypothesis=SimpleNamespace(hypothesis_id="h_initial_crossing"),
                )
                self.register_buffer("decision_threshold", torch.tensor(0.20))

            @staticmethod
            def mean_hard_trajectory_score(features: torch.Tensor) -> torch.Tensor:
                return torch.amax(features[..., 0], dim=-1)

            @staticmethod
            def mean_trajectory_score(features: torch.Tensor, beta: float) -> torch.Tensor:
                del beta
                return torch.amax(features[..., 0], dim=-1)

            @classmethod
            def trajectory_uncertainty(cls, features: torch.Tensor, beta: float) -> torch.Tensor:
                del beta
                return torch.zeros_like(cls.mean_hard_trajectory_score(features))

            @classmethod
            def hard_trajectory_uncertainty(cls, features: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(cls.mean_hard_trajectory_score(features))

            @staticmethod
            def mean_state_score(features: torch.Tensor) -> torch.Tensor:
                return features[..., 0]

        phase = np.linspace(0.0, 1.0, 24, dtype=np.float32)
        expert = straight(0.0)
        expert.states[:, 1] = 0.30 * np.sin(np.pi * phase) ** 2
        falsifier = HypothesisFalsifier(
            FeatureLibrary(),
            FalsifierConfig(
                steps=1,
                expert_weight=0.0,
                smoothness_weight=0.0,
                length_weight=0.0,
                boundary_weight=1.0,
                uncertainty_weight=0.0,
                step_penalty_weight=0.0,
                workspace_penalty_weight=0.0,
                invariance_weight=0.0,
                false_unsafe_trust_radius=0.50,
                false_unsafe_hard_margin=0.05,
                false_unsafe_smooth_margin_weight=0.0,
            ),
            (0.0, 10.0),
            (-4.0, 4.0),
            torch.device("cpu"),
        )
        result = falsifier.generate(
            FakeYMaxEnsemble(),
            expert,
            InterventionSpec("h_initial_crossing", "model_false_unsafe"),
            initialization_mix=0.0,
            restart_index=0,
        )
        self.assertEqual(result.trajectory.metadata["hard_margin_checkpoint_step"], 0)
        self.assertTrue(result.trajectory.metadata["optimization_hard_margin_achieved"])

        class RejectFirstCheckpointFalsifier(HypothesisFalsifier):
            def __init__(self) -> None:
                super().__init__(
                    falsifier.library,
                    falsifier.config,
                    falsifier.workspace_x,
                    falsifier.workspace_y,
                    falsifier.device,
                )
                self.validation_calls = 0

            def validate(
                self,
                trajectory: Trajectory,
                anchor: Trajectory,
            ) -> tuple[bool, str]:
                self.validation_calls += 1
                if self.validation_calls == 1:
                    return False, "synthetic_early_invalid"
                return super().validate(trajectory, anchor)

        validating_falsifier = RejectFirstCheckpointFalsifier()
        later_result = validating_falsifier.generate(
            FakeYMaxEnsemble(),
            expert,
            InterventionSpec("h_initial_crossing", "model_false_unsafe"),
            initialization_mix=0.0,
            restart_index=0,
        )
        self.assertEqual(later_result.trajectory.metadata["hard_margin_checkpoint_step"], 1)
        self.assertEqual(
            later_result.trajectory.metadata["rejected_hard_margin_checkpoints"],
            {"synthetic_early_invalid": 1},
        )

    def test_falsifier_records_source_model_witness_before_oracle(self) -> None:
        library = FeatureLibrary()
        hypothesis = ConstraintHypothesis(
            "h_witness", "planar region", ("x_position", "y_position"),
            "joint", "forbidden_region", "max", "mlp", "enter region", "witness audit"
        )
        registry = LearnerRegistry(
            library, hidden_dims=(8,), ensemble_size=1, seed=31, device=torch.device("cpu")
        )
        ensemble = registry.ensure(compile_hypothesis(hypothesis, library))
        falsifier = HypothesisFalsifier(
            library,
            FalsifierConfig(steps=1, false_unsafe_trust_radius=0.08),
            (0.0, 10.0),
            (-4.0, 4.0),
            torch.device("cpu"),
        )
        result = falsifier.generate(
            ensemble,
            straight(1.0),
            InterventionSpec(hypothesis.hypothesis_id, "model_false_safe"),
            initialization_mix=0.1,
        )
        index = result.trajectory.metadata["source_witness_index"]
        self.assertGreaterEqual(index, 0)
        self.assertLess(index, len(result.trajectory.states))
        self.assertEqual(
            result.trajectory.metadata["source_witness_kind"],
            "model_argmax_before_oracle",
        )

    def test_warmup_witness_is_maximum_intervention_not_data_novelty(self) -> None:
        expert = straight(1.0)
        candidate = generate_warmup_candidate(
            expert,
            8,
            np.random.default_rng(4),
            (0.0, 10.0),
            (-4.0, 4.0),
        )
        expected = int(np.argmax(np.linalg.norm(candidate.states - expert.states, axis=1)))
        self.assertEqual(candidate.metadata["source_witness_index"], expected)
        self.assertEqual(
            candidate.metadata["source_witness_kind"],
            "intervention_max_deformation",
        )

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
    def test_impossible_finalization_holdout_requirement_fails_fast(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "finalization_minimum_selection_per_label exceeds",
        ):
            SemanticNumericCEGIS(
                task_description="test",
                feature_library=FeatureLibrary(),
                reasoner=None,
                registry=None,
                trainer_config=TrainerConfig(),
                evidence_compiler=None,
                falsifier=SimpleNamespace(
                    config=SimpleNamespace(false_unsafe_hard_margin=0.05)
                ),
                oracle=None,
                evaluation_oracle=None,
                experts=[straight(1.0)],
                heldout_experts=[],
                workspace_x=(0.0, 10.0),
                workspace_y=(-4.0, 4.0),
                loop_config=LoopConfig(
                    finalize_qualified_champion=True,
                    warmup_validation_safe_count=2,
                    warmup_validation_violation_count=2,
                    final_calibration_safe_count=2,
                    final_calibration_violation_count=2,
                    finalization_minimum_calibration_per_label=2,
                    finalization_minimum_selection_per_label=3,
                ),
                output_dir="unused",
                seed=0,
                device=torch.device("cpu"),
            )

    @staticmethod
    def _curved_expert(name: str, offset: float = 0.0) -> Trajectory:
        phase = np.linspace(0.0, 1.0, 24, dtype=np.float32)
        states = np.column_stack(
            (
                2.0 * phase,
                offset + 0.8 * np.sin(np.pi * phase),
            )
        ).astype(np.float32)
        actions = np.zeros_like(states)
        actions[:-1] = np.diff(states, axis=0)
        return Trajectory(states, actions, metadata={"trajectory_id": name})

    def test_warmup_candidates_are_deterministic_demo_relative_pairs(self) -> None:
        expert = self._curved_expert("expert_pair")
        toward = generate_warmup_candidate(
            expert,
            4,
            np.random.default_rng(1),
            (0.0, 2.0),
            (-2.0, 2.0),
        )
        away = generate_warmup_candidate(
            expert,
            5,
            np.random.default_rng(999),
            (0.0, 2.0),
            (-2.0, 2.0),
        )
        repeated = generate_warmup_candidate(
            expert,
            4,
            np.random.default_rng(12345),
            (0.0, 2.0),
            (-2.0, 2.0),
        )
        line = np.linspace(expert.states[0], expert.states[-1], len(expert.states))
        alpha = 0.06
        self.assertTrue(np.allclose(toward.states - line, (1.0 - alpha) * (expert.states - line)))
        self.assertTrue(np.allclose(away.states - line, (1.0 + alpha) * (expert.states - line)))
        self.assertTrue(np.array_equal(toward.states, repeated.states))
        self.assertEqual(toward.metadata["warmup_pair_index"], 2)
        self.assertEqual(away.metadata["warmup_pair_index"], 2)
        self.assertEqual(toward.metadata["warmup_direction"], "toward_chord")
        self.assertEqual(away.metadata["warmup_direction"], "continue_detour")
        self.assertTrue(np.array_equal(toward.states[[0, -1]], expert.states[[0, -1]]))
        self.assertTrue(np.allclose(toward.actions[:-1], np.diff(toward.states, axis=0)))

    def test_warmup_pairing_reaches_fixed_label_coverage_and_partitions_holdouts(self) -> None:
        class StateOnlyMembershipOracle:
            def __init__(self) -> None:
                self.query_count = 0

            def query(self, trajectory: Trajectory) -> int:
                self.query_count += 1
                return SAFE_LABEL if float(np.max(trajectory.states[:, 1])) > 0.8 else VIOLATION_LABEL

        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            warmup_queries=14,
            max_warmup_queries=14,
            minimum_safe_query_count=5,
            minimum_violation_query_count=5,
            warmup_validation_safe_count=2,
            warmup_validation_violation_count=2,
            final_calibration_safe_count=2,
            final_calibration_violation_count=2,
        )
        controller.experts = [
            self._curved_expert(
                f"expert_{index}",
                0.10 if index == 3 else 0.0,
            )
            for index in range(5)
        ]
        controller.buffer = QueryBuffer()
        controller.rng = np.random.default_rng(73)
        controller.workspace_x = (0.0, 2.0)
        controller.workspace_y = (-2.0, 2.0)
        controller.oracle = StateOnlyMembershipOracle()
        controller.progress = lambda value: None
        controller.stage_diagnostics = []
        with tempfile.TemporaryDirectory() as temporary:
            controller.output_dir = Path(temporary)
            controller._collect_warmup()
        self.assertEqual(controller.oracle.query_count, 14)
        self.assertEqual(controller.buffer.label_counts(), {"safe": 8, "violation": 6})
        records = controller.buffer.records
        for pair_index in range(7):
            left = records[2 * pair_index].trajectory.metadata
            right = records[2 * pair_index + 1].trajectory.metadata
            self.assertEqual(left["expert_id"], right["expert_id"])
            self.assertEqual(left["alpha"], right["alpha"])
            self.assertEqual(left["warmup_direction"], "toward_chord")
            self.assertEqual(right["warmup_direction"], "continue_detour")
        sources = [record.source for record in records]
        self.assertEqual(sources.count("warmup"), 6)
        self.assertEqual(sources.count("warmup_validation"), 4)
        self.assertEqual(sources.count("final_calibration"), 4)
        pair_ids_by_source = {
            source: {
                int(record.trajectory.metadata["warmup_pair_index"])
                for record in records
                if record.source == source
            }
            for source in ("warmup", "warmup_validation", "final_calibration")
        }
        self.assertEqual(pair_ids_by_source["warmup"], {0, 1, 3})
        self.assertEqual(pair_ids_by_source["warmup_validation"], {2, 4})
        self.assertEqual(pair_ids_by_source["final_calibration"], {5, 6})
        self.assertTrue(
            pair_ids_by_source["warmup"].isdisjoint(
                pair_ids_by_source["warmup_validation"]
            )
        )
        self.assertTrue(
            pair_ids_by_source["warmup"].isdisjoint(
                pair_ids_by_source["final_calibration"]
            )
        )
        self.assertTrue(
            pair_ids_by_source["warmup_validation"].isdisjoint(
                pair_ids_by_source["final_calibration"]
            )
        )

    def test_warmup_minimum_label_count_controls_the_query_target(self) -> None:
        class StateOnlyMembershipOracle:
            def __init__(self) -> None:
                self.query_count = 0

            def query(self, trajectory: Trajectory) -> int:
                self.query_count += 1
                return SAFE_LABEL if float(np.max(trajectory.states[:, 1])) > 0.8 else VIOLATION_LABEL

        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            warmup_queries=2,
            max_warmup_queries=14,
            minimum_label_count=6,
            minimum_safe_query_count=0,
            minimum_violation_query_count=0,
            warmup_validation_safe_count=0,
            warmup_validation_violation_count=0,
            final_calibration_safe_count=0,
            final_calibration_violation_count=0,
        )
        controller.experts = [self._curved_expert(f"expert_{index}") for index in range(5)]
        controller.buffer = QueryBuffer()
        controller.rng = np.random.default_rng(7)
        controller.workspace_x = (0.0, 2.0)
        controller.workspace_y = (-2.0, 2.0)
        controller.oracle = StateOnlyMembershipOracle()
        controller.progress = lambda value: None
        controller.stage_diagnostics = []
        with tempfile.TemporaryDirectory() as temporary:
            controller.output_dir = Path(temporary)
            controller._collect_warmup()
        self.assertEqual(controller.oracle.query_count, 12)
        self.assertEqual(controller.buffer.label_counts(), {"safe": 6, "violation": 6})

    def test_impossible_warmup_targets_fail_before_querying_oracle(self) -> None:
        class CountingOracle:
            def __init__(self) -> None:
                self.query_count = 0

            def query(self, trajectory: Trajectory) -> int:
                del trajectory
                self.query_count += 1
                return SAFE_LABEL

        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            warmup_queries=4,
            max_warmup_queries=4,
            minimum_label_count=3,
            minimum_safe_query_count=0,
            minimum_violation_query_count=0,
            warmup_validation_safe_count=0,
            warmup_validation_violation_count=0,
            final_calibration_safe_count=0,
            final_calibration_violation_count=0,
        )
        controller.experts = [self._curved_expert("expert")]
        controller.buffer = QueryBuffer()
        controller.rng = np.random.default_rng(7)
        controller.workspace_x = (0.0, 2.0)
        controller.workspace_y = (-2.0, 2.0)
        controller.oracle = CountingOracle()
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            controller._collect_warmup()
        self.assertEqual(controller.oracle.query_count, 0)

    def test_first_thirty_warmup_queries_do_not_repeat_trajectories(self) -> None:
        experts = [self._curved_expert(f"expert_{index}") for index in range(5)]
        fingerprints = []
        for index in range(30):
            trajectory = generate_warmup_candidate(
                experts[(index // 2) % len(experts)],
                index,
                np.random.default_rng(index),
                (0.0, 2.0),
                (-2.0, 2.0),
            )
            fingerprints.append(trajectory.states.tobytes())
        self.assertEqual(len(set(fingerprints)), 30)

    def test_straight_expert_uses_distinct_chord_normal_warmup_fallback(self) -> None:
        expert = straight(0.0)
        trajectories = [
            generate_warmup_candidate(
                expert,
                index,
                np.random.default_rng(index),
                (0.0, 2.0),
                (-2.0, 2.0),
            )
            for index in range(50)
        ]
        self.assertEqual(len({trajectory.states.tobytes() for trajectory in trajectories}), 50)
        self.assertTrue(
            all(
                trajectory.metadata["warmup_basis"] == "chord_normal_fallback"
                for trajectory in trajectories
            )
        )
        self.assertEqual(
            {trajectories[0].metadata["warmup_direction"], trajectories[1].metadata["warmup_direction"]},
            {"negative_chord_normal", "positive_chord_normal"},
        )

    def test_warmup_split_never_breaks_a_family_to_force_label_coverage(self) -> None:
        labels_by_pair = ((SAFE_LABEL, SAFE_LABEL), (VIOLATION_LABEL, VIOLATION_LABEL), (SAFE_LABEL, VIOLATION_LABEL))
        records = []
        for pair_index, labels in enumerate(labels_by_pair):
            for member_index, label in enumerate(labels):
                trajectory = straight(float(pair_index + member_index / 10.0))
                trajectory.metadata.update(
                    {
                        "source": "warmup",
                        "warmup_pair_index": pair_index,
                        "warmup_direction": (
                            "toward_chord" if member_index == 0 else "continue_detour"
                        ),
                        "expert_id": f"expert_{pair_index}",
                        "alpha": 0.02 * (pair_index + 1),
                    }
                )
                records.append(QueryRecord(trajectory, label, "warmup", 0))
        groups, errors = SemanticNumericCEGIS._warmup_pair_groups(records)
        self.assertEqual(errors, [])
        _, singleton_errors = SemanticNumericCEGIS._warmup_pair_groups(records[:1])
        self.assertTrue(any("exactly two" in error for error in singleton_errors))
        with self.assertRaisesRegex(ValueError, "no whole-family partition"):
            SemanticNumericCEGIS._choose_warmup_group_roles(
                groups,
                {
                    "validation_safe": 1,
                    "validation_violation": 1,
                    "calibration_safe": 1,
                    "calibration_violation": 1,
                },
            )

    def test_warmup_uses_remaining_budget_until_whole_family_split_is_feasible(self) -> None:
        class SequenceOracle:
            def __init__(self) -> None:
                self.query_count = 0
                self.labels = (
                    SAFE_LABEL,
                    SAFE_LABEL,
                    VIOLATION_LABEL,
                    VIOLATION_LABEL,
                    SAFE_LABEL,
                    VIOLATION_LABEL,
                    SAFE_LABEL,
                    VIOLATION_LABEL,
                )

            def query(self, trajectory: Trajectory) -> int:
                del trajectory
                label = self.labels[self.query_count]
                self.query_count += 1
                return label

        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            warmup_queries=6,
            max_warmup_queries=8,
            minimum_label_count=0,
            minimum_safe_query_count=0,
            minimum_violation_query_count=0,
            warmup_validation_safe_count=1,
            warmup_validation_violation_count=1,
            final_calibration_safe_count=1,
            final_calibration_violation_count=1,
        )
        controller.experts = [self._curved_expert(f"expert_{index}") for index in range(4)]
        controller.buffer = QueryBuffer()
        controller.rng = np.random.default_rng(7)
        controller.workspace_x = (0.0, 2.0)
        controller.workspace_y = (-2.0, 2.0)
        controller.oracle = SequenceOracle()
        controller.progress = lambda value: None
        controller.stage_diagnostics = []
        with tempfile.TemporaryDirectory() as temporary:
            controller.output_dir = Path(temporary)
            controller._collect_warmup()
        self.assertEqual(controller.oracle.query_count, 8)
        roles_by_pair = {}
        for record in controller.buffer.records:
            pair_index = int(record.trajectory.metadata["warmup_pair_index"])
            roles_by_pair.setdefault(pair_index, set()).add(record.source)
        self.assertTrue(all(len(roles) == 1 for roles in roles_by_pair.values()))

    def test_warmup_failure_persists_actual_and_target_counts(self) -> None:
        class AlwaysViolationOracle:
            def __init__(self) -> None:
                self.query_count = 0

            def query(self, trajectory: Trajectory) -> int:
                del trajectory
                self.query_count += 1
                return VIOLATION_LABEL

        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            warmup_queries=4,
            max_warmup_queries=4,
            minimum_label_count=1,
            minimum_safe_query_count=1,
            minimum_violation_query_count=1,
            warmup_validation_safe_count=0,
            warmup_validation_violation_count=0,
            final_calibration_safe_count=0,
            final_calibration_violation_count=0,
        )
        controller.experts = [self._curved_expert("expert_failure")]
        controller.buffer = QueryBuffer()
        controller.rng = np.random.default_rng(73)
        controller.workspace_x = (0.0, 2.0)
        controller.workspace_y = (-2.0, 2.0)
        controller.oracle = AlwaysViolationOracle()
        with tempfile.TemporaryDirectory() as temporary:
            controller.output_dir = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "actual=.*target="):
                controller._collect_warmup()
            diagnostic_path = Path(temporary) / "warmup_failure_diagnostics.json"
            self.assertTrue(diagnostic_path.exists())
            payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["actual_counts"], {"safe": 0, "violation": 4})
        self.assertEqual(payload["target_counts"], {"safe": 1, "violation": 1})
        self.assertEqual(len(payload["records"]), 4)

    def test_query_pool_contains_both_false_unsafe_and_shortcut_probes(self) -> None:
        controller = object.__new__(SemanticNumericCEGIS)
        hypothesis = canonical_initial_hypotheses()[0]
        controller.pending_interventions = {}
        kinds = [controller._intervention_for(hypothesis, 1, index).kind for index in range(3)]
        self.assertEqual(kinds, ["model_false_unsafe", "shortcut", "model_false_safe"])
        controller.pending_interventions = {
            hypothesis.hypothesis_id: [
                InterventionSpec(hypothesis.hypothesis_id, "boundary_uncertainty")
            ]
        }
        self.assertEqual(controller._intervention_for(hypothesis, 2, 0).kind, "boundary_uncertainty")
        self.assertEqual(controller._intervention_for(hypothesis, 2, 1).kind, "model_false_unsafe")

    def test_composite_false_unsafe_pending_intervention_gets_explicit_clause(self) -> None:
        hypothesis = ConstraintHypothesis(
            "h_pending_composite",
            "equality plus speed bound",
            ("y_position", "speed"),
            "joint",
            "equality_band",
            "max",
            "linear",
            "stress a clause",
            "Semantic interventions may omit a clause ID.",
            clauses=(
                ConstraintClause(
                    "c_equal", ("y_position",), "joint", "equality_band",
                    "max", "linear", "move y", "target equality",
                ),
                ConstraintClause(
                    "c_speed", ("speed",), "joint", "upper_bound",
                    "max", "linear", "increase speed", "target speed",
                ),
            ),
        )
        controller = object.__new__(SemanticNumericCEGIS)
        controller.pending_interventions = {
            hypothesis.hypothesis_id: [
                InterventionSpec(
                    hypothesis.hypothesis_id,
                    "model_false_unsafe",
                    variable="speed",
                    clause_id=None,
                    rationale="LLM omitted clause_id.",
                )
            ]
        }
        resolved = controller._intervention_for(hypothesis, 1, 0)
        self.assertEqual(resolved.clause_id, "c_speed")
        self.assertEqual(resolved.variable, "speed")
        controller._consume_pending(resolved)
        self.assertEqual(controller.pending_interventions[hypothesis.hypothesis_id], [])

        controller.pending_interventions[hypothesis.hypothesis_id] = [
            InterventionSpec(
                hypothesis.hypothesis_id,
                "model_false_unsafe",
                variable="speed",
                clause_id="c_equal",
                rationale="LLM mixed clause and variable.",
            )
        ]
        aligned = controller._intervention_for(hypothesis, 1, 0)
        self.assertEqual(aligned.clause_id, "c_equal")
        self.assertEqual(aligned.variable, "y_position")

        controller.pending_interventions[hypothesis.hypothesis_id] = []
        round_one = controller._intervention_for(hypothesis, 1, 0)
        round_two = controller._intervention_for(hypothesis, 2, 0)
        self.assertEqual(round_one.clause_id, "c_equal")
        self.assertEqual(round_two.clause_id, "c_speed")

        five_clause_hypothesis = ConstraintHypothesis(
            "h_five_clause_schedule",
            "five independently probed bounds",
            ("y_position",),
            "joint",
            "upper_bound",
            "max",
            "linear",
            "stress each bound",
            "Regression test for multiple false-unsafe occurrences in one pool.",
            clauses=tuple(
                ConstraintClause(
                    f"c_{index}",
                    ("y_position",),
                    "joint",
                    "upper_bound",
                    "max",
                    "linear",
                    "increase y",
                    f"bound {index}",
                )
                for index in range(5)
            ),
        )
        controller.pending_interventions[five_clause_hypothesis.hypothesis_id] = []
        round_one_probes = [
            controller._intervention_for(five_clause_hypothesis, 1, query_index)
            for query_index in (0, 5, 10, 15, 20, 25)
        ]
        round_two_probes = [
            controller._intervention_for(five_clause_hypothesis, 2, query_index)
            for query_index in (0, 5, 10, 15, 20)
        ]
        self.assertEqual(
            [probe.kind for probe in round_one_probes],
            ["model_false_unsafe"] * 6,
        )
        self.assertEqual(
            [probe.clause_id for probe in round_one_probes],
            ["c_0", "c_1", "c_2", "c_3", "c_4", "c_0"],
        )
        self.assertEqual(
            [probe.clause_id for probe in round_two_probes],
            ["c_1", "c_2", "c_3", "c_4", "c_0"],
        )

        controller.pending_interventions[five_clause_hypothesis.hypothesis_id] = [
            InterventionSpec(
                five_clause_hypothesis.hypothesis_id,
                "boundary_uncertainty",
            )
        ]
        probes_after_pending = [
            controller._intervention_for(five_clause_hypothesis, 1, query_index)
            for query_index in (1, 6, 11, 16, 21)
        ]
        self.assertEqual(
            [probe.kind for probe in probes_after_pending],
            ["model_false_unsafe"] * 5,
        )
        self.assertEqual(
            [probe.clause_id for probe in probes_after_pending],
            ["c_0", "c_1", "c_2", "c_3", "c_4"],
        )

        unresolved = InterventionSpec(
            hypothesis.hypothesis_id,
            "model_false_unsafe",
            clause_id=None,
        )
        with self.assertRaisesRegex(ValueError, "explicit clause_id"):
            controller._synthesize_best(hypothesis, straight(0.0), unresolved, 1)

        trajectory = straight(0.1)
        trajectory.metadata.update(
            {
                "source_anchor_margin_satisfied": True,
                "generation_hard_margin_satisfied": True,
                "query_hard_margin_satisfied": True,
                "safe_query_causal_rejector_ids": [hypothesis.hypothesis_id],
                "query_target_clause_id": None,
            }
        )
        result = FalsifierResult(
            trajectory,
            "model_false_unsafe",
            hypothesis.hypothesis_id,
            0.0,
            0.0,
            0.0,
            0.0,
            True,
            "ok",
        )
        candidate = AcquisitionCandidate(
            hypothesis,
            unresolved,
            result,
            {hypothesis.hypothesis_id: VIOLATION_LABEL},
            {hypothesis.hypothesis_id: 1.0},
            {hypothesis.hypothesis_id: 0.0},
            1.0,
            {},
        )
        self.assertFalse(controller._candidate_is_queryable(candidate))

    @staticmethod
    def _radius_ladder_fixture(
        threshold: float,
        hard_margin: float,
        radii: tuple[float, ...],
        *,
        valid: bool = True,
        events: list[str] | None = None,
    ) -> tuple[SemanticNumericCEGIS, ConstraintHypothesis, Trajectory, list[float]]:
        hypothesis = canonical_initial_hypotheses()[0]
        calls: list[float] = []

        class RadiusRegistry:
            def __init__(self) -> None:
                self.models = {
                    hypothesis.hypothesis_id: SimpleNamespace(
                        decision_threshold=torch.tensor(threshold)
                    )
                }

            @staticmethod
            def predict(
                hypothesis_id: str,
                trajectory: Trajectory,
            ) -> tuple[int, float, float]:
                del hypothesis_id
                score = float(trajectory.metadata.get("synthetic_hard_score", 0.0))
                return int(score > threshold), score, 0.0

        class RecordingFalsifier:
            def __init__(self) -> None:
                self.config = FalsifierConfig(
                    false_unsafe_trust_radius=max(radii),
                    false_unsafe_radius_ladder=radii,
                    false_unsafe_anchor_margin=0.0,
                    false_unsafe_hard_margin=hard_margin,
                )

            def false_unsafe_radii(self) -> tuple[float, ...]:
                return HypothesisFalsifier.false_unsafe_radii(self)  # type: ignore[arg-type]

            def generate(
                self,
                ensemble: object,
                expert: Trajectory,
                intervention: InterventionSpec,
                *,
                initialization_mix: float,
                restart_index: int,
                trust_radius: float | None = None,
            ) -> FalsifierResult:
                del ensemble, initialization_mix, restart_index
                assert trust_radius is not None
                radius = float(trust_radius)
                calls.append(radius)
                if events is not None:
                    events.append(f"radius:{radius:.2f}")
                states = expert.states.copy()
                states[1:-1, 1] += radius
                actions = np.zeros_like(states)
                actions[:-1] = np.diff(states, axis=0)
                trajectory = Trajectory(
                    states,
                    actions,
                    metadata={
                        "synthetic_hard_score": radius,
                        "trust_radius": radius,
                        "max_expert_deviation": radius,
                        # A zero local gradient at an early rung must not stop
                        # the radius ladder: a larger trust region can still
                        # reach a piecewise model boundary.
                        "falsifier_reachability_status": (
                            "zero_initial_crossing_gradient"
                            if radius < threshold + hard_margin
                            else "hard_margin_checkpoint"
                        ),
                    },
                )
                return FalsifierResult(
                    trajectory,
                    intervention.kind,
                    hypothesis.hypothesis_id,
                    1.0,
                    radius,
                    radius,
                    0.0,
                    valid,
                    "ok" if valid else "synthetic_invalid",
                )

        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(falsifier_restarts=1)
        controller.registry = RadiusRegistry()
        controller.falsifier = RecordingFalsifier()
        controller.rng = np.random.default_rng(0)
        controller.workspace_x = (0.0, 10.0)
        controller.workspace_y = (-4.0, 4.0)
        return controller, hypothesis, straight(0.0), calls

    def test_radius_ladder_stops_at_first_hard_margin_crossing(self) -> None:
        controller, hypothesis, expert, calls = self._radius_ladder_fixture(
            0.09,
            0.02,
            (0.03, 0.06, 0.12, 0.24),
        )
        result = controller._synthesize_best(
            hypothesis,
            expert,
            InterventionSpec(hypothesis.hypothesis_id, "model_false_unsafe"),
            1,
        )
        self.assertEqual(calls, [0.03, 0.06, 0.12])
        self.assertAlmostEqual(result.trajectory.metadata["selected_trust_radius"], 0.12)
        self.assertTrue(result.trajectory.metadata["generation_hard_margin_satisfied"])
        self.assertEqual(
            result.trajectory.metadata["radius_ladder_status"],
            "first_hard_margin_crossing",
        )

    def test_composite_ladder_ignores_earlier_nuisance_clause_crossing(self) -> None:
        hypothesis = ConstraintHypothesis(
            "h_ladder_composite",
            "target plus nuisance",
            ("y_position", "speed"),
            "joint",
            "equality_band",
            "max",
            "linear",
            "stress target",
            "The nuisance clause crosses at the first radius.",
            clauses=(
                ConstraintClause(
                    "c_target", ("y_position",), "joint", "equality_band",
                    "max", "linear", "move y", "target clause",
                ),
                ConstraintClause(
                    "c_nuisance", ("speed",), "joint", "upper_bound",
                    "max", "linear", "increase speed", "nuisance clause",
                ),
            ),
        )
        threshold = 0.09
        hard_margin = 0.02
        radii = (0.03, 0.06, 0.12, 0.24)
        calls: list[float] = []

        class CompositeRegistry:
            def __init__(self) -> None:
                self.models = {
                    hypothesis.hypothesis_id: SimpleNamespace(
                        decision_threshold=torch.tensor(threshold)
                    )
                }

            @staticmethod
            def predict(
                hypothesis_id: str,
                trajectory: Trajectory,
            ) -> tuple[int, float, float]:
                del hypothesis_id
                full_score = float(trajectory.metadata.get("synthetic_full_score", 0.0))
                return int(full_score > threshold), full_score, 0.0

        class CompositeRecordingFalsifier:
            def __init__(self) -> None:
                self.config = FalsifierConfig(
                    false_unsafe_trust_radius=max(radii),
                    false_unsafe_radius_ladder=radii,
                    false_unsafe_anchor_margin=0.0,
                    false_unsafe_hard_margin=hard_margin,
                )

            def false_unsafe_radii(self) -> tuple[float, ...]:
                return HypothesisFalsifier.false_unsafe_radii(self)  # type: ignore[arg-type]

            @staticmethod
            def validate(trajectory: Trajectory, anchor: Trajectory) -> tuple[bool, str]:
                del trajectory, anchor
                return True, "ok"

            def generate(
                self,
                ensemble: object,
                expert: Trajectory,
                intervention: InterventionSpec,
                *,
                initialization_mix: float,
                restart_index: int,
                trust_radius: float | None = None,
            ) -> FalsifierResult:
                del ensemble, initialization_mix, restart_index
                assert trust_radius is not None
                radius = float(trust_radius)
                calls.append(radius)
                trajectory = straight(radius)
                trajectory.metadata.update(
                    {
                        # The nuisance clause makes the full composite reject
                        # immediately, but the requested clause reaches margin
                        # only at radius .12.
                        "synthetic_full_score": 1.0,
                        "source_candidate_target_hard_score": radius,
                        "trust_radius": radius,
                        "max_expert_deviation": radius,
                    }
                )
                return FalsifierResult(
                    trajectory,
                    intervention.kind,
                    hypothesis.hypothesis_id,
                    1.0,
                    radius,
                    1.0,
                    0.0,
                    True,
                    "ok",
                )

        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(falsifier_restarts=1)
        controller.registry = CompositeRegistry()
        controller.falsifier = CompositeRecordingFalsifier()
        controller.rng = np.random.default_rng(0)
        controller.workspace_x = (0.0, 10.0)
        controller.workspace_y = (-4.0, 4.0)
        result = controller._synthesize_best(
            hypothesis,
            straight(0.0),
            InterventionSpec(
                hypothesis.hypothesis_id,
                "model_false_unsafe",
                variable="y_position",
                clause_id="c_target",
            ),
            1,
        )
        self.assertEqual(calls, [0.03, 0.06, 0.12])
        self.assertAlmostEqual(result.trajectory.metadata["selected_trust_radius"], 0.12)
        self.assertTrue(result.trajectory.metadata["generation_hard_margin_satisfied"])

    def test_radius_ladder_exhaustion_does_not_claim_a_crossing(self) -> None:
        controller, hypothesis, expert, calls = self._radius_ladder_fixture(
            0.50,
            0.02,
            (0.04, 0.08, 0.16),
        )
        result = controller._synthesize_best(
            hypothesis,
            expert,
            InterventionSpec(hypothesis.hypothesis_id, "model_false_unsafe"),
            1,
        )
        self.assertEqual(calls, [0.04, 0.08, 0.16])
        self.assertAlmostEqual(result.trajectory.metadata["selected_trust_radius"], 0.16)
        self.assertFalse(result.trajectory.metadata["generation_hard_margin_satisfied"])
        self.assertEqual(
            result.trajectory.metadata["radius_ladder_status"],
            "exhausted_without_hard_margin",
        )
        candidate = AcquisitionCandidate(
            hypothesis,
            InterventionSpec(hypothesis.hypothesis_id, "model_false_unsafe"),
            result,
            {hypothesis.hypothesis_id: SAFE_LABEL},
            {hypothesis.hypothesis_id: 0.16},
            {hypothesis.hypothesis_id: 0.0},
            1.0,
            {},
        )
        self.assertFalse(controller._candidate_is_queryable(candidate))

    def test_false_unsafe_fallback_preserves_all_radius_attempt_diagnostics(self) -> None:
        controller, hypothesis, expert, calls = self._radius_ladder_fixture(
            0.50,
            0.10,
            (0.10, 0.20),
            valid=False,
        )
        result = controller._synthesize_best(
            hypothesis,
            expert,
            InterventionSpec(hypothesis.hypothesis_id, "model_false_unsafe"),
            1,
        )
        self.assertEqual(calls, [0.10, 0.20])
        self.assertEqual(len(result.trajectory.metadata["radius_ladder_attempts"]), 2)
        self.assertIn(
            "no_valid_candidate_fallback",
            result.trajectory.metadata["radius_ladder_status"],
        )
        self.assertFalse(result.trajectory.metadata["generation_hard_margin_satisfied"])

    def test_radius_ladder_spends_one_oracle_query_for_one_candidate(self) -> None:
        events: list[str] = []
        controller, hypothesis, expert, calls = self._radius_ladder_fixture(
            0.09,
            0.02,
            (0.03, 0.06, 0.12, 0.24),
            events=events,
        )

        class CountingOracle:
            def __init__(self) -> None:
                self.query_count = 0

            def query(self, trajectory: Trajectory) -> int:
                del trajectory
                self.query_count += 1
                events.append("oracle")
                return SAFE_LABEL

        def certify_query(
            result: FalsifierResult,
            anchor: Trajectory,
            active: list[ConstraintHypothesis],
        ) -> FalsifierResult:
            del anchor, active
            result.trajectory.metadata.update(
                {
                    "query_hard_margin_satisfied": True,
                    "safe_query_causal_rejector_ids": [hypothesis.hypothesis_id],
                }
            )
            return result

        controller.config = LoopConfig(
            candidate_pool_per_hypothesis=1,
            query_hypothesis_beam=1,
            oracle_query_budget_per_round=1,
            falsifier_restarts=1,
            minimum_safe_label_fraction=0.0,
            minimum_violation_label_fraction=0.0,
            reserve_label_seeking_queries=False,
            candidate_history_deduplication_rms=0.0,
        )
        controller.experts = [expert]
        controller.buffer = QueryBuffer()
        controller.query_priorities = {}
        controller.pending_interventions = {}
        controller.oracle = CountingOracle()
        controller.progress = lambda message: None
        controller.query_diagnostics = []
        controller._refine_false_unsafe_to_nearest_boundary = certify_query
        controller._collect_hypothesis_queries([hypothesis], 1)

        self.assertEqual(calls, [0.03, 0.06, 0.12])
        self.assertEqual(events, ["radius:0.03", "radius:0.06", "radius:0.12", "oracle"])
        self.assertEqual(controller.oracle.query_count, 1)
        self.assertEqual(len(controller.buffer.records), 1)
        self.assertEqual(len(controller.query_diagnostics), 1)
        self.assertTrue(controller.query_diagnostics[0]["queried"])

    @staticmethod
    def _acquisition_candidate(
        hypothesis: ConstraintHypothesis,
        name: str,
        kind: str,
        y: float,
        acquisition_score: float,
        prediction: int,
    ) -> AcquisitionCandidate:
        trajectory = straight(y, source=name)
        trajectory.metadata.update(
            {
                "name": name,
                "trust_radius": 0.08 if kind == "model_false_unsafe" else None,
                "max_expert_deviation": 0.02 if kind == "model_false_unsafe" else None,
                "source_anchor_margin_satisfied": bool(
                    kind == "model_false_unsafe" and prediction == VIOLATION_LABEL
                ),
                "generation_hard_margin_satisfied": bool(
                    kind == "model_false_unsafe" and prediction == VIOLATION_LABEL
                ),
                "query_hard_margin_satisfied": bool(
                    kind == "model_false_unsafe" and prediction == VIOLATION_LABEL
                ),
                "safe_query_causal_rejector_ids": (
                    [hypothesis.hypothesis_id]
                    if kind == "model_false_unsafe" and prediction == VIOLATION_LABEL
                    else []
                ),
            }
        )
        intervention = InterventionSpec(hypothesis.hypothesis_id, kind, rationale=name)
        result = FalsifierResult(
            trajectory,
            kind,
            hypothesis.hypothesis_id,
            0.0,
            0.0,
            0.0,
            0.0,
            True,
            "ok",
        )
        return AcquisitionCandidate(
            hypothesis,
            intervention,
            result,
            {hypothesis.hypothesis_id: prediction},
            {hypothesis.hypothesis_id: 0.0},
            {hypothesis.hypothesis_id: 0.0},
            acquisition_score,
            {},
        )

    def test_safe_acquisition_retries_after_oracle_returns_violation(self) -> None:
        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            oracle_query_budget_per_round=3,
            minimum_safe_label_fraction=0.20,
            minimum_violation_label_fraction=0.0,
            candidate_history_deduplication_rms=0.0,
        )
        controller.buffer = QueryBuffer()
        hypothesis = canonical_initial_hypotheses()[0]
        safe_probe_1 = self._acquisition_candidate(
            hypothesis, "safe_probe_1", "model_false_unsafe", 1.0, 0.90, VIOLATION_LABEL
        )
        safe_probe_2 = self._acquisition_candidate(
            hypothesis, "safe_probe_2", "model_false_unsafe", 2.0, 0.80, VIOLATION_LABEL
        )
        global_candidate = self._acquisition_candidate(
            hypothesis, "global", "model_false_safe", 3.0, 0.99, SAFE_LABEL
        )
        candidates = [global_candidate, safe_probe_1, safe_probe_2]
        selected: list[AcquisitionCandidate] = []

        first, reason, details = controller._choose_next_acquisition(
            candidates, selected, {"safe": 1, "violation": 5}, 3
        )
        self.assertIs(first, safe_probe_1)
        self.assertEqual(reason, "adaptive_label_balance_safe")
        self.assertEqual(details["label_deficits_before_query"]["safe"], 1)
        selected.append(first)
        controller.buffer.add(QueryRecord(first.result.trajectory, VIOLATION_LABEL, first.intervention.kind, 1))

        second, reason, _ = controller._choose_next_acquisition(
            candidates, selected, {"safe": 1, "violation": 6}, 2
        )
        self.assertIs(second, safe_probe_2)
        self.assertEqual(reason, "adaptive_label_balance_safe")
        selected.append(second)
        controller.buffer.add(QueryRecord(second.result.trajectory, SAFE_LABEL, second.intervention.kind, 1))

        third, reason, details = controller._choose_next_acquisition(
            candidates, selected, {"safe": 2, "violation": 6}, 1
        )
        self.assertIs(third, global_candidate)
        self.assertEqual(reason, "global_acquisition")
        self.assertEqual(details["label_deficits_before_query"]["safe"], 0)

    def test_safe_acquisition_does_not_override_global_score_when_balance_is_sufficient(self) -> None:
        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            oracle_query_budget_per_round=2,
            minimum_safe_label_fraction=0.35,
            minimum_violation_label_fraction=0.0,
            candidate_history_deduplication_rms=0.0,
        )
        controller.buffer = QueryBuffer()
        hypothesis = canonical_initial_hypotheses()[0]
        safe_probe = self._acquisition_candidate(
            hypothesis, "safe_probe", "model_false_unsafe", 1.0, 0.10, VIOLATION_LABEL
        )
        global_1 = self._acquisition_candidate(
            hypothesis, "global_1", "model_false_safe", 2.0, 0.99, SAFE_LABEL
        )
        global_2 = self._acquisition_candidate(
            hypothesis, "global_2", "shortcut", 3.0, 0.98, SAFE_LABEL
        )
        candidates = [safe_probe, global_1, global_2]
        selected: list[AcquisitionCandidate] = []

        first, reason, _ = controller._choose_next_acquisition(
            candidates, selected, {"safe": 5, "violation": 5}, 2
        )
        self.assertIs(first, global_1)
        self.assertEqual(reason, "global_acquisition")
        selected.append(first)
        second, reason, _ = controller._choose_next_acquisition(
            candidates, selected, {"safe": 5, "violation": 6}, 1
        )
        self.assertIs(second, global_2)
        self.assertEqual(reason, "global_acquisition")

    def test_all_model_safe_candidate_cannot_consume_safe_balance_slot(self) -> None:
        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            minimum_safe_label_fraction=0.40,
            minimum_violation_label_fraction=0.0,
            candidate_history_deduplication_rms=0.0,
        )
        controller.buffer = QueryBuffer()
        hypothesis = canonical_initial_hypotheses()[0]
        noncausal_safe_probe = self._acquisition_candidate(
            hypothesis, "all_models_safe", "model_false_unsafe", 1.0, 0.80, SAFE_LABEL
        )
        global_candidate = self._acquisition_candidate(
            hypothesis, "global", "model_false_safe", 2.0, 0.90, SAFE_LABEL
        )
        components = controller._label_acquisition_components(noncausal_safe_probe)
        self.assertFalse(components["safe_query_eligible"])
        selected, reason, _ = controller._choose_next_acquisition(
            [noncausal_safe_probe, global_candidate],
            [],
            {"safe": 1, "violation": 5},
            2,
        )
        self.assertIs(selected, global_candidate)
        self.assertEqual(reason, "global_acquisition")

    def test_generation_certificate_cannot_replace_refined_query_certificate(self) -> None:
        hypothesis = canonical_initial_hypotheses()[0]
        candidate = self._acquisition_candidate(
            hypothesis,
            "stale_endpoint_certificate",
            "model_false_unsafe",
            1.0,
            0.80,
            VIOLATION_LABEL,
        )
        candidate.result.trajectory.metadata["generation_hard_margin_satisfied"] = True
        candidate.result.trajectory.metadata["query_hard_margin_satisfied"] = False
        self.assertFalse(SemanticNumericCEGIS._candidate_is_queryable(candidate))

    def test_safe_balance_slot_cap_preserves_global_information_budget(self) -> None:
        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            minimum_safe_label_fraction=0.50,
            minimum_violation_label_fraction=0.0,
            maximum_label_balance_queries_per_round=2,
            candidate_history_deduplication_rms=0.0,
        )
        controller.buffer = QueryBuffer()
        hypothesis = canonical_initial_hypotheses()[0]
        safe_probe = self._acquisition_candidate(
            hypothesis, "safe_probe", "model_false_unsafe", 1.0, 0.80, VIOLATION_LABEL
        )
        global_candidate = self._acquisition_candidate(
            hypothesis, "global", "model_false_safe", 2.0, 0.99, SAFE_LABEL
        )
        selected, reason, details = controller._choose_next_acquisition(
            [safe_probe, global_candidate],
            [],
            {"safe": 1, "violation": 8},
            2,
            label_balance_queries_used=2,
        )
        self.assertIs(selected, global_candidate)
        self.assertEqual(reason, "global_acquisition")
        self.assertEqual(details["maximum_label_balance_queries_per_round"], 2)

    def test_confirmed_safe_rejection_signature_is_not_queried_twice_in_one_round(self) -> None:
        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            minimum_safe_label_fraction=0.50,
            minimum_violation_label_fraction=0.0,
            candidate_history_deduplication_rms=0.0,
        )
        controller.buffer = QueryBuffer()
        hypothesis = canonical_initial_hypotheses()[0]
        safe_probe_1 = self._acquisition_candidate(
            hypothesis, "safe_probe_1", "model_false_unsafe", 1.0, 0.80, VIOLATION_LABEL
        )
        safe_probe_2 = self._acquisition_candidate(
            hypothesis, "safe_probe_2", "model_false_unsafe", 2.0, 0.70, VIOLATION_LABEL
        )
        # This voter rejects the first trajectory but is not a causal
        # expert-safe-to-query crossing, so it must not alter deduplication.
        safe_probe_1.predictions["h_noncausal_voter"] = VIOLATION_LABEL
        global_candidate = self._acquisition_candidate(
            hypothesis, "global", "model_false_safe", 3.0, 0.99, SAFE_LABEL
        )
        selected, reason, _ = controller._choose_next_acquisition(
            [safe_probe_1, safe_probe_2, global_candidate],
            [safe_probe_1],
            {"safe": 2, "violation": 8},
            2,
            label_balance_queries_used=1,
            selected_labels={id(safe_probe_1): SAFE_LABEL},
        )
        self.assertIs(selected, global_candidate)
        self.assertEqual(reason, "global_acquisition")

    def test_confirmed_safe_clause_does_not_deduplicate_other_composite_clause(self) -> None:
        hypothesis = ConstraintHypothesis(
            "h_clause_signature",
            "two independent clause targets",
            ("y_position", "speed"),
            "joint",
            "equality_band",
            "max",
            "linear",
            "stress one clause",
            "Clause-aware safe-query signature regression.",
            clauses=(
                ConstraintClause(
                    "c_equal", ("y_position",), "joint", "equality_band",
                    "max", "linear", "move y", "first clause",
                ),
                ConstraintClause(
                    "c_speed", ("speed",), "joint", "upper_bound",
                    "max", "linear", "increase speed", "second clause",
                ),
            ),
        )

        def probe(name: str, y: float, clause_id: str, score: float) -> AcquisitionCandidate:
            trajectory = straight(y, source=name)
            trajectory.metadata.update(
                {
                    "name": name,
                    "trust_radius": 0.08,
                    "max_expert_deviation": 0.02,
                    "source_anchor_margin_satisfied": True,
                    "generation_hard_margin_satisfied": True,
                    "query_hard_margin_satisfied": True,
                    "query_target_clause_id": clause_id,
                    "safe_query_causal_rejector_ids": [hypothesis.hypothesis_id],
                }
            )
            intervention = InterventionSpec(
                hypothesis.hypothesis_id,
                "model_false_unsafe",
                clause_id=clause_id,
            )
            result = FalsifierResult(
                trajectory,
                intervention.kind,
                hypothesis.hypothesis_id,
                0.0,
                0.0,
                1.0,
                0.0,
                True,
                "ok",
            )
            return AcquisitionCandidate(
                hypothesis,
                intervention,
                result,
                {hypothesis.hypothesis_id: VIOLATION_LABEL},
                {hypothesis.hypothesis_id: 1.0},
                {hypothesis.hypothesis_id: 0.0},
                score,
                {},
            )

        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            minimum_safe_label_fraction=0.50,
            minimum_violation_label_fraction=0.0,
            candidate_history_deduplication_rms=0.0,
        )
        controller.buffer = QueryBuffer()
        equal_probe = probe("equal", 1.0, "c_equal", 0.80)
        speed_probe = probe("speed", 2.0, "c_speed", 0.70)
        global_candidate = self._acquisition_candidate(
            hypothesis,
            "global",
            "model_false_safe",
            3.0,
            0.99,
            SAFE_LABEL,
        )
        selected, reason, _ = controller._choose_next_acquisition(
            [equal_probe, speed_probe, global_candidate],
            [equal_probe],
            {"safe": 2, "violation": 8},
            2,
            label_balance_queries_used=1,
            selected_labels={id(equal_probe): SAFE_LABEL},
        )
        self.assertIs(selected, speed_probe)
        self.assertEqual(reason, "adaptive_label_balance_safe")

    def test_noncausal_safe_boundary_candidate_does_not_suppress_causal_probe(self) -> None:
        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            minimum_safe_label_fraction=0.50,
            minimum_violation_label_fraction=0.0,
            candidate_history_deduplication_rms=0.0,
        )
        controller.buffer = QueryBuffer()
        hypothesis = canonical_initial_hypotheses()[0]
        noncausal_boundary = self._acquisition_candidate(
            hypothesis,
            "noncausal_boundary",
            "boundary_uncertainty",
            1.0,
            0.80,
            VIOLATION_LABEL,
        )
        causal_probe = self._acquisition_candidate(
            hypothesis,
            "causal_probe",
            "model_false_unsafe",
            2.0,
            0.70,
            VIOLATION_LABEL,
        )
        global_candidate = self._acquisition_candidate(
            hypothesis,
            "global",
            "model_false_safe",
            3.0,
            0.99,
            SAFE_LABEL,
        )
        selected, reason, _ = controller._choose_next_acquisition(
            [noncausal_boundary, causal_probe, global_candidate],
            [noncausal_boundary],
            {"safe": 2, "violation": 8},
            2,
            label_balance_queries_used=1,
            selected_labels={id(noncausal_boundary): SAFE_LABEL},
        )
        self.assertIs(selected, causal_probe)
        self.assertEqual(reason, "adaptive_label_balance_safe")

    def test_safe_acquisition_balance_excludes_calibration_and_selection_holdouts(self) -> None:
        controller = object.__new__(SemanticNumericCEGIS)
        controller.buffer = QueryBuffer()
        controller.buffer.add(QueryRecord(straight(1.0), SAFE_LABEL, "warmup", 0))
        for index in range(5):
            controller.buffer.add(QueryRecord(straight(-1.0 - index), VIOLATION_LABEL, "warmup", 0))
        for index in range(4):
            controller.buffer.add(
                QueryRecord(straight(10.0 + index), SAFE_LABEL, "warmup_validation", 0)
            )
            controller.buffer.add(
                QueryRecord(straight(20.0 + index), SAFE_LABEL, "final_calibration", 0)
            )
        self.assertEqual(controller.buffer.label_counts(), {"safe": 9, "violation": 5})
        self.assertEqual(controller._trainable_label_counts(), {"safe": 1, "violation": 5})

    def test_false_unsafe_probe_is_refined_to_nearest_model_boundary(self) -> None:
        expert = straight(0.0)
        endpoint_states = expert.states.copy()
        phase = np.linspace(0.0, 1.0, len(endpoint_states), dtype=np.float32)
        bump = np.sin(np.pi * phase) ** 2
        endpoint_states[:, 1] += 0.08 * bump / np.max(bump)
        endpoint = Trajectory(
            endpoint_states,
            metadata={
                "source": "model_false_unsafe",
                "trust_radius": 0.08,
                "max_expert_deviation": 0.08,
                "generation_hard_margin_target": 0.09,
                "generation_achieved_hard_margin": 0.05,
                "generation_hard_margin_satisfied": True,
            },
        )
        hypothesis = canonical_initial_hypotheses()[0]

        class DistanceRegistry:
            def __init__(self, anchor: Trajectory, hypothesis_id: str) -> None:
                self.anchor = anchor
                self.models = {
                    hypothesis_id: type(
                        "ThresholdModel",
                        (),
                        {"decision_threshold": torch.tensor(0.04)},
                    )()
                }

            def predict(self, hypothesis_id: str, trajectory: Trajectory) -> tuple[int, float, float]:
                distance = float(
                    np.max(np.linalg.norm(trajectory.states - self.anchor.states, axis=1))
                )
                threshold = float(self.models[hypothesis_id].decision_threshold.item())
                return int(distance > threshold), distance, 0.0

        class ValidatingFalsifier:
            @staticmethod
            def validate(trajectory: Trajectory, anchor: Trajectory) -> tuple[bool, str]:
                del trajectory, anchor
                return True, "ok"

        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            safe_query_boundary_bisection_steps=12,
            safe_query_boundary_margin=0.0,
        )
        controller.registry = DistanceRegistry(expert, hypothesis.hypothesis_id)
        controller.falsifier = ValidatingFalsifier()
        result = FalsifierResult(
            endpoint,
            "model_false_unsafe",
            hypothesis.hypothesis_id,
            1.0,
            0.5,
            0.08,
            0.0,
            True,
            "ok",
        )
        refined = controller._refine_false_unsafe_to_nearest_boundary(
            result,
            expert,
            [hypothesis],
        )
        refined_distance = float(
            np.max(np.linalg.norm(refined.trajectory.states - expert.states, axis=1))
        )
        self.assertTrue(
            refined.trajectory.metadata.get("boundary_refined", False),
            refined.trajectory.metadata,
        )
        self.assertGreaterEqual(refined_distance, 0.04)
        self.assertLess(refined_distance, 0.041)
        self.assertLess(refined_distance, 0.08)
        self.assertAlmostEqual(
            refined.trajectory.metadata["generation_hard_margin_target"],
            0.09,
        )
        self.assertAlmostEqual(
            refined.trajectory.metadata["query_hard_margin_target"],
            0.04,
            places=6,
        )
        self.assertTrue(refined.trajectory.metadata["query_hard_margin_satisfied"])
        self.assertEqual(
            controller.registry.predict(hypothesis.hypothesis_id, refined.trajectory)[0],
            VIOLATION_LABEL,
        )

    def test_boundary_refinement_finds_first_nonmonotone_crossing(self) -> None:
        expert = straight(0.0)
        endpoint_states = expert.states.copy()
        phase = np.linspace(0.0, 1.0, len(endpoint_states), dtype=np.float32)
        bump = np.sin(np.pi * phase) ** 2
        endpoint_states[:, 1] += 0.08 * bump / np.max(bump)
        endpoint = Trajectory(
            endpoint_states,
            metadata={
                "source": "model_false_unsafe",
                "trust_radius": 0.08,
                "max_expert_deviation": 0.08,
            },
        )
        nonmonotone = canonical_initial_hypotheses()[0]
        late = canonical_initial_hypotheses()[1]

        class NonmonotoneRegistry:
            def __init__(self) -> None:
                self.models = {
                    nonmonotone.hypothesis_id: SimpleNamespace(
                        decision_threshold=torch.tensor(0.5)
                    ),
                    late.hypothesis_id: SimpleNamespace(
                        decision_threshold=torch.tensor(0.5)
                    ),
                }

            @staticmethod
            def predict(
                hypothesis_id: str,
                trajectory: Trajectory,
            ) -> tuple[int, float, float]:
                deformation = float(
                    np.max(np.linalg.norm(trajectory.states - expert.states, axis=1))
                )
                alpha = deformation / 0.08
                if hypothesis_id == nonmonotone.hypothesis_id:
                    reject = (0.25 <= alpha <= 0.35) or alpha >= 0.75
                else:
                    reject = alpha >= 0.40
                score = 1.0 if reject else 0.0
                return int(reject), score, 0.0

        class ValidatingFalsifier:
            @staticmethod
            def validate(trajectory: Trajectory, anchor: Trajectory) -> tuple[bool, str]:
                del trajectory, anchor
                return True, "ok"

        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            safe_query_boundary_bisection_steps=12,
            safe_query_boundary_scan_points=64,
            safe_query_boundary_margin=0.0,
        )
        controller.registry = NonmonotoneRegistry()
        controller.falsifier = ValidatingFalsifier()
        result = FalsifierResult(
            endpoint,
            "model_false_unsafe",
            nonmonotone.hypothesis_id,
            1.0,
            0.5,
            1.0,
            0.0,
            True,
            "ok",
        )
        refined = controller._refine_false_unsafe_to_nearest_boundary(
            result,
            expert,
            [nonmonotone, late],
        )
        alpha = float(refined.trajectory.metadata["boundary_refinement_alpha"])
        self.assertGreaterEqual(alpha, 0.25)
        self.assertLess(alpha, 0.251)
        self.assertEqual(
            refined.trajectory.metadata["boundary_trigger_hypothesis_id"],
            nonmonotone.hypothesis_id,
        )
        self.assertEqual(
            controller.registry.predict(nonmonotone.hypothesis_id, refined.trajectory)[0],
            VIOLATION_LABEL,
        )

    def test_boundary_refinement_crosses_named_clause_not_earlier_nuisance_clause(self) -> None:
        expert = straight(0.0)
        endpoint_states = expert.states.copy()
        phase = np.linspace(0.0, 1.0, len(endpoint_states), dtype=np.float32)
        bump = np.sin(np.pi * phase) ** 2
        endpoint_states[:, 1] += 0.08 * bump / np.max(bump)
        hypothesis = ConstraintHypothesis(
            "h_clause_refinement",
            "target equality plus nuisance bound",
            ("y_position", "speed"),
            "joint",
            "equality_band",
            "max",
            "linear",
            "stress one clause",
            "Regression test for clause-specific boundary refinement.",
            clauses=(
                ConstraintClause(
                    "c_target",
                    ("y_position",),
                    "joint",
                    "equality_band",
                    "max",
                    "linear",
                    "move y",
                    "The requested clause.",
                ),
                ConstraintClause(
                    "c_nuisance",
                    ("speed",),
                    "joint",
                    "upper_bound",
                    "max",
                    "linear",
                    "move quickly",
                    "An earlier unrelated crossing.",
                ),
            ),
        )

        class ClauseRegistry:
            def __init__(self) -> None:
                self.models = {
                    hypothesis.hypothesis_id: SimpleNamespace(
                        decision_threshold=torch.tensor(0.5)
                    )
                }

            @staticmethod
            def _alpha(trajectory: Trajectory) -> float:
                deformation = float(
                    np.max(np.linalg.norm(trajectory.states - expert.states, axis=1))
                )
                return deformation / 0.08

            def predict(
                self,
                hypothesis_id: str,
                trajectory: Trajectory,
            ) -> tuple[int, float, float]:
                del hypothesis_id
                alpha = self._alpha(trajectory)
                # The nuisance clause rejects from alpha=.10, while the named
                # target clause does not reject until alpha=.60.
                full_score = 1.0 if alpha >= 0.10 else 0.0
                return int(full_score > 0.5), full_score, 0.0

            def hard_clause_score(
                self,
                hypothesis_id: str,
                trajectory: Trajectory,
                clause_id: str,
            ) -> float:
                del hypothesis_id
                alpha = self._alpha(trajectory)
                if clause_id == "c_target":
                    return 1.0 if alpha >= 0.60 else 0.0
                return 1.0 if alpha >= 0.10 else 0.0

        class ValidatingFalsifier:
            @staticmethod
            def validate(trajectory: Trajectory, anchor: Trajectory) -> tuple[bool, str]:
                del trajectory, anchor
                return True, "ok"

        endpoint = Trajectory(
            endpoint_states,
            metadata={
                "source": "model_false_unsafe",
                "trust_radius": 0.08,
                "max_expert_deviation": 0.08,
                "optimization_target_clause_id": "c_target",
                "generation_hard_margin_satisfied": True,
            },
        )
        controller = object.__new__(SemanticNumericCEGIS)
        controller.config = LoopConfig(
            safe_query_boundary_bisection_steps=12,
            safe_query_boundary_scan_points=64,
            safe_query_boundary_margin=0.0,
        )
        controller.registry = ClauseRegistry()
        controller.falsifier = ValidatingFalsifier()
        result = FalsifierResult(
            endpoint,
            "model_false_unsafe",
            hypothesis.hypothesis_id,
            1.0,
            0.5,
            1.0,
            0.0,
            True,
            "ok",
        )
        refined = controller._refine_false_unsafe_to_nearest_boundary(
            result,
            expert,
            [hypothesis],
        )
        alpha = float(refined.trajectory.metadata["boundary_refinement_alpha"])
        self.assertGreaterEqual(alpha, 0.60)
        self.assertLess(alpha, 0.601)
        self.assertEqual(refined.trajectory.metadata["boundary_trigger_clause_id"], "c_target")
        self.assertTrue(refined.trajectory.metadata["query_hard_margin_satisfied"])
        self.assertEqual(
            refined.trajectory.metadata["safe_query_causal_rejector_ids"],
            [hypothesis.hypothesis_id],
        )

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

    def test_prequential_evidence_does_not_rescore_training_labels(self) -> None:
        library = FeatureLibrary()
        hypothesis = canonical_initial_hypotheses()[4]
        registry = LearnerRegistry(library, hidden_dims=(8,), ensemble_size=1, seed=8, device=torch.device("cpu"))
        registry.ensure(compile_hypothesis(hypothesis, library))
        records = [
            QueryRecord(
                straight(1.0),
                SAFE_LABEL,
                "audit",
                0,
                predictions_before_query={hypothesis.hypothesis_id: SAFE_LABEL},
                scores_before_query={hypothesis.hypothesis_id: -0.2},
            ),
            QueryRecord(
                straight(0.0),
                VIOLATION_LABEL,
                "audit",
                0,
                predictions_before_query={hypothesis.hypothesis_id: VIOLATION_LABEL},
                scores_before_query={hypothesis.hypothesis_id: 0.2},
            ),
        ]
        evidence = EvidenceCompiler(
            library, EvidenceConfig(minimum_per_label=1), torch.device("cpu")
        ).compile(registry, [hypothesis.hypothesis_id], [], records)[0]
        self.assertEqual(evidence.balanced_accuracy, 1.0)
        self.assertEqual(evidence.prequential_count, 2)

    def test_rolling_prequential_window_drops_obsolete_outer_round_predictions(self) -> None:
        library = FeatureLibrary()
        hypothesis = canonical_initial_hypotheses()[4]
        registry = LearnerRegistry(library, hidden_dims=(8,), ensemble_size=1, seed=9, device=torch.device("cpu"))
        registry.ensure(compile_hypothesis(hypothesis, library))
        records = [
            QueryRecord(
                straight(0.0), VIOLATION_LABEL, "old", 1,
                predictions_before_query={hypothesis.hypothesis_id: SAFE_LABEL},
                scores_before_query={hypothesis.hypothesis_id: -0.2},
            ),
            QueryRecord(
                straight(0.0), VIOLATION_LABEL, "recent", 2,
                predictions_before_query={hypothesis.hypothesis_id: VIOLATION_LABEL},
                scores_before_query={hypothesis.hypothesis_id: 0.2},
            ),
            QueryRecord(
                straight(1.0), SAFE_LABEL, "recent", 2,
                predictions_before_query={hypothesis.hypothesis_id: SAFE_LABEL},
                scores_before_query={hypothesis.hypothesis_id: -0.2},
            ),
        ]
        evidence = EvidenceCompiler(
            library,
            EvidenceConfig(minimum_per_label=1, prequential_window_rounds=1),
            torch.device("cpu"),
        ).compile(registry, [hypothesis.hypothesis_id], [], records)[0]
        self.assertEqual(evidence.prequential_count, 2)
        self.assertEqual(evidence.balanced_accuracy, 1.0)

    def test_nested_feature_superset_requires_material_evidence_gain(self) -> None:
        library = FeatureLibrary()
        simple = ConstraintHypothesis(
            "h_xy_simple", "planar obstacle", ("x_position", "y_position"),
            "joint", "forbidden_region", "max", "mlp", "enter region", "simple spatial explanation"
        )
        superset = ConstraintHypothesis(
            "h_xyp_superset", "phase obstacle", ("x_position", "y_position", "progress"),
            "joint", "forbidden_region", "max", "mlp", "enter phase region", "nested explanation"
        )
        registry = LearnerRegistry(library, hidden_dims=(8,), ensemble_size=1, seed=19, device=torch.device("cpu"))
        for hypothesis in (simple, superset):
            registry.ensure(compile_hypothesis(hypothesis, library))
        records = [
            QueryRecord(
                straight(1.0), SAFE_LABEL, "audit", 1,
                predictions_before_query={simple.hypothesis_id: SAFE_LABEL, superset.hypothesis_id: SAFE_LABEL},
                scores_before_query={simple.hypothesis_id: -0.2, superset.hypothesis_id: -0.2},
            ),
            QueryRecord(
                straight(0.0), VIOLATION_LABEL, "audit", 1,
                predictions_before_query={simple.hypothesis_id: VIOLATION_LABEL, superset.hypothesis_id: VIOLATION_LABEL},
                scores_before_query={simple.hypothesis_id: 0.2, superset.hypothesis_id: 0.2},
            ),
        ]
        evidence = EvidenceCompiler(
            library,
            EvidenceConfig(minimum_per_label=1, nested_minimum_balanced_accuracy_gain=0.08),
            torch.device("cpu"),
        ).compile(registry, [simple.hypothesis_id, superset.hypothesis_id], [], records)
        by_id = {item.hypothesis_id: item for item in evidence}
        self.assertTrue(by_id[simple.hypothesis_id].champion_eligible)
        self.assertFalse(by_id[superset.hypothesis_id].champion_eligible)
        self.assertIn(
            "nested_without_material_evidence_gain",
            by_id[superset.hypothesis_id].ineligibility_reasons,
        )

    def test_progress_proxy_cannot_replace_physical_coordinate_without_gain(self) -> None:
        library = FeatureLibrary()
        spatial = ConstraintHypothesis(
            "h_xy_physical", "planar obstacle", ("x_position", "y_position"),
            "joint", "forbidden_region", "max", "mlp", "enter region", "physical state"
        )
        proxy = ConstraintHypothesis(
            "h_yp_proxy", "phase obstacle", ("y_position", "progress"),
            "joint", "forbidden_region", "max", "mlp", "enter phase region", "time proxy"
        )
        registry = LearnerRegistry(library, hidden_dims=(8,), ensemble_size=1, seed=23, device=torch.device("cpu"))
        for hypothesis in (spatial, proxy):
            registry.ensure(compile_hypothesis(hypothesis, library))
        records = [
            QueryRecord(
                straight(1.0), SAFE_LABEL, "audit", 1,
                predictions_before_query={spatial.hypothesis_id: SAFE_LABEL, proxy.hypothesis_id: SAFE_LABEL},
                scores_before_query={spatial.hypothesis_id: -0.2, proxy.hypothesis_id: -0.2},
            ),
            QueryRecord(
                straight(0.0), VIOLATION_LABEL, "audit", 1,
                predictions_before_query={spatial.hypothesis_id: VIOLATION_LABEL, proxy.hypothesis_id: VIOLATION_LABEL},
                scores_before_query={spatial.hypothesis_id: 0.2, proxy.hypothesis_id: 0.2},
            ),
        ]
        evidence = EvidenceCompiler(
            library,
            EvidenceConfig(minimum_per_label=1, progress_proxy_minimum_balanced_accuracy_gain=0.08),
            torch.device("cpu"),
        ).compile(registry, [spatial.hypothesis_id, proxy.hypothesis_id], [], records)
        by_id = {item.hypothesis_id: item for item in evidence}
        self.assertTrue(by_id[spatial.hypothesis_id].champion_eligible)
        self.assertFalse(by_id[proxy.hypothesis_id].champion_eligible)
        self.assertIn(
            "task_progress_proxy_without_material_evidence_gain",
            by_id[proxy.hypothesis_id].ineligibility_reasons,
        )

    def test_velocity_proxy_needs_large_gain_over_persistent_position_model(self) -> None:
        library = FeatureLibrary()
        position = ConstraintHypothesis(
            "h_position_cause", "position obstacle", ("x_position", "y_position"),
            "joint", "forbidden_region", "max", "mlp", "enter region", "persistent obstacle"
        )
        velocity = ConstraintHypothesis(
            "h_velocity_proxy", "velocity proxy", ("x_velocity", "y_velocity"),
            "joint", "forbidden_region", "max", "mlp", "enter velocity regime", "derived proxy"
        )
        registry = LearnerRegistry(library, hidden_dims=(8,), ensemble_size=1, seed=29, device=torch.device("cpu"))
        for hypothesis in (position, velocity):
            registry.ensure(compile_hypothesis(hypothesis, library))
        records = [
            QueryRecord(
                straight(1.0), SAFE_LABEL, "audit", 1,
                predictions_before_query={position.hypothesis_id: SAFE_LABEL, velocity.hypothesis_id: SAFE_LABEL},
                scores_before_query={position.hypothesis_id: -0.2, velocity.hypothesis_id: -0.2},
            ),
            QueryRecord(
                straight(0.0), VIOLATION_LABEL, "audit", 1,
                predictions_before_query={position.hypothesis_id: VIOLATION_LABEL, velocity.hypothesis_id: VIOLATION_LABEL},
                scores_before_query={position.hypothesis_id: 0.2, velocity.hypothesis_id: 0.2},
            ),
        ]
        evidence = EvidenceCompiler(
            library,
            EvidenceConfig(minimum_per_label=1, dynamics_proxy_minimum_balanced_accuracy_gain=0.20),
            torch.device("cpu"),
        ).compile(registry, [position.hypothesis_id, velocity.hypothesis_id], [], records)
        by_id = {item.hypothesis_id: item for item in evidence}
        self.assertTrue(by_id[position.hypothesis_id].champion_eligible)
        self.assertFalse(by_id[velocity.hypothesis_id].champion_eligible)
        self.assertIn(
            "derived_dynamics_proxy_without_material_evidence_gain",
            by_id[velocity.hypothesis_id].ineligibility_reasons,
        )

    def test_terminal_representation_collision_rejects_endpoint_proxy_but_not_spatial_max(self) -> None:
        library = FeatureLibrary()
        terminal = ConstraintHypothesis(
            "h_terminal_proxy",
            "terminal proxy",
            ("x_position", "y_position"),
            "joint",
            "equality_band",
            "last",
            "linear",
            "deviate from terminal bands",
            "terminal-only competing explanation",
            clauses=(
                ConstraintClause(
                    "terminal_x", ("x_position",), "joint", "equality_band", "last", "linear",
                    "deviate in x", "terminal x band"
                ),
                ConstraintClause(
                    "terminal_y", ("y_position",), "joint", "equality_band", "last", "linear",
                    "deviate in y", "terminal y band"
                ),
            ),
        )
        spatial = ConstraintHypothesis(
            "h_spatial_path", "spatial path", ("x_position", "y_position"),
            "joint", "forbidden_region", "max", "mlp", "enter region", "path-dependent explanation"
        )
        x = np.linspace(0.0, 2.0, 24, dtype=np.float32)
        phase = np.linspace(0.0, 1.0, 24, dtype=np.float32)
        safe = Trajectory(np.column_stack((x, np.sin(np.pi * phase))).astype(np.float32))
        violation = Trajectory(np.column_stack((x, np.zeros_like(x))).astype(np.float32))
        registry = LearnerRegistry(library, hidden_dims=(8,), ensemble_size=1, seed=41, device=torch.device("cpu"))
        for hypothesis in (terminal, spatial):
            registry.ensure(compile_hypothesis(hypothesis, library))
        records = [
            QueryRecord(
                safe, SAFE_LABEL, "audit", 1,
                predictions_before_query={terminal.hypothesis_id: SAFE_LABEL, spatial.hypothesis_id: SAFE_LABEL},
                scores_before_query={terminal.hypothesis_id: -0.2, spatial.hypothesis_id: -0.2},
            ),
            QueryRecord(
                violation, VIOLATION_LABEL, "shortcut", 1,
                predictions_before_query={terminal.hypothesis_id: VIOLATION_LABEL, spatial.hypothesis_id: VIOLATION_LABEL},
                scores_before_query={terminal.hypothesis_id: 0.2, spatial.hypothesis_id: 0.2},
            ),
        ]
        evidence = EvidenceCompiler(
            library,
            EvidenceConfig(
                minimum_per_label=1,
                champion_minimum_safe_accuracy=0.0,
                champion_minimum_violation_recall=0.0,
                champion_minimum_expert_safe_rate=0.0,
                champion_minimum_fit_expert_safe_rate=0.0,
            ),
            torch.device("cpu"),
        ).compile(registry, [terminal.hypothesis_id, spatial.hypothesis_id], [], records)
        by_id = {item.hypothesis_id: item for item in evidence}
        self.assertFalse(by_id[terminal.hypothesis_id].champion_eligible)
        self.assertEqual(by_id[terminal.hypothesis_id].contradictory_representation_group_count, 1)
        self.assertIn(
            "terminal_invariance_contradicted_by_oracle",
            by_id[terminal.hypothesis_id].ineligibility_reasons,
        )
        self.assertNotIn(
            "terminal_invariance_contradicted_by_oracle",
            by_id[spatial.hypothesis_id].ineligibility_reasons,
        )

    def test_convex_hull_containment_handles_polygon_line_and_interval(self) -> None:
        triangle = np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float64
        )
        inside = np.asarray(
            [[0.5, 0.2], [1.0, 0.4], [1.5, 0.2]], dtype=np.float64
        )
        outside = np.asarray([[1.0, 1.1]], dtype=np.float64)
        self.assertTrue(_points_inside_convex_hull(triangle, inside, 1.0e-8))
        self.assertFalse(_points_inside_convex_hull(triangle, outside, 1.0e-8))
        self.assertFalse(
            _points_inside_convex_hull(
                triangle, np.asarray([[1.0, 1.0 + 5.0e-9]]), 1.0e-8
            )
        )

        line = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        self.assertTrue(
            _points_inside_convex_hull(
                line, np.asarray([[0.5, 0.0], [1.5, 0.0]]), 1.0e-8
            )
        )
        self.assertFalse(
            _points_inside_convex_hull(
                line, np.asarray([[1.0, 0.01]]), 1.0e-8
            )
        )
        interval = np.asarray([[-1.0], [2.0]])
        self.assertTrue(
            _points_inside_convex_hull(
                interval, np.asarray([[0.0], [1.5]]), 1.0e-8
            )
        )
        self.assertFalse(
            _points_inside_convex_hull(
                interval, np.asarray([[2.1]]), 1.0e-8
            )
        )
        self.assertFalse(
            _points_inside_convex_hull(
                interval, np.asarray([[2.0 + 5.0e-9]]), 1.0e-8
            )
        )

    def test_linear_max_support_gate_rejects_ambiguous_controls(self) -> None:
        library = FeatureLibrary()
        device = torch.device("cpu")
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            EvidenceCompiler(
                library,
                EvidenceConfig(linear_max_support_gate_enforced=1),
                device,
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            EvidenceCompiler(
                library,
                EvidenceConfig(linear_max_minimum_contradictory_anchors=1.9),
                device,
            )

    def test_linear_max_support_order_gate_rejects_affine_not_spatial(self) -> None:
        library = FeatureLibrary()
        affine = ConstraintHypothesis(
            "h_affine_hull",
            "affine spatial exclusion",
            ("x_position", "y_position"),
            "joint",
            "forbidden_region",
            "max",
            "linear",
            "enter halfspace",
            "affine support-order test",
        )
        spatial = ConstraintHypothesis(
            "h_mlp_hull",
            "nonlinear spatial exclusion",
            ("x_position", "y_position"),
            "joint",
            "forbidden_region",
            "max",
            "mlp",
            "enter learned region",
            "nonlinear comparison",
        )
        registry = LearnerRegistry(
            library,
            hidden_dims=(8,),
            ensemble_size=1,
            seed=47,
            device=torch.device("cpu"),
        )
        for hypothesis in (affine, spatial):
            registry.ensure(compile_hypothesis(hypothesis, library))

        anchor_a = Trajectory(
            np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32),
            metadata={"trajectory_id": "anchor_a"},
        )
        anchor_b = Trajectory(
            np.asarray([[0.0, 2.0], [1.0, 3.0], [2.0, 2.0]], dtype=np.float32),
            metadata={"trajectory_id": "anchor_b"},
        )

        def violation(anchor_id: str, y_offset: float) -> QueryRecord:
            trajectory = Trajectory(
                np.asarray(
                    [
                        [0.5, y_offset + 0.2],
                        [1.0, y_offset + 0.4],
                        [1.5, y_offset + 0.2],
                    ],
                    dtype=np.float32,
                ),
                metadata={"expert_id": anchor_id},
            )
            return QueryRecord(
                trajectory,
                VIOLATION_LABEL,
                "warmup_validation",
                0,
                predictions_before_query={
                    affine.hypothesis_id: VIOLATION_LABEL,
                    spatial.hypothesis_id: VIOLATION_LABEL,
                },
                scores_before_query={
                    affine.hypothesis_id: 0.2,
                    spatial.hypothesis_id: 0.2,
                },
            )

        records = [violation("anchor_a", 0.0), violation("anchor_b", 2.0)]
        common = dict(
            minimum_per_label=0,
            champion_minimum_safe_accuracy=0.0,
            champion_minimum_violation_recall=0.0,
            champion_minimum_expert_safe_rate=0.0,
            champion_minimum_fit_expert_safe_rate=0.0,
        )
        gated = EvidenceCompiler(
            library,
            EvidenceConfig(
                **common,
                linear_max_minimum_contradictory_anchors=2,
            ),
            torch.device("cpu"),
        ).compile(
            registry,
            [affine.hypothesis_id, spatial.hypothesis_id],
            [anchor_a, anchor_b],
            records,
        )
        ungated = EvidenceCompiler(
            library,
            EvidenceConfig(
                **common,
                linear_max_minimum_contradictory_anchors=2,
                linear_max_support_gate_enforced=False,
            ),
            torch.device("cpu"),
        ).compile(
            registry,
            [affine.hypothesis_id, spatial.hypothesis_id],
            [anchor_a, anchor_b],
            records,
        )
        by_id = {item.hypothesis_id: item for item in gated}
        baseline = {item.hypothesis_id: item for item in ungated}
        affine_evidence = by_id[affine.hypothesis_id]
        self.assertFalse(affine_evidence.champion_eligible)
        self.assertIn(
            "linear_max_support_order_contradicted_by_oracle",
            affine_evidence.ineligibility_reasons,
        )
        self.assertEqual(affine_evidence.linear_max_support_pair_count, 2)
        self.assertEqual(affine_evidence.linear_max_support_contradiction_count, 2)
        self.assertEqual(affine_evidence.linear_max_support_distinct_anchor_count, 2)
        self.assertTrue(affine_evidence.linear_max_support_gate_triggered)
        self.assertTrue(affine_evidence.linear_max_support_gate_applied)
        self.assertTrue(by_id[spatial.hypothesis_id].champion_eligible)
        self.assertTrue(
            baseline[affine.hypothesis_id].linear_max_support_gate_triggered
        )
        self.assertFalse(
            baseline[affine.hypothesis_id].linear_max_support_gate_applied
        )
        self.assertTrue(baseline[affine.hypothesis_id].champion_eligible)
        self.assertEqual(
            affine_evidence.selection_score,
            baseline[affine.hypothesis_id].selection_score,
        )
        self.assertEqual(
            affine_evidence.query_priority,
            baseline[affine.hypothesis_id].query_priority,
        )

    def test_linear_max_support_gate_requires_distinct_noncalibration_anchors(self) -> None:
        library = FeatureLibrary()
        affine = ConstraintHypothesis(
            "h_affine_repeat",
            "affine spatial exclusion",
            ("x_position", "y_position"),
            "joint",
            "forbidden_region",
            "max",
            "linear",
            "enter halfspace",
            "distinct-anchor guard",
        )
        registry = LearnerRegistry(
            library,
            hidden_dims=(8,),
            ensemble_size=1,
            seed=53,
            device=torch.device("cpu"),
        )
        registry.ensure(compile_hypothesis(affine, library))
        anchor = Trajectory(
            np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32),
            metadata={"trajectory_id": "only_anchor"},
        )
        candidate = Trajectory(
            np.asarray([[0.5, 0.2], [1.0, 0.4], [1.5, 0.2]], dtype=np.float32),
            metadata={"expert_id": "only_anchor"},
        )
        records = [
            QueryRecord(candidate.copy(), VIOLATION_LABEL, source, 0)
            for source in ("warmup_validation", "warmup_validation", "final_calibration")
        ]
        evidence = EvidenceCompiler(
            library,
            EvidenceConfig(
                minimum_per_label=0,
                champion_minimum_safe_accuracy=0.0,
                champion_minimum_violation_recall=0.0,
                champion_minimum_expert_safe_rate=0.0,
                champion_minimum_fit_expert_safe_rate=0.0,
                linear_max_minimum_contradictory_anchors=2,
            ),
            torch.device("cpu"),
        ).compile(registry, [affine.hypothesis_id], [anchor], records)[0]
        self.assertTrue(evidence.champion_eligible)
        self.assertEqual(evidence.linear_max_support_pair_count, 2)
        self.assertEqual(evidence.linear_max_support_contradiction_count, 2)
        self.assertEqual(evidence.linear_max_support_distinct_anchor_count, 1)

    def test_structurally_rejected_simple_model_cannot_dominate_superset(self) -> None:
        library = FeatureLibrary()
        simple = ConstraintHypothesis(
            "h_linear_x_impossible",
            "one-dimensional affine exclusion",
            ("x_position",),
            "joint",
            "forbidden_region",
            "max",
            "linear",
            "enter interval support",
            "structurally contradicted simpler model",
        )
        superset = ConstraintHypothesis(
            "h_linear_xy_possible",
            "two-dimensional affine exclusion",
            ("x_position", "y_position"),
            "joint",
            "forbidden_region",
            "max",
            "linear",
            "enter planar support",
            "candidate outside the planar anchor hull",
        )
        registry = LearnerRegistry(
            library,
            hidden_dims=(8,),
            ensemble_size=1,
            seed=59,
            device=torch.device("cpu"),
        )
        for hypothesis in (simple, superset):
            registry.ensure(compile_hypothesis(hypothesis, library))

        anchors = [
            Trajectory(
                np.asarray([[0.0, offset], [1.0, offset], [2.0, offset]], dtype=np.float32),
                metadata={"trajectory_id": f"anchor_{index}"},
            )
            for index, offset in enumerate((0.0, 2.0))
        ]
        records = []
        for index, offset in enumerate((0.0, 2.0)):
            records.append(
                QueryRecord(
                    Trajectory(
                        np.asarray(
                            [[0.5, offset + 0.5], [1.0, offset + 0.5], [1.5, offset + 0.5]],
                            dtype=np.float32,
                        ),
                        metadata={"expert_id": f"anchor_{index}"},
                    ),
                    VIOLATION_LABEL,
                    "warmup_validation",
                    0,
                    predictions_before_query={
                        simple.hypothesis_id: VIOLATION_LABEL,
                        superset.hypothesis_id: VIOLATION_LABEL,
                    },
                    scores_before_query={
                        simple.hypothesis_id: 0.2,
                        superset.hypothesis_id: 0.2,
                    },
                )
            )
        evidence = EvidenceCompiler(
            library,
            EvidenceConfig(
                minimum_per_label=0,
                champion_minimum_safe_accuracy=0.0,
                champion_minimum_violation_recall=0.0,
                champion_minimum_expert_safe_rate=0.0,
                champion_minimum_fit_expert_safe_rate=0.0,
                nested_minimum_balanced_accuracy_gain=1.0,
            ),
            torch.device("cpu"),
        ).compile(
            registry,
            [simple.hypothesis_id, superset.hypothesis_id],
            anchors,
            records,
        )
        by_id = {item.hypothesis_id: item for item in evidence}
        self.assertFalse(by_id[simple.hypothesis_id].champion_eligible)
        self.assertIn(
            "linear_max_support_order_contradicted_by_oracle",
            by_id[simple.hypothesis_id].ineligibility_reasons,
        )
        self.assertTrue(by_id[superset.hypothesis_id].champion_eligible)
        self.assertNotIn(
            "nested_without_material_evidence_gain",
            by_id[superset.hypothesis_id].ineligibility_reasons,
        )

    def test_fit_expert_inconsistency_blocks_champion_eligibility(self) -> None:
        library = FeatureLibrary()
        hypothesis = ConstraintHypothesis(
            "h_y_floor_gate", "y floor", ("y_position",), "joint", "lower_bound", "max", "linear",
            "decrease y", "qualification gate test"
        )
        registry = LearnerRegistry(library, hidden_dims=(8,), ensemble_size=1, seed=8, device=torch.device("cpu"))
        ensemble = registry.ensure(compile_hypothesis(hypothesis, library))
        with torch.no_grad():
            ensemble.members[0].clause_heads[0].threshold.zero_()
        records = [
            QueryRecord(
                straight(1.0), SAFE_LABEL, "audit", 0,
                predictions_before_query={hypothesis.hypothesis_id: SAFE_LABEL},
                scores_before_query={hypothesis.hypothesis_id: -0.2},
            ),
            QueryRecord(
                straight(-1.0), VIOLATION_LABEL, "audit", 0,
                predictions_before_query={hypothesis.hypothesis_id: VIOLATION_LABEL},
                scores_before_query={hypothesis.hypothesis_id: 0.2},
            ),
        ]
        evidence = EvidenceCompiler(
            library, EvidenceConfig(minimum_per_label=1), torch.device("cpu")
        ).compile(
            registry,
            [hypothesis.hypothesis_id],
            [straight(1.0)],
            records,
            [straight(-1.0)],
        )[0]
        self.assertEqual(evidence.fit_expert_safe_rate, 0.0)
        self.assertFalse(evidence.champion_eligible)
        self.assertIn("fit_expert_safe_rate_below_gate", evidence.ineligibility_reasons)

    def test_invalid_revision_fallback_preserves_hypothesis_bank(self) -> None:
        library = FeatureLibrary()
        hypotheses = canonical_initial_hypotheses()[:4]
        bank = HypothesisBank.from_hypotheses(hypotheses, library)
        evidence = [
            HypothesisEvidence(
                item.hypothesis_id, 0.5, 0.25, 0.75, 0.5, 0.5, 1, 3, 0.1, 0.0,
                0.5, 1, 2, 0.5, True,
                champion_eligible=False,
                ineligibility_reasons=("safe_accuracy_below_gate",),
            )
            for item in hypotheses
        ]
        reasoner = object.__new__(LocalQwenSemanticReasoner)
        reasoner.config = SemanticConfig(allow_fallback=True)
        reasoner.interactions = []
        reasoner._fallback = EvidencePolicyReasoner(reasoner.config)
        reasoner._generate = lambda prompt: '{"actions":['
        actions = reasoner.revise("avoid obstacle", library, bank, {"ranking": []}, evidence, 1)
        self.assertTrue(reasoner.interactions[0]["used_fallback"])
        self.assertTrue(all(action.action in {"retain_and_query", "propose_intervention"} for action in actions))
        bank.apply_actions(actions, library, outer_round=1)
        self.assertEqual(len(bank.active()), len(hypotheses))

    def test_prompt_disambiguates_temporal_max_from_feature_peak(self) -> None:
        prompt = build_initial_prompt("avoid an obstacle", FeatureLibrary(), 4)
        self.assertIn("max means ANY time step", prompt)
        self.assertIn("Never encode it as lower_bound + max", prompt)

    def test_policy_can_compose_heterogeneous_constraints(self) -> None:
        library = FeatureLibrary()
        left = ConstraintHypothesis(
            "h_equal_y", "hold y", ("y_position",), "joint", "equality_band", "max", "linear",
            "move y", "A learned equality-like band."
        )
        right = ConstraintHypothesis(
            "h_upper_speed", "limit speed", ("speed",), "joint", "upper_bound", "max", "linear",
            "increase speed", "A learned one-sided limit."
        )
        bank = HypothesisBank.from_hypotheses([left, right], library)
        evidence = [
            HypothesisEvidence(item.hypothesis_id, 0.7, 0.7, 0.7, 1.0, 0.3, 1, 1, 0.2, 0.01, 0.5, 4, 2, 0.7, True)
            for item in (left, right)
        ]
        report = {
            "pair_complementarity": [
                {"hypothesis_ids": [left.hypothesis_id, right.hypothesis_id], "gain_over_best_single": 0.2}
            ]
        }
        policy = EvidencePolicyReasoner(SemanticConfig(beam_width=2, minimum_composition_gain=0.05))
        actions = policy.revise("hold y and limit speed", library, bank, report, evidence, 1)
        self.assertTrue(any(action.action == "compose_hypotheses" for action in actions))
        bank.apply_actions(actions, library, outer_round=1)
        composites = [item for item in bank.active() if len(item.atomic_clauses()) == 2]
        self.assertEqual(len(composites), 1)

    def test_openai_backend_requests_strict_structured_output(self) -> None:
        class Response:
            output_text = json.dumps(
                {
                    "hypotheses": [
                        normalize_hypothesis_payload(
                            {"id": "h_speed_api", "name": "speed upper", "variable": "speed", "relation": "upper"}
                        ),
                        normalize_hypothesis_payload(
                            {"id": "h_xy_api", "name": "planar region", "variables": ["x_position", "y_position"]}
                        ),
                    ]
                }
            )
            id = "resp_test"
            model = "gpt-test"
            usage = None

        class Responses:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return Response()

        class Client:
            def __init__(self) -> None:
                self.responses = Responses()

        client = Client()
        reasoner = OpenAISemanticReasoner(
            "gpt-test", SemanticConfig(allow_fallback=False), client=client
        )
        hypotheses = reasoner.propose_initial("avoid obstacle", FeatureLibrary())
        self.assertEqual(len(hypotheses), 2)
        output_format = client.responses.kwargs["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        self.assertFalse(reasoner.interactions[0]["used_fallback"])

    def test_frozen_bank_replays_only_round_zero_hypotheses_without_llm_interactions(self) -> None:
        library = FeatureLibrary()
        hypotheses = canonical_initial_hypotheses()[:2]
        bank = HypothesisBank.from_hypotheses(hypotheses, library)
        bank.audit_log.append(
            {
                "outer_round": 1,
                "event": "add",
                "hypothesis_id": hypotheses[0].hypothesis_id,
                "reason": "must not be replayed twice",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "hypothesis_bank.json"
            artifact.write_text(json.dumps(bank.to_dict()), encoding="utf-8")
            reasoner = FrozenBankSemanticReasoner(artifact)
            replayed = reasoner.propose_initial("unused", library)
        self.assertEqual(
            [item.hypothesis_id for item in replayed],
            [item.hypothesis_id for item in hypotheses],
        )
        self.assertEqual(reasoner.interactions, [])
        self.assertEqual(reasoner.source_manifest["llm_calls"], 0)
        self.assertEqual(len(reasoner.source_manifest["sha256"]), 64)

    def test_qwen_partial_bank_is_augmentation_not_full_fallback(self) -> None:
        raw = json.dumps(
            {
                "hypotheses": [
                    {
                        "id": "h_bad",
                        "variables": ["x_position", "y_position"],
                        "relation": "equality_band",
                    },
                    {"id": "h_speed_qwen", "variable": "speed", "relation": "upper"},
                ]
            }
        )
        reasoner = object.__new__(LocalQwenSemanticReasoner)
        reasoner.config = SemanticConfig(max_initial_hypotheses=2, allow_fallback=True)
        reasoner.interactions = []
        reasoner._fallback = EvidencePolicyReasoner(reasoner.config)
        reasoner._generate = lambda prompt: raw
        hypotheses = reasoner.propose_initial("avoid obstacle", FeatureLibrary())
        self.assertEqual(len(hypotheses), 2)
        self.assertEqual(reasoner.interactions[0]["accepted_llm_count"], 1)
        self.assertFalse(reasoner.interactions[0]["used_fallback"])
        self.assertTrue(reasoner.interactions[0]["used_augmentation"])

    def test_equivalent_llm_structures_are_canonicalized(self) -> None:
        scalar = normalize_hypothesis_payload(
            {
                "id": "h_scalar",
                "variables": ["speed"],
                "coupling": "independent",
                "relation": "upper_bound",
                "model_family": "linear",
            }
        )
        planar = normalize_hypothesis_payload(
            {
                "id": "h_planar",
                "variables": ["x_position", "y_position"],
                "coupling": "joint",
                "relation": "upper_bound",
                "model_family": "linear",
            }
        )
        self.assertEqual(scalar["coupling"], "joint")
        self.assertEqual(planar["relation"], "forbidden_region")
        compile_hypothesis(hypothesis_from_dict(scalar), FeatureLibrary())
        compile_hypothesis(hypothesis_from_dict(planar), FeatureLibrary())

    def test_qwen_multiple_champions_are_repaired_without_full_fallback(self) -> None:
        library = FeatureLibrary()
        hypotheses = canonical_initial_hypotheses()[:2]
        bank = HypothesisBank.from_hypotheses(hypotheses, library)
        evidence = [
            HypothesisEvidence(item.hypothesis_id, 0.6, 0.6, 0.6, 0.8, 0.4, 1, 1, 0.2, 0.01, 0.5, 4, 2, 0.6, True)
            for item in hypotheses
        ]
        raw = json.dumps(
            {
                "actions": [
                    {"action": "retain_and_query", "target": hypotheses[0].hypothesis_id},
                    {"action": "retain_and_query", "target": hypotheses[1].hypothesis_id},
                ]
            }
        )
        reasoner = object.__new__(LocalQwenSemanticReasoner)
        reasoner.config = SemanticConfig(beam_width=1, allow_fallback=True)
        reasoner.interactions = []
        reasoner._fallback = EvidencePolicyReasoner(reasoner.config)
        reasoner._generate = lambda prompt: raw
        actions = reasoner.revise(
            "avoid obstacle",
            library,
            bank,
            {"ranking": [hypotheses[0].hypothesis_id], "pair_complementarity": []},
            evidence,
            1,
        )
        self.assertEqual(sum(action.action == "retain_and_query" for action in actions), 1)
        self.assertFalse(reasoner.interactions[0]["used_fallback"])
        self.assertTrue(reasoner.interactions[0]["policy_augmented"])


if __name__ == "__main__":
    unittest.main()
