from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT))

from tools.generate_carrywater_active import (
    _generate_experts,
    _generate_public_candidate_pool,
    _private_rule,
)
from llm_modulo_cegis.carrywater_active import validate_trajectory
from llm_modulo_cegis.data import (
    FeatureLibrary,
    load_candidate_pool,
    load_expert_dataset,
    load_task_spec,
)
from llm_modulo_cegis.falsifier import FalsifierConfig, FalsifierResult
from llm_modulo_cegis.loop import LoopConfig, SemanticNumericCEGIS
from llm_modulo_cegis.oracle import RuleEvaluationOracle
from llm_modulo_cegis.pool_falsifier import PoolHypothesisFalsifier
from llm_modulo_cegis.semantic import canonical_initial_hypotheses
from llm_modulo_cegis.types import (
    InterventionSpec,
    SAFE_LABEL,
    Trajectory,
    VIOLATION_LABEL,
)


SUITE_ROOT = PACKAGE_ROOT / "data" / "CarryWaterActive"
PUBLIC_ROOT = SUITE_ROOT / "public" / "carrywater_active"
PRIVATE_ROOT = SUITE_ROOT / "private" / "carrywater_active"


class CarryWaterActiveContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_task_spec(PUBLIC_ROOT)
        cls.library = FeatureLibrary.from_task_spec(cls.spec)
        cls.experts = load_expert_dataset(PUBLIC_ROOT, "all")
        cls.candidates = load_candidate_pool(PUBLIC_ROOT)
        cls.oracle = RuleEvaluationOracle.from_private_files(
            PRIVATE_ROOT / "oracle.json",
            PRIVATE_ROOT / "evaluation_trajectories.npz",
        )

    def test_v2_task_contract_and_fixed_splits(self) -> None:
        self.assertEqual(self.spec.schema_version, 2)
        self.assertEqual(self.spec.raw_state_dimension, 12)
        self.assertEqual(self.spec.action_dimension, 6)
        self.assertEqual(self.spec.action_horizon, "transition")
        self.assertEqual(self.spec.trajectory_adapter, "carrywater_active_v1")
        self.assertAlmostEqual(self.spec.dt, 0.05)
        splits = json.loads((PUBLIC_ROOT / "splits.json").read_text(encoding="utf-8"))
        self.assertEqual({key: len(value) for key, value in splits.items()}, {
            "train": 40,
            "validation": 12,
            "test": 12,
        })
        self.assertEqual(len(self.experts), 64)

    def test_every_public_rollout_is_dynamically_valid(self) -> None:
        invalid = [
            (item.metadata.get("trajectory_id"), validate_trajectory(item))
            for item in [*self.experts, *self.candidates]
            if not validate_trajectory(item).valid
        ]
        self.assertEqual(invalid, [])
        self.assertTrue(all(item.actions.shape == (119, 6) for item in self.experts))
        self.assertTrue(all(item.states.shape == (120, 12) for item in self.candidates))

    def test_public_candidates_are_unlabeled_complete_pairs(self) -> None:
        with np.load(PUBLIC_ROOT / "candidate_trajectories.npz", allow_pickle=False) as archive:
            self.assertEqual(
                set(archive.files),
                {
                    "actions",
                    "lengths",
                    "observations",
                    "pair_ids",
                    "pair_members",
                    "phase_ids",
                    "reference_xyz",
                    "trajectory_ids",
                },
            )
            self.assertNotIn("labels", archive.files)
            pair_ids = archive["pair_ids"].astype(str)
            pair_members = archive["pair_members"]
        self.assertEqual(len(self.candidates), 512)
        for start in range(0, len(pair_ids), 2):
            self.assertEqual(pair_ids[start], pair_ids[start + 1])
            self.assertEqual(set(map(int, pair_members[start : start + 2])), {0, 1})

    def test_public_json_contains_no_private_answer(self) -> None:
        forbidden_keys = {
            "private_seed",
            "clauses",
            "expected_structure",
            "half_width",
            "threshold",
            "clause_labels",
            "normalized_severities",
        }

        def visit(value: object) -> None:
            if isinstance(value, dict):
                self.assertFalse(forbidden_keys & set(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for path in PUBLIC_ROOT.glob("*.json"):
            visit(json.loads(path.read_text(encoding="utf-8")))
            self.assertNotIn("917431", path.read_text(encoding="utf-8"))

    def test_experts_are_safe_and_warmup_prefix_has_both_labels(self) -> None:
        self.assertTrue(all(self.oracle.label(item) == SAFE_LABEL for item in self.experts))
        prefix = np.asarray(
            [self.oracle.label(item) for item in self.candidates[:12]], dtype=np.int64
        )
        self.assertEqual(np.bincount(prefix, minlength=2).tolist(), [3, 9])

    def test_feature_numpy_torch_parity_and_gradients(self) -> None:
        states = self.candidates[0].states
        numpy_values = self.library.numpy_features(states, self.library.names)
        tensor = torch.tensor(states, dtype=torch.float32, requires_grad=True)
        torch_values = self.library.torch_features(tensor, self.library.names)
        self.assertEqual(tuple(torch_values.shape), (120, 18))
        self.assertTrue(np.allclose(numpy_values, torch_values.detach().numpy(), atol=2.0e-6))
        torch_values.sum().backward()
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(bool(torch.all(torch.isfinite(tensor.grad))))

    def test_task_specific_fallback_contains_atomic_and_composite_rules(self) -> None:
        hypotheses = canonical_initial_hypotheses(self.library)
        self.assertEqual(len(hypotheses), 4)
        self.assertEqual(
            {item.hypothesis_id for item in hypotheses},
            {
                "h_target_dz_band",
                "h_speed_3d",
                "h_tilt_vertical",
                "h_carrywater_composite",
            },
        )
        composite = next(item for item in hypotheses if item.clauses)
        self.assertEqual(len(composite.clauses), 3)
        self.assertEqual(composite.composition, "any_violation")

    def test_pool_warmup_uses_new_global_rollouts(self) -> None:
        falsifier = PoolHypothesisFalsifier(
            self.library,
            FalsifierConfig(steps=1),
            self.candidates,
            torch.device("cpu"),
            validator=validate_trajectory,
        )
        first = falsifier.warmup_candidate(
            self.experts[0], 0, np.random.default_rng(7)
        )
        second = falsifier.warmup_candidate(
            self.experts[0], 1, np.random.default_rng(7)
        )
        self.assertEqual(first.metadata["warmup_pair_index"], 0)
        self.assertEqual(second.metadata["warmup_pair_index"], 0)
        self.assertEqual(first.metadata["warmup_basis"], "independent_control_space_rollout_pair")
        self.assertFalse(np.array_equal(first.states, self.experts[0].states))
        self.assertTrue(validate_trajectory(first).valid)

    def test_pool_rank_windows_advance_across_rounds_and_beyond_64(self) -> None:
        self.assertEqual(
            PoolHypothesisFalsifier.candidate_rank_offset(
                outer_round=1,
                pool_slot=0,
                pool_size=49,
                restarts=1,
            ),
            0,
        )
        self.assertEqual(
            PoolHypothesisFalsifier.candidate_rank_offset(
                outer_round=2,
                pool_slot=16,
                pool_size=49,
                restarts=1,
            ),
            65,
        )

        class RankedFakeEnsemble:
            def __init__(self) -> None:
                self.compiled = SimpleNamespace(
                    variables=("target_dz",),
                    clauses=(),
                    hypothesis=SimpleNamespace(hypothesis_id="h_rank_test"),
                )
                self.decision_threshold = torch.tensor(-1.0)

            @staticmethod
            def mean_hard_trajectory_score(features: torch.Tensor) -> torch.Tensor:
                if features.ndim == 2:
                    return torch.tensor(0.0, device=features.device)
                return torch.arange(
                    features.shape[0], dtype=features.dtype, device=features.device
                )

            @staticmethod
            def mean_trajectory_score(
                features: torch.Tensor,
                beta: float,
            ) -> torch.Tensor:
                del beta
                return RankedFakeEnsemble.mean_hard_trajectory_score(features)

            @staticmethod
            def trajectory_uncertainty(features: torch.Tensor, beta: float) -> torch.Tensor:
                del beta
                return torch.zeros(features.shape[0], device=features.device)

            @staticmethod
            def mean_state_score(features: torch.Tensor) -> torch.Tensor:
                return torch.zeros(features.shape[0], device=features.device)

        falsifier = PoolHypothesisFalsifier(
            self.library,
            FalsifierConfig(steps=1),
            self.candidates,
            torch.device("cpu"),
        )
        result = falsifier.generate(
            RankedFakeEnsemble(),
            self.experts[0],
            InterventionSpec(
                target_hypothesis_id="h_rank_test",
                kind="boundary_uncertainty",
            ),
            initialization_mix=0.0,
            restart_index=65,
        )
        self.assertEqual(result.trajectory.metadata["pool_selection_requested_rank"], 65)
        self.assertEqual(result.trajectory.metadata["pool_selection_rank"], 65)
        self.assertEqual(
            result.trajectory.metadata["pool_candidate_id"],
            self.candidates[65].metadata["trajectory_id"],
        )

    def test_validated_pool_endpoint_gets_query_side_hard_certificate(self) -> None:
        composite = next(
            item
            for item in canonical_initial_hypotheses(self.library)
            if item.hypothesis_id == "h_carrywater_composite"
        )
        expert = self.experts[0]
        candidate = self.candidates[20].copy()
        candidate.metadata.update(
            {
                "optimization_target_clause_id": "height_band",
                "source_anchor_margin_satisfied": True,
                "generation_hard_margin_satisfied": True,
            }
        )

        class FrozenCrossingRegistry:
            def __init__(self) -> None:
                self.models = {
                    composite.hypothesis_id: SimpleNamespace(
                        decision_threshold=torch.tensor(0.0)
                    )
                }

            @staticmethod
            def predict(
                hypothesis_id: str,
                trajectory: Trajectory,
            ) -> tuple[int, float, float]:
                del hypothesis_id
                score = -0.2 if trajectory is expert else 0.2
                return int(score > 0.0), score, 0.0

            @staticmethod
            def hard_clause_score(
                hypothesis_id: str,
                trajectory: Trajectory,
                clause_id: str,
            ) -> float:
                del hypothesis_id, clause_id
                return -0.2 if trajectory is expert else 0.2

        controller = object.__new__(SemanticNumericCEGIS)
        controller.falsifier = SimpleNamespace(
            uses_validated_global_rollout_pool=True
        )
        controller.registry = FrozenCrossingRegistry()
        controller.config = LoopConfig(safe_query_boundary_margin=0.02)
        result = FalsifierResult(
            candidate,
            "model_false_unsafe",
            composite.hypothesis_id,
            0.0,
            0.0,
            0.2,
            0.0,
            True,
            "valid_public_rollout",
        )

        certified = controller._refine_false_unsafe_to_nearest_boundary(
            result,
            expert,
            [composite],
        )

        self.assertIs(certified.trajectory, candidate)
        self.assertEqual(
            candidate.metadata["boundary_refinement_status"],
            "public_pool_endpoint_certified",
        )
        self.assertTrue(candidate.metadata["query_hard_margin_satisfied"])
        self.assertEqual(
            candidate.metadata["safe_query_causal_rejector_ids"],
            [composite.hypothesis_id],
        )
        self.assertEqual(candidate.metadata["query_target_clause_id"], "height_band")


class CarryWaterActivePrivateEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with np.load(PRIVATE_ROOT / "evaluation_trajectories.npz", allow_pickle=False) as archive:
            cls.arrays = {name: archive[name].copy() for name in archive.files}
        cls.oracle = RuleEvaluationOracle.from_private_files(PRIVATE_ROOT / "oracle.json")

    def test_private_bank_is_balanced_and_clause_stratified(self) -> None:
        labels = self.arrays["labels"]
        clauses = self.arrays["clause_labels"]
        roles = self.arrays["pair_roles"].astype(str)
        targets = self.arrays["pair_targets"].astype(str)
        self.assertEqual(labels.shape, (1024,))
        self.assertEqual(np.bincount(labels, minlength=2).tolist(), [512, 512])
        self.assertTrue(np.all(clauses[roles == "safe"] == 0))
        expected = {
            "height_only": (1, 0, 0),
            "speed_only": (0, 1, 0),
            "tilt_only": (0, 0, 1),
        }
        for target, pattern in expected.items():
            selected = (roles == "violation") & (targets == target)
            unique = {tuple(map(int, row)) for row in clauses[selected]}
            self.assertEqual(unique, {pattern})
            self.assertEqual(int(np.sum(selected)), 128)
        multi = clauses[(roles == "violation") & (targets == "multi_clause")]
        values, counts = np.unique(multi, axis=0, return_counts=True)
        self.assertEqual(
            {tuple(map(int, row)): int(count) for row, count in zip(values, counts)},
            {(1, 1, 0): 32, (1, 0, 1): 32, (0, 1, 1): 32, (1, 1, 1): 32},
        )

    def _pair_indices(self, target: str) -> list[tuple[int, int]]:
        pair_ids = self.arrays["pair_ids"].astype(str)
        roles = self.arrays["pair_roles"].astype(str)
        targets = self.arrays["pair_targets"].astype(str)
        result: list[tuple[int, int]] = []
        for pair_id in sorted(set(pair_ids[targets == target])):
            selected = np.flatnonzero(pair_ids == pair_id)
            self.assertEqual(len(selected), 2)
            safe = int(selected[np.flatnonzero(roles[selected] == "safe")[0]])
            violation = int(selected[np.flatnonzero(roles[selected] == "violation")[0]])
            result.append((safe, violation))
        return result

    def test_world_height_proxy_has_exact_opposite_label_collisions(self) -> None:
        observations = self.arrays["observations"]
        labels = self.arrays["labels"]
        pairs = self._pair_indices("height_only")
        self.assertEqual(len(pairs), 128)
        for safe, violation in pairs:
            self.assertTrue(np.array_equal(observations[safe, :, 2], observations[violation, :, 2]))
            self.assertFalse(np.array_equal(observations[safe, :, 5], observations[violation, :, 5]))
            self.assertEqual({int(labels[safe]), int(labels[violation])}, {0, 1})

    def test_single_velocity_component_proxies_have_exact_collisions(self) -> None:
        observations = self.arrays["observations"]
        held_axes = self.arrays["held_velocity_axes"]
        axis_counts = np.zeros(3, dtype=np.int64)
        for safe, violation in self._pair_indices("speed_only"):
            axis = int(held_axes[safe])
            self.assertEqual(axis, int(held_axes[violation]))
            axis_counts[axis] += 1
            self.assertTrue(
                np.array_equal(
                    observations[safe, :, 6 + axis],
                    observations[violation, :, 6 + axis],
                )
            )
            safe_speed = np.linalg.norm(observations[safe, :, 6:9], axis=1)
            violation_speed = np.linalg.norm(observations[violation, :, 6:9], axis=1)
            self.assertGreater(float(np.max(violation_speed)), float(np.max(safe_speed)))
        self.assertTrue(np.all(axis_counts > 0), axis_counts.tolist())

    def test_yaw_proxy_has_exact_opposite_label_collisions(self) -> None:
        observations = self.arrays["observations"]
        for safe, violation in self._pair_indices("tilt_only"):
            self.assertTrue(np.array_equal(observations[safe, :, 11], observations[violation, :, 11]))
            self.assertFalse(
                np.array_equal(observations[safe, :, 9:11], observations[violation, :, 9:11])
            )

    def test_private_rollouts_replay_and_labels_match_analytic_oracle(self) -> None:
        observations = self.arrays["observations"]
        actions = self.arrays["actions"]
        labels = self.arrays["labels"]
        for index in range(len(labels)):
            trajectory = Trajectory(observations[index], actions[index], dt=0.05)
            validity = validate_trajectory(trajectory)
            self.assertTrue(validity.valid, (index, validity))
            self.assertEqual(self.oracle.label(trajectory), int(labels[index]))

    def test_public_generation_arrays_do_not_depend_on_private_seed(self) -> None:
        rule_a = _private_rule(np.random.default_rng(11))
        rule_b = _private_rule(np.random.default_rng(29))
        experts_a, _ = _generate_experts(rule_a, np.random.default_rng(101))
        experts_b, _ = _generate_experts(rule_b, np.random.default_rng(101))
        self.assertTrue(
            np.array_equal(
                np.stack([item.observations for item in experts_a]),
                np.stack([item.observations for item in experts_b]),
            )
        )
        candidates_a, _ = _generate_public_candidate_pool(
            rule_a,
            np.random.default_rng(303),
            np.random.default_rng(305),
        )
        candidates_b, _ = _generate_public_candidate_pool(
            rule_b,
            np.random.default_rng(303),
            np.random.default_rng(305),
        )
        for key in candidates_a:
            self.assertTrue(np.array_equal(candidates_a[key], candidates_b[key]), key)


if __name__ == "__main__":
    unittest.main()
