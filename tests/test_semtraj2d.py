from __future__ import annotations

import json
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT))

from tools.generate_semtraj2d import (
    _derive_private_seed,
    _deterministic_npz,
    _file_sha256,
    _private_rule,
)
from llm_modulo_cegis.data import (
    FeatureLibrary,
    SEMTRAJ_FEATURE_SPECS,
    TaskSpec,
    load_expert_dataset,
    load_task_spec,
)
from llm_modulo_cegis.evaluation import (
    _binary_average_precision,
    _binary_auroc,
    evaluate_boundary,
    plot_boundary,
)
from llm_modulo_cegis.oracle import DeferredEvaluationOracle, RuleEvaluationOracle
from llm_modulo_cegis.structure_evaluation import evaluate_structure
from llm_modulo_cegis.types import SAFE_LABEL, Trajectory, VIOLATION_LABEL


SUITE_ROOT = PACKAGE_ROOT / "data" / "SemTraj2D"


class SemTrajFeatureTests(unittest.TestCase):
    def test_twelve_features_are_finite_and_differentiable(self) -> None:
        library = FeatureLibrary(SEMTRAJ_FEATURE_SPECS)
        self.assertEqual(len(library.names), 12)
        states = torch.tensor(
            [[0.0, 0.0], [0.1, 0.02], [0.22, 0.01], [0.35, 0.04]],
            dtype=torch.float32,
            requires_grad=True,
        )
        features = library.torch_features(states, library.names)
        self.assertEqual(tuple(features.shape), (4, 12))
        self.assertTrue(bool(torch.all(torch.isfinite(features))))
        features.sum().backward()
        self.assertIsNotNone(states.grad)
        self.assertTrue(bool(torch.all(torch.isfinite(states.grad))))


class SemTrajDatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = json.loads((SUITE_ROOT / "suite_manifest.json").read_text(encoding="utf-8"))
        cls.task_ids = [str(row["task_instance_id"]) for row in cls.suite["tasks"]]

    def test_suite_contains_supported_families_and_open_set_control(self) -> None:
        self.assertEqual(
            set(self.task_ids),
            {
                "disk_clean",
                "disk_upper_proxy",
                "diagonal_halfspace",
                "lane_band",
                "speed_limit",
                "disk_and_speed",
                "lane_and_speed",
                "eventually_visit_open_set",
            },
        )
        self.assertEqual(int(self.suite["task_count"]), 8)
        self.assertIn("public_generation_seed", self.suite)
        self.assertNotIn("dataset_seed", self.suite)
        public_suite = json.loads(
            (SUITE_ROOT / "public" / "suite_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(int(public_suite["task_count"]), 8)
        serialized_public = json.dumps(public_suite).lower()
        self.assertNotIn("evaluation_array", serialized_public)
        self.assertNotIn("oracle", serialized_public)

    def test_public_task_specs_and_archives_obey_contract(self) -> None:
        forbidden_top_level = {
            "ground_truth",
            "clauses",
            "constraint",
            "obstacle_center",
            "radius",
            "threshold",
        }
        for task_id in self.task_ids:
            public_dir = SUITE_ROOT / "public" / task_id
            spec_payload = json.loads((public_dir / "task_spec.json").read_text(encoding="utf-8"))
            self.assertFalse(forbidden_top_level & set(spec_payload))
            public_manifest = json.loads((public_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(forbidden_top_level & set(public_manifest))
            self.assertIn("public_generation_seed", public_manifest)
            self.assertNotIn("dataset_seed", public_manifest)
            for relative_path in public_manifest["learner_visible_files"]:
                normalized = Path(str(relative_path))
                self.assertFalse(normalized.is_absolute())
                self.assertNotIn("private", [part.lower() for part in normalized.parts])
            for relative_path, expected_hash in public_manifest["integrity"].items():
                self.assertEqual(_file_sha256(public_dir / relative_path), expected_hash)
            spec = load_task_spec(public_dir)
            self.assertEqual(spec.task_instance_id, task_id)
            self.assertEqual(spec.horizon, 100)
            self.assertEqual(len(spec.feature_specs), 12)
            all_experts = load_expert_dataset(public_dir, "all")
            self.assertEqual(len(all_experts), 30)
            splits = json.loads((public_dir / "splits.json").read_text(encoding="utf-8"))
            split_sets = [set(values) for values in splits.values()]
            self.assertFalse(split_sets[0] & split_sets[1])
            self.assertFalse(split_sets[0] & split_sets[2])
            self.assertFalse(split_sets[1] & split_sets[2])

    def test_task_spec_rejects_unknown_fields_at_every_schema_level(self) -> None:
        public_dir = SUITE_ROOT / "public" / "disk_clean"
        payload = json.loads((public_dir / "task_spec.json").read_text(encoding="utf-8"))
        top_level = copy.deepcopy(payload)
        top_level["radius"] = 1.5
        with self.assertRaises(ValueError):
            TaskSpec.from_dict(top_level)
        nested = copy.deepcopy(payload)
        nested["feature_schema"][0]["private_threshold"] = 0.2
        with self.assertRaises(ValueError):
            TaskSpec.from_dict(nested)

    def test_all_experts_are_safe_under_private_oracle(self) -> None:
        for task_id in self.task_ids:
            public_dir = SUITE_ROOT / "public" / task_id
            private_dir = SUITE_ROOT / "private" / task_id
            oracle = RuleEvaluationOracle.from_private_files(private_dir / "oracle.json")
            for expert in load_expert_dataset(public_dir, "all"):
                self.assertEqual(oracle.label(expert), SAFE_LABEL, task_id)

    def test_clean_and_confounded_disk_have_identical_private_test_bank(self) -> None:
        clean = SUITE_ROOT / "private" / "disk_clean"
        proxy = SUITE_ROOT / "private" / "disk_upper_proxy"
        clean_manifest = json.loads((clean / "manifest.json").read_text(encoding="utf-8"))
        proxy_manifest = json.loads((proxy / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            clean_manifest["evaluation_array_sha256"],
            proxy_manifest["evaluation_array_sha256"],
        )
        clean_rule = json.loads((clean / "oracle.json").read_text(encoding="utf-8"))
        proxy_rule = json.loads((proxy / "oracle.json").read_text(encoding="utf-8"))
        self.assertEqual(clean_rule, proxy_rule)
        self.assertNotIn("expected_structure", clean_rule)
        self.assertEqual(
            json.loads((clean / "expected_structure.json").read_text(encoding="utf-8")),
            json.loads((proxy / "expected_structure.json").read_text(encoding="utf-8")),
        )
        clean_spec = load_task_spec(SUITE_ROOT / "public" / "disk_clean")
        proxy_spec = load_task_spec(SUITE_ROOT / "public" / "disk_upper_proxy")
        self.assertEqual(clean_spec.task_description, proxy_spec.task_description)
        self.assertEqual(
            _file_sha256(clean / "evaluation_trajectories.npz"),
            _file_sha256(proxy / "evaluation_trajectories.npz"),
        )

    def test_clean_experts_cover_both_sides_but_proxy_experts_are_upper_only(self) -> None:
        rule = json.loads(
            (SUITE_ROOT / "private" / "disk_clean" / "oracle.json").read_text(encoding="utf-8")
        )
        center = np.asarray(rule["clauses"][0]["center"], dtype=float)

        def route_signs(task_id: str) -> list[int]:
            signs: list[int] = []
            for expert in load_expert_dataset(SUITE_ROOT / "public" / task_id, "all"):
                index = int(np.argmin(np.abs(expert.states[:, 0] - center[0])))
                signs.append(1 if expert.states[index, 1] > center[1] else -1)
            return signs

        clean_signs = route_signs("disk_clean")
        proxy_signs = route_signs("disk_upper_proxy")
        self.assertIn(-1, clean_signs)
        self.assertIn(1, clean_signs)
        self.assertEqual(set(proxy_signs), {1})
        with np.load(
            SUITE_ROOT / "public" / "disk_clean" / "expert_trajectories.npz",
            allow_pickle=False,
        ) as clean_archive, np.load(
            SUITE_ROOT / "public" / "disk_upper_proxy" / "expert_trajectories.npz",
            allow_pickle=False,
        ) as proxy_archive:
            clean_states = clean_archive["observations"]
            proxy_states = proxy_archive["observations"]
            self.assertTrue(np.array_equal(clean_states[..., 0], proxy_states[..., 0]))
            self.assertTrue(np.array_equal(clean_states[:, (0, -1), :], proxy_states[:, (0, -1), :]))
        self.assertEqual(
            json.loads((SUITE_ROOT / "public" / "disk_clean" / "splits.json").read_text(encoding="utf-8")),
            json.loads(
                (SUITE_ROOT / "public" / "disk_upper_proxy" / "splits.json").read_text(
                    encoding="utf-8"
                )
            ),
        )

    def test_private_evaluation_is_balanced_and_counterfactual_endpoints_match(self) -> None:
        for task_id in self.task_ids:
            private_dir = SUITE_ROOT / "private" / task_id
            oracle = RuleEvaluationOracle.from_private_files(
                private_dir / "oracle.json",
                private_dir / "evaluation_trajectories.npz",
            )
            bank = oracle.evaluation_trajectories()
            self.assertIsNotNone(bank)
            observations, labels, groups, _ = bank
            self.assertEqual(int(np.sum(labels == SAFE_LABEL)), int(np.sum(labels == VIOLATION_LABEL)))
            with np.load(private_dir / "evaluation_trajectories.npz", allow_pickle=False) as archive:
                pair_ids = archive["pair_ids"].astype(str)
                pair_roles = archive["pair_roles"].astype(str)
                for pair_id in sorted(set(pair_ids) - {""}):
                    selected = np.flatnonzero(pair_ids == pair_id)
                    self.assertEqual(len(selected), 2)
                    self.assertEqual(set(pair_roles[selected]), {"safe", "violation"})
                    self.assertTrue(np.allclose(observations[selected[0], 0], observations[selected[1], 0]))
                    self.assertTrue(np.allclose(observations[selected[0], -1], observations[selected[1], -1]))
                    self.assertEqual(set(labels[selected].tolist()), {SAFE_LABEL, VIOLATION_LABEL})
            for group in set(groups.tolist()):
                selected = groups == group
                self.assertGreater(int(np.sum(selected & (labels == SAFE_LABEL))), 0)
                self.assertGreater(int(np.sum(selected & (labels == VIOLATION_LABEL))), 0)

    def test_composite_bank_contains_isolated_and_joint_violations(self) -> None:
        tasks = {
            "disk_and_speed": {
                "spatial_only": (1, 0),
                "speed_only": (0, 1),
                "multi_clause": (1, 1),
            },
            "lane_and_speed": {
                "lane_only": (1, 0),
                "speed_only": (0, 1),
                "multi_clause": (1, 1),
            },
        }
        for task_id, expected in tasks.items():
            path = SUITE_ROOT / "private" / task_id / "evaluation_trajectories.npz"
            with np.load(path, allow_pickle=False) as archive:
                groups = archive["groups"].astype(str)
                clause_labels = archive["clause_labels"]
                labels = archive["labels"]
                for group, pattern in expected.items():
                    selected = (groups == group) & (labels == VIOLATION_LABEL)
                    self.assertGreater(int(np.sum(selected)), 0)
                    self.assertTrue(np.all(clause_labels[selected] == np.asarray(pattern)))

    def test_open_set_private_contract_marks_structure_unrepresentable(self) -> None:
        payload = json.loads(
            (SUITE_ROOT / "private" / "eventually_visit_open_set" / "expected_structure.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(bool(payload["representable"]))

    def test_deterministic_npz_writer_is_byte_stable(self) -> None:
        arrays = {
            "a": np.arange(12, dtype=np.float32).reshape(3, 4),
            "b": np.asarray(["x", "y"], dtype="U2"),
        }
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.npz"
            right = Path(directory) / "right.npz"
            _deterministic_npz(left, arrays)
            _deterministic_npz(right, arrays)
            self.assertEqual(_file_sha256(left), _file_sha256(right))

    def test_private_rule_seed_is_independent_of_public_generation_seed(self) -> None:
        left_seed = _derive_private_seed(0x123456789ABCDEF0123456789ABCDEF0, "speed_limit:rule")
        right_seed = _derive_private_seed(0xFEDCBA9876543210FEDCBA9876543210, "speed_limit:rule")
        self.assertNotEqual(left_seed, right_seed)
        self.assertNotEqual(_private_rule("speed_limit", left_seed), _private_rule("speed_limit", right_seed))

    def test_speed_truth_is_not_recoverable_from_one_velocity_component(self) -> None:
        path = SUITE_ROOT / "private" / "speed_limit" / "evaluation_trajectories.npz"
        with np.load(path, allow_pickle=False) as archive:
            states = archive["observations"].astype(np.float64)
            labels = archive["labels"].astype(np.int64)
        velocity = np.diff(states, axis=1)
        scores = {
            "speed": np.max(np.linalg.norm(velocity, axis=-1), axis=1),
            "positive_x": np.max(velocity[..., 0], axis=1),
            "negative_x": np.max(-velocity[..., 0], axis=1),
            "positive_y": np.max(velocity[..., 1], axis=1),
            "negative_y": np.max(-velocity[..., 1], axis=1),
        }

        def best_balanced_accuracy(values: np.ndarray) -> float:
            unique = np.unique(values)
            thresholds = np.concatenate(
                (
                    [unique[0] - 1.0e-9],
                    0.5 * (unique[:-1] + unique[1:]),
                    [unique[-1] + 1.0e-9],
                )
            )
            return max(
                0.5
                * (
                    np.mean((values <= threshold)[labels == SAFE_LABEL])
                    + np.mean((values > threshold)[labels == VIOLATION_LABEL])
                )
                for threshold in thresholds
            )

        self.assertGreater(best_balanced_accuracy(scores["speed"]), 0.999)
        for proxy_name in ("positive_x", "negative_x", "positive_y", "negative_y"):
            self.assertLess(best_balanced_accuracy(scores[proxy_name]), 0.90, proxy_name)


class RuleOracleTests(unittest.TestCase):
    def test_deferred_evaluator_exposes_no_private_truth(self) -> None:
        evaluator = DeferredEvaluationOracle()
        self.assertFalse(evaluator.supports_state_grid)
        self.assertIsNone(evaluator.evaluation_trajectories())
        self.assertIsNone(evaluator.evaluation_geometry())
        with self.assertRaises(RuntimeError):
            evaluator.state_violation_mask(np.zeros((1, 2), dtype=np.float32))

    def test_atomic_and_composite_rules(self) -> None:
        payload = {
            "schema_version": 1,
            "composition": "any_violation",
            "clauses": [
                {"clause_id": "lane", "kind": "equality_band", "variable": "y_position", "center": 0.0, "half_width": 0.5},
                {"clause_id": "speed", "kind": "speed_upper_bound", "threshold": 0.2},
            ],
        }
        oracle = RuleEvaluationOracle(payload)
        safe = Trajectory(np.column_stack((np.linspace(0.0, 1.0, 11), np.zeros(11))))
        lane_violation = Trajectory(
            np.column_stack(
                (
                    np.linspace(0.0, 1.0, 101),
                    0.6 * np.sin(np.linspace(0.0, np.pi, 101)),
                )
            )
        )
        speed_violation = Trajectory(np.asarray([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0]], dtype=np.float32))
        self.assertEqual(oracle.label(safe), SAFE_LABEL)
        self.assertEqual(oracle.clause_labels(lane_violation), {"lane": 1, "speed": 0})
        self.assertEqual(oracle.clause_labels(speed_violation), {"lane": 0, "speed": 1})
        membership = oracle.membership_view()
        self.assertFalse(hasattr(membership, "evaluation_geometry"))
        self.assertEqual(membership.query(safe), SAFE_LABEL)
        self.assertEqual(membership.query(lane_violation), VIOLATION_LABEL)
        self.assertEqual(membership.query_count, 2)

    def test_circle_checks_continuous_segments(self) -> None:
        payload = {
            "schema_version": 1,
            "composition": "any_violation",
            "clauses": [
                {"clause_id": "disk", "kind": "circle_exclusion", "center": [0.0, 0.0], "radius": 0.5}
            ],
        }
        oracle = RuleEvaluationOracle(payload)
        crossing = Trajectory(np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
        self.assertEqual(oracle.label(crossing), VIOLATION_LABEL)

    @staticmethod
    def _carrywater_states(
        *,
        reference_dz: float = 0.0,
        speed: float = 0.2,
        roll: float = 0.05,
    ) -> np.ndarray:
        states = np.zeros((5, 12), dtype=np.float32)
        states[:, 5] = reference_dz
        states[:, 6] = speed
        states[:, 9] = roll
        return states

    @staticmethod
    def _carrywater_payload(*, aliases: bool = False) -> dict[str, object]:
        if aliases:
            clauses = [
                {
                    "clause_id": "height",
                    "kind": "relative_height_band",
                    "center": 0.0,
                    "half_width": 0.1,
                },
                {
                    "clause_id": "speed",
                    "kind": "observed_speed_upper_bound",
                    "threshold": 0.5,
                },
                {
                    "clause_id": "tilt",
                    "kind": "tilt_from_vertical_upper_bound",
                    "threshold": 0.2,
                },
            ]
        else:
            clauses = [
                {
                    "clause_id": "height",
                    "kind": "equality_band",
                    "feature": "reference_dz",
                    "center": 0.0,
                    "half_width": 0.1,
                },
                {
                    "clause_id": "speed",
                    "kind": "l2_upper_bound",
                    "features": ["vx", "vy", "vz"],
                    "threshold": 0.5,
                },
                {
                    "clause_id": "tilt",
                    "kind": "upright_tilt_upper_bound",
                    "features": ["roll", "pitch"],
                    "formula": "acos(cos(roll)*cos(pitch))",
                    "threshold": 0.2,
                },
            ]
        return {
            "schema_version": 1,
            "task_instance_id": "carrywater_active",
            "composition": "any_violation",
            "observation_dimension": 12,
            "action_dimension": 6,
            "horizon": 5,
            "dt": 0.05,
            "clauses": clauses,
        }

    def test_carrywater_active_clause_names_and_canonical_schema_agree(self) -> None:
        trajectories = [
            Trajectory(self._carrywater_states(), dt=0.05),
            Trajectory(self._carrywater_states(reference_dz=0.11), dt=0.05),
            Trajectory(self._carrywater_states(speed=0.6), dt=0.05),
            Trajectory(self._carrywater_states(roll=0.25), dt=0.05),
        ]
        expected_clause_labels = [
            {"height": 0, "speed": 0, "tilt": 0},
            {"height": 1, "speed": 0, "tilt": 0},
            {"height": 0, "speed": 1, "tilt": 0},
            {"height": 0, "speed": 0, "tilt": 1},
        ]
        for aliases in (False, True):
            oracle = RuleEvaluationOracle(self._carrywater_payload(aliases=aliases))
            self.assertFalse(oracle.supports_state_grid)
            self.assertEqual(
                [oracle.clause_labels(trajectory) for trajectory in trajectories],
                expected_clause_labels,
            )
            self.assertEqual(
                [oracle.label(trajectory) for trajectory in trajectories],
                [SAFE_LABEL, VIOLATION_LABEL, VIOLATION_LABEL, VIOLATION_LABEL],
            )
            with self.assertRaises(NotImplementedError):
                oracle.state_violation_mask(np.zeros((3, 2), dtype=np.float32))

    def test_nonplanar_private_bank_skips_grid_and_keeps_trajectory_metrics(self) -> None:
        observations = np.stack(
            [
                self._carrywater_states(),
                self._carrywater_states(reference_dz=0.11),
                self._carrywater_states(speed=0.6),
                self._carrywater_states(roll=0.25),
            ]
        )
        labels = np.asarray(
            [SAFE_LABEL, VIOLATION_LABEL, VIOLATION_LABEL, VIOLATION_LABEL],
            dtype=np.int64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "evaluation_trajectories.npz"
            np.savez_compressed(
                archive_path,
                observations=observations,
                labels=labels,
                groups=np.asarray(["all"] * len(labels)),
                trajectory_ids=np.asarray([f"active-{index}" for index in range(len(labels))]),
            )
            oracle = RuleEvaluationOracle(
                self._carrywater_payload(),
                archive_path,
            )
            bank = oracle.evaluation_trajectories()
            self.assertIsNotNone(bank)
            assert bank is not None
            self.assertEqual(bank[0].shape, (4, 5, 12))

            class NonPlanarLibrary:
                raw_state_dimension = 12
                is_planar = False

                @staticmethod
                def grid_features(*args: object, **kwargs: object) -> np.ndarray:
                    raise AssertionError("non-planar evaluation must not call grid_features")

                @staticmethod
                def torch_features(states: torch.Tensor, variables: tuple[str, ...]) -> torch.Tensor:
                    del variables
                    height = torch.abs(states[..., 5]) - 0.1
                    speed = torch.linalg.vector_norm(states[..., 6:9], dim=-1) - 0.5
                    tilt = torch.acos(
                        torch.clamp(
                            torch.cos(states[..., 9]) * torch.cos(states[..., 10]),
                            -1.0,
                            1.0,
                        )
                    ) - 0.2
                    return torch.stack((height, speed, tilt), dim=-1)

            class NonPlanarEnsemble:
                compiled = SimpleNamespace(variables=("height", "speed", "tilt"))
                decision_threshold = torch.tensor(0.0)

                @staticmethod
                def mean_state_score(features: torch.Tensor) -> torch.Tensor:
                    del features
                    raise AssertionError("non-planar evaluation must not score a planar state grid")

                @staticmethod
                def mean_hard_trajectory_score(features: torch.Tensor) -> torch.Tensor:
                    return torch.max(features)

                @staticmethod
                def predict_features(features: torch.Tensor) -> int:
                    return int(torch.max(features).item() > 0.0)

            metrics, grid = evaluate_boundary(
                NonPlanarEnsemble(),  # type: ignore[arg-type]
                NonPlanarLibrary(),  # type: ignore[arg-type]
                oracle,
                [Trajectory(observations[0], dt=0.05)],
                (-1.0, 1.0),
                (-1.0, 1.0),
                8,
                torch.device("cpu"),
            )
            self.assertEqual(metrics.evaluation_trajectory_count, 4)
            self.assertAlmostEqual(metrics.trajectory_balanced_accuracy, 1.0)
            self.assertAlmostEqual(metrics.trajectory_safe_accuracy, 1.0)
            self.assertAlmostEqual(metrics.trajectory_violation_recall, 1.0)
            self.assertAlmostEqual(metrics.heldout_expert_safe_rate, 1.0)
            self.assertTrue(all(array.size == 0 for array in grid))

            plot_path = Path(temporary) / "nonplanar.png"
            plot_boundary(
                plot_path,
                grid,
                [],
                [],
                oracle,
                "CarryWaterActive trajectory-only evaluation",
            )
            self.assertTrue(plot_path.is_file())


class MetricTests(unittest.TestCase):
    def test_average_precision_is_permutation_invariant_for_ties(self) -> None:
        labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
        scores = np.ones(4, dtype=np.float64)
        self.assertAlmostEqual(_binary_average_precision(labels, scores), 0.5)
        order = np.asarray([3, 1, 0, 2])
        self.assertAlmostEqual(
            _binary_average_precision(labels[order], scores[order]),
            0.5,
        )
        self.assertAlmostEqual(_binary_auroc(labels, scores), 0.5)

    def test_rank_metrics_handle_perfect_and_reversed_scores(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        perfect = np.asarray([0.1, 0.2, 0.8, 0.9])
        reversed_scores = -perfect
        self.assertAlmostEqual(_binary_average_precision(labels, perfect), 1.0)
        self.assertAlmostEqual(_binary_auroc(labels, perfect), 1.0)
        self.assertAlmostEqual(_binary_auroc(labels, reversed_scores), 0.0)

    def test_structure_scoring_is_permutation_invariant(self) -> None:
        lane = {
            "variables": ["y_position"],
            "coupling": "joint",
            "relation": "equality_band",
            "temporal_operator": "max",
            "model_family": "linear",
        }
        speed = {
            "variables": ["speed"],
            "coupling": "joint",
            "relation": "upper_bound",
            "temporal_operator": "max",
            "model_family": "linear",
        }
        expected = {
            "representable": True,
            "composition": "any_violation",
            "clauses": [lane, speed],
        }
        predicted = {
            "composition": "any_violation",
            "clauses": [
                {"clause_id": "speed", **speed},
                {"clause_id": "lane", **lane, "coupling": "independent"},
            ],
        }
        score = evaluate_structure(predicted, expected, selection_status="qualified")
        self.assertTrue(score["exact_structure_recovery"])
        self.assertTrue(score["qualified_exact_structure_recovery"])
        self.assertEqual(score["component_accuracy"]["variables"], 1.0)

        unqualified = evaluate_structure(predicted, expected, selection_status="inconclusive")
        self.assertTrue(unqualified["exact_structure_recovery"])
        self.assertFalse(unqualified["qualified_exact_structure_recovery"])

    def test_representable_wrong_qualified_structure_is_erroneous(self) -> None:
        expected = {
            "representable": True,
            "composition": "any_violation",
            "clauses": [
                {
                    "variables": ["target_dz"],
                    "coupling": "joint",
                    "relation": "equality_band",
                    "temporal_operator": "max",
                    "model_family": "linear",
                }
            ],
        }
        wrong = {
            "composition": "any_violation",
            "variables": ["z_position"],
            "coupling": "joint",
            "relation": "equality_band",
            "temporal_operator": "max",
            "model_family": "linear",
        }
        score = evaluate_structure(wrong, expected, selection_status="qualified")
        self.assertFalse(score["exact_structure_recovery"])
        self.assertFalse(score["qualified_exact_structure_recovery"])
        self.assertTrue(score["erroneous_qualified_champion"])

        exact = {
            "composition": "any_violation",
            "variables": ["target_dz"],
            "coupling": "joint",
            "relation": "equality_band",
            "temporal_operator": "max",
            "model_family": "linear",
        }
        exact_score = evaluate_structure(exact, expected, selection_status="qualified")
        self.assertTrue(exact_score["exact_structure_recovery"])
        self.assertTrue(exact_score["qualified_exact_structure_recovery"])
        self.assertFalse(exact_score["erroneous_qualified_champion"])

    def test_open_set_structure_scoring_rewards_inconclusive(self) -> None:
        expected = {"representable": False, "clauses": []}
        score = evaluate_structure(None, expected, selection_status="inconclusive")
        self.assertTrue(score["correct_abstention"])
        self.assertFalse(score["erroneous_qualified_champion"])
        self.assertIsNone(score["exact_structure_recovery"])
        self.assertIsNone(score["qualified_exact_structure_recovery"])


if __name__ == "__main__":
    unittest.main()
