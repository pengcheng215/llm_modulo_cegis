"""Regression contract for the intentionally small project root."""

from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_project_root_contains_only_primary_training_entrypoints(self) -> None:
        root_python_files = {path.name for path in PROJECT_ROOT.glob("*.py")}
        self.assertEqual(
            root_python_files,
            {
                "run_obstacle_avoid.py",
                "run_semtraj2d.py",
                "run_carrywater_active.py",
            },
        )

    def test_offline_tools_and_experiment_runners_are_grouped(self) -> None:
        expected_tools = {
            "evaluate_carrywater_active.py",
            "evaluate_semtraj2d.py",
            "generate_carrywater_active.py",
            "generate_semtraj2d.py",
            "replay_finalization.py",
        }
        expected_experiments = {
            "run_ablation.py",
            "run_carrywater_q48_multiseed.py",
            "run_falsifier_multiseed.py",
            "run_linear_max_support_gate_multiseed.py",
            "run_numeric_fitting_multiseed.py",
            "run_violation_pooling_multiseed.py",
        }
        self.assertTrue(
            expected_tools.issubset(
                {path.name for path in (PROJECT_ROOT / "tools").glob("*.py")}
            )
        )
        self.assertTrue(
            expected_experiments.issubset(
                {path.name for path in (PROJECT_ROOT / "experiments").glob("*.py")}
            )
        )


if __name__ == "__main__":
    unittest.main()
