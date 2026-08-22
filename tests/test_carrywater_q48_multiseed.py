from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT))

from llm_modulo_cegis.data import load_candidate_pool
from experiments.run_carrywater_q48_multiseed import (
    ARMS,
    AuditError,
    CORRECT_HYPOTHESIS_ID,
    DEFAULT_PLAN,
    audit_run,
    build_posthoc_command,
    config_differences,
    diagnose,
    file_sha256,
    load_yaml,
    repository_path,
    validate_plan,
)


PUBLIC_ROOT = PACKAGE_ROOT / "data" / "CarryWaterActive" / "public" / "carrywater_active"


def _row(arm: str, *, qualified_exact: bool, erroneous: bool, private: bool) -> dict[str, object]:
    return {
        "arm": arm,
        "qualified_exact": qualified_exact,
        "erroneous_qualified": erroneous,
        "private_success": private,
    }


class CarryWaterQ48PlanTests(unittest.TestCase):
    def test_real_plan_has_only_registered_treatment_differences(self) -> None:
        plan = load_yaml(DEFAULT_PLAN)
        report = validate_plan(plan, DEFAULT_PLAN)
        self.assertTrue(report["valid"])
        self.assertTrue(report["correct_composite_shared_exactly"])
        self.assertTrue(report["query_budget_matched"])
        self.assertFalse(report["compute_budget_matched"])
        self.assertEqual(report["queries_per_run"], 48)

    def test_config_difference_detector_exposes_unregistered_change(self) -> None:
        left = {"trainer": {"epochs": 80}, "loop": {"candidate_pool_per_hypothesis": 49}}
        right = {"trainer": {"epochs": 81}, "loop": {"candidate_pool_per_hypothesis": 7}}
        self.assertEqual(
            config_differences(left, right),
            {"trainer.epochs", "loop.candidate_pool_per_hypothesis"},
        )

    def test_posthoc_commands_keep_official_and_diagnostic_outputs_separate(self) -> None:
        output = Path("example")
        official = build_posthoc_command(output, diagnostic_all=False)
        diagnostic = build_posthoc_command(output, diagnostic_all=True)
        self.assertNotIn("--diagnostic-all-hypotheses", official)
        self.assertIn("--diagnostic-all-hypotheses", diagnostic)
        self.assertIn("posthoc_all_hypotheses_diagnostic.json", diagnostic[-1])


class CarryWaterQ48DiagnosisTests(unittest.TestCase):
    def _rows(
        self,
        *,
        correct_qe: int = 5,
        correct_private: int = 5,
        full_qe: int = 5,
        full_private: int = 5,
        full_wrong: int = 0,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index in range(5):
            rows.append(
                _row(
                    ARMS[0],
                    qualified_exact=index < correct_qe,
                    erroneous=False,
                    private=index < correct_private,
                )
            )
            rows.append(
                _row(
                    ARMS[1],
                    qualified_exact=index < full_qe,
                    erroneous=index < full_wrong,
                    private=index < full_private,
                )
            )
        return rows

    def test_decision_tree_localizes_numeric_failure_before_structure_ranking(self) -> None:
        result = diagnose(self._rows(correct_qe=3, full_wrong=2))
        self.assertEqual(result["code"], "numeric_fitting_or_acquisition_instability")

    def test_decision_tree_localizes_proxy_selection_after_numeric_ceiling_passes(self) -> None:
        result = diagnose(self._rows(full_qe=3, full_wrong=1))
        self.assertEqual(result["code"], "structure_ranking_or_proxy_rejection_failure")

    def test_decision_tree_releases_baselines_only_after_both_arms_pass(self) -> None:
        result = diagnose(self._rows())
        self.assertEqual(result["code"], "ready_for_matched_baselines")
        self.assertEqual(result["stability_target_count"], 4)


class CarryWaterQ48ArtifactAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = load_yaml(DEFAULT_PLAN)
        cls.candidates = load_candidate_pool(PUBLIC_ROOT)

    def _write_json(self, path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _make_valid_run(self, output_root: Path) -> Path:
        arm = ARMS[0]
        seed = 7
        output = output_root / arm / f"seed_{seed}"
        output.mkdir(parents=True)
        selected = self.candidates[:48]
        observations = np.stack([item.states for item in selected]).astype(np.float32)
        actions = np.stack([item.actions for item in selected]).astype(np.float32)
        labels = np.asarray(
            [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1] + [0, 1] * 18,
            dtype=np.int64,
        )
        rounds = np.repeat(np.arange(4, dtype=np.int64), 12)
        config = load_yaml(repository_path(self.plan["arms"][arm]["config"]))
        config["seed"] = seed
        config["output_dir"] = str(output.resolve())
        np.savez_compressed(
            output / "oracle_queries.npz",
            observations=observations,
            actions=actions,
            labels=labels,
            outer_rounds=rounds,
        )
        query_log = []
        warmup_sources = ["warmup"] * 12
        warmup_roles = ["warmup_training"] * 12
        role_specs = (
            ("final_calibration", "final_threshold_calibration"),
            ("warmup_validation", "heldout_structure_selection"),
        )
        unassigned = set(range(12))
        for source, evidence_role in role_specs:
            for label_name, label in (("safe", 0), ("violation", 1)):
                count = int(config["loop"].get(f"{source}_{label_name}_count", 0))
                selected_indices = [
                    index
                    for index in sorted(unassigned)
                    if int(labels[index]) == label
                ][:count]
                if len(selected_indices) != count:
                    raise AssertionError(
                        f"synthetic warmup cannot allocate {source}:{label_name}={count}"
                    )
                for index in selected_indices:
                    warmup_sources[index] = source
                    warmup_roles[index] = evidence_role
                    unassigned.remove(index)
        for index, (item, label, outer_round) in enumerate(zip(selected, labels, rounds)):
            source = warmup_sources[index] if index < 12 else "active_query"
            evidence_role = warmup_roles[index] if index < 12 else "active_query"
            query_log.append(
                {
                    "label": int(label),
                    "source": source,
                    "outer_round": int(outer_round),
                    "trajectory_metadata": {
                        "pool_candidate_id": item.metadata["trajectory_id"],
                        "evidence_role": evidence_role,
                    },
                }
            )
        self._write_json(output / "oracle_query_log.json", query_log)
        result = {
            "oracle_queries": 48,
            "llm_interactions": 0,
            "llm_fallbacks": 0,
            "llm_augmentations": 0,
            "selection_status": "qualified",
            "champion_eligible": True,
            "champion_hypothesis_id": CORRECT_HYPOTHESIS_ID,
            "decision_threshold": 0.125,
        }
        self._write_json(output / "result.json", result)
        self._write_json(output / "semantic_interactions.json", [])
        self._write_json(output / "hypothesis_bank.json", {})
        (output / "constraint_models.pt").write_bytes(b"synthetic-test-checkpoint")
        selection_evidence = {
            "hypothesis_id": CORRECT_HYPOTHESIS_ID,
            "balanced_accuracy": 0.75,
            "safe_accuracy": 0.8,
            "violation_recall": 0.7,
            "expert_safe_rate": 0.95,
            "fit_expert_safe_rate": 0.96,
            "selection_score": 0.75,
            "champion_eligible": True,
            "ineligibility_reasons": [],
        }
        self._write_json(
            output / "evidence_history.json",
            [
                {
                    "outer_round": 3,
                    "selection_status": "qualified",
                    "qualified_ranking": [CORRECT_HYPOTHESIS_ID],
                    "hypotheses": [selection_evidence],
                }
            ],
        )
        checkpoint_row = {
            "outer_round": 3,
            "hypothesis_id": CORRECT_HYPOTHESIS_ID,
            "selection_key": [0.75, 3],
            "selection_evidence": selection_evidence,
            "decision_threshold": 0.125,
            "selected_as_best_so_far": True,
        }
        self._write_json(
            output / "qualified_checkpoint_history.json", [checkpoint_row]
        )
        self._write_json(
            output / "stage_diagnostics.json",
            [
                {
                    "stage": "best_qualified_checkpoint_restore",
                    "restored_outer_round": 3,
                    "champion_hypothesis_id": CORRECT_HYPOTHESIS_ID,
                    "selection_key": [0.75, 3],
                    "selection_evidence": selection_evidence,
                    "decision_threshold": 0.125,
                }
            ],
        )
        self._write_json(output / "evaluation_history.json", [])
        self._write_json(output / "query_diagnostics.json", [])
        self._write_json(output / "threshold_calibration_history.json", [])
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        inputs = {
            f"llm_modulo_cegis/data/CarryWaterActive/public/carrywater_active/{path.name}": file_sha256(path)
            for path in PUBLIC_ROOT.iterdir()
            if path.is_file()
        }
        implementation = {
            "files": {"synthetic_runner.py": hashlib.sha256(b"same").hexdigest()},
            "inputs": inputs,
            "evaluation_only_inputs": {},
            "runtime": {
                "seed": seed,
                "private_evaluation_mode": "deferred_posthoc",
                "environment": {
                    "PYTHONHASHSEED": str(seed),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            },
        }
        self._write_json(output / "implementation_manifest.json", implementation)
        freeze_names = (
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
        freeze = {
            "training_complete": True,
            "champion_hypothesis_id": CORRECT_HYPOTHESIS_ID,
            "selection_status": "qualified",
            "private_evaluation_mode": "deferred_posthoc",
            "private_evaluation_loaded_before_freeze": False,
            "training_artifact_sha256": {
                name: file_sha256(output / name) for name in freeze_names
            },
        }
        self._write_json(output / "freeze_manifest.json", freeze)
        champion_metrics = {
            "heldout_expert_safe_rate": 0.95,
            "trajectory_balanced_accuracy": 0.8,
            "trajectory_safe_accuracy": 0.9,
            "trajectory_violation_recall": 0.7,
            "trajectory_exact_pair_accuracy": 0.6,
            "trajectory_pair_ranking_accuracy": 0.75,
            "trajectory_worst_pair_target_balanced_accuracy": 0.65,
            "trajectory_minimum_clause_recall": 0.7,
            "trajectory_pair_target_balanced_accuracy": {
                "height_only": 0.65,
                "speed_only": 0.72,
                "tilt_only": 0.70,
            },
            "trajectory_clause_recall": {"height": 0.7, "speed": 0.75, "tilt": 0.72},
        }
        structure = {
            "exact_structure_recovery": True,
            "qualified_exact_structure_recovery": True,
            "erroneous_qualified_champion": False,
        }
        official = {
            "evaluation_protocol": "frozen_checkpoint_posthoc_private",
            "champion_hypothesis_id": CORRECT_HYPOTHESIS_ID,
            "selection_status_frozen_before_private_evaluation": "qualified",
            "frozen_artifact_sha256": {
                "constraint_models.pt": file_sha256(output / "constraint_models.pt"),
                "result.json": file_sha256(output / "result.json"),
            },
            "champion_metrics": champion_metrics,
            "structure_metrics": structure,
            "diagnostic_all_hypotheses_enabled": False,
        }
        diagnostic = copy.deepcopy(official)
        diagnostic["diagnostic_all_hypotheses_enabled"] = True
        diagnostic["all_hypothesis_metrics_diagnostic_only"] = {
            CORRECT_HYPOTHESIS_ID: champion_metrics
        }
        self._write_json(output / "posthoc_evaluation.json", official)
        self._write_json(output / "posthoc_all_hypotheses_diagnostic.json", diagnostic)
        stages = {}
        for stage in ("training", "official_posthoc", "diagnostic_all_posthoc"):
            stdout = f"{stage}_stdout.log"
            stderr = f"{stage}_stderr.log"
            (output / stdout).write_text("", encoding="utf-8")
            (output / stderr).write_text("", encoding="utf-8")
            stages[stage] = {
                "returncode": 0,
                "duration_seconds": 0.1,
                "stdout": stdout,
                "stderr": stderr,
            }
        self._write_json(
            output / "q48_run_metadata.json",
            {
                "arm": arm,
                "seed": seed,
                "total_duration_seconds": 0.3,
                "stages": stages,
            },
        )
        return output

    def test_strict_artifact_audit_accepts_a_complete_q48_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            output = self._make_valid_run(output_root)
            row = audit_run(self.plan, output_root, ARMS[0], 7)
            self.assertEqual(row["oracle_queries"], 48)
            self.assertEqual(row["unique_trajectory_count"], 48)
            self.assertTrue(row["qualified_exact"])
            self.assertTrue(row["private_success"])
            self.assertTrue(row["qualified_checkpoint_restored"])
            self.assertFalse(row["qualified_checkpoint_restored_from_earlier_round"])
            self.assertTrue(row["all_q48_audit_artifacts_sealed"])
            config = yaml.safe_load(
                (output / "resolved_config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                row["warmup_role_label_counts"]["final_calibration:safe"],
                int(config["loop"]["final_calibration_safe_count"]),
            )
            self.assertEqual(
                row["warmup_role_label_counts"]["final_calibration:violation"],
                int(config["loop"]["final_calibration_violation_count"]),
            )

    def test_strict_artifact_audit_rejects_round_underfill_or_overfill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            output = self._make_valid_run(output_root)
            with np.load(output / "oracle_queries.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            arrays["outer_rounds"][-1] = 2
            np.savez_compressed(output / "oracle_queries.npz", **arrays)
            freeze = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
            freeze["training_artifact_sha256"]["oracle_queries.npz"] = file_sha256(
                output / "oracle_queries.npz"
            )
            self._write_json(output / "freeze_manifest.json", freeze)
            with self.assertRaisesRegex(AuditError, "round distribution"):
                audit_run(self.plan, output_root, ARMS[0], 7)

    def test_strict_artifact_audit_rejects_unsealed_q48_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            output = self._make_valid_run(output_root)
            freeze = json.loads(
                (output / "freeze_manifest.json").read_text(encoding="utf-8")
            )
            del freeze["training_artifact_sha256"]["threshold_calibration_history.json"]
            self._write_json(output / "freeze_manifest.json", freeze)
            with self.assertRaisesRegex(AuditError, "six audit artifacts"):
                audit_run(self.plan, output_root, ARMS[0], 7)

    def test_audit_uses_restored_qualified_checkpoint_when_last_refit_regresses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            output = self._make_valid_run(output_root)
            earlier = {
                "hypothesis_id": CORRECT_HYPOTHESIS_ID,
                "balanced_accuracy": 0.82,
                "safe_accuracy": 0.90,
                "violation_recall": 0.74,
                "expert_safe_rate": 0.98,
                "fit_expert_safe_rate": 0.97,
                "selection_score": 0.82,
                "champion_eligible": True,
                "ineligibility_reasons": [],
            }
            strongest = {
                "hypothesis_id": CORRECT_HYPOTHESIS_ID,
                "balanced_accuracy": 0.84,
                "safe_accuracy": 0.92,
                "violation_recall": 0.76,
                "expert_safe_rate": 0.99,
                "fit_expert_safe_rate": 0.98,
                "selection_score": 0.90,
                "champion_eligible": True,
                "ineligibility_reasons": [],
            }
            regressed = {
                "hypothesis_id": CORRECT_HYPOTHESIS_ID,
                "balanced_accuracy": 0.50,
                "safe_accuracy": 1.0,
                "violation_recall": 0.0,
                "expert_safe_rate": 1.0,
                "fit_expert_safe_rate": 1.0,
                "selection_score": 0.40,
                "champion_eligible": False,
                "ineligibility_reasons": ["violation_recall_below_gate"],
            }
            self._write_json(
                output / "evidence_history.json",
                [
                    {
                        "outer_round": 1,
                        "selection_status": "qualified",
                        "qualified_ranking": [CORRECT_HYPOTHESIS_ID],
                        "hypotheses": [strongest],
                    },
                    {
                        "outer_round": 2,
                        "selection_status": "qualified",
                        "qualified_ranking": [CORRECT_HYPOTHESIS_ID],
                        "hypotheses": [earlier],
                    },
                    {
                        "outer_round": 3,
                        "selection_status": "inconclusive",
                        "qualified_ranking": [],
                        "hypotheses": [regressed],
                    },
                ],
            )
            checkpoint = {
                "outer_round": 1,
                "hypothesis_id": CORRECT_HYPOTHESIS_ID,
                "selection_key": [0.90, 1],
                "selection_evidence": strongest,
                "decision_threshold": 0.125,
                "selected_as_best_so_far": True,
            }
            weaker_checkpoint = {
                "outer_round": 2,
                "hypothesis_id": CORRECT_HYPOTHESIS_ID,
                "selection_key": [0.82, 2],
                "selection_evidence": earlier,
                "decision_threshold": 0.2,
                "selected_as_best_so_far": False,
            }
            self._write_json(
                output / "qualified_checkpoint_history.json",
                [checkpoint, weaker_checkpoint],
            )
            self._write_json(
                output / "stage_diagnostics.json",
                [
                    {
                        "stage": "best_qualified_checkpoint_restore",
                        "restored_outer_round": 1,
                        "champion_hypothesis_id": CORRECT_HYPOTHESIS_ID,
                        "selection_key": [0.90, 1],
                        "selection_evidence": strongest,
                        "decision_threshold": 0.125,
                    }
                ],
            )
            freeze = json.loads(
                (output / "freeze_manifest.json").read_text(encoding="utf-8")
            )
            for name in (
                "evidence_history.json",
                "qualified_checkpoint_history.json",
                "stage_diagnostics.json",
            ):
                freeze["training_artifact_sha256"][name] = file_sha256(output / name)
            self._write_json(output / "freeze_manifest.json", freeze)
            row = audit_run(self.plan, output_root, ARMS[0], 7)
            self.assertAlmostEqual(row["public_balanced_accuracy"], 0.84)
            self.assertTrue(row["qualified_checkpoint_restored"])
            self.assertTrue(row["qualified_checkpoint_restored_from_earlier_round"])
            self.assertEqual(
                row["public_evidence_provenance"]["last_round_selection_status"],
                "inconclusive",
            )

    def test_strict_artifact_audit_rejects_mislabeled_calibration_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            output = self._make_valid_run(output_root)
            query_log = json.loads(
                (output / "oracle_query_log.json").read_text(encoding="utf-8")
            )
            calibration_index = next(
                index
                for index, entry in enumerate(query_log[:12])
                if entry["source"] == "final_calibration"
            )
            query_log[calibration_index]["trajectory_metadata"][
                "evidence_role"
            ] = "warmup_training"
            self._write_json(output / "oracle_query_log.json", query_log)
            freeze = json.loads(
                (output / "freeze_manifest.json").read_text(encoding="utf-8")
            )
            freeze["training_artifact_sha256"]["oracle_query_log.json"] = file_sha256(
                output / "oracle_query_log.json"
            )
            self._write_json(output / "freeze_manifest.json", freeze)
            with self.assertRaisesRegex(AuditError, "source/evidence-role mismatch"):
                audit_run(self.plan, output_root, ARMS[0], 7)


if __name__ == "__main__":
    unittest.main()
