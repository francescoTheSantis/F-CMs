import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from plot_architecture_ablation import (
    _reviewer_loss_table,
    audit_controlled_design,
)
from src.architecture_ablation import (
    AblationMetricConfig,
    ArchitectureAblationRecorder,
    canonical_structural_metrics,
    compute_ablation_metrics,
)


class AblationMetricTests(unittest.TestCase):
    def test_shift_metrics_and_post_only_recovery(self):
        rounds = list(range(1, 13))
        losses = [0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
        config = AblationMetricConfig(
            pre_window=3,
            post_peak_window=3,
            smoothing_window=2,
            stability_window=2,
            final_window=2,
        )

        metrics = compute_ablation_metrics(
            rounds, losses, shift_round=6, metric_config=config
        )

        self.assertAlmostEqual(metrics["pre_shift_loss"], 0.6)
        self.assertAlmostEqual(metrics["post_shift_peak_loss"], 0.9)
        self.assertEqual(metrics["post_shift_peak_round"], 6)
        self.assertAlmostEqual(metrics["loss_spike"], 0.3)
        self.assertAlmostEqual(metrics["minimum_post_shift_loss"], 0.2)
        self.assertEqual(metrics["minimum_post_shift_loss_round"], 12)
        # The first full post-only trailing average <= 0.6 is at round 9:
        # mean(rounds 8, 9) = 0.55.  Round 10 remains below the baseline.
        self.assertEqual(metrics["recovery_round"], 9)
        self.assertEqual(metrics["recovery_time"], 3)
        self.assertEqual(metrics["recovery_time_rounds"], 3)
        self.assertTrue(metrics["recovered"])
        self.assertAlmostEqual(metrics["cumulative_excess_loss"], 0.5)
        self.assertAlmostEqual(metrics["final_window_loss"], 0.25)
        self.assertAlmostEqual(metrics["final_delta"], -0.35)

    def test_recovery_requires_complete_stability_window(self):
        metrics = compute_ablation_metrics(
            rounds=[1, 2, 3, 4, 5, 6, 7],
            client_mean_losses=[0.5, 0.5, 0.4, 0.4, 0.8, 0.4, 0.4],
            shift_round=3,
            metric_config={
                "pre_window": 2,
                "post_peak_window": 5,
                "smoothing_window": 1,
                "stability_window": 3,
                "final_window": 2,
            },
        )

        self.assertFalse(metrics["recovered"])
        self.assertIsNone(metrics["recovery_round"])
        self.assertIsNone(metrics["recovery_time"])

    def test_horizon_limits_peak_minimum_and_final_window(self):
        metrics = compute_ablation_metrics(
            rounds=range(1, 10),
            client_mean_losses=[1.0, 1.0, 1.0, 1.4, 1.2, 0.9, 0.8, 2.0, 0.1],
            shift_round=4,
            metric_config={
                "pre_window": 3,
                "post_peak_window": 2,
                "smoothing_window": 1,
                "stability_window": 1,
                "post_horizon": 4,
                "final_window": 2,
            },
        )

        self.assertEqual(metrics["post_shift_horizon_end_round"], 7)
        self.assertEqual(metrics["post_shift_peak_round"], 4)
        self.assertAlmostEqual(metrics["post_shift_peak_loss"], 1.4)
        self.assertEqual(metrics["minimum_post_shift_loss_round"], 7)
        self.assertAlmostEqual(metrics["minimum_post_shift_loss"], 0.8)
        self.assertAlmostEqual(metrics["final_window_loss"], 0.85)

    def test_missing_pre_curve_produces_json_safe_convertible_metrics(self):
        metrics = compute_ablation_metrics([5, 6], [1.0, 0.8], shift_round=5)
        self.assertTrue(math.isnan(metrics["pre_shift_loss"]))
        self.assertTrue(math.isnan(metrics["loss_spike"]))
        self.assertIsNone(metrics["recovery_round"])


class StructuralMetricTests(unittest.TestCase):
    def test_nested_counts_are_normalized_and_fractions_derived(self):
        metrics = canonical_structural_metrics(
            {
                "parameter_counts": {
                    "total_post_shift": 100,
                    "preserved": 70,
                    "newly_initialized": 20,
                    "fully_reinitialized": 10,
                },
                "added_concepts": ["c3", "c4"],
                "affected_modules": ["m1", "m2", "m3"],
                "changed_tensors": {"a": {}, "b": {}},
                "function_preserving": False,
                "fallbacks": [{"reason": "unsafe"}],
            }
        )

        self.assertEqual(metrics["parameters_total"], 100)
        self.assertEqual(metrics["parameters_preserved"], 70)
        self.assertAlmostEqual(metrics["parameters_preserved_fraction"], 0.7)
        self.assertAlmostEqual(metrics["parameters_newly_initialized_fraction"], 0.2)
        self.assertAlmostEqual(metrics["parameters_fully_reinitialized_fraction"], 0.1)
        self.assertEqual(metrics["number_added_concepts"], 2)
        self.assertEqual(metrics["number_affected_modules"], 3)
        self.assertEqual(metrics["number_changed_tensors"], 2)
        self.assertFalse(metrics["function_preserving"])
        self.assertEqual(metrics["fallback_count"], 1)


class PlotAuditTests(unittest.TestCase):
    def test_controlled_design_audit_checks_all_methods_and_client_losses(self):
        methods = ("full_reinit", "partial_random", "partial_zero")
        round_rows = []
        client_rows = []
        for method in methods:
            for round_number, loss in ((1, 0.8), (2, 0.7), (3, 0.9)):
                common = {
                    "dataset": "asia",
                    "architecture": "cgm",
                    "split_group": "10_50_to_100",
                    "method": method,
                    "seed": 1,
                    "drift_round": 3,
                    "round": round_number,
                }
                round_rows.append(
                    {
                        **common,
                        "client_mean_loss": loss,
                        "client_std_loss": 0.0,
                        "n_clients": 1,
                    }
                )
                client_rows.append(
                    {**common, "client_id": 11, "task_loss": loss}
                )

        result = audit_controlled_design(
            pd.DataFrame(client_rows),
            pd.DataFrame(round_rows),
            tolerance=1e-8,
        )
        self.assertEqual(result.iloc[0]["status"], "pass")

        client_rows[0]["task_loss"] += 0.01
        result = audit_controlled_design(
            pd.DataFrame(client_rows),
            pd.DataFrame(round_rows),
            tolerance=1e-8,
        )
        self.assertEqual(result.iloc[0]["status"], "fail")
        self.assertIn("pre_shift_client_loss_mismatch", result.iloc[0]["reasons"])

    def test_reviewer_table_exposes_recovery_censoring(self):
        summary = pd.DataFrame(
            [
                {
                    "method": "partial_zero",
                    "n_seeds": 2,
                    "recovered_mean": 0.5,
                    "recovered_std": 0.7071,
                    "recovered_n": 2,
                    "recovery_time_mean": 3.0,
                    "recovery_time_std": 0.0,
                    "recovery_time_n": 1,
                }
            ]
        )
        table = _reviewer_loss_table(summary)
        self.assertEqual(table.iloc[0]["recovery_rate"], "50.0% (1/2)")
        self.assertIn("(n=1/2)", table.iloc[0]["recovery_time"])


class RecorderTests(unittest.TestCase):
    def test_recorder_writes_tidy_round_and_seed_files(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "architecture_ablation"
            cfg = {
                "seed": 7,
                "dataset": {"name": "asia"},
                "model": {"name": "cgm"},
                "architecture_ablation": {
                    "pre_shift_window": 2,
                    "post_peak_window": 2,
                    "smoothing_window": 1,
                    "stability_window": 1,
                    "final_window": 2,
                    "post_shift_horizon": None,
                },
                "learning": {
                    "subgraphs": {
                        "rnd_drift": 3,
                        "client_selection_mode_predrift": "first_no_add",
                        "architecture_update_strategy": "partial_zero",
                    }
                },
            }
            recorder = ArchitectureAblationRecorder(
                cfg,
                output_dir=output_dir,
            )
            recorder.record_round(1, {11: 0.8, 12: 1.0})
            recorder.record_round(
                2,
                [
                    {"client_id": 11, "task_loss": 0.6, "n_labeled": 4},
                    {"client_id": 12, "task_loss": float("nan"), "reason": "missing"},
                ],
            )
            recorder.record_round(3, {11: 1.2, 12: 0.8})
            recorder.record_round(4, {11: 0.5, 12: 0.7})
            recorder.record_shift_snapshot(
                "post_migration_pre_local_training",
                3,
                {11: 1.4, 12: 1.0},
            )
            recorder.set_structural_metadata(
                {
                    "parameters_total": 20,
                    "parameters_preserved": 15,
                    "parameters_newly_initialized": 5,
                    "parameters_fully_reinitialized": 0,
                    "added_concepts": ["either"],
                    "affected_modules": ["dysp"],
                    "changed_tensors": ["node_mlps.dysp.0.weight"],
                }
            )
            metrics = recorder.finalize()

            expected_files = {
                "config.json",
                "experiment.json",
                "validation_client_losses.csv",
                "validation_round_summary.csv",
                "shift_snapshot_client_losses.csv",
                "shift_snapshot_summary.csv",
                "structural_change.json",
                "seed_metrics.csv",
                "seed_metrics.json",
            }
            self.assertEqual(
                expected_files,
                {path.name for path in output_dir.iterdir()},
            )

            with (output_dir / "validation_client_losses.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                client_rows = list(csv.DictReader(handle))
            self.assertEqual(len(client_rows), 8)
            self.assertEqual(client_rows[0]["dataset"], "asia")
            self.assertEqual(client_rows[0]["method"], "partial_zero")

            with (output_dir / "validation_round_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                round_rows = list(csv.DictReader(handle))
            self.assertEqual(len(round_rows), 4)
            self.assertAlmostEqual(float(round_rows[0]["mean_task_loss"]), 0.9)
            self.assertAlmostEqual(float(round_rows[0]["std_task_loss"]), 0.1)
            self.assertEqual(round_rows[1]["n_evaluated_clients"], "1")

            with (output_dir / "seed_metrics.json").open(encoding="utf-8") as handle:
                saved_metrics = json.load(handle)
            self.assertEqual(saved_metrics["number_added_concepts"], 1)
            self.assertAlmostEqual(saved_metrics["parameters_preserved_fraction"], 0.75)
            self.assertEqual(saved_metrics["post_shift_peak_round"], 3)
            self.assertEqual(saved_metrics["split_group"], "10_50_to_100")
            self.assertAlmostEqual(saved_metrics["immediate_post_migration_loss"], 1.2)
            self.assertAlmostEqual(saved_metrics["immediate_loss_spike"], 0.45)
            self.assertEqual(saved_metrics["immediate_post_migration_client_count"], 2)
            self.assertEqual(metrics["dataset"], "asia")

            # JSON uses null rather than non-standard NaN tokens.
            all_json = (output_dir / "seed_metrics.json").read_text(encoding="utf-8")
            self.assertNotIn("NaN", all_json)


if __name__ == "__main__":
    unittest.main()
