import unittest

from scripts.summarize_structural_privacy import summarize


class StructuralPrivacySummaryTests(unittest.TestCase):
    def test_changes_are_paired_by_seed_against_infinite_epsilon(self):
        rows = [
            {
                "dataset": "asia",
                "model": "c2bm",
                "seed": 1,
                "epsilon": "inf",
                "task_accuracy": 0.80,
                "graph_disagreement_rate": 0.0,
                "changed_pair_report_rate": 0.0,
            },
            {
                "dataset": "asia",
                "model": "c2bm",
                "seed": 2,
                "epsilon": "inf",
                "task_accuracy": 0.70,
                "graph_disagreement_rate": 0.0,
                "changed_pair_report_rate": 0.0,
            },
            {
                "dataset": "asia",
                "model": "c2bm",
                "seed": 1,
                "epsilon": 4.0,
                "task_accuracy": 0.79,
                "graph_disagreement_rate": 0.10,
                "changed_pair_report_rate": 0.04,
            },
            {
                "dataset": "asia",
                "model": "c2bm",
                "seed": 2,
                "epsilon": 4.0,
                "task_accuracy": 0.68,
                "graph_disagreement_rate": 0.20,
                "changed_pair_report_rate": 0.03,
            },
        ]

        result = summarize(rows)
        finite = next(row for row in result if row["epsilon"] == 4.0)
        self.assertEqual(finite["n_runs"], 2)
        self.assertAlmostEqual(finite["task_accuracy_pct_mean"], 73.5)
        self.assertAlmostEqual(finite["change_from_non_private_pp_mean"], -1.5)
        self.assertEqual(finite["n_paired"], 2)
        self.assertAlmostEqual(finite["graph_disagreement_pct_mean"], 15.0)


if __name__ == "__main__":
    unittest.main()
