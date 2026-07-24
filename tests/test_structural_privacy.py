import math
import unittest

import numpy as np
import pandas as pd

from src.structural_privacy import (
    graph_pair_disagreement,
    privatize_graph_edge_states,
    privatize_local_graphs,
    randomized_response_probabilities,
)


def make_graph() -> pd.DataFrame:
    # a -> b, c -> b, and no edge between a and c
    return pd.DataFrame(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
        dtype=int,
    )


class StructuralPrivacyTests(unittest.TestCase):
    def test_probability_formula_and_exact_privacy_ratio(self):
        epsilon = 1.7
        p, q = randomized_response_probabilities(epsilon)
        self.assertAlmostEqual(p + 2.0 * q, 1.0)
        self.assertAlmostEqual(p / q, math.exp(epsilon))

        # For any two possible inputs and any reported category, the largest
        # likelihood ratio is p/q = exp(epsilon).
        likelihoods = np.array(
            [
                [p, q, q],
                [q, p, q],
                [q, q, p],
            ]
        )
        ratios = likelihoods[:, None, :] / likelihoods[None, :, :]
        self.assertLessEqual(float(ratios.max()), math.exp(epsilon) + 1e-12)

    def test_zero_epsilon_is_uniform(self):
        p, q = randomized_response_probabilities(0.0)
        self.assertAlmostEqual(p, 1.0 / 3.0)
        self.assertAlmostEqual(q, 1.0 / 3.0)

    def test_infinite_epsilon_is_identity(self):
        graph = make_graph()
        private_graph, stats = privatize_graph_edge_states(
            graph,
            epsilon=math.inf,
            rng=np.random.default_rng(8),
        )
        pd.testing.assert_frame_equal(private_graph, graph)
        self.assertEqual(stats["changed_pair_reports"], 0)

    def test_seeded_local_privatization_is_reproducible(self):
        graphs = [make_graph(), make_graph()]
        first, first_report = privatize_local_graphs(graphs, epsilon=1.0, seed=123)
        second, second_report = privatize_local_graphs(graphs, epsilon=1.0, seed=123)

        for first_graph, second_graph in zip(first, second):
            pd.testing.assert_frame_equal(first_graph, second_graph)
        self.assertEqual(first_report, second_report)

    def test_empirical_randomized_response_distribution(self):
        epsilon = 1.2
        p, q = randomized_response_probabilities(epsilon)
        graph = pd.DataFrame(
            [[0, 1], [0, 0]],
            index=["a", "b"],
            columns=["a", "b"],
            dtype=int,
        )
        rng = np.random.default_rng(91)
        counts = np.zeros(3, dtype=int)

        for _ in range(20_000):
            private_graph, _ = privatize_graph_edge_states(graph, epsilon, rng)
            if private_graph.loc["a", "b"] == 1:
                counts[0] += 1
            elif private_graph.loc["b", "a"] == 1:
                counts[1] += 1
            else:
                counts[2] += 1

        frequencies = counts / counts.sum()
        np.testing.assert_allclose(frequencies, [p, q, q], atol=0.012)

    def test_pair_disagreement_counts_reversal_once(self):
        first = make_graph()
        second = make_graph()
        second.loc["a", "b"] = 0
        second.loc["b", "a"] = 1
        result = graph_pair_disagreement(first, second)
        self.assertEqual(result["differing_pairs"], 1)
        self.assertEqual(result["edge_reversals"], 1)
        self.assertAlmostEqual(result["disagreement_rate"], 1.0 / 3.0)

    def test_invalid_bidirectional_pair_is_rejected(self):
        graph = make_graph()
        graph.loc["b", "a"] = 1
        with self.assertRaisesRegex(ValueError, "both directions"):
            privatize_graph_edge_states(graph, epsilon=2.0)


if __name__ == "__main__":
    unittest.main()

