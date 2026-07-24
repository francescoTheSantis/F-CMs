import unittest

import numpy as np

from src.utils import aggregate_modulewise


class ModulewiseAggregationTests(unittest.TestCase):
    def test_parameters_use_only_clients_that_updated_them(self):
        results = [
            {
                "parameters": [
                    np.array([2.0]),
                    np.array([20.0]),
                    np.array([4.0]),
                ],
                "num_examples": 1,
                "trainable_keys": {"concept_a.weight"},
                "modality": None,
            },
            {
                "parameters": [
                    np.array([8.0]),
                    np.array([6.0]),
                    np.array([12.0]),
                ],
                "num_examples": 3,
                "trainable_keys": {"concept_b.weight"},
                "modality": None,
            },
        ]

        aggregated = aggregate_modulewise(
            results,
            parameter_keys=[
                "concept_a.weight",
                "concept_b.weight",
                "running_count",
            ],
            reference_parameters=[
                np.array([100.0]),
                np.array([100.0]),
                np.array([100.0]),
            ],
            parameter_names={"concept_a.weight", "concept_b.weight"},
        )

        np.testing.assert_allclose(aggregated[0], [2.0])
        np.testing.assert_allclose(aggregated[1], [6.0])
        np.testing.assert_allclose(aggregated[2], [10.0])

    def test_parameter_without_an_updating_client_keeps_global_value(self):
        aggregated = aggregate_modulewise(
            [
                {
                    "parameters": [np.array([3.0])],
                    "num_examples": 5,
                    "trainable_keys": set(),
                    "modality": None,
                }
            ],
            parameter_keys=["concept_a.weight"],
            reference_parameters=[np.array([11.0])],
            parameter_names={"concept_a.weight"},
        )
        np.testing.assert_allclose(aggregated[0], [11.0])


if __name__ == "__main__":
    unittest.main()
