import unittest
from unittest.mock import Mock

import torch

from src.engines.predictor import _update_concept_metric_if_available


class PredictorInterventionMetricTests(unittest.TestCase):
    def test_missing_root_prediction_is_skipped(self):
        metric = Mock()
        target = torch.tensor([0, 1])

        updated = _update_concept_metric_if_available(
            metric,
            {"predicted_child": torch.tensor([[0.8, 0.2], [0.1, 0.9]])},
            "private_graph_root",
            target,
        )

        self.assertFalse(updated)
        metric.update.assert_not_called()

    def test_available_child_prediction_updates_metric(self):
        metric = Mock()
        prediction = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        target = torch.tensor([0, 1])

        updated = _update_concept_metric_if_available(
            metric,
            {"child": prediction},
            "child",
            target,
        )

        self.assertTrue(updated)
        metric.update.assert_called_once_with(prediction, target)


if __name__ == "__main__":
    unittest.main()
