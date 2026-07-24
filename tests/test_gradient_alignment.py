import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.gradient_alignment import (
    _full_pass_gradient,
    compute_gradient_alignment,
    save_gradient_alignment,
)
from src.models.cbm import CBM


class _DictDataset(Dataset):
    def __init__(self, x, c, y):
        self.x = x
        self.c = c
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {
            "x": self.x[index],
            "c": self.c[index],
            "y": self.y[index],
        }


def _model():
    torch.manual_seed(3)
    return CBM(
        input_size=3,
        hidden_size=5,
        concept_hidden_size=4,
        output_size=2,
        n_layers_encoder=1,
        n_layers_concept_encoder=1,
        n_layers_decoder=1,
        concept_loss_weight=0.5,
        c_info={"names": ["c0", "c1"], "cardinality": [2, 2]},
        y_info={"names": ["y"], "cardinality": [2]},
        c_name_index={"c0": 0, "c1": 1, "y": 2},
        name="cbm_mlp",
    )


def _loaders(partial=False):
    generator = torch.Generator().manual_seed(7)
    loaders = []
    for client in range(2):
        x = torch.randn(8, 3, generator=generator) + client * 0.25
        c = torch.randint(0, 2, (8, 2), generator=generator).float()
        if partial and client == 0:
            c[:, 1] = -1
        y = torch.randint(0, 2, (8,), generator=generator)
        loaders.append(
            DataLoader(
                _DictDataset(x, c, y),
                batch_size=4,
                shuffle=False,
            )
        )
    return loaders


class GradientAlignmentTest(unittest.TestCase):
    def test_full_pass_gradient_is_example_weighted_across_uneven_batches(self):
        generator = torch.Generator().manual_seed(17)
        dataset = _DictDataset(
            torch.randn(5, 3, generator=generator),
            torch.randint(0, 2, (5, 2), generator=generator).float(),
            torch.randint(0, 2, (5,), generator=generator),
        )
        model = _model().eval()
        gradient_small_batches, _, loss_small_batches = _full_pass_gradient(
            copy.deepcopy(model),
            DataLoader(dataset, batch_size=2, shuffle=False),
            torch.device("cpu"),
            max_batches=None,
        )
        gradient_single_batch, _, loss_single_batch = _full_pass_gradient(
            copy.deepcopy(model),
            DataLoader(dataset, batch_size=5, shuffle=False),
            torch.device("cpu"),
            max_batches=None,
        )
        self.assertTrue(
            torch.allclose(
                gradient_small_batches,
                gradient_single_batch,
                atol=1e-8,
                rtol=1e-6,
            )
        )
        self.assertAlmostEqual(loss_small_batches, loss_single_batch, places=6)

    def test_exact_alignment_when_all_clients_update_all_coordinates(self):
        record = compute_gradient_alignment(
            global_model=_model(),
            train_dataloaders=_loaders(partial=False),
            client_indices=[0, 1],
            y_to_freeze=[False, False],
            learning_mode="local_federated",
            freezing=True,
            device="cpu",
            round_index=1,
        )
        self.assertLess(record["overall"]["l2_distance"], 1e-10)
        self.assertAlmostEqual(record["overall"]["cosine_similarity"], 1.0, places=10)
        self.assertEqual(record["diagnostics"]["uncovered_coordinates"], 0)

    def test_partial_supervision_produces_finite_mismatch_and_outputs(self):
        record = compute_gradient_alignment(
            global_model=_model(),
            train_dataloaders=_loaders(partial=True),
            client_indices=[0, 1],
            y_to_freeze=[False, False],
            learning_mode="local_federated",
            freezing=True,
            device="cpu",
            round_index=4,
        )
        self.assertGreater(record["overall"]["l2_distance"], 0.0)
        self.assertGreaterEqual(record["overall"]["cosine_similarity"], -1.0)
        self.assertLessEqual(record["overall"]["cosine_similarity"], 1.0)
        self.assertGreaterEqual(record["overall"]["angle_degrees"], 0.0)
        self.assertLessEqual(record["overall"]["angle_degrees"], 180.0)
        self.assertGreaterEqual(
            record["overall"]["scale_adjusted_relative_l2"],
            0.0,
        )
        self.assertIn("concept/c1", record["modules"])

        with tempfile.TemporaryDirectory() as directory:
            summary = save_gradient_alignment([record], output_dir=directory)
            self.assertEqual(summary["n_measured_rounds"], 1)
            summary_path = Path(directory) / "gradient_alignment_summary.json"
            with summary_path.open() as handle:
                saved = json.load(handle)
            self.assertEqual(saved["measured_rounds"], [4])


if __name__ == "__main__":
    unittest.main()
