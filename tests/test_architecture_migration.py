"""Standard-library tests for structural parameter migration.

Run with::

    python -m unittest tests.test_architecture_migration
"""

from __future__ import annotations

from collections import OrderedDict
import json
import unittest

import torch

from src.architecture_migration import (
    migrate_architecture_state,
    migrate_identity_axis_blocks,
)


class _MetadataModel:
    """Small model-shaped object that avoids importing the training stack."""

    def __init__(
        self,
        *,
        name,
        names,
        cardinalities,
        edges,
        roots,
        state,
        concept_hidden_size,
        task="y",
        prop_type=None,
    ):
        self.name = name
        self.combo_info = {"names": list(names), "cardinality": list(cardinalities)}
        self.c_info = {
            "names": [node for node in names if node != task],
            "cardinality": [
                cardinalities[index]
                for index, node in enumerate(names)
                if node != task
            ],
        }
        self.y_info = {
            "names": [task],
            "cardinality": [cardinalities[list(names).index(task)]],
        }
        self.roots_info = {"names": list(roots)}
        self.concept_hidden_size = concept_hidden_size
        if prop_type is not None:
            self.prop_type = prop_type

        index = {node: position for position, node in enumerate(names)}
        self.graph = torch.zeros(len(names), len(names), dtype=torch.int64)
        for parent, child in edges:
            self.graph[index[parent], index[child]] = 1
        self._state = OrderedDict((key, value.clone()) for key, value in state.items())

    def state_dict(self):
        return OrderedDict((key, value.clone()) for key, value in self._state.items())

    def named_parameters(self):
        # The migration code needs names for accounting, not live Parameters.
        return iter(self._state.items())


class IdentityAxisMigrationTest(unittest.TestCase):
    def test_input_expansion_random_and_zero(self):
        old = torch.arange(30, dtype=torch.float32).reshape(3, 10)
        destination = torch.full((3, 12), 17.0)
        old_ids = list(range(10))
        new_ids = list(range(12))

        random_result, random_report = migrate_identity_axis_blocks(
            old,
            destination,
            old_identities=old_ids,
            new_identities=new_ids,
            axis=1,
            strategy="partial_random",
        )
        torch.testing.assert_close(random_result[:, :10], old)
        torch.testing.assert_close(random_result[:, 10:], destination[:, 10:])
        self.assertEqual(random_report["preserved_elements"], 30)
        self.assertEqual(random_report["newly_initialized_elements"], 6)
        self.assertEqual(random_report["fully_reinitialized_elements"], 0)
        self.assertEqual(random_report["zeroed_elements"], 0)

        zero_result, zero_report = migrate_identity_axis_blocks(
            old,
            destination,
            old_identities=old_ids,
            new_identities=new_ids,
            axis=1,
            strategy="partial_zero",
        )
        torch.testing.assert_close(zero_result[:, :10], old)
        torch.testing.assert_close(zero_result[:, 10:], torch.zeros(3, 2))
        self.assertEqual(zero_report["zeroed_elements"], 6)

    def test_reorder_uses_identity_not_position(self):
        old = torch.tensor([[1.0, 2.0, 3.0]])
        destination = torch.full((1, 4), -5.0)
        result, report = migrate_identity_axis_blocks(
            old,
            destination,
            old_identities=["a", "b", "c"],
            new_identities=["c", "new", "a", "b"],
            axis=1,
            strategy="partial_random",
        )
        torch.testing.assert_close(result, torch.tensor([[3.0, -5.0, 1.0, 2.0]]))
        self.assertEqual(report["copied_blocks"], ["c", "a", "b"])
        self.assertEqual(report["new_blocks"], ["new"])


class CGMMigrationTest(unittest.TestCase):
    key = "concept_encoders.y.c_encoder.mlp.0.affinity.weight"

    @staticmethod
    def _models(destination_value=-7.0):
        # Old parent order: a, b. New graph-label/parent order: b, c, a.
        old_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        new_weight = torch.full((4, 6), destination_value)
        old_model = _MetadataModel(
            name="cgm",
            names=["a", "b", "y"],
            cardinalities=[2, 2, 2],
            edges=[("a", "y"), ("b", "y")],
            roots=["a", "b"],
            state={CGMMigrationTest.key: old_weight},
            concept_hidden_size=2,
        )
        new_model = _MetadataModel(
            name="cgm_multi",
            names=["b", "c", "a", "y"],
            cardinalities=[2, 2, 2, 2],
            edges=[("b", "y"), ("c", "y"), ("a", "y")],
            roots=["b", "c", "a"],
            state={CGMMigrationTest.key: new_weight},
            concept_hidden_size=2,
        )
        return old_model, new_model, old_weight, new_weight

    def test_parent_columns_random_and_zero(self):
        old_model, new_model, old_weight, new_weight = self._models()

        random_state, random_meta = migrate_architecture_state(
            old_model.state_dict(), old_model, new_model, strategy="partial_random"
        )
        migrated = random_state[self.key]
        torch.testing.assert_close(migrated[:, 0:2], old_weight[:, 2:4])  # b
        torch.testing.assert_close(migrated[:, 2:4], new_weight[:, 2:4])  # c
        torch.testing.assert_close(migrated[:, 4:6], old_weight[:, 0:2])  # a
        self.assertFalse(random_meta["function_preserving"])
        self.assertIn(
            "random_nonzero_new_parent_blocks",
            random_meta["function_preservation_reasons"],
        )

        zero_state, zero_meta = migrate_architecture_state(
            old_model.state_dict(), old_model, new_model, strategy="partial_zero"
        )
        migrated = zero_state[self.key]
        torch.testing.assert_close(migrated[:, 0:2], old_weight[:, 2:4])
        torch.testing.assert_close(migrated[:, 2:4], torch.zeros(4, 2))
        torch.testing.assert_close(migrated[:, 4:6], old_weight[:, 0:2])
        self.assertTrue(zero_meta["function_preserving"])
        counts = zero_meta["parameter_counts"]
        self.assertEqual(counts["preserved"], 16)
        self.assertEqual(counts["newly_initialized"], 8)
        self.assertEqual(counts["fully_reinitialized"], 0)
        self.assertEqual(counts["total"], 24)
        self.assertEqual(zero_meta["added_concepts"], ["c"])
        self.assertAlmostEqual(
            zero_meta["structural_change"]["realized_pre_concept_coverage"],
            2 / 3,
        )
        self.assertEqual(
            zero_meta["structural_change"]["realized_post_concept_coverage"],
            1.0,
        )
        json.dumps(zero_meta)  # metadata must be persistence-safe

    def test_full_reinit_is_strict_legacy_shape_matching(self):
        old_model, new_model, _, new_weight = self._models()
        state, metadata = migrate_architecture_state(
            old_model.state_dict(), old_model, new_model, strategy="full_reinit"
        )
        torch.testing.assert_close(state[self.key], new_weight)
        self.assertEqual(metadata["parameter_counts"]["fully_reinitialized"], 24)
        self.assertEqual(metadata["parameter_counts"]["preserved"], 0)


class C2BMMigrationTest(unittest.TestCase):
    old_prefix = "propagators.1.y"
    new_prefix = "propagators.2.y"

    @staticmethod
    def _models(destination_value=-11.0):
        # Old rows are [child state][p2(3 states), p1(2 states)].
        # New rows are [child state][p1(2), p3(2), p2(3)].
        old_readout_weight = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        old_readout_bias = torch.arange(10, dtype=torch.float32) + 100
        old_hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        new_readout_weight = torch.full((14, 2), destination_value)
        new_readout_bias = torch.full((14,), destination_value)
        new_hidden = torch.full((2, 4), destination_value)

        old_state = {
            f"{C2BMMigrationTest.old_prefix}.mlp.0.affinity.weight": old_hidden,
            f"{C2BMMigrationTest.old_prefix}.readout.weight": old_readout_weight,
            f"{C2BMMigrationTest.old_prefix}.readout.bias": old_readout_bias,
        }
        new_state = {
            f"{C2BMMigrationTest.new_prefix}.mlp.0.affinity.weight": new_hidden,
            f"{C2BMMigrationTest.new_prefix}.readout.weight": new_readout_weight,
            f"{C2BMMigrationTest.new_prefix}.readout.bias": new_readout_bias,
        }
        old_model = _MetadataModel(
            name="c2bm",
            names=["p2", "p1", "y"],
            cardinalities=[3, 2, 2],
            edges=[("p2", "y"), ("p1", "y")],
            roots=["p2", "p1"],
            state=old_state,
            concept_hidden_size=2,
            prop_type="equations",
        )
        new_model = _MetadataModel(
            name="c2bm_multi",
            names=["p1", "p3", "p2", "y"],
            cardinalities=[2, 2, 3, 2],
            edges=[("p1", "y"), ("p3", "y"), ("p2", "y")],
            roots=["p1", "p3", "p2"],
            state=new_state,
            concept_hidden_size=2,
            prop_type="equations",
        )
        return old_model, new_model, old_state, new_state

    def test_equation_rows_and_level_path_are_mapped_by_identity(self):
        old_model, new_model, old_state, new_state = self._models()
        migrated, metadata = migrate_architecture_state(
            old_model.state_dict(), old_model, new_model, strategy="partial_zero"
        )

        old_weight = old_state[f"{self.old_prefix}.readout.weight"]
        new_weight = migrated[f"{self.new_prefix}.readout.weight"]
        old_bias = old_state[f"{self.old_prefix}.readout.bias"]
        new_bias = migrated[f"{self.new_prefix}.readout.bias"]
        for child_state in range(2):
            old_base = child_state * 5
            new_base = child_state * 7
            # p1 moves from old offset 3 to new offset 0.
            torch.testing.assert_close(
                new_weight[new_base : new_base + 2],
                old_weight[old_base + 3 : old_base + 5],
            )
            torch.testing.assert_close(
                new_bias[new_base : new_base + 2],
                old_bias[old_base + 3 : old_base + 5],
            )
            # p3 is genuinely new and zeroed (both coefficient and bias).
            torch.testing.assert_close(new_weight[new_base + 2 : new_base + 4], torch.zeros(2, 2))
            torch.testing.assert_close(new_bias[new_base + 2 : new_base + 4], torch.zeros(2))
            # p2 moves from old offset 0 to new offset 4.
            torch.testing.assert_close(
                new_weight[new_base + 4 : new_base + 7],
                old_weight[old_base : old_base + 3],
            )
            torch.testing.assert_close(
                new_bias[new_base + 4 : new_base + 7],
                old_bias[old_base : old_base + 3],
            )

        torch.testing.assert_close(
            migrated[f"{self.new_prefix}.mlp.0.affinity.weight"],
            old_state[f"{self.old_prefix}.mlp.0.affinity.weight"],
        )
        self.assertTrue(metadata["function_preserving"])
        self.assertEqual(metadata["parameter_counts"]["fully_reinitialized"], 0)
        self.assertIn(
            f"{self.new_prefix}.mlp.0.affinity.weight",
            metadata["changed_tensors"],
        )
        json.dumps(metadata)

    def test_legacy_does_not_follow_changed_level_path(self):
        old_model, new_model, _, new_state = self._models()
        migrated, metadata = migrate_architecture_state(
            old_model.state_dict(), old_model, new_model, strategy="full_reinit"
        )
        for key, value in new_state.items():
            torch.testing.assert_close(migrated[key], value)
        self.assertEqual(
            metadata["parameter_counts"]["fully_reinitialized"],
            sum(value.numel() for value in new_state.values()),
        )
        self.assertEqual(metadata["parameter_counts"]["newly_initialized"], 0)


if __name__ == "__main__":
    unittest.main()
