import unittest

import torch

from src.data.concept_perturbation import (
    ConceptPerturbationSpec,
    concept_perturbation_seed,
    perturb_concept_annotations,
)


class ConceptPerturbationTests(unittest.TestCase):
    def setUp(self):
        self.annotations = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [1.0, 1.0, -1.0],
                [0.0, 2.0, -1.0],
                [1.0, 0.0, -1.0],
            ]
        )
        self.cardinalities = [2, 3, 2]

    def test_missing_masks_only_available_entries(self):
        output, report = perturb_concept_annotations(
            self.annotations,
            self.cardinalities,
            ConceptPerturbationSpec(mode="missing", probability=1.0),
            seed=123,
        )
        self.assertTrue(output[:, :2].eq(-1).all())
        self.assertTrue(output[:, 2].eq(-1).all())
        self.assertEqual(report["eligible"], 8)
        self.assertEqual(report["perturbed"], 8)

    def test_noise_always_selects_a_different_valid_class(self):
        output, report = perturb_concept_annotations(
            self.annotations,
            self.cardinalities,
            ConceptPerturbationSpec(mode="noise", probability=1.0),
            seed=123,
        )
        self.assertTrue(output[:, 0].ne(self.annotations[:, 0]).all())
        self.assertTrue(output[:, 1].ne(self.annotations[:, 1]).all())
        self.assertTrue(output[:, 0].ge(0).all() and output[:, 0].lt(2).all())
        self.assertTrue(output[:, 1].ge(0).all() and output[:, 1].lt(3).all())
        self.assertTrue(output[:, 2].eq(-1).all())
        self.assertEqual(report["perturbed"], 8)

    def test_same_seed_is_deterministic_and_probabilities_are_nested(self):
        low_spec = ConceptPerturbationSpec(mode="missing", probability=0.3)
        high_spec = ConceptPerturbationSpec(mode="missing", probability=0.9)
        low, _ = perturb_concept_annotations(
            self.annotations, self.cardinalities, low_spec, seed=456
        )
        low_repeat, _ = perturb_concept_annotations(
            self.annotations, self.cardinalities, low_spec, seed=456
        )
        high, _ = perturb_concept_annotations(
            self.annotations, self.cardinalities, high_spec, seed=456
        )
        self.assertTrue(torch.equal(low, low_repeat))
        newly_missing_low = low.eq(-1) & self.annotations.ne(-1)
        newly_missing_high = high.eq(-1) & self.annotations.ne(-1)
        self.assertTrue((newly_missing_low & ~newly_missing_high).sum().eq(0))

    def test_client_seeds_are_stable_and_distinct(self):
        first = concept_perturbation_seed(3, 1, 17_041)
        self.assertEqual(first, concept_perturbation_seed(3, 1, 17_041))
        self.assertNotEqual(first, concept_perturbation_seed(3, 2, 17_041))


if __name__ == "__main__":
    unittest.main()
