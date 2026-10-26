import copy
import importlib
import itertools
import random
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.generate_split import CustomDataset, build_client_subgraph_ids, generate_split
from src.utils import (
    build_local_graphs,
    dataprocess_auditing,
    filter_dataloaders_by_concepts,
    load_dataloaders,
    update_config_from_data_subgroup_clients,
)


FLAGS = [False, True, False, True, False]
EXPECTED = {
    "first_no_add": [0] * 10,
    "all_no_add": [0, 2, 4, 0, 2] * 2,
    "all": [1, 3, 1, 3, 1, 0, 2, 4, 0, 2],
}
CLIENTS = list(range(11, 21))


class FixedClientAssignmentTests(unittest.TestCase):
    def test_every_mode_uses_the_all_cohort_and_repeats_its_original_subgraphs(self):
        for mode, expected in EXPECTED.items():
            with self.subTest(mode=mode):
                count, assignments, clients = build_client_subgraph_ids(
                    10, FLAGS, dataset_client_multiplier=2, client_selection_mode=mode,
                )
                self.assertEqual(count, 20)
                self.assertEqual(clients, CLIENTS)
                self.assertEqual(assignments[10:], expected)
                self.assertEqual(assignments[:10], [0, 2, 4, 0, 2, 4, 0, 2, 4, 0])

    def test_all_phase_pairs_keep_the_same_clients_and_independent_assignments(self):
        for pre, post in itertools.product(EXPECTED, repeat=2):
            with self.subTest(pre=pre, post=post):
                count, pre_ids, pre_clients, post_clients = build_client_subgraph_ids(
                    10, FLAGS, dataset_client_multiplier=2,
                    client_selection_mode=pre, client_selection_mode_postdrift=post,
                )
                self.assertEqual(count, 20)
                self.assertEqual(pre_clients, CLIENTS)
                self.assertEqual(post_clients, CLIENTS)
                self.assertEqual(pre_ids[10:], EXPECTED[pre])
                _, post_ids, _ = build_client_subgraph_ids(
                    10, FLAGS, dataset_client_multiplier=2, client_selection_mode=post,
                )
                self.assertEqual(post_ids[10:], EXPECTED[post])

    def test_all_cohort_is_computed_from_saved_population(self):
        count, _, clients = build_client_subgraph_ids(
            8, FLAGS, dataset_client_multiplier=2, client_selection_mode="first_no_add",
        )
        self.assertEqual(count, 16)
        self.assertEqual(clients, list(range(7, 17)))


class PhaseAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        split_module = importlib.import_module("src.data.generate_split")
        self.patches = [patch.object(split_module, "CACHE", self.cache),
                        patch("src.utils.CACHE", self.cache)]
        for mock in self.patches:
            mock.start()
            self.addCleanup(mock.stop)
        self.names = ["a", "b", "c", "d", "e"]
        self.graph = pd.DataFrame(0, index=self.names + ["y"], columns=self.names + ["y"])
        self.graph.loc["a", "b"] = 1
        self.graph.loc["b", "y"] = 1
        self.graph.loc["d", "y"] = 1
        self.source_c = torch.arange(300).reshape(60, 5) % 3
        self.source_y = torch.arange(60) % 2
        data = {}
        for split in ("train", "val", "test"):
            data[split] = CustomDataset(
                torch.arange(120).reshape(60, 2).float(), self.source_c.clone(),
                self.source_y.clone(), self.graph,
            )
            data[split].X = data[split].x
        self.dataset = SimpleNamespace(
            data=data, c_info={"names": self.names, "cardinality": [3] * 5},
            y_info={"names": ["y"], "cardinality": [2]},
        )

    def prepare(self, pre="first_no_add", post="all", drift_round=10, shared_test=False):
        cfg = OmegaConf.create({
            "seed": 1,
            "dataset": {"name": "synthetic", "batch_size": 2, "per_client_testset": not shared_test,
                        "per_client_reduce_fraction": 0.75},
            "model": {"name": "cbm_mlp"}, "engine": {"model": "${model}"},
            "learning": {
                "mode": "local_federated", "n_clients": 10,
                "annotation_assumption": "task_included",
                "settings": {"canary_fraction": 0.5, "sia_samples_per_client": 1,
                             "sia": True, "n_canaries": -1, "n_rounds": 60},
                "subgraphs": {
                    "dataset_client_multiplier": 2, "drift_add_nodes_ratio": 0.5,
                    "client_selection_mode_predrift": pre,
                    "client_selection_mode_postdrift": post, "rnd_drift": drift_round,
                    "dict_subgraph_with_add_nodes": {"number_add_nodes": 2},
                    "structural_concept_coverage_ranges": {
                        "first_no_add": [0.1, 0.5], "all_no_add": [0.51, 0.75], "all": [1, 1],
                    },
                    "aggregate_graph": {"local_graphs": "from_true_graph",
                                        "perc_clients_alterations": 0, "graph_alteration_prob": 0},
                },
            },
        })
        subgraphs = {"subgraph_1": [0, 1, 5], "subgraph_2": [0, 1, 3, 4, 5],
                     "subgraph_3": [1, 2], "subgraph_4": [2, 3, 4], "subgraph_5": [2, 5]}
        concept_names = {key: [self.graph.columns[i] for i in values]
                         for key, values in subgraphs.items()}
        split_module = importlib.import_module("src.data.generate_split")
        split_module._validate_structural_concept_coverage_ranges(cfg, self.graph, 5, subgraphs, FLAGS)
        _, names, _, _, _ = generate_split(
            cfg, [self.dataset], self.graph, 5,
            precomputed_subgraphs=(copy.deepcopy(subgraphs), concept_names, FLAGS, [3, 4], ["d", "e"]),
        )
        root = str(self.cache / "synthetic" / "task_included")
        post_root = cfg.learning.subgraphs.get("postdrift_cache_path", root)
        loaders = {"predrift": load_dataloaders(cfg, root, 20),
                   "postdrift": load_dataloaders(cfg, post_root, 20)}
        phase_cfgs = {}
        for phase in ("predrift", "postdrift"):
            phase_cfgs[phase] = update_config_from_data_subgroup_clients(
                cfg, CLIENTS, [self.dataset], names, self.names + ["y"],
                cache_path=root if phase == "predrift" else post_root,
            )
        return cfg, names, root, loaders, phase_cfgs

    def test_phase_masks_preserve_samples_after_auditing_and_recover_new_annotations(self):
        cfg, names, root, loaders, phase_cfgs = self.prepare()
        views = {}
        reference = None
        for phase, phase_cfg in phase_cfgs.items():
            torch.manual_seed(4)
            random.seed(4)
            audited, _, in_out, _ = dataprocess_auditing(loaders[phase][0], 20, cfg)
            if reference is None:
                reference, reference_in_out = audited, in_out
                torch_state, random_state = torch.get_rng_state(), random.getstate()
            else:
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
                self.assertEqual(random.getstate(), random_state)
                for actual, expected in zip(in_out, reference_in_out):
                    torch.testing.assert_close(actual, expected)
            phase_path = phase_cfg.learning.subgraphs.cache_path
            views[phase] = filter_dataloaders_by_concepts(
                list(audited), phase_cfg, CLIENTS, names, self.names, phase_path,
            )
            mode = "first_no_add" if phase == "predrift" else "all"
            for client_id in CLIENTS:
                index = client_id - 1
                expected_subgraph = EXPECTED[mode][client_id - 11] + 1
                visible = names[f"subgraph_{expected_subgraph}"]
                self.assertEqual(list(phase_cfg.engine.c_names_id[client_id]), visible)
                view = views[phase][index].dataset
                self.assertEqual(len(view), len(audited[index].dataset))
                for sample_index in range(len(view)):
                    sample = view[sample_index]
                    original = reference[index].dataset[sample_index]
                    torch.testing.assert_close(sample["x"], original["x"])
                    row = int(sample["x"][0]) // 2
                    expected_c = torch.full((5,), -1, dtype=torch.long)
                    for name in visible:
                        col = self.names.index(name)
                        expected_c[col] = self.source_c[row, col]
                    torch.testing.assert_close(sample["c"], expected_c)
                    expected_y = -1 if expected_subgraph in (3, 4) else self.source_y[row].item()
                    self.assertEqual(sample["y"].item(), expected_y)
            graphs, _ = build_local_graphs(CLIENTS, phase_cfg, views[phase], "y", self.graph)
            for client_id, graph in graphs.items():
                self.assertEqual(set(graph.columns), set(phase_cfg.engine.c_names_id[client_id]) | {"y"})
        # Client 12 changes both concept availability and task availability.
        self.assertTrue((views["predrift"][11].dataset.c[:, :2] != -1).all())
        self.assertTrue((next(iter(views["predrift"][11]))["y"] != -1).all())
        self.assertTrue((views["postdrift"][11].dataset.c[:, :2] == -1).all())
        self.assertTrue((next(iter(views["postdrift"][11]))["y"] == -1).all())

    def test_validation_and_test_use_their_own_phase_annotations(self):
        for pre, post in (("first_no_add", "all_no_add"), ("all_no_add", "all")):
            cfg, names, root, loaders, phase_cfgs = self.prepare(pre, post)
            self.assertEqual(list(cfg.learning.subgraphs.predrift_client_ids), CLIENTS)
            self.assertEqual(list(cfg.learning.subgraphs.postdrift_client_ids), CLIENTS)
            for phase, phase_cfg in phase_cfgs.items():
                for original in loaders[phase][1:]:
                    views = filter_dataloaders_by_concepts(
                        list(original), phase_cfg, CLIENTS, names, self.names,
                        phase_cfg.learning.subgraphs.cache_path,
                    )
                    for client_id in CLIENTS:
                        batch = next(iter(views[client_id - 1]))
                        self.assertEqual(batch["c"].shape[1], len(phase_cfg.model.c_info.names))
                        self.assertEqual(len(views[client_id - 1].dataset), len(original[client_id - 1].dataset))

    def test_shared_test_set_uses_phase_union_without_client_specific_masks(self):
        _, names, root, loaders, phase_cfgs = self.prepare(shared_test=True)
        for phase, count in (("predrift", 2), ("postdrift", 5)):
            views = filter_dataloaders_by_concepts(
                list(loaders[phase][2]), phase_cfgs[phase], CLIENTS, names, self.names,
                phase_cfgs[phase].learning.subgraphs.cache_path, loader_client_ids=[1],
            )
            batch = next(iter(views[0]))
            self.assertEqual(batch["c"].shape[1], count)
            self.assertTrue((batch["c"] != -1).all())

    def test_static_runs_assign_subgraphs_before_the_original_save_function(self):
        for drift_round, active_phase, expected_count in ((1000, "predrift", 2), (1, "postdrift", 5)):
            cfg, _, root, _, phase_cfgs = self.prepare(drift_round=drift_round)
            self.assertNotIn("postdrift_cache_path", cfg.learning.subgraphs)
            self.assertTrue(all(path.is_file() for path in Path(root).iterdir()))
            self.assertEqual(len(phase_cfgs[active_phase].model.c_info.names), expected_count)


if __name__ == "__main__":
    unittest.main()
