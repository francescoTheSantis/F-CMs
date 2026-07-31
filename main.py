import itertools
import shutil
import shutil
import sys
import numpy as np
import torch
import os
import warnings
import hydra # type: ignore
import pickle
import networkx as nx
from torch.utils.data import DataLoader
from src.causal_discovery.causal_discovery_block import causal_discovery
from src.completion.completion_block import complete_graph_with_llm
from src.data.utils import static_graph_collate
from pytorch_lightning.loggers import WandbLogger # type: ignore
from src.trainer import Trainer
from src.plots_mia import plot_and_save_max_mia, plot_and_save_max_sia
import subprocess
import json
from typing import Dict, List, Any, Optional, Tuple

from hydra.utils import instantiate, call # type: ignore
from omegaconf import DictConfig, open_dict, OmegaConf # type: ignore

import pickle
import warnings
import time

from src.utils import (
    model_is_causal,
    seed_everything, 
    maybe_freeze_parameters, 
    maybe_make_private,
    update_config_from_data_subgroup_clients,
    aggregate_graph_proposals,
    aggregate, 
    aggregate_multimodal,
    get_parameters, 
    set_parameters, 
    set_old_parameters,
    load_dataloaders,
    identify_subgraph,
    dataprocess_auditing,
    initialize_mia_results,
    plot_training_metrics,
    compute_validation_loss,
    filter_dataloaders_by_concepts,
    build_local_graphs,
    maybe_update_config_with_graph_subgroup_clients,
    _print_concept_availability,
)

from src.dra import (
    run_dra_attack,
    summarize_dra_results,
)
from src.fedcbm import run_fedcbm_baseline
from src.fcl import run_fcl_baseline
from src.metrics import _evaluate_graph_against_truth
from src.architecture_ablation import ArchitectureAblationRecorder

# data loading
from src.data.dataset_block import get_dataset

#from src.server import get_evaluate_fn
#from src.strategy import CustomFedAvgWithModelSaving
from src.utils import clean_empty_configs
from src.data.dataset_block import get_dataset
from src.utils import get_intervention_policy, remove_cycles, remove_problematic_edges, get_split_paths, get_split_paths_fl
from src.utils import clean_empty_configs, update_config_from_data, maybe_update_config_with_graph, update_intervention_policy_and_graph, update_config_from_client
from src.data.utils import update_datasets, construct_combined_true_graph
from src.plots import maybe_plot_graph
from src.my_hydra import parse_hyperparams
from src.data.generate_split import generate_split, get_subgraph_dict, generate_split_with_fallback
from collections import OrderedDict

from env import CACHE

# Suppress specific warning
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")

TEST_RESULT_FILENAMES = [
    "y_accuracy.pkl",
    "c_accuracy.pkl",
    "single_c_interventions_on_y.pkl",
    "single_OODc_interventions_on_y.pkl",
    "single_IDc_interventions_on_y.pkl",
    "level_interventions_on_y.pkl",
    "level_ID_interventions_on_y.pkl",
    "level_OOD_interventions_on_y.pkl",
    "level_interventions_on_c.pkl",
    "level_ID_interventions_on_c_OOD.pkl",
    "level_OOD_interventions_on_c_ID.pkl",
    "cumulative_interventions_on_y.pkl",
    "cumulative_interventions_on_c.pkl",
]


def _is_numeric_scalar(value):
    return isinstance(value, (int, float, np.integer, np.floating))


def _compute_unmasked_validation_task_loss(engine, dataloader, device):
    """Evaluate task loss without exposing validation-only labels to training.

    This lightweight evaluator is used once, immediately after architectural
    migration and before any local optimization.  It mirrors the validation
    forward pass (zero interventions) but avoids attaching another Lightning
    trainer, so it does not modify optimizer state or consume random samples.
    """

    if dataloader is None:
        return float("nan"), 0

    was_training = engine.training
    engine.to(device)
    engine.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in dataloader:
            moved_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            x, c, y, modality = engine._unpack_batch(moved_batch)
            intervention_index = torch.zeros(c.shape, device=device)
            inputs = engine._build_model_inputs(
                x,
                c,
                intervention_index=intervention_index,
                modality=modality,
            )
            y_output, c_output = engine.forward(**inputs)
            y_hat_loss, _ = engine.model.filter_output_for_loss(y_output, c_output)
            y_eval = moved_batch.get("y_eval", y).flatten().long()
            losses = engine.model._compute_task_loss(
                y_hat_loss,
                y_eval,
                reduction="none",
                ignore_index=-1,
            )
            if losses is None:
                continue
            valid = (y_eval != -1) & torch.isfinite(losses)
            total_loss += float(losses[valid].sum().detach().cpu())
            total_count += int(valid.sum().detach().cpu())

    engine.train(was_training)
    if total_count == 0:
        return float("nan"), 0
    return total_loss / total_count, total_count


def _weighted_average_test_artifacts(values, weights):
    non_null = [(value, weight) for value, weight in zip(values, weights) if value is not None]
    if not non_null:
        return None

    sample_value = non_null[0][0]

    if isinstance(sample_value, dict):
        aggregated = {}
        keys = set()
        for value, _ in non_null:
            keys.update(value.keys())
        for key in keys:
            key_values = [value.get(key) for value, _ in non_null]
            aggregated_value = _weighted_average_test_artifacts(key_values, weights)
            if aggregated_value is not None:
                aggregated[key] = aggregated_value
        return aggregated

    if _is_numeric_scalar(sample_value):
        weighted_sum = 0.0
        total_weight = 0.0
        for value, weight in non_null:
            value_float = float(value)
            if np.isnan(value_float):
                continue
            weighted_sum += value_float * weight
            total_weight += weight
        if total_weight == 0:
            return float("nan")
        return weighted_sum / total_weight

    return sample_value


def _collect_test_artifacts(result_dir: str):
    artifacts = {}
    for filename in TEST_RESULT_FILENAMES:
        filepath = os.path.join(result_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as handle:
                artifacts[filename] = pickle.load(handle)
    return artifacts


def _aggregate_and_save_test_artifacts(result_dir: str, per_loader_artifacts, per_loader_weights):
    if not per_loader_artifacts:
        return {}

    aggregated = {}
    for filename in TEST_RESULT_FILENAMES:
        values = [artifacts.get(filename) for artifacts in per_loader_artifacts]
        aggregated_value = _weighted_average_test_artifacts(values, per_loader_weights)
        if aggregated_value is None:
            continue
        aggregated[filename] = aggregated_value
        filepath = os.path.join(result_dir, filename)
        with open(filepath, "wb") as handle:
            pickle.dump(aggregated_value, handle)

    aggregated_summary = {}
    if "y_accuracy.pkl" in aggregated and isinstance(aggregated["y_accuracy.pkl"], dict):
        aggregated_summary["test/y/y_accuracy"] = aggregated["y_accuracy.pkl"].get("_baseline", np.nan)
    if "c_accuracy.pkl" in aggregated and isinstance(aggregated["c_accuracy.pkl"], dict):
        for concept_name, value in aggregated["c_accuracy.pkl"].items():
            aggregated_summary[f"test/c/{concept_name}"] = value

    with open(os.path.join(result_dir, "aggregated_test_metrics.json"), "w") as handle:
        json.dump(aggregated_summary, handle, indent=2)

    return aggregated_summary


def _extract_task_labels(dataset) -> Optional[torch.Tensor]:
    labels = getattr(dataset, "y", None)
    if labels is not None:
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        return labels.detach().cpu().view(-1)

    subset_indices = getattr(dataset, "indices", None)
    subset_parent = getattr(dataset, "dataset", None)
    if subset_indices is not None and subset_parent is not None:
        parent_labels = _extract_task_labels(subset_parent)
        if parent_labels is None:
            return None
        return parent_labels[torch.as_tensor(subset_indices, dtype=torch.long)]

    child_datasets = getattr(dataset, "datasets", None)
    if child_datasets is not None:
        child_labels = []
        for child_dataset in child_datasets:
            child_tensor = _extract_task_labels(child_dataset)
            if child_tensor is None:
                return None
            child_labels.append(child_tensor)
        if child_labels:
            return torch.cat(child_labels)

    return None


def _dataloaders_have_task_labels(dataloaders, client_ids=None):
    """Check whether selected client datasets contain an unmasked task label."""
    if not isinstance(dataloaders, (list, tuple)):
        dataloaders = [dataloaders]
    selected_loaders = dataloaders if client_ids is None else [
        dataloaders[client_id - 1]
        for client_id in client_ids
        if 1 <= client_id <= len(dataloaders)
    ]
    for dataloader in selected_loaders:
        labels = _extract_task_labels(dataloader.dataset)
        if labels is not None and torch.any(labels != -1):
            return True
    return False


def _binary_label_counts(labels: torch.Tensor) -> Tuple[int, int, int]:
    labels = labels.to(torch.float32).view(-1)
    labels = labels[~torch.isnan(labels)]
    labels = labels.round().to(torch.int64)
    class_0 = int((labels == 0).sum().item())
    class_1 = int((labels == 1).sum().item())
    total = class_0 + class_1
    return class_0, class_1, total


def _print_task_label_distribution(
    dataloaders,
    task_name: str,
    split_name: str,
    per_client: bool = True,
) -> None:
    if not dataloaders:
        return

    print(f"\033[94mTask label distribution for {split_name} split ({task_name}):\033[0m")
    overall_class_0 = 0
    overall_class_1 = 0

    for client_idx, loader in enumerate(dataloaders, start=1):
        labels = _extract_task_labels(loader.dataset)
        if labels is None:
            if per_client:
                print(f"\033[94mClient {client_idx}: unable to read task labels\033[0m")
            continue

        class_0, class_1, total = _binary_label_counts(labels)
        overall_class_0 += class_0
        overall_class_1 += class_1

        if per_client:
            class_0_pct = (100.0 * class_0 / total) if total > 0 else 0.0
            class_1_pct = (100.0 * class_1 / total) if total > 0 else 0.0
            print(
                f"\033[94mClient {client_idx}: class 0 = {class_0} ({class_0_pct:.2f}%), "
                f"class 1 = {class_1} ({class_1_pct:.2f}%), total = {total}\033[0m"
            )

    overall_total = overall_class_0 + overall_class_1
    overall_class_0_pct = (100.0 * overall_class_0 / overall_total) if overall_total > 0 else 0.0
    overall_class_1_pct = (100.0 * overall_class_1 / overall_total) if overall_total > 0 else 0.0
    print(
        f"\033[94mOverall {split_name}: class 0 = {overall_class_0} ({overall_class_0_pct:.2f}%), "
        f"class 1 = {overall_class_1} ({overall_class_1_pct:.2f}%), total = {overall_total}\033[0m"
    )



@hydra.main(config_path="conf", config_name="test", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # various preliminaries, it set the seed for reproducibility
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    os.makedirs('results', exist_ok=True)
    if torch.cuda.is_available():
        device = f"cuda:{cfg.trainer.devices[0]}" 
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    with open_dict(cfg): cfg.update(device=device)
    print(f"Using {cfg.device} device")

    # adjust config
    cfg = clean_empty_configs(cfg)

    # instantiate the dataset, split into train, val, test
    # preprocess all of them and save the preprocessed dataset
    dataset, true_graph, dataset_directory = get_dataset(cfg.dataset, cfg.device, seed=cfg.seed)
    


    combined_dataset = OmegaConf.select(cfg, 'combined_datasets.other_datasets', default=None)
    if combined_dataset is not None:
        if cfg.learning.mode == "centralized":
            raise ValueError("Combined datasets are not supported in centralized learning mode.")

        if cfg.dataset.name not in ['colormnist', 'fashionmnist']:
            raise ValueError("Combined datasets are only supported for the 'colormnist' and 'fashionmnist' datasets.")

        datasets = {} 
        datasets[0] = dataset
        base_conf_path = hydra.utils.get_original_cwd() + "/conf/dataset"

        for i, ds_name in enumerate(cfg.combined_datasets.other_datasets):
            if ds_name not in ['colormnist', 'fashionmnist']:
                raise ValueError(f"Dataset {ds_name} is not supported.")
            if ds_name == cfg.dataset.name:
                warnings.warn(f"Dataset {ds_name} is the same as the original dataset. Skipping it.")
                continue
            ds_path = os.path.join(base_conf_path, f"{ds_name}.yaml")
            original_cfg = OmegaConf.load(ds_path)
            overrides = cfg.combined_datasets.get(ds_name, {})
            merged_cfg = OmegaConf.merge(original_cfg, overrides)
            new_dataset, _, _ = get_dataset(merged_cfg, cfg.device, seed=cfg.seed)

            # consistency checks
            # check at least two variables are in common
            original_variables = dataset.c_info['names']+dataset.y_info['names']
            new_variables = new_dataset.c_info['names']+new_dataset.y_info['names']
            common_variables = set(original_variables) & set(new_variables)
            if len(common_variables) < 2:
                raise ValueError(f"Dataset {ds_name} does not have at least two variables in common with the original dataset.")
            else:
                datasets = update_datasets(datasets, new_dataset, cfg.combined_datasets)
                graph = construct_combined_true_graph(datasets, cfg.dataset, cfg.combined_datasets)
            
    else:
        datasets = {0: dataset}
    
    print(OmegaConf.to_yaml(cfg))
    

    if cfg.dataset.load_graph:
        try:
            if true_graph is None or cfg.dataset.load_true_graph == False:
                with open(os.path.join(dataset_directory, "learned_graph.pkl"), 'rb') as f:
                    graph = pickle.load(f)
                    
            else:
                graph = true_graph
        except FileNotFoundError:
            print("Graph file not found. Change the config or run graph learning.")
    else:
        # graph construction
        if len(datasets)>1:
            raise NotImplementedError("Multiple datasets are not supported in the current version for graph construction.")
        else:
            if true_graph is None or cfg.dataset.load_true_graph == False:
                # estimate causal graph with causal structural learning algorithms
                graph = causal_discovery(cfg, dataset, true_graph, save_file_name="graph_causal_discovery.pkl")
                #if true_graph is not None:
                #    hamming = hamming_distance(true_graph, predicted_graph)
                #    print('(after CD) structural hamming distance: ', hamming)    

                # complete the causal graph with LLM and RAG
                graph = complete_graph_with_llm(cfg, graph, cfg.dataset.name)
                

                #if true_graph is not None:
                #    hamming = hamming_distance(true_graph, completed_graph)
                #     print('(after LLM + RAG) structural hamming distance: ', hamming)
                graph, dataset = remove_problematic_edges(graph, dataset)
                y_index = graph.columns.get_loc(datasets[0].y_info['names'][0])
                graph = remove_cycles(graph, y_index)
                #graph = completed_graph
                with open(os.path.join(dataset_directory, "learned_graph.pkl"), 'wb') as f:
                    pickle.dump(graph, f)
            else:
                graph = true_graph

    # interv graph must be always the true graph if available
    if true_graph is not None:
        interv_graph = true_graph.copy()
    else:
        interv_graph = graph.copy()
    centralized_c_dict = {name: idx for idx, name in enumerate(datasets[0].c_info['names'])}
            
    # get the causal graph
    #if cfg.dataset.load_true_graph:
    #    graph = true_graph
    #else:
    #    if cfg.dataset.load_graph:
    #        with open(os.path.join(dataset_directory, "graph.pkl"), 'rb') as f:
    #            graph = pickle.load(f)
    #    else:
    #        # estimate causal graph with causal structural learning algorithms
    #        predicted_graph = causal_discovery(cfg, dataset, true_graph)
    #        #if true_graph is not None:
    #        #    hamming = hamming_distance(true_graph, predicted_graph)
    #        #    print('(after CD) structural hamming distance: ', hamming)    

    #        # complete the causal graph with LLM and RAG
    #        completed_graph = complete_graph_with_llm(cfg, predicted_graph, cfg.dataset.name)
    #        #if true_graph is not None:
    #        #    hamming = hamming_distance(true_graph, completed_graph)
    #        #     print('(after LLM + RAG) structural hamming distance: ', hamming)
    #        graph = completed_graph
    #
    #        # save graph
    #        with open(os.path.join(dataset_directory, "graph.pkl"), 'wb') as f:
    #            pickle.dump(graph, f)

    # fix the graph
    # (part 1): remove bidirected and undirected edges + add virtual nodes
    # edge can only be directed at this stage, the following function is just here in 
    # case the CD + LLM + RAG pipeline is modified and could produce bidirected or undirected edges
    #graph, dataset = remove_problematic_edges(graph, dataset)
    y_index = graph.columns.get_loc(datasets[0].y_info['names'][0])
    maybe_plot_graph(graph, 'graph')
    if cfg.dataset.name in ["siim_pneumothorax", "skincon", "cheXpert"] and cfg.learning.mode == "localized":
            true_graph = graph

      # it is ok also for multimodal because c_info and y_info contain all the variables of the datasets
        # insert
    #y_index = len(graph)-1
    
    ## (part 2): remove cycles
    #graph = remove_cycles(graph, y_index)

    #if true_graph is not None:
        #hamming = hamming_distance(true_graph, graph)
        #print('(after fix) structural hamming distance: ', hamming)
    
    #partitions = get_partitions(graph, y_index, n_clients = 5)
    #col_to_index = {col: idx for idx, col in enumerate(graph.columns)}    
    #partitions_named = {
    #    term: [graph.columns[i] for i in indices]
    #    for term, indices in partitions.items()
    #}

    # use the graph to define an intervention policy at test time
    interv_policy, ip_names = get_intervention_policy(interv_graph, y_index)
    print('intervention policy:', interv_policy)
    print('intervention policy names:', ip_names)

    for i, dataset in datasets.items():
        [dataset.data[split].register_graph(graph) for split in dataset.data]

    # We split the data by selecting a sub-graph for each split
    if cfg.learning.mode == "centralized":
        subgraphs, subgraphs_concept_names, subgraphs_with_add_nodes, add_nodes_values, add_nodes_names = None, None, None, None, None
    else:
        # seed_everything(cfg.get("seed"))
        # subgraphs, subgraphs_concept_names, subgraphs_with_add_nodes, add_nodes_values, add_nodes_names = generate_split(cfg, datasets, graph, y_index)
        if cfg.learning.get("seed_plot_interventions") is not None:
            seed_everything(cfg.learning.get("seed_plot_interventions"))
        subgraphs, subgraphs_concept_names, subgraphs_with_add_nodes, add_nodes_values, add_nodes_names = \
            generate_split_with_fallback(
                cfg, datasets, graph, y_index,
                seed_everything_fn=seed_everything,  # <-- pass your seeding function
                step=100,
                max_tries=100,
            )  
        if cfg.learning.get("seed_plot_interventions") is not None:
            seed_everything(cfg.get("seed")) 
                 
        print("seed_plot_interventions:", cfg.learning.get("seed_plot_interventions"))
        
        ## Save subgraphs_concept_names, add_nodes_values, and subgraphs_with_add_nodes
        #model_name = cfg.model._target_.split('.')[-1] if hasattr(cfg.model, '_target_') else 'model'
        #subgraphs_file = f"subgraphs_{cfg.dataset.name}_{model_name}_seed_{cfg.seed}.json"
        #with open(subgraphs_file, 'w') as f:
        #    json.dump({
        #        "subgraphs_concept_names": subgraphs_concept_names,
        #        "add_nodes_names": add_nodes_names,
        #        "subgraphs_with_add_nodes": subgraphs_with_add_nodes,
        #        "dataset_name": cfg.dataset.name,
        #        "model_type": model_name
        #    }, f, indent=2)
        #print(f"Saved subgraphs data to {subgraphs_file}")

    # update config based on the dataset
    # e.g., set input and output size of the model
    cfg = update_config_from_data(cfg, datasets, subgraphs_concept_names)
    if cfg.learning.mode == 'localized':
        interv_policy, graph = update_intervention_policy_and_graph(cfg, interv_policy, graph, datasets)
        maybe_plot_graph(graph, 'graph_localized')

    cfg = maybe_update_config_with_graph(cfg, graph, interv_policy)
    # check consistency
    if len(datasets) > 1:
        raise NotImplementedError("Multiple datasets are not supported in the current version for training.")
    else:
        combo_info = {'names': datasets[0].c_info['names'] + datasets[0].y_info['names'],
                        'cardinality': datasets[0].c_info['cardinality'] + datasets[0].y_info['cardinality']}
        assert list(cfg.engine.model.c_name_index.keys()) == combo_info['names'], "Concept names are ordered differently in the engine configuration and in the dataset one."

    ############ data block ########################################################################################
    #if combined_dataset is None:
        # Load the unique test-set (if not combined dataset)
        # test_dataloader = DataLoader(dataset.data['test'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
        # save the test dataloader
        #test_path = os.path.join(str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption), "test_dataloader.pkl")
        #with open(test_path, 'wb') as f:
        #    pickle.dump(test_dataloader, f)

    # Load the test dataloader for the specific client
    path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
    
    #test_path = get_split_paths(cfg, path, True)
    #with open(test_path, 'rb') as f:
    #    test_dataloader = pickle.load(f)

    # Add graph columns to engine config if available
    with open_dict(cfg):
        # order true_graph columns following the topological order of the graph
        #G = nx.from_pandas_adjacency(true_graph, create_using=nx.DiGraph)
        #ordered_nodes = list(nx.topological_sort(G))
        # eliminate task from the ordered columns
        #ordered_nodes = [node for node in ordered_nodes if node != datasets[0].y_info['names'][0]]
        # flatten get_intervention_policy output
        if cfg.dataset.name in ["siim_pneumothorax", "skincon", "cheXpert", "cheXpert_multi"]:
            # true_graph = graph
            if cfg.learning.mode != "localized":
                true_graph = graph
        ordered_nodes = list(itertools.chain.from_iterable(get_intervention_policy(true_graph, y_index)[0]))
        # take the names
        ordered_nodes = [true_graph.columns[idx] for idx in ordered_nodes]
        cfg.engine.centralized_topological_order = ordered_nodes
        cfg.engine.centralized_c_dict = centralized_c_dict 
        # if cfg.dataset.name == "siim_pneumothorax":
        #     true_graph = true_graph.loc[ordered_nodes, ordered_nodes] # c2bm graph
    

    ############ training block ########################################################################################

    # If the training is centralized
    if cfg.learning.mode in ['centralized', 'localized']:
        
        if cfg.learning.mode == 'localized':
            #path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
            print("\033[93mLocalized training\033[0m")
            # Load only the training and validation split specified by the local training parameters
            # From cache get the dataloader
            train_path, val_path, test_path = get_split_paths(cfg, path)
            # if the file is not found, raise an error
            if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
                raise FileNotFoundError(f"File {train_path} or {val_path} or {test_path} not found")
            # Load the dataloaders
            with open(train_path, 'rb') as f:
                train_dataloader = pickle.load(f)
                print("train_dataloader", train_dataloader)
                print("train_dataloader length", len(train_dataloader))
            with open(val_path, 'rb') as f:
                val_dataloader = pickle.load(f)   
            with open(test_path, 'rb') as f:
                test_dataloader = pickle.load(f)
        else:
            # Load all the training and validation splits
            train_dataloader = DataLoader(datasets[0].data['train'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
            val_dataloader = DataLoader(datasets[0].data['val'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
            test_dataloader = DataLoader(datasets[0].data['test'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)

        #batch = next(iter(val_dataloader))
        #print(batch['y'][0])
        engine = instantiate(cfg.engine)
        if hasattr(datasets[0], 'class_weights') and datasets[0].class_weights is not None:
            engine.model.class_weights = datasets[0].class_weights
        if hasattr(datasets[0], 'concept_class_weights') and datasets[0].concept_class_weights:
            engine.model.concept_class_weights = datasets[0].concept_class_weights
        train_dataloader, privacy_engine = maybe_make_private(
            engine, train_dataloader, cfg, epochs=cfg.trainer.max_epochs
        )
        
        try:
            trainer = Trainer(cfg)
            trainer.logger.log_hyperparams(parse_hyperparams(cfg))
            # ---- train
            trainer.fit(engine, train_dataloader, val_dataloader)
            # ----- test
            trainer.test(engine, test_dataloader, ckpt_path='best')
            if privacy_engine is not None:
                spent_eps = privacy_engine.get_epsilon(getattr(engine, "dp_delta", None))
                print(
                    f"\033[96m[DP] Spent ε={spent_eps:.3f} for δ={getattr(engine, 'dp_delta', None)}\033[0m"
                )
            trainer.logger.finalize("success")
        finally:
            if isinstance(trainer.logger, WandbLogger):
                trainer.logger.experiment.finish()

        try:

            try:
                essential_concepts = ordered_nodes
            except NameError:
                essential_concepts = None
            if not essential_concepts:
                essential_concepts = OmegaConf.select(cfg, "engine.centralized_topological_order", default=[]) or []
            task_name = datasets[0].y_info["names"][0]
            essential_set = set(essential_concepts)
            essential_set.add(task_name)

            model_concepts = set()
            predicted_concepts = getattr(engine.model, "predicted_concepts", None)
            if predicted_concepts:
                model_concepts.update(predicted_concepts)
            if not model_concepts:
                cfg_for_coverage = cfg
                engine_c_names = getattr(engine, "c_names_id", None)
                if engine_c_names is None:
                    engine_c_names = OmegaConf.select(cfg_for_coverage, "engine.c_names_id", default=None)
                if engine_c_names is None:
                    engine_c_names = OmegaConf.select(cfg, "engine.c_names_id", default=None)
                if engine_c_names:
                    engine_c_names = list(engine_c_names.values())[0]
                    model_concepts.update(engine_c_names)
            if _dataloaders_have_task_labels(train_dataloader):
                model_concepts.add(task_name)

            covered_concepts = model_concepts & essential_set
            concept_coverage = float(len(covered_concepts) / len(essential_set)) if len(essential_set) > 0 else float('nan')


            additional_metrics = {
                "concept_coverage": concept_coverage,
                "structural_concept_coverage": OmegaConf.select(
                    cfg,
                    "learning.subgraphs.structural_concept_coverage",
                    default=float("nan"),
                ),
            }
            os.makedirs("results", exist_ok=True)
            with open("results/additional_metrics.json", "w") as fp:
                json.dump(additional_metrics, fp, indent=2)

        except Exception as e:
            print(f"\033[91m[WARN] Failed to save additional metrics: {e}\033[0m")
                
                
    elif cfg.learning.mode == 'local_federated':
        
        # hyperparameters   
        n_rounds = int(cfg.learning.settings.n_rounds)
        n_clients = cfg.learning.n_clients 
        patience = int(cfg.learning.settings.patience)
        architecture_update_strategy = str(
            cfg.learning.subgraphs.get(
                "architecture_update_strategy", "full_reinit"
            )
        )
        architecture_ablation_enabled = bool(
            OmegaConf.select(
                cfg, "architecture_ablation.enabled", default=False
            )
        )
        # The local-FL loop mutates trainer.max_epochs/patience below.  Preserve
        # the resolved pre-mutation configuration so saved artifacts describe
        # the actual federated horizon rather than the local trainer settings.
        architecture_ablation_cfg_snapshot = (
            OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            if architecture_ablation_enabled
            else None
        )
        record_shift_snapshot = bool(
            OmegaConf.select(
                cfg,
                "architecture_ablation.record_shift_snapshot",
                default=True,
            )
        )
        cfg.trainer.max_epochs = cfg.learning.settings.local_epochs
        cfg.trainer.patience = 0
        architecture_ablation_recorder = None
        # use_concepts = True # whether to use concept information in the MIA attacks
        
        num_threads = cfg.learning.settings.num_threads
        print(f"\033[93mLocal Federated training with {n_clients} clients\033[0m")
        
        # set seed for reproducibility
        torch.set_num_threads(num_threads)
        # seed_everything(cfg.seed)
        
        # read client data
        n_saved_clients = cfg.learning.subgraphs.get(
            "n_clients_saved",
            n_clients * cfg.learning.subgraphs.get('dataset_client_multiplier', 1),
        )
        train_dataloaders, val_dataloaders, test_dataloaders = load_dataloaders(
            cfg, path, n_saved_clients
        )
        task_name = datasets[0].y_info["names"][0]
        _print_task_label_distribution(train_dataloaders, task_name, "train", per_client=True)
        _print_task_label_distribution(val_dataloaders, task_name, "val", per_client=False)
        _print_task_label_distribution(test_dataloaders, task_name, "test", per_client=False)
        train_dataloaders, canary_loaders, true_in_outs, sia_loader = dataprocess_auditing(train_dataloaders, n_saved_clients, cfg) # NOTE: for the moment we are reducing the training data size
        _print_task_label_distribution(train_dataloaders, task_name, "train after auditing", per_client=True)
        print("\033[94mNumber of samples per client:\033[0m")
        for client_idx, train_loader in enumerate(train_dataloaders, start=1):
            print(f"\033[94mClient {client_idx}: {len(train_loader.dataset)} samples\033[0m")

        # identify if y is present or not for the clients
        #y_present = []
        #for cid in range(len(val_dataloaders)):
        #    if round_val_dataloaders[cid].dataset.y[0]!=-1:
        #        y_present.append(True)
        #    else:
        #         y_present.append(False)
       
        t0 = time.time()
        best_loss = float('inf')
        best_round = 0
        no_improvement_count = 0
        sia_accuracies = []
        history = {"round": [], "loss_val_avg": [], "loss_val_client": {}, "y_acc_val_avg": [], "y_acc_val_client": {}}
        n_dataset_clients = len(train_dataloaders)

        # determine clients for possible predrift and postdrift phases
        if cfg.learning.subgraphs.rnd_drift > 1:
            predrift_clients = [
                int(client_id) for client_id in cfg.learning.subgraphs.get(
                    "predrift_client_ids", range(1, n_clients + 1)
                )
            ]
            postdrift_clients = []

            if cfg.learning.subgraphs.rnd_drift <= n_rounds:
                postdrift_clients = [
                    int(client_id) for client_id in cfg.learning.subgraphs.get(
                        "postdrift_client_ids", range(1, n_dataset_clients + 1)
                    )
                ]

                # check postdrift clients have subgraphs that cover all the subgraphs
                #postdrift_subgraphs = set()
                #for cid in postdrift_clients:
                #    subgraph_id = identify_subgraph(path, cid)
                #    subgraph = subgraphs_concept_names[f'subgraph_{subgraph_id}']
                #    # I want to add the subgraph, not the nodes
                #    postdrift_subgraphs.add(frozenset(subgraph))

                # check if postdrift_subgraphs cover all subgraphs, not nodes but subgraphs
                #all_subgraphs = set()
                #for sg in subgraphs_concept_names.values():
                #    all_subgraphs.add(frozenset(sg))

                #postdrift_mode = cfg.learning.subgraphs.get(
                 #   'client_selection_mode_postdrift',
                 #   cfg.learning.subgraphs.get('client_selection_mode', 'all'),
                #)
                #if postdrift_mode == 'all' and postdrift_subgraphs != all_subgraphs:
                #    raise ValueError("Post-drift clients selected with mode 'all' do not cover all subgraphs. Adjust the generated clients to include all subgraphs.")

        else:
            predrift_clients = None
            postdrift_clients = list(range(1, n_clients + 1))

        tracked_client_ids = list(dict.fromkeys(
            (predrift_clients or []) + postdrift_clients
        ))
        for client_id in tracked_client_ids:
            history["loss_val_client"][client_id] = []
            history["y_acc_val_client"][client_id] = []
        mia_accuracies, mia_epsilons = initialize_mia_results(n_dataset_clients)


        # determine node order
        node_order = datasets[0].c_info["names"] + datasets[0].y_info["names"]

        # update c_name_index, c_names_id, c_names_ood dictionaries based on predrift clients
        cfg_predrift = update_config_from_data_subgroup_clients(cfg, 
                                                           predrift_clients, 
                                                           datasets, 
                                                           subgraphs_concept_names, 
                                                           node_order)
        cfg_postdrift = update_config_from_data_subgroup_clients(
            cfg, postdrift_clients, datasets, subgraphs_concept_names, node_order
        )


        # timing metrics (saved for c2bm and cem)
        timing_metrics: Dict[str, float] = {}

        # determine whether to use graph aggregation, see if there is the dictionary "aggregate_graph" with local_graphs not none
        use_graph_agg = False
        graph_eval_metrics: Dict[str, Dict[str, float]] = {}
        if hasattr(cfg.learning.subgraphs, "aggregate_graph") and model_is_causal(cfg.model):
            agg_graph_cfg = cfg.learning.subgraphs.aggregate_graph
            if agg_graph_cfg is not None and hasattr(agg_graph_cfg, "local_graphs"):
                if agg_graph_cfg.local_graphs is not None and agg_graph_cfg.local_graphs != "none":
                    use_graph_agg = True

        if use_graph_agg:
            
            # build local graphs for all clients
            if predrift_clients is None:
                all_clients = postdrift_clients
            else:
                all_clients = list(dict.fromkeys(predrift_clients + postdrift_clients))

            local_graphs, local_weights = build_local_graphs(all_clients, 
                                                             cfg, 
                                                             train_dataloaders, 
                                                             datasets[0].y_info["names"][0], 
                                                             graph)

            # aggregate graphs for pre-drift and post-drift clients
            # predrift
            t_agg_predrift = time.time()
            graph_predrift, _ = aggregate_graph_proposals(
                client_selection = predrift_clients,
                local_graphs=local_graphs,
                weights=local_weights,
                config_input=cfg_predrift,
                task_node=datasets[0].y_info["names"][0]
            )
            t_agg_predrift = time.time() - t_agg_predrift
            print(f"\033[94m[Timing] aggregate_graph_proposals (predrift): {t_agg_predrift:.3f}s\033[0m")
            timing_metrics["aggregate_graph_predrift"] = t_agg_predrift

            maybe_plot_graph(graph_predrift, 'graph_predrift')
            eval_res = _evaluate_graph_against_truth(
                graph_predrift, true_graph, "aggregated pre-drift graph", key="graph_predrift"
            )
            if eval_res is not None:
                k, metrics = eval_res
                graph_eval_metrics[k] = metrics

            # postdrift
            t_agg_postdrift = time.time()
            graph_postdrift, _ = aggregate_graph_proposals(
                client_selection = postdrift_clients or predrift_clients,
                local_graphs=local_graphs,
                weights=local_weights,
                config_input=cfg_postdrift,
                task_node=datasets[0].y_info["names"][0]
            )
            t_agg_postdrift = time.time() - t_agg_postdrift
            print(f"\033[94m[Timing] aggregate_graph_proposals (postdrift): {t_agg_postdrift:.3f}s\033[0m")
            if cfg.learning.subgraphs.rnd_drift <= n_rounds:
                timing_metrics["aggregate_graph_postdrift"] = t_agg_postdrift

            maybe_plot_graph(graph_postdrift, 'graph_postdrift')
            eval_res = _evaluate_graph_against_truth(
                graph_postdrift, true_graph, "aggregated post-drift graph", key="graph_postdrift"
            )
            if eval_res is not None:
                k, metrics = eval_res
                graph_eval_metrics[k] = metrics

            if graph_eval_metrics:
                with open(os.path.join("results", "graph_metrics.json"), "w") as fp:
                    json.dump(graph_eval_metrics, fp, indent=2)

            # update intervention policy for pre-drift clients
            interv_policy_predrift, ip_names_predrift = get_intervention_policy(graph_predrift, 
                                                                                y_index = graph_predrift.columns.get_loc(datasets[0].y_info["names"][0]) if graph_predrift is not None else None)
            # take in consideration that the intervention policy is already constructed relative to the graph_predrift with the node indices referred to it
            interv_policy_predrift_constructed = True
            interv_policy_postdrift, _ = get_intervention_policy(
                graph_postdrift,
                y_index=graph_postdrift.columns.get_loc(datasets[0].y_info["names"][0])
                if graph_postdrift is not None else None,
            )
            interv_policy_postdrift_constructed = True
        
        else:
            # update graph and intervention policy based on predrift clients
            interv_policy_predrift, graph_predrift = update_intervention_policy_and_graph(
                    cfg_predrift, interv_policy, graph, datasets
            )
            # the intervention policy is not constructed from the graph_predrift, it is just a selection of the one relative to the global graph -> the node indices are still related to the global graph
            interv_policy_predrift_constructed = False
            interv_policy_postdrift, graph_postdrift = update_intervention_policy_and_graph(
                cfg_postdrift, interv_policy, graph, datasets
            )
            interv_policy_postdrift_constructed = False


        # update config predrift with the graph and intervention policy updated based on predrift clients
        cfg_predrift = maybe_update_config_with_graph_subgroup_clients(cfg_predrift, predrift_clients, graph_predrift,interv_policy_predrift,  interv_policy_predrift_constructed, datasets)
        cfg_postdrift = maybe_update_config_with_graph_subgroup_clients(
            cfg_postdrift, postdrift_clients, graph_postdrift,
            interv_policy_postdrift, interv_policy_postdrift_constructed, datasets
        )
        
        # stop code now
        # sys.exit(0)

        # Build independent phase views from the complete loader population.
        # A client may belong to both phases without being copied or renumbered.
        train_dataloaders_predrift = filter_dataloaders_by_concepts(
            list(train_dataloaders), cfg_predrift, predrift_clients,
            subgraphs_concept_names, datasets[0].c_info["names"], path,
        )
        val_dataloaders_predrift = filter_dataloaders_by_concepts(
            list(val_dataloaders), cfg_predrift, predrift_clients,
            subgraphs_concept_names, datasets[0].c_info["names"], path,
        )
        train_dataloaders_postdrift = filter_dataloaders_by_concepts(
            list(train_dataloaders), cfg_postdrift, postdrift_clients,
            subgraphs_concept_names, datasets[0].c_info["names"], path,
        )
        val_dataloaders_postdrift = filter_dataloaders_by_concepts(
            list(val_dataloaders), cfg_postdrift, postdrift_clients,
            subgraphs_concept_names, datasets[0].c_info["names"], path,
        )

        # Keep separate test views as well: the best checkpoint may belong to
        # either phase, independently of whether drift occurred during training.
        per_client_testset = (
            OmegaConf.select(cfg, "combined_datasets.other_datasets", default=None) is not None
            or OmegaConf.select(cfg, "dataset.per_client_testset", default=False)
        )
        test_dataloaders_predrift = filter_dataloaders_by_concepts(
            list(test_dataloaders), cfg_predrift, predrift_clients,
            subgraphs_concept_names, datasets[0].c_info["names"], path,
            loader_client_ids=None if per_client_testset else [1],
        )
        test_dataloaders_postdrift = filter_dataloaders_by_concepts(
            list(test_dataloaders), cfg_postdrift, postdrift_clients,
            subgraphs_concept_names, datasets[0].c_info["names"], path,
            loader_client_ids=None if per_client_testset else [1],
        )

        if architecture_ablation_enabled:
            if int(cfg.learning.subgraphs.rnd_drift) > int(n_rounds):
                raise ValueError(
                    "architecture_ablation.enabled requires rnd_drift to occur "
                    "within learning.settings.n_rounds."
                )
            recorder_cfg = architecture_ablation_cfg_snapshot
            metric_cfg = OmegaConf.select(
                recorder_cfg, "architecture_ablation", default={}
            )
            metric_cfg = OmegaConf.to_container(metric_cfg, resolve=True)
            architecture_ablation_recorder = ArchitectureAblationRecorder(
                recorder_cfg,
                method=architecture_update_strategy,
                metric_config=metric_cfg,
                experiment_fields={
                    "validation_population": "active training-federation clients",
                    "client_aggregation": "unweighted arithmetic mean",
                    "client_uncertainty": "population standard deviation (ddof=0)",
                    "predrift_client_ids": list(predrift_clients or []),
                    "postdrift_client_ids": list(postdrift_clients or []),
                    "federated_rounds": n_rounds,
                    "federated_patience": patience,
                    "local_epochs_per_round": int(
                        cfg.learning.settings.local_epochs
                    ),
                },
            )

        for i in range(len(train_dataloaders)):
            batch = next(iter(train_dataloaders[i]))
            print(f"batch of client {i}: {batch['y'][0:3]}")

        # start federated learning rounds
        init_cfg = cfg_predrift if cfg_predrift is not None else cfg_postdrift
        t_predrift_init = time.time()
        init_engine = instantiate(init_cfg.engine)
        t_predrift_init = time.time() - t_predrift_init
        print(f"\033[94m[Timing] Predrift model instantiation: {t_predrift_init:.3f}s\033[0m")
        timing_metrics["predrift_model_instantiation"] = t_predrift_init
        param_count_predrift = sum(p.numel() for p in init_engine.model.parameters())
        try:
            param_count_postdrift = sum(p.numel() for p in instantiate(cfg_postdrift.engine).model.parameters())
        except Exception:
            param_count_postdrift = param_count_predrift
        global_params = get_parameters(init_engine)
        global_param_keys = list(init_engine.model.state_dict().keys())
        drift_debug_printed = False
        t_start_training = time.time()
        for rnd in range(1, n_rounds + 1):
            print(f"\033[92m\n--> ROUND {rnd}/{n_rounds}\033[0m")
            client_params: List[Tuple[List[torch.Tensor], int]] = []
            val_losses, val_accs, sizes = [], [], []
            
            # ------------------------------------------------------------
            # Setup configuration based on drift round
            # ------------------------------------------------------------
            if rnd < cfg.learning.subgraphs.rnd_drift:
                cfg_round = cfg_predrift
                round_client_ids = predrift_clients
                round_train_dataloaders = train_dataloaders_predrift
                round_val_dataloaders = val_dataloaders_predrift
            else:
                if rnd == cfg.learning.subgraphs.rnd_drift:
                    print("\033[93mDrift occurred: switching to new client data distributions\033[0m")
                cfg_round = cfg_postdrift
                round_client_ids = postdrift_clients
                round_train_dataloaders = train_dataloaders_postdrift
                round_val_dataloaders = val_dataloaders_postdrift

            # Config IDs are 1-based; dataloader arrays are 0-based.
            round_client_indices = [client_id - 1 for client_id in round_client_ids]
            n_clients_round = len(round_client_indices)
            if rnd == cfg.learning.subgraphs.rnd_drift:
                t_init_model = time.time()
            engine = instantiate(cfg_round.engine)
            if hasattr(datasets[0], 'class_weights') and datasets[0].class_weights is not None:
                engine.model.class_weights = datasets[0].class_weights
            if hasattr(datasets[0], 'concept_class_weights') and datasets[0].concept_class_weights:
                engine.model.concept_class_weights = datasets[0].concept_class_weights
            engine.model.to(cfg.device)
            if rnd == cfg.learning.subgraphs.rnd_drift:
                t_init_model = time.time() - t_init_model
                print(f"\033[94m[Timing] Postdrift model instantiation: {t_init_model:.3f}s\033[0m")
                timing_metrics["postdrift_model_instantiation"] = t_init_model

            # if (rnd >= cfg.learning.subgraphs.rnd_drift) and True:
            if  cfg.learning.subgraphs.rnd_drift >=1:
                print("\033[95m[Drift Debug] Checking concept label availability (train/val) for post-drift clients\033[0m")
                for cid in round_client_indices:
                    _print_concept_availability("train", round_train_dataloaders[cid], cfg_round, cid)
                    if round_val_dataloaders[cid] is not None:
                        _print_concept_availability("val", round_val_dataloaders[cid], cfg_round, cid)
                drift_debug_printed = True
            
            # ------------------------------------------------------------
            # local training (sequentially)
            # ------------------------------------------------------------
            print(f"\033[93mLocal training on {n_clients_round} clients\033[0m")
            shift_snapshot_rows = []
            for n, cid in enumerate(round_client_indices):
                # clone global params → local model
                update_config_from_client(cfg_round, datasets, cid)
                local_engine = instantiate(cfg_round.engine)
                client_id = round_client_ids[n]
                local_engine.cid = client_id
                if hasattr(datasets[0], 'class_weights') and datasets[0].class_weights is not None:
                    local_engine.model.class_weights = datasets[0].class_weights
                if hasattr(datasets[0], 'concept_class_weights') and datasets[0].concept_class_weights:
                    local_engine.model.concept_class_weights = datasets[0].concept_class_weights
                # first training
                if rnd == cfg.learning.subgraphs.rnd_drift:
                    # load new architecture and update only those parameters that were present before
                    t_set_params = time.time()
                    migration_report = set_old_parameters(
                        local_engine,
                        global_params,
                        global_param_keys,
                        verbose=(cid == round_client_indices[0]),
                        strategy=architecture_update_strategy,
                        old_model=init_engine.model,
                    )
                    if (
                        architecture_ablation_recorder is not None
                        and cid == round_client_indices[0]
                    ):
                        architecture_ablation_recorder.set_structural_metadata(
                            migration_report
                        )
                    t_set_params = time.time() - t_set_params
                    if cid == round_client_indices[0]:
                        print(f"\033[94m[Timing] Postdrift set_old_parameters (client {cid}): {t_set_params:.3f}s\033[0m")
                        timing_metrics["postdrift_set_old_parameters"] = t_set_params
                else:
                    set_parameters(local_engine, global_params)
                    
                local_engine.model.to(cfg.device)

                if (
                    architecture_ablation_recorder is not None
                    and record_shift_snapshot
                    and rnd == cfg.learning.subgraphs.rnd_drift
                ):
                    immediate_loss, immediate_count = (
                        _compute_unmasked_validation_task_loss(
                            local_engine,
                            round_val_dataloaders[cid],
                            cfg.device,
                        )
                    )
                    shift_snapshot_rows.append(
                        {
                            "client_id": client_id,
                            "task_loss": immediate_loss,
                            "n_labeled_samples": immediate_count,
                            "eligible": True,
                            "phase": "postdrift",
                            "reason": None
                            if np.isfinite(immediate_loss)
                            else "immediate_task_loss_unavailable",
                        }
                    )

                # freeze if required
                maybe_freeze_parameters(
                    train_dataloader = round_train_dataloaders[cid],
                    y_to_freeze = True if round_val_dataloaders[cid].dataset.y[0]==-1 else False, # to change if we can incorporate y in round_train_dataloaders[cid].dataset
                    model=local_engine.model,
                    learning=cfg.learning.mode,
                    freezing=cfg.learning.settings.freezing,
                )

                train_loader, privacy_engine = maybe_make_private(
                    local_engine,
                    round_train_dataloaders[cid],
                    cfg_round,
                    epochs=cfg.trainer.max_epochs,
                )

                # local train
                trainer = Trainer(cfg, client_id=client_id)
                trainer.logger.log_hyperparams(parse_hyperparams(cfg)) 
                trainer.fit(local_engine, train_loader)
                if privacy_engine is not None:
                    spent_eps = privacy_engine.get_epsilon(getattr(local_engine, "dp_delta", None))
                    print(
                        f"\033[96m[DP][Client {client_id}] Spent ε={spent_eps:.3f} for δ={getattr(local_engine, 'dp_delta', None)}\033[0m"
                    )
                local_engine.model.to(cfg.device) # put back to device
                n_samples = len(round_train_dataloaders[cid].dataset)
                client_modality = getattr(round_train_dataloaders[cid].dataset, "modality", None)
                                    
                # # local validation
                # if round_val_dataloaders[cid] is not None:
                #     avg_loss = compute_validation_loss(local_engine.model, round_val_dataloaders[cid], cfg)
                #     history["loss_val_client"][n].append(avg_loss)
    
                # collect weights for aggregation
                if hasattr(local_engine.model, "modality_encoders"):
                    client_params.append((get_parameters(local_engine), n_samples, client_modality))
                else:
                    client_params.append((get_parameters(local_engine), n_samples))

            if shift_snapshot_rows:
                architecture_ablation_recorder.record_shift_snapshot(
                    "post_migration_pre_local_training",
                    rnd,
                    shift_snapshot_rows,
                )

            
            # # ------------------------------------------------------------
            # # Privacy Attack: MIA
            # # ------------------------------------------------------------
            # if cfg.learning.settings.mia:
            #     print(f"\033[93mRunning Membership Inference Attack (MIA)\033[0m")
                
            #     set_parameters(local_engine, global_params)

            # if cfg.learning.settings.mia:
            #     #global_vec = flat_trainable_params_tensor(local_engine.model, cfg.device)

            #     start_n_client = 0 if rnd < cfg.learning.subgraphs.rnd_drift else n_clients 
            #     for cid in round_client_indices:
            #         # normalize client update vector
            #         true_in_out = true_in_outs[cid].float().numpy()
            #         set_parameters(local_engine, client_params[cid][0])
            #         client_vec = flat_trainable_params_tensor(local_engine.model, cfg.device)
            #         client_update = client_vec - global_vec
            #         client_update = client_update / torch.tensor(np.linalg.norm(client_update.cpu()), device=cfg.device) 

            #         # white-box attack (accumulate over the whole canary loader)
            #         set_parameters(local_engine, global_params)
            #         client_model = local_engine.model.to(cfg.device)
            #         scores_whitebox_list = []
            #         for batch in canary_loaders[cid]:
            #             scores_whitebox_list.append(score_whitebox_batch(batch, client_model, client_update, cfg, use_concepts=use_concepts))
            #         scores_whitebox = np.concatenate(scores_whitebox_list, axis=0)
            #         set_parameters(local_engine, client_params[cid][0])

            #         # black-box baseline (negative loss) accumulated over loader
            #         client_model = local_engine.model.to(cfg.device)
            #         scores_blackbox_loss = score_blackbox_loss_loader(canary_loaders[cid], client_model, cfg, use_concepts=use_concepts)

            #         # black-box concept-entropy (uses ONLY concepts if available, else falls back to label entropy)
            #         # scores_blackbox_concept = score_blackbox_concept_entropy_loader(canary_loaders[cid], client_model, cfg)

                    # # black-box shadow MLP (features = label confidences/margins/entropy/-loss + concept stats if available)
                    # shadow_epochs = getattr(getattr(cfg, "learning").settings, "mia_shadow_epochs", 100)
                    # scores_blackbox_shadow = shadow_mlp_scores_loader(
                    #     canary_loaders[cid],
                    #     client_model,
                    #     cfg,
                    #     y_mem_labels=true_in_out,
                    #     use_concepts=use_concepts,
                    #     epochs=shadow_epochs,
                    #     batch_size=64,
                    #     lr=1e-4,
                    #     k_folds=5,
                    #     scores_whitebox_list=None,  # no need to use them.. no effect observed in practice on asia
                    # )

            #         # evaluate white-box
            #         accuracy_mia, privacy_estimate = evaluate_privacy(scores_whitebox, true_in_out, cfg)
            #         print(f"Client {cid} - MIA accuracy (whitebox): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
            #         mia_accuracies['whitebox'][cid].append(accuracy_mia)
            #         mia_epsilons['whitebox'][cid].append(privacy_estimate)

            #         # evaluate black-box baseline (loss)
            #         accuracy_mia, privacy_estimate = evaluate_privacy(scores_blackbox_loss, true_in_out, cfg)
            #         print(f"Client {cid} - MIA accuracy (blackbox-loss): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
            #         mia_accuracies['blackbox'][cid].append(accuracy_mia)
            #         mia_epsilons['blackbox'][cid].append(privacy_estimate)

            #         # evaluate black-box concept-entropy
            #         # accuracy_mia, privacy_estimate = evaluate_privacy(scores_blackbox_concept, true_in_out, cfg)
            #         # print(f"Client {cid} - MIA accuracy (blackbox-concept): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
            #         # mia_accuracies['blackbox_concept'][cid].append(accuracy_mia)
            #         # mia_epsilons['blackbox_concept'][cid].append(privacy_estimate)

            #         # evaluate black-box shadow MLP
            #         accuracy_mia, privacy_estimate = evaluate_privacy(scores_blackbox_shadow, true_in_out, cfg)
            #         print(f"Client {cid} - MIA accuracy (blackbox-shadow): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
            #         mia_accuracies['blackbox_shadow'][cid].append(accuracy_mia)
            #         mia_epsilons['blackbox_shadow'][cid].append(privacy_estimate)
            
            # # ------------------------------------------------------------
            # # Privacy Attack: SIA
            # # ------------------------------------------------------------
            # if cfg.learning.settings.sia:
            #     print(f"\033[93mRunning Source Inference Attack (SIA)\033[0m")

            #     sia_accuracies.append(run_sia_attack(
            #         local_engine=local_engine,  # instantiated engine
            #         sia_loader=sia_loader,          # returned by dataprocess_auditing
            #         client_params=client_params,    # local models from this round
            #         cfg=cfg,
            #         use_concepts=use_concepts,
            #     ))
            #     print(f"\033[92mSIA accuracy this round: {sia_accuracies[-1]:.4f}\033[0m")  
            

            # ------------------------------------------------------------
            # FedAvg aggregation
            # ------------------------------------------------------------
            print(f"\033[93mAggregating local models\033[0m")
            current_param_keys = list(local_engine.model.state_dict().keys())
            if client_params and len(client_params[0]) == 3:
                global_params = aggregate_multimodal(
                    client_params,
                    current_param_keys,
                    reference_parameters=get_parameters(local_engine),
                )
            else:
                global_params = aggregate(client_params)
            global_param_keys = current_param_keys
            print("Saving global model parameters")
            params_dict = zip(local_engine.model.state_dict().keys(), global_params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            local_engine.model.load_state_dict(state_dict, strict=True)
            
            # ------------------------------------------------------------
            # FedAvg aggregation on client validation sets
            # ------------------------------------------------------------
            print(f"\033[93mEvaluating on client validation sets\033[0m")
            local_engine = instantiate(cfg_round.engine)
            set_parameters(local_engine, global_params)
            local_engine.model.to(cfg.device)
            for tracked_client_id in tracked_client_ids:
                history["loss_val_client"][tracked_client_id].append(float("nan"))
                history["y_acc_val_client"][tracked_client_id].append(float("nan"))

            ablation_client_rows = []
            for client_id, cid in zip(round_client_ids, round_client_indices):
                local_engine.cid = client_id
                val_metrics = trainer.validate(local_engine, round_val_dataloaders[cid])[0]  #{'val/c/asia': 0.0, 'val/c/bronc': 0.0, 'val/c/either': 0.0, 'val/c/lung': 0.0, 'val/c/smoke': 0.0, 'val/c/tub': 0.0, 'val/c/xray': 0.0, 'val_loss': nan}
                val_losses.append(val_metrics['val_loss'])
                val_accs.append(val_metrics.get('val/y/y_accuracy', np.nan))  
                 # log per-client val metrics
                history["loss_val_client"][client_id][-1] = val_metrics["val_loss"]
                history["y_acc_val_client"][client_id][-1] = val_metrics.get("val/y/y_accuracy", np.nan)
                validation_size = len(round_val_dataloaders[cid].dataset)
                sizes.append(validation_size)

                task_loss_value = val_metrics.get("val_task_loss", float("nan"))
                if torch.is_tensor(task_loss_value):
                    task_loss_value = task_loss_value.detach().cpu().item()
                try:
                    task_loss_value = float(task_loss_value)
                except (TypeError, ValueError):
                    task_loss_value = float("nan")
                ablation_client_rows.append(
                    {
                        "client_id": client_id,
                        "task_loss": task_loss_value,
                        "n_labeled_samples": validation_size,
                        "eligible": True,
                        "phase": "predrift"
                        if rnd < cfg.learning.subgraphs.rnd_drift
                        else "postdrift",
                        "reason": None
                        if np.isfinite(task_loss_value)
                        else "val_task_loss_unavailable",
                    }
                )

            # log aggregated val metrics (weighted)
            w_loss = sum(l * s for l, s in zip(val_losses, sizes)) / sum(sizes)
            val_accs = np.asarray(val_accs, dtype=float); sizes = np.asarray(sizes, dtype=float)
            mask = ~np.isnan(val_accs)          # keep only clients that actually have an accuracy
            if mask.any():
                y_acc = (val_accs[mask] * sizes[mask]).sum() / sizes[mask].sum()
            else:
                y_acc = np.nan
            history["round"].append(rnd)
            history["loss_val_avg"].append(w_loss)
            history["y_acc_val_avg"].append(y_acc)
            print(f"\033[92m✅ aggregated  val_loss={w_loss:.4f}, val/y/y_accuracy={y_acc:.4f}  \033[0m")
            if architecture_ablation_recorder is not None:
                ablation_summary = architecture_ablation_recorder.record_round(
                    rnd, ablation_client_rows
                )
                print(
                    "\033[96m[Architecture ablation] "
                    f"task_loss_mean={ablation_summary['mean_task_loss']:.4f}, "
                    f"client_std={ablation_summary['std_task_loss']:.4f}, "
                    f"n={ablation_summary['n_evaluated_clients']}\033[0m"
                )

            # check improvement
            check_improvements = False
            if cfg.learning.subgraphs.rnd_drift > n_rounds:
                check_improvements = True
            elif rnd >= cfg.learning.subgraphs.rnd_drift:
                check_improvements = True

            if check_improvements:
                if w_loss < best_loss:
                    best_loss = w_loss
                    best_round = rnd
                    no_improvement_count = 0
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(local_engine.model.state_dict(), f"checkpoints/model_round_{rnd}.pth")
                else:
                    no_improvement_count += 1

            # early stopping if no improvement after 'patience' rounds
            if no_improvement_count >= patience:
                print(f"\033[91mEarly stopping triggered at round {rnd}.\033[0m")
                break


        if architecture_ablation_recorder is not None:
            architecture_ablation_recorder.finalize()


        # ------------------------------------------------------------
        # Trim logged metrics up to the best round (drop overfitting tail)
        # ------------------------------------------------------------
        last_round_executed = history["round"][-1] if len(history["round"]) > 0 else 0
        try:
            # Find index of best_round within history["round"], fallback to argmin
            if len(history["round"]) == 0:
                keep_upto = 0
            else:
                if best_round in history["round"]:
                    keep_upto = history["round"].index(best_round)
                else:
                    keep_upto = int(np.argmin(history["loss_val_avg"]))
            n_keep = keep_upto + 1

            # Trim history
            print(f"\033[94mValidation loss history before trimming: {history['loss_val_avg']}\033[0m")
            history["round"] = history["round"][:n_keep]
            history["loss_val_avg"] = history["loss_val_avg"][:n_keep]
            history["y_acc_val_avg"] = history["y_acc_val_avg"][:n_keep]
            for client_id in tracked_client_ids:
                history["loss_val_client"][client_id] = history["loss_val_client"][client_id][:n_keep]
                history["y_acc_val_client"][client_id] = history["y_acc_val_client"][client_id][:n_keep]
            # save history to json
            with open("results/training_history.json", "w") as fp:
                json.dump(history, fp, indent=2)

            # Trim SIA (per-round)
            if isinstance(sia_accuracies, list) and len(sia_accuracies) > 0:
                sia_accuracies[:] = sia_accuracies[:n_keep]

            # Trim MIA (per-round, per-client, per-variant)
            if isinstance(mia_accuracies, dict):
                for variant, per_client in mia_accuracies.items():
                    for cid in range(len(per_client)):
                        per_client[cid] = per_client[cid][:n_keep]
            if isinstance(mia_epsilons, dict):
                for variant, per_client in mia_epsilons.items():
                    for cid in range(len(per_client)):
                        per_client[cid] = per_client[cid][:n_keep]

            print(f"\033[96mTrimmed metrics to best round {best_round} (keeping {n_keep} rounds, removed {max(0, len(history['loss_val_avg']) - n_keep)}).\033[0m")
        except Exception as e:
            print(f"\033[91m[WARN] Failed trimming metrics: {e}\033[0m")


        # Ablation figures are deliberately generated only by the standalone
        # results reader so plotting choices never require retraining.
        if not architecture_ablation_enabled:
            plot_training_metrics(history)
        # ------------------------------------------------------------
        # Final evaluation on the test set
        # ------------------------------------------------------------
        print(f"\033[93mFinal evaluation on the test set\033[0m")
        # Load the best model
        ind_min_loss = np.argmin(history["loss_val_avg"])
        # best_round = history["round"][ind_min_loss]
        print(f"\033[92mBest round: {best_round} with loss {history['loss_val_avg'][ind_min_loss]:.4f}\033[0m")
        cfg_eval = cfg_predrift if best_round < cfg.learning.subgraphs.rnd_drift else cfg_postdrift
        local_engine = instantiate(cfg_eval.engine)
        local_engine.model.load_state_dict(torch.load(f"checkpoints/model_round_{best_round}.pth", weights_only=False, map_location="cpu"))

        # Evaluate the phase associated with the selected checkpoint. Keep the
        # original 1-based client ID next to every per-client test loader.
        evaluate_predrift = best_round < cfg.learning.subgraphs.rnd_drift
        eval_client_ids = predrift_clients if evaluate_predrift else postdrift_clients
        eval_test_dataloaders = (
            test_dataloaders_predrift if evaluate_predrift
            else test_dataloaders_postdrift
        )
        if per_client_testset:
            test_pairs = [
                (client_id, eval_test_dataloaders[client_id - 1])
                for client_id in eval_client_ids
            ]
        else:
            test_pairs = [(None, eval_test_dataloaders[0])]

        per_loader_test_artifacts = []
        per_loader_test_weights = []
        per_client_test_artifacts = {}
        checkpoint_state = local_engine.model.state_dict()
        for client_id, test_dataloader in test_pairs:
            for filename in TEST_RESULT_FILENAMES:
                artifact_path = os.path.join("results", filename)
                if os.path.exists(artifact_path):
                    os.remove(artifact_path)
            test_engine = instantiate(cfg_eval.engine)
            test_engine.model.load_state_dict(checkpoint_state)
            test_engine.cid = client_id
            test_trainer = Trainer(cfg, client_id=client_id)
            test_trainer.test(test_engine, test_dataloader)
            artifacts = _collect_test_artifacts("results")
            artifact_key = "global" if client_id is None else str(client_id)
            per_client_test_artifacts[artifact_key] = artifacts
            per_loader_test_artifacts.append(artifacts)
            per_loader_test_weights.append(len(test_dataloader.dataset))

        with open("results/test_client_mapping.json", "w") as handle:
            json.dump({
                "phase": "predrift" if evaluate_predrift else "postdrift",
                "per_client_testset": bool(per_client_testset),
                "client_ids": [client_id for client_id, _ in test_pairs],
            }, handle, indent=2)
        with open("results/per_client_test_artifacts.pkl", "wb") as handle:
            pickle.dump(per_client_test_artifacts, handle)
        per_client_test_summary = {
            client_id: {
                "test/y/y_accuracy": artifacts.get("y_accuracy.pkl", {}).get("_baseline", float("nan")),
                **{
                    f"test/c/{name}": value
                    for name, value in artifacts.get("c_accuracy.pkl", {}).items()
                },
            }
            for client_id, artifacts in per_client_test_artifacts.items()
        }
        with open("results/per_client_test_metrics.json", "w") as handle:
            json.dump(per_client_test_summary, handle, indent=2)

        aggregated_test_summary = _aggregate_and_save_test_artifacts(
            "results",
            per_loader_test_artifacts,
            per_loader_test_weights,
        )
        if "test/y/y_accuracy" in aggregated_test_summary:
            print(
                f"\033[92mAggregated test/y/y_accuracy="
                f"{aggregated_test_summary['test/y/y_accuracy']:.4f}\033[0m"
            )
        print(f"\033[90mFinished! Training time: {round((time.time() - t0)/60, 2)} minutes\033[0m")

        # Save additional drift-related metrics
        try:
            last_round = last_round_executed
            drift_happened = cfg.learning.subgraphs.rnd_drift <= last_round

            try:
                essential_concepts = ordered_nodes
            except NameError:
                essential_concepts = None
            if not essential_concepts:
                essential_concepts = OmegaConf.select(cfg, "engine.centralized_topological_order", default=[]) or []
            task_name = datasets[0].y_info["names"][0]
            essential_set = set(essential_concepts)
            essential_set.add(task_name)

            model_concepts = set()
            predicted_concepts = getattr(local_engine.model, "predicted_concepts", None)
            if predicted_concepts:
                model_concepts.update(predicted_concepts)
            if not model_concepts:
                cfg_for_coverage = cfg_eval if cfg_eval is not None else cfg
                engine_c_names = getattr(local_engine, "c_names_all", None)
                if engine_c_names is None:
                    engine_c_names = OmegaConf.select(cfg_for_coverage, "engine.c_names_all", default=None)
                if engine_c_names is None:
                    engine_c_names = OmegaConf.select(cfg, "engine.c_names_all", default=None)
                if engine_c_names:
                    model_concepts.update(engine_c_names)
            coverage_client_ids = (
                predrift_clients
                if best_round < cfg.learning.subgraphs.rnd_drift
                else postdrift_clients
            )
            if _dataloaders_have_task_labels(train_dataloaders, coverage_client_ids):
                model_concepts.add(task_name)

            covered_concepts = model_concepts & essential_set
            concept_coverage = float(len(covered_concepts) / len(essential_set)) if len(essential_set) > 0 else float('nan')

            params_change_ratio = 0.0
            if drift_happened and param_count_predrift > 0:
                delta_params = max(0, param_count_postdrift - param_count_predrift)
                params_change_ratio = float(delta_params / param_count_predrift)

            additional_metrics = {
                "concept_coverage": concept_coverage,
                "structural_concept_coverage": OmegaConf.select(
                    cfg,
                    "learning.subgraphs.structural_concept_coverage",
                    default=float("nan"),
                ),
                "percent_params_changed": params_change_ratio,
                "drift_happened": drift_happened,
                "last_round": last_round,
            }
            os.makedirs("results", exist_ok=True)
            with open("results/additional_metrics.json", "w") as fp:
                json.dump(additional_metrics, fp, indent=2)

            # Save timing metrics (for c2bm and cem)
            if cfg.model.name in ('c2bm', 'cem'):
                if "postdrift_model_instantiation" in timing_metrics and "postdrift_set_old_parameters" in timing_metrics:
                    timing_metrics["postdrift_init_total"] = timing_metrics["postdrift_model_instantiation"] + timing_metrics["postdrift_set_old_parameters"]
                timing_metrics["total_training_time"] = time.time() - t_start_training
                timing_metrics["n_rounds_executed"] = last_round_executed
                timing_metrics["model_name"] = cfg.model.name
                with open("results/timing_metrics.json", "w") as fp:
                    json.dump(timing_metrics, fp, indent=2)
        except Exception as e:
            print(f"\033[91m[WARN] Failed to save additional metrics: {e}\033[0m")
    
        # Evaluate the model on the client datasets
        #test_losses, sizes = [], []
    #{'val/c/asia': 0.0, 'val/c/bronc': 0.0, 'val/c/either': 0.0, 'val/c/lung': 0.0, 'val/c/smoke': 0.0, 'val/c/tub': 0.0, 'val/c/xray': 0.0, 'val_loss': nan}
            #test_losses.append(test_metrics['test_loss'])
            #sizes.append(len(test_dataloaders[cid].dataset))

            # log aggregated val metrics (weighted)
            #t_loss = sum(l * s for l, s in zip(test_losses, sizes)) / sum(sizes)

        print(f"\033[90mFinished! Training time: {round((time.time() - t0)/60, 2)} minutes\033[0m")
        
        
        # # save mia results
        # if cfg.learning.settings.mia:
        #     print(f"Saving MIA results: {os.getcwd() + '/mia_results.json'}")
        #     with open("mia_results.json", "w") as fp:
        #         json.dump(
        #             {
        #                 "accuracies": mia_accuracies,  
        #                 "epsilons":   mia_epsilons,
        #             }, fp, indent=2)
            
        #     # plot MIA results
        #     plot_and_save_max_mia(show=False)
        
        # # save sia results
        # if cfg.learning.settings.sia:
        #     print(f"Saving SIA results: {os.getcwd() + '/sia_results.json'}")
        #     with open("sia_results.json", "w") as fp:
        #         json.dump(
        #             {
        #                 "accuracies": sia_accuracies,
        #             }, fp, indent=2)
            
        #     # plot SIA results
        #     plot_and_save_max_sia(out_json="sia_max.json", show=False)
        

        # # ------------------------------------------------------------
        # # Run DRA attacks
        # # ------------------------------------------------------------ 
        # if cfg.learning.settings.dra:
            
        #     # Perform DRA on each client test set for n_samples
        #     for testid in range(len(test_dataloaders)):
        #         dra_results = run_dra_attack(
        #             test_dataloader=test_dataloaders[testid],
        #             model=instantiate(cfg.engine).model,
        #             device=cfg.device,
        #             methods=("DLG","iDLG"),
        #             max_attacks_per_loader=min(cfg.learning.settings.dra_samples_per_loader, len(test_dataloaders[testid].dataset)),
        #             iters=300,
        #             lr=1.0,
        #             early_stop_tol=1e-6,
        #             log_every=2000,
        #         )
        #         # save the dict dra_results 
        #         with open(f"dra_results_client_{testid}.json", "w") as fp:
        #             json.dump(dra_results, fp, indent=2)
        #         summarize_dra_results(f"dra_results_client_{testid}.json", metrics=("mse", "loss"))

  
    elif cfg.learning.mode == 'FedCBM':
        n_rounds = int(cfg.learning.settings.n_rounds)
        n_clients = int(cfg.learning.n_clients)
        dataset_client_multiplier = int(cfg.learning.subgraphs.get('dataset_client_multiplier', 1))
        num_threads = int(cfg.learning.settings.get("num_threads", 1))
        torch.set_num_threads(num_threads)

        print(f"\033[93mFedCBM baseline with {n_clients} clients\033[0m")
        if int(cfg.learning.subgraphs.get("rnd_drift", 0)) <= n_rounds:
            raise ValueError(
                "[FedCBM] This baseline is static. "
                "Set learning.subgraphs.rnd_drift > learning.settings.n_rounds "
                "(or trainer.max_epochs) to prevent drift during FedCBM runs."
            )

        train_dataloaders, val_dataloaders, test_dataloaders = load_dataloaders(
            cfg,
            path,
            n_clients * dataset_client_multiplier,
        )

        available_clients = min(n_clients, len(train_dataloaders))
        active_client_indices = list(range(available_clients))
        if available_clients < n_clients:
            print(
                f"\033[91m[FedCBM] Requested {n_clients} clients but found {available_clients}."
                " Proceeding with available clients.\033[0m"
            )

        fedcbm_metrics = run_fedcbm_baseline(
            cfg=cfg,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            test_dataloaders=test_dataloaders,
            active_client_indices=active_client_indices,
            concept_names=datasets[0].c_info["names"],
            essential_concepts=ordered_nodes,
            device=cfg.device,
        )

        print(
            "\033[92mFedCBM completed: "
            f"task_accuracy_weighted={fedcbm_metrics.get('task_accuracy_weighted', float('nan')):.4f}, "
            f"task_accuracy_macro={fedcbm_metrics.get('task_accuracy_macro', float('nan')):.4f}, "
            f"heads_trained={fedcbm_metrics.get('n_clients_with_local_head', 0)}/"
            f"{fedcbm_metrics.get('n_active_clients', len(active_client_indices))}\033[0m"
        )

    elif cfg.learning.mode == 'FCL':
        n_rounds = int(cfg.learning.settings.n_rounds)
        n_clients = int(cfg.learning.n_clients)
        dataset_client_multiplier = int(cfg.learning.subgraphs.get('dataset_client_multiplier', 1))
        num_threads = int(cfg.learning.settings.get("num_threads", 1))
        torch.set_num_threads(num_threads)

        print(f"\033[93mFCL baseline with {n_clients} clients\033[0m")
        if int(cfg.learning.subgraphs.get("rnd_drift", 0)) <= n_rounds:
            raise ValueError(
                "[FCL] This baseline is static. "
                "Set learning.subgraphs.rnd_drift > learning.settings.n_rounds "
                "(or trainer.max_epochs) to prevent drift during FCL runs."
            )

        train_dataloaders, val_dataloaders, test_dataloaders = load_dataloaders(
            cfg,
            path,
            n_clients * dataset_client_multiplier,
        )

        available_clients = min(n_clients, len(train_dataloaders))
        active_client_indices = list(range(available_clients))
        if available_clients < n_clients:
            print(
                f"\033[91m[FCL] Requested {n_clients} clients but found {available_clients}."
                " Proceeding with available clients.\033[0m"
            )

        fcl_metrics = run_fcl_baseline(
            cfg=cfg,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            test_dataloaders=test_dataloaders,
            active_client_indices=active_client_indices,
            concept_names=datasets[0].c_info["names"],
            n_classes=int(datasets[0].y_info["cardinality"][0]),
            device=cfg.device,
        )

        print(
            "\033[92mFCL completed: "
            f"task_accuracy_weighted={fcl_metrics.get('task_accuracy_weighted', float('nan')):.4f}, "
            f"task_accuracy_macro={fcl_metrics.get('task_accuracy_macro', float('nan')):.4f}, "
            f"active_clients={fcl_metrics.get('n_active_clients', len(active_client_indices))}\033[0m"
        )

    elif cfg.learning.mode == 'federated':

        # Add path to cfg
        with open_dict(cfg):
            cfg.path = os.getcwd()

        # Save the full configuration to a temporary file
        config_filepath = "temp_config.yaml"
        with open(config_filepath, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        os.environ["config_path"] = os.path.join(os.getcwd(),config_filepath)
        print(f"Config file saved to {os.path.join(os.getcwd(),config_filepath)}")

        # Instantiate the FL training
        print("\033[93mFederated training\033[0m")
        # subprocess.Popen(["bash", "../../../../../src/fl_training.sh"])
        subprocess.run(["bash", "../../../../../src/fl_training.sh"])

        print("\033[93mFederated training completed.\033[0m")
        
        # Delete the temporary config file
        os.remove(config_filepath)
    else:
        raise ValueError(
            "The learning mode is not supported. Please choose one of the following: "
            "centralized, localized, local_federated, FedCBM, FCL, federated."
        )

    # delete any created checkpoints
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")

if __name__ == "__main__":
    main()
