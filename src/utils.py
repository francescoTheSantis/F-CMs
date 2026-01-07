from scipy import datasets
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
from omegaconf import DictConfig, open_dict, OmegaConf  # type: ignore
from src.my_hydra import parse_hyperparams, target_classname
from src.metrics import edge_type
from env import CACHE
import os
import random
import warnings
from src.data.generate_split import get_subgraph_dict
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional

import re
from functools import reduce
from flwr.common import NDArrays
from collections import OrderedDict
import shutil
import pickle
import copy
import math
import scipy

from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    ConcatDataset,
    random_split,
)


def load_dataloaders(cfg: DictConfig, path: str, n_clients: int):
    combined_dataset = OmegaConf.select(cfg, 'combined_datasets.other_datasets', default=None)
    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []

    # if cfg.combined_datasets is None then there is a unique test_dataloader
    for client_id in range(1,n_clients+1):
        train_path, val_path, test_path = get_split_paths_fl(cfg, path, client_id)
        # if the file is not found, raise an error
        if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"File {train_path} or {val_path} or {test_path} not found")
        # Load the dataloaders
        with open(train_path, 'rb') as f:
            train_dataloader = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_dataloader = pickle.load(f)

        train_dataloaders.append(train_dataloader)
        val_dataloaders.append(val_dataloader)

        if combined_dataset is not None:
            with open(test_path, 'rb') as f:
                test_dataloader = pickle.load(f)
            test_dataloaders.append(test_dataloader)
        else:
            if client_id == 1:
                with open(test_path, 'rb') as f:
                    test_dataloader = pickle.load(f)
                test_dataloaders.append(test_dataloader)

    return train_dataloaders, val_dataloaders, test_dataloaders

            
def remove_checkpoints(best_round: int):
    for file in os.listdir("checkpoints"):
        path = os.path.join("checkpoints", file)
        if file != f"model_round_{best_round}.pth":
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def get_parameters(engine):
    return [val.cpu().numpy() for _, val in engine.model.state_dict().items()]


def set_parameters(engine, parameters):
    params_dict = zip(engine.model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    engine.model.load_state_dict(state_dict, strict=True)


def set_old_parameters(engine, parameters, parameter_keys, verbose: bool = False):
    """
    Load only the parameters that are compatible with the current model.
    Useful when the architecture expands and new modules are introduced.
    """
    if parameter_keys is None:
        raise ValueError("parameter_keys must be provided to map previous parameters.")
    if len(parameters) != len(parameter_keys):
        warnings.warn(
            f"Parameter/key length mismatch: {len(parameters)} params vs {len(parameter_keys)} keys. "
            "Proceeding with the shortest length."
        )

    new_state = engine.model.state_dict()
    loaded = 0
    missing = 0
    mismatched = 0

    for k, v in zip(parameter_keys, parameters):
        if k not in new_state:
            print(f"Key {k} not found in the current model state_dict.")
            missing += 1
            continue
        old_tensor = torch.tensor(v)
        if new_state[k].shape != old_tensor.shape:
            print(f"Shape mismatch for key {k}: expected {new_state[k].shape}, got {old_tensor.shape}.")
            mismatched += 1
            continue
        if old_tensor.dtype != new_state[k].dtype:
            old_tensor = old_tensor.to(new_state[k].dtype)
        new_state[k] = old_tensor
        loaded += 1

    engine.model.load_state_dict(new_state, strict=False)
    if verbose:
        total = len(parameter_keys)
        print(
            f"\033[96mLoaded {loaded}/{total} params from previous model "
            f"(missing={missing}, shape_mismatch={mismatched}).\033[0m"
        )
    return {"loaded": loaded, "missing": missing, "mismatched": mismatched}


def model_has_concepts(model):
    name = model.name
    if name in ['blackbox_multi', 'cbm_linear', 'cbm_mlp', 'cem', 'c2bm', 'cgm']:
        return True
    elif name in ['blackbox']:
        return False
    else:
        raise ValueError(f"Unknown model type: {name}")


def model_is_causal(model):
    name = model.name
    if name in ['c2bm', 'cgm']:
        return True
    elif name in ['blackbox', 'blackbox_multi', 'cem', 'cbm_linear', 'cbm_mlp']:
        return False
    else:
        raise ValueError(f"Unknown model type: {name}")


def clean_empty_configs(cfg: DictConfig) -> DictConfig:
    """ can be used to set default values for missing keys """
    with open_dict(cfg):
        if not cfg.get('causal_discovery'):
            cfg.update(causal_discovery = None)
        if not cfg.get('llm'):
            cfg.update(llm = None)
        if not cfg.get('rag'):
            cfg.update(rag = None)
    return cfg


# def update_config_from_model(cfg: DictConfig) -> DictConfig:
#     """ can be used to update the config based on the model """
#     if not model_is_causal(cfg.model):
#         cfg.causal_discovery = None
#     return cfg


def extract_number_between_substrings(s, start_substring='trainset_', end_substring='_subgraph'):
    pattern = re.escape(start_substring) + r'(\d+)' + re.escape(end_substring)
    match = re.search(pattern, s)
    if match:
        return int(match.group(1))
    return None


def identify_subgraph(path, client_id):
    for file in os.listdir(path):
        if extract_number_between_substrings(file) == client_id:
            # Get the substring between "subgraph_" and "."
            subgraph_id = file.split('subgraph_')[1].split('.')[0]
            return subgraph_id
    return None

def update_config_from_client(cfg: DictConfig, datasets, cid: int) -> DictConfig:
    """ can be used to update the config based on the client id """
    if cfg.learning.mode=="local_federated" and len(datasets)>1:
        with open_dict(cfg):
            dataset = datasets[(cid) % len(datasets)]
            cfg.engine.model.update(input_size = dataset.data["train"].X.shape[-1] if dataset.data["train"].X is not None else None)
    return cfg


def update_config_from_data(cfg: DictConfig, datasets, subgraphs_concept_names, subset_concepts = None, subset_clients = None, node_order = None) -> DictConfig:
    """ can be used to update the config based on the data, e.g., set input and output size """ 

    with open_dict(cfg):

        if cfg.learning.mode=="localized":
            if len(datasets)>1:
                dataset = datasets[(cfg.client_id-1) % len(datasets)]
            else:
                dataset = datasets[0]

            input_size = dataset.data["train"].X.shape[-1] if dataset.data["train"].X is not None else None

            original_c_names = dataset.c_info['names']
            path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
            # Get the subgraph given the client id
            subgraph_id = identify_subgraph(path, cfg.client_id)
            updated_c_names = subgraphs_concept_names['subgraph_'+subgraph_id]
            #c_cardinality = [card for card, name in zip(dataset.c_info['cardinality'], dataset.c_info['names']) if name in updated_c_names]
            c_combined = [(name, card) for name, card in zip(dataset.c_info['names'], dataset.c_info['cardinality']) if name in updated_c_names]
            c_info = {'names': [x for x, _ in c_combined], 'cardinality': [card for _, card in c_combined]}

            # The list of names for in-distribution concepts of the client
            c_names_id = dict()
            #c_cardinality = dict()
            # The list of names for out-of-distribution concepts for the client
            c_names_ood = dict()
            ## Get the subgraph given the client id 
            c_names_id = {cfg.client_id: [name for name in dataset.c_info['names'] if name in updated_c_names]}
            #c_cardinality = {cfg.client_id: [card for card, name in zip(dataset.c_info['cardinality'], dataset.c_info['names']) if name in updated_c_names]}
            c_names_ood = {cfg.client_id: [name for name in dataset.c_info['names'] if name not in updated_c_names]}
            c_names_all = original_c_names
      
        elif cfg.learning.mode=="local_federated":
            
            if subset_concepts is None:
                original_c_names = datasets[0].c_info['names']
                c_info = datasets[0].c_info
                total_clients = cfg.learning.n_clients*cfg.learning.subgraphs.get('dataset_client_multiplier', 1)
            else:
                #order subset_concepts according to node_order
                if node_order is not None:
                    subset_concepts = [name for name in node_order if name in subset_concepts]
                original_c_names = subset_concepts
                c_info = {'names': subset_concepts, 'cardinality': [datasets[0].c_info['cardinality'][datasets[0].c_info['names'].index(name)] for name in subset_concepts]}
                total_clients = len(subset_clients)

            path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
            # just initialize it with dataset[0], then I'll update configuration and model with the different clients
            input_size = datasets[0].data["train"].X.shape[-1] if datasets[0].data["train"].X is not None else None
            #input_size = dict()
            # The list of names for in-distribution concepts (concepts that the client has in its subgraph)
            c_names_id = dict()
            #c_cardinality = dict()
            # The list of names for out-of-distribution concepts (concepts that the client has not in its subgraph)
            c_names_ood = dict()
            # The list of names for all concepts (in-distribution and out-of-distribution)
            c_names_all = original_c_names

            for id in range(1, total_clients+1):
                if subset_concepts is None:
                    if len(datasets)>1:
                        dataset = datasets[(id-1) % len(datasets)]
                    else:
                        dataset = datasets[0]
                    available_concepts = dataset.c_info['names']
                else:
                    available_concepts = subset_concepts
                
                #input_size[id-1] = dataset.data["train"].X.shape[-1] if dataset.data["train"].X is not None else None
                subgraph_id = identify_subgraph(path, id)
                if subgraph_id is not None:
                    updated_c_names = subgraphs_concept_names['subgraph_'+subgraph_id]
                    c_names_id[id] = [name for name in available_concepts if name in updated_c_names]
                    c_names_ood[id] = [name for name in available_concepts if name not in updated_c_names]

            
            ## The list of names for out-of-distribution concepts (concepts that the client has never seen before)
            #c_names_ood = [name for name in dataset.c_info['names'] if name not in updated_c_names]
            #c_names_all = c_names + c_names_ood
    
        else:
            dataset = datasets[0]
            input_size = dataset.data["train"].X.shape[-1] if dataset.data["train"].X is not None else None
            #input_size = dataset.data["train"].X.shape[-1] if dataset.data["train"].X is not None else None
            c_info = dataset.c_info
            original_c_names = dataset.c_info['names']
            # For non-localized/federated learning, we assume all clients have the same concepts
            c_names_id = {1: original_c_names}  
            #c_cardinality = {1: dataset.c_info['cardinality']}
            c_names_ood = {1: []}  # No out-of-distribution concepts
            c_names_all = original_c_names  # All concepts are in-distribution

        cfg.engine.model.update(
            input_size =  input_size,
            output_size = datasets[0].y_info['cardinality'][0], # we assume single class classification
            c_info = c_info,
            y_info = datasets[0].y_info,
            c_name_index = {name: i for i, name in enumerate(c_names_all + datasets[0].y_info['names'])},
        )
        cfg.engine.update(
            c_names_id = c_names_id,
            c_names_ood = c_names_ood,
            c_names_all = c_names_all,
            c_name_index = {name: i for i, name in enumerate(c_names_all)},
            learning_modality = cfg.learning.mode
        )
        #if not( cfg.learning.mode=="local_federated" and len(datasets)>1):
        #    cfg.engine.model.update(input_size = datasets[0].data["train"].X.shape[-1] if datasets[0].data["train"].X is not None else None)

    return cfg


def maybe_update_config_with_graph(cfg: DictConfig, graph, interv_policy) -> DictConfig:
    """ can be used to update the config based on the graph """
    if graph is not None:
        if model_is_causal(cfg.model):
            with open_dict(cfg):
                cfg.engine.model.update(
                    graph = graph.values.tolist(),
                    graph_labels = graph.index.tolist()
                )
    if interv_policy:
        with open_dict(cfg):
            cfg.engine.update(
                test_interv_policy = interv_policy
            )
    return cfg


def update_intervention_policy_and_graph(cfg, interv_policy, graph, datasets):

    if cfg is None:
        return None, None
    # Get the subgraph given the client id
    #for file in os.listdir(path):
    #    if ('trainset_'+str(cfg.client_id)) in file:
            # Get the substring between "subgraph_" and "."
    #        subgraph_id = file.split('subgraph_')[1].split('.')[0]
    #c_index = subgraphs['subgraph_'+subgraph_id]  
    #c_names = subgraphs_concept_names['subgraph_'+subgraph_id] 

    path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
    c_names = cfg.model.c_info['names']
    original_c_names = datasets[0].c_info['names']
    dict_names_to_index = {name: i for i, name in enumerate(original_c_names)}
    c_index = [dict_names_to_index[name] for name in c_names]

    # Update policy
    updated_policy = []
    #c_name_idx = {k:v for k, v in zip(c_index, range(len(c_names)))}
    for level in interv_policy:
        level_policy = []
        for i, node in enumerate(level):
            if node in c_index:
                #level_policy.append(c_name_idx[node])
                level_policy.append(node)
        if len(level_policy) > 0:
            updated_policy.append(level_policy)
                
    # Update graph
    updated_graph = graph.loc[c_names+cfg.model.y_info['names'], c_names+cfg.model.y_info['names']]
    return updated_policy, updated_graph

def update_config_from_data_subgroup_clients(cfg, subgroup_clients, datasets, subgraphs_concept_names, node_order ):
    
    if subgroup_clients is None:
        return None

    # Get concepts from subgroup clients subgraphs
    subgroup_concepts = set()
    for cid in subgroup_clients:
        path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
        subgraph_id = identify_subgraph(path, cid)
        if subgraph_id is not None:
            client_concepts = subgraphs_concept_names[f'subgraph_{subgraph_id}']
            subgroup_concepts.update(client_concepts)
    
    # Create a temporary cfg with filtered c_info for use with update_intervention_policy_and_graph
    subgroup_concepts_list = list(subgroup_concepts)
    cfg_predrift = copy.deepcopy(cfg)
    cfg_predrift = update_config_from_data(cfg_predrift, datasets, subgraphs_concept_names, subset_concepts = subgroup_concepts_list, subset_clients = subgroup_clients, node_order = node_order) 
       
    return cfg_predrift

def maybe_update_config_with_graph_subgroup_clients(cfg_predrift, predrift_clients, graph_predrift,policy_predrift, datasets):
         
    if predrift_clients is None:
        return cfg_predrift
    
    # Update cfg with filtered graph and policy
    cfg_predrift = maybe_update_config_with_graph(cfg_predrift, graph_predrift, policy_predrift)

    assert cfg_predrift.engine.model.graph_labels == list(cfg_predrift.model.c_name_index.keys())
    # replace in cfg_predrift.test_interv_policy the indices with respect to the current graph
    if cfg_predrift.engine.get('test_interv_policy', None) is not None:
        name_to_index = cfg_predrift.model.c_name_index
        original_index_to_name = {i: name for i, name in enumerate(datasets[0].c_info['names'])}
        updated_policy = []
        for level in cfg_predrift.engine.test_interv_policy:
            level_policy = []
            for node in level:
                node_name = original_index_to_name[node]
                if node_name in cfg_predrift.engine.model.graph_labels:
                    level_policy.append(name_to_index[node_name])
            if len(level_policy) > 0:
                 updated_policy.append(level_policy)
        with open_dict(cfg_predrift):
            cfg_predrift.engine.test_interv_policy = updated_policy

    return cfg_predrift


def filter_dataloaders_by_concepts(train_dataloaders, val_dataloaders, cfg_predrift, subgroup_clients, subgraphs_concept_names, all_concept_names, cache_path):
    """
    Filter dataloaders for clients in subgroup_clients to keep only concepts from their subgraphs.
    Uses a custom collate_fn to filter concepts dynamically during batch creation.
    
    Args:
        train_dataloaders: list of training DataLoaders
        val_dataloaders: list of validation DataLoaders
        subgroup_clients: list of client IDs to filter (1-indexed)
        subgraphs_concept_names: dict mapping subgraph IDs to concept names
        all_concept_names: list of all concept names in original order
        cache_path: path to cache directory for identifying subgraphs
    
    Returns:
        tuple: (filtered_train_dataloaders, filtered_val_dataloaders)
    """
    from src.data.utils import create_filtering_collate_fn

    if cfg_predrift is None or subgroup_clients is None:
        return train_dataloaders, val_dataloaders
    
    # Compute subgroup_concepts from subgraphs of clients in subgroup_clients
    subgroup_concepts = set()
    for cid in subgroup_clients:
        subgraph_id = identify_subgraph(cache_path, cid)
        if subgraph_id is not None:
            client_concepts = subgraphs_concept_names[f'subgraph_{subgraph_id}']
            subgroup_concepts.update(client_concepts)
    
    # Create concept mask - indices to keep
    concept_indices_to_keep = [i for i, name in enumerate(all_concept_names) if name in subgroup_concepts]
    
    print(f"\033[96m[filter_dataloaders] Filtering concepts for clients {subgroup_clients}\033[0m")
    print(f"\033[96m[filter_dataloaders] Keeping {len(concept_indices_to_keep)}/{len(all_concept_names)} concepts: {subgroup_concepts}\033[0m")
    
    for cid in subgroup_clients:
        loader_idx = cid - 1  # convert from 1-indexed to 0-indexed
        
        if loader_idx >= len(train_dataloaders):
            warnings.warn(f"Client {cid} not found in dataloaders (index {loader_idx} >= {len(train_dataloaders)})")
            continue
            
        # Filter training dataloader by wrapping collate_fn
        if train_dataloaders[loader_idx] is not None:
            old_loader = train_dataloaders[loader_idx]
            dataset = old_loader.dataset
            
            # Create filtering collate function
            original_collate_fn = old_loader.collate_fn
            filtering_collate_fn = create_filtering_collate_fn(original_collate_fn, concept_indices_to_keep, cfg_predrift, all_concept_names)
            
            # Create new DataLoader with filtering collate_fn
            train_dataloaders[loader_idx] = DataLoader(
                dataset,
                batch_size=old_loader.batch_size,
                shuffle=True if hasattr(old_loader.sampler, '_shuffle') else False,
                num_workers=old_loader.num_workers,
                pin_memory=old_loader.pin_memory,
                drop_last=old_loader.drop_last,
                collate_fn=filtering_collate_fn
            )           
        
        # Filter validation dataloader by wrapping collate_fn
        if val_dataloaders[loader_idx] is not None:
            old_loader = val_dataloaders[loader_idx]
            dataset = old_loader.dataset
            
            # Create filtering collate function
            original_collate_fn = old_loader.collate_fn
            filtering_collate_fn = create_filtering_collate_fn(original_collate_fn, concept_indices_to_keep, cfg_predrift, all_concept_names)
            
            # Create new DataLoader with filtering collate_fn
            val_dataloaders[loader_idx] = DataLoader(
                dataset,
                batch_size=old_loader.batch_size,
                shuffle=False,
                num_workers=old_loader.num_workers,
                pin_memory=old_loader.pin_memory,
                drop_last=old_loader.drop_last,
                collate_fn=filtering_collate_fn
            )
    
    return train_dataloaders, val_dataloaders

def get_parents(graph, i):
    # get the indices of the parents of the node i
    return graph[:,i].nonzero().squeeze(1)


def get_roots(graph):
    return graph.sum(dim=0) == 0


def dfs(node, adj_matrix, visited, stack, remove):
    visited[node] = True
    stack[node] = True
    for neighbor in range(len(adj_matrix)):
        if adj_matrix[neighbor][node] == 1:
            if not visited[neighbor]:
                if dfs(neighbor, adj_matrix, visited, stack, remove):
                    return True
            elif stack[neighbor]:
                if remove:
                    adj_matrix[neighbor][node] = 0
                    print(f'The cycle has been broken by removing the edge: {neighbor} -> {node}')
                return True
    stack[node] = False
    return False


def contains_cycle(adj_matrix):
    visited = [False] * len(adj_matrix)
    stack = [False] * len(adj_matrix)
    for node in range(len(adj_matrix)):
        if not visited[node]:
            if dfs(node, adj_matrix, visited, stack, False):
                return True
    return False


def remove_cycles(graph, start_node):
    """
    This function removes the cycles in the graph by removing the last visited edges
    before the cycle is detected
    Args:
        graph (Dataframe): the adjacency matrix of the graph
        start_node: the index of the task node
    Returns:
        graph (Dataframe): the adjacency matrix of the graph without cycles
    """
    adj_matrix = graph.values
    if contains_cycle(adj_matrix):
        while contains_cycle(adj_matrix):
            visited = [False] * len(adj_matrix)
            stack = [False] * len(adj_matrix)
            dfs(start_node, adj_matrix, visited, stack, True)
    else:
        print('there are no cycles in the graph, therefore the graph is left untouched')
    graph = pd.DataFrame(adj_matrix, index=graph.index, columns=graph.columns, dtype=int)
    return graph




def aggregate_graph_proposals(
    client_selection: Optional[List[float]],
    local_graphs: List[pd.DataFrame],
    weights: Optional[List[float]] = None,
    cfg: Optional[List[str]] = None,
    task_node: Optional[str] = None,
):
    if not local_graphs:
        raise ValueError("local_graphs cannot be empty.")
    if weights is None:
        weights = [1.0] * len(local_graphs)
    if len(weights) != len(local_graphs):
        raise ValueError("weights must match local_graphs length.")

    # Use client selection to filter local_graphs and weights
    if client_selection is None:
        return None, None
    
    selected_graphs = []
    selected_weights = []
    
    for client in client_selection:
        selected_graphs.append(local_graphs[client-1])
        selected_weights.append(weights[client-1])

    
    if not selected_graphs:
        raise ValueError("No clients selected based on client_selection.")
    
    local_graphs = selected_graphs
    weights = selected_weights
    
    # Get union of all nodes from local graphs
    all_nodes = set()
    for graph in local_graphs:
        all_nodes.update(graph.index.tolist())
        all_nodes.update(graph.columns.tolist())
    

    # Ensure all nodes from local graphs are in the same order of cfg.engine.c_names_index
    if cfg is not None and task_node is not None:
        node_order = cfg.engine.c_names_index.keys()
        node_order = [node for node in node_order if node in all_nodes]
        node_order.append(task_node)
    else:
        node_order = sorted(all_nodes)
    
    n_nodes = len(node_order)
    
    
    # Initialize matrices for aggregation
    # For each pair (i,j) where i<j, we track 3 options: i->j, j->i, no edge
    weighted_votes_forward = np.zeros((n_nodes, n_nodes))  # votes for edge i->j
    weighted_votes_backward = np.zeros((n_nodes, n_nodes))  # votes for edge j->i
    weighted_votes_noedge = np.zeros((n_nodes, n_nodes))  # votes for no edge
    total_weight_pairs = np.zeros((n_nodes, n_nodes))  # total weight for each pair
    
    # Aggregate votes from each local graph
    for graph_idx, local_graph in enumerate(local_graphs):
        client_weight = weights[graph_idx]
        
        for i, node_i in enumerate(node_order):
            for j, node_j in enumerate(node_order):
                if i >= j:  # Only consider i < j to avoid double counting
                    continue
                
                # Check presence of nodes in local graph
                has_couple = (node_i in local_graph.index and node_j in local_graph.columns)

                
                edge_i_j = 0
                edge_j_i = 0
                
                if has_couple:
                    edge_i_j = local_graph.loc[node_i, node_j]
                    edge_j_i = local_graph.loc[node_j, node_i]
                
                # Count this client's vote for the pair (i,j)
                if has_couple:
                    total_weight_pairs[i, j] += client_weight
                    
                    if edge_i_j == 1 and edge_j_i == 0:
                        # Vote for i->j
                        weighted_votes_forward[i, j] += client_weight
                    elif edge_j_i == 1 and edge_i_j == 0:
                        # Vote for j->i
                        weighted_votes_backward[i, j] += client_weight
                    else:
                        # Vote for no edge (both 0 or both 1 which is invalid, treat as no edge)
                        weighted_votes_noedge[i, j] += client_weight
                    
  
    # Initialize adjacency matrix based on weighted majority voting
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    uncertain_edges = []
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Only process each pair once
            if total_weight_pairs[i, j] == 0:
                continue
            
            # Get votes for the 3 options
            vote_forward = weighted_votes_forward[i, j]
            vote_backward = weighted_votes_backward[i, j]
            vote_noedge = weighted_votes_noedge[i, j]
            
            # Find the option with maximum votes (weighted majority wins)
            max_vote = max(vote_forward, vote_backward, vote_noedge)
            
            # Check for ties
            tie_count = sum([
                vote_forward == max_vote,
                vote_backward == max_vote,
                vote_noedge == max_vote
            ])
            
            if tie_count > 1:
                raise NotImplementedError("Conflict resolution for ties is not implemented.")

            else:
                # Use simple majority without conflict resolution
                if vote_forward == max_vote and vote_forward > 0:
                    adj[i, j] = 1
                elif vote_backward == max_vote and vote_backward > 0:
                    adj[j, i] = 1
    
    # Handle cycles if DAG is required
    graph_df = pd.DataFrame(adj, index=node_order, columns=node_order, dtype=int)
    start_node = n_nodes - 1
    graph_df = remove_cycles(graph_df, start_node)
    
    # Check that task node has at least one parent
    if task_node is not None:
        if task_node in graph_df.columns:
            # Get incoming edges to task node (parents)
            task_column = graph_df[task_node]
            num_parents = task_column.sum()
            
            if num_parents == 0:
                raise ValueError(f"Task node '{task_node}' has no parents in the aggregated graph. "
                               f"At least one parent is required for the task node.")

    meta = {
        "weights": pd.DataFrame(total_weight_pairs, index=node_order, columns=node_order),
        "votes_forward": pd.DataFrame(weighted_votes_forward, index=node_order, columns=node_order),
        "votes_backward": pd.DataFrame(weighted_votes_backward, index=node_order, columns=node_order),
        "votes_noedge": pd.DataFrame(weighted_votes_noedge, index=node_order, columns=node_order),
        "uncertain_edges": uncertain_edges,
    }
    return graph_df, meta


def remove_problematic_edges(graph, dataset):
    graph, virtual_c_names = common_cause_nodes(graph)
    if virtual_c_names:
        for split in dataset.data:
            dataset.data[split].c = torch.cat([torch.full((len(dataset.data[split].c), len(virtual_c_names)), float('nan')), 
                                            dataset.data[split].c], 
                                            dim=1)
        dataset.c_info['names'] = virtual_c_names + dataset.c_info['names']
        dataset.c_info['cardinality'] = [2]*len(virtual_c_names) + dataset.c_info['cardinality']
    else:
        print('therefore no virtual nodes where added and the ground truth graph and C are left untouched') 
    return graph, dataset


def get_graph_levels(graph, task_node):
    # extract the subgraph of the task node
    involeved_nodes, roots = get_task_graph(graph, task_node)
    # get the levels of the graph
    levels = get_levels(graph, involeved_nodes, roots)
    # check if the levels and roots are correct
    check_graph(levels, graph)
    return levels


def common_cause_nodes(graph):
    """
    This function removes the bi-directed and undirected edges in the graph
    and adds virtual nodes (roots) to model the common cause
    Args:
        adj_mat: the adjacency matrix of the graph
    Returns:
        adj_mat: the adjacency matrix of the graph with virtual
    """
    adj_mat = graph.values
    virtual_node_names = []
    pairs = []

    for i in range(len(adj_mat)):
        for j in range(i, len(adj_mat)):
            e_t = edge_type(adj_mat, i, j)
            if e_t in ['i<->j', 'i-j']: # if edge is undirected or bi-directed
                adj_mat[i, j] = 0 # the edge is removed
                adj_mat[j, i] = 0
                print(f'The {e_t} edge between {i} and {j} has been removed')
                pairs.append((i, j))
    
    for cnt, pair in enumerate(pairs):
        i = pair[0]+cnt
        j = pair[1]+cnt
        vector = np.zeros(len(adj_mat)); vector[i] = 1; vector[j] = 1
        adj_mat = np.row_stack([vector, adj_mat])
        adj_mat = np.column_stack([np.zeros(len(adj_mat)), adj_mat]) 
        virtual_node_names.append(f'#virtual_{cnt}') # add virtual node name
        print(f'The virtual node {virtual_node_names[-1]} has been added')


    if len(virtual_node_names) == 0:
        print('no undirected or bidirected edges where found')
    else:
        labels = virtual_node_names + list(graph.index)
        graph = pd.DataFrame(adj_mat, index=labels, columns=labels, dtype=int)
    return graph, virtual_node_names


def get_task_graph(graph, task_node):
    # start from the last in the graph, i.e., the index of the task
    branches = [[task_node]]
    nodes = [task_node]
    roots = []
    while True:
        parents = [get_parents(graph, node) for node in nodes]
        parents = torch.unique(torch.cat(parents)).tolist()
        # if parents are roots, add them later
        roots_to_add = [p for p in parents if len(get_parents(graph, p))==0]
        parents = [p for p in parents if p not in roots_to_add]
        roots = roots + roots_to_add
        if len(parents) == 0:
            break
        # remove nodes from previous branches if they are neessary at this step
        branches = [[node for node in level if node not in parents] for level in branches]
        # add nodes to the level
        branches.append(parents)
        nodes = parents
    roots = list(set(roots))
    involved_nodes = [node for branch in branches for node in branch]
    return involved_nodes, roots


def get_levels(graph, involved_nodes, roots):
    # start from the top of the graph
    # 1st level: get all nodes that requires only the roots to be computed
    # 2nd level: then get all nodes that only requires the root and the previous nodes
    # and so on...
    previous_level = roots
    levels = []
    while len(involved_nodes) > 0:
        level = [node for node in involved_nodes if set(get_parents(graph, node).tolist()).issubset(previous_level)]
        levels.append(level)
        previous_level = previous_level + level
        involved_nodes = [node for node in involved_nodes if node not in previous_level]
    levels.insert(0, roots)
    return levels


def check_graph(graph_levels, true_graph):
    roots = graph_levels[0].copy()
    levels = graph_levels[1:].copy()

    # check roots found are a subset of the total roots
    all_roots = torch.where(get_roots(true_graph))[0].tolist()
    assert set(roots).issubset(all_roots), \
        "The roots found are not a subset of the total roots"

    levels.insert(0, roots)
    # check all nodes appear only ones in the graph levels
    involved_nodes = [node for branch in levels for node in branch]
    assert [involved_nodes.count(node) for node in involved_nodes] == [1]*len(involved_nodes), \
        "Some nodes appear more than once in the graph levels"

    for level in levels:
        if level == roots:
            continue
        for node in level:
            parents = get_parents(true_graph, node).tolist()
            # check parents appear before their children in the graph levels
            for parent in parents:
                node_index = [levels.index(l) for l in levels if node in l]; assert len(node_index) == 1; node_index = node_index[0]
                parent_index = [levels.index(l) for l in levels if parent in l]; assert len(parent_index) == 1; parent_index = parent_index[0]
                assert node_index > parent_index, \
                f"Parent {parent} appear after their children {node} in the graph level {level}"

            # check at least one parent is in the previous level
            assert any([p in levels[levels.index(level)-1] for p in parents]), \
                f"At least one parent of {node} should be in the previous level"

            # check if the position of the nodes in the graph levels correspond to
            # the number of edges to a root
            node_index = [levels.index(l) for l in levels if node in l]; assert len(node_index) == 1; node_index = node_index[0]
            len_node_to_roots = 0
            while True:
                len_node_to_roots += 1
                if set(parents).issubset(set(roots)):
                    break
                parents = [get_parents(true_graph, parent) for parent in parents]
                parents = torch.unique(torch.cat(parents)).tolist()
            assert node_index == len_node_to_roots, \
                f"The position of the node {node} in the graph levels is not correct"
  
                    
def get_intervention_policy(graph, y_index):
    # get the levels of the graph
    torch_values_graph = torch.tensor(graph.values)
    levels = get_graph_levels(torch_values_graph, y_index)
    # remove the task node
    levels = levels[:-1]
    # get the names of the nodes
    names = list(graph.index)
    level_names = [[names[i] for i in level] for level in levels]

    # filter virtual roots from intervention policy
    levels = [[node for node in level if '#virtual_' not in names[node]] for level in levels]
    level_names = [[names[i] for i in level] for level in levels] 
    return levels, level_names


def extract_between(text, split):
    if split=='train':
        substring1 = 'trainset_'
    elif split=='val':
        substring1 = 'valset_'
    elif split=='test':
        substring1 = 'testset_'
    substring2 = '_subgraph'
    start = text.find(substring1)
    end = text.find(substring2, start + len(substring1))
    
    if start != -1 and end != -1:
        return text[start + len(substring1):end]
    else:
        return None
   
    
def get_split_paths(cfg, path):
    combined_dataset = OmegaConf.select(cfg, 'combined_datasets.other_datasets', default=None)
    root = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
    if combined_dataset is None:
        test_path = os.path.join(root, f"test.pkl")

    for file in os.listdir(path):
        if extract_between(file, 'train') == str(cfg.client_id):
            train_path = os.path.join(path, file)
        if extract_between(file, 'val') == str(cfg.client_id):
            val_path = os.path.join(path, file)
        
        if combined_dataset is not None:
            if extract_between(file, 'test') == str(cfg.client_id):
                test_path = os.path.join(path, file)


    
    return train_path, val_path, test_path

# to compact with the previous one
def get_split_paths_fl(cfg, path, client_id):
    combined_dataset = OmegaConf.select(cfg, 'combined_datasets.other_datasets', default=None)
    for file in os.listdir(path):
        if extract_between(file, 'train') == str(client_id):
            train_path = os.path.join(path, file)
        if extract_between(file, 'val') == str(client_id):
            val_path = os.path.join(path, file)

        if combined_dataset is not None:
            if extract_between(file, 'test') == str(client_id):
                test_path = os.path.join(path, file)
        else:
            root = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
            test_path = os.path.join(root, f"test.pkl")

    return train_path, val_path, test_path


def seed_everything(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# plot and save plot on server side
def plot_loss_and_accuracy(
        loss: List[float],
        accuracy: List[float],
        show: bool = True,
        fold=0):
    
    # # Plot loss separately
    plt.figure(figsize=(12, 6))
    plt.plot(loss, label='Loss', color='blue')
    min_loss_index = loss.index(min(loss))
    plt.scatter(min_loss_index, loss[min_loss_index], color='red', marker='*', s=100, label='Min Loss')
    
    # Labels and title for loss
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Distributed Loss (Weighted Average on Test-Set)')
    plt.legend()
    
    # Save the loss plot
    loss_plot_path = f"images/distributed_loss.png"
    plt.savefig(loss_plot_path)
    if show:
        plt.show()
    else:
        plt.close()

    # Plot accuracy separately
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy, label='Accuracy', color='green')
    max_accuracy_index = accuracy.index(max(accuracy))
    plt.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=100, label='Max Accuracy')
    
    # Labels and title for accuracy
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Distributed Accuracy (Weighted Average on Test-Set)')
    plt.legend()
    
    # Save the accuracy plot
    accuracy_plot_path = f"images/distributed_accuracy.png"
    plt.savefig(accuracy_plot_path)
    if show:
        plt.show()
    else:
        plt.close()

    # Print out server-side information
    print(f"\n\033[1;34mServer Side\033[0m \nMinimum Loss occurred at round {min_loss_index + 1} with a loss value of {loss[min_loss_index]:.3f} \nMaximum Accuracy occurred at round {max_accuracy_index + 1} with an accuracy value of {accuracy[max_accuracy_index]*100:.2f}\n")
    
    return min_loss_index + 1, max_accuracy_index + 1


# create folders
def create_folders():
    # print current working directory
    print("Current working directory:", os.getcwd())
    os.makedirs('images', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('histories', exist_ok=True)


def maybe_freeze_parameters(train_dataloader,  y_to_freeze, model, learning, freezing = True):
    """
    This function freezes the model parameters related to the concepts masked for the client when learning = 'federated'
    Args:
        c (torch.Tensor): concept labels
        model (nn.Module): the model
        learning (str): the learning mode
        freezing (bool): whether to freeze the parameters or not     
    Returns:
        None
    """
    # voglio applicare filtering collate a train_dataloader
    new_dataloader = copy.deepcopy(train_dataloader)
    batch = next(iter(new_dataloader))
    c = batch['c'] if 'c' in batch else None


    if (learning == "local_federated" or learning=="federated") and freezing:

        c_indices_to_freeze = torch.where(c[0] == -1)[0].tolist()
        if y_to_freeze:
            y_index = len(model.c_info['names'])  # assuming y is after all concepts
            c_indices_to_freeze.append(int(y_index))

        for param in model.parameters():
            param.requires_grad = True

        if model.name=="cbm_linear" or model.name =="cbm_mlp":
            c_keys = list(model.c_mlp.keys())
            c_to_freeze = [c_keys[i] for i in c_indices_to_freeze if 0 <= i < len(c_keys)]
            for name, mlp in model.c_mlp.items():
                    if name in c_to_freeze:
                        for param in mlp.parameters():
                            param.requires_grad = False
            if y_to_freeze and hasattr(model, "decoder"):
                for param in model.decoder.parameters():
                    param.requires_grad = False
            print("Parameters frozen for concepts:", c_to_freeze)

        if model.name=="cem":
            c_keys = list(model.concept_encoders.keys())
            c_to_freeze = [c_keys[i] for i in c_indices_to_freeze if 0 <= i < len(c_keys)]
            for name, concept_encoder in model.concept_encoders.items():
                if name in c_to_freeze:
                    for param in concept_encoder.parameters():
                        param.requires_grad = False
            if y_to_freeze and hasattr(model, "decoder"):
                for param in model.decoder.parameters():
                    param.requires_grad = False
            print("Parameters frozen for concepts:", c_to_freeze)
        
        if model.name=="c2bm":
            c_keys = list(model.concept_encoders.keys())
            c_to_freeze = [c_keys[i] for i in c_indices_to_freeze if 0 <= i < len(c_keys)]

            for name, concept_encoder in model.concept_encoders.items():
                if name in c_to_freeze:
                    for param in concept_encoder.parameters():
                        param.requires_grad = False

            for _, level in model.propagators.items(): 
                for c_name, propagator in level.items():
                    if c_name in c_to_freeze:
                        for param in propagator.parameters():
                            param.requires_grad = False

            print("Parameters frozen for concepts:", c_to_freeze)

            # check
            #for name, param in model.named_parameters():
            #    print(f"{name}: requires_grad = {param.requires_grad}")

    return None


def parameters_to_1d(parameters):
    return np.concatenate([x.flatten() for x in parameters])


def score_blackbox_batch(batch, model, cfg, use_concepts=True):
    '''
    Computes membership inference attack scores for a batch by 
    computing the negative log-likelihood of the model's predictions.
    '''
    with torch.no_grad():
        x = batch['x'].to(cfg.device)
        y = batch['y'].to(cfg.device)
        c = batch['c'].to(cfg.device) if 'c' in batch else None

        # Forward pass
        y_hat, c_hat = model(x)
        y_hat_loss, c_hat_loss = model.filter_output_for_loss(y_hat, c_hat)
        if not use_concepts: # supervision only on y, no concepts
            losses = model._compute_task_loss(y_hat_loss, y, reduction='none')
        else:
            losses = model.loss(y_hat_loss, y, c_hat_loss, c, reduction='none')
        return -losses.cpu().numpy()


# def project_update_to_trainable(client_update: torch.Tensor, model: torch.nn.Module, device=None):
#     """
#     Slice a full 1D update vector (all params) down to only the segments
#     corresponding to trainable (requires_grad=True) parameters, in the same
#     order used by model.parameters(). Re-normalises the result.
#     """
#     if device is None:
#         device = client_update.device
#     pieces = []
#     offset = 0
#     for p in model.parameters():
#         n = p.numel()
#         if p.requires_grad:
#             pieces.append(client_update[offset:offset + n])
#         offset += n
#     if not pieces:
#         raise ValueError("Model has no trainable parameters (requires_grad=True).")
#     v = torch.cat(pieces).to(device)
#     v = v / (torch.linalg.norm(v) + 1e-12)
#     return v


def flat_trainable_params_tensor(model, device):
    parts = [p.detach().to(device).flatten() for p in model.parameters() if p.requires_grad]
    return torch.cat(parts) if parts else torch.tensor([], device=device)

def score_whitebox_batch(batch, model, client_update, cfg, use_concepts=True):
    '''
    Computes membership inference attack scores for a batch by 
    computing the inner product between the 'pseudogradient'
    represented by each client update and the true gradients
    for each sample in the batch.
    '''

    # --- project update to trainable params & normalise ---
    # client_update = project_update_to_trainable(client_update, model, device=cfg.device)

    # Move data
    x = batch['x'].to(cfg.device)
    y = batch['y'].to(cfg.device)
    c = batch['c'].to(cfg.device) if 'c' in batch else None

    # Pre-allocate scores array on GPU
    batch_size = len(y)
    scores = torch.zeros(batch_size, device=cfg.device) # Zeroing directly on GPU

    # Forward pass
    y_hat, c_hat = model(x)
    y_hat_loss, c_hat_loss = model.filter_output_for_loss(y_hat, c_hat)
    if not use_concepts: # supervision only on y, no concepts
        losses = model._compute_task_loss(y_hat_loss, y, reduction='none')
    else:
        losses = model.loss(y_hat_loss, y, c_hat_loss, c, reduction='none')  # Use 'none' to get per-sample losses
    
    params = [p for p in model.parameters() if p.requires_grad] # ADDED line because some parameters may be frozen

    for i, loss in enumerate(losses):
        grad_vector = torch.autograd.grad(
            loss, 
            # model.parameters(),
            params,
            retain_graph=True,
            allow_unused=True # Allow unused gradients, otherwise an error is raised while computing the gradients.
        )

        # Since some of the gradient vectors may be None (due to some part of the model being frozen), 
        # we replace them with zero vectors.
        grad_vector = [
            g if g is not None else torch.zeros_like(p)
            # for g, p in zip(grad_vector, model.parameters())
            for g, p in zip(grad_vector, params)
        ]
        
        # Flatten and concatenate gradients
        with torch.no_grad():
            flat_grad = torch.cat([g.flatten() for g in grad_vector])
            scores[i] = -torch.dot(flat_grad, client_update) # Dot product directly on GPU
    
    return scores.cpu().numpy()


class _SIASampleDataset(Dataset):
    """ A thin wrapper to expose (sample, target, c, cid) or dict+cid. """

    def __init__(self, base_ds: Dataset, base_indices: List[int], cid: int):
        self.base_ds = base_ds
        self.indices = base_indices
        self.cid = cid
        if hasattr(base_ds, "c"):
            self.c = base_ds.c[base_indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.base_ds[self.indices[idx]]
        
        # dict → copy and add key(s)
        if isinstance(item, dict):
            sample = dict(item)  # shallow copy
            sample["cid"] = self.cid
            if hasattr(self.base_ds, "c"):
                sample["c"] = self.base_ds.c[self.indices[idx]]
            return sample

        else:
            raise RuntimeError("Unsupported item type in _SIASampleDataset: expected dict, got {}".format(type(item)))


def _clone_loader(template_loader, dataset, *, shuffle):
    """Return a DataLoader with the same runtime parameters as `template_loader`."""
    return DataLoader(
        dataset,
        batch_size=template_loader.batch_size,
        shuffle=shuffle,
        num_workers=template_loader.num_workers,
        pin_memory=template_loader.pin_memory,
        drop_last=template_loader.drop_last,
        collate_fn=template_loader.collate_fn,
        worker_init_fn=template_loader.worker_init_fn,
        persistent_workers=getattr(template_loader, "persistent_workers", False),
    )


def dataprocess_auditing(train_dataloaders, n_clients, cfg):
    """
    Produces:
      • subsampled_train_loaders : list[DataLoader]
      • canary_loaders          : list[DataLoader]
      • true_in_outs            : list[torch.BoolTensor]
      • sia_loader              : DataLoader   
    """
    frac = cfg.learning.settings.canary_fraction
    sia_k = cfg.learning.settings.sia_samples_per_client

    subsampled_train_loaders, canary_loaders, true_in_outs = [], [], []
    sia_datasets: List[_SIASampleDataset] = []

    for cid in range(n_clients):
        base_loader  = train_dataloaders[cid]
        base_dataset = base_loader.dataset
        n_total      = len(base_dataset)

        # ---------- 1. split into canaries / non-canaries -------------------
        n_canaries   = max(1, int(round(frac * n_total)))
        n_non_can    = n_total - n_canaries
        canaries_ds, non_canaries_ds = random_split(
            base_dataset, [n_canaries, n_non_can]
        )

        if hasattr(base_dataset, "c"):
            full_c = base_dataset.c
            canaries_ds.c      = full_c[canaries_ds.indices]
            non_canaries_ds.c  = full_c[non_canaries_ds.indices]

        # ---------- 2. subsample canaries "in" / "out" ----------------------
        true_in_out = torch.rand(n_canaries) < 0.5
        keep_idx    = true_in_out.nonzero(as_tuple=False).squeeze(1)

        canaries_in_ds = Subset(canaries_ds, keep_idx)
        if hasattr(canaries_ds, "c"):
            canaries_in_ds.c = canaries_ds.c[keep_idx]

        # ---------- 3. build *training* dataset & loaders -------------------
        combined_ds = ConcatDataset([non_canaries_ds, canaries_in_ds])
        if hasattr(base_dataset, "c"):
            combined_ds.c = torch.cat([non_canaries_ds.c, canaries_in_ds.c])

        subsampled_train_loaders.append(
            _clone_loader(base_loader, combined_ds, shuffle=True)
        )
        canary_loaders.append(
            _clone_loader(base_loader, canaries_ds, shuffle=False)
        )
        true_in_outs.append(true_in_out)
        cfg.learning.settings.n_canaries = n_canaries

        # ---------- 4. build SIA samples for this client --------------------
        if cfg.learning.settings.sia:
            pool = non_canaries_ds.indices    # only non-canaries
            sel  = random.sample(pool, min(sia_k, len(pool)))
            sia_datasets.append(_SIASampleDataset(base_dataset, sel, cid))

    # ---------- 5. concatenate SIA datasets from all clients ---------------
    if sia_datasets:
        sia_dataset = ConcatDataset(sia_datasets)
        if hasattr(train_dataloaders[0].dataset, "c"):
            sia_dataset.c = torch.cat([d.c for d in sia_datasets])

        sia_loader = _clone_loader(
            train_dataloaders[0],   # template with proper collate_fn
            sia_dataset,
            shuffle=False           # keep deterministic order
        )
    else:
        sia_loader = None

    return subsampled_train_loaders, canary_loaders, true_in_outs, sia_loader


def p_value_DP_audit(m, r, v, eps, delta):
    """
    Computes the p-value for the audit hypothesis test under differential privacy.
    The implementation follows Appendix D from:
    https://arxiv.org/pdf/2305.08846

    Args:
        m (int): Total number of examples, each included independently with probability 0.5.
        r (int): Number of guesses made by the auditor (excluding abstentions).
        v (int): Number of correct guesses by the auditor.
        eps (float): Differential privacy epsilon parameter.
        delta (float): Differential privacy delta parameter.

    Returns:
        float: p-value, i.e., the probability of observing at least v correct guesses under the null hypothesis.
    """
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1/(1+math.exp(-eps)) # accuracy of eps-DP randomized response
    beta = scipy.stats.binom.sf(v-1, r, q) # = P[Binomial(r, q) >= v]
    alpha = 0
    sum = 0 # = P[v > Binomial(r, q) >= v - i]
    for i in range(1, v + 1):
        sum = sum + scipy.stats.binom.pmf(v - i, r, q)
        if sum > i * alpha:
            alpha = sum / i
    p = beta + alpha * delta * 2 * m
    return min(p, 1)


def get_eps_audit(m, r, v, delta, p):
    """
    Computes a lower bound on epsilon (eps) for which the observed audit results
    would not be (eps, delta)-differentially private at confidence level 1-p.

    Args:
        m (int): Total number of examples, each included independently with probability 0.5.
        r (int): Number of guesses made by the auditor (excluding abstentions).
        v (int): Number of correct guesses by the auditor.
        delta (float): Differential privacy delta parameter.
        p (float): 1 - confidence level (e.g., p=0.05 for 95% confidence).

    Returns:
        float: Lower bound on epsilon (eps) such that the mechanism is not (eps, delta)-DP
               with probability at least 1-p.
    """
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p < 1
    eps_min = 0 # maintain p_value_DP(eps_min) < p
    eps_max = 1 # maintain p_value_DP(eps_max) >= p
    while p_value_DP_audit(m, r, v, eps_max, delta) < p: eps_max = eps_max + 1
    for _ in range(30): # binary search
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min


def evaluate_privacy(scores, true_in_out, cfg):
    n_canaries = cfg.learning.settings.n_canaries
    ground_truth = copy.deepcopy(true_in_out)
    score_indices_sorted = np.argsort(scores)[::-1]
    classified_in = score_indices_sorted[:int(n_canaries * cfg.learning.settings.k_plus + 1)]
    classified_out = score_indices_sorted[int(n_canaries * (1 - cfg.learning.settings.k_min) + 1):]
    abstained = np.setdiff1d(score_indices_sorted, np.concatenate((classified_in, classified_out)))
    classification = np.zeros(n_canaries)
    classification[classified_in] = 1
    classification[abstained] = 2
    ground_truth[abstained] = 2
    W = ground_truth == classification
    num_correct = W.sum() - len(abstained)
    accuracy_mia = num_correct / (n_canaries - len(abstained))
    
    # tpr = np.sum(classification == true_in_out) / len(canaries_in_idx)
    # tnr = np.sum((1 - classification) == (1 - true_in_out)) / len(canaries_out_idx)
    # fpr = np.sum(classification == (1 - true_in_out)) / len(canaries_out_idx)
    # fnr = np.sum((1 - classification) == true_in_out) / len(canaries_in_idx)

    # compute empirical privacy estimate, which should be < epsilon w/ high probability
    privacy_estimate = get_eps_audit(
        m=n_canaries,
        r=n_canaries - len(abstained),
        v=num_correct,
        delta=cfg.learning.settings.delta,
        p=0.05)
    
    # Kairouz privacy estimate from https://proceedings.mlr.press/v37/kairouz15.html
    # privacy_estimate = np.max([np.log(1 - cfg.delta - fpr) - np.log(fnr), 
                        # np.log(1 - cfg.delta - fnr) - np.log(fpr)])
                        
    return accuracy_mia, privacy_estimate


def initialize_mia_results(n_clients):
    """ Initialize dictionaries to store MIA results for each client """
    if n_clients <= 0:
        raise ValueError("Number of clients must be greater than 0")

    mia_accuracies = {"whitebox": {}, "blackbox": {}, "blackbox_shadow": {}} # "blackbox_concept": {}
    mia_epsilons = {"whitebox": {}, "blackbox": {}, "blackbox_shadow": {}} #  "blackbox_concept": {}
    for cid in range(n_clients):
        mia_accuracies["whitebox"][cid] = []
        mia_accuracies["blackbox"][cid] = []
        # mia_accuracies["blackbox_concept"][cid] = []
        mia_accuracies["blackbox_shadow"][cid] = []
        mia_epsilons["whitebox"][cid] = []
        mia_epsilons["blackbox"][cid] = []
        # mia_epsilons["blackbox_concept"][cid] = []
        mia_epsilons["blackbox_shadow"][cid] = []

    return mia_accuracies, mia_epsilons


@torch.no_grad()
def _sia_batch_loss(batch, model, cfg, use_concepts=True):
    """
    Returns the per-sample task&concept loss for a batch, **CPU numpy**.
    Accepts either tuple- or dict-style batches produced by `sia_loader`.
    """
    # ------------- unpack --------------------------------------------------
    if isinstance(batch, dict):
        x = batch["x"].to(cfg.device)
        y = batch["y"].to(cfg.device)
        c = batch.get("c")
        c = c.to(cfg.device) if c is not None else None
    else:                                   # tuple or list
        raise RuntimeError("Unsupported batch type in _sia_batch_loss: expected dict, got {}".format(type(batch)))
    
    x, y = x.to(cfg.device), y.to(cfg.device)

    # ------------- forward & loss -----------------------------------------
    y_hat, c_hat = model(x)
    y_hat_loss, c_hat_loss = model.filter_output_for_loss(y_hat, c_hat)
    if not use_concepts: # supervision only on y, no concepts
        losses = model._compute_task_loss(y_hat_loss, y, reduction="none", ignore_index=-1)
    else:
        losses = model.loss(y_hat_loss, y, c_hat_loss, c, reduction="none", ignore_index=-1)  # shape = (B,)

    return losses.detach().cpu().numpy() 


def _sia_true_cids(dataset):
    """Return a NumPy array with the cid for every sample in the dataset."""
    if isinstance(dataset, ConcatDataset):
        cids = []
        for sub in dataset.datasets:          # each is _SIASampleDataset
            cids.extend([sub.cid] * len(sub))
        return np.asarray(cids, dtype=np.int64)
    else:                                     # single _SIASampleDataset
        return np.asarray([dataset.cid] * len(dataset), dtype=np.int64)
    

def run_sia_attack(
    local_engine,
    sia_loader: torch.utils.data.DataLoader,
    client_params: List[Tuple[List[torch.Tensor], int]],
    cfg,
    use_concepts: bool = True,
) -> float:
    """
    Args
    ----
    sia_loader     : the joint loader built in `dataprocess_auditing`
    client_params  : list[(parameters_list, n_samples)] – exactly what you
                     already build for FedAvg
    cfg            : Hydra / OmegaConf config (for `device`, `engine`, ...)

    Returns
    -------
    accuracy : proportion of samples whose source client is predicted
               correctly (lower loss → chosen client)
    """
    n_clients    = len(client_params)
    n_samples    = len(sia_loader.dataset)
    losses_all   = np.empty((n_samples, n_clients), dtype=np.float32)

    # We'll fill losses column-wise (one client at a time)
    for cid in range(n_clients):
        set_parameters(local_engine, client_params[cid][0])
        local_engine.model.to(cfg.device)
        local_engine.model.eval()

        # collect losses for *all* SIA samples with this client’s model
        losses_client = []
        for batch in sia_loader:            # the loader is NOT shuffled
            losses_client.append(_sia_batch_loss(batch, local_engine.model, cfg, use_concepts=use_concepts))
        losses_all[:, cid] = np.concatenate(losses_client)

    # ----------- prediction & accuracy -------------------------------------
    pred_cid   = losses_all.argmin(axis=1)           # (n_samples,)
    true_cid = _sia_true_cids(sia_loader.dataset)

    accuracy   = (pred_cid == true_cid).mean()

    return accuracy


# --------------------- BLACK-BOX MIA HELPERS (concept-entropy + shadow MLP) ---------------------
def _as_label_probs(y_like: torch.Tensor) -> torch.Tensor:
    """
    Ensure we have class probabilities on the last dimension.
    If the last dim already (approximately) sums to 1, treat as probs; else softmax.
    """
    s = y_like.sum(dim=-1, keepdim=True)
    if torch.allclose(s.mean(), torch.ones_like(s.mean()), atol=1e-3, rtol=1e-3):
        p = y_like
    else:
        p = torch.softmax(y_like, dim=-1)
    return p.clamp(1e-12, 1 - 1e-12)

def _iter_concept_probs(c_hat) -> list[torch.Tensor]:
    """
    Yield a list of concept probability tensors, each shaped (B, C_k),
    without assuming equal cardinality across concepts. Accepts:
      - dict[str, Tensor] where each Tensor is (B, C_k)
      - Tensor of shape (B, K, C) → K concepts with C categories each
      - list/tuple of Tensors [(B, C_k), ...]
    We ensure probabilities per concept using softmax if needed.
    """
    if c_hat is None:
        return []
    outs = []
    if isinstance(c_hat, dict):
        for _, v in sorted(c_hat.items(), key=lambda kv: kv[0]):
            if not isinstance(v, torch.Tensor):
                continue
            outs.append(_as_label_probs(v))
    elif isinstance(c_hat, (list, tuple)):
        for v in c_hat:
            if isinstance(v, torch.Tensor):
                outs.append(_as_label_probs(v))
    elif isinstance(c_hat, torch.Tensor):
        if c_hat.ndim == 3:  # (B, K, C)
            B, K, C = c_hat.shape
            v = _as_label_probs(c_hat)              # (B, K, C)
            outs = [v[:, k, :] for k in range(K)]   # list of (B, C)
        elif c_hat.ndim == 2:  # (B, C) single concept
            outs = [_as_label_probs(c_hat)]
        else:
            outs = []
    else:
        outs = []
    return outs

@torch.no_grad()
def score_blackbox_concept_entropy_loader(loader, model, cfg):
    """
    Compute a black-box score using ONLY concept outputs:
      score = - mean categorical entropy across concepts (higher → more likely IN).
    Falls back to -entropy(label probs) if concepts are unavailable.
    Returns a NumPy array of length len(loader.dataset) in loader order.
    """
    scores = []
    model.eval()
    for batch in loader:
        x = batch['x'].to(cfg.device)
        y_hat, c_hat = model(x)
        # try concept path first
        concept_list = _iter_concept_probs(c_hat)
        if len(concept_list) > 0:
            ents = []
            for p_k in concept_list:  # (B, C_k)
                ent_k = -(p_k * torch.log(p_k)).sum(dim=-1)  # (B,)
                ents.append(ent_k)
            ent_mat = torch.stack(ents, dim=-1)              # (B, K)
            ent_mean = ent_mat.mean(dim=-1)                  # (B,)
            scores.append((-ent_mean).detach().cpu().numpy())
        else:
            # fallback: label entropy
            y_probs = _as_label_probs(y_hat)
            ent = -(y_probs * torch.log(y_probs)).sum(dim=-1)  # (B,)
            scores.append((-ent).detach().cpu().numpy())
    return np.concatenate(scores, axis=0)

@torch.no_grad()
def score_blackbox_loss_loader(loader, model, cfg, use_concepts=True):
    """
    Wrapper around score_blackbox_batch that accumulates over the whole loader
    and returns a concatenated NumPy vector in loader order.
    """
    all_scores = []
    for batch in loader:
        all_scores.append(score_blackbox_batch(batch, model, cfg, use_concepts=use_concepts))
    return np.concatenate(all_scores, axis=0)

def _extract_bb_features_from_loader(
    loader,
    model,
    cfg,
    use_concepts: bool = True,
    raw_concept_probs: bool = False,
):
    """
    Build per-sample features for a shadow attacker, in loader order.

    Always includes LABEL-HEAD features (computed from probabilities):
      - y_conf (top1 prob), y_margin (p1 - p2), y_entropy, y_brier, -loss_task, y_correct(0/1)

    If concept outputs are available and use_concepts=True, also include:
      - concept confidence stats: mean/std over concepts of max-prob per concept
      - concept entropy   stats: mean/std over concepts of categorical entropy
      - fraction of concepts with conf >= 0.9
      - -loss_concept
      - (optional) raw concatenated concept probabilities (may overfit when data is small)

    Returns
    -------
    X : np.ndarray of shape (N, D)
    y_true : np.ndarray of shape (N,)
    """
    X = []
    y_true_all = []
    model.eval()
    
    for batch in loader:
        x = batch['x'].to(cfg.device)
        y = batch['y'].to(cfg.device)                   # (B,)
        c = batch.get('c')
        c = c.to(cfg.device) if c is not None else None

        # forward
        y_hat, c_hat = model(x)                         # probabilities already
        y_probs = _as_label_probs(y_hat)

        # base label features
        y_sorted, _ = torch.sort(y_probs, dim=-1, descending=True)
        y_conf   = y_sorted[:, 0]                                           # (B,)
        y_margin = (y_sorted[:, 0] - y_sorted[:, 1]) if y_probs.shape[-1] > 1 else y_sorted[:, 0]
        y_ent    = -(y_probs * torch.log(y_probs)).sum(dim=-1)              # (B,)
        y_onehot = F.one_hot(y.long(), num_classes=y_probs.shape[-1]).float()
        y_brier  = ((y_probs - y_onehot) ** 2).sum(dim=-1)                  # (B,)
        y_pred   = y_probs.argmax(dim=-1)
        y_corr   = (y_pred == y).float()                                    # (B,)

        # losses from model (task, concept, mixed)
        y_hat_loss, c_hat_loss = model.filter_output_for_loss(y_hat, c_hat)
        # ask for multi_output=True if supported; otherwise fall back gracefully
        try:
            loss_task, loss_concept, loss_mixed = model.loss(
                y_hat_loss, y, c_hat_loss, c, reduction='none', multi_output=True
            )
        except TypeError:
            loss = model.loss(y_hat_loss, y, c_hat_loss, c, reduction='none')
            loss_task, loss_concept, loss_mixed = loss, torch.zeros_like(loss), loss

        feats = [
            y_conf, y_margin, y_ent, y_brier,
            -loss_task, y_corr
        ]

        # use_concepts = False
        if use_concepts:
            concept_list = _iter_concept_probs(c_hat)  # list of (B, C_k)
            if len(concept_list) > 0:
                # per-concept confidence (max prob) and entropy
                confs = [p_k.max(dim=-1).values for p_k in concept_list]              # list of (B,)
                ents  = [-(p_k * torch.log(p_k)).sum(dim=-1) for p_k in concept_list] # list of (B,)
                confs_mat = torch.stack(confs, dim=-1)     # (B, K)
                ents_mat  = torch.stack(ents,  dim=-1)     # (B, K)
                near_bin  = (confs_mat >= 0.9).float().mean(dim=-1)  # (B,)

                feats += [
                    confs_mat.mean(dim=-1),   # (B,)
                    confs_mat.std(dim=-1),    # (B,)
                    ents_mat.mean(dim=-1),    # (B,)
                    ents_mat.std(dim=-1),     # (B,)
                    near_bin,                 # (B,)
                    -loss_concept,            # (B,)
                ]

                if raw_concept_probs:
                    # concatenate raw probs in a fixed order
                    concat = torch.cat(concept_list, dim=-1)   # (B, sum_k C_k)
                    feats.append(concat)

        # stack along feature dim
        F_batch = torch.cat([f.float().unsqueeze(-1) if f.dim() == 1 else f.float() for f in feats], dim=-1)  # (B, D)
        X.append(F_batch.detach().cpu())
        y_true_all.append(y.detach().cpu())

    X = torch.cat(X, dim=0).numpy()
    y_true = torch.cat(y_true_all, dim=0).long().numpy()
    return X, y_true

class _ShadowMLP(nn.Module):
    def __init__(self, d, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# def shadow_mlp_scores_loader(  # NOTA: HERE: questa funzione e' stata usata per asia
#     loader,
#     model,
#     cfg,
#     y_mem_labels: np.ndarray,
#     use_concepts: bool = True,
#     epochs: int = 100,
#     batch_size: int = 64,
#     lr: float = 1e-3,
#     k_folds: int = 10,
#     class_conditional: bool = True,
#     weight_decay: float = 1e-3,
#     dropout: float = 0.1,
#     standardize: bool = True,
#     raw_concept_probs: bool = False,
#     scores_whitebox_list: list = None
# ):
#     """
#     Train a shadow MLP with K-fold CV and return out-of-fold membership scores
#     for ALL samples in loader order (sigmoid logits).

#     If class_conditional=True, we train one attacker per true label and
#     use the class-specific attacker to score samples of that class.

#     Optionally, scores_whitebox_list (white-box scores) can be appended as an additional feature
#     to the attacker; they are concatenated and aligned in loader order and added as a new column to the features.
#     """
#     # Create a single-batch dataloader to avoid feature dimension mismatches
#     dataset = loader.dataset
#     single_batch_loader = torch.utils.data.DataLoader(
#         dataset, 
#         batch_size=len(dataset),  # Process all samples in one batch
#         shuffle=False,  # Preserve order
#         collate_fn=loader.collate_fn if hasattr(loader, 'collate_fn') else None
#     )
    
#     # extract features (+ y_true labels for class-conditional training)
#     X, y_true = _extract_bb_features_from_loader(
#         single_batch_loader, model, cfg,
#         use_concepts=use_concepts,
#         raw_concept_probs=raw_concept_probs
#     )
#     # ---- Optional: append white-box scores as an additional feature ----
#     y_mem_np = np.asarray(y_mem_labels)  # we'll possibly trim this if needed
#     if scores_whitebox_list is not None and len(scores_whitebox_list) > 0:
#         if isinstance(scores_whitebox_list, np.ndarray):
#             wb_scores = scores_whitebox_list.reshape(-1)
#         else:
#             wb_scores = np.concatenate([np.asarray(s).reshape(-1) for s in scores_whitebox_list], axis=0)
#         # align lengths if needed
#         if wb_scores.shape[0] != X.shape[0]:
#             min_n = min(wb_scores.shape[0], X.shape[0], y_true.shape[0], y_mem_np.shape[0])
#             print(f"\033[93m[shadow-mlp] Warning: length mismatch (X={X.shape[0]}, wb={wb_scores.shape[0]}). Trimming to {min_n}.\033[0m")
#             wb_scores = wb_scores[:min_n]
#             X = X[:min_n]
#             y_true = y_true[:min_n]
#             y_mem_np = y_mem_np[:min_n]
#         # append as a feature column
#         X = np.concatenate([X, wb_scores.astype(np.float32)[:, None]], axis=1)
#     y_mem = torch.from_numpy(y_mem_np.astype(np.float32))
#     n = len(y_mem)
#     scores = torch.zeros(n, dtype=torch.float32)

#     # helper to run CV for a given subset of indices
#     def _cv_scores_for_indices(indices: np.ndarray) -> torch.Tensor:
#         if len(indices) == 0:
#             return torch.zeros(0, dtype=torch.float32)
#         # create stratified folds across membership labels within the subset
#         idx0 = indices[y_mem_labels[indices] == 0]
#         idx1 = indices[y_mem_labels[indices] == 1]
#         # guard for tiny splits
#         folds0 = np.array_split(idx0, k_folds) if len(idx0) >= k_folds else [idx0]
#         folds1 = np.array_split(idx1, k_folds) if len(idx1) >= k_folds else [idx1]

#         out = torch.zeros(len(indices), dtype=torch.float32)
#         # map from absolute indices to local positions for writing back
#         abs_to_local = {abs_i: j for j, abs_i in enumerate(indices)}

#         num_folds = max(len(folds0), len(folds1))
#         for k in range(num_folds):
#             val_idx_abs = np.concatenate([
#                 folds0[k % len(folds0)] if len(folds0) > 0 else np.array([], dtype=int),
#                 folds1[k % len(folds1)] if len(folds1) > 0 else np.array([], dtype=int),
#             ])
#             train_idx_abs = np.setdiff1d(indices, val_idx_abs)

#             X_train = torch.from_numpy(X[train_idx_abs]).float().to(cfg.device)
#             y_train = y_mem[train_idx_abs].to(cfg.device)
#             X_val   = torch.from_numpy(X[val_idx_abs]).float().to(cfg.device)

#             # standardize per fold
#             if standardize and X_train.numel() > 0:
#                 mu = X_train.mean(dim=0, keepdim=True)
#                 sd = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
#                 X_train = (X_train - mu) / sd
#                 X_val   = (X_val   - mu) / sd

#             attacker = _ShadowMLP(X_train.shape[1], hidden=64, dropout=dropout).to(cfg.device)
#             pos = y_train.sum()
#             neg = len(y_train) - pos
#             pos_weight = (neg / (pos + 1e-8)).clamp(min=0.0, max=1e6)
#             criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#             optim = torch.optim.Adam(attacker.parameters(), lr=lr, weight_decay=weight_decay)

#             ds = torch.utils.data.TensorDataset(X_train, y_train)
#             print(ds)
#             print(len(ds))
#             dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

#             attacker.train()
#             best_val = float('inf')
#             patience_counter = 0
#             for _ in range(epochs):
#                 for xb, yb in dl:
#                     optim.zero_grad()
#                     logits = attacker(xb)
#                     loss = criterion(logits, yb)
#                     loss.backward()
#                     optim.step()

#                 with torch.no_grad():
#                     attacker.eval()
#                     if len(X_val) > 0:
#                         val_logits = attacker(X_val)
#                         val_loss = criterion(val_logits, y_mem[val_idx_abs].to(cfg.device))
#                         if val_loss < best_val - 1e-5:
#                             best_val = val_loss
#                             patience_counter = 0
#                         else:
#                             patience_counter += 1
#                         attacker.train()
#                         if patience_counter >= 5:
#                             break

#             attacker.eval()
#             with torch.no_grad():
#                 s = torch.sigmoid(attacker(X_val)).detach().cpu()   # (|val_idx_abs|,)
#             # write back to out-of-fold vector
#             for j, abs_i in enumerate(val_idx_abs):
#                 out_idx = abs_to_local[abs_i]
#                 out[out_idx] = s[j]

#         return out

#     if class_conditional:
#         classes = np.unique(y_true)
#         for cls in classes:
#             cls_indices = np.where(y_true == cls)[0]
#             out_scores = _cv_scores_for_indices(cls_indices)
#             scores[cls_indices] = out_scores
#     else:
#         all_indices = np.arange(n)
#         scores[:] = _cv_scores_for_indices(all_indices)

#     return scores.numpy()
# -------------------------------------------------------------------------------------------------

def shadow_mlp_scores_loader(
    loader,
    model,
    cfg,
    y_mem_labels: np.ndarray,
    use_concepts: bool = True,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    k_folds: int = 10,
    class_conditional: bool = True,
    weight_decay: float = 1e-3,
    dropout: float = 0.1,
    standardize: bool = True,
    raw_concept_probs: bool = False,
    scores_whitebox_list: list = None
):
    """
    Train a shadow MLP with K-fold CV and return out-of-fold membership scores
    for ALL samples in loader order (sigmoid logits).

    If class_conditional=True, we train one attacker per true label and
    use the class-specific attacker to score samples of that class.

    Optionally, scores_whitebox_list can be appended as a feature.
    """
    # Build a single-batch loader to keep feature dims consistent
    dataset = loader.dataset
    single_batch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=getattr(loader, "collate_fn", None),
    )

    # Extract features (+ labels for class-conditional attackers)
    X, y_true = _extract_bb_features_from_loader(
        single_batch_loader, model, cfg,
        use_concepts=use_concepts,
        raw_concept_probs=raw_concept_probs
    )

    # Optionally append white-box scores as a feature
    y_mem_np = np.asarray(y_mem_labels)
    if scores_whitebox_list is not None and len(scores_whitebox_list) > 0:
        wb_scores = (
            scores_whitebox_list.reshape(-1)
            if isinstance(scores_whitebox_list, np.ndarray)
            else np.concatenate([np.asarray(s).reshape(-1) for s in scores_whitebox_list], axis=0)
        )
        if wb_scores.shape[0] != X.shape[0]:
            min_n = min(wb_scores.shape[0], X.shape[0], y_true.shape[0], y_mem_np.shape[0])
            print(f"\033[93m[shadow-mlp] Warning: length mismatch (X={X.shape[0]}, wb={wb_scores.shape[0]}). Trimming to {min_n}.\033[0m")
            wb_scores = wb_scores[:min_n]
            X        = X[:min_n]
            y_true   = y_true[:min_n]
            y_mem_np = y_mem_np[:min_n]
        X = np.concatenate([X, wb_scores.astype(np.float32)[:, None]], axis=1)

    y_mem = torch.from_numpy(y_mem_np.astype(np.float32))
    n = len(y_mem)
    scores = torch.zeros(n, dtype=torch.float32)

    # ---------- robust CV helper ----------
    def _cv_scores_for_indices(indices: np.ndarray) -> torch.Tensor:
        # nothing to score
        if len(indices) == 0:
            return torch.zeros(0, dtype=torch.float32)
        # if subset too small, default to 0.5
        if len(indices) < 2:
            return torch.full((len(indices),), 0.5, dtype=torch.float32)

        # stratify by membership when possible
        mem_sub = y_mem_np[indices]
        idx0 = indices[mem_sub == 0]
        idx1 = indices[mem_sub == 1]

        fold_pairs = []  # list of (train_idx_abs, val_idx_abs)

        if len(idx0) == 0 or len(idx1) == 0:
            # no stratification possible → simple K-fold on the subset
            n_folds = min(k_folds, max(2, len(indices)))
            simple_folds = [f for f in np.array_split(indices, n_folds) if len(f) > 0]
            for v in simple_folds:
                t = np.setdiff1d(indices, v)
                if len(t) == 0:
                    # move 1 sample from val→train
                    if len(v) > 1:
                        t, v = v[:1], v[1:]
                    else:
                        continue  # still empty; skip
                fold_pairs.append((t, v))
        else:
            # stratified folds with no empty splits
            n_folds0 = min(k_folds, len(idx0))
            n_folds1 = min(k_folds, len(idx1))
            folds0 = np.array_split(idx0, n_folds0)
            folds1 = np.array_split(idx1, n_folds1)
            num_folds = max(n_folds0, n_folds1)
            for k in range(num_folds):
                v = np.concatenate([folds0[k % n_folds0], folds1[k % n_folds1]])
                if len(v) == 0:
                    continue
                t = np.setdiff1d(indices, v)
                if len(t) == 0:
                    if len(v) > 1:
                        t, v = v[:1], v[1:]
                    else:
                        continue
                fold_pairs.append((t, v))

        # If we still failed to build any fold, default to 0.5
        if len(fold_pairs) == 0:
            return torch.full((len(indices),), 0.5, dtype=torch.float32)

        out = torch.zeros(len(indices), dtype=torch.float32)
        abs_to_local = {abs_i: j for j, abs_i in enumerate(indices)}

        for train_idx_abs, val_idx_abs in fold_pairs:
            if len(val_idx_abs) == 0:
                continue
            # tensors
            X_train = torch.from_numpy(X[train_idx_abs]).float().to(cfg.device)
            y_train = y_mem[train_idx_abs].to(cfg.device)
            X_val   = torch.from_numpy(X[val_idx_abs]).float().to(cfg.device)
            y_val   = y_mem[val_idx_abs].to(cfg.device)

            # standardize per fold
            if standardize and X_train.numel() > 0:
                mu = X_train.mean(dim=0, keepdim=True)
                sd = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
                X_train = (X_train - mu) / sd
                X_val   = (X_val   - mu) / sd

            # if training set is tiny, clamp batch_size
            ds = torch.utils.data.TensorDataset(X_train, y_train)
            if len(ds) == 0:
                # cannot train this fold, skip
                continue
            dl = torch.utils.data.DataLoader(ds, batch_size=max(1, min(batch_size, len(ds))), shuffle=True)

            attacker = _ShadowMLP(X_train.shape[1], hidden=64, dropout=dropout).to(cfg.device)
            pos = y_train.sum()
            neg = len(y_train) - pos
            pos_weight = (neg / (pos + 1e-8)).clamp(min=0.0, max=1e6)
            criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optim      = torch.optim.Adam(attacker.parameters(), lr=lr, weight_decay=weight_decay)

            attacker.train()
            best_val = float('inf')
            patience_counter = 0
            for _ in range(epochs):
                for xb, yb in dl:
                    optim.zero_grad()
                    logits = attacker(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optim.step()

                with torch.no_grad():
                    attacker.eval()
                    if len(X_val) > 0:
                        val_logits = attacker(X_val)
                        val_loss = criterion(val_logits, y_val)
                        if val_loss < best_val - 1e-5:
                            best_val = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        attacker.train()
                        if patience_counter >= 5:
                            break

            attacker.eval()
            with torch.no_grad():
                s = torch.sigmoid(attacker(X_val)).detach().cpu()   # (|val_idx_abs|,)
            for j, abs_i in enumerate(val_idx_abs):
                out[abs_to_local[abs_i]] = s[j]

        return out

    if class_conditional:
        classes = np.unique(y_true)
        for cls in classes:
            cls_indices = np.where(y_true == cls)[0]
            out_scores = _cv_scores_for_indices(cls_indices)
            scores[cls_indices] = out_scores
    else:
        all_indices = np.arange(n)
        scores[:] = _cv_scores_for_indices(all_indices)

    return scores.numpy()


def compute_validation_loss(model, val_loader, cfg) -> float:
    """
    Compute average *task* validation loss for a given model and validation loader.

    - Assumes forward returns (y_hat, c_hat) (already probs in your code).
    - Uses model.filter_output_for_loss(...) and model.loss(...).
    - If model.loss does not support multi_output=True, falls back gracefully.

    Returns:
        float: average loss over all samples, or +inf if loader is None/empty.
    """
    if val_loader is None:
        return float('inf')

    model.eval()
    model.to(cfg.device)
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(cfg.device)
            y = batch['y'].to(cfg.device)
            c = batch.get('c')
            c = c.to(cfg.device) if c is not None else None

            y_hat, c_hat = model(x)
            y_hat_loss, c_hat_loss = model.filter_output_for_loss(y_hat, c_hat)

            try:
                loss_task, loss_concept, loss_mixed = model.loss(
                    y_hat_loss, y, c_hat_loss, c,
                    reduction='mean', multi_output=True
                )
                if loss_task is None:
                    raise TypeError
                loss_value = loss_task
            except TypeError:
                loss_value = model.loss(
                    y_hat_loss, y, c_hat_loss, c,
                    reduction='mean'
                )

            bs = y.shape[0]
            total_loss += float(loss_value) * bs
            total_samples += bs

    return (total_loss / total_samples) if total_samples > 0 else float('inf')

def plot_training_metrics(history: Dict[str, Any], save_dir: str = ".") -> None:
    """
    Create two figures:
    1. A figure with subplots showing the validation loss trend for each client
    2. A figure showing the average validation loss trend across all clients
    
    Args:
        history: Dictionary containing training history with keys:
            - "round": List of round numbers
            - "loss_val_avg": List of average validation losses per round
            - "loss_val_client": Dict mapping client IDs to lists of validation losses
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    rounds = history["round"]
    
    # 1. Per-client validation loss plot
    n_clients = len(history["loss_val_client"])
    fig_height = max(4, min(12, 2 * n_clients))
    fig, axes = plt.subplots(n_clients, 1, figsize=(10, fig_height), sharex=True)
    
    # Ensure axes is always a list-like object even with one client
    if n_clients == 1:
        axes = [axes]
    
    for cid, ax in enumerate(axes):
        client_losses = history["loss_val_client"][cid]
        ax.plot(rounds, client_losses, 'o-', label=f'Client {cid}')
        ax.set_ylabel('Validation Loss')
        ax.set_title(f'Client {cid}')
        ax.grid(True, linestyle='--', alpha=0.7)
        
    axes[-1].set_xlabel('Round')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/client_validation_losses.png", dpi=300)
    
    # 2. Average validation loss across clients
    plt.figure(figsize=(10, 6))
    avg_loss = np.mean([history["loss_val_client"][cid] for cid in range(n_clients)], axis=0)
    plt.plot(rounds, avg_loss, 'o-', color='red', 
             linewidth=2, label='Average Validation Loss')
    
    # Optionally overlay individual client trends for comparison
    for cid in range(n_clients):
        client_losses = history["loss_val_client"][cid]
        plt.plot(rounds, client_losses, '--', alpha=0.3, label=f'Client {cid}')
    
    plt.xlabel('Round')
    plt.ylabel('Validation Loss')
    plt.title('Average Validation Loss Across Clients')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/average_validation_loss.png", dpi=300)
    
    plt.close('all')
    print(f"Training plots saved to {save_dir}/")

def build_local_graphs(client_ids, cfg, train_dataloaders, y_presence, y_name, graph = None):
    local_graphs = []
    local_weights = []
    for client_id in client_ids:
        loader = train_dataloaders[client_id - 1]
        node_names = cfg.engine.c_names_id[client_id]
        #node_names = list(client_nodes)
        if not node_names:
            continue
        if y_presence[client_id - 1]:
            node_names.append(y_name)
        if graph is not None:
            local_graphs.append(graph.loc[node_names, node_names])
        else:
            raise NotImplementedError("Graph construction from data is not implemented.")
        local_weights.append(len(loader.dataset))
    return local_graphs, local_weights
