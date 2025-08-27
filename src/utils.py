import torch
import random
import numpy as np
import pandas as pd
from omegaconf import DictConfig, open_dict, OmegaConf
from src.my_hydra import parse_hyperparams, target_classname
from src.metrics import edge_type
from env import CACHE
import os
import random
from src.data.generate_split import get_subgraph_dict
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

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


def model_has_concepts(model):
    name = model.name
    if name in ['blackbox_multi', 'cbm_linear', 'cbm_mlp', 'cem', 'c2bm']:
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


def update_config_from_data(cfg: DictConfig, datasets, subgraphs, subgraphs_concept_names) -> DictConfig:
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
            
            original_c_names = datasets[0].c_info['names']
            c_info = datasets[0].c_info

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
            for id in range(1, cfg.learning.n_clients + 1):
                if len(datasets)>1:
                    dataset = datasets[(id-1) % len(datasets)]
                else:
                    dataset = datasets[0]
                #input_size[id-1] = dataset.data["train"].X.shape[-1] if dataset.data["train"].X is not None else None
                subgraph_id = identify_subgraph(path, id)
                if subgraph_id is not None:
                    updated_c_names = subgraphs_concept_names['subgraph_'+subgraph_id]
                    c_names_id[id] = [name for name in dataset.c_info['names'] if name in updated_c_names]
                    c_names_ood[id] = [name for name in dataset.c_info['names'] if name not in updated_c_names]

            
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


def update_intervention_policy_and_graph(cfg, interv_policy, graph, subgraphs, subgraphs_concept_names):
    path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)

    # Get the subgraph given the client id
    for file in os.listdir(path):
        if ('trainset_'+str(cfg.client_id)) in file:
            # Get the substring between "subgraph_" and "."
            subgraph_id = file.split('subgraph_')[1].split('.')[0]
    c_index = subgraphs['subgraph_'+subgraph_id]  
    c_names = subgraphs_concept_names['subgraph_'+subgraph_id] 

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


def maybe_freeze_parameters(c, model, learning, freezing = True):
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
    if (learning == "local_federated" or learning=="federated") and freezing:

        c_indices_to_freeze = torch.where(c[0] == -1)[0]

        for param in model.parameters():
            param.requires_grad = True

        if model.name=="cbm_linear" or model.name =="cbm_mlp":
            c_keys = list(model.c_mlp.keys())
            c_to_freeze = [c_keys[i] for i in c_indices_to_freeze]
            for name, mlp in model.c_mlp.items():
                    if name in c_to_freeze:
                        for param in mlp.parameters():
                            param.requires_grad = False
            print("Parameters frozen for concepts:", c_to_freeze)

        if model.name=="cem":
            c_keys = list(model.concept_encoders.keys())
            c_to_freeze = [c_keys[i] for i in c_indices_to_freeze]
            for name, concept_encoder in model.concept_encoders.items():
                if name in c_to_freeze:
                    for param in concept_encoder.parameters():
                        param.requires_grad = False
            print("Parameters frozen for concepts:", c_to_freeze)
        
        if model.name=="c2bm":
            c_keys = list(model.concept_encoders.keys())
            c_to_freeze = [c_keys[i] for i in c_indices_to_freeze]

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


def score_blackbox_batch(batch, model, cfg):
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

def score_whitebox_batch(batch, model, client_update, cfg):
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


def dataprocess_auditing(train_dataloaders, cfg):
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

    for cid in range(cfg.learning.n_clients):
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
    
    mia_accuracies = {"whitebox": {}, "blackbox": {}}
    mia_epsilons = {"whitebox": {}, "blackbox": {}}
    for cid in range(n_clients):
        mia_accuracies["whitebox"][cid] = []
        mia_accuracies["blackbox"][cid] = []
        mia_epsilons["whitebox"][cid] = []
        mia_epsilons["blackbox"][cid] = []
        
    return mia_accuracies, mia_epsilons


@torch.no_grad()
def _sia_batch_loss(batch, model, cfg):
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
    losses = model.loss(
        y_hat_loss, y, c_hat_loss, c, reduction="none", ignore_index=-1
    )  # shape = (B,)

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
            losses_client.append(_sia_batch_loss(batch, local_engine.model, cfg))
        losses_all[:, cid] = np.concatenate(losses_client)

    # ----------- prediction & accuracy -------------------------------------
    pred_cid   = losses_all.argmin(axis=1)           # (n_samples,)
    true_cid = _sia_true_cids(sia_loader.dataset)

    accuracy   = (pred_cid == true_cid).mean()

    return accuracy

