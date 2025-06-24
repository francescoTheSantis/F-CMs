import torch
import random
import numpy as np
import pandas as pd
from omegaconf import DictConfig, open_dict
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


def load_dataloaders(cfg: DictConfig, path: str, n_clients: int):
    train_dataloaders = []
    val_dataloaders = []
    for client_id in range(1,n_clients+1):
        train_path, val_path = get_split_paths_fl(cfg, path, client_id)
        # if the file is not found, raise an error
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(f"File {train_path} or {val_path} not found")
        # Load the dataloaders
        with open(train_path, 'rb') as f:
            train_dataloader = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_dataloader = pickle.load(f)
            
        train_dataloaders.append(train_dataloader)
        val_dataloaders.append(val_dataloader)
    return train_dataloaders, val_dataloaders
            
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
    if target_classname(model) in ['BlackBox_Multi', 'CBM', 'CEM', 'C2BM', 'SCBM']:
        return True
    elif target_classname(model) in ['BlackBox']:
        return False
    else:
        raise ValueError(f"Unknown model type: {target_classname(model)}")
    

def model_is_causal(model):
    if target_classname(model) in ['C2BM']:
        return True
    elif target_classname(model) in ['BlackBox', 'BlackBox_Multi', 'CEM', 'CBM', 'SCBM']:
        return False
    else:
        raise ValueError(f"Unknown model type: {target_classname(model)}")

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

def update_config_from_data(cfg: DictConfig, dataset, subgraphs, subgraphs_concept_names) -> DictConfig:
    """ can be used to update the config based on the data, e.g., set input and output size """
    original_c_names = dataset.c_info['names']
    with open_dict(cfg):
        if cfg.learning.mode=='localized':
            path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
            # Get the subgraph giventhe client id
            subgraph_id = identify_subgraph(path, cfg.client_id) #file.split('subgraph_')[1].split('.')[0]
            #_, updated_c_names = get_subgraph_dict(cfg)  
            updated_c_names = subgraphs_concept_names['subgraph_'+subgraph_id]       
            c_names = [name for name in dataset.c_info['names'] if name in updated_c_names]
            c_cardinality = [card for card, name in zip(dataset.c_info['cardinality'], dataset.c_info['names']) if name in c_names]
            
            # The list of names for out-of-distribution concepts (concepts that the client has never seen before)
            c_names_ood = [name for name in dataset.c_info['names'] if name not in updated_c_names]
            c_info = {'names': c_names, 'cardinality': c_cardinality, 'c_names_ood': c_names_ood}
        else:
            c_info = dataset.c_info
            c_names = original_c_names

        cfg.engine.model.update(
            input_size = dataset.data["train"].X.shape[-1] if dataset.data["train"].X is not None else None,
            output_size = dataset.y_info['cardinality'][0], # we assume single class classification
            c_info = c_info,
            y_info = dataset.y_info,
            c_name_index = {name: i for i, name in enumerate(original_c_names)},
        )
        cfg.engine.update(
            c_names = c_names,
            c_names_ood = c_info.get('c_names_ood', []),
            c_name_index = {name: i for i, name in enumerate(original_c_names)}
        )
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
    c_name_idx = {k:v for k, v in zip(c_index, range(len(c_names)))}
    for level in interv_policy:
        level_policy = []
        for i, node in enumerate(level):
            if node in c_index:
                level_policy.append(c_name_idx[node])
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
    
def get_split_paths(cfg, path, test=False):
    if not test:
        for file in os.listdir(path):
            if extract_between(file, 'train') == str(cfg.client_id):
                train_path = os.path.join(path, file)
            if extract_between(file, 'val') == str(cfg.client_id):
                val_path = os.path.join(path, file)
        return train_path, val_path
    else:
        for file in os.listdir(path):
            if extract_between(file, 'test') == str(cfg.client_id):
                test_path = os.path.join(path, file)
        return test_path


def get_split_paths_fl(cfg, path, client_id):
    for file in os.listdir(path):
        if extract_between(file, 'train') == str(client_id):
            train_path = os.path.join(path, file)
        if extract_between(file, 'val') == str(client_id):
            val_path = os.path.join(path, file)
    return train_path, val_path

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
    if learning == "federated" and freezing:

        c_indices_to_freeze = torch.where(c[0] == -1)[0]

        for param in model.parameters():
            param.requires_grad = True

        if model.__class__.__name__=="CBM":
            c_keys = list(model.c_mlp.keys())
            c_to_freeze = [c_keys[i] for i in c_indices_to_freeze]
            for name, mlp in model.c_mlp.items():
                    if name in c_to_freeze:
                        for param in mlp.parameters():
                            param.requires_grad = False
            print("Parameters frozen for concepts:", c_to_freeze)

        if model.__class__.__name__=="CEM":
            c_keys = list(model.concept_encoders.keys())
            c_to_freeze = [c_keys[i] for i in c_indices_to_freeze]
            for name, concept_encoder in model.concept_encoders.items():
                if name in c_to_freeze:
                    for param in concept_encoder.parameters():
                        param.requires_grad = False
            print("Parameters frozen for concepts:", c_to_freeze)
        
        if model.__class__.__name__=="C2BM":
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
