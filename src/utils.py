import torch
import numpy as np
import pandas as pd
from omegaconf import DictConfig, open_dict
from src.hydra import parse_hyperparams, target_classname
from src.metrics import edge_type
from env import CACHE
import os
from src.data.generate_split import get_subgraph_dict

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

def update_config_from_data(cfg: DictConfig, dataset) -> DictConfig:
    """ can be used to update the config based on the data, e.g., set input and output size """
    with open_dict(cfg):
        if cfg.learning.mode=='localized':
            path = str(CACHE / cfg.dataset.name)
            # Get the subgraph giventhe client id
            for file in os.listdir(path):
                if ('trainset_'+str(cfg.learning.client_id)) in file:
                    # Get the substring between "subgraph_" and "."
                    subgraph_id = file.split('subgraph_')[1].split('.')[0]
            _, updated_c_names = get_subgraph_dict(cfg)  
            updated_c_names = updated_c_names['subgraph_'+subgraph_id]       
            c_names = [name for name in dataset.c_info['names'] if name in updated_c_names]
            c_cardinality = [card for card, name in zip(dataset.c_info['cardinality'], dataset.c_info['names']) if name in c_names]
            c_info = {'names': c_names, 'cardinality': c_cardinality}
        else:
            c_info = dataset.c_info
            c_names = dataset.c_info['names']

        cfg.engine.model.update(
            input_size = dataset.data["train"].X.shape[-1] if dataset.data["train"].X is not None else None,
            output_size = dataset.y_info['cardinality'][0], # we assume single class classification
            c_info = c_info,
            y_info = dataset.y_info,
        )
        cfg.engine.update(
            c_names = c_names
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

def update_intervention_policy_and_graph(cfg: DictConfig, interv_policy, graph):
    path = str(CACHE / cfg.dataset.name)
    # Get the subgraph giventhe client id
    for file in os.listdir(path):
        if ('trainset_'+str(cfg.learning.client_id)) in file:
            # Get the substring between "subgraph_" and "."
            subgraph_id = file.split('subgraph_')[1].split('.')[0]
    c_index, c_names = get_subgraph_dict(cfg) 
    c_index = c_index['subgraph_'+subgraph_id]  
    c_names = c_names['subgraph_'+subgraph_id] 

    # Update policy
    updated_policy = []
    for level in interv_policy:
        level_policy = []
        for i, node in enumerate(level):
            if node in c_index:
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