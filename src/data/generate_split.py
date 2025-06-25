import torch
import numpy as np
from env import CACHE
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from src.data.utils import static_graph_collate
import pickle
import os
import random

def get_connected_subgraph(graph, task_index):
    from src.utils import get_parents
    """
    This function generate a connected subgraph from a graph, which could be the original graph or a subgraph.
    Args:
        graph: The graph.
        task_index: The index of the task for the subgraph.
    Returns:
        subgraph: A list of nodes in the subgraph.
    """
    torch_graph = torch.tensor(graph.values)
    nodes = [task_index]
    subgraph = [task_index]
    while True:
        # partition the graph starting from the task node
        # get the parents of the nodes
        parents = [get_parents(torch_graph, node) for node in nodes]
        # eliminate empty tensors
        parents = [t for t in parents if t.numel() > 0]
        # if there are no parents, break the loop
        if len(parents) == 0:
            break
        # Choose randomly between the two options for number_of_parents_to_keep
        number_of_parents_to_keep = random.choices([
            max((len(parents) + 1) // 2, 1),
            max((len(parents) + 1) // 2, 2)],
            weights= [0.6,0.4],
            k=1
        )[0]
        # select, for each node, at least a random parent
        parents = [list({random.choice(nodes) for _ in range(number_of_parents_to_keep)}) for nodes in parents]
        # remove duplicates and flatten the list
        parents = [int(item) for sublist in parents for item in sublist]
        parents = list(set(parents))
        subgraph.extend(parents)
        nodes = parents
    # flatten the list and obtain unique values
    subgraph = sorted(list(set(subgraph)))

    return subgraph

def get_subgraphs(graph, y_index, n_subgraphs, modality = 'random_nodes', concept_in_common = False, task_in_common= True):
    """
    This function generates n_subgraphs from the original graph.

    If modality is 'random_nodes', the function generates subgraphs composed by random nodes in the graphs.
    If modality is 'connected_nodes', the function generates subgraphs composed by connected nodes in the graphs (trees).

    The following conditions must be satisfied:
    - Each subgraph have at least two nodes
    - If task_in_common is False, the union of all the subgraphs must cover all the nodes in the original graph
    - If task_in_common is True, the union of all the subgraphs must cover all the nodes in the graph starting from y_index
    - If concept_in_common is True and modality is 'connected_nodes', the subgraphs must have at least one node in common with another subgraph generated before.

    Args:
        graph: The original graph.
        y_index: The index of the task node in the original graph.
        n_subgraphs: The number of subgraphs to generate.
        modality: The modality to use for generating the subgraphs. It can be 'random_nodes' or 'connected_nodes'.
        concept_in_common: If True, the subgraphs must have at least one node in common with another subgraph generated before.
        
    Returns:
        subgraphs: A dictionary with indices as keys and lists of nodes for each subgraph as values
        subgraphs_concept_names: A dictionary with indices as keys and the names of the nodes of each subgraph as values
    """
    from src.utils import get_roots, get_task_graph
    nodes_covered = set()
    n_subgraphs_generated = 0
    subgraphs = []

    if task_in_common:
        nodes_to_cover = get_task_graph(torch.tensor(graph.values), task_node = y_index)
        # get nodes in a single list
        nodes_to_cover = [int(item) for sublist in nodes_to_cover for item in sublist]
    else:
        nodes_to_cover = list(range(len(graph)))
    
    # eliminate y_index from nodes_to_cover
    nodes_to_cover.remove(y_index)

    # Step 1: Generate enough subgraphs to cover all nodes
    while len(nodes_covered)< len(nodes_to_cover) or n_subgraphs_generated < n_subgraphs:

        if modality == 'random_nodes':
            # Randomly select a subgroup of nodes
            subgraph = random.sample(nodes_to_cover, random.randint(2, len(nodes_to_cover)))         
        if modality == 'connected_nodes':
            if not task_in_common:
                # Randomly select a node that is not a root to start the subgraph
                roots = get_roots(torch.tensor(graph.values))
                task_index = random.choice([node for node in nodes_to_cover if node not in np.where(roots)[0]])
            else:
                task_index = y_index
                       
            subgraph = get_connected_subgraph(graph, task_index)
            if y_index in subgraph:
                # if y_index is in the subgraph, remove it
                subgraph.remove(y_index)
            # if concept_in_common, add the subgraph only if it has at least one node in common with another subgraph
            if concept_in_common:
                # If the subgraph has no nodes in common with any other subgraph, skip it
                if n_subgraphs_generated!=0 and not any(set(subgraph).intersection(set(s)) for s in subgraphs):
                    continue
       
        subgraphs = subgraphs + [subgraph]

        nodes_covered = nodes_covered.union(set(subgraph))
        n_subgraphs_generated += 1



    # Step 2: Merge subgraphs until reaching desired number
    while len(subgraphs) > n_subgraphs:

        if modality == 'random_nodes':
            # Sort by length so smaller ones are merged first
            subgraphs = sorted(subgraphs, key=len)
            # Merge the two smallest
            first = subgraphs.pop(0)
            second = subgraphs.pop(0)
            merged = list(set(first + second))
            subgraphs.append(merged)
            
            
        if modality == 'connected_nodes':
            # Sort by length so smaller ones are merged first
            subgraphs = sorted(subgraphs, key=len)
            # Try to merge the ones that have at least a node in common
            merged = False
            for i in range(len(subgraphs)):
                for j in range(i+1, len(subgraphs)):
                    if len(set(subgraphs[i]).intersection(set(subgraphs[j]))) > 0:
                        merged = True
                        merged_subgraph = list(set(subgraphs[i] + subgraphs[j]))
                        subgraphs.pop(j)
                        subgraphs.pop(i)
                        subgraphs.append(merged_subgraph)
                        break
                if merged:
                    break

            # If no merge is possible, just eliminate the smallest subgraph
            if not merged:
                subgraphs.pop(0)

    # return a dictionary with soubgroups as keys and the nodes as values
    subgraphs = {f'subgraph_{i+1}': s for i, s in enumerate(subgraphs)}
    # return a dictionary with the subgroups as keys and the nodes names as values
    subgraphs_concept_names = {f'subgraph_{i+1}':[graph.columns[node_idx] for node_idx in s] for i, s in enumerate(subgraphs.values())}

    return subgraphs, subgraphs_concept_names

def generate_split(cfg, dataset, graph, y_index):

    n = cfg.learning.n_clients
    # Get the subgraph for each client
    subgraphs, subgraphs_concept_names = get_subgraphs(graph, y_index, round(n/2)+1, 
                                 modality=cfg.learning.subgraphs.modality,
                                 concept_in_common=cfg.learning.subgraphs.concept_in_common,
                                 task_in_common=cfg.learning.subgraphs.task_in_common)
    # Check on the subgraphs
    assert len(subgraphs) < n, "Number of subgraphs must be lower than n"

    # If the task is not included, select some subgraphs to mask the y variable
    if not cfg.learning.annotation_assumption == "task_included":
        r = random.randint(1, (len(subgraphs)-1))  # Randomly select r subgraphs to mask y variable
        subgraphs_task_excluded = random.sample(range(1, len(subgraphs) + 1), r)
    else:
        subgraphs_task_excluded = None
    
    split_and_save(cfg, dataset, graph, 'train', n, subgraphs, subgraphs_task_excluded)
    split_and_save(cfg, dataset, graph, 'val',n, subgraphs, subgraphs_task_excluded)
    split_and_save(cfg, dataset, graph, 'test', n, subgraphs, subgraphs_task_excluded)
    # Save the dataloader for the unique, real test-set
    test_dataloader = DataLoader(dataset.data['test'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
    root = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
    path = os.path.join(root, f"test.pkl")
    with open(path, 'wb') as f:
        pickle.dump(test_dataloader, f)

    return subgraphs, subgraphs_concept_names

def split_and_save(cfg, data, graph, set,n, subgraphs = None, subgraphs_task_excluded = None):
        
        # Create as many splits as the number of clients     
        x, c, y = [], [], []
        for row in data.data[set]:
            x.append(row['x'].unsqueeze(0))
            c.append(row['c'].unsqueeze(0))
            y.append(row['y'].unsqueeze(0))
        x = torch.cat(x, dim=0)
        c = torch.cat(c, dim=0)
        y = torch.cat(y, dim=0)

        # Ensure the tensors can be evenly split
        assert x.size(0) == c.size(0) == y.size(0), "Tensors must have the same number of rows"

        # Get the disctionary containing the subgraphs given the dataset's name
        #subgraphs, _ = get_subgraph_dict(cfg)
        # Get the index of the y variable in the graph
        #y_index = graph.columns.get_loc(cfg.dataset.loader.task_name)
        # Get the subgraphs based on the graph and y_index

        # Generate indices for splitting
        indices = torch.arange(x.size(0))

        # Calculate split sizes for uneven splits
        split_sizes = [(x.size(0) + i) // n for i in range(n)] 

        # Split indices into uneven parts
        split_indices = torch.split(indices, split_sizes)

        # Split tensors using the indices
        x_splits = [x[idx] for idx in split_indices]
        c_splits = [c[idx] for idx in split_indices]
        y_splits = [y[idx] for idx in split_indices]

        if subgraphs is None:
            raise ValueError("`subgraphs` cannot be None.")

        # For each split, create a dataloader containing x, c, y 
        for i in range(n):
            j = i % len(subgraphs)
            masked_c_splits = apply_mask(c_splits[i], subgraphs[f'subgraph_{j+1}'])

            if subgraphs_task_excluded is not None and ((j+1) in subgraphs_task_excluded):
                # If the task is not included, mask the y variable as well
                masked_y_splits = -1 * torch.ones_like(y_splits[i])  # Mask y variable
            else:
                masked_y_splits = y_splits[i]
        
            dataloader = DataLoader(
                CustomDataset(x_splits[i], masked_c_splits, masked_y_splits, graph),
                batch_size=cfg.dataset.batch_size,
                collate_fn=static_graph_collate
            )


            # Create directory if it doesn't exist
            root = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
            os.makedirs(os.path.join(root), exist_ok=True)
            # Store the dataloader in the 
            path = os.path.join(root, f"{set}set_{i+1}_subgraph_{j+1}.pkl") # Start to count from 1
            with open(path, 'wb') as f:
                pickle.dump(dataloader, f)

def get_subgraph_dict(cfg):
    if cfg.dataset.name == 'asia':
        subgraphs = {
             'subgraph_1': [0,1,5],
             'subgraph_2': [2,3,5],
             'subgraph_3': [2,4]
        }
        subgraphs_concept_names = {
            'subgraph_1': ['asia', 'tub', 'either'],
            'subgraph_2': ['smoke', 'lung', 'either'],
            'subgraph_3': ['smoke', 'bronc']   
        }
    else:
        pass
    return subgraphs, subgraphs_concept_names

def apply_mask(tensor, keep):
    """
    Apply a mask to a tensor by zeroing out columns not in the keep list.
    Args:
        tensor: The input tensor.
        keep: List of column indices to keep.
    Returns:
        The masked tensor.
    """
    mask = torch.ones(tensor.size(1), dtype=torch.bool)
    mask[keep] = False
    masked_tensor = tensor.clone()
    masked_tensor[:, mask] = -1
    return masked_tensor

class CustomDataset(Dataset):
    def __init__(self, x, c, y, graph):
        self.x = x
        self.c = c
        self.y = y
        self.graph = graph

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'c': self.c[idx],
            'y': self.y[idx],
            'graph': self.graph
        }