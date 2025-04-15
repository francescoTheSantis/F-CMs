import torch
from env import CACHE
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from src.data.utils import static_graph_collate
import pickle
import os
import random

def get_nodes_subgroups(graph, y_index, n_subgroups, modality= 'task_included'):
    """
    This function generates n_subgroups of nodes from the graph with the following properties:
    - Each subgroup must have at least two nodes
    - The union of all subgroups must be equal to the original graph
    - if modality is 'task_included', the task node must be included in each subgroup
    - if modality is 'task_excluded', the task node must be excluded from each subgroup
    """
    nodes_covered = set()
    n_subgroups_generated = 0
    subgroups = []

    nodes_to_cover = list(range(len(graph)))
    # eliminate y_index from nodes_to_cover
    
    nodes_to_cover.remove(y_index)


    # Step 1: Generate enough subgroups to cover all nodes
    while len(nodes_covered)< len(nodes_to_cover) or n_subgroups_generated < n_subgroups:
        # Randomly select a subgroup of nodes
        subgroup = random.sample(nodes_to_cover, random.randint(2, len(nodes_to_cover)))
        # check if the subgroup has at least one node in common with another subgroup

        subgroups = subgroups + [subgroup]
        nodes_covered = nodes_covered.union(set(subgroup))
        n_subgroups_generated += 1


    # Step 2: Merge subgroups until reaching desired number
    while len(subgroups) > n_subgroups:
        # Sort by length so smaller ones are merged first
        subgroups = sorted(subgroups, key=len)
        # Merge the two smallest
        first = subgroups.pop(0)
        second = subgroups.pop(0)
        merged = list(set(first + second))
        subgroups.append(merged)

    if modality == 'task_included':
        # add the task node to each subgroup
        for i in range(len(subgroups)):
            if y_index not in subgroups[i]:
                subgroups[i].append(y_index)

    # return a dictionary with soubgroups as keys and the nodes as values
    subgraphs = {f'subgraph_{i+1}': subgroup for i, subgroup in enumerate(subgroups)}
    # return a dictionary with the subgroups as keys and the nodes names as values
    subgraphs_concept_names = {f'subgraph_{i+1}':[graph.columns[node_idx] for node_idx in subgroup] for i, subgroup in enumerate(subgraphs.values())}

    return subgraphs, subgraphs_concept_names

def generate_split(cfg, dataset, graph):
    split_and_save(cfg, dataset, graph, 'train')
    split_and_save(cfg, dataset, graph, 'val')
    split_and_save(cfg, dataset, graph, 'test')
    # Save the dataloader for the unique, real test-set
    test_dataloader = DataLoader(dataset.data['test'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
    root = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
    path = os.path.join(root, f"test.pkl")
    with open(path, 'wb') as f:
        pickle.dump(test_dataloader, f)

def split_and_save(cfg, data, graph, set):
        # Create as many splits as the number of clients     
        x, c, y = [], [], []
        for row in data.data[set]:
            x.append(row['x'].unsqueeze(0))
            c.append(row['c'].unsqueeze(0))
            y.append(row['y'].unsqueeze(0))
        x = torch.cat(x, dim=0)
        c = torch.cat(c, dim=0)
        y = torch.cat(y, dim=0)

        # create n random splits from the preivous tensors
        n = cfg.learning.n_clients

        # Get the disctionary containing the subgraphs given the dataset's name
        subgraphs, _ = get_subgraph_dict(cfg)
        # Get the index of the y variable in the graph
        #y_index = graph.columns.get_loc(cfg.dataset.loader.task_name)
        #subgraphs, _ = get_nodes_subgroups(graph, y_index = y_index, n_subgroups =3, modality = 'task_excluded')

        # Ensure the tensors can be evenly split
        assert x.size(0) == c.size(0) == y.size(0), "Tensors must have the same number of rows"
        assert len(subgraphs) < n, "Number of subgraphs must be lower than n"

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

        # For each split, create a dataloader containing x, c, y 
        for i in range(n):
            j = i % len(subgraphs)
            masked_c_splits = apply_mask(c_splits[i], subgraphs[f'subgraph_{j+1}'])

            dataloader = DataLoader(
                CustomDataset(x_splits[i], masked_c_splits, y_splits[i], graph),
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