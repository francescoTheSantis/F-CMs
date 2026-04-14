import torch
import numpy as np
from env import CACHE
from torch.utils.data import DataLoader
from typing import Callable, Any, Tuple, Optional
from torch.utils.data import Dataset
from src.data.utils import static_graph_collate
import pickle
import os
import random
import shutil
import random
import math

def dfs_forward(torch_graph, start_node, end_node, visited=None, randomize=True, nodes_not_allowed = []):
    """
    DFS forward traversal to find a path from start_node to end_node.
    
    Args:
        torch_graph: torch tensor adjacency matrix
        start_node: current node index
        end_node: target node index
        visited: set of visited nodes
        randomize: if True, shuffle successors to get different paths
    
    Returns:
        List of node indices representing the path, or None if no path exists
    """
    if visited is None:
        visited = set()
    
    if start_node == end_node:
        return [start_node]
    
    if start_node in visited :
        return None
    
    visited.add(start_node)

    if start_node in nodes_not_allowed:
        return None
    

    # Get successors: nodes where torch_graph[start_node][neighbor] == 1
    successors = torch.where(torch_graph[start_node] == 1)[0].tolist()
    
    # Randomize order of exploration
    if randomize:
        random.shuffle(successors)

    if end_node is None and len(successors) == 0:
        return [start_node]
    
    for neighbor in successors:
        if neighbor not in visited:
            result = dfs_forward(torch_graph, neighbor, end_node, visited, randomize, nodes_not_allowed=nodes_not_allowed)
            if result is not None:
                return [start_node] + result
    
    return None

def dfs_backward(torch_graph, start_node, visited=None, randomize=True, nodes_not_allowed = []):
    """
    DFS backward traversal to find a root node (node with no parents).
    Can include multiple parents to create branching paths.
    
    Args:
        torch_graph: torch tensor adjacency matrix
        start_node: current node index
        visited: set of visited nodes
        randomize: if True, shuffle parents to get different roots
        branch_probability: probability of including additional branches (0.0 to 1.0)
    
    Returns:
        Root node index, or None if no root is found
    """
    if visited is None:
        visited = set()

    if start_node in visited:
        return None

    visited.add(start_node)
    
    if start_node in nodes_not_allowed:
        return None
    
    # Get parents: nodes where torch_graph[parent][start_node] == 1
    parents = torch.where(torch_graph[:, start_node] == 1)[0].tolist()
    
    # If no parents, this is a root
    if len(parents) == 0:
        return [start_node]
    
    # Randomize order of exploration
    if randomize:
        random.shuffle(parents)
    
    # Recursively search parents
    for parent in parents:
        if parent not in visited:
            result = dfs_backward(torch_graph, parent, visited, randomize, nodes_not_allowed=nodes_not_allowed)
            if result is not None:
                return [start_node] + result
    
    return None


def find_all_parent_child_pairs(graph, y_index_graph):
    """
    Find all parent-child node pairs in the graph and retain only thos who belong to y_index_graph.
    
    Args:
        graph: pandas DataFrame representing adjacency matrix where graph[i][j] = 1 means i -> j (i is parent of j)
    
    Returns:
        List of tuples (parent, child) representing all parent-child relationships in the graph
    """
    # Convert to torch tensor
    torch_graph = torch.tensor(graph.values)
    
    parent_child_pairs = []
    parents = range(torch_graph.size(0))
    # eliminate from parent those not in y_index_graph
    parents = [p for p in parents if p in y_index_graph]
    
    # Iterate through all nodes
    for parent in parents:
        # Find all children of the current parent (where graph[parent][child] == 1)
        children = torch.where(torch_graph[parent] == 1)[0].tolist()
        
        # Eliminate from children those not in y_index_graph
        children = [c for c in children if c in y_index_graph]
        for child in children:
            parent_child_pairs.append((parent, child))
    
    return parent_child_pairs


def find_path_to_target_or_leaf(graph, start_node, end_node=None, randomize=False, nodes_not_allowed = []):
    """
    Find a path from start_node to end_node (or to a leaf if end_node is None).
    Can include multiple branches.
    
    Args:
        graph: pandas DataFrame representing adjacency matrix
        start_node: Starting node index
        end_node: Target node index (if None, find path to any leaf)
        randomize: if True, return a random path (non-deterministic)
        branch_probability: probability of including additional branches (0.0 to 1.0)
    
    Returns:
        List of node indices representing the path, or None if no path exists
    """
    # Convert to torch tensor
    torch_graph = torch.tensor(graph.values)

    if start_node in nodes_not_allowed or end_node in nodes_not_allowed:
        return None
    
    if start_node >= torch_graph.size(0):
        return None
    
    if end_node is not None and end_node >= torch_graph.size(0):
        return None
    
    if end_node is not None and start_node == end_node:
        return [start_node]
    
    return dfs_forward(torch_graph, start_node, end_node, randomize=randomize, nodes_not_allowed=nodes_not_allowed)


def find_path_to_root(graph, start_node, randomize=True, nodes_not_allowed = []):
    """
    Find the root node that connects to start_node by traversing backwards.
    A root is a node with no incoming edges.
    
    Args:
        graph: pandas DataFrame representing adjacency matrix
        start_node: Starting node index
        randomize: if True, return a random root (non-deterministic)
    
    Returns:
        Root node index, or None if no root is found
    """
    # Convert to torch tensor
    torch_graph = torch.tensor(graph.values)
    
    if start_node >= torch_graph.size(0):
        return None
    
    # Check if start_node is already a root (no parents)
    parents = torch.where(torch_graph[:, start_node] == 1)[0]
    if parents.numel() == 0:
        return [start_node]
    
    return dfs_backward(torch_graph, start_node, randomize=randomize, nodes_not_allowed=nodes_not_allowed)

def generate_base_subgraph(graph, task_indices, randomize = True, nodes_not_allowed = []):
    """
    Generate a base subgraph connecting roots to task nodes.
    
    Args:
        graph: pandas DataFrame representing adjacency matrix
        task_indices: list of task node indices
        randomize: if True, randomize path selection
    
    Returns:
        List of node indices forming the subgraph
    """
    # nodes are numeric, graph.index are the names of the concepts
    # find nodes of the graph
    
    if any(i not in list(range(len(graph))) for i in task_indices):
        raise ValueError("One or more task indices are not in the graph.")
 
    subgraph = []
    
    for i in task_indices:

        # calculate number of childrens of nodes
        max_n_childrens = 0
        for node in range(len(graph)):
            childrens = torch.where(torch.tensor(graph.values)[node,:] == 1)[0].tolist()
            if len(childrens) > max_n_childrens:
                max_n_childrens = len(childrens)

        max_random_paths = max(1, round(((len(graph) + max_n_childrens) / 10) ** 0.9))
        random_node = random.randint(1, max_random_paths)
        task_path = []

        for attempt in range(random_node):

            path_to_root = find_path_to_root(graph, start_node=i, randomize=randomize, nodes_not_allowed=nodes_not_allowed)
            # reverse path to have from root to node
            path_to_root= path_to_root[::-1] if path_to_root is not None else None
            root = path_to_root[0] if path_to_root is not None else None
            if root is None:
                raise ValueError(f"I cannot find a path to root for node {graph.columns[i]} that not include nodes_not_allowed {list(graph.columns[nodes_not_allowed])}")
 
            #current_path = find_path_to_target_or_leaf(graph, start_node=root, end_node=i, randomize=True, nodes_not_allowed=nodes_not_allowed)
            #if path_to_root is not None:
            task_path = list(set(task_path + path_to_root))

            if task_path is None or len(task_path) == 0:
                raise ValueError(f"No path found from root {root} to node {i}")
        
        subgraph = list(set(subgraph + task_path))
    
    return subgraph

def generate_add_nodes_values(graph, torch_graph, y_index_graph, y_index, add_nodes_modality, add_nodes_number, add_nodes_values= [], old_add_nodes_values = []):
    """
    Generate add_nodes_values based on the specified modality. 
    NOTE: The additional nodes are selected from the y_index subgraph because they are considered only in cgm and c2bm.
    
    Args:
        graph: pandas DataFrame representing adjacency matrix
        torch_graph: torch tensor adjacency matrix
        y_index_graph: list of nodes in y_index subgraph
        y_index: target node index
        add_nodes_modality: 'random' or other modality
        add_nodes_number: number of nodes to add
    
    Returns:
        List of node indices to add
    """
    from src.utils import get_parents

    if add_nodes_modality == 'random':
        possible_nodes = [node for node in y_index_graph if node != y_index and node not in old_add_nodes_values]
        if possible_nodes == []:
            raise ValueError("It is not possible to select additional nodes with the specified modality. Please change modality or reduce number_add_nodes.")
    elif add_nodes_modality == 'at_least_one_parent_and_child' or add_nodes_modality == 'connection':
        # Select nodes that have at least one parent and one child
        possible_nodes = [] 
        connections = []
        for node in y_index_graph:
                childrens =  torch.where(torch_graph[node,:] == 1)[0].tolist()
                parents = get_parents(torch_graph, node).tolist()
                if len(childrens) != 0 and len(parents) != 0:
                    possible_nodes.append(node)
                    connections.append(len(parents)+ len(childrens))

        # filter out old_add_nodes_values
        possible_nodes = [node for node in possible_nodes if node not in old_add_nodes_values and node != y_index]
        if possible_nodes == []:
            raise ValueError("It is not possible to select additional nodes with the specified modality. Please change modality or reduce number_add_nodes.")

        if add_nodes_modality == 'connection':
            # order nodes in base of connections and select top add_nodes_number
            possible_nodes = sorted(possible_nodes, key=lambda x: connections[possible_nodes.index(x)], reverse=True)    
            return possible_nodes[:add_nodes_number]
    elif add_nodes_modality == 'specific_nodes':
        if not add_nodes_values:
            raise ValueError("add_nodes_values must be provided when using 'specific_nodes' modality.")
        if any(node not in y_index_graph for node in add_nodes_values):
            raise ValueError("One or more specified additional nodes are not in the y_index subgraph. Please check specific_nodes_names.")
        if len(add_nodes_values) < add_nodes_number:
            raise ValueError("The number of specified additional nodes is less than number_add_nodes.")
        if any(node in old_add_nodes_values for node in add_nodes_values):
            raise ValueError("One or more specified additional nodes have already been used in previous attempts. Please provide different specific_nodes_names.")
        if any(node == y_index for node in add_nodes_values):
            raise ValueError("The specified additional nodes cannot include the task node.")
        if len(add_nodes_values) > add_nodes_number:
            return random.sample(add_nodes_values, add_nodes_number)
        possible_nodes = add_nodes_values
      
    else:
        raise ValueError("Unsupported modality for dict_subgraph_with_add_nodes. Supported modalities are 'connected_nodes' and 'random'")

    return random.sample(possible_nodes, min(add_nodes_number, len(possible_nodes)))


def add_additional_nodes_to_subgraph(graph, subgraph, add_nodes_values, randomize=True):
    """
    Extend subgraph by including additional nodes with their paths.
    
    Args:
        graph: pandas DataFrame representing adjacency matrix
        subgraph: current subgraph as list of node indices
        add_nodes_values: nodes to add
        randomize: if True, randomize path selection
    
    Returns:
        Extended subgraph as list of node indices
    """
    extended_subgraph = subgraph.copy()

    for node in add_nodes_values:
        if node in extended_subgraph:
            continue
        
        # Find a leaf reachable from this node
        path_to_leaf = find_path_to_target_or_leaf(graph, start_node=node, randomize=randomize)
        if path_to_leaf is None:
            continue
        
        # Get paths to and from the additional node
        path_to_add_node = find_path_to_root(graph, start_node=node, randomize=randomize)
        # reverse path_to_add_node to have from root to node
        path_to_add_node = path_to_add_node[::-1] if path_to_add_node is not None else []
        # just select a part of path to leaf
        random_length = random.randint(2, len(path_to_leaf))
        path_from_add_node = path_to_leaf[:random_length]
        
        if path_to_add_node and path_from_add_node:
            extended_subgraph = list(set(extended_subgraph + path_to_add_node + path_from_add_node))
    
    return extended_subgraph

def get_subgraphs(graph, y_index, min_number_subgraphs = 3, max_number_subgraphs = 10, modality = 'random_nodes', randomly_eliminate_task_from_subgraphs = True, dict_subgraph_with_add_nodes = {}):
    """
    This function generates n_subgraphs from the original graph.

    If modality is 'random_nodes', the function generates subgraphs composed by random nodes in the graphs.
    If modality is 'connected_nodes', the function generates subgraphs composed by connected nodes in the graphs (trees).

    The following conditions must be satisfied:
    - Each subgraph have at least two nodes
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
        add_nodes_values: The list of additional nodes used
        subgraphs_with_add_nodes: List of boolean indicating which subgraphs contain additional nodes
    """
    from src.utils import get_roots, get_task_graph

    if min_number_subgraphs > max_number_subgraphs:
        raise ValueError("min_number_subgraphs must be less than or equal to max_number_subgraphs.")

    ### INITIALIZATION ###
    torch_graph = torch.tensor(graph.values)

    nodes_covered = set()
    n_subgraphs_generated = 0
    subgraphs = []
    subgraphs_with_add_nodes = []

    # subgraph_reaching_task = random.choice([0,1]) #task in common
    # select nodes to cover
    nodes_to_cover = list(range(len(graph)))
    nodes_to_cover.sort()

    # get the graph starting from y_index, i.e., the one considered in c2bm and cgm
    y_index_graph = get_task_graph(torch.tensor(graph.values), task_node = y_index)
    y_index_graph = [int(item) for sublist in y_index_graph for item in sublist]
    roots = get_roots(torch_graph)
    roots = torch.nonzero(roots).squeeze()

    # get edges to cover
    couples_parents_children = find_all_parent_child_pairs(graph, y_index_graph)
    couples_covered = set()
    subgraph_from_missing_root = None
    n_subgraphs_add_nodes_to_generate = 0
    n_subgraphs_add_nodes_generated = 0

    # manage additional nodes to include in specific subgraphs
    if dict_subgraph_with_add_nodes:
        add_nodes_modality = dict_subgraph_with_add_nodes.get('modality', 'random')
        add_nodes_number = dict_subgraph_with_add_nodes.get('number_add_nodes', 1)
        n_subgraphs_add_nodes_to_generate = dict_subgraph_with_add_nodes.get('number_subgraphs_add_nodes', 1)
        n_subgraphs_add_nodes_generated = 0
        if add_nodes_modality== 'specific_nodes':
            custom_nodes = dict_subgraph_with_add_nodes.get('specific_nodes_names', [])
            custom_values = [graph.columns.get_loc(name) for name in custom_nodes if name in graph.columns]
        else:
            custom_values = []


        # Validate parameters for additional nodes
        if add_nodes_number <1:
            raise ValueError("number_add_nodes must be at least 1")
        if add_nodes_number >= len(y_index_graph)-1:
            raise ValueError("number_add_nodes exceeds the number of available nodes in the y_index subgraph minus one: {}".format(len(y_index_graph)-1))
        if n_subgraphs_add_nodes_to_generate < 1:
            raise ValueError("number_subgraphs_add_nodes must be at least 1")
        
        # at least one subgraph without additional nodes
        if n_subgraphs_add_nodes_to_generate > (max_number_subgraphs-1):
            raise ValueError("Number of subgraphs with additional nodes to generate exceeds the maximum number of subgraphs-1, i.e., number of clients-1: {}".format(max_number_subgraphs -1 ))
        if n_subgraphs_add_nodes_to_generate < 1:
            raise ValueError("Number of subgraphs with additional nodes to generate must be at least 1")
           
        # Generate initial add_nodes_values
        add_nodes_values = generate_add_nodes_values(
            graph, torch_graph, y_index_graph, y_index, 
            add_nodes_modality, add_nodes_number, add_nodes_values= custom_values
        )

    else:
        add_nodes_values = []



    ### STEP 1: GENERATE SUBGRAPHS UNTIL COVERING ALL NODES IN THE GRAPH AND COVERING ALL EDGES AND THE NUMBER OF SUBGRAPHS IS AT LEAST min_number_subgraphs ###
    max_retries = 3
    retry_count = 0
    hist_add_nodes_values = []


    def check_condition():
        base_condition = len(nodes_covered) < len(nodes_to_cover) or n_subgraphs_generated < min_number_subgraphs or len(couples_covered) < len(couples_parents_children)
        if dict_subgraph_with_add_nodes:
            return base_condition or n_subgraphs_add_nodes_generated < n_subgraphs_add_nodes_to_generate
        return base_condition
    
    def get_pairs_in_subgraph(subgraph, all_pairs):
        """Helper function to find which parent-child pairs are contained in a subgraph."""
        pairs_in_subgraph = set()
        for parent, child in all_pairs:
            if parent in subgraph and child in subgraph:
                pairs_in_subgraph.add((parent, child))
        return pairs_in_subgraph

    while_iterations = 0
    while check_condition():
        while_iterations = while_iterations +1
        if while_iterations > 500:
            raise ValueError("Exceeded maximum iterations while generating subgraphs. Please revise additional nodes logic or the number of subgraphs to generate.")
        
        try:
            # generate a subgraph
            if modality == 'random_nodes':
                raise NotImplementedError("Modality 'random_nodes' is not implemented in this version.")
                #subgraph = random.sample(nodes_to_cover, random.randint(2, len(nodes_to_cover)))
                
            elif modality == 'connected_nodes':
                # Generate task indices

                # Randomly include some node chosen in the whole graph (excluding roots and nodes in y_index) in such a way to be able to cover the whole graph
                subgraph_from_missing_root_none = False
                task_indices = []
                remaining_nodes = [node for node in nodes_to_cover if node not in roots and node not in y_index_graph and node not in nodes_covered]
                if len(remaining_nodes) != 0:
                    random_number = random.randint(0, max(1, (len(remaining_nodes)//3)))
                    task_indices = random.sample([node for node in remaining_nodes if node not in roots and node not in y_index_graph], random_number)

                # Add remaining roots if any
                if remaining_nodes == []:
                    
                    # there can still be missing nodes that are roots not in y_index_graph
                    missing_roots = [node for node in nodes_to_cover if (node not in nodes_covered) and (node in roots)]
                    if missing_roots !=[]:
                        # add as task_indices a descendant of missing_roots
                        childrens = []
                        for node in missing_roots:
                            if node in y_index_graph:
                                # Generate the subgraph directly from this root to y_index
                                subgraph_from_missing_root = find_path_to_target_or_leaf(graph, start_node=node, end_node=y_index, randomize=True, nodes_not_allowed=add_nodes_values)
                                if subgraph_from_missing_root is None:
                                    if retry_count == max_retries:
                                        subgraph_from_missing_root_none = True
                                        subgraph_from_missing_root = find_path_to_target_or_leaf(graph, start_node=node, end_node=y_index, randomize=True)
                                        print(f" Warning: There are problems in generating subgraphs from root {graph.columns[node]} to task {graph.columns[y_index]} that do not include additional nodes {list(graph.columns[add_nodes_values])}. I will add a subgraph with additional nodes.")
                                    else:
                                        raise ValueError(f"Could not generate subgraph from root {graph.columns[node]} to task {graph.columns[y_index]} that do not include additional nodes {list(graph.columns[add_nodes_values])}.")
                                break
                            else:
                                curr_childrens = torch.where(torch_graph[node,:] == 1)[0].tolist()
                                childrens.extend(curr_childrens)
                        
                        if subgraph_from_missing_root is None:
                            if childrens == []:
                                childrens = [node]
                            random_childrens = random.sample(childrens, max(len(childrens)//3, 1))
                            task_indices.extend(random_childrens) 
                            
            

                # Include the main task
                #if task_in_common:
                task_indices.append(y_index)
                #else:
                    # Ensure one subgraph reaches the task
                    #if n_subgraphs_generated == subgraph_reaching_task:
                    #    task_indices.append(y_index)
                    #else:
                    #    if not dict_subgraph_with_add_nodes:
                    #        task_indices.append(random.choice([node for node in y_index_graph if node not in roots]))
                    #    else:
                    #        if n_subgraphs_generated % 2 == 0 and n_subgraphs_add_nodes_generated < n_subgraphs_add_nodes_to_generate:
                    #            task_indices.append(random.choice([node for node in y_index_graph if node not in roots]))
                    #        else:
                    #            # Exclude from task_indices the nodes in add_nodes_values
                    #            if len(task_indices)!=0:
                    #                task_indices = [node for node in task_indices if node not in add_nodes_values]
                    #            task_indices.append(random.choice([node for node in y_index_graph if node not in roots and node not in add_nodes_values]))


                # Handle additional nodes if required
                if not dict_subgraph_with_add_nodes:
                    # Simple case: no additional nodes
                    if subgraph_from_missing_root is not None:
                        subgraph = subgraph_from_missing_root
                    else:
                        subgraph = generate_base_subgraph(graph, task_indices, randomize=True)
                    has_add_nodes = False

                else:
                    # Complex case: include or exclude additional nodes
                    if (n_subgraphs_generated % 2 == 0 and n_subgraphs_add_nodes_generated < n_subgraphs_add_nodes_to_generate) or subgraph_from_missing_root_none:
                        # Include additional nodes
                        if subgraph_from_missing_root is not None:
                            subgraph = subgraph_from_missing_root
                        else:
                            subgraph = generate_base_subgraph(graph, task_indices, randomize=True)
                        subgraph = add_additional_nodes_to_subgraph(graph, subgraph, add_nodes_values, randomize=True)
                        has_add_nodes = True
                    
                    else:
                        # NOTE: PAY ATTENTION WITH SUBGRAPHS FROM MISSING ROOT
                        # Exclude additional nodes
                        if subgraph_from_missing_root is not None:
                            subgraph = subgraph_from_missing_root
                        else:
                            subgraph = generate_base_subgraph(graph, task_indices, randomize=True, 
                                                            nodes_not_allowed=add_nodes_values)
                        has_add_nodes = False

            if while_iterations > 100:
                raise ValueError("Exceeded maximum iterations. I will change the add_nodes_values and restart subgraph generation.")
      
            # Check for duplicate subgraphs
            if set(subgraph) in map(set, subgraphs):
                continue

            # there could be some missing nodes/ couples difficult to cover, add them forcibly
            if n_subgraphs_generated > 20 and  n_subgraphs_generated % 2 == 0:
                last_missing_nodes = [node for node in nodes_to_cover if node not in nodes_covered]
                last_missing_couples = [pair for pair in couples_parents_children if pair not in couples_covered]
                if last_missing_nodes != []:
                    for node in last_missing_nodes:
                        subgraph = add_additional_nodes_to_subgraph(graph, subgraph, [node], randomize=True)
                        break
                if last_missing_couples != []:
                    for pair in last_missing_couples:
                        subgraph = add_additional_nodes_to_subgraph(graph, subgraph, [pair[0], pair[1]], randomize=True)
                        break
                subgraph = add_additional_nodes_to_subgraph(graph, subgraph, add_nodes_values, randomize=True)
                has_add_nodes = True
                print("Forcibly added missing nodes/couples to subgraph to ensure coverage. There could be some more subgraph with add_nodes.")
             

            # Add subgraph
            subgraphs.append(subgraph)
            subgraphs_with_add_nodes.append(has_add_nodes)
            nodes_covered = nodes_covered.union(set(subgraph))
            # Update covered parent-child pairs
            pairs_in_current_subgraph = get_pairs_in_subgraph(subgraph, couples_parents_children)
            couples_covered = couples_covered.union(pairs_in_current_subgraph)
            n_subgraphs_generated += 1
            if dict_subgraph_with_add_nodes:
                n_subgraphs_add_nodes_generated += int(has_add_nodes)
            retry_count = 0  # Reset on success
            subgraph_from_missing_root = None


        except (ValueError, IndexError, AttributeError) as e:
            retry_count += 1
            hist_add_nodes_values.append(add_nodes_values[0])
            print(f"Error generating subgraph (attempt {retry_count}/{max_retries}): {e}")
            
            if retry_count > max_retries:
                print(f"Max retries reached. Restarting subgraph generation from the beginning.")
                
                # Re-initialize all variables
                nodes_covered = set()
                n_subgraphs_generated = 0
                subgraphs = []
                subgraphs_with_add_nodes = []
                couples_covered = set()
                subgraph_from_missing_root = None
                retry_count = 0
                #hist_add_nodes_values = []
                
                if dict_subgraph_with_add_nodes:
                    n_subgraphs_add_nodes_generated = 0
                    # Regenerate add_nodes_values from scratch
                    if add_nodes_modality == 'specific_nodes':
                        raise ValueError("nodes specified in specific_nodes_names caused repeated failures. Please revise the provided nodes.")
                    else:
                        custom_values = []
                    
                    add_nodes_values = generate_add_nodes_values(
                        graph, torch_graph, y_index_graph, y_index,
                        add_nodes_modality, add_nodes_number, old_add_nodes_values=hist_add_nodes_values
                    )
                    print(f"Restarted with new add_nodes_values: {list(graph.columns[add_nodes_values])}")
                    while_iterations = 0
                
                continue

 
    ### STEP 2: MERGE SUBGRAPHS IF NEEDED ###
    print(f"Generated {len(subgraphs)} subgraphs. Max allowed: {max_number_subgraphs}")

    if len(subgraphs) > max_number_subgraphs:
        print("Warning: Number of subgraphs exceeds the maximum allowed. Some subgraphs will be merged trying to mantain balance between those with and without additional nodes (if present).")
        # Separate subgraphs with and without additional nodes
        subgraphs_with_add = [(i, sg) for i, sg in enumerate(subgraphs) if subgraphs_with_add_nodes[i]]
        subgraphs_without_add = [(i, sg) for i, sg in enumerate(subgraphs) if not subgraphs_with_add_nodes[i]]
        
        print(f"Pre-merging subgraphs WITH additional nodes: {len(subgraphs_with_add)}")
        print(f"Pre-merging subgraphs WITHOUT additional nodes: {len(subgraphs_without_add)}")


        # First, try to merge within groups
        # Merge subgraphs WITH additional nodes
        # while len(subgraphs_with_add) > n_subgraphs_add_nodes_to_generate and len(subgraphs_with_add) > 1:
        #     subgraphs_with_add = sorted(subgraphs_with_add, key=lambda x: len(x[1]))
        #     first = subgraphs_with_add.pop(0)
        #     second = subgraphs_with_add.pop(0)
        #     merged = (first[0], list(set(first[1] + second[1])))
        #     subgraphs_with_add.append(merged)

        # Merge subgraphs WITHOUT additional nodes
        while len(subgraphs_with_add) + len(subgraphs_without_add) > max_number_subgraphs and len(subgraphs_without_add) > 1:
            subgraphs_without_add = sorted(subgraphs_without_add, key=lambda x: len(x[1]))
            first = subgraphs_without_add.pop(0)
            second = subgraphs_without_add.pop(0)
            merged = (first[0], list(set(first[1] + second[1])))
            subgraphs_without_add.append(merged)

        # Check if we still exceed max_number_subgraphs
        #total_after_merge = len(subgraphs_with_add) + len(subgraphs_without_add)

        #if total_after_merge > max_number_subgraphs:
        #    print(f"Still have {total_after_merge} subgraphs after group merge. Further merging needed.")
        #    # Merge across groups if necessary
        #    all_subgraphs = subgraphs_with_add + subgraphs_without_add
            
        #    while len(all_subgraphs) > max_number_subgraphs:
        #        all_subgraphs = sorted(all_subgraphs, key=lambda x: len(x[1]))
        #        first = all_subgraphs.pop(0)
        #        second = all_subgraphs.pop(0)
        #        # The merged subgraph has additional nodes if either had them
        #        has_add = first[0] in [idx for idx, _ in subgraphs_with_add] or second[0] in [idx for idx, _ in subgraphs_with_add]
        #        merged = (first[0], list(set(first[1] + second[1])))
        #        all_subgraphs.append(merged)
                
                # Update tracking
        #        if has_add:
        #            # merge the merged with one of the subgraphs_with_add
        #            subgraphs_with_add = [sg for sg in all_subgraphs if sg[0] in [idx for idx, _ in subgraphs_with_add] or sg == merged]
        #        
        #    subgraphs = [sg for _, sg in all_subgraphs]
        #    subgraphs_with_add_nodes = [any(node in add_nodes_values for node in sg) for sg in subgraphs]

        #else:


        # Recombine
        subgraphs = [sg for _, sg in subgraphs_with_add] + [sg for _, sg in subgraphs_without_add]
        subgraphs_with_add_nodes = [True] * len(subgraphs_with_add) + [False] * len(subgraphs_without_add)

        
        # Ensure at least one subgraph with and one without additional nodes
        if dict_subgraph_with_add_nodes:
            has_with = any(subgraphs_with_add_nodes)
            has_without = any(not flag for flag in subgraphs_with_add_nodes)
            
            if not has_with or not has_without:
                raise ValueError("After merging, subgraphs do not contain both types (with and without additional nodes) as required.")

    print("=== Subgraph generation completed ===")
    print("Summary:")
    print(f"Final: {len(subgraphs)} subgraphs")
    if len(add_nodes_values) > 0:
        print("Additional nodes used:", [graph.columns[node_idx] for node_idx in add_nodes_values])
        print(f"Subgraphs with additional nodes: {sum(subgraphs_with_add_nodes)}")
        print(f"Subgraphs without additional nodes: {sum(not flag for flag in subgraphs_with_add_nodes)}")
    #print(f"Parent-child pairs covered: {len(couples_covered)}/{len(couples_parents_children)}")

    # return a dictionary with soubgroups as keys and the nodes as values
    subgraphs = {f'subgraph_{i+1}': s for i, s in enumerate(subgraphs)}
    # return a dictionary with the subgroups as keys and the nodes names as values
    subgraphs_concept_names = {f'subgraph_{i+1}':[graph.columns[node_idx] for node_idx in s] for i, s in enumerate(subgraphs.values())}

    # Randomly eliminate task from some subgraphs if needed but only if I do not eliminate couples parents-children
    if randomly_eliminate_task_from_subgraphs:
        #assert task_in_common, "randomly_eliminate_task_from_subgraphs can be True only if task_in_common is True, otherwise there are already subgraphs without the task."
        # Get parents of y_index, note: they are in y_index_graph by definition
        parents_of_task = [parent for parent, child in couples_parents_children if child == y_index]
        
        for i in range(len(subgraphs)):
            if y_index in subgraphs[f'subgraph_{i+1}']:
                # Check if removing y_index would eliminate a parent-child pair
                # This happens if there's a parent of y_index in this subgraph that is not in any other subgraph
                can_remove = True
                for parent in parents_of_task:
                    if parent in subgraphs[f'subgraph_{i+1}']:
                        # Check if this parent exists in another subgraph
                        parent_in_other_subgraph = False
                        for j in range(len(subgraphs)):
                            if j != i and parent in subgraphs[f'subgraph_{j+1}'] and y_index in subgraphs[f'subgraph_{j+1}']:
                                parent_in_other_subgraph = True
                                break
                        if not parent_in_other_subgraph:
                            # This parent-child pair would be lost
                            can_remove = False
                            break
                
                # Only remove if it doesn't eliminate any parent-child pair and randomly decide
                if can_remove and random.random() < 0.5:
                    subgraphs[f'subgraph_{i+1}'].remove(y_index)
                    subgraphs_concept_names[f'subgraph_{i+1}'].remove(graph.columns[y_index])

    add_nodes_names = [graph.columns[node_idx] for node_idx in add_nodes_values]


    ### FINAL VALIDATION CHECKS ###
    print("\n=== Performing final validation checks ===")


    # 0. Check that number of subgraphs is less than n
    assert len(subgraphs) <= max_number_subgraphs, "Number of subgraphs must be lower than n"

    # 0.bis Check that if there are add_nodes, at least one subgraph has them and one doesn't
    if dict_subgraph_with_add_nodes:
        has_with = any(subgraphs_with_add_nodes)
        has_without = any(not flag for flag in subgraphs_with_add_nodes)
        if not has_with:
            raise ValueError("At least one subgraph with additional nodes is required, but none found.")
        if not has_without:
            raise ValueError("At least one subgraph without additional nodes is required, but none found.")
        print(f"✓ Subgraphs with and without additional nodes are present.")
    
    # 1. Check that all nodes are covered by the union of subgraphs
    all_nodes_in_graph = set(range(len(graph)))
    all_nodes_covered = set()
    for sg in subgraphs.values():
        all_nodes_covered.update(sg)
    
    missing_nodes = all_nodes_in_graph - all_nodes_covered
    if missing_nodes:
        missing_node_names = [graph.columns[node] for node in missing_nodes]
        raise ValueError(f"The following nodes are not covered by any subgraph: {missing_node_names} (indices: {missing_nodes})")
    print(f"✓ All {len(all_nodes_in_graph)} nodes are covered by subgraphs")
    
    # 2. Check that all parent-child couples are covered by the union of subgraphs
    covered_couples = set()
    for sg in subgraphs.values():
        for parent, child in couples_parents_children:
            if parent in sg and child in sg:
                covered_couples.add((parent, child))
    
    missing_couples = set(couples_parents_children) - covered_couples
    if missing_couples:
        missing_couples_names = [(graph.columns[p], graph.columns[c]) for p, c in missing_couples]
        raise ValueError(f"The following parent-child pairs are not covered by any subgraph: {missing_couples_names}")
    print(f"✓ All {len(couples_parents_children)} parent-child pairs are covered")
    
    # 3. Check that subgraphs with add_nodes actually contain them and those without don't
    if dict_subgraph_with_add_nodes:
        for i, (key, sg) in enumerate(subgraphs.items()):
            should_have_add_nodes = subgraphs_with_add_nodes[i]
            has_all_add_nodes = all(node in sg for node in add_nodes_values)
            has_at_least_one_add_node = any(node in sg for node in add_nodes_values)

            
            if should_have_add_nodes and not has_all_add_nodes:
                raise ValueError(f"Subgraph {key} is marked as having additional nodes but doesn't contain any of {add_nodes_values}")
            
            if not should_have_add_nodes and has_at_least_one_add_node:
                found_add_nodes = [node for node in sg if node in add_nodes_values]
                raise ValueError(f"Subgraph {key} is marked as NOT having additional nodes but contains {found_add_nodes}")
        
        print(f"✓ Additional nodes validation passed:")
        if len(add_nodes_values) > 0:
            print(f"  - {sum(subgraphs_with_add_nodes)} subgraphs with additional nodes contain them")
            print(f"  - {sum(not flag for flag in subgraphs_with_add_nodes)} subgraphs without additional nodes don't contain them")
    
    print("=== All validation checks passed ===\n")

    return subgraphs, subgraphs_concept_names, subgraphs_with_add_nodes, add_nodes_values, add_nodes_names

def build_client_subgraph_ids(
    n_train_clients,
    subgraphs_with_add_nodes,
    dataset_client_multiplier=1,
    drift_add_nodes_ratio=0.5,
    ):
    
    if dataset_client_multiplier < 1:
        raise ValueError("dataset_client_multiplier must be >= 1.")
    if drift_add_nodes_ratio < 0 or drift_add_nodes_ratio > 1:
        raise ValueError("drift_add_nodes_ratio must be between 0 and 1.")

    n_dataset_clients = n_train_clients * dataset_client_multiplier
    if n_dataset_clients == n_train_clients:
        return n_dataset_clients, None

    add_ids = [i for i, flag in enumerate(subgraphs_with_add_nodes) if flag]
    no_add_ids = [i for i, flag in enumerate(subgraphs_with_add_nodes) if not flag]
    if not no_add_ids:
        raise ValueError("No subgraphs without additional nodes are available.")

    n_extra_clients = n_dataset_clients - n_train_clients
    n_extra_add = int(round(n_extra_clients * drift_add_nodes_ratio))
    n_extra_add = max(0, min(n_extra_add, n_extra_clients))
    n_extra_no_add = n_extra_clients - n_extra_add

    def cycle_ids(ids, count):
        return [ids[i % len(ids)] for i in range(count)]

    client_subgraph_ids = []
    client_subgraph_ids.extend(cycle_ids(no_add_ids, n_train_clients))

    if n_extra_add > 0 and not add_ids:
        print("Warning: drift_add_nodes_ratio requested, but no subgraphs with additional nodes were generated.")
        n_extra_no_add = n_extra_clients
        n_extra_add = 0

    if n_extra_add > 0:
        client_subgraph_ids.extend(cycle_ids(add_ids, n_extra_add))
    if n_extra_no_add > 0:
        client_subgraph_ids.extend(cycle_ids(no_add_ids, n_extra_no_add))

    return n_dataset_clients, client_subgraph_ids

def build_client_subgraph_ids_for_datasets(
    n_train_clients,
    n_datasets,
    dataset_client_multiplier=1,
    drift_dataset_ratio=0.5,
):
    if dataset_client_multiplier < 1:
        raise ValueError("dataset_client_multiplier must be >= 1.")
    if drift_dataset_ratio < 0 or drift_dataset_ratio > 1:
        raise ValueError("drift_dataset_ratio must be between 0 and 1.")
    if n_datasets < 2:
        raise ValueError("At least two datasets are required.")

    n_dataset_clients = n_train_clients * dataset_client_multiplier
    client_subgraph_ids = [0] * n_train_clients

    n_extra_clients = n_dataset_clients - n_train_clients
    if n_extra_clients == 0:
        return n_dataset_clients, client_subgraph_ids

    n_extra_drift = int(round(n_extra_clients * drift_dataset_ratio))
    n_extra_drift = max(0, min(n_extra_drift, n_extra_clients))
    n_extra_remaining = n_extra_clients - n_extra_drift

    extra_ids = []
    if n_extra_drift > 0:
        extra_ids.extend([1] * n_extra_drift)

    if n_extra_remaining > 0:
        if n_datasets == 2:
            extra_ids.extend([0] * n_extra_remaining)
        else:
            other_ids = list(range(2, n_datasets))
            extra_ids.extend([other_ids[i % len(other_ids)] for i in range(n_extra_remaining)])

    client_subgraph_ids.extend(extra_ids)
    return n_dataset_clients, client_subgraph_ids

def generate_split(cfg, datasets, graph, y_index):

    n = cfg.learning.n_clients
    dataset_client_multiplier = int(cfg.learning.subgraphs.get('dataset_client_multiplier', 1))
    drift_add_nodes_ratio = cfg.learning.subgraphs.get('drift_add_nodes_ratio', 0.5)
    n_dataset_clients = n
    client_subgraph_ids = None

    if len(datasets)>1:
        # if cfg.learning.subgraphs.get('dict_subgraph_with_add_nodes', {}) != {}:
        #     raise NotImplementedError("When multiple datasets are used, it is not possible to use additional nodes in subgraphs.")
        
        # Create a subgraph for each client containing all the variables and values from one dataset
        subgraphs = {}
        subgraphs_concept_names = {}
        for i, dataset in enumerate(datasets.values()):
           # get variables name understanding columns that are not all equal to -1
           subgraphs_concept_names[f'subgraph_{i+1}'] = [dataset.c_info['names'][col] for col in range(dataset.data['train'].c.size(1)) if not torch.all(dataset.data['train'].c[:,col] == -1)]
           subgraphs[f'subgraph_{i+1}'] = [dataset.c_info['names'].index(name) for name in subgraphs_concept_names[f'subgraph_{i+1}']]
           # add y if it has values different from -1
           #if not torch.all(dataset.data['train'].y == -1):
           #    subgraphs_concept_names[f'subgraph_{i+1}'] += dataset.y_info['names']
           #    subgraphs[f'subgraph_{i+1}'] += [y_index]
        
        subgraphs_with_add_nodes = [False] * len(subgraphs)
        add_nodes_values = []
        add_nodes_names = []
        indices_subgraphs_reaching_task = list(range(1, len(subgraphs) + 1))
        combined_cfg = cfg.get('combined_datasets', {})
        drift_dataset_ratio = combined_cfg.get('drift_dataset_ratio', 0.5)
        n_dataset_clients, client_subgraph_ids = build_client_subgraph_ids_for_datasets(
            n_train_clients=n,
            n_datasets=len(subgraphs),
            dataset_client_multiplier=dataset_client_multiplier,
            drift_dataset_ratio=drift_dataset_ratio,
        )

        n_extra_clients = n_dataset_clients - n
        if n_extra_clients > 0:
            extra_counts = {i: client_subgraph_ids[n:].count(i) for i in range(len(subgraphs))}
            extra_desc = ", ".join(
                f"{count} on subgraph_{idx + 1}"
                for idx, count in extra_counts.items()
                if count > 0
            )
            print(
                f"Dataset clients: {n_dataset_clients} (base {n} on subgraph_1, "
                f"extra {n_extra_clients}: {extra_desc})"
            )

    else:
        if cfg.learning.subgraphs.rnd_drift > 1:
           assert cfg.learning.subgraphs.get('dict_subgraph_with_add_nodes', {}) != {}, "To use rnd_drift > 1, you must specify dict_subgraph_with_add_nodes in the config."
        # Get the subgraph for each client
        subgraphs, subgraphs_concept_names, subgraphs_with_add_nodes, add_nodes_values, add_nodes_names = get_subgraphs(graph, 
                                                                         y_index, 
                                                                         min_number_subgraphs= cfg.learning.subgraphs.get('min_number_subgraphs', 2),
                                                                         max_number_subgraphs = cfg.learning.subgraphs.get('max_number_subgraphs', cfg.learning.n_clients),
                                                                         modality=cfg.learning.subgraphs.modality,
                                                                         #concept_in_common=cfg.learning.subgraphs.concept_in_common,
                                                                         #task_in_common=cfg.learning.subgraphs.task_in_common,
                                                                         randomly_eliminate_task_from_subgraphs=cfg.learning.subgraphs.get('randomly_eliminate_task_from_subgraphs', True),
                                                                         dict_subgraph_with_add_nodes=cfg.learning.subgraphs.get('dict_subgraph_with_add_nodes', {})
                                    )
        
        # eliminate y_index from each subgraph but save from which I eliminated it
        indices_subgraphs_reaching_task = []
        for key in subgraphs.keys():
            if y_index in subgraphs[key]:
                indices_subgraphs_reaching_task.append(int(key.split('_')[1]))
                subgraphs[key].remove(y_index)
                subgraphs_concept_names[key].remove(graph.columns[y_index])
        
    # If the task is not included, select some subgraphs to mask the y variable
    #if not cfg.learning.annotation_assumption == "task_included":
        #r = random.randint(1, (len(subgraphs)-1))  # Randomly select r subgraphs to mask y variable
        #subgraphs_task_excluded = random.sample(range(1, len(subgraphs) + 1), r)
        # subgraphs_task_excluded are those that do not reach the task, so the complementary of indices_subgraphs_reaching_task
    subgraphs_task_excluded = [i for i in range(1, len(subgraphs) + 1) if i not in indices_subgraphs_reaching_task]
    #else:
    #    subgraphs_task_excluded = None

    if len(datasets) == 1 and dataset_client_multiplier != 1:
        n_dataset_clients, client_subgraph_ids = build_client_subgraph_ids(
            n_train_clients=n,
            subgraphs_with_add_nodes=subgraphs_with_add_nodes,
            dataset_client_multiplier=dataset_client_multiplier,
            drift_add_nodes_ratio=drift_add_nodes_ratio,
        )

        n_extra_clients = n_dataset_clients - n
        n_extra_add = sum(
            1 for idx in client_subgraph_ids[n:] if subgraphs_with_add_nodes[idx]
        )
        print(
            f"Dataset clients: {n_dataset_clients} (base {n} without additional nodes, "
            f"extra {n_extra_clients}: {n_extra_add} with additional nodes)"
        )

    print("\nClient concept coverage:")
    for i in range(n_dataset_clients):
        if client_subgraph_ids is None:
            subgraph_idx = i % len(subgraphs)
        else:
            subgraph_idx = client_subgraph_ids[i]
        subgraph_key = f"subgraph_{subgraph_idx + 1}"
        concepts = subgraphs_concept_names.get(subgraph_key, [])
        print(f"client {i + 1}: {subgraph_key} -> {concepts}")
    
    # clean the directory
    root = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
    path = os.path.join(root)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    split_and_save(cfg, datasets, graph, 'train', n_dataset_clients, subgraphs, subgraphs_task_excluded, root, client_subgraph_ids)
    split_and_save(cfg, datasets, graph, 'val', n_dataset_clients, subgraphs, subgraphs_task_excluded, root, client_subgraph_ids)
    split_and_save(cfg, datasets, graph, 'test', n_dataset_clients, subgraphs, subgraphs_task_excluded, root, client_subgraph_ids)
    # Save the dataloader for the unique, real test-set (if not combined datasets)
    if len(datasets) == 1:
        test_dataloader = DataLoader(datasets[0].data['test'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
        root = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
        path = os.path.join(root, f"test.pkl")
        with open(path, 'wb') as f:
            pickle.dump(test_dataloader, f)

    return subgraphs, subgraphs_concept_names, subgraphs_with_add_nodes, add_nodes_values, add_nodes_names

def split_and_save(cfg, datasets, graph, set, n, subgraphs = None, subgraphs_task_excluded = None, root = None, client_subgraph_ids = None):

        if len(datasets)==1:
            dataset = datasets[0]
            # Create as many splits as the number of clients

            x, c, y = [], [], []
            for row in dataset.data[set]:
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

        # Swap concept values for a fraction of clients: invert class labels within selected concepts
        swapping_clients = cfg.learning.get('swapping_clients', 0.0)
        swapping_concepts = cfg.learning.get('swapping_concepts', 0.0)
        swapping_factor = cfg.learning.get('swapping_factor', 0.0)
        # Per-concept mapping: concept_idx -> set of client indices that will swap it
        swap_concept_clients = {}

        if swapping_clients > 0 and swapping_concepts > 0 and set == 'train':
            # 1) Collect all available (non-masked) concept indices across all subgraphs
            all_available_concepts = list(range(len(dataset.c_info['names']))) if len(datasets) == 1 else []

            # 2) Choose which concepts to swap
            n_swap = max(1, round(len(all_available_concepts) * swapping_concepts))
            swap_concept_indices = random.sample(all_available_concepts, n_swap)

            # 3) For each concept, find eligible clients (those who have it in their subgraph)
            #    and select swapping_clients fraction of them
            for col in swap_concept_indices:
                eligible = []
                for i in range(n):
                    j = i % len(subgraphs) if client_subgraph_ids is None else client_subgraph_ids[i]
                    client_concepts = subgraphs[f'subgraph_{j+1}']
                    if col in client_concepts:
                        eligible.append(i)
                if eligible:
                    n_sel = max(1, round(len(eligible) * swapping_clients))
                    n_sel = min(n_sel, len(eligible)-1) # Ensure at least one client remains unchanged
                    selected = random.sample(eligible, n_sel)
                    swap_concept_clients[col] = {s for s in selected}
                    print(f"[Swap] Concept {col}: eligible {len(eligible)}, selected clients {sorted(selected)}")

            print(f"[Swap] Concepts to swap: {swap_concept_indices}, factor: {swapping_factor}")

        # For each split, create a dataloader containing x, c, y 
        for i in range(n):
            if client_subgraph_ids is None:
                j = i % len(subgraphs)
            else:
                j = client_subgraph_ids[i]
                # print(f"Client {i+1} uses subgraph {j+1}")

            if len(datasets) == 1:
                masked_c_splits = apply_mask(c_splits[i], subgraphs[f'subgraph_{j+1}'])

                if subgraphs_task_excluded is not None and ((j+1) in subgraphs_task_excluded):
                    # If the task is not included, mask the y variable as well
                    masked_y_splits = -1 * torch.ones_like(y_splits[i])  # Mask y variable
                else:
                    masked_y_splits = y_splits[i]
                
                x_i = x_splits[i]
            else:
                x_i = datasets[j].data[set].X
                masked_c_splits = datasets[j].data[set].c

                if subgraphs_task_excluded is not None and ((j+1) in subgraphs_task_excluded):
                    # If the task is not included, mask the y variable as well
                    masked_y_splits = -1 * torch.ones_like(datasets[j].data[set].y)  # Mask y variable
                else:
                    masked_y_splits = datasets[j].data[set].y
                
            if swap_concept_clients:
                # Determine which concepts this client should swap
                concepts_to_swap_here = [col for col, clients in swap_concept_clients.items() if i in clients]
                if concepts_to_swap_here:
                    rng = torch.Generator()
                    rng.manual_seed(42 + i)
                    for col in concepts_to_swap_here:
                        valid = masked_c_splits[:, col] != -1
                        if valid.any():
                            vals = masked_c_splits[valid, col]
                            unique_vals = vals.unique().sort()[0]
                            if len(unique_vals) >= 2:
                                v0, v1 = unique_vals[0], unique_vals[1]
                                new_vals = vals.clone()
                                # Only swap a fraction of the valid values
                                n_valid = valid.sum().item()
                                n_to_swap = max(1, round(n_valid * swapping_factor))
                                swap_row_indices = torch.randperm(n_valid, generator=rng)[:n_to_swap]
                                swap_mask = torch.zeros(n_valid, dtype=torch.bool)
                                swap_mask[swap_row_indices] = True
                                new_vals[swap_mask & (vals == v0)] = v1
                                new_vals[swap_mask & (vals == v1)] = v0
                                new_vals[~swap_mask] = vals[~swap_mask]
                                masked_c_splits[valid, col] = new_vals
                    print(f"[Swap] Client {i+1}: swapped concepts {concepts_to_swap_here}, factor {swapping_factor}")

            dataloader = DataLoader(
                CustomDataset(x_i, masked_c_splits, masked_y_splits, graph),
                batch_size=cfg.dataset.batch_size,
                collate_fn=static_graph_collate
            )

            # Store the dataloader in the 
            path = os.path.join(root, f"{set}set_{i+1}_subgraph_{j+1}.pkl") # Start to count from 1
            # print(f"Saving dataloader for {set} set, client {i+1}, subgraph {j+1} at {path}")
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

from typing import Tuple, Any, Optional, Callable

def generate_split_with_fallback(
    cfg: dict,
    datasets: Any,
    graph: Any,
    y_index: Any,
    *,
    seed_key: str = "seed",
    step: int = 100,
    max_tries: int = 20,
    seed_everything_fn: Optional[Callable[[int], None]] = None,
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Calls `generate_split(cfg, datasets, graph, y_index)` with seed fallbacks.
    On each failure, tries seed = base_seed + k*step for k=0..max_tries-1.
    Always restores the original seed in cfg and reseeds it (if seed_everything_fn is given).
    """
    if cfg.learning.get("seed_plot_interventions") is not None:
        original_seed = cfg.learning.get("seed_plot_interventions")
    else:
        original_seed = cfg.get(seed_key, 0)
    base_seed = int(original_seed) if original_seed is not None else 0

    last_err: Optional[Exception] = None
    try:
        for k in range(max_tries):
            trial_seed = base_seed + k * step
            try:
                cfg[seed_key] = trial_seed
                if seed_everything_fn is not None:
                    seed_everything_fn(trial_seed)

                return generate_split(cfg, datasets, graph, y_index)

            except Exception as e:
                last_err = e
                print(e)

        raise RuntimeError(
            f"generate_split failed after {max_tries} attempts "
            f"(base_seed={base_seed}, step={step}, key='{seed_key}')."
        ) from last_err

    finally:
        cfg[seed_key] = original_seed
        if seed_everything_fn is not None:
            seed_everything_fn(int(original_seed) if original_seed is not None else 0)
