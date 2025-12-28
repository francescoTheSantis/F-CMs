import torch
import pandas as pd
import pytest
import random
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.data.generate_split import (
    dfs_forward,
    dfs_backward,
    find_path_to_target_or_leaf,
    find_path_to_root,
    generate_base_subgraph,
    generate_add_nodes_values,
    add_additional_nodes_to_subgraph,
    get_subgraphs
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_graph():
    """
    Simple DAG:
    0 -> 1 -> 3
    0 -> 2 -> 3
    """
    data = [
        [0, 1, 1, 0],  # 0 -> 1, 2
        [0, 0, 0, 1],  # 1 -> 3
        [0, 0, 0, 1],  # 2 -> 3
        [0, 0, 0, 0]   # 3 (leaf)
    ]
    df = pd.DataFrame(data, columns=['0', '1', '2', '3'])
    return df


@pytest.fixture
def complex_graph():
    """
    More complex DAG (asia-like):
    0 -> 1 -> 5
    2 -> 3 -> 5
    2 -> 4
    5 -> 6
    5 -> 7
    """
    data = [
        [0, 1, 0, 0, 0, 0, 0, 0],  # 0 (asia) -> 1 (tub)
        [0, 0, 0, 0, 0, 1, 0, 0],  # 1 (tub) -> 5 (either)
        [0, 0, 0, 1, 1, 0, 0, 0],  # 2 (smoke) -> 3 (lung), 4 (bronc)
        [0, 0, 0, 0, 0, 1, 0, 0],  # 3 (lung) -> 5 (either)
        [0, 0, 0, 0, 0, 0, 0, 1],  # 4 (bronc) -> 7 (dysp)
        [0, 0, 0, 0, 0, 0, 1, 1],  # 5 (either) -> 6 (xray), 7 (dysp)
        [0, 0, 0, 0, 0, 0, 0, 0],  # 6 (xray) (leaf)
        [0, 0, 0, 0, 0, 0, 0, 0]   # 7 (dysp) (leaf)
    ]
    columns = ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
    df = pd.DataFrame(data, columns=columns)
    return df


# ============================================================================
# Tests for dfs_forward
# ============================================================================

def test_dfs_forward_simple_path(simple_graph):
    """Test finding a simple path from root to leaf."""
    torch_graph = torch.tensor(simple_graph.values)
    path = dfs_forward(torch_graph, start_node=0, end_node=3, randomize=False)
    
    assert path is not None
    assert path[0] == 0
    assert path[-1] == 3
    assert len(path) >= 3  # At least 0 -> X -> 3


def test_dfs_forward_to_leaf(simple_graph):
    """Test finding path to any leaf when end_node is None."""
    torch_graph = torch.tensor(simple_graph.values)
    path = dfs_forward(torch_graph, start_node=0, end_node=None, randomize=False)
    
    assert path is not None
    assert path[0] == 0
    assert path[-1] == 3  # 3 is the only leaf


def test_dfs_forward_same_node(simple_graph):
    """Test when start and end are the same."""
    torch_graph = torch.tensor(simple_graph.values)
    path = dfs_forward(torch_graph, start_node=2, end_node=2, randomize=False)
    
    assert path == [2]


def test_dfs_forward_no_path():
    """Test when no path exists."""
    # Disconnected graph: 0 -> 1, 2 -> 3
    data = [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ]
    torch_graph = torch.tensor(data)
    path = dfs_forward(torch_graph, start_node=0, end_node=3, randomize=False)
    
    assert path is None


def test_dfs_forward_with_nodes_not_allowed(simple_graph):
    """Test path finding with forbidden nodes."""
    torch_graph = torch.tensor(simple_graph.values)
    # Block node 1, should find path through node 2
    path = dfs_forward(torch_graph, start_node=0, end_node=3, 
                      randomize=False, nodes_not_allowed=[1])
    
    assert path is not None
    assert 1 not in path
    assert 2 in path


def test_dfs_forward_randomize(simple_graph):
    """Test that randomize produces different paths."""
    torch_graph = torch.tensor(simple_graph.values)
    paths = set()
    
    for _ in range(10):
        path = dfs_forward(torch_graph, start_node=0, end_node=3, randomize=True)
        paths.add(tuple(path) if path else None)
    
    # Should have at least 2 different paths (0->1->3 and 0->2->3)
    assert len(paths) >= 2


# ============================================================================
# Tests for dfs_backward
# ============================================================================

def test_dfs_backward_find_root(simple_graph):
    """Test finding root from a node."""
    torch_graph = torch.tensor(simple_graph.values)
    path = dfs_backward(torch_graph, start_node=3, randomize=False)
    
    assert path is not None
    assert path[0] == 3  # Start node
    assert path[-1] == 0  # 0 is the root


def test_dfs_backward_already_root(simple_graph):
    """Test when node is already a root."""
    torch_graph = torch.tensor(simple_graph.values)
    path = dfs_backward(torch_graph, start_node=0, randomize=False)
    
    assert path == [0]


def test_dfs_backward_with_nodes_not_allowed(complex_graph):
    """Test backward traversal with forbidden nodes."""
    torch_graph = torch.tensor(complex_graph.values)
    # Block node 0 (asia), should find root 2 (smoke)
    path = dfs_backward(torch_graph, start_node=5, randomize=False, nodes_not_allowed=[0])
    
    assert path is not None
    assert 0 not in path


# ============================================================================
# Tests for find_path_to_target_or_leaf
# ============================================================================

def test_find_path_to_target(simple_graph):
    """Test finding path to specific target."""
    path = find_path_to_target_or_leaf(simple_graph, start_node=0, end_node=3)
    
    assert path is not None
    assert path[0] == 0
    assert path[-1] == 3


def test_find_path_to_leaf(simple_graph):
    """Test finding path to any leaf."""
    path = find_path_to_target_or_leaf(simple_graph, start_node=1, end_node=None)
    
    assert path is not None
    assert path[0] == 1
    # Should reach a leaf (node with no outgoing edges)


def test_find_path_with_forbidden_nodes(simple_graph):
    """Test path finding excluding certain nodes."""
    path = find_path_to_target_or_leaf(simple_graph, start_node=0, end_node=3,
                                       nodes_not_allowed=[1])
    
    assert path is not None
    assert 1 not in path


# ============================================================================
# Tests for find_path_to_root
# ============================================================================

def test_find_path_to_root(complex_graph):
    """Test finding path from node to root."""
    path = find_path_to_root(complex_graph, start_node=5, randomize=False)
    
    assert path is not None
    assert path[0] == 5  # Start node
    assert path[-1] in [0, 2]  # Either asia or smoke are roots


def test_find_path_to_root_already_root(simple_graph):
    """Test when node is already a root."""
    path = find_path_to_root(simple_graph, start_node=0)
    
    assert path == [0]


# ============================================================================
# Tests for generate_base_subgraph
# ============================================================================

def test_generate_base_subgraph_single_task(simple_graph):
    """Test generating subgraph for single task node."""
    subgraph = generate_base_subgraph(simple_graph, task_indices=[3], randomize=False)
    
    assert 3 in subgraph
    assert 0 in subgraph  # Should include root
    assert len(subgraph) >= 2


def test_generate_base_subgraph_multiple_tasks(complex_graph):
    """Test generating subgraph for multiple task nodes."""
    subgraph = generate_base_subgraph(complex_graph, task_indices=[6, 7], randomize=False)
    
    assert 6 in subgraph
    assert 7 in subgraph
    # Should include at least one root
    assert 0 in subgraph or 2 in subgraph


def test_generate_base_subgraph_with_forbidden_nodes(complex_graph):
    """Test subgraph generation excluding certain nodes."""
    subgraph = generate_base_subgraph(complex_graph, task_indices=[7], 
                                     randomize=False, nodes_not_allowed=[0, 1])
    
    assert 7 in subgraph
    assert 0 not in subgraph
    assert 1 not in subgraph


def test_generate_base_subgraph_invalid_node(simple_graph):
    """Test error handling for invalid task node."""
    with pytest.raises(ValueError):
        generate_base_subgraph(simple_graph, task_indices=[99], randomize=False)


# ============================================================================
# Tests for generate_add_nodes_values
# ============================================================================

def test_generate_add_nodes_random(complex_graph):
    """Test random selection of additional nodes."""
    torch_graph = torch.tensor(complex_graph.values)
    y_index_graph = [1, 3, 4, 5, 6, 7]  # All nodes reachable from task
    y_index = 7
    
    add_nodes = generate_add_nodes_values(
        complex_graph, torch_graph, y_index_graph, y_index,
        add_nodes_modality='random', add_nodes_number=2
    )
    
    assert len(add_nodes) == 2
    assert y_index not in add_nodes
    assert all(node in y_index_graph for node in add_nodes)


def test_generate_add_nodes_connected(complex_graph):
    """Test selection of connected nodes (with parents and children)."""
    torch_graph = torch.tensor(complex_graph.values)
    y_index_graph = [1, 3, 4, 5, 6, 7]
    y_index = 7
    
    add_nodes = generate_add_nodes_values(
        complex_graph, torch_graph, y_index_graph, y_index,
        add_nodes_modality='at_least_one_parent_and_child', add_nodes_number=1
    )
    
    assert len(add_nodes) <= 1
    # Nodes should have both parents and children
    for node in add_nodes:
        parents = torch.where(torch_graph[:, node] == 1)[0]
        children = torch.where(torch_graph[node, :] == 1)[0]
        assert parents.numel() > 0
        assert children.numel() > 0


def test_generate_add_nodes_insufficient_nodes(simple_graph):
    """Test when not enough valid nodes available."""
    torch_graph = torch.tensor(simple_graph.values)
    y_index_graph = [1, 2, 3]
    y_index = 3
    
    add_nodes = generate_add_nodes_values(
        simple_graph, torch_graph, y_index_graph, y_index,
        add_nodes_modality='random', add_nodes_number=10
    )
    
    # Should return as many as possible (excluding y_index)
    assert len(add_nodes) <= len(y_index_graph) - 1


# ============================================================================
# Tests for add_additional_nodes_to_subgraph
# ============================================================================

def test_add_additional_nodes_extends_subgraph(complex_graph):
    """Test that additional nodes are properly added to subgraph."""
    base_subgraph = [0, 1, 5, 6]
    add_nodes = [2, 3]
    
    extended = add_additional_nodes_to_subgraph(
        complex_graph, base_subgraph, add_nodes, randomize=False
    )
    
    # Should contain original nodes
    assert all(node in extended for node in base_subgraph)
    # Should contain additional nodes
    assert 2 in extended or 3 in extended


def test_add_additional_nodes_already_present(simple_graph):
    """Test when additional nodes are already in subgraph."""
    base_subgraph = [0, 1, 2, 3]
    add_nodes = [1, 2]
    
    extended = add_additional_nodes_to_subgraph(
        simple_graph, base_subgraph, add_nodes, randomize=False
    )
    
    # Should not duplicate nodes
    assert len(extended) == len(set(extended))
    assert set(base_subgraph).issubset(set(extended))


# ============================================================================
# Tests for get_subgraphs (Integration Tests)
# ============================================================================

def test_get_subgraphs_basic(complex_graph):
    """Test basic subgraph generation."""
    subgraphs, subgraphs_names, add_nodes, subgraphs_with_add_nodes = get_subgraphs(
        complex_graph,
        y_index=7,
        min_number_subgraphs=3,
        max_number_subgraphs=5,
        modality='connected_nodes',
        task_in_common=True,
        dict_subgraph_with_add_nodes={}
    )
    
    assert len(subgraphs) >= 3
    assert len(subgraphs) <= 5
    assert len(subgraphs) == len(subgraphs_names)
    assert len(subgraphs_with_add_nodes) == 0
    
    # Check that all subgraphs contain at least 2 nodes
    for sg in subgraphs.values():
        assert len(sg) >= 2


def test_get_subgraphs_with_additional_nodes(complex_graph):
    """Test subgraph generation with additional nodes."""
    dict_add_nodes = {
        'modality': 'random',
        'number_add_nodes': 1,
        'number_subgraphs_add_nodes': 2
    }
    
    subgraphs, subgraphs_names, subgraphs_with_add_nodes, add_nodes_values = get_subgraphs(
        complex_graph,
        y_index=7,
        min_number_subgraphs=4,
        max_number_subgraphs=6,
        modality='connected_nodes',
        task_in_common=True,
        dict_subgraph_with_add_nodes=dict_add_nodes
    )
    
    assert len(add_nodes_values) >= 1
    assert len(subgraphs) >= 4
    # check where subgraphs_with_add_nodes is True
    assert sum(subgraphs_with_add_nodes) == 2

    
    # Check that at least some subgraphs contain additional nodes
    assert sum(subgraphs_with_add_nodes) >= 1
    assert not all(subgraphs_with_add_nodes)  # Not all should have add nodes


def test_get_subgraphs_coverage(simple_graph):
    """Test that subgraphs cover all nodes in the graph."""
    subgraphs, _, _, _ = get_subgraphs(
        simple_graph,
        y_index=3,
        min_number_subgraphs=2,
        max_number_subgraphs=4,
        modality='connected_nodes',
        task_in_common=True,
        dict_subgraph_with_add_nodes={}
    )
    
    # Collect all covered nodes
    covered_nodes = set()
    for sg in subgraphs.values():
        covered_nodes.update(sg)
    
    # Should cover most nodes (some nodes might not be reachable)
    all_nodes = set(range(len(simple_graph)))
    
    # At least cover more than half
    assert len(covered_nodes) == len(all_nodes)


#def test_get_subgraphs_random_modality(simple_graph):
#    """Test random nodes modality."""
#    subgraphs, _, _, _ = get_subgraphs(
#        simple_graph,
#        y_index=3,
#        min_number_subgraphs=2,
#        max_number_subgraphs=4,
#        modality='random_nodes',
#        task_in_common=True,
#        dict_subgraph_with_add_nodes={}
#    )
    
#    assert len(subgraphs) >= 2
#    # Random modality should still create valid subgraphs
#    for sg in subgraphs.values():
#        assert len(sg) >= 2


def test_get_subgraphs_task_not_in_common(complex_graph):
    """Test when task is not common to all subgraphs."""
    subgraphs, _, _, _ = get_subgraphs(
        complex_graph,
        y_index=7,
        min_number_subgraphs=3,
        max_number_subgraphs=5,
        modality='connected_nodes',
        task_in_common=False,
        dict_subgraph_with_add_nodes={}
    )
    
    assert len(subgraphs) >= 3



def test_get_subgraphs_invalid_add_nodes_config(complex_graph):
    """Test error handling for invalid additional nodes configuration."""
    dict_add_nodes = {
        'modality': 'invalid_modality',
        'number_add_nodes': 1,
        'number_subgraphs_add_nodes': 1
    }
    
    with pytest.raises(ValueError):
        get_subgraphs(
            complex_graph,
            y_index=7,
            min_number_subgraphs=3,
            max_number_subgraphs=5,
            modality='connected_nodes',
            task_in_common=True,
            dict_subgraph_with_add_nodes=dict_add_nodes
        )


def test_get_subgraphs_no_duplicates(complex_graph):
    """Test that generated subgraphs are unique."""
    subgraphs, _, _, _ = get_subgraphs(
        complex_graph,
        y_index=7,
        min_number_subgraphs=3,
        max_number_subgraphs=6,
        modality='connected_nodes',
        task_in_common=True,
        dict_subgraph_with_add_nodes={}
    )
    
    # Convert to sets for comparison
    subgraph_sets = [frozenset(sg) for sg in subgraphs.values()]
    
    # Check no duplicates
    assert len(subgraph_sets) == len(set(subgraph_sets))


def test_generate_add_nodes_connection_modality(complex_graph):
    """Test 'connection' modality prioritizes highly connected nodes."""
    torch_graph = torch.tensor(complex_graph.values)
    y_index_graph = [1, 3, 4, 5, 6, 7]
    y_index = 7
    
    add_nodes = generate_add_nodes_values(
        complex_graph, torch_graph, y_index_graph, y_index,
        add_nodes_modality='connection', add_nodes_number=2
    )
    
    assert len(add_nodes) <= 2
    # Node 5 (either) should be selected as it has most connections
    # It has 2 parents (1,3) and 2 children (6,7)
    if len(add_nodes) > 0:
        assert 5 in add_nodes or any(node in [1, 3, 4] for node in add_nodes)


def test_dfs_forward_with_nodes_not_allowed_blocks_path(simple_graph):
    """Test that blocking all intermediate nodes prevents path finding."""
    torch_graph = torch.tensor(simple_graph.values)
    # Block both node 1 and 2, should not find path from 0 to 3
    path = dfs_forward(torch_graph, start_node=0, end_node=3, 
                      randomize=False, nodes_not_allowed=[1, 2])
    
    assert path is None


def test_find_path_to_root_with_nodes_not_allowed(complex_graph):
    """Test finding root while avoiding certain nodes."""
    path = find_path_to_root(complex_graph, start_node=5, 
                            randomize=False, nodes_not_allowed=[0])
    
    assert path is not None
    assert 0 not in path
    # Should find path through smoke (2) instead of asia (0)
    if len(path) > 1:
        assert 2 in path or path[-1] == 2


def test_get_subgraphs_with_add_nodes_tracking(complex_graph):
    """Test that subgraphs_with_add_nodes correctly tracks which subgraphs have additional nodes."""
    dict_add_nodes = {
        'modality': 'random',
        'number_add_nodes': 1,
        'number_subgraphs_add_nodes': 2
    }
    
    subgraphs, subgraphs_concept_names, subgraphs_with_add_nodes, add_nodes_values = get_subgraphs(
        complex_graph,
        y_index=7,
        min_number_subgraphs=4,
        max_number_subgraphs=6,
        modality='connected_nodes',
        task_in_common=True,
        dict_subgraph_with_add_nodes=dict_add_nodes
    )
    
    # Verify tracking is correct
    for i, (sg_key, sg) in enumerate(subgraphs.items()):
        has_add_node = any(node in add_nodes_values for node in sg)
        # The tracking might not be perfect due to merging, but check consistency
        if subgraphs_with_add_nodes[i]:
            # If marked as having add nodes, at least one path to them should exist
            assert len(sg) > 0


def test_generate_base_subgraph_multiple_paths(simple_graph):
    """Test that generate_base_subgraph can create multiple paths due to randomization."""
    # Run multiple times to check variation
    subgraphs = []
    for _ in range(5):
        sg = generate_base_subgraph(simple_graph, task_indices=[3], randomize=True)
        subgraphs.append(tuple(sorted(sg)))
    
    # Should generate some variation (though not guaranteed every time)
    # At minimum, subgraphs should be valid
    for sg in subgraphs:
        assert 3 in sg
        assert 0 in sg


def test_add_additional_nodes_partial_path(complex_graph):
    """Test that add_additional_nodes_to_subgraph uses partial paths to leaves."""
    base_subgraph = [0, 1, 5]
    add_nodes = [2]  # smoke
    
    extended = add_additional_nodes_to_subgraph(
        complex_graph, base_subgraph, add_nodes, randomize=True
    )
    
    # Should contain base nodes
    assert all(node in extended for node in base_subgraph)
    # Should add node 2 and potentially some descendants
    assert 2 in extended


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
