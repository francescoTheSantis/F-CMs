"""Edge-level local differential privacy for client concept graphs.

The client graph is represented as a directed binary adjacency matrix.  For
each unordered pair of nodes there are exactly three supported relation
states: ``i -> j``, ``j -> i``, and no edge.  Applying k-ary randomized
response independently to every pair provides epsilon edge-LDP for the
fixed-node-set adjacency relation in which two graphs differ in one pair.

This module intentionally does not try to hide the client node/concept set.
Doing so requires a separate protocol because the set of reported pairs
itself reveals which concepts are present at a client.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


FORWARD = 0
BACKWARD = 1
NO_EDGE = 2
RELATION_NAMES = ("forward", "backward", "no_edge")


def parse_structural_epsilon(value: Any) -> float:
    """Parse and validate a structural privacy epsilon value."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"inf", "+inf", "infinity", "+infinity", ".inf", "+.inf"}:
            return math.inf
    epsilon = float(value)
    if math.isnan(epsilon) or epsilon < 0:
        raise ValueError("Structural privacy epsilon must be non-negative.")
    return epsilon


def randomized_response_probabilities(epsilon: Any) -> Tuple[float, float]:
    """Return truth probability ``p`` and per-alternative probability ``q``.

    The numerically stable form avoids overflow for large finite epsilon:

    ``p = 1 / (1 + 2 exp(-epsilon))`` and
    ``q = exp(-epsilon) / (1 + 2 exp(-epsilon))``.
    """
    epsilon = parse_structural_epsilon(epsilon)
    if math.isinf(epsilon):
        return 1.0, 0.0
    exp_negative_epsilon = math.exp(-epsilon)
    denominator = 1.0 + 2.0 * exp_negative_epsilon
    return 1.0 / denominator, exp_negative_epsilon / denominator


def _validate_adjacency_graph(graph: pd.DataFrame) -> np.ndarray:
    if not isinstance(graph, pd.DataFrame):
        raise TypeError("A local graph must be a pandas DataFrame.")
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("A local graph adjacency matrix must be square.")
    if list(graph.index) != list(graph.columns):
        raise ValueError("A local graph must use the same ordered labels for rows and columns.")

    adjacency = graph.to_numpy(copy=True)
    if not np.isin(adjacency, [0, 1]).all():
        raise ValueError("Three-way randomized response requires a binary adjacency matrix.")
    if np.any(np.diag(adjacency) != 0):
        raise ValueError("Self-loops are not supported by the structural privacy mechanism.")

    for i in range(adjacency.shape[0]):
        for j in range(i + 1, adjacency.shape[0]):
            if adjacency[i, j] == 1 and adjacency[j, i] == 1:
                raise ValueError(
                    f"Pair ({graph.index[i]!r}, {graph.index[j]!r}) has edges in both "
                    "directions; expected one of forward, backward, or no edge."
                )
    return adjacency.astype(np.int8, copy=False)


def _relation_state(adjacency: np.ndarray, i: int, j: int) -> int:
    if adjacency[i, j] == 1:
        return FORWARD
    if adjacency[j, i] == 1:
        return BACKWARD
    return NO_EDGE


def _write_relation_state(adjacency: np.ndarray, i: int, j: int, state: int) -> None:
    adjacency[i, j] = 0
    adjacency[j, i] = 0
    if state == FORWARD:
        adjacency[i, j] = 1
    elif state == BACKWARD:
        adjacency[j, i] = 1
    elif state != NO_EDGE:
        raise ValueError(f"Unknown relation state: {state}")


def privatize_graph_edge_states(
    graph: pd.DataFrame,
    epsilon: Any,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Privatize all unordered pair states in one fixed-node-set local graph.

    This is the client-side mechanism.  No cycle removal or other
    data-dependent repair is performed before transmission.  Any repair at
    the server must use only the privatized reports, which is DP-safe
    post-processing.
    """
    epsilon = parse_structural_epsilon(epsilon)
    p, q = randomized_response_probabilities(epsilon)
    adjacency = _validate_adjacency_graph(graph)
    privatized = np.zeros_like(adjacency, dtype=np.int8)
    generator = rng if rng is not None else np.random.default_rng()

    changed_pairs = 0
    reported_counts = np.zeros(3, dtype=np.int64)
    n_nodes = adjacency.shape[0]

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            true_state = _relation_state(adjacency, i, j)
            if math.isinf(epsilon):
                reported_state = true_state
            else:
                probabilities = np.full(3, q, dtype=np.float64)
                probabilities[true_state] = p
                # Normalize defensively against the last bits of floating-point
                # roundoff rejected by Generator.choice on some NumPy versions.
                probabilities /= probabilities.sum()
                reported_state = int(generator.choice(3, p=probabilities))

            _write_relation_state(privatized, i, j, reported_state)
            changed_pairs += int(reported_state != true_state)
            reported_counts[reported_state] += 1

    n_pairs = n_nodes * (n_nodes - 1) // 2
    stats = {
        "pair_reports": int(n_pairs),
        "changed_pair_reports": int(changed_pairs),
        "reported_relation_counts": {
            name: int(reported_counts[index]) for index, name in enumerate(RELATION_NAMES)
        },
    }
    privatized_graph = pd.DataFrame(
        privatized,
        index=graph.index.copy(),
        columns=graph.columns.copy(),
        dtype=int,
    )
    return privatized_graph, stats


def privatize_local_graphs(
    local_graphs: Sequence[pd.DataFrame],
    epsilon: Any,
    seed: Optional[int] = None,
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """Privatize a sequence of local graphs with independent client RNGs."""
    epsilon = parse_structural_epsilon(epsilon)
    p, q = randomized_response_probabilities(epsilon)
    seed_sequence = np.random.SeedSequence(seed)
    client_seed_sequences = seed_sequence.spawn(len(local_graphs))

    privatized_graphs: List[pd.DataFrame] = []
    total_pairs = 0
    total_changed = 0
    reported_counts = {name: 0 for name in RELATION_NAMES}

    for graph, client_seed in zip(local_graphs, client_seed_sequences):
        privatized_graph, stats = privatize_graph_edge_states(
            graph,
            epsilon=epsilon,
            rng=np.random.default_rng(client_seed),
        )
        privatized_graphs.append(privatized_graph)
        total_pairs += stats["pair_reports"]
        total_changed += stats["changed_pair_reports"]
        for relation, count in stats["reported_relation_counts"].items():
            reported_counts[relation] += count

    changed_rate = float(total_changed / total_pairs) if total_pairs else 0.0
    epsilon_json: Any = "inf" if math.isinf(epsilon) else float(epsilon)
    report = {
        "enabled": True,
        "mechanism": "three_way_randomized_response",
        "privacy_unit": "one_unordered_pair_relation",
        "epsilon": epsilon_json,
        "truth_probability_p": float(p),
        "alternative_probability_q": float(q),
        "expected_frequency_transform": {
            "intercept": float(q),
            "slope": float(p - q),
            "formula": "E[private_frequency_r] = q + (p - q) * frequency_r",
        },
        "expected_changed_pair_rate": float(2.0 * q),
        "number_of_client_graphs": int(len(local_graphs)),
        "pair_reports": int(total_pairs),
        "changed_pair_reports": int(total_changed),
        "changed_pair_report_rate": changed_rate,
        "reported_relation_counts": reported_counts,
    }
    return privatized_graphs, report


def graph_pair_disagreement(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare two graphs using one unit per unordered three-state pair.

    A reversal therefore counts as one disagreeing relation, matching the
    edge-neighbor definition used by the privacy mechanism.
    """
    node_order = list(first.index)
    node_order.extend(node for node in second.index if node not in first.index)
    first_aligned = first.reindex(index=node_order, columns=node_order, fill_value=0)
    second_aligned = second.reindex(index=node_order, columns=node_order, fill_value=0)
    first_adjacency = _validate_adjacency_graph(first_aligned)
    second_adjacency = _validate_adjacency_graph(second_aligned)

    differing_pairs = 0
    reversals = 0
    additions = 0
    deletions = 0
    n_nodes = len(node_order)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            first_state = _relation_state(first_adjacency, i, j)
            second_state = _relation_state(second_adjacency, i, j)
            if first_state == second_state:
                continue
            differing_pairs += 1
            if first_state == NO_EDGE:
                additions += 1
            elif second_state == NO_EDGE:
                deletions += 1
            else:
                reversals += 1

    total_pairs = n_nodes * (n_nodes - 1) // 2
    return {
        "differing_pairs": int(differing_pairs),
        "total_pairs": int(total_pairs),
        "disagreement_rate": float(differing_pairs / total_pairs) if total_pairs else 0.0,
        "edge_additions": int(additions),
        "edge_deletions": int(deletions),
        "edge_reversals": int(reversals),
    }
