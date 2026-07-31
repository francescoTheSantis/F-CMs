"""Parameter migration for structural concept-model updates.

The training code historically copied a tensor only when its state-dict key and
complete shape matched.  This module keeps that behaviour as ``full_reinit``
and provides two semantic, block-wise alternatives:

``partial_random``
    Copy blocks with the same meaning and retain the *already initialized*
    destination values for new blocks.

``partial_zero``
    Copy blocks with the same meaning and zero genuinely new, safely located
    blocks.  Unsafe blocks retain their destination initialization.

No function in this file samples random numbers.  Consequently, callers can
instantiate the destination model once and compare the three strategies
without changing the random-number stream.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import re
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


MIGRATION_STRATEGIES = ("full_reinit", "partial_random", "partial_zero")

_CGM_INPUT_WEIGHT_RE = re.compile(
    r"^concept_encoders\.([^.]+)\.c_encoder\.mlp\.0\.affinity\.weight$"
)
_PROPAGATOR_RE = re.compile(r"^propagators\.([^.]+)\.([^.]+)\.(.+)$")


def _as_tensor_like(value: Any, reference: torch.Tensor) -> torch.Tensor:
    """Clone ``value`` on the dtype/device of ``reference``."""

    if torch.is_tensor(value):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(device=reference.device, dtype=reference.dtype).clone()


def _json_identity(value: Hashable) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_json_identity(item) for item in value]
    return str(value)


def _axis_offsets(
    identities: Sequence[Hashable],
    block_sizes: Optional[Mapping[Hashable, int]],
) -> Tuple[Dict[Hashable, Tuple[int, int]], int]:
    if len(set(identities)) != len(identities):
        raise ValueError("Block identities must be unique within an axis.")

    offsets: Dict[Hashable, Tuple[int, int]] = {}
    start = 0
    for identity in identities:
        size = 1 if block_sizes is None else int(block_sizes[identity])
        if size < 0:
            raise ValueError(f"Negative block size for identity {identity!r}.")
        offsets[identity] = (start, start + size)
        start += size
    return offsets, start


def migrate_identity_axis_blocks(
    old_tensor: torch.Tensor,
    new_tensor: torch.Tensor,
    *,
    old_identities: Sequence[Hashable],
    new_identities: Sequence[Hashable],
    axis: int,
    strategy: str,
    old_block_sizes: Optional[Mapping[Hashable, int]] = None,
    new_block_sizes: Optional[Mapping[Hashable, int]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Migrate named contiguous blocks along one tensor axis.

    Blocks shared by identity are copied only when their sizes agree.  Blocks
    found only in the destination retain its initializer for
    ``partial_random`` and are zeroed for ``partial_zero``.  A shared identity
    with a changed block size is deliberately *not* copied or zeroed because
    its within-block semantics are unknown.

    The returned element categories are mutually exclusive and cover every
    destination element:

    - ``preserved_elements``: copied from a semantic predecessor;
    - ``newly_initialized_elements``: destination blocks with no predecessor;
    - ``fully_reinitialized_elements``: a predecessor existed but was unsafe
      to map.
    """

    if strategy not in ("partial_random", "partial_zero"):
        raise ValueError(
            "migrate_identity_axis_blocks supports partial_random and "
            "partial_zero only."
        )
    if old_tensor.ndim != new_tensor.ndim:
        return new_tensor.detach().clone(), {
            "mapping_safe": False,
            "reason": "rank_mismatch",
            "preserved_elements": 0,
            "newly_initialized_elements": 0,
            "fully_reinitialized_elements": int(new_tensor.numel()),
            "zeroed_elements": 0,
            "copied_blocks": [],
            "new_blocks": [],
            "removed_blocks": [_json_identity(item) for item in old_identities],
            "unsafe_blocks": [],
        }

    normalized_axis = axis if axis >= 0 else new_tensor.ndim + axis
    if normalized_axis < 0 or normalized_axis >= new_tensor.ndim:
        raise ValueError(f"Axis {axis} is invalid for rank-{new_tensor.ndim} tensor.")

    old_other_shape = tuple(
        size for index, size in enumerate(old_tensor.shape) if index != normalized_axis
    )
    new_other_shape = tuple(
        size for index, size in enumerate(new_tensor.shape) if index != normalized_axis
    )
    if old_other_shape != new_other_shape:
        return new_tensor.detach().clone(), {
            "mapping_safe": False,
            "reason": "non_migrated_axes_shape_mismatch",
            "preserved_elements": 0,
            "newly_initialized_elements": 0,
            "fully_reinitialized_elements": int(new_tensor.numel()),
            "zeroed_elements": 0,
            "copied_blocks": [],
            "new_blocks": [],
            "removed_blocks": [_json_identity(item) for item in old_identities],
            "unsafe_blocks": [],
        }

    old_offsets, old_extent = _axis_offsets(old_identities, old_block_sizes)
    new_offsets, new_extent = _axis_offsets(new_identities, new_block_sizes)
    if old_extent != old_tensor.shape[normalized_axis] or new_extent != new_tensor.shape[normalized_axis]:
        return new_tensor.detach().clone(), {
            "mapping_safe": False,
            "reason": "block_layout_does_not_cover_axis",
            "preserved_elements": 0,
            "newly_initialized_elements": 0,
            "fully_reinitialized_elements": int(new_tensor.numel()),
            "zeroed_elements": 0,
            "copied_blocks": [],
            "new_blocks": [],
            "removed_blocks": [_json_identity(item) for item in old_identities],
            "unsafe_blocks": [],
        }

    result = new_tensor.detach().clone()
    old_cast = _as_tensor_like(old_tensor, new_tensor)
    elements_per_axis_entry = int(new_tensor.numel() // max(1, new_extent))
    preserved = 0
    newly_initialized = 0
    fully_reinitialized = 0
    zeroed = 0
    copied_blocks: List[Any] = []
    new_blocks: List[Any] = []
    unsafe_blocks: List[Any] = []

    for identity in new_identities:
        new_start, new_stop = new_offsets[identity]
        new_size = new_stop - new_start
        destination_slice = [slice(None)] * new_tensor.ndim
        destination_slice[normalized_axis] = slice(new_start, new_stop)
        block_elements = new_size * elements_per_axis_entry

        if identity not in old_offsets:
            new_blocks.append(_json_identity(identity))
            newly_initialized += block_elements
            if strategy == "partial_zero":
                result[tuple(destination_slice)] = 0
                zeroed += block_elements
            continue

        old_start, old_stop = old_offsets[identity]
        old_size = old_stop - old_start
        if old_size != new_size:
            unsafe_blocks.append(_json_identity(identity))
            fully_reinitialized += block_elements
            continue

        source_slice = [slice(None)] * old_tensor.ndim
        source_slice[normalized_axis] = slice(old_start, old_stop)
        result[tuple(destination_slice)] = old_cast[tuple(source_slice)]
        copied_blocks.append(_json_identity(identity))
        preserved += block_elements

    removed_blocks = [
        _json_identity(identity) for identity in old_identities if identity not in new_offsets
    ]
    return result, {
        "mapping_safe": True,
        "reason": None,
        "preserved_elements": int(preserved),
        "newly_initialized_elements": int(newly_initialized),
        "fully_reinitialized_elements": int(fully_reinitialized),
        "zeroed_elements": int(zeroed),
        "copied_blocks": copied_blocks,
        "new_blocks": new_blocks,
        "removed_blocks": removed_blocks,
        "unsafe_blocks": unsafe_blocks,
    }


@dataclass(frozen=True)
class _GraphMetadata:
    names: Tuple[str, ...]
    cardinalities: Dict[str, int]
    parents: Dict[str, Tuple[str, ...]]
    roots: Tuple[str, ...]
    concept_names: Tuple[str, ...]
    task_names: Tuple[str, ...]
    concept_hidden_size: Optional[int]
    prop_type: Optional[str]


def _mapping_get(mapping: Any, key: str, default: Any) -> Any:
    if mapping is None:
        return default
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    try:
        return mapping[key]
    except (KeyError, TypeError):
        return getattr(mapping, key, default)


def _extract_graph_metadata(model: Any) -> Optional[_GraphMetadata]:
    combo_info = getattr(model, "combo_info", None)
    names = list(_mapping_get(combo_info, "names", []))
    cardinality_values = list(_mapping_get(combo_info, "cardinality", []))
    graph = getattr(model, "graph", None)
    if not names or len(cardinality_values) != len(names) or graph is None:
        return None

    graph_tensor = torch.as_tensor(graph)
    if graph_tensor.ndim != 2 or tuple(graph_tensor.shape) != (len(names), len(names)):
        return None

    parents: Dict[str, Tuple[str, ...]] = {}
    for child_index, child_name in enumerate(names):
        parent_indices = torch.nonzero(graph_tensor[:, child_index], as_tuple=False).flatten()
        parents[child_name] = tuple(names[int(index)] for index in parent_indices.tolist())

    roots_info = getattr(model, "roots_info", None)
    roots = tuple(_mapping_get(roots_info, "names", []))
    c_info = getattr(model, "c_info", None)
    y_info = getattr(model, "y_info", None)
    concept_names = tuple(_mapping_get(c_info, "names", []))
    task_names = tuple(_mapping_get(y_info, "names", []))
    if not concept_names:
        concept_names = tuple(name for name in names if name not in task_names)

    hidden_size = getattr(model, "concept_hidden_size", None)
    return _GraphMetadata(
        names=tuple(str(name) for name in names),
        cardinalities={str(name): int(value) for name, value in zip(names, cardinality_values)},
        parents={str(name): tuple(str(parent) for parent in value) for name, value in parents.items()},
        roots=tuple(str(name) for name in roots),
        concept_names=tuple(str(name) for name in concept_names),
        task_names=tuple(str(name) for name in task_names),
        concept_hidden_size=int(hidden_size) if hidden_size is not None else None,
        prop_type=str(getattr(model, "prop_type", "")).lower() or None,
    )


def _model_kind(model: Any) -> str:
    identity = f"{getattr(model, 'name', '')} {model.__class__.__name__}".lower()
    if "c2bm" in identity:
        return "c2bm"
    if "cgm" in identity:
        return "cgm"
    return "generic"


def _parameter_names(model: Any, state_keys: Iterable[str]) -> set[str]:
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        try:
            return {str(name) for name, _ in named_parameters()}
        except TypeError:
            return {str(name) for name, _ in named_parameters(recurse=True)}
    # This fallback keeps the utility usable with lightweight metadata objects.
    return set(state_keys)


def _propagator_index(state: Mapping[str, Any]) -> Dict[Tuple[str, str], str]:
    result: Dict[Tuple[str, str], str] = {}
    for key in state:
        match = _PROPAGATOR_RE.match(key)
        if match is None:
            continue
        _, node_name, suffix = match.groups()
        semantic_key = (node_name, suffix)
        # A valid C2BM has one propagator per node.  Preserve ambiguity as a
        # missing semantic mapping instead of silently selecting one.
        if semantic_key in result:
            result[semantic_key] = ""
        else:
            result[semantic_key] = key
    return {key: value for key, value in result.items() if value}


def _record(
    *,
    new_key: str,
    old_key: Optional[str],
    tensor: torch.Tensor,
    action: str,
    reason: Optional[str],
    counts: Mapping[str, int],
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "new_key": new_key,
        "old_key": old_key,
        "shape": [int(value) for value in tensor.shape],
        "elements": int(tensor.numel()),
        "action": action,
        "reason": reason,
        "preserved_elements": int(counts.get("preserved_elements", 0)),
        "newly_initialized_elements": int(counts.get("newly_initialized_elements", 0)),
        "fully_reinitialized_elements": int(counts.get("fully_reinitialized_elements", 0)),
        "zeroed_elements": int(counts.get("zeroed_elements", 0)),
        "details": dict(details or {}),
    }


def _whole_counts(tensor: torch.Tensor, category: str) -> Dict[str, int]:
    counts = {
        "preserved_elements": 0,
        "newly_initialized_elements": 0,
        "fully_reinitialized_elements": 0,
        "zeroed_elements": 0,
    }
    counts[category] = int(tensor.numel())
    return counts


def _structural_diff(
    old: Optional[_GraphMetadata], new: Optional[_GraphMetadata]
) -> Dict[str, Any]:
    if old is None or new is None:
        return {
            "metadata_available": False,
            "old_nodes": [],
            "new_nodes": [],
            "old_concepts": [],
            "new_concepts": [],
            "realized_pre_concept_coverage": None,
            "realized_post_concept_coverage": None,
            "added_nodes": [],
            "removed_nodes": [],
            "added_concepts": [],
            "removed_concepts": [],
            "parent_changes": [],
            "root_transitions": [],
            "affected_modules": [],
        }

    old_names = set(old.names)
    new_names = set(new.names)
    shared_names = old_names & new_names
    parent_changes: List[Dict[str, Any]] = []
    root_transitions: List[Dict[str, Any]] = []
    affected = set(new_names - old_names)
    old_roots = set(old.roots)
    new_roots = set(new.roots)

    for name in sorted(shared_names):
        old_parents = old.parents.get(name, ())
        new_parents = new.parents.get(name, ())
        if old_parents != new_parents:
            old_parent_set = set(old_parents)
            new_parent_set = set(new_parents)
            parent_changes.append(
                {
                    "node": name,
                    "old_parents": list(old_parents),
                    "new_parents": list(new_parents),
                    "added_parents": sorted(new_parent_set - old_parent_set),
                    "removed_parents": sorted(old_parent_set - new_parent_set),
                    "order_changed": (
                        old_parent_set == new_parent_set and old_parents != new_parents
                    ),
                }
            )
            affected.add(name)
        if (name in old_roots) != (name in new_roots):
            root_transitions.append(
                {
                    "node": name,
                    "old_is_root": name in old_roots,
                    "new_is_root": name in new_roots,
                }
            )
            affected.add(name)

    added_concepts = sorted(set(new.concept_names) - set(old.concept_names))
    removed_concepts = sorted(set(old.concept_names) - set(new.concept_names))
    new_concept_set = set(new.concept_names)
    coverage_denominator = len(new_concept_set)
    return {
        "metadata_available": True,
        "old_nodes": list(old.names),
        "new_nodes": list(new.names),
        "old_concepts": list(old.concept_names),
        "new_concepts": list(new.concept_names),
        "realized_pre_concept_coverage": (
            len(set(old.concept_names) & new_concept_set) / coverage_denominator
            if coverage_denominator
            else None
        ),
        "realized_post_concept_coverage": 1.0
        if coverage_denominator
        else None,
        "added_nodes": sorted(new_names - old_names),
        "removed_nodes": sorted(old_names - new_names),
        "added_concepts": added_concepts,
        "removed_concepts": removed_concepts,
        "parent_changes": parent_changes,
        "root_transitions": root_transitions,
        "affected_modules": sorted(affected),
    }


def migrate_architecture_state(
    old_state: Mapping[str, Any],
    old_model: Any,
    new_model: Any,
    *,
    strategy: str = "full_reinit",
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, Any]]:
    """Return a migrated destination state and JSON-serializable metadata.

    Parameters
    ----------
    old_state:
        Mapping from the pre-shift model's state-dict keys to tensors/arrays.
    old_model, new_model:
        Model objects providing graph metadata and, for ``new_model``, the
        freshly initialized destination ``state_dict``.
    strategy:
        One of :data:`MIGRATION_STRATEGIES`.

    Notes
    -----
    ``full_reinit`` is intentionally strict legacy behaviour: only an exact
    key and exact full shape are copied.  It never performs semantic remapping.
    """

    if strategy not in MIGRATION_STRATEGIES:
        raise ValueError(
            f"Unknown migration strategy {strategy!r}; expected one of "
            f"{MIGRATION_STRATEGIES}."
        )

    destination_state = OrderedDict(
        (str(key), value.detach().clone())
        for key, value in new_model.state_dict().items()
    )
    old_state_lookup = {str(key): value for key, value in old_state.items()}
    old_metadata = _extract_graph_metadata(old_model)
    new_metadata = _extract_graph_metadata(new_model)
    old_kind = _model_kind(old_model)
    new_kind = _model_kind(new_model)
    kind = new_kind if old_kind == new_kind else "generic"
    structural = _structural_diff(old_metadata, new_metadata)
    old_propagators = _propagator_index(old_state_lookup)
    parameter_names = _parameter_names(new_model, destination_state.keys())

    records: List[Dict[str, Any]] = []
    fallbacks: List[Dict[str, Any]] = []

    def add_fallback(tensor: Optional[str], module: Optional[str], reason: str) -> None:
        item = {"tensor": tensor, "module": module, "reason": reason}
        if item not in fallbacks:
            fallbacks.append(item)

    def legacy_for_key(new_key: str, destination: torch.Tensor, reason_prefix: str = "legacy"):
        if new_key in old_state_lookup:
            source = torch.as_tensor(old_state_lookup[new_key])
            if tuple(source.shape) == tuple(destination.shape):
                migrated = _as_tensor_like(source, destination)
                return migrated, _record(
                    new_key=new_key,
                    old_key=new_key,
                    tensor=destination,
                    action="copied_exact",
                    reason=f"{reason_prefix}_exact_key_and_shape",
                    counts=_whole_counts(destination, "preserved_elements"),
                )
            return destination, _record(
                new_key=new_key,
                old_key=new_key,
                tensor=destination,
                action="fully_reinitialized",
                reason=f"{reason_prefix}_shape_mismatch",
                counts=_whole_counts(destination, "fully_reinitialized_elements"),
                details={"old_shape": [int(value) for value in source.shape]},
            )
        return destination, _record(
            new_key=new_key,
            old_key=None,
            tensor=destination,
            action="newly_initialized",
            reason=f"{reason_prefix}_no_exact_key",
            counts=_whole_counts(destination, "newly_initialized_elements"),
        )

    for new_key, destination in destination_state.items():
        # Legacy behaviour must not inspect semantic identities.
        if strategy == "full_reinit":
            migrated, tensor_record = legacy_for_key(new_key, destination)
            # A missing exact C2BM level path may still have a semantic source;
            # classify it as discarded rather than genuinely new.
            prop_match = _PROPAGATOR_RE.match(new_key)
            if prop_match is not None and tensor_record["old_key"] is None:
                _, node_name, suffix = prop_match.groups()
                semantic_key = old_propagators.get((node_name, suffix))
                if semantic_key is not None:
                    tensor_record.update(
                        old_key=semantic_key,
                        action="fully_reinitialized",
                        reason="legacy_key_changed_with_topological_level",
                        preserved_elements=0,
                        newly_initialized_elements=0,
                        fully_reinitialized_elements=int(destination.numel()),
                    )
            destination_state[new_key] = migrated
            records.append(tensor_record)
            continue

        # CGM: semantically migrate first-input columns by parent name.
        cgm_match = _CGM_INPUT_WEIGHT_RE.match(new_key) if kind == "cgm" else None
        if cgm_match is not None:
            node_name = cgm_match.group(1)
            source_value = old_state_lookup.get(new_key)
            if source_value is None or old_metadata is None or new_metadata is None:
                migrated, tensor_record = legacy_for_key(
                    new_key, destination, reason_prefix="cgm_metadata_fallback"
                )
                if node_name in (set(old_metadata.names) if old_metadata else set()):
                    add_fallback(new_key, node_name, "missing_cgm_tensor_or_graph_metadata")
                destination_state[new_key] = migrated
                records.append(tensor_record)
                continue

            source = torch.as_tensor(source_value)
            old_root = node_name in set(old_metadata.roots)
            new_root = node_name in set(new_metadata.roots)
            if old_root != new_root:
                add_fallback(new_key, node_name, "root_status_transition")
                # No semantic mapping exists between an encoder representation
                # and a parent-embedding concatenation.  Preserve the exact
                # legacy fallback (which may copy a coincidentally equal shape).
                migrated, tensor_record = legacy_for_key(
                    new_key,
                    destination,
                    reason_prefix="root_status_transition_fallback",
                )
                destination_state[new_key] = migrated
                records.append(tensor_record)
                continue

            if node_name not in old_metadata.cardinalities or node_name not in new_metadata.cardinalities:
                add_fallback(new_key, node_name, "node_cardinality_metadata_missing")
                migrated, tensor_record = legacy_for_key(
                    new_key, destination, reason_prefix="cgm_layout_fallback"
                )
                destination_state[new_key] = migrated
                records.append(tensor_record)
                continue

            old_hidden = old_metadata.concept_hidden_size
            new_hidden = new_metadata.concept_hidden_size
            layout_safe = (
                source.ndim == 2
                and destination.ndim == 2
                and old_hidden is not None
                and old_hidden == new_hidden
                and old_metadata.cardinalities[node_name]
                == new_metadata.cardinalities[node_name]
            )
            if not layout_safe:
                add_fallback(new_key, node_name, "cgm_hidden_or_cardinality_layout_changed")
                migrated, tensor_record = legacy_for_key(
                    new_key,
                    destination,
                    reason_prefix="cgm_hidden_or_cardinality_fallback",
                )
                destination_state[new_key] = migrated
                records.append(tensor_record)
                continue

            if old_root and new_root:
                if tuple(source.shape) == tuple(destination.shape):
                    destination_state[new_key] = _as_tensor_like(source, destination)
                    records.append(
                        _record(
                            new_key=new_key,
                            old_key=new_key,
                            tensor=destination,
                            action="copied_exact",
                            reason="shared_root_encoder_input",
                            counts=_whole_counts(destination, "preserved_elements"),
                        )
                    )
                else:
                    add_fallback(new_key, node_name, "root_encoder_input_shape_changed")
                    records.append(
                        _record(
                            new_key=new_key,
                            old_key=new_key,
                            tensor=destination,
                            action="fully_reinitialized",
                            reason="root_encoder_input_shape_changed",
                            counts=_whole_counts(destination, "fully_reinitialized_elements"),
                        )
                    )
                continue

            old_parents = old_metadata.parents.get(node_name, ())
            new_parents = new_metadata.parents.get(node_name, ())
            migrated, block_report = migrate_identity_axis_blocks(
                source,
                destination,
                old_identities=old_parents,
                new_identities=new_parents,
                axis=1,
                strategy=strategy,
                old_block_sizes={parent: int(old_hidden) for parent in old_parents},
                new_block_sizes={parent: int(new_hidden) for parent in new_parents},
            )
            if not block_report["mapping_safe"]:
                add_fallback(new_key, node_name, str(block_report["reason"]))
            action = (
                "copied_exact"
                if old_parents == new_parents
                and block_report["preserved_elements"] == destination.numel()
                else "mapped_parent_columns"
            )
            destination_state[new_key] = migrated
            records.append(
                _record(
                    new_key=new_key,
                    old_key=new_key,
                    tensor=destination,
                    action=action,
                    reason="cgm_parent_identity_mapping",
                    counts=block_report,
                    details=block_report,
                )
            )
            continue

        # C2BM: locate propagators by node identity, independent of level path.
        prop_match = _PROPAGATOR_RE.match(new_key) if kind == "c2bm" else None
        if prop_match is not None:
            _, node_name, suffix = prop_match.groups()
            semantic_old_key = old_propagators.get((node_name, suffix))
            if semantic_old_key is None:
                destination_state[new_key] = destination
                records.append(
                    _record(
                        new_key=new_key,
                        old_key=None,
                        tensor=destination,
                        action="newly_initialized",
                        reason="new_c2bm_propagator_or_no_semantic_predecessor",
                        counts=_whole_counts(destination, "newly_initialized_elements"),
                    )
                )
                continue

            source = torch.as_tensor(old_state_lookup[semantic_old_key])
            old_prop_type = old_metadata.prop_type if old_metadata else None
            new_prop_type = new_metadata.prop_type if new_metadata else None
            if (
                old_metadata is None
                or new_metadata is None
                or old_prop_type != new_prop_type
                or new_prop_type not in ("equations", "embeddings")
            ):
                add_fallback(new_key, node_name, "unsupported_or_changed_c2bm_propagator_layout")
                migrated, tensor_record = legacy_for_key(
                    new_key, destination, reason_prefix="c2bm_layout_fallback"
                )
                # If the exact path changed, legacy leaves a semantic predecessor
                # behind and this is a full reinitialization, not a new tensor.
                if tensor_record["old_key"] is None:
                    tensor_record.update(
                        old_key=semantic_old_key,
                        action="fully_reinitialized",
                        reason="unsafe_c2bm_layout_and_level_path_changed",
                        newly_initialized_elements=0,
                        fully_reinitialized_elements=int(destination.numel()),
                    )
                destination_state[new_key] = migrated
                records.append(tensor_record)
                continue

            is_equation_readout = new_prop_type == "equations" and suffix in (
                "readout.weight",
                "readout.bias",
            )
            if is_equation_readout:
                old_child_cardinality = old_metadata.cardinalities.get(node_name)
                new_child_cardinality = new_metadata.cardinalities.get(node_name)
                if (
                    old_child_cardinality is None
                    or old_child_cardinality != new_child_cardinality
                ):
                    add_fallback(new_key, node_name, "child_cardinality_changed")
                    migrated, tensor_record = legacy_for_key(
                        new_key,
                        destination,
                        reason_prefix="child_cardinality_fallback",
                    )
                    destination_state[new_key] = migrated
                    records.append(tensor_record)
                    continue

                old_parents = old_metadata.parents.get(node_name, ())
                new_parents = new_metadata.parents.get(node_name, ())
                old_identities = [
                    (child_state, parent)
                    for child_state in range(int(old_child_cardinality))
                    for parent in old_parents
                ]
                new_identities = [
                    (child_state, parent)
                    for child_state in range(int(new_child_cardinality))
                    for parent in new_parents
                ]
                old_sizes = {
                    identity: old_metadata.cardinalities[identity[1]]
                    for identity in old_identities
                }
                new_sizes = {
                    identity: new_metadata.cardinalities[identity[1]]
                    for identity in new_identities
                }
                migrated, block_report = migrate_identity_axis_blocks(
                    source,
                    destination,
                    old_identities=old_identities,
                    new_identities=new_identities,
                    axis=0,
                    strategy=strategy,
                    old_block_sizes=old_sizes,
                    new_block_sizes=new_sizes,
                )
                if not block_report["mapping_safe"] or block_report["unsafe_blocks"]:
                    add_fallback(
                        new_key,
                        node_name,
                        str(block_report["reason"] or "parent_cardinality_changed"),
                    )
                action = (
                    "copied_semantic_level_path"
                    if old_parents == new_parents
                    and semantic_old_key != new_key
                    and block_report["preserved_elements"] == destination.numel()
                    else (
                        "copied_exact"
                        if old_parents == new_parents
                        and semantic_old_key == new_key
                        and block_report["preserved_elements"] == destination.numel()
                        else "mapped_equation_rows"
                    )
                )
                destination_state[new_key] = migrated
                records.append(
                    _record(
                        new_key=new_key,
                        old_key=semantic_old_key,
                        tensor=destination,
                        action=action,
                        reason="c2bm_child_parent_state_identity_mapping",
                        counts=block_report,
                        details=block_report,
                    )
                )
                continue

            # Hidden equation layers and every embeddings propagator tensor are
            # parent-independent.  Node identity is sufficient across levels.
            if tuple(source.shape) == tuple(destination.shape):
                destination_state[new_key] = _as_tensor_like(source, destination)
                records.append(
                    _record(
                        new_key=new_key,
                        old_key=semantic_old_key,
                        tensor=destination,
                        action=(
                            "copied_exact"
                            if semantic_old_key == new_key
                            else "copied_semantic_level_path"
                        ),
                        reason="c2bm_node_identity_mapping",
                        counts=_whole_counts(destination, "preserved_elements"),
                    )
                )
            else:
                add_fallback(new_key, node_name, "c2bm_parent_independent_tensor_shape_changed")
                records.append(
                    _record(
                        new_key=new_key,
                        old_key=semantic_old_key,
                        tensor=destination,
                        action="fully_reinitialized",
                        reason="c2bm_parent_independent_tensor_shape_changed",
                        counts=_whole_counts(destination, "fully_reinitialized_elements"),
                        details={"old_shape": [int(value) for value in source.shape]},
                    )
                )
            continue

        # Unchanged modules and unsupported architectures retain legacy exact
        # key/shape semantics.  This also handles newly introduced modules.
        migrated, tensor_record = legacy_for_key(new_key, destination)
        if tensor_record["action"] == "fully_reinitialized":
            add_fallback(
                new_key,
                None,
                "no_safe_semantic_mapping; retained_legacy_shape_mismatch_behavior",
            )
        destination_state[new_key] = migrated
        records.append(tensor_record)

    # Add structural fallbacks that may not own a destination tensor (e.g. a
    # non-root becoming a root removes its C2BM propagator entirely).
    for transition in structural["root_transitions"]:
        add_fallback(None, transition["node"], "root_status_transition")
    parent_probabilities_are_unused = (
        kind == "c2bm"
        and new_metadata is not None
        and new_metadata.prop_type == "embeddings"
    )
    for change in structural["parent_changes"]:
        if change["removed_parents"] and not parent_probabilities_are_unused:
            add_fallback(None, change["node"], "removed_parent_prevents_exact_function_preservation")

    preserved_parameters = 0
    newly_initialized_parameters = 0
    fully_reinitialized_parameters = 0
    total_parameters = 0
    for tensor_record in records:
        is_parameter = tensor_record["new_key"] in parameter_names
        tensor_record["is_parameter"] = is_parameter
        if not is_parameter:
            continue
        total_parameters += tensor_record["elements"]
        preserved_parameters += tensor_record["preserved_elements"]
        newly_initialized_parameters += tensor_record["newly_initialized_elements"]
        fully_reinitialized_parameters += tensor_record["fully_reinitialized_elements"]

    categorized = (
        preserved_parameters
        + newly_initialized_parameters
        + fully_reinitialized_parameters
    )
    if categorized != total_parameters:
        raise RuntimeError(
            "Internal migration accounting error: parameter categories do not "
            f"cover the destination model ({categorized} != {total_parameters})."
        )

    denominator = float(total_parameters) if total_parameters else 1.0
    parameter_counts = {
        "total": int(total_parameters),
        "preserved": int(preserved_parameters),
        "newly_initialized": int(newly_initialized_parameters),
        "fully_reinitialized": int(fully_reinitialized_parameters),
        "preserved_fraction": float(preserved_parameters / denominator),
        "newly_initialized_fraction": float(newly_initialized_parameters / denominator),
        "fully_reinitialized_fraction": float(fully_reinitialized_parameters / denominator),
    }

    preservation_reasons: List[str] = []
    if old_kind != new_kind:
        preservation_reasons.append("model_kind_changed")
    if not structural["metadata_available"] and kind in ("cgm", "c2bm"):
        preservation_reasons.append("graph_metadata_unavailable")
    if structural["removed_nodes"]:
        preservation_reasons.append("nodes_removed")
    if structural["root_transitions"]:
        preservation_reasons.append("root_status_transition")
    if fully_reinitialized_parameters:
        preservation_reasons.append("semantic_predecessor_parameters_reinitialized")
    if fallbacks:
        preservation_reasons.append("unsafe_semantic_mapping_fallback")

    relevant_parent_changes = structural["parent_changes"]
    if kind == "c2bm" and new_metadata and new_metadata.prop_type == "embeddings":
        # The embeddings branch does not consume parent probabilities.
        relevant_parent_changes = []
    if any(change["removed_parents"] for change in relevant_parent_changes):
        preservation_reasons.append("parents_removed")
    has_added_parent = any(change["added_parents"] for change in relevant_parent_changes)
    has_order_change = any(change["order_changed"] for change in relevant_parent_changes)
    if strategy == "partial_random" and has_added_parent:
        preservation_reasons.append("random_nonzero_new_parent_blocks")
    if strategy == "full_reinit" and relevant_parent_changes:
        preservation_reasons.append("legacy_parent_interface_update")
    if strategy == "full_reinit" and has_order_change:
        preservation_reasons.append("legacy_positional_copy_after_parent_reorder")
    if strategy == "partial_zero":
        unsafe_new_blocks = [
            record
            for record in records
            if record["action"] in ("mapped_parent_columns", "mapped_equation_rows")
            and record["newly_initialized_elements"] > record["zeroed_elements"]
        ]
        if unsafe_new_blocks:
            preservation_reasons.append("new_parent_blocks_not_safely_zeroed")

    # Keep reason ordering stable while removing duplicates.
    preservation_reasons = list(dict.fromkeys(preservation_reasons))
    affected_nodes = set(structural["affected_modules"])
    changed_tensors: List[str] = []
    for record in records:
        semantically_changed = record["action"] != "copied_exact"
        cgm_key_match = _CGM_INPUT_WEIGHT_RE.match(record["new_key"])
        if cgm_key_match is not None and cgm_key_match.group(1) in affected_nodes:
            semantically_changed = True
        propagator_key_match = _PROPAGATOR_RE.match(record["new_key"])
        if propagator_key_match is not None and propagator_key_match.group(2) in affected_nodes:
            semantically_changed = True
        record["semantically_changed"] = semantically_changed
        if semantically_changed:
            changed_tensors.append(record["new_key"])
    metadata: Dict[str, Any] = {
        "strategy": strategy,
        "model_kind": kind,
        "old_model_kind": old_kind,
        "new_model_kind": new_kind,
        "parameter_counts": parameter_counts,
        "tensor_records": records,
        "changed_tensors": changed_tensors,
        "changed_tensor_count": len(changed_tensors),
        "added_concepts": structural["added_concepts"],
        "added_concept_count": len(structural["added_concepts"]),
        "affected_modules": structural["affected_modules"],
        "affected_module_count": len(structural["affected_modules"]),
        "structural_change": structural,
        "fallbacks": fallbacks,
        "fallback_count": len(fallbacks),
        "function_preserving": not preservation_reasons,
        "function_preservation_reasons": preservation_reasons,
        "rng_samples_drawn": 0,
    }
    return destination_state, metadata


__all__ = [
    "MIGRATION_STRATEGIES",
    "migrate_architecture_state",
    "migrate_identity_axis_blocks",
]
