"""Empirical gradient-alignment diagnostics for the static F-CM analysis.

The reference objective used here is the sample-size-weighted pooled local
objective

    F_pool(w) = sum_k n_k / sum_r n_r * F_k(w).

For every measured communication round, each F_k gradient is evaluated at the
same broadcast model before local training.  A client gradient is the
example-weighted gradient over one complete pass through its training
dataloader.  The module-wise direction uses the trainable-coordinate masks
produced by the same freezing logic as local training and normalizes every
coordinate over the clients that update it.

This is an observed-trajectory diagnostic.  It does not verify the uniform
"for every w" bound in Assumption (S3).
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from src.utils import maybe_freeze_parameters


ALIGNMENT_METRIC_NAMES = (
    "zeta_hat",
    "zeta_squared_hat",
    "l2_distance",
    "squared_l2_distance",
    "relative_l2_distance",
    "relative_squared_l2_distance",
    "cosine_similarity",
    "angle_degrees",
    "direction_distance",
    "norm_ratio",
    "optimal_federated_rescaling",
    "scale_adjusted_relative_l2",
    "reference_norm",
    "federated_norm",
    "inner_product",
)


@contextmanager
def _preserve_rng_state():
    """Prevent the audit pass from changing the subsequent training RNG stream."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _move_tensor(value: Any, device: torch.device) -> Any:
    return value.to(device) if torch.is_tensor(value) else value


def _model_inputs(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    x = _move_tensor(batch["x"], device)
    c = _move_tensor(batch.get("c"), device)
    inputs: Dict[str, Any] = {
        "x": x,
        "c": c,
        # The rebuttal experiment sets intervention_prob=0.  Constructing the
        # zero mask explicitly also keeps the audit objective deterministic if
        # a caller forgets that override.
        "intervention_index": torch.zeros_like(c) if c is not None else None,
    }
    modality = batch.get("modality")
    if modality is not None:
        inputs["modality"] = modality
    return inputs


def _batch_loss(model: nn.Module, batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    inputs = _model_inputs(batch, device)
    y = _move_tensor(batch["y"], device)
    c = inputs["c"]
    y_output, c_output = model(**inputs)
    y_hat, c_hat = model.filter_output_for_loss(y_output, c_output)
    loss = model.loss(y_hat, y, c_hat, c)
    if loss is None:
        raise RuntimeError(
            "The client objective returned no loss. This usually means that the "
            "selected model requires task supervision but the client has no task labels."
        )
    if loss.ndim != 0:
        loss = loss.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss encountered during alignment audit: {loss}")
    return loss


def _full_pass_gradient(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    device: torch.device,
    max_batches: Optional[int],
) -> Tuple[torch.Tensor, int, float]:
    """Return the example-weighted gradient over a complete client-data pass."""

    try:
        available_batches = len(dataloader)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("Alignment audit requires a dataloader with a finite length.") from exc

    n_batches = available_batches if max_batches is None else min(available_batches, max_batches)
    if n_batches <= 0:
        raise ValueError("Alignment audit received an empty client dataloader.")

    model.zero_grad(set_to_none=True)
    weighted_loss_sum = 0.0
    n_examples = 0
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= n_batches:
            break
        loss = _batch_loss(model, batch, device)
        batch_size = int(batch["x"].shape[0])
        if batch_size <= 0:
            continue
        (loss * float(batch_size)).backward()
        weighted_loss_sum += float(loss.detach().cpu()) * batch_size
        n_examples += batch_size

    if n_examples <= 0:
        raise ValueError("Alignment audit received no examples from the client dataloader.")

    chunks: List[torch.Tensor] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            chunks.append(torch.zeros(parameter.numel(), dtype=torch.float64))
        else:
            chunks.append(
                parameter.grad.detach().reshape(-1).cpu().to(torch.float64)
                / float(n_examples)
            )

    model.zero_grad(set_to_none=True)
    return torch.cat(chunks), n_batches, weighted_loss_sum / float(n_examples)


def _trainable_coordinate_mask(
    model: nn.Module,
    dataloader: Any,
    y_to_freeze: bool,
    learning_mode: str,
    freezing: bool,
) -> torch.Tensor:
    """Build the P^(k) mask using the local-training freezing implementation."""

    original_flags = [parameter.requires_grad for parameter in model.parameters()]
    with _preserve_rng_state():
        maybe_freeze_parameters(
            train_dataloader=dataloader,
            y_to_freeze=y_to_freeze,
            model=model,
            learning=learning_mode,
            freezing=freezing,
        )

    mask_chunks = [
        torch.full((parameter.numel(),), parameter.requires_grad, dtype=torch.bool)
        for parameter in model.parameters()
    ]

    # The unmasked local gradient is needed for grad F_k and grad F_pool.
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    mask = torch.cat(mask_chunks)

    # This model is a private audit copy, but restore the flags for callers that
    # use the helper independently.
    for parameter, flag in zip(model.parameters(), original_flags):
        parameter.requires_grad_(flag)
    return mask


def _module_name(parameter_name: str) -> str:
    """Map architecture-specific parameter names to readable module blocks."""

    parts = parameter_name.split(".")
    if parameter_name.startswith("modality_encoders.") and len(parts) >= 2:
        return f"encoder/{parts[1]}"
    if parameter_name.startswith("encoder."):
        return "encoder"
    if parameter_name.startswith("c_mlp.") and len(parts) >= 2:
        return f"concept/{parts[1]}"
    if parameter_name.startswith("concept_encoders.") and len(parts) >= 2:
        return f"variable/{parts[1]}"
    if parameter_name.startswith("propagators.") and len(parts) >= 3:
        return f"variable/{parts[2]}"
    if parameter_name.startswith("decoder."):
        return "task"
    if parameter_name.startswith("classifier."):
        return "task"
    return parts[0]


def _parameter_layout(model: nn.Module) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    names: List[str] = []
    layout: List[Tuple[int, int, str]] = []
    offset = 0
    for name, parameter in model.named_parameters():
        names.append(name)
        end = offset + parameter.numel()
        layout.append((offset, end, _module_name(name)))
        offset = end
    return names, layout


def _alignment_metrics(
    federated: torch.Tensor,
    reference: torch.Tensor,
    eps: float,
) -> Dict[str, float]:
    delta = federated - reference
    squared_distance = float(torch.dot(delta, delta))
    l2_distance = math.sqrt(max(0.0, squared_distance))
    reference_sq = float(torch.dot(reference, reference))
    federated_sq = float(torch.dot(federated, federated))
    reference_norm = math.sqrt(max(0.0, reference_sq))
    federated_norm = math.sqrt(max(0.0, federated_sq))
    inner_product = float(torch.dot(federated, reference))

    if reference_norm <= eps and federated_norm <= eps:
        cosine = 1.0
        direction_distance = 0.0
        angle_degrees = 0.0
        optimal_rescaling = 1.0
    elif reference_norm <= eps or federated_norm <= eps:
        cosine = 0.0
        direction_distance = math.sqrt(2.0)
        angle_degrees = 90.0
        optimal_rescaling = 0.0
    else:
        cosine = inner_product / (reference_norm * federated_norm)
        cosine = max(-1.0, min(1.0, cosine))
        normalized_delta = federated / federated_norm - reference / reference_norm
        direction_distance = float(torch.linalg.vector_norm(normalized_delta))
        angle_degrees = math.degrees(math.acos(cosine))
        optimal_rescaling = max(0.0, inner_product / federated_sq)

    scale_adjusted_delta = optimal_rescaling * federated - reference
    scale_adjusted_relative_l2 = float(
        torch.linalg.vector_norm(scale_adjusted_delta)
    ) / max(reference_norm, eps)

    return {
        "zeta_hat": l2_distance,
        "zeta_squared_hat": squared_distance,
        "l2_distance": l2_distance,
        "squared_l2_distance": squared_distance,
        "relative_l2_distance": l2_distance / max(reference_norm, eps),
        "relative_squared_l2_distance": squared_distance / max(reference_sq, eps),
        "cosine_similarity": cosine,
        "angle_degrees": angle_degrees,
        "direction_distance": direction_distance,
        "norm_ratio": federated_norm / max(reference_norm, eps),
        "optimal_federated_rescaling": optimal_rescaling,
        "scale_adjusted_relative_l2": scale_adjusted_relative_l2,
        "reference_norm": reference_norm,
        "federated_norm": federated_norm,
        "inner_product": inner_product,
    }


def _module_metrics(
    federated: torch.Tensor,
    reference: torch.Tensor,
    layout: Sequence[Tuple[int, int, str]],
    eps: float,
) -> Dict[str, Dict[str, float]]:
    ranges: Dict[str, List[Tuple[int, int]]] = {}
    for start, end, module in layout:
        ranges.setdefault(module, []).append((start, end))

    metrics: Dict[str, Dict[str, float]] = {}
    for module, module_ranges in ranges.items():
        federated_chunks = [federated[start:end] for start, end in module_ranges]
        reference_chunks = [reference[start:end] for start, end in module_ranges]
        metrics[module] = _alignment_metrics(
            torch.cat(federated_chunks),
            torch.cat(reference_chunks),
            eps,
        )
    return metrics


def compute_gradient_alignment(
    *,
    global_model: nn.Module,
    train_dataloaders: Sequence[Any],
    client_indices: Sequence[int],
    y_to_freeze: Sequence[bool],
    learning_mode: str,
    freezing: bool,
    device: str | torch.device,
    round_index: int,
    max_batches: Optional[int] = None,
    include_module_metrics: bool = True,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Compute the empirical S3 mismatch at a broadcast model.

    ``client_indices`` are zero-based indices into ``train_dataloaders`` and
    ``y_to_freeze`` has one entry in the same order.
    """

    if len(client_indices) == 0:
        raise ValueError("At least one client is required for gradient alignment.")
    if len(client_indices) != len(y_to_freeze):
        raise ValueError("client_indices and y_to_freeze must have equal lengths.")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive or None.")

    audit_device = torch.device(device)
    global_model = copy.deepcopy(global_model).to(audit_device)
    global_model.eval()
    parameter_names, layout = _parameter_layout(global_model)
    total_parameters = sum(parameter.numel() for parameter in global_model.parameters())
    if total_parameters == 0:
        raise ValueError("Cannot audit a model with no parameters.")

    gradients: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    sample_counts: List[int] = []
    client_details: List[Dict[str, Any]] = []

    for client_index, freeze_task in zip(client_indices, y_to_freeze):
        dataloader = train_dataloaders[client_index]
        client_model = copy.deepcopy(global_model)
        client_mask = _trainable_coordinate_mask(
            client_model,
            dataloader,
            y_to_freeze=freeze_task,
            learning_mode=learning_mode,
            freezing=freezing,
        )
        for parameter in client_model.parameters():
            parameter.requires_grad_(True)

        # Keep dataloader shuffling and any dataset-side randomness from changing
        # the local training that follows this read-only audit.
        with _preserve_rng_state():
            gradient, used_batches, mean_loss = _full_pass_gradient(
                client_model,
                dataloader,
                audit_device,
                max_batches=max_batches,
            )

        if gradient.numel() != total_parameters or client_mask.numel() != total_parameters:
            raise RuntimeError("Client gradient/mask layout does not match the broadcast model.")

        n_samples = len(dataloader.dataset)
        gradients.append(gradient)
        masks.append(client_mask)
        sample_counts.append(n_samples)
        client_details.append(
            {
                "client_index": int(client_index),
                "client_id": int(client_index + 1),
                "n_samples": int(n_samples),
                "n_batches": int(used_batches),
                "mean_loss": mean_loss,
                "trainable_coordinate_fraction": float(client_mask.to(torch.float64).mean()),
                "task_frozen": bool(freeze_task),
            }
        )
        del client_model

    sample_weights = torch.tensor(sample_counts, dtype=torch.float64)
    total_samples = float(sample_weights.sum())
    stacked_gradients = torch.stack(gradients)
    stacked_masks = torch.stack(masks).to(torch.float64)

    # grad F_pool(w_t) for F_pool = sum_k n_k / N * F_k.
    reference_gradient = (
        stacked_gradients * (sample_weights / total_samples).unsqueeze(1)
    ).sum(dim=0)

    # sum_k R^(k) grad F_k, with R weights normalized independently on
    # every trainable coordinate.
    coordinate_denominator = (
        stacked_masks * sample_weights.unsqueeze(1)
    ).sum(dim=0)
    federated_numerator = (
        stacked_gradients * stacked_masks * sample_weights.unsqueeze(1)
    ).sum(dim=0)
    covered = coordinate_denominator > 0
    federated_gradient = torch.zeros_like(reference_gradient)
    federated_gradient[covered] = (
        federated_numerator[covered] / coordinate_denominator[covered]
    )

    overall = _alignment_metrics(federated_gradient, reference_gradient, eps)
    modules = (
        _module_metrics(federated_gradient, reference_gradient, layout, eps)
        if include_module_metrics
        else {}
    )

    return {
        "round": int(round_index),
        "definition": {
            "reference": "sample-size-weighted pooled empirical local objective gradient",
            "federated": "coordinate-wise module-normalized masked client gradient",
            "local_gradient": (
                "example-weighted gradient over a complete client-data pass "
                "at the broadcast model"
            ),
            "interventions": "disabled (zero intervention mask)",
            "scope": "observed trajectory; not a uniform estimate over all parameters w",
        },
        "overall": overall,
        "modules": modules,
        "diagnostics": {
            "n_clients": len(client_indices),
            "total_client_samples": int(sum(sample_counts)),
            "n_parameters": int(total_parameters),
            "covered_coordinate_fraction": float(covered.to(torch.float64).mean()),
            "uncovered_coordinates": int((~covered).sum()),
            "max_batches_per_client": max_batches,
            "parameter_names": parameter_names,
        },
        "clients": client_details,
    }


def should_measure_alignment(settings: Mapping[str, Any], round_index: int) -> bool:
    if not bool(settings.get("enabled", False)):
        return False
    explicit_rounds = settings.get("rounds")
    if explicit_rounds is not None:
        return round_index in {int(value) for value in explicit_rounds}
    every = int(settings.get("every_n_rounds", 1))
    if every <= 0:
        raise ValueError("gradient_alignment.every_n_rounds must be positive.")
    return (round_index - 1) % every == 0


def _series_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize_gradient_alignment(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(records) == 0:
        return {"n_measured_rounds": 0}

    summary: Dict[str, Any] = {
        "n_measured_rounds": len(records),
        "measured_rounds": [int(record["round"]) for record in records],
        "experiment": dict(records[0].get("experiment", {})),
        "metrics": {},
        "modules": {},
    }
    for metric_name in ALIGNMENT_METRIC_NAMES:
        values = [float(record["overall"][metric_name]) for record in records]
        summary["metrics"][metric_name] = _series_summary(values)

    module_names = sorted(
        {
            module_name
            for record in records
            for module_name in record.get("modules", {})
        }
    )
    for module_name in module_names:
        module_records = [
            record["modules"][module_name]
            for record in records
            if module_name in record.get("modules", {})
        ]
        summary["modules"][module_name] = {
            metric_name: _series_summary(
                [float(module[metric_name]) for module in module_records]
            )
            for metric_name in ALIGNMENT_METRIC_NAMES
        }

    coverages = [
        float(record["diagnostics"]["covered_coordinate_fraction"])
        for record in records
    ]
    uncovered = [
        int(record["diagnostics"]["uncovered_coordinates"])
        for record in records
    ]
    summary["coverage"] = {
        "covered_coordinate_fraction_min": min(coverages),
        "covered_coordinate_fraction_max": max(coverages),
        "uncovered_coordinates_max": max(uncovered),
    }
    elapsed = [
        float(record["elapsed_seconds"])
        for record in records
        if "elapsed_seconds" in record
    ]
    if elapsed:
        summary["audit_elapsed_seconds"] = _series_summary(elapsed)

    max_squared = summary["metrics"]["squared_l2_distance"]["max"]
    summary["trajectory_diagnostic"] = {
        "zeta_squared_max_observed": max_squared,
        "zeta_max_observed": math.sqrt(max(0.0, max_squared)),
        "qualification": (
            "Maximum mismatch on measured broadcast iterates only; this is not "
            "an upper bound over all w."
        ),
    }
    return summary


def save_gradient_alignment(
    records: Sequence[Mapping[str, Any]],
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Persist crash-resilient round data plus reviewer-ready summaries."""

    os.makedirs(output_dir, exist_ok=True)
    records_path = os.path.join(output_dir, "gradient_alignment_rounds.json")
    summary_path = os.path.join(output_dir, "gradient_alignment_summary.json")
    csv_path = os.path.join(output_dir, "gradient_alignment_rounds.csv")

    with open(records_path, "w") as handle:
        json.dump(list(records), handle, indent=2)

    summary = summarize_gradient_alignment(records)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    fieldnames = ["round", *ALIGNMENT_METRIC_NAMES]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {"round": int(record["round"])}
            row.update(
                {
                    metric_name: float(record["overall"][metric_name])
                    for metric_name in ALIGNMENT_METRIC_NAMES
                }
            )
            writer.writerow(row)

    return summary
