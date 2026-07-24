"""Controlled perturbations of concept annotations used for training.

The project represents an unavailable concept annotation with ``-1``.  These
utilities perturb only annotations that are already available, so they preserve
the client-level concept supervision pattern created by ``generate_split.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch


VALID_MODES = {"none", "missing", "noise"}


@dataclass(frozen=True)
class ConceptPerturbationSpec:
    mode: str = "none"
    probability: float = 0.0
    seed_offset: int = 17_041
    ignore_index: int = -1

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "ConceptPerturbationSpec":
        if config is None:
            return cls()

        spec = cls(
            mode=str(config.get("mode", "none")).lower(),
            probability=float(config.get("probability", 0.0)),
            seed_offset=int(config.get("seed_offset", 17_041)),
            ignore_index=int(config.get("ignore_index", -1)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Unknown concept perturbation mode '{self.mode}'. "
                f"Expected one of {sorted(VALID_MODES)}."
            )
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Concept perturbation probability must be in [0, 1].")
        if self.mode == "none" and self.probability != 0.0:
            raise ValueError("mode='none' requires probability=0.")


def concept_perturbation_seed(
    experiment_seed: int,
    client_id: int,
    seed_offset: int,
) -> int:
    """Return a stable client-specific seed independent of Python hash state."""

    if client_id < 1:
        raise ValueError("client_id must be one-indexed and positive.")
    modulus = 2**63 - 1
    return (
        int(seed_offset)
        + int(experiment_seed) * 1_000_003
        + int(client_id) * 10_007
    ) % modulus


def perturb_concept_annotations(
    annotations: torch.Tensor,
    cardinalities: Sequence[int],
    spec: ConceptPerturbationSpec,
    *,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Mask or corrupt available concept annotations.

    Selection is i.i.d. over available sample-concept entries.  For noise,
    every selected class is replaced uniformly by one of the other valid
    classes.  Drawing one selection tensor and one replacement tensor for the
    full input also makes experiments at increasing probabilities nested when
    they use the same seed.
    """

    spec.validate()
    if annotations.ndim != 2:
        raise ValueError(
            "Concept annotations must have shape [n_samples, n_concepts], "
            f"received {tuple(annotations.shape)}."
        )
    if annotations.shape[1] != len(cardinalities):
        raise ValueError(
            "The number of concept cardinalities does not match the annotation "
            f"columns: {len(cardinalities)} != {annotations.shape[1]}."
        )

    output = annotations.clone()
    finite = torch.isfinite(output)
    available = finite & output.ne(spec.ignore_index)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    selection_uniform = torch.rand(
        output.shape,
        generator=generator,
        device="cpu",
        dtype=torch.float64,
    ).to(output.device)
    replacement_uniform = torch.rand(
        output.shape,
        generator=generator,
        device="cpu",
        dtype=torch.float64,
    ).to(output.device)
    selected = available & selection_uniform.lt(spec.probability)

    if spec.mode == "missing":
        output[selected] = spec.ignore_index
    elif spec.mode == "noise":
        for concept_idx, cardinality_value in enumerate(cardinalities):
            cardinality = int(cardinality_value)
            if cardinality < 2:
                raise ValueError(
                    f"Concept column {concept_idx} has invalid cardinality {cardinality}; "
                    "noise requires at least two valid classes."
                )

            column_available = available[:, concept_idx]
            if column_available.any():
                labels = output[column_available, concept_idx]
                rounded = labels.round()
                if not torch.allclose(labels, rounded):
                    raise ValueError(
                        f"Concept column {concept_idx} contains non-integer labels."
                    )
                if (rounded.lt(0) | rounded.ge(cardinality)).any():
                    raise ValueError(
                        f"Concept column {concept_idx} contains a label outside "
                        f"[0, {cardinality - 1}]."
                    )

            column_selected = selected[:, concept_idx]
            if not column_selected.any():
                continue

            old_labels = output[column_selected, concept_idx].long()
            offsets = (
                replacement_uniform[column_selected, concept_idx]
                .mul(cardinality - 1)
                .floor()
                .long()
                .add(1)
            )
            output[column_selected, concept_idx] = (
                (old_labels + offsets) % cardinality
            ).to(output.dtype)
    elif spec.mode != "none":
        raise AssertionError(f"Unhandled perturbation mode: {spec.mode}")

    per_concept = []
    for concept_idx in range(annotations.shape[1]):
        eligible_count = int(available[:, concept_idx].sum().item())
        selected_count = int(selected[:, concept_idx].sum().item())
        per_concept.append(
            {
                "concept_index": concept_idx,
                "eligible": eligible_count,
                "perturbed": selected_count,
                "realized_probability": (
                    selected_count / eligible_count if eligible_count else 0.0
                ),
            }
        )

    eligible_total = int(available.sum().item())
    perturbed_total = int(selected.sum().item())
    report = {
        "mode": spec.mode,
        "requested_probability": spec.probability,
        "eligible": eligible_total,
        "perturbed": perturbed_total,
        "realized_probability": (
            perturbed_total / eligible_total if eligible_total else 0.0
        ),
        "seed": int(seed),
        "per_concept": per_concept,
    }
    return output, report

