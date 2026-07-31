"""Persistence and curve metrics for architecture-initialization ablations.

This module intentionally has no dependency on the training loop.  The loop only
needs to provide one task-validation loss per eligible client and round.  All
aggregation is unweighted across clients, as required by the ablation protocol.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


DEFAULT_OUTPUT_DIR = Path("results") / "architecture_ablation"


@dataclass(frozen=True)
class AblationMetricConfig:
    """Window definitions used to summarize a seed-level loss curve.

    ``post_horizon`` counts recorded post-shift rounds, including the shift
    round.  ``None`` uses every recorded post-shift round.  Recovery smoothing
    is trailing and post-shift-only; a full smoothing window is required.
    """

    pre_window: int = 5
    post_peak_window: int = 5
    smoothing_window: int = 5
    stability_window: int = 5
    post_horizon: Optional[int] = None
    final_window: int = 5

    def __post_init__(self) -> None:
        for name in (
            "pre_window",
            "post_peak_window",
            "smoothing_window",
            "stability_window",
            "final_window",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.post_horizon is not None:
            if (
                not isinstance(self.post_horizon, int)
                or isinstance(self.post_horizon, bool)
                or self.post_horizon <= 0
            ):
                raise ValueError(
                    "post_horizon must be None or a positive integer, "
                    f"got {self.post_horizon!r}."
                )

    @classmethod
    def from_value(
        cls, value: Optional[Union["AblationMetricConfig", Mapping[str, Any]]]
    ) -> "AblationMetricConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("metric_config must be an AblationMetricConfig or mapping.")

        aliases = {
            "pre_shift_window": "pre_window",
            "peak_window": "post_peak_window",
            "post_shift_peak_window": "post_peak_window",
            "recovery_smoothing_window": "smoothing_window",
            "recovery_stability_window": "stability_window",
            "post_shift_horizon": "post_horizon",
            "final_window_size": "final_window",
        }
        normalized: Dict[str, Any] = {}
        valid_fields = set(cls.__dataclass_fields__)
        for key, item in value.items():
            normalized_key = aliases.get(str(key), str(key))
            if normalized_key in valid_fields:
                normalized[normalized_key] = item
        return cls(**normalized)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_finite(value: Any) -> bool:
    value_float = _as_float(value)
    return value_float is not None and math.isfinite(value_float)


def _finite_mean(values: Sequence[float]) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def compute_ablation_metrics(
    rounds: Sequence[Union[int, float]],
    client_mean_losses: Sequence[Union[int, float]],
    shift_round: Union[int, float],
    *,
    metric_config: Optional[
        Union[AblationMetricConfig, Mapping[str, Any]]
    ] = None,
    pre_window: Optional[int] = None,
    post_peak_window: Optional[int] = None,
    smoothing_window: Optional[int] = None,
    stability_window: Optional[int] = None,
    post_horizon: Optional[int] = None,
    final_window: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute seed-level shift metrics from an unweighted client-mean curve.

    The shift round is the first post-shift round.  Peak loss is restricted to
    the configured initial post-shift peak window.  Minimum, cumulative excess,
    recovery, and final loss use the configured post-shift horizon.

    Recovery is the first endpoint of a *full*, trailing, post-shift-only
    smoothing window whose value is at or below the pre-shift baseline and is
    followed by ``stability_window - 1`` consecutive smoothed values that also
    remain at or below the baseline.  Missing/non-consecutive rounds invalidate
    any smoothing window that crosses them.
    """

    config = AblationMetricConfig.from_value(metric_config)
    overrides = {
        "pre_window": pre_window,
        "post_peak_window": post_peak_window,
        "smoothing_window": smoothing_window,
        "stability_window": stability_window,
        "post_horizon": post_horizon,
        "final_window": final_window,
    }
    if any(value is not None for value in overrides.values()):
        config_values = asdict(config)
        config_values.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        config = AblationMetricConfig(**config_values)

    if len(rounds) != len(client_mean_losses):
        raise ValueError(
            "rounds and client_mean_losses must have equal length: "
            f"{len(rounds)} != {len(client_mean_losses)}."
        )

    pairs: List[Tuple[float, float]] = []
    for round_value, loss_value in zip(rounds, client_mean_losses):
        round_float = _as_float(round_value)
        loss_float = _as_float(loss_value)
        if round_float is None:
            raise ValueError(f"Invalid round value: {round_value!r}.")
        pairs.append(
            (
                round_float,
                float("nan") if loss_float is None else loss_float,
            )
        )
    pairs.sort(key=lambda item: item[0])
    sorted_rounds = [item[0] for item in pairs]
    if len(set(sorted_rounds)) != len(sorted_rounds):
        raise ValueError("rounds must be unique.")

    shift_float = float(shift_round)
    pre_pairs = [item for item in pairs if item[0] < shift_float]
    pre_window_pairs = pre_pairs[-config.pre_window :]
    pre_values = [loss for _, loss in pre_window_pairs if math.isfinite(loss)]
    pre_shift_loss = _finite_mean(pre_values)

    post_pairs = [item for item in pairs if item[0] >= shift_float]
    if config.post_horizon is not None:
        post_pairs = post_pairs[: config.post_horizon]

    peak_pairs = post_pairs[: config.post_peak_window]
    finite_peak_pairs = [item for item in peak_pairs if math.isfinite(item[1])]
    if finite_peak_pairs:
        post_shift_peak_round, post_shift_peak_loss = max(
            finite_peak_pairs, key=lambda item: item[1]
        )
    else:
        post_shift_peak_round, post_shift_peak_loss = None, float("nan")

    finite_post_pairs = [item for item in post_pairs if math.isfinite(item[1])]
    if finite_post_pairs:
        minimum_post_shift_round, minimum_post_shift_loss = min(
            finite_post_pairs, key=lambda item: item[1]
        )
    else:
        minimum_post_shift_round, minimum_post_shift_loss = None, float("nan")

    if math.isfinite(pre_shift_loss) and math.isfinite(post_shift_peak_loss):
        loss_spike = post_shift_peak_loss - pre_shift_loss
    else:
        loss_spike = float("nan")

    if math.isfinite(pre_shift_loss):
        cumulative_excess_loss = float(
            sum(max(0.0, loss - pre_shift_loss) for _, loss in finite_post_pairs)
        )
    else:
        cumulative_excess_loss = float("nan")

    final_pairs = post_pairs[-config.final_window :]
    final_values = [loss for _, loss in final_pairs if math.isfinite(loss)]
    final_window_loss = _finite_mean(final_values)
    if math.isfinite(pre_shift_loss) and math.isfinite(final_window_loss):
        final_delta = final_window_loss - pre_shift_loss
    else:
        final_delta = float("nan")

    # Each entry corresponds to one post-shift round.  None denotes a missing,
    # incomplete, or non-consecutive trailing smoothing window.
    smoothed: List[Optional[float]] = [None] * len(post_pairs)
    smooth_width = config.smoothing_window
    for index in range(smooth_width - 1, len(post_pairs)):
        window = post_pairs[index - smooth_width + 1 : index + 1]
        window_rounds = [item[0] for item in window]
        window_losses = [item[1] for item in window]
        consecutive = all(
            math.isclose(window_rounds[pos] - window_rounds[pos - 1], 1.0)
            for pos in range(1, len(window_rounds))
        )
        if consecutive and all(math.isfinite(loss) for loss in window_losses):
            smoothed[index] = float(np.mean(window_losses))

    recovery_round: Optional[float] = None
    recovery_smoothed_loss: Optional[float] = None
    if math.isfinite(pre_shift_loss):
        for index, smooth_loss in enumerate(smoothed):
            if smooth_loss is None or smooth_loss > pre_shift_loss:
                continue
            stable_end = index + config.stability_window
            if stable_end > len(smoothed):
                continue
            stable_values = smoothed[index:stable_end]
            stable_rounds = [round_value for round_value, _ in post_pairs[index:stable_end]]
            stable_consecutive = all(
                math.isclose(stable_rounds[pos] - stable_rounds[pos - 1], 1.0)
                for pos in range(1, len(stable_rounds))
            )
            if (
                stable_consecutive
                and all(value is not None for value in stable_values)
                and all(value <= pre_shift_loss for value in stable_values if value is not None)
            ):
                recovery_round = post_pairs[index][0]
                recovery_smoothed_loss = smooth_loss
                break

    recovery_time = (
        recovery_round - shift_float if recovery_round is not None else None
    )

    def _round_output(value: Optional[float]) -> Optional[Union[int, float]]:
        if value is None:
            return None
        if float(value).is_integer():
            return int(value)
        return float(value)

    result: Dict[str, Any] = {
        "shift_round": _round_output(shift_float),
        "pre_shift_loss": pre_shift_loss,
        "pre_shift_window_count": len(pre_values),
        "pre_shift_window_start_round": _round_output(pre_window_pairs[0][0])
        if pre_window_pairs
        else None,
        "pre_shift_window_end_round": _round_output(pre_window_pairs[-1][0])
        if pre_window_pairs
        else None,
        "post_shift_peak_loss": post_shift_peak_loss,
        "post_shift_peak_round": _round_output(post_shift_peak_round),
        "loss_spike": loss_spike,
        "minimum_post_shift_loss": minimum_post_shift_loss,
        "minimum_post_shift_loss_round": _round_output(minimum_post_shift_round),
        "recovered": recovery_round is not None,
        "recovery_round": _round_output(recovery_round),
        "recovery_time": _round_output(recovery_time),
        "recovery_time_rounds": _round_output(recovery_time),
        "recovery_smoothed_loss": recovery_smoothed_loss,
        "cumulative_excess_loss": cumulative_excess_loss,
        "final_window_loss": final_window_loss,
        "final_delta": final_delta,
        "post_shift_rounds_evaluated": len(finite_post_pairs),
        "post_shift_horizon_start_round": _round_output(post_pairs[0][0])
        if post_pairs
        else None,
        "post_shift_horizon_end_round": _round_output(post_pairs[-1][0])
        if post_pairs
        else None,
        **asdict(config),
    }
    return result


def _config_select(config: Any, path: str, default: Any = None) -> Any:
    if config is None:
        return default
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.select(config, path, default=default)
    except (ImportError, TypeError, ValueError):
        pass

    current = config
    for component in path.split("."):
        if isinstance(current, Mapping):
            if component not in current:
                return default
            current = current[component]
        elif hasattr(current, component):
            current = getattr(current, component)
        else:
            return default
    return current


def _first_config_value(config: Any, paths: Sequence[str], default: Any = None) -> Any:
    sentinel = object()
    for path in paths:
        value = _config_select(config, path, sentinel)
        if value is not sentinel and value is not None:
            return value
    return default


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _resolved_config(config: Any) -> Any:
    if config is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return _json_safe(OmegaConf.to_container(config, resolve=True))
    except (ImportError, TypeError, ValueError):
        pass
    return _json_safe(config)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    safe = _json_safe(value)
    if safe is None:
        return ""
    if isinstance(safe, (dict, list)):
        return json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return safe


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
    os.replace(temporary, path)


def _lookup_nested(mapping: Mapping[str, Any], paths: Sequence[str]) -> Any:
    sentinel = object()
    for path in paths:
        current: Any = mapping
        for component in path.split("."):
            if not isinstance(current, Mapping) or component not in current:
                current = sentinel
                break
            current = current[component]
        if current is not sentinel and current is not None:
            return current
    return None


def _count_or_length(value: Any) -> Optional[Union[int, float]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        numeric = _as_float(value)
        return numeric
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    ):
        return len(value)
    numeric = _as_float(value)
    if numeric is None:
        return None
    return int(numeric) if numeric.is_integer() else numeric


def canonical_structural_metrics(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Extract canonical scalar shift fields from a migration report.

    Raw metadata is always persisted separately, so this permissive normalizer
    only controls which structural values are copied into ``seed_metrics``.
    """

    if not metadata:
        return {}

    count_paths = {
        "parameters_total": (
            "parameters_total",
            "total_parameters",
            "post_shift_total_parameters",
            "total_post_shift_parameters",
            "total_numel",
            "parameter_counts.total",
            "parameter_counts.total_post_shift",
            "parameter_counts.total_parameters",
        ),
        "parameters_preserved": (
            "parameters_preserved",
            "preserved_parameters",
            "preserved_numel",
            "parameter_counts.preserved",
            "parameter_counts.parameters_preserved",
        ),
        "parameters_newly_initialized": (
            "parameters_newly_initialized",
            "newly_initialized_parameters",
            "newly_initialized_numel",
            "new_parameters",
            "parameter_counts.newly_initialized",
            "parameter_counts.new",
        ),
        "parameters_fully_reinitialized": (
            "parameters_fully_reinitialized",
            "fully_reinitialized_parameters",
            "fully_reinitialized_numel",
            "reinitialized_parameters",
            "parameter_counts.fully_reinitialized",
            "parameter_counts.reinitialized",
        ),
    }
    result: Dict[str, Any] = {}
    for output_name, paths in count_paths.items():
        value = _count_or_length(_lookup_nested(metadata, paths))
        if value is not None:
            result[output_name] = value

    if "parameters_total" not in result:
        category_names = (
            "parameters_preserved",
            "parameters_newly_initialized",
            "parameters_fully_reinitialized",
        )
        if all(name in result for name in category_names):
            result["parameters_total"] = sum(float(result[name]) for name in category_names)

    fraction_paths = {
        "parameters_preserved_fraction": (
            "parameters_preserved_fraction",
            "fraction_parameters_preserved",
            "preserved_fraction",
            "parameter_counts.preserved_fraction",
            "parameter_fractions.preserved",
        ),
        "parameters_newly_initialized_fraction": (
            "parameters_newly_initialized_fraction",
            "fraction_parameters_newly_initialized",
            "newly_initialized_fraction",
            "parameter_counts.newly_initialized_fraction",
            "parameter_fractions.newly_initialized",
        ),
        "parameters_fully_reinitialized_fraction": (
            "parameters_fully_reinitialized_fraction",
            "fraction_parameters_fully_reinitialized",
            "fully_reinitialized_fraction",
            "parameter_counts.fully_reinitialized_fraction",
            "parameter_fractions.fully_reinitialized",
        ),
    }
    for output_name, paths in fraction_paths.items():
        value = _as_float(_lookup_nested(metadata, paths))
        if value is not None:
            result[output_name] = value

    total = _as_float(result.get("parameters_total"))
    if total is not None and total > 0:
        for category in (
            "preserved",
            "newly_initialized",
            "fully_reinitialized",
        ):
            count_name = f"parameters_{category}"
            fraction_name = f"parameters_{category}_fraction"
            if count_name in result and fraction_name not in result:
                result[fraction_name] = float(result[count_name]) / total

    object_paths = {
        "number_added_concepts": (
            "number_added_concepts",
            "n_added_concepts",
            "added_concept_count",
            "added_concepts",
            "structural_counts.added_concepts",
        ),
        "number_affected_modules": (
            "number_affected_modules",
            "n_affected_modules",
            "affected_module_count",
            "affected_modules",
            "structural_counts.affected_modules",
        ),
        "number_changed_tensors": (
            "number_changed_tensors",
            "n_changed_tensors",
            "changed_tensor_count",
            "changed_tensors",
            "structural_counts.changed_tensors",
        ),
    }
    for output_name, paths in object_paths.items():
        value = _count_or_length(_lookup_nested(metadata, paths))
        if value is not None:
            result[output_name] = value

    for output_name, paths in {
        "realized_pre_concept_coverage": (
            "realized_pre_concept_coverage",
            "structural_change.realized_pre_concept_coverage",
        ),
        "realized_post_concept_coverage": (
            "realized_post_concept_coverage",
            "structural_change.realized_post_concept_coverage",
        ),
    }.items():
        value = _as_float(_lookup_nested(metadata, paths))
        if value is not None:
            result[output_name] = value

    function_preserving = _lookup_nested(
        metadata, ("function_preserving",)
    )
    if isinstance(function_preserving, (bool, np.bool_)):
        result["function_preserving"] = bool(function_preserving)
    fallback_count = _count_or_length(
        _lookup_nested(metadata, ("fallback_count", "fallbacks"))
    )
    if fallback_count is not None:
        result["fallback_count"] = fallback_count

    return result


class ArchitectureAblationRecorder:
    """Record one seed/method run without coupling plotting to training.

    Parameters can be supplied explicitly or inferred from a Hydra/OmegaConf
    configuration.  Explicit values always win.  ``record_round`` persists both
    tidy CSV files immediately, so a partially completed job remains plottable.
    """

    CLIENT_FIELDS = (
        "dataset",
        "architecture",
        "split_group",
        "method",
        "seed",
        "drift_round",
        "round",
        "client_id",
        "task_loss",
        "n_labeled_samples",
        "eligible",
        "evaluated",
        "phase",
        "reason",
    )
    ROUND_FIELDS = (
        "dataset",
        "architecture",
        "split_group",
        "method",
        "seed",
        "drift_round",
        "round",
        "mean_task_loss",
        "std_task_loss",
        "n_evaluated_clients",
        "n_eligible_clients",
    )
    SHIFT_SNAPSHOT_CLIENT_FIELDS = (
        "dataset",
        "architecture",
        "split_group",
        "method",
        "seed",
        "drift_round",
        "stage",
        "round",
        "client_id",
        "task_loss",
        "n_labeled_samples",
        "eligible",
        "evaluated",
        "phase",
        "reason",
    )
    SHIFT_SNAPSHOT_SUMMARY_FIELDS = (
        "dataset",
        "architecture",
        "split_group",
        "method",
        "seed",
        "drift_round",
        "stage",
        "round",
        "mean_task_loss",
        "std_task_loss",
        "n_evaluated_clients",
        "n_eligible_clients",
    )

    def __init__(
        self,
        cfg: Any = None,
        *,
        output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
        dataset: Optional[str] = None,
        architecture: Optional[str] = None,
        split_group: Optional[str] = None,
        method: Optional[str] = None,
        seed: Optional[int] = None,
        drift_round: Optional[int] = None,
        metric_config: Optional[
            Union[AblationMetricConfig, Mapping[str, Any]]
        ] = None,
        experiment_fields: Optional[Mapping[str, Any]] = None,
    ) -> None:
        section = _first_config_value(
            cfg,
            (
                "architecture_ablation",
                "learning.architecture_ablation",
                "learning.subgraphs.architecture_ablation",
            ),
            default={},
        )
        if not isinstance(section, Mapping):
            try:
                section = dict(section)
            except (TypeError, ValueError):
                section = {}

        section_metrics = section.get(
            "metrics", section.get("metric_config", section)
        )
        self.metric_config = AblationMetricConfig.from_value(
            metric_config if metric_config is not None else section_metrics
        )

        inferred_dataset = _first_config_value(cfg, ("dataset.name", "dataset"))
        inferred_architecture = _first_config_value(
            cfg, ("model.name", "architecture", "model._target_")
        )
        if isinstance(inferred_architecture, str) and "." in inferred_architecture:
            inferred_architecture = inferred_architecture.rsplit(".", 1)[-1]
        inferred_method = _first_config_value(
            cfg,
            (
                "architecture_ablation.method",
                "learning.architecture_ablation.method",
                "learning.subgraphs.architecture_ablation.method",
                "learning.subgraphs.architecture_update_strategy",
                "learning.subgraphs.architecture_update_init_strategy",
                "learning.subgraphs.initialization_strategy",
                "learning.subgraphs.parameter_transfer_strategy",
            ),
            default=section.get("method"),
        )
        inferred_split_group = _first_config_value(
            cfg,
            (
                "architecture_ablation.split_group",
                "learning.architecture_ablation.split_group",
                "learning.subgraphs.architecture_ablation.split_group",
                "learning.subgraphs.split_group",
            ),
            default=section.get("split_group"),
        )
        predrift_mode = _config_select(
            cfg, "learning.subgraphs.client_selection_mode_predrift"
        )
        if inferred_split_group is None:
            inferred_split_group = {
                "first_no_add": "10_50_to_100",
                "all_no_add": "51_75_to_100",
            }.get(predrift_mode, predrift_mode)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.common: Dict[str, Any] = {
            "dataset": dataset if dataset is not None else inferred_dataset,
            "architecture": architecture
            if architecture is not None
            else inferred_architecture,
            "split_group": split_group
            if split_group is not None
            else inferred_split_group,
            "method": method if method is not None else inferred_method,
            "seed": seed if seed is not None else _config_select(cfg, "seed"),
            "drift_round": drift_round
            if drift_round is not None
            else _config_select(cfg, "learning.subgraphs.rnd_drift"),
        }
        if self.common["drift_round"] is None:
            raise ValueError("drift_round must be supplied explicitly or through cfg.")

        self._resolved_cfg = _resolved_config(cfg)
        self._experiment_fields = dict(experiment_fields or {})
        self._client_rows_by_round: Dict[float, List[Dict[str, Any]]] = {}
        self._summary_by_round: Dict[float, Dict[str, Any]] = {}
        self._shift_snapshot_rows: Dict[str, List[Dict[str, Any]]] = {}
        self._shift_snapshot_summaries: Dict[str, Dict[str, Any]] = {}
        self._structural_metadata: Dict[str, Any] = {}
        self._write_experiment_files()
        self._persist_validation()
        self._persist_shift_snapshots()

    def _write_experiment_files(self) -> None:
        _atomic_json(self.output_dir / "config.json", self._resolved_cfg)
        experiment = {
            "schema_version": 1,
            **self.common,
            "metric_config": asdict(self.metric_config),
            **self._experiment_fields,
        }
        _atomic_json(self.output_dir / "experiment.json", experiment)

    @staticmethod
    def _normalize_client_rows(client_rows: Any) -> List[Dict[str, Any]]:
        if isinstance(client_rows, Mapping):
            if "client_id" in client_rows:
                inputs: List[Any] = [client_rows]
            else:
                inputs = []
                for client_id, value in client_rows.items():
                    if isinstance(value, Mapping):
                        inputs.append({"client_id": client_id, **dict(value)})
                    else:
                        inputs.append({"client_id": client_id, "task_loss": value})
        else:
            inputs = list(client_rows)

        normalized: List[Dict[str, Any]] = []
        for item in inputs:
            if isinstance(item, Mapping):
                row = dict(item)
            elif isinstance(item, (tuple, list)) and len(item) in (2, 3):
                row = {"client_id": item[0], "task_loss": item[1]}
                if len(item) == 3:
                    row["n_labeled_samples"] = item[2]
            else:
                raise TypeError(
                    "Each client row must be a mapping or (client_id, loss[, n_labeled]) tuple."
                )
            if "client_id" not in row:
                raise ValueError(f"Client row is missing client_id: {row!r}.")

            loss = None
            for key in ("task_loss", "validation_task_loss", "val_task_loss", "loss"):
                if key in row:
                    loss = _as_float(row[key])
                    break
            n_labeled = None
            for key in (
                "n_labeled_samples",
                "n_labeled",
                "labeled_samples",
                "count",
            ):
                if key in row:
                    n_labeled = _count_or_length(row[key])
                    break
            eligible = bool(row.get("eligible", True))
            has_labels = n_labeled is None or float(n_labeled) > 0
            evaluated = eligible and has_labels and _is_finite(loss)
            normalized.append(
                {
                    "client_id": row["client_id"],
                    "task_loss": loss,
                    "n_labeled_samples": n_labeled,
                    "eligible": eligible,
                    "evaluated": evaluated,
                    "phase": row.get("phase"),
                    "reason": row.get("reason"),
                }
            )
        return normalized

    def record_round(self, round_number: Union[int, float], client_rows: Any) -> Dict[str, Any]:
        """Record and immediately persist client task losses for one round.

        ``client_rows`` may be ``{client_id: loss}``, ``{client_id: {...}}``, or
        an iterable of row mappings/tuples.  Re-recording a round replaces it.
        """

        round_float = _as_float(round_number)
        if round_float is None or not math.isfinite(round_float):
            raise ValueError(f"Invalid round_number: {round_number!r}.")
        normalized = self._normalize_client_rows(client_rows)
        rows: List[Dict[str, Any]] = []
        for row in normalized:
            rows.append({**self.common, "round": round_number, **row})

        evaluated_losses = [
            float(row["task_loss"]) for row in rows if row["evaluated"]
        ]
        if evaluated_losses:
            mean_loss = float(np.mean(evaluated_losses))
            std_loss = float(np.std(evaluated_losses, ddof=0))
        else:
            mean_loss = float("nan")
            std_loss = float("nan")
        summary = {
            **self.common,
            "round": round_number,
            "mean_task_loss": mean_loss,
            "std_task_loss": std_loss,
            "n_evaluated_clients": len(evaluated_losses),
            "n_eligible_clients": sum(bool(row["eligible"]) for row in rows),
        }
        self._client_rows_by_round[round_float] = rows
        self._summary_by_round[round_float] = summary
        self._persist_validation()
        return dict(summary)

    def _persist_validation(self) -> None:
        client_rows = [
            row
            for round_key in sorted(self._client_rows_by_round)
            for row in self._client_rows_by_round[round_key]
        ]
        summary_rows = [
            self._summary_by_round[round_key]
            for round_key in sorted(self._summary_by_round)
        ]
        _atomic_csv(
            self.output_dir / "validation_client_losses.csv",
            self.CLIENT_FIELDS,
            client_rows,
        )
        _atomic_csv(
            self.output_dir / "validation_round_summary.csv",
            self.ROUND_FIELDS,
            summary_rows,
        )

    def record_shift_snapshot(
        self,
        stage: str,
        round_number: Union[int, float],
        client_rows: Any,
    ) -> Dict[str, Any]:
        """Persist an auxiliary loss snapshot at the structural update.

        The primary round curve remains the post-FedAvg validation trajectory.
        This snapshot measures the migrated models before local optimization,
        which isolates the immediate initialization discontinuity.
        """

        if not stage or not isinstance(stage, str):
            raise ValueError("stage must be a non-empty string.")
        round_float = _as_float(round_number)
        if round_float is None or not math.isfinite(round_float):
            raise ValueError(f"Invalid round_number: {round_number!r}.")

        normalized = self._normalize_client_rows(client_rows)
        rows = [
            {
                **self.common,
                "stage": stage,
                "round": round_number,
                **row,
            }
            for row in normalized
        ]
        evaluated_losses = [
            float(row["task_loss"]) for row in rows if row["evaluated"]
        ]
        summary = {
            **self.common,
            "stage": stage,
            "round": round_number,
            "mean_task_loss": float(np.mean(evaluated_losses))
            if evaluated_losses
            else float("nan"),
            "std_task_loss": float(np.std(evaluated_losses, ddof=0))
            if evaluated_losses
            else float("nan"),
            "n_evaluated_clients": len(evaluated_losses),
            "n_eligible_clients": sum(bool(row["eligible"]) for row in rows),
        }
        self._shift_snapshot_rows[stage] = rows
        self._shift_snapshot_summaries[stage] = summary
        self._persist_shift_snapshots()
        return dict(summary)

    def _persist_shift_snapshots(self) -> None:
        rows = [
            row
            for stage in sorted(self._shift_snapshot_rows)
            for row in self._shift_snapshot_rows[stage]
        ]
        summaries = [
            self._shift_snapshot_summaries[stage]
            for stage in sorted(self._shift_snapshot_summaries)
        ]
        _atomic_csv(
            self.output_dir / "shift_snapshot_client_losses.csv",
            self.SHIFT_SNAPSHOT_CLIENT_FIELDS,
            rows,
        )
        _atomic_csv(
            self.output_dir / "shift_snapshot_summary.csv",
            self.SHIFT_SNAPSHOT_SUMMARY_FIELDS,
            summaries,
        )

    def set_structural_metadata(self, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        """Merge and persist structural/migration metadata immediately."""

        if not isinstance(metadata, Mapping):
            raise TypeError("structural metadata must be a mapping.")
        self._structural_metadata.update(dict(metadata))
        canonical = canonical_structural_metrics(self._structural_metadata)
        payload = {
            "metadata": self._structural_metadata,
            "seed_metric_fields": canonical,
        }
        _atomic_json(self.output_dir / "structural_change.json", payload)
        return canonical

    def finalize(self) -> Dict[str, Any]:
        """Compute and save the final per-seed metric row, then return it."""

        ordered = [
            self._summary_by_round[key] for key in sorted(self._summary_by_round)
        ]
        curve_metrics = compute_ablation_metrics(
            [row["round"] for row in ordered],
            [row["mean_task_loss"] for row in ordered],
            shift_round=self.common["drift_round"],
            metric_config=self.metric_config,
        )
        structural_metrics = canonical_structural_metrics(self._structural_metadata)
        seed_metrics: Dict[str, Any] = {
            **self.common,
            "n_rounds_recorded": len(ordered),
            "first_recorded_round": ordered[0]["round"] if ordered else None,
            "last_recorded_round": ordered[-1]["round"] if ordered else None,
            **curve_metrics,
            **structural_metrics,
        }
        immediate = self._shift_snapshot_summaries.get(
            "post_migration_pre_local_training"
        )
        if immediate is not None:
            immediate_loss = float(immediate["mean_task_loss"])
            seed_metrics.update(
                {
                    "immediate_post_migration_loss": immediate_loss,
                    "immediate_post_migration_client_std": float(
                        immediate["std_task_loss"]
                    ),
                    "immediate_post_migration_client_count": int(
                        immediate["n_evaluated_clients"]
                    ),
                    "immediate_loss_spike": immediate_loss
                    - float(curve_metrics["pre_shift_loss"])
                    if math.isfinite(immediate_loss)
                    and math.isfinite(float(curve_metrics["pre_shift_loss"]))
                    else float("nan"),
                }
            )
        _atomic_json(self.output_dir / "seed_metrics.json", seed_metrics)
        _atomic_csv(
            self.output_dir / "seed_metrics.csv",
            tuple(seed_metrics.keys()),
            [seed_metrics],
        )
        # Re-persist in case the caller populated metadata before finalize but
        # intentionally skipped set_structural_metadata (empty is not written).
        if self._structural_metadata:
            self.set_structural_metadata(self._structural_metadata)
        return seed_metrics


__all__ = [
    "AblationMetricConfig",
    "ArchitectureAblationRecorder",
    "canonical_structural_metrics",
    "compute_ablation_metrics",
]
