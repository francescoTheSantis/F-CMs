#!/usr/bin/env python3
"""Post-hoc plots and tables for the architecture-initialization ablation.

The script only reads artifacts produced under ``results/architecture_ablation``;
it never imports the training pipeline and never requires a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ("full_reinit", "partial_random", "partial_zero")
METHOD_STYLES = {
    "full_reinit": {"label": "full_reinit", "color": "#d62728"},
    "partial_random": {"label": "partial_random", "color": "#1f77b4"},
    "partial_zero": {"label": "partial_zero", "color": "#2ca02c"},
}
IDENTITY_COLUMNS = (
    "dataset",
    "architecture",
    "split_group",
    "method",
    "seed",
)
PER_SEED_GROUP = ("dataset", "architecture", "split_group", "seed")
ACROSS_SEED_GROUP = ("dataset", "architecture", "split_group")
ROUND_ALIASES = {
    "client_mean_loss": ("mean_task_loss", "client_mean_loss", "mean_validation_task_loss"),
    "client_std_loss": ("std_task_loss", "client_std_loss", "std_validation_task_loss"),
    "n_clients": ("n_evaluated_clients", "n_clients", "client_count"),
}
PREFERRED_METRICS = (
    "pre_shift_loss",
    "immediate_post_migration_loss",
    "immediate_loss_spike",
    "immediate_post_migration_client_std",
    "immediate_post_migration_client_count",
    "post_shift_peak_loss",
    "loss_spike",
    "post_shift_peak_round",
    "minimum_post_shift_loss",
    "minimum_post_shift_loss_round",
    "recovered",
    "recovery_round",
    "recovery_time",
    "cumulative_excess_loss",
    "final_window_loss",
    "final_delta",
    "parameters_preserved",
    "parameters_preserved_fraction",
    "parameters_newly_initialized",
    "parameters_newly_initialized_fraction",
    "parameters_fully_reinitialized",
    "parameters_fully_reinitialized_fraction",
    "number_added_concepts",
    "number_affected_modules",
    "number_changed_tensors",
    "realized_pre_concept_coverage",
    "realized_post_concept_coverage",
    "function_preserving",
    "fallback_count",
)
REVIEWER_LOSS_METRICS = (
    "pre_shift_loss",
    "immediate_post_migration_loss",
    "immediate_loss_spike",
    "post_shift_peak_loss",
    "loss_spike",
    "post_shift_peak_round",
    "minimum_post_shift_loss",
    "minimum_post_shift_loss_round",
    "recovery_time",
    "cumulative_excess_loss",
    "final_window_loss",
    "final_delta",
)
REVIEWER_STRUCTURAL_METRICS = (
    "parameters_total",
    "parameters_preserved",
    "parameters_preserved_fraction",
    "parameters_newly_initialized",
    "parameters_newly_initialized_fraction",
    "parameters_fully_reinitialized",
    "parameters_fully_reinitialized_fraction",
    "number_added_concepts",
    "number_affected_modules",
    "number_changed_tensors",
    "realized_pre_concept_coverage",
    "realized_post_concept_coverage",
    "fallback_count",
)


def _warn(message: str) -> None:
    print(f"[architecture-ablation] WARNING: {message}", file=sys.stderr)


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "unknown"


def _method_sort_key(value: Any) -> Tuple[int, str]:
    text = str(value)
    try:
        return METHOD_ORDER.index(text), text
    except ValueError:
        return len(METHOD_ORDER), text


def discover_artifact_dirs(roots: Sequence[Path]) -> List[Path]:
    """Find unique ``results/architecture_ablation`` directories recursively."""

    found = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            _warn(f"input path does not exist: {root}")
            continue
        if root.is_file():
            root = root.parent
        if root.name == "architecture_ablation" and root.parent.name == "results":
            found.add(root)
        direct = root / "results" / "architecture_ablation"
        if direct.is_dir():
            found.add(direct.resolve())
        for candidate in root.rglob("architecture_ablation"):
            if candidate.is_dir() and candidate.parent.name == "results":
                found.add(candidate.resolve())
    return sorted(found)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _warn(f"cannot read {path}: {exc}")
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = mapping
        for component in path.split("."):
            if not isinstance(current, Mapping) or component not in current:
                current = None
                break
            current = current[component]
        if current is not None and not isinstance(current, Mapping):
            return current
    return None


def _metadata_from_experiment(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    predrift_mode = _nested(
        payload,
        "client_selection_mode_predrift",
        "learning.subgraphs.client_selection_mode_predrift",
        "config.learning.subgraphs.client_selection_mode_predrift",
    )
    split_group = _nested(payload, "split_group", "experiment.split_group")
    if split_group is None:
        split_group = {
            "first_no_add": "10_50_to_100",
            "all_no_add": "51_75_to_100",
        }.get(predrift_mode, predrift_mode)
    return {
        "dataset": _nested(payload, "dataset", "dataset.name", "experiment.dataset"),
        "architecture": _nested(
            payload, "architecture", "model", "model.name", "experiment.architecture"
        ),
        "split_group": split_group,
        "method": _nested(
            payload,
            "method",
            "experiment.method",
            "learning.subgraphs.architecture_update_strategy",
        ),
        "seed": _nested(payload, "seed", "experiment.seed"),
        "drift_round": _nested(
            payload,
            "drift_round",
            "shift_round",
            "learning.subgraphs.rnd_drift",
            "experiment.drift_round",
        ),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        _warn(f"cannot read {path}: {exc}")
        return pd.DataFrame()


def _fill_metadata(frame: pd.DataFrame, metadata: Mapping[str, Any], source: Path) -> pd.DataFrame:
    frame = frame.copy()
    aliases = {"architecture": ("model",), "method": ("strategy",)}
    for target, candidates in aliases.items():
        if target not in frame:
            for candidate in candidates:
                if candidate in frame:
                    frame[target] = frame[candidate]
                    break
    for key, value in metadata.items():
        if key not in frame:
            frame[key] = value
        elif value is not None:
            frame[key] = frame[key].where(frame[key].notna(), value)
    frame["artifact_dir"] = str(source)
    return frame


def _coalesce_column(frame: pd.DataFrame, target: str, candidates: Iterable[str]) -> None:
    available = [candidate for candidate in candidates if candidate in frame]
    if not available:
        return
    if target not in frame:
        frame[target] = frame[available[0]]
        available = available[1:]
    for candidate in available:
        frame[target] = frame[target].where(frame[target].notna(), frame[candidate])


def _normalize_round_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for target, candidates in ROUND_ALIASES.items():
        _coalesce_column(frame, target, candidates)
    for column in ("round", "client_mean_loss", "client_std_loss", "n_clients", "drift_round", "seed"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _summary_from_clients(client_frame: pd.DataFrame) -> pd.DataFrame:
    required = set(IDENTITY_COLUMNS) | {"round", "task_loss", "drift_round"}
    if client_frame.empty or not required.issubset(client_frame.columns):
        return pd.DataFrame()
    data = client_frame.copy()
    if "evaluated" in data:
        evaluated = data["evaluated"].astype(str).str.lower().isin(("true", "1", "yes"))
        data = data[evaluated]
    data["task_loss"] = pd.to_numeric(data["task_loss"], errors="coerce")
    data = data[np.isfinite(data["task_loss"])]
    group_columns = list(IDENTITY_COLUMNS) + ["drift_round", "round"]
    rows = []
    for key, group in data.groupby(group_columns, dropna=False, sort=False):
        values = group["task_loss"].to_numpy(dtype=float)
        row = dict(zip(group_columns, key))
        row.update(
            client_mean_loss=float(np.mean(values)),
            client_std_loss=float(np.std(values, ddof=0)),
            n_clients=int(values.size),
            artifact_dir=group["artifact_dir"].iloc[-1],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_artifacts(artifact_dirs: Sequence[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clients: List[pd.DataFrame] = []
    rounds: List[pd.DataFrame] = []
    metrics: List[pd.DataFrame] = []
    for artifact_dir in artifact_dirs:
        experiment_path = artifact_dir / "experiment.json"
        metadata = _metadata_from_experiment(experiment_path)
        if not experiment_path.is_file():
            _warn(f"missing experiment.json in {artifact_dir}")
        client_path = artifact_dir / "validation_client_losses.csv"
        round_path = artifact_dir / "validation_round_summary.csv"
        metric_path = artifact_dir / "seed_metrics.csv"
        if client_path.is_file():
            clients.append(_fill_metadata(_read_csv(client_path), metadata, artifact_dir))
        if round_path.is_file():
            rounds.append(
                _normalize_round_summary(
                    _fill_metadata(_read_csv(round_path), metadata, artifact_dir)
                )
            )
        if metric_path.is_file():
            metrics.append(_fill_metadata(_read_csv(metric_path), metadata, artifact_dir))
        else:
            _warn(f"run is incomplete (no seed_metrics.csv): {artifact_dir}")

    client_frame = pd.concat(clients, ignore_index=True, sort=False) if clients else pd.DataFrame()
    round_frame = pd.concat(rounds, ignore_index=True, sort=False) if rounds else pd.DataFrame()
    metric_frame = pd.concat(metrics, ignore_index=True, sort=False) if metrics else pd.DataFrame()
    if round_frame.empty and not client_frame.empty:
        round_frame = _summary_from_clients(client_frame)

    for frame in (client_frame, round_frame, metric_frame):
        if "seed" in frame:
            frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce")
        if "drift_round" in frame:
            frame["drift_round"] = pd.to_numeric(frame["drift_round"], errors="coerce")

    if not round_frame.empty:
        subset = [column for column in (*IDENTITY_COLUMNS, "round") if column in round_frame]
        duplicates = round_frame.duplicated(subset=subset, keep=False)
        if duplicates.any():
            conflicting = sorted(
                round_frame.loc[duplicates, "artifact_dir"].astype(str).unique()
            )
            raise ValueError(
                "Duplicate run identities were discovered across artifact "
                "directories. Narrow the input paths to one sweep per identity: "
                f"{conflicting}"
            )
        round_frame = round_frame.drop_duplicates(subset=subset, keep="last")
    if not metric_frame.empty:
        subset = [column for column in IDENTITY_COLUMNS if column in metric_frame]
        duplicates = metric_frame.duplicated(subset=subset, keep=False)
        if duplicates.any():
            conflicting = sorted(
                metric_frame.loc[duplicates, "artifact_dir"].astype(str).unique()
            )
            raise ValueError(
                "Duplicate completed run identities were discovered. Narrow the "
                f"input paths to one sweep per identity: {conflicting}"
            )
        metric_frame = metric_frame.drop_duplicates(subset=subset, keep="last")
    return client_frame, round_frame, metric_frame


def _validate_round_frame(frame: pd.DataFrame) -> None:
    required = set(IDENTITY_COLUMNS) | {
        "drift_round",
        "round",
        "client_mean_loss",
        "client_std_loss",
        "n_clients",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"round-summary artifacts are missing columns: {missing}")
    missing_meta = frame[list(IDENTITY_COLUMNS)].isna().any(axis=1)
    if missing_meta.any():
        dirs = sorted(frame.loc[missing_meta, "artifact_dir"].astype(str).unique())
        raise ValueError(f"missing experiment identity metadata in: {dirs}")


def audit_controlled_design(
    clients: pd.DataFrame,
    rounds: pd.DataFrame,
    *,
    tolerance: float,
    metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Check that each comparison is complete and identical before the shift."""

    expected_methods = set(METHOD_ORDER)
    rows: List[Dict[str, Any]] = []
    for key, round_group in rounds.groupby(
        list(PER_SEED_GROUP), dropna=False, sort=True
    ):
        dataset, architecture, split_group, seed = key
        shift = _drift_round(round_group)
        methods = set(round_group["method"].astype(str))
        completed_methods = set(methods)
        if metrics is not None:
            completed_methods = set()
            if not metrics.empty:
                metric_mask = np.ones(len(metrics), dtype=bool)
                for column, value in zip(PER_SEED_GROUP, key):
                    metric_mask &= (
                        metrics[column].astype(str).to_numpy() == str(value)
                    )
                completed_methods = set(
                    metrics.loc[metric_mask, "method"].astype(str)
                )
        reasons: List[str] = []
        if methods != expected_methods:
            reasons.append(
                "methods=" + ",".join(sorted(methods, key=_method_sort_key))
            )
        if completed_methods != expected_methods:
            reasons.append(
                "completed_metrics="
                + ",".join(sorted(completed_methods, key=_method_sort_key))
            )

        round_sets = {
            method: set(
                pd.to_numeric(
                    round_group.loc[
                        round_group["method"].astype(str) == method, "round"
                    ],
                    errors="coerce",
                ).dropna()
            )
            for method in methods
        }
        same_round_grid = bool(round_sets) and len(
            {tuple(sorted(values)) for values in round_sets.values()}
        ) == 1
        if not same_round_grid:
            reasons.append("round_grid_mismatch")

        pre_rounds = round_group[round_group["round"] < shift].copy()
        mean_pivot = pre_rounds.pivot_table(
            index="round",
            columns="method",
            values="client_mean_loss",
            aggfunc="last",
        )
        mean_pivot = mean_pivot.reindex(columns=METHOD_ORDER)
        complete_mean = mean_pivot.dropna()
        max_mean_difference = (
            float((complete_mean.max(axis=1) - complete_mean.min(axis=1)).max())
            if not complete_mean.empty
            else float("nan")
        )

        group_clients = pd.DataFrame()
        if not clients.empty:
            mask = np.ones(len(clients), dtype=bool)
            for column, value in zip(PER_SEED_GROUP, key):
                mask &= clients[column].astype(str).to_numpy() == str(value)
            group_clients = clients.loc[mask].copy()
        if not group_clients.empty:
            group_clients["round"] = pd.to_numeric(
                group_clients["round"], errors="coerce"
            )
            group_clients["task_loss"] = pd.to_numeric(
                group_clients["task_loss"], errors="coerce"
            )
            group_clients["client_id"] = group_clients["client_id"].astype(str)
            pre_clients = group_clients[group_clients["round"] < shift]
            client_grids = {
                method: set(
                    zip(
                        method_group["round"],
                        method_group["client_id"],
                    )
                )
                for method, method_group in pre_clients.groupby(
                    pre_clients["method"].astype(str)
                )
            }
            same_pre_client_grid = bool(client_grids) and len(
                {tuple(sorted(values)) for values in client_grids.values()}
            ) == 1
            client_pivot = pre_clients.pivot_table(
                index=["round", "client_id"],
                columns="method",
                values="task_loss",
                aggfunc="last",
            ).reindex(columns=METHOD_ORDER)
            complete_clients = client_pivot.dropna()
            max_client_difference = (
                float(
                    (
                        complete_clients.max(axis=1)
                        - complete_clients.min(axis=1)
                    ).max()
                )
                if not complete_clients.empty
                else float("nan")
            )
        else:
            same_pre_client_grid = False
            max_client_difference = float("nan")
            reasons.append("raw_client_rows_unavailable")

        if methods == expected_methods:
            if not same_pre_client_grid:
                reasons.append("pre_shift_client_grid_mismatch")
            if (
                not math.isfinite(max_mean_difference)
                or max_mean_difference > tolerance
            ):
                reasons.append("pre_shift_mean_mismatch")
            if (
                not math.isfinite(max_client_difference)
                or max_client_difference > tolerance
            ):
                reasons.append("pre_shift_client_loss_mismatch")

        if methods != expected_methods or completed_methods != expected_methods:
            status = "incomplete"
        elif reasons:
            status = "fail"
        else:
            status = "pass"
        rows.append(
            {
                "dataset": dataset,
                "architecture": architecture,
                "split_group": split_group,
                "seed": seed,
                "drift_round": shift,
                "status": status,
                "methods_present": ",".join(
                    sorted(methods, key=_method_sort_key)
                ),
                "completed_metric_methods": ",".join(
                    sorted(completed_methods, key=_method_sort_key)
                ),
                "matching_round_grid": same_round_grid,
                "matching_pre_shift_client_grid": same_pre_client_grid,
                "max_abs_pre_shift_client_mean_difference": max_mean_difference,
                "max_abs_pre_shift_client_loss_difference": max_client_difference,
                "tolerance": tolerance,
                "reasons": ";".join(dict.fromkeys(reasons)),
            }
        )
    return pd.DataFrame(rows)


def _markdown_value(value: Any) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return f"{float(value):.4f}"
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No completed metric rows were found._\n"
    columns = list(frame.columns)
    header = "| " + " | ".join(str(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join((header, separator, *rows)) + "\n"


def _write_table(frame: pd.DataFrame, csv_path: Path, markdown_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(f"# {title}\n\n{_markdown_table(frame)}", encoding="utf-8")


def _ordered_metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    excluded = {"artifact_dir", "dataset", "architecture", "split_group", "seed", "drift_round"}
    preferred = [column for column in PREFERRED_METRICS if column in frame]
    remaining = [
        column for column in frame.columns if column not in excluded | {"method", *preferred}
    ]
    output = frame[["method", *preferred, *remaining]].copy()
    output["__order"] = output["method"].map(_method_sort_key)
    return output.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def _method_sequence(values: Iterable[Any]) -> List[str]:
    return sorted({str(value) for value in values}, key=_method_sort_key)


def _drift_round(group: pd.DataFrame) -> float:
    values = pd.to_numeric(group["drift_round"], errors="coerce").dropna().unique()
    if len(values) != 1:
        _warn(f"expected one drift round, found {values.tolist()}; using the first")
    return float(values[0]) if len(values) else float("nan")


def _save_figure(fig: plt.Figure, base_path: Path, formats: Sequence[str], dpi: int) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"bbox_inches": "tight"}
        if extension.lower() in ("png", "jpg", "jpeg"):
            kwargs["dpi"] = dpi
        fig.savefig(base_path.with_suffix(f".{extension.lower()}"), **kwargs)
    plt.close(fig)


def plot_per_seed(
    rounds: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> int:
    count = 0
    for key, group in rounds.groupby(list(PER_SEED_GROUP), dropna=False, sort=True):
        dataset, architecture, split_group, seed = key
        group = group.sort_values(["method", "round"])
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for method in _method_sequence(group["method"]):
            method_group = group[group["method"].astype(str) == method].sort_values("round")
            x = method_group["round"].to_numpy(dtype=float)
            mean = method_group["client_mean_loss"].to_numpy(dtype=float)
            std = method_group["client_std_loss"].fillna(0.0).to_numpy(dtype=float)
            style = METHOD_STYLES.get(method, {"label": method, "color": None})
            line = ax.plot(x, mean, linewidth=2.0, label=style["label"], color=style["color"])[0]
            ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.18)
        shift = _drift_round(group)
        if math.isfinite(shift):
            ax.axvline(shift, color="black", linestyle="--", linewidth=1.4, label="structural shift")
        ax.set_xlabel("Federated round")
        ax.set_ylabel("Client-mean validation task loss")
        ax.set_title(f"{dataset} · {architecture} · {split_group} · seed {int(seed)}")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()

        group_dir = output_dir / "per_seed" / _slug(dataset) / _slug(architecture) / _slug(split_group)
        base = group_dir / f"seed_{int(seed)}"
        _save_figure(fig, base, formats, dpi)

        if metrics.empty:
            metric_table = pd.DataFrame()
        else:
            mask = np.ones(len(metrics), dtype=bool)
            for column, value in zip(PER_SEED_GROUP, key):
                mask &= metrics[column].astype(str).to_numpy() == str(value)
            metric_table = _ordered_metric_table(metrics.loc[mask])
        _write_table(
            metric_table,
            group_dir / f"seed_{int(seed)}_metrics.csv",
            group_dir / f"seed_{int(seed)}_metrics.md",
            f"Metrics: {dataset}, {architecture}, {split_group}, seed {int(seed)}",
        )
        count += 1
    return count


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=[*ACROSS_SEED_GROUP, "method", "n_seeds"])
    group_columns = [*ACROSS_SEED_GROUP, "method"]
    ignored = set(group_columns) | {
        "seed",
        "artifact_dir",
        "drift_round",
        "shift_round",
        "pre_shift_window",
        "post_peak_window",
        "smoothing_window",
        "stability_window",
        "final_window",
        "post_shift_horizon",
    }
    numeric_columns = []
    numeric_data: Dict[str, pd.Series] = {}
    for column in metrics.columns:
        if column in ignored:
            continue
        series = metrics[column]
        if pd.api.types.is_bool_dtype(series):
            numeric = series.astype(float)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            numeric_columns.append(column)
            numeric_data[column] = numeric

    working = metrics.copy()
    for column, series in numeric_data.items():
        working[column] = series
    rows: List[Dict[str, Any]] = []
    for key, group in working.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, key))
        row["n_seeds"] = int(group["seed"].nunique())
        drift_values = pd.to_numeric(group.get("drift_round"), errors="coerce").dropna().unique()
        row["drift_round"] = drift_values[0] if len(drift_values) == 1 else np.nan
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{column}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else (0.0 if values.size else np.nan)
            )
            row[f"{column}_n"] = int(values.size)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _formatted_summary_stat(
    row: pd.Series,
    metric: str,
    *,
    percentage: bool = False,
    show_available_n: bool = False,
) -> str:
    mean_key, std_key, n_key = (
        f"{metric}_mean",
        f"{metric}_std",
        f"{metric}_n",
    )
    if mean_key not in row.index or pd.isna(row[mean_key]):
        return "—"
    mean = float(row[mean_key])
    std = float(row.get(std_key, np.nan))
    scale = 100.0 if percentage else 1.0
    suffix = "%" if percentage else ""
    value = f"{scale * mean:.4f} ± {scale * std:.4f}{suffix}"
    if show_available_n:
        available = int(row.get(n_key, 0))
        total = int(row["n_seeds"])
        value += f" (n={available}/{total})"
    return value


def _reviewer_loss_table(summary_group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_group.sort_values(
        "method", key=lambda series: series.map(_method_sort_key)
    ).iterrows():
        output: Dict[str, Any] = {"method": row["method"], "seeds": int(row["n_seeds"])}
        recovered_mean = row.get("recovered_mean", np.nan)
        recovered_n = int(row.get("recovered_n", row["n_seeds"]))
        if pd.isna(recovered_mean):
            output["recovery_rate"] = "—"
        else:
            recovered_count = int(round(float(recovered_mean) * recovered_n))
            output["recovery_rate"] = (
                f"{100.0 * float(recovered_mean):.1f}% "
                f"({recovered_count}/{recovered_n})"
            )
        for metric in REVIEWER_LOSS_METRICS:
            if f"{metric}_mean" not in row.index:
                continue
            output[metric] = _formatted_summary_stat(
                row,
                metric,
                show_available_n=(metric == "recovery_time"),
            )
        rows.append(output)
    return pd.DataFrame(rows)


def _reviewer_structural_table(summary_group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_group.sort_values(
        "method", key=lambda series: series.map(_method_sort_key)
    ).iterrows():
        output: Dict[str, Any] = {"method": row["method"], "seeds": int(row["n_seeds"])}
        preserving_mean = row.get("function_preserving_mean", np.nan)
        preserving_n = int(row.get("function_preserving_n", row["n_seeds"]))
        if not pd.isna(preserving_mean):
            preserving_count = int(round(float(preserving_mean) * preserving_n))
            output["function_preserving_rate"] = (
                f"{100.0 * float(preserving_mean):.1f}% "
                f"({preserving_count}/{preserving_n})"
            )
        for metric in REVIEWER_STRUCTURAL_METRICS:
            if f"{metric}_mean" not in row.index:
                continue
            output[metric] = _formatted_summary_stat(
                row,
                metric,
                percentage=(metric.endswith("_fraction") or metric.endswith("_coverage")),
            )
        rows.append(output)
    return pd.DataFrame(rows)


def write_reviewer_tables(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Architecture-update initialization ablation",
        "",
        "Values are mean ± sample standard deviation across seed-level metrics. "
        "Fractions and realized coverage are percentages. Recovery time is conditional "
        "on recovery within the measured horizon and therefore includes its available "
        "seed count; recovery rate is reported separately. A dash means unavailable.",
        "",
    ]
    for key, group in summary.groupby(list(ACROSS_SEED_GROUP), dropna=False, sort=True):
        dataset, architecture, split_group = key
        lines.extend(
            [
                f"## {dataset} · {architecture} · {split_group}",
                "",
                "### Loss dynamics",
                "",
                _markdown_table(_reviewer_loss_table(group)).rstrip(),
                "",
                "### Structural accounting",
                "",
                _markdown_table(_reviewer_structural_table(group)).rstrip(),
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_across_seed(
    rounds: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> int:
    count = 0
    for key, group in rounds.groupby(list(ACROSS_SEED_GROUP), dropna=False, sort=True):
        dataset, architecture, split_group = key
        seed_level = (
            group.groupby(["method", "seed", "round"], as_index=False, dropna=False)[
                "client_mean_loss"
            ]
            .mean()
            .sort_values(["method", "seed", "round"])
        )
        curve_rows = []
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for method in _method_sequence(seed_level["method"]):
            method_group = seed_level[seed_level["method"].astype(str) == method]
            grouped = method_group.groupby("round", sort=True)["client_mean_loss"]
            curve = grouped.agg(["mean", "std", "count"]).reset_index()
            curve["std"] = curve["std"].fillna(0.0)
            curve.insert(0, "method", method)
            curve = curve.rename(
                columns={"mean": "seed_mean_loss", "std": "seed_std_loss", "count": "n_seeds"}
            )
            curve_rows.append(curve)
            x = curve["round"].to_numpy(dtype=float)
            mean = curve["seed_mean_loss"].to_numpy(dtype=float)
            std = curve["seed_std_loss"].to_numpy(dtype=float)
            style = METHOD_STYLES.get(method, {"label": method, "color": None})
            line = ax.plot(x, mean, linewidth=2.0, label=style["label"], color=style["color"])[0]
            ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.18)
        shift = _drift_round(group)
        if math.isfinite(shift):
            ax.axvline(shift, color="black", linestyle="--", linewidth=1.4, label="structural shift")
        ax.set_xlabel("Federated round")
        ax.set_ylabel("Mean seed-level client-mean task loss")
        ax.set_title(f"{dataset} · {architecture} · {split_group}\nshading: ±1 SD across seeds")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()

        group_dir = output_dir / "across_seed" / _slug(dataset) / _slug(architecture) / _slug(split_group)
        base = group_dir / "mean_across_seeds"
        _save_figure(fig, base, formats, dpi)
        pd.concat(curve_rows, ignore_index=True).to_csv(group_dir / "mean_across_seeds_curve.csv", index=False)

        if summary_metrics.empty:
            metric_table = pd.DataFrame()
        else:
            mask = np.ones(len(summary_metrics), dtype=bool)
            for column, value in zip(ACROSS_SEED_GROUP, key):
                mask &= summary_metrics[column].astype(str).to_numpy() == str(value)
            metric_table = summary_metrics.loc[mask].copy()
            metric_table["__order"] = metric_table["method"].map(_method_sort_key)
            metric_table = metric_table.sort_values("__order").drop(columns="__order")
        _write_table(
            metric_table,
            group_dir / "mean_across_seeds_metrics.csv",
            group_dir / "mean_across_seeds_metrics.md",
            f"Across-seed metrics: {dataset}, {architecture}, {split_group}",
        )
        count += 1
    return count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate architecture-initialization ablation plots and tables from saved artifacts."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Hydra sweep/run directories (or architecture_ablation directories) to scan recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("architecture_ablation_report"),
        help="Report directory (default: architecture_ablation_report).",
    )
    parser.add_argument(
        "--across-seed",
        action="store_true",
        help="Also plot the mean across seed-level client means, shaded by ±1 sample SD across seeds.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated figure formats (default: png,pdf).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster formats (default: 300).")
    parser.add_argument(
        "--pre-shift-tolerance",
        type=float,
        default=1e-8,
        help=(
            "Absolute tolerance for the controlled-design pre-shift equality "
            "audit (default: 1e-8)."
        ),
    )
    parser.add_argument(
        "--strict-design",
        action="store_true",
        help=(
            "Fail instead of only warning when a comparison is missing a method "
            "or violates the controlled-design checks."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    formats = tuple(part.strip().lower() for part in args.formats.split(",") if part.strip())
    if not formats:
        raise ValueError("--formats must contain at least one extension")
    artifact_dirs = discover_artifact_dirs(args.inputs)
    if not artifact_dirs:
        raise SystemExit("No results/architecture_ablation directories were found.")
    print(f"[architecture-ablation] found {len(artifact_dirs)} artifact directories")
    clients, rounds, metrics = load_artifacts(artifact_dirs)
    if rounds.empty:
        raise SystemExit("No validation round summaries were found.")
    _validate_round_frame(rounds)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    design_checks = audit_controlled_design(
        clients,
        rounds,
        tolerance=args.pre_shift_tolerance,
        metrics=metrics,
    )
    design_checks.to_csv(output_dir / "controlled_design_checks.csv", index=False)
    for row in design_checks.itertuples(index=False):
        if row.status != "pass":
            _warn(
                "controlled-design check "
                f"{row.status} for {row.dataset}/{row.architecture}/"
                f"{row.split_group}/seed={row.seed}: {row.reasons}"
            )
    if args.strict_design and (design_checks["status"] != "pass").any():
        raise SystemExit(
            "Controlled-design checks failed; see controlled_design_checks.csv."
        )
    per_seed_count = plot_per_seed(rounds, metrics, output_dir, formats, args.dpi)

    metrics_per_seed = metrics.copy()
    if not metrics_per_seed.empty:
        order = [column for column in (*IDENTITY_COLUMNS, "drift_round") if column in metrics_per_seed]
        rest = [column for column in metrics_per_seed if column not in {*order, "artifact_dir"}]
        metrics_per_seed = metrics_per_seed[[*order, *rest, "artifact_dir"]]
        metrics_per_seed = metrics_per_seed.sort_values(list(IDENTITY_COLUMNS)).reset_index(drop=True)
    metrics_per_seed.to_csv(output_dir / "metrics_per_seed.csv", index=False)

    metrics_summary = summarize_metrics(metrics)
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    write_reviewer_tables(metrics_summary, output_dir / "reviewer_tables.md")

    across_count = 0
    if args.across_seed:
        across_count = plot_across_seed(
            rounds, metrics_summary, output_dir, formats, args.dpi
        )
    print(
        f"[architecture-ablation] wrote {per_seed_count} per-seed figures and "
        f"{across_count} across-seed figures to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
