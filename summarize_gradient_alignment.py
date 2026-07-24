"""Aggregate gradient-alignment diagnostics across Hydra runs and seeds.

Example:
    python summarize_gradient_alignment.py \
        outputs/multirun/2026-07-24/12-00-00 \
        --output-dir rebuttal_alignment_summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


RUN_METRICS = (
    "zeta_squared_max_observed",
    "zeta_max_observed",
    "relative_l2_mean",
    "relative_l2_max",
    "relative_squared_l2_mean",
    "relative_squared_l2_max",
    "cosine_mean",
    "cosine_min",
    "angle_degrees_mean",
    "direction_distance_mean",
    "norm_ratio_mean",
    "scale_adjusted_relative_l2_mean",
)

TRAJECTORY_METRICS = (
    "squared_l2_distance",
    "relative_l2_distance",
    "cosine_similarity",
    "angle_degrees",
    "norm_ratio",
    "scale_adjusted_relative_l2",
    "reference_norm",
    "federated_norm",
)


def _discover_round_files(roots: Iterable[str]) -> List[Path]:
    files = set()
    for root_value in roots:
        root = Path(root_value).expanduser()
        if root.is_file() and root.name == "gradient_alignment_rounds.json":
            files.add(root.resolve())
        elif root.is_dir():
            files.update(
                path.resolve()
                for path in root.rglob("gradient_alignment_rounds.json")
            )
    return sorted(files)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _run_summary(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        records = json.load(handle)
    if not records:
        raise ValueError(f"No alignment records found in {path}")

    first = records[0]
    experiment = first.get("experiment", {})
    squared = [float(record["overall"]["squared_l2_distance"]) for record in records]
    relative = [float(record["overall"]["relative_l2_distance"]) for record in records]
    relative_squared = [
        float(record["overall"]["relative_squared_l2_distance"])
        for record in records
    ]
    cosine = [float(record["overall"]["cosine_similarity"]) for record in records]
    angle = [
        float(
            record["overall"].get(
                "angle_degrees",
                math.degrees(
                    math.acos(
                        max(
                            -1.0,
                            min(1.0, float(record["overall"]["cosine_similarity"])),
                        )
                    )
                ),
            )
        )
        for record in records
    ]
    direction = [float(record["overall"]["direction_distance"]) for record in records]
    norm_ratio = [
        float(
            record["overall"].get(
                "norm_ratio",
                float(record["overall"]["federated_norm"])
                / max(float(record["overall"]["reference_norm"]), 1e-12),
            )
        )
        for record in records
    ]
    scale_adjusted = [
        float(
            record["overall"].get(
                "scale_adjusted_relative_l2",
                math.sqrt(
                    max(
                        0.0,
                        1.0
                        - max(
                            0.0,
                            float(record["overall"]["cosine_similarity"]),
                        )
                        ** 2,
                    )
                ),
            )
        )
        for record in records
    ]
    max_squared = max(squared)

    return {
        "path": str(path),
        "dataset": str(experiment.get("dataset", "unknown")),
        "model": str(experiment.get("model", "unknown")),
        "seed": experiment.get("seed"),
        "n_rounds": len(records),
        "round_first": int(records[0]["round"]),
        "round_last": int(records[-1]["round"]),
        "zeta_squared_max_observed": max_squared,
        "zeta_max_observed": math.sqrt(max(0.0, max_squared)),
        "relative_l2_mean": _mean(relative),
        "relative_l2_max": max(relative),
        "relative_squared_l2_mean": _mean(relative_squared),
        "relative_squared_l2_max": max(relative_squared),
        "cosine_mean": _mean(cosine),
        "cosine_min": min(cosine),
        "angle_degrees_mean": _mean(angle),
        "direction_distance_mean": _mean(direction),
        "norm_ratio_mean": _mean(norm_ratio),
        "scale_adjusted_relative_l2_mean": _mean(scale_adjusted),
    }


def _mean_se(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= 1:
        standard_error = 0.0
    else:
        standard_error = float(np.std(array, ddof=1) / math.sqrt(len(array)))
    return {
        "mean": float(np.mean(array)),
        "standard_error": standard_error,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _derived_overall_metric(overall: Mapping[str, Any], metric: str) -> float:
    if metric in overall:
        return float(overall[metric])
    cosine = max(-1.0, min(1.0, float(overall["cosine_similarity"])))
    if metric == "angle_degrees":
        return math.degrees(math.acos(cosine))
    if metric == "norm_ratio":
        return float(overall["federated_norm"]) / max(
            float(overall["reference_norm"]),
            1e-12,
        )
    if metric == "scale_adjusted_relative_l2":
        return math.sqrt(max(0.0, 1.0 - max(0.0, cosine) ** 2))
    raise KeyError(metric)


def _trajectory_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], Dict[str, List[float]]] = {}
    seed_sets: Dict[Tuple[str, str, int], set] = {}
    for path in paths:
        with path.open() as handle:
            records = json.load(handle)
        if not records:
            continue
        experiment = records[0].get("experiment", {})
        dataset = str(experiment.get("dataset", "unknown"))
        model = str(experiment.get("model", "unknown"))
        seed = experiment.get("seed")
        for record in records:
            key = (dataset, model, int(record["round"]))
            metric_values = grouped.setdefault(
                key,
                {metric: [] for metric in TRAJECTORY_METRICS},
            )
            seed_sets.setdefault(key, set()).add(seed)
            for metric in TRAJECTORY_METRICS:
                metric_values[metric].append(
                    _derived_overall_metric(record["overall"], metric)
                )

    rows: List[Dict[str, Any]] = []
    for (dataset, model, round_index), metrics in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "round": round_index,
            "n_seeds": len(seed_sets[(dataset, model, round_index)]),
        }
        for metric, values in metrics.items():
            stats = _mean_se(values)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_se"] = stats["standard_error"]
        rows.append(row)
    return rows


def _group_summaries(run_summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for run in run_summaries:
        grouped.setdefault((str(run["dataset"]), str(run["model"])), []).append(run)

    output: List[Dict[str, Any]] = []
    for (dataset, model), runs in sorted(grouped.items()):
        entry: Dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "n_seeds": len(runs),
            "seeds": [run["seed"] for run in runs],
            "metrics": {},
        }
        for metric in RUN_METRICS:
            entry["metrics"][metric] = _mean_se(
                [float(run[metric]) for run in runs]
            )
        output.append(entry)
    return output


def _write_run_csv(run_summaries: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "model",
        "seed",
        "n_rounds",
        "round_first",
        "round_last",
        *RUN_METRICS,
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in run_summaries:
            writer.writerow({key: run.get(key) for key in fieldnames})


def _write_group_csv(groups: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = ["dataset", "model", "n_seeds"]
    for metric in RUN_METRICS:
        fieldnames.extend((f"{metric}_mean", f"{metric}_se"))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            row: Dict[str, Any] = {
                "dataset": group["dataset"],
                "model": group["model"],
                "n_seeds": group["n_seeds"],
            }
            for metric in RUN_METRICS:
                row[f"{metric}_mean"] = group["metrics"][metric]["mean"]
                row[f"{metric}_se"] = group["metrics"][metric]["standard_error"]
            writer.writerow(row)


def _write_trajectory_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = ["dataset", "model", "round", "n_seeds"]
    for metric in TRAJECTORY_METRICS:
        fieldnames.extend((f"{metric}_mean", f"{metric}_se"))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_trajectory_figure(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping trajectory figures.")
        return

    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["model"])), []).append(row)
    if not groups:
        return

    fig, axes = plt.subplots(
        len(groups),
        2,
        figsize=(9.0, 3.0 * len(groups)),
        squeeze=False,
    )
    for row_index, ((dataset, model), group_rows) in enumerate(sorted(groups.items())):
        group_rows = sorted(group_rows, key=lambda row: int(row["round"]))
        rounds = np.asarray([row["round"] for row in group_rows], dtype=float)

        zeta_mean = np.asarray(
            [row["squared_l2_distance_mean"] for row in group_rows],
            dtype=float,
        )
        zeta_se = np.asarray(
            [row["squared_l2_distance_se"] for row in group_rows],
            dtype=float,
        )
        axes[row_index, 0].plot(rounds, zeta_mean, color="#2166ac", linewidth=1.8)
        axes[row_index, 0].fill_between(
            rounds,
            np.maximum(0.0, zeta_mean - zeta_se),
            zeta_mean + zeta_se,
            color="#2166ac",
            alpha=0.2,
        )
        axes[row_index, 0].set_ylabel(r"$\widehat{\zeta}_t^2$")
        axes[row_index, 0].set_title(f"{dataset} / {model}")
        axes[row_index, 0].grid(alpha=0.25)

        cosine_mean = np.asarray(
            [row["cosine_similarity_mean"] for row in group_rows],
            dtype=float,
        )
        cosine_se = np.asarray(
            [row["cosine_similarity_se"] for row in group_rows],
            dtype=float,
        )
        axes[row_index, 1].plot(
            rounds,
            cosine_mean,
            color="#b2182b",
            linewidth=1.8,
        )
        axes[row_index, 1].fill_between(
            rounds,
            np.maximum(-1.0, cosine_mean - cosine_se),
            np.minimum(1.0, cosine_mean + cosine_se),
            color="#b2182b",
            alpha=0.2,
        )
        axes[row_index, 1].set_ylim(-0.05, 1.05)
        axes[row_index, 1].set_ylabel("cosine similarity")
        axes[row_index, 1].grid(alpha=0.25)
        for axis in axes[row_index]:
            axis.set_xlabel("communication round")

    fig.tight_layout()
    fig.savefig(output_dir / "gradient_alignment_trajectories.pdf")
    fig.savefig(output_dir / "gradient_alignment_trajectories.png", dpi=200)
    plt.close(fig)


def _latex_value(metric: Mapping[str, float], precision: int = 4) -> str:
    return (
        f"{metric['mean']:.{precision}f} "
        f"$\\pm$ {metric['standard_error']:.{precision}f}"
    )


def _write_latex(groups: Sequence[Mapping[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            r"Dataset & Model & $\max_t \widehat{\zeta}_t^2$ "
            r"& Relative L2 & Cosine & Min. cosine \\"
        ),
        r"\midrule",
    ]
    for group in groups:
        metrics = group["metrics"]
        dataset = str(group["dataset"]).replace("_", r"\_")
        model = str(group["model"]).replace("_", r"\_")
        lines.append(
            f"{dataset} & {model} & "
            f"{_latex_value(metrics['zeta_squared_max_observed'], 5)} & "
            f"{_latex_value(metrics['relative_l2_mean'], 4)} & "
            f"{_latex_value(metrics['cosine_mean'], 4)} & "
            f"{_latex_value(metrics['cosine_min'], 4)} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="+",
        help="Hydra output roots or individual gradient_alignment_rounds.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="rebuttal_alignment_summary",
        help="Directory for JSON, CSV, and LaTeX summaries.",
    )
    args = parser.parse_args()

    round_files = _discover_round_files(args.roots)
    if not round_files:
        raise FileNotFoundError(
            "No gradient_alignment_rounds.json files were found under the supplied roots."
        )

    run_summaries = [_run_summary(path) for path in round_files]
    groups = _group_summaries(run_summaries)
    trajectory_rows = _trajectory_rows(round_files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "qualification": (
            "All zeta values are maxima on measured broadcast iterates, not "
            "uniform upper bounds over the parameter space."
        ),
        "runs": run_summaries,
        "groups": groups,
    }
    (output_dir / "gradient_alignment_aggregate.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    _write_run_csv(run_summaries, output_dir / "gradient_alignment_runs.csv")
    _write_group_csv(groups, output_dir / "gradient_alignment_groups.csv")
    _write_trajectory_csv(
        trajectory_rows,
        output_dir / "gradient_alignment_trajectory.csv",
    )
    _write_latex(groups, output_dir / "gradient_alignment_table.tex")
    _write_trajectory_figure(trajectory_rows, output_dir)

    print(f"Aggregated {len(run_summaries)} runs into {output_dir}")
    for group in groups:
        metrics = group["metrics"]
        print(
            f"{group['dataset']} / {group['model']} ({group['n_seeds']} seeds): "
            f"relative L2={_latex_value(metrics['relative_l2_mean'])}, "
            f"cosine={_latex_value(metrics['cosine_mean'])}"
        )


if __name__ == "__main__":
    main()
