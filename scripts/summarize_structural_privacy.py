#!/usr/bin/env python3
"""Summarize structural-LDP Hydra runs into rebuttal-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


def nested_get(mapping: Dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def read_task_accuracy(run_dir: Path) -> Optional[float]:
    summary_path = run_dir / "results" / "aggregated_test_metrics.json"
    if summary_path.exists():
        with summary_path.open() as handle:
            summary = json.load(handle)
        value = summary.get("test/y/y_accuracy")
        return None if value is None else float(value)

    legacy_path = run_dir / "results" / "y_accuracy.pkl"
    if legacy_path.exists():
        with legacy_path.open("rb") as handle:
            value = pickle.load(handle)
        if isinstance(value, dict) and "_baseline" in value:
            return float(value["_baseline"])
    return None


def epsilon_sort_value(value: Any) -> float:
    if isinstance(value, str) and value.lower() in {"inf", ".inf", "infinity"}:
        return math.inf
    return float(value)


def discover_rows(roots: Iterable[Path], phase: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for root in roots:
        for report_path in root.rglob("results/structural_privacy.json"):
            run_dir = report_path.parent.parent
            resolved = run_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            config_path = run_dir / ".hydra" / "config.yaml"
            if not config_path.exists():
                print(f"Skipping {run_dir}: missing .hydra/config.yaml")
                continue
            with config_path.open() as handle:
                config = yaml.safe_load(handle)
            with report_path.open() as handle:
                report = json.load(handle)

            accuracy = read_task_accuracy(run_dir)
            disagreement = nested_get(
                report,
                f"aggregate_graph_disagreement.{phase}.disagreement_rate",
            )
            completion_status = (
                "complete" if accuracy is not None else "incomplete_missing_task_metrics"
            )
            rows.append(
                {
                    "run_dir": str(resolved),
                    "dataset": nested_get(config, "dataset.name", "unknown"),
                    "model": nested_get(config, "model.name", "unknown"),
                    "seed": int(config.get("seed", -1)),
                    "epsilon": report.get("epsilon"),
                    "task_accuracy": accuracy,
                    "graph_disagreement_rate": (
                        None if disagreement is None else float(disagreement)
                    ),
                    "changed_pair_report_rate": report.get("changed_pair_report_rate"),
                    "phase": phase,
                    "completion_status": completion_status,
                }
            )
    return rows


def mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None, None, 0
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return statistics.mean(clean), std, len(clean)


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    non_private_accuracy: Dict[Tuple[str, str, int], float] = {}
    for row in rows:
        if math.isinf(epsilon_sort_value(row["epsilon"])) and row["task_accuracy"] is not None:
            non_private_accuracy[(row["dataset"], row["model"], row["seed"])] = float(
                row["task_accuracy"]
            )

    grouped: Dict[Tuple[str, str, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["epsilon"])].append(row)

    summaries: List[Dict[str, Any]] = []
    for (dataset, model, epsilon), group in grouped.items():
        acc_mean, acc_std, n_accuracy = mean_std(row["task_accuracy"] for row in group)
        graph_mean, graph_std, n_graph = mean_std(
            row["graph_disagreement_rate"] for row in group
        )
        changed_mean, changed_std, _ = mean_std(
            row["changed_pair_report_rate"] for row in group
        )

        paired_changes = []
        for row in group:
            baseline = non_private_accuracy.get((dataset, model, row["seed"]))
            if baseline is not None and row["task_accuracy"] is not None:
                paired_changes.append(100.0 * (float(row["task_accuracy"]) - baseline))
        change_mean, change_std, n_paired = mean_std(paired_changes)

        summaries.append(
            {
                "dataset": dataset,
                "model": model,
                "epsilon": epsilon,
                "n_runs": len(group),
                "n_accuracy": n_accuracy,
                "task_accuracy_pct_mean": None if acc_mean is None else 100.0 * acc_mean,
                "task_accuracy_pct_std": None if acc_std is None else 100.0 * acc_std,
                "change_from_non_private_pp_mean": change_mean,
                "change_from_non_private_pp_std": change_std,
                "n_paired": n_paired,
                "graph_disagreement_pct_mean": (
                    None if graph_mean is None else 100.0 * graph_mean
                ),
                "graph_disagreement_pct_std": (
                    None if graph_std is None else 100.0 * graph_std
                ),
                "n_graph": n_graph,
                "changed_pair_report_pct_mean": (
                    None if changed_mean is None else 100.0 * changed_mean
                ),
                "changed_pair_report_pct_std": (
                    None if changed_std is None else 100.0 * changed_std
                ),
            }
        )

    return sorted(
        summaries,
        key=lambda row: (
            str(row["dataset"]).lower(),
            str(row["model"]).lower(),
            -epsilon_sort_value(row["epsilon"]),
        ),
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or std is None:
        return "–"
    return f"{mean:.2f} ± {std:.2f}"


def write_markdown(path: Path, rows: List[Dict[str, Any]], phase: str) -> None:
    lines = [
        f"# Structural privacy results ({phase})",
        "",
        "| Dataset | Model | $\\varepsilon_s$ | Complete task runs | Task accuracy (%) | Change vs. non-private (pp) | Graph disagreement (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        epsilon = row["epsilon"]
        epsilon_text = "$\\infty$" if math.isinf(epsilon_sort_value(epsilon)) else str(epsilon)
        lines.append(
            "| {dataset} | {model} | {epsilon} | {complete}/{total} | {accuracy} | {change} | {graph} |".format(
                dataset=row["dataset"],
                model=row["model"],
                epsilon=epsilon_text,
                complete=row["n_accuracy"],
                total=row["n_runs"],
                accuracy=format_mean_std(
                    row["task_accuracy_pct_mean"], row["task_accuracy_pct_std"]
                ),
                change=format_mean_std(
                    row["change_from_non_private_pp_mean"],
                    row["change_from_non_private_pp_std"],
                ),
                graph=format_mean_std(
                    row["graph_disagreement_pct_mean"],
                    row["graph_disagreement_pct_std"],
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Hydra run or multirun directories to scan recursively.",
    )
    parser.add_argument(
        "--phase",
        choices=("predrift", "postdrift"),
        default="postdrift",
        help="Aggregate graph phase used for disagreement (default: postdrift).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("structural_privacy_summary"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Write aggregate tables even when one or more discovered runs are "
            "missing final task metrics. By default this is treated as an error."
        ),
    )
    args = parser.parse_args()

    rows = discover_rows(args.roots, args.phase)
    if not rows:
        raise SystemExit("No completed structural-privacy runs were found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", rows)

    incomplete = [row for row in rows if row["completion_status"] != "complete"]
    if incomplete and not args.allow_incomplete:
        print("Incomplete structural-privacy runs (graph report exists, task metrics missing):")
        for row in incomplete:
            print(
                "  dataset={dataset} model={model} seed={seed} epsilon={epsilon}: "
                "{run_dir}".format(**row)
            )
        raise SystemExit(
            f"{len(incomplete)} incomplete run(s) found. Their graph aggregation "
            "completed, but training/final evaluation did not write "
            "results/aggregated_test_metrics.json. See runs.csv; fix or rerun "
            "them, or pass --allow-incomplete to produce a partial table."
        )

    summaries = summarize(rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_markdown(args.output_dir / "table.md", summaries, args.phase)
    print(f"Found {len(rows)} runs; wrote summaries to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
