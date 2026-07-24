#!/usr/bin/env python3
"""Run and summarize the concept-annotation sensitivity experiment.

Examples
--------
Run the complete five-seed grid and write paste-ready tables:

    python scripts/concept_noise_rebuttal.py run

Resume an interrupted grid:

    python scripts/concept_noise_rebuttal.py run --output-dir rebuttal_results/concept_noise

Regenerate tables without training:

    python scripts/concept_noise_rebuttal.py summarize \
        --output-dir rebuttal_results/concept_noise
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATES = (0.1, 0.3, 0.6, 0.9)
METRIC_NAMES = (
    "task_accuracy",
    "concept_accuracy",
    "intervention_label_accuracy",
    "concept_coverage",
)
DISPLAY_METRICS = (
    "task_accuracy",
    "concept_accuracy",
    "intervention_label_accuracy",
)
DATASET_SETTINGS = {
    "asia": {
        "epochs": 120,
        "patience": 10,
        "drift_round": 10,
        "n_add_node_subgraphs": 1,
    },
    "alarm": {
        "epochs": 200,
        "patience": 15,
        "drift_round": 20,
        "n_add_node_subgraphs": 3,
    },
}


def _probability_tag(probability: float) -> str:
    return f"{probability:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _run_directory(
    output_dir: Path,
    dataset: str,
    model: str,
    mode: str,
    probability: float,
    seed: int,
) -> Path:
    condition = "clean" if mode == "none" else f"{mode}_{_probability_tag(probability)}"
    return output_dir / "runs" / dataset / model / condition / f"seed_{seed}"


def _experiment_specs(
    datasets: Sequence[str],
    models: Sequence[str],
    seeds: Sequence[int],
    rates: Sequence[float],
) -> Iterable[Tuple[str, str, str, float, int]]:
    conditions = [("none", 0.0)]
    conditions.extend(("missing", probability) for probability in rates)
    conditions.extend(("noise", probability) for probability in rates)
    for dataset in datasets:
        for model in models:
            for mode, probability in conditions:
                for seed in seeds:
                    yield dataset, model, mode, probability, seed


def _required_artifacts(run_dir: Path) -> List[Path]:
    return [
        run_dir / "graph.pkl",
        run_dir / "results" / "y_accuracy.pkl",
        run_dir / "results" / "c_accuracy.pkl",
        run_dir / "results" / "cumulative_interventions_on_y.pkl",
        run_dir / "results" / "cumulative_interventions_on_c.pkl",
        run_dir / "results" / "additional_metrics.json",
        run_dir / "results" / "concept_annotation_perturbation.json",
    ]


def _is_complete(run_dir: Path) -> bool:
    return all(path.exists() for path in _required_artifacts(run_dir))


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backfill_run_fingerprints(run_dir: Path) -> None:
    spec_path = run_dir / "run_spec.json"
    if not spec_path.exists():
        return
    payload = json.loads(spec_path.read_text())
    graph_hash = _sha256(run_dir / "graph.pkl")
    cached_dataset_path = payload.get("cached_dataset")
    if not cached_dataset_path and {"dataset", "seed"} <= payload.keys():
        cached_dataset_path = str(
            _cached_dataset_path(str(payload["dataset"]), int(payload["seed"]))
        )
        payload["cached_dataset"] = cached_dataset_path
        changed = True
    else:
        changed = False
    dataset_hash = (
        _sha256(Path(cached_dataset_path)) if cached_dataset_path else None
    )
    for key, value in (
        ("graph_sha256", graph_hash),
        ("dataset_sha256", dataset_hash),
    ):
        if value is not None and payload.get(key) != value:
            payload[key] = value
            changed = True
    if changed:
        spec_path.write_text(json.dumps(payload, indent=2))


def _cached_dataset_path(dataset: str, seed: int) -> Path:
    cache_root = os.environ.get("FEDERATED_C2BM_CACHE")
    if cache_root:
        root = Path(cache_root).expanduser()
    else:
        xdg_cache = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        ).expanduser()
        root = xdg_cache / "federated_c2bm"
    return root / dataset / f"preprocessed_dataset_{seed}.pkl"


def _stream_process(command: Sequence[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return process.wait()


def run_grid(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = list(
        _experiment_specs(args.datasets, args.models, args.seeds, args.rates)
    )
    print(f"Experiment root: {output_dir}")
    print(f"Grid size: {len(specs)} runs")

    failures = []
    for run_index, (dataset, model, mode, probability, seed) in enumerate(specs, 1):
        run_dir = _run_directory(
            output_dir, dataset, model, mode, probability, seed
        )
        if _is_complete(run_dir) and not args.rerun:
            _backfill_run_fingerprints(run_dir)
            print(f"[{run_index}/{len(specs)}] complete, skipping: {run_dir}")
            continue

        settings = DATASET_SETTINGS[dataset]
        epochs = args.max_epochs or settings["epochs"]
        cached_dataset = _cached_dataset_path(dataset, seed)
        load_embeddings = cached_dataset.exists()
        command = [
            args.python,
            str(REPO_ROOT / "main.py"),
            "--config-name=concept_noise_rebuttal",
            f"dataset={dataset}",
            f"model={model}",
            "learning=local_federated",
            f"dataset.load_embeddings={str(load_embeddings).lower()}",
            f"seed={seed}",
            f"learning.concept_annotation_perturbation.mode={mode}",
            f"learning.concept_annotation_perturbation.probability={probability}",
            f"learning.subgraphs.rnd_drift={settings['drift_round']}",
            (
                "learning.subgraphs.dict_subgraph_with_add_nodes."
                f"number_subgraphs_add_nodes={settings['n_add_node_subgraphs']}"
            ),
            f"trainer.max_epochs={epochs}",
            f"trainer.patience={settings['patience']}",
            f"hydra.run.dir={run_dir}",
        ]
        run_dir.mkdir(parents=True, exist_ok=True)
        spec_payload = {
            "dataset": dataset,
            "model": model,
            "mode": mode,
            "probability": probability,
            "seed": seed,
            "epochs": epochs,
            "cached_dataset": str(cached_dataset),
            "loaded_cached_dataset": load_embeddings,
            "command": command,
            "status": "planned" if args.dry_run else "running",
        }
        spec_path = run_dir / "run_spec.json"
        spec_path.write_text(json.dumps(spec_payload, indent=2))

        print(
            f"[{run_index}/{len(specs)}] {dataset}/{model}/{mode}/"
            f"{probability:g}/seed={seed}"
        )
        if args.dry_run:
            print(" ".join(command))
            continue

        exit_code = _stream_process(
            command,
            cwd=REPO_ROOT,
            log_path=run_dir / "run.log",
        )
        spec_payload["exit_code"] = exit_code
        spec_payload["status"] = (
            "complete" if exit_code == 0 and _is_complete(run_dir) else "failed"
        )
        if spec_payload["status"] == "complete":
            spec_payload["graph_sha256"] = _sha256(run_dir / "graph.pkl")
            spec_payload["dataset_sha256"] = _sha256(cached_dataset)
        spec_path.write_text(json.dumps(spec_payload, indent=2))
        if spec_payload["status"] == "failed":
            failures.append(str(run_dir))
            print(f"Run failed or produced incomplete artifacts: {run_dir}")
            if args.fail_fast:
                raise SystemExit(exit_code or 1)

    if args.dry_run:
        return
    summarize(output_dir)
    if failures:
        print(f"Completed with {len(failures)} failed/incomplete runs:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean_finite(values: Iterable[object]) -> float:
    finite = [
        converted
        for value in values
        if (converted := _finite_float(value)) is not None
    ]
    return fmean(finite) if finite else float("nan")


def _parse_intervention_key(key: str) -> Tuple[int, str]:
    prefix, separator, concept_name = str(key).partition("_")
    if not separator or not prefix.isdigit():
        raise ValueError(f"Unexpected cumulative intervention key: {key!r}")
    return int(prefix), concept_name


def _intervention_label_accuracy(
    task_baseline: float,
    concept_baseline: Mapping[str, object],
    cumulative_task: Mapping[str, object],
    cumulative_concept: Mapping[str, object],
    evaluated_concepts: Sequence[str],
) -> Tuple[float, List[float]]:
    """Macro-average remaining predicted labels, including the task.

    Intervened concepts are corrections supplied by a user, so they are
    excluded from the remaining-label accuracy at and after their intervention.
    The reported scalar is the mean over the complete trajectory, baseline
    included.
    """

    baseline_values = [task_baseline]
    baseline_values.extend(concept_baseline.get(name) for name in evaluated_concepts)
    trajectory = [_mean_finite(baseline_values)]
    ordered_steps = sorted(
        (
            (_parse_intervention_key(key)[0], _parse_intervention_key(key)[1], key)
            for key in cumulative_task
        ),
        key=lambda item: item[0],
    )
    intervened = set()

    for _, concept_name, task_key in ordered_steps:
        intervened.add(concept_name)
        values = [cumulative_task[task_key]]
        for evaluated_concept in evaluated_concepts:
            if evaluated_concept in intervened:
                continue
            values.append(
                cumulative_concept.get(f"{task_key}/{evaluated_concept}")
            )
        trajectory.append(_mean_finite(values))

    return _mean_finite(trajectory), trajectory


def _extract_run_metrics(run_dir: Path) -> Dict[str, object]:
    spec = json.loads((run_dir / "run_spec.json").read_text())
    results_dir = run_dir / "results"
    task_artifact = _load_pickle(results_dir / "y_accuracy.pkl")
    concept_artifact = _load_pickle(results_dir / "c_accuracy.pkl")
    task_accuracy = float(task_artifact["_baseline"])
    cumulative_task = _load_pickle(
        results_dir / "cumulative_interventions_on_y.pkl"
    )
    cumulative_concept = _load_pickle(
        results_dir / "cumulative_interventions_on_c.pkl"
    )
    task_relevant = [
        _parse_intervention_key(key)[1] for key in cumulative_task
    ]
    evaluated_concepts = [
        name
        for name in task_relevant
        if _finite_float(concept_artifact.get(name)) is not None
    ]
    if not evaluated_concepts:
        evaluated_concepts = [
            name
            for name, value in concept_artifact.items()
            if _finite_float(value) is not None
        ]

    concept_accuracy = _mean_finite(
        concept_artifact.get(name) for name in evaluated_concepts
    )
    intervention_accuracy, trajectory = _intervention_label_accuracy(
        task_accuracy,
        concept_artifact,
        cumulative_task,
        cumulative_concept,
        evaluated_concepts,
    )
    additional = json.loads(
        (results_dir / "additional_metrics.json").read_text()
    )
    perturbation = json.loads(
        (results_dir / "concept_annotation_perturbation.json").read_text()
    )

    return {
        **{
            key: spec[key]
            for key in ("dataset", "model", "mode", "probability", "seed")
        },
        "task_accuracy": task_accuracy,
        "concept_accuracy": concept_accuracy,
        "intervention_label_accuracy": intervention_accuracy,
        "intervention_trajectory": trajectory,
        "concept_coverage": float(additional["concept_coverage"]),
        "requested_probability": float(
            perturbation["requested_probability"]
        ),
        "realized_probability": float(perturbation["realized_probability"]),
        "eligible_annotations": int(perturbation["eligible"]),
        "perturbed_annotations": int(perturbation["perturbed"]),
        "evaluated_concepts": evaluated_concepts,
        "graph_sha256": spec.get("graph_sha256"),
        "dataset_sha256": spec.get("dataset_sha256"),
        "validation_and_test_unchanged": bool(
            perturbation["validation_and_test_unchanged"]
        ),
        "run_dir": str(run_dir),
    }


def _discover_runs(output_dir: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    records = []
    incomplete = []
    for spec_path in sorted((output_dir / "runs").glob("**/run_spec.json")):
        run_dir = spec_path.parent
        if not _is_complete(run_dir):
            incomplete.append(str(run_dir))
            continue
        try:
            records.append(_extract_run_metrics(run_dir))
        except Exception as error:
            incomplete.append(f"{run_dir}: {type(error).__name__}: {error}")
    return records, incomplete


def _aggregate(records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["dataset"],
            record["model"],
            record["mode"],
            float(record["probability"]),
        )
        grouped[key].append(record)

    summaries = []
    for (dataset, model, mode, probability), group in sorted(grouped.items()):
        row: Dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "mode": mode,
            "probability": probability,
            "n_runs": len(group),
            "seeds": [int(record["seed"]) for record in group],
        }
        for metric_name in METRIC_NAMES + ("realized_probability",):
            values = [
                float(record[metric_name])
                for record in group
                if _finite_float(record[metric_name]) is not None
            ]
            row[f"{metric_name}_mean"] = fmean(values) if values else float("nan")
            row[f"{metric_name}_std"] = (
                stdev(values) if len(values) > 1 else float("nan")
            )
            row[f"{metric_name}_sem"] = (
                row[f"{metric_name}_std"] / math.sqrt(len(values))
                if len(values) > 1
                else float("nan")
            )
        summaries.append(row)
    return summaries


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_percent(mean: object, sem: object) -> str:
    mean_value = _finite_float(mean)
    sem_value = _finite_float(sem)
    if mean_value is None:
        return "N/A"
    if sem_value is None:
        return f"{100 * mean_value:.1f}"
    return f"{100 * mean_value:.1f}±{100 * sem_value:.1f}"


def _summary_lookup(
    summaries: Sequence[Mapping[str, object]],
) -> Dict[Tuple[str, str, str, float], Mapping[str, object]]:
    return {
        (
            str(row["dataset"]),
            str(row["model"]),
            str(row["mode"]),
            float(row["probability"]),
        ): row
        for row in summaries
    }


def _table_cell(row: Mapping[str, object] | None) -> str:
    if row is None:
        return "N/A"
    return " / ".join(
        _format_percent(
            row[f"{metric_name}_mean"],
            row[f"{metric_name}_sem"],
        )
        for metric_name in DISPLAY_METRICS
    )


def _write_markdown_table(
    path: Path,
    summaries: Sequence[Mapping[str, object]],
    mode: str,
    rates: Sequence[float],
) -> None:
    lookup = _summary_lookup(summaries)
    datasets_models = sorted(
        {
            (str(row["dataset"]), str(row["model"]))
            for row in summaries
            if str(row["mode"]) in {"none", mode}
        }
    )
    rate_symbol = "\\mu" if mode == "missing" else "\\eta"
    lines = [
        (
            "| Dataset | Model | Clean reference | "
            + " | ".join(f"${rate_symbol}={rate:g}$" for rate in rates)
            + " |"
        ),
        "| --- | --- | ---: | " + " | ".join("---:" for _ in rates) + " |",
    ]
    for dataset, model in datasets_models:
        display_model = "C$^2$BM" if model == "c2bm" else model.upper()
        clean = lookup.get((dataset, model, "none", 0.0))
        cells = [
            _table_cell(lookup.get((dataset, model, mode, rate)))
            for rate in rates
        ]
        lines.append(
            f"| {dataset.upper()} | {display_model} | {_table_cell(clean)} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "Entries are **task / clean-test concept / intervention-trajectory "
            "label accuracy (%)**, reported as mean±standard error over seeds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_coverage_table(
    path: Path,
    summaries: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "| Dataset | Model | Condition | Probability | Coverage (%) | Seeds |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {str(row['dataset']).upper()} | {str(row['model']).upper()} | "
            f"{row['mode']} | {float(row['probability']):g} | "
            f"{_format_percent(row['concept_coverage_mean'], row['concept_coverage_sem'])} | "
            f"{int(row['n_runs'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _range_text(values: Sequence[float]) -> str:
    if not values:
        return "N/A"
    return f"{min(values):+.1f} to {max(values):+.1f} pp"


def _rate_span_text(rates: Sequence[float]) -> str:
    percentages = sorted({int(round(100 * rate)) for rate in rates})
    if not percentages:
        return "completed"
    if len(percentages) == 1:
        return f"{percentages[0]}%"
    return f"{percentages[0]}–{percentages[-1]}%"


def _write_findings(
    path: Path,
    summaries: Sequence[Mapping[str, object]],
    rates: Sequence[float],
) -> None:
    lookup = _summary_lookup(summaries)
    settings = sorted(
        {
            (str(row["dataset"]), str(row["model"]))
            for row in summaries
            if str(row["mode"]) == "none"
        }
    )
    lines = ["# Automatically computed findings", ""]
    for mode in ("missing", "noise"):
        lines.append(f"## {mode.capitalize()} annotations")
        lines.append("")
        for rate in rates:
            deltas = {metric: [] for metric in DISPLAY_METRICS}
            for dataset, model in settings:
                clean = lookup.get((dataset, model, "none", 0.0))
                perturbed = lookup.get((dataset, model, mode, rate))
                if clean is None or perturbed is None:
                    continue
                for metric in DISPLAY_METRICS:
                    deltas[metric].append(
                        100
                        * (
                            float(perturbed[f"{metric}_mean"])
                            - float(clean[f"{metric}_mean"])
                        )
                    )
            lines.append(
                f"- At {int(rate * 100)}%: task delta "
                f"{_range_text(deltas['task_accuracy'])}; concept delta "
                f"{_range_text(deltas['concept_accuracy'])}; intervention-label "
                f"delta {_range_text(deltas['intervention_label_accuracy'])}."
            )
        lines.append("")

    coverage_values = [
        100 * float(row["concept_coverage_mean"])
        for row in summaries
        if _finite_float(row["concept_coverage_mean"]) is not None
    ]
    realized_errors = [
        100
        * abs(
            float(row["realized_probability_mean"]) - float(row["probability"])
        )
        for row in summaries
    ]
    lines.extend(
        [
            "## Sanity checks",
            "",
            f"- Coverage across all completed conditions: "
            f"{min(coverage_values):.1f}%–{max(coverage_values):.1f}%."
            if coverage_values
            else "- Coverage unavailable.",
            f"- Largest absolute requested-vs-realized perturbation-rate gap: "
            f"{max(realized_errors):.2f} percentage points."
            if realized_errors
            else "- Perturbation-rate check unavailable.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_control_audit(
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[
            (str(record["dataset"]), str(record["model"]), int(record["seed"]))
        ].append(record)

    lines = [
        "# Paired-control audit",
        "",
        "| Dataset | Model | Seed | Conditions | Dataset fixed | Graph fixed | Val/test unperturbed |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for (dataset, model, seed), group in sorted(grouped.items()):
        dataset_hashes = {
            str(record["dataset_sha256"])
            for record in group
            if record.get("dataset_sha256")
        }
        graph_hashes = {
            str(record["graph_sha256"])
            for record in group
            if record.get("graph_sha256")
        }
        val_test_unchanged = all(
            bool(record["validation_and_test_unchanged"]) for record in group
        )
        lines.append(
            f"| {dataset.upper()} | {model.upper()} | {seed} | {len(group)} | "
            f"{'yes' if len(dataset_hashes) == 1 else 'unavailable' if not dataset_hashes else 'NO'} | "
            f"{'yes' if len(graph_hashes) == 1 else 'unavailable' if not graph_hashes else 'NO'} | "
            f"{'yes' if val_test_unchanged else 'NO'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _delta_range(
    summaries: Sequence[Mapping[str, object]],
    *,
    mode: str,
    rates: Sequence[float],
    metric: str,
) -> str:
    lookup = _summary_lookup(summaries)
    settings = {
        (str(row["dataset"]), str(row["model"]))
        for row in summaries
        if str(row["mode"]) == "none"
    }
    deltas = []
    for dataset, model in settings:
        clean = lookup.get((dataset, model, "none", 0.0))
        if clean is None:
            continue
        for rate in rates:
            perturbed = lookup.get((dataset, model, mode, rate))
            if perturbed is None:
                continue
            deltas.append(
                100
                * (
                    float(perturbed[f"{metric}_mean"])
                    - float(clean[f"{metric}_mean"])
                )
            )
    return _range_text(deltas)


def _write_rebuttal_draft(
    path: Path,
    summaries: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    completed_rates = sorted(
        {
            float(row["probability"])
            for row in summaries
            if str(row["mode"]) in {"missing", "noise"}
        }
    )
    low_rates = [rate for rate in completed_rates if rate <= 0.3]
    high_rates = [rate for rate in completed_rates if rate >= 0.6]
    coverage_values = [
        100 * float(row["concept_coverage_mean"])
        for row in summaries
        if _finite_float(row["concept_coverage_mean"]) is not None
    ]

    lines = [
        "# Rebuttal-ready concept-label sensitivity response",
        "",
        (
            "F-CMs natively support heterogeneous **client-level** partial concept "
            "supervision: a client optimizes only modules associated with variables "
            "it supervises, and the server aggregates each module only over clients "
            "that updated it. We agree that this does not by itself establish "
            "robustness to **sample-level** missing or incorrect annotations. The "
            "paper's concept-label swaps test structured inter-client semantic "
            "disagreement, while its edge additions, removals, and reversals test "
            "structural errors; we therefore added the complementary controlled "
            "experiment requested by the reviewer."
        ),
        "",
        (
            "On ASIA and ALARM, for both bipartite F-CMs (CEM) and graph-based "
            "F-CMs (C$^2$BM), we keep task labels, client partitions, local/global "
            "graphs, validation data, and test data fixed within each paired seed, "
            "and perturb only concept annotations in the training split. Each "
            f"available annotation is independently masked with $\\mu\\in"
            f"\\{{{','.join(f'{rate:g}' for rate in completed_rates)}\\}}$ or "
            f"corrupted with $\\eta\\in\\{{{','.join(f'{rate:g}' for rate in completed_rates)}"
            "\\}$. A corrupted multiclass label is drawn uniformly from the other "
            "valid classes (binary variables therefore undergo a flip). We report "
            "mean±standard error over paired seeds. Each table entry is task "
            "accuracy / clean-test concept accuracy / mean label accuracy over the "
            "full cumulative intervention trajectory. We use Bayesian-network "
            "datasets because they provide genuinely clean concept ground truth, "
            "letting us calibrate the injected error exactly rather than treating "
            "medical pseudo-labels as an error-free reference. Note also that "
            "$\\eta=0.9$ is an intentionally extreme regime: for a binary concept, "
            "90% corruption makes its training labels predominantly inverted."
        ),
        "",
    ]

    if low_rates:
        for mode, label in (("missing", "Missing"), ("noise", "Noisy")):
            lines.append(
                f"For {label.lower()} annotations at {_rate_span_text(low_rates)}, "
                "the change from the "
                f"paired clean reference across dataset/model combinations is "
                f"{_delta_range(summaries, mode=mode, rates=low_rates, metric='task_accuracy')} "
                f"for task accuracy, "
                f"{_delta_range(summaries, mode=mode, rates=low_rates, metric='concept_accuracy')} "
                f"for concept accuracy, and "
                f"{_delta_range(summaries, mode=mode, rates=low_rates, metric='intervention_label_accuracy')} "
                "for intervention-trajectory label accuracy."
            )
            lines.append("")
    if high_rates:
        lines.extend(
            [
                (
                    "As expected, the 60–90% settings are stress tests beyond the "
                    "moderate-noise regime and show the following ranges:"
                ),
                "",
            ]
        )
        for mode, label in (("missing", "missing"), ("noise", "noisy")):
            lines.append(
                f"- {label.capitalize()}: task "
                f"{_delta_range(summaries, mode=mode, rates=high_rates, metric='task_accuracy')}; "
                f"concept "
                f"{_delta_range(summaries, mode=mode, rates=high_rates, metric='concept_accuracy')}; "
                f"intervention-label "
                f"{_delta_range(summaries, mode=mode, rates=high_rates, metric='intervention_label_accuracy')}."
            )
        lines.append("")

    if coverage_values:
        lines.extend(
            [
                (
                    f"Concept coverage is {min(coverage_values):.1f}%–"
                    f"{max(coverage_values):.1f}% across completed conditions. This "
                    "invariance is expected: coverage measures whether the architecture "
                    "predicts each task-relevant concept, whereas annotation corruption "
                    "changes prediction quality rather than the declared concept set. "
                    "The concept and intervention metrics above are therefore the "
                    "relevant evidence about annotation quality."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Missing concept annotations",
            "",
            (output_dir / "missing_annotations.md").read_text().strip(),
            "",
            "## Noisy concept annotations",
            "",
            (output_dir / "noisy_annotations.md").read_text().strip(),
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def summarize(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, incomplete = _discover_runs(output_dir)
    if not records:
        raise SystemExit(f"No complete runs found under {output_dir / 'runs'}")

    summaries = _aggregate(records)
    per_seed_fields = [
        "dataset",
        "model",
        "mode",
        "probability",
        "seed",
        "task_accuracy",
        "concept_accuracy",
        "intervention_label_accuracy",
        "concept_coverage",
        "requested_probability",
        "realized_probability",
        "eligible_annotations",
        "perturbed_annotations",
        "graph_sha256",
        "dataset_sha256",
        "validation_and_test_unchanged",
        "run_dir",
    ]
    _write_csv(output_dir / "per_seed_results.csv", records, per_seed_fields)

    summary_fields = [
        "dataset",
        "model",
        "mode",
        "probability",
        "n_runs",
        "seeds",
    ]
    for metric_name in METRIC_NAMES + ("realized_probability",):
        summary_fields.extend(
            (
                f"{metric_name}_mean",
                f"{metric_name}_std",
                f"{metric_name}_sem",
            )
        )
    _write_csv(output_dir / "summary.csv", summaries, summary_fields)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True)
    )

    available_rates = sorted(
        {
            float(row["probability"])
            for row in summaries
            if row["mode"] in {"missing", "noise"}
        }
    )
    _write_markdown_table(
        output_dir / "missing_annotations.md",
        summaries,
        mode="missing",
        rates=available_rates,
    )
    _write_markdown_table(
        output_dir / "noisy_annotations.md",
        summaries,
        mode="noise",
        rates=available_rates,
    )
    _write_coverage_table(output_dir / "coverage.md", summaries)
    _write_control_audit(output_dir / "control_audit.md", records)
    _write_findings(
        output_dir / "findings.md",
        summaries,
        rates=available_rates,
    )
    _write_rebuttal_draft(
        output_dir / "rebuttal_draft.md",
        summaries,
        output_dir,
    )
    (output_dir / "incomplete_runs.txt").write_text(
        "\n".join(incomplete) + ("\n" if incomplete else "")
    )

    print(f"Summarized {len(records)} complete runs into {output_dir}")
    print(f"Paste-ready table: {output_dir / 'missing_annotations.md'}")
    print(f"Paste-ready table: {output_dir / 'noisy_annotations.md'}")
    print(f"Paste-ready response: {output_dir / 'rebuttal_draft.md'}")
    if incomplete:
        print(
            f"Warning: {len(incomplete)} runs were incomplete; see "
            f"{output_dir / 'incomplete_runs.txt'}"
        )


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "rebuttal_results" / "concept_noise",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run/resume the experiment grid")
    _common_parser(run_parser)
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SETTINGS), default=["asia", "alarm"])
    run_parser.add_argument("--models", nargs="+", choices=["cem", "c2bm"], default=["cem", "c2bm"])
    run_parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    run_parser.add_argument("--rates", nargs="+", type=float, default=list(DEFAULT_RATES))
    run_parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="override paper-matched dataset-specific training horizons",
    )
    run_parser.add_argument("--rerun", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")

    summarize_parser = subparsers.add_parser(
        "summarize", help="summarize already completed runs"
    )
    _common_parser(summarize_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        invalid_rates = [rate for rate in args.rates if not 0.0 < rate <= 1.0]
        if invalid_rates:
            parser.error(f"rates must be in (0, 1], received {invalid_rates}")
        run_grid(args)
    else:
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
