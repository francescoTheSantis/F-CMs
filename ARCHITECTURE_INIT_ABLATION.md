# Architecture-update initialization ablation

This experiment isolates the parameter-transfer rule used at the structural-shift
round. Everything before the shift is shared across the three methods:

- `full_reinit`: preserve same-name/same-shape tensors and fully reinitialize a
  changed-shape tensor (the backward-compatible default).
- `partial_random`: copy semantically corresponding old blocks and leave only new
  blocks at their normal random initialization.
- `partial_zero`: copy corresponding blocks and zero only new blocks when doing so
  preserves the pre-shift function.

The two configured shift groups are `10_50_to_100` (`first_no_add -> all`) and
`51_75_to_100` (`all_no_add -> all`). Both configurations require the post-shift
client set to cover exactly 100% of structural nodes.

## Run the sweeps

From the repository root:

```bash
conda activate private_CF
python main.py --config-name=rebuttal_arch_init_asia -m
python main.py --config-name=rebuttal_arch_init_alarm -m
```

Each command runs 60 jobs: five seeds, two architectures (`cgm`, `c2bm`), three
initialization methods, and two shift groups. Every job uses 60 federated rounds,
with the structural shift at round 10 and early stopping disabled for this horizon.
To run a smaller prespecified seed set, override the sweep dimension, for example:

```bash
python main.py --config-name=rebuttal_arch_init_asia -m seed=1,2,3
```

Hydra writes runs below:

```text
outputs/architecture_init_ablation/{asia,alarm}/DATE/TIME/JOB_ID/
```

Do not launch jobs for the same dataset concurrently. Dataset split generation uses
a shared per-dataset cache directory. Hydra's default basic sweeper executes the jobs
sequentially.

## Saved artifacts

Every completed run stores the following under
`results/architecture_ablation/` inside its Hydra run directory:

- `experiment.json`: experiment identity, drift round, and metric-window config;
- `validation_client_losses.csv`: raw per-round/per-client validation task loss;
- `validation_round_summary.csv`: arithmetic client mean, population standard
  deviation, and evaluated-client count for every round;
- `shift_snapshot_client_losses.csv` and `shift_snapshot_summary.csv`: the
  task loss immediately after migration and before any local optimization at
  the shift, using the exact per-client migrated models;
- `structural_change.json`: migration decisions and structural-change metadata;
- `seed_metrics.csv` and `seed_metrics.json`: final metrics for that seed and method;
- `config.json`: resolved training configuration.

These files are sufficient to regenerate all figures and tables without retraining.

## Regenerate plots and tables

Scan any number of run or sweep roots recursively:

```bash
python plot_architecture_ablation.py \
  outputs/architecture_init_ablation/asia/DATE/TIME \
  outputs/architecture_init_ablation/alarm/DATE/TIME \
  --output-dir architecture_ablation_report \
  --across-seed \
  --strict-design
```

By default, figures are written as PNG and PDF. Use, for example,
`--formats png --dpi 200` for a quicker preview.
Pass one completed sweep directory per dataset; the script rejects duplicate
dataset/architecture/split/method/seed identities rather than mixing reruns.

The report contains:

- one figure for each seed/dataset/architecture/shift group under `per_seed/`;
- a CSV and Markdown metric table adjacent to every per-seed figure;
- optional across-seed figures and their source curves/tables under `across_seed/`;
- `metrics_per_seed.csv`, the combined tidy per-seed metrics;
- `metrics_summary.csv`, mean and sample standard deviation across seeds;
- `reviewer_tables.md`, compact copy-ready mean ± standard-deviation tables.
- `controlled_design_checks.csv`, an audit that all three methods are present,
  have matching round/client grids, and have equal pre-shift per-client losses.

In a per-seed figure, each line is the arithmetic mean validation task loss across
eligible federation clients and the band is ±1 population standard deviation across
those clients. In an across-seed figure, each seed first contributes one client-mean
curve; the line is the mean of those seed-level curves and the band is ±1 sample
standard deviation across seeds. Clients are never pooled across seeds.

The primary round-level value at the shift is measured after that round's local
training and FedAvg, as in the existing training loop. The auxiliary
`immediate_post_migration_loss` and `immediate_loss_spike` metrics isolate the
initialization discontinuity before local training can smooth it out.

The active-federation mean intentionally follows the requested client population:
the pre-shift and post-shift cohorts can differ as late clients arrive. Therefore,
raw within-method spikes combine architectural change with federation composition;
the controlled causal comparison is the difference among strategies under the same
seed/split. The raw client CSV supports a common-client sensitivity analysis without
retraining if desired.

Parameter fractions use the post-shift model parameter count as their denominator.
Parameters removed with an old parent are absent from that denominator; such cases
are logged as function-preservation fallbacks in `structural_change.json`.

Without `--strict-design`, the plotting script warns about incomplete/failed checks
and still plots round data persisted before interruption. The strict flag is
recommended for final rebuttal tables. A run contributes to final metric tables only
after `seed_metrics.csv` has been written. Recovery time is summarized only over
recovered seeds and is always accompanied by the recovery rate and its available-seed
count, avoiding silent censoring.
