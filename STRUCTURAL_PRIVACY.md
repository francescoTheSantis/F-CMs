# Structural privacy experiments

This repository implements optional edge-level local differential privacy
(LDP) for the client graphs used by graph-based F-CMs (C2BM and CGM).

## Mechanism and guarantee

For each unordered pair of nodes in a client's fixed node set, the local
binary adjacency matrix is represented by one of three states:

1. `i -> j`
2. `j -> i`
3. no edge

The client reports the true state with probability

\[
p=\frac{e^{\varepsilon_s}}{e^{\varepsilon_s}+2}
\]

and each of the other two states with probability

\[
q=\frac{1}{e^{\varepsilon_s}+2}.
\]

For any two input states and any output state, the likelihood ratio is at
most \(p/q=e^{\varepsilon_s}\). Because all unchanged pair factors cancel,
the complete transmitted adjacency matrix is
\(\varepsilon_s\)-edge-LDP when neighboring graphs have the same node set and
differ in one unordered pair state. The weighted maximum-consensus
aggregation, random tie-breaking, cycle removal, and task-parent repair use
only privatized reports and are therefore post-processing.

The guarantee does **not** hide the client concept/node set, graph size, or
client aggregation weight. Graphs differing in \(d\) pair states have the
standard group bound \(d\varepsilon_s\), and repeated releases compose.
The present repository aggregates binary relations; continuous edge
confidence values would require a separately specified bounded/numerical
mechanism.

This guarantee is separate from the repository's sample-level DP-SGD option
for model updates. Enabling both protects two different released objects
under their respective neighboring relations; the two epsilon values should
not be presented as though they were one interchangeable privacy budget.

The implementation is in
[`src/structural_privacy.py`](src/structural_privacy.py). The call in
[`main.py`](main.py) marks the simulated client-to-server boundary: model
construction receives only privatized local graphs when the mechanism is
enabled.

## Rebuttal sweeps

Each sweep runs C2BM and CGM for five seeds at
\(\varepsilon_s\in\{\infty,8,4,2,1\}\). The infinite-epsilon runs are paired
non-private baselines. This is preferable to subtracting a paper mean
obtained from potentially different seeds or settings.

```bash
python main.py --config-name structural_privacy_asia
python main.py --config-name structural_privacy_alarm
python main.py --config-name structural_privacy_chexpert
```

The three configurations inherit the dataset-specific settings from
`test_asia.yaml`, `test_alarm.yaml`, and `test_chexpert.yaml`, respectively.
The CheXpert configuration deliberately uses the same cached proxy graph as
the paper (`~/.cache/federated_c2bm/cheXpert/learned_graph.pkl`). If that
artifact is absent, first reproduce the paper's graph-learning step with
`dataset.load_graph=false`, then run the sweep with the cached graph. Do not
silently substitute a different proxy when comparing task-accuracy changes.
To use a different GPU:

```bash
python main.py --config-name structural_privacy_asia 'trainer.devices=[1]'
```

For a single smoke run:

```bash
python main.py --config-name default \
  dataset=asia \
  model=c2bm \
  learning=local_federated \
  dataset.load_embeddings=true \
  trainer.max_epochs=2 \
  learning.subgraphs.rnd_drift=1000 \
  learning.subgraphs.aggregate_graph.local_graphs=from_true_graph \
  learning.subgraphs.aggregate_graph.perc_clients_alterations=0.1 \
  learning.subgraphs.aggregate_graph.graph_alteration_prob=0.1 \
  learning.subgraphs.aggregate_graph.structural_privacy.enabled=true \
  learning.subgraphs.aggregate_graph.structural_privacy.epsilon=4.0 \
  'trainer.devices=[0]'
```

Every completed private run writes:

- `results/aggregated_test_metrics.json`: task accuracy;
- `results/structural_privacy.json`: mechanism parameters, sampled report
  change rate, DP-safe post-processing events, and graph disagreement;
- `results/graph_metrics.json`: disagreement with a ground-truth graph when
  one is available.

`structural_privacy.json` compares the private aggregate with the non-private
maximum-consensus graph built from the same simulated clients. This oracle
comparison is strictly an **offline benchmark evaluation** and must be
disabled in a real deployment by setting
`evaluate_against_non_private=false`.

## Generate the table

Point the summarizer at one or more Hydra multirun directories:

```bash
python scripts/summarize_structural_privacy.py \
  outputs/multirun/2026-07-24/12-00-00 \
  outputs/multirun/2026-07-24/13-00-00 \
  outputs/multirun/2026-07-24/14-00-00
```

It writes:

- `structural_privacy_summary/runs.csv`;
- `structural_privacy_summary/summary.csv`;
- `structural_privacy_summary/table.md`.

Task-accuracy changes are paired by dataset, model, and seed against the
\(\varepsilon_s=\infty\) run. Graph disagreement is the percentage of
unordered three-state relations that differ from the paired non-private
aggregate; an orientation reversal counts once.

## Tests

The tests check the exact privacy likelihood ratio, the
\(\varepsilon_s=0\) and \(\infty\) limits, empirical report probabilities,
seed reproducibility, input validation, and graph-disagreement semantics:

```bash
python -m unittest tests.test_structural_privacy
```
