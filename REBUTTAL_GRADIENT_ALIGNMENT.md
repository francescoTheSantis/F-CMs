# Empirical Assumption (S3) experiment

## What can be quantified

Assumption (S3) is uniform over the entire parameter space, so no finite
training experiment can establish its smallest valid global constant. What can
be measured exactly on the empirical training objectives is the mismatch at
every broadcast iterate visited by training.

For client \(k\), the audit makes one complete, example-weighted pass over its
training data at the broadcast model \(w_t\), before any local update, and
computes

\[
g_{k,t}=\nabla F_k(w_t).
\]

The explicit reference objective is the pooled empirical objective

\[
F_{\mathrm{pool}}(w)=\sum_k\frac{n_k}{N}F_k(w),\qquad
g^{\mathrm{pool}}_t=\sum_k\frac{n_k}{N}g_{k,t}.
\]

For coordinate \(a\), let \(m_{k,a}\in\{0,1\}\) denote whether the same
freezing logic used in local training lets client \(k\) update that coordinate.
The module-wise direction in (S3) is computed as

\[
d_{t,a}=
\frac{\sum_k n_k m_{k,a}g_{k,t,a}}
     {\sum_k n_k m_{k,a}},
\]

with a zero value only when no client covers the coordinate. The primary
diagnostic is therefore

\[
\widehat\zeta_t^2
=\left\|d_t-g^{\mathrm{pool}}_t\right\|_2^2,\qquad
\widehat\zeta_{\mathrm{traj}}^2
=\max_{t\in\mathcal T}\widehat\zeta_t^2.
\]

The maximum is an observed-trajectory diagnostic, not a proof of the
supremum over all \(w\). In particular, it should be described as
“maximum observed squared mismatch” rather than as a certified global upper
bound on \(\zeta^2\).

## Metrics recorded

Every measured round stores:

- \(\widehat\zeta_t\) and \(\widehat\zeta_t^2\);
- relative L2 and relative squared L2 mismatch, normalized by the pooled
  gradient norm;
- cosine similarity and angle in degrees;
- the distance between unit-normalized directions;
- the norm ratio \(\|d_t\|/\|g_t^{\mathrm{pool}}\|\);
- the relative mismatch after optimally rescaling \(d_t\), which separates
  angular disagreement from blockwise magnitude rescaling;
- both gradient norms, parameter coverage, per-client sample counts and masks;
- all the same alignment metrics for the encoder, task head, and each
  concept/variable module.

Absolute \(\widehat\zeta_t^2\) is the quantity corresponding directly to
(S3), but it depends on the loss normalization. It should be reported together
with a dimensionless relative metric and cosine similarity.

## Rebuttal protocol

The three dedicated configurations use a deliberately simple and auditable
static protocol:

- 10 fixed clients and fixed heterogeneous supervision masks;
- one local epoch per communication round;
- one full client-data gradient pass immediately before that local epoch;
- SGD without momentum and zero training/test interventions;
- an audit at every communication round;
- five seeds;
- 30 rounds for Asia and 40 rounds for Alarm and CheXpert;
- batch size 512;
- the same CBM architecture across the three datasets;
- complete parameter coverage is checked and stored in every record.

The extra audit costs roughly one local epoch per measured round: it performs
one forward/backward pass over each active client's complete experimental
training split, but no optimizer step. There is no separate centralized data
pass; the pooled gradient is formed exactly from the full client gradients.

CheXpert uses a reproducible 10% subset to keep five full-pass, 40-round runs
tractable. “Full dataset” therefore means the complete training split of this
declared CheXpert experiment, not a minibatch estimate. Its deterministic
bipartite concept-to-task graph is used only to generate the client supervision
subsets; the CBM itself does not use causal graph propagation.

## Running the experiments

From the repository root, using the project environment:

```bash
python main.py --config-name rebuttal_alignment_asia
python main.py --config-name rebuttal_alignment_alarm
python main.py --config-name rebuttal_alignment_chexpert
```

The configurations use Hydra multirun and launch seeds 1--5. The first
CheXpert run builds a reduced, seed-specific embedding cache. Once all five
caches exist, exact repeat runs can add `dataset.load_embeddings=true`.

For a short integration check:

```bash
python main.py --config-name rebuttal_alignment_asia \
  hydra.mode=RUN seed=1 trainer.max_epochs=1 trainer.patience=1 \
  learning.settings.n_rounds=1 learning.settings.patience=1
```

Each run writes the following files under its Hydra output directory:

- `results/gradient_alignment_rounds.json`;
- `results/gradient_alignment_rounds.csv`;
- `results/gradient_alignment_summary.json`.

Aggregate any collection of completed Hydra output directories with:

```bash
python summarize_gradient_alignment.py \
  PATH_TO_ASIA_MULTIRUN PATH_TO_ALARM_MULTIRUN PATH_TO_CHEXPERT_MULTIRUN \
  --output-dir rebuttal_alignment_summary
```

This produces seed-level and grouped CSV/JSON summaries, a LaTeX table, a
round-wise trajectory CSV, and PDF/PNG plots of
\(\widehat\zeta_t^2\) and cosine similarity.

## Recommended reporting

The main table should report, for each dataset, the mean and standard error
across seeds of:

1. \(\max_t\widehat\zeta_t^2\);
2. mean relative L2 mismatch;
3. mean cosine similarity and minimum trajectory cosine;
4. mean scale-adjusted relative L2 mismatch.

Also report that the covered-coordinate fraction was 1.0 in every retained
run. The trajectory figure is useful because a single maximum can be dominated
by an early round or by a round where the reference gradient is already small.

Suggested qualification for the rebuttal:

> We agree that (S3) cannot be judged from the theorem alone. We therefore
> measured its pointwise discrepancy at every broadcast iterate. Before local
> training, every client computed a full-pass gradient at the common global
> model. We compared the module-normalized aggregate with the gradient of the
> explicitly defined sample-size-weighted pooled empirical objective. We
> report the maximum observed squared discrepancy, relative discrepancy, and
> cosine similarity over five seeds. These measurements do not certify the
> uniform bound for every parameter vector, but they directly quantify the
> mismatch on the optimization trajectories to which the convergence result
> is applied.

Replace the last sentence of the first paragraph with the actual aggregate
numbers after all five-seed runs have completed.
