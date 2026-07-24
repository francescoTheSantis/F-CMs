# F-CMs: Federated Concept-Based Models

This is the official implementation of the paper:

> **Federated Concept-Based Models: Interpretable Models with Distributed Supervision**  
> Dario Fenoglio*, Arianna Casanova*, Francesco De Santis, Gabriele Dominici, Johannes Schneider, Pietro Barbiero, Giovanni De Felice, Marc Langheinrich, Martin Gjoreski  
> [arXiv:2602.04093](https://arxiv.org/abs/2602.04093)

`*` Equal contribution.

## Overview

Federated Concept-Based Models (F-CMs) train interpretable concept-based models in federated settings where concept supervision is distributed across clients and may evolve over time. The framework supports partial concept annotations, heterogeneous clients, dynamic client participation, and architecture updates when new concepts or dependencies appear. At a high level, the F-CMs pipeline performs three steps during federated training:

- **Graph aggregation:** client-provided concept graphs are combined into a shared graph.
- **Dynamic architecture adaptation:** concept and task modules are expanded or rewired when the global concept space changes.
- **Module-wise training and aggregation:** clients update only the concept/task modules for which they have supervision; the server aggregates each module only from clients that updated it.

The code instantiates F-CMs with several concept-based architectures, including CBM, CEM, CGM, and C2BM, and includes centralized, localized, static federated, FedCBM, and FCL baselines.


## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/francescoTheSantis/F-CMs.git
cd F-CMs
conda env create -f environment.yaml
conda activate fcms
```

Alternatively, install the Python dependencies with pip:

```bash
pip install -r requirements.txt
```

## Data and Cache

The project stores processed datasets, embeddings, graphs, and client splits in `~/.cache/federated_c2bm`. You can override this location with:

```bash
export FEDERATED_C2BM_CACHE=/path/to/cache
```

For a first run on a dataset, set:

```bash
dataset.load_embeddings=false
```

This creates the cached preprocessed dataset. Later runs can use:

```bash
dataset.load_embeddings=true
```

The Bayesian-network datasets used in the paper (`asia`, `sachs`, `alarm`, `insurance`, `hailfinder`) are handled through `bnlearn` and local BIF files. Image datasets such as SIIM-Pneumothorax and CheXpert may require manual dataset access or Kaggle/Hugging Face credentials, depending on the loader.

## Configuration

Experiments are configured with Hydra. The main configuration groups are:

- `dataset`: choose a dataset, e.g. `asia`, `sachs`, `alarm`, `insurance`, `hailfinder`, `siim_pneumothorax`, `cheXpert`, `cheXpert_multi`.
- `model`: choose an architecture, e.g. `cbm_mlp`, `cem`, `cgm`, `c2bm`, or their multimodal variants.
- `learning`: choose a training regime:
  - `centralized`: pooled-data upper bound.
  - `localized`: one independent model per client.
  - `local_federated`: F-CMs / static F-CMs simulation used in the paper.
  - `FedCBM`: external static FedCBM baseline.
  - `FCL`: external static FCL baseline.
  - `federated`: Flower-based server/client execution.

Important parameters:

- `learning.n_clients`: number of active clients.
- `learning.subgraphs.rnd_drift`: round at which new clients/concepts enter. Set it within the training horizon for F-CMs with drift; set it larger than `trainer.max_epochs` for static/no-drift runs.
- `learning.subgraphs.aggregate_graph`: local graph aggregation settings.
- `trainer.max_epochs`: number of communication rounds for federated modes.
- `trainer.devices`: GPU ids used by PyTorch Lightning.
- `engine.optim_kwargs.lr`: learning rate.

## Running Experiments

Use `--config-name default` for a single controlled run. For example, run F-CMs on Asia with C2BM:

```bash
python main.py --config-name default \
  dataset=asia \
  model=c2bm \
  learning=local_federated \
  dataset.load_embeddings=false \
  trainer.max_epochs=20 \
  learning.subgraphs.rnd_drift=10 \
  'trainer.devices=[0]'
```

After the cache has been created, rerun with `dataset.load_embeddings=true`:

```bash
python main.py --config-name default \
  dataset=asia \
  model=c2bm \
  learning=local_federated \
  dataset.load_embeddings=true \
  trainer.max_epochs=120 \
  learning.subgraphs.rnd_drift=10 \
  'trainer.devices=[0]'
```

To run the static federated variant, keep the same learning mode but move drift beyond the training horizon:

```bash
python main.py --config-name default \
  dataset=asia \
  model=c2bm \
  learning=local_federated \
  dataset.load_embeddings=true \
  trainer.max_epochs=120 \
  learning.subgraphs.rnd_drift=1000 \
  'trainer.devices=[0]'
```

To run the external baselines:

```bash
python main.py --config-name default dataset=asia model=c2bm learning=FedCBM dataset.load_embeddings=true learning.subgraphs.rnd_drift=1000
python main.py --config-name default dataset=asia model=c2bm learning=FCL dataset.load_embeddings=true learning.subgraphs.rnd_drift=1000
```

For `FedCBM` and `FCL`, `learning.subgraphs.rnd_drift` must be larger than the number of rounds, because these baselines are implemented as static federated methods.

The paper sweep templates are in `conf/test_*.yaml`. For example:

```bash
python main.py --config-name test_sachs
python main.py --config-name test_alarm
python main.py --config-name test_siim
python main.py --config-name test_chexpert_multi
```

Before launching a sweep, check the selected GPUs, seeds, datasets, models, and `dataset.load_embeddings` flag in the corresponding config file.

## Outputs

Hydra writes each run under:

```text
outputs/<date>/<time>/
outputs/multirun/<date>/<time>/<job_id>/
```

Inside each run directory, the code writes metrics and artifacts such as:

```text
results/y_accuracy.pkl
results/c_accuracy.pkl
results/training_history.json
results/additional_metrics.json
results/graph_metrics.json
results/aggregated_test_metrics.json
graph.pkl
```

Use `show_results.py` to aggregate completed sweep folders and generate the tables/figures. Edit the `paths` list in that script to point to your Hydra output folders.

## Structural Privacy

Graph-based F-CMs can optionally privatize every client's transmitted
three-state pair relations with edge-level local differential privacy.
Rebuttal sweeps for C2BM and CGM on Asia, Alarm, and CheXpert, together with
the precise guarantee, output metrics, and table-generation commands, are
documented in [STRUCTURAL_PRIVACY.md](STRUCTURAL_PRIVACY.md).

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this repository, please cite:

```bibtex
@article{fenoglio2026fcms,
  title   = {Federated Concept-Based Models: Interpretable Models with Distributed Supervision},
  author  = {Fenoglio, Dario and Casanova, Arianna and De Santis, Francesco and Dominici, Gabriele and Schneider, Johannes and Barbiero, Pietro and De Felice, Giovanni and Langheinrich, Marc and Gjoreski, Martin},
  journal = {arXiv preprint arXiv:2602.04093},
  year    = {2026}
}
```
