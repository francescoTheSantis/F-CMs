# Federated-C2BM

## Reproducing the Experiments

To reproduce the experiments described in this repository, follow these steps:

1. **Create the conda environment:**
  ```bash
  conda env create -f environment.yaml
  ```
2. **Activate the environment:**
  ```bash
  conda activate federated_c2bm
  ```
3. **Run the main experiment script:**
  ```bash
  python main.py
  ```

This will execute all the experiments defined in the `sweep.yaml` file.