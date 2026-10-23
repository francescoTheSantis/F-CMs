"""
MMIST_ccRCC multimodal dataset implementation.

link to download processed data: https://multi-modal-ist.github.io/datasets/ccRCC/
link to the paper: https://arxiv.org/pdf/2405.01658v1
"""

import io
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from env import CACHE


RAW_MMIST_DIR = CACHE / "MMIST_ccRCC"

# Rename the condition using the following mapping:
MAPPING = {
    "gender": "gender",
    "age_diag": "age",
    "grade": "tumor_grade",
    "cancer_history": "cancer_history",
    "ajcc_path_tumor_pt": "T",
    "ajcc_path_nodes_pn": "N",
    "ajcc_clin_metastasis_cm": "M",
    "ajcc_path_metastasis_pm": "Mp",
    "ajcc_path_tumor_stage": "ajcc_tumor_stage",
    "race": "race",
    "VHL_mutation": "VHL",
    "PBMR1_mutation": "PBRM1",
    "TTN_mutation": "TTN",
    "vital_status_12": "vital_status_12",
}

# values in mapping except vital_status_12 which is the target variable
CONCEPT_NAMES = list(MAPPING.values())
CONCEPT_NAMES.remove("vital_status_12")

TARGET_NAME = "vital_status_12"

URL = "https://github.com/Multi-Modal-IST/multi-modal-ist.github.io/tree/master/datasets/ccRCC/Code"

# ----------------------------------------------------------------------------
# Data download functions
# ---------------------------------------------------------------------------

def download_github_subfolder(github_url, destination="."):
    # Parse the GitHub URL
    parts = github_url.replace("https://github.com/", "").split("/tree/")
    repo = parts[0]                          # USER/REPO
    branch, subfolder = parts[1].split("/", 1)  # BRANCH, path/to/folder

    # Download the ZIP of the full repo
    zip_url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
    with urllib.request.urlopen(zip_url) as response:
        zip_data = response.read()

    # Extract only the target subfolder
    repo_name = repo.split("/")[1]
    prefix = f"{repo_name}-{branch}/{subfolder}/"  # path inside the ZIP

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for file in zf.namelist():
            if file.startswith(prefix) and not file.endswith("/"):
                # Reconstruct relative path
                relative_path = file[len(prefix):]
                dest_path = os.path.join(destination, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with zf.open(file) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())


def download_data(destination: Path) -> None:
    """
    Download the raw MMIST_ccRCC data into `destination` if not already present.
    """
    destination.mkdir(parents=True, exist_ok=True)
    if not any(destination.glob("*")):
        print(f"Downloading MMIST_ccRCC data to {destination}...")
        download_github_subfolder(URL, destination=destination)

# -----------------------------------------------------------------------------
# Utility functions for loading and processing the raw data files into tensors.
# ---------------------------------------------------------------------------

def load_npz(paths):
    return {Path(p).stem: dict(np.load(p, allow_pickle=True)) for p in paths}


# ---------------------------------------------------------------------------
# Per-split Dataset wrapper
# ---------------------------------------------------------------------------

class _MMISTccRCCSplit(Dataset):
    """
    Lightweight PyTorch Dataset wrapping prebuilt tensors for one split.

    Exposes the interface expected by generate_split.py:
      - Iteration: yields dicts with 'x', 'c', 'y', 'graph' per sample.
      - Direct tensor attributes: .X, .c, .y (for bulk access).
      - .register_graph() to attach the per-client causal subgraph.
    """

    def __init__(self, X: torch.Tensor, ct: torch.Tensor, wsi: torch.Tensor, mri: torch.Tensor, c: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X          # (N, 2048)  — concatenated CT + WSI + MRI
        self.ct = ct        # (N, 512)
        self.wsi = wsi      # (N, 1024)
        self.mri = mri      # (N, 512)
        self.c = c          # (N, n_concepts)
        self.y = y          # (N, 1)
        self.graph: Dict = {}

    def register_graph(self, graph: Dict) -> None:
        """Store the per-client causal subgraph (called by the FL training loop)."""
        self.graph = graph

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "x": self.X[idx],
            "ct": self.ct[idx],
            "wsi": self.wsi[idx],
            "mri": self.mri[idx],
            "c": self.c[idx],
            "y": self.y[idx],
            "graph": self.graph,
        }


# ---------------------------------------------------------------------------
# Outer container class
# ---------------------------------------------------------------------------

class MMISTccRCC:
    """
    Outer container for the MMIST_ccRCC multimodal dataset.

    Responsibilities:
      - Hold dataset-level metadata (c_info, y_info, modalities).
      - Trigger data download and validation in __init__.
      - Expose split() to build train / val / test _MMISTccRCC objects.
      - Expose set_active_modality() to switch all splits between 'image'
        and 'text' at once.
      - Expose load_ground_truth_graph() (returns None — no known causal graph).

    Instantiated by Hydra from conf/dataset/mmist_ccrcc.yaml via
    `loader._target_: src.data.datasets.mmist_ccrcc.MMISTccRCC`.
    """

    def __init__(
        self,
        active_modality: str = "image",
        seed: int = 1,
    ):
        self.active_modality = active_modality
        self.seed = seed

        # NOTE: The cardinalities are updated below
        self.c_info = {
            "names": CONCEPT_NAMES,
            "cardinality": [None] * len(CONCEPT_NAMES), 
        }
        # NOTE: The target variable is binary (vital_status_12: alive vs deceased at 12 months).
        self.y_info = {"names": [TARGET_NAME], "cardinality": [2]}
        self.modalities = ["image"]
        self.data: Dict[str, "_MMISTccRCCSplit"] = {}

        RAW_MMIST_DIR.mkdir(parents=True, exist_ok=True)

        # data are downloaded if not already present in RAW_MMIST_DIR
        download_data(RAW_MMIST_DIR)

        # Load the CSV file
        df = pd.read_csv(os.path.join(RAW_MMIST_DIR, "clinical+genomic_split.csv"))
        
        # cancer_history has been stored as NaN if not present, turn it to 0 (no history).
        df['cancer_history'] = df['cancer_history'].fillna(0)

        # there are few NaNs (17) for age_diag. We remove those rows.
        df = df.dropna()
        
        # Turn one hot encoded race columns into categorical variables (0, 1, ..., #races-1)
        race_columns = [col for col in df.columns if col.startswith('race_')]
        # create value to index mapping
        race_mapping = {col: idx for idx, col in enumerate(race_columns)}

        df['race'] = df[race_columns].idxmax(axis=1).apply(lambda x: race_mapping[x])
        df = df.drop(columns=race_columns)

        # certain columns have values lower than 0, add 1 to each value to make them non-negative categorical variables
        for col in df.columns:
            if df[col].dtype in [int, float] and (df[col] < 0).any():
                df[col] = df[col] + 1
            # convert column to int if it is float but has only integer values
            if df[col].dtype == float and (df[col] % 1 == 0).all():
                df[col] = df[col].astype(int)

        # Rename columns using MAPPING
        df = df.rename(columns=MAPPING)

        # Store cardinality for each concept in c_info
        cardinalities = []
        for col in CONCEPT_NAMES:
            if col != TARGET_NAME:
                cardinalities.append(df[col].nunique())
        self.c_info['cardinality'] = cardinalities

        # Split into train/test following the split column in the CSV
        train_df = df[df['Split'] == 'train'].drop(columns=['Split'])
        test_df = df[df['Split'] == 'test'].drop(columns=['Split'])

        # turn all variables to int
        for col in [x for x in train_df.columns if x != 'case_id']:
            train_df[col] = train_df[col].astype(int)
        for col in [x for x in test_df.columns if x != 'case_id']:
            test_df[col] = test_df[col].astype(int)

        print("Reading processed CT, WSI and MRI data...")

        ct_path = os.path.join(RAW_MMIST_DIR, "Features", "CT Features")
        ct_paths = [os.path.join(ct_path, f) for f in os.listdir(ct_path) if f.endswith('.npz')]
        ct_data = load_npz(ct_paths)
        # Process keys: eliminate the last term after "-" ("-" included)
        ct_data = {"-".join(k.split("-")[:-1]): v for k, v in ct_data.items()}

        wsi_path = os.path.join(RAW_MMIST_DIR, "Features", "WSI Features")
        wsi_paths = [os.path.join(wsi_path, f) for f in os.listdir(wsi_path) if f.endswith('.npz')]
        wsi_npz = load_npz(wsi_paths)
        # Read the WSI_patientfiles.csv file
        wsi_patientfiles_path = os.path.join(RAW_MMIST_DIR, "WSI_patientfiles.csv")
        wsi_patientfiles_df = pd.read_csv(wsi_patientfiles_path)
        # Substitute the keys in the wsi_data with the value in case id column of the wsi_patientfiles_df file
        wsi_data = {}
        for k, v in wsi_npz.items():
            if (k+".npz") in wsi_patientfiles_df['chosen_exam'].values:
                wsi_data[wsi_patientfiles_df[wsi_patientfiles_df['chosen_exam'] == (k+".npz")]['case_id'].values[0]] = v

        mri_path = os.path.join(RAW_MMIST_DIR, "Features", "MRI Features")
        mri_paths = [os.path.join(mri_path, f) for f in os.listdir(mri_path) if f.endswith('.npz')]
        mri_data = load_npz(mri_paths)
        mri_data = {"-".join(k.split("-")[:-1]): v for k, v in mri_data.items()}

        # Now split the train in train and val with 90/10 split, stratifying on the target variable.
        case_ids = train_df[['case_id', 'vital_status_12']].drop_duplicates()
        train_patients, val_patients = train_test_split(
            case_ids, test_size=0.10, random_state=self.seed, shuffle=True, stratify=case_ids['vital_status_12']
        )

        train_val_df = train_df.copy()
        train_df = train_val_df[train_val_df['case_id'].isin(train_patients['case_id'])].reset_index(drop=True)
        val_df = train_val_df[train_val_df['case_id'].isin(val_patients['case_id'])].reset_index(drop=True) 

        # Now concat all the embeddings for each patient to make the x, use the columns of the CSV containing the tabular data to make c and vital_status_12 to make y. 
        # Store x, c and y as tensors in self.data for each split.
        for split_name, split_df in zip(["train", 'val', "test"], [train_df, val_df, test_df]):
            x_list = []
            ct_list = []
            wsi_list = []
            mri_list = []
            c_list = []
            y_list = []
            for _, row in split_df.iterrows():
                case_id = row['case_id']
                ct_embed = ct_data[case_id]['arr_0'].flatten() if case_id in ct_data else np.zeros(512)
                wsi_embed = wsi_data[case_id]['arr_0'].flatten() if case_id in wsi_data else np.zeros(1024)
                mri_embed = mri_data[case_id]['arr_0'].flatten() if case_id in mri_data else np.zeros(512)
                x_list.append(np.concatenate([ct_embed, wsi_embed, mri_embed]))
                ct_list.append(ct_embed)
                wsi_list.append(wsi_embed)
                mri_list.append(mri_embed)
                c_list.append(row[CONCEPT_NAMES].values)
                y_list.append(row[TARGET_NAME])
            split_ds = _MMISTccRCCSplit(
                X=torch.tensor(np.array(x_list), dtype=torch.float32),
                ct=torch.tensor(np.array(ct_list), dtype=torch.float32),
                wsi=torch.tensor(np.array(wsi_list), dtype=torch.float32),
                mri=torch.tensor(np.array(mri_list), dtype=torch.float32),
                c=torch.tensor(np.array(c_list, dtype=np.float32)),
                y=torch.tensor(y_list, dtype=torch.float32).unsqueeze(1),
            )
            self.data[split_name] = split_ds
        print("Finished loading data.")


    def load_ground_truth_graph(self):
        """
        Return the ground-truth causal adjacency matrix, or None if unknown.
        """
        self.adj = None
        return self.adj

    def set_active_modality(self, modality: str) -> None:
        """Update the active modality label on the container (no-op on splits)."""
        self.active_modality = modality
