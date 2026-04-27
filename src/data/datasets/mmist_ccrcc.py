"""
MMIST_ccRCC multimodal dataset implementation.

link to download processed data: https://multi-modal-ist.github.io/datasets/ccRCC/
link to the paper: https://arxiv.org/pdf/2405.01658v1
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn, seed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import urllib.request
import zipfile
import io


from env import CACHE

try:
    from torchvision import transforms
    import torchvision.models as tv_models
    from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
except ImportError:  # pragma: no cover
    transforms = None
    tv_models = None
    ResNet18_Weights = None
    ResNet50_Weights = None


# ---------------------------------------------------------------------------
# Constants — fill in the actual values for MMIST_ccRCC
# ---------------------------------------------------------------------------

RAW_MMIST_DIR = CACHE / "MMIST_ccRCC"
PROCESSED_MULTIMODAL_DIR = CACHE / "MMIST_ccRCC_multi"

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

# Filenames written to PROCESSED_MULTIMODAL_DIR
MERGED_FILENAME = "mmist_ccrcc_merged.csv"
TRAIN_SPLIT_FILENAME = "train_split.csv"
VAL_SPLIT_FILENAME = "val_split.csv"
TEST_SPLIT_FILENAME = "test_split.csv"

# ----------------------------------------------------------------------------
# Data download functions
# ---------------------------------------------------------------------------

def download_github_subfolder(github_url, destination="."):
    # Parse the GitHub URL
    # e.g. https://github.com/USER/REPO/tree/BRANCH/path/to/folder
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
# Metadata / split builders
# ---------------------------------------------------------------------------

def build_multimodal_metadata(
    raw_root: Path = RAW_MMIST_DIR,
    processed_root: Path = PROCESSED_MULTIMODAL_DIR,
    seed: int = 42,
    force_rebuild: bool = False,
) -> Path:
    """
    Merge image-level metadata with text (reports / clinical notes) into a
    single CSV saved to processed_root / MERGED_FILENAME.

    Returns the path to the merged CSV.

    Steps to implement:
      1. Call _validate_required_files(raw_root) to assert raw data is present.
      2. Load the raw label / metadata CSV(s) into a DataFrame.
      3. Load the text data (reports, clinical notes, …) and merge it with the
         image metadata on a shared key (patient_id, sample_id, …).
      4. Derive / rename columns so the final DataFrame has at least:
             - "sample_id"   : unique identifier per row
             - "img_id"      : relative path to the image file
             - "patient_id"  : patient-level grouping key (used for split)
             - "text_input"  : the primary text string fed to the text encoder
             - TARGET_NAME   : integer label (0 / 1)
             - *CONCEPT_NAMES: one column per concept, integer-valued
      5. Handle NaNs, binarise uncertain labels, balance classes if needed.
      6. Save the merged DataFrame to metadata_path and return the path.

    TODO: implement steps 1-6 for the actual MMIST_ccRCC data format.
    """
    processed_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    metadata_path = processed_root / MERGED_FILENAME

    if metadata_path.exists() and not force_rebuild:
        return metadata_path

    # TODO: load raw CSVs / JSON / DICOM headers
    # TODO: merge image metadata with text
    # TODO: clean, binarise, balance
    # TODO: save to metadata_path

    raise NotImplementedError("build_multimodal_metadata() is not yet implemented.")


def create_patient_splits(
    metadata_path: Path,
    processed_root: Path = PROCESSED_MULTIMODAL_DIR,
    seed: int = 42,
) -> Dict[str, Path]:
    """
    Partition samples into train / val / test by patient_id (no patient leaks
    across splits) and write one CSV per split containing only the "sample_id"
    column.

    Returns a dict  {'train': Path, 'val': Path, 'test': Path}.

    Steps to implement:
      1. Read the merged metadata CSV.
      2. Collect unique patient_id values.
      3. Split patients 70 / 10 / 20 (train / val / test) with train_test_split,
         stratifying if class imbalance is a concern.
      4. Map each patient to its split and filter the DataFrame accordingly.
      5. Save each split's sample_id list to a CSV and return the paths.

    TODO: adjust split ratios if required by the dataset size.
    """
    processed_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metadata_path)
    unique_patients = df["patient_id"].dropna().unique()

    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=0.30, random_state=seed, shuffle=True
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=2 / 3, random_state=seed, shuffle=True
    )

    split_map = {"train": set(train_patients), "val": set(val_patients), "test": set(test_patients)}
    split_files = {
        "train": processed_root / TRAIN_SPLIT_FILENAME,
        "val": processed_root / VAL_SPLIT_FILENAME,
        "test": processed_root / TEST_SPLIT_FILENAME,
    }
    for split_name, patients in split_map.items():
        split_df = df[df["patient_id"].isin(patients)][["sample_id"]].copy()
        split_df.to_csv(split_files[split_name], index=False)

    return split_files

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

        self.c_info = {
            "names": CONCEPT_NAMES,
            "cardinality": [2] * len(CONCEPT_NAMES), 
        }
        self.y_info = {"names": [TARGET_NAME], "cardinality": [2]}
        self.modalities = ["image", "image"]
        self.data: Dict[str, "_MMISTccRCC"] = {}

        RAW_MMIST_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_MULTIMODAL_DIR.mkdir(parents=True, exist_ok=True)

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

        # Split into train/test following the split column in the CSV
        train_df = df[df['Split'] == 'train'].drop(columns=['Split'])
        test_df = df[df['Split'] == 'test'].drop(columns=['Split'])

        # Rename columns using MAPPING
        train_df = train_df.rename(columns=MAPPING)
        test_df = test_df.rename(columns=MAPPING)
        
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
            self.data[split_name] = {
                "x": torch.tensor(x_list, dtype=torch.float32),
                "ct": torch.tensor(ct_list, dtype=torch.float32),
                "wsi": torch.tensor(wsi_list, dtype=torch.float32),
                "mri": torch.tensor(mri_list, dtype=torch.float32),
                "c": torch.tensor(c_list, dtype=torch.float32),
                "y": torch.tensor(y_list, dtype=torch.float32).unsqueeze(1),
            }
        print("Finished loading data.")


    def load_ground_truth_graph(self):
        """
        Return the ground-truth causal adjacency matrix, or None if unknown.
        """
        self.adj = None
        return self.adj

    def set_active_modality(self, modality: str) -> None:
        """
        Switch all split datasets to `modality` ('image' or 'text').
        """
        if modality not in self.modalities:
            raise ValueError(f"active_modality must be one of {self.modalities}")
        self.active_modality = modality
        for split_dataset in self.data.values():
            split_dataset.set_active_modality(modality)

    def split(self, seed: Optional[int] = None) -> None:
        """
        Build the train / val / test splits and store them in self.data.

        Steps:
          1. Call build_multimodal_metadata() to produce / load the merged CSV.
          2. Call create_patient_splits() to get per-split sample_id CSVs.
          3. Instantiate one _MMISTccRCC per split and store in self.data.

        TODO: pass any additional constructor arguments that _MMISTccRCC needs.
        """
        seed = 42 if seed is None else seed

        metadata_path = build_multimodal_metadata(
            raw_root=RAW_MMIST_DIR,
            processed_root=PROCESSED_MULTIMODAL_DIR,
            seed=seed,
            force_rebuild=False,
        )
        split_files = create_patient_splits(metadata_path, PROCESSED_MULTIMODAL_DIR, seed=seed)

        for split_name, split_csv in split_files.items():
            transform = self.transform["train"] if split_name == "train" else self.transform["test"]
            self.data[split_name] = _MMISTccRCC(
                raw_root=RAW_MMIST_DIR,
                metadata_path=metadata_path,
                split_csv_path=split_csv,
                split=split_name,
                transform=transform,
                target_transform=self.target_transform,
                concept_names=self.c_info["names"],
                active_modality=self.active_modality,
            )


# ---------------------------------------------------------------------------
# Inner per-split PyTorch Dataset
# ---------------------------------------------------------------------------

class _MMISTccRCC(Dataset):
    """
    PyTorch Dataset for one split (train / val / test) of MMIST_ccRCC.

    Mirrors _CheXpertMulti from cheXpert_multi.py.

    Key attributes set after generate_multimodal_embeddings() runs:
      X_image  — (N, img_embed_dim) tensor of image embeddings
      X_text   — (N, txt_embed_dim) tensor of text embeddings
      X        — alias pointing to whichever modality is currently active
      c        — (N, n_concepts) float tensor of concept labels
      y        — (N, 1)          float tensor of task labels
    """

    def __init__(
        self,
        raw_root: Path,
        metadata_path: Path,
        split_csv_path: Path,
        split: str,
        transform=None,
        target_transform=None,
        concept_names: Optional[List[str]] = None,
        active_modality: str = "image",
        text_column: str = "text_input",
    ):
        self.raw_root = raw_root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.concept_names = concept_names or CONCEPT_NAMES
        self.active_modality = active_modality
        self.text_column = text_column
        self.graph: Dict = {}

        # Load and filter metadata to this split's sample IDs
        self.df = pd.read_csv(metadata_path)
        split_ids = pd.read_csv(split_csv_path)["sample_id"].tolist()
        self.df = self.df[self.df["sample_id"].isin(split_ids)].reset_index(drop=True)

        # Populated by generate_multimodal_embeddings()
        self.X: Optional[torch.Tensor] = None
        self.X_image: Optional[torch.Tensor] = None
        self.X_text: Optional[torch.Tensor] = None
        self.c: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

    def set_active_modality(self, modality: str) -> None:
        """
        Point self.X to the pre-computed embedding tensor for `modality`.
        Called by MMISTccRCC.set_active_modality() and by
        generate_multimodal_embeddings() after each split is processed.
        """
        self.active_modality = modality
        if modality == "image":
            self.X = self.X_image
        elif modality == "text":
            self.X = self.X_text
        else:
            raise ValueError("active_modality must be 'image' or 'text'")

    def update_lists(self) -> None:
        """
        Convert the concept and label columns of self.df into tensors.
        Must be called after generate_multimodal_embeddings() in preprocessing.py.

        TODO: verify that all CONCEPT_NAMES columns exist in self.df and contain
              integer-valued labels (0 / 1 / …). Adjust dtype if needed.
        """
        self.c = torch.tensor(
            self.df[self.concept_names].values, dtype=torch.float32
        )
        self.y = torch.tensor(
            self.df[TARGET_NAME].values, dtype=torch.float32
        ).unsqueeze(1)

    def register_graph(self, graph: Dict) -> None:
        """Store a per-client subgraph dict (set by the federated training loop)."""
        self.graph = graph

    def _load_image(self, sample: pd.Series) -> torch.Tensor:
        """
        Load and pre-process the raw image for `sample` from disk.

        Called by __getitem__ when X_image is None (i.e. before embedding
        generation) or when active_modality == 'image' during embedding gen.

        TODO:
          - Resolve the absolute image path from self.raw_root and
            sample["img_id"] (adjust the column name if different).
          - Open with PIL, apply any domain-specific pre-processing
            (e.g. stain normalisation for histopathology, windowing for CT).
          - Apply self.transform if set.
          - Return a (C, H, W) float tensor.
        """
        # TODO: adjust path construction to match the raw data layout
        img_path = self.raw_root / sample["img_id"]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _get_active_input(self, idx: int, sample: pd.Series) -> torch.Tensor:
        """
        Return the input tensor for the currently active modality.

        If embeddings have already been computed (X_image / X_text is not None),
        return the pre-computed vector; otherwise fall back to loading raw data.
        """
        if self.active_modality == "image":
            if self.X_image is not None:
                return self.X_image[idx]
            return self._load_image(sample)

        if self.active_modality == "text":
            if self.X_text is None:
                raise RuntimeError(
                    "Text modality requested before text embeddings were generated. "
                    "Run generate_multimodal_embeddings(..., generate_text=True) first."
                )
            return self.X_text[idx]

        raise ValueError(f"Unsupported active_modality: {self.active_modality}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        """
        Return a dict with keys expected by the training engine:
          x              — input embedding / raw tensor for the active modality
          c              — concept label vector
          y              — task label scalar
          graph          — per-client subgraph dict (may be empty)
          sample_id      — unique sample identifier
          text           — raw text string for this sample
          patient_id     — patient-level grouping key
          active_modality— which modality is currently active

        TODO: add any extra keys required by downstream engines or metrics.
        """
        sample = self.df.iloc[idx]
        x = self._get_active_input(idx, sample)
        concepts = torch.from_numpy(
            sample[self.concept_names].values.astype(np.float32)
        )
        label = torch.tensor(sample[TARGET_NAME], dtype=torch.float32)

        return {
            "x": x,
            "c": concepts,
            "y": label,
            "graph": self.graph,
            "sample_id": sample["sample_id"],
            "text": sample[self.text_column],
            "patient_id": sample["patient_id"],
            "active_modality": self.active_modality,
        }
