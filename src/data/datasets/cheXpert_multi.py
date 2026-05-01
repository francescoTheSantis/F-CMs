import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import PIL
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from env import CACHE
from src.data.model_weights import ensure_resnet18_imagenet_weights

try:
    from torchvision import transforms
    import torchvision.models as tv_models
    from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
except ImportError:  # pragma: no cover - depends on runtime environment
    transforms = None
    tv_models = None
    ResNet18_Weights = None
    ResNet50_Weights = None

try:
    import torchxrayvision as xrv
except ImportError:  # pragma: no cover - depends on runtime environment
    xrv = None


RAW_CHEXPERT_DIR = CACHE / "cheXpert"
PROCESSED_MULTIMODAL_DIR = CACHE / "cheXpert_multi"
AUXILIARY_DATA_URL = "https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1" # manually download and place in RAW_CHEXPERT_DIR if not using automatic download

TARGET_NAME = "No Finding"
CONCEPT_NAMES = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

REQUIRED_RAW_FILES = {
    "train.csv": "file",
    "valid.csv": "file",
    "df_chexpert_plus_240401.csv": "file",
    "chexbert_labels": "dir",
    "radgraph-XL-annotations": "dir",
}

MERGED_FILENAME = "cheXpert_multi_merged.csv"
STATS_FILENAME = "cheXpert_multi_stats.json"
TRAIN_SPLIT_FILENAME = "custom_train_multimodal.csv"
VAL_SPLIT_FILENAME = "custom_val_multimodal.csv"
TEST_SPLIT_FILENAME = "custom_test_multimodal.csv"

TEXT_SOURCE_OPTIONS = {
    "report",
    "impression",
    "findings",
    "impression_findings",
    "findings_impression",
}

transResize = 224



def _balance_task_classes(
    df: pd.DataFrame,
    target_name: str = TARGET_NAME,
    seed: int = 42,
):
    class_counts_before = {
        int(label): int(count)
        for label, count in df[target_name].value_counts().sort_index().items()
    }

    if len(class_counts_before) < 2:
        return df.reset_index(drop=True), {
            "applied": False,
            "reason": "single_class_only",
            "class_counts_before": class_counts_before,
            "class_counts_after": class_counts_before,
        }

    majority_label = max(class_counts_before, key=class_counts_before.get)
    minority_label = min(class_counts_before, key=class_counts_before.get)
    majority_df = df[df[target_name] == majority_label]
    minority_df = df[df[target_name] == minority_label]

    target_majority_size = min(len(majority_df), len(minority_df))
    if target_majority_size < len(majority_df):
        majority_sampled = majority_df.sample(n=target_majority_size, random_state=seed)
    else:
        majority_sampled = majority_df.copy()

    balanced_df = pd.concat([minority_df, majority_sampled], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    class_counts_after = {
        int(label): int(count)
        for label, count in balanced_df[target_name].value_counts().sort_index().items()
    }

    return balanced_df, {
        "applied": True,
        "majority_label": int(majority_label),
        "minority_label": int(minority_label),
        "class_counts_before": class_counts_before,
        "class_counts_after": class_counts_after,
        "rows_removed": int(len(df) - len(balanced_df)),
    }

def _balance_task_classes_by_patient(df, target_name=TARGET_NAME, seed=42):
    patient_labels = (
        df.groupby("patient_id")[target_name]
        .max()
        .reset_index()
    )

    patient_counts_before = {
        int(label): int(count)
        for label, count in patient_labels[target_name].value_counts().sort_index().items()
    }
    image_counts_before = {
        int(label): int(count)
        for label, count in df[target_name].value_counts().sort_index().items()
    }
    if len(patient_counts_before) < 2:
        return df.reset_index(drop=True), {
            "applied": False,
            "reason": "single_class_only",
            "patient_counts_before": patient_counts_before,
            "patient_counts_after": patient_counts_before,
            "image_counts_before": image_counts_before,
            "image_counts_after": image_counts_before,
        }

    min_patient_count = min(patient_counts_before.values())

    sampled_patients = (
        patient_labels
        .groupby(target_name, group_keys=False)
        .apply(lambda x: x.sample(n=min_patient_count, random_state=seed))
        .reset_index(drop=True)
    )

    patient_balanced_df = df[df["patient_id"].isin(sampled_patients["patient_id"])]
    image_counts_after_patient_balance = {
        int(label): int(count)
        for label, count in patient_balanced_df[target_name].value_counts().sort_index().items()
    }
    min_image_count = min(image_counts_after_patient_balance.values())

    balanced_df = (
        patient_balanced_df
        .groupby(target_name, group_keys=False)
        .apply(lambda x: x.sample(n=min_image_count, random_state=seed))
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    patient_counts_after = {
        int(label): int(count)
        for label, count in sampled_patients[target_name].value_counts().sort_index().items()
    }
    image_counts_after = {
        int(label): int(count)
        for label, count in balanced_df[target_name].value_counts().sort_index().items()
    }

    return balanced_df, {
        "applied": True,
        "unit": "patient_then_image",
        "patient_counts_before": patient_counts_before,
        "patient_counts_after": patient_counts_after,
        "image_counts_before": image_counts_before,
        "image_counts_after_patient_balance": image_counts_after_patient_balance,
        "image_counts_after": image_counts_after,
        "patients_removed": int(patient_labels["patient_id"].nunique() - sampled_patients["patient_id"].nunique()),
        "rows_removed": int(len(df) - len(balanced_df)),
    }



def _require_torchvision() -> None:
    if transforms is None or tv_models is None:
        raise ImportError(
            "torchvision is required for CheXpert multimodal image loading/embedding generation."
        )


train_transform = None
test_transform = None
if transforms is not None:
    train_transform = transforms.Compose(
        [
            #transforms.RandomAffine(degrees=(0, 5), translate=(0.05, 0.05), shear=5),
            #transforms.RandomHorizontalFlip(),
            transforms.Resize(transResize),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(transResize),
            transforms.ToTensor(),
        ]
    )


class Identity(nn.Module):
    def forward(self, x):
        return x


class InputImgEncoder(nn.Module):
    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.weights_name = getattr(original_model, "weights", None)
        if self.weights_name == "densenet121-res224-all":
            self.features = original_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.weights_name == "densenet121-res224-all":
            x = self.avgpool(x)
        return torch.flatten(x, 1)


def _normalize_rel_path(path: str) -> str:
    path = str(path)
    return re.sub(r"^CheXpert-v1\.0-small/", "", path)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _clean_text(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _compose_primary_text(row: pd.Series, text_source: str) -> Optional[str]:
    report = row["report_text"]
    impression = row["section_impression_text"]
    findings = row["section_findings_text"]

    if text_source == "report":
        return report or impression or findings
    if text_source == "impression":
        return impression or findings or report
    if text_source == "findings":
        return findings or impression or report
    if text_source == "impression_findings":
        parts = [p for p in [impression, findings] if p]
        return " ".join(parts) if parts else report
    if text_source == "findings_impression":
        parts = [p for p in [findings, impression] if p]
        return " ".join(parts) if parts else report
    raise ValueError(f"Unsupported text source: {text_source}")


def _load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle]


def _load_chexbert_labels(raw_root: Path) -> pd.DataFrame:
    merged = None
    variants = {
        "report": raw_root / "chexbert_labels" / "report_fixed.json",
        "impression": raw_root / "chexbert_labels" / "impression_fixed.json",
        "findings": raw_root / "chexbert_labels" / "findings_fixed.json",
    }
    for variant, path in variants.items():
        variant_df = pd.DataFrame.from_records(_load_jsonl(path))
        variant_df["image_path_rel"] = variant_df["path_to_image"].map(_normalize_rel_path)
        renamed_columns = {}
        for column in variant_df.columns:
            if column in {"path_to_image", "image_path_rel"}:
                continue
            renamed_columns[column] = f"chexbert_{variant}_{_slugify(column)}"
        variant_df = variant_df.rename(columns=renamed_columns).drop(columns=["path_to_image"])
        merged = variant_df if merged is None else merged.merge(variant_df, on="image_path_rel", how="outer")
    return merged


def _validate_required_files(raw_root: Path) -> None:
    missing = []
    for name, expected_type in REQUIRED_RAW_FILES.items():
        path = raw_root / name
        if expected_type == "file" and not path.is_file():
            missing.append(name)
        elif expected_type == "dir" and not path.is_dir():
            # if there is the same directory with a .zip extension, consider it as present and unzip it
            zip_path = raw_root / f"{name}.zip"
            if zip_path.is_file():
                # voglio che mi crei una cartella che si chiama come name e che ci metta dentro i file che ci sono dentro il zip, se c'è già una cartella con quel nome allora mi deve sovrascrivere i file al suo interno
                #shutil.rmtree(raw_root / name, ignore_errors=True)
                shutil.unpack_archive(str(zip_path), str(raw_root / name))
                continue
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            "CheXpert multimodal preprocessing requires the Kaggle CheXpert-small files plus the "
            "CheXpert+ additions in the same directory.\n"
            f"Missing under {raw_root}: {missing}\n"
            f"Download the auxiliary files from: {AUXILIARY_DATA_URL}"
        )


def download_base_data(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "train.csv").exists() and (destination / "valid.csv").exists():
        return

    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "kagglehub is required to download the base CheXpert-small dataset automatically. "
            "Install it or place train.csv/valid.csv manually in the cache directory."
        ) from exc

    downloaded_path = Path(kagglehub.dataset_download("ashery/chexpert"))
    for item in downloaded_path.iterdir():
        shutil.move(str(item), str(destination / item.name))


def build_multimodal_metadata(
    raw_root: Path = RAW_CHEXPERT_DIR,
    processed_root: Path = PROCESSED_MULTIMODAL_DIR,
    text_source: str = "report",
    seed: int = 42,
    force_rebuild: bool = False,
) -> Path:
    if text_source not in TEXT_SOURCE_OPTIONS:
        raise ValueError(f"text_source must be one of {sorted(TEXT_SOURCE_OPTIONS)}")

    processed_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    metadata_path = processed_root / MERGED_FILENAME
    stats_path = processed_root / STATS_FILENAME

    download_base_data(raw_root)
    _validate_required_files(raw_root)

    if metadata_path.exists() and not force_rebuild:
        return metadata_path

    train_df = pd.read_csv(raw_root / "train.csv")
    valid_df = pd.read_csv(raw_root / "valid.csv")
    reports_df = pd.read_csv(raw_root / "df_chexpert_plus_240401.csv")

    images_df = pd.concat([train_df, valid_df], ignore_index=True)
    images_df["image_path_rel"] = images_df["Path"].map(_normalize_rel_path)
    images_df["img_id"] = images_df["image_path_rel"]
    images_df["patient_id"] = images_df["image_path_rel"].str.extract(r"(patient\d+)")
    images_df["study_id"] = images_df["image_path_rel"].str.extract(r"(study\d+)")
    images_df["view_id"] = images_df["image_path_rel"].str.extract(r"(view\d+_[^/]+\.jpg)")
    images_df["source_split"] = images_df["image_path_rel"].str.extract(r"^(train|valid)")

    reports_df["image_path_rel"] = reports_df["path_to_image"].map(_normalize_rel_path)
    reports_df["report_text"] = reports_df["report"].map(_clean_text)
    reports_df["section_findings_text"] = reports_df["section_findings"].map(_clean_text)
    reports_df["section_impression_text"] = reports_df["section_impression"].map(_clean_text)
    reports_df["text_input"] = reports_df.apply(lambda row: _compose_primary_text(row, text_source), axis=1)

    report_columns = [
        "image_path_rel",
        "report_text",
        "section_findings_text",
        "section_impression_text",
        "text_input",
        "split",
    ]
    merged_df = images_df.merge(reports_df[report_columns], on="image_path_rel", how="left")
    merged_df = merged_df.rename(columns={"split": "report_split"})

    chexbert_df = _load_chexbert_labels(raw_root)
    merged_df = merged_df.merge(chexbert_df, on="image_path_rel", how="left")

    rows_before_text_filter = len(merged_df)
    merged_df = merged_df[merged_df["text_input"].notna()].reset_index(drop=True)

    merged_df[CONCEPT_NAMES] = merged_df[CONCEPT_NAMES].fillna(0)
    merged_df[CONCEPT_NAMES] = merged_df[CONCEPT_NAMES].replace(-1, 0)
    merged_df[TARGET_NAME] = np.where(merged_df[TARGET_NAME] == 1, 0, 1)
    merged_df["sample_id"] = merged_df["img_id"]
    #merged_df, balancing_stats = _balance_task_classes(merged_df, target_name=TARGET_NAME, seed=seed)

    merged_df.to_csv(metadata_path, index=False)

    stats = {
        "text_source": text_source,
        "rows_before_text_filter": int(rows_before_text_filter),
        "rows_after_text_filter": int(len(merged_df)),
        "dropped_missing_text_rows": int(rows_before_text_filter - len(merged_df)),
        "unique_patients": int(merged_df["patient_id"].nunique()),
        "unique_studies": int(merged_df[["patient_id", "study_id"]].drop_duplicates().shape[0]),
        "multi_view_studies": int(merged_df.groupby(["patient_id", "study_id"]).size().gt(1).sum()),
        "chexbert_columns": [column for column in merged_df.columns if column.startswith("chexbert_")],
        "radgraph_status": "validated_only_not_merged_yet"
    #    "task_balancing": None,
    }
    
    with open(stats_path, "w") as handle:
        json.dump(stats, handle, indent=2)

    return metadata_path


def create_patient_splits(
    metadata_path: Path,
    processed_root: Path = PROCESSED_MULTIMODAL_DIR,
    seed: int = 42,
) -> Dict[str, Path]:
    
    stats_path = processed_root / STATS_FILENAME
    processed_root.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(metadata_path)
    

    # stratified split

    #unique_patients = df["patient_id"].dropna().unique()
    #train_patients, temp_patients = train_test_split(
    #    unique_patients, test_size=0.30, random_state=seed, shuffle=True
    #)
    #val_patients, test_patients = train_test_split(
    #    temp_patients, test_size=2 / 3, random_state=seed, shuffle=True
    #)
    patient_labels = (
        df.groupby("patient_id")[TARGET_NAME]
        .max()
        .reset_index()
    )

    train_patients, temp_patients = train_test_split(
        patient_labels["patient_id"],
        test_size=0.30,
        random_state=seed,
        shuffle=True,
        stratify=patient_labels[TARGET_NAME],
    )

    temp_labels = patient_labels[
        patient_labels["patient_id"].isin(temp_patients)
    ]

    val_patients, test_patients = train_test_split(
        temp_labels["patient_id"],
        test_size= 1/2,
        random_state=seed,
        shuffle=True,
        stratify=temp_labels[TARGET_NAME],
    )

    # split dataframes based on patient splits
    train_df = df[df["patient_id"].isin(train_patients)].copy()
    val_df = df[df["patient_id"].isin(val_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()


    # balance training set
    train_df_balanced, train_balancing_stats = _balance_task_classes_by_patient(
        train_df,
        target_name=TARGET_NAME,
        seed=seed,
    )

    # balance val set
    val_df_balanced, val_balancing_stats = _balance_task_classes_by_patient(
        val_df,
        target_name=TARGET_NAME,
        seed=seed,
    )

    #save the splits

    #re-built the dataframe 
    new_df = pd.concat(
        [train_df_balanced, val_df_balanced, test_df],
        ignore_index=True,
    )

    print(
        f"Train (balanced): {len(train_df_balanced)} | "
        f"Val (balanced): {len(val_df_balanced)} | Test: {len(test_df)} | "
        f"Total: {len(new_df)}"
    )

    #split_map = {
    #    "train": set(train_patients),
    #    "val": set(val_patients),
    #    "test": set(test_patients),
    #}

    split_files = {
        "train": processed_root / TRAIN_SPLIT_FILENAME,
        "val": processed_root / VAL_SPLIT_FILENAME,
        "test": processed_root / TEST_SPLIT_FILENAME,
    }

    train_df_balanced['sample_id'].to_csv(split_files["train"], index=False)
    val_df_balanced['sample_id'].to_csv(split_files["val"], index=False)
    test_df['sample_id'].to_csv(split_files["test"], index=False)

    # overwrite metadata
    new_df.to_csv(metadata_path, index=False)

    # update stats with balancing info
    with open(stats_path, "r") as handle:
        stats = json.load(handle)
    stats["task_balancing"] = {
        "train_balancing": train_balancing_stats,
        "val_balancing": val_balancing_stats,
    }
    with open(stats_path, "w") as handle:
        json.dump(stats, handle, indent=2)


    return split_files


def _build_image_encoder(backbone: str, device: str) -> nn.Module:
    _require_torchvision()

    if backbone == "resnet18":
        input_encoder = tv_models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif backbone == "resnet18_cxr":
        model_directory = Path(__file__).resolve().parents[2] / "models"
        weights_path = ensure_resnet18_imagenet_weights(model_directory)
        input_encoder_res = tv_models.resnet18(weights=None)
        input_encoder_res.load_state_dict(
            torch.load(
                weights_path,
                map_location="cpu",
                weights_only=False,
            )
        )
        n_features = input_encoder_res.fc.in_features
        projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, 256, bias=False),
        )
        input_encoder_res.fc = Identity()
        input_encoder = nn.Sequential(input_encoder_res, projector)
    elif backbone == "resnet50":
        input_encoder = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)
    elif backbone == "res224-all":
        if xrv is None:
            raise ImportError("torchxrayvision is required for the res224-all backbone.")
        input_encoder = xrv.models.DenseNet(weights="densenet121-res224-all")
    elif backbone == "densenet121":
        input_encoder = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.DEFAULT)
        input_encoder.classifier = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported image backbone: {backbone}")

    model = InputImgEncoder(input_encoder).to(device)
    model.eval()
    return model


def _mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _pool_text_features(
    hidden_state: torch.Tensor,
    tokenized_batch: Dict[str, torch.Tensor],
    pooling: str,
) -> torch.Tensor:
    if pooling == "cls":
        return hidden_state[:, 0]
    if pooling == "mean":
        return _mean_pool(hidden_state, tokenized_batch["attention_mask"])
    raise ValueError(f"Unsupported text pooling: {pooling}")


def _generate_image_embeddings(split_dataset, model, batch_size: int, device: str) -> torch.Tensor:
    original_modality = split_dataset.active_modality
    split_dataset.set_active_modality("image")

    data_loader = DataLoader(split_dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"{split_dataset.split}-image-emb"):
            images = batch["x"].to(device)
            embeddings.append(model(images).cpu())

    split_dataset.set_active_modality(original_modality)
    return torch.cat(embeddings, dim=0)


def _generate_text_embeddings(
    split_dataset,
    tokenizer,
    model,
    batch_size: int,
    device: str,
    text_column: str,
    max_length: int,
    pooling: str,
) -> torch.Tensor:
    texts = split_dataset.df[text_column].tolist()
    embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=f"{split_dataset.split}-text-emb"):
            batch_text = texts[start : start + batch_size]
            tokenized = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            hidden_state = model(**tokenized)["last_hidden_state"]
            embeddings.append(_pool_text_features(hidden_state, tokenized, pooling).cpu())

    return torch.cat(embeddings, dim=0)


def generate_multimodal_embeddings(
    dataset,
    device: str = "cpu",
    image_backbone: str = "resnet18_cxr",
    text_backbone: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
    text_pooling: str = "cls",
    image_batch_size: int = 32,
    text_batch_size: int = 8,
    text_max_length: int = 512,
    text_column: str = "text_input",
    generate_image: bool = True,
    generate_text: bool = True,
) -> object:
    image_model = _build_image_encoder(image_backbone, device) if generate_image else None
    if generate_text:
        tokenizer = AutoTokenizer.from_pretrained(text_backbone, cache_dir=str(CACHE / "huggingface"), trust_remote_code=True)
        text_model = AutoModel.from_pretrained(text_backbone, cache_dir=str(CACHE / "huggingface"), trust_remote_code=True).to(device)
        text_model.eval()
    else:
        tokenizer = None
        text_model = None

    for split_name, split_dataset in dataset.data.items():
        if generate_image:
            split_dataset.X_image = _generate_image_embeddings(
                split_dataset, image_model, image_batch_size, device
            )
        if generate_text:
            split_dataset.X_text = _generate_text_embeddings(
                split_dataset,
                tokenizer,
                text_model,
                text_batch_size,
                device,
                text_column,
                text_max_length,
                text_pooling,
            )
        split_dataset.set_active_modality(split_dataset.active_modality)
        dataset.data[split_name] = split_dataset

    return dataset


class CheXpertMulti:
    """
    Multimodal CheXpert dataset.

    This first-step integration keeps the existing image-only CheXpert pipeline untouched and
    prepares a paired image-report dataset with precomputed image/text embeddings.

    Current design choices:
    - samples remain image-level, with the report replicated across all views of the same study
    - concept/task supervision stays aligned with the existing image CheXpert setup
    - image/text embeddings are stored separately (`X_image`, `X_text`)
    - `active_modality` selects which embedding is exposed through `dataset.X`
    - RadGraph files are validated but not consumed yet; their format is better handled in the
      next architecture-design step
    """

    def __init__(
        self,
        train_transform=train_transform,
        test_transform=test_transform,
        target_transform=None,
        val_size: float = 0.1,
        ftune_size: float = 0.0,
        ftune_val_size: float = 0.0,
        active_modality: str = "image",
        text_source: str = "report",
    ):
        self.transform = {"train": train_transform, "test": test_transform}
        self.target_transform = target_transform
        self.val_size = val_size
        self.ftune_size = ftune_size
        self.ftune_val_size = ftune_val_size
        self.active_modality = active_modality
        self.text_source = text_source

        self.c_info = {'names': CONCEPT_NAMES, 
        'cardinality':  [2] * len(CONCEPT_NAMES)}

        self.y_info = {"names": [TARGET_NAME], "cardinality": [2]}
        self.modalities = ["image", "text"]
        self.data = {}

        RAW_CHEXPERT_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_MULTIMODAL_DIR.mkdir(parents=True, exist_ok=True)
        download_base_data(RAW_CHEXPERT_DIR) # check if raw data is present
        _validate_required_files(RAW_CHEXPERT_DIR)

    def load_ground_truth_graph(self):
        self.adj = None
        return self.adj

    def set_active_modality(self, modality: str) -> None:
        if modality not in self.modalities:
            raise ValueError(f"active_modality must be one of {self.modalities}")
        self.active_modality = modality
        for split_dataset in self.data.values():
            split_dataset.set_active_modality(modality)

    def split(self, seed: Optional[int] = None):
        seed = 42 if seed is None else seed
        metadata_path = build_multimodal_metadata(
            raw_root=RAW_CHEXPERT_DIR,
            processed_root=PROCESSED_MULTIMODAL_DIR,
            text_source=self.text_source,
            seed=seed,
            force_rebuild=True,
        )
        split_files = create_patient_splits(metadata_path, PROCESSED_MULTIMODAL_DIR, seed=seed)

        self.data["train"] = _CheXpertMulti(
            raw_root=RAW_CHEXPERT_DIR,
            metadata_path=metadata_path,
            split_csv_path=split_files["train"],
            split="train",
            transform=self.transform["train"],
            target_transform=self.target_transform,
            concept_names=self.c_info["names"],
            active_modality=self.active_modality,
        )
        self.data["val"] = _CheXpertMulti(
            raw_root=RAW_CHEXPERT_DIR,
            metadata_path=metadata_path,
            split_csv_path=split_files["val"],
            split="val",
            transform=self.transform["test"],
            target_transform=self.target_transform,
            concept_names=self.c_info["names"],
            active_modality=self.active_modality,
        )
        self.data["test"] = _CheXpertMulti(
            raw_root=RAW_CHEXPERT_DIR,
            metadata_path=metadata_path,
            split_csv_path=split_files["test"],
            split="test",
            transform=self.transform["test"],
            target_transform=self.target_transform,
            concept_names=self.c_info["names"],
            active_modality=self.active_modality,
        )


class _CheXpertMulti(Dataset):
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
        self.graph = {}

        self.df = pd.read_csv(metadata_path)
        split_ids = pd.read_csv(split_csv_path)["sample_id"].tolist()
        self.df = self.df[self.df["sample_id"].isin(split_ids)].reset_index(drop=True)

        self.X = None
        self.X_image = None
        self.X_text = None
        self.c = None
        self.y = None

    def set_active_modality(self, modality: str) -> None:
        self.active_modality = modality
        if modality == "image":
            self.X = self.X_image
        elif modality == "text":
            self.X = self.X_text
        else:
            raise ValueError("active_modality must be either 'image' or 'text'")

    def update_lists(self):
        self.c = torch.tensor(self.df[self.concept_names].values, dtype=torch.float32)
        self.y = torch.tensor(self.df[TARGET_NAME].values, dtype=torch.float32).unsqueeze(1)

    def register_graph(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.df)

    def _load_image(self, sample: pd.Series) -> torch.Tensor:
        _require_torchvision()
        img_path = self.raw_root / sample["img_id"]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        r_min = max(0, (height - width) / 2)
        r_max = min(height, (height + width) / 2)
        c_min = max(0, (width - height) / 2)
        c_max = min(width, (width + height) / 2)
        image = image.crop((c_min, r_min, c_max, r_max))
        image = image.resize((transResize, transResize))
        image = PIL.ImageOps.equalize(image)
        image = image.convert("L").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _get_active_input(self, idx: int, sample: pd.Series):
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

        raise ValueError(f"Unsupported active modality: {self.active_modality}")

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        x = self._get_active_input(idx, sample)
        concepts = torch.from_numpy(sample[self.concept_names].values.astype(np.float32))
        label = torch.tensor(sample[TARGET_NAME], dtype=torch.float32)

        return {
            "x": x,
            "c": concepts,
            "y": label,
            "graph": self.graph,
            "sample_id": sample["sample_id"],
            "img_id": sample["img_id"],
            "text": sample[self.text_column],
            "patient_id": sample["patient_id"],
            "study_id": sample["study_id"],
            "active_modality": self.active_modality,
        }
