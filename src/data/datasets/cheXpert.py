import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import Dataset
import os
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import requests
from typing import Union
from torch_geometric.utils import to_dense_adj
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from src.data.utils import split_dataset
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from env import CACHE
import PIL
from PIL import Image
from pathlib import Path
import kagglehub
import shutil

# download
CHEXPERT_DIR = CACHE / "cheXpert"
TARGET_NAME = "No Finding"
CONCEPT_NAMES = ['Enlarged Cardiomediastinum',
                 'Cardiomegaly',
                 'Lung Opacity',
                 'Lung Lesion',
                 'Edema',
                 'Consolidation',
                 'Pneumonia',
                 'Atelectasis',
                 'Pneumothorax',
                'Pleural Effusion',
                'Pleural Other',
                'Fracture',
                'Support Devices']

def _normalize_rel_path(path: str) -> str:
    path = str(path)
    return re.sub(r"^CheXpert-v1\.0-small/", "", path)

def _balance_task_classes_by_patient(df, target_name=TARGET_NAME, seed=42):
    patient_labels = (
        df.groupby("patient_id")[target_name]
        .max()
        .reset_index()
    )

    patient_counts = {
        int(label): int(count)
        for label, count in patient_labels[target_name].value_counts().sort_index().items()
    }

    if len(patient_counts) < 2:
        return df.reset_index(drop=True)

    min_patient_count = min(patient_counts.values())

    sampled_patients = (
        patient_labels
        .groupby(target_name, group_keys=False)
        .apply(lambda x: x.sample(n=min_patient_count, random_state=seed))
        .reset_index(drop=True)
    )

    patient_balanced_df = df[df["patient_id"].isin(sampled_patients["patient_id"])]
    image_counts = {
        int(label): int(count)
        for label, count in patient_balanced_df[target_name].value_counts().sort_index().items()
    }
    min_image_count = min(image_counts.values())

    balanced_df = (
        patient_balanced_df
        .groupby(target_name, group_keys=False)
        .apply(lambda x: x.sample(n=min_image_count, random_state=seed))
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    return balanced_df

#---- Transformations ----
transResize = 224

train_transform = transforms.Compose([
    #transforms.RandomAffine(degrees=(0, 5), translate=(0.05, 0.05), shear=5),
    #transforms.RandomHorizontalFlip(),
    transforms.Resize(transResize),
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(transResize),
    #transforms.CenterCrop(224),
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
])
                 

def download_data(destination):
    try:       # Check if train.csv and valid.csv are in the destination
        if not os.path.exists(os.path.join(destination, "train.csv")) or not os.path.exists(os.path.join(destination, "valid.csv")):
            print(f"Dataset not found at {destination}. Downloading...")
            base_cache = CACHE.parent
            path = kagglehub.dataset_download("ashery/chexpert")           
            path = Path(path)
            for item in path.iterdir():
                dest_path = destination / item.name
                # Se è un file o cartella, usa shutil.move
                shutil.move(str(item), str(dest_path))
        else:
            print(f"Dataset already exists at {destination}. Skipping download.")
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")  

def clean_and_split_data(data_path, seed=42):

    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "valid.csv"))
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    full_df = full_df.reset_index(drop=True)

    # Extract patient_id from path
    full_df["image_path_rel"] = full_df["Path"].map(_normalize_rel_path)
    full_df["patient_id"] = full_df["image_path_rel"].str.extract(r"(patient\d+)")

    # Fill NaNs as absent, uncertain (-1) as present (U-Ones strategy from CheXpert paper)
    full_df[CONCEPT_NAMES] = full_df[CONCEPT_NAMES].fillna(0)
    full_df[CONCEPT_NAMES] = full_df[CONCEPT_NAMES].replace(-1, 0)
    full_df[TARGET_NAME] = np.where(full_df[TARGET_NAME] == 1, 0, 1)

    # Extract relative path
    full_df["img_id"] = full_df["Path"].apply(lambda x: x.split("/", 1)[1] if "/" in x else x)
    full_df = full_df.drop(columns=["Path"])

    # --- Patient-level stratified split (70% train, 10% val, 20% test) ---
    patient_labels = (
        full_df.groupby("patient_id")[TARGET_NAME]
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

    temp_labels = patient_labels[patient_labels["patient_id"].isin(temp_patients)]

    val_patients, test_patients = train_test_split(
        temp_labels["patient_id"],
        test_size=1 / 2,
        random_state=seed,
        shuffle=True,
        stratify=temp_labels[TARGET_NAME],
    )

    train_df_split = full_df[full_df["patient_id"].isin(train_patients)].copy()
    val_df_split = full_df[full_df["patient_id"].isin(val_patients)].copy()
    test_df_split = full_df[full_df["patient_id"].isin(test_patients)].copy()

    # --- Balance train and val by patient, leave test natural ---
    train_df_balanced = _balance_task_classes_by_patient(train_df_split, seed=seed)
    val_df_balanced = _balance_task_classes_by_patient(val_df_split, seed=seed)

    print(
        f"[CheXpert] Train (balanced): {len(train_df_balanced)} | "
        f"Val (balanced): {len(val_df_balanced)} | Test: {len(test_df_split)}"
    )

    # Save merged metadata and split indices
    new_df = pd.concat([train_df_balanced, val_df_balanced, test_df_split], ignore_index=True)
    new_df.to_csv(os.path.join(data_path, "cheXpert_merged.csv"), index=False)

    train_df_balanced[["img_id"]].to_csv(os.path.join(data_path, "custom_train.csv"), index=False)
    val_df_balanced[["img_id"]].to_csv(os.path.join(data_path, "custom_val.csv"), index=False)
    test_df_split[["img_id"]].to_csv(os.path.join(data_path, "custom_test.csv"), index=False)

    return None, None

class CheXpert():
    """
    The CheXpert dataset is a large dataset of chest X-rays with labeled observations.
    The concept labels are the various observations in the X-rays.
    The task is to predict the presence of these observations.


    Attributes:
        transform: The transformations to apply to the images. Default is None.
        target_transform: The transformations to apply to the target labels. Default is None.
        ftune_size: The proportion of the test set to include in the finetuning set. Default is 0.1.
        val_size: The proportion of the training set to include in the validation set. Default is 0.1.
        coloring: A dictionary with coloring options: 'train' and 'test'.
    """
    def __init__(self, 
                 train_transform = train_transform,
                 test_transform = test_transform,
                 target_transform = None,
                 val_size: float = 0.1, # proportion of the training set to include in the validation set
                 ftune_size: float = 0., # proportion of the test set to include in the finetuning set
                 ftune_val_size: float = 0., # proportion of the finetuning set to include in the finetuning validation set
        ):

        self.transform = {'train': train_transform, 'test': test_transform}
        self.target_transform = target_transform
        self.ftune_size = ftune_size
        self.val_size = val_size
        self.ftune_val_size = ftune_val_size
        self.class_weights = None
        self.concept_class_weights = {}

        self.c_info = {'names': CONCEPT_NAMES, 
        'cardinality':  [2] * len(CONCEPT_NAMES)}

        self.y_info = {'names': [TARGET_NAME],
                       'cardinality': [2]}
        

        self.data = {}

        #files preprocessing
        os.makedirs(CHEXPERT_DIR, exist_ok=True)
        download_data(destination = CHEXPERT_DIR)


    def load_ground_truth_graph(self):
        self.adj = None
        return self.adj

    def split(self, seed=None):
        """ 
        Split the dataset into training, validation and test sets 
        """
        self.class_weights, self.concept_class_weights = clean_and_split_data(CHEXPERT_DIR, seed=seed)
        self.data['train'] = _CheXpert(root = CHEXPERT_DIR, 
                                        split = 'train', 
                                        transform = self.transform['train'],
                                        target_transform = self.target_transform,
                                        concept_names = self.c_info['names'])
        
        self.data['val'] = _CheXpert(root = CHEXPERT_DIR, 
                                split = 'val', 
                                transform = self.transform['test'],
                                target_transform = self.target_transform,
                                concept_names = self.c_info['names'])
        self.data['test'] = _CheXpert(root = CHEXPERT_DIR, 
                                      split = 'test', 
                                      transform = self.transform['test'],
                                      target_transform = self.target_transform,
                                      concept_names = self.c_info['names'])
        
        #self.data['train'], self.data['val'] = split_dataset(self.data['train'], self.val_size)
        #self.data['val'].split_type = 'val'
        if self.ftune_size !=0:
            self.data['test'], self.data['ftune'] = split_dataset(self.data['test'], self.ftune_size)
            self.data['ftune'].split_type = 'ftune'
            self.data['ftune'], self.data['ftune_val'] = split_dataset(self.data['ftune'], self.ftune_val_size)
            self.data['ftune_val'].split_type = 'ftune_val'


class _CheXpert(Dataset):
    """CheXpert Dataset.

    Args:
        root         (str): Root directory of dataset.
        csv_path     (str): Path to the metadata CSV file. Defaults to `{root}/ddi_metadata.csv`
        transform         : Function to transform and collate image input. (can use test_transform from this file) 
    """
    def __init__(self, root, split, csv_path = None, transform = None, target_transform = None, concept_names = None):    
        #assert filter >= 0.
        #assert filter <= 1.0    

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.concept_names = concept_names
        self.graph = {}

        if csv_path is None:
            csv_path = os.path.join(root, "cheXpert_merged.csv")
        self.df = pd.read_csv(csv_path)

        if self.split == 'train':
            train_images = pd.read_csv(os.path.join(root, "custom_train.csv"))["img_id"].tolist()
            self.df = self.df[self.df["img_id"].apply(lambda x: x in train_images)]
        elif self.split == 'val':
            val_images = pd.read_csv(os.path.join(root, "custom_val.csv"))["img_id"].tolist()
            self.df = self.df[self.df["img_id"].apply(lambda x: x in val_images)]
        else:
            test_images = pd.read_csv(os.path.join(root, "custom_test.csv"))["img_id"].tolist()
            self.df = self.df[self.df["img_id"].apply(lambda x: x in test_images)]

        self.df = self.df.reset_index(drop=True)

        self.X = None
        self.c = None
        self.y = None
    
    def update_lists(self):
        # Initialize the concepts and labels
        self.c = []
        self.y = []

        for i in range(len(self.df)):
            out = self.__getitem__(i)
            self.c.append(out['c'])
            self.y.append(out['y'])

        # concatena in tensori
        self.c = torch.cat([c.unsqueeze(0) for c in self.c], dim=0)
        self.y = torch.tensor(self.df[TARGET_NAME].values, dtype=torch.float32).unsqueeze(1)

    def register_graph(self, graph):
        self.graph = graph

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.df)

    def __getitem__(self, idx):
        # Get sample
        sample = self.df.iloc[idx]

        # Get image ID
        img_id = sample["img_id"]

        # Get image path
        img_path = os.path.join(CHEXPERT_DIR, img_id)
        # open image
        if self.X is None:
            image = Image.open(img_path).convert("RGB")
            width, height = image.size
            r_min = max(0,(height-width)/2)
            r_max = min(height,(height+width)/2)
            c_min = max(0,(width-height)/2)
            c_max = min(width,(width+height)/2)
            image = image.crop((c_min,r_min,c_max,r_max))

            image = image.resize((transResize,transResize))
            image = PIL.ImageOps.equalize(image)
            image = image.convert('L').convert('RGB')
            if self.transform:
                image = self.transform(image) 
        else:
            image = self.X[idx]

        # Get skin concepts
        concepts = torch.from_numpy(sample[self.concept_names].values.astype(np.float32))

        # Get the target label
        label = torch.tensor(sample[TARGET_NAME], dtype=torch.float32)

        return {"x": image,  "c": concepts, "y": label, "graph": self.graph}