import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from typing import Union
from torch_geometric.utils import to_dense_adj
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

#---- Transformations ----
transResize = 224

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=(0, 5), translate=(0.05, 0.05), shear=5),
    transforms.RandomHorizontalFlip(),
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

    # Filter frontal and AP/PA images
    full_df = full_df[full_df["Frontal/Lateral"] == "Frontal"]
    full_df = full_df.drop(columns=["Frontal/Lateral", "AP/PA"])

    # Discretize Age
    #full_df["Age"] = full_df["Age"].apply(lambda x: 0.0 if x < 30 else (1.0 if x < 60 else 2.0))

    # Map Sex
    #full_df['Sex'] = full_df['Sex'].map({'Male': 0, 'Female': 1}) 

    # Eliminate duplicates
    subject_ids = []
    for index, row in full_df.iterrows():
        path_parts = row['Path'].split("/")
        patient_part = next(part for part in path_parts if "patient" in part)
        subject_ids.append(patient_part[len("patient"):])
    full_df['subject_id'] = subject_ids

    # Droping duplicate patient recordings except for the last visit
    full_df = full_df.drop_duplicates(subset=['subject_id'], keep='last')
    full_df = full_df.sort_values(by=['subject_id']).reset_index(drop=True)
    
    # Fill NaNs as absent, uncertain (-1) as present (U-Ones strategy from CheXpert paper)
    full_df[CONCEPT_NAMES] = full_df[CONCEPT_NAMES].fillna(0)
    full_df[CONCEPT_NAMES] = full_df[CONCEPT_NAMES].replace(-1, 1)
    # if full_df[TERGET_NAME] is 0 replace it with 1, otherwise put 0
    full_df[TARGET_NAME] = np.where(full_df[TARGET_NAME] == 1, 0, 1)

    # Extract relative path
    full_df["img_id"] = full_df["Path"].apply(lambda x: x.split("/", 1)[1] if "/" in x else x) # correct in this way it just keep the name without /train
    full_df = full_df.drop(columns=["Path"])  

    # # Balance: reduce minority to 1/3, then undersample majority to 1.5x minority
    majority = full_df[full_df[TARGET_NAME] == 1]
    minority = full_df[full_df[TARGET_NAME] == 0]

    # # Reduce minority class to 1/3
    # target_minority_size = int(len(minority) / 3)
    # minority = minority.sample(n=target_minority_size, random_state=42)
    # print(f"[CheXpert] Minority reduced: {len(full_df[full_df[TARGET_NAME] == 0])}->{len(minority)}")

    # # Reduce majority class to 1.5x minority, with stratified sampling
    target_majority_size = min(len(majority), int(len(minority)))

    if target_majority_size < len(majority):
         majority_sampled = majority.sample(n=target_majority_size, random_state=42)
         print(f"[CheXpert] Majority reduced: {len(majority)}->{len(majority_sampled)}")
    else:
         majority_sampled = majority

    full_df_balanced = pd.concat([minority, majority_sampled], ignore_index=True)
    print(f"[CheXpert] Total balanced: {len(full_df_balanced)}")

    # No balancing — use all data, handle imbalance via class weights in loss
    #full_df_all = full_df.copy()
    #n_pos_task = (full_df_all[TARGET_NAME] == 1).sum()
    #n_neg_task = (full_df_all[TARGET_NAME] == 0).sum()
    #n_total_task = n_pos_task + n_neg_task
    #print(f"[CheXpert] Using full dataset: {n_total_task} samples (target=1: {n_pos_task}, target=0: {n_neg_task})")

    full_df_all = full_df_balanced.copy()
    # Shuffle and split into train, val and test
    full_df_all = full_df_all.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(full_df_all) * (0.7))  # 70% train
    val_idx = int(len(full_df_all) * (0.8))     # 10% val, 20% test
    full_df_all.iloc[:split_idx][["img_id"]].to_csv(os.path.join(data_path, "custom_train.csv"), index=False)
    full_df_all.iloc[split_idx:val_idx][["img_id"]].to_csv(os.path.join(data_path, "custom_val.csv"), index=False)
    full_df_all.iloc[val_idx:][["img_id"]].to_csv(os.path.join(data_path, "custom_test.csv"), index=False)
    
    full_df_all.to_csv(os.path.join(data_path, "cheXpert_merged.csv"), index=False)

    # Compute task class weights (balanced: n_samples / (2 * n_class_samples))
    #import torch
    #train_df = full_df_all.iloc[:split_idx]
    #n_pos_train = (train_df[TARGET_NAME] == 1).sum()
    #n_neg_train = (train_df[TARGET_NAME] == 0).sum()
    #n_total_train = n_pos_train + n_neg_train
    #w_neg = n_total_train / (2.0 * n_neg_train)
    #w_pos = n_total_train / (2.0 * n_pos_train)
    #class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)
    #print(f"[CheXpert] Task weights: neg={w_neg:.2f} ({n_neg_train}), pos={w_pos:.2f} ({n_pos_train})")

    # Compute per-concept class weights
    #concept_class_weights = {}
    #for c_name in CONCEPT_NAMES:
    #    vals = train_df[c_name].values
    #    n_pos = (vals == 1).sum()
    #    n_neg = (vals == 0).sum()
    #    n_total = n_pos + n_neg
    #    if n_pos > 0 and n_neg > 0:
    #        c_w_neg = n_total / (2.0 * n_neg)
    #        c_w_pos = n_total / (2.0 * n_pos)
    #        concept_class_weights[c_name] = torch.tensor([c_w_neg, c_w_pos], dtype=torch.float32)
    #        print(f"[CheXpert] Concept weight {c_name}: neg={c_w_neg:.2f}, pos={c_w_pos:.2f} (pos_rate={n_pos/n_total*100:.1f}%)")
    class_weights = None
    concept_class_weights = None
    return class_weights, concept_class_weights

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
