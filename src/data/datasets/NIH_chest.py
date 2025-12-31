from env import CACHE

import torch
import torchvision
from torchvision.transforms import Compose
from torchvision import transforms
import numpy as np
import pandas as pd
import torchxrayvision as xrv
from typing import Union
from skimage.io import imread
import matplotlib.pyplot as plt

from src.data.utils import split_dataset
import shutil
import os
import zipfile
import re
import ast
from PIL import Image


"""
NIH-chest Dataset Loader
** THIS DATASET NEEDS TO BE DOWNLOADED BEFORE BEING ABLE TO USE THE LOADER **

################################################################################
## DOWNLOAD INSTRUCTIONS
################################################################################

There are two files to download:

1) The original NIH Chest X-ray dataset (images + labels):
https://www.kaggle.com/datasets/nih-chest-xrays/data
You need to download the .zip file and put it in the directory specified in the code below

2) The file that contains the concept annotations:
https://huggingface.co/datasets/ttumyche/CheXStruct
You need to download the directory nih_cxr14 containing the .csv files and put it in the directory specified in the code below


################################################################################
"""
### PATHS TO SAVE THE DATA ###
DATA_DIRECTORY = CACHE / "NIH_chest"/ "raw_data"
DATA_DIRECTORY.mkdir(exist_ok=True, parents=True)


### TABULAR DATA DEFINITION ###
# data preparation: dictionary to change columns names of "descending_aorta_tortuous.csv" to not have overlapping names
DESCENDING_AORTA_TORTUOUS_TABULAR_DATA_DICTIONARY = {
    "point_1": "point_a1",
    "point_2": "point_a2",
    "point_3": "point_a3",
    "point_4": "point_a4",
    "point_5": "point_a5",
    "point_6": "point_a6"
}

# raw tabular data columns = Anatomical landmarks directly extracted from the radiological images, they still need to be preprocessed to have one column for each landmark
RAW_TABULAR_DATA_COLUMNS = {"heart_xmin", "heart_xmax", "lung_xmin", "lung_xmax",
                        "mediastinum_xmin", "mediastinum_xmax",
                        "mediastinum_lung_xmin", "mediastinum_lung_xmax",
                        "point_c1", "point_c2", "point_c3",
                        "point_t1", "point_t2", "point_t3", "point_t4", "point_t5", "point_t6","point_t7", "point_t8", "point_t9", 
                        "direction_per_pnt",
                        "aortic_knob_xmin", "aortic_knob_xmax", "trachea_point_left", "trachea_point_right",
                        "heart_point", "trachea_point",
                        "desc_aorta_point_left","desc_aorta_point_right",
                        "descending_aorta_tortuous",
                        "point_a1", "point_a2", "point_a3", "point_a4", "point_a5", "point_a6"
                        }

#TABULAR_DATA_COLUMNS = 


### CONCEPTS AND TASK DEFINITION ###
# concepts considered in the CBMs models = Diagnostic indices
# they correspond to the labels of each .csv file with the corresponding name
TOTAL_CONCEPTS = {"Patient Gender",
            "cardiomegaly",
            "mediastinal_widening",
            "carina_angle",
            "trachea_deviation",
            "aortic_knob_enlargement",
            "ascending_aorta_enlargement",
            "descending_aorta_enlargement",
            "descending_aorta_tortuous"
            }

CONCEPTS_FOR_IMAGES_MODALITY = TOTAL_CONCEPTS - {"Patient Gender"}
CONCEPTS_FOR_TABULAR_MODALITY = TOTAL_CONCEPTS

# data preparation: dictionary used to create a single multilabel task from the 14 disease categories
TASK_DICTIONARY = {'Atelectasis': 0, 
 'Cardiomegaly': 1, 
 'Effusion': 2, 
 'Infiltration': 3, 
 'Mass': 4, 
 'Nodule': 5, 
 'Pneumonia': 6, 
 'Pneumothorax': 7, 
 'Consolidation': 8, 
 'Edema': 9, 
 'Emphysema': 10,
 'Fibrosis': 11,
 'Pleural_Thickening': 12,
 'Hernia': 13,
 "No Finding": 14
}

def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024].

    Call xrv.utils.normalize moving forward.
    """
    return xrv.utils.normalize(img, maxval, reshape)

def process_diagnosis(diagnosis_str):
    """
    Process diagnosis string: if multiple diagnoses separated by |, randomly pick one.
    Then replace diagnosis name with corresponding number from TASK_DICTIONARY.
    """
    return TASK_DICTIONARY.get(diagnosis_str, np.nan)

def preprocess_tabular_data(df):
    """
    Preprocess tabular data in the dataframe:
    a) Process direction_per_pnt: split into separate columns for each point with values 0 (flat), 1 (right), 2 (left)
    b) Split point columns into _x and _y components
    c) Process Patient Gender: M->0, F->1
    d) Merge duplicate _x and _y columns by taking their mean
    e) Replace True/False with 1/0
    """
    # Replace True/False with 1/0
    df = df.replace({True: 1, False: 0})
    

    # a) Process direction_per_pnt: split it into separate columns for each point
    if 'direction_per_pnt' in df.columns:
        # Map direction values: flat->0, right->1, left->2
        direction_mapping = {'flat': 0, 'right': 1, 'left': 2}
        
        # Extract all directions at once using apply
        def extract_directions(val):
            if pd.notna(val):
                directions = re.findall(r"'([^']*)'", val)
                return [direction_mapping.get(d, np.nan) for d in directions]
            return []
        
        directions_series = df['direction_per_pnt'].apply(extract_directions)
        
        # Find max number of directions to create columns
        max_directions = directions_series.apply(len).max() if len(directions_series) > 0 else 0
        
        # Create all direction columns at once
        for i in range(1, int(max_directions) + 1):
            col_name = f'direction_point_{i}_trachea'
            df[col_name] = directions_series.apply(lambda x: x[i-1] if len(x) >= i else np.nan)
        
        df = df.drop(columns=['direction_per_pnt'])
    
    # b) Split point columns into _x and _y components and not direction_point_{i}_trachea
    point_cols = [col for col in df.columns if "point" in col and not col.endswith('_x') and not col.endswith('_y') and not col.startswith('direction_point_')]
    
    for col in point_cols:
        if col in df.columns:  # Check if column still exists
            # Use vectorized string operations for faster processing
            # Remove brackets and split by comma
            temp = df[col].astype(str).str.strip('()[]').str.split(',', expand=True)
            if temp.shape[1] >= 2:
                df[f'{col}_x'] = pd.to_numeric(temp[0], errors='coerce')
                df[f'{col}_y'] = pd.to_numeric(temp[1], errors='coerce')
            df = df.drop(columns=[col])
    
    # c) Process Patient Gender: M->0, F->1
    if 'Patient Gender' in df.columns:
        df['Patient Gender'] = df['Patient Gender'].apply(lambda x: 0 if x == 'M' else 1)
    
    return df

class NIHChestDataset():
    """
    The NIH Chest dataset is a collection of chest X-ray images with associated labels.
    The concept labels are the disease categories present in the images.
    The task is to predict the presence of diseases based on the X-ray images.

    Attributes:
        transform: The transformations to apply to the images. Default is None.
        target_transform: The transformations to apply to the target labels. Default is None.
        ftune_size: The proportion of the test set to include in the finetuning set. Default is 0.1.
        val_size: The proportion of the training set to include in the validation set. Default is 0.1.
        coloring: A dictionary with coloring options: 'train' and 'test'.
    """

    def __init__(self, 
                 modality: str = 'image', # 'image' or 'text
                 transform: Union[Compose, torch.nn.Module] = None,
                 target_transform: Union[Compose, torch.nn.Module] = None, 
                 val_size: float = 0.1, # proportion of the training set to include in the validation set
                 ftune_size: float = 0., # proportion of the test set to include in the finetuning set
                 ftune_val_size: float = 0., # proportion of the finetuning set to include in the finetuning validation set
                 coloring: dict = {'train': {'mode': 'custom', 'kwargs': {'custom_digits': [1,3,5,7]}}, 
                                   'test' : {'mode': 'random', 'kwargs': {'random_prob': 0.5}}}):
        
        # check if data have been downloaded
        if not os.path.exists(DATA_DIRECTORY / "archive.zip"):
            raise FileNotFoundError(f"Please download the NIH Chest X-ray dataset from https://www.kaggle.com/datasets/nih-chest-xrays/data and place the archive.zip file in {DATA_DIRECTORY}")
        if not os.path.exists(DATA_DIRECTORY / "nih_cxr14"):
            raise FileNotFoundError(f"Please download the concept annotations nih_cxr14 from https://huggingface.co/datasets/ttumyche/CheXStruct and place the nih_cxr14 directory in {DATA_DIRECTORY}")

        # unzip archive.zip if not already done
        extract_to = str(str(DATA_DIRECTORY) + "/original_nih_chest")
        if not os.path.exists(extract_to):         
            with zipfile.ZipFile(str(DATA_DIRECTORY) + "/archive.zip", 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    try:
                        zip_ref.extract(file_info, extract_to)
                    except zipfile.BadZipFile:
                        print(f"Corrupted file skipped: {file_info.filename}")
            
            # put all the images in a single folder
            if not os.path.exists(os.path.join(DATA_DIRECTORY, "original_nih_chest", "images")):
                os.makedirs(os.path.join(DATA_DIRECTORY, "original_nih_chest", "images"))
                for i in range(2,13):
                    folder_name = f"images_{i:03d}"
                    folder_path = os.path.join(extract_to, folder_name, 'images')
                    for filename in os.listdir(folder_path):
                        shutil.move(os.path.join(folder_path, filename), os.path.join(DATA_DIRECTORY, "original_nih_chest", "images"))
                    shutil.rmtree(folder_path)
                    shutil.rmtree(os.path.join(extract_to, folder_name))  
                


            # create a subfolder for the radiological findings
            os.makedirs(str(DATA_DIRECTORY) + "/nih_cxr14/radiological_findings", exist_ok=True)
            shutil.move(os.path.join(DATA_DIRECTORY, "nih_cxr14", "cardiomegaly.csv"), str(DATA_DIRECTORY) + "/nih_cxr14/radiological_findings")
            shutil.move(os.path.join(DATA_DIRECTORY, "nih_cxr14", "mediastinal_widening.csv"), str(DATA_DIRECTORY) + "/nih_cxr14/radiological_findings")
            shutil.move(os.path.join(DATA_DIRECTORY, "nih_cxr14", "carina_angle.csv"), str(DATA_DIRECTORY) + "/nih_cxr14/radiological_findings")
            shutil.move(os.path.join(DATA_DIRECTORY, "nih_cxr14", "trachea_deviation.csv"), str(DATA_DIRECTORY) + "/nih_cxr14/radiological_findings")
            shutil.move(os.path.join(DATA_DIRECTORY, "nih_cxr14", "aortic_knob_enlargement.csv"), str(DATA_DIRECTORY) + "/nih_cxr14/radiological_findings")
            shutil.move(os.path.join(DATA_DIRECTORY, "nih_cxr14", "ascending_aorta_enlargement.csv"), str(DATA_DIRECTORY) + "/nih_cxr14/radiological_findings")
            shutil.move(os.path.join(DATA_DIRECTORY, "nih_cxr14", "descending_aorta_enlargement.csv"), str(DATA_DIRECTORY) + "/nih_cxr14/radiological_findings")
            shutil.move(os.path.join(DATA_DIRECTORY, "nih_cxr14", "descending_aorta_tortuous.csv"), str(DATA_DIRECTORY) + "/nih_cxr14/radiological_findings")
        else:
            print("Files already extracted.")

        self.modality = modality
        self.tabular_columns = []
        self.transform = transform
        self.target_transform = target_transform
        self.ftune_size = ftune_size
        self.val_size = val_size
        self.ftune_val_size = ftune_val_size

        if modality not in ['image', 'tabular']:
            raise ValueError("Modality must be either 'image' or 'tabular'")
        elif modality == 'image':
            self.c_info = {'names': CONCEPTS_FOR_IMAGES_MODALITY,   
            'cardinality': [2,2,2,2,2,2,2,2]} # 8 concepts
        else:
            self.c_info = {'names': CONCEPTS_FOR_TABULAR_MODALITY,
            'cardinality': [2,2,2,2,2,2,2,2,2]} # 9 concepts


        self.y_info = {'names': ['diagnosis'], 'cardinality': [15]} # 14 disease categories + No Finding

        self.data = {}

    def load_ground_truth_graph(self):
        raise NotImplementedError("There is no ground truth graph for the NIH-chest dataset.")


    def split(self):
        """ 
        Split the dataset into training, validation and test sets 
        """
        self.data['train'] = _NIH_chest(root = str(CACHE / "NIH_chest"), 
                                        modality = self.modality,
                                        tabular_columns= self.tabular_columns,
                                        concepts_names= self.c_info['names'],
                                        task_names = self.y_info['names'],
                                        train = True)
        self.data['test'] = _NIH_chest(root = str(CACHE / "NIH_chest"),
                                       modality = self.modality,
                                       tabular_columns= self.tabular_columns,
                                       concepts_names= self.c_info['names'],
                                       task_names = self.y_info['names'],
                                       train = False)
        self.data['train'], self.data['val'] = split_dataset(self.data['train'], self.val_size)
        self.data['val'].split_type = 'val'
        if self.ftune_size !=0:
            self.data['test'], self.data['ftune'] = split_dataset(self.data['test'], self.ftune_size)
            self.data['ftune'].split_type = 'ftune'
            self.data['ftune'], self.data['ftune_val'] = split_dataset(self.data['ftune'], self.ftune_val_size)
            self.data['ftune_val'].split_type = 'ftune_val'


class _NIH_chest():
    def __init__(self, 
                 root: str,
                 modality: str, # 'image' or 'text
                 tabular_columns: list,
                 concepts_names: list,
                 task_names: list,
                 train: bool = False):
        
        self.modality = modality
        self.split_type = 'train' if train else 'test'
        self.graph = {}
        self.root = root
        self.concepts_names = concepts_names
        self.task_names = task_names
        self.tabular_columns = tabular_columns

        
        ### LOAD DATAFRAME (TRAIN, TEST) ###
        # load images ids for train and test
        if self.split_type == 'train':
           images = pd.read_csv(DATA_DIRECTORY / "original_nih_chest/train_val_list.txt", header=None, names=["img_id"])
        else:
           images = pd.read_csv(DATA_DIRECTORY / "original_nih_chest/test_list.txt", header=None, names=["img_id"])          
        self.df = images.copy()

        # clean images
        self.df = self.df[self.df['img_id'] != '00005299_000.png'].reset_index(drop=True)

        ### ADD TASK TO THE DATAFRAME ###
        # load the task labels
        labels = pd.read_csv(DATA_DIRECTORY / "original_nih_chest/Data_Entry_2017.csv")
        labels = labels.rename(columns={'Image Index': 'img_id', 'Finding Labels': 'diagnosis'}) 
        self.df = pd.merge(self.df, labels, on='img_id', how='left')
        # keep only img_id, diagnosis (task_names) and concepts_names columns
        columns_to_keep = ['img_id'] + list(self.task_names) + list(self.concepts_names)
        self.df = self.df[[col for col in columns_to_keep if col in self.df.columns]]
        
        # Filter: keep only samples with single diagnosis (no "|" character)
        # Note: checked that all the diagnoses are still represented in the training set after this filtering
        self.df = self.df[~self.df['diagnosis'].str.contains('|', regex=False, na=False)].reset_index(drop=True)
        print(f"After filtering for single diagnoses: {len(self.df)} samples")

        
        # Replace diagnosis name with corresponding number from TASK_DICTIONARY
        self.df['diagnosis'] = self.df['diagnosis'].apply(lambda x: TASK_DICTIONARY.get(x, np.nan) if pd.notna(x) else np.nan)

        # eliminate rows with NaN diagnosis
        self.df = self.df.dropna(subset=['diagnosis']).reset_index(drop=True)

        ### ADD CONCEPTS TO THE DATAFRAME ###
        # load the concepts
        for file in os.listdir(DATA_DIRECTORY / "nih_cxr14/radiological_findings"):
            # read each concepts file
            concept_file = pd.read_csv(DATA_DIRECTORY / "nih_cxr14/radiological_findings" / file)
            file_name = file.split('.')[0]

            # change names of the columns in tabular data files if needed
            if file_name == 'carina_angle':
                # rename point_1, point_2, point_3 as point_c1, point_c2, point_c3
                concept_file = concept_file.rename(columns={f"point_{i}": f"point_c{i}" for i in range(1,4)})
            elif file_name == 'descending_aorta_tortuous':
                concept_file = concept_file.rename(columns={f"point_{i}": f"point_a{i}" for i in range(1,7)})
            elif  file_name == 'trachea_deviation':
                # rename point_1,...point_9 as point_1_trachea,...,point_9_trachea
                concept_file = concept_file.rename(columns={f"point_{i}": f"point_t{i}" for i in range(1,10)})
            elif file_name == 'mediastinal_widening':
                # rename lung_xmin and lung_xmax as mediastinum_lung_xmin and mediastinum_lung_xmax
                concept_file = concept_file.rename(columns={"lung_xmin": "mediastinum_lung_xmin", "lung_xmax": "mediastinum_lung_xmax"})
            elif file_name == 'descending_aorta_enlargement':
                # eliminate trachea_pint_right and trachea_point_left columns
                concept_file = concept_file.drop(columns=['trachea_point_right', 'trachea_point_left'], errors='ignore')


            # rename images and labels for merging
            concept_file.rename(columns={'image_file': 'img_id'}, inplace=True)
            concept_file['img_id'] = concept_file['img_id'].apply(lambda x: x + '.png')
            concept_file.rename(columns={'label': file_name}, inplace=True)

            # add concepts and raw tabular data to concepts_df
            # drop from concept_file all the columns that are not in RAW_TABULAR_DATA_COLUMNS or in TOTAL_CONCEPTS
            if self.modality == 'tabular':
                concept_file = concept_file[[col for col in concept_file.columns if col in RAW_TABULAR_DATA_COLUMNS or col in self.concepts_names or col == 'img_id']]
            else:
                concept_file = concept_file[[col for col in concept_file.columns if col in self.concepts_names or col == 'img_id']]   
            self.df = pd.merge(self.df, concept_file, on='img_id', how='left')
            
        # preprocess tabular data
        self.df = preprocess_tabular_data(self.df)

        # create the column 'img_path' by joining root + 'images/' + img_id
        self.df['img_path'] = self.df['img_id'].apply(lambda x: os.path.join(self.root, 'raw_data/original_nih_chest/images', x))
        self.df = self.df.drop(columns=['img_id'])

        # eliminate rows with NaN in any columns
        self.df = self.df.dropna().reset_index(drop=True)

        # create X,c,y
        # select columns in the order specified in self.concepts_names and self.task_names
        self.c = torch.tensor(self.df[list(self.concepts_names)].values.astype(np.float32)) 
        self.y = self.df[list(self.task_names)].values.astype(np.float32)
        if modality == 'image':
            self.X = self.df['img_path'].values
            self.tabular_columns = None
        else:
            # keep names of the columns for tabular data
            if self.tabular_columns == []:
                self.tabular_columns = [col for col in self.df.columns if col not in list(self.concepts_names) + list(self.task_names) + ['img_path']]
            self.X = self.df[self.tabular_columns].values.astype(np.float32)


    def register_graph(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.modality == 'image':
            img_path = self.X[idx]
            try:
                transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
                img = imread(img_path)
                img= normalize(img, maxval=255, reshape=True)
                img = transform(img)
                img = torch.from_numpy(img)       
            except:
                print(f"Error loading image: {img_path}")
                raise FileNotFoundError(f"Image not found: {img_path}")  
            x = img
        else:
            x = self.X[idx]
            # all the columns that are not concepts or tasks
            x = torch.FloatTensor(x)

        y = torch.FloatTensor(self.y[idx])
        c = torch.FloatTensor(self.c[idx])

        return {"x": x, "c": c, "y": y, "graph": self.graph}

    def collate_fn(self, instances):
        #print(f"collate_fn called with {len(instances)} instances")
        xs = torch.stack([ins["x"] for ins in instances], dim=0)
        c = torch.stack([ins["c"] for ins in instances], dim=0)
        labels = torch.stack([ins["y"] for ins in instances], dim=0)
        graph = instances[0]["graph"]
        return {"x": xs, "c": c, "y": labels, "graph" : graph}
    





