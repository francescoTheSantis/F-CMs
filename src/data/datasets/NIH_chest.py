from env import CACHE

import torch
import torchvision
from torchvision.transforms import Compose
from torchvision import transforms
import numpy as np
import pandas as pd
import torchxrayvision as xrv
from typing import Union

from src.data.utils import split_dataset
import shutil
import os
import zipfile
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
DATA_DIRECTORY = CACHE / "NIH_chest"/ "raw_data"
DATA_DIRECTORY.mkdir(exist_ok=True, parents=True)

# determine which columns are considered as tabular data
TABULAR_DATA_DICTIONARY = {'cardiomegaly': ['ctr'],
                        'mediastinal_widening': ['mcr'],
                        'carina_angle': ['angle'],
                        'trachea_deviation': ['direction'],
                        'aortic_knob_enlargement': ['ratio'],
                        'ascending_aorta_enlargement': ['ratio'],
                        'descending_aorta_enlargement': ['ratio'],
                        'descending_aorta_tortuous': ['curvature']}

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
        self.transform = transform
        self.target_transform = target_transform
        self.ftune_size = ftune_size
        self.val_size = val_size
        self.ftune_val_size = ftune_val_size

        self.c_info = {'names': ['Patient Age',
                                 'Male',
                                 'ascending_aorta_enlargement',
                                 'cardiomegaly',
                                 'aortic_knob_enlargement',
                                'descending_aorta_tortuous',
                                'trachea_deviation',
                                'carina_angle',
                                'mediastinal_widening',
                                'descending_aorta_enlargement'],
        'cardinality': [100,2,2,2,2,2,2,2,2,2]} # 10 concepts

        self.y_info = {'names': ['Atelectasis', 
                                 'Cardiomegaly', 
                                 'Effusion',
                                 'Infiltration', 
                                 'Mass',
                                 'Nodule',
                                 'Pneumonia',
                                 'Pneumothorax',
                                 'Consolidation',
                                 'Edema',
                                 'Emphysema',
                                 'Fibrosis',
                                 'Pleural_Thickening',
                                 'Hernia'], 'cardinality': [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]} # 14 disease categories + No Finding

        self.data = {}

    def load_ground_truth_graph(self):
        raise NotImplementedError("There is no ground truth graph for the NIH-chest dataset.")


    def split(self):
        """ 
        Split the dataset into training, validation and test sets 
        """
        self.data['train'] = _NIH_chest(root = str(CACHE / "NIH_chest"), 
                                        modality = self.modality,
                                        concepts_names= self.c_info['names'],
                                        task_names = self.y_info['names'],
                                        train = True)
        self.data['test'] = _NIH_chest(root = str(CACHE / "NIH_chest"),
                                       modality = self.modality,
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
                 concepts_names: list,
                 task_names: list,
                 train: bool = False):
        
        self.modality = modality
        self.split_type = 'train' if train else 'test'
        self.graph = {}
        self.root = root
        self.concepts_names = concepts_names
        self.task_names = task_names
        
        # load images ids for train and test
        if self.split_type == 'train':
           images = pd.read_csv(DATA_DIRECTORY / "original_nih_chest/train_val_list.txt", header=None, names=["img_id"])
        else:
           images = pd.read_csv(DATA_DIRECTORY / "original_nih_chest/test_list.txt", header=None, names=["img_id"])
            
        self.df = images.copy()

        # load the labels
        labels = pd.read_csv(DATA_DIRECTORY / "original_nih_chest/Data_Entry_2017.csv")
        labels = labels.rename(columns={'Image Index': 'img_id', 'Finding Labels': 'disease'}) 
        self.df = pd.merge(self.df, labels, on='img_id', how='left')
        self.df = self.df.drop(columns=['Follow-up #', 'Patient ID', 'View Position', 'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 'Unnamed: 11'])
        for disease in self.task_names:
            self.df[f"Task_{disease}"] = self.df['disease'].apply(lambda x: 1 if disease in x.split('|') else 0)
        # drop the original 'disease' column
        self.df = self.df.drop(columns=['disease'])

        # load the concepts
        for file in os.listdir(DATA_DIRECTORY / "nih_cxr14/radiological_findings"):
            concepts_file = pd.read_csv(DATA_DIRECTORY / "nih_cxr14/radiological_findings" / file)
            file_name = file.split('.')[0]     

            # for each file, only the label is kept as concept
            concepts_file.rename(columns={'image_file': 'img_id'}, inplace=True)
            concepts_file['img_id'] = concepts_file['img_id'].apply(lambda x: x + '.png')
            concepts = concepts_file[['img_id'] + ['label']]
            concepts.rename(columns={'label': file_name}, inplace=True)

            # add concepts to self.df
            self.df = pd.merge(self.df, concepts, on='img_id', how='left')

            if modality == 'tabular':
                if file_name not in TABULAR_DATA_DICTIONARY.keys():
                    continue
                else:
                    # add to self.df the columns of concepts_file that are in the TABULAR_DATA_DICTIONARY
                    tabular_file_addition = concepts_file[['img_id'] + concepts_file.columns.intersection(TABULAR_DATA_DICTIONARY[file_name]).tolist()]

                # change names of the columns in tabular_file_addition
                if file_name in ['ascending_aorta_enlargement', 'descending_aorta_enlargement', 'aortic_knob_enlargement']:
                    # eliminate "enlargement" from the file_name if it is present
                    file_name = file_name.replace('_enlargement', '')
                    tabular_file_addition.rename(columns={'ratio': f'{file_name}_ratio'}, inplace=True)
                elif file_name in ['descending_aorta_tortuous', 'trachea_deviation']:
                    file_name = file_name.replace('_tortuous', '')
                    file_name = file_name.replace('_angle', '')
                    tabular_file_addition = tabular_file_addition.rename(columns=lambda x: file_name + '_' + x if x != 'img_id'else x)
                elif file_name == 'carina_angle':
                    tabular_file_addition.rename(columns={'angle': 'carina_angle_num'}, inplace=True)

                self.df = pd.merge(self.df, tabular_file_addition, on='img_id', how='left')


        # clean data
        # replace True/False with 1/0
        self.df = self.df.replace({True: 1, False: 0})

        # replace 'M'/'F' with 1/0
        self.df['Patient Gender'] = self.df['Patient Gender'].replace({'M': 1, 'F': 0})
        self.df.rename(columns={'Patient Gender': 'Male'}, inplace=True)

        # manage identical columns found in concepts_files (e.g. heart_ratio_x and heart_ratio_y) by taking the mean of the two columns
        for col in self.df.columns:
            if col.endswith('_x'):
                col_y = col[:-2] + '_y'
                self.df[col[:-2]] = self.df[[col, col_y]].mean(axis=1)
                self.df = self.df.drop(columns=[col, col_y])
                
        # eliminate rows with NaN values and reset the index
        self.df = self.df.dropna().reset_index(drop=True)


        # create the column 'img_path' by joining root + 'images/' + img_id
        self.df['img_path'] = self.df['img_id'].apply(lambda x: os.path.join(self.root, 'raw_data/original_nih_chest/images', x))
        self.df = self.df.drop(columns=['img_id'])

        # save tabular data
        if modality == 'tabular':
            trachea_direction_mapping = {'flat': 0, 'right': 1, 'left': 2, 'left&right': 3, 'right&left': 4}
            self.df['trachea_deviation_direction'] = self.df['trachea_deviation_direction'].map(trachea_direction_mapping)
            self.df.drop(columns = ['img_path'], inplace=True) # drop img_path column
            self.df.to_csv(os.path.join(DATA_DIRECTORY, f"tabular_data_{self.split_type}.csv"), index=False)

        # create X,c,y
        if modality == 'image':
            self.X = self.df['img_path'].values
        else:
            feature_columns = [col for col in self.df.columns if col not in self.concepts_names + [f"Task_{task}" for task in self.task_names]]
            self.X = torch.tensor(self.df[feature_columns].values.astype(np.float32))
        self.c = torch.tensor(self.df[self.concepts_names].values.astype(np.float32))
        self.y = torch.tensor(self.df[[f"Task_{task}" for task in self.task_names]].values.astype(np.float32))


    def register_graph(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.modality == 'image':
            img_path = self.X[idx]

            try:
                transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
                img = Image.open(img_path).convert("L")
                img = np.array(img)
                img =  xrv.datasets.normalize(img, 255)
                img = img[None, ...]
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
        xs = torch.stack([ins["x"] for ins in instances], dim=0)
        c = torch.stack([ins["c"] for ins in instances], dim=0)
        labels = torch.stack([ins["y"] for ins in instances], dim=0)
        graph = instances[0]["graph"]
        return {"x": xs, "c": c, "y": labels, "graph" : graph}
    





