from sqlalchemy import values
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from env import CACHE

DATA_DIRECTORY = CACHE / "derm7pt"
IMAGES_DIRECTORY = DATA_DIRECTORY / "images"
meta_data = DATA_DIRECTORY / "meta.csv"
train_csv = DATA_DIRECTORY / "train_indexes.csv"
valid_csv = DATA_DIRECTORY / "valid_indexes.csv"
test_csv = DATA_DIRECTORY / "test_indexes.csv"
TASK_NAME = "diagnosis"
CONCEPT_NAMES = ["pigment_network",
            "streaks",
            "dots_and_globules",
            "blue_whitish_veil",
            "regression_structures"]
TASK_VALUES = ["nevus", "melanoma"]

CONCEPT_MAPPINGS = {
            "pigment_network": { "absent": 0, "typical": 1, "atypical": 2},
            "streaks": {"absent": 0, "regular": 1, "irregular": 2},
            "dots_and_globules": {"absent": 0, "regular": 1, "irregular": 2},
            "blue_whitish_veil": {"absent": 0, "present": 1},
            "regression_structures": {"absent": 0, "present": 1},
        }

CLASS_MAPPING = {"nevus": 0, "melanoma": 1}


class Derm7ptDataset():
    def __init__(self, 
                 test_size:  float = 0.2, # proportion of the dataset to include in the test set
                 val_size: float = 0.1, # proportion of the training set to include in the validation set
                 ftune_size: float = 0., # proportion of the test set to include in the finetuning set
                 ftune_val_size: float = 0. # proportion of the finetuning set to include in the finetuning validation set
                    ):
        
    
        self.test_size = test_size
        self.val_size = val_size
        self.ftune_size = ftune_size
        
        c_names = CONCEPT_NAMES  # Names of the concepts
        
        y_name = [TASK_NAME]
        c_cardinalities = [3,3,3,2,2]  # Cardinalities for all c_names
        y_cardinality = [len(TASK_VALUES)]  # Cardinality for the task_name
        self.c_info = {'names': c_names, 
                       'cardinality': c_cardinalities}
        self.y_info = {'names': y_name,
                       'cardinality': y_cardinality}  
        self.data = {}
        self.subgraphs_concept_names = {}
        
        
        
    def load_ground_truth_graph(self): 
        return None
    
    def split(self, **kwargs):
        """ 
        Split the dataset into training, validation and test sets 
        """
        # self.bn_model is biased after this point
        self.data['train'] = _Derm7ptDataset(csv_file=train_csv,
                                             meta_data=meta_data
        )
        self.data['val'] = _Derm7ptDataset(csv_file=valid_csv,
                                           meta_data=meta_data
        )
        self.data['test'] = _Derm7ptDataset(csv_file=test_csv,
                                            meta_data=meta_data
        )
        self.data['train'].split_type = 'train'
        self.data['val'].split_type = 'val'

class _Derm7ptDataset(Dataset):
    """
    A custom Pytorch dataset for reading data from a CSV file regarding the Derm7pt dataset.

    Args:
        csv_file (str): Path to the CSV file containing the data.
        img_extension (str): The file extension of the images.
        path_to_images (str): The path to the images folder.

    Attributes:
        data (DataFrame): Pandas DataFrame containing the data from the CSV file.

    """
    
    def __init__(self, csv_file, meta_data):
 
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.X = None

        # load data and process it
        indexes = pd.read_csv(csv_file)
        self.data = pd.read_csv(meta_data)
        # rename the column "diagnosis" to "labels"
        self.data = self.data.rename(columns={"diagnosis": "labels"})
        # rename the column "derm" to "image_path"
        self.data = self.data.rename(columns={"derm": "image_path"})

        ## Filter data
        # filter data such that index is in the indexes of the csv file, correct the code
        self.data = self.data.iloc[indexes['indexes']]
        # fix the index
        self.data = self.data.reset_index(drop=True)
        # change all the labels containing "nevus" to "nevus" and all the labels containing "melanoma" to "melanoma
        self.data["labels"] = self.data["labels"].apply(lambda x: "nevus" if "nevus" in x else ("melanoma" if "melanoma" in x else x))
        # filter data in such a way diagnosis is either nevus or melanoma
        #self.data = self.data[self.data['labels'].isin(TASK_VALUES)]
        # filter only the columns we need: diagnosis, cliic, concepts
        self.data = self.data[CONCEPT_NAMES + ["labels", "image_path"]]
        self.data = self.data.reset_index(drop=True)

        # apply concept and label mappings
        self.data["labels"] = self.data["labels"].map(CLASS_MAPPING)
        for concept in CONCEPT_NAMES:
            self.data[concept] = self.data[concept].map(CONCEPT_MAPPINGS[concept])
        
        # Filter out rows where the image file doesn't exist
        def check_image_exists(img_path):
            full_path = IMAGES_DIRECTORY / img_path
            return os.path.exists(full_path)
        
        # Keep only rows where the image exists
        valid_images = self.data["image_path"].apply(check_image_exists)
        if not valid_images.all():
            missing_count = (~valid_images).sum()
            print(f"Warning: {missing_count} images not found and will be skipped")
            self.data = self.data[valid_images].reset_index(drop=True)
        
        self.graph = {}

    def register_graph(self, graph):
        self.graph = graph
    
    def update_lists(self):
        """
        Initialize the concepts and labels as tensors by iterating through all data.
        """
        self.c = []
        self.y = []
        
        for i in range(len(self.data)):
            out = self.__getitem__(i)
            self.c.append(out['c'])
            self.y.append(out['y'])
        
        self.c = torch.stack(self.c, dim=0)
        self.y = torch.stack(self.y, dim=0).unsqueeze(-1)
        self.c = self.c.type(torch.int)
        self.y = self.y.type(torch.int)
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.data)
    
    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        img_path = row["image_path"]
        label = row["labels"].astype(np.float32)
        concepts = row[CONCEPT_NAMES].values.astype(np.float32)
        concepts = torch.tensor(concepts, dtype=torch.float32)
        
        
        if self.X is None:
            img_full_path = IMAGES_DIRECTORY / img_path
            image = Image.open(img_full_path).convert("RGB")
            if self.transform:
                image = self.transform(image) 
        else:
            image = self.X[idx]

        # Make the concept ints
        concepts = concepts.type(torch.long)
        # transform everything to tensors
        label = torch.tensor(label)   


        return {'x':image, 'c':concepts, 'y':label, 'graph':self.graph}

