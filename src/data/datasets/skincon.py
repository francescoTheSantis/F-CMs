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
from urllib.parse import urlparse
from env import CACHE
import PIL
from PIL import Image

# To work with this dataset:
# 1. Download the dataset from https://skincon-dataset.github.io/ (Fitzpatrick17k images)
ROOT = CACHE / "skincon"
IMAGES_DIRECTORY = ROOT / "images" 
os.makedirs(IMAGES_DIRECTORY, exist_ok=True)
TARGET_CLASSES = ["malignant", "benign", "non-neoplastic"]
CLASS_MAPPING = {"benign": 0, "malignant": 1, "non-neoplastic": 2}

# create a unique file containing both target, images and annotations
def create_unique_csv(root, annotations_path=None, target_path=None, concept_names=None):
    # read the two dataframes
    if annotations_path is None:
        annotations_path = os.path.join(root, "fitzpatrick17k_original_annotations.csv")
    if target_path is None:
        target_path = os.path.join(root, "fitzpatrick17k_images_labels.csv")
    
    annotations_df = pd.read_csv(annotations_path)
    target_df = pd.read_csv(target_path)
    
    # preprocess them
    # eliminate ".jpg" from ImageID
    annotations_df['ImageID'] = annotations_df['ImageID'].apply(lambda x: x[:-4])
    # in target_df create imageID column by taking just the string after "/" of column "url"
    #target_df["image_name"] = target_df["url"].str.split("/").str[-1]
    # rename column "md5hash" in target_df as "ImageID"
    target_df = target_df.rename(columns={"md5hash": "ImageID"})
    
    # merge the two dataframes on the ImageID column
    merged_df = pd.merge(annotations_df, target_df, on="ImageID", how="inner")
    # re-establish index
    merged_df = merged_df.reset_index(drop=True)
    # rename "ImageID" from the merged_df as "img_id"
    merged_df = merged_df.rename(columns={"ImageID": "img_id"})
    # eliminate images that have 1 in "Do not consider this image" column
    merged_df = merged_df[merged_df["Do not consider this image"] != 1]
    merged_df = merged_df.reset_index(drop=True)
    # transform all the columns names in lowercase
    merged_df.columns = merged_df.columns.str.lower()
    # just keep the columns "img_id", "three_partition_label" and concept_names columns 
    if concept_names is not None:
        merged_df = merged_df[["img_id", "three_partition_label", "url"] + concept_names]

    # rename "three_partition_label" column as "label"
    merged_df = merged_df.rename(columns={"three_partition_label": "label"})
    # select only labels in TARGET_CLASSES
    merged_df = merged_df[merged_df["label"].isin(TARGET_CLASSES)]
    merged_df.to_csv(os.path.join(root, "skincon_merged.csv"), index=False)

def download_fitzpatrick_images(csv_path=None):
    # download images in the url column of the csv file and save them in the root directory
    df = pd.read_csv(csv_path)
    url_column = "url"
    # create also a csv file with the id of the image and if it is train or test
    # add a time bar to the download process
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = row[url_column]
        if not isinstance(url, str) or url.strip() == "":
            continue
        
        filename = row['img_id'] + ".jpg"
        save_path = os.path.join(IMAGES_DIRECTORY, filename)
        
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

def generate_train_test_split(csv_path, train_csv_path, test_csv_path, test_size=0.2):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataset
    split_idx = int(len(df) * (1 - test_size))
    # save only img_id
    df.iloc[:split_idx][["img_id"]].to_csv(train_csv_path, index=False)
    df.iloc[split_idx:][["img_id"]].to_csv(test_csv_path, index=False)


class SkinConDataset():
    """
    The color MNIST dataset is a modified version of the MNIST dataset where each digit is colored either red or green.
    The concept labels are the digit and the color of the digit.
    The task is to predict whether the digit is even or odd.

    Attributes:
        transform: The transformations to apply to the images. Default is None.
        target_transform: The transformations to apply to the target labels. Default is None.
        ftune_size: The proportion of the test set to include in the finetuning set. Default is 0.1.
        val_size: The proportion of the training set to include in the validation set. Default is 0.1.
        coloring: A dictionary with coloring options: 'train' and 'test'.
    """
    def __init__(self, 
                 transform: Union[Compose, torch.nn.Module] = None,
                 target_transform: Union[Compose, torch.nn.Module] = None, 
                 val_size: float = 0.1, # proportion of the training set to include in the validation set
                 ftune_size: float = 0., # proportion of the test set to include in the finetuning set
                 ftune_val_size: float = 0., # proportion of the finetuning set to include in the finetuning validation set
        ):

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        self.target_transform = target_transform
        self.ftune_size = ftune_size
        self.val_size = val_size
        self.ftune_val_size = ftune_val_size

        self.c_info = {'names':  ["papule", "plaque", "pustule", "bulla", "patch", "nodule", 
                                  "ulcer", "crust", "erosion", "atrophy", "exudate", "telangiectasia", 
                                  "scale", "scar", "friable", "dome-shaped", "brown(hyperpigmentation)", 
                                  "white(hypopigmentation)", "purple", "yellow", "black", "erythema"], 
        'cardinality': [2] * 22}

        self.y_info = {'names': ["malignant"],
                       'cardinality': [2]}
        

        self.data = {}

        # files preprocessing
        merged_csv_path = os.path.join(ROOT, "skincon_merged.csv")
        if not os.path.exists(merged_csv_path):
            annotations_path = os.path.join(ROOT, "fitzpatrick17k_original_annotations.csv")
            target_path = os.path.join(ROOT, "fitzpatrick17k_images_labels.csv")
            create_unique_csv(root=ROOT, annotations_path=annotations_path, target_path=target_path, concept_names=self.c_info['names'])

        # if images are not already downloaded, download them. Check if I have the same number of images in the directory as in the csv file, if not, redownload them.
        if len(os.listdir(IMAGES_DIRECTORY)) < 500:
            print("Downloading images...")
            download_fitzpatrick_images(csv_path = merged_csv_path)
            print("Download completed.")
        
        # add a check that eliminates from the csv file the images that are not present in the directory
        df = pd.read_csv(merged_csv_path)
        available_images = set(os.listdir(IMAGES_DIRECTORY))
        df = df[df["img_id"].apply(lambda x: x + ".jpg" in available_images)]
        df.to_csv(merged_csv_path, index=False)
     
        # if files with train/test split do not exist, create them
        train_csv_path = os.path.join(ROOT, "skincon_train.csv")
        test_csv_path = os.path.join(ROOT, "skincon_test.csv")
        if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
            csv_path = merged_csv_path
            generate_train_test_split(csv_path, train_csv_path, test_csv_path)


    def load_ground_truth_graph(self):
        self.adj = None
        return self.adj

    def split(self):
        """ 
        Split the dataset into training, validation and test sets 
        """
        self.data['train'] = _SkinConDataset(root = ROOT, 
                                            train = True, 
                                            transform = self.transform,
                                            target_transform = self.target_transform,
                                            concept_names = self.c_info['names'])
        self.data['test'] = _SkinConDataset(root = ROOT, 
                                            train = False, 
                                            transform = self.transform,
                                            target_transform = self.target_transform,
                                            concept_names = self.c_info['names'])
        self.data['train'], self.data['val'] = split_dataset(self.data['train'], self.val_size)
        self.data['val'].split_type = 'val'
        if self.ftune_size !=0:
            self.data['test'], self.data['ftune'] = split_dataset(self.data['test'], self.ftune_size)
            self.data['ftune'].split_type = 'ftune'
            self.data['ftune'], self.data['ftune_val'] = split_dataset(self.data['ftune'], self.ftune_val_size)
            self.data['ftune_val'].split_type = 'ftune_val'


class _SkinConDataset(Dataset):
    """SkinCon Dataset.

    Args:
        root         (str): Root directory of dataset.
        csv_path     (str): Path to the metadata CSV file. Defaults to `{root}/ddi_metadata.csv`
        transform         : Function to transform and collate image input. (can use test_transform from this file) 
    """
    def __init__(self, root, train, csv_path = None, transform = None, target_transform = None, concept_names = None):    
        #assert filter >= 0.
        #assert filter <= 1.0    

        self.split = 'train' if train else 'test'
        self.transform = transform
        self.target_transform = target_transform
        self.concept_names = concept_names

        if csv_path is None:
            csv_path = os.path.join(root, "skincon_merged.csv")
        self.df = pd.read_csv(csv_path)

        if train:
            train_images = pd.read_csv(os.path.join(ROOT, "skincon_train.csv"))["img_id"].tolist()
            self.df = self.df[self.df["img_id"].apply(lambda x: x in train_images)]

        else:
            test_images = pd.read_csv(os.path.join(ROOT, "skincon_test.csv"))["img_id"].tolist()
            self.df = self.df[self.df["img_id"].apply(lambda x: x in test_images)]

        self.df = self.df.reset_index(drop=True)
        # convert the "label" column to 0 for benign and 1 for malignant using CLASS_MAPPING
        self.df["label"] = self.df["label"].apply(lambda x: CLASS_MAPPING[x])
        # Attention: unbalanced dataset for the moment
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
        self.c = torch.cat([c.unsqueeze(0) for c in self.c], dim=0).type(torch.int)
        self.y = torch.cat([y.unsqueeze(0) for y in self.y], dim=0).unsqueeze(-1).type(torch.int)

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
        img_path = os.path.join(IMAGES_DIRECTORY, img_id + ".jpg")
        # open image
        if self.X is None:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image) 
        else:
            image = self.X[idx]

        # Get skin concepts
        concepts = torch.from_numpy(sample[self.concept_names].values.astype(np.float32))
        concepts = concepts.type(torch.long)

        # Get the target label
        label = torch.tensor(sample["label"], dtype=torch.long)

        return {"x": image,  "c": concepts, "y": label}

