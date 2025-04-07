from env import CACHE
from copy import deepcopy
import os
import json
import torch
from torch.utils.data import DataLoader
import torchvision.models as tv_models
from torchvision.models.resnet import ResNet50_Weights
import numpy as np
# progress bar
from tqdm import tqdm

from src.data.utils import reduce_dataset
from src.data.autoencoder import AutoencoderTrainer, scale_embeddings

def maybe_reduce(reduce_fraction, dataset):
    # random sample a fraction of the dataset
    if reduce_fraction is not None:
        for split, data in dataset.data.items():
            # get the number of samples to be split
            n_split = int(reduce_fraction * len(data))
            # get the indices of samples to be split
            index_split = np.random.choice(len(data), n_split, replace=False)
            data = reduce_dataset(data, index_split)
            dataset.data[split] = data
    return dataset

def preprocess_dataset(cfg, _dataset, device) -> dict:
    """
    Preprocess the dataset.
    Args:
        cfg: Dictionary with the configuration.
        dataset: Dictionary with the dataset splits.
    Returns:
        processed_dataset: Dictionary with the preprocessed dataset splits.
    """
    dataset = deepcopy(_dataset)

    print('preprocessing data...')

    # colormnist
    dataset_name = cfg.dataset.get('name')

    if dataset_name in ['asia', 'alarm', 'sachs', 'hailfinder', 'insurance']:
        dataset = maybe_reduce(cfg.dataset.get('reduce_fraction', None), dataset)
        
        all_var = dataset.c_info['names'] + dataset.y_info['names'] # variables have been reordered
                                                                    # when the dataset was created
        # for most datasets, encode only the concepts variables, exclude the task
        selected_var = dataset.c_info['names']
        if dataset_name=='asia':
            pass
            # selected_var = ['asia', 'smoke']
        elif dataset_name=='alarm':
            pass
            # selected_var = ['MINVOLSET', 'DISCONNECT', 'PULMEMBOLUS', \
            #                 'INTUBATION', 'KINKEDTUBE', 'ANAPHYLAXIS', \
            #                 'FIO2', 'INSUFFANESTH', 'LVFAILURE', 'HYPOVOLEMIA', \
            #                 'ERRLOWOUTPUT', 'ERRCAUTER']
        elif dataset_name=='sachs_ood':
            pass
        selected_var_index = [all_var.index(var) for var in selected_var]

        autoencoder_trainer = AutoencoderTrainer(autoencoder_cfg=cfg.dataset.autoencoder,
                                                 input_shape=len(selected_var), 
                                                 device=device)
        dataset.split()
        dataset = autoencoder_trainer.train(dataset=dataset, 
                                            selected_var_index=selected_var_index)
        dataset = scale_embeddings(dataset)

        # avoid empty spaces in the concepts names
        dataset.c_info['names'] = [concept.replace(' ', '_') for concept in dataset.c_info['names']]


        #dataset = maybe_reduce(cfg.dataset.get('reduce_fraction', None), dataset)
        #dataset = generate_img_embeddings(dataset, batch_size=cfg.dataset.get('batch_size'), device=device)
    else:
        raise ValueError(f"Preprocessing is missing for dataset: {cfg.dataset.get('name')}")
    
    print('done')

    print(f"Concepts: {dataset.c_info['names']}")
    print(f"Task: {dataset.y_info['names']}")
    return dataset
