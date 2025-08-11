from env import CACHE
from copy import deepcopy
import os
import json
import torch
from torch.utils.data import DataLoader
import torchvision.models as tv_models
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights
import numpy as np
# progress bar
from tqdm import tqdm

from src.models.layers.pretrained import InputImgEncoder
from src.data.utils import reduce_dataset
from src.data.datasets.colormnist import update_concept_names_ColorMNIST, onehot_to_concepts_ColorMNIST
from src.data.datasets.fashionmnist import update_concept_names_FashionMNIST, onehot_to_concepts_FashionMNIST
from src.data.autoencoder import AutoencoderTrainer, scale_embeddings
#from src.data.labelfree_preprocessing import load_pretrained_clip_model, generate_img_embeddings_and_assign_concepts
#from src.completion.concepts_retrieval import concepts_generation, filtering_concepts_from_llm
#from src.data.datasets.synthetic import get_synthetic_datasets, SyntheticDatasetContainer

def generate_img_embeddings(dataset: torch.utils.data.Dataset,
                           batch_size: int = 32,
                           device: str = 'cpu',
                           backbone: str = 'resnet18') -> None:
    
    if backbone == 'resnet18':
        input_encoder = tv_models.resnet18(weights= ResNet18_Weights.DEFAULT)
    elif backbone == 'resnet50':
        input_encoder = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = InputImgEncoder(input_encoder).to(device)
    model.eval()

    for split, data in dataset.data.items():
        data = _generate_img_embeddings(data, model, batch_size, device)
        dataset.data[split] = data
    return dataset

def _generate_img_embeddings(dataset, model, batch_size, device) -> None:
    """
    Preprocess an image dataset using a given input encoder.
    Args:
        dataset: dataset object.
        input_encoder: input encoder model.
        batch_size: batch size.
        device: device to run the model on.
    Returns:
        None
    """

    # Load dataset
    data_loader = DataLoader(dataset, batch_size=batch_size)

    # Extract embeddings
    embeddings = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader)):
            images = batch['x'].to(device)
            # TODO: check this handles colors correctly
            emb = model(images)
            embeddings.append(emb)
                
    # Concatenate and save embeddings
    embeddings = torch.cat(embeddings, dim=0).cpu()
    dataset.X = embeddings
    return dataset

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

def preprocess_dataset(dataset_cfg, _dataset, device, backbone ='resnet18') -> dict:
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
    dataset_name = dataset_cfg.get('name').replace('_ood', '')
    if dataset_name == 'colormnist':
        is_complex_coloring =  dataset_cfg.loader.coloring.train.mode == 'complex'
        dataset.split()
        if  dataset_cfg.get('onehot_to_concepts') == True: 

            dataset = update_concept_names_ColorMNIST(dataset, 
                                                      is_complex_coloring )
        dataset = maybe_reduce(dataset_cfg.get('reduce_fraction', None), dataset)
        dataset = generate_img_embeddings(dataset, 
                                          batch_size=256, 
                                          device=device,
                                          backbone=backbone)
        if dataset_cfg.get('onehot_to_concepts') == True:
            dataset = onehot_to_concepts_ColorMNIST(dataset, 
                                                    is_complex_coloring)
            
    elif dataset_name == 'fashionmnist':
        is_complex_coloring =  dataset_cfg.loader.coloring.train.mode == 'complex'
        dataset.split()
        if dataset_cfg.get('onehot_to_concepts') == True: 

            dataset = update_concept_names_FashionMNIST(dataset, 
                                                      is_complex_coloring)
        dataset = maybe_reduce(dataset_cfg.get('reduce_fraction', None), dataset)
        dataset = generate_img_embeddings(dataset, 
                                          batch_size=256, 
                                          device=device,
                                          backbone=backbone)
        if dataset_cfg.get('onehot_to_concepts') == True:
            dataset = onehot_to_concepts_FashionMNIST(dataset, 
                                                    is_complex_coloring)

    elif dataset_name in ['asia', 'alarm', 'sachs', 'hailfinder', 'insurance']:
        dataset = maybe_reduce(dataset_cfg.get('reduce_fraction', None), dataset)
        
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

        autoencoder_trainer = AutoencoderTrainer(autoencoder_cfg= dataset_cfg.autoencoder,
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
        raise ValueError(f"Preprocessing is missing for dataset: {dataset_cfg.get('name')}")
    
    print('done')

    print(f"Concepts: {dataset.c_info['names']}")
    print(f"Task: {dataset.y_info['names']}")
    return dataset
