from env import CACHE
from copy import deepcopy
import os
import json
import torch
from torch.utils.data import DataLoader
import torchvision.models as tv_models
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights
import torchxrayvision as xrv
import numpy as np
# progress bar
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from src.models.layers.pretrained import InputImgEncoder
from src.models.layers.base import MLP
from src.data.utils import reduce_dataset, change_task
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
    elif backbone == 'res224-all':
        input_encoder = xrv.models.DenseNet(weights="densenet121-res224-all")
    else:
        raise ValueError(f"Backbone {backbone} not supported for image embeddings generation.")

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
    # take the name of the dataset
    if hasattr(dataset, 'collate_fn'):
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size)

    # Extract embeddings
    embeddings = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader)):
            images = batch['x'].to(device)
            # TODO: check this handles colors correctly
            #check if the dataset root contains "NIH_chest"
            emb = model(images)
            embeddings.append(emb)
                
    # Concatenate and save embeddings
    embeddings = torch.cat(embeddings, dim=0).cpu()
    dataset.X = embeddings
    return dataset

def generate_tabular_embeddings(dataset: torch.utils.data.Dataset,
                                batch_size: int = 32,
                                device: str = 'cpu',
                                hidden_size: int = 128,
                                n_layers: int = 3,
                                epochs: int = 50,
                                lr: float = 1e-3) -> None:
    """
    Generate embeddings from tabular data using a trained MLP model.
    The model is trained to predict labels from tabular input, and embeddings
    are extracted from the penultimate layer (before the classification layer).
    
    Args:
        dataset: dataset object with tabular data
        batch_size: batch size for training and embedding extraction
        device: device to run the model on
        hidden_size: hidden layer size for the MLP
        n_layers: number of layers in the MLP encoder
        epochs: number of training epochs
        lr: learning rate
    """
    import torch.nn.functional as F
    import torch.optim as optim
    
    # Get input size from the training data
    input_size = dataset.data['train'].X.shape[1]
    
    # Determine actual number of unique classes in the training data
    train_labels = dataset.data['train'].y
    unique_labels = np.unique(train_labels)
    output_size = int(max(unique_labels)) + 1  # e.g., if labels are 0-14, we need 15 classes
    
    print(f"Input size: {input_size}")
    print(f"Unique labels in training: {unique_labels}")
    print(f"Number of output classes: {output_size}")
    print(f"Label range: [{train_labels.min():.0f}, {train_labels.max():.0f}]")
    
    # Create MLP model: encoder + classifier
    encoder = MLP(input_size=input_size,
                  hidden_size=hidden_size,
                  output_size=hidden_size,
                  n_layers=n_layers,
                  activation='leaky_relu')
    
    classifier = torch.nn.Linear(hidden_size, output_size)
    
    model = torch.nn.Sequential(encoder, classifier).to(device)
    
    # Train the model
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_data = dataset.data['train']
    if hasattr(train_data, 'collate_fn'):
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    print("Training tabular embedding model...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_samples = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch['x'].to(device)
            y = batch['y'].squeeze().long().to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += y.size(0)
            correct += (predicted == y).sum().item()
        
        acc = 100 * correct / total_samples
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%")
    
    print("Extracting embeddings...")
    # Extract embeddings using only the encoder (excluding classifier)
    encoder.eval()
    for split, data in dataset.data.items():
        data = _generate_tabular_embeddings(data, encoder, batch_size, device)
        dataset.data[split] = data
    
    return dataset

def _generate_tabular_embeddings(dataset, encoder, batch_size, device):
    """
    Extract embeddings from tabular data using the trained encoder.
    """
    if hasattr(dataset, 'collate_fn'):
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size)
    
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            x = batch['x'].to(device)
            emb = encoder(x)
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
            
            change_task(dataset, task= 'color')

        else:
            raise ValueError("The FashionMNIST dataset requires onehot_to_concepts to be set to True to change the task to color; otherwise, clothing is not causally connected to any concept")
    

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
        var_index = [all_var.index(var) for var in all_var]

        autoencoder_trainer = AutoencoderTrainer(autoencoder_cfg= dataset_cfg.autoencoder,
                                                 input_shape=len(all_var), 
                                                 device=device)
        dataset.split()
        dataset = autoencoder_trainer.train(dataset=dataset, 
                                            selected_var_index=var_index)
        dataset = scale_embeddings(dataset)

        # avoid empty spaces in the concepts names
        dataset.c_info['names'] = [concept.replace(' ', '_') for concept in dataset.c_info['names']]


        #dataset = maybe_reduce(cfg.dataset.get('reduce_fraction', None), dataset)
        #dataset = generate_img_embeddings(dataset, batch_size=cfg.dataset.get('batch_size'), device=device)

    elif dataset_name == 'nih_chest_images' or dataset_name == 'nih_chest_tabular':
        
        dataset.split()
        dataset = maybe_reduce(dataset_cfg.get('reduce_fraction', None), dataset)
        # check modality

        if dataset_name == 'nih_chest_images':
            backbone = 'res224-all'
            dataset = generate_img_embeddings(dataset, 
                                            batch_size= dataset_cfg.get('batch_size', 32), 
                                            device=device,
                                            backbone=backbone)

        else:
            scaler = StandardScaler()
            X_train = dataset.data['train'].X

            scaler.fit(X_train)

            for split in dataset.data:
                dataset.data[split].X = scaler.transform(dataset.data[split].X)

            dataset = generate_tabular_embeddings(dataset,
                                                  batch_size= dataset_cfg.get('batch_size', 32),
                                                  device=device)

    else:
        raise ValueError(f"Preprocessing is missing for dataset: {dataset_cfg.get('name')}")
    
    print('done')

    print(f"Concepts: {dataset.c_info['names']}")
    print(f"Task: {dataset.y_info['names']}")
    return dataset
