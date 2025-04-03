from env import CACHE

import os
import pickle
from hydra.utils import instantiate

from src.data.preprocessing import preprocess_dataset
from src.plots import maybe_plot_graph

def get_dataset(cfg):
    """
    1) instantiate the dataset, 
    2) split into train, val, test
    3) preprocess all of them 
    4) save the preprocessed dataset.
    Alternatively, if the 'cfg.dataset.load_embeddings' option is provided,
    load the stored dataset.
    Args:
        cfg: DictConfig
    Returns:
        dataset: the preprocessed dataset
    """
    dataset_directory = os.path.join(str(CACHE / cfg.dataset.name))
    os.makedirs(dataset_directory, exist_ok=True)

    destination_path = os.path.join(dataset_directory, "preprocessed_dataset.pkl")
    if cfg.dataset.get('load_embeddings') == False:
        dataset = instantiate(cfg.dataset.loader)
        dataset = preprocess_dataset(cfg, dataset, device=cfg.device)
        with open(destination_path, 'wb') as f: 
            pickle.dump(dataset, f)
    else:
        with open(destination_path, 'rb') as f: 
            dataset = pickle.load(f)
    
    true_graph = dataset.load_ground_truth_graph()
    maybe_plot_graph(true_graph, 'true_graph')
    return dataset, true_graph, dataset_directory