import random
import numpy as np
import torch
import os
import warnings
import hydra
import pickle
from torch.utils.data import DataLoader
from src.data.utils import static_graph_collate
from pytorch_lightning.loggers import WandbLogger
from src.trainer import Trainer

from hydra.utils import instantiate, call
from omegaconf import DictConfig, open_dict, OmegaConf

import pickle
from pathlib import Path
import warnings

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig

#from src.server import get_evaluate_fn
#from src.strategy import CustomFedAvgWithModelSaving
from src.utils import clean_empty_configs
from src.data.dataset_block import get_dataset
from src.utils import get_intervention_policy, remove_cycles, remove_problematic_edges, get_split_paths, get_partitions
from src.utils import clean_empty_configs, update_config_from_data, maybe_update_config_with_graph, update_intervention_policy_and_graph
from src.plots import maybe_plot_graph
from src.hydra import parse_hyperparams
from src.data.generate_split import generate_split, get_subgraph_dict

from env import CACHE

# Suppress specific warning
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")
    
def seed_everything(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@hydra.main(config_path="conf", config_name="my_sweep", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # various preliminaries, it set the seed for reproducibility
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    os.mkdir('results')
    with open_dict(cfg): cfg.update(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {cfg.device} device")

    # adjust config
    cfg = clean_empty_configs(cfg)

    # instantiate the dataset, split into train, val, test
    # preprocess all of them and save the preprocessed dataset
    dataset, true_graph, dataset_directory = get_dataset(cfg)
    graph = true_graph

    print(OmegaConf.to_yaml(cfg))

    graph, dataset = remove_problematic_edges(graph, dataset)
    # (part 2): remove cycles
    y_index = list(graph.index).index(dataset.y_info['names'][0]); assert y_index == len(graph) - 1
    graph = remove_cycles(graph, y_index)
    maybe_plot_graph(graph, 'fixed_graph')

    #partitions = get_partitions(graph, y_index, n_clients = 5)
    #col_to_index = {col: idx for idx, col in enumerate(graph.columns)}    
    #partitions_named = {
    #    term: [graph.columns[i] for i in indices]
    #    for term, indices in partitions.items()
    #}

    # use the graph to define an intervention policy at test time
    interv_policy, ip_names = get_intervention_policy(graph, y_index)
    print('intervention policy:', interv_policy)
    print('intervention policy names:', ip_names)

    # update config based on the dataset
    # e.g., set input and output size of the model
    cfg = update_config_from_data(cfg, dataset)
    if cfg.learning.mode == 'localized':
        interv_policy, graph = update_intervention_policy_and_graph(cfg, interv_policy, graph)
    cfg = maybe_update_config_with_graph(cfg, graph, interv_policy)

    ############ data block ########################################################################################
    [dataset.data[split].register_graph(graph) for split in dataset.data]
           
    # We split the data by selecting a sub-graph for each split
    generate_split(cfg, dataset, graph)

    # If the training is centralized
    if cfg.learning.mode in ['centralized', 'localized']:
        if cfg.learning.mode == 'localized':
            # Load only the training and validation split specified by the local training parameters
            # From cache get the dataloader
            path = str(CACHE / cfg.dataset.name)
            train_path, val_path = get_split_paths(cfg, path)
            # if the file is not found, raise an error
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                raise FileNotFoundError(f"File {train_path} or {val_path} not found")
            # Load the dataloaders
            with open(train_path, 'rb') as f:
                train_dataloader = pickle.load(f)
            with open(val_path, 'rb') as f:
                val_dataloader = pickle.load(f)            
        else:
            # Load all the training and validation splits
            train_dataloader = DataLoader(dataset.data['train'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
            val_dataloader = DataLoader(dataset.data['val'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
        # Load the unique test-set
        test_dataloader = DataLoader(dataset.data['test'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)

        engine = instantiate(cfg.engine)
        try:
            trainer = Trainer(cfg)
            trainer.logger.log_hyperparams(parse_hyperparams(cfg))
            # ---- train
            trainer.fit(engine, train_dataloader, val_dataloader)
            # ----- test
            trainer.test(engine, test_dataloader)
            trainer.logger.finalize("success")
        finally:
            if isinstance(trainer.logger, WandbLogger):
                trainer.logger.experiment.finish()
    elif cfg.learning.mode == 'federated':
        pass
    else:
        raise ValueError('The learning mode is not supported. Please choose one of the following: centralized, localized, federated.')


if __name__ == "__main__":
    main()
