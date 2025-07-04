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
import subprocess

from hydra.utils import instantiate, call
from omegaconf import DictConfig, open_dict, OmegaConf

import pickle
from pathlib import Path
import warnings
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from src.utils import seed_everything, maybe_freeze_parameters, aggregate, get_parameters, set_parameters, remove_checkpoints, load_dataloaders

#from src.server import get_evaluate_fn
#from src.strategy import CustomFedAvgWithModelSaving
from src.utils import clean_empty_configs
from src.data.dataset_block import get_dataset
from src.utils import get_intervention_policy, remove_cycles, remove_problematic_edges, get_split_paths, get_split_paths_fl
from src.utils import clean_empty_configs, update_config_from_data, maybe_update_config_with_graph, update_intervention_policy_and_graph
from src.plots import maybe_plot_graph
from src.my_hydra import parse_hyperparams
from src.data.generate_split import generate_split, get_subgraph_dict
from collections import OrderedDict
from typing import List, Dict, Tuple
import copy

from env import CACHE
import shutil

# Suppress specific warning
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")
    

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

    [dataset.data[split].register_graph(graph) for split in dataset.data]
                    
    # We split the data by selecting a sub-graph for each split
    subgraphs, subgraphs_concept_names = generate_split(cfg, dataset, graph, y_index)

    # update config based on the dataset
    # e.g., set input and output size of the model
    cfg = update_config_from_data(cfg, dataset, subgraphs, subgraphs_concept_names)
    if cfg.learning.mode == 'localized':
        interv_policy, graph = update_intervention_policy_and_graph(cfg, interv_policy, graph, subgraphs, subgraphs_concept_names)  

    cfg = maybe_update_config_with_graph(cfg, graph, interv_policy)

    ############ data block ########################################################################################

    # Load the unique test-set
    test_dataloader = DataLoader(dataset.data['test'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
    # save the test dataloader
    test_path = os.path.join(str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption), "test_dataloader.pkl")
    with open(test_path, 'wb') as f:
        pickle.dump(test_dataloader, f)

    # Load the test dataloader for the specific client
    path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
    #test_path = get_split_paths(cfg, path, True)
    #with open(test_path, 'rb') as f:
    #    test_dataloader = pickle.load(f)

    # If the training is centralized
    if cfg.learning.mode in ['centralized', 'localized']:
        if cfg.learning.mode == 'localized':
            print("\033[93mLocalized training\033[0m")
            # Load only the training and validation split specified by the local training parameters
            # From cache get the dataloader
            train_path, val_path = get_split_paths(cfg, path)
            # if the file is not found, raise an error
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                raise FileNotFoundError(f"File {train_path} or {val_path} not found")
            # Load the dataloaders
            with open(train_path, 'rb') as f:
                train_dataloader = pickle.load(f)
                print("train_dataloader", train_dataloader)
                print("train_dataloader length", len(train_dataloader))
            with open(val_path, 'rb') as f:
                val_dataloader = pickle.load(f)            
        else:
            # Load all the training and validation splits
            train_dataloader = DataLoader(dataset.data['train'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
            val_dataloader = DataLoader(dataset.data['val'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)

        engine = instantiate(cfg.engine)
        try:
            trainer = Trainer(cfg)
            trainer.logger.log_hyperparams(parse_hyperparams(cfg))
            # ---- train
            trainer.fit(engine, train_dataloader, val_dataloader)
            # ----- test
            trainer.test(engine, test_dataloader, ckpt_path='best')
            trainer.logger.finalize("success")
        finally:
            if isinstance(trainer.logger, WandbLogger):
                trainer.logger.experiment.finish()
                
                
    elif cfg.learning.mode == 'local_federated':
        
        # hyperparameters   
        n_rounds = cfg.learning.settings.n_rounds
        n_clients = cfg.learning.n_clients 
        patience = cfg.learning.settings.patience
        cfg.trainer.max_epochs = cfg.learning.settings.local_epochs
        cfg.trainer.patience = 0
        path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
        num_threads = cfg.learning.settings.num_threads
        print(f"\033[93mLocal Federated training with {n_clients} clients\033[0m")
        
        # set seed for reproducibility
        torch.set_num_threads(num_threads)
        seed_everything(cfg.seed)
        
        # read client data
        train_dataloaders, val_dataloaders = load_dataloaders(cfg, path, n_clients)
        
        # try with and without these two lines
        engine = instantiate(cfg.engine)
        engine.model.to(cfg.device)
                
        t0 = time.time()
        best_loss = float('inf')
        best_round = 0
        no_improvement_count = 0
        history = {"round": [], "loss_val_avg": []}
        global_params = get_parameters(instantiate(cfg.engine))
        for rnd in range(1, n_rounds + 1):
            print(f"\033[92m\n--> ROUND {rnd}/{n_rounds}\033[0m")
            client_params: List[Tuple[List[torch.Tensor], int]] = []
            val_losses, sizes = [], []
            
            # ------------------------------------------------------------
            # local training (sequentially)
            # ------------------------------------------------------------
            print(f"\033[93mLocal training on {n_clients} clients\033[0m")
            for cid in range(n_clients):
                # clone global params → local model
                local_engine = instantiate(cfg.engine)
                set_parameters(local_engine, global_params)
                local_engine.model.to(cfg.device)

                # freeze if required
                maybe_freeze_parameters(
                    c=train_dataloaders[cid].dataset.c,
                    model=local_engine.model,
                    learning=cfg.learning.mode,
                    freezing=cfg.learning.settings.freezing,
                )

                # local train
                trainer = Trainer(cfg, client_id=cid)
                trainer.logger.log_hyperparams(parse_hyperparams(cfg)) 
                trainer.fit(local_engine, train_dataloaders[cid])
                n_samples = len(train_dataloaders[cid].dataset)

                # collect weights for aggregation
                client_params.append((get_parameters(local_engine), n_samples))
                
            # ------------------------------------------------------------
            # FedAvg aggregation
            # ------------------------------------------------------------
            print(f"\033[93mAggregating local models\033[0m")
            global_params = aggregate(client_params)
            print("Saving global model parameters")
            params_dict = zip(local_engine.model.state_dict().keys(), global_params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            local_engine.model.load_state_dict(state_dict, strict=True)
            
            # ------------------------------------------------------------
            # FedAvg aggregation on client validation sets
            # ------------------------------------------------------------
            print(f"\033[93mEvaluating on client validation sets\033[0m")
            local_engine = instantiate(cfg.engine)
            set_parameters(local_engine, global_params)
            local_engine.model.to(cfg.device)
            for cid in range(n_clients):
                val_metrics = trainer.validate(local_engine, val_dataloaders[cid])[0]  #{'val/c/asia': 0.0, 'val/c/bronc': 0.0, 'val/c/either': 0.0, 'val/c/lung': 0.0, 'val/c/smoke': 0.0, 'val/c/tub': 0.0, 'val/c/xray': 0.0, 'val_loss': nan}
                val_losses.append(val_metrics['val_loss'])
                sizes.append(len(val_dataloaders[cid].dataset))

            # log aggregated val metrics (weighted)
            w_loss = sum(l * s for l, s in zip(val_losses, sizes)) / sum(sizes)
            history["round"].append(rnd)
            history["loss_val_avg"].append(w_loss)
            print(f"\033[92m✅ aggregated  val_loss={w_loss:.4f}\033[0m")

            # check improvement
            if w_loss < best_loss:
                best_loss = w_loss
                best_round = rnd
                no_improvement_count = 0
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(local_engine.model.state_dict(), f"checkpoints/model_round_{rnd}.pth")
            else:
                no_improvement_count += 1

            # early stopping if no improvement after 'patience' rounds
            if no_improvement_count >= patience:
                print(f"\033[91mEarly stopping triggered at round {rnd}.\033[0m")
                break

        # ------------------------------------------------------------
        # Final evaluation on the test set
        # ------------------------------------------------------------
        print(f"\033[93mFinal evaluation on the test set\033[0m")
        # Load the best model
        ind_min_loss = np.argmin(history["loss_val_avg"])
        # best_round = history["round"][ind_min_loss]
        print(f"\033[92mBest round: {best_round} with loss {history['loss_val_avg'][ind_min_loss]:.4f}\033[0m")
        local_engine = instantiate(cfg.engine)
        local_engine.model.load_state_dict(torch.load(f"checkpoints/model_round_{best_round}.pth", weights_only=False))

        # Evaluate the model on the client datasets    
        trainer.test(engine, test_dataloader)
        print(f"\033[90mFinished! Training time: {round((time.time() - t0)/60, 2)} minutes\033[0m")
        
        
    elif cfg.learning.mode == 'federated':

        # Add path to cfg
        with open_dict(cfg):
            cfg.path = os.getcwd()

        # Save the full configuration to a temporary file
        config_filepath = "temp_config.yaml"
        with open(config_filepath, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        os.environ["config_path"] = os.path.join(os.getcwd(),config_filepath)
        print(f"Config file saved to {os.path.join(os.getcwd(),config_filepath)}")

        # Instantiate the FL training
        print("\033[93mFederated training\033[0m")
        # subprocess.Popen(["bash", "../../../../../src/fl_training.sh"])
        subprocess.run(["bash", "../../../../../src/fl_training.sh"])

        print("\033[93mFederated training completed.\033[0m")
        
        # Delete the temporary config file
        os.remove(config_filepath)
    else:
        raise ValueError('The learning mode is not supported. Please choose one of the following: centralized, localized, federated.')


if __name__ == "__main__":
    main()
