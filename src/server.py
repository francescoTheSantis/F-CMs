"""
CFL implementation of fedavg, server side.

Code to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
run the appopriate client code (client.py).
"""

from typing import List, Tuple, Union, Optional, Dict
import numpy as np
# Compat for NumPy 2.0 removal; flwr still expects np.float_.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
import argparse
import torch
from torch.utils.data import DataLoader
from logging import WARNING
from collections import OrderedDict
import json
import time
from functools import reduce

import flwr as fl
from flwr.common import Parameters, Scalar, Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
)


import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig, open_dict, OmegaConf
from omegaconf import OmegaConf
from omegaconf import DictConfig
import pickle

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.utils import get_intervention_policy, remove_cycles, remove_problematic_edges, get_split_paths_fl
from src.my_hydra import parse_hyperparams
from src.utils import seed_everything, create_folders, plot_loss_and_accuracy, get_split_paths
from src.trainer import Trainer
from env import CACHE
import subprocess



# Custom weighted average function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    # validities = [num_examples * m["validity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Custom aggregate function
def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, saving_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model  # used for saving checkpoints
        self.saving_path = saving_path  # used for saving checkpoints
    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        ################################################################################
        # Federated averaging aggregation
        ################################################################################
        # Federated averaging - from traditional code
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_parameters_global = ndarrays_to_parameters(aggregate(weights_results))   # Global aggregation - traditional - no clustering
        
        # Aggregate custom metrics if aggregation fn was provided   
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
        ################################################################################
        # Save model
        ################################################################################
        if aggregated_parameters_global is not None:

            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters_global)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Save the model. TODO: save only best accuracy model and loss model
            torch.save(self.model.state_dict(), f"{self.saving_path}/model_round_{server_round}.pth")
        
        return aggregated_parameters_global, aggregated_metrics





@hydra.main(config_path="../conf", config_name="my_sweep", version_base="1.3")
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    
    # Read and Merge the external configuration
    config_filepath = os.getenv("config_path", "default_config.yaml") #TODO
    # config_filepath = "/home/dario/Desktop/Federated-C2BM/outputs/multirun/2025-04-09/17-46-33/0/temp_config.yaml"
    cfg = OmegaConf.load(config_filepath)
    # cfg = OmegaConf.merge(cfg, cfg_overrides) #TODO: merge?
    with open_dict(cfg): 
        root = cfg.path
    os.chdir(root)
    
    # hyperparameters
    num_threads = cfg.learning.settings.num_threads
    n_rounds = cfg.learning.settings.n_rounds
    local_epochs = cfg.learning.settings.local_epochs
    n_clients = cfg.learning.n_clients
    ip = cfg.learning.ip 
    port = cfg.learning.port
    print(f"\033[94mServer will run on {ip}:{port} with {n_clients} clients, {n_rounds} rounds and {local_epochs} local epochs.\033[0m")
    
    # various preliminaries, it set the seed for reproducibility
    torch.set_num_threads(num_threads)
    seed_everything(cfg.seed)
    create_folders()
    if torch.cuda.is_available():
        device = f"cuda:{cfg.trainer.devices[0]}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    with open_dict(cfg): cfg.update(device=device)
    print(f"Server uses {cfg.device} device")
    
    # Load test dataloader
    test_path = os.path.join(str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption), "test_dataloader.pkl")
    #path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
    #test_path = get_split_paths(cfg, path, True)
    with open(test_path, 'rb') as f:
        test_dataloader = pickle.load(f)
    
    # instantiate the engine
    engine = hydra.utils.instantiate(cfg.engine) 
    model = engine.model 

    # instantiate the trainer
    trainer = Trainer(cfg)
    trainer.logger.log_hyperparams(parse_hyperparams(cfg)) 
    
    # Config_client
    def fit_config(
            server_round: int
        ) -> Dict[str, Scalar]:
        """
            Generate training configuration dict for each round.
        """
        config = {
            "current_round": server_round,
            "local_epochs": local_epochs,
            "tot_rounds": n_rounds,
        }
        return config

    # Define strategy
    strategy = SaveModelStrategy(
        # self defined
        model=model,
        saving_path="checkpoints",
        # super
        min_fit_clients=n_clients, # always all training
        min_evaluate_clients=n_clients, # always all evaluating
        min_available_clients=n_clients, # always all available
        evaluate_metrics_aggregation_fn=weighted_average, #TODO
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
    )

    # Start Flower server and (finish all training and evaluation)
    history = fl.server.start_server(
        server_address=f"{ip}:{port}",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
    )

    # Convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    with open(f'histories/distributed_metrics.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)

    # Plot client training loss and accuracy
    # utils.plot_all_clients_metrics(fold=args.fold)

    # Plots and Evaluation the model on the client datasets, (averaged)
    best_loss_round, best_acc_round = plot_loss_and_accuracy(loss, accuracy, show=False)
    model.load_state_dict(torch.load(f"checkpoints/model_round_{best_loss_round}.pth", weights_only=False, map_location="cpu"))
    engine.model = model # probably unnecessary

    # Evaluate the model on the client datasets    
    trainer.test(engine, test_dataloader)
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    
    # Optionally, send a signal to terminate all clients after training is done
    def kill_clients():
        try:
            # This will kill all python processes running client.py
            subprocess.run(["pkill", "-u", "dario", "-f", "client.py"])
            print("All client.py processes have been terminated.")
        except Exception as e:
            print(f"Failed to kill client.py processes: {e}")
            
    # kill_clients()
    
    
if __name__ == "__main__":
    main()
