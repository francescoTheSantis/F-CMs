"""
This code creates a Flower client that can be used to train a model locally and share the updated 
model with the server. When it is started, it connects to the Flower server and waits for instructions.
If the server sends a model, the client trains the model locally and sends back the updated model.
If abilitated, at the end of the training the client evaluates the last model, and plots the 
metrics during the training.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
have this code running.
"""

from collections import OrderedDict

import torch
import flwr as fl

import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
from omegaconf import OmegaConf
from omegaconf import DictConfig
import pickle

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.utils import get_split_paths_fl
from src.my_hydra import parse_hyperparams
from src.utils import seed_everything
from src.trainer import Trainer
from env import CACHE


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
            engine,
            trainer,
            client_id,
            train_dataloader,
            val_dataloader,
        ):
        self.engine = engine
        self.trainer = trainer
        self.client_id = client_id # [0,cfg.n_clients]
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader


    # get the parameters of the model
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.engine.model.state_dict().items()]

    # set the parameters of the model
    def set_parameters(self, parameters):
        params_dict = zip(self.engine.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.engine.model.load_state_dict(state_dict, strict=True)

    # local training
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Local training   
        self.trainer.fit(self.engine, self.train_dataloader)

        return self.get_parameters(config), len(self.train_dataloader.dataset), {}
    
    # local evaluation after aggregation
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        cur_round = config["current_round"]

        # Local evaluation
        self.trainer.validate(self.engine, self.val_dataloader)
        
        # Get the loss and accuracy
        loss_trad = self.engine.trainer.callback_metrics["val_loss"].item()
        # accuracy_trad = self.engine.trainer.callback_metrics["test/c/asia"].item()
        accuracy_trad = 0.5

        return float(loss_trad), len(self.val_dataloader.dataset), {
            "accuracy": float(accuracy_trad),
        }

# main
@hydra.main(config_path="../conf", config_name="my_sweep", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Get client id
    client_id = cfg.client_id
    
    # Read and Merge the external configuration
    config_filepath = os.getenv("config_path", "default_config.yaml") #TODO
    # config_filepath = "/home/dario/Desktop/Federated-C2BM/outputs/multirun/2025-04-09/17-46-33/0/temp_config.yaml"
    cfg = OmegaConf.load(config_filepath)
    # cfg = OmegaConf.merge(cfg, cfg_overrides) #TODO: merge?
    
    # TODO: write properly in the federated.yaml
    cfg.trainer.max_epochs = cfg.get("local_epochs", 1)
    cfg.trainer.patience = 0
    
    # various preliminaries, it set the seed for reproducibility
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    os.makedirs('results', exist_ok=True)
    with open_dict(cfg): cfg.update(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Client {client_id} uses {cfg.device} device")
    
    # Load client data
    path = str(CACHE / cfg.dataset.name)
    train_path, val_path = get_split_paths_fl(cfg, path, client_id)
    # if the file is not found, raise an error
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"File {train_path} or {val_path} not found")
    # Load the dataloaders
    with open(train_path, 'rb') as f:
        train_dataloader = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_dataloader = pickle.load(f)  
            
    # instantiate the engine
    engine = hydra.utils.instantiate(cfg.engine)  

    # instantiate the trainer
    trainer = Trainer(cfg)
    trainer.logger.log_hyperparams(parse_hyperparams(cfg))  
    
    # Start Flower client
    client = FlowerClient(
        engine=engine,
        trainer=trainer,
        client_id=client_id,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    ).to_client()
    
    fl.client.start_client(server_address=f"{cfg.get('ip','0.0.0.0')}:{cfg.get('port', '8018')}", client=client) # local host

if __name__ == "__main__":
    main()
