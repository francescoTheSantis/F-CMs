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

import argparse
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import flwr as fl

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import public.config as cfg
import public.utils as utils
import public.models as models


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
        model,
        client_id,
        train_loader,
        val_loader,
        device
        ):
        self.model = model
        self.client_id = client_id # [0,cfg.n_clients]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.drifting_log = []

        # plot
        self.metrics = {
            "rounds": [],
            "loss": [],
            "accuracy": []
        }

    # override
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # override
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # override
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        cur_round = config["current_round"]

        # Train the model   
        for epoch in range(config["local_epochs"]):
            models.simple_train(model=self.model,
                                device=self.device,
                                train_loader=self.train_loader, 
                                optimizer=torch.optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum),
                                epoch=epoch,
                                client_id=self.client_id)

        return self.get_parameters(config), len(self.train_loader.dataset), {}
    
    # override
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        cur_round = config["current_round"]

        loss_trad, accuracy_trad, f1_score_trad = models.simple_test(self.model, self.device, self.val_loader)

        # quick check results and save for plot
        print(f"Client {self.client_id} - Round {cur_round} - Loss: {loss_trad:.4f}, Accuracy: {accuracy_trad:.4f}")
        self.metrics["rounds"].append(cur_round)
        self.metrics["loss"].append(loss_trad)
        self.metrics["accuracy"].append(accuracy_trad)
        np.save(f"results/{cfg.default_path}/client_{self.client_id}_metrics.npy", self.metrics)

        return float(loss_trad), len(self.val_loader.dataset), {
            "accuracy": float(accuracy_trad),
            "f1_score": float(f1_score_trad)
        }

# main
def main() -> None:
    # Get client id
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--id",
        type=int,
        choices=range(0, cfg.n_clients),
        required=True,
        help="Specifies the artificial data partition",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=False,
        default=0,
        help="Specifies the fold number of the cross-validation",
    )
    args = parser.parse_args()

    # Load device, model and data
    utils.set_seed(cfg.random_seed + args.fold)
    device = utils.check_gpu(client_id=args.id)
    in_channels = utils.get_in_channels()
    model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                          input_size=cfg.input_size).to(device)
    
    # Load client data
    data = np.load(f"../../data/cur_datasets/client_{args.id}.npy", allow_pickle=True).item()
    
    # Split the data into training and testing subsets
    train_data, val_data = train_test_split(
       data, test_size=cfg.client_eval_ratio, random_state=cfg.random_seed
    )
        
    # reduce client data
    if cfg.n_samples_clients > 0:
        train_features = train_features[:cfg.n_samples_clients]
        train_labels = train_labels[:cfg.n_samples_clients]
        
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Start Flower client
    client = FlowerClient(model=model,
                          client_id=args.id,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          device=device
                          ).to_client()
    
    fl.client.start_client(server_address=f"{cfg.ip}:{cfg.port}", client=client) # local host

if __name__ == "__main__":
    main()
