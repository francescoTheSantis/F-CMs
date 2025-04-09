from env import CACHE
import flwr as fl
import numpy as np


class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                    cfg.dataset.name,
                    cfg.model._target_,
                    engine,
                    trainer,
                    client_id,
                    train_loader,
                    val_loader
                ):

        self.dataset_name = cfg.dataset.name
        self.model_name = cfg.model._target_.split('.')[-1]
        self.client_id = client_id # [0,cfg.n_clients]
        self.engine = engine
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader

    # override
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.engine.model.state_dict().items()]

    # override
    #def set_parameters(self, parameters):
    #    params_dict = zip(self.model.state_dict().keys(), parameters)
    #    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    #    self.model.load_state_dict(state_dict, strict=True)

    # override
    def fit(self, parameters, config):
        #self.set_parameters(parameters)
        self.trainer.fit(self.engine, self.train_dataloader, self.val_dataloader)
        return self.get_parameters(config), len(self.train_loader.dataset), {}
    
    # override
    def evaluate(self, parameters, config):
        #self.set_parameters(parameters)
        #cur_round = config["current_round"]

        # evaluate the model
        test_results = self.trainer.test(self.engine, self.test_dataloader)
        test_metrics = test_results[0] if isinstance(test_results, list) else test_results
        test_metrics = {k: v.item() if hasattr(v, "item") else v for k, v in test_metrics.items()}

        # save results
        np.save(f"CACHE/{self.dataset.name}/{self.model_name}/seed/client_{self.client_id}_metrics.npy", self.test_results)

        return len(self.val_loader.dataset), test_results