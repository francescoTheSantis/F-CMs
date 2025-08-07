from time import time

from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer as _Trainer_
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger
from torch import cuda

from env import PROJECT_NAME, WANDB_ENTITY
from hydra.core.hydra_config import HydraConfig
from src.my_hydra import parse_hyperparams
from wandb.sdk.lib.runid import generate_id
import os
import torch

class GradientMonitor_afterB(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        norms = []
        for p in pl_module.parameters():
            if p.grad is not None:
                norms.append(p.grad.norm().item())
        print(f"Gradient Norms after backward: {norms}")       
        
def _get_logger(cfg: DictConfig):
    name = f"seed{cfg.get('seed', '')}.{int(time())}"
    group_format = (
        "{dataset}.{causal_discovery}.{llm}.{rag}."
        "{model}.h{hidden_size}.lr{lr}"
    )
    group = group_format.format(**parse_hyperparams(cfg))
    if cfg.get("notes") is not None:
        group = f"{group}.{cfg.notes}"
    if cfg.trainer.logger == "wandb":
        logger = WandbLogger(
            project=PROJECT_NAME,
            entity=WANDB_ENTITY,
            log_model=True,
            id=generate_id(),
            save_dir=HydraConfig.get().runtime.output_dir,
            name=name,
            group=group,
        )
    else:
        raise ValueError(f"Unknown logger {cfg.trainer.logger}")
    return logger


class Trainer(_Trainer_):
    def __init__(self, cfg: DictConfig, client_id: int = None):
        callbacks = []
        if cfg.trainer.get("monitor", None) is not None:
            if cfg.trainer.get("patience", None) is not None:
                callbacks.append(
                    EarlyStopping(
                        monitor=cfg.trainer.monitor,
                        patience=cfg.trainer.patience,
                    )
                )                
                callbacks.append(
                ModelCheckpoint(
                    dirpath="checkpoints",
                    every_n_epochs=None,
                    monitor=cfg.trainer.monitor,
                    save_top_k=1,
                    mode="min",
                    save_last=True,
                    save_weights_only=False,
                )
            )
        callbacks.append(
            LearningRateMonitor(
                logging_interval="step",
            )
        )
        
        # for federated 
        if client_id is not None:
            # print(f"\033[94mClient ID: {client_id} in trainer\033[0m")
            ckpt_dir = os.path.join("checkpoints", f"client_{client_id}")
            callbacks.append(
                ModelCheckpoint(
                    dirpath=ckpt_dir,
                    every_n_epochs=None,
                    monitor="train_loss",
                    save_top_k=1,
                    mode="min",
                    save_last=True,
                    save_weights_only=False,
                )
            )
        callbacks.append(GradientMonitor_afterB())

        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = cfg.trainer.get("devices", torch.cuda.device_count())
        elif torch.backends.mps.is_available():      # Apple Silicon
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            dev_cfg = cfg.trainer.get("devices")
            if dev_cfg in (None, "auto"):
                devices = None           
            else:
                devices = int(dev_cfg)
                if devices <= 0:        
                    devices = 1
        
        if cfg.trainer.get("logger") is not None:
            logger = _get_logger(cfg)
        else:
            logger = DummyLogger()
        trainer_kwargs = {
            k: v
            for k, v in cfg.trainer.items()
            if k not in ["monitor", "patience", "logger"]
        }
        super().__init__(
            callbacks=callbacks,
            accelerator=accelerator,
            logger=logger,
            **trainer_kwargs,
        )
    
    def get_model(self):
        return self.model
