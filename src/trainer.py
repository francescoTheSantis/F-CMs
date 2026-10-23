from time import time

from omegaconf import DictConfig, ListConfig
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
        # if norms:
        #     total_norm = (sum(n ** 2 for n in norms) ** 0.5)
        #     max_norm = max(norms)
        #     print(f"[GradNorm] step={trainer.global_step} total={total_norm:.6f} max_param={max_norm:.6f}")


# ------------------------ EpochLossPrinter callback ------------------------
class EpochLossPrinter(pl.Callback):
    """Prints train/val loss at the end of each training epoch.
    Relies on metrics logged in the LightningModule via self.log(...).
    It tries common keys like 'train_loss', 'train_loss_epoch', 'val_loss', 'val_loss_epoch'.
    """
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        def _to_float(x):
            if x is None:
                return None
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().item()
            except Exception:
                pass
            try:
                return float(x)
            except Exception:
                return None

        train_loss = metrics.get("train_loss_epoch", metrics.get("train_loss", None))
        val_loss = metrics.get("val_loss", metrics.get("val_loss_epoch", None))

        train_loss_f = _to_float(train_loss)
        val_loss_f = _to_float(val_loss)

        train_str = f"{train_loss_f:.6f}" if isinstance(train_loss_f, (int, float)) else "N/A"
        val_str = f"{val_loss_f:.6f}" if isinstance(val_loss_f, (int, float)) else "N/A"

        epoch = getattr(trainer, "current_epoch", None)
        epoch_str = str(epoch) if epoch is not None else "?"
        print(f"\n[Epoch {epoch_str}] train_loss={train_str} | val_loss={val_str}")
        
def _get_logger(cfg: DictConfig):
    name = f"seed{cfg.get('seed', '')}.{int(time())}"
    # group_format = (
    #     "{dataset}.{causal_discovery}.{llm}.{rag}."
    #     "{model}.h{hidden_size}.lr{lr}"
    # )
    group_format = (
        "{dataset}.{model}.{learning}"
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
        if cfg.trainer.get("monitor", None) is not None and cfg.learning.mode != 'local_federated':
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
        
        # for federated 
        else:
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
        callbacks.append(EpochLossPrinter())
        callbacks.append(
            LearningRateMonitor(
                logging_interval="step",
            )
        )
        
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
            elif isinstance(dev_cfg, (list, tuple, ListConfig)):
                devices = max(1, len(dev_cfg))
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
            if k not in ["monitor", "patience", "logger", "devices"]
        }
        super().__init__(
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            **trainer_kwargs,
        )
    
    def get_model(self):
        return self.model
