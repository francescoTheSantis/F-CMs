import torch
import torch.nn as nn

from src.models.layers.base import MLP
from typing import Dict

class BlackBox_Multi(nn.Module):
    """
    Opaque model with multiple outputs. It predicts both task and concept labels.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=2,
                 n_layers_encoder=1,
                 n_layers_decoder=1,
                 activation='leaky_relu',
                 concept_loss_weight=0.5,
                 c_info={},
                 y_info={},
                 c_name_index=None,
                 name: str = 'BlackBox_Multi'):
        super(BlackBox_Multi, self).__init__()
        
        # to be stored for every model
        self.has_concepts = False
        self.is_causal = False
        self.c_name_index = c_name_index
        self.name = name

        self.concept_names = c_info['names']
        self.concept_cardinality = c_info['cardinality']
        self.task_cardinality = y_info['cardinality']
        self.virtual_roots = [name for name in c_info['names'] if name.startswith('#virtual_')]
        vr_cardinality = [c_info['cardinality'][c_info['names'].index(name)] for name in self.virtual_roots]
        self.concept_loss_weight = concept_loss_weight

        output_size = output_size + sum(self.concept_cardinality) - sum(vr_cardinality)
        alt_hidden_size = int(output_size * 2)

        self.encoder = MLP(input_size=input_size,
                           hidden_size=max(hidden_size, alt_hidden_size),
                           n_layers=n_layers_encoder,
                           activation=activation)

        self.decoder = MLP(input_size=max(hidden_size, alt_hidden_size),
                           hidden_size=alt_hidden_size,
                           output_size=output_size,
                           n_layers=n_layers_decoder,
                           activation=activation)


    def forward(self, x, c=None, intervention_index=None):
        y_hidden = self.encoder(x)
        all_hat_logits = self.decoder(y_hidden)
        y_hat_probs = torch.softmax(all_hat_logits[:, :self.task_cardinality[0]], dim=1)
        c_hat_probs = {}
        for i, name in enumerate(self.concept_names):
            if name in self.virtual_roots: continue
            c_hat_probs[name] = torch.softmax(all_hat_logits[:, self.task_cardinality[0] + sum(self.concept_cardinality[:i]):
                                                               self.task_cardinality[0] + sum(self.concept_cardinality[:i+1])], dim=1)
        return y_hat_probs, c_hat_probs
    
    def filter_output_for_loss(self, y_output, c_output):
        return y_output, c_output
    
    def filter_output_for_metric(self, y_output, c_output):
        return y_output, c_output
    
    # def loss(self, y_hat, y, c_hat_dict, c):
    #     """Compute loss function.
    #     Args:
    #         y_hat (torch.Tensor): Predicted task logits
    #         y (torch.Tensor): True task labels
    #         c_hat_dict (Dict): Predicted concept logits
    #         c (torch.Tensor): True concept labels"""
    #     y = y.flatten().long()
    #     # c = c.long() # later to avoid nan to disappear
    #     loss_form = torch.nn.NLLLoss()

    #     # -- task loss
    #     y_hat = torch.log(y_hat + 1e-6)

    #     # -- concepts loss
    #     concept_loss = 0
    #     for name, c_hat in c_hat_dict.items():
    #         if not (c[:,self.c_name_index[name]].long()!=-1).sum()==0:
    #             c_hat = torch.log(c_hat + 1e-6)
    #             concept_loss += loss_form(c_hat, c[:,self.c_name_index[name]].long()) 

    #     if y[y== -1].numel() != 0:
    #         total_loss = concept_loss
    #     else:
    #         task_loss = loss_form(y_hat, y)
    #         total_loss = self.concept_loss_weight * concept_loss + (1-self.concept_loss_weight) * task_loss

    #     return total_loss

    def loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        c_hat_dict: Dict[str, torch.Tensor],
        c: torch.Tensor,
        reduction: str = "mean",          # "mean" | "sum" | "none"
        ignore_index: int = -100
    ):
        """
        Task-and-concept loss with an optional per-sample output.

        Args
        ----
        y_hat : (B, n_classes) logits **before** soft-max
        y     : (B,) task labels  (−1 marks “missing”)
        c_hat_dict : {name: (B, n_c_values) logits}
        c     : (B, n_concepts)   (−1 marks “missing”)
        reduction : how to reduce losses:
            - "mean" (default) : scalar identical to the old code
            - "none"           : length-B tensor with the loss for each sample
            - "sum"            : scalar sum of all samples
        """

        y = y.flatten().long()                       # (B,)

        # ----- helper that works for both reductions ---------------------------
        def nll(pred_log, tgt):
            return torch.nn.functional.nll_loss(
                pred_log, tgt, reduction=reduction, ignore_index=-1
            )                                        # shape → () or (B,)

        # ----- task loss --------------------------------------------------------
        y_hat_log = torch.log_softmax(y_hat, dim=1)  # log-p
        task_loss = None
        if (y != -1).any():                          # at least one labelled sample
            task_loss = nll(y_hat_log, y)

        # ----- concept loss -----------------------------------------------------
        if reduction == "none":
            # keep running total per sample
            concept_loss = torch.zeros_like(y_hat_log[:, 0])   # (B,)
        else:
            concept_loss = 0.0

        for name, c_hat in c_hat_dict.items():
            idx   = self.c_name_index[name]
            label = c[:, idx].long()                # (B,)
            if (label != -1).sum() == 0:
                continue                            # this concept is fully missing

            c_hat_log = torch.log_softmax(c_hat, dim=1)

            closs = nll(c_hat_log, label)           # shape as chosen above

            # accumulate
            if reduction == "none":
                concept_loss = concept_loss + closs  # per-sample add
            else:
                concept_loss = concept_loss + closs  # scalar add

        # ----- final mixture ----------------------------------------------------
        # If *all* task labels are missing, return only concept loss
        if (y == -1).all():
            total_loss = concept_loss
        else:
            if reduction == "none":
                total_loss = (
                    self.concept_loss_weight * concept_loss
                    + (1.0 - self.concept_loss_weight) * task_loss
                )                                    # element-wise
            else:
                total_loss = (
                    self.concept_loss_weight * concept_loss
                    + (1.0 - self.concept_loss_weight) * task_loss
                )

        return total_loss
