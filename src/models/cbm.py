import torch
import torch.nn as nn
from src.models.layers.base import MLP
from src.models.layers.intervention import maybe_intervene
from typing import Dict

class CBM(nn.Module):
    """
    Concept bottleneck model. It predicts both task and concept labels.
    """
    def __init__(self, 
                 input_size, 
                 hidden_size,
                 concept_hidden_size,
                 output_size=2,
                 n_layers_encoder=1,
                 n_layers_concept_encoder=1,
                 n_layers_decoder=1,
                 activation='leaky_relu',
                 concept_loss_weight=0.5,
                 decoder_type='mlp',
                 c_info={},
                 y_info={},
                 c_name_index=None,
                 name: str = 'CBM'):
        super(CBM, self).__init__()

        # to be stored for every model
        self.has_concepts = True
        self.is_causal = False
        self.c_name_index = c_name_index
        self.name = name

        # concepts info
        self.concept_names = c_info['names']
        self.virtual_roots = [name for name in c_info['names'] if name.startswith('#virtual_')]
        self.concept_loss_weight = concept_loss_weight
        self.filtered_c_info = {
            "names": [name for name in c_info["names"] if name not in self.virtual_roots],
            "cardinality": [card for name, card in zip(c_info["names"], c_info["cardinality"]) if name not in self.virtual_roots]
        }

        # Encoder
        self.encoder = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers_encoder,
                           activation=activation)
        
        ## Concept encoder
        #self.c_mlp = MLP(input_size=hidden_size,
        #                 hidden_size=concept_hidden_size,
        #                 output_size=sum(self.filtered_c_info['cardinality']),
        #                 n_layers=n_layers_concept_encoder,
        #                 activation=activation)

        # Concept encoders, one for each concept
        self.c_mlp = nn.ModuleDict()
        for name in self.filtered_c_info['names']:
            self.c_mlp[name] = MLP(input_size=hidden_size,
                                   hidden_size=concept_hidden_size,
                                   output_size=self.filtered_c_info['cardinality'][self.filtered_c_info['names'].index(name)],
                                   n_layers=n_layers_concept_encoder,
                                   activation=activation)
        
        # Decoder
        if decoder_type == 'mlp':
            self.decoder = MLP(input_size=sum(self.filtered_c_info['cardinality']),
                               hidden_size=sum(self.filtered_c_info['cardinality'])//2,
                               output_size=output_size,
                               n_layers=n_layers_decoder,
                               activation=activation)
        elif decoder_type == 'linear':
            self.decoder = nn.Linear(sum(self.filtered_c_info['cardinality']), 
                                     output_size)
        else:
            raise ValueError(f"Decoder type {decoder_type} not supported")

      
    def forward(self, x, c=None, intervention_index=None):
        x_encoded = self.encoder(x)
        # filter out virtual roots from c and intervention_index
        filtered_c = c[:, [v for k, v in self.c_name_index.items() if k in self.concept_names and k not in self.virtual_roots]]
        filtered_intervention_index = intervention_index[:, [v for k, v in self.c_name_index.items() if k in self.concept_names and k not in self.virtual_roots]]
        
        #c_hat_logits = self.c_mlp(x_encoded)
        c_hat_logits = {}
        c_hat_probs = {}
        for i, name in enumerate(self.filtered_c_info['names']):
            # Get the logits for the current concept
            #c_hat_probs[name] = torch.softmax(c_hat_logits[:,sum(self.filtered_c_info['cardinality'][:i]):sum(self.filtered_c_info['cardinality'][:i+1])], dim=1)
            c_hat_logits[name] = self.c_mlp[name](x_encoded)
            c_hat_probs[name] = torch.softmax(c_hat_logits[name], dim=1)
            # Maybe intervene on concepts probabilities
            if filtered_c[:,i] is not None and filtered_intervention_index[:,i] is not None:
                c_hat_probs[name] = maybe_intervene(c_hat_probs[name], filtered_c[:,i], filtered_intervention_index[:,i])
        
        c_probs_concat = torch.cat(list(c_hat_probs.values()), dim=1)
        # Task predictions
        y_hat_logits = self.decoder(c_probs_concat)
        y_hat_probs = torch.softmax(y_hat_logits, dim=1)
        return y_hat_probs, c_hat_probs
    
    def filter_output_for_loss(self, y_output, c_output):
        """Filter output for loss function"""
        return y_output, c_output
    
    def filter_output_for_metric(self, y_output, c_output):
        """Filter output for metric function"""
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
    #         total_loss =  self.concept_loss_weight * concept_loss + (1-self.concept_loss_weight) * task_loss
    #     return total_loss
    
    def loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        c_hat_dict: Dict[str, torch.Tensor],
        c: torch.Tensor,
        reduction: str = "mean",          # "mean" | "sum" | "none"
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
                pred_log, tgt, reduction=reduction
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

