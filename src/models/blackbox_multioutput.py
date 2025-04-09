import torch
import torch.nn as nn

from src.models.layers.base import MLP

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
                 c_name_index=None):
        super(BlackBox_Multi, self).__init__()
        
        # to be stored for every model
        self.has_concepts = False
        self.is_causal = False
        self.c_name_index = c_name_index

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
    
    def loss(self, y_hat, y, c_hat_dict, c):
        """Compute loss function.
        Args:
            y_hat (torch.Tensor): Predicted task logits
            y (torch.Tensor): True task labels
            c_hat_dict (Dict): Predicted concept logits
            c (torch.Tensor): True concept labels"""
        y = y.flatten().long()
        # c = c.long() # later to avoid nan to disappear
        loss_form = torch.nn.NLLLoss()

        # -- task loss
        y_hat = torch.log(y_hat + 1e-6)
        task_loss = loss_form(y_hat, y)

        # -- concepts loss
        concept_loss = 0
        for name, c_hat in c_hat_dict.items():
            if not (c[:,self.c_name_index[name]].long()!=-1).sum()==0:
                c_hat = torch.log(c_hat + 1e-6)
                concept_loss += loss_form(c_hat, c[:,self.c_name_index[name]].long())    

        return self.concept_loss_weight * concept_loss + (1-self.concept_loss_weight) * task_loss