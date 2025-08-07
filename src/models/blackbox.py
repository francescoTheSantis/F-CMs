import torch
import torch.nn as nn

from src.models.layers.base import MLP

class BlackBox(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=2,
                 n_layers_encoder=1,
                 n_layers_decoder=1,
                 activation='leaky_relu',
                 c_info={},
                 y_info={},
                 name: str = 'BlackBox',
                 c_name_index=None):
        super(BlackBox, self).__init__()
        
        # to be stored for every model
        self.has_concepts = False
        self.is_causal = False
        self.name = name
        
        self.encoder = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers_encoder,
                           activation=activation)
        
        self.decoder = MLP(input_size=hidden_size,
                           hidden_size=hidden_size//2,
                           output_size=output_size,
                           n_layers=n_layers_decoder,
                           activation=activation)


    def forward(self, x, c=None, intervention_index=None):
        y_hidden = self.encoder(x)
        y_hat = self.decoder(y_hidden)
        return y_hat, None
    
    def filter_output_for_loss(self, y_output, c_output):
        return y_output, None
    
    def filter_output_for_metric(self, y_output, c_output):
        return y_output, None

    def loss(self, y_hat, y, c_hat_dict, c, reduction='mean', ignore_index=-100):
        """Compute the loss function.
        Args:
            y_hat: Predicted logits.
            y: True labels.
            (unused) c_hat_dict: Predicted concept probabilities.
            (unused) c: True concept values."""
        y = y.flatten().long()
        # cross entropy
        return torch.nn.functional.cross_entropy(y_hat, y, reduction=reduction, ignore_index=ignore_index)
