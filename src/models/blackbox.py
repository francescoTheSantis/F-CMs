import torch
import torch.nn as nn

from src.models.base import BaseModel
from src.models.layers.base import MLP

class BlackBox(BaseModel):
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
        
        super(BlackBox, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            n_layers_encoder=n_layers_encoder,
            activation=activation,
            c_info=c_info,
            y_info=y_info,
            c_name_index=c_name_index,
            name=name
        )
        
        # BlackBox specific properties
        self.has_concepts = False
        self.is_causal = False
        self.n_layers_decoder = n_layers_decoder
        
        # Build the model
        self._build_model()

    def _build_model(self):
        """Build the BlackBox model architecture."""
        # Encoder is already created in BaseModel
        
        # Create decoder
        self.decoder = MLP(input_size=self.hidden_size,
                           hidden_size=self.hidden_size//2,
                           output_size=self.output_size,
                           n_layers=self.n_layers_decoder,
                           activation=self.activation)

    def forward(self, x, c=None, intervention_index=None):
        """
        Forward pass of the BlackBox model.
        
        Args:
            x: Input data tensor
            c: Concept labels (unused for BlackBox)
            intervention_index: Intervention indices (unused for BlackBox)
            
        Returns:
            Tuple of (task_predictions, None) since BlackBox doesn't predict concepts
        """
        y_hidden = self.encoder(x)
        y_hat = self.decoder(y_hidden)
        return y_hat, None
    
    def filter_output_for_loss(self, y_output, c_output):
        """Filter outputs for loss computation - BlackBox only uses task output."""
        return y_output, None
    
    def filter_output_for_metric(self, y_output, c_output):
        """Filter outputs for metric computation - BlackBox only uses task output."""
        return y_output, None

    def loss(self, y_hat, y, c_hat_dict, c, reduction='mean', ignore_index=-100):
        """
        Compute the loss function for BlackBox model.
        
        Args:
            y_hat: Predicted logits.
            y: True labels.
            c_hat_dict: Predicted concept probabilities (unused for BlackBox).
            c: True concept values (unused for BlackBox).
            reduction: Loss reduction method.
            ignore_index: Index to ignore in loss computation.
            
        Returns:
            Cross-entropy loss for the task prediction.
        """

        y = y.flatten().long()

        loss =  self._compute_nll_loss(
            pred_log=torch.nn.functional.log_softmax(y_hat, dim=-1),
            target=y,
            reduction=reduction,
            ignore_index=ignore_index
        )

        return loss