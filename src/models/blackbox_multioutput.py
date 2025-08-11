import torch
import torch.nn as nn

from src.models.base import BaseModel
from src.models.layers.base import MLP
from typing import Dict, Optional, Tuple

class BlackBox_Multi(BaseModel):
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
        
        super(BlackBox_Multi, self).__init__(
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
        
        # BlackBox_Multi specific properties
        self.has_concepts = False  # It predicts concepts but doesn't use them for reasoning
        self.is_causal = False
        self.n_layers_decoder = n_layers_decoder
        self.concept_cardinality = c_info['cardinality']
        self.task_cardinality = y_info['cardinality']
        
        # Setup concept loss weight
        self._setup_concept_loss_weight(concept_loss_weight)
        
        # Build the model
        self._build_model()

    def _build_model(self):
        """Build the BlackBox_Multi model architecture."""
        # Encoder is already created in BaseModel

        # Calculate total output size (task + concepts)
        total_output_size = self.output_size + sum(self.concept_cardinality)
        alt_hidden_size = int(total_output_size * 2)

        # Update encoder with potentially larger hidden size
        self.encoder = MLP(
            input_size=self.input_size,
            hidden_size=max(self.hidden_size, alt_hidden_size),
            n_layers=self.n_layers_encoder,
            activation=self.activation
        )

        # Create decoder
        self.decoder = MLP(
            input_size=max(self.hidden_size, alt_hidden_size),
            hidden_size=alt_hidden_size,
            output_size=total_output_size,
            n_layers=self.n_layers_decoder,
            activation=self.activation
        )

    def forward(self, x, c=None, intervention_index=None):
        """
        Forward pass of the BlackBox_Multi model.
        
        Args:
            x: Input data tensor
            c: Concept labels (unused for BlackBox_Multi forward pass)
            intervention_index: Intervention indices (unused for BlackBox_Multi)
            
        Returns:
            Tuple of (task_predictions, concept_predictions)
        """
        y_hidden = self.encoder(x)
        all_hat_logits = self.decoder(y_hidden)
        
        # Extract task predictions
        y_hat_probs = torch.softmax(all_hat_logits[:, :self.task_cardinality[0]], dim=1)
        
        breakpoint()

        # Extract concept predictions
        c_hat_probs = {}
        for i, name in enumerate(self.concept_names):
            start_idx = self.task_cardinality[0] + sum(self.concept_cardinality[:i])
            end_idx = self.task_cardinality[0] + sum(self.concept_cardinality[:i+1])
            c_hat_probs[name] = torch.softmax(all_hat_logits[:, start_idx:end_idx], dim=1)
            
        return y_hat_probs, c_hat_probs
    
    def filter_output_for_loss(self, y_output, c_output):
        """Filter outputs for loss computation."""
        return y_output, c_output
    
    def filter_output_for_metric(self, y_output, c_output):
        """Filter outputs for metric computation."""
        return y_output, c_output

    def loss(self,
             y_hat: torch.Tensor,
             y: torch.Tensor,
             c_hat_dict: Dict[str, torch.Tensor],
             c: torch.Tensor,
             reduction: str = "mean",
             ignore_index: int = -1) -> torch.Tensor:
        """
        Compute the loss function for BlackBox_Multi model.
        """

        loss = self._concept_based_loss(
            y_hat,
            y,
            c_hat_dict,
            c,
            reduction,
            ignore_index
        )

        return loss