import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from src.models.base import BaseModel
from src.models.layers.base import MLP
from src.models.layers.c_encoder import ConceptBlock
from typing import Dict, Optional, Tuple

class CEM(BaseModel):
    """
    Adaptation of the Concept Embedding Model (CEM) (https://arxiv.org/abs/2209.09056)
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
                 c_info={},
                 y_info={},
                 c_name_index=None,
                 name: str = 'CEM'):
        
        super(CEM, self).__init__(
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

        # CEM specific properties
        self.has_concepts = True
        self.is_causal = False
        self.concept_hidden_size = concept_hidden_size
        self.n_layers_concept_encoder = n_layers_concept_encoder
        self.n_layers_decoder = n_layers_decoder
        
        # Setup concept loss weight
        self._setup_concept_loss_weight(concept_loss_weight)
        
        # Build the model
        self._build_model()

    def _build_model(self):
        """Build the CEM model architecture."""
        # Encoder is already created in BaseModel
        
        # Concept encoders, one for each concept
        self.concept_encoders = nn.ModuleDict()
        for name in self.concept_names:
            concept_idx = self.concept_names.index(name)
            self.concept_encoders[name] = ConceptBlock(
                input_size=self.hidden_size,
                hidden_size=self.concept_hidden_size,
                n_layers=self.n_layers_concept_encoder,
                activation=self.activation,
                c_cardinality=self.c_info['cardinality'][concept_idx]
            )
        
        # Decoder
        if self.output_size is not None:
            n_concepts = len(self.concept_names)
            self.decoder = MLP(
                input_size=n_concepts * self.concept_hidden_size,
                hidden_size=n_concepts * self.concept_hidden_size // 2,
                output_size=self.output_size,
                n_layers=self.n_layers_decoder,
                activation=self.activation
            )
    
    def forward(self, x, c=None, intervention_index=None):
        """
        Forward pass of the CEM model.
        
        Args:
            x: Input data tensor
            c: Concept labels (optional)
            intervention_index: Intervention indices (optional)
            
        Returns:
            Tuple of (task_predictions, concept_predictions)
        """
        # Encode input, get the latent features
        x_encoded = self.encoder(x)

        # Update intervention_index according to the annotation availability
        intervention_index = self._concept_availability_checker(c, intervention_index)

        # create embeddings and probabilities for each concept
        c_embeddings, c_hat_probs = {}, {}
        for name in self.concept_names:

            # The concept annotation tensor c has shape (B, #total_concepts).
            # For this reason we need the self.c_name_index to access the position related
            # to the i-th concept
            i = self.c_name_index[name]
            
            # Handle case where c or intervention_index might be None
            c_input = c[:, i] if c is not None else None
            intervention_input = intervention_index[:, i] if intervention_index is not None else None
            
            c_embeddings[name], c_hat_probs[name] = self.concept_encoders[name](
                x_encoded, 
                c_input, 
                intervention_input
            )
        
        # Concatenate all concept embeddings
        concat_embeddings = torch.cat(list(c_embeddings.values()), dim=1)
        
        # Decode, get task logits
        y_hat_logits = self.decoder(concat_embeddings)
        y_hat_probs = torch.softmax(y_hat_logits, dim=1)
        
        return y_hat_probs, c_hat_probs
    
    def filter_output_for_loss(self, y_output, c_output):
        """Filter output for loss function"""
        return y_output, c_output
    
    def filter_output_for_metric(self, y_output, c_output):
        """Filter output for metric function"""
        return y_output, c_output

    def loss(self,
             y_hat: torch.Tensor,
             y: torch.Tensor,
             c_hat_dict: Dict[str, torch.Tensor],
             c: torch.Tensor,
             reduction: str = "mean",
             ignore_index: int = -1) -> torch.Tensor:
        """
        Compute the loss function for CEM model.
        
        Args:
            y_hat: Predicted task logits/probabilities
            y: True task labels
            c_hat_dict: Predicted concept probabilities
            c: True concept labels
            reduction: Loss reduction method ("mean", "sum", "none")
            ignore_index: Index to ignore in loss computation
            
        Returns:
            Total loss combining task and concept losses
        """
        y = y.flatten().long()

        # ----- task loss --------------------------------------------------------
        y_hat_log = torch.log_softmax(y_hat, dim=1)
        task_loss = None
        if (y != ignore_index).any():  # at least one labelled sample
            task_loss = self._compute_nll_loss(y_hat_log, y, reduction, ignore_index)

        # ----- concept loss -----------------------------------------------------
        concept_loss = self._compute_concept_loss(c_hat_dict, c, reduction, ignore_index)

        # ----- final mixture ----------------------------------------------------
        # If *all* task labels are missing, return only concept loss
        if (y == ignore_index).all():
            total_loss = concept_loss
        else:
            total_loss = self._compute_total_loss(task_loss, concept_loss, reduction)

        return total_loss