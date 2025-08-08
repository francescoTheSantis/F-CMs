import torch
import torch.nn as nn
from src.models.base import BaseModel
from src.models.layers.base import MLP
from src.models.layers.intervention import maybe_intervene
from typing import Dict, Optional, Tuple

class CBM(BaseModel):
    """
    Concept Bottleneck Model. It predicts both task and concept labels.
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 concept_hidden_size: int,
                 output_size: int = 2,
                 n_layers_encoder: int = 1,
                 n_layers_concept_encoder: int = 1,
                 n_layers_decoder: int = 1,
                 activation: str = 'leaky_relu',
                 concept_loss_weight: float = 0.5,
                 decoder_type: str = 'mlp',
                 c_info: Dict = {},
                 y_info: Dict = {},
                 c_name_index: Optional[Dict] = None,
                 name: str = 'CBM'):
        
        # Initialize base model
        super(CBM, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            c_info=c_info,
            y_info=y_info,
            c_name_index=c_name_index,
            name=name
        )

        # CBM-specific properties
        self.has_concepts = True
        self.is_causal = False
        self.concept_hidden_size = concept_hidden_size
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_concept_encoder = n_layers_concept_encoder
        self.n_layers_decoder = n_layers_decoder
        self.decoder_type = decoder_type

        # Setup concept loss weight
        self._setup_concept_loss_weight(concept_loss_weight)
        
        # Filter out virtual roots from concept info
        self.filtered_c_info = {
            "names": [name for name in c_info["names"] if name not in self.virtual_roots],
            "cardinality": [card for name, card in zip(c_info["names"], c_info["cardinality"]) 
                           if name not in self.virtual_roots]
        }

        # Build model architecture
        self._build_model()

    def _build_model(self):
        """Build the CBM architecture."""
        # Encoder
        self.encoder = MLP(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers_encoder,
            activation=self.activation
        )
        
        # Concept encoders - one for each concept
        self.c_mlp = nn.ModuleDict()
        for i, name in enumerate(self.filtered_c_info['names']):
            self.c_mlp[name] = MLP(
                input_size=self.hidden_size,
                hidden_size=self.concept_hidden_size,
                output_size=self.filtered_c_info['cardinality'][i],
                n_layers=self.n_layers_concept_encoder,
                activation=self.activation
            )
        
        # Decoder
        total_concept_dim = sum(self.filtered_c_info['cardinality'])
        if self.decoder_type == 'mlp':
            self.decoder = MLP(
                input_size=total_concept_dim,
                hidden_size=total_concept_dim // 2,
                output_size=self.output_size,
                n_layers=self.n_layers_decoder,
                activation=self.activation
            )
        elif self.decoder_type == 'linear':
            self.decoder = nn.Linear(total_concept_dim, self.output_size)
        else:
            raise ValueError(f"Decoder type {self.decoder_type} not supported")

    def forward(self, 
                x: torch.Tensor, 
                c: Optional[torch.Tensor] = None, 
                intervention_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the CBM.
        
        Args:
            x: Input data tensor
            c: Concept labels (optional)
            intervention_index: Intervention indices (optional)
            
        Returns:
            Tuple of (task_predictions, concept_predictions)
        """
        # Encode input
        x_encoded = self.encoder(x)
        
        # Filter out virtual roots from c and intervention_index if provided
        filtered_c = None
        filtered_intervention_index = None
        
        if c is not None:
            filtered_indices = [v for k, v in self.c_name_index.items() 
                              if k in self.concept_names and k not in self.virtual_roots]
            filtered_c = c[:, filtered_indices]
        
        if intervention_index is not None:
            filtered_indices = [v for k, v in self.c_name_index.items() 
                              if k in self.concept_names and k not in self.virtual_roots]
            filtered_intervention_index = intervention_index[:, filtered_indices]

        # Predict concepts
        c_hat_logits = {}
        c_hat_probs = {}
        
        for i, name in enumerate(self.filtered_c_info['names']):
            # Get logits for current concept
            c_hat_logits[name] = self.c_mlp[name](x_encoded)
            c_hat_probs[name] = torch.softmax(c_hat_logits[name], dim=1)
            
            # Maybe intervene on concept probabilities
            if (filtered_c is not None and filtered_intervention_index is not None and 
                i < filtered_c.shape[1] and i < filtered_intervention_index.shape[1]):
                c_hat_probs[name] = maybe_intervene(
                    c_hat_probs[name], 
                    filtered_c[:, i], 
                    filtered_intervention_index[:, i]
                )
        
        # Concatenate concept probabilities
        c_probs_concat = torch.cat(list(c_hat_probs.values()), dim=1)
        
        # Task predictions
        y_hat_logits = self.decoder(c_probs_concat)
        y_hat_probs = torch.softmax(y_hat_logits, dim=1)
        
        return y_hat_probs, c_hat_probs

    def loss(self, 
             y_hat: torch.Tensor,
             y: torch.Tensor,
             c_hat_dict: Optional[Dict[str, torch.Tensor]],
             c: Optional[torch.Tensor],
             reduction: str = "mean",
             ignore_index: int = -1) -> torch.Tensor:
        """
        Compute the CBM loss function.
        
        Args:
            y_hat: Predicted task probabilities
            y: True task labels
            c_hat_dict: Predicted concept probabilities
            c: True concept labels
            reduction: Loss reduction method
            ignore_index: Index to ignore in loss computation
            
        Returns:
            Computed loss tensor
        """
        y = y.flatten().long()
        
        # Task loss
        y_hat_log = torch.log_softmax(y_hat, dim=1)
        task_loss = None
        
        if (y != ignore_index).any():  # At least one labeled sample
            task_loss = self._compute_nll_loss(y_hat_log, y, reduction, ignore_index)
        
        # Concept loss
        concept_loss = torch.tensor(0.0, device=y_hat.device)
        
        if c_hat_dict and c is not None:
            for name, c_hat in c_hat_dict.items():
                if name in self.c_name_index:
                    concept_idx = self.c_name_index[name]
                    concept_labels = c[:, concept_idx].long()
                    
                    # Skip if no valid labels
                    if (concept_labels != ignore_index).sum() == 0:
                        continue
                    
                    # Compute concept loss
                    c_hat_log = torch.log(c_hat + 1e-6)
                    concept_loss += torch.nn.functional.nll_loss(
                        c_hat_log, concept_labels, reduction="mean", ignore_index=ignore_index
                    )
        
        # Total loss
        if (y == ignore_index).all():  # No task labels
            total_loss = concept_loss
        else:
            if task_loss is not None:
                total_loss = (
                    self.concept_loss_weight * concept_loss
                    + (1.0 - self.concept_loss_weight) * task_loss
                )
            else:
                total_loss = concept_loss
        
        return total_loss