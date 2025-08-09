import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.models.layers.base import MLP
from typing import Dict, Optional, Tuple

class BaseModel(nn.Module, ABC):
    """
    Base class for all the models: both blackboxes and concept-based models.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int = 2,
                 n_layers_encoder: int = 1,
                 activation: str = 'leaky_relu',
                 c_info: Dict = {},
                 y_info: Dict = {},
                 c_name_index: Optional[Dict] = None,
                 name: str = 'BaseModel'):
        super(BaseModel, self).__init__()
        
        # Common model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.c_info = c_info
        self.y_info = y_info
        self.c_name_index = c_name_index
        self.name = name
        
        # To be set by subclasses
        self.has_concepts = False
        self.is_causal = False
        
        # Concept-related properties (for concept-based models)
        if c_info:
            self.concept_names = c_info.get('names', [])
            self.virtual_roots = [name for name in self.concept_names if name.startswith('#virtual_')]
        else:
            self.concept_names = []
            self.virtual_roots = []

        # Encoder: each model will implement the same encoder
        self.encoder = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers_encoder,
                           activation=activation)
    
    @abstractmethod
    def _build_model(self):
        """
        Abstract method to build the model architecture.
        Each subclass should implement this method.
        """
        pass

    @abstractmethod
    def forward(self, 
                x: torch.Tensor, 
                c: Optional[torch.Tensor] = None, 
                intervention_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass of the model.
        
        Args:
            x: Input data tensor
            c: Concept labels (optional)
            intervention_index: Intervention indices (optional)
            
        Returns:
            Tuple of (task_predictions, concept_predictions)
        """
        pass
    
    def filter_output_for_loss(self, 
                              y_output: torch.Tensor, 
                              c_output: Optional[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Filter model outputs for loss computation.
        Default implementation returns outputs as-is.
        
        Args:
            y_output: Task predictions
            c_output: Concept predictions
            
        Returns:
            Filtered outputs for loss computation
        """
        return y_output, c_output
    
    def filter_output_for_metric(self, 
                                y_output: torch.Tensor, 
                                c_output: Optional[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Filter model outputs for metric computation.
        Default implementation returns outputs as-is.
        
        Args:
            y_output: Task predictions
            c_output: Concept predictions
            
        Returns:
            Filtered outputs for metric computation
        """
        return y_output, c_output
    
    @abstractmethod
    def loss(self, 
             y_hat: torch.Tensor,
             y: torch.Tensor,
             c_hat_dict: Optional[Dict[str, torch.Tensor]],
             c: Optional[torch.Tensor],
             reduction: str = "mean",
             ignore_index: int = -100) -> torch.Tensor:
        """
        Compute the loss function.
        
        Args:
            y_hat: Predicted task logits/probabilities
            y: True task labels
            c_hat_dict: Predicted concept probabilities (optional)
            c: True concept labels (optional)
            reduction: Loss reduction method ("mean", "sum", "none")
            ignore_index: Index to ignore in loss computation
            
        Returns:
            Computed loss tensor
        """
        pass
    
    def _setup_concept_loss_weight(self, concept_loss_weight: float = 0.5):
        """
        Setup concept loss weight for concept-based models.
        
        Args:
            concept_loss_weight: Weight for concept loss in total loss
        """
        self.concept_loss_weight = concept_loss_weight
    
    def _compute_nll_loss(self, 
                         pred_log: torch.Tensor, 
                         target: torch.Tensor, 
                         reduction: str = "mean", 
                         ignore_index: int = -100) -> torch.Tensor:
        """
        Helper function to compute NLL loss with specified reduction.
        
        Args:
            pred_log: Log probabilities
            target: Target labels
            reduction: Reduction method
            ignore_index: Index to ignore
            
        Returns:
            Computed NLL loss
        """
        return torch.nn.functional.nll_loss(
            pred_log, target, reduction=reduction, ignore_index=ignore_index
        )
    
    def _compute_concept_loss(self, 
                             c_hat_dict: Dict[str, torch.Tensor], 
                             c: torch.Tensor, 
                             reduction: str = "mean", 
                             ignore_index: int = -1) -> torch.Tensor:
        """
        Helper function to compute concept loss for concept-based models.
        
        Args:
            c_hat_dict: Predicted concept probabilities
            c: True concept labels
            reduction: Reduction method
            ignore_index: Index to ignore
            
        Returns:
            Computed concept loss
        """
    
        if not self.has_concepts or not c_hat_dict:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        if reduction == "none":
            # Initialize per-sample concept loss
            batch_size = next(iter(c_hat_dict.values())).size(0)
            concept_loss = torch.zeros(batch_size, device=next(self.parameters()).device)
        else:
            concept_loss = 0.0
        
        for name, c_hat in c_hat_dict.items():
            if name not in self.c_name_index:
                continue
                
            idx = self.c_name_index[name]
            label = c[:, idx].long()
            
            # Skip if all labels are missing
            if (label != ignore_index).sum() == 0:
                continue
            
            c_hat_log = torch.log_softmax(c_hat, dim=1)
            concept_loss_i = self._compute_nll_loss(c_hat_log, label, reduction, ignore_index)
            
            if reduction == "none":
                concept_loss = concept_loss + concept_loss_i
            else:
                concept_loss = concept_loss + concept_loss_i
        
        return concept_loss

    def _compute_total_loss(self,
                           task_loss: Optional[torch.Tensor],
                           concept_loss: torch.Tensor,
                           reduction: str = "mean") -> torch.Tensor:
        """
        Compute the total loss combining task and concept losses.
        
        Args:
            task_loss: Task loss tensor
            concept_loss: Concept loss tensor
            reduction: Reduction method
            
        Returns:
            Total loss tensor
        """

        if reduction == "none":
            total_loss = (
                self.concept_loss_weight * concept_loss
                + (1.0 - self.concept_loss_weight) * task_loss
            )  # element-wise                                  
        else:
            total_loss = (
                self.concept_loss_weight * concept_loss
                + (1.0 - self.concept_loss_weight) * task_loss
            )

        return total_loss