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
        self.class_weights = None  # To be set by subclasses if needed
        self.concept_class_weights = {}  # Per-concept class weights {name: tensor([w_neg, w_pos])}
        
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

    def _concept_availability_checker(self, c, intervention_index):
        """
        Check for label absence and update intervention index accordingly.
        """
        # If the concept label is -1, it means the client has no access to the concept label.
        # For this reason the rand int cannot be applied.
        if c is not None and intervention_index is not None:
            for i, name in enumerate(list(self.c_name_index.keys())):
                # first, check if the name is a task variable
                if name in self.y_info['names']:
                    continue
                # Check if concept annotations are available
                if not (c[:,self.c_name_index[name]].long()!=-1).sum()==0:
                    continue
                else:
                    intervention_index[:,i] = torch.zeros_like(intervention_index[:,i], dtype=torch.int64)
        return intervention_index
    
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
             ignore_index: int = -1) -> torch.Tensor:
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
                         weight: Optional[torch.Tensor] = None,
                         reduction: str = "mean", 
                         ignore_index: int = -1) -> torch.Tensor:
        """
        Helper function to compute NLL loss with specified reduction.
        
        Args:
            pred_log: Log probabilities
            target: Target labels
            weight: Class weights tensor (optional)
            reduction: Reduction method
            ignore_index: Index to ignore
            
        Returns:
            Computed NLL loss
        """
        if weight is not None:
            weight = weight.to(pred_log.device)
            return torch.nn.functional.nll_loss(
                pred_log, target, weight=weight, reduction=reduction, ignore_index=ignore_index
            )
        else:
            return torch.nn.functional.nll_loss(
                pred_log, target, reduction=reduction, ignore_index=ignore_index
            )
        
    def _compute_task_loss(self, 
                          y_hat: torch.Tensor, 
                          y: torch.Tensor, 
                          reduction: str = "mean", 
                          ignore_index: int = -1) -> torch.Tensor:
        #y = y.flatten().long()

        # ----- task loss --------------------------------------------------------
        y_hat_log = torch.log(y_hat + 1e-6)
        task_loss = None
        if (y != ignore_index).any():  # at least one labelled sample
            task_loss = self._compute_nll_loss(y_hat_log, y, weight=self.class_weights, reduction=reduction, ignore_index=ignore_index)

        return task_loss

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
        
        count = 0
        for name, c_hat in c_hat_dict.items():
            if name not in self.c_name_index:
                continue
                
            idx = self.c_name_index[name]
            label = c[:, idx].long()
            
            # Skip if all labels are missing
            if (label != ignore_index).sum() == 0:
                continue

            c_hat_log = torch.log(c_hat + 1e-6)
            c_weight = self.concept_class_weights.get(name, None)
            concept_loss_i = self._compute_nll_loss(c_hat_log, label, weight=c_weight, reduction=reduction, ignore_index=ignore_index)
            
            if reduction == "none":
                concept_loss = concept_loss + concept_loss_i
                count += 1
            else:
                concept_loss = concept_loss + concept_loss_i
                count += 1
        
        concept_loss = concept_loss / max(count, 1)
        
        return concept_loss

    def _mix_losses(self,
                    task_loss: Optional[torch.Tensor],
                    concept_loss: torch.Tensor,
                    reduction: str = "mean") -> torch.Tensor:
        """
        Mix task and concept losses.

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
    
    def _concept_based_loss(self,
             y_hat: torch.Tensor,
             y: torch.Tensor,
             c_hat_dict: Dict[str, torch.Tensor],
             c: torch.Tensor,
             reduction: str = "mean",
             ignore_index: int = -1,
             multi_output: bool = False) -> torch.Tensor:
        """
        Compute the loss function for Concept-based models.
        
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
        task_loss = self._compute_task_loss(y_hat, y, reduction, ignore_index)

        # ----- concept loss -----------------------------------------------------
        concept_loss = self._compute_concept_loss(c_hat_dict, c, reduction, ignore_index)

        # ----- final mixture ----------------------------------------------------
        # If *all* task labels are missing, return only concept loss
        if (y == ignore_index).all():
            total_loss = concept_loss
        else:
            total_loss = self._mix_losses(task_loss, concept_loss, reduction)

        if multi_output:
            return task_loss, concept_loss, total_loss
        else:   
            return total_loss
