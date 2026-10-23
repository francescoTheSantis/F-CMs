import torch
import torch.nn as nn
from src.models.base import BaseModel
from src.models.layers.base import Dense, MLP
from src.models.layers.c_encoder import ConceptBlock
from src.models.layers.intervention import maybe_intervene
from src.utils import get_graph_levels, get_parents
from typing import Dict, Optional, Tuple


class CGM(BaseModel):
    """
    CGM: It propagates the information through a predefined graph of concepts.
    """
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 concept_hidden_size,
                 output_size=2,
                 n_layers_encoder=1,
                 n_layers_concept_encoder=1,
                 n_layers_propagation=1,
                 activation='leaky_relu',
                 concept_loss_weight=0.5,
                 c_info={},
                 y_info={},
                 graph=None,
                 graph_labels=None,
                 cat_latent=False,
                 c_name_index=None,
                 name: str = 'CGM'):
        
        super(CGM, self).__init__(
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
        
        # CGM specific properties
        self.has_concepts = True
        self.is_causal = True
        self.concept_hidden_size = concept_hidden_size
        self.n_layers_concept_encoder = n_layers_concept_encoder
        self.n_layers_propagation = n_layers_propagation
        self.graph = torch.Tensor(graph).int() if graph is not None else None
        self.graph_labels = graph_labels
        self.cat_latent = cat_latent
        
        # Setup concept loss weight
        self._setup_concept_loss_weight(concept_loss_weight)
        
        # Build the model
        self._build_model()

    def _build_model(self):
        """Build the CGM model architecture."""
        # Encoder is already created in BaseModel
        
        # define concepts info parameters
        self.c_names = self.c_info['names'] # used later to retrieve which are concepts 
        self.y_names = self.y_info['names'] # and which are targets
        self.combo_info = {'names': self.c_info['names'] + self.y_info['names'],
                           'cardinality': self.c_info['cardinality'] + self.y_info['cardinality']}
        
        # check order self.combo_info['names'] matches graph_labels
        assert self.combo_info['names'] == self.graph_labels
        if len(self.combo_info['names']) == len(self.c_name_index.keys()):
            # check they have the same order
            assert self.combo_info['names'] == list(self.c_name_index.keys())
        
        # sort c_names and graph_labels
        #c2bm_graph_ordered = sorted(self.c_names + self.y_names)
        #graph_labels_ordered = sorted(self.graph_labels)
        #assert c2bm_graph_ordered == graph_labels_ordered

        # indentify levels and roots
        # get levels
        task_index = self.combo_info['names'].index(self.y_names[0])
        self.graph_levels = get_graph_levels(self.graph, task_index)
        self.roots = self.graph_levels[0]
        # flat graph_levels
        flat_levels = [node for level in self.graph_levels for node in level]
        # find self.predicted_concepts from self.c_name_index
        # invert the dictionary self.c_name_index
        #index_c_name = {v:k for k,v in self.c_names.items()}
        self.predicted_concepts = [self.graph_labels[i] for i in flat_levels ]
        # eliminate task from predicted concepts
        self.predicted_concepts = [c for c in self.predicted_concepts if c != self.y_names[0]]
        self.roots_info = {'names': [name for i, name in enumerate(self.combo_info['names']) 
                                     if i in self.roots], 
                           'cardinality': [card for i, card in enumerate(self.combo_info['cardinality']) 
                                           if i in self.roots]}
        if self.y_names[0] in self.roots_info['names']:
            raise ValueError('The target variable cannot be a root concept')

        # create the dictionary of concept encoders
        self.concept_encoders = nn.ModuleDict()
        for name in self.combo_info['names']:
            if name in self.roots_info['names']:
                self.concept_encoders[name] = ConceptBlock(
                    input_size=self.hidden_size,
                    hidden_size=self.concept_hidden_size,
                    n_layers=self.n_layers_concept_encoder,
                    activation=self.activation,
                    c_cardinality=[c for n, c in zip(self.combo_info['names'], self.combo_info['cardinality']) if n==name][0] #self.combo_info['cardinality'][concept_idx]
                )   
            else:
                node_idx = self.combo_info['names'].index(name)
                parents = get_parents(self.graph, node_idx).tolist()
                self.concept_encoders[name] = ConceptBlock(
                    input_size=len(parents) * self.concept_hidden_size,
                    hidden_size=self.concept_hidden_size,
                    n_layers=self.n_layers_concept_encoder,
                    activation=self.activation,
                    c_cardinality=[c for n, c in zip(self.combo_info['names'], self.combo_info['cardinality']) if n==name][0] #self.combo_info['cardinality'][concept_idx]
                )               

    def forward(self, x, c=None, intervention_index=None):
        """
        Forward pass of the CGM model.
        
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

        c_embs, c_probs = {}, {}

        for level in self.graph_levels: # skip the task level
            for i in level:
                name = self.combo_info['names'][i]
                c_index = self.c_name_index[name]

                if name in self.roots_info['names']:
                    concept_encoder_input = x_encoded
                    c_int = c[:,c_index] if c is not None else None
                    int_idx = intervention_index[:,c_index] if intervention_index is not None else None
                elif name in self.c_names:
                    p_indices = get_parents(self.graph, i).tolist()
                    p_names = [self.combo_info['names'][p] for p in p_indices]
                    concept_encoder_input = torch.cat([c_embs[p_name] for p_name in p_names], dim=1)
                    c_int = c[:,c_index] if c is not None else None
                    int_idx = intervention_index[:,c_index] if intervention_index is not None else None
                else: # it's the task variable
                    p_indices = get_parents(self.graph, i).tolist()
                    p_names = [self.combo_info['names'][p] for p in p_indices]
                    concept_encoder_input = torch.cat([c_embs[p_name] for p_name in p_names], dim=1)
                    c_int = None
                    int_idx = None

                c_embs[name], c_probs[name] = self.concept_encoders[name](
                    concept_encoder_input, 
                    c_int, 
                    int_idx,
                    to_return=['embs', 'probs']
                )

                # Update the probabilities if there is an intervention
                if name not in self.y_names and c is not None and intervention_index is not None:
                    c_probs[name] = maybe_intervene(c_probs[name], c[:,c_index], intervention_index[:,c_index]) 

        # Decode, get task logits
        y_hat_probs = c_probs[self.y_names[0]]
        # filter virtual roots
        c_hat_probs = {k:v for k,v in c_probs.items() if k in self.c_names and k not in self.virtual_roots}
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
             ignore_index: int = -1,
             multi_output: bool = False) -> torch.Tensor:
        """
        Compute the loss function for C2BM model.
        """

        loss = self._concept_based_loss(
            y_hat,
            y,
            c_hat_dict,
            c,
            reduction,
            ignore_index,
            multi_output=multi_output
        )

        return loss
