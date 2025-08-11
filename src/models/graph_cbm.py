import torch
import torch.nn as nn
from src.models.base import BaseModel
from src.models.layers.base import Dense, MLP
from src.models.layers.c_encoder import ConceptBlock
from src.models.layers.intervention import maybe_intervene
from src.utils import get_graph_levels, get_parents
from typing import Dict, Optional, Tuple


class GraphCBM(BaseModel):
    """
    Graph CBM: It propagates the information through a predefined graph of concepts.
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
                 prop_type='linear',
                 cat_latent=False,
                 c_name_index=None,
                 name: str = 'GraphCBM'):
        
        super(GraphCBM, self).__init__(
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
        
        # C2BM specific properties
        self.has_concepts = True
        self.is_causal = True
        self.concept_hidden_size = concept_hidden_size
        self.n_layers_concept_encoder = n_layers_concept_encoder
        self.n_layers_propagation = n_layers_propagation
        self.graph = torch.Tensor(graph).int() if graph is not None else None
        self.graph_labels = graph_labels
        self.prop_type = prop_type
        self.cat_latent = cat_latent
        
        # Setup concept loss weight
        self._setup_concept_loss_weight(concept_loss_weight)
        
        # Build the model
        self._build_model()

    def _build_model(self):
        """Build the C2BM model architecture."""
        # Encoder is already created in BaseModel
        
        # define concepts info parameters
        self.c_names = self.c_info['names'] # used later to retrieve which are concepts 
        self.y_names = self.y_info['names'] # and which are targets
        self.combo_info = {'names': self.c_info['names'] + self.y_info['names'],
                           'cardinality': self.c_info['cardinality'] + self.y_info['cardinality']}
        
        # sort c_names and graph_labels
        c2bm_graph = sorted(self.c_names + self.y_names)
        graph_labels = sorted(self.graph_labels)
        assert c2bm_graph == graph_labels

        # Concept encoders, one for each concept
        self.concept_encoders = nn.ModuleDict()
        for name in self.combo_info['names']:
            concept_idx = self.combo_info['names'].index(name)
            self.concept_encoders[name] = ConceptBlock(
                input_size=self.hidden_size,
                hidden_size=self.concept_hidden_size,
                n_layers=self.n_layers_concept_encoder,
                activation=self.activation,
                c_cardinality=self.combo_info['cardinality'][concept_idx]
            )

        # get levels
        task_index = self.combo_info['names'].index(self.y_names[0])
        graph_levels = get_graph_levels(self.graph, task_index)
        self.roots = graph_levels[0]
        self.roots_info = {'names': [name for i, name in enumerate(self.combo_info['names']) 
                                     if i in self.roots], 
                           'cardinality': [card for i, card in enumerate(self.combo_info['cardinality']) 
                                           if i in self.roots]}
        if self.y_names[0] in self.roots_info['names']:
            raise ValueError('The target variable cannot be a root concept')
        
        # get list of propagators
        self.propagators = nn.ModuleDict()
        for i in range(1, len(graph_levels)):
            level = graph_levels[i]
            self.propagators[str(i)] = nn.ModuleDict()
            for node in level:
                node_name = self.combo_info['names'][node]
                parents = get_parents(self.graph, node).tolist()
                node_cardinality = self.combo_info['cardinality'][node]
                parents_cardinality = [self.combo_info['cardinality'][p] for p in parents]
                
                if self.prop_type == 'embeddings':
                    self.propagators[str(i)][node_name] = MLP(
                        input_size=node_cardinality*self.concept_hidden_size,
                        hidden_size=self.concept_hidden_size,
                        output_size=node_cardinality,
                        n_layers=self.n_layers_propagation,
                        activation=self.activation
                    )
                elif self.prop_type == 'equations':
                    self.propagators[str(i)][node_name] = MLP(
                        input_size=node_cardinality*self.concept_hidden_size,
                        hidden_size=self.concept_hidden_size,
                        output_size=sum(parents_cardinality)*node_cardinality,
                        n_layers=self.n_layers_propagation,
                        activation=self.activation
                    )   

    def forward(self, x, c=None, intervention_index=None):
        """
        Forward pass of the C2BM model.
        
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

        c_embs, c_probs, c_values_emb = {}, {}, {}

        for name in self.combo_info['names']:

            # The concept annotation tensor c has shape (B, #total_concepts).
            # For this reason we need the self.c_name_index to access the position related
            # to the i-th concept
            i = self.c_name_index[name]
            
            # create embeddings and probabilities for each root concept
            # this assumes the task is last in the name list
            if name in self.roots_info['names']:
                c_input = c[:,i] if c is not None else None
                intervention_input = intervention_index[:,i] if intervention_index is not None else None
                c_embs[name], c_probs[name] = self.concept_encoders[name](
                    x_encoded, 
                    c_input, 
                    intervention_input,
                    to_return=['embs', 'probs']
                )
            elif name in self.c_names:
                # create latent for each non-root concept    
                # remember not to intervene on the task
                c_input = c[:,i] if name in self.c_names and c is not None else None
                intervention_input = intervention_index[:,i] if name in self.c_names and intervention_index is not None else None
                c_values_emb[name] = self.concept_encoders[name](
                    x_encoded, 
                    c_input, 
                    intervention_input,
                    to_return=['values_embs']
                )
            else: # it's the task variable
                c_values_emb[name] = self.concept_encoders[name](
                    x_encoded, 
                    None, 
                    None,
                    to_return=['values_embs']
                )

        # propagate the information through the causal graph
        for _, level in self.propagators.items():
            # update all nodes in the level
            for c_name, propagator in level.items():
                c_index = self.combo_info['names'].index(c_name)
                p_indices = get_parents(self.graph, c_index).tolist()
                p_names = [self.combo_info['names'][p] for p in p_indices]
                
                c_cardinality = self.combo_info['cardinality'][c_index]
                p_cardinality = [self.combo_info['cardinality'][p] for p in p_indices]
                # propagate embeddings
                c_prop_parents = torch.cat([c_probs[p_name] for p_name in p_names], dim=1).unsqueeze(-1)
                if self.prop_type == 'embeddings':
                    logits = propagator(c_values_emb[c_name]) # shape: (batch_size, c_cardinality)
                    c_probs[c_name] = torch.softmax(logits, dim=1)
                elif self.prop_type == 'equations':
                    weights = propagator(c_values_emb[c_name])
                    weights = weights.reshape(-1, c_cardinality, sum(p_cardinality))
                    c_probs[c_name] = torch.softmax(torch.matmul(weights, c_prop_parents).squeeze(-1), dim=1)

                if c_name not in self.y_names and c is not None and intervention_index is not None:
                    c_probs[c_name] = maybe_intervene(c_probs[c_name], c[:,c_index], intervention_index[:,c_index]) 

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
             ignore_index: int = -1) -> torch.Tensor:
        """
        Compute the loss function for C2BM model.
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