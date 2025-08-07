import torch
import torch.nn as nn
from src.models.layers.base import Dense, MLP
from src.models.layers.c_encoder import ConceptBlock
from src.models.layers.intervention import maybe_intervene
from src.utils import get_graph_levels, get_parents
from typing import Dict


class C2BM(nn.Module):
    """
    Causal version of the CEM model. 
    It propagates the information through a predefined causal graph.
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
                 name: str = 'C2BM'):
        super(C2BM, self).__init__()
        
        # to be stored for every model
        self.has_concepts = True
        self.is_causal = True
        self.c_name_index = c_name_index
        self.name = name

        # define concepts info parameters
        self.c_names = c_info['names'] # used later to retrieve which are concepts 
        self.y_names = y_info['names'] # and which are targerts
        self.virtual_roots = [name for name in c_info['names'] if name.startswith('#virtual_')]
        self.combo_info = {'names': c_info['names'] + y_info['names'],
                           'cardinality': c_info['cardinality'] + y_info['cardinality']}
        
        # sort c_names and graph_labels
        c2bm_graph = sorted(self.c_names + self.y_names)
        graph_labels = sorted(graph_labels)
        assert c2bm_graph == graph_labels

        # Encoder
        self.encoder = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers_encoder,
                           activation=activation)

        # Concept encoders, one for each concept
        self.concept_encoders = nn.ModuleDict()
        for name in self.combo_info['names']:
            self.concept_encoders[name] = ConceptBlock(input_size=hidden_size,
                                                       hidden_size=concept_hidden_size,
                                                       n_layers=n_layers_concept_encoder,
                                                       activation=activation,
                                                       c_cardinality=self.combo_info['cardinality'][self.combo_info['names'].index(name)])
        self.concept_hidden_size = concept_hidden_size
        self.concept_loss_weight = concept_loss_weight

        # get levels
        self.graph = torch.Tensor(graph).int()
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
        self.prop_type = prop_type
        self.cat_latent = cat_latent
        if prop_type in ['dense', 'mlp', 'embeddings', 'equations']:
            self.propagators = nn.ModuleDict()
            for i in range(1, len(graph_levels)):
                level = graph_levels[i]
                self.propagators[str(i)] = nn.ModuleDict()
                for node in level:
                    node_name = self.combo_info['names'][node]
                    parents = get_parents(self.graph, node).tolist()
                    node_cardinality = self.combo_info['cardinality'][node]
                    parents_cardinality = [self.combo_info['cardinality'][p] for p in parents]
                    if prop_type == 'dense':
                        prop_input_size = sum(parents_cardinality) + hidden_size if cat_latent else sum(parents_cardinality)
                        self.propagators[str(i)][node_name] = Dense(input_size = prop_input_size, 
                                                                    output_size = node_cardinality,
                                                                    activation = activation)
                    elif prop_type == 'mlp':
                        prop_input_size = sum(parents_cardinality) + hidden_size if cat_latent else sum(parents_cardinality)
                        self.propagators[str(i)][node_name] = MLP(input_size = prop_input_size,
                                                                  hidden_size = sum(parents_cardinality)*2,
                                                                  output_size = node_cardinality,
                                                                  n_layers = n_layers_propagation,
                                                                  activation = activation)
                    elif prop_type == 'embeddings':
                        self.propagators[str(i)][node_name] = MLP(input_size = len(parents)*concept_hidden_size,
                                                                  hidden_size = concept_hidden_size,
                                                                  output_size = node_cardinality,
                                                                  n_layers = n_layers_propagation,
                                                                  activation = activation)
                        # TODO: embeddings for the latent factors if there are undirected edges
                    elif prop_type == 'equations':
                        self.propagators[str(i)][node_name] = MLP(input_size = node_cardinality*concept_hidden_size,
                                                                  hidden_size = concept_hidden_size,
                                                                  output_size = sum(parents_cardinality)*node_cardinality,
                                                                  n_layers = n_layers_propagation,
                                                                  activation = activation)
        else:
            raise ValueError('invalid prop_type')     
        
    def label_absence_checker(self, c, intervention_index, name, i):
        # If the concept label is -1, it means the client has no access to the concept label.
        # For this reason the rand int cannot be applied.
        if name not in self.y_names:
            if not (c[:,self.c_name_index[name]].long()!=-1).sum()==0:
                # It means there are no -1 in the batch for the specific concept
                pass
            else:
                intervention_index[:,i] = torch.zeros_like(intervention_index[:,i], dtype=torch.int64)
        return intervention_index

    def forward(self, x, c=None, intervention_index=None):
        # Encode input, get the latent features
        x_encoded = self.encoder(x)

        c_embs, c_probs, c_values_emb = {}, {}, {}
        for i, name in enumerate(self.combo_info['names']):

            # update intervention index according to the concept label availability
            if intervention_index is not None:
                intervention_index = self.label_absence_checker(c, intervention_index, name, i)

            # create embeddings and probabilities for each root concept
            # this assumes the task is last in the name list
            if name in self.roots_info['names']:
                c_embs[name], c_probs[name] =  self.concept_encoders[name](x_encoded, 
                                                                           c[:,i], 
                                                                           intervention_index[:,i],
                                                                           to_return=['embs', 'probs'])
            else:
                # create latent for each non-root concept    
                # remember not to intervene on the task
                c_values_emb[name] =  self.concept_encoders[name](x_encoded, 
                                                                  c[:,i] if name in self.c_names else None, 
                                                                  intervention_index[:,i] if name in self.c_names else None,
                                                                  to_return=['values_embs'])
                
        # propagate the information through the causal graph
        for _, level in self.propagators.items():
            # update all nodes in the level
            for c_name, propagator in level.items():
                c_index = self.combo_info['names'].index(c_name)
                p_indices = get_parents(self.graph, c_index).tolist()
                p_names = [self.combo_info['names'][p] for p in p_indices]
                
                if self.prop_type =='dense' or self.prop_type == 'mlp':
                    # propagate probabilities
                    c_prob_parents = torch.cat([c_probs[p_name] for p_name in p_names], dim=1)
                    if self.cat_latent: 
                        c_prob_parents = torch.cat([c_prob_parents, x_encoded], dim=1)
                    c_probs[c_name] = torch.softmax(propagator(c_prob_parents), dim=1)
                    if c_name not in self.y_names:
                        c_probs[c_name] = maybe_intervene(c_probs[c_name], c[:,c_index], intervention_index[:,c_index])

                elif self.prop_type == 'embeddings':
                    c_cardinality = self.combo_info['cardinality'][c_index]
                    # propagate embeddings
                    c_embs_parents = torch.cat([c_embs[p_name] for p_name in p_names], dim=1)
                    c_probs[c_name] = torch.softmax(propagator(c_embs_parents), dim=1)
                    if c_name not in self.y_names:
                        c_probs[c_name] = maybe_intervene(c_probs[c_name], c[:,c_index], intervention_index[:,c_index])
                    # compute the weighted average between concepts embedding and probabilities
                    first = c_values_emb[c_name].reshape(-1, c_cardinality, self.concept_hidden_size)
                    second = c_probs[c_name].unsqueeze(-1)
                    c_embs[c_name] = (first * second).sum(dim=1)

                elif self.prop_type == 'equations':
                    c_cardinality = self.combo_info['cardinality'][c_index]
                    p_cardinality = [self.combo_info['cardinality'][p] for p in p_indices]
                    # propagate embeddings
                    c_prop_parents = torch.cat([c_probs[p_name] for p_name in p_names], dim=1).unsqueeze(-1)
                    weights = propagator(c_values_emb[c_name])
                    weights = weights.reshape(-1, c_cardinality, sum(p_cardinality))
                    c_probs[c_name] = torch.softmax(torch.matmul(weights, c_prop_parents).squeeze(-1), dim=1)
                    if c_name not in self.y_names:
                        c_probs[c_name] = maybe_intervene(c_probs[c_name], c[:,c_index], intervention_index[:,c_index])                   
                    # first = c_values_emb[c_name].reshape(-1, c_cardinality, self.concept_hidden_size)
                    # second = c_probs[c_name].unsqueeze(-1)
                    # c_embs[c_name] = (first * second).sum(dim=1)

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

    # def loss(self, y_hat, y, c_hat_dict, c):
    #     """Compute loss function.
    #     Args:
    #         y_hat (torch.Tensor): Predicted task logits
    #         y (torch.Tensor): True task labels
    #         c_hat_dict (Dict): Predicted concept logits
    #         c (torch.Tensor): True concept labels"""
    #     y = y.flatten().long()
    #     # c = c.long() # later to avoid nan to disappear
    #     loss_form = torch.nn.NLLLoss()

    #     # -- task loss
    #     y_hat = torch.log(y_hat + 1e-6)

    #     # -- concepts loss
    #     concept_loss = 0
    #     for name, c_hat in c_hat_dict.items():
    #         if not (c[:,self.c_name_index[name]].long()!=-1).sum()==0:
    #             c_hat = torch.log(c_hat + 1e-6)
    #             concept_loss += loss_form(c_hat, c[:,self.c_name_index[name]].long()) 

    #     if y[y== -1].numel() != 0:
    #         total_loss = concept_loss
    #     else:
    #          task_loss = loss_form(y_hat, y)
    #          total_loss = self.concept_loss_weight * concept_loss + (1-self.concept_loss_weight) * task_loss
    #     return total_loss
    
    def loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        c_hat_dict: Dict[str, torch.Tensor],
        c: torch.Tensor,
        reduction: str = "mean",          # "mean" | "sum" | "none"
        ignore_index: int = -100
    ):
        """
        Task-and-concept loss with an optional per-sample output.

        Args
        ----
        y_hat : (B, n_classes) logits **before** soft-max
        y     : (B,) task labels  (−1 marks “missing”)
        c_hat_dict : {name: (B, n_c_values) logits}
        c     : (B, n_concepts)   (−1 marks “missing”)
        reduction : how to reduce losses:
            - "mean" (default) : scalar identical to the old code
            - "none"           : length-B tensor with the loss for each sample
            - "sum"            : scalar sum of all samples
        """

        y = y.flatten().long()                       # (B,)

        # ----- helper that works for both reductions ---------------------------
        def nll(pred_log, tgt):
            return torch.nn.functional.nll_loss(
                pred_log, tgt, reduction=reduction, ignore_index=ignore_index
            )                                        # shape → () or (B,)

        # ----- task loss --------------------------------------------------------
        y_hat_log = torch.log_softmax(y_hat, dim=1)  # log-p
        task_loss = None
        if (y != -1).any():                          # at least one labelled sample
            task_loss = nll(y_hat_log, y)

        # ----- concept loss -----------------------------------------------------
        if reduction == "none":
            # keep running total per sample
            concept_loss = torch.zeros_like(y_hat_log[:, 0])   # (B,)
        else:
            concept_loss = 0.0

        for name, c_hat in c_hat_dict.items():
            idx   = self.c_name_index[name]
            label = c[:, idx].long()                # (B,)
            if (label != -1).sum() == 0:
                continue                            # this concept is fully missing

            c_hat_log = torch.log_softmax(c_hat, dim=1)

            closs = nll(c_hat_log, label)           # shape as chosen above

            # accumulate
            if reduction == "none":
                concept_loss = concept_loss + closs  # per-sample add
            else:
                concept_loss = concept_loss + closs  # scalar add

        # ----- final mixture ----------------------------------------------------
        # If *all* task labels are missing, return only concept loss
        if (y == -1).all():
            total_loss = concept_loss
        else:
            if reduction == "none":
                total_loss = (
                    self.concept_loss_weight * concept_loss
                    + (1.0 - self.concept_loss_weight) * task_loss
                )                                    # element-wise
            else:
                total_loss = (
                    self.concept_loss_weight * concept_loss
                    + (1.0 - self.concept_loss_weight) * task_loss
                )

        return total_loss

