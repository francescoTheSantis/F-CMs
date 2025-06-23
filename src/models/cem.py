import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from src.models.layers.base import MLP
from src.models.layers.c_encoder import ConceptBlock

class CEM(nn.Module):
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
        super(CEM, self).__init__()

        # to be stored for every model
        self.has_concepts = True
        self.is_causal = False
        self.c_name_index = c_name_index
        self.name = name

        # concepts info
        self.concept_names = c_info['names']
        self.virtual_roots = [name for name in c_info['names'] if name.startswith('#virtual_')]
        self.concept_loss_weight = concept_loss_weight

        # Encoder
        self.encoder = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers_encoder,
                           activation=activation)
        
        # Concept encoders, one for each concept
        self.concept_encoders = nn.ModuleDict()
        for name in self.concept_names:
            if name in self.virtual_roots: continue
            self.concept_encoders[name] = ConceptBlock(input_size=hidden_size,
                                                       hidden_size=concept_hidden_size,
                                                       n_layers=n_layers_concept_encoder,
                                                       activation=activation,
                                                       c_cardinality=c_info['cardinality'][c_info['names'].index(name)])
        # Decoder
        if output_size is not None:
            n_concepts = len(self.concept_names) - len(self.virtual_roots)
            self.decoder = MLP(input_size=n_concepts*concept_hidden_size,
                               hidden_size=n_concepts*concept_hidden_size//2,
                               output_size=output_size,
                               n_layers=n_layers_decoder,
                               activation=activation)
    
    
    def forward(self, x, c=None, intervention_index=None):
        """Forward pass of the model.
        Args:
            x (torch.Tensor): Input data
            c (torch.Tensor): Concept labels
            intervention_index (torch.Tensor): Intervention index
        Retrun:
            y_hat_probs (torch.Tensor): Predicted task probabilities
            c_hat_probs (Dict): Predicted concept probabilities
        """
        # Encode input, get the latent features
        x_encoded = self.encoder(x)

        # create embeddings and probabilities for each concept
        c_embeddings, c_hat_probs = {}, {}
        for name in self.concept_names:
            if name in self.virtual_roots: continue
            i = self.c_name_index[name]
            c_embeddings[name], c_hat_probs[name] = self.concept_encoders[name](x_encoded, 
                                                                                c[:,i], 
                                                                                intervention_index[:,i])
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

    def loss(self, y_hat, y, c_hat_dict, c):
        """Compute loss function.
        Args:
            y_hat (torch.Tensor): Predicted task logits
            y (torch.Tensor): True task labels
            c_hat_dict (Dict): Predicted concept logits
            c (torch.Tensor): True concept labels"""
        y = y.flatten().long()
        # c = c.long() # later to avoid nan to disappear
        loss_form = torch.nn.NLLLoss()

        # -- task loss
        y_hat = torch.log(y_hat + 1e-6)
        

        # -- concepts loss
        concept_loss = 0
        for name, c_hat in c_hat_dict.items():
            if not (c[:,self.c_name_index[name]].long()!=-1).sum()==0:
                c_hat = torch.log(c_hat + 1e-6)
                concept_loss += loss_form(c_hat, c[:,self.c_name_index[name]].long())    

        if y[y== -1].numel() != 0:
            total_loss = concept_loss
        else:
            task_loss = loss_form(y_hat, y)
            total_loss = self.concept_loss_weight * concept_loss + (1-self.concept_loss_weight) * task_loss
        return total_loss