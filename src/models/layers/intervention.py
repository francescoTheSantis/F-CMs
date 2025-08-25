import torch
from torch.nn.functional import one_hot

def get_test_intervention_index(c_shape, c_index, values=None):   
    """
    Get intervention index for test time intervention.
    Args:
        c_shape: shape of the concept tensor
        c_index: (int or list[int]) indices of concepts to intervene on
        values: (int or Tensor or str) or list of (int or Tensor or str), if None, it only return the intervention index
                                                                          if int, set all intervened concepts to this value
                                                                          if Tensor, set intervened concepts to this tensor
                                                                          if 'random', set intervened concepts to random values
    """       
    intervention_index = torch.zeros(c_shape)
    c_values = torch.full(c_shape, float('nan'))
    if isinstance(c_index, int):
        c_index = [c_index]
    if values is not None and not isinstance(values, list):
        values = [values]

    if c_index:
        for i in range(len(c_index)):
            # indices
            index_i = c_index[i]
            intervention_index[:,index_i] = 1
            # values
            if values is not None:
                values_i = values[i]
                if isinstance(values_i, int):
                    c_values[:, index_i] = torch.ones(c_shape[0]) * values_i
                elif isinstance(values, torch.Tensor):
                    c_values[:, index_i] = values_i
                elif values == 'random':
                    raise NotImplementedError
                
    if values is not None:
        return intervention_index.to("cuda" if torch.cuda.is_available() else "cpu"), c_values.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return intervention_index.to("cuda" if torch.cuda.is_available() else "cpu")

def maybe_intervene(c_pred_probs, c, intervention_index):
    # check if intervention index is not all zeros (non interventions) and ground truth is not all nans (virutal roots)
    if not torch.all(intervention_index == 0) and not torch.all(torch.isnan(c)):
        if torch.any(torch.isnan(c)): raise ValueError("Intervention with nan ground truth is not allowed")
        concept_cardinality = c_pred_probs.shape[1]
        index = intervention_index.bool().unsqueeze(1).repeat(1,concept_cardinality)
        index = index.to(c_pred_probs.device)
        # TODO: take a look here
        # print("c min:", c.min().item(), "c max:", c.max().item())
        # print("Concept cardinality:", concept_cardinality)
        c = c.clamp(min=0)
        # (you have negative values in c)
        c_one_hot = one_hot(c.long(), concept_cardinality)   
        c_pred_probs = torch.where(index, c_one_hot, c_pred_probs)
    return c_pred_probs

