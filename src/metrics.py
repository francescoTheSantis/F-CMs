import torch
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape

class ClassificationAccuracy(Metric):
    """
    Classification Accuracy is a standard metric that measures the proportion of correct predictions
    made by the model.
    """
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, 
               preds: torch.Tensor, 
               target: torch.Tensor):
        # check if both preds and target are not nan
        if torch.isnan(preds).any() or torch.isnan(target).any():
            self.correct += 0
            self.total += 1
        else:
            # Manage Monte Carlo approximation tensor
            if len(preds.shape)>2:
                preds = preds.mean(dim=-1)
            preds = preds.argmax(dim=-1)
            target = target.flatten().long()
            _check_same_shape(preds, target)
            correct = preds.eq(target).sum()
            self.correct += correct
            self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total



def residual_concept_causal_effect(cace_metric_before, cace_metric_after):
    """
    Compute the residual concept causal effect between two concepts.
    Args:
        cace_metric_before: ConceptCausalEffect metric before the do-intervention on the inner concept
        cace_metric_after: ConceptCausalEffect metric after do-intervention on the inner concept
    """
    cace_before = cace_metric_before.compute()
    cace_after = cace_metric_after.compute()
    return cace_after / cace_before


class ConceptCausalEffect(Metric):
    """
    Concept Causal Effect (CaCE) is a metric that measures the causal effect between concept pairs
    or between a concept and the task.
    NOTE: only works on binary concepts.
    """
    def __init__(self):
        super().__init__()
        self.add_state("preds_do_1", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("preds_do_0", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, 
               preds_do_1: torch.Tensor, 
               preds_do_0: torch.Tensor):
        _check_same_shape(preds_do_1, preds_do_0)
        # expected value = 1*p(output=1|do(1)) + 0*(1-p(output=1|do(1))
        self.preds_do_1 += preds_do_1[:,1].sum()
        # expected value = 1*p(output=1|do(0)) + 0*(1-p(output=1|do(0))
        self.preds_do_0 += preds_do_0[:,1].sum()
        self.total += preds_do_1.size()[0]

    def compute(self):
        return (self.preds_do_1.float() / self.total) - (self.preds_do_0.float()  / self.total)




def edge_type(graph, i, j):
    if graph[i,j]==1 and graph[j,i]==0:
        return 'i->j'
    elif graph[i,j]==0 and graph[j,i]==1:
        return 'i<-j'
    elif (graph[i,j]==-1 and graph[j,i]==-1) or (graph[i,j]==1 and graph[j,i]==1):
        return 'i-j'
    elif graph[i,j]==0 and graph[j,i]==0:
        return '/'
    else:
        raise ValueError(f'invalid edge type {i}, {j}')

# graph similairty metrics
def hamming_distance(first, second):
    """Compute the graph edit distance between two partially direceted graphs"""
    first = first.loc[[row for row in first.index if '#virtual_' not in row],
                      [col for col in first.columns if '#virtual_' not in col]]
    first = torch.Tensor(first.values)
    second = second.loc[[row for row in second.index if '#virtual_' not in row],
                        [col for col in second.columns if '#virtual_' not in col]]
    second = torch.Tensor(second.values)
    assert (first.diag() == 0).all() and (second.diag() == 0).all()
    assert first.size() == second.size()
    N = first.size(0)
    cost = 0
    count = 0
    for i in range(N):
        for j in range(i, N):
            if i==j: continue
            if edge_type(first, i, j)==edge_type(second, i, j): continue
            else:
                count += 1
                # edge was directed
                if edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='i-j': cost += 1./5.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='i-j': cost += 1./5.
                elif edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='i<-j': cost += 1./3.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='i->j': cost += 1./3.
                # edge was undirected
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='i->j': cost += 1./4. 
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='i<-j': cost += 1./4.
                # there was no edge
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i-j': cost += 1./2.
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i->j': cost += 1
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i<-j': cost += 1

                else:  
                    raise ValueError(f'invalid combination of edge types {i}, {j}')
    
    # cost = cost / (N*(N-1))/2
    return cost, count