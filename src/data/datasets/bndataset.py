from env import CACHE
import os
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj
import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

from src.data.utils import split_dataset

class BNDataset():
    def __init__(self, 
                 dag_name = 'asia',
                 task_name = 'dysp',
                 dataset_n_samples: int = 10000,
                 test_size:  float = 0.2, # proportion of the dataset to include in the test set
                 val_size: float = 0.1, # proportion of the training set to include in the validation set
                 ftune_size: float = 0., # proportion of the test set to include in the finetuning set
                 ftune_val_size: float = 0., # proportion of the finetuning set to include in the finetuning validation set
                 bias: dict = {'train': {'mode': False, 'kwargs': {}},
                               'test': {'mode': False, 'kwargs': {}}}):
        
        self.dag_name = dag_name
        if dag_name in ['asia', 'alarm', 'andes', 'sachs', 'water']:
            bn_model_dict = bn.import_DAG(self.dag_name)
        else:
            path = os.path.join(CACHE, 'bnlearn_bif_datasets', f'{dag_name}.bif')
            bn_model_dict = bn.import_DAG(path)
        self.bn_model_dict = bn_model_dict
        self.bn_model = bn_model_dict["model"]
        
        self.dataset_n_samples = dataset_n_samples
        self.test_size = test_size
        self.val_size = val_size
        self.ftune_size = ftune_size
        
        c_names = [name for name in list(self.bn_model.nodes()) if name != task_name]  # All except the last node
        y_name = [task_name]
        c_cardinalities = [int(self.bn_model.get_cardinality()[node]) for node in c_names]  # Cardinalities for all c_names
        y_cardinality = [int(self.bn_model.get_cardinality()[y_name[0]])]
        self.c_info = {'names': c_names, 
                       'cardinality': c_cardinalities}
        self.y_info = {'names': y_name,
                       'cardinality': y_cardinality}  
        self.data = {}
        self.subgraphs_concept_names = {}
        
        
    def load_ground_truth_graph(self): 
        node_labels = self.c_info['names'] + self.y_info['names']
        # reorder the adjacency matrix to match the new node labels
        adj_pandas = self.bn_model_dict['adjmat'].loc[node_labels, node_labels]
        self.adj = adj_pandas.astype(int)
        return self.adj
    
    def split(self):
        """ 
        Split the dataset into training, validation and test sets 
        """
        # self.bn_model is biased after this point
        self.data['train'] = _BNDataset(bn_model = self.bn_model,
                                        n_samples = int(self.dataset_n_samples*(1-self.val_size-self.test_size)),
                                        concept_names = self.c_info['names'],
                                        task_name = self.y_info['names'][0]
        )
        self.data['val'] = _BNDataset(bn_model = self.bn_model,
                                      n_samples = int(self.dataset_n_samples*self.val_size),
                                      concept_names = self.c_info['names'],
                                      task_name = self.y_info['names'][0],
        )
        self.data['test'] = _BNDataset(bn_model = self.bn_model,
                                       n_samples = int(self.dataset_n_samples*self.test_size),
                                       concept_names = self.c_info['names'],
                                       task_name = self.y_info['names'][0],
        )
        self.data['train'].split_type = 'train'
        self.data['val'].split_type = 'val'

class _BNDataset(torch.utils.data.Dataset):

    def __init__(self,
                    bn_model: dict,
                    n_samples: int, 
                    task_name: str,
                    concept_names: list,
                    bias_kwargs: dict = {}):
        
        super().__init__()

        self.bn_model = bn_model.copy()
        self.n_samples = n_samples
        self.bias_kwargs = bias_kwargs
        self.split_type = ""
        self.graph = []

        inference = BayesianModelSampling(self.bn_model)
        self.data = inference.forward_sample(size=self.n_samples)
        assert self.data.loc[:,concept_names].columns.tolist() == concept_names, "Concept names do not match!"
        #assert concept_names == self.c_info['names'], "Concept names do not match!"
        reordered_names = concept_names + [task_name]
        self.y = torch.Tensor(self.data.loc[:,task_name].values).float().unsqueeze(1)
        self.c = torch.Tensor(self.data.loc[:,concept_names].values).float()
        self.X = torch.Tensor(self.data.loc[:,reordered_names].values).float()
 
    def register_graph(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        c = self.c[index]
        y = self.y[index]
        return {'x':x, 'c':c, 'y':y, 'graph':self.graph}
    