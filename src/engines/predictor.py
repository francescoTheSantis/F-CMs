from typing import Any, Optional, Mapping, Type
import pickle
import itertools

import torch
from copy import deepcopy
from torch import nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.collections import _remove_prefix
import pytorch_lightning as pl
from src.data.generate_split import get_subgraph_dict
from env import CACHE

from src.models.layers.intervention import get_test_intervention_index

class Predictor(pl.LightningModule):    
    def __init__(self,
                model: Optional[nn.Module] = None,
                metrics: Optional[Mapping[str, Metric]] = None,
                optim_class: Optional[Type] = None,
                optim_kwargs: Optional[Mapping] = None,
                scheduler_class: Optional[Type] = None,
                scheduler_kwargs: Optional[Mapping] = None,
                intervention_prob: Optional[float] = 0.2,
                #c_names: Optional[list] = None,
                test_interv_policy: Optional[str] = None,
                test_interv_noise: Optional[float] = 0.,
                c_name_index: Optional[Mapping[str, int]] = None,
                c_names_id: Optional[Mapping[str, str]] = None,
                c_names_ood: Optional[Mapping[str, str]] = None,
                c_names_all: Optional[list] = None,
                annotation_assumption: Optional[str] = None,
                learning_modality: Optional[str] = 'localized',
                cid: Optional[int] = 1,
                centralized_topological_order: Optional[list] = None,
                centralized_c_dict: Optional[dict] = None,
                ):
        super(Predictor, self).__init__()         
        self.model = model
        self.save_hyperparameters(ignore=["model"], logger=False)

        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or dict()
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or dict()
        self.annotation_assumption = annotation_assumption

        # for regularization
        self.intervention_prob = intervention_prob
        # store the intervention policy
        self.test_interv_policy = test_interv_policy
        self.test_interv_noise = test_interv_noise  

        #self.c_names = c_names
        self.n_concepts = len(c_names_all)
        self.c_name_index = c_name_index

        # create an index to name mapping for concepts
        #self.c_index_to_name_map = {i: name for i, name in zip(c_name_index.values(), c_name_index.keys())}
        #self.id_c_idxes = list({k:v for k,v in self.c_index_to_name_map.items() if v in self.c_names_all}.keys())

        self.c_names_id = c_names_id 
        self.c_names_ood = c_names_ood 
        self.c_names_all = c_names_all
        self.clients = range(1, len(self.c_names_ood)+2)  # +1 for the case where there are no OOD concepts

        self.learning_modality = learning_modality
        self.centralized_topological_order = centralized_topological_order
        self.centralized_c_dict = centralized_c_dict
        if metrics is None:
            metrics = dict()

        self._set_metrics(metrics)

        self.cid = cid
        self.level_intervention_id_annotations = {}
        self.level_intervention_ood_annotations = {}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    @staticmethod
    def _check_metric(metric):
        metric = metric.clone()
        metric.reset()
        return metric
    
    def set_id_ood_interventions(self):
        id_concepts = self.c_index_id
        id_interventions = []
        self.ood_interventions = []
        for level in self.test_interv_policy:
            level_id_list = []
            level_ood_list = []
            for concept in level:
                if concept in id_concepts:
                    level_id_list.append(concept)
                else:
                    level_ood_list.append(concept)
            if level_ood_list != []:
                # the OOD interventions is the list of concepts that are not in the ID concepts
                self.ood_interventions.append(level_ood_list)
            if level_id_list != []:
                id_interventions.append(level_id_list)
        # the intervention policy is updated to the ID concepts (defined by the client available concepts)
        self.test_interv_policy = id_interventions

    def _set_metrics(self, metrics):
        # --- accuracy metrics ---
        y_acc_metrics = {'y_accuracy': metrics.get('classification_acc')}
        # we want to compute the accuracy on both ID and OOD concepts (if any)
        c_acc_metrics = {k: metrics.get('classification_acc') for k in self.c_names_all}

        # task accuracy metrics
        self.train_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="train/y/")
        self.val_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="val/y/")
        self.test_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="test/y/")
        
        # --- concept accuracy metrics ---
        self.train_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="train/c/")
        self.val_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="val/c/")
        self.test_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="test/c/")      
            
        if self.model.has_concepts:
            # --- ground truth intervention metrics ---
            c_acc_metrics['_baseline'] = metrics.get('classification_acc')
            c_acc_levels_metrics = {f'level {n}': metrics.get('classification_acc')
                                    for n in range(0, len(self.test_interv_policy)+1)}
            
            # task accuracy after invervention on each individual concept 
            # (one metric for each concept)
            self.test_intervention_single_y = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
                prefix="test_intervention/single/y/")
            # task accuracy after intervention of each graph level
            self.test_intervention_level_y = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in c_acc_levels_metrics.items()},
                prefix="test_intervention/level/y/")
            
            # task accuracy after intervention on id/ood concepts of each graph level
            if self.learning_modality not in ["centralized", "localized"]:
                self.test_intervention_id_level_y = {}
                self.test_intervention_ood_level_y = {}
                for client_id in range(1,len(self.c_names_ood)+1):
                    self.test_intervention_id_level_y[f'client {client_id}'] = MetricCollection(
                        metrics={k: self._check_metric(m) for k, m in c_acc_levels_metrics.items()},
                        prefix=f"test_intervention/id_level/y/client_{client_id}/")
                    self.test_intervention_ood_level_y[f'client {client_id}'] = MetricCollection(
                        metrics={k: self._check_metric(m) for k, m in c_acc_levels_metrics.items()},
                        prefix=f"test_intervention/ood_level/y/client_{client_id}/")
                
            # individual child concept accuracy after
            # intervention on ancestors in the graph
            childs_per_level = {}
            for l in range(0, len(self.test_interv_policy)+1):
                childs = list(itertools.chain(*self.test_interv_policy[l:]))
                for child in childs:
                    child_name = self.c_names_all[child]
                    childs_per_level[f'level {l}/child {child_name}'] = metrics.get('classification_acc')
            self.test_intervention_level_c = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in childs_per_level.items()},
                prefix="test_intervention/level/c/")
            
            # individual child ood concept accuracy after
            # intervention on id ancestors in the graph for each client
            if self.learning_modality not in ["centralized", "localized"]:
                self.test_intervention_id_level_c_ood ={}
                childs_per_level_ood = {}
                for client_id in range(1,len(self.c_names_ood)+1):
                    ood_concepts = self.c_names_ood[client_id]
                    # create a dictionary childs
                    childs_per_level_ood[client_id] = {}
                    for l in range(0, len(self.test_interv_policy)+1):
                        childs = list(itertools.chain(*self.test_interv_policy[l:]))
                        # filter out IID concepts for each client
                        # extract indices from ood_concepts
                        ood_indices = [i for i, c in enumerate(self.c_names_all) if c in ood_concepts]
                        childs =  [child for child in childs if child in ood_indices]
                        if len(childs)!=0:
                            for child in childs:
                                child_name = self.c_names_all[child] 
                                childs_per_level_ood[client_id][f'level {l}/child {child_name}'] = metrics.get('classification_acc')
                    self.test_intervention_id_level_c_ood[f'client {client_id}'] = MetricCollection(
                        metrics={k: self._check_metric(m) for k, m in childs_per_level_ood[client_id].items()},
                        prefix=f"test_intervention/id_level/c_ood/client_{client_id}/")
                
                
                # individual child iid concept accuracy after
                # intervention on ood ancestors in the graph for each client
                self.test_intervention_ood_level_c_id ={}
                childs_per_level_id = {}
                for client_id in range(1,len(self.c_names_ood)+1):
                    id_concepts = self.c_names_id[client_id]
                    # create a dictionary childs
                    childs_per_level_id[client_id] = {}
                    for l in range(0, len(self.test_interv_policy)+1):
                        childs = list(itertools.chain(*self.test_interv_policy[l:]))
                        # filter out IID concepts for each client
                        id_indices = [i for i, c in enumerate(self.c_names_all) if c in id_concepts]
                        childs =  [child for child in childs if child in id_indices]
                        if len(childs)!=0:
                            for child in childs:
                                child_name = self.c_names_all[child] 
                                childs_per_level_id[client_id][f'level {l}/child {child_name}'] = metrics.get('classification_acc')
                    self.test_intervention_ood_level_c_id[f'client {client_id}'] = MetricCollection(
                        metrics={k: self._check_metric(m) for k, m in childs_per_level_id[client_id].items()},
                        prefix=f"test_intervention/ood_level/c_id/client_{client_id}/")

            
            # cumulative interventions: task accuracy
            cumulative_y_metrics = {}
            cumulative_count = 0
            for c_name in self.centralized_topological_order:
                if c_name in self.model.virtual_roots: continue
                cumulative_count += 1
                cumulative_y_metrics[f'{cumulative_count}_{c_name}'] = metrics.get('classification_acc')
            
            self.test_intervention_cumulative_y = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in cumulative_y_metrics.items()},
                prefix="test_intervention/cumulative/y/")
            
            # cumulative interventions: concept accuracy
            cumulative_c_metrics = {}
            cumulative_count = 0
            for c_name_i in self.centralized_topological_order:
                if c_name_i in self.model.virtual_roots: continue
                cumulative_count += 1
                for c_name_j in self.centralized_topological_order:
                    if c_name_j in self.model.virtual_roots: continue
                    cumulative_c_metrics[f'{cumulative_count}_{c_name_i}/{c_name_j}'] = metrics.get('classification_acc')
            
            self.test_intervention_cumulative_c = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in cumulative_c_metrics.items()},
                prefix="test_intervention/cumulative/c/")

            # --- fairness metrics ---
            self.cace = MetricCollection(
                metrics = {'before': self._check_metric(metrics.get('cace')),
                           'after': self._check_metric(metrics.get('cace'))},   
                prefix="test_intervention/cace/")

    def log_metrics(self, metrics, **kwargs):
        """"""
        self.log_dict(
            metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True, **kwargs
        )

    def log_loss(self, name, loss, **kwargs):
        """"""
        self.log(
            name + "_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            **kwargs,
        )

    def _unpack_batch(self, batch):
        """
        Unpack a batch into data and preprocessing dictionaries.
        """
        return batch['x'], batch['c'], batch['y']
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # add batch_size to batch
        if isinstance(batch, dict):
            batch['batch_size'] = batch['x'].shape[0]
        else:
            raise NotImplementedError("Only dict batches are supported")
        return batch

    def get_intervention_index(self, c_shape, step):
        """
        Get intervention index for training time intervention.
        Args:
            c_shape: shape of the concept tensor
            step: (str) 'train' or 'val'
        """
        # for regularization only
        if step=='train':
            intervention_index = torch.bernoulli(torch.ones(c_shape) * self.intervention_prob)
        else:
            intervention_index = torch.zeros(c_shape)
        return intervention_index.to("cuda" if torch.cuda.is_available() else "cpu")
    
    #def _remove_node_id_ood(self, nodes):
    #    for node in nodes:
    #        if node not in self.id_c_idxes:
    #            # the node is an OOD concept, we remove it from the list of nodes
    #            nodes.remove(node)
    #    return nodes

    def test_intervention(self, batch):
        if self.model.has_concepts:
            x, c, y = self._unpack_batch(batch)
            # maybe add noise
            if self.test_interv_noise > 0:
                x = x + torch.randn_like(x) * self.test_interv_noise

            # baseline task accuracy
            # do not intervene
            intervention_index = get_test_intervention_index(c.shape, [])
            inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
            # forward pass with intervention at test time
            y_output, c_output = self.forward(**inputs)
            y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
            # update metric after intervention:
            # after interveening on concept c_name_i, how well can we predict y
            self.test_intervention_single_y['_baseline'].update(y_hat, y)           

            # interventions on individual concepts
            for i, c_name_i in [(i, name) for name, i in self.c_name_index.items() if name in self.c_names_all]:
                if c_name_i in self.model.virtual_roots: continue
                first_key = next(iter(self.c_names_id))
                if self.learning_modality == 'localized' and c_name_i not in self.c_names_id[first_key]:
                    # The concept is not in the ID concepts, therefore interveaning on this concept
                    # does not have any effect on the task. 
                    # By the way, we just avoid to compute the intervention index but we still perform the forward pass
                    print(f"Skipping intervention on {c_name_i} as it is not in the ID concepts")
                    intervention_index = torch.zeros(c.shape, dtype=c.dtype, device=c.device)
                else:
                    # intervene on concept c_name_i
                    intervention_index = get_test_intervention_index(c.shape, i)
                inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
                # forward pass with intervention at test time
                y_output, c_output = self.forward(**inputs)
                y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                self.test_intervention_single_y[c_name_i].update(y_hat, y)
                
                #self.test_intervention_single_y[c_name_i].to(c.device)
                # update metric after intervention:
                # after interveening on concept c_name_i, how well can we predict y
                #self.test_intervention_single_y[c_name_i].reset()
                #self.test_intervention_single_y[c_name_i].update(y_hat, y)
                #value = self.test_intervention_single_y[c_name_i].compute()
                #print(value)
                #print(id(self.test_intervention_single_y[c_name_i]))
                

            # single interventions cumulative
            cumulative_indices = []
            number_of_interventions = 0
            for c_name in self.centralized_topological_order:
                number_of_interventions += 1
                if c_name in self.model.virtual_roots: continue
                # intervene on concept c_name_i in a cumulative way
                #if c.shape[1]< len(self.centralized_topological_order):
                if c_name in self.c_name_index:
                    cumulative_indices.append(self.c_name_index[c_name])
                intervention_index = torch.zeros(c.shape, dtype=c.dtype, device=c.device)
                for idx in cumulative_indices:
                    intervention_index += get_test_intervention_index(c.shape, idx)
                inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
                # forward pass with intervention at test time
                y_output, c_output = self.forward(**inputs)
                y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                self.test_intervention_cumulative_y[str(number_of_interventions) + "_" + c_name].update(y_hat, y)
                for c_name_j in self.centralized_topological_order:
                    if c_name_j in self.c_name_index:
                        index_in_c = self.c_name_index[c_name_j]
                    else:
                        continue
                    if c_name_j in self.model.virtual_roots: continue
                    if c_name_j not in c_hat.keys():
                        continue
                    self.test_intervention_cumulative_c[str(number_of_interventions) + "_" + c_name +"/" + c_name_j].update(c_hat[c_name_j], c[:,index_in_c])
                
            # level intervention
            # NOTICE: if self.learning_modality== "localized", self.interv_policy has been updated to the subgraph
            device = c.device
            if self.learning_modality == "localized":
                self.clients = [first_key]
            elif self.learning_modality == "centralized":
                self.clients = [1]
            else:
                self.clients = range(1, len(self.c_names_ood)+2)  # +1 for the case where there are no OOD concepts
            possible_clients = deepcopy(self.clients)  # copy the list of clients
            # If there are multiple test sets, select only the clients with the correct one
            if (c == -1).all(dim=0).any():
                # extract columns of c where there are all -1
                mask = (c == -1).all(dim=0)
                # take indices where mask == True
                mask_true = torch.nonzero(mask, as_tuple=True)[0].tolist()
                # take indices in c_names_all corresponding to c_names_ood[client_id] for each client
                client_ood_indices = {}
                ood_keys = list(self.c_names_ood.keys())
                for client in possible_clients[:-1]:
                    key = ood_keys[client-1]
                    client_ood_indices[client] = [self.c_names_all.index(c_name) for c_name in self.c_names_ood[key]]

                client_ood_indices[len(self.c_names_ood)+1] = mask_true
                possible_clients = [client_id for client_id in possible_clients if (client_ood_indices[client_id] == mask_true)]
                self.clients = possible_clients

            for client_id in self.clients:
                self.level_intervention_id_annotations[client_id] = {}
                self.level_intervention_ood_annotations[client_id] = {}
                if client_id!= len(self.c_names_ood)+1:
                    if self.learning_modality not in ["centralized", "localized"]:
                        id_concepts = self.c_names_id[client_id]
                        ood_concepts = self.c_names_ood[client_id]
                        id_indices = [i for i, c in enumerate(self.c_names_all) if c in id_concepts]
                        ood_indices = [i for i, c in enumerate(self.c_names_all) if c in ood_concepts]
                # client_id == len(self.c_names_ood)+1 does not exist. It is created to calculate level intervention without difference id/ood concepts
                for l in range(0, len(self.test_interv_policy)+1):
                    # get the nodes to intervene on
                    nodes = list(itertools.chain(*self.test_interv_policy[:l]))
                    if client_id == len(self.c_names_ood)+1 or self.learning_modality in ["centralized", "localized"]:
                        # intervene on all the concepts of the level
                        intervention_index = get_test_intervention_index(c.shape, nodes)
                        inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
                        y_output, c_output = self.forward(**inputs)
                        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                        # update metric after intervention:
                        # after interveening on a level of the graph, how well can we predict y
                        self.test_intervention_level_y[f'level {l}'].update(y_hat, y)
                    else:
                        if self.learning_modality not in ["centralized", "localized"]:
                            
                            # intervene on id and ood concepts of the level
                            nodes_id = [node for node in nodes if node in id_indices]
                            nodes_ood = [node for node in nodes if node in ood_indices]
                            if len(nodes_id)==0:
                                self.level_intervention_id_annotations[client_id][l] ="empty"
                            if len(nodes_ood)==0:
                                self.level_intervention_ood_annotations[client_id][l] ="empty"
                            intervention_index_id = get_test_intervention_index(c.shape, nodes_id)
                            intervention_index_ood = get_test_intervention_index(c.shape, nodes_ood)
                            inputs_id = {'x':x, 'c':c, 'intervention_index':intervention_index_id}
                            inputs_ood = {'x':x, 'c':c, 'intervention_index':intervention_index_ood}
                            y_output_id, c_output_id = self.forward(**inputs_id)
                            y_output_ood, c_output_ood = self.forward(**inputs_ood)
                            y_hat_id, c_hat_id = self.model.filter_output_for_metric(y_output_id, c_output_id)
                            y_hat_ood, c_hat_ood = self.model.filter_output_for_metric(y_output_ood, c_output_ood)
                            self.test_intervention_id_level_y[f'client {client_id}'][f'level {l}'].to(device).update(y_hat_id, y)
                            self.test_intervention_ood_level_y[f'client {client_id}'][f'level {l}'].to(device).update(y_hat_ood, y)
                    # update metric after intervention:
                    # after interveening on a level of the graph, how well can we predict each child concept
                    childs = list(itertools.chain(*self.test_interv_policy[l:]))
                    for child_index in childs:
                        if self.learning_modality not in ["centralized", "localized"]:
                            if self.c_names_all[child_index] not in c_hat_id.keys():
                                continue
                        if client_id == len(self.c_names_ood)+1 or self.learning_modality in ["centralized", "localized"]:
                            c_name = self.c_names_all[child_index]
                            self.test_intervention_level_c[f'level {l}/child {c_name}'].update(c_hat[c_name], c[:,child_index])
                        else:
                            if self.learning_modality not in ["centralized", "localized"]:
                                if child_index in id_indices:
                                    c_name = self.c_names_all[child_index]
                                    self.test_intervention_ood_level_c_id[f'client {client_id}'][f'level {l}/child {c_name}'].to(device).update(c_hat_ood[c_name], c[:,child_index])
                                elif child_index in ood_indices:
                                    c_name = self.c_names_all[child_index]
                                    self.test_intervention_id_level_c_ood[f'client {client_id}'][f'level {l}/child {c_name}'].to(device).update(c_hat_id[c_name], c[:,child_index])

                        #if c_name in self.c_names_id[1]:
                        #    self.test_intervention_level_c[f'level {l}/child {c_name}'].update(c_hat[c_name], c[:,child_index])
                        #else:
                        #    # There is no improvement on the concept accuracy for OOD concepts
                        #    self.test_intervention_level_c[f'level {l}/child {c_name}'].update(torch.Tensor([float('nan')]), 
                        #                                                                       torch.Tensor([float('nan')]))
                    
                    # level intervention for all clients on iid and ood concepts separately

    # NOT APPLICABLE NOW
    def test_intervention_fairness(self, batch):
        if self.model.has_concepts:
            x, c, y = self._unpack_batch(batch)

            # get a concept pair i,j (node j has to be a bottleneck for node i to the task)
            i = self.c_names.index('Attractive')
            j = self.c_names.index('Qualified')

            # compute the cace before the do-intervention on concept j
            # different do-interventions on concept i, effect on the task
            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=1)
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_before_do_1, _ = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=0)
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_before_do_0, _ = self.model.filter_output_for_metric(y_output, c_output)
            self.cace['before'].update(y_hat_before_do_1, y_hat_before_do_0)

            # on causal models like causal cem, because of the way they are implemented, is not necessary to strip eedges
            # after interventions, as interventions fix the values of the concept and previous calculations are useless
            # at most there is a little overhead in the forward pass
            # if self.model.is_causal:
            #     self.model.remove_edges(j)

            # compute the cace after the do-intervention on concept j
            # different do-interventions on concept i, effect on the task
            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,1])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_after_do_1, _ = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,0])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_after_do_0, _ = self.model.filter_output_for_metric(y_output, c_output)
            self.cace['after'].update(y_hat_after_do_1, y_hat_after_do_0)

            self.log_metrics(self.cace, batch_size=batch['batch_size'])


    def update_and_log_metrics(self, step, y_hat, y, c_hat, c, batch, calculate_c_metrics= True, calculate_y_metrics= True):
        if calculate_y_metrics:
            # update and log task metrics
            y_collection = getattr(self, f"{step}_y_metrics")
            y_collection.update(y_hat, y)
            self.log_metrics(y_collection, batch_size=batch['batch_size'])
        if calculate_c_metrics and self.model.has_concepts:
            # update and log concept metrics
            c_collection = getattr(self, f"{step}_c_metrics")
            # log metrics for all predicted concepts 
            # (not necessarily all concepts, some models predicts only a subset of concepts)
            for k, v in c_hat.items():
                c_collection[k].update(v, c[:,self.c_name_index[k]])
            self.log_metrics(c_collection, batch_size=batch['batch_size'])

    def shared_step(self, batch, step):
        x, c, y = self._unpack_batch(batch)
        intervention_index = self.get_intervention_index(c.shape, step=step)
        inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
        # model forward
        y_output, c_output = self.forward(**inputs)
        # Compute loss
        y_hat_loss, c_hat_loss = self.model.filter_output_for_loss(y_output, c_output)
        loss = self.model.loss(y_hat_loss, y, c_hat_loss, c)
        return loss, y_output, c_output, y, c

    def training_step(self, batch, batch_idx):
        loss, y_output, c_output, y, c = self.shared_step(batch, step='train')
        if torch.isnan(loss).any():
            print(f'at epoc: {self.current_epoch}, batch: {batch_idx}')
            print('Loss has nan')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)

        if y[y== -1].numel() != 0:
            self.update_and_log_metrics("train", y_hat, y, c_hat, c, batch, calculate_c_metrics = True, calculate_y_metrics = False)
        else:
            self.update_and_log_metrics("train", y_hat, y, c_hat, c, batch)          
        self.log_loss("train", loss, batch_size=batch['batch_size'])
        
        # check parameter freezing
        #print("c", c[0])
        #for name, param in self.model.named_parameters():
        #    print(f"{name}: requires_grad = {param.requires_grad}") 
        
        return loss
    
    def on_train_epoch_end(self):
        # Set the current epoch for SCBM and update the list of concept probs for computing the concept percentiles
        if type(self.model).__name__ == 'SCBM':
            self.model.training_epoch = self.current_epoch
            self.model.concept_pred = torch.cat(self.model.concept_pred_tmp, dim=0) 
            self.model.concept_pred_tmp = []        

    def validation_step(self, batch, batch_idx):
        val_loss, y_output, c_output, y, c = self.shared_step(batch, step='val')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        if y[y== -1].numel() != 0:
            self.update_and_log_metrics("val", y_hat, y, c_hat, c, batch, calculate_c_metrics = True, calculate_y_metrics = False)
        else:
            self.update_and_log_metrics("val", y_hat, y, c_hat, c, batch) 
        self.log_loss("val", val_loss, batch_size=batch['batch_size'])
        return val_loss
    
    def test_step(self, batch, batch_idx):
        test_loss, y_output, c_output, y, c = self.shared_step(batch, step='test')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        #print("c_hat_bronc", c_hat["bronc"][0:5])
        if y[y== -1].numel() != 0:
            self.update_and_log_metrics("test", y_hat, y, c_hat, c, batch, calculate_c_metrics = True, calculate_y_metrics = False)
        else:
            self.update_and_log_metrics("test", y_hat, y, c_hat, c, batch) 
        self.log_loss("test", test_loss, batch_size=batch['batch_size'])
        # test-time interventions
        self.test_intervention(batch)
            
        # DA RIVEDERE
        #if 'Qualified' in self.c_names:
        #    self.test_intervention_fairness(batch)
        return test_loss

    def on_test_epoch_end(self): #***reset metrics???
        # baseline task accuracy
        y_baseline = self.test_y_metrics['y_accuracy'].compute().item()
        print(f"Baseline task accuracy: {y_baseline}")
        pickle.dump({'_baseline':y_baseline}, open(f'results/y_accuracy.pkl', 'wb'))

        # baseline concept accuracy
        c_baseline = {}
        for k, metric in self.test_c_metrics.items():
            k = _remove_prefix(k, self.test_c_metrics.prefix)
            c_baseline[k] = metric.compute().item()
            print(f"Baseline concept accuracy for {k}: {c_baseline[k]}")
        pickle.dump(c_baseline, open(f'results/c_accuracy.pkl', 'wb'))

        if self.model.has_concepts:
            # task accuracy after invervention on each individual concept
            y_int = {}
            for k, metric in self.test_intervention_single_y.items():
                c_name = _remove_prefix(k, self.test_intervention_single_y.prefix)
                y_int[c_name] = metric.compute().item()
                first_key = next(iter(self.c_names_id)) # if learning_modality is localized we have only one client
                if self.learning_modality == "localized" and c_name not in self.c_names_id[first_key]:
                    continue
                print(f"Task accuracy after intervention on {c_name}: {y_int[c_name]}")
            pickle.dump(y_int, open(f'results/single_c_interventions_on_y.pkl', 'wb'))

            # if local_federated task accuracy after intervention on each individual ood concept for each client
            if self.learning_modality not in ["centralized", "localized"]:
                if len(self.c_names_ood) != 0:          
                    y_int_OOD = dict()
                    for client_id in self.clients[:-1]:  # exclude the last client which is not a real client
                        y_int_OOD[client_id] = dict()
                        for c_name in self.c_names_ood[client_id]:                
                            y_int_OOD[client_id][c_name] = y_int[c_name]
                            print(f"Task accuracy for client {client_id} after intervention on ood {c_name}: {y_int_OOD[client_id][c_name]}")
                    pickle.dump(y_int_OOD, open(f'results/single_OODc_interventions_on_y.pkl', 'wb'))

                    # task accuracy after intervention on each individual id concept for each client
                    y_int_ID = dict()
                    for client_id in self.clients[:-1]:  # exclude the last client which is not a real client
                        y_int_ID[client_id] = dict()
                        for c_name in self.c_names_id[client_id]:
                            y_int_ID[client_id][c_name] = y_int[c_name]
                            print(f"Task accuracy for client {client_id} after intervention on id {c_name}: {y_int_ID[client_id][c_name]}")
                    pickle.dump(y_int_ID, open(f'results/single_IDc_interventions_on_y.pkl', 'wb'))     

            # task accuracy after intervention of each graph level
            y_int = {}
            for k, metric in self.test_intervention_level_y.items():
                level = _remove_prefix(k, self.test_intervention_level_y.prefix)
                y_int[level] = metric.compute().item()
                print(f"Task accuracy after intervention on {level}: {y_int[level]}")
            pickle.dump(y_int, open(f'results/level_interventions_on_y.pkl', 'wb'))

            if self.learning_modality not in ["centralized", "localized"]:
                # task accuracy after intervention on each individual id concept of each graph level for each client 
                y_int_ID_level = {}
                for client_id in self.clients[:-1]:  # exclude the last client which is not a real client
                    y_int_ID_level[client_id] = {}
                    for k, metric in self.test_intervention_id_level_y[f'client {client_id}'].items():
                        level = _remove_prefix(k, self.test_intervention_id_level_y[f'client {client_id}'].prefix)
                        y_int_ID_level[client_id][f'id_{level}'] = metric.compute().item()
                        num_level = int(level.replace("level ", ""))
                        print(f"Task accuracy for client {client_id} after intervention on id_{level}{' (empty)' if num_level in self.level_intervention_id_annotations[client_id].keys() else ''}: {y_int_ID_level[client_id][f'id_{level}']}")
                pickle.dump(y_int_ID_level, open(f'results/level_ID_interventions_on_y.pkl', 'wb'))

                # task accuracy after intervention on each individual ood concept of each graph level for each client
                y_int_OOD_level = {}
                for client_id in self.clients[:-1]:  # exclude the last client which is not a real client
                    y_int_OOD_level[client_id] = {}
                    for k, metric in self.test_intervention_ood_level_y[f'client {client_id}'].items():
                        level = _remove_prefix(k, self.test_intervention_ood_level_y[f'client {client_id}'].prefix)
                        y_int_OOD_level[client_id][f'ood_{level}'] = metric.compute().item()
                        num_level = int(level.replace("level ", ""))
                        print(f"Task accuracy for client {client_id} after intervention on ood {level}{' (empty)' if num_level in self.level_intervention_ood_annotations[client_id].keys() else ''}: {y_int_OOD_level[client_id][f'ood_{level}']}")
                pickle.dump(y_int_OOD_level, open(f'results/level_OOD_interventions_on_y.pkl', 'wb'))

            # individual child concept accuracy after
            # intervention on ancestors in the graph
            c_int = {}
            for k, metric in self.test_intervention_level_c.items():
                level_child = _remove_prefix(k, self.test_intervention_level_c.prefix)
                c_int[level_child] = metric.compute().item()
                print(f"Concept accuracy after intervention on {level_child}: {c_int[level_child]}")
            pickle.dump(c_int, open(f'results/level_interventions_on_c.pkl', 'wb'))

            if self.learning_modality not in ["centralized", "localized"]:
                # individual ood child concept accuracy after
                # intervention on id ancestors in the graph, for each client
                c_int_id_level_c_ood = {}
                for client_id in self.clients[:-1]:  # exclude the last client which is not a real client
                    c_int_id_level_c_ood[client_id] = {}
                    for k, metric in self.test_intervention_id_level_c_ood[f'client {client_id}'].items():
                        level_child = _remove_prefix(k, self.test_intervention_id_level_c_ood[f'client {client_id}'].prefix)
                        c_int_id_level_c_ood[client_id][f'id_{level_child}'] = metric.compute().item()
                        # extract "level" from level_child. It's the string before "/" in level_child
                        level = level_child.split("/")[0]
                        level = level.replace("level ", "")
                        print(f"Concept accuracy for client {client_id} after intervention on id {level_child}{' (empty)' if int(level) in self.level_intervention_id_annotations[client_id].keys() else ''}: {c_int_id_level_c_ood[client_id][f'id_{level_child}']}")
                pickle.dump(c_int_id_level_c_ood, open(f'results/level_ID_interventions_on_c_OOD.pkl', 'wb'))

                # individual id child concept accuracy after
                # intervention on ood ancestors in the graph, for each client
                c_int_ood_level_c_id ={}

                for client_id in self.clients[:-1]:  # exclude the last client which is not a real client
                    c_int_ood_level_c_id[client_id] = {}
                    for k, metric in self.test_intervention_ood_level_c_id[f'client {client_id}'].items():
                        level_child = _remove_prefix(k, self.test_intervention_ood_level_c_id[f'client {client_id}'].prefix)
                        c_int_ood_level_c_id[client_id][f'ood_{level_child}'] = metric.compute().item()
                        level = level_child.split("/")[0]
                        level = level.replace("level ", "")
                        print(f"Concept accuracy for client {client_id} after intervention on ood {level_child}{' (empty)' if int(level) in self.level_intervention_ood_annotations[client_id].keys() else ''}: {c_int_ood_level_c_id[client_id][f'ood_{level_child}']}")
                pickle.dump(c_int_ood_level_c_id, open(f'results/level_OOD_interventions_on_c_ID.pkl', 'wb'))

            # cumulative interventions on task accuracy
            y_int_cumulative = {}
            for k, metric in self.test_intervention_cumulative_y.items():
                key = _remove_prefix(k, self.test_intervention_cumulative_y.prefix)
                y_int_cumulative[key] = metric.compute().item()
                print(f"Task accuracy after cumulative intervention {key}: {y_int_cumulative[key]}")
            pickle.dump(y_int_cumulative, open(f'results/cumulative_interventions_on_y.pkl', 'wb'))

            # cumulative interventions on concept accuracy
            c_int_cumulative = {}
            for k, metric in self.test_intervention_cumulative_c.items():
                key = _remove_prefix(k, self.test_intervention_cumulative_c.prefix)
                c_int_cumulative[key] = metric.compute().item()
                print(f"Concept accuracy after cumulative intervention {key}: {c_int_cumulative[key]}")
            pickle.dump(c_int_cumulative, open(f'results/cumulative_interventions_on_c.pkl', 'wb'))

            # save graph and concepts
            # DA RIVEDERE
            # Load existing graph.pkl if it exists, otherwise create new dict
            try:
                with open("graph.pkl", 'rb') as f:
                    graph_data = pickle.load(f)
            except FileNotFoundError:
                graph_data = {}
            
            # Update only these keys without overwriting other data
            graph_data.update({
                'concepts': self.c_names_all,
                'policy': self.test_interv_policy,
                'centralized_topological_order': self.centralized_topological_order,
                'learning_modality': self.learning_modality
            })
            
            # Save updated data
            pickle.dump(graph_data, open("graph.pkl", 'wb'))

    def configure_optimizers(self):
        """"""
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg["optimizer"] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop("monitor", None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg["lr_scheduler"] = scheduler
            if metric is not None:
                cfg["monitor"] = metric
        return cfg


#    def prepare_for_test(self, cfg):
#        """
#       Prepare the model for test time. This is called before the test step.
#        """
#        self.model.eval()
#
#
#        # Get the subgraph for the client
#        #concept_ids, concept_names = get_subgraph_dict(cfg)
#        path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
#        #concept_id_list = concept_ids['subgraph_'+identify_subgraph(path, cfg.client_id)]
#        #concept_name_list = concept_names['subgraph_'+identify_subgraph(path, cfg.client_id)]
#
#
#        # Store the ids and names of the ID concepts
#        #self.c_index_id = concept_id_list
#        #self.c_name_id = [name for name in self.c_names if name in concept_name_list]
#
#        # Store the ids and names of the OOD concepts
#        #self.c_index_ood = [id for id in range(self.n_concepts) if id not in concept_id_list]
#        #self.c_name_ood = [name for name in self.c_names if name not in concept_name_list]
#
#
#       # Update the ID & OOD interventions
#        self.set_id_ood_interventions()


