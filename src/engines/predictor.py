from typing import Any, Optional, Mapping, Type
import pickle
import itertools

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.collections import _remove_prefix
import pytorch_lightning as pl
from src.utils import identify_subgraph
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
                c_names: Optional[list] = None,
                test_interv_policy: Optional[str] = None,
                test_interv_noise: Optional[float] = 0.,
                c_name_index: Optional[Mapping[str, int]] = None,
                c_names_ood: Optional[list] = None,
                annotation_assumption: Optional[str] = None
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

        self.c_names = c_names
        self.n_concepts = len(c_names)
        self.c_name_index = c_name_index

        # create an index to name mapping for concepts
        self.c_index_to_name_map = {i: name for i, name in zip(c_name_index.values(), c_name_index.keys())}
        self.id_c_idxes = list({k:v for k,v in self.c_index_to_name_map.items() if v in self.c_names}.keys())

        self.c_names_ood = c_names_ood if c_names_ood is not None else []
        self.c_names_all = c_names + self.c_names_ood # equal to self.c_names if c_names_ood is None or empty

        if metrics is None:
            metrics = dict()
        self._set_metrics(metrics)

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
        c_acc_metrics = {k: metrics.get('classification_acc') for k in self.c_names+self.c_names_ood}

        # task accuracy metrics
        if self.annotation_assumption == "task_included":
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
    
    def _remove_node_id_ood(self, nodes):
        for node in nodes:
            if node not in self.id_c_idxes:
                # the node is an OOD concept, we remove it from the list of nodes
                nodes.remove(node)
        return nodes

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

                if c_name_i not in self.c_names:
                    # The concept is not in the ID concepts, therefore interveaning on this concept
                    # does not have any effect on the task. 
                    # By the way, we just avoid to compute the intervention index but we still perform the forward pass
                    intervention_index = torch.zeros(c.shape, dtype=c.dtype, device=c.device)
                else:
                    # intervene on concept c_name_i
                    intervention_index = get_test_intervention_index(c.shape, i)
                inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
                # forward pass with intervention at test time
                y_output, c_output = self.forward(**inputs)
                y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                # update metric after intervention:
                # after interveening on concept c_name_i, how well can we predict y
                self.test_intervention_single_y[c_name_i].update(y_hat, y)

            # level intervention
            for l in range(0, len(self.test_interv_policy)+1):
                nodes = list(itertools.chain(*self.test_interv_policy[:l]))
                # If some nodes are OOD concepts, we remove them from the nodes list in order to not intervene on them
                nodes = self._remove_node_id_ood(nodes)
                intervention_index = get_test_intervention_index(c.shape, nodes)
                inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
                # forward pass with intervention at test time
                y_output, c_output = self.forward(**inputs)
                y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                # update metric after intervention:
                # after interveening on a level of the graph, how well can we predict y
                self.test_intervention_level_y[f'level {l}'].update(y_hat, y)
                # update metric after intervention:
                # after interveening on a level of the graph, how well can we predict each child concept
                childs = list(itertools.chain(*self.test_interv_policy[l:]))
                for child_index in childs:
                    c_name = self.c_names_all[child_index]
                    if c_name in self.c_names:
                        self.test_intervention_level_c[f'level {l}/child {c_name}'].update(c_hat[c_name], c[:,child_index])
                    else:
                        # There is no improvement on the concept accuracy for OOD concepts
                        self.test_intervention_level_c[f'level {l}/child {c_name}'].update(torch.Tensor([float('nan')]), 
                                                                                           torch.Tensor([float('nan')]))


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
        if calculate_c_metrics:
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

        if self.annotation_assumption=="task_included":
            self.update_and_log_metrics("train", y_hat, y, c_hat, c, batch)
        else:
            self.update_and_log_metrics("train", y_hat, y, c_hat, c, batch, calculate_c_metrics = True, calculate_y_metrics = False)
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
        if self.annotation_assumption=="task_included":
            self.update_and_log_metrics("val", y_hat, y, c_hat, c, batch)
        else:
            self.update_and_log_metrics("val", y_hat, y, c_hat, c, batch, calculate_c_metrics = True, calculate_y_metrics = False)
 
        #self.update_and_log_metrics("val", y_hat, y, c_hat, c, batch)
        self.log_loss("val", val_loss, batch_size=batch['batch_size'])
        return val_loss
    
    def test_step(self, batch, batch_idx):
        test_loss, y_output, c_output, y, c = self.shared_step(batch, step='test')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        self.update_and_log_metrics("test", y_hat, y, c_hat, c, batch)
        self.log_loss("test", test_loss, batch_size=batch['batch_size'])
        # test-time interventions
        self.test_intervention(batch)
            
        if 'Qualified' in self.c_names:
            self.test_intervention_fairness(batch)
        return test_loss

    def on_test_epoch_end(self):
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
                print(f"Task accuracy after intervention on {c_name}: {y_int[c_name]}")
            pickle.dump(y_int, open(f'results/single_c_interventions_on_y.pkl', 'wb'))

            # task accuracy after intervention of each graph level
            y_int = {}
            for k, metric in self.test_intervention_level_y.items():
                level = _remove_prefix(k, self.test_intervention_level_y.prefix)
                y_int[level] = metric.compute().item()
                print(f"Task accuracy after intervention on {level}: {y_int[level]}")
            pickle.dump(y_int, open(f'results/level_interventions_on_y.pkl', 'wb'))

            # individual child concept accuracy after
            # intervention on ancestors in the graph
            c_int = {}
            for k, metric in self.test_intervention_level_c.items():
                level = _remove_prefix(k, self.test_intervention_level_c.prefix)
                c_int[level] = metric.compute().item()
                print(f"Concept accuracy after intervention on {level}: {c_int[level]}")
            pickle.dump(c_int, open(f'results/level_interventions_on_c.pkl', 'wb'))

            # save graph and concepts
            pickle.dump({'concepts':self.c_names,
                         'policy':self.test_interv_policy}, open("graph.pkl", 'wb'))

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

    def prepare_for_test(self, cfg):
        """
        Prepare the model for test time. This is called before the test step.
        """
        self.model.eval()

        # Get the subgraph for the client
        concept_ids, concept_names = get_subgraph_dict(cfg)
        path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
        concept_id_list = concept_ids['subgraph_'+identify_subgraph(path, cfg.client_id)]
        concept_name_list = concept_names['subgraph_'+identify_subgraph(path, cfg.client_id)]

        # Store the ids and names of the ID concepts
        self.c_index_id = concept_id_list
        self.c_name_id = [name for name in self.c_names if name in concept_name_list]

        # Store the ids and names of the OOD concepts
        self.c_index_ood = [id for id in range(self.n_concepts) if id not in concept_id_list]
        self.c_name_ood = [name for name in self.c_names if name not in concept_name_list]

        # Update the ID & OOD interventions
        self.set_id_ood_interventions()


