import numpy as np
import torch
import os
import warnings
import hydra # type: ignore
import pickle
from torch.utils.data import DataLoader
from src.causal_discovery.causal_discovery_block import causal_discovery
from src.completion.completion_block import complete_graph_with_llm
from src.data.utils import static_graph_collate
from pytorch_lightning.loggers import WandbLogger # type: ignore
from src.trainer import Trainer
from src.plots_mia import plot_and_save_max_mia, plot_and_save_max_sia
import subprocess
import json
from typing import Dict, List, Any

from hydra.utils import instantiate, call # type: ignore
from omegaconf import DictConfig, open_dict, OmegaConf # type: ignore

import pickle
import warnings
import time

from src.utils import (
    model_is_causal,
    seed_everything, 
    maybe_freeze_parameters, 
    update_config_from_data_subgroup_clients,
    aggregate_graph_proposals,
    aggregate, 
    get_parameters, 
    set_parameters, 
    set_old_parameters,
    load_dataloaders,
    identify_subgraph,
    dataprocess_auditing,
    initialize_mia_results,
    plot_training_metrics,
    compute_validation_loss,
    filter_dataloaders_by_concepts,
    build_local_graphs,
    maybe_update_config_with_graph_subgroup_clients
)

from src.dra import (
    run_dra_attack,
    summarize_dra_results,
)

# data loading
from src.data.dataset_block import get_dataset

#from src.server import get_evaluate_fn
#from src.strategy import CustomFedAvgWithModelSaving
from src.utils import clean_empty_configs
from src.data.dataset_block import get_dataset
from src.utils import get_intervention_policy, remove_cycles, remove_problematic_edges, get_split_paths, get_split_paths_fl
from src.utils import clean_empty_configs, update_config_from_data, maybe_update_config_with_graph, update_intervention_policy_and_graph, update_config_from_client
from src.data.utils import update_datasets, construct_combined_true_graph
from src.plots import maybe_plot_graph
from src.my_hydra import parse_hyperparams
from src.data.generate_split import generate_split, get_subgraph_dict, generate_split_with_fallback
from collections import OrderedDict
from typing import List, Dict, Tuple

from env import CACHE

# Suppress specific warning
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")


def _print_concept_availability(tag, loader, cfg, cid):
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "c") or dataset.c is None:
        print(f"\033[95m[{tag}] client {cid}: no concept labels in dataset.\033[0m")
        return
    c = dataset.c
    if c.numel() == 0:
        print(f"\033[95m[{tag}] client {cid}: empty concept tensor.\033[0m")
        return
    # A concept is unavailable if its column is all -1
    unavailable = (c == -1).all(dim=0)
    n_total = int(unavailable.numel())
    n_unavail = int(unavailable.sum().item())
    n_avail = n_total - n_unavail
    if hasattr(cfg, "engine") and hasattr(cfg.engine, "model") and hasattr(cfg.engine.model, "c_info"):
        names = cfg.engine.model.c_info.get("names", [])
    else:
        names = []
    unavailable_names = [names[i] for i in range(min(len(names), n_total)) if unavailable[i]]
    print(
        f"\033[95m[{tag}] client {cid}: concepts available {n_avail}/{n_total}, "
        f"unavailable {n_unavail}/{n_total}.\033[0m"
    )
    if unavailable_names:
        print(f"\033[95m[{tag}] client {cid} unavailable concepts: {unavailable_names}\033[0m")
   

@hydra.main(config_path="conf", config_name="test", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # various preliminaries, it set the seed for reproducibility
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    os.mkdir('results')
    with open_dict(cfg): cfg.update(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {cfg.device} device")

    # adjust config
    cfg = clean_empty_configs(cfg)

    # instantiate the dataset, split into train, val, test
    # preprocess all of them and save the preprocessed dataset
    dataset, true_graph, dataset_directory = get_dataset(cfg.dataset, cfg.device, seed=cfg.seed)


    combined_dataset = OmegaConf.select(cfg, 'combined_datasets.other_datasets', default=None)
    if combined_dataset is not None:
        if cfg.learning.mode == "centralized":
            raise ValueError("Combined datasets are not supported in centralized learning mode.")

        if cfg.dataset.name not in ['colormnist', 'fashionmnist']:
            raise ValueError("Combined datasets are only supported for the 'colormnist' and 'fashionmnist' datasets.")

        datasets = {} 
        datasets[0] = dataset
        base_conf_path = hydra.utils.get_original_cwd() + "/conf/dataset"

        for i, ds_name in enumerate(cfg.combined_datasets.other_datasets):
            if ds_name not in ['colormnist', 'fashionmnist']:
                raise ValueError(f"Dataset {ds_name} is not supported.")
            if ds_name == cfg.dataset.name:
                warnings.warn(f"Dataset {ds_name} is the same as the original dataset. Skipping it.")
                continue
            ds_path = os.path.join(base_conf_path, f"{ds_name}.yaml")
            original_cfg = OmegaConf.load(ds_path)
            overrides = cfg.combined_datasets.get(ds_name, {})
            merged_cfg = OmegaConf.merge(original_cfg, overrides)
            new_dataset, _, _ = get_dataset(merged_cfg, cfg.device, seed=cfg.seed)

            # consistency checks
            # check at least two variables are in common
            original_variables = dataset.c_info['names']+dataset.y_info['names']
            new_variables = new_dataset.c_info['names']+new_dataset.y_info['names']
            common_variables = set(original_variables) & set(new_variables)
            if len(common_variables) < 2:
                raise ValueError(f"Dataset {ds_name} does not have at least two variables in common with the original dataset.")
            else:
                datasets = update_datasets(datasets, new_dataset, cfg.combined_datasets)
                graph = construct_combined_true_graph(datasets, cfg.dataset, cfg.combined_datasets)
            
    else:
        datasets = {0: dataset}
    
    print(OmegaConf.to_yaml(cfg))
    

    if cfg.dataset.load_graph:
        try:
            if true_graph is None or cfg.dataset.load_true_graph == False:
                with open(os.path.join(dataset_directory, "learned_graph.pkl"), 'rb') as f:
                    graph = pickle.load(f)

            else:
                graph = true_graph
        except FileNotFoundError:
            print("Graph file not found. Change the config or run graph learning.")
    else:
        # graph construction
        if len(datasets)>1:
            raise NotImplementedError("Multiple datasets are not supported in the current version for graph construction.")
        else:
            if true_graph is None or cfg.dataset.load_true_graph == False:
                # estimate causal graph with causal structural learning algorithms
                graph = causal_discovery(cfg, dataset, true_graph, save_file_name="graph_causal_discovery.pkl")
                #if true_graph is not None:
                #    hamming = hamming_distance(true_graph, predicted_graph)
                #    print('(after CD) structural hamming distance: ', hamming)    

                # complete the causal graph with LLM and RAG
                graph = complete_graph_with_llm(cfg, graph, cfg.dataset.name)
                #if true_graph is not None:
                #    hamming = hamming_distance(true_graph, completed_graph)
                #     print('(after LLM + RAG) structural hamming distance: ', hamming)
                graph, dataset = remove_problematic_edges(graph, dataset)
                y_index = graph.columns.get_loc(datasets[0].y_info['names'][0])
                graph = remove_cycles(graph, y_index)
                #graph = completed_graph
                with open(os.path.join(dataset_directory, "learned_graph.pkl"), 'wb') as f:
                    pickle.dump(graph, f)
            else:
                graph = true_graph


    
    # interv graph must be always the true graph if available
    if true_graph is not None:
       interv_graph = true_graph.copy()
    else:
       interv_graph = graph.copy()
            
    # get the causal graph
    #if cfg.dataset.load_true_graph:
    #    graph = true_graph
    #else:
    #    if cfg.dataset.load_graph:
    #        with open(os.path.join(dataset_directory, "graph.pkl"), 'rb') as f:
    #            graph = pickle.load(f)
    #    else:
    #        # estimate causal graph with causal structural learning algorithms
    #        predicted_graph = causal_discovery(cfg, dataset, true_graph)
    #        #if true_graph is not None:
    #        #    hamming = hamming_distance(true_graph, predicted_graph)
    #        #    print('(after CD) structural hamming distance: ', hamming)    

    #        # complete the causal graph with LLM and RAG
    #        completed_graph = complete_graph_with_llm(cfg, predicted_graph, cfg.dataset.name)
    #        #if true_graph is not None:
    #        #    hamming = hamming_distance(true_graph, completed_graph)
    #        #     print('(after LLM + RAG) structural hamming distance: ', hamming)
    #        graph = completed_graph
    #
    #        # save graph
    #        with open(os.path.join(dataset_directory, "graph.pkl"), 'wb') as f:
    #            pickle.dump(graph, f)

    # fix the graph
    # (part 1): remove bidirected and undirected edges + add virtual nodes
    # edge can only be directed at this stage, the following function is just here in 
    # case the CD + LLM + RAG pipeline is modified and could produce bidirected or undirected edges
    #graph, dataset = remove_problematic_edges(graph, dataset)
    y_index = graph.columns.get_loc(datasets[0].y_info['names'][0])
    maybe_plot_graph(graph, 'graph')

      # it is ok also for multimodal because c_info and y_info contain all the variables of the datasets
        # insert
    #y_index = len(graph)-1
    
    ## (part 2): remove cycles
    #graph = remove_cycles(graph, y_index)

    #if true_graph is not None:
        #hamming = hamming_distance(true_graph, graph)
        #print('(after fix) structural hamming distance: ', hamming)
    
    #partitions = get_partitions(graph, y_index, n_clients = 5)
    #col_to_index = {col: idx for idx, col in enumerate(graph.columns)}    
    #partitions_named = {
    #    term: [graph.columns[i] for i in indices]
    #    for term, indices in partitions.items()
    #}

    # use the graph to define an intervention policy at test time
    interv_policy, ip_names = get_intervention_policy(interv_graph, y_index)
    print('intervention policy:', interv_policy)
    print('intervention policy names:', ip_names)

    for i, dataset in datasets.items():
        [dataset.data[split].register_graph(graph) for split in dataset.data]

    # We split the data by selecting a sub-graph for each split
    if cfg.learning.mode == "centralized":
        subgraphs, subgraphs_concept_names, subgraphs_with_add_nodes, add_nodes_values, add_nodes_names = None, None, None, None, None
    else:
        # seed_everything(cfg.get("seed"))
        # subgraphs, subgraphs_concept_names, subgraphs_with_add_nodes, add_nodes_values, add_nodes_names = generate_split(cfg, datasets, graph, y_index)
        subgraphs, subgraphs_concept_names, subgraphs_with_add_nodes, add_nodes_values, add_nodes_names = \
            generate_split_with_fallback(
                cfg, datasets, graph, y_index,
                seed_everything_fn=seed_everything,  # <-- pass your seeding function
                step=100,
                max_tries=20,
            )        
        
        
        ## Save subgraphs_concept_names, add_nodes_values, and subgraphs_with_add_nodes
        #model_name = cfg.model._target_.split('.')[-1] if hasattr(cfg.model, '_target_') else 'model'
        #subgraphs_file = f"subgraphs_{cfg.dataset.name}_{model_name}_seed_{cfg.seed}.json"
        #with open(subgraphs_file, 'w') as f:
        #    json.dump({
        #        "subgraphs_concept_names": subgraphs_concept_names,
        #        "add_nodes_names": add_nodes_names,
        #        "subgraphs_with_add_nodes": subgraphs_with_add_nodes,
        #        "dataset_name": cfg.dataset.name,
        #        "model_type": model_name
        #    }, f, indent=2)
        #print(f"Saved subgraphs data to {subgraphs_file}")

    # update config based on the dataset
    # e.g., set input and output size of the model
    cfg = update_config_from_data(cfg, datasets, subgraphs_concept_names)
    if cfg.learning.mode == 'localized':
        interv_policy, graph = update_intervention_policy_and_graph(cfg, interv_policy, graph, datasets)  

    cfg = maybe_update_config_with_graph(cfg, graph, interv_policy)
    # check consistency
    if len(datasets) > 1:
        raise NotImplementedError("Multiple datasets are not supported in the current version for training.")
    else:
        combo_info = {'names': datasets[0].c_info['names'] + datasets[0].y_info['names'],
                        'cardinality': datasets[0].c_info['cardinality'] + datasets[0].y_info['cardinality']}
        assert list(cfg.engine.model.c_name_index.keys()) == combo_info['names'], "Concept names are ordered differently in the engine configuration and in the dataset one."

    ############ data block ########################################################################################
    #if combined_dataset is None:
        # Load the unique test-set (if not combined dataset)
        # test_dataloader = DataLoader(dataset.data['test'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
        # save the test dataloader
        #test_path = os.path.join(str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption), "test_dataloader.pkl")
        #with open(test_path, 'wb') as f:
        #    pickle.dump(test_dataloader, f)

    # Load the test dataloader for the specific client
    path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
    
    #test_path = get_split_paths(cfg, path, True)
    #with open(test_path, 'rb') as f:
    #    test_dataloader = pickle.load(f)

    # If the training is centralized
    if cfg.learning.mode in ['centralized', 'localized']:
        
        if cfg.learning.mode == 'localized':
            #path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
            print("\033[93mLocalized training\033[0m")
            # Load only the training and validation split specified by the local training parameters
            # From cache get the dataloader
            train_path, val_path, test_path = get_split_paths(cfg, path)
            # if the file is not found, raise an error
            if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
                raise FileNotFoundError(f"File {train_path} or {val_path} or {test_path} not found")
            # Load the dataloaders
            with open(train_path, 'rb') as f:
                train_dataloader = pickle.load(f)
                print("train_dataloader", train_dataloader)
                print("train_dataloader length", len(train_dataloader))
            with open(val_path, 'rb') as f:
                val_dataloader = pickle.load(f)   
            with open(test_path, 'rb') as f:
                test_dataloader = pickle.load(f)
        else:
            # Load all the training and validation splits
            train_dataloader = DataLoader(datasets[0].data['train'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
            val_dataloader = DataLoader(datasets[0].data['val'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
            test_dataloader = DataLoader(datasets[0].data['test'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)

        engine = instantiate(cfg.engine)
        
        
        try:
            trainer = Trainer(cfg)
            trainer.logger.log_hyperparams(parse_hyperparams(cfg))
            # ---- train
            trainer.fit(engine, train_dataloader, val_dataloader)
            # ----- test
            trainer.test(engine, test_dataloader, ckpt_path='best')
            trainer.logger.finalize("success")
        finally:
            if isinstance(trainer.logger, WandbLogger):
                trainer.logger.experiment.finish()
                
                
    elif cfg.learning.mode == 'local_federated':
        
        # hyperparameters   
        n_rounds = cfg.learning.settings.n_rounds 
        n_clients = cfg.learning.n_clients 
        patience = cfg.learning.settings.patience
        cfg.trainer.max_epochs = cfg.learning.settings.local_epochs
        cfg.trainer.patience = 0
        # use_concepts = True # whether to use concept information in the MIA attacks
        
        num_threads = cfg.learning.settings.num_threads
        print(f"\033[93mLocal Federated training with {n_clients} clients\033[0m")
        
        # set seed for reproducibility
        torch.set_num_threads(num_threads)
        # seed_everything(cfg.seed)
        
        # read client data
        train_dataloaders, val_dataloaders, test_dataloaders = load_dataloaders(cfg, path, n_clients * cfg.learning.subgraphs.get('dataset_client_multiplier', 1))
        train_dataloaders, canary_loaders, true_in_outs, sia_loader = dataprocess_auditing(train_dataloaders, n_clients * cfg.learning.subgraphs.get('dataset_client_multiplier', 1), cfg) # NOTE: for the moment we are reducing the training data size
        print("\033[94mNumber of samples per client:\033[0m")
        for i in range(len(train_dataloaders)):
            print(f"\033[94mClient {i}: {len(train_dataloaders[i].dataset)} samples\033[0m")

        # identify if y is present or not for the clients
        #y_present = []
        #for cid in range(len(val_dataloaders)):
        #    if val_dataloaders[cid].dataset.y[0]!=-1:
        #        y_present.append(True)
        #    else:
        #         y_present.append(False)
       
        t0 = time.time()
        best_loss = float('inf')
        best_round = 0
        no_improvement_count = 0
        sia_accuracies = []
        history = {"round": [], "loss_val_avg": [], "loss_val_client": {}, "y_acc_val_avg": [], "y_acc_val_client": {}}
        for cid in range(n_clients):
            history["loss_val_client"][cid] = []
            history["y_acc_val_client"][cid] = []
        mia_accuracies, mia_epsilons = initialize_mia_results(n_clients)
        n_dataset_clients = len(train_dataloaders)

        # determine clients for possible predrift and postdrift phases
        if cfg.learning.subgraphs.rnd_drift > 1:
            predrift_clients = list(range(1, n_clients + 1))
            postdrift_clients = list(range(n_clients + 1, min(2 * n_clients, n_dataset_clients) + 1))

            # check postdrift clients have subgraphs that cover all the subgraphs
            postdrift_subgraphs = set()
            for cid in postdrift_clients:
                subgraph_id = identify_subgraph(path, cid)
                subgraph = subgraphs_concept_names[f'subgraph_{subgraph_id}']
                # I want to add the subgraph, not the nodes
                postdrift_subgraphs.add(frozenset(subgraph))
     
            # check if postdrift_subgraphs cover all subgraphs, not nodes but subgraphs
            all_subgraphs = set()
            for sg in subgraphs_concept_names.values():
                all_subgraphs.add(frozenset(sg))

            if postdrift_subgraphs != all_subgraphs:
                raise ValueError("Post-drift clients do not cover all subgraphs. Adjust post-drift clients to include all subgraphs.")

        else:
            predrift_clients = None
            postdrift_clients = list(range(1, n_clients + 1))
            

        # determine node order
        node_order = datasets[0].c_info["names"] + datasets[0].y_info["names"]

        # update c_name_index, c_names_id, c_names_ood dictionaries based on predrift clients
        cfg_predrift = update_config_from_data_subgroup_clients(cfg, 
                                                           predrift_clients, 
                                                           datasets, 
                                                           subgraphs_concept_names, 
                                                           node_order)
        cfg_postdrift = cfg


        # determine whether to use graph aggregation, see if there is the dictionary "aggregate_graph" with local_graphs not none
        use_graph_agg = False
        if hasattr(cfg.learning.subgraphs, "aggregate_graph") and model_is_causal(cfg.model):
            agg_graph_cfg = cfg.learning.subgraphs.aggregate_graph
            if agg_graph_cfg is not None and hasattr(agg_graph_cfg, "local_graphs"):
                if agg_graph_cfg.local_graphs is not None and agg_graph_cfg.local_graphs != "none":
                    use_graph_agg = True

        if use_graph_agg:
            
            # build local graphs for all clients
            if predrift_clients is None:
                all_clients = postdrift_clients
            else:
                all_clients = predrift_clients + postdrift_clients

            local_graphs, local_weights = build_local_graphs(all_clients, 
                                                             cfg, 
                                                             train_dataloaders, 
                                                             datasets[0].y_info["names"][0], 
                                                             graph)

            # aggregate graphs for pre-drift and post-drift clients
            # predrift
            graph_predrift, _ = aggregate_graph_proposals(
                client_selection = predrift_clients,
                local_graphs=local_graphs,
                weights=local_weights,
                config_input=cfg_predrift,
                task_node=datasets[0].y_info["names"][0]
            )

            maybe_plot_graph(graph_predrift, 'graph_predrift')

            # postdrift
            graph_postdrift, _ = aggregate_graph_proposals(
                client_selection = postdrift_clients,
                local_graphs=local_graphs,
                weights=local_weights,
                config_input=cfg_postdrift,
                task_node=datasets[0].y_info["names"][0]
            )

            maybe_plot_graph(graph_postdrift, 'graph_postdrift')

            # update intervention policy for pre-drift clients
            interv_policy_predrift, ip_names_predrift = get_intervention_policy(graph_predrift, 
                                                                                y_index = graph_predrift.columns.get_loc(datasets[0].y_info["names"][0]) if graph_predrift is not None else None)
            # take in consideration that the intervention policy is already constructed relative to the graph_predrift with the node indices referred to it
            interv_policy_predrift_constructed = True
            # Note: intervention policy postidrft remains the one on the true graph to guarantee consistency among the models            
            cfg = maybe_update_config_with_graph(cfg, graph_postdrift, interv_policy)
        
        else:
            # update graph and intervention policy based on predrift clients
            interv_policy_predrift, graph_predrift = update_intervention_policy_and_graph(
                    cfg_predrift, interv_policy, graph, datasets
            )
            # the intervention policy is not constructed from the graph_predrift, it is just a selection of the one relative to the global graph -> the node indices are still related to the global graph
            interv_policy_predrift_constructed = False


        # update config predrift with the graph and intervention policy updated based on predrift clients
        cfg_predrift = maybe_update_config_with_graph_subgroup_clients(cfg_predrift, predrift_clients, graph_predrift,interv_policy_predrift,  interv_policy_predrift_constructed, datasets)
        

        # Filter dataloaders for predrift clients
        train_dataloaders = filter_dataloaders_by_concepts(
            train_dataloaders, 
            cfg_predrift,
            predrift_clients,
            subgraphs_concept_names,
            datasets[0].c_info['names'],
            path
        )

        val_dataloaders = filter_dataloaders_by_concepts(
            val_dataloaders, 
            cfg_predrift,
            predrift_clients,
            subgraphs_concept_names,
            datasets[0].c_info['names'],
            path
        )

        # if there is just the pre-drift phase, also filter test dataloaders
        if cfg.learning.subgraphs.rnd_drift > n_rounds:
            test_dataloaders = filter_dataloaders_by_concepts(
                test_dataloaders, 
                cfg_predrift,
                predrift_clients,
                subgraphs_concept_names,
                datasets[0].c_info['names'],
                path
            )


        # start federated learning rounds
        init_cfg = cfg_predrift if cfg_predrift is not None else cfg_postdrift
        init_engine = instantiate(init_cfg.engine)
        global_params = get_parameters(init_engine)
        global_param_keys = list(init_engine.model.state_dict().keys())
        drift_debug_printed = False
        for rnd in range(1, n_rounds + 1):
            print(f"\033[92m\n--> ROUND {rnd}/{n_rounds}\033[0m")
            client_params: List[Tuple[List[torch.Tensor], int]] = []
            val_losses, val_accs, sizes = [], [], []
            
            # ------------------------------------------------------------
            # Setup configuration based on drift round
            # ------------------------------------------------------------
            if rnd < cfg.learning.subgraphs.rnd_drift:
                # pre-drift phase: use only first n_clients info: concepts, subgraph, etc...
                start_n_client = 0
                #if cfg_predrift is None:
                #    cfg_predrift = update_config_with_subgroup_clients(
                #        cfg,
                #        graph,
                #        datasets,
                #        subgraphs_concept_names,
                #        interv_policy,
                #        subgroup_clients=predrift_clients,
                #    )
                cfg_round = cfg_predrift
            else:
                # post-drift phase: use last n_clients info: concepts, subgraph, etc...
                if rnd == cfg.learning.subgraphs.rnd_drift:
                    print("\033[93mDrift occurred: switching to new client data distributions\033[0m")
                if cfg.learning.subgraphs.rnd_drift>1:
                    start_n_client = n_clients
                else:
                    start_n_client = 0
                cfg_round = cfg_postdrift
            engine = instantiate(cfg_round.engine)
            engine.model.to(cfg.device)

            # if (rnd >= cfg.learning.subgraphs.rnd_drift) and True:
            if  cfg.learning.subgraphs.rnd_drift >=1:
                print("\033[95m[Drift Debug] Checking concept label availability (train/val) for post-drift clients\033[0m")
                for cid in range(start_n_client, start_n_client + n_clients):
                    _print_concept_availability("train", train_dataloaders[cid], cfg_round, cid)
                    if val_dataloaders[cid] is not None:
                        _print_concept_availability("val", val_dataloaders[cid], cfg_round, cid)
                drift_debug_printed = True
            
            # ------------------------------------------------------------
            # local training (sequentially)
            # ------------------------------------------------------------
            print(f"\033[93mLocal training on {n_clients} clients\033[0m") 
            for n, cid in enumerate(range(start_n_client, start_n_client + n_clients)):
                # clone global params → local model
                update_config_from_client(cfg_round, datasets, cid)
                local_engine = instantiate(cfg_round.engine)
                # first training
                if rnd == cfg.learning.subgraphs.rnd_drift:
                    # load new architecture and update only those parameters that were present before
                    set_old_parameters(
                        local_engine,
                        global_params,
                        global_param_keys,
                        verbose=(cid == start_n_client),
                    )
                else:
                    set_parameters(local_engine, global_params)
                    
                local_engine.model.to(cfg.device)

                # freeze if required
                maybe_freeze_parameters(
                    train_dataloader = train_dataloaders[cid],
                    y_to_freeze = True if val_dataloaders[cid].dataset.y[0]==-1 else False, # to change if we can incorporate y in train_dataloaders[cid].dataset
                    model=local_engine.model,
                    learning=cfg.learning.mode,
                    freezing=cfg.learning.settings.freezing,
                )

                # local train
                trainer = Trainer(cfg, client_id=cid)
                trainer.logger.log_hyperparams(parse_hyperparams(cfg)) 
                trainer.fit(local_engine, train_dataloaders[cid])
                local_engine.model.to(cfg.device) # put back to device
                n_samples = len(train_dataloaders[cid].dataset)
                                    
                # # local validation
                # if val_dataloaders[cid] is not None:
                #     avg_loss = compute_validation_loss(local_engine.model, val_dataloaders[cid], cfg)
                #     history["loss_val_client"][n].append(avg_loss)
    
                # collect weights for aggregation
                client_params.append((get_parameters(local_engine), n_samples))

            
            # # ------------------------------------------------------------
            # # Privacy Attack: MIA
            # # ------------------------------------------------------------
            # if cfg.learning.settings.mia:
            #     print(f"\033[93mRunning Membership Inference Attack (MIA)\033[0m")
                
            #     set_parameters(local_engine, global_params)

            # if cfg.learning.settings.mia:
            #     #global_vec = flat_trainable_params_tensor(local_engine.model, cfg.device)

            #     start_n_client = 0 if rnd < cfg.learning.subgraphs.rnd_drift else n_clients 
            #     for cid in range(start_n_client, start_n_client + n_clients):
            #         # normalize client update vector
            #         true_in_out = true_in_outs[cid].float().numpy()
            #         set_parameters(local_engine, client_params[cid][0])
            #         client_vec = flat_trainable_params_tensor(local_engine.model, cfg.device)
            #         client_update = client_vec - global_vec
            #         client_update = client_update / torch.tensor(np.linalg.norm(client_update.cpu()), device=cfg.device) 

            #         # white-box attack (accumulate over the whole canary loader)
            #         set_parameters(local_engine, global_params)
            #         client_model = local_engine.model.to(cfg.device)
            #         scores_whitebox_list = []
            #         for batch in canary_loaders[cid]:
            #             scores_whitebox_list.append(score_whitebox_batch(batch, client_model, client_update, cfg, use_concepts=use_concepts))
            #         scores_whitebox = np.concatenate(scores_whitebox_list, axis=0)
            #         set_parameters(local_engine, client_params[cid][0])

            #         # black-box baseline (negative loss) accumulated over loader
            #         client_model = local_engine.model.to(cfg.device)
            #         scores_blackbox_loss = score_blackbox_loss_loader(canary_loaders[cid], client_model, cfg, use_concepts=use_concepts)

            #         # black-box concept-entropy (uses ONLY concepts if available, else falls back to label entropy)
            #         # scores_blackbox_concept = score_blackbox_concept_entropy_loader(canary_loaders[cid], client_model, cfg)

                    # # black-box shadow MLP (features = label confidences/margins/entropy/-loss + concept stats if available)
                    # shadow_epochs = getattr(getattr(cfg, "learning").settings, "mia_shadow_epochs", 100)
                    # scores_blackbox_shadow = shadow_mlp_scores_loader(
                    #     canary_loaders[cid],
                    #     client_model,
                    #     cfg,
                    #     y_mem_labels=true_in_out,
                    #     use_concepts=use_concepts,
                    #     epochs=shadow_epochs,
                    #     batch_size=64,
                    #     lr=1e-4,
                    #     k_folds=5,
                    #     scores_whitebox_list=None,  # no need to use them.. no effect observed in practice on asia
                    # )

            #         # evaluate white-box
            #         accuracy_mia, privacy_estimate = evaluate_privacy(scores_whitebox, true_in_out, cfg)
            #         print(f"Client {cid} - MIA accuracy (whitebox): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
            #         mia_accuracies['whitebox'][cid].append(accuracy_mia)
            #         mia_epsilons['whitebox'][cid].append(privacy_estimate)

            #         # evaluate black-box baseline (loss)
            #         accuracy_mia, privacy_estimate = evaluate_privacy(scores_blackbox_loss, true_in_out, cfg)
            #         print(f"Client {cid} - MIA accuracy (blackbox-loss): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
            #         mia_accuracies['blackbox'][cid].append(accuracy_mia)
            #         mia_epsilons['blackbox'][cid].append(privacy_estimate)

            #         # evaluate black-box concept-entropy
            #         # accuracy_mia, privacy_estimate = evaluate_privacy(scores_blackbox_concept, true_in_out, cfg)
            #         # print(f"Client {cid} - MIA accuracy (blackbox-concept): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
            #         # mia_accuracies['blackbox_concept'][cid].append(accuracy_mia)
            #         # mia_epsilons['blackbox_concept'][cid].append(privacy_estimate)

            #         # evaluate black-box shadow MLP
            #         accuracy_mia, privacy_estimate = evaluate_privacy(scores_blackbox_shadow, true_in_out, cfg)
            #         print(f"Client {cid} - MIA accuracy (blackbox-shadow): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
            #         mia_accuracies['blackbox_shadow'][cid].append(accuracy_mia)
            #         mia_epsilons['blackbox_shadow'][cid].append(privacy_estimate)
            
            # # ------------------------------------------------------------
            # # Privacy Attack: SIA
            # # ------------------------------------------------------------
            # if cfg.learning.settings.sia:
            #     print(f"\033[93mRunning Source Inference Attack (SIA)\033[0m")

            #     sia_accuracies.append(run_sia_attack(
            #         local_engine=local_engine,  # instantiated engine
            #         sia_loader=sia_loader,          # returned by dataprocess_auditing
            #         client_params=client_params,    # local models from this round
            #         cfg=cfg,
            #         use_concepts=use_concepts,
            #     ))
            #     print(f"\033[92mSIA accuracy this round: {sia_accuracies[-1]:.4f}\033[0m")  
            

            # ------------------------------------------------------------
            # FedAvg aggregation
            # ------------------------------------------------------------
            print(f"\033[93mAggregating local models\033[0m")
            global_params = aggregate(client_params)
            global_param_keys = list(local_engine.model.state_dict().keys())
            print("Saving global model parameters")
            params_dict = zip(local_engine.model.state_dict().keys(), global_params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            local_engine.model.load_state_dict(state_dict, strict=True)
            
            # ------------------------------------------------------------
            # FedAvg aggregation on client validation sets
            # ------------------------------------------------------------
            print(f"\033[93mEvaluating on client validation sets\033[0m")
            local_engine = instantiate(cfg_round.engine)
            set_parameters(local_engine, global_params)
            local_engine.model.to(cfg.device)
            for cid in range(start_n_client, start_n_client + n_clients):
                val_metrics = trainer.validate(local_engine, val_dataloaders[cid])[0]  #{'val/c/asia': 0.0, 'val/c/bronc': 0.0, 'val/c/either': 0.0, 'val/c/lung': 0.0, 'val/c/smoke': 0.0, 'val/c/tub': 0.0, 'val/c/xray': 0.0, 'val_loss': nan}
                val_losses.append(val_metrics['val_loss'])
                val_accs.append(val_metrics.get('val/y/y_accuracy', np.nan))  
                 # log per-client val metrics
                history["loss_val_client"][cid - start_n_client].append(val_metrics['val_loss'])
                history["y_acc_val_client"][cid - start_n_client].append(val_metrics.get('val/y/y_accuracy', np.nan))
                sizes.append(len(val_dataloaders[cid].dataset))

            # log aggregated val metrics (weighted)
            w_loss = sum(l * s for l, s in zip(val_losses, sizes)) / sum(sizes)
            val_accs = np.asarray(val_accs, dtype=float); sizes = np.asarray(sizes, dtype=float)
            mask = ~np.isnan(val_accs)          # keep only clients that actually have an accuracy
            if mask.any():
                y_acc = (val_accs[mask] * sizes[mask]).sum() / sizes[mask].sum()
            else:
                y_acc = np.nan
            history["round"].append(rnd)
            history["loss_val_avg"].append(w_loss)
            history["y_acc_val_avg"].append(y_acc)
            print(f"\033[92m✅ aggregated  val_loss={w_loss:.4f}, val/y/y_accuracy={y_acc:.4f}  \033[0m")

            # check improvement
            check_improvements = False
            if cfg.learning.subgraphs.rnd_drift > n_rounds:
                check_improvements = True
            elif rnd > cfg.learning.subgraphs.rnd_drift:
                check_improvements = True

            if check_improvements:
                if w_loss < best_loss:
                    best_loss = w_loss
                    best_round = rnd
                    no_improvement_count = 0
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(local_engine.model.state_dict(), f"checkpoints/model_round_{rnd}.pth")
                else:
                    no_improvement_count += 1

            # early stopping if no improvement after 'patience' rounds
            if no_improvement_count >= patience:
                print(f"\033[91mEarly stopping triggered at round {rnd}.\033[0m")
                break


        # ------------------------------------------------------------
        # Trim logged metrics up to the best round (drop overfitting tail)
        # ------------------------------------------------------------
        try:
            # Find index of best_round within history["round"], fallback to argmin
            if len(history["round"]) == 0:
                keep_upto = 0
            else:
                if best_round in history["round"]:
                    keep_upto = history["round"].index(best_round)
                else:
                    keep_upto = int(np.argmin(history["loss_val_avg"]))
            n_keep = keep_upto + 1

            # Trim history
            print(f"\033[94mValidation loss history before trimming: {history['loss_val_avg']}\033[0m")
            history["round"] = history["round"][:n_keep]
            history["loss_val_avg"] = history["loss_val_avg"][:n_keep]
            history["y_acc_val_avg"] = history["y_acc_val_avg"][:n_keep]
            for cid in range(n_clients):
                if cid in history["loss_val_client"]:
                    history["loss_val_client"][cid] = history["loss_val_client"][cid][:n_keep]
                    history["y_acc_val_client"][cid] = history["y_acc_val_client"][cid][:n_keep]
            # save history to json
            with open("results/training_history.json", "w") as fp:
                json.dump(history, fp, indent=2)

            # Trim SIA (per-round)
            if isinstance(sia_accuracies, list) and len(sia_accuracies) > 0:
                sia_accuracies[:] = sia_accuracies[:n_keep]

            # Trim MIA (per-round, per-client, per-variant)
            if isinstance(mia_accuracies, dict):
                for variant, per_client in mia_accuracies.items():
                    for cid in range(len(per_client)):
                        per_client[cid] = per_client[cid][:n_keep]
            if isinstance(mia_epsilons, dict):
                for variant, per_client in mia_epsilons.items():
                    for cid in range(len(per_client)):
                        per_client[cid] = per_client[cid][:n_keep]

            print(f"\033[96mTrimmed metrics to best round {best_round} (keeping {n_keep} rounds, removed {max(0, len(history['loss_val_avg']) - n_keep)}).\033[0m")
        except Exception as e:
            print(f"\033[91m[WARN] Failed trimming metrics: {e}\033[0m")


        plot_training_metrics(history)
        # ------------------------------------------------------------
        # Final evaluation on the test set
        # ------------------------------------------------------------
        print(f"\033[93mFinal evaluation on the test set\033[0m")
        # Load the best model
        ind_min_loss = np.argmin(history["loss_val_avg"])
        # best_round = history["round"][ind_min_loss]
        print(f"\033[92mBest round: {best_round} with loss {history['loss_val_avg'][ind_min_loss]:.4f}\033[0m")
        cfg_eval = cfg_predrift if best_round < cfg.learning.subgraphs.rnd_drift else cfg
        local_engine = instantiate(cfg_eval.engine)
        local_engine.model.load_state_dict(torch.load(f"checkpoints/model_round_{best_round}.pth", weights_only=False))

        # Evaluate the model on the client datasets 
        for testid in range(len(test_dataloaders)):
            test_dataloader = test_dataloaders[testid]   
            trainer.test(local_engine, test_dataloader)        
        print(f"\033[90mFinished! Training time: {round((time.time() - t0)/60, 2)} minutes\033[0m")
    
        # Evaluate the model on the client datasets
        #test_losses, sizes = [], []
    #{'val/c/asia': 0.0, 'val/c/bronc': 0.0, 'val/c/either': 0.0, 'val/c/lung': 0.0, 'val/c/smoke': 0.0, 'val/c/tub': 0.0, 'val/c/xray': 0.0, 'val_loss': nan}
            #test_losses.append(test_metrics['test_loss'])
            #sizes.append(len(test_dataloaders[cid].dataset))

            # log aggregated val metrics (weighted)
            #t_loss = sum(l * s for l, s in zip(test_losses, sizes)) / sum(sizes)

        print(f"\033[90mFinished! Training time: {round((time.time() - t0)/60, 2)} minutes\033[0m")
        
        
        # # save mia results
        # if cfg.learning.settings.mia:
        #     print(f"Saving MIA results: {os.getcwd() + '/mia_results.json'}")
        #     with open("mia_results.json", "w") as fp:
        #         json.dump(
        #             {
        #                 "accuracies": mia_accuracies,  
        #                 "epsilons":   mia_epsilons,
        #             }, fp, indent=2)
            
        #     # plot MIA results
        #     plot_and_save_max_mia(show=False)
        
        # # save sia results
        # if cfg.learning.settings.sia:
        #     print(f"Saving SIA results: {os.getcwd() + '/sia_results.json'}")
        #     with open("sia_results.json", "w") as fp:
        #         json.dump(
        #             {
        #                 "accuracies": sia_accuracies,
        #             }, fp, indent=2)
            
        #     # plot SIA results
        #     plot_and_save_max_sia(out_json="sia_max.json", show=False)
        

        # # ------------------------------------------------------------
        # # Run DRA attacks
        # # ------------------------------------------------------------ 
        # if cfg.learning.settings.dra:
            
        #     # Perform DRA on each client test set for n_samples
        #     for testid in range(len(test_dataloaders)):
        #         dra_results = run_dra_attack(
        #             test_dataloader=test_dataloaders[testid],
        #             model=instantiate(cfg.engine).model,
        #             device=cfg.device,
        #             methods=("DLG","iDLG"),
        #             max_attacks_per_loader=min(cfg.learning.settings.dra_samples_per_loader, len(test_dataloaders[testid].dataset)),
        #             iters=300,
        #             lr=1.0,
        #             early_stop_tol=1e-6,
        #             log_every=2000,
        #         )
        #         # save the dict dra_results 
        #         with open(f"dra_results_client_{testid}.json", "w") as fp:
        #             json.dump(dra_results, fp, indent=2)
        #         summarize_dra_results(f"dra_results_client_{testid}.json", metrics=("mse", "loss"))

  
    elif cfg.learning.mode == 'federated':

        # Add path to cfg
        with open_dict(cfg):
            cfg.path = os.getcwd()

        # Save the full configuration to a temporary file
        config_filepath = "temp_config.yaml"
        with open(config_filepath, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        os.environ["config_path"] = os.path.join(os.getcwd(),config_filepath)
        print(f"Config file saved to {os.path.join(os.getcwd(),config_filepath)}")

        # Instantiate the FL training
        print("\033[93mFederated training\033[0m")
        # subprocess.Popen(["bash", "../../../../../src/fl_training.sh"])
        subprocess.run(["bash", "../../../../../src/fl_training.sh"])

        print("\033[93mFederated training completed.\033[0m")
        
        # Delete the temporary config file
        os.remove(config_filepath)
    else:
        raise ValueError('The learning mode is not supported. Please choose one of the following: centralized, localized, federated.')


if __name__ == "__main__":
    main()
