import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import hydra
import pickle
from torch.utils.data import DataLoader
from src.data.utils import static_graph_collate
from pytorch_lightning.loggers import WandbLogger
from src.trainer import Trainer
from src.plots_mia import plot_and_save_max_mia, plot_and_save_max_sia
import subprocess
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from hydra.utils import instantiate, call
from omegaconf import DictConfig, open_dict, OmegaConf

import pickle
from pathlib import Path
import warnings
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from src.utils import (
    seed_everything, 
    maybe_freeze_parameters, 
    aggregate, 
    get_parameters, 
    set_parameters, 
    load_dataloaders,
    score_blackbox_batch,
    score_whitebox_batch,
    dataprocess_auditing,
    evaluate_privacy,
    initialize_mia_results,
    run_sia_attack,
    flat_trainable_params_tensor,
    plot_training_metrics,
    compute_validation_loss,
    shadow_mlp_scores_loader,
    score_blackbox_loss_loader,
)

from src.dra import (
    run_dra_attack,
    summarize_dra_results,
)

# data loading
from src.data.dataset_block import get_dataset

# causal discovery
#from src.causal_discovery.causal_discovery_block import causal_discovery

# graph completion block
#from src.completion.completion_block import complete_graph_with_llm

#from src.server import get_evaluate_fn
#from src.strategy import CustomFedAvgWithModelSaving
from src.utils import clean_empty_configs
from src.data.dataset_block import get_dataset
from src.utils import get_intervention_policy, remove_cycles, remove_problematic_edges, get_split_paths, get_split_paths_fl
from src.utils import clean_empty_configs, update_config_from_data, maybe_update_config_with_graph, update_intervention_policy_and_graph, update_config_from_client
from src.data.utils import update_datasets, construct_combined_true_graph
from src.plots import maybe_plot_graph
from src.my_hydra import parse_hyperparams
from src.data.generate_split import generate_split, get_subgraph_dict
from collections import OrderedDict
from typing import List, Dict, Tuple
import copy

from env import CACHE
import shutil

# Suppress specific warning
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")
   

@hydra.main(config_path="conf", config_name="my_sweep", version_base="1.3")
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
    dataset, true_graph, dataset_directory = get_dataset(cfg.dataset, cfg.device)
    graph = true_graph

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
            new_dataset, _, _ = get_dataset(merged_cfg, cfg.device)

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
    maybe_plot_graph(graph, 'graph')

    y_index = graph.columns.get_loc(datasets[0].y_info['names'][0])  # it is ok also for multimodal because c_info and y_info contain all the variables of the datasets
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
    interv_policy, ip_names = get_intervention_policy(graph, y_index)
    print('intervention policy:', interv_policy)
    print('intervention policy names:', ip_names)

    for i, dataset in datasets.items():
        [dataset.data[split].register_graph(graph) for split in dataset.data]

    # We split the data by selecting a sub-graph for each split
    if cfg.learning.mode == "centralized":
        subgraphs, subgraphs_concept_names = None, None
    else:
        subgraphs, subgraphs_concept_names = generate_split(cfg, datasets, graph, y_index)

    # update config based on the dataset
    # e.g., set input and output size of the model
    cfg = update_config_from_data(cfg, datasets, subgraphs, subgraphs_concept_names)
    if cfg.learning.mode == 'localized':
        interv_policy, graph = update_intervention_policy_and_graph(cfg, interv_policy, graph, subgraphs, subgraphs_concept_names)  

    cfg = maybe_update_config_with_graph(cfg, graph, interv_policy)

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
        use_concepts = True # whether to use concept information in the MIA attacks
        
        num_threads = cfg.learning.settings.num_threads
        print(f"\033[93mLocal Federated training with {n_clients} clients\033[0m")
        
        # set seed for reproducibility
        torch.set_num_threads(num_threads)
        seed_everything(cfg.seed)
        
        # read client data
        train_dataloaders, val_dataloaders, test_dataloaders = load_dataloaders(cfg, path, n_clients)
        train_dataloaders, canary_loaders, true_in_outs, sia_loader = dataprocess_auditing(train_dataloaders, cfg) # NOTE: for the moment we are reducing the training data size
        print("\033[94mNumber of samples per client:\033[0m")
        for i in range(len(train_dataloaders)):
            print(f"\033[94mClient {i}: {len(train_dataloaders[i].dataset)} samples\033[0m")

        # try with and without these two lines
        engine = instantiate(cfg.engine)
        engine.model.to(cfg.device)
                
        t0 = time.time()
        best_loss = float('inf')
        best_round = 0
        no_improvement_count = 0
        sia_accuracies = []
        history = {"round": [], "loss_val_avg": [], "loss_val_client": {}}
        for cid in range(n_clients):
            history["loss_val_client"][cid] = []
        mia_accuracies, mia_epsilons = initialize_mia_results(n_clients)
        global_params = get_parameters(instantiate(cfg.engine))
        for rnd in range(1, n_rounds + 1):
            print(f"\033[92m\n--> ROUND {rnd}/{n_rounds}\033[0m")
            client_params: List[Tuple[List[torch.Tensor], int]] = []
            val_losses, sizes = [], []
            
            # ------------------------------------------------------------
            # local training (sequentially)
            # ------------------------------------------------------------
            print(f"\033[93mLocal training on {n_clients} clients\033[0m")
            for cid in range(n_clients):
                # clone global params → local model
                update_config_from_client(cfg, datasets, cid)
                local_engine = instantiate(cfg.engine)
                #print(local_engine.client_id)
                set_parameters(local_engine, global_params)
                local_engine.model.to(cfg.device)
                #local_engine.client_id = cid

                # freeze if required
                maybe_freeze_parameters(
                    c=train_dataloaders[cid].dataset.c,
                    model=local_engine.model,
                    learning=cfg.learning.mode,
                    freezing=cfg.learning.settings.freezing,
                )

                # local train
                trainer = Trainer(cfg, client_id=cid)
                trainer.logger.log_hyperparams(parse_hyperparams(cfg)) 
                trainer.fit(local_engine, train_dataloaders[cid])
                n_samples = len(train_dataloaders[cid].dataset)
                    
                # local validation
                if val_dataloaders[cid] is not None:
                    avg_loss = compute_validation_loss(local_engine.model, val_dataloaders[cid], cfg)
                    history["loss_val_client"][cid].append(avg_loss)
    
                # collect weights for aggregation
                client_params.append((get_parameters(local_engine), n_samples))

            
            # ------------------------------------------------------------
            # Privacy Attack: MIA
            # ------------------------------------------------------------
            if cfg.learning.settings.mia:
                print(f"\033[93mRunning Membership Inference Attack (MIA)\033[0m")
                
                set_parameters(local_engine, global_params)
                global_vec = flat_trainable_params_tensor(local_engine.model, cfg.device)

                for cid in range(n_clients):
                    # normalize client update vector
                    true_in_out = true_in_outs[cid].float().numpy()
                    set_parameters(local_engine, client_params[cid][0])
                    client_vec = flat_trainable_params_tensor(local_engine.model, cfg.device)
                    client_update = client_vec - global_vec
                    client_update = client_update / np.linalg.norm(client_update) 

                    # white-box attack (accumulate over the whole canary loader)
                    set_parameters(local_engine, global_params)
                    client_model = local_engine.model.to(cfg.device)
                    scores_whitebox_list = []
                    for batch in canary_loaders[cid]:
                        scores_whitebox_list.append(score_whitebox_batch(batch, client_model, client_update, cfg, use_concepts=use_concepts))
                    scores_whitebox = np.concatenate(scores_whitebox_list, axis=0)
                    set_parameters(local_engine, client_params[cid][0])

                    # black-box baseline (negative loss) accumulated over loader
                    client_model = local_engine.model.to(cfg.device)
                    scores_blackbox_loss = score_blackbox_loss_loader(canary_loaders[cid], client_model, cfg, use_concepts=use_concepts)

                    # black-box concept-entropy (uses ONLY concepts if available, else falls back to label entropy)
                    # scores_blackbox_concept = score_blackbox_concept_entropy_loader(canary_loaders[cid], client_model, cfg)

                    # black-box shadow MLP (features = label confidences/margins/entropy/-loss + concept stats if available)
                    shadow_epochs = getattr(getattr(cfg, "learning").settings, "mia_shadow_epochs", 100)
                    scores_blackbox_shadow = shadow_mlp_scores_loader(
                        canary_loaders[cid],
                        client_model,
                        cfg,
                        y_mem_labels=true_in_out,
                        use_concepts=use_concepts,
                        epochs=shadow_epochs,
                        batch_size=64,
                        lr=1e-4,
                        k_folds=10,
                        scores_whitebox_list=None,  # no need to use them.. no effect observed in practice on asia
                    )

                    # evaluate white-box
                    accuracy_mia, privacy_estimate = evaluate_privacy(scores_whitebox, true_in_out, cfg)
                    print(f"Client {cid} - MIA accuracy (whitebox): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
                    mia_accuracies['whitebox'][cid].append(accuracy_mia)
                    mia_epsilons['whitebox'][cid].append(privacy_estimate)

                    # evaluate black-box baseline (loss)
                    accuracy_mia, privacy_estimate = evaluate_privacy(scores_blackbox_loss, true_in_out, cfg)
                    print(f"Client {cid} - MIA accuracy (blackbox-loss): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
                    mia_accuracies['blackbox'][cid].append(accuracy_mia)
                    mia_epsilons['blackbox'][cid].append(privacy_estimate)

                    # evaluate black-box concept-entropy
                    # accuracy_mia, privacy_estimate = evaluate_privacy(scores_blackbox_concept, true_in_out, cfg)
                    # print(f"Client {cid} - MIA accuracy (blackbox-concept): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
                    # mia_accuracies['blackbox_concept'][cid].append(accuracy_mia)
                    # mia_epsilons['blackbox_concept'][cid].append(privacy_estimate)

                    # evaluate black-box shadow MLP
                    accuracy_mia, privacy_estimate = evaluate_privacy(scores_blackbox_shadow, true_in_out, cfg)
                    print(f"Client {cid} - MIA accuracy (blackbox-shadow): {accuracy_mia:.4f}, epsilon: {privacy_estimate:.4f}")
                    mia_accuracies['blackbox_shadow'][cid].append(accuracy_mia)
                    mia_epsilons['blackbox_shadow'][cid].append(privacy_estimate)
            
            # ------------------------------------------------------------
            # Privacy Attack: SIA
            # ------------------------------------------------------------
            if cfg.learning.settings.sia:
                print(f"\033[93mRunning Source Inference Attack (SIA)\033[0m")

                sia_accuracies.append(run_sia_attack(
                    local_engine=local_engine,  # instantiated engine
                    sia_loader=sia_loader,          # returned by dataprocess_auditing
                    client_params=client_params,    # local models from this round
                    cfg=cfg,
                    use_concepts=use_concepts,
                ))
                print(f"\033[92mSIA accuracy this round: {sia_accuracies[-1]:.4f}\033[0m")  
            

            # ------------------------------------------------------------
            # FedAvg aggregation
            # ------------------------------------------------------------
            print(f"\033[93mAggregating local models\033[0m")
            global_params = aggregate(client_params)
            print("Saving global model parameters")
            params_dict = zip(local_engine.model.state_dict().keys(), global_params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            local_engine.model.load_state_dict(state_dict, strict=True)
            
            # ------------------------------------------------------------
            # FedAvg aggregation on client validation sets
            # ------------------------------------------------------------
            print(f"\033[93mEvaluating on client validation sets\033[0m")
            local_engine = instantiate(cfg.engine)
            set_parameters(local_engine, global_params)
            local_engine.model.to(cfg.device)
            for cid in range(n_clients):
                val_metrics = trainer.validate(local_engine, val_dataloaders[cid])[0]  #{'val/c/asia': 0.0, 'val/c/bronc': 0.0, 'val/c/either': 0.0, 'val/c/lung': 0.0, 'val/c/smoke': 0.0, 'val/c/tub': 0.0, 'val/c/xray': 0.0, 'val_loss': nan}
                val_losses.append(val_metrics['val_loss'])
                sizes.append(len(val_dataloaders[cid].dataset))

            # log aggregated val metrics (weighted)
            w_loss = sum(l * s for l, s in zip(val_losses, sizes)) / sum(sizes)
            history["round"].append(rnd)
            history["loss_val_avg"].append(w_loss)
            print(f"\033[92m✅ aggregated  val_loss={w_loss:.4f}\033[0m")

            # check improvement
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
            history["round"] = history["round"][:n_keep]
            history["loss_val_avg"] = history["loss_val_avg"][:n_keep]
            for cid in range(n_clients):
                if cid in history["loss_val_client"]:
                    history["loss_val_client"][cid] = history["loss_val_client"][cid][:n_keep]

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
        local_engine = instantiate(cfg.engine)
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
        
        
        # save mia results
        if cfg.learning.settings.mia:
            print(f"Saving MIA results: {os.getcwd() + '/mia_results.json'}")
            with open("mia_results.json", "w") as fp:
                json.dump(
                    {
                        "accuracies": mia_accuracies,  
                        "epsilons":   mia_epsilons,
                    }, fp, indent=2)
            
            # plot MIA results
            plot_and_save_max_mia(show=False)
        
        # save sia results
        if cfg.learning.settings.sia:
            print(f"Saving SIA results: {os.getcwd() + '/sia_results.json'}")
            with open("sia_results.json", "w") as fp:
                json.dump(
                    {
                        "accuracies": sia_accuracies,
                    }, fp, indent=2)
            
            # plot SIA results
            plot_and_save_max_sia(out_json="sia_max.json", show=False)
        

        # ------------------------------------------------------------
        # Run DRA attacks
        # ------------------------------------------------------------ 
        if cfg.learning.settings.dra:
            
            # Perform DRA on each client test set for n_samples
            for testid in range(len(test_dataloaders)):
                dra_results = run_dra_attack(
                    test_dataloader=test_dataloaders[testid],
                    model=instantiate(cfg.engine).model,
                    device=cfg.device,
                    methods=("DLG","iDLG"),
                    max_attacks_per_loader=min(cfg.learning.settings.dra_samples_per_loader, len(test_dataloaders[testid].dataset)),
                    iters=300,
                    lr=1.0,
                    early_stop_tol=1e-6,
                    log_every=2000,
                )
                # save the dict dra_results 
                with open(f"dra_results_client_{testid}.json", "w") as fp:
                    json.dump(dra_results, fp, indent=2)
                summarize_dra_results(f"dra_results_client_{testid}.json", metrics=("mse", "loss"))

  
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
