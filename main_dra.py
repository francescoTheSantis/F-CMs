import torch
import os
import warnings
import hydra
import json
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict, OmegaConf
import warnings
import hydra
from src.utils import (
    seed_everything, 
    maybe_freeze_parameters, 
    load_dataloaders,
    dataprocess_auditing,
)

from src.dra import (
    run_dra_attack,
    summarize_dra_results,
)

"""

Main script to only run DRA attacks in a local federated learning setting.

"""

# data loading
from src.data.dataset_block import get_dataset
from src.utils import clean_empty_configs
from src.data.dataset_block import get_dataset
from src.utils import get_intervention_policy
from src.utils import clean_empty_configs, update_config_from_data, maybe_update_config_with_graph, update_intervention_policy_and_graph
from src.data.utils import update_datasets, construct_combined_true_graph
from src.plots import maybe_plot_graph
from src.data.generate_split import generate_split
from env import CACHE

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

    maybe_plot_graph(graph, 'graph')

    y_index = graph.columns.get_loc(datasets[0].y_info['names'][0])  # it is ok also for multimodal because c_info and y_info contain all the variables of the datasets

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

    # Load the test dataloader for the specific client
    path = str(CACHE / cfg.dataset.name / cfg.learning.annotation_assumption)
                
    if cfg.learning.mode == 'local_federated':
        
        # hyperparameters   
        n_rounds = cfg.learning.settings.n_rounds
        n_clients = cfg.learning.n_clients 
        patience = cfg.learning.settings.patience
        cfg.trainer.max_epochs = cfg.learning.settings.local_epochs
        cfg.trainer.patience = 0
        
        num_threads = cfg.learning.settings.num_threads
        print(f"\033[93mLocal Federated training with {n_clients} clients\033[0m")
        
        # set seed for reproducibility
        torch.set_num_threads(num_threads)
        seed_everything(cfg.seed)
        
        # read client data
        train_dataloaders, val_dataloaders, test_dataloaders = load_dataloaders(cfg, path, n_clients)
        train_dataloaders, canary_loaders, true_in_outs, sia_loader = dataprocess_auditing(train_dataloaders, cfg) # NOTE: for the moment we are reducing the training data size

        # try with and without these two lines
        engine = instantiate(cfg.engine)
        engine.model.to(cfg.device)
                
        # freeze if required
        # maybe_freeze_parameters(
        #     c=train_dataloaders[0].dataset.c,
        #     model=local_engine.model,
        #     learning=cfg.learning.mode,
        #     freezing=cfg.learning.settings.freezing,
        # )

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

    else:
        raise ValueError('The learning mode is not supported. Please choose one of the following: local_federated.')


if __name__ == "__main__":
    main()
