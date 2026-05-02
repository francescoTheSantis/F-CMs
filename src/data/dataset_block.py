# from env import CACHE

# import os
# import pickle
# from hydra.utils import instantiate

# from src.data.preprocessing import preprocess_dataset
# from src.plots import maybe_plot_graph

# def get_dataset(dataset_cfg, device_cfg, seed):
#     """
#     1) instantiate the dataset, 
#     2) split into train, val, test
#     3) preprocess all of them 
#     4) save the preprocessed dataset.
#     Alternatively, if the 'cfg.dataset.load_embeddings' option is provided,
#     load the stored dataset.
#     Args:
#         cfg: DictConfig
#     Returns:
#         dataset: the preprocessed dataset
#     """
#     dataset_directory = os.path.join(str(CACHE / dataset_cfg.name))
#     os.makedirs(dataset_directory, exist_ok=True)

#     destination_path = os.path.join(dataset_directory, f"preprocessed_dataset_{seed}.pkl")
#     if dataset_cfg.get('load_embeddings') == False:
#         dataset = instantiate(dataset_cfg.loader)
#         dataset = preprocess_dataset(dataset_cfg, 
#                                 dataset, 
#                                 device=device_cfg,
#                                 seed=seed)
#         with open(destination_path, 'wb') as f: 
#             pickle.dump(dataset, f)
#     else:
#         with open(destination_path, 'rb') as f: 
#             dataset = pickle.load(f)
    
#     # store in the cache the c_info of the dataset (it will be used by the show_results script)
#     c_info_path = os.path.join(dataset_directory, "c_info.pkl")
#     with open(c_info_path, 'wb') as f: 
#         pickle.dump(dataset.c_info, f)

#     true_graph = dataset.load_ground_truth_graph()
#     maybe_plot_graph(true_graph, 'true_graph')
#     return dataset, true_graph, dataset_directory


from env import CACHE

import os
import pickle
from hydra.utils import instantiate

from src.data.preprocessing import preprocess_dataset
from src.plots import maybe_plot_graph


def _preprocessed_cache_suffix(dataset_cfg):
    if dataset_cfg.get("name") != "cheXpert_multi":
        return ""

    loader_cfg = dataset_cfg.get("loader", {})
    loader_all_balanced = loader_cfg is not None and loader_cfg.get("all_balanced", False)
    if dataset_cfg.get("all_balanced", False) or loader_all_balanced:
        return "_all_balanced"

    return ""


def get_dataset(dataset_cfg, device_cfg, seed):
    """
    1) instantiate the dataset, 
    2) split into train, val, test
    3) preprocess all of them 
    4) save the preprocessed dataset.
    Alternatively, if the 'cfg.dataset.load_embeddings' option is provided,
    load the stored dataset.
    Args:
        cfg: DictConfig
    Returns:
        dataset: the preprocessed dataset
    """
    dataset_directory = os.path.join(str(CACHE / dataset_cfg.name))
    os.makedirs(dataset_directory, exist_ok=True)

    cache_suffix = _preprocessed_cache_suffix(dataset_cfg)
    destination_path = os.path.join(dataset_directory, f"preprocessed_dataset_{seed}{cache_suffix}.pkl")
    if dataset_cfg.get('load_embeddings') == False:
        dataset = instantiate(dataset_cfg.loader)
        dataset = preprocess_dataset(dataset_cfg, 
                                dataset, 
                                device=device_cfg,
                                seed=seed)
        with open(destination_path, 'wb') as f: 
            pickle.dump(dataset, f)
    else:
        with open(destination_path, 'rb') as f: 
            dataset = pickle.load(f)
    
    # store in the cache the c_info of the dataset (it will be used by the show_results script)
    c_info_path = os.path.join(dataset_directory, "c_info.pkl")
    with open(c_info_path, 'wb') as f: 
        pickle.dump(dataset.c_info, f)

    true_graph = dataset.load_ground_truth_graph()
    maybe_plot_graph(true_graph, 'true_graph')
    return dataset, true_graph, dataset_directory
