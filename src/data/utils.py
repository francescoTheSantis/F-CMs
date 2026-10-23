import torch
import numpy as np
import pandas as pd
from copy import deepcopy
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_adj

def static_graph_collate(batch):
    result = {
        "x": torch.stack([item["x"] for item in batch]),
        "c": torch.stack([item["c"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "graph": batch[0]["graph"],  # Add the graph once
    }
    if "modality" in batch[0]:
        modalities = [item["modality"] for item in batch]
        result["modality"] = modalities[0] if len(set(modalities)) == 1 else modalities
    return result

def create_filtering_collate_fn(original_collate_fn, concept_indices_to_keep, cfg_predrift, all_concept_names):
    """
    Create a collate_fn that filters concepts to keep only specified indices.
    
    Args:
        original_collate_fn: The original collate function
        concept_indices_to_keep: List of concept indices to keep
    
    Returns:
        A wrapper collate function that filters concepts
    """
    def filtering_collate(batch):
        # Call original collate
        result = original_collate_fn(batch)
        
        # Filter concepts if present
        if 'c' in result and result['c'] is not None:
            result['c'] = result['c'][:, concept_indices_to_keep]
            # order result['c'] according to cfg.predrift.c_name_index
            #now the column name of result['c'] corresponds to all_concept_names[concept_indices_to_keep], I want to order them according to cfg_predrift.c_name_index
            if cfg_predrift is not None and 'c_name_index' in cfg_predrift.model:
                c_name_index = cfg_predrift.model['c_name_index']
                # get the names of the concepts in result['c']
                current_c_names = [all_concept_names[i] for i in concept_indices_to_keep]
                # get the indices to reorder current_c_names according to c_name_index
                reorder_indices = [current_c_names.index(name) for name in c_name_index if name in current_c_names]
                result['c'] = result['c'][:, reorder_indices]
        
        if 'graph' in result and cfg_predrift is not None and hasattr(cfg_predrift.engine.model, 'graph'):
            result['graph'] = cfg_predrift.engine.model.graph.copy()  # Ensure graph is included from cfg_predrift
        return result
    
    return filtering_collate

def reduce_dataset(_dataset, index_to_keep):
    dataset = deepcopy(_dataset)
    if dataset.X is not None:
        dataset.X = dataset.X[index_to_keep]
    if hasattr(dataset, 'X_image') and dataset.X_image is not None:
        dataset.X_image = dataset.X_image[index_to_keep]
    if hasattr(dataset, 'X_text') and dataset.X_text is not None:
        dataset.X_text = dataset.X_text[index_to_keep]
    if dataset.c is not None:
        dataset.c = dataset.c[index_to_keep]
    if dataset.y is not None:
        dataset.y = dataset.y[index_to_keep]
    if hasattr(dataset, 'df'):
        dataset.df = dataset.df.iloc[index_to_keep]
        dataset.df = dataset.df.reset_index(drop = True)
    return dataset

def split_dataset(_dataset, split_size):
    len_dataset = len(_dataset)
    # get the number of samples to be split
    n_split = int(split_size * len_dataset)
    # get the indices of samples to be split
    index_split = np.random.choice(len_dataset, n_split, replace=False)
    # get the indices of the training (or test) samples
    # np.setdiff1d returns a sorted array; shuffle so the retained order depends on the seed
    index_original = np.setdiff1d(np.arange(len_dataset), index_split)
    # np.random.shuffle(index_original)

    dataset_split = reduce_dataset(_dataset, index_split)
    dataset_original = reduce_dataset(_dataset, index_original)

    return dataset_original, dataset_split

def random_coloring(coloring_kwargs):
    return "red" if random.random() < coloring_kwargs['random_prob'] else "green"

def custom_coloring(values, coloring_kwargs):
    colors_results = []
    for index in range(len(values)):
        value = values[index]
        col_dict = coloring_kwargs[index]
        if col_dict["mode"]=="single values":
                if value in col_dict["values"]:
                    colors_results.append('red')
                else:
                    colors_results.append('green')
        elif col_dict["mode"]=="interval":
            if col_dict["values"][0]<= value and value <= col_dict["values"][1]:
                colors_results.append('red')
            else:
                colors_results.append('green')
        else:
            raise ValueError(f"invalid coloring_kwargs.")
    
    if colors_results.count('red') == len(colors_results):
        return 'red'
    else:
        return 'green'
    
def complex_coloring(scale, possible_scales, possible_degrees, possible_colors, coloring_kwargs):
    """
    Generate a random degree and color for an image based on the scale chosen for the image and the possible scales, degrees, and colors for the dataset.
    From coloring_kwargs, it will choose the degree and color that corresponds to the scale chosen for the image.
    Specifically, it will check the scale against the intervals defined in coloring_kwargs['scale'] and choose the corresponding degree and color
    specified in coloring_kwargs['degrees'] and coloring_kwargs['colors'].
    If there is not a unique degree and color for the scale, it will choose randomly among the associated degrees and colors.
    If the scale is not in the intervals defined in coloring_kwargs['scale'], it will choose randomly among the remaining degrees and colors not associated with any scale.
    Args:
        scale (float): the scale of the image
        possible_scales (list): the possible scales an image can have
        possible_degrees (list): the possible degrees an image can have
        possible_colors (list): the possible colors an image can have
        coloring_kwargs (dict): a dictionary with the scale, degree, and color kwargs
    Returns:
        degree: the degree of the image
        color: the color of the image
    """

    scale_kwargs = coloring_kwargs["scales"]
    degree_kwargs = coloring_kwargs["degrees"]
    color_kwargs = coloring_kwargs["colors"]

    degree = None
    color = None
    remaining_degrees = set(possible_degrees)
    remaining_colors = set(possible_colors)

    # check not overlapping scale_kwargs
    sorted_scale_kwargs = sorted(scale_kwargs, key=lambda x: x[0])
    for i in range(len(sorted_scale_kwargs) - 1):
        if sorted_scale_kwargs[i][1] >= sorted_scale_kwargs[i + 1][0]:
            raise ValueError("scale_kwargs should not overlap. Please check the intervals in scale_kwargs.")

    for index in range(len(scale_kwargs)):
        #check if all the scale values scale_kwargs is an interval [a,b] in scale_kwargs are in the possible scales
        scale_min = scale_kwargs[index][0]
        scale_max = scale_kwargs[index][1]
        degree_min = degree_kwargs[index][0]
        degree_max = degree_kwargs[index][1]
        degree_range = degree_max - degree_min

        if scale_max < scale_min:
            raise ValueError("scale_max should be greater than scale_min in scale_kwargs.")
        if scale_max < min(possible_scales) or scale_min > max(possible_scales):
            raise ValueError("scale_kwargs should be in the range of possible_scales:", possible_scales)
        if color_kwargs[index] not in possible_colors:
            raise ValueError("color_kwargs should be in the range of possible_colors:", possible_colors)
        
        
        # filter possible degrees for this interval
        possible_degrees_image = [d for d in possible_degrees if degree_min <= d <= degree_max]
        if len(possible_degrees_image) == 0:
            raise ValueError(f"No possible degrees in the range of degree_kwargs: {degree_range}")
        
        remaining_degrees -= set(possible_degrees_image)
        remaining_colors.discard(color_kwargs[index])
        
        if scale_min <= scale <= scale_max:
            degree = random.choice(possible_degrees_image)
            color = color_kwargs[index]


    if degree is None or color is None:
        if len(remaining_degrees) == 0 or len(remaining_colors) == 0:
            raise ValueError("No remaining degrees or colors to choose from for scales not specified in scales_kwargs. Review the scale_kwargs, degree_kwargs, and color_kwargs.")
        else:
            # assign a random degree from the remaining degrees
            degree = random.choice(remaining_degrees)
            # assign a random color from the remaining colors
            color = random.choice(remaining_colors)
    return degree, color
        
def transform_and_colorize(image: torch.Tensor, color: str, scale: float, degree: float) -> torch.Tensor:
    if image.dtype == "uint8":
        image = torch.tensor(image, dtype=torch.float32)
        colored_image = torch.zeros(3, 64, 64)
    else:
        colored_image = torch.zeros(3, 28, 28)  # Create an image with 3 channels (RGB)
    if color == 'red':
        colored_image[0] = image  # Red channel
    elif color == 'green':
        colored_image[1] = image  # Green channel
    
    if scale is not None and degree is not None:
        affine_transformer = v2.RandomAffine(degrees = (degree,degree), scale=(scale, scale))
        transformed_image = affine_transformer(colored_image)
    else:
        transformed_image = colored_image

    #plot images -- for debugging purposes
    #colored_image_plt = colored_image.permute(1, 2, 0).numpy()
    #colored_image_plt = colored_image_plt / 255.0  # Normalize to [0, 1] for plotting
    #transformed_image_plt = transformed_image.permute(1, 2, 0).numpy()
    #transformed_image_plt = transformed_image_plt / 255.0  # Normalize to [0, 1] for plotting
    #plt.imsave("colored_image.png", colored_image_plt)
    #plt.imsave("transformed_image.png", transformed_image_plt)
    return transformed_image

def change_task(dataset, task):
    '''
    Assign one of the dataset’s concepts as its new task.
    Args:
        - dataset: The dataset to modify.
        - task: The new task to assign to the dataset.
    '''
    if dataset.y_info['names'][0]==task:
        return None

    y_index = dataset.c_info['names'].index(task) if task in dataset.c_info['names'] else None

    temp_dataset = deepcopy(dataset)

    if y_index is None:
        raise ValueError(f"Task {task} not found in dataset.")
    
    for split in dataset.data:
        old_y_values = temp_dataset.data[split].y
        new_y_values = temp_dataset.data[split].c[:, y_index]
        if torch.is_tensor(old_y_values):
            old_y_values = old_y_values.view(-1)
        else:
            old_y_values = np.asarray(old_y_values).reshape(-1)
        if torch.is_tensor(new_y_values):
            new_y_values = new_y_values.view(-1, 1)
        else:
            new_y_values = np.asarray(new_y_values).reshape(-1, 1)
        dataset.data[split].y = new_y_values
        dataset.data[split].c[:, y_index] = old_y_values

    new_y_cardinality = [dataset.c_info['cardinality'][y_index]]
    old_y_cardinality = dataset.y_info['cardinality'][0]
    dataset.y_info['cardinality'] = new_y_cardinality
    dataset.c_info['cardinality'][y_index] = old_y_cardinality
    old_y_name = dataset.y_info['names'][0]
    dataset.y_info['names'] = [task]
    dataset.c_info['names'][y_index] = old_y_name

    

    return None

def expand_dataset(dataset, combined_names, combined_cardinalities):
    '''
    Expands the dataset to include all concepts in combined names.
    Args:
        - dataset: The dataset to expand.
        - combined_names: The names of the concepts to include in the dataset.
        - combined_cardinalities: The cardinalities of the concepts to include in the dataset.
    '''

    old_names = np.array(dataset.c_info['names'])

    # discover the order of old_names in combined_names
    order = [combined_names.index(name) for name in old_names if name in combined_names]

    for split in dataset.data:
        # place the old_data in the correct place of the new dataset
        old_data = dataset.data[split].c
        expanded_data = torch.full((old_data.shape[0], len(combined_names)), -1.0, dtype=old_data.dtype, device=old_data.device)
        expanded_data[:, order] = old_data
        dataset.data[split].c = expanded_data

    # adjust the names and the cardinality of the new concepts
    dataset.c_info['names'] = combined_names
    dataset.c_info['cardinality'] = combined_cardinalities

    return None

def update_datasets(old_datasets, new_dataset, cfg_combined_datasets):

    datasets = deepcopy(old_datasets)
    m_new_dataset = deepcopy(new_dataset)

    if cfg_combined_datasets.get('common_task', None) is not None:
        task = cfg_combined_datasets.common_task
        change_task(m_new_dataset, task)
        for key, dataset in datasets.items():
            change_task(dataset, task)
        
        combined_c_names = sorted(list(set(datasets[0].c_info['names']) | set(m_new_dataset.c_info['names'])))
        combined_cardinalities = [datasets[0].c_info['cardinality'][datasets[0].c_info['names'].index(name)] 
                                  if name in datasets[0].c_info['names'] 
                                  else 0 for name in combined_c_names]
        combined_cardinalities = [m_new_dataset.c_info['cardinality'][m_new_dataset.c_info['names'].index(name)]
                                  if name in m_new_dataset.c_info['names']
                                  else combined_cardinalities[combined_c_names.index(name)]
                                  for name in combined_c_names]
        for key,dataset in datasets.items():
            expand_dataset(dataset, combined_c_names, combined_cardinalities)
            datasets[key] = dataset
        expand_dataset(m_new_dataset, combined_c_names, combined_cardinalities)

        datasets[len(datasets)] = m_new_dataset
    else:
        raise ValueError("cfg_combined_datasets.common_task is not defined. Please define it in the configuration file.")

    return datasets

def construct_combined_true_graph(datasets, cfg_dataset, cfg_combined_datasets):
    if (cfg_dataset.name=='colormnist' and cfg_combined_datasets.other_datasets[0]=='fashionmnist') or \
        (cfg_dataset.name=='fashionmnist' and cfg_combined_datasets.other_datasets[0]=='colormnist'):
        node_labels = datasets[0].c_info['names'] + datasets[0].y_info['names']
        edges = [[4, 1], [1, 5], [2,3]]
        edge_index = torch.tensor(edges).t()
        values = to_dense_adj(edge_index)[0]
        adj = pd.DataFrame(values, index=node_labels, columns=node_labels, dtype=int)
    else:
        raise ValueError(f"Dataset {cfg_dataset.name} is not supported for combined true graph construction.")

    return adj



def modify_class_values(var):
   var_values = sorted(set(var))
   var_mapping = {old_val: new_val for new_val, old_val in enumerate(var_values)}
   return [var_mapping[val] for val in var]
