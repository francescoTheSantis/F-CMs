import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scienceplots

warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])

def delta_single_c_interventions_on_y(d):
    baseline = d['_baseline']
    delta_dict = {k:(v - baseline) for k, v in d.items()}
    # Remove the baseline from the delta_dict
    delta_dict.pop('_baseline', None)
    return delta_dict


def plot_single_c_on_y(input, custom_order, model_styles, folder=None):
    """
    This function will plot the results of the interventions on the y variable for each model and dataset
    It will create a grid of subplots, one for each dataset, and plot the results
    for each model with different colors and markers.
    The function will also create a single legend below the plots with the model names and styles.
    Each row will represent a learning method, each column a dataset and in each plot multiple lines will represent the models.
    """

    # The blackbox multi model is not included in the plot as it does not allow for interventions
    input = input[['seed', 'dataset', 'model', 'learning', 'single_c_interventions_on_y']].dropna()
    
    # compute the delta with respect to the _baseline and eliminate it
    input['single_c_interventions_on_y'] = input['single_c_interventions_on_y'].apply(delta_single_c_interventions_on_y)

    # Create a list of unique datasets and learning methods
    datasets = input['dataset'].unique()
    learning_methods = input['learning'].unique()

    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(len(learning_methods), len(datasets), hspace=0.4, wspace=0.4)

    axes = [fig.add_subplot(gs[i, j]) for i in range(len(learning_methods)) for j in range(len(datasets))]

    # Iterate over each learning method and dataset to create subplots
    for i, learning_method in enumerate(learning_methods):
        for j, dataset in enumerate(datasets):
            ax = axes[i * len(datasets) + j]
            subset = input[(input['learning'] == learning_method) & (input['dataset'] == dataset)]

            # Iterate over each model in the subset and plot the results
            for model in model_styles.keys():
                if model not in subset['model'].values:
                    continue

                model_subset = subset[subset['model'] == model]
                if model_subset.empty:
                    continue

                interventions = model_subset['single_c_interventions_on_y'].iloc[0]
                x_labels = list(interventions.keys())
                x = np.arange(len(x_labels))  # the x locations for the groups

                # Width of each bar
                bar_width = 0.2

                # Adjust bar position by model index (to avoid overlap)
                model_idx = list(model_styles.keys()).index(model)  # assumes you have a list of model names
                offset = model_idx * bar_width

                # Heights (values) for the current model
                bar_heights = [interventions[label] for label in x_labels]

                # Plot each model's bars with horizontal offset
                ax.bar(
                    x + offset,
                    bar_heights,
                    width=bar_width,
                    label=model,
                    color=model_styles[model]['color'],
                    alpha=0.7
                )

                # Set x-axis ticks and labels centered between groups
                ax.set_xticks(x + bar_width * (len(list(model_styles.keys())) - 1) / 2)
                ax.set_xticklabels(x_labels, rotation=45, ha='right')

            ax.set_title(f"{learning_method} - {dataset}")
            ax.set_xlabel("Concept Names")
            ax.set_ylabel("Single C Interventions on Y")
            ax.set_xticks(range(len(interventions)))
            ax.set_xticklabels(interventions.keys(), rotation=0, ha='right')
            ax.legend()
            ax.grid(True)
    # save the figure if a folder is provided
    if folder:
        plt.savefig(f"{folder}/single_c_interventions_on_y_{learning_method}_{dataset}.png", bbox_inches='tight')
    else:
        raise ValueError("Folder path is required to save the figure.")