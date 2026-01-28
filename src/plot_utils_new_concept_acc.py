import pickle
from xml.parsers.expat import model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scienceplots
import seaborn as sns
import os
import math
import json
from env import CACHE
import yaml
from statistics import NormalDist


warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])


################################################
############## FUNCTIONS #######################
################################################

def delta_single_c_interventions_on_y(d):
    baseline = d['_baseline']
    delta_dict = {k:(v - baseline) for k, v in d.items()}
    # Remove the baseline from the delta_dict
    delta_dict.pop('_baseline', None)
    return delta_dict

def reorder(learning_methods, clients_flag=False, clients_perspective=False):
    # the custom order is: centralized, local_federated
    # or: localized_1, localized_2, ..., localized_n
    if clients_flag:
        if clients_perspective:
            custom_order = [f'federated_{i+1}' for i in range(len(learning_methods))]
        else:
            custom_order = [f'localized_{i+1}' for i in range(len(learning_methods))]
    else:
        custom_order = ['centralized', 'local_federated', 'localized']
    # Reorder the learning methods according to the custom order
    ordered_learning_methods = [method for method in custom_order if method in learning_methods]
    return ordered_learning_methods

def rename_learning_methods(learning_method):
    for method in learning_method:
        if method == 'centralized':
            renamed_method = ['Centralized']
        elif method.startswith('localized'):
            # Extract the number from the method name
            num = method.split('_')[-1]
            renamed_method = [f'Localized (cl. {num})'] if any(char.isdigit() for char in num) else ['Localized']
        elif 'federated' in method and any(char.isdigit() for char in method):
            # Extract the number from the method name
            num = method.split('_')[-1]
            renamed_method = [f'Federated (cl. {num})']
        elif method == 'local_federated':
            renamed_method = ['Federated']
        else:
            raise ValueError(f"Unknown learning method: {method}")
    return renamed_method

def average_over_seed(input):
    d = {}
    for el in input:
        for k, v in el.items():
            if k not in d:
                d[k] = []
            d[k].append(v)

    mean = {k: np.mean(v) for k, v in d.items()}
    std = {k: np.std(v) for k, v in d.items()}
    return mean, std

def single_c_plot(
    input,
    custom_order,
    model_styles,
    folder=None,
    figsize=(20, 15),
    title_size=16,
    label_size=14,
    tick_size=14,
    legend_size=14,
    legend_bgcolor='lightgray',
    legend_edgecolor='black',
    legend_alpha=0.3,
    plot_name="default",
    clients_flag=False,
    client_perspective=False
):

    if input is None or input.empty:
        print(f"[WARN] No data available for {plot_name}; skipping plot.")
        return

    datasets = input['dataset'].unique()
    # Reorder datasets according to custom order
    datasets = sorted(datasets, key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
    learning_methods = input['learning'].unique()

    if plot_name=='single_c_interventions_on_y':
        learning_methods = reorder(learning_methods, clients_flag, client_perspective)

    # Check if there's any data to plot
    if len(learning_methods) == 0 or len(datasets) == 0:
        print(f"Warning: No data available to plot for {plot_name}. Skipping plot.")
        return

    # change models' names according to model_styles
    input['model'] = input['model'].apply(lambda x: model_styles[x]['name'] if x in model_styles else x)

    n_rows = len(learning_methods)
    n_cols = len(datasets)

    if n_rows == 0 or n_cols == 0 or len(model_styles) == 0:
        print(f"[WARN] {plot_name}: nothing to plot (datasets={n_cols}, learning_methods={n_rows}, models={len(model_styles)}).")
        return

    # Spacing tweaks to avoid label/title collisions
    axis_label_pad = 3
    title_pad = 15

    figsize = (6 * n_cols, 5 * n_rows)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.6, wspace=0.6)
    axes = [fig.add_subplot(gs[i, j]) for i in range(n_rows) for j in range(n_cols)]

    handles_labels = []

    for i, learning_method in enumerate(learning_methods):
        for j, dataset in enumerate(datasets):
            ax = axes[i * n_cols + j]
            subset = input[(input['learning'] == learning_method) & (input['dataset'] == dataset)]

            # Get unique x_labels for this subplot
            all_x_labels = set()
            for model in model_styles.values():
                if model['name'] in subset['model'].values:
                    model_subset = subset[subset['model'] == model['name']]
                    if not model_subset.empty:
                        interventions, _ = average_over_seed(model_subset[plot_name])
                        all_x_labels.update(interventions.keys())
            
            x_labels = sorted(list(all_x_labels))
            x = np.arange(len(x_labels))

            group_width = 0.8  # Total width for each group of bars
            bar_width = group_width / len(model_styles)  # Width of individual bars
            
            for model_idx, model in enumerate(model_styles.values()):
                if model['name'] not in subset['model'].values:
                    continue

                model_subset = subset[subset['model'] == model['name']]
                if model_subset.empty:
                    continue

                interventions, interventions_std = average_over_seed(model_subset[plot_name])
                
                # Calculate offset for this specific model
                offset = (model_idx - (len(model_styles) - 1) / 2) * bar_width

                bar_heights = [interventions.get(label, 0) for label in x_labels]
                bar_errors = [interventions_std.get(label, 0) / np.sqrt(len(model_subset)) for label in x_labels] # 1.96 *

                color = {el['name']:el['color'] for el in model_styles.values()}[model['name']]

                bars = ax.bar(
                    x + offset,
                    bar_heights,
                    yerr=bar_errors,
                    width=bar_width,
                    label=model['name'],
                    color=color,
                    alpha=0.7,
                    error_kw={'elinewidth': 0.5}  # Reduce the thickness of the error bars
                )

                if i == 0 and j == 0:
                    handles_labels.append(([bars[0], model['name']]))

            if len(x_labels) > 15:
                # Convert string IDs to numbers
                x_labels_numeric = list(range(len(x_labels)))
                ax.set_xticks(x_labels_numeric[::5])  # Show a tick every 5 ticks
                ax.set_xticklabels(x_labels_numeric[::5], rotation=0, ha='right', fontsize=tick_size)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=tick_size)
            
            ax.tick_params(axis='y', labelsize=tick_size)
            ax.minorticks_off()
            ax.grid(True)

            # Show x-axis label only for last row
            if i == n_rows - 1:
                ax.set_xlabel("Concept Names", fontsize=label_size, labelpad=axis_label_pad)

            # Show dataset name only in top row
            if i == 0:
                ax.set_title(dataset, fontsize=title_size, pad=title_pad)
            else:
                ax.set_title("")

            if j==0:
                ax.set_ylabel("$\\Delta$ on $y$", fontsize=label_size, labelpad=axis_label_pad)                

        # Get average vertical position of current row
        row_axes = [axes[i * n_cols + j] for j in range(n_cols)]
        bbox = [ax.get_position() for ax in row_axes]
        y_middle = np.mean([b.y0 + b.height / 2 for b in bbox])

        # Dynamically determine a good x-position based on left-most subplot
        leftmost_ax = row_axes[0].get_position()
        x_pos = leftmost_ax.x0 - 0.14  # Decrease this to get closer (0.02–0.03 usually works well)

        # Rename the learning methods
        if plot_name == 'single_c_interventions_on_y':
            learning_method = rename_learning_methods([learning_method])[0]

        fig.text(
            x_pos,
            y_middle,
            learning_method,
            va='center',
            ha='right',
            rotation='vertical',
            fontsize=title_size
        )

    # Shared legend with background
    handles, labels = zip(*handles_labels)
    legend = fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(model_styles),
        bbox_to_anchor=(0.5, -0.13),
        fontsize=legend_size,
        frameon=True
    )
    legend.get_frame().set_facecolor(legend_bgcolor)
    legend.get_frame().set_edgecolor(legend_edgecolor)
    legend.get_frame().set_alpha(legend_alpha)

    # Make space for row labels and legend
    plt.tight_layout(rect=[0.1, 0.12, 0.98, 0.98], pad=1.4, h_pad=1.2, w_pad=1.0)

    if folder:
        if clients_flag:
            if client_perspective:
                plt.savefig(f"{folder}/{plot_name}_clients_perspective.pdf", bbox_inches='tight')
            else:
                plt.savefig(f"{folder}/{plot_name}_clients.pdf", bbox_inches='tight')
        else:
            plt.savefig(f"{folder}/{plot_name}.pdf", bbox_inches='tight')
    else:
        raise ValueError("Folder path is required to save the figure.")


def plot_single_c_on_y(
    input,
    custom_order,
    model_styles,
    folder=None,
    figsize=(20, 15),
    title_size=16,
    label_size=14,
    tick_size=14,
    legend_size=14,
    legend_bgcolor='lightgray',
    legend_edgecolor='black',
    legend_alpha=0.3
):
    
    # Count the number of different clients
    n_clients = len(input[input['learning'].str.startswith('localized')]['learning'].unique())

    # Aggregate localized over the different clients
    # Rename all localized_idx to localized
    input_global = input.copy()
    input_global['learning'] = input['learning'].apply(lambda x: 'localized' if x.startswith('localized') else x)
    input_global = input_global[['seed', 'dataset', 'model', 'learning', 'single_c_interventions_on_y']].dropna()
    input_global['single_c_interventions_on_y'] = input_global['single_c_interventions_on_y'].apply(delta_single_c_interventions_on_y)
    single_c_plot(
        input_global,
        custom_order,
        model_styles,
        folder=folder,
        figsize=figsize,
        title_size=title_size,
        label_size=label_size,
        tick_size=tick_size,
        legend_size=legend_size,
        legend_bgcolor=legend_bgcolor,
        legend_edgecolor=legend_edgecolor,
        legend_alpha=legend_alpha,
        plot_name="single_c_interventions_on_y"
    )

    # Now show all the clients for localized
    inputs_clients = input.copy()
    # filter just localized
    inputs_clients = inputs_clients[inputs_clients['learning'].str.startswith('localized')]
    inputs_clients = inputs_clients[['seed', 'dataset', 'model', 'learning', 'single_c_interventions_on_y']].dropna()
    inputs_clients['single_c_interventions_on_y'] = inputs_clients['single_c_interventions_on_y'].apply(delta_single_c_interventions_on_y)
    single_c_plot(
        inputs_clients,
        custom_order,
        model_styles,
        folder=folder,
        figsize=figsize,
        title_size=title_size,
        label_size=label_size,
        tick_size=tick_size,
        legend_size=legend_size,
        legend_bgcolor=legend_bgcolor,
        legend_edgecolor=legend_edgecolor,
        legend_alpha=legend_alpha,
        plot_name="single_c_interventions_on_y",
        clients_flag=True
    )

    # Now i want to analyze the federated learning approach (local_federated).
    # Specifically, i want to show the now which are the OOD concepts for each client and store them in a dicitonary

    # filter only the clients
    input_clients = input[input['learning'].str.startswith('localized')]

    # group by dataset, learning and take the first concept_acc of each group
    ood_concepts = {}
    for (dataset, learning), group in input_clients.groupby(['dataset', 'learning']):
        d = group.iloc[0]['concept_acc']
        d = [k for k, v in d.items() if np.isnan(v)]
        # rename the learning as federated_idx
        learning_name = learning.replace('localized', 'federated')
        ood_concepts[(dataset, learning_name)] = d

    # Now i want to maintain only the federated 
    input_global = input.copy()
    input_global = input_global[input_global['learning'] == 'local_federated']
    input_global = input_global[['seed', 'dataset', 'model', 'learning', 'single_c_interventions_on_y']].dropna()
    input_global['single_c_interventions_on_y'] = input_global['single_c_interventions_on_y'].apply(delta_single_c_interventions_on_y)

    # for each row, i want to have as many rows as the number of clients. Each row will show the interventions for the OOD concepts of that client
    input_global['learning'] = [['federated_'+str(i) for i in range(1, n_clients+1)] for _ in range(len(input_global))]
    input_global = input_global.explode('learning').reset_index(drop=True)

    # for each row, maintain in the single_c_interventions_on_y only the concepts that are OOD for that client
    input_global['single_c_interventions_on_y'] = input_global.apply(
        lambda row: {k: v for k, v in row['single_c_interventions_on_y'].items() if k in ood_concepts.get((row['dataset'], row['learning']), [])},
        axis=1
    )
    single_c_plot(
        input_global,
        custom_order,
        model_styles,
        folder=folder,
        figsize=figsize,
        title_size=title_size,
        label_size=label_size,
        tick_size=tick_size,
        legend_size=legend_size,
        legend_bgcolor=legend_bgcolor,
        legend_edgecolor=legend_edgecolor,
        legend_alpha=legend_alpha,
        plot_name="single_c_interventions_on_y",
        clients_flag=True,
        client_perspective=True
    )


def plot_cumulative_single_c_on_y_OLD(
    input,
    custom_order,
    model_styles,
    folder=None,
    figsize=(20, 15),
    title_size=16,
    label_size=14,
    tick_size=14,
    legend_size=14,
    legend_bgcolor='lightgray',
    legend_edgecolor='black',
    legend_alpha=0.3,
    localized_client_id=None
):
    """
    Plot cumulative delta of single concept interventions on y.
    Concepts are ordered according to input['graph'].
    If interventions for some concepts are missing, the line is dashed between those points.
    
    Args:
        localized_client_id: ID of the specific localized client to plot (e.g., 1, 2, 3).
                            If None, only centralized and federated methods are plotted.
    """
    
    # Filter data based on localized_client_id
    input_global = input.copy()
    
    if localized_client_id is not None:
        # Keep only the specified localized client
        localized_name = f'localized_{localized_client_id}'
        input_global = input_global[
            (input_global['learning'] == 'centralized') |
            (input_global['learning'] == 'local_federated') |
            (input_global['learning'] == localized_name)
        ]
        # Rename the specific localized client to 'localized' for plotting
        input_global['learning'] = input_global['learning'].apply(
            lambda x: 'localized' if x == localized_name else x
        )
    else:
        # Keep only centralized and federated (no localized)
        input_global = input_global[
            (input_global['learning'] == 'centralized') |
            (input_global['learning'] == 'local_federated')
        ]
    
    input_global = input_global[['seed', 'dataset', 'model', 'learning', 'single_c_interventions_on_y', 'graph']].dropna()
    input_global['single_c_interventions_on_y'] = input_global['single_c_interventions_on_y'].apply(delta_single_c_interventions_on_y)

    # Reorder datasets and learning methods
    datasets = input_global['dataset'].unique()
    datasets = sorted(datasets, key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
    learning_methods = reorder(input_global['learning'].unique(), False, False)

    # Check if there's any data to plot
    if len(learning_methods) == 0 or len(datasets) == 0:
        print(f"Warning: No data available to plot for cumulative_single_c_interventions_on_y. Skipping plot.")
        return

    # Change models' names according to model_styles
    input_global['model'] = input_global['model'].apply(lambda x: model_styles[x]['name'] if x in model_styles else x)

    if len(learning_methods) == 0 or len(model_styles) == 0:
        print(f"[WARN] cumulative_single_c_interventions_on_y: nothing to plot.")
        return

    # Spacing tweaks
    axis_label_pad = 3
    title_pad = 15

    # Create a separate plot for each dataset
    for dataset in datasets:
        n_rows = len(learning_methods)
        n_cols = 1  # One column per dataset

        figsize = (8, 5 * n_rows)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.6, wspace=0.6)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]

        handles_labels = []

        for i, learning_method in enumerate(learning_methods):
            ax = axes[i]
            subset = input_global[(input_global['learning'] == learning_method) & (input_global['dataset'] == dataset)]

            if subset.empty:
                ax.axis('off')
                continue

            # Get the concept order from graph
            # Take the first row's graph (assuming same structure for same dataset/learning)
            graph = subset.iloc[0]['graph']
            concept_order = list(graph.keys()) if isinstance(graph, dict) else []
            
            if not concept_order:
                ax.axis('off')
                continue

            x = np.arange(len(concept_order))

            for model_idx, model in enumerate(model_styles.values()):
                if model['name'] not in subset['model'].values:
                    continue

                model_subset = subset[subset['model'] == model['name']]
                if model_subset.empty:
                    continue

                # Average over seeds
                interventions_list = model_subset['single_c_interventions_on_y'].tolist()
                
                # Compute cumulative values for each seed
                cumulative_values_per_seed = []
                for interventions_dict in interventions_list:
                    cumulative = 0
                    cumulative_values = []
                    missing_mask = []
                    for concept in concept_order:
                        if concept in interventions_dict:
                            cumulative += interventions_dict[concept]
                            cumulative_values.append(cumulative)
                            missing_mask.append(False)
                        else:
                            cumulative_values.append(cumulative)  # Keep previous value
                            missing_mask.append(True)
                    cumulative_values_per_seed.append((cumulative_values, missing_mask))
                
                # Average over seeds
                mean_cumulative = np.mean([cv for cv, _ in cumulative_values_per_seed], axis=0)
                std_cumulative = np.std([cv for cv, _ in cumulative_values_per_seed], axis=0)
                stderr_cumulative = 1.96 * std_cumulative / np.sqrt(len(cumulative_values_per_seed))
                
                # Aggregate missing mask (if any seed is missing, mark as missing)
                missing_mask = np.any([mm for _, mm in cumulative_values_per_seed], axis=0)

                color = model['color']

                # Plot line with segments: solid where data exists, dashed where missing
                for k in range(len(x) - 1):
                    if missing_mask[k] or missing_mask[k+1]:
                        linestyle = '--'
                    else:
                        linestyle = '-'
                    
                    ax.plot(
                        x[k:k+2],
                        mean_cumulative[k:k+2],
                        color=color,
                        linestyle=linestyle,
                        linewidth=2,
                        marker='o',
                        markersize=5
                    )

                # Plot error band with different alpha based on missing data
                for k in range(len(x) - 1):
                    alpha_value = 0.2 if (missing_mask[k] or missing_mask[k+1]) else 0.4
                    ax.fill_between(
                        x[k:k+2],
                        (mean_cumulative - stderr_cumulative)[k:k+2],
                        (mean_cumulative + stderr_cumulative)[k:k+2],
                        color=color,
                        alpha=alpha_value
                    )

                if i == 0:
                    # Create a dummy line for legend
                    line, = ax.plot([], [], color=color, linestyle='-', linewidth=2, marker='o', label=model['name'])
                    handles_labels.append((line, model['name']))

            ax.set_xticks(x)
            ax.set_xticklabels(concept_order, rotation=45, ha='right', fontsize=tick_size)
            ax.tick_params(axis='y', labelsize=tick_size)
            ax.minorticks_off()
            ax.grid(True, alpha=0.3)

            # Show x-axis label only for last row
            if i == n_rows - 1:
                ax.set_xlabel("Concept Names", fontsize=label_size, labelpad=axis_label_pad)

            # Show dataset name only in top row
            if i == 0:
                ax.set_title(dataset, fontsize=title_size, pad=title_pad)

            ax.set_ylabel("Cumulative $\\Delta$ on $y$", fontsize=label_size, labelpad=axis_label_pad)

            # Add row label for learning method on the left
            learning_method_name = rename_learning_methods([learning_method])[0]
            ax_pos = ax.get_position()
            fig.text(
                ax_pos.x0 - 0.12,
                ax_pos.y0 + ax_pos.height / 2,
                learning_method_name,
                va='center',
                ha='right',
                rotation='vertical',
                fontsize=title_size
            )

        # Shared legend
        if handles_labels:
            handles, labels = zip(*handles_labels)
            legend = fig.legend(
                handles,
                labels,
                loc='lower center',
                ncol=len(model_styles),
                bbox_to_anchor=(0.5, -0.08),
                fontsize=legend_size,
                frameon=True
            )
            legend.get_frame().set_facecolor(legend_bgcolor)
            legend.get_frame().set_edgecolor(legend_edgecolor)
            legend.get_frame().set_alpha(legend_alpha)

        plt.tight_layout(rect=[0.15, 0.08, 0.98, 0.98], pad=1.4, h_pad=1.2, w_pad=1.0)

        if folder:
            if localized_client_id is not None:
                plt.savefig(f"{folder}/cumulative_single_c_interventions_on_y_{dataset}_client_{localized_client_id}.pdf", bbox_inches='tight')
            else:
                plt.savefig(f"{folder}/cumulative_single_c_interventions_on_y_{dataset}.pdf", bbox_inches='tight')
        else:
            raise ValueError("Folder path is required to save the figure.")
        
        plt.close(fig)


def plot_single_architecture_multi_modality_OLD(
    input,
    custom_order,
    architecture_name,
    rnd_drift_values=None,
    folder=None,
    figsize=(20, 15),
    title_size=16,
    label_size=14,
    tick_size=14,
    legend_size=14,
    legend_bgcolor='lightgray',
    legend_edgecolor='black',
    legend_alpha=0.3,
    localized_client_id=None
):
    """
    Plot cumulative delta for a single architecture with multiple learning modalities/configurations.
    
    Args:
        architecture_name: Name of the architecture to plot (e.g., 'c2bm', 'cbm')
        rnd_drift_values: Dictionary mapping indices to rnd_drift values, e.g., {0: 0, 1: 10}
                          or a single value to use for all rows. If None, checks input data.
        localized_client_id: ID of the specific localized client to consider
    """
    
    # Filter data for the specified architecture
    input_filtered = input[input['model'] == architecture_name].copy()
    
    if input_filtered.empty:
        print(f"Warning: No data available for architecture {architecture_name}. Skipping plot.")
        return
    
    # Filter based on localized_client_id
    if localized_client_id is not None:
        localized_name = f'localized_{localized_client_id}'
        input_filtered = input_filtered[
            (input_filtered['learning'] == 'centralized') |
            (input_filtered['learning'] == 'local_federated') |
            (input_filtered['learning'] == localized_name)
        ]
        input_filtered['learning'] = input_filtered['learning'].apply(
            lambda x: 'localized' if x == localized_name else x
        )
    
    # Handle rnd_drift information
    if 'rnd_drift' not in input_filtered.columns:
        if rnd_drift_values is not None:
            # If rnd_drift_values is a dict, use it to map values
            if isinstance(rnd_drift_values, dict):
                input_filtered['rnd_drift'] = input_filtered.index.map(lambda x: rnd_drift_values.get(x, 0))
            else:
                # If it's a single value, use it for all rows
                input_filtered['rnd_drift'] = rnd_drift_values
        else:
            print(f"Warning: 'rnd_drift' column not found in input data and no rnd_drift_values provided. All local_federated will be treated as no drift.")
            input_filtered['rnd_drift'] = 0
    
    # Also get n_rounds information if available
    if 'n_rounds' not in input_filtered.columns:
        print(f"Warning: 'n_rounds' column not found in input data. Will assume rnd_drift comparison without n_rounds.")
        input_filtered['n_rounds'] = float('inf')  # Default to infinity so rnd_drift > n_rounds is always false
    
    # Create a new column that distinguishes local_federated by rnd_drift
    def create_learning_label(row):
        if row['learning'] == 'local_federated':
            # Convert to numeric to ensure comparison works
            rnd_drift = float(row['rnd_drift']) if row['rnd_drift'] is not None else 0
            n_rounds = float(row['n_rounds']) if row['n_rounds'] is not None else float('inf')
            
            # No drift if rnd_drift == 0 or rnd_drift > n_rounds (drift never occurs)
            if rnd_drift == 0 or rnd_drift > n_rounds:
                return 'local_federated_no_drift'
            else:
                return 'local_federated_drift'
        return row['learning']
    
    input_filtered['learning_label'] = input_filtered.apply(create_learning_label, axis=1)
    
    # Keep rnd_drift column in the filtered data
    columns_to_keep = ['seed', 'dataset', 'model', 'learning', 'learning_label', 'rnd_drift', 'n_rounds', 'single_c_interventions_on_y', 'graph', 'true_graph']
    input_filtered = input_filtered[[col for col in columns_to_keep if col in input_filtered.columns]]
    input_filtered = input_filtered.dropna(subset=['single_c_interventions_on_y', 'graph'])
    input_filtered['single_c_interventions_on_y'] = input_filtered['single_c_interventions_on_y'].apply(delta_single_c_interventions_on_y)

    datasets = input_filtered['dataset'].unique()
    datasets = sorted(datasets, key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
    learning_methods = input_filtered['learning_label'].unique()

    if len(learning_methods) == 0 or len(datasets) == 0:
        print(f"Warning: No data available to plot. Skipping.")
        return

    # Define colors for different learning methods
    learning_colors = {
        'centralized': '#1f77b4',
        'localized': '#d62728',
        'local_federated_no_drift': '#ff7f0e',
        'local_federated_drift': '#2ca02c'
    }
    
    # Define display names for learning methods
    learning_display_names = {
        'centralized': 'Centralized',
        'localized': 'Localized',
        'local_federated_no_drift': 'Federated (no drift)',
        'local_federated_drift': 'Federated (with drift)'
    }

    axis_label_pad = 3
    title_pad = 15

    # Create a separate plot for each dataset
    for dataset in datasets:
        figsize = (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        handles_labels = []

        subset = input_filtered[input_filtered['dataset'] == dataset]
        
        if subset.empty:
            plt.close(fig)
            continue

        # Determine concept order from true_graph if available, otherwise use graph
        if 'true_graph' in subset.columns and subset.iloc[0]['true_graph'] is not None:
            true_graph = subset.iloc[0]['true_graph']
            # true_graph is now a list of concept names (saved in main.py as list(true_graph.columns))
            if isinstance(true_graph, list):
                concept_order = true_graph 
            else:
                # Fallback if it's some other format
                concept_order = list(true_graph)
        else:
            # Fallback to using graph
            all_graphs = subset['graph'].tolist()
            concept_order = list(all_graphs[0].keys()) if isinstance(all_graphs[0], dict) else all_graphs[0]
        
        # eliminate from concept_order the last concept of true_graph, i.e., the true task
        #concept_order = concept_order[:-1]

        if not concept_order:
            plt.close(fig)
            continue

        x = np.arange(len(concept_order))



        for learning_method in learning_methods:
            method_subset = subset[subset['learning_label'] == learning_method]
            
            if method_subset.empty:
                continue

            # Average over seeds
            interventions_list = method_subset['single_c_interventions_on_y'].tolist()
            graphs_list = method_subset['graph'].tolist()
            
            # Compute cumulative values for each seed
            cumulative_values_per_seed = []
            for interventions_dict, method_graph in zip(interventions_list, graphs_list):
                # Determine which concepts are available for this learning method (from its graph)
                method_concepts = list(method_graph.keys()) if isinstance(method_graph, dict) else method_graph
                
                cumulative = 0
                cumulative_values = []
                missing_mask = []
                for concept in concept_order:
                    # A concept is available ONLY if it's in both the method's graph AND the interventions dict
                    if concept in method_concepts and concept in interventions_dict:
                        cumulative += interventions_dict[concept]
                        cumulative_values.append(cumulative)
                        missing_mask.append(False)
                    else:
                        # Concept not available for this learning method
                        cumulative_values.append(cumulative)
                        missing_mask.append(True)
                cumulative_values_per_seed.append((cumulative_values, missing_mask))
            
            # Calculate mean and std considering only seeds where concept is present
            mean_cumulative = []
            std_cumulative = []
            stderr_cumulative = []
            
            for concept_idx in range(len(concept_order)):
                # Get values for this concept only from seeds where it's present (not missing)
                values_for_concept = []
                for cumulative_values, missing_mask_seed in cumulative_values_per_seed:
                    if not missing_mask_seed[concept_idx]:  # Concept is present in this seed
                        values_for_concept.append(cumulative_values[concept_idx])
                
                if len(values_for_concept) > 0:
                    mean_cumulative.append(np.mean(values_for_concept))
                    std_cumulative.append(np.std(values_for_concept))
                    stderr_cumulative.append(1.96 * np.std(values_for_concept) / np.sqrt(len(values_for_concept)))
                else:
                    # No valid seeds for this concept - use 0
                    mean_cumulative.append(0)
                    std_cumulative.append(0)
                    stderr_cumulative.append(0)
            
            mean_cumulative = np.array(mean_cumulative)
            std_cumulative = np.array(std_cumulative)
            stderr_cumulative = np.array(stderr_cumulative)
            
            # Aggregate missing mask: a concept is missing if it's missing in ALL seeds
            missing_mask = np.all([mm for _, mm in cumulative_values_per_seed], axis=0)

            color = learning_colors.get(learning_method, '#333333')
            learning_method_name = learning_display_names.get(learning_method, learning_method)

            # Plot line with segments
            for k in range(len(x) - 1):
                if missing_mask[k+1]:
                    linestyle = '--'
                else:
                    linestyle = '-'
                
                ax.plot(
                    x[k:k+2],
                    mean_cumulative[k:k+2],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    marker='o',
                    markersize=6
                )

            # Plot error band with different alpha based on missing data
            for k in range(len(x) - 1):
                alpha_value = 0.2 if (missing_mask[k+1]) else 0.4
                ax.fill_between(
                    x[k:k+2],
                    (mean_cumulative - stderr_cumulative)[k:k+2],
                    (mean_cumulative + stderr_cumulative)[k:k+2],
                    color=color,
                    alpha=alpha_value
                )

            # Create dummy line for legend
            line, = ax.plot([], [], color=color, linestyle='-', linewidth=2, marker='o', label=learning_method_name)
            handles_labels.append((line, learning_method_name))

        ax.set_xticks(x)
        ax.set_xticklabels(concept_order, rotation=45, ha='right', fontsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        ax.minorticks_off()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Concept Names", fontsize=label_size, labelpad=axis_label_pad)
        ax.set_ylabel("Cumulative $\\Delta$ on $y$", fontsize=label_size, labelpad=axis_label_pad)
        ax.set_title(f"{dataset} - {architecture_name}", fontsize=title_size, pad=title_pad)

        if handles_labels:
            handles, labels = zip(*handles_labels)
            legend = ax.legend(
                handles,
                labels,
                loc='best',
                fontsize=legend_size,
                frameon=True
            )
            legend.get_frame().set_facecolor(legend_bgcolor)
            legend.get_frame().set_edgecolor(legend_edgecolor)
            legend.get_frame().set_alpha(legend_alpha)

        plt.tight_layout()

        if folder:
            if localized_client_id is not None:
                plt.savefig(f"{folder}/cumulative_{architecture_name}_multi_modality_{dataset}_client_{localized_client_id}.pdf", bbox_inches='tight')
            else:
                plt.savefig(f"{folder}/cumulative_{architecture_name}_multi_modality_{dataset}.pdf", bbox_inches='tight')
        else:
            raise ValueError("Folder path is required to save the figure.")
        
        plt.close(fig)


def plot_cumulative_accuracy_multi_modality(
    input,
    custom_order,
    architecture_name,
    c_info,
    rnd_drift_values=None,
    variable = 'labels',
    folder=None,
    figsize=(20, 15),
    title_size=16,
    label_size=14,
    tick_size=14,
    legend_size=14,
    legend_bgcolor='lightgray',
    legend_edgecolor='black',
    legend_alpha=0.3,
    localized_client_id=None
):
    """
    Plot average cumulative concept accuracies and task interventions for a single architecture 
    with multiple learning modalities. For each intervention level (e.g., '1_asia'), plot:
    - Average of all concept accuracies at that level (e.g., mean of '1_asia/bronc', '1_asia/asia', ...)
    - Task intervention value at that level (e.g., '1_asia' from cumulative_task_interventions)
    
    NaN values in concept accuracies are replaced with worst classifier (1/cardinality).
    
    Args:
        input: DataFrame containing the results
        architecture_name: Name of the architecture to plot
        c_info: Dictionary with concept cardinality information per dataset
        rnd_drift_values: Dictionary mapping indices to rnd_drift values
        localized_client_id: ID of the specific localized client to consider
    """
    
    # Filter data for the specified architecture
    input_filtered = input[input['model'] == architecture_name].copy()
    
    if input_filtered.empty:
        print(f"Warning: No data available for architecture {architecture_name}. Skipping plot.")
        return
    
  
    # Handle rnd_drift information
    if 'rnd_drift' not in input_filtered.columns:
        if rnd_drift_values is not None:
            if isinstance(rnd_drift_values, dict):
                input_filtered['rnd_drift'] = input_filtered.index.map(lambda x: rnd_drift_values.get(x, 0))
            else:
                input_filtered['rnd_drift'] = rnd_drift_values
        else:
            input_filtered['rnd_drift'] = 0
    
    if 'n_rounds' not in input_filtered.columns:
        input_filtered['n_rounds'] = float('inf')
    
    # Create learning label
    def create_learning_label(row):
        # Distinguish local_federated by rnd_drift
        if row['learning'] == 'local_federated':
            rnd_drift = float(row['rnd_drift']) if row['rnd_drift'] is not None else 0
            n_rounds = float(row['n_rounds']) if row['n_rounds'] is not None else float('inf')
            
            if rnd_drift == 0 or rnd_drift > n_rounds:
                return 'local_federated_no_drift'
            else:
                return 'local_federated_drift'

        # Check for multiple localized clients
        localized_clients = [col for col in input_filtered['learning'].unique() if col.startswith('localized_')]
        if len(localized_clients) > 1:
            print(f"Warning: Multiple localized clients found: {localized_clients}. This may affect the plot.")
        # Return 'localized' for all of them
        if row['learning'].startswith('localized_'):
            return 'localized'
        
        return row['learning']
    
    input_filtered['learning_label'] = input_filtered.apply(create_learning_label, axis=1)
    
    # Keep relevant columns
    required_columns = ['seed', 'dataset', 'model', 'learning', 'learning_label', 
                       'cumulative_task_interventions', 'cumulative_concept_interventions', 'graph', 'concept_acc', 'task_acc', 'predicted_concepts']
    optional_columns = ['rnd_drift', 'n_rounds']
    columns_to_keep = [col for col in required_columns + optional_columns if col in input_filtered.columns]
    
    input_filtered = input_filtered[columns_to_keep]
    input_filtered = input_filtered.dropna(subset=['cumulative_task_interventions', 'cumulative_concept_interventions'])

    datasets = input_filtered['dataset'].unique()
    datasets = sorted(datasets, key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
    learning_methods = input_filtered['learning_label'].unique()

    if len(learning_methods) == 0 or len(datasets) == 0:
        print(f"Warning: No data available to plot. Skipping.")
        return

    # Define colors for different learning methods
    learning_colors = {
        'centralized': '#2ca02c',
        'localized': '#d62728',
        'local_federated_no_drift': '#ff7f0e',
        'local_federated_drift': '#1f77b4'
    }
    
    # Define display names for learning methods
    learning_display_names = {
        'centralized': 'Centralized',
        'localized': 'Localized',
        'local_federated_no_drift': 'Federated (no drift)',
        'local_federated_drift': 'Federated (with drift)'
    }

    axis_label_pad = 3
    title_pad = 15

    # Create a separate plot for each dataset
    for dataset in datasets:
        figsize = (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        handles_labels = []

        subset = input_filtered[input_filtered['dataset'] == dataset]
        
        if subset.empty:
            plt.close(fig)
            continue

        # Get intervention levels from cumulative_task_interventions
        first_row_interventions = subset['cumulative_task_interventions'].iloc[0]
        if isinstance(first_row_interventions, dict):
            intervention_c_order = list(first_row_interventions.keys())
        else:
            print(f"Warning: cumulative_task_interventions is not a dict for dataset {dataset}. Skipping.")
            plt.close(fig)
            continue

        if not intervention_c_order:
            plt.close(fig)
            continue

        # Add baseline level to intervention order
        intervention_c_order_with_baseline = ['0_baseline'] + intervention_c_order
        x = np.arange(len(intervention_c_order_with_baseline))
        
        for learning_method in learning_methods:
            method_subset = subset[subset['learning_label'] == learning_method]
            
            if method_subset.empty:
                continue

            # Process data for each seed
            label_avg_per_seed = []
            missing_masks_per_seed = []
            

            for idx, row in method_subset.iterrows():
                cumulative_concept_dict = row['cumulative_concept_interventions']
                cumulative_task_dict = row['cumulative_task_interventions']
                baseline_concept_acc = row['concept_acc'] if 'concept_acc' in row else None
                baseline_task_acc = row['task_acc'] if 'task_acc' in row else None
                method_graph = row['predicted_concepts'] if 'predicted_concepts' in row else None

                assert set(baseline_concept_acc)== set(method_graph), "Mismatch between concept_acc keys and predicted_concepts keys"
                
                if not isinstance(cumulative_concept_dict, dict) or not isinstance(cumulative_task_dict, dict):
                    continue
                
                # Build missing mask for this seed
                missing_mask_seed = np.zeros(len(intervention_c_order), dtype=bool)
                if method_graph is not None:
                    # Convert method_graph to a list if it's a dict, listconfig, or other iterable
                    if isinstance(method_graph, dict):
                        method_concepts = list(method_graph.keys())
                    else:
                        # Handle listconfig, list, or other iterables
                        method_concepts = list(method_graph)
                    
                    # Convert to set for fast lookup
                    method_concepts_set = set(str(c) for c in method_concepts)
                    
                    # Mark intervention levels that are NOT in this seed's graph as missing
                    for i, level in enumerate(intervention_c_order):
                        # Extract clean concept name from intervention level (remove numeric prefix like "1_asia" -> "asia")
                        level_str = str(level)
                        if '_' in level_str and level_str.split('_')[0].isdigit():
                            clean_level = '_'.join(level_str.split('_')[1:])
                        else:
                            clean_level = level_str
                        
                        # Level is missing if clean name is NOT in method_concepts
                        if clean_level not in method_concepts_set:
                            missing_mask_seed[i] = True
                
                # add a False at the beginning to missing_mask_seed for baseline
                missing_mask_seed = np.insert(missing_mask_seed, 0, False)
                missing_masks_per_seed.append(missing_mask_seed)
                
               
                # For each intervention level, compute average concept accuracy
                concept_avg_for_level = []
                task_values_for_level = []
                
                # Iterate over levels including baseline
                for level in intervention_c_order_with_baseline:
                    # Get all concept accuracies for this level (e.g., all '1_asia/*')
                    concept_values = []
                    for key, value in cumulative_concept_dict.items():
                        if level =="0_baseline" and key.startswith(str(1)):
                            # For baseline, consider all concepts
                            concept_name = key.split('/')[-1]
                            if concept_name in baseline_concept_acc.keys():
                                value = np.nan
                                value = baseline_concept_acc.get(concept_name) if baseline_concept_acc is not None else None
                            else:
                                value = np.nan
                            
                        elif key.startswith(f"{level}/"):
                            # Extract concept name
                            concept_name = key.split('/')[-1]
                        else:
                            continue
                            
                        # Replace NaN with worst classifier
                        #if np.isnan(value):
                        #    if dataset.lower() in c_info and c_info[dataset.lower()] is not None:
                        #        try:
                        #            concept_idx = c_info[dataset.lower()]['names'].index(concept_name)
                        #            concept_cardinality = c_info[dataset.lower()]['cardinality'][concept_idx]
                        #            value = 1.0 / concept_cardinality
                        #        except (ValueError, KeyError, IndexError):
                        #            value = 0.5  # Default fallback
                        #    else:
                        #         value = 0.5  # Default fallback
                            
                        concept_values.append(value)
                    
                    # Compute average for this level
                    if concept_values:
                        concept_avg_for_level.append(float(np.mean([x for x in concept_values if not math.isnan(x)])))
                    else:
                        concept_avg_for_level.append(0.0)
                    
                    # Get task intervention value
                    if level == '0_baseline':
                        task_value = baseline_task_acc if baseline_task_acc is not None else 0.0
                    else:
                        task_value = cumulative_task_dict.get(level)
                    task_values_for_level.append(task_value)
                
                # Calculate label values as mean of concept avg and task value for each level
                if variable == 'labels':
                    label_values_for_level = [(c + t) / 2 for c, t in zip(concept_avg_for_level, task_values_for_level)]
                else:
                    label_values_for_level = task_values_for_level
                label_avg_per_seed.append(label_values_for_level)
            
            if not label_avg_per_seed:
                continue
            
            # Calculate mean and standard error across seeds
            label_array = np.array(label_avg_per_seed)
            mean_label = np.mean(label_array, axis=0)
            std_label = np.std(label_array, axis=0)
            stderr_label = 1.96*std_label / np.sqrt(len(label_avg_per_seed)) #1.96 *

            # Aggregate missing mask: a concept is missing if it's missing in ALL seeds
            if missing_masks_per_seed:
                missing_mask = np.all(missing_masks_per_seed, axis=0)
            else:
                missing_mask = np.zeros(len(intervention_c_order), dtype=bool)

            color = learning_colors.get(learning_method, '#333333')
            learning_method_name = learning_display_names.get(learning_method, learning_method)

            # Plot label average line with segments (dashed when concept is missing)
            for k in range(len(x)-1):
                # Use dashed line if next level is missing (skip k=0 which is baseline)
                if missing_mask[k+1]:
                    linestyle = '--'
                else:
                    linestyle = '-'
                
                ax.plot(
                    x[k:k+2],
                    mean_label[k:k+2],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    marker='o',
                    markersize=6
                )

            # Plot error band with different alpha based on missing data
            for k in range(len(x) - 1):
                # Check if next level is missing (skip k=0 which is baseline)
                alpha_value = 0.2 if (missing_mask[k+1]) else 0.4
                ax.fill_between(
                    x[k:k+2],
                    (mean_label - stderr_label)[k:k+2],
                    (mean_label + stderr_label)[k:k+2],
                    color=color,
                    alpha=alpha_value
                )

            # Create dummy line for legend
            line, = ax.plot([], [], color=color, linestyle='-', linewidth=2, marker='o', label=learning_method_name)
            handles_labels.append((line, learning_method_name))

        # Use only indices for x-axis labels
        level_labels = list(range(len(intervention_c_order_with_baseline)))
        
        # replace _ with space
        architecture_name = architecture_name.replace("_", " ")
        
        ax.set_xticks(x)
        ax.set_xticklabels(level_labels, rotation=0, ha='center', fontsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        ax.minorticks_off()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Number of intervened concepts", fontsize=label_size, labelpad=axis_label_pad)
        title = "Label" if variable == 'labels' else "Task"
        ax.set_ylabel(f"{title} Accuracy (%)", fontsize=label_size, labelpad=axis_label_pad)
        ax.set_title(f"{dataset} - {architecture_name} ({title} Accuracy)", fontsize=title_size, pad=title_pad)

        if handles_labels:
            handles, labels = zip(*handles_labels)
            legend = ax.legend(
                handles,
                labels,
                loc='best',
                fontsize=legend_size,
                frameon=True,
                ncol=1
            )
            legend.get_frame().set_facecolor(legend_bgcolor)
            legend.get_frame().set_edgecolor(legend_edgecolor)
            legend.get_frame().set_alpha(legend_alpha)

        plt.tight_layout()

        if folder:
            if localized_client_id is not None:
                plt.savefig(f"{folder}/cumulative_{variable}_acc_{architecture_name}_multi_modality_{dataset}_client_{localized_client_id}.pdf", bbox_inches='tight')
            else:
                plt.savefig(f"{folder}/cumulative_{variable}_acc_{architecture_name}_multi_modality_{dataset}.pdf", bbox_inches='tight')
        else:
            raise ValueError("Folder path is required to save the figure.")
        
        plt.close(fig)


def plot_cumulative_accuracy_multi_model(
    input,
    custom_order,
    learning_modality,
    c_info,
    rnd_drift_values=None,
    variable = 'labels',
    folder=None,
    figsize=(20, 15),
    title_size=16,
    label_size=14,
    tick_size=14,
    legend_size=14,
    legend_bgcolor='lightgray',
    legend_edgecolor='black',
    legend_alpha=0.3,
    localized_client_id=None
):
    """
    Plot average cumulative concept accuracies and task interventions for a single architecture 
    with multiple learning modalities. For each intervention level (e.g., '1_asia'), plot:
    - Average of all concept accuracies at that level (e.g., mean of '1_asia/bronc', '1_asia/asia', ...)
    - Task intervention value at that level (e.g., '1_asia' from cumulative_task_interventions)
    
    NaN values in concept accuracies are replaced with worst classifier (1/cardinality).
    
    Args:
        input: DataFrame containing the results
        architecture_name: Name of the architecture to plot
        c_info: Dictionary with concept cardinality information per dataset
        rnd_drift_values: Dictionary mapping indices to rnd_drift values
        localized_client_id: ID of the specific localized client to consider
    """
    
    # Filter data for the specified architecture
    input_filtered = input.copy()
    
    if input_filtered.empty:
        print(f"Warning: No data available.")
        return
    
  
    # Handle rnd_drift information
    if 'rnd_drift' not in input_filtered.columns:
        if rnd_drift_values is not None:
            if isinstance(rnd_drift_values, dict):
                input_filtered['rnd_drift'] = input_filtered.index.map(lambda x: rnd_drift_values.get(x, 0))
            else:
                input_filtered['rnd_drift'] = rnd_drift_values
        else:
            input_filtered['rnd_drift'] = 0
    
    if 'n_rounds' not in input_filtered.columns:
        input_filtered['n_rounds'] = float('inf')
    
    # Create learning label
    def create_learning_label(row):
        # Distinguish local_federated by rnd_drift
        if row['learning'] == 'local_federated':
            rnd_drift = float(row['rnd_drift']) if row['rnd_drift'] is not None else 0
            n_rounds = float(row['n_rounds']) if row['n_rounds'] is not None else float('inf')
            
            if rnd_drift == 0 or rnd_drift > n_rounds:
                return 'local_federated_no_drift'
            else:
                return 'local_federated_drift'

        # Check for multiple localized clients
        localized_clients = [col for col in input_filtered['learning'].unique() if col.startswith('localized_')]
        if len(localized_clients) > 1:
            print(f"Warning: Multiple localized clients found: {localized_clients}. This may affect the plot.")
        # Return 'localized' for all of them
        if row['learning'].startswith('localized_'):
            return 'localized'
        
        return row['learning']
    
    input_filtered['learning_label'] = input_filtered.apply(create_learning_label, axis=1)
    input_filtered = input_filtered[input_filtered['learning_label'] == learning_modality]

    # Keep relevant columns
    required_columns = ['seed', 'dataset', 'model', 'learning', 'learning_label', 
                       'cumulative_task_interventions', 'cumulative_concept_interventions', 'graph', 'concept_acc', 'task_acc', 'predicted_concepts']
    optional_columns = ['rnd_drift', 'n_rounds']
    columns_to_keep = [col for col in required_columns + optional_columns if col in input_filtered.columns]
    
    input_filtered = input_filtered[columns_to_keep]
    input_filtered = input_filtered.dropna(subset=['cumulative_task_interventions', 'cumulative_concept_interventions'])

    datasets = input_filtered['dataset'].unique()
    datasets = sorted(datasets, key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
    models = input_filtered['model'].unique()

    if len(models) == 0 or len(datasets) == 0:
        print(f"Warning: No data available to plot. Skipping.")
        return

    # Define colors for different learning methods
    model_colors = {
        'c2bm': '#2ca02c',
        'cbm_mlp': '#9467bd',
        'cbm_linear': '#ff7f0e',
        'cem': '#1f77b4',
        'blackbox': '#d62728',
        'cgm': '#00008B'
    }
    
    # Define display names for learning methods
    model_display_names = {
    'c2bm': 'C2BM',
    'cbm_mlp': 'CBM+MLP',
    'cbm_linear': 'CBM+Linear',
    'cem': 'CEM',
    'blackbox': 'BlackBox',
    'cgm': 'CGM'
    }


    axis_label_pad = 3
    title_pad = 15

    # Create a separate plot for each dataset
    for dataset in datasets:
        figsize = (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        handles_labels = []

        subset = input_filtered[input_filtered['dataset'] == dataset]
        
        if subset.empty:
            plt.close(fig)
            continue

        # Get intervention levels from cumulative_task_interventions
        first_row_interventions = subset['cumulative_task_interventions'].iloc[0]
        if isinstance(first_row_interventions, dict):
            intervention_c_order = list(first_row_interventions.keys())
        else:
            print(f"Warning: cumulative_task_interventions is not a dict for dataset {dataset}. Skipping.")
            plt.close(fig)
            continue

        if not intervention_c_order:
            plt.close(fig)
            continue

        # Add baseline level to intervention order
        intervention_c_order_with_baseline = ['0_baseline'] + intervention_c_order
        x = np.arange(len(intervention_c_order_with_baseline))
        
        for model in models:
            method_subset = subset[subset['model'] == model]
            
            if method_subset.empty:
                continue

            # Process data for each seed
            label_avg_per_seed = []
            missing_masks_per_seed = []
            

            for idx, row in method_subset.iterrows():
                cumulative_concept_dict = row['cumulative_concept_interventions']
                cumulative_task_dict = row['cumulative_task_interventions']
                baseline_concept_acc = row['concept_acc'] if 'concept_acc' in row else None
                baseline_task_acc = row['task_acc'] if 'task_acc' in row else None
                method_graph = row['predicted_concepts'] if 'predicted_concepts' in row else None

                assert set(baseline_concept_acc)== set(method_graph), "Mismatch between concept_acc keys and predicted_concepts keys"
                
                if not isinstance(cumulative_concept_dict, dict) or not isinstance(cumulative_task_dict, dict):
                    continue
                
                # Build missing mask for this seed
                missing_mask_seed = np.zeros(len(intervention_c_order), dtype=bool)
                if method_graph is not None:
                    # Convert method_graph to a list if it's a dict, listconfig, or other iterable
                    if isinstance(method_graph, dict):
                        method_concepts = list(method_graph.keys())
                    else:
                        # Handle listconfig, list, or other iterables
                        method_concepts = list(method_graph)
                    
                    # Convert to set for fast lookup
                    method_concepts_set = set(str(c) for c in method_concepts)
                    
                    # Mark intervention levels that are NOT in this seed's graph as missing
                    for i, level in enumerate(intervention_c_order):
                        # Extract clean concept name from intervention level (remove numeric prefix like "1_asia" -> "asia")
                        level_str = str(level)
                        if '_' in level_str and level_str.split('_')[0].isdigit():
                            clean_level = '_'.join(level_str.split('_')[1:])
                        else:
                            clean_level = level_str
                        
                        # Level is missing if clean name is NOT in method_concepts
                        if clean_level not in method_concepts_set:
                            missing_mask_seed[i] = True
                
                # add a False at the beginning to missing_mask_seed for baseline
                missing_mask_seed = np.insert(missing_mask_seed, 0, False)
                missing_masks_per_seed.append(missing_mask_seed)
                
               
                # For each intervention level, compute average concept accuracy
                concept_avg_for_level = []
                task_values_for_level = []
                
                # Iterate over levels including baseline
                for level in intervention_c_order_with_baseline:
                    # Get all concept accuracies for this level (e.g., all '1_asia/*')
                    concept_values = []
                    for key, value in cumulative_concept_dict.items():
                        if level =="0_baseline" and key.startswith(str(1)):
                            # For baseline, consider all concepts
                            concept_name = key.split('/')[-1]
                            if concept_name in baseline_concept_acc.keys():
                                value = np.nan
                                value = baseline_concept_acc.get(concept_name) if baseline_concept_acc is not None else None
                            else:
                                value = np.nan
                            
                        elif key.startswith(f"{level}/"):
                            # Extract concept name
                            concept_name = key.split('/')[-1]
                        else:
                            continue
                            
                        # Replace NaN with worst classifier
                        #if np.isnan(value):
                        #    if dataset.lower() in c_info and c_info[dataset.lower()] is not None:
                        #        try:
                        #            concept_idx = c_info[dataset.lower()]['names'].index(concept_name)
                        #            concept_cardinality = c_info[dataset.lower()]['cardinality'][concept_idx]
                        #            value = 1.0 / concept_cardinality
                        #        except (ValueError, KeyError, IndexError):
                        #            value = 0.5  # Default fallback
                        #     else:
                        #         value = 0.5  # Default fallback
                            
                        concept_values.append(value)
                    
                    # Compute average for this level
                    if concept_values:
                        concept_avg_for_level.append(float(np.mean([x for x in concept_values if not math.isnan(x)])))
                    else:
                        concept_avg_for_level.append(0.0)
                    
                    # Get task intervention value
                    if level == '0_baseline':
                        task_value = baseline_task_acc if baseline_task_acc is not None else 0.0
                    else:
                        task_value = cumulative_task_dict.get(level)
                    task_values_for_level.append(task_value)
                
                # Calculate label values as mean of concept avg and task value for each level
                if variable == 'labels':
                    label_values_for_level = [(c + t) / 2 for c, t in zip(concept_avg_for_level, task_values_for_level)]
                else:
                    label_values_for_level = task_values_for_level
                label_avg_per_seed.append(label_values_for_level)
            
            if not label_avg_per_seed:
                continue
            
            # Calculate mean and standard error across seeds
            label_array = np.array(label_avg_per_seed)
            mean_label = np.mean(label_array, axis=0)
            std_label = np.std(label_array, axis=0)
            stderr_label = 1.96*std_label / np.sqrt(len(label_avg_per_seed)) #1.96 *

            # Aggregate missing mask: a concept is missing if it's missing in ALL seeds
            if missing_masks_per_seed:
                missing_mask = np.all(missing_masks_per_seed, axis=0)
            else:
                missing_mask = np.zeros(len(intervention_c_order), dtype=bool)

            color = model_colors.get(model, '#333333')
            model_name = model_display_names.get(model, model)
            model_name = model_name.replace("_", " ")

            # Plot label average line with segments (dashed when concept is missing)
            for k in range(len(x)-1):
                # Use dashed line if next level is missing (skip k=0 which is baseline)
                if missing_mask[k+1]:
                    linestyle = '--'
                else:
                    linestyle = '-'
                
                ax.plot(
                    x[k:k+2],
                    mean_label[k:k+2],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    marker='o',
                    markersize=6
                )

            # Plot error band with different alpha based on missing data
            for k in range(len(x) - 1):
                # Check if next level is missing (skip k=0 which is baseline)
                alpha_value = 0.2 if (missing_mask[k+1]) else 0.4
                ax.fill_between(
                    x[k:k+2],
                    (mean_label - stderr_label)[k:k+2],
                    (mean_label + stderr_label)[k:k+2],
                    color=color,
                    alpha=alpha_value
                )

            # Create dummy line for legend
            line, = ax.plot([], [], color=color, linestyle='-', linewidth=2, marker='o', label=model_name)
            handles_labels.append((line, model_name))

        # Use only indices for x-axis labels
        level_labels = list(range(len(intervention_c_order_with_baseline)))
        
        learning_modality = learning_modality.replace("_", " ")

        ax.set_xticks(x)
        ax.set_xticklabels(level_labels, rotation=0, ha='center', fontsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        ax.minorticks_off()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Number of intervened concepts", fontsize=label_size, labelpad=axis_label_pad)
        title = "Label" if variable == 'labels' else "Task"
        ax.set_ylabel(f"{title} Accuracy (%)", fontsize=label_size, labelpad=axis_label_pad)
        ax.set_title(f"{dataset} - {learning_modality} ({title} Accuracy)", fontsize=title_size, pad=title_pad)

        if handles_labels:
            handles, labels = zip(*handles_labels)
            legend = ax.legend(
                handles,
                labels,
                loc='best',
                fontsize=legend_size,
                frameon=True,
                ncol=1
            )
            legend.get_frame().set_facecolor(legend_bgcolor)
            legend.get_frame().set_edgecolor(legend_edgecolor)
            legend.get_frame().set_alpha(legend_alpha)

        plt.tight_layout()

        if folder:
            if localized_client_id is not None:
                plt.savefig(f"{folder}/cumulative_{variable}_acc_{learning_modality}_multi_model_{dataset}_client_{localized_client_id}.pdf", bbox_inches='tight')
            else:
                plt.savefig(f"{folder}/cumulative_{variable}_acc_{learning_modality}_multi_model_{dataset}.pdf", bbox_inches='tight')
        else:
            raise ValueError("Folder path is required to save the figure.")
        
        plt.close(fig)

def delta_single_c_interventions_on_y_id_ood(d, base):
    baseline = base['_baseline']
    delta_dict = {}
    for k, v in base.items():
        if k =='_baseline':
            continue
        if k in d:
            delta_dict[k] = d[k] - baseline
        if k not in d:
            delta_dict[k] = 0

    return delta_dict

def level_interventions_plot(
    input,
    custom_order,
    model_styles,
    folder=None,
    figsize=(20, 15),
    title_size=16,
    label_size=14,
    tick_size=14,
    legend_size=14,
    legend_bgcolor='lightgray',
    legend_edgecolor='black',
    legend_alpha=0.3,
    plot_name="level_interventions_on_y",
    clients_flag=False
):

    if input is None or input.empty:
        print(f"[WARN] No data available for {plot_name}; skipping plot.")
        return

    datasets = input['dataset'].unique()
    # Reorder datasets according to custom order
    datasets = sorted(datasets, key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
    learning_methods = input['learning'].unique()

    learning_methods = reorder(learning_methods, clients_flag)

    # change models' names according to model_styles
    input['model'] = input['model'].apply(lambda x: model_styles[x]['name'] if x in model_styles else x)

    n_rows = len(learning_methods)
    n_cols = len(datasets)

    if n_rows == 0 or n_cols == 0 or len(model_styles) == 0:
        print(f"[WARN] {plot_name}: nothing to plot (datasets={n_cols}, learning_methods={n_rows}, models={len(model_styles)}).")
        return

    # Spacing tweaks to keep labels and titles from overlapping
    axis_label_pad = 5
    title_pad = 15

    figsize = (6 * n_cols, 5 * n_rows)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.6, wspace=0.6)
    axes = [fig.add_subplot(gs[i, j]) for i in range(n_rows) for j in range(n_cols)]

    handles_labels = []

    for i, learning_method in enumerate(learning_methods):
        for j, dataset in enumerate(datasets):
            ax = axes[i * n_cols + j]
            subset = input[(input['learning'] == learning_method) & (input['dataset'] == dataset)]

            # Get unique x_labels (intervention levels) for this subplot
            all_x_labels = set()
            for model in model_styles.values():
                if model['name'] in subset['model'].values:
                    model_subset = subset[subset['model'] == model['name']]
                    if not model_subset.empty:
                        interventions, _ = average_over_seed(model_subset[plot_name])
                        all_x_labels.update(interventions.keys())
            
            x_labels = sorted(list(all_x_labels))
            x = np.arange(len(x_labels))
            
            for model_idx, model in enumerate(model_styles.values()):
                if model['name'] not in subset['model'].values:
                    continue

                model_subset = subset[subset['model'] == model['name']]
                if model_subset.empty:
                    continue

                interventions, interventions_std = average_over_seed(model_subset[plot_name])

                line_heights = [interventions.get(label, 0) for label in x_labels]
                line_errors = [interventions_std.get(label, 0) / np.sqrt(len(model_subset)) for label in x_labels] # 1.96 *

                color = {el['name']:el['color'] for el in model_styles.values()}[model['name']]

                # Plot line with solid linestyle
                line = ax.plot(
                    x,
                    line_heights,
                    label=model['name'],
                    color=color,
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    alpha=0.8,
                    linestyle='-'
                )

                # Add error bars
                ax.errorbar(
                    x,
                    line_heights,
                    yerr=line_errors,
                    color=color,
                    alpha=0.6,
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                    fmt='none'
                )

                if i == 0 and j == 0:
                    handles_labels.append((line[0], model['name']))

            if len(x_labels) > 15:
                # Convert string IDs to numbers
                x_labels_numeric = list(range(len(x_labels)))
                ax.set_xticks(x_labels_numeric[::5])  # Show a tick every 5 ticks
                ax.set_xticklabels(x_labels_numeric[::5], rotation=0, ha='right', fontsize=tick_size)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, rotation=0, ha='right', fontsize=tick_size)
            
            ax.tick_params(axis='y', labelsize=tick_size)
            ax.minorticks_off()
            ax.grid(True, alpha=0.3)

            # Show x-axis label only for last row
            if i == n_rows - 1:
                ax.set_xlabel("Intervention Levels", fontsize=label_size, labelpad=axis_label_pad)

            # Show dataset name only in top row
            if i == 0:
                ax.set_title(dataset, fontsize=title_size, pad=title_pad)
            else:
                ax.set_title("")

            if j==0:
                if plot_name == 'level_interventions_on_y':
                    ax.set_ylabel("$\\Delta$ on $y$", fontsize=label_size, labelpad=axis_label_pad)
                elif plot_name == 'level_interventions_on_c':
                    ax.set_ylabel("$\\Delta$ on $c$", fontsize=label_size, labelpad=axis_label_pad)

        # Get average vertical position of current row
        row_axes = [axes[i * n_cols + j] for j in range(n_cols)]
        bbox = [ax.get_position() for ax in row_axes]
        y_middle = np.mean([b.y0 + b.height / 2 for b in bbox])

        # Dynamically determine a good x-position based on left-most subplot
        leftmost_ax = row_axes[0].get_position()
        x_pos = leftmost_ax.x0 - 0.12  # Decrease this to get closer (0.02–0.03 usually works well)

        # Rename the learning methods
        learning_method = rename_learning_methods([learning_method])[0]

        fig.text(
            x_pos,
            y_middle,
            learning_method,
            va='center',
            ha='right',
            rotation='vertical',
            fontsize=title_size
        )

    # Shared legend with background
    handles, labels = zip(*handles_labels)
    legend = fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(model_styles),
        bbox_to_anchor=(0.5, -0.08),
        fontsize=legend_size,
        frameon=True
    )
    legend.get_frame().set_facecolor(legend_bgcolor)
    legend.get_frame().set_edgecolor(legend_edgecolor)
    legend.get_frame().set_alpha(legend_alpha)

    # Make space for row labels and legend
    plt.tight_layout(rect=[0.1, 0.12, 0.98, 0.98], pad=1.4, h_pad=1.2, w_pad=1.0)

    if folder:
        if clients_flag:
            plt.savefig(f"{folder}/{plot_name}_clients.pdf", bbox_inches='tight')
        else:
            plt.savefig(f"{folder}/{plot_name}.pdf", bbox_inches='tight')
    else:
        raise ValueError("Folder path is required to save the figure.")

def delta_level_interventions_on_y(d):
    baseline = d['level 0']
    delta_dict = {int(k.split(' ')[-1]):(v - baseline) for k, v in d.items()}
    # Remove the baseline from the delta_dict
    delta_dict.pop(0, None)
    return delta_dict

def delta_level_interventions_on_c(d, dataset=None, c_info=None):
    # formal dictionary in key=(level, concept): value=accuracy
    dict = {(int(k.split('/')[0].replace('level ','')), k.split('/')[1].replace('child ','')):v for k,v in d.items()}

    # list of levels
    levels = list(set([k[0] for k in dict.keys()]))
    # list of concepts
    concepts = list(set([k[1] for k in dict.keys()]))
    
    # get the baseline for each concept
    baselines = {k[1]:v for k,v in dict.items() if k[0] == 0}

    # compute the delta for each level and concept
    delta_dict = {}
    for level in levels:
        for concept in concepts:
            if (level, concept) in dict:
                delta_dict[(level, concept)] = dict[(level, concept)] - baselines.get(concept, 0)

    # Average over the level
    avg_dict = {}
    intervened_concepts = 0
    for level in levels+[len(levels)]:
        values = [v for (l, c), v in dict.items() if l == level]
        # add to the values the as many 1 as the number of concepts that are missing at that level with respect to the previous one
        if level > 0:
            intervened_concepts += len([v for (l, c), v in dict.items() if l == level-1]) - len(values)
            values += [1.0] * (intervened_concepts)
        if values:
            avg_dict[level] = np.mean(values)

    return avg_dict

def plot_level_interventions(
    input, 
    custom_order,
    model_styles,
    folder=None,
    c_info=None,
    id=False,
    figsize=(15, 10),
    title_size=16,
    label_size=14,
    tick_size=14,
    legend_size=14,
    legend_bgcolor='lightgray',
    legend_edgecolor='black',
    legend_alpha=0.3,
):
    
    # Count the number of different clients
    n_clients = len(input[input['learning'].str.startswith('localized')]['learning'].unique())

    ## Level interventions on y ##

    # Aggregate localized over the different clients
    # Rename all localized_idx to localized
    input_global = input.copy()
    input_global['learning'] = input['learning'].apply(lambda x: 'localized' if x.startswith('localized') else x)
    input_global = input_global[['seed', 'dataset', 'model', 'learning', 'level_interventions_on_y']].dropna()
    input_global['level_interventions_on_y'] = input_global['level_interventions_on_y'].apply(delta_level_interventions_on_y)

    level_interventions_plot(
        input_global,
        custom_order,
        model_styles,
        folder=folder,
        figsize=figsize,
        title_size=title_size,
        label_size=label_size,
        tick_size=tick_size,
        legend_size=legend_size,
        legend_bgcolor=legend_bgcolor,
        legend_edgecolor=legend_edgecolor,
        legend_alpha=legend_alpha,
        plot_name="level_interventions_on_y"
    )

    ## Level interventions on c ##
    input_global = input.copy()
    input_global['learning'] = input['learning'].apply(lambda x: 'localized' if x.startswith('localized') else x)
    input_global = input_global[['seed', 'dataset', 'model', 'learning', 'level_interventions_on_c']].dropna()
    # Average the accuracy of the concepts for each level
    input_global['level_interventions_on_c'] = input_global.apply(
        lambda row: delta_level_interventions_on_c(row['level_interventions_on_c'], row['dataset'], c_info), 
        axis=1
    )

    level_interventions_plot(
        input_global,
        custom_order,
        model_styles,
        folder=folder,
        figsize=figsize,
        title_size=title_size,
        label_size=label_size,
        tick_size=tick_size,
        legend_size=legend_size,
        legend_bgcolor=legend_bgcolor,
        legend_edgecolor=legend_edgecolor,
        legend_alpha=legend_alpha,
        plot_name="level_interventions_on_c"
    )

    # NON SO SE ABBIA SENSO
    ## Level interventions considering effects only on OOD concepts##

    # # Get OOD for each client
    # # filter only the clients
    # input_clients = input[input['learning'].str.startswith('localized')]
    # # group by dataset, learning and take the first concept_acc of each group
    # ood_concepts = {}
    # for (dataset, learning), group in input_clients.groupby(['dataset', 'learning']):
    #     d = group.iloc[0]['concept_acc']
    #     d = [k for k, v in d.items() if np.isnan(v)]
    #     # rename the learning as federated_idx
    #     learning_name = learning.replace('localized', 'federated')
    #     ood_concepts[(dataset, learning_name)] = d

    # Label level interventions
    


def compute_statistics(
       performance,
       eliminate=False
):
    
    # Filter data for 'task' and 'concept'
    task_df = performance.rename(columns={'task_acc': 'accuracy'})
    concept_df = performance.rename(columns={'agg_concept': 'accuracy'})

    # Compute mean and std for 'task'
    task_stats = task_df.groupby(['model', 'dataset', 'learning']).agg(
        avg_accuracy_task=('accuracy', 'mean'),
        std_accuracy_task=('accuracy', 'std'),
        total_occurrences=('accuracy', 'count'),
        concept_left_out=('concept_left_out', 'mean')
    ).reset_index().fillna(0)

    # compute confidence intervals
    task_stats['ci_task'] = 1.96 * task_stats['std_accuracy_task'] / np.sqrt(task_stats['total_occurrences']) # 1.96 *

    # Compute mean and std for 'concept'
    concept_stats = concept_df.groupby(['model', 'dataset', 'learning']).agg(
        avg_accuracy_concept=('accuracy', 'mean'),
        std_accuracy_concept=('accuracy', 'std'),
        total_occurrences=('accuracy', 'count'),
        concept_left_out=('concept_left_out', 'mean')
    ).reset_index().fillna(0)

    # compute confidence intervals
    concept_stats['ci_concept'] = 1.96*concept_stats['std_accuracy_concept'] / np.sqrt(concept_stats['total_occurrences'])

    # Aggregated concepts and task performance
    label_stats = concept_df.groupby(['model', 'dataset', 'learning']).agg(
        avg_accuracy_label=('agg_label', 'mean'),
        std_accuracy_label=('agg_label', 'std'),
        total_occurrences=('agg_label', 'count'),
        concept_left_out=('concept_left_out', 'mean')
    ).reset_index().fillna(0)

    # compute confidence intervals
    label_stats['ci_label'] = 1.96*label_stats['std_accuracy_label'] / np.sqrt(label_stats['total_occurrences']) # 1.96 *

    return task_stats, concept_stats, label_stats    


def produce_accuracy_tables(performance):

    # Apply the following averaging only to centralied and federated, eliminate local results
    performance_centr_fed = performance[performance['learning'].isin(['centralized', 'local_federated'])]

    # Centralized and federated only
    task_stats, concept_stats, label_stats = compute_statistics(
        performance=performance_centr_fed,
    )

    # Localized only
    performance_local = performance[performance['learning'].str.startswith('localized')]
    # all localized are renamed to 'localized'
    performance_local['learning'] = 'localized'

    # Centralized and federated only
    task_stats_local, concept_stats_local, label_stats_local = compute_statistics(
        performance=performance_local,
    )

    # concatenate the localized stats to the previous ones
    task_stats = pd.concat([task_stats, task_stats_local])
    concept_stats = pd.concat([concept_stats, concept_stats_local])
    label_stats = pd.concat([label_stats, label_stats_local])

    return task_stats, concept_stats, label_stats



def tabular_task_and_concept_accuracy( 
        complete_task_stats, 
        complete_concept_stats, 
        complete_label_stats,
        custom_order, 
        model_styles, 
        visualization_folder,
        label,
    ):

    for learning in complete_task_stats['learning'].unique():

        # Filter the task and concept stats to maintain only the rows with the current learning method
        task_stats = complete_task_stats[complete_task_stats['learning'] == learning]
        concept_stats = complete_concept_stats[complete_concept_stats['learning'] == learning]
        label_stats = complete_label_stats[complete_label_stats['learning'] == learning]

        print(f'\n\nLearning method: {learning}')

        if label == 'task':
            ########## Task Accuracy Table ##########

            task_avg = task_stats[['model', 'dataset', 'avg_accuracy_task']]
            task_std = task_stats[['model', 'dataset', 'ci_task']]

            # Merge task_avg and task_std dataframes on 'model' and 'dataset'
            merged_task = pd.merge(task_avg, task_std, on=['model', 'dataset'])

            # Create a pivot table with the desired format
            pivot_table_avg = task_avg.pivot(index='model', columns='dataset', values=['avg_accuracy_task'])
            pivot_table_avg.columns = pivot_table_avg.columns.get_level_values(1)
            pivot_table_std = task_std.pivot(index='model', columns='dataset', values=['ci_task'])
            pivot_table_std.columns = pivot_table_std.columns.get_level_values(1)

            final_table = pd.DataFrame()
            for i, row in pivot_table_avg.iterrows():
                d={}
                for j in pivot_table_std.columns:
                    acc = row[j]*100
                    std = pivot_table_std.loc[i, j]*100
                    d[j] = f"{acc:.2f} ± {std:.2f}"
                # add a column to the final_table dataframe called row.name which contains d
                final_table = pd.concat([final_table, pd.DataFrame(d, index=[row.name])], axis=0)
                
            # Reindex the columns of final_table according to the custom order
            final_table = final_table.reindex(columns=custom_order)

            # Replace the model and dataset names
            final_table.index = final_table.index.map(lambda x: model_styles[x]['name'] if x in model_styles else x)

            print('\n\nTask Accuracy Table:')
            print('-------------------')
            print(final_table)

            # store the table in a csv file
            result_file = f'{visualization_folder}/{learning}/task_accuracy.csv'
            if not os.path.exists(os.path.dirname(result_file)):
                os.makedirs(os.path.dirname(result_file))
            final_table.to_csv(result_file, index=True)

        elif label == 'concepts':
            ########## Concept Accuracy Table ##########

            task_avg = concept_stats[['model', 'dataset', 'avg_accuracy_concept']]
            task_std = concept_stats[['model', 'dataset', 'ci_concept']]

            # Merge task_avg and task_std dataframes on 'model' and 'dataset'
            merged_task = pd.merge(task_avg, task_std, on=['model', 'dataset'])

            # Create a pivot table with the desired format
            pivot_table_avg = task_avg.pivot(index='model', columns='dataset', values=['avg_accuracy_concept'])
            pivot_table_avg.columns = pivot_table_avg.columns.get_level_values(1)
            pivot_table_std = task_std.pivot(index='model', columns='dataset', values=['ci_concept'])
            pivot_table_std.columns = pivot_table_std.columns.get_level_values(1)

            final_table = pd.DataFrame()
            for i, row in pivot_table_avg.iterrows():
                d={}
                for j in pivot_table_std.columns:
                    acc = row[j]*100
                    std = pivot_table_std.loc[i, j]*100
                    d[j] = f"{acc:.2f} ± {std:.2f}"
                # add a column to the final_table dataframe called row.name which contains d
                final_table = pd.concat([final_table, pd.DataFrame(d, index=[row.name])], axis=0)
                
            # Reindex the columns of final_table according to the custom order
            final_table = final_table.reindex(columns=custom_order)

            # Replace the model and dataset names
            final_table.index = final_table.index.map(lambda x: model_styles[x]['name'] if x in model_styles else x)

            print('\n\nConcept Accuracy Table:')
            print('-------------------')
            print(final_table)

            # store the table in a csv file
            result_file = f'{visualization_folder}/{learning}/concept_accuracy.csv'
            if not os.path.exists(os.path.dirname(result_file)):
                os.makedirs(os.path.dirname(result_file))
            final_table.to_csv(result_file, index=True)

        elif label == 'labels':
            ############ Label Accuracy Table ##########
            label_avg = label_stats[['model', 'dataset', 'avg_accuracy_label']]
            label_std = label_stats[['model', 'dataset', 'ci_label']]   

            # Merge label_avg and label_std dataframes on 'model' and 'dataset'
            merged_label = pd.merge(label_avg, label_std, on=['model', 'dataset'])
            # Create a pivot table with the desired format
            pivot_table_avg = label_avg.pivot(index='model', columns='dataset', values=['avg_accuracy_label'])
            pivot_table_avg.columns = pivot_table_avg.columns.get_level_values(1)
            pivot_table_std = label_std.pivot(index='model', columns='dataset', values=['ci_label'])
            pivot_table_std.columns = pivot_table_std.columns.get_level_values(1)

            final_table = pd.DataFrame()
            for i, row in pivot_table_avg.iterrows():
                d={}
                for j in pivot_table_std.columns:
                    acc = row[j]*100
                    std = pivot_table_std.loc[i, j]*100
                    d[j] = f"{acc:.2f} ± {std:.2f}"
                # add a column to the final_table dataframe called row.name which contains d
                final_table = pd.concat([final_table, pd.DataFrame(d, index=[row.name])], axis=0)

            # Reindex the columns of final_table according to the custom order
            final_table = final_table.reindex(columns=custom_order)

            # Replace the model and dataset names
            final_table.index = final_table.index.map(lambda x: model_styles[x]['name'] if x in model_styles else x)

            print('\n\nLabel Accuracy Table:')
            print('-------------------')
            print(final_table)

            # store the table in a csv file
            result_file = f'{visualization_folder}/{learning}/label_accuracy.csv'
            if not os.path.exists(os.path.dirname(result_file)):
                os.makedirs(os.path.dirname(result_file))
            final_table.to_csv(result_file, index=True)
    

def compute_drift_statistics(performance):
    coverage_stats = pd.DataFrame()
    param_change_stats = pd.DataFrame()

    if 'concept_coverage' in performance.columns:
        coverage_df = performance[['model', 'dataset', 'learning', 'concept_coverage']].dropna(subset=['concept_coverage'])
        if not coverage_df.empty:
            coverage_stats = coverage_df.groupby(['model', 'dataset', 'learning']).agg(
                avg_coverage=('concept_coverage', 'mean'),
                std_coverage=('concept_coverage', 'std'),
                total_occurrences=('concept_coverage', 'count')
            ).reset_index().fillna(0)
            coverage_stats['ci_coverage'] = 1.96*coverage_stats['std_coverage'] / np.sqrt(coverage_stats['total_occurrences'])

    if 'percent_params_changed' in performance.columns:
        params_df = performance[['model', 'dataset', 'learning', 'percent_params_changed']].dropna(subset=['percent_params_changed'])
        if not params_df.empty:
            param_change_stats = params_df.groupby(['model', 'dataset', 'learning']).agg(
                avg_param_change=('percent_params_changed', 'mean'),
                std_param_change=('percent_params_changed', 'std'),
                total_occurrences=('percent_params_changed', 'count')
            ).reset_index().fillna(0)
            param_change_stats['ci_param_change'] = 1.96* param_change_stats['std_param_change'] / np.sqrt(param_change_stats['total_occurrences'])

    return coverage_stats, param_change_stats


def tabular_drift_metrics(
        coverage_stats,
        param_change_stats,
        custom_order,
        model_styles,
        visualization_folder,
    ):

    learnings = set()
    if not coverage_stats.empty:
        learnings.update(coverage_stats['learning'].unique())
    if not param_change_stats.empty:
        learnings.update(param_change_stats['learning'].unique())

    for learning in learnings:
        if not coverage_stats.empty:
            cov_subset = coverage_stats[coverage_stats['learning'] == learning]
            if not cov_subset.empty:
                cov_avg = cov_subset[['model', 'dataset', 'avg_coverage']]
                cov_ci = cov_subset[['model', 'dataset', 'ci_coverage']]

                pivot_avg = cov_avg.pivot(index='model', columns='dataset', values='avg_coverage')
                pivot_ci = cov_ci.pivot(index='model', columns='dataset', values='ci_coverage')

                final_table = pd.DataFrame()
                for idx, row in pivot_avg.iterrows():
                    row_dict = {}
                    for dataset in pivot_ci.columns:
                        acc = row.get(dataset, np.nan) * 100
                        ci = pivot_ci.loc[idx, dataset] * 100 if dataset in pivot_ci.columns else np.nan
                        row_dict[dataset] = f"{acc:.2f} ± {ci:.2f}" if not np.isnan(acc) else "N/A"
                    final_table = pd.concat([final_table, pd.DataFrame(row_dict, index=[idx])], axis=0)

                final_table = final_table.reindex(columns=custom_order)
                final_table.index = final_table.index.map(lambda x: model_styles[x]['name'] if x in model_styles else x)

                print('\n\nConcept Coverage Table:')
                print('----------------------')
                print(final_table)

                result_file = f'{visualization_folder}/{learning}/concept_coverage.csv'
                if not os.path.exists(os.path.dirname(result_file)):
                    os.makedirs(os.path.dirname(result_file))
                final_table.to_csv(result_file, index=True)

        if not param_change_stats.empty:
            param_subset = param_change_stats[param_change_stats['learning'] == learning]
            if not param_subset.empty:
                param_avg = param_subset[['model', 'dataset', 'avg_param_change']]
                param_ci = param_subset[['model', 'dataset', 'ci_param_change']]

                pivot_avg = param_avg.pivot(index='model', columns='dataset', values='avg_param_change')
                pivot_ci = param_ci.pivot(index='model', columns='dataset', values='ci_param_change')

                final_table = pd.DataFrame()
                for idx, row in pivot_avg.iterrows():
                    row_dict = {}
                    for dataset in pivot_ci.columns:
                        acc = row.get(dataset, np.nan) * 100
                        ci = pivot_ci.loc[idx, dataset] * 100 if dataset in pivot_ci.columns else np.nan
                        row_dict[dataset] = f"{acc:.2f} ± {ci:.2f}" if not np.isnan(acc) else "N/A"
                    final_table = pd.concat([final_table, pd.DataFrame(row_dict, index=[idx])], axis=0)

                final_table = final_table.reindex(columns=custom_order)
                final_table.index = final_table.index.map(lambda x: model_styles[x]['name'] if x in model_styles else x)

                print('\n\n% Parameters Changed Table:')
                print('--------------------------')
                print(final_table)

                result_file = f'{visualization_folder}/{learning}/percent_params_changed.csv'
                if not os.path.exists(os.path.dirname(result_file)):
                    os.makedirs(os.path.dirname(result_file))
                final_table.to_csv(result_file, index=True)


def tabular_graph_metrics(
        performance,
        custom_order,
        model_styles,
        visualization_folder,
    ):
    stage_specs = {
        "graph_predrift": "Aggregated pre-drift graph",
        "graph_postdrift": "Aggregated post-drift graph",
    }
    metric_specs = [
        ("hamming_cost", "Hamming cost", "{:.3f}", "{:.3f}"),
        ("avg_cost", "Avg cost", "{:.3f}", "{:.3f}"),
        ("differing_pairs", "Differing pairs", "{:.1f}", "{:.1f}"),
    ]
    preferred_model_order = [model_styles[k]['name'] for k in model_styles]

    for learning in performance['learning'].unique():
        perf_learn = performance[performance['learning'] == learning]

        for stage_key, stage_label in stage_specs.items():
            metric_columns = [f"{stage_key}_{m[0]}" for m in metric_specs]
            available_cols = [c for c in metric_columns if c in perf_learn.columns]
            if not available_cols:
                continue

            subset = perf_learn[['model', 'dataset'] + available_cols]
            subset = subset.dropna(subset=available_cols, how='all')
            if subset.empty:
                continue

            agg = subset.groupby(['model', 'dataset'])[available_cols].agg(['mean', 'std'])

            models = list(agg.index.get_level_values('model').unique())
            datasets_present = list(agg.index.get_level_values('dataset').unique())
            columns = list(custom_order)
            for ds in datasets_present:
                if ds not in columns:
                    columns.append(ds)

            final_table = pd.DataFrame(index=models, columns=columns)
            for model in final_table.index:
                for dataset in final_table.columns:
                    cell = "N/A"
                    if (model, dataset) in agg.index:
                        row = agg.loc[(model, dataset)]
                        parts = []
                        for metric_name, metric_label, mean_fmt, std_fmt in metric_specs:
                            col = f"{stage_key}_{metric_name}"
                            if (col, 'mean') not in agg.columns:
                                continue
                            mean_val = row.get((col, 'mean'), np.nan)
                            std_val = row.get((col, 'std'), np.nan)
                            if np.isnan(mean_val):
                                continue
                            std_val = 0.0 if np.isnan(std_val) else std_val
                            parts.append(f"{metric_label}: {mean_fmt.format(mean_val)} ± {std_fmt.format(std_val)}")
                        if parts:
                            cell = " | ".join(parts)
                    final_table.loc[model, dataset] = cell

            existing_models = [m for m in preferred_model_order if m in final_table.index]
            remaining_models = [m for m in final_table.index if m not in existing_models]
            final_table = final_table.loc[existing_models + remaining_models]

            print(f"\n\nGraph Metrics Table — Learning: {learning} — {stage_label}")
            print('------------------------------------------------------------')
            print(final_table)

            result_file = f"{visualization_folder}/{learning}/graph_metrics_{stage_key}.csv"
            if not os.path.exists(os.path.dirname(result_file)):
                os.makedirs(os.path.dirname(result_file))
            final_table.to_csv(result_file, index=True)

    
#############################################
############ PRIVACY ATTACKS ################
#############################################

def privacy_attacks(
       performance,
       model_styles,
       custom_order,
       visualization_folder
):    
    for learning in performance['learning'].unique():

        # Keep only required columns if present
        privacy_cols = [
            'model', 'dataset', 'learning',
            'sia_acc',
            'mia_whitebox_acc', 'mia_blackbox_acc', 'mia_shadow_acc',
            'dra_dlg_mse_mean',
            'dra_idlg_mse_mean',
        ]
        available_cols = [c for c in privacy_cols if c in performance.columns]
        privacy_df = performance[available_cols].copy()

        # Drop rows where everything is NaN for privacy metrics
        metric_cols = [c for c in available_cols if c not in ['model','dataset','learning']]
        if metric_cols:
            privacy_df = privacy_df.dropna(subset=metric_cols, how='all')

        if not privacy_df.empty and metric_cols:
            # Aggregate across folds (runs) → mean and std per (model, dataset, learning)
            mean_agg = {c: 'mean' for c in metric_cols}
            std_agg  = {c: 'std'  for c in metric_cols}
            grp = ['model', 'dataset', 'learning']
            privacy_mean = privacy_df.groupby(grp, as_index=False).agg(mean_agg)
            privacy_std  = privacy_df.groupby(grp, as_index=False).agg(std_agg).fillna(0)

            # For each learning method and dataset, create a table:
            attack_columns = [
                ('SIA (acc)', 'sia_acc', 'sia_acc'),
                ('MIA WB (acc)', 'mia_whitebox_acc', 'mia_whitebox_acc'),
                ('MIA BB (acc)', 'mia_blackbox_acc', 'mia_blackbox_acc'),
                ('MIA Shadow (acc)', 'mia_shadow_acc', 'mia_shadow_acc'),
                ('DRA DLG (MSE)', 'dra_dlg_mse_mean', 'dra_dlg_mse_mean'),
                ('DRA iDLG (MSE)', 'dra_idlg_mse_mean', 'dra_idlg_mse_mean'),
            ]

            # Which metrics are accuracies
            acc_metrics = {'sia_acc', 'mia_whitebox_acc', 'mia_blackbox_acc', 'mia_shadow_acc'}

            # Helper to format cells as mean ± std
            def _fmt_cell(mean_val, std_val, is_accuracy=False):
                if np.isnan(mean_val):
                    return "N/A"
                if is_accuracy:
                    m = mean_val * 100.0
                    s = 0.0 if (std_val is None or np.isnan(std_val)) else std_val * 100.0
                    return f"{m:.2f} ± {s:.2f}"
                else:
                    s = 0.0 if (std_val is None or np.isnan(std_val)) else std_val
                    return f"{mean_val:.3f} ± {s:.3f}"

            # Order models by model_styles if possible (after earlier rename to pretty names)
            preferred_model_order = [model_styles[k]['name'] for k in model_styles]

            for learning in sorted(privacy_mean['learning'].unique()):
                for dataset_name in custom_order:
                    sub_m = privacy_mean[(privacy_mean['learning'] == learning) & (privacy_mean['dataset'] == dataset_name)]
                    if sub_m.empty:
                        continue
                    sub_s = privacy_std[(privacy_std['learning'] == learning) & (privacy_std['dataset'] == dataset_name)]

                    # index by model
                    sub_m = sub_m.set_index('model')
                    sub_s = sub_s.set_index('model')

                    # Build the printable table with formatted strings
                    table_rows = {}
                    for model_name in sub_m.index.unique():
                        row = {}
                        for col_title, mean_key, std_key in attack_columns:
                            is_acc = mean_key in acc_metrics
                            mean_val = sub_m.loc[model_name, mean_key] if mean_key in sub_m.columns else np.nan
                            std_val  = sub_s.loc[model_name, std_key] if std_key in sub_s.columns else np.nan
                            # Handle duplicated index (multiple rows) by taking mean again
                            if isinstance(mean_val, pd.Series):
                                mean_val = float(mean_val.mean())
                            if isinstance(std_val, pd.Series):
                                std_val = float(std_val.mean())
                            row[col_title] = _fmt_cell(mean_val, std_val, is_accuracy=is_acc)
                        table_rows[model_name] = row

                    privacy_table = pd.DataFrame.from_dict(table_rows, orient='index')
                    # Reindex rows by preferred model order if names match
                    existing_models = [m for m in preferred_model_order if m in privacy_table.index]
                    remaining = [m for m in privacy_table.index if m not in existing_models]
                    privacy_table = privacy_table.reindex(existing_models + remaining)

                    print(f"\n\nPrivacy Attacks Table — Learning: {learning} — Dataset: {dataset_name}")
                    print('------------------------------------------------------------')
                    print(privacy_table)

                    # Save to CSV
                    result_file = f"{visualization_folder}/{learning}/privacy_{dataset_name}.csv"
                    if not os.path.exists(os.path.dirname(result_file)):
                        os.makedirs(os.path.dirname(result_file))
                    privacy_table.to_csv(result_file, index=True)
        else:
            print("\n[INFO] No privacy metrics found to tabulate.")




########### FEDERATED PLOTS ######################

# ---------- DRA: compute common kept indices across experiments ----------
def _is_finite_number(x):
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False

def _derive_excluded_from_raw(raw_json_path, methods=('DLG', 'iDLG'), metrics=('mse','loss')):
    """
    Fallback: compute excluded indices by scanning the raw per-sample results,
    excluding any sample with non-finite metric in ANY requested method/metric.
    Returns (excluded_indices_set, n_total_aligned)
    """
    if not os.path.exists(raw_json_path):
        return set(), 0
    try:
        with open(raw_json_path, 'r') as f:
            data = json.load(f)
        # ensure lists for methods; skip missing methods
        method_lists = [data[m] for m in methods if m in data and isinstance(data[m], list)]
        if not method_lists:
            return set(), 0
        n_total = min(len(lst) for lst in method_lists)
        excluded = set()
        for i in range(n_total):
            ok = True
            for m in methods:
                lst = data.get(m, [])
                if i >= len(lst) or not isinstance(lst[i], dict):
                    ok = False
                    break
                for met in metrics:
                    if not _is_finite_number(lst[i].get(met, None)):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                excluded.add(i)
        return excluded, n_total
    except Exception:
        return set(), 0

def compute_common_kept_indices(experiment_paths, methods=('DLG','iDLG'), metrics=('mse','loss')):
    """
    From all 'dra_results_summary.json' (or raw files if summary missing), take the UNION
    of excluded indices, restricted to the minimum aligned length across experiments.
    Return the sorted list of kept indices common to all experiments.
    """
    union_excluded = set()
    min_n_total = None

    for exp in experiment_paths:
        sum_path = os.path.join(exp, 'dra_results_summary.json')
        if os.path.exists(sum_path):
            try:
                with open(sum_path, 'r') as f:
                    summ = json.load(f)
                n_total = int(summ.get('counts', {}).get('n_total_aligned', 0))
                excl = set(int(i) for i in summ.get('counts', {}).get('excluded_indices', []))
                min_n_total = n_total if (min_n_total is None) else min(min_n_total, n_total)
                union_excluded |= excl
                continue
            except Exception:
                pass

        # Fallback to raw client results if summary is missing/broken
        raw_path = os.path.join(exp, 'dra_results_client_0.json')
        if not os.path.exists(raw_path):
            raw_path = os.path.join(exp, 'dra_results.json')
        excl, n_total = _derive_excluded_from_raw(raw_path, methods=methods, metrics=metrics)
        if n_total > 0:
            min_n_total = n_total if (min_n_total is None) else min(min_n_total, n_total)
            union_excluded |= excl

    if min_n_total is None:
        # Nothing found; return empty (=no common indices)
        return []

    # Restrict union to valid range [0, min_n_total)
    union_excluded = {i for i in union_excluded if 0 <= i < min_n_total}
    kept = sorted(set(range(min_n_total)) - union_excluded)
    print(f"[DRA] Common kept indices across experiments: {len(kept)} / {min_n_total} (excluded union size={len(union_excluded)})")
    return kept

def _dra_mean_over_indices(raw_json_path, method, metric, indices):
    """
    Compute mean of `metric` for `method` using only `indices`.
    Returns np.nan if unavailable.
    """
    if not os.path.exists(raw_json_path) or not indices:
        return np.nan
    try:
        with open(raw_json_path, 'r') as f:
            data = json.load(f)
        lst = data.get(method, [])
        vals = []
        for i in indices:
            if i < 0 or i >= len(lst) or not isinstance(lst[i], dict):
                continue
            v = lst[i].get(metric, None)
            if _is_finite_number(v):
                vals.append(float(v))
        if not vals:
            return np.nan
        return float(np.mean(vals))
    except Exception:
        return np.nan



####################################################
################# SETUP STUFF ######################
####################################################

def setup_results(paths, visualization_folder):
    # if folder doesn't exist, create it
    if not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)

    # Load the results folder form the paths
    exps_path = []
    lmr_paths = []
    for path in paths:
        exps = os.listdir(path)
        exps_path += [os.path.join(path, exp) for exp in exps if 'multirun' not in exp]
    return exps_path

def load_exps(exps_path, n_clients=5, args=None):
    performance = []

    # Precompute once for this run
    common_kept_indices = compute_common_kept_indices(exps_path, methods=('DLG','iDLG'), metrics=('mse','loss'))
    valid_concepts = []

    # search if there is 'c2bm' in the config files of the run, if it is save the name of the concepts
    valid_concepts = {}
    missing_concept_exps = set()

    for exp in exps_path:
        conf_file = os.path.join(exp, '.hydra/config.yaml')      
        if os.path.exists(conf_file):
            with open(conf_file, 'r') as file:
                conf = yaml.safe_load(file)
                dataset = conf['dataset']['name']
                seed = conf['seed']
                model = conf['model']['name']
                training_modality = conf['learning']['mode']
                rnd_drift = conf['learning'].get('subgraphs', {}).get('rnd_drift', 0)
                n_rounds = conf['trainer'].get('max_epochs', float('inf'))
                training_modality_full = training_modality
                if 'localized' in training_modality:
                    training_modality_full = training_modality + '_' + str(conf['client_id'])
                elif "federated" in training_modality or "local_federated" in training_modality:
                    training_modality_full = training_modality + f'_rnddrift{rnd_drift}_nrounds{n_rounds}'
                #key = dataset + '_' + str(seed) + '_' + training_modality_full
                key = dataset + '' + str(seed) + '' + training_modality_full +'_' + model
                if key not in valid_concepts.keys():
                    valid_concepts[key]= []
                if conf['model']['name'] == 'c2bm' or conf['model']['name'] == 'cgm':
                    result_file = os.path.join(exp, 'results') 
                    concept_file = os.path.join(result_file, 'c_accuracy.pkl')
                    if not os.path.exists(concept_file):
                        print(f"[load_exps] Missing concept accuracy file: {concept_file}. Skipping this experiment.")
                        missing_concept_exps.add(exp)
                        continue
                    with open(concept_file, 'rb') as file:
                        # save in valid concepts the name of the concepts in concept file
                        concept_results = pickle.load(file)
                    valid_concepts[key] = [k for k, v in concept_results.items() if _is_finite_number(v)]

                
                    

    for exp in exps_path:
        if exp in missing_concept_exps:
            continue
        d = {}
        conf_file = os.path.join(exp, '.hydra/config.yaml')
        result_file = os.path.join(exp, 'results')  
        if os.path.exists(conf_file) and os.path.exists(result_file):
            try:
                with open(conf_file, 'r') as file:
                    conf = yaml.safe_load(file)
                d['seed'] = conf['seed']
                d['dataset'] = conf['dataset']['name']
                d['model'] = conf['model']['name']

                if 'localized' in conf['learning']['mode']:
                    d['learning'] = conf['learning']['mode'] + '_' + str(conf['client_id'])
                else:
                    d['learning'] = conf['learning']['mode']

                # Extract rnd_drift information
                try:
                    if 'subgraphs' in conf['learning'] and 'rnd_drift' in conf['learning']['subgraphs']:
                        d['rnd_drift'] = conf['learning']['subgraphs']['rnd_drift']
                    else:
                        d['rnd_drift'] = 0
                except (KeyError, TypeError):
                    d['rnd_drift'] = 0

                # Extract n_rounds information (corresponds to max_epochs)
                try:
                    if 'max_epochs' in conf['trainer']:
                        d['n_rounds'] = int(conf['trainer']['max_epochs'])
                    else:
                        d['n_rounds'] = float('inf')  # Default to infinity if not found
                except (KeyError, TypeError):
                    d['n_rounds'] = float('inf')

                # Build the key to match valid_concepts (same logic as above)
                training_modality = conf['learning']['mode']
                training_modality_full = training_modality
                if 'localized' in training_modality:
                    training_modality_full = training_modality + '_' + str(conf['client_id'])
                elif "federated" in training_modality or "local_federated" in training_modality:
                    training_modality_full = training_modality + f'_rnddrift{d["rnd_drift"]}_nrounds{d["n_rounds"]}'
                #key = d['dataset'] + '_' + str(d['seed']) + '_' + training_modality_full
                key = d['dataset'] + '' + str(d['seed']) + '' + training_modality_full +'_' + d['model']

                # Initialize graph similarity metrics
                d['graph_predrift_hamming_cost'] = np.nan
                d['graph_predrift_avg_cost'] = np.nan
                d['graph_predrift_differing_pairs'] = np.nan
                d['graph_postdrift_hamming_cost'] = np.nan
                d['graph_postdrift_avg_cost'] = np.nan
                d['graph_postdrift_differing_pairs'] = np.nan

                # Concept results
                concept_file = os.path.join(result_file, 'c_accuracy.pkl')
                if not os.path.exists(concept_file):
                    print(f"[load_exps] Missing concept accuracy file: {concept_file}. Skipping this experiment.")
                    missing_concept_exps.add(exp)
                    continue
                with open(concept_file, 'rb') as file:
                    concept_results = pickle.load(file)

                # Select the last row of the dataframe where we test the model
                if len(valid_concepts.get(key, [])) > 0:
                    # filter only the valid concepts
                    concept_results = {k:v for k,v in concept_results.items() if k in valid_concepts[key]}
                d['concept_acc'] = concept_results

                # Task results
                task_file = os.path.join(result_file, 'y_accuracy.pkl')
                with open(task_file, 'rb') as file:
                    task_results = pickle.load(file)

                d['task_acc'] = task_results['_baseline']

                # Additional drift metrics (optional)
                d['concept_coverage'] = np.nan
                d['percent_params_changed'] = np.nan
                additional_metrics_path = os.path.join(result_file, "additional_metrics.json")
                if os.path.exists(additional_metrics_path):
                    try:
                        with open(additional_metrics_path, "r") as f:
                            additional_metrics = json.load(f)
                        d['concept_coverage'] = float(additional_metrics.get("concept_coverage", np.nan))
                        d['percent_params_changed'] = float(additional_metrics.get("percent_params_changed", np.nan))
                    except Exception:
                        pass

                try:
                    # Collect graph
                    graph_file = os.path.join(exp, 'graph.pkl')
                    with open(graph_file, 'rb') as file:
                        graph_results = pickle.load(file)
                    d['graph'] = graph_results['concepts']
                    d['predicted_concepts'] = graph_results.get('predicted_concepts', None)
                    # Extract true_graph_columns from graph.pkl if available
                    d['true_graph'] = graph_results.get('centralized_topological_order', None)
                except FileNotFoundError:
                    d['graph'] = None
                    d['true_graph'] = None

                # Graph similarity metrics (predrift/postdrift)
                graph_metrics_path = os.path.join(result_file, "graph_metrics.json")
                if os.path.exists(graph_metrics_path):
                    try:
                        with open(graph_metrics_path, "r") as f:
                            graph_metrics = json.load(f)

                        stage_map = {
                            "graph_predrift": "graph_predrift",
                            "aggregated_pre_drift_graph": "graph_predrift",
                            "pre_drift": "graph_predrift",
                            "predrift": "graph_predrift",
                            "graph_postdrift": "graph_postdrift",
                            "aggregated_post_drift_graph": "graph_postdrift",
                            "post_drift": "graph_postdrift",
                            "postdrift": "graph_postdrift",
                        }

                        def _assign_metrics(prefix, metrics_dict):
                            if not isinstance(metrics_dict, dict):
                                return
                            d[f"{prefix}_hamming_cost"] = float(metrics_dict.get("hamming_cost", np.nan))
                            d[f"{prefix}_avg_cost"] = float(metrics_dict.get("avg_cost", np.nan))
                            d[f"{prefix}_differing_pairs"] = float(metrics_dict.get("differing_pairs", np.nan))

                        for raw_key, metrics_dict in graph_metrics.items():
                            stage_key = stage_map.get(str(raw_key).lower())
                            if stage_key:
                                _assign_metrics(stage_key, metrics_dict)
                    except Exception:
                        pass

                ###### Collect the results for the interventions
                single_id_on_y_file =  os.path.join(exp, 'results', 'single_IDc_interventions_on_y.pkl')
                #[os.path.join(exp, 'results', f'client_{client}_single_IDc_interventions_on_y.pkl') \
                #                        for client in range(1,n_clients)] # The maximum number of clients has to be known
                single_ood_on_y_file = os.path.join(exp, 'results', 'single_OODc_interventions_on_y.pkl')
                #single_ood_on_y_files = [os.path.join(exp, 'results', f'client_{client}_single_OODc_interventions_on_y.pkl') \
                #                        for client in range(1,n_clients)] # The maximum number of clients has to be known

                level_interventions_on_c_file = os.path.join(exp, 'results', 'level_interventions_on_c.pkl')

                level_interventions_on_y_file = os.path.join(exp, 'results', 'level_interventions_on_y.pkl')

                single_c_interventions_on_y_file = os.path.join(exp, 'results', 'single_c_interventions_on_y.pkl')

                # now read all those files 
                d['single_id_on_y'] = []
                d['single_ood_on_y'] = []
                #for client in range(1, n_clients):
                if os.path.exists(single_id_on_y_file):
                    with open(single_id_on_y_file, 'rb') as f:
                        data = pickle.load(f)
                        for client in range(1, n_clients+1):
                            # filter only the concepts in valid concepts
                            if len(valid_concepts[key]) > 0:
                                data[client] = {k:v for k,v in data[client].items() if k in valid_concepts[key] or k == "_baseline"}
                        d['single_id_on_y'] = data
                           

                if os.path.exists(single_ood_on_y_file):
                    with open(single_ood_on_y_file, 'rb') as f:
                        data = pickle.load(f)
                        for client in range(1, n_clients+1):
                            # filter only the concepts in valid concepts
                            if len(valid_concepts[key]) > 0:
                                data[client] = {k:v for k,v in data[client].items() if k in valid_concepts[key] or k == "_baseline"}
                        d['single_ood_on_y'] = data

                if d['single_id_on_y'] == []:
                    d['single_id_on_y'] = None
                if d['single_ood_on_y'] == []:
                    d['single_ood_on_y'] = None

                if os.path.exists(level_interventions_on_c_file):
                    with open(level_interventions_on_c_file, 'rb') as f:
                        d['level_interventions_on_c'] = pickle.load(f)
                else:
                    d['level_interventions_on_c'] = None

                if os.path.exists(level_interventions_on_y_file):
                    with open(level_interventions_on_y_file, 'rb') as f:
                        d['level_interventions_on_y'] = pickle.load(f)
                else:
                    d['level_interventions_on_y'] = None

                if os.path.exists(single_c_interventions_on_y_file):
                    with open(single_c_interventions_on_y_file, 'rb') as f:
                        single_c_interventions_on_y = pickle.load(f)
                    # eliminate concepts not in valid concepts
                    if len(valid_concepts[key]) > 0:
                        single_c_interventions_on_y = {k:v for k,v in single_c_interventions_on_y.items() if k in valid_concepts[key] or k == "_baseline"}
                    d['single_c_interventions_on_y'] = single_c_interventions_on_y
                else:
                    d['single_c_interventions_on_y'] = None

                # Load cumulative task interventions
                cumulative_task_interventions_file = os.path.join(exp, 'results', 'cumulative_interventions_on_y.pkl')
                if os.path.exists(cumulative_task_interventions_file):
                    with open(cumulative_task_interventions_file, 'rb') as f:
                        cumulative_interventions_on_y = pickle.load(f)
                    # cumulative_data should contain 'values' and 'concept_names'
                    d['cumulative_task_interventions'] = cumulative_interventions_on_y
                    #d['concept_names'] = cumulative_interventions_on_y.get('concept_names', None)
                else:
                    d['cumulative_task_interventions'] = None
                    #d['concept_names'] = None

                # Load cumulative concept interventions
                cumulative_concept_interventions_file = os.path.join(exp, 'results', 'cumulative_interventions_on_c.pkl')
                if os.path.exists(cumulative_concept_interventions_file):
                    with open(cumulative_concept_interventions_file, 'rb') as f:
                        cumulative_interventions_on_c = pickle.load(f)
                    d['cumulative_concept_interventions'] = cumulative_interventions_on_c
                else:
                    d['cumulative_concept_interventions'] = None


                ######################################################################
                ######################################################################
                # ------------------ Privacy attacks: SIA, MIA, DRA ------------------
                # Defaults
                d['sia_acc'] = np.nan
                d['mia_whitebox_acc'] = np.nan
                d['mia_blackbox_acc'] = np.nan
                d['mia_shadow_acc'] = np.nan
                d['dra_dlg_mse_mean'] = np.nan
                d['dra_dlg_mse_ci'] = np.nan
                d['dra_idlg_mse_mean'] = np.nan
                d['dra_idlg_mse_ci'] = np.nan

                # Paths
                sia_path = os.path.join(exp, 'sia_max.json')
                mia_summary_path = os.path.join(exp, 'mia_summary.json')
                mia_max_path = os.path.join(exp, 'mia_max.json')  # optional alt name
                dra_summary_path = os.path.join(exp, 'dra_results_summary.json')

                # --- SIA (accuracy) ---
                if os.path.exists(sia_path):
                    try:
                        with open(sia_path, 'r') as f:
                            sia_json = json.load(f)
                        # expect {"max_sia_accuracy": float}
                        if 'max_sia_accuracy' in sia_json:
                            d['sia_acc'] = float(sia_json['max_sia_accuracy'])
                    except Exception:
                        pass

                # --- MIA (accuracy) ---
                def _get_mia_acc(mia_json, key):
                    """Return a single accuracy for `key` ('whitebox'|'blackbox'), preferring worst_case.max_mia_accuracy then mean_across_clients.max_mia_accuracy."""
                    try:
                        if key in mia_json:
                            if args.mia_metric == "max":
                                if 'worst_case' in mia_json[key] and 'max_mia_accuracy' in mia_json[key]['worst_case']:
                                    return float(mia_json[key]['worst_case']['max_mia_accuracy'])
                            elif args.mia_metric == "mean":
                                if 'mean_across_clients' in mia_json[key] and 'max_mia_accuracy' in mia_json[key]['mean_across_clients']:
                                    return float(mia_json[key]['mean_across_clients']['max_mia_accuracy'])
                            else:
                                raise ValueError(f"Unknown MIA metric: {args.mia_metric}")
                    except Exception:
                        return np.nan
                    return np.nan

                mia_json = None
                if os.path.exists(mia_summary_path):
                    try:
                        with open(mia_summary_path, 'r') as f:
                            mia_json = json.load(f)
                    except Exception:
                        mia_json = None
                elif os.path.exists(mia_max_path):
                    try:
                        with open(mia_max_path, 'r') as f:
                            mia_json = json.load(f)
                    except Exception:
                        mia_json = None

                if isinstance(mia_json, dict):
                    d['mia_whitebox_acc'] = _get_mia_acc(mia_json, 'whitebox')
                    d['mia_blackbox_acc'] = _get_mia_acc(mia_json, 'blackbox')
                    d['mia_shadow_acc'] = _get_mia_acc(mia_json, 'blackbox_shadow')

                # --- DRA (MSE) recomputed on COMMON kept indices across experiments ---
                raw_dra_path = os.path.join(exp, 'dra_results_client_0.json')
                if not os.path.exists(raw_dra_path):
                    raw_dra_path = os.path.join(exp, 'dra_results.json')

                d['dra_dlg_mse_mean']  = _dra_mean_over_indices(raw_dra_path, 'DLG',  'mse', common_kept_indices)
                d['dra_idlg_mse_mean'] = _dra_mean_over_indices(raw_dra_path, 'iDLG', 'mse', common_kept_indices)
                ######################################################################
                ######################################################################

                performance.append(d)

            except:
                pass

    performance = pd.DataFrame(performance)

    # Read from the cache the c_info for each dataset
    c_info = {}
    for dataset in performance['dataset'].unique():
        dataset_directory = os.path.join(CACHE, dataset)
        c_info_path = os.path.join(dataset_directory, "c_info.pkl")
        if os.path.exists(c_info_path):
            with open(c_info_path, 'rb') as f: 
                c_info_pkl = pickle.load(f)
            c_info[dataset] = c_info_pkl
        else:
            c_info[dataset] = None

    # Fill empty graph entries by matching on dataset
    for idx, row in performance.iterrows():
        if row['graph'] is None:
            # Find rows with the same dataset that have a non-None graph
            matching_rows = performance[(performance['dataset'] == row['dataset']) & 
                                    (performance['graph'].notna())]
            if not matching_rows.empty:
                # Use the first matching graph
                performance.at[idx, 'graph'] = matching_rows.iloc[0]['graph']

    def _format_results(row, graph, dataset, model, count_nan=False, task=None, worst_classifier=False):
        if model != 'blackbox':
            # Build values list based on the graph - include all concepts from graph
            # If a concept is not in row (filtered out), it's a missing concept
            values = []
            for c_name in graph:
                if c_name in row:
                    values.append(row[c_name])
                else:
                    # Concept is missing (was filtered out) - will be treated as OOD
                    values.append(np.nan)
            
            if count_nan:
                return sum([1 for x in values if math.isnan(x)]) if values else 0
            else:
                if worst_classifier:
                    # Replace missing concepts with random classifier performance
                    if dataset is not None and dataset in c_info and c_info[dataset] is not None:
                        for idx, c_name in enumerate(graph):
                            if math.isnan(values[idx]):
                                concept_cardinality_idx = c_info[dataset]['names'].index(c_name)
                                concept_cardinality = c_info[dataset]['cardinality'][concept_cardinality_idx]
                                values[idx] = 1.0/concept_cardinality
                else:
                    # eliminate nan values before computing the average
                    values = [x for x in values if not math.isnan(x)]
                if task is not None:
                    values = values + [task]
                return np.mean([x for x in values if not math.isnan(x)]) if values else 0
        else:
            return 0


    # Aggregate the results in concept_acc and task_acc for each row.
    # In the aggregation, we compute the expected value over the concepts. 
    # For missing concepts (NaN values), we predict them using a random classifier 
    # (uniform distribution based on concept cardinality: 1/cardinality).
    performance['concept_left_out'] = performance.apply(lambda x: _format_results(x['concept_acc'], x['true_graph'], x['dataset'], x['model'], count_nan=True), axis=1)
    performance['agg_concept'] = performance.apply(lambda x: _format_results(x['concept_acc'], x['true_graph'], x['dataset'], x['model'], count_nan=False, worst_classifier=False), axis=1)
    performance['agg_label'] = performance.apply(lambda x: _format_results(x['concept_acc'], x['true_graph'], x['dataset'], x['model'], count_nan=False, task=x['task_acc'], worst_classifier=False), axis=1)

    return performance, c_info

def apply_styles(performance, dataset_styles, model_styles, custom_order):

    def _get_df_name(df):
        try:
            return dataset_styles[df]['name']
        except KeyError:
            raise ValueError(f"Unknown dataset name: {df}")
    
    # Rename the datasets in the performance dataframe
    performance['dataset'] = performance['dataset'].apply(_get_df_name)

    # Filter the performance dataframe to keep only the models in model_styles 
    # and datasets in custom_order.
    performance = performance[performance['model'].isin(model_styles.keys()) & performance['dataset'].isin(custom_order)]

    # change the name of the model
    performance['model'] = performance['model'].apply(lambda x: model_styles[x]['name'] if x in model_styles else x)
    
    return performance



def _find_history_files(paths, history_filename="training_history.json"):
    history_files = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[training_history] Skipping missing path: {path}")
            continue
        if os.path.isfile(path):
            if os.path.basename(path) == history_filename:
                history_files.append(path)
            continue

        candidate = os.path.join(path, "results", history_filename)
        if os.path.isfile(candidate):
            history_files.append(candidate)
            continue

        for root, _, files in os.walk(path):
            if history_filename in files:
                history_files.append(os.path.join(root, history_filename))

    return sorted(set(history_files))


def _standardize_history(history):
    for key in ("loss_val_client", "y_acc_val_client"):
        if key not in history:
            continue
        raw = history[key]
        if isinstance(raw, dict):
            standardized = {}
            for k, v in raw.items():
                try:
                    k = int(k)
                except (TypeError, ValueError):
                    pass
                standardized[k] = v
            history[key] = standardized
        elif isinstance(raw, list):
            history[key] = {i: v for i, v in enumerate(raw)}
    return history


def _find_experiment_root(history_path, max_depth=5):
    current = os.path.dirname(history_path)
    for _ in range(max_depth):
        conf_path = os.path.join(current, ".hydra", "config.yaml")
        if os.path.isfile(conf_path):
            return current, conf_path
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None, None


def _load_histories_by_model(history_files):
    histories_by_model = {}
    dataset_names = set()
    for path in history_files:
        try:
            with open(path, "r") as fp:
                history = json.load(fp)
            history = _standardize_history(history)
            if "round" not in history:
                print(f"[training_history] Missing 'round' in {path}, skipping.")
                continue
            _, conf_path = _find_experiment_root(path)
            model_name = None
            dataset_name = None
            if conf_path is not None:
                try:
                    with open(conf_path, "r") as file:
                        conf = yaml.safe_load(file)
                    model_name = conf.get("model", {}).get("name")
                    dataset_name = conf.get("dataset", {}).get("name")
                except Exception as exc:
                    print(f"[training_history] Failed to read config for {path}: {exc}")
            if not model_name:
                print(f"[training_history] Missing model name for {path}, skipping.")
                continue
            histories_by_model.setdefault(model_name, []).append(history)
            if dataset_name:
                dataset_names.add(dataset_name)
        except Exception as exc:
            print(f"[training_history] Failed to load {path}: {exc}")
    if len(dataset_names) > 1:
        print(f"[training_history] Multiple datasets detected: {sorted(dataset_names)}")
    return histories_by_model


def _common_rounds(histories):
    round_lists = [history.get("round", []) for history in histories]
    if not round_lists:
        return []

    round_sets = [set(rounds) for rounds in round_lists]
    base_rounds = list(round_lists[0])
    common = [r for r in base_rounds if all(r in rs for rs in round_sets[1:])]
    if len(common) != len(base_rounds):
        print(f"[training_history] Using {len(common)} common rounds across histories.")
    return common


def _collect_client_ids(histories, key="loss_val_client", mode="intersection"):
    client_sets = []
    for history in histories:
        clients = history.get(key, {})
        if isinstance(clients, dict):
            client_sets.append(set(clients.keys()))
    if not client_sets:
        return []

    if mode == "union":
        union = set.union(*client_sets)
        if any(union != s for s in client_sets):
            print(f"[training_history] Client ids differ across histories, using {len(union)} union clients.")
        return sorted(union)

    common = set.intersection(*client_sets)
    if not common:
        union = set.union(*client_sets)
        print("[training_history] No common client ids found, using union.")
        return sorted(union)

    if any(common != s for s in client_sets):
        print(f"[training_history] Client ids differ across histories, using {len(common)} common clients.")
    return sorted(common)


def _aligned_series(histories, rounds, key, client_id=None):
    aligned = []
    for history in histories:
        round_to_idx = {r: i for i, r in enumerate(history.get("round", []))}
        if client_id is None:
            series = history.get(key)
        else:
            series = history.get(key, {}).get(client_id)

        if series is None:
            aligned.append([np.nan] * len(rounds))
            continue

        values = []
        for r in rounds:
            idx = round_to_idx.get(r)
            if idx is None or idx >= len(series):
                values.append(np.nan)
            else:
                values.append(series[idx])
        aligned.append(values)

    return np.asarray(aligned, dtype=float)


def _ci_multiplier(confidence, n):
    n = np.asarray(n)
    alpha = 1.0 - confidence

    if n.size == 0:
        return n.astype(float)

    try:
        from scipy.stats import t

        multipliers = t.ppf(1.0 - alpha / 2.0, df=np.maximum(n - 1, 1))
    except Exception:
        z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        multipliers = np.full_like(n, z_value, dtype=float)

    multipliers = np.where(n > 1, multipliers, 0.0)
    return multipliers


def _mean_and_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    n = np.sum(~np.isnan(values), axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    sem = np.where(n > 1, std / np.sqrt(n), 0.0)
    ci = _ci_multiplier(confidence, n) * sem
    return mean, ci, n

def plot_training_metrics_across_seeds(
    paths,
    save_dir="figs",
    history_filename="training_history.json",
    confidence=0.95,
    model_styles=None,
    font_sizes=None,
):
    """
    Aggregate training histories across seeds per model and plot mean with confidence intervals.

    Args:
        paths: List of multirun directories, experiment directories, or direct history files.
        save_dir: Directory to save plots.
        history_filename: Name of the history file saved under each experiment results folder.
        confidence: Confidence level for the interval (default: 0.95).
        model_styles: Mapping from model name to style metadata (marker, color, size, name).
        font_sizes: Dict with keys base/title/label/tick/legend to control font sizes.
    """

    marker_size = 6
    model_styles = {
        'cem': {'marker': 'P', 'name': 'CEM', 'color': 'tab:blue', 'size': marker_size},
        'cbm_linear': {'marker': '*', 'name': 'CBM+Linear', 'color': 'tab:red', 'size': marker_size},
        'cbm_mlp': {'marker': '^', 'name': 'CBM+MLP', 'color': 'tab:purple', 'size': marker_size},
        'blackbox': {'marker': 'o', 'name': 'BlackBox', 'color': 'tab:black', 'size': marker_size},
        'blackbox_multi': {'marker': 'o', 'name': 'BlackBox (Multi)', 'color': 'tab:grey', 'size': marker_size},
        'cgm': {'marker': 'D', 'name': 'CGM', 'color': 'tab:orange', 'size': marker_size},
        'c2bm': {'marker': 's', 'name': 'C2BM', 'color': 'tab:green', 'size': marker_size},
    }
    
    os.makedirs(save_dir, exist_ok=True)
    history_files = _find_history_files(paths, history_filename=history_filename)
    histories_by_model = _load_histories_by_model(history_files)

    if not histories_by_model:
        print("[training_history] No histories loaded.")
        return

    if model_styles is None:
        model_styles = globals().get("model_styles", {})

    if model_styles:
        model_order = list(model_styles.keys())
        ordered_models = [m for m in model_order if m in histories_by_model]
        skipped_models = sorted([m for m in histories_by_model.keys() if m not in model_styles])
        if skipped_models:
            print(f"[training_history] Skipping models without styles: {skipped_models}")
        histories_by_model = {m: histories_by_model[m] for m in ordered_models}
        models = ordered_models
    else:
        models = sorted(histories_by_model.keys())

    if not models:
        print("[training_history] No models found after filtering.")
        return

    all_histories = []
    for model in models:
        model_histories = histories_by_model.get(model, [])
        all_histories.extend(model_histories)
        print(f"[training_history] {model}: {len(model_histories)} histories")

    rounds = _common_rounds(all_histories)
    if not rounds:
        print("[training_history] No common rounds across histories.")
        return

    client_ids = _collect_client_ids(all_histories, key="loss_val_client", mode="union")
    print(f"[training_history] Aggregating {len(all_histories)} histories across {len(rounds)} rounds.")

    default_font_sizes = {
        "base": 16,
        "title": 18,
        "label": 16,
        "tick": 14,
        "legend": 14,
    }
    if font_sizes is None:
        font_sizes = default_font_sizes
    else:
        font_sizes = {**default_font_sizes, **font_sizes}

    rc_backup = plt.rcParams.copy()
    plt.rcParams.update({
        "font.size": font_sizes["base"],
        "axes.titlesize": font_sizes["title"],
        "axes.labelsize": font_sizes["label"],
        "xtick.labelsize": font_sizes["tick"],
        "ytick.labelsize": font_sizes["tick"],
        "legend.fontsize": font_sizes["legend"],
    })

    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    if color_cycle is not None:
        colors = color_cycle.by_key().get("color", [])
    else:
        colors = []
    if not colors:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    def _style_for(model_name, idx):
        style = model_styles.get(model_name, {})
        color = style.get("color", colors[idx % len(colors)])
        marker = style.get("marker", "o")
        label = style.get("name", model_name)
        size = style.get("size", 8)
        return color, marker, label, size

    # 1. Per-client validation loss with confidence intervals.
    try:
        if client_ids:
            fig_height = max(4, min(12, 2 * len(client_ids)))
            fig, axes = plt.subplots(len(client_ids), 1, figsize=(10, fig_height), sharex=True)
            if len(client_ids) == 1:
                axes = [axes]

            legend_items = {}
            for cid, ax in zip(client_ids, axes):
                for idx, model_name in enumerate(models):
                    model_histories = histories_by_model.get(model_name, [])
                    values = _aligned_series(model_histories, rounds, "loss_val_client", client_id=cid)
                    if np.all(np.isnan(values)):
                        continue
                    mean, ci, _ = _mean_and_ci(values, confidence=confidence)
                    color, marker, label, size = _style_for(model_name, idx)
                    line = ax.plot(
                        rounds,
                        mean,
                        marker=marker,
                        color=color,
                        linewidth=2,
                        markersize=size,
                        label=label,
                    )[0]
                    ax.fill_between(rounds, mean - ci, mean + ci, color=color, alpha=0.2)
                    legend_items.setdefault(label, line)

                ax.set_ylabel("Validation Loss")
                ax.set_title(f"Client {cid}")
                ax.grid(True, linestyle="--", alpha=0.7)

            axes[-1].set_xlabel("Round")
            if legend_items:
                fig.legend(
                    list(legend_items.values()),
                    list(legend_items.keys()),
                    loc="lower center",
                    ncol=min(len(legend_items), 3),
                    frameon=True,
                    bbox_to_anchor=(0.5, -0.02),
                )
                fig.tight_layout(rect=[0, 0.05, 1, 1])
            else:
                fig.tight_layout()
            fig.savefig(f"{save_dir}/client_validation_losses.png", dpi=300, bbox_inches="tight")

        # 2. Average validation loss across clients with confidence intervals.
        plt.figure(figsize=(10, 6))
        for idx, model_name in enumerate(models):
            model_histories = histories_by_model.get(model_name, [])
            avg_values = _aligned_series(model_histories, rounds, "loss_val_avg")
            if np.all(np.isnan(avg_values)):
                continue
            avg_mean, avg_ci, _ = _mean_and_ci(avg_values, confidence=confidence)
            color, marker, label, size = _style_for(model_name, idx)
            plt.plot(
                rounds,
                avg_mean,
                marker=marker,
                color=color,
                linewidth=2,
                markersize=size,
                label=label,
            )
            plt.fill_between(rounds, avg_mean - avg_ci, avg_mean + avg_ci, color=color, alpha=0.2)

        plt.xlabel("Round")
        plt.ylabel("Validation Loss")
        plt.title("Average Validation Loss Across Clients")
        plt.legend(ncol=min(len(models), 3), frameon=True)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/average_validation_loss.png", dpi=300, bbox_inches="tight")

        # 3. Average validation accuracy across clients with confidence intervals.
        plotted_any = False
        plt.figure(figsize=(10, 6))
        for idx, model_name in enumerate(models):
            model_histories = histories_by_model.get(model_name, [])
            acc_values = _aligned_series(model_histories, rounds, "y_acc_val_avg")
            if np.all(np.isnan(acc_values)):
                continue
            acc_mean, acc_ci, _ = _mean_and_ci(acc_values, confidence=confidence)
            color, marker, label, size = _style_for(model_name, idx)
            plt.plot(
                rounds,
                acc_mean,
                marker=marker,
                color=color,
                linewidth=2,
                markersize=size,
                label=label,
            )
            plt.fill_between(rounds, acc_mean - acc_ci, acc_mean + acc_ci, color=color, alpha=0.2)
            plotted_any = True

        if plotted_any:
            plt.xlabel("Round")
            plt.ylabel("Validation Accuracy")
            plt.title("Average Validation Accuracy Across Clients")
            plt.legend(ncol=min(len(models), 3), frameon=True)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/average_validation_accuracy.png", dpi=300, bbox_inches="tight")
        else:
            print("[training_history] Accuracy history is empty or NaN; skipping accuracy plot.")
            plt.close()
    finally:
        plt.close("all")
        plt.rcParams.update(rc_backup)
        print(f"Training plots saved to {save_dir}/")
