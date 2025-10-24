import pickle
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

    datasets = input['dataset'].unique()
    # Reorder datasets according to custom order
    datasets = sorted(datasets, key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
    learning_methods = input['learning'].unique()

    if plot_name=='single_c_interventions_on_y':
        learning_methods = reorder(learning_methods, clients_flag, client_perspective)

    # change models' names according to model_styles
    input['model'] = input['model'].apply(lambda x: model_styles[x]['name'] if x in model_styles else x)

    n_rows = len(learning_methods)
    n_cols = len(datasets)

    figsize = (6 * n_cols, 5 * n_rows)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.4)
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
                bar_errors = [1.96 * interventions_std.get(label, 0) / np.sqrt(len(model_subset)) for label in x_labels]

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
                ax.set_xlabel("Concept Names", fontsize=label_size)

            # Show dataset name only in top row
            if i == 0:
                ax.set_title(dataset, fontsize=title_size)
            else:
                ax.set_title("")

            if j==0:
                ax.set_ylabel("$\\Delta$ on $y$", fontsize=label_size)                

        # Get average vertical position of current row
        row_axes = [axes[i * n_cols + j] for j in range(n_cols)]
        bbox = [ax.get_position() for ax in row_axes]
        y_middle = np.mean([b.y0 + b.height / 2 for b in bbox])

        # Dynamically determine a good x-position based on left-most subplot
        leftmost_ax = row_axes[0].get_position()
        x_pos = leftmost_ax.x0 - 0.04  # Decrease this to get closer (0.02–0.03 usually works well)

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
        bbox_to_anchor=(0.5, -0.02),
        fontsize=legend_size,
        frameon=True
    )
    legend.get_frame().set_facecolor(legend_bgcolor)
    legend.get_frame().set_edgecolor(legend_edgecolor)
    legend.get_frame().set_alpha(legend_alpha)

    # Make space for row labels and legend
    plt.tight_layout(rect=[0.08, 0.07, 1, 1])

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

    datasets = input['dataset'].unique()
    # Reorder datasets according to custom order
    datasets = sorted(datasets, key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
    learning_methods = input['learning'].unique()

    learning_methods = reorder(learning_methods, clients_flag)

    # change models' names according to model_styles
    input['model'] = input['model'].apply(lambda x: model_styles[x]['name'] if x in model_styles else x)

    n_rows = len(learning_methods)
    n_cols = len(datasets)

    figsize = (6 * n_cols, 5 * n_rows)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.4)
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
                line_errors = [1.96 * interventions_std.get(label, 0) / np.sqrt(len(model_subset)) for label in x_labels]

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
                ax.set_xlabel("Intervention Levels", fontsize=label_size)

            # Show dataset name only in top row
            if i == 0:
                ax.set_title(dataset, fontsize=title_size)
            else:
                ax.set_title("")

            if j==0:
                if plot_name == 'level_interventions_on_y':
                    ax.set_ylabel("$\\Delta$ on $y$", fontsize=label_size)
                elif plot_name == 'level_interventions_on_c':
                    ax.set_ylabel("$\\Delta$ on $c$", fontsize=label_size)

        # Get average vertical position of current row
        row_axes = [axes[i * n_cols + j] for j in range(n_cols)]
        bbox = [ax.get_position() for ax in row_axes]
        y_middle = np.mean([b.y0 + b.height / 2 for b in bbox])

        # Dynamically determine a good x-position based on left-most subplot
        leftmost_ax = row_axes[0].get_position()
        x_pos = leftmost_ax.x0 - 0.04  # Decrease this to get closer (0.02–0.03 usually works well)

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
        bbox_to_anchor=(0.5, -0.02),
        fontsize=legend_size,
        frameon=True
    )
    legend.get_frame().set_facecolor(legend_bgcolor)
    legend.get_frame().set_edgecolor(legend_edgecolor)
    legend.get_frame().set_alpha(legend_alpha)

    # Make space for row labels and legend
    plt.tight_layout(rect=[0.08, 0.07, 1, 1])

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
    task_stats['ci_task'] = 1.96 * task_stats['std_accuracy_task'] / np.sqrt(task_stats['total_occurrences'])

    # Compute mean and std for 'concept'
    concept_stats = concept_df.groupby(['model', 'dataset', 'learning']).agg(
        avg_accuracy_concept=('accuracy', 'mean'),
        std_accuracy_concept=('accuracy', 'std'),
        total_occurrences=('accuracy', 'count'),
        concept_left_out=('concept_left_out', 'mean')
    ).reset_index().fillna(0)

    # compute confidence intervals
    concept_stats['ci_concept'] = 1.96 * concept_stats['std_accuracy_concept'] / np.sqrt(concept_stats['total_occurrences'])

    # Aggregated concepts and task performance
    label_stats = concept_df.groupby(['model', 'dataset', 'learning']).agg(
        avg_accuracy_label=('agg_label', 'mean'),
        std_accuracy_label=('agg_label', 'std'),
        total_occurrences=('agg_label', 'count'),
        concept_left_out=('concept_left_out', 'mean')
    ).reset_index().fillna(0)

    # compute confidence intervals
    label_stats['ci_label'] = 1.96 * label_stats['std_accuracy_label'] / np.sqrt(label_stats['total_occurrences'])

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

    for exp in exps_path:
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

                # Concept results
                concept_file = os.path.join(result_file, 'c_accuracy.pkl')
                with open(concept_file, 'rb') as file:
                    concept_results = pickle.load(file)

                # Select the last row of the dataframe where we test the model
                d['concept_acc'] = concept_results

                # Task results
                task_file = os.path.join(result_file, 'y_accuracy.pkl')
                with open(task_file, 'rb') as file:
                    task_results = pickle.load(file)

                d['task_acc'] = task_results['_baseline']

                try:
                    # Collect graph
                    graph_file = os.path.join(exp, 'graph.pkl')
                    with open(graph_file, 'rb') as file:
                        graph_results = pickle.load(file)
                    d['graph'] = graph_results['concepts']
                except FileNotFoundError:
                    d['graph'] = None

                ###### Collect the results for the interventions
                single_id_on_y_files = [os.path.join(exp, 'results', f'client_{client}_single_IDc_interventions_on_y.pkl') \
                                        for client in range(1,n_clients)] # The maximum number of clients has to be known
                single_ood_on_y_files = [os.path.join(exp, 'results', f'client_{client}_single_OODc_interventions_on_y.pkl') \
                                        for client in range(1,n_clients)] # The maximum number of clients has to be known

                level_interventions_on_c_file = os.path.join(exp, 'results', 'level_interventions_on_c.pkl')

                level_interventions_on_y_file = os.path.join(exp, 'results', 'level_interventions_on_y.pkl')

                single_c_interventions_on_y_file = os.path.join(exp, 'results', 'single_c_interventions_on_y.pkl')

                # now read all those files 
                d['single_id_on_y'] = []
                d['single_ood_on_y'] = []
                for client in range(1, n_clients):
                    if os.path.exists(single_id_on_y_files[client-1]):
                        with open(single_id_on_y_files[client-1], 'rb') as f:
                            d['single_id_on_y'].append(pickle.load(f))

                    if os.path.exists(single_ood_on_y_files[client-1]):
                        with open(single_ood_on_y_files[client-1], 'rb') as f:
                            d['single_ood_on_y'].append(pickle.load(f))

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
                        d['single_c_interventions_on_y'] = pickle.load(f)
                else:
                    d['single_c_interventions_on_y'] = None


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
            values = [v for k,v in row.items() if k in row]
            if count_nan:
                return sum([1 for x in values if math.isnan(x)]) if values else 0
            else:
                if worst_classifier:
                    # replace the NaN values with the performance of the worst classifier
                    if dataset is not None and dataset in c_info and c_info[dataset] is not None:
                        for idx, c_name in enumerate(graph):
                            concept_cardinality_idx = c_info[dataset]['names'].index(c_name)
                            concept_cardinality = c_info[dataset]['cardinality'][concept_cardinality_idx]
                            if math.isnan(values[idx]):
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
    # In the aggregation, we simply compute the expected value over the concepts on which
    # the model was trained. Therefore, if the model didn't train on a concept (NaN value), we ignore it in the average.
    performance['concept_left_out'] = performance.apply(lambda x: _format_results(x['concept_acc'], x['graph'], x['dataset'], x['model'], count_nan=True), axis=1)
    performance['agg_concept'] = performance.apply(lambda x: _format_results(x['concept_acc'], x['graph'], x['dataset'], x['model'], count_nan=False), axis=1)
    performance['agg_label'] = performance.apply(lambda x: _format_results(x['concept_acc'], x['graph'], x['dataset'], x['model'], count_nan=False, task=x['task_acc']), axis=1)

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