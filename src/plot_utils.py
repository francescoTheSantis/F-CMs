import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scienceplots
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])

def delta_single_c_interventions_on_y(d):
    baseline = d['_baseline']
    delta_dict = {k:(v - baseline) for k, v in d.items()}
    # Remove the baseline from the delta_dict
    delta_dict.pop('_baseline', None)
    return delta_dict

def reorder(learning_methods):
    n = len(learning_methods) -2 # the numbeer of clients is given by removing centralized and federated
    # the custom order is: centralized, local_federated, localized_1, localized_2, ..., localized_n
    custom_order = ['centralized', 'local_federated'] + [f'localized_{i+1}' for i in range(n)]
    # Reorder the learning methods according to the custom order
    ordered_learning_methods = [method for method in custom_order if method in learning_methods]
    return ordered_learning_methods

def rename_learning_methods(learning_method):
    for method in learning_method:
        if method == 'centralized':
            renamed_method = ['Centralized']
        elif method == 'local_federated':
            renamed_method = ['Federated']
        elif method.startswith('localized'):
            # Extract the number from the method name
            num = method.split('_')[-1]
            renamed_method = [f'Localized (cl. {num})']
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
    plot_name="default"
):

    datasets = input['dataset'].unique()
    # Reorder datasets according to custom order
    datasets = sorted(datasets, key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
    learning_methods = input['learning'].unique()

    if plot_name=='single_c_interventions_on_y':
        learning_methods = reorder(learning_methods)

    n_rows = len(learning_methods)
    n_cols = len(datasets)

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
    input = input[['seed', 'dataset', 'model', 'learning', 'single_c_interventions_on_y']].dropna()
    input['single_c_interventions_on_y'] = input['single_c_interventions_on_y'].apply(delta_single_c_interventions_on_y)
    single_c_plot(
        input,
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

def plot_single_id_ood_on_y(
    input, 
    custom_order,
    model_styles,
    folder=None,
    id=False,
    figsize=(15, 10),
    title_size=16,
    label_size=14,
    tick_size=14,
    legend_size=14,
    legend_bgcolor='lightgray',
    legend_edgecolor='black',
    legend_alpha=0.3
):
    str_id = 'id' if id else 'ood'
    plot_name = f"single_c_interventions_on_y_{str_id}"

    input = input[['seed', 'dataset', 'model', 'learning', f'single_{str_id}_on_y', 'single_c_interventions_on_y']].dropna()
    # Explode the 'single_id_on_y' column to create multiple rows for each list element
    input = input.explode(f'single_{str_id}_on_y').reset_index(drop=True)

    # Update the 'learning' column with the index of the element in the list
    input['learning'] = input.groupby(['seed', 'dataset', 'model', 'learning']).cumcount().apply(lambda x: f"cl. {x}")

    input[f'single_{str_id}_on_y'] = input.apply(
        lambda row: delta_single_c_interventions_on_y_id_ood(row[f'single_{str_id}_on_y'], row['single_c_interventions_on_y']), 
        axis=1
    )


    # Rename f'single_{str_id}_on_y' to f"single_c_interventions_on_y_{str_id}"
    input = input.rename(columns={f'single_{str_id}_on_y': f"single_c_interventions_on_y_{str_id}"})

    single_c_plot(
        input,
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
        plot_name=plot_name
    )