import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
import yaml
import json
from matplotlib.ticker import FuncFormatter
from src.plot_utils import *
import argparse
#from env import CACHE

warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--mia_metric", type=str, default="max", choices=["max", "mean"],
                    help="Metric to use for MIA results (default: max)")
args = parser.parse_args()


# List the paths containing the sweeps' results
paths = [
    # "outputs/multirun/2026-04-16/13-02-51_asia", #OK
    # "outputs/multirun/2026-04-16/13-02-46_asia_s", #OK
    # "outputs/multirun/2026-04-16/13-58-53_ins", #OK
    # "outputs/multirun/2026-04-16/13-58-54_ins_s", #OK
    # "outputs/multirun/2026-04-18/12-49-24_ha", #OK but skipped 1 cgm and 1 c2bm
    # "outputs/multirun/2026-04-18/12-49-30_ha_s", #OK
    # "outputs/multirun/2026-04-20/10-45-55_sachs", #OK
    # "outputs/multirun/2026-04-20/10-46-16_sachs_s", #OK
    # "outputs/multirun/2026-04-20/19-21-00_alarm", #OK
    # "outputs/multirun/2026-04-20/19-21-16_alarm_s", #OK 
    # "outputs/multirun/2026-04-21/11-11-20_siim", #OK but skipped seed 1 for all models/learning modes
    # "outputs/multirun/2026-04-21/11-11-42_siim_s", #OK but skipped seed 1 for all models/learning modes
    # "outputs/multirun/2026-04-22/15-49-04_asia_bas", 
    # "outputs/multirun/2026-04-22/15-51-34_sachs_bas",
    # "outputs/multirun/2026-04-22/15-53-59_alarm_bas",
    # "outputs/multirun/2026-04-22/15-56-11_ins_bas",
    # "outputs/multirun/2026-04-22/15-59-30_hail_bas",
    # "outputs/multirun/2026-04-22/16-03-56_siim_bas",
    #
    # - balanced training and val and unbalanced (original) test
    # "outputs/multirun/2026-04-30/22-37-56_v1_all",
    # "outputs/multirun/2026-04-30/22-38-19_v1_static",
    # "outputs/multirun/2026-04-30/23-49-02_v2_all", # benino
    # "outputs/multirun/2026-05-01/06-40-22_v2_static",
    # "outputs/multirun/2026-04-30/23-49-25_v3_all",
    # "outputs/multirun/2026-05-01/06-40-45_v3_static",
    # "outputs/multirun/2026-05-01/03-28-00_v4_all",
    "outputs/multirun/2026-05-01/10-16-39_v4_static",
    # "outputs/multirun/2026-05-01/03-28-22_v5_all",
    # "outputs/multirun/2026-05-01/10-17-02_v5_static",
    
]

# folder to save processed results
visualization_folder = 'figs/'

# maximum number of clients
n_clients = 10

# load the paths of each experiments
exps_path = setup_results(paths, visualization_folder)

# load the experiment results
performance, c_info = load_exps(exps_path, n_clients=n_clients, args=args)

# Debug: Check drift configuration
print("\n[DEBUG] Loaded experiments configuration:")
if 'rnd_drift' in performance.columns and 'n_rounds' in performance.columns:
    drift_summary = performance[['dataset', 'learning', 'rnd_drift', 'n_rounds']].drop_duplicates()
    print(drift_summary)
    print(f"\nUnique rnd_drift values: {performance['rnd_drift'].unique()}")
    print(f"Unique n_rounds values: {performance['n_rounds'].unique()}")
print()

######### Dataset and model styles #########

# Define a dictionary to associate marker, name, and color to each model.
# If the experiment you run does not contain a model, just remove it from the dictionary.
# If you want to add a new model, just add it to the dictionary.
marker_size = 14
model_styles = {
    'cem': {'marker': 'P', 'name': 'CEM', 'color': 'tab:blue', 'size': marker_size},
    'cem_multi': {'marker': 'P', 'name': 'CEM (Multi)', 'color': 'steelblue', 'size': marker_size},
    'cbm_linear': {'marker': '*', 'name': 'CBM+Linear', 'color': 'tab:red', 'size': marker_size},
    'cbm_mlp': {'marker': '^', 'name': 'CBM+MLP', 'color': 'tab:purple', 'size': marker_size},
    'cbm_linear_multi': {'marker': '*', 'name': 'CBM+Linear (Multi)', 'color': 'tab:pink', 'size': marker_size},
    'cbm_mlp_multi': {'marker': '^', 'name': 'CBM+MLP (Multi)', 'color': 'tab:brown', 'size': marker_size},
    'blackbox': {'marker': 'o', 'name': 'BlackBox', 'color': 'tab:black', 'size': marker_size},
    'blackbox_multi': {'marker': 'o', 'name': 'BlackBox (Multi)', 'color': 'tab:grey', 'size': marker_size},
    'blackbox_multi_multi': {'marker': 'o', 'name': 'BlackBox (MultiModal)', 'color': 'tab:olive', 'size': marker_size},
    'cgm': {'marker': 'D', 'name': 'CGM', 'color': 'tab:orange', 'size': marker_size},
    'cgm_multi': {'marker': 'D', 'name': 'CGM (Multi)', 'color': 'royalblue', 'size': marker_size},
    'c2bm': {'marker': 's', 'name': 'C2BM', 'color': 'tab:green', 'size': marker_size},
    'c2bm_multi': {'marker': 's', 'name': 'C2BM (Multi)', 'color': 'forestgreen', 'size': marker_size},
}
dataset_styles = {
    'sachs': {'name': 'Sachs'},
    'asia': {'name': 'Asia'},
    'alarm': {'name': 'Alarm'},
    'hailfinder': {'name': 'Hailfinder'},
    'insurance': {'name': 'Insurance'},
    # 'nih_chest_images': {'name': 'NIH Chest X-Ray'},
    # 'cub_causal_struct': {'name': 'CUB_CAUSAL'},
    'siim_pneumothorax': {'name': 'SIIM-Pneumothorax'},
    'cheXpert_multi': {'name': 'CheXpert-Multi'},
}

# Define the custom order
# If the experiment you run does not contain a dataset, just remove it from the list.
custom_order = [
    'Asia',
    'Sachs',
    'Alarm',
    'Insurance',
    'Hailfinder',
    # 'NIH Chest X-Ray',
    # 'CUB_CAUSAL',
    'SIIM-Pneumothorax',
    'CheXpert-Multi',
]

apply_styles(performance, dataset_styles, model_styles, custom_order)


########## Task & Concept Accuracy Plot ##########
complete_task_stats, complete_concept_stats, complete_label_stats = produce_accuracy_tables(performance)
kwargs_tables = {
    'complete_task_stats': complete_task_stats,
    'complete_concept_stats': complete_concept_stats,
    'complete_label_stats': complete_label_stats,
    'custom_order': custom_order,
    'model_styles': model_styles,
    'visualization_folder': visualization_folder,
}

# task
kwargs_tables['label'] = 'task'
tabular_task_and_concept_accuracy(**kwargs_tables)

# concepts
kwargs_tables['label'] = 'concepts'
tabular_task_and_concept_accuracy(**kwargs_tables)

# labels
kwargs_tables['label'] = 'labels'
tabular_task_and_concept_accuracy(**kwargs_tables)

# balanced task accuracy
kwargs_tables['label'] = 'balanced_task'
tabular_task_and_concept_accuracy(**kwargs_tables)

# Concept coverage and parameter change tables
coverage_stats, param_change_stats = compute_drift_statistics(performance)
tabular_drift_metrics(
    coverage_stats,
    param_change_stats,
    custom_order,
    model_styles,
    visualization_folder,
)


########## Intervention plots ##########
# FedCBM runs can legitimately miss intervention artifacts; keep accuracy tables
# and skip intervention plots only when no intervention data is available.
has_intervention_data = (
    ('cumulative_task_interventions' in performance.columns and performance['cumulative_task_interventions'].notna().any())
    and ('cumulative_concept_interventions' in performance.columns and performance['cumulative_concept_interventions'].notna().any())
)

if not has_intervention_data:
    print("[show_results] No intervention artifacts found. Skipping intervention plots.")
else:
# Eliminate blackbox and blackbox_multi from the model style
#model_styles = {k: v for k, v in model_styles.items() if k not in ['blackbox', 'blackbox_multi']}

# Eliminate blackbox and blackbox_multi from the performance dataframe
#performance = performance[performance['model'].isin(['blackbox', 'blackbox_multi']) == False]

### Intervention plot for single c interventions on y ###
#plot_single_c_on_y(performance, custom_order, model_styles, visualization_folder)

### Intervention plot for level interventions ###
#plot_level_interventions(performance, custom_order, model_styles, visualization_folder, c_info)

### Cumulative intervention plots ###

# Collect data for grid plot
    plot_data_dict = {}

    # plot cumulative interventions for each architecture, multi learning modalities
    for architecture in performance['model'].unique():
        data = plot_cumulative_accuracy_multi_modality(
            performance,
            custom_order,
            architecture_name=architecture,
            variable='task',
            c_info=c_info,
            folder=visualization_folder,
            return_data=True,  # Return data for grid plot
        )
        if data:
            plot_data_dict.update(data)

    # Create grid plot with CEM and C2BM only
    grid_model_names = ['cem','c2bm']  # Only CEM and C2BM
    datasets = ['Asia','Hailfinder', "SIIM-Pneumothorax"]
    # Sort datasets according to custom_order
    datasets = sorted(datasets, key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))


    plot_cumulative_accuracy_grid_multi_modality(
        plot_data_dict=plot_data_dict,
        model_names=grid_model_names,
        datasets=datasets,
        variable='task',
        folder=visualization_folder,
    )

    # plot cumulative interventions for all architectures together, single learning modality


    data_2 = plot_cumulative_accuracy_multi_model(
            performance,
            custom_order,
            learning_modality='local_federated_drift',
            variable='task',
            c_info=c_info,
            folder=visualization_folder,
            return_data=True  # Return data for grid plot
            #seeds_to_average = [3,4,5],
        )

    plot_cumulative_accuracy_grid_multi_model(
        plot_data_dict=data_2,
        learning_modalities=['local_federated_drift'],
        datasets=datasets,
        variable='task',
        folder=visualization_folder,
    )





# def plot_intervention_results(df, metric='accuracy', title_font=None, label_font=None, tick_font=None, legend_font=None):
#     unique_noises = [0]
#     unique_datasets = custom_order
#     n_cols = len(unique_noises)
#     n_rows = len(unique_datasets)
#     fig, axes = plt.subplots(n_cols, n_rows, figsize=(30, 7), sharex=True, sharey=False)
    
#     for i, dataset in enumerate(unique_datasets):
#         for j, noise in enumerate(unique_noises):
#             ax = axes[i] #axes[i, j] if n_rows > 1 else axes[j]
#             data = df[(df['noise'] == noise) & (df['dataset'] == dataset)]
#             grouped_data = data.groupby(['p_int', 'model']).agg(
#                 mean_metric=(metric, 'mean'),
#                 std_metric=(metric, 'std')
#             ).reset_index().fillna(0)
#             for model in grouped_data['model'].unique():
#                 model_data = grouped_data[grouped_data['model'] == model]
#                 style = model_styles.get(model, {'marker': 'o', 'color': 'black', 'size': 10, 'name': model})
#                 ax.plot(model_data['p_int'], model_data['mean_metric'], color=style['color'], linestyle='-', alpha=0.5)
#                 ax.scatter(model_data['p_int'], model_data['mean_metric'], marker=style['marker'], color=style['color'], s=style['size']**2, label=style['name'], edgecolor='black', alpha=0.5)
#                 ax.fill_between(model_data['p_int'], model_data['mean_metric'] - model_data['std_metric'], model_data['mean_metric'] + model_data['std_metric'], color=style['color'], alpha=0.2)
#             ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
#             if j == 0:
#                 ax.set_title(f'{get_df_name(dataset)}', fontsize=label_font['size'])
#             if i == 0:
#                 ax.set_ylabel('Task Acc', fontsize=label_font['size'])
#             ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
#             ax.minorticks_off()
#             ax.grid(True)
#             ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    
#     # Create a single legend below the plots
#     handles, labels = [], []
#     for ax in axes.flatten():
#         for handle, label in zip(*ax.get_legend_handles_labels()):
#             if label not in labels:
#                 handles.append(handle)
#                 labels.append(label)
#     for handle in handles:
#         handle.set_alpha(1)  # Remove transparency from legend markers

#     # Create custom legend handles
#     custom_handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=style['size']+10, label=style['name'], markeredgewidth=0.5, markeredgecolor='black') for style in model_styles.values()]

#     # Create a single legend below the plots
#     fig.legend(
#         handles=custom_handles,
#         loc='lower center',
#         ncol=(len(custom_handles) + 1) // 2,  # Split legend into two rows
#         fontsize=tick_font['size'],
#         frameon=True,
#         bbox_to_anchor=(0.5, -0.2),
#         columnspacing=1.0,
#         handletextpad=0.5
#     )

#     plt.tight_layout()
#     plt.savefig('figs/intervention_id.pdf')
#     plt.show()

# # Call the function with the desired metric and font properties
# legend_font = {'size': 44}
# title_font = {'size': 36, 'weight': 'bold'}
# label_font = {'size': 44}
# tick_font = {'size': 28}
# plot_intervention_results(performance, metric='accuracy', title_font=title_font, label_font=label_font, tick_font=tick_font, legend_font=legend_font)


plot_training_metrics_across_seeds(
    paths=paths,
    save_dir="figs",
    confidence=0.95,
    model_styles=model_styles,
)
