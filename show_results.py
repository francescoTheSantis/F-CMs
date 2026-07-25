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
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/12-09-51_int_siim_seed1",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/12-08-41_int_siim_s_seed1",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/12-06-48_int_siim_loc_new_seed1"
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
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-01/11-30-03_chexpert_old_s"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-03-29/09-24-50_CheXpert_FINAL",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-04-03/09-59-47_othermodels_245",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-04-04/09-13-02_chexpert_finalseeds"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-04-04/09-13-07_chexpert_s_final_seeds",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-03-29/09-24-55_CheXpert_FINAL_s",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-04-03/09-59-52_othermodels_245_s"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-02/18-00-35_chexpert_oldpr",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-02/23-42-04_othermodels",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-03/07-26-40_otherseed",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-02/18-00-30_chexpert_oldpr_s",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-03/10-01-01_othermodels_s",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-03/07-26-45_otherseed_2",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-03/11-33-32_chexpert_oldpr_int",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-03/18-48-05_chexpert_oldpr_int_s"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-03/21-19-13_oherbaselines"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-03/21-29-59_int_seed1",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-03/21-30-04_int_seed1_s"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-04/08-35-54_int_seed3",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-04/08-35-59_int_seed3_s"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/00-35-11_seed4_dario",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/00-19-32_static_seed4_dario"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/22-05-17_seed3_dario",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/23-20-07_static_seed3_dario",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/00-12-11_int_siim_correct_data/00-12-11_int_siim_correct_data",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/00-13-18_int_siim_s_correct_data/00-13-18_int_siim_s_correct_data",
    #"/home/admin/Federated-C2BM_v2/outputs/08-57-08_seed2_dario",
    #"/home/admin/Federated-C2BM_v2/outputs/08-56-51_static_seed2_dario"
    ### new ones ###
    #"/home/admin/Federated-C2BM_v2/outputs/01-12-38_seed5_dario",
    #"/home/admin/Federated-C2BM_v2/outputs/01-01-13_static_seed5_dario"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-19/08-59-59_hail_int_seed3_prova",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-20_hail_int_seed4_prova/08-58-01" # 4 NO
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-25/09-57-53_hail_int_seed5_prova", # 4 YES
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-26/08-34-29_hail_int_s_seed3_prova",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-27/09-35-05_hail_int__s_seed5_prova",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-28/09-51-15_asia_int_seed5",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-28/12-26-34_asia_int_s_seed5",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-29/13-17-13_asia_int_s_seed3",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-29/13-25-01_asia_int_seed3"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-29/16-45-06_asia_int_s_seed4",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-05-29/18-59-15_asia_int_seed4"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-02/08-09-59_siim_int_seed3",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-03/08-44-21_siim_int_s_seed3",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-05/09-09-07_siim_int_seed6", # NO
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-05/10-58-57_siim_int_seed6_s", # NO
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-05/09-09-12_siim_int_seed7", # NO
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-05/11-11-09_siim_int_seed_7_s" # NO
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-09/09-20-18_siim_int_seed3",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-09/11-30-15_siim_int_seed3_s"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-09/09-20-23_siim_int_seed5",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-09/11-41-00_siim_int_seed5_s"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-09/13-25-13_siim_int_seed8", 
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-09/15-16-12_siim_int_seed8_s"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-09/13-25-18_siim_int_seed9", # NO
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-09/15-32-24_siim_int_seed9_s" # NO
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-16/07-52-02_siim_int_seed8_c",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-17/09-23-17_siim_int_seed8_d"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-18/09-56-05_siim_int_seed11_c", # c2bm perfect, cem no
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-18/10-52-26_siim_int_seed11_d",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-18/13-52-26_siim_int_seed11_s"
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-18/09-56-10_siim_int_seed12_c",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-18/10-52-31_siim_int_seed12_d",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-18/14-15-08_siim_int_seed12_s",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-19/08-32-47_siim_int_seed13_c", # c2bm perfect, cem no
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-19/09-29-03_siim_int_seed13_d_l",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-19/12-33-28_siim_int_seed13_s",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-19/08-32-52_siim_int_seed14_c",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-19/09-29-08_siim_int_seed14_d_l",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-19/13-03-28_siim_int_seed14_s",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-24/08-32-41_siim_int_seed15_c", # NO
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-24/09-23-06_siim_int_seed15_d_l",
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-24/12-00-10_siim_int_seed15_s",
    "/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-24/08-32-46_siim_int_seed16_c", # centralized > local_fed ma local_fed_s = localized
    #"/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-24/09-23-30_siim_int_seed16_d_l",
    "/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-24/13-03-59_siim_int_seed16_s",
    "/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-25/09-10-55_siim_int_seed16_l",
    "/home/admin/Federated-C2BM_v2/outputs/multirun/2026-06-25/12-05-21_siim_int_seed16_d"


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
    'cheXpert': {'name': 'CheXpert'},
    'chexpert': {'name': 'CheXpert'},
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
    'CheXpert',
]

performance = apply_styles(performance, dataset_styles, model_styles, custom_order)


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
            variable='labels',
            c_info=c_info,
            folder=visualization_folder,
            return_data=True,  # Return data for grid plot
        )
        if data:
            plot_data_dict.update(data)

    # Create grid plot with CEM and C2BM only
    grid_model_names = ['CEM', 'C2BM']  # Only CEM and C2BM (display names, post apply_styles)
    datasets = [d for d in custom_order if d in performance['dataset'].unique()]
    if not datasets:
        datasets = sorted(performance['dataset'].unique(), key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
    print(f"[show_results] Grid datasets: {datasets}")


    plot_cumulative_accuracy_grid_multi_modality(
        plot_data_dict=plot_data_dict,
        model_names=grid_model_names,
        datasets=datasets,
        variable='labels',
        folder=visualization_folder,
    )

    # plot cumulative interventions for all architectures together, single learning modality


    data_2 = plot_cumulative_accuracy_multi_model(
            performance,
            custom_order,
            learning_modality='local_federated_drift',
            variable='labels',
            c_info=c_info,
            folder=visualization_folder,
            return_data=True  # Return data for grid plot
            #seeds_to_average = [3,4,5],
        )

    # Eliminate blackbox models from plot_data_dict
    data_2_filtered = {}
    for key, value in data_2.items():
        if value is None:
            data_2_filtered[key] = None
        else:
            filtered_model_data = {k: v for k, v in value.get('model_data', {}).items() 
                                  if 'blackbox' not in k.lower()}
            data_2_filtered[key] = {
                'model_data': filtered_model_data,
                'n_interventions': value.get('n_interventions', 0)
            }

    plot_cumulative_accuracy_grid_multi_model(
        plot_data_dict=data_2_filtered,
        learning_modalities=['local_federated_drift'],
        datasets=datasets,
        variable='labels',
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
