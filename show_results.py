import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
import yaml
import json
from matplotlib.ticker import FuncFormatter
import pickle
import math
from statistics import NormalDist
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
    #"/home/admin/Federated-C2BM/outputs/multirun/2025-11-10/18-57-50",
    #"/home/admin/Federated-C2BM/outputs/multirun/2025-11-11/18-33-54",
    # "/home/admin/Federated-C2BM/outputs/multirun/2025-11-12/07-57-37",
    # "/home/admin/Federated-C2BM/outputs/multirun/2025-11-12/08-22-54",
    # "/home/admin/Federated-C2BM/outputs/multirun/2025-11-12/09-30-44",
    # "/home/admin/Federated-C2BM/outputs/multirun/2025-11-12/13-18-41",
    # "/home/admin/Federated-C2BM/outputs/multirun/2025-11-12/16-46-58",
    # "/home/admin/Federated-C2BM/outputs/multirun/2025-12-01/03-26-27"
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2026-01-08/12-46-39"
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2026-01-13/17-00-51"
    "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2026-01-13/17-07-41"
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2026-01-13/16-14-19_test_2"
]

# folder to save processed results
visualization_folder = 'figs/'

# maximum number of clients
n_clients = 10

# load the paths of each experiments
exps_path = setup_results(paths, visualization_folder)

# load the experiment results
performance, c_info = load_exps(exps_path, n_clients=n_clients, args=args)

######### Dataset and model styles #########

# Define a dictionary to associate marker, name, and color to each model.
# If the experiment you run does not contain a model, just remove it from the dictionary.
# If you want to add a new model, just add it to the dictionary.
marker_size = 14
model_styles = {
    'cem': {'marker': 'P', 'name': 'CEM', 'color': 'tab:blue', 'size': marker_size},
    'cbm_linear': {'marker': '*', 'name': 'CBM+Linear', 'color': 'tab:red', 'size': marker_size},
    'cbm_mlp': {'marker': '^', 'name': 'CBM+MLP', 'color': 'tab:purple', 'size': marker_size},
    'blackbox': {'marker': 'o', 'name': 'BlackBox', 'color': 'tab:black', 'size': marker_size},
    'blackbox_multi': {'marker': 'o', 'name': 'BlackBox (Multi)', 'color': 'tab:grey', 'size': marker_size},
    'cgm': {'marker': 'D', 'name': 'CGM', 'color': 'tab:orange', 'size': marker_size},
    'c2bm': {'marker': 's', 'name': 'C2BM', 'color': 'tab:green', 'size': marker_size},
}
dataset_styles = {
    'sachs': {'name': 'Sachs'},
    'asia': {'name': 'Asia'},
    'alarm': {'name': 'Alarm'},
    'hailfinder': {'name': 'Hailfinder'},
    'insurance': {'name': 'Insurance'},
}

# Define the custom order
# If the experiment you run does not contain a dataset, just remove it from the list.
custom_order = [
    'Asia',
    'Sachs',
    'Alarm',
    'Insurance',
    'Hailfinder',
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

########## Intervention plots ##########
# Eliminate blackbox and blackbox_multi from the model style
model_styles = {k: v for k, v in model_styles.items() if k not in ['blackbox', 'blackbox_multi']}

# Eliminate blackbox and blackbox_multi from the performance dataframe
performance = performance[performance['model'].isin(['blackbox', 'blackbox_multi']) == False]

### Intervention plot for single c interventions on y ###
plot_single_c_on_y(performance, custom_order, model_styles, visualization_folder)

### Intervention plot for level interventions ###
plot_level_interventions(performance, custom_order, model_styles, visualization_folder, c_info)







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


plot_training_metrics_across_seeds(
    paths=paths,
    save_dir="figs",
    confidence=0.95,
    model_styles=model_styles,
)
