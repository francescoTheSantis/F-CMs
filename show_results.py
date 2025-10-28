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
from src.plot_utils import *
import argparse
from src.utils import select_round_indices_by_accuracy

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--mia_metric", type=str, default="max", choices=["max", "mean"],
                    help="Metric to use for MIA results (default: max)")
parser.add_argument("--metric_mode", type=str, default="train", choices=["train", "val"],
                    help="Metric mode to use for results (default: train)")
args = parser.parse_args()

warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])

def _extract_series(entry):
    if isinstance(entry, dict):
        rounds = entry.get("rounds", [])
        values = entry.get("values", [])
    else:
        if isinstance(entry, (list, tuple, np.ndarray)):
            values = list(entry)
        else:
            values = []
        rounds = list(range(1, len(values) + 1))
    return rounds, values


def _resolve_selected_indices(meta, series_len, default_indices=None, metric_mode=None):
    meta = meta or {}
    if not isinstance(meta, dict):
        meta = dict(meta)
    original_target_metric = meta.get("target_metric")
    if metric_mode in ("train", "val"):
        meta["target_metric"] = metric_mode

    selected = meta.get("selected_indices")
    if isinstance(selected, list):
        should_ignore_cached = (
            metric_mode in ("train", "val")
            and original_target_metric not in (None, metric_mode)
        )
        if not should_ignore_cached:
            filtered = [int(idx) for idx in selected if isinstance(idx, (int, float)) and 0 <= int(idx) < series_len]
            if filtered:
                return filtered

    target_accuracy = meta.get("target_accuracy")
    tolerance = meta.get("tolerance", 0.01)
    target_metric = meta.get("target_metric", "val")

    if target_accuracy is None:
        if default_indices is not None:
            return [int(idx) for idx in default_indices if isinstance(idx, (int, float)) and 0 <= int(idx) < series_len]
        return list(range(series_len))

    accuracy_val = meta.get("accuracy_val_avg", [])
    accuracy_train = meta.get("accuracy_train_avg", [])
    metric_series = accuracy_train if target_metric == "train" else accuracy_val
    round_numbers = meta.get("all_rounds", list(range(1, len(metric_series) + 1)))

    try:
        indices = select_round_indices_by_accuracy(
            metric_series,
            round_numbers,
            target_accuracy=target_accuracy,
            tolerance=tolerance,
        )
    except Exception:
        indices = []
    indices = [int(idx) for idx in indices if isinstance(idx, (int, float)) and 0 <= int(idx) < series_len]
    if indices:
        return indices
    if default_indices is not None:
        return [int(idx) for idx in default_indices if isinstance(idx, (int, float)) and 0 <= int(idx) < series_len]
    return []


def _aggregate_mia_attack(acc_dict, indices, mia_metric_mode):
    if not isinstance(acc_dict, dict) or not acc_dict:
        return np.nan
    if not indices:
        return np.nan

    rows = []
    for cid in sorted(acc_dict.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        _, values = _extract_series(acc_dict[cid])
        row = []
        for idx in indices:
            if 0 <= idx < len(values):
                row.append(float(values[idx]))
            else:
                row.append(np.nan)
        rows.append(row)

    if not rows:
        return np.nan
    arr = np.array(rows, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan

    if mia_metric_mode == "max":
        return float(np.nanmax(arr))

    mean_per_round = np.nanmean(arr, axis=0)
    if np.all(np.isnan(mean_per_round)):
        return np.nan
    return float(np.nanmax(mean_per_round))


def _compute_mia_metrics_from_results(mia_data, mia_metric_mode, metric_mode=None):
    accuracies = mia_data.get("accuracies")
    meta = mia_data.get("meta", {}) or {}
    if not isinstance(accuracies, dict):
        return None, None, meta

    if not isinstance(meta, dict):
        meta = dict(meta)
    if metric_mode in ("train", "val"):
        effective_metric_mode = metric_mode
    else:
        existing = meta.get("target_metric")
        effective_metric_mode = existing if existing in ("train", "val") else "val"
    meta["target_metric"] = effective_metric_mode

    primary_series_key = "accuracy_train_avg" if effective_metric_mode == "train" else "accuracy_val_avg"
    fallback_series_key = "accuracy_val_avg" if primary_series_key == "accuracy_train_avg" else "accuracy_train_avg"

    series_len = len(meta.get(primary_series_key, []))
    if series_len == 0:
        series_len = len(meta.get(fallback_series_key, []))
    if series_len == 0:
        for attack_dict in accuracies.values():
            if isinstance(attack_dict, dict) and attack_dict:
                first_entry = next(iter(attack_dict.values()))
                _, values = _extract_series(first_entry)
                series_len = len(values)
                if series_len:
                    break

    indices = _resolve_selected_indices(meta, series_len, metric_mode=effective_metric_mode)
    metrics = {}
    for attack in ["whitebox", "blackbox", "blackbox_shadow"]:
        metrics[attack] = _aggregate_mia_attack(accuracies.get(attack, {}), indices, mia_metric_mode)

    if isinstance(meta, dict):
        meta["selected_indices"] = indices
        all_rounds = meta.get("all_rounds", list(range(1, series_len + 1)))
        selected_rounds = [all_rounds[idx] for idx in indices if 0 <= idx < len(all_rounds)]
        if selected_rounds:
            meta["selected_rounds"] = selected_rounds

        train_series = meta.get("accuracy_train_avg", [])
        val_series = meta.get("accuracy_val_avg", [])
        train_values = [train_series[idx] for idx in indices if 0 <= idx < len(train_series)]
        val_values = [val_series[idx] for idx in indices if 0 <= idx < len(val_series)]
        if train_values:
            meta["selected_round_train_accuracies"] = train_values
        if val_values:
            meta["selected_round_val_accuracies"] = val_values

        metric_series = train_series if effective_metric_mode == "train" else val_series
        metric_values = [metric_series[idx] for idx in indices if 0 <= idx < len(metric_series)]
        if metric_values:
            meta["selected_round_metric_values"] = metric_values

    return metrics, indices, meta


def _compute_sia_metric_from_results(sia_data, default_indices=None, default_meta=None, metric_mode=None):
    accuracies = sia_data.get("accuracies")
    if not isinstance(accuracies, list):
        return np.nan
    meta = sia_data.get("meta", {}) or {}
    if not meta and default_meta:
        meta = default_meta

    series_len = len(accuracies)
    indices = _resolve_selected_indices(meta, series_len, default_indices=default_indices, metric_mode=metric_mode)
    if not indices:
        if meta.get("target_accuracy") is not None:
            return np.nan
        indices = list(range(series_len))

    values = [accuracies[idx] for idx in indices if 0 <= idx < series_len]
    if not values:
        return np.nan
    return float(np.nanmax(values))

# List the paths containing the results
paths = [
    "/home/edoardog/projects/Federated-C2BM/outputs/multirun/2025-10-27/17-27-07"
    #"/home/edoardog/projects/Federated-C2BM/outputs/multirun/2025-10-27/14-29-24"
    #"/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-09-05/22-13-56" # no concept infor for mia e sia
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-09-05/17-16-48" # same as the next but with no concept info for mia
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-09-05/15-00-18" # same as the next but with correct shadow results for cmb, multi_output is enabled
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-09-05/10-02-00" # same as the next but with whitebox score in shadow attack
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-09-04/15-42-45" # good results asia 3 folds, shared with others
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-09-03/22-52-33" # good results asia 1 fold
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-08-29/16-35-07" # 2000 con 20 epochs
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-08-29/14-48-26" # 6000
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-08-29/11-23-47_asia1000_samples" #1000 samples asia
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-08-29/13-33-07" #4000 samples asia
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-08-26/22-32-35",
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-08-28/11-02-38"
    # "/Users/dariofenoglio/Library/CloudStorage/OneDrive-USI/PC/Desktop/USI_Locale/Federated-C2BM/outputs/multirun/2025-08-26/18-31-14_good_asia"
]

# maximum number of clients
n_clients = 10

visualization_folder = 'figs/'
# if folder doesn't exist, create it
if not os.path.exists(visualization_folder):
    os.makedirs(visualization_folder)

###### Collect results regarding concept/task performance ######

exps_path = []
lmr_paths = []
for path in paths:
    exps = os.listdir(path)
    exps_path += [os.path.join(path, exp) for exp in exps if 'multirun' not in exp]

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

# Precompute once for this run
common_kept_indices = compute_common_kept_indices(exps_path, methods=('DLG','iDLG'), metrics=('mse','loss'))

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

performance = []

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
                d['learning'] = conf['learning']['mode'] + '_' + str(d['seed']) # TODO: change to client_id
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
            d['mia_reference_rounds'] = None
            d['mia_reference_metric'] = None
            d['mia_target_metric'] = None
            d['mia_target_accuracy'] = np.nan

            # Paths
            sia_results_path = os.path.join(exp, 'sia_results.json')
            sia_max_path = os.path.join(exp, 'sia_max.json')
            mia_results_path = os.path.join(exp, 'mia_results.json')
            mia_summary_path = os.path.join(exp, 'mia_summary.json')
            mia_max_path = os.path.join(exp, 'mia_max.json')  # optional alt name
            dra_summary_path = os.path.join(exp, 'dra_results_summary.json')

            # --- MIA (accuracy) ---
            mia_selected_indices = None
            mia_meta = {}
            mia_metrics = None
            if os.path.exists(mia_results_path):
                try:
                    with open(mia_results_path, 'r') as f:
                        mia_results_data = json.load(f)
                    mia_metrics, mia_selected_indices, mia_meta = _compute_mia_metrics_from_results(
                        mia_results_data,
                        args.mia_metric,
                        metric_mode=args.metric_mode,
                    )
                    if mia_metrics is not None:
                        d['mia_whitebox_acc'] = mia_metrics.get('whitebox', np.nan)
                        d['mia_blackbox_acc'] = mia_metrics.get('blackbox', np.nan)
                        d['mia_shadow_acc'] = mia_metrics.get('blackbox_shadow', np.nan)
                except Exception:
                    mia_metrics = None
                    mia_selected_indices = None
                    mia_meta = {}

            if (mia_metrics is None) or (not isinstance(mia_metrics, dict)):
                def _legacy_mia_acc(mia_json, key):
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
                    d['mia_whitebox_acc'] = _legacy_mia_acc(mia_json, 'whitebox')
                    d['mia_blackbox_acc'] = _legacy_mia_acc(mia_json, 'blackbox')
                    d['mia_shadow_acc'] = _legacy_mia_acc(mia_json, 'blackbox_shadow')

            if isinstance(mia_meta, dict) and mia_meta:
                rounds_list = mia_meta.get("selected_rounds")
                if (not rounds_list) and mia_selected_indices is not None:
                    all_rounds_meta = mia_meta.get("all_rounds", [])
                    if not all_rounds_meta:
                        all_rounds_meta = list(range(1, len(mia_meta.get("accuracy_val_avg", [])) + 1))
                    rounds_list = []
                    for idx in mia_selected_indices:
                        if 0 <= idx < len(all_rounds_meta):
                            rounds_list.append(all_rounds_meta[idx])
                if rounds_list:
                    d['mia_reference_rounds'] = ", ".join(str(r) for r in rounds_list)

                metric_vals = mia_meta.get("selected_round_metric_values")
                target_metric_meta = mia_meta.get("target_metric")
                if (not metric_vals) and mia_selected_indices is not None:
                    series = mia_meta.get("accuracy_train_avg" if target_metric_meta == "train" else "accuracy_val_avg", [])
                    metric_vals = []
                    for idx in mia_selected_indices:
                        if 0 <= idx < len(series):
                            metric_vals.append(series[idx])
                if metric_vals:
                    formatted_vals = [f"{val:.4f}" for val in metric_vals if isinstance(val, (int, float, np.floating))]
                    if formatted_vals:
                        d['mia_reference_metric'] = ", ".join(formatted_vals)

                if target_metric_meta:
                    d['mia_target_metric'] = target_metric_meta
                target_acc_meta = mia_meta.get("target_accuracy")
                if target_acc_meta is not None:
                    d['mia_target_accuracy'] = target_acc_meta

            # --- SIA (accuracy) ---
            if os.path.exists(sia_results_path):
                try:
                    with open(sia_results_path, 'r') as f:
                        sia_results_data = json.load(f)
                    d['sia_acc'] = _compute_sia_metric_from_results(
                        sia_results_data,
                        default_indices=mia_selected_indices,
                        default_meta=mia_meta,
                        metric_mode=args.metric_mode,
                    )
                except Exception:
                    pass
            if np.isnan(d['sia_acc']) and os.path.exists(sia_max_path):
                try:
                    with open(sia_max_path, 'r') as f:
                        sia_json = json.load(f)
                    if 'max_sia_accuracy' in sia_json:
                        d['sia_acc'] = float(sia_json['max_sia_accuracy'])
                except Exception:
                    pass

            # --- DRA (MSE) recomputed on COMMON kept indices across experiments ---
            raw_dra_path = os.path.join(exp, 'dra_results_client_0.json')
            if not os.path.exists(raw_dra_path):
                raw_dra_path = os.path.join(exp, 'dra_results.json')

            d['dra_dlg_mse_mean']  = _dra_mean_over_indices(raw_dra_path, 'DLG',  'mse', common_kept_indices)
            d['dra_idlg_mse_mean'] = _dra_mean_over_indices(raw_dra_path, 'iDLG', 'mse', common_kept_indices)

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

            performance.append(d)

        except:
            pass


# Format the collected list of experiments into a DataFrame
performance = pd.DataFrame(performance)

# Fill empty graph entries by matching on dataset
for idx, row in performance.iterrows():
    if row['graph'] is None:
        # Find rows with the same dataset that have a non-None graph
        matching_rows = performance[(performance['dataset'] == row['dataset']) & 
                                   (performance['graph'].notna())]
        if not matching_rows.empty:
            # Use the first matching graph
            performance.at[idx, 'graph'] = matching_rows.iloc[0]['graph']

def format_results(row, graph, count_nan=False, task=None):
    values = [v for k,v in row.items() if k in row]
    if count_nan:
        return sum([1 for x in values if math.isnan(x)]) if values else 0
    else:
        if task is not None:
            values = values + [task]
        return np.mean([x for x in values if not math.isnan(x)]) if values else 0


# Aggregate the results in concept_acc and task_acc for each row.
# For the localized training we ELIMINATE the NaN values for the OOD concepts and compute the mean over the remaining concepts.
performance['concept_left_out'] = performance.apply(lambda x: format_results(x['concept_acc'], x['graph'], count_nan=True), axis=1)
performance['agg_concept'] = performance.apply(lambda x: format_results(x['concept_acc'], x['graph'], count_nan=False), axis=1)
performance['agg_label'] = performance.apply(lambda x: format_results(x['concept_acc'], x['graph'], count_nan=False, task=x['task_acc']), axis=1)

######### Dataset and model styles #########

def get_df_name(df):
    if df=='sachs':
        return 'Sachs'
    elif df=='asia':
        return 'Asia'
    elif df=='alarm':
        return 'Alarm'
    elif df=='hailfinder':
        return 'Hailfinder'
    elif df=='insurance':
        return 'Insurance'
    else:
        raise ValueError(f"Unknown dataset name: {df}")

# Rename the datasets in the performance dataframe
performance['dataset'] = performance['dataset'].apply(get_df_name)

marker_size = 14

# Define a dictionary to associate marker, name, and color to each model.
# If the experiment you run does not contain a model, just remove it from the dictionary.
# If you want to add a new model, just add it to the dictionary.
model_styles = {
    'cem': {'marker': 'P', 'name': 'CEM', 'color': 'tab:blue', 'size': marker_size},
    'cbm_linear': {'marker': '*', 'name': 'CBM+Linear', 'color': 'tab:red', 'size': marker_size},
    'cbm_mlp': {'marker': '^', 'name': 'CBM+MLP', 'color': 'tab:purple', 'size': marker_size},
    'blackbox': {'marker': 'o', 'name': 'BlackBox', 'color': 'tab:black', 'size': marker_size},
    'blackbox_multi': {'marker': 'o', 'name': 'BlackBox (Multi)', 'color': 'tab:grey', 'size': marker_size},
    'cgm': {'marker': 'D', 'name': 'CGM', 'color': 'tab:orange', 'size': marker_size},
    'c2bm': {'marker': 's', 'name': 'C2BM', 'color': 'tab:green', 'size': marker_size},
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

# Filter the performance dataframe to keep only the models in model_styles 
# and datasets in custom_order.
performance = performance[performance['model'].isin(model_styles.keys()) & performance['dataset'].isin(custom_order)]

# change the name of the model
performance['model'] = performance['model'].apply(lambda x: model_styles[x]['name'] if x in model_styles else x)

########## Task & Concept Accuracy Plot ##########

#df = performance.copy()
# Filter data for 'task' and 'concept'
task_df = performance.copy()
task_df = task_df.rename(columns={'task_acc': 'accuracy'})
concept_df = performance.copy()
concept_df = concept_df.rename(columns={'agg_concept': 'accuracy'})

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

complete_task_stats = task_stats.copy()
complete_concept_stats = concept_stats.copy()
complete_label_stats = label_stats.copy()

for learning in performance['learning'].unique():

    # Filter the task and concept stats to maintain only the rows with the current learning method
    task_stats = complete_task_stats[complete_task_stats['learning'] == learning]
    concept_stats = complete_concept_stats[complete_concept_stats['learning'] == learning]
    label_stats = complete_label_stats[complete_label_stats['learning'] == learning]

    print(f'\n\nLearning method: {learning}')

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

    # print('\n\nConcept Accuracy Table:')
    # print('-------------------')
    # print(final_table)

    # store the table in a csv file
    result_file = f'{visualization_folder}/{learning}/concept_accuracy.csv'
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    final_table.to_csv(result_file, index=True)

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


    ########## Privacy Attack Tables ##########
    # Keep only required columns if present
    privacy_cols = [
        'model', 'dataset', 'learning',
        'sia_acc',
        'mia_whitebox_acc', 'mia_blackbox_acc', 'mia_shadow_acc',
        'dra_dlg_mse_mean',
        'dra_idlg_mse_mean',
        'mia_target_metric',
        'mia_target_accuracy',
        'mia_reference_rounds',
        'mia_reference_metric',
    ]
    available_cols = [c for c in privacy_cols if c in performance.columns]
    privacy_df = performance[available_cols].copy()

    info_columns = ['mia_target_metric', 'mia_target_accuracy', 'mia_reference_rounds', 'mia_reference_metric']
    metric_cols = [c for c in available_cols if c not in ['model','dataset','learning'] + info_columns]
    info_cols_present = [c for c in info_columns if c in available_cols]

    drop_cols = metric_cols if metric_cols else info_cols_present
    if drop_cols:
        privacy_df = privacy_df.dropna(subset=drop_cols, how='all')

    if not privacy_df.empty and (metric_cols or info_cols_present):
        # Aggregate across folds (runs) → mean and std per (model, dataset, learning)
        mean_agg = {c: 'mean' for c in metric_cols}
        info_agg = {c: 'first' for c in info_cols_present}
        agg_dict = {}
        agg_dict.update(mean_agg)
        agg_dict.update(info_agg)
        grp = ['model', 'dataset', 'learning']
        privacy_mean = privacy_df.groupby(grp, as_index=False).agg(agg_dict)

        if metric_cols:
            std_agg  = {c: 'std'  for c in metric_cols}
            privacy_std  = privacy_df.groupby(grp, as_index=False).agg(std_agg).fillna(0)
        else:
            privacy_std = pd.DataFrame()

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
                if not privacy_std.empty and not sub_s.empty and 'model' in sub_s.columns:
                    sub_s = sub_s.set_index('model')
                else:
                    sub_s = pd.DataFrame(index=sub_m.index)

                # Build the printable table with formatted strings
                table_rows = {}
                for model_name in sub_m.index.unique():
                    row = {}
                    for col_title, mean_key, std_key in attack_columns:
                        is_acc = mean_key in acc_metrics
                        mean_val = sub_m.loc[model_name, mean_key] if mean_key in sub_m.columns else np.nan
                        std_val  = sub_s.loc[model_name, std_key] if (not sub_s.empty and std_key in sub_s.columns) else np.nan
                        # Handle duplicated index (multiple rows) by taking mean again
                        if isinstance(mean_val, pd.Series):
                            mean_val = float(mean_val.mean())
                        if isinstance(std_val, pd.Series):
                            std_val = float(std_val.mean())
                        row[col_title] = _fmt_cell(mean_val, std_val, is_accuracy=is_acc)

                    def _scalar(value):
                        if isinstance(value, pd.Series):
                            return value.iloc[0]
                        return value

                    rounds_val = _scalar(sub_m.loc[model_name, 'mia_reference_rounds']) if 'mia_reference_rounds' in sub_m.columns else None
                    metric_val = _scalar(sub_m.loc[model_name, 'mia_reference_metric']) if 'mia_reference_metric' in sub_m.columns else None
                    target_metric_val = _scalar(sub_m.loc[model_name, 'mia_target_metric']) if 'mia_target_metric' in sub_m.columns else None
                    target_acc_val = _scalar(sub_m.loc[model_name, 'mia_target_accuracy']) if 'mia_target_accuracy' in sub_m.columns else None

                    row['MIA rounds'] = rounds_val if (isinstance(rounds_val, str) and rounds_val.strip()) else 'N/A'

                    if isinstance(metric_val, str) and metric_val.strip():
                        parts = [p.strip() for p in metric_val.split(',') if p.strip()]
                        formatted = []
                        for part in parts:
                            try:
                                formatted.append(f"{float(part)*100:.2f}")
                            except ValueError:
                                formatted.append(part)
                        row['MIA ref acc (%)'] = '; '.join(formatted) if formatted else 'N/A'
                    else:
                        row['MIA ref acc (%)'] = 'N/A'

                    row['Target metric'] = target_metric_val if (isinstance(target_metric_val, str) and target_metric_val) else 'N/A'
                    if isinstance(target_acc_val, (int, float, np.floating)) and not np.isnan(target_acc_val):
                        row['Target acc (%)'] = f"{float(target_acc_val)*100:.2f}"
                    else:
                        row['Target acc (%)'] = 'N/A'

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



########## Intervention plots ##########

# Eliminate blackbox and blackbox_multi from the model style
model_styles = {k: v for k, v in model_styles.items() if k not in ['blackbox', 'blackbox_multi']}

# Eliminate blackbox and blackbox_multi from the performance dataframe
performance = performance[performance['model'].isin(['blackbox', 'blackbox_multi']) == False]

### Intervention plot for single c interventions on y ###
plot_single_c_on_y(performance, custom_order, model_styles, visualization_folder)

### Intervention plot for level interventions on y ###
#plot_level_interventions_on_y(performance, custom_order, model_styles, visualization_folder)

### Intervention plot for single ID interventions on y ###
plot_single_id_ood_on_y(performance, custom_order, model_styles, visualization_folder, id=True)

### Intervention plot for single OOD interventions on y ###
#plot_single_id_ood_on_y(performance, custom_order, model_styles, visualization_folder, id=False)

### Intervention plot for level interventions on c ###
#plot_level_interventions_on_c(performance, custom_order, model_styles, visualization_folder)













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
