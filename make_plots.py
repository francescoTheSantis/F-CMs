import pickle
import pandas as pd
import os
import time
import numpy as np
import plotly.io as pio
from src.plots import plot_intervention, plot_level_intervention

"""
task accuracy after invervention on each individual concept     
"""
title = {'celeba': 'CelebA',
         'colormnist': 'ColorMNIST',
         'colormnist_ood': 'ColorMNIST',
         'asia': 'Asia',
         'asia_true': 'Asia',
         'alarm': 'Alarm',
         'alarm_true': 'Alarm',
         'sachs': 'Sachs',
         'sachs_true': 'Sachs',
         'hailfinder': 'Hailfinder',
         'hailfinder_true': 'Hailfinder',
         'insurance': 'Insurance',
         'insurance_true': 'Insurance',
         'pneumothorax': 'Pneumothorax'
}

def cumulative_improvement(means, stds):
    """
    This function computes the cumulative improvement of the means and stds
    Args:
        means: the means of the metrics
        stds: the stds of the metrics
    Returns:
        cum_improvement: the cumulative improvement of the means and stds
    """
    for model in means.keys():
        values_means = np.array(list(means[model].values()))
        values_stds = np.array(list(stds[model].values()))
        for level in means[model].keys():
            means[model][level] = values_means[:level+1].sum()
            stds[model][level] = values_stds[:level+1].sum()
    return means, stds

folder = 'plots_DEF_cumulative'
os.makedirs(folder, exist_ok=True)
root_result_dir =   {   
                        # 'colormnist': { 
                        #     'blackbox':   [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [0,10,20,30,40]],
                        #     'cbm_linear': [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [1,11,21,31,41]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [2,12,22,32,42]],
                        #     'cem':        [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [3,13,23,33,43]],
                        #     'crm':        [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [4,14,24,34,44]]
                        #     },
                        # 'colormnist_ood': { 
                        #     'blackbox':   [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [5,15,25,35,45]],
                        #     'cbm_linear': [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [6,16,26,36,46]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [7,17,27,37,47]],
                        #     'cem':        [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [8,18,28,38,48]],
                        #     'crm':        [f'outputs/multirun/2025-01-27/21-28-23_colormnist/{i}' for i in [9,19,29,39,49]]
                        # },
                        'celeba': {       
                            'blackbox':   [f'outputs/multirun/2025-01-27/21-17-33_blackbox_celeba/{i}' for i in [0,1,2,3,4]],
                            'cbm_linear': [f'outputs/multirun/2025-01-27/02-27-30_celeba/{i}' for i in [1,11,21,31,41]],
                            'cbm_mlp':    [f'outputs/multirun/2025-01-27/02-27-30_celeba/{i}' for i in [2,12,22,32,42]],
                            'cem':        [f'outputs/multirun/2025-01-27/02-27-30_celeba/{i}' for i in [3,13,23,33,43]],
                            'crm':        [f'outputs/multirun/2025-01-27/02-27-30_celeba/{i}' for i in [4,14,24,34,44]]
                        },
                        # true graphs
                        # 'asia_true': { 
                        #     'blackbox':   [f'outputs/multirun/2025-01-27/19-56-39_blackbox_bn/{i}' for i in [0,5,10,15,20]],
                        #     'cbm_linear': [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [1,26,51,76,101]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [2,27,52,77,102]],
                        #     'cem':        [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [3,28,53,78,103]],
                        #     'crm':        [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [4,29,54,79,104]]
                        # },
                        # 'sachs_true': { 
                        #     'blackbox':   [f'outputs/multirun/2025-01-27/19-56-39_blackbox_bn/{i}' for i in [1,6,11,16,21]],
                        #     'cbm_linear': [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [6,31,56,81,106]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [7,32,57,82,107]],
                        #     'cem':        [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [8,33,58,83,108]],
                        #     'crm':        [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [9,34,59,84,109]]
                        # },
                        # 'insurance_true': { 
                        #     'blackbox':   [f'outputs/multirun/2025-01-27/19-56-39_blackbox_bn/{i}' for i in [2,7,12,17,22]],
                        #     'cbm_linear': [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [11,36,61,86,111]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [12,37,62,87,112]],
                        #     'cem':        [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [13,38,63,88,113]],
                        #     'crm':        [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [14,39,64,89,114]]
                        # },
                        # 'alarm_true': { 
                        #     'blackbox':   [f'outputs/multirun/2025-01-27/19-56-39_blackbox_bn/{i}' for i in [3,8,13,18,23]],
                        #     'cbm_linear': [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [16,41,66,91,116]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [17,42,67,92,117]],
                        #     'cem':        [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [18,43,68,93,118]],
                        #     'crm':        [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [19,44,69,94,119]]
                        # },
                        # 'hailfinder_true': { 
                        #     'blackbox':   [f'outputs/multirun/2025-01-27/19-56-39_blackbox_bn/{i}' for i in [4,9,14,19,24]],
                        #     'cbm_linear': [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [21,46,71,96,121]],
                        #     'cbm_mlp':    [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [22,47,72,97,122]],
                        #     'cem':        [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [23,48,73,98,123]],
                        #     'crm':        [f'outputs/multirun/2025-01-27/12-08-38_bn_true/{i}' for i in [24,49,74,99,124]]
                        # },
                        # # learned graph
                        'asia': { 
                            'blackbox':   [f'outputs/multirun/2025-01-27/19-56-39_blackbox_bn/{i}' for i in [0,5,10,15,20]],
                            'cbm_linear': [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [1,26,51,76,101]],
                            'cbm_mlp':    [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [2,27,52,77,102]],
                            'cem':        [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [3,28,53,78,103]],
                            'crm':        [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [4,29,54,79,104]]
                        },
                        'sachs': { 
                            'blackbox':   [f'outputs/multirun/2025-01-27/19-56-39_blackbox_bn/{i}' for i in [1,6,11,16,21]],
                            'cbm_linear': [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [6,31,56,81,106]],
                            'cbm_mlp':    [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [7,32,57,82,107]],
                            'cem':        [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [8,33,58,83,108]],
                            'crm':        [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [9,34,59,84,109]]
                        },
                        'insurance': { 
                            'blackbox':   [f'outputs/multirun/2025-01-27/19-56-39_blackbox_bn/{i}' for i in [2,7,12,17,22]],
                            'cbm_linear': [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [11,36,61,86,111]],
                            'cbm_mlp':    [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [12,37,62,87,112]],
                            'cem':        [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [13,38,63,88,113]],
                            'crm':        [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [14,39,64,89,114]]
                        },
                        'alarm': { 
                            'blackbox':   [f'outputs/multirun/2025-01-27/19-56-39_blackbox_bn/{i}' for i in [3,8,13,18,23]],
                            'cbm_linear': [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [16,41,66,91,116]],
                            'cbm_mlp':    [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [17,42,67,92,117]],
                            'cem':        [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [18,43,68,93,118]],
                            'crm':        [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [19,44,69,94,119]]
                        },
                        'hailfinder': { 
                            'blackbox':   [f'outputs/multirun/2025-01-27/19-56-39_blackbox_bn/{i}' for i in [4,9,14,19,24]],
                            'cbm_linear': [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [21,46,71,96,121]],
                            'cbm_mlp':    [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [22,47,72,97,122]],
                            'cem':        [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [23,48,73,98,123]],
                            'crm':        [f'outputs/multirun/2025-01-27/12-08-56_bn_learned/{i}' for i in [24,49,74,99,124]]
                        },
                        'pneumothorax': {
                            'blackbox':   [f'outputs/multirun/2025-01-29/00-44-17_pneumo/{i}' for i in [0,5,10,15,20]],
                            'cbm_linear': [f'outputs/multirun/2025-01-29/00-44-17_pneumo/{i}' for i in [1,6,11,16,21]],
                            'cbm_mlp':    [f'outputs/multirun/2025-01-29/00-44-17_pneumo/{i}' for i in [2,7,12,17,22]],
                            'cem':        [f'outputs/multirun/2025-01-29/00-44-17_pneumo/{i}' for i in [3,8,13,18,23]],
                            'crm':        [f'outputs/multirun/2025-01-29/00-44-17_pneumo/{i}' for i in [4,9,14,19,24]]
                        }
                    }

std_mean = True
std_95 = True
cumulative = True

label_acc_results = pd.DataFrame(index=['blackbox', 'cbm_linear', 'cbm_mlp', 'cem', 'crm'], 
                            columns=root_result_dir.keys())
task_acc_results = pd.DataFrame(index=['blackbox', 'cbm_linear', 'cbm_mlp', 'cem', 'crm'], 
                                columns=root_result_dir.keys())
# acc_results_noisy = pd.DataFrame(index=['cbm_linear', 'cbm_mlp', 'cem', 'crm'], 
#                                     columns=root_result_dir.keys())

for dataset in root_result_dir.keys():
    print(f'----{dataset}-----')
    root_result_dir_d = root_result_dir[dataset]

    # average accuracy
    # print('Average accuracy')
    for model in root_result_dir_d.keys():
        # print(f'--{model}--')
        average = []
        average_noisy = []
        average_task = []
        for i, run in enumerate(root_result_dir_d[model]):
            single = []
            task_acc = pickle.load(open(run + '/results/y_accuracy.pkl', 'rb'))['_baseline']
            single.append(task_acc)
            # and all valid concepts
            c_accuracy = pickle.load(open(run + '/results/c_accuracy.pkl', 'rb'))
            valid_concepts = [k for k, v in pickle.load(open(root_result_dir_d['crm'][i] + '/results/c_accuracy.pkl', 'rb')).items() if not np.isnan(v)]
            for c in valid_concepts:
                single.append(c_accuracy[c])
            average.append(np.array(single).mean())
            average_task.append(task_acc)

            # if model != 'blackbox':
            #     average_noisy.append(pickle.load(open(run + '/results/single_c_interventions_on_y.pkl', 'rb'))['_baseline'])

        # compute the average
        std = np.array(average).std(ddof=1)
        # std_noisy = np.array(average_noisy).std(ddof=1)
        if std_mean:
            std = std / np.sqrt(len(average))
        if std_95:
            std = 1.96 * std
            # std_noisy = std_noisy / np.sqrt(len(average))
        label_acc_results.loc[model, dataset] = f'{round(np.array(average).mean()*100,2)} ± {round(std*100,2)}'

        std = np.array(average_task).std(ddof=1)
        if std_mean:
            std = std / np.sqrt(len(average_task))
        if std_95:
            std = 1.96 * std
        task_acc_results.loc[model, dataset] = f'{round(np.array(average_task).mean()*100,2)} ± {round(std*100,2)}'
        # acc_results_noisy.loc[model, dataset] = f'{round(np.array(average_noisy).mean()*100,2)} ± {round(std_noisy*100,2)}'   






    res_to_plot = {model:{} for model in root_result_dir_d.keys()}
    std_to_plot = {model:{} for model in root_result_dir_d.keys()}
    # single concept interventions
    # print('Single concept interventions')
    for model in res_to_plot.keys():
        # print(f'--{model}--')
        average = []
        if model != 'blackbox':
            for i, run in enumerate(root_result_dir_d[model]):
                y_baseline = pickle.load(open(run + '/results/y_accuracy.pkl', 'rb'))['_baseline']
                # print(f"({dataset}) (run {i+1}) Baseline y test accuracy for {model}: {y_baseline}")
                y_acc = pickle.load(open(run + '/results/single_c_interventions_on_y.pkl', 'rb'))
                y_nosy_baseline = y_acc['_baseline']
                # print(f"({dataset}) (run {i+1}) Baseline y noisy-test accuracy for {model}: {y_nosy_baseline}")
                y_delta = {}
                for c_name in y_acc.keys():
                    if c_name != '_baseline':
                        y_delta[c_name] = ((y_acc[c_name] - y_nosy_baseline)/y_nosy_baseline)*100.
                average.append(y_delta)
            # compute the average
            res_to_plot[model] = {key: np.array([d[key] for d in average]).mean() for key in average[0].keys()}
            std_to_plot[model] = {key: np.array([d[key] for d in average]).std(ddof=1) for key in average[0].keys()}
            if std_mean:
                std_to_plot[model] = {key: v / np.sqrt(len(average)) for key, v in std_to_plot[model].items()}
            if std_95:
                std_to_plot[model] = {key: 1.96 * v for key, v in std_to_plot[model].items()}
    fig = plot_intervention(res_to_plot, # accuracy delta after noise wrt to true baseline
                            std_to_plot,
                            f'{title[dataset]}')
    # write a random plot before the real one (to avoid weird box in the pdf)
    pio.write_image(fig, f"{folder}/{dataset}_SI_on_y.pdf")
    # wait 0.5 seconds
    time.sleep(1)
    # save a figure of 600dpi, with 2.0 inches, and  height 0.75inches
    pio.write_image(fig, f"{folder}/{dataset}_SI_on_y.pdf", width=2.6*600, height=1.5*600, scale=1)

    
    res_to_plot = {model:{} for model in root_result_dir_d.keys()}
    std_to_plot = {model:{} for model in root_result_dir_d.keys()}
    # level interventions on y
    # print('Level interventions on y')
    for model in res_to_plot.keys():
        # print(f'--{model}--')
        average = []
        if model != 'blackbox':
            for i, run in enumerate(root_result_dir_d[model]):
                file = pickle.load(open(run + '/graph.pkl', 'rb'))
                policy = file['policy']
                # print(f"({dataset}) (run {i+1}) Policy for {model}: {policy}")
                y_baseline = pickle.load(open(run + '/results/y_accuracy.pkl', 'rb'))['_baseline']
                # print(f"({dataset}) (run {i+1}) Baseline y test accuracy for {model}: {y_baseline}")
                y_acc = pickle.load(open(run + '/results/level_interventions_on_y.pkl', 'rb'))
                y_nosy_baseline = y_acc['level 0']
                assert y_nosy_baseline == pickle.load(open(run + '/results/single_c_interventions_on_y.pkl', 'rb'))['_baseline']
                # print(f"({dataset}) (run {i+1}) Baseline y noisy-test accuracy for {model}: {y_nosy_baseline}")
                y_delta = {}
                for name in y_acc.keys():
                    level_number = int(name.split(' ')[-1])
                    level = policy[level_number-1]
                    label = level_number #f'{[c_names[i] for i in level]}'
                    y_delta[label] = ((y_acc[name] - y_nosy_baseline)/y_nosy_baseline)*100. # y_acc[name]*100
                average.append(y_delta)
            # compute the average
            res_to_plot[model] = {key: np.array([d[key] for d in average]).mean() for key in average[0].keys()}
            std_to_plot[model] = {key: np.array([d[key] for d in average]).std(ddof=1) for key in average[0].keys()}
            if std_mean:
                std_to_plot[model] = {key: v / np.sqrt(len(average)) for key, v in std_to_plot[model].items()}
            if std_95:
                std_to_plot[model] = {key: 1.96 * v for key, v in std_to_plot[model].items()}
    res_to_plot['blackbox'] = {key: 0 for key in res_to_plot['crm'].keys()}
    std_to_plot['blackbox'] = {key: 0 for key in std_to_plot['crm'].keys()}
    # compute cumulative improvement
    if cumulative:
        res_to_plot, std_to_plot = cumulative_improvement(res_to_plot, std_to_plot)
    fig = plot_level_intervention(res_to_plot, 
                                  std_to_plot,
                                  'Cumul. improv. (%) on task acc.',
                                  f'{title[dataset]}')
    # write a random plot before the real one (to avoid weird box in the pdf)
    pio.write_image(fig, f"{folder}/{dataset}_LI_on_y.pdf")
    # wait 0.5 seconds
    time.sleep(1)   
    pio.write_image(fig, f"{folder}/{dataset}_LI_on_y.pdf", width=2.0*600, height=2.0*600, scale=1)



    res_to_plot = {model:{} for model in root_result_dir_d.keys()}
    std_to_plot = {model:{} for model in root_result_dir_d.keys()}
    # level interventions on concepts
    # print('Level interventions on concepts')
    for model in res_to_plot.keys():
        # print(f'--{model}--')
        average = []
        if model != 'blackbox':
            for i, run in enumerate(root_result_dir_d[model]):
                file = pickle.load(open(run + '/graph.pkl', 'rb'))
                policy = file['policy']
                concepts = file['concepts']
                # print(f"({dataset}) (run {i+1}) Policy for {model}: {policy}")
                c_baseline = pickle.load(open(run + '/results/c_accuracy.pkl', 'rb'))
                # print(f"({dataset}) (run {i+1}) Baseline c test accuracy for {model}: {np.array(list(c_baseline.values())).mean()}")
                c_acc = pickle.load(open(run + '/results/level_interventions_on_c.pkl', 'rb'))
                c_noisy_baseline = {key.split('/')[1].split('child ')[-1]: c_acc[key] 
                                    for key in c_acc.keys() if 'level 0' in key}
                # print(f"({dataset}) (run {i+1}) Baseline c noisy-test accuracy for {model}: {np.array(list(c_noisy_baseline.values())).mean()}")
                c_delta = {}
                for level in range(len(policy)):
                    childs = [key.split('child ')[-1] for key in c_acc.keys() if f'level {level}' in key]
                    temp = [(c_acc[f'level {level}/child {c_name}'] - c_noisy_baseline[c_name])/c_noisy_baseline[c_name]*100.
                            for c_name in childs]
                    c_delta[level] = sum(temp)/len(temp)
                average.append(c_delta)
            # compute the average
            res_to_plot[model] = {key: np.array([d[key] for d in average]).mean() for key in average[0].keys()}
            std_to_plot[model] = {key: np.array([d[key] for d in average]).std(ddof=1) for key in average[0].keys()}
            if std_mean:
                std_to_plot[model] = {key: v / np.sqrt(len(average)) for key, v in std_to_plot[model].items()}
            if std_95:
                std_to_plot[model] = {key: 1.96 * v for key, v in std_to_plot[model].items()}
    res_to_plot['blackbox'] = {key: 0 for key in res_to_plot['crm'].keys()}
    std_to_plot['blackbox'] = {key: 0 for key in std_to_plot['crm'].keys()}
    if cumulative:
        res_to_plot, std_to_plot = cumulative_improvement(res_to_plot, std_to_plot)
    fig = plot_level_intervention(res_to_plot, 
                                  std_to_plot,
                                  'Cumul. improv. (%) on concept acc.',
                                  f'{title[dataset]}')
    # write a random plot before the real one (to avoid weird box in the pdf)
    pio.write_image(fig, f"{folder}/{dataset}_LI_on_c.pdf")
    # wait 0.5 seconds
    time.sleep(1)   
    pio.write_image(fig, f"{folder}/{dataset}_LI_on_c.pdf", width=2.0*600, height=2.0*600, scale=1)





    res_to_plot = {model:{} for model in root_result_dir_d.keys()}
    std_to_plot = {model:{} for model in root_result_dir_d.keys()}
    # level interventions on Y + concepts
    # print('Level interventions on concepts')
    for model in res_to_plot.keys():
        average = []
        if model != 'blackbox':
            for i, run in enumerate(root_result_dir_d[model]):
                file = pickle.load(open(run + '/graph.pkl', 'rb'))
                policy = file['policy']
                concepts = file['concepts']

                # extract y
                y_baseline = pickle.load(open(run + '/results/y_accuracy.pkl', 'rb'))['_baseline']
                y_acc = pickle.load(open(run + '/results/level_interventions_on_y.pkl', 'rb'))
                y_nosy_baseline = y_acc['level 0']

                # extract c
                c_baseline = pickle.load(open(run + '/results/c_accuracy.pkl', 'rb'))
                # print(f"({dataset}) (run {i+1}) Baseline c test accuracy for {model}: {np.array(list(c_baseline.values())).mean()}")
                c_acc = pickle.load(open(run + '/results/level_interventions_on_c.pkl', 'rb'))
                c_noisy_baseline = {key.split('/')[1].split('child ')[-1]: c_acc[key] 
                                    for key in c_acc.keys() if 'level 0' in key}

                assert y_nosy_baseline == pickle.load(open(run + '/results/single_c_interventions_on_y.pkl', 'rb'))['_baseline']
                single = []
                c_delta = {}
                for level in range(len(policy)+1):
                    # append task improvement
                    temp = [((y_acc[f'level {level}'] - y_nosy_baseline)/y_nosy_baseline)*100]    #[y_acc[f'level {level}']*100]
                    # append concepts improvement
                    if level < len(policy):
                        childs = [key.split('child ')[-1] for key in c_acc.keys() if f'level {level}' in key]
                        temp += [(c_acc[f'level {level}/child {c_name}'] - c_noisy_baseline[c_name])/c_noisy_baseline[c_name]*100.   #[c_acc[f'level {level}/child {c_name}']*100
                                for c_name in childs]
                    c_delta[level] = sum(temp)/len(temp)
                average.append(c_delta)
            # compute the average
            res_to_plot[model] = {key: np.array([d[key] for d in average]).mean() for key in average[0].keys()}
            std_to_plot[model] = {key: np.array([d[key] for d in average]).std(ddof=1) for key in average[0].keys()}
            if std_mean:
                std_to_plot[model] = {key: v / np.sqrt(len(average)) for key, v in std_to_plot[model].items()}
            if std_95:
                std_to_plot[model] = {key: 1.96 * v for key, v in std_to_plot[model].items()}
    res_to_plot['blackbox'] = {key: 0 for key in res_to_plot['crm'].keys()}
    std_to_plot['blackbox'] = {key: 0 for key in std_to_plot['crm'].keys()}
    if cumulative:
        res_to_plot, std_to_plot = cumulative_improvement(res_to_plot, std_to_plot)
    fig = plot_level_intervention(res_to_plot, 
                                  std_to_plot,
                                  'Cumul. improv. (%) on label acc.',
                                  f'{title[dataset]}')
    # write a random plot before the real one (to avoid weird box in the pdf)
    pio.write_image(fig, f"{folder}/{dataset}_LI_on_both.pdf")
    # wait 0.5 seconds
    time.sleep(1)   
    pio.write_image(fig, f"{folder}/{dataset}_LI_on_both.pdf", width=2.0*600, height=2.0*600, scale=1)




print('-- Label accuracy (concepts + task) --')
print(label_acc_results)
print('')
print('-- Task accuracy --')
print(task_acc_results)

# print('-- after noise is injected at test time --')
# print(acc_results_noisy)

