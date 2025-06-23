import pickle
import pandas as pd
import os
import time
import numpy as np
import plotly.io as pio
from src.plots import plot_intervention, plot_level_intervention
import yaml

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

# List the paths containing the results
paths = [
    "/home/fdesantis/projects/Federated-C2BM/outputs/multirun/2025-06-23/09-26-22",
]

###### Collect results ######

possible_datasets = title.keys()
possible_learning_modes = ['localized', 'federated', 'centralized']
possible_models = ['blackbox', 'cbm_linear', 'cbm_mlp', 'cem', 'crm']

exps_path = []
lmr_paths = []
for path in paths:
    exps = os.listdir(path)
    exps_path += [os.path.join(path, exp) for exp in exps if 'multirun' not in exp]

root_result_dir = {
    possible_learning_mode: {dataset: {model: [] for model in possible_models} for dataset in possible_datasets} for possible_learning_mode in possible_learning_modes
}

for exp_path in exps_path:
    # Load the config file
    with open(os.path.join(exp_path, '.hydra', 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    conf_dataset = config['dataset']['name']
    conf_model = config['model']['name']
    conf_learning = config['learning']['mode']
    root_result_dir[conf_learning][conf_dataset][conf_model] += [exp_path]


std_mean = True
std_95 = True
cumulative = True


for learning_mode in possible_learning_modes:
    label_acc_results = pd.DataFrame(index=possible_models, 
                                columns=possible_datasets)
    task_acc_results = pd.DataFrame(index=possible_models, 
                                    columns=possible_datasets)
    # acc_results_noisy = pd.DataFrame(index=['cbm_linear', 'cbm_mlp', 'cem', 'crm'], 
    #                                     columns=root_result_dir.keys())
    for dataset in possible_datasets:
        print(f'----{dataset}-----')
        root_result_dir_d = root_result_dir[learning_mode][dataset]

        # average accuracy
        # print('Average accuracy')
        for model in possible_models:
            try:
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
            except:
                print(f'Data for {model}, {dataset} not found.')

        res_to_plot = {model:{} for model in root_result_dir_d.keys()}
        std_to_plot = {model:{} for model in root_result_dir_d.keys()}
        # single concept interventions
        # print('Single concept interventions')
        for model in res_to_plot.keys():
            try:
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
            except:
                print(f'Data for {model}, {dataset} not found.')

        fig = plot_intervention(res_to_plot, # accuracy delta after noise wrt to true baseline
                                std_to_plot,
                                f'{title[dataset]}')
        # write a random plot before the real one (to avoid weird box in the pdf)
        sub_folder = f'{folder}/{learning_mode}'
        os.makedirs(sub_folder, exist_ok=True)
        pio.write_image(fig, f"{folder}/{learning_mode}/{dataset}_SI_on_y.pdf")
        # wait 0.5 seconds
        time.sleep(1)
        # save a figure of 600dpi, with 2.0 inches, and  height 0.75inches
        pio.write_image(fig, f"{folder}/{learning_mode}/{dataset}_SI_on_y.pdf", width=2.6*600, height=1.5*600, scale=1)
        
        res_to_plot = {model:{} for model in root_result_dir_d.keys()}
        std_to_plot = {model:{} for model in root_result_dir_d.keys()}
        # level interventions on y
        # print('Level interventions on y')
        for model in possible_models:
            # print(f'--{model}--')
            average = []
            try:
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
            except:
                print(f'Data for {model}, {dataset} not found.')

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
        for model in possible_models:
            try:
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
            except:
                print(f'Data for {model}, {dataset} not found.')

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
        for model in possible_models:
            average = []
            try:
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
            except:
                print(f'Data for {model}, {dataset} not found.')
                
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

