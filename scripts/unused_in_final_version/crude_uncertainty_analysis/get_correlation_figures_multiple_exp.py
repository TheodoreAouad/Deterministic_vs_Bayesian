"""
This file computes the graph of accuracy (y) against uncertainty (x). It is used to get a figure across multiple
experiments.
To use:
- change the 'to change' parameters
- run the file in console
"""

import pathlib
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_from_file, get_file_and_dir_path_in_dir, get_unc_key
from scripts.utils import get_deadzone

#### TO CHANGE ######

group_nbs = ['205', '207', '208']
# group_nbs = ['193','209', '195']
show_fig = False
save_fig = True
info_supp = '_few_shot'
#####################

nb_of_exps = len(group_nbs)
save_path = pathlib.Path(f'results/correlations_figures/few_shot')
save_path.mkdir(parents=True, exist_ok=True)


def get_all_path_exps(group_nbs, result_dir):
    all_path_exps = []
    for group_nb in group_nbs:
        all_path_exps += list(set(get_file_and_dir_path_in_dir(join(result_dir, group_nb))[1]))
    return all_path_exps


results = pd.DataFrame(columns=['exp_nb', 'exp_type', 'loss_type', 'nb_of_data', 'acc', 'vr', 'pe', 'mi'])

result_dir = 'polyaxon_results/groups/'
all_exps = get_all_path_exps(group_nbs, result_dir)

for exp in all_exps:
    try:
        all_results = load_from_file(join(exp, 'results.pkl'))
        arguments = load_from_file(join(exp, 'arguments.pkl'))
    except FileNotFoundError:
        print(exp, 'not found')
        continue
    exp_nb = exp.split('/')[-1]
    if 'split_labels' in arguments.keys():
        exp_type = 'unseen_classes'
    elif 'dataset' in arguments.keys():
        exp_type = 'unseen_dataset'
    else:
        exp_type = 'random'
    results = results.append(pd.DataFrame.from_dict({
        'exp_nb': [exp_nb],
        'exp_type': [exp_type],
        'loss_type': [arguments['loss_type']],
        'nb_of_data': [all_results[get_unc_key(all_results, 'nb_of_data')]][0],
        'acc': [all_results['eval accuracy'][0].mean()],
        'vr': [all_results[get_unc_key(all_results.columns, 'seen vr')][0].mean()],
        'pe': [all_results[get_unc_key(all_results.columns, 'seen pe')][0].mean()],
        'mi': [all_results[get_unc_key(all_results.columns, 'seen mi')][0].mean()],
    }))
    results = results.append(pd.DataFrame.from_dict({
        'exp_nb': [exp_nb],
        'exp_type': [exp_type],
        'loss_type': [arguments['loss_type']],
        'nb_of_data': [all_results[get_unc_key(all_results, 'nb_of_data')]][0],
        'acc': [0],
        'vr': [all_results[get_unc_key(all_results.columns, 'unseen vr')][0].mean()],
        'pe': [all_results[get_unc_key(all_results.columns, 'unseen pe')][0].mean()],
        'mi': [all_results[get_unc_key(all_results.columns, 'unseen mi')][0].mean()],
    }))

loss_types = ['exp', 'uniform', 'criterion']
useful_arguments = {key: value for (key, value) in arguments.items() if key not in [
    'dataset',
    'split_train',
    'loss_type',
]}

# TODO: factorize the plotting using object API of matplotlib
for exp_type in ['random', 'unseen_classes', 'unseen_dataset']:
    results_of_type = results[results['exp_type'] == exp_type]
    if len(results_of_type) == 0:
        continue

    plt.figure(figsize=(15, 15))
    plt.suptitle('Accuracy vs uncertainty, size of dataset' + f'\n Exp Type: {exp_type}, {useful_arguments}', wrap=True)
    for i, loss_type in enumerate(loss_types):
        results_of_loss = results_of_type[results_of_type['loss_type'] == loss_type]
        plt.subplot(3, 3, 3 * i + 1)
        plt.scatter(results_of_loss['vr'], results_of_loss['acc'], c=results_of_loss['nb_of_data'])
        cbar = plt.colorbar()
        cbar.set_label('Size of trainset', rotation=270)
        plt.ylabel('acc')
        plt.xlabel('unc')
        max_seen, min_unseen, deadzone = get_deadzone(results_of_loss["vr"], results_of_loss["acc"] == 0)
        plt.title(
            f'VR - {loss_type}. deadzone: {round(deadzone, 2)}')
        if max_seen < min_unseen:
            plt.axvspan(max_seen, min_unseen, color='green', alpha=0.2)
        else:
            plt.axvspan(min_unseen, max_seen, color='red', alpha=0.2)

        plt.subplot(3, 3, 3 * i + 2)
        plt.scatter(results_of_loss['pe'], results_of_loss['acc'], c=results_of_loss['nb_of_data'])
        max_seen, min_unseen, deadzone = get_deadzone(results_of_loss["pe"], results_of_loss["acc"] == 0)
        plt.title(
            f'PE - {loss_type}. deadzone: {round(deadzone, 2)}')
        if max_seen < min_unseen:
            plt.axvspan(max_seen, min_unseen, color='green', alpha=0.2)
        else:
            plt.axvspan(min_unseen, max_seen, color='red', alpha=0.2)
        cbar.set_label('Size of trainset', rotation=270)
        plt.ylabel('acc')
        plt.xlabel('unc')
        plt.subplot(3, 3, 3 * i + 3)
        plt.scatter(results_of_loss['mi'], results_of_loss['acc'], c=results_of_loss['nb_of_data'])
        max_seen, min_unseen, deadzone = get_deadzone(results_of_loss["mi"], results_of_loss["acc"] == 0)
        plt.title(
            f'MI - {loss_type}. deadzone: {round(deadzone, 2)}')
        if max_seen < min_unseen:
            plt.axvspan(max_seen, min_unseen, color='green', alpha=0.2)
        else:
            plt.axvspan(min_unseen, max_seen, color='red', alpha=0.2)
        cbar.set_label('Size of trainset', rotation=270)
        plt.ylabel('acc')
        plt.xlabel('unc')
    if save_fig:
        (save_path / exp_type).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / exp_type / ('accuracy_vs_unc_with_size_of_dataset' + info_supp + '.png'))
    if show_fig:
        plt.show()

    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Accuracy vs size of dataset' + f'\n Exp Type: {exp_type}, {useful_arguments}', wrap=True)
    for i, loss_type in enumerate(loss_types):
        results_of_loss = results_of_type[results_of_type['loss_type'] == loss_type]
        results_of_loss = results_of_loss[results_of_loss['acc'] != 0]
        plt.subplot(1, 3, i + 1)
        plt.scatter(results_of_loss['nb_of_data'], results_of_loss['acc'])
        plt.title(f'{loss_type}')
        plt.ylabel('acc')
        plt.xlabel('dataset size')
    if save_fig:
        (save_path / exp_type).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / exp_type / ('accuracy_vs_size_of_dataset' + info_supp + '.png'))
    if show_fig:
        plt.show()

    plt.figure(figsize=(20, 14))
    plt.suptitle(f'Uncertainty vs size of dataset' + f'\n Exp Type: {exp_type}, {useful_arguments}', wrap=True)
    for i, loss_type in enumerate(loss_types):
        results_of_loss = results_of_type[results_of_type['loss_type'] == loss_type]
        seen_or_unseen = results_of_loss['acc'] != 0
        plt.subplot(3, 3, 3 * i + 1)
        plt.scatter(results_of_loss['nb_of_data'], results_of_loss['vr'], c=seen_or_unseen, alpha=0.5, )
        plt.ylabel('unc')
        plt.xlabel('size of trainset')
        plt.title(f'VR - {loss_type}')
        cbar = plt.colorbar()
        cbar.set_label('Seen 1 / Unseen 0', rotation=270)
        plt.subplot(3, 3, 3 * i + 2)
        plt.scatter(results_of_loss['nb_of_data'], results_of_loss['pe'], c=seen_or_unseen, alpha=0.5, )
        plt.title(f'PE - {loss_type}')
        plt.ylabel('unc')
        plt.xlabel('size of trainset')
        cbar = plt.colorbar()
        cbar.set_label('Seen 1 / Unseen 0', rotation=270)
        plt.subplot(3, 3, 3 * i + 3)
        plt.scatter(results_of_loss['nb_of_data'], results_of_loss['mi'], c=seen_or_unseen, alpha=0.5, )
        plt.title(f'MI - {loss_type}')
        plt.ylabel('unc')
        plt.xlabel('size of trainset')
        cbar = plt.colorbar()
        cbar.set_label('Seen 1 / Unseen 0', rotation=270)
    if save_fig:
        (save_path / exp_type).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / exp_type / ('unc_vs_size_of_dataset' + info_supp + '.png'))
    if show_fig:
        plt.show()
