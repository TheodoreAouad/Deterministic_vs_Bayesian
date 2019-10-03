"""
This file performs the computation to compare a model to the tables of Recent Advances in Open Set Recognition: A
Survey, table IV: COMPARISON AMONG THE REPRESENTATIVE OSR METHODS USING DEPTH FEATURES.

For each prediction, we try if it is from the training set or not. We compute the
Area Under ROC,  True Positive Rate against False Positive Rate

"""
import os
import pathlib

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from scripts.utils import get_trained_model_and_args_and_groupnb, get_evalloader_seen, get_evalloader_unseen, \
    get_determinism
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_all_uncertainty_measures_not_bayesian, get_all_uncertainty_measures, \
    get_predictions_from_multiple_tests
from src.utils import save_to_file

# exp_nbs = [20701, 20701, 20702, 20703, 20704]    # cifar10 criterion
# exp_nbs = [20695, 20696, 20697, 20698, 20699]    # cifar10 uniform
# exp_nbs = [20694, 20693, 20692, 20691, 20690]    # cifar10 exponential
# exp_nbs = [20724, 20723, 20722, 20725, 20726]    # cifar10 determinist
# exp_nbs = [20709, 20707, 20706, 20708, 20705]    # mnist criterion
# exp_nbs = [20715, 20716, 20718, 20719, 20717]    # mnist uniform
exp_nbs = [20712, 20713, 20714, 20711, 20710]    # mnist exponential

path_to_res = 'polyaxon_results/groups'
number_of_tests = 20
is_determinist = exp_nbs == [20724, 20723, 20722, 20725, 20726]
only_ood = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

if is_determinist:
    number_of_tests = 1
    get_uncs = get_all_uncertainty_measures_not_bayesian
    unc_names = ['sr', 'pe']
else:
    get_uncs = get_all_uncertainty_measures
    unc_names = ['sr', 'vr', 'pe', 'mi']
path_to_outputs = pathlib.Path(f'temp/softmax_outputs/{number_of_tests}')

all_xs = dict({k:[] for k in unc_names})
all_ys = dict({k:[] for k in unc_names})
all_uncs = {}

for exp_nb in exp_nbs:

    bay_net_trained, arguments, _ = get_trained_model_and_args_and_groupnb(exp_nb, path_to_res)
    

    if os.path.exists(path_to_outputs / f'{exp_nb}/true_labels_seen.pt'):
        true_labels_seen = torch.load(path_to_outputs / f'{exp_nb}/true_labels_seen.pt')
        all_outputs_seen = torch.load(path_to_outputs / f'{exp_nb}/all_outputs_seen.pt')
        all_outputs_unseen = torch.load(path_to_outputs / f'{exp_nb}/all_outputs_unseen.pt')

    else:
        (path_to_outputs / f'{exp_nb}').mkdir(exist_ok=True, parents=True)
        evalloader_seen = get_evalloader_seen(arguments)
        # BE CAREFUL: in the paper, the process is tested on the enterity of the unseen classes
        evalloader_unseen = get_evalloader_unseen(arguments)
        true_labels_seen, all_outputs_seen = eval_bayesian(
            model=bay_net_trained,
            evalloader=evalloader_seen,
            number_of_tests=number_of_tests,
            return_accuracy=False,
            device=device,
            verbose=True,
        )

        _, all_outputs_unseen = eval_bayesian(
            model=bay_net_trained,
            evalloader=evalloader_unseen,
            number_of_tests=number_of_tests,
            device=device,
            verbose=True,
        )

        torch.save(true_labels_seen, path_to_outputs / f'{exp_nb}/true_labels_seen.pt')
        torch.save(all_outputs_seen, path_to_outputs / f'{exp_nb}/all_outputs_seen.pt')
        torch.save(all_outputs_unseen, path_to_outputs / f'{exp_nb}/all_outputs_unseen.pt')

    preds = get_predictions_from_multiple_tests(all_outputs_seen).float()

    all_outputs_true = all_outputs_seen
    all_outputs_false = all_outputs_unseen
    if not only_ood:
        all_outputs_true = all_outputs_seen[:, preds == true_labels_seen, :]
        all_outputs_false = torch.cat((
            all_outputs_seen[:, preds != true_labels_seen, :],
            all_outputs_unseen
        ), 1)

    uncs_true = get_uncs(all_outputs_true)
    uncs_false = get_uncs(all_outputs_false)

    aucs = []
    all_uncs[exp_nb] = pd.DataFrame()

    for idx, unc_name in enumerate(unc_names):
        all_uncs[exp_nb] = all_uncs[exp_nb].append(
            pd.DataFrame()
            .assign(unc=torch.cat((uncs_true[idx], uncs_false[idx])))
            .assign(is_ood=torch.cat((torch.zeros_like(uncs_true[idx]), torch.ones_like(uncs_false[idx]))))
            .assign(unc_name=unc_name)
            .assign(number_of_tests=number_of_tests)
        )
    
        this_df = all_uncs[exp_nb].loc[all_uncs[exp_nb].unc_name == unc_name]
        positives = this_df.is_ood.sum()
        negatives = (1-this_df.is_ood).sum()
        grouped_unc = this_df.groupby('unc')
    
        to_plot = (
            grouped_unc
            .sum()
            .assign(n=grouped_unc.apply(lambda df: len(df)))
            .assign(tp=lambda df: df.iloc[::-1].is_ood.cumsum())
            .assign(fp=lambda df: df.iloc[::-1].n.cumsum() - df.tp)
    
        ).reset_index()
    
        all_xs[unc_name].append((to_plot.fp/negatives))
        all_ys[unc_name].append((to_plot.tp/positives))
        # xs = to_plot.fp/negatives
        # ys = to_plot.tp/positives



if not is_determinist:
    unc_names = ['sr', 'pe', 'vr', 'mi']

fig, axs = plt.subplots(len(unc_names), 1, figsize=(10, 15))

for idx, unc_name in enumerate(unc_names):
    xs_means = pd.DataFrame(all_xs[unc_name]).mean(0)
    ys_means = pd.DataFrame(all_ys[unc_name]).mean(0)
    axs[idx].plot(xs_means, ys_means)
    axs[idx].plot([0, 1], [0, 1], linestyle='dashed')
    axs[idx].set_ylim(bottom=0, top=1)
    axs[idx].set_xlim(left=0, right=1)
    axs[idx].set_ylabel('tpr')
    axs[idx].set_xlabel('fpr')
    cur_auc = metrics.auc(xs_means.iloc[xs_means.argsort()], ys_means.iloc[xs_means.argsort()])
    aucs.append(cur_auc)
    axs[idx].set_title(f'{unc_name}, AUC: {round(100 * cur_auc, 2)} %')

fig.suptitle(f'')
fig.show()
print(aucs)

save_fig = False
if save_fig:
    trainset = 'cifar10'
    save_path = f'rapport/open_set_recognition/{trainset}/'
    save_path = pathlib.Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    if is_determinist:
        fig.savefig(save_path / f'determinist_classes.png')
        save_to_file(fig, save_path / f'determinist_classes.pkl')
    else:
        fig.savefig(save_path/f'{arguments["loss_type"]}_classes.png')
        save_to_file(fig, save_path/f'{arguments["loss_type"]}_classes.pkl')




