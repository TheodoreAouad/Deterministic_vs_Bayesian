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
from matplotlib import cm
from matplotlib.lines import Line2D

from scripts.utils import get_trained_model_and_args_and_groupnb, get_evalloader_seen, get_evalloader_unseen, get_args
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_all_uncertainty_measures_not_bayesian, get_all_uncertainty_measures, \
    get_predictions_from_multiple_tests
from src.utils import save_to_file

################## RHO = -8, STD_PRIOR = 0.55 ################
# These experiments consider the best hyperparameters


# exp_nbs = [20701, 20701, 20702, 20703, 20704]    # cifar10 criterion
# exp_nbs = [20695, 20696, 20697, 20698, 20699]    # cifar10 uniform
# exp_nbs = [20694, 20693, 20692, 20691, 20690]    # cifar10 exponential
# exp_nbs_det = [20724, 20723, 20722, 20725, 20726]    # cifar10 determinist
# exp_nbs = [20709, 20707, 20706, 20708, 20705]    # mnist criterion
# exp_nbs = [20715, 20716, 20718, 20719, 20717]    # mnist uniform
# exp_nbs = [20712, 20713, 20714, 20711, 20710]    # mnist exponential
# exp_nbs_det = [20736, 20738, 20739, 20737, 20735]    # mnist determinist

################## RHO = -10 STD_PRIOR = 0.55 ################
# These experiments consider the hyperparameters that match the determinist accuracy for CIFAR10

# exp_nbs = [20765, 20764, 20768, 20766, 20767]   # cifar10 criterion
# exp_nbs =  20777, 20778, 20776, 20774, 20775]   # cifar10 uniform
exp_nbs = [20770, 20771, 20773, 20772, 20769]   # cifar10 exponential
exp_nbs_det = [20724, 20723, 20722, 20725, 20726]    # cifar10 determinist
# exp_nbs = [20749, 20753, 20752, 20750, 20751]    # mnist criterion
# exp_nbs = [20762, 20763, 20761, 20759, 20760]    # mnist uniform
# exp_nbs = [20754, 20755, 20757, 20758, 20756]    # mnist exponential
# exp_nbs_det = [20736, 20738, 20739, 20737, 20735]    # mnist determinist

# exp_nbs_det = ['20869', '20867', '20866', '20868', '20870'] # mnist det no batch norm
# exp_nbs_det = ['20872', '20875', '20874', '20873', '20871'] # cifar10 det no batch norm

arg = get_args(exp_nbs[0])
trainset = arg['trainset']
save_path1 = pathlib.Path(f'rapport/open_set_recognition/')
show_fig = True
save_fig = True

# exp_nbs = [
#     20777, 20770, 20778, 20771, 20776, 20749, 20754, 20753, 20765, 20762,
#     20763, 20764, 20752, 20755, 20773, 20774, 20775, 20772, 20750, 20768,
#     20757, 20761, 20759, 20766, 20758, 20767, 20760, 20769, 20756, 20751
# ]

path_to_res = 'polyaxon_results/groups'
number_of_tests = 100

typs = ['seen_unseen', 'true_false', 'true_false_unseen']
res = pd.DataFrame(columns=['rho', 'std_prior', 'T', 'loss', 'unc_name',
                            'mnist_seen_unseen', 'cifar10_seen_unseen', 'mnist_true_false', 'cifar10_true_false',
                            'mnist_true_false_unseen', 'cifar10_true_false_unseen'],
                   )

loss_type = arg['loss_type']
rho = arg['rho']
std_prior = arg['std_prior']


for typs_idx in range(3):
    typ = typs[typs_idx]

    save_path = save_path1 / f'{trainset}/{typ}/{arg["loss_type"]}'

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    device = torch.device(device)
    print(device)

    unc_names_det = ['sr', 'pe']
    unc_names = ['sr', 'vr', 'pe', 'mi']
    path_to_outputs = pathlib.Path(f'temp/softmax_outputs/')


    def compute_tps_fps(exp_nbs, get_uncs, unc_names, number_of_tests):
        all_xs = dict({k: [] for k in unc_names})
        all_ys = dict({k: [] for k in unc_names})
        all_xs_pr = dict({k: [] for k in unc_names})
        all_ys_pr = dict({k: [] for k in unc_names})
        all_uncs = {}
        for exp_nb in exp_nbs:

            bay_net_trained, arguments, _ = get_trained_model_and_args_and_groupnb(exp_nb, path_to_res)

            if number_of_tests < 100 and os.path.exists(path_to_outputs / '100' / f'{exp_nb}/true_labels_seen.pt'):

                true_labels_seen = torch.load(path_to_outputs / f'100' / f'{exp_nb}/true_labels_seen.pt')
                all_outputs_seen = torch.load(path_to_outputs / f'100' / f'{exp_nb}/all_outputs_seen.pt')
                all_outputs_unseen = torch.load(path_to_outputs / f'100' / f'{exp_nb}/all_outputs_unseen.pt')

                random_idx = np.arange(100)
                np.random.shuffle(random_idx)
                random_idx = random_idx[:number_of_tests]
                all_outputs_seen = all_outputs_seen[random_idx]
                all_outputs_unseen = all_outputs_unseen[random_idx]
            elif os.path.exists(path_to_outputs / f'{number_of_tests}' / f'{exp_nb}/true_labels_seen.pt'):
                true_labels_seen = torch.load(path_to_outputs / f'{number_of_tests}' / f'{exp_nb}/true_labels_seen.pt')
                all_outputs_seen = torch.load(path_to_outputs / f'{number_of_tests}' / f'{exp_nb}/all_outputs_seen.pt')
                all_outputs_unseen = torch.load(path_to_outputs / f'{number_of_tests}' / f'{exp_nb}/all_outputs_unseen.pt')
            else:
                (path_to_outputs / f'{number_of_tests}' / f'{exp_nb}').mkdir(exist_ok=True, parents=True)
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

                torch.save(true_labels_seen, path_to_outputs / f'{number_of_tests}' / f'{exp_nb}/true_labels_seen.pt')
                torch.save(all_outputs_seen, path_to_outputs / f'{number_of_tests}' / f'{exp_nb}/all_outputs_seen.pt')
                torch.save(all_outputs_unseen, path_to_outputs / f'{number_of_tests}' / f'{exp_nb}/all_outputs_unseen.pt')

            preds = get_predictions_from_multiple_tests(all_outputs_seen).float()

            if typ == 'seen_unseen':
                all_outputs_true = all_outputs_seen
                all_outputs_false = all_outputs_unseen

            elif typ == 'true_false':
                all_outputs_true = all_outputs_seen[:, preds == true_labels_seen, :]
                all_outputs_false = all_outputs_seen[:, preds != true_labels_seen, :]
            else:
                all_outputs_true = all_outputs_seen[:, preds == true_labels_seen, :]
                all_outputs_false = torch.cat((
                    all_outputs_seen[:, preds != true_labels_seen, :],
                    all_outputs_unseen
                ), 1)

            uncs_true = get_uncs(all_outputs_true)
            uncs_false = get_uncs(all_outputs_false)

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
                negatives = (1 - this_df.is_ood).sum()
                grouped_unc = this_df.groupby('unc')

                to_plot = (
                    grouped_unc
                        .sum()
                        .assign(n=grouped_unc.apply(lambda df: len(df)))
                        .assign(tp=lambda df: df.iloc[::-1].is_ood.cumsum())
                        .assign(fp=lambda df: df.iloc[::-1].n.cumsum() - df.tp)
                        .assign(fn=lambda df: df.is_ood.cumsum())
                        .assign(precision=lambda df: df.tp / (df.tp + df.fp))
                        .assign(recall= lambda df: df.tp / (df.tp + df.fn))

                ).reset_index()

                all_xs[unc_name].append((to_plot.fp / negatives))
                all_ys[unc_name].append((to_plot.tp / positives))
                all_xs_pr[unc_name].append((to_plot.recall))
                all_ys_pr[unc_name].append(to_plot.precision)
                # xs = to_plot.fp/negatives
                # ys = to_plot.tp/positives
        return all_xs, all_ys, all_xs_pr, all_ys_pr


    all_xs_bay, all_ys_bay, all_xs_pr, all_ys_pr = compute_tps_fps(exp_nbs, get_all_uncertainty_measures, unc_names, number_of_tests)
    all_xs_det, all_ys_det, _, _ = compute_tps_fps(exp_nbs_det, get_all_uncertainty_measures_not_bayesian, unc_names_det, 1)

    if show_fig:

        unc_names = ['sr', 'pe', 'vr', 'mi']

        cmaps = ['Reds', 'Blues']
        cmap_position = np.linspace(0.3, 0.8, len(exp_nbs))
        cmap_det = cm.get_cmap(cmaps[0])
        cmap_bay = cm.get_cmap(cmaps[1])

        aucs = np.zeros((len(unc_names), len(exp_nbs)))
        auprs = np.zeros((len(unc_names), len(exp_nbs)))
        aucs_det = np.zeros((len(unc_names), len(exp_nbs)))
        for idx_unc, unc_name in enumerate(unc_names):
            fig, axs = plt.subplots(1, 1, figsize=(7, 7))
            # xs_means = pd.DataFrame(all_xs_bay[unc_name]).mean(0)
            # ys_means = pd.DataFrame(all_ys_bay[unc_name]).mean(0)
            #
            # axs[idx].plot(xs_means.iloc[xs_means.argsort()], ys_means.iloc[xs_means.argsort()])
            axs.plot([0, 1], [0, 1], linestyle='dashed', c='black')
            axs.set_ylim(bottom=0, top=1)
            axs.set_xlim(left=0, right=1)
            axs.set_ylabel('tpr')
            axs.set_xlabel('fpr')

            title = f'{typ}, {unc_name}\n'
            legend_elements = []
            if unc_name in unc_names_det:
                for idx_exp, (xs, ys) in enumerate(zip(all_xs_det[unc_name], all_ys_det[unc_name])):
                    axs.plot(xs, ys, label=f'det, {exp_nbs[idx_exp]}', linestyle='dotted', c=cmap_det(cmap_position[idx_exp]))
                    aucs_det[idx_unc, idx_exp] = metrics.auc(xs, ys)
                legend_elements += [Line2D([0], [0], color=cmap_det(0.5), linestyle='dotted', label='deterministic')]
                title += f'Det AUC: {round(100 * aucs_det[idx_unc].mean(), 2)}  +- {round(100 * 1.97 * aucs_det[idx_unc].std() /5, 2)} %, '
            for idx_exp, (xs, ys) in enumerate(zip(all_xs_bay[unc_name], all_ys_bay[unc_name])):
                axs.plot(xs, ys, label=exp_nbs[idx_exp], c=cmap_bay(cmap_position[idx_exp]))
                aucs[idx_unc, idx_exp] = metrics.auc(xs, ys)
            for idx_exp, (xs, ys) in enumerate(zip(all_xs_pr[unc_name], all_ys_pr[unc_name])):
                # axs.plot(xs, ys, label=exp_nbs[idx_exp], c=cmap_bay(cmap_position[idx_exp]))
                auprs[idx_unc, idx_exp] = metrics.auc(xs, ys)
            legend_elements += [Line2D([0], [0], color=cmap_bay(0.5), label='bayesian')]
            # cur_auc = metrics.auc(xs_means.iloc[xs_means.argsort()], ys_means.iloc[xs_means.argsort()])
            axs.set_title(title + f'Bay AUC: {round(100 * aucs[idx_unc].mean(), 2)} +- {round(100 * 1.97 * aucs[idx_unc].std() / 5, 2)} %, T={number_of_tests}')
            axs.legend(handles=legend_elements)

            fig.suptitle(f'')
            if save_fig:
                (save_path / f'{unc_name}').mkdir(exist_ok=True, parents=True)
                fig.savefig(save_path / f'{unc_name}' / f'roc_{typ}_{arg["loss_type"]}_{unc_name}_T{number_of_tests}.png')
                save_to_file(fig, save_path / f'{unc_name}' / f'roc_{typ}_{arg["loss_type"]}_{unc_name}_T{number_of_tests}.pkl')
                lp = arg['loss_type']
                print(f"Fig saved in {save_path / f'{unc_name}' / f'roc_{typ}_{lp}_{unc_name}_T{number_of_tests}.png'}")
            plt.close(fig)

            # res.loc[
            #     [res.rho == rho, res.std_prior == std_prior, res.T == number_of_tests, res.unc_name == unc_name],
            #     [f'{trainset}+_{typ}']
            # ] = [aucs[idx_unc].mean(), 1.97 * aucs[idx_unc].std() / 5]

    print(aucs.mean(1))
    print(aucs.std(1) * 1.97 / 5)
    print(aucs_det.mean(1))
    print(aucs_det.std(1) * 1.97 / 5)
    print(auprs.mean(1))


