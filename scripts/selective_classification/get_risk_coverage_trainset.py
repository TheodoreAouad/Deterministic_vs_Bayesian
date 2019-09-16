"""
This file computes the bounds for a given risk and delta, for the given experiments. The accuracy / coverage are
computed on the trainset.
To use it:
- change the 'to change' parameters
- run the code in the console
"""
import os
import pathlib
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import pandas as pd
import torch
from tqdm import tqdm

from scripts.utils import get_trained_model_and_args_and_groupnb, get_args
from src.dataset_manager.get_data import get_mnist, get_cifar10
from src.risk_control import bound_animate
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_predictions_from_multiple_tests, \
    get_all_uncertainty_measures
from src.utils import convert_tensor_to_float, plot_density_on_ax

###### TO CHANGE ######################################################

# exp_nbs = ['3713', '3719', '3749', '3778', '3716', '3722', '3752', '3781', '3842', '3851', '3861', '3864']
exp_nbs = ['4795']
number_of_tests_list = [20]
number_of_tests_to_print = [10, 20]
rstars = [0.13076923]
# rstars = [0.05, 0.05897436, 0.06794872, 0.07692308, 0.08589744,
#           0.09487179, 0.10384615, 0.11282051, 0.12179487, 0.13076923,
#           0.13974359, 0.14871795, 0.15769231, 0.16666667, 0.17564103,
#           0.18461538, 0.19358974, 0.2025641, 0.21153846, 0.22051282,
#           0.22948718, 0.23846154, 0.2474359, 0.25641026, 0.26538462,
#           0.27435897, 0.28333333, 0.29230769, 0.30128205, 0.31025641,
#           0.31923077, 0.32820513, 0.33717949, 0.34615385, 0.35512821,
#           0.36410256, 0.37307692, 0.38205128, 0.39102564, ]
# rstars = [0.05, 0.05897436, 0.06794872, 0.07692308, 0.08589744,
#           0.09487179, 0.10384615, 0.11282051, 0.12179487, 0.13076923,
#           0.13974359, 0.14871795, 0.15769231, 0.16666667, 0.17564103,]

delta = 0.01

do_computation = False
recompute_outputs = False
verbose = False

save_csv = True
show_fig = True
do_save_animation = True
save_fig = True
figsize = (10, 6)

save_csv_path = 'results/risk_coverage/cifar10'
save_fig_path = 'results/risk_coverage/cifar10'
GPUPATH = '/output/sicara/BayesianFewShotExperiments/groups/'
CPUPATH = 'polyaxon_results/groups'
#######################################################################


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

if device == torch.device('cuda'):
    path = GPUPATH
elif device == torch.device('cpu'):
    path = CPUPATH
else:
    assert False

save_csv_path = pathlib.Path(save_csv_path)
if not os.path.exists(save_csv_path / 'results_train.csv'):
    results_train = pd.DataFrame(columns=['exp', 'unc', 'threshold', 'risk', 'acc', 'coverage', 'time', 'number_of_tests'])
    results_eval = pd.DataFrame(columns=['exp', 'unc', 'threshold', 'risk', 'acc', 'coverage', 'time', 'number_of_tests'])
    if save_csv:
        save_csv_path.mkdir(exist_ok=True, parents=True)
        results_train.to_csv(save_csv_path / 'results_train.csv')
        results_eval.to_csv(save_csv_path / 'results_eval.csv')
else:
    results_train = pd.read_csv(save_csv_path / 'results_train.csv', )
    results_train = results_train.filter(regex=r'^(?!Unnamed)')
    results_train.to_csv(save_csv_path / 'results_train_backup.csv')
    results_eval = pd.read_csv(save_csv_path / 'results_eval.csv', )
    results_eval = results_eval.filter(regex=r'^(?!Unnamed)')
    results_eval.to_csv(save_csv_path / 'results_eval_backup.csv')

global_start = time.time()


def save_animation(unc, correct_preds, risks, bounds, coverages, thetas, figsize, save_path):
    """
    This function saves the animation of the threshold finding.
    Args:

        correct_preds (array-like): size (size_of_batch): array with 1 if the prediction is correct, 0 if false
        risks (array-like): size (nb of iterations in bound_animate): the evolution of risks across iterations
        bounds (array-like): size (nb of iterations in bound_animate): the evolution of bounds across iterations
        coverages (array-like): size (nb of iterations in bound_animate): the evolution of coverage across iterations
        thetas (array-like): size (nb of iterations in bound_animate): the evolution of threshold across iterations
        figsize (Tuple): size of the plt.figure
        save_path (str): path where to save the animation

    """
    fig_anim = plt.figure(figsize=figsize)
    fig_anim.suptitle(f'{arguments["loss_type"], arguments["type_of_unseen"]}, \n'
                      f'{arguments}', wrap=True)
    ax = fig_anim.add_subplot(111)
    unc_correct = unc[correct_preds == 1]
    unc_false = unc[correct_preds == 0]

    def animate(i):
        """
        Iteration function to animate with matplotlib.animation
        Args:
            i (int): index on which to iterate

        """
        ax.clear()
        plot_density_on_ax(ax, [unc_correct, unc_false], hist=True, labels=['correct', 'false'])
        ax.annotate(f'Risk: {round(100 * risks[i], 2)}% \n'
                    f'Bound: {round(100 * bounds[i], 2)}% \n'
                    f'R*: {100*rstar}% \n'
                    f'Coverage {round(100 * coverages[i], 2)}%',
                    xy=(0.6, 0.6),
                    xycoords='axes fraction')
        ax.axvline(-thetas[i], color='r', label='theta')
        ax.legend()
        return ax

    anim = animation.FuncAnimation(fig_anim, animate, frames=len(thetas), )
    anim.save(save_path, writer='pillow', fps=4)


if do_computation:
    for exp_nb in exp_nbs:
        print(exp_nb)
        for number_of_tests in number_of_tests_list:
            bay_net, arguments, _ = get_trained_model_and_args_and_groupnb(exp_nb, exp_path=path)
            if recompute_outputs:

                split_labels = arguments.get('split_labels', 10)
                if arguments['trainset'] == 'mnist':
                    get_trainset = get_mnist
                elif arguments['trainset'] == 'cifar10':
                    get_trainset = get_cifar10
                else:
                    assert False, 'trainset not recognized'

                trainloader_seen, _, evalloader_seen = get_trainset(
                    train_labels=range(split_labels),
                    eval_labels=range(split_labels),
                    batch_size=128,
                )

                bay_net.to(device)

                true_labels_train, all_outputs_train = eval_bayesian(
                    bay_net,
                    trainloader_seen,
                    number_of_tests=number_of_tests,
                    return_accuracy=False,
                    device=device,
                    verbose=True,
                )
                labels_predicted_train = get_predictions_from_multiple_tests(all_outputs_train).float()

                true_labels_eval, all_outputs_eval = eval_bayesian(
                    bay_net,
                    evalloader_seen,
                    number_of_tests=number_of_tests,
                    return_accuracy=False,
                    device=device,
                    verbose=True,
                )
                labels_predicted_eval = get_predictions_from_multiple_tests(all_outputs_eval).float()

            correct_preds_train = (labels_predicted_train == true_labels_train).float()
            residuals = 1 - correct_preds_train
            correct_preds_eval = (labels_predicted_eval == true_labels_eval).float()

            uncs_train = get_all_uncertainty_measures(all_outputs_train)
            uncs_eval = get_all_uncertainty_measures(all_outputs_eval)
            for idx_risk, rstar in enumerate(tqdm(rstars)):
                for unc_train, unc_eval, unc_name in zip(uncs_train, uncs_eval, ['vr', 'pe', 'mi']):
                    start = time.time()
                    thetas, bounds, risks, coverages = bound_animate(rstar, delta, -unc_train, residuals, verbose=verbose,
                                                                     max_iter=10,
                                                                     precision=1e-5, )
                    threshold = thetas[-1]
                    acc_train = correct_preds_train[-unc_train > threshold].mean()  # .sum() / correct_preds_train.size(0)
                    coverage_train = (-unc_train >= threshold).sum().float() / unc_train.size(0)
                    new_res_train = pd.DataFrame.from_dict({
                        'exp': [exp_nb],
                        'unc': [unc_name],
                        'delta': [delta],
                        'threshold': [threshold],
                        'risk': [rstar],
                        'acc': [acc_train],
                        'coverage': [coverage_train],
                        'time': [time.time() - start],
                        'number_of_tests': [number_of_tests],
                        'loss_type': [arguments['loss_type']],
                    })
                    convert_tensor_to_float(new_res_train)
                    results_train = results_train.append(new_res_train, sort=True)

                    acc_eval = correct_preds_eval[-unc_eval > threshold].mean()
                    coverage_eval = (-unc_eval >= threshold).sum().float() / unc_eval.size(0)
                    new_res_eval = pd.DataFrame.from_dict({
                        'exp': [exp_nb],
                        'unc': [unc_name],
                        'delta': [delta],
                        'threshold': [threshold],
                        'risk': [rstar],
                        'acc': [acc_eval],
                        'coverage': [coverage_eval],
                        'time': [time.time() - start],
                        'number_of_tests': [number_of_tests],
                        'loss_type': [arguments['loss_type']],
                    })
                    convert_tensor_to_float(new_res_eval)
                    results_eval = results_eval.append(new_res_eval, sort=True)

                    if do_save_animation:
                        save_animation_path = pathlib.Path(save_fig_path + '/animation')
                        save_animation_path.mkdir(exist_ok=True, parents=True)
                        save_animation_path = save_animation_path / f'{exp_nb}_{unc_name}_{idx_risk}_' \
                            f'finding_threshold.gif'
                        save_animation(unc_train, correct_preds_train, risks, bounds, coverages, thetas, figsize, save_animation_path)

                if save_csv:
                    results_train.to_csv(save_csv_path / 'results_train.csv')
                    results_eval.to_csv(save_csv_path / 'results_eval.csv')
        print(f'Time since start: {time.time() - global_start}')


# theta = rc.get_selection_threshold(
#     bay_net,
#     trainloader,
#     rstar,
#     delta,
#     uncertainty_function,
#     number_of_tests,
#     verbose,
#     device
# )


results_train = pd.read_csv(save_csv_path / 'results_train.csv', )
def plot_acc_cov(number_of_tests_to_print, exp_nb, results, figsize=figsize,):
    fig = plt.figure(figsize=figsize)
    arguments = get_args(exp_nb, path)
    fig.suptitle(f'Acc-Coverage, w.r.t. nb of tests and uncertainty measure\n'
                 f'{dict({k: v for k, v in arguments.items() if k not in ["type_of_unseen", "number_of_tests"]})}',
                 wrap=True)
    ax = fig.add_subplot(1, 1, 1)
    axs = {
        'vr': ax,
        'pe': ax,
        'mi': ax,
    }
    styles = ['dotted', (0, (5, 1)), ]  # size: len(number_of_tests_to_plot)
    markers = ['D', 'v', ]  # size: len(number_of_tests_to_plot)
    cmaps = ['cool', 'hot', ]  # size: len(number_of_tests_to_plot)
    cmaps_position = [0.3, 0.5, 0.7]  # size: number of uncertainty measures
    for style, nb_of_test in enumerate(number_of_tests_to_print):
        exp_nb = int(exp_nb)
        results_of_this_exp = results.query(f'exp == {exp_nb} & number_of_tests == {nb_of_test}')

        cmap = cm.get_cmap(cmaps[style])
        # axs = {
        #     'vr': fig.add_subplot(3, 1, 1),
        #     'pe': fig.add_subplot(3, 1, 2),
        #     'mi': fig.add_subplot(3, 1, 3),
        # }
        for unc_idx, (unc, ax) in enumerate(axs.items()):
            results_of_this_unc = results_of_this_exp[results_of_this_exp['unc'] == unc]
            # ax.set_title(unc)
            xs = results_of_this_unc['coverage'].astype(float).values
            ys = results_of_this_unc['acc'].astype(float).values
            ax.plot(
                xs[xs.argsort()],
                ys[xs.argsort()],
                linestyle=styles[style],
                marker=markers[style],
                label=f'{unc}-{nb_of_test}',
                c=cmap(cmaps_position[unc_idx])
            )
    ax.set_xlabel('coverage')
    ax.set_ylabel('accuracy')
    ax.legend()
    return fig


if show_fig or save_fig:
    for exp_nb in exp_nbs:
        fig = plot_acc_cov(number_of_tests_to_print, exp_nb, results_train, figsize=figsize)
        if save_fig:
            save_fig_path = pathlib.Path(save_fig_path)
            save_fig_path.mkdir(exist_ok=True, parents=True)
            fig.savefig(save_fig_path / f'{exp_nb}-acc-coverage-train.png')
        if show_fig:
            fig.show()

results_eval = pd.read_csv(save_csv_path / 'results_eval.csv', )
if show_fig or save_fig:
    for exp_nb in exp_nbs:
        fig = plot_acc_cov(number_of_tests_to_print, exp_nb, results_eval, figsize=figsize)
        if save_fig:
            save_fig_path = pathlib.Path(save_fig_path)
            save_fig_path.mkdir(exist_ok=True, parents=True)
            fig.savefig(save_fig_path / f'{exp_nb}-acc-coverage-eval.png')
        if show_fig:
            fig.show()


def f(x):
    """
    This function transforms the string of tensors x into a string of floats.
    Args:
        x (str): the string with the word 'tensor' we want to see disappear.

    Returns:
        str: all 'tensor' strings are gone.
    """
    if type(x) == str:
        if 'tensor' in x:
            return float(x.replace('tensor', '').replace('(', '').replace(')', ''))
        else:
            return x
    else:
        return x
