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
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scripts.utils import get_trained_model_and_args_and_groupnb
from src.dataset_manager.get_data import get_mnist, get_cifar10
from src.risk_control import bound_animate
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_predictions_from_multiple_tests, \
    get_all_uncertainty_measures_bayesian
from src.utils import convert_tensor_to_float, plot_density_on_ax

###### TO CHANGE ######################################################

# these_exp_nbs = ['3713', '3719', '3749', '3778', '3716', '3722', '3752', '3781', '3842', '3851', '3861', '3864']
these_exp_nbs = ['19815']
number_of_tests = 10
nb_of_runs = 1
# rstars = [0.13076923]
rstars = np.linspace(0.1, 0.5, 50)

delta = 0.01

recompute_outputs = True
verbose = False

save_csv = True
do_save_animation = False
figsize = (10, 6)

save_csv_path = 'results/risk_coverage/'
save_fig_path = 'results/risk_coverage/'
GPUPATH = '/output/sicara/BayesianFewShotExperiments/groups/'
CPUPATH = 'polyaxon_results/groups'
#######################################################################

path_to_exps = CPUPATH


def save_animation(
        arguments,
        rstar,
        unc,
        correct_preds,
        risks,
        bounds,
        coverages,
        thetas,
        figsize,
        save_path,
):
    """
    This function saves the animation of the threshold finding.
    Args:
        arguments (dict): arguments of the experiment
        rstar (float): maximum acceptable risk
        unc (array-like): size (size_of_batch): the uncertainty for each sample
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
                    f'R*: {100 * rstar}% \n'
                    f'Coverage {round(100 * coverages[i], 2)}%',
                    xy=(0.6, 0.6),
                    xycoords='axes fraction')
        ax.axvline(-thetas[i], color='r', label='theta')
        ax.legend()
        return ax

    anim = animation.FuncAnimation(fig_anim, animate, frames=len(thetas), )
    anim.save(save_path, writer='pillow', fps=4)


def main(
        exp_nbs=None,
        path_to_exps=path_to_exps,
        path_to_results=save_csv_path,
        nb_of_runs=nb_of_runs,
        number_of_tests=number_of_tests,
        rstars=rstars,
        delta=delta,
        recompute_outputs=recompute_outputs,
        verbose=verbose,
        save_csv=save_csv,
        do_save_animation=do_save_animation,
        device='cpu',
):
    """
    Performs selective classification given a trained network and testset. Computes different threshold depending on
    different accepted risks.
    Args:
        exp_nbs (int || str): number of the experiment
        path_to_exps (str): path to the experiment groups
        path_to_results (str): path to save the results
        nb_of_runs (int): number of times to perform the same operation to get a confidence interval
        number_of_tests (int): number of inferences for each predictions
        rstars (list): list of float of accepted risks
        delta (float): probability of being higher than the upper bound
        recompute_outputs (Bool): whether or not we compute the outputs of train / test set. Put False if it is already
                                  computed and you don't want to loose time.
        verbose (Bool): show or not progress bar
        save_csv (Bool): save or not in csv
        do_save_animation (Bool): save or not the animation of finding the threshold.
        device (torch.device): gpu or cpu

    """
    if exp_nbs is None:
        exp_nbs = these_exp_nbs
    save_csv_path = pathlib.Path(path_to_results)
    save_fig_path = pathlib.Path(path_to_results)

    if not os.path.exists(save_csv_path / 'results_train.csv'):
        results_train = pd.DataFrame(
            columns=['exp', 'unc', 'threshold', 'risk', 'acc', 'coverage', 'time', 'number_of_tests'])
        results_eval = pd.DataFrame(
            columns=['exp', 'unc', 'threshold', 'risk', 'acc', 'coverage', 'time', 'number_of_tests'])
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
    for _ in range(nb_of_runs):
        for exp_nb in exp_nbs:
            print(exp_nb)
            bay_net, arguments, _ = get_trained_model_and_args_and_groupnb(exp_nb, exp_path=path_to_exps)
            if recompute_outputs:

                split_labels = arguments.get('split_labels', 10)
                if arguments.get('trainset', 'mnist') == 'mnist':
                    get_trainset = get_mnist
                elif arguments.get('trainset', 'mnist') == 'cifar10':
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

            uncs_train = get_all_uncertainty_measures_bayesian(all_outputs_train)
            uncs_eval = get_all_uncertainty_measures_bayesian(all_outputs_eval)
            for idx_risk, rstar in enumerate(tqdm(rstars)):
                for unc_train, unc_eval, unc_name in zip(uncs_train, uncs_eval, ['vr', 'pe', 'mi']):
                    start = time.time()
                    thetas, bounds, risks, coverages = bound_animate(rstar, delta, -unc_train, residuals,
                                                                     verbose=verbose,
                                                                     max_iter=10,
                                                                     precision=1e-5, )
                    threshold = thetas[-1]
                    acc_train = correct_preds_train[
                        -unc_train > threshold].mean()  # .sum() / correct_preds_train.size(0)
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
                        'loss_type': [arguments.get('loss_type', 'criterion')],
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
                        'loss_type': [arguments.get('loss_type', 'criterion')],
                    })
                    convert_tensor_to_float(new_res_eval)
                    results_eval = results_eval.append(new_res_eval, sort=True)

                    if do_save_animation:
                        save_animation_path = save_fig_path / 'animation'
                        save_animation_path.mkdir(exist_ok=True, parents=True)
                        save_animation_path = save_animation_path / f'{exp_nb}_{unc_name}_{idx_risk}_' \
                            f'finding_threshold.gif'
                        save_animation(arguments, rstar, unc_train, correct_preds_train, risks, bounds, coverages,
                                       thetas, figsize,
                                       save_animation_path)

                if save_csv:
                    results_train.to_csv(save_csv_path / 'results_train.csv')
                    results_eval.to_csv(save_csv_path / 'results_eval.csv')
            print(f'Time since start: {time.time() - global_start}')


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    device = torch.device(device)
    print(device)

    if device == torch.device('cuda'):
        path_to_exps = GPUPATH
    elif device == torch.device('cpu'):
        path_to_exps = CPUPATH

    main(path_to_exps=path_to_exps, path_to_results=save_csv_path, device=device, )
