import os
import pathlib
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import torch
from tqdm import tqdm

from scripts.utils import get_trained_model_and_args_and_groupnb, get_args
from src.dataset_manager.get_data import get_mnist
from src.risk_control import bound_animate
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import compute_predictive_entropy, get_predictions_from_multiple_tests, \
    get_all_uncertainty_measures
from src.utils import convert_tensor_to_float, plot_density_on_ax

###### TO CHANGE #########

exp_nbs = ['3713', '3719', '3749', '3778', '3716', '3722', '3752', '3781', '3842', '3851', '3861', '3864']
# exp_nbs = ['3713']
number_of_tests = 10
# rstars = np.linspace(0.01, 0.03, 50)
rstars = [0.01]
delta = 0.01

do_computation = True
verbose = False

save_csv = True
show_fig = False
save_animation = True
save_fig = False
figsize = (10, 6)

save_csv_path = 'results/risk_coverage/'
save_fig_path = 'results/risk_coverage/'
GPUPATH = '/output/sicara/BayesianFewShotExperiments/groups/'
CPUPATH = 'polyaxon_results/groups'
##########################


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

if device == torch.device('cuda'):
    path = GPUPATH
if device == torch.device('cpu'):
    path = CPUPATH

save_csv_path = pathlib.Path(save_csv_path)
if not os.path.exists(save_csv_path):
    results = pd.DataFrame(columns=['exp', 'unc', 'threshold', 'risk', 'acc', 'coverage', 'time'])
    if save_csv:
        save_csv_path.mkdir()
else:
    results = pd.read_csv(save_csv_path / 'results.csv', )
    results = results.filter(regex=r'^(?!Unnamed)')
    results.to_csv(save_csv_path / 'results_backup.csv')

global_start = time.time()
if do_computation:
    for exp_nb in exp_nbs:
        print(exp_nb)
        bay_net, arguments, _ = get_trained_model_and_args_and_groupnb(exp_nb, exp_path=path)
        split_labels = arguments.get('split_labels', 10)
        trainloader, _, _ = get_mnist(train_labels=range(split_labels), eval_labels=(), split_val=0, batch_size=128)
        bay_net.to(device)
        uncertainty_function = compute_predictive_entropy

        true_labels, all_outputs_train = eval_bayesian(
            bay_net,
            trainloader,
            number_of_tests=number_of_tests,
            return_accuracy=False,
            device=device,
            verbose=True,
        )
        labels_predicted = get_predictions_from_multiple_tests(all_outputs_train).float()
        correct_preds = (labels_predicted == true_labels).float()
        residuals = 1 - correct_preds

        uncs = get_all_uncertainty_measures(all_outputs_train)
        for idx_risk, rstar in enumerate(tqdm(rstars)):
            for unc, unc_name in zip(uncs, ['vr', 'pe', 'mi']):
                start = time.time()
                thetas, bounds, risks, coverages = bound_animate(rstar, delta, -unc, residuals, verbose=verbose,
                                                                 max_iter=10,
                                                                 precision=1e-5, )
                threshold = thetas[-1]
                acc = correct_preds[-unc > threshold].mean()  # .sum() / correct_preds.size(0)
                coverage = (-unc >= threshold).sum().float() / unc.size(0)
                new_res = pd.DataFrame.from_dict({
                    'exp': [exp_nb],
                    'unc': [unc_name],
                    'delta': [delta],
                    'threshold': [threshold],
                    'risk': [rstar],
                    'acc': [acc],
                    'coverage': [coverage],
                    'time': [time.time() - start],
                })
                convert_tensor_to_float(new_res)
                results = results.append(new_res, sort=True)

                if save_animation:
                    fig_anim = plt.figure(figsize=figsize)
                    fig_anim.suptitle(f'{arguments["loss_type"], arguments["type_of_unseen"]}, \n'
                                      f'{arguments}', wrap=True)
                    ax = fig_anim.add_subplot(111)
                    unc_correct = unc[correct_preds == 1]
                    unc_false = unc[correct_preds == 0]


                    def animate(i):
                        ax.clear()
                        plot_density_on_ax(ax, [unc_correct, unc_false], ['correct', 'false'])
                        ax.annotate(f'Risk: {round(100 * risks[i], 2)}% \n'
                                    f'Bound: {round(100 * bounds[i], 2)}% \n'
                                    f'R*: {rstar} \n'
                                    f'Coverage {round(100 * coverages[i], 2)}%',
                                    xy=(0.6, 0.6),
                                    xycoords='axes fraction')
                        ax.axvline(-thetas[i], color='r', label='theta')
                        ax.legend()
                        return ax


                    anim = animation.FuncAnimation(fig_anim, animate, frames=len(thetas), )
                    save_animation_path = pathlib.Path(save_fig_path + '/animation')
                    save_animation_path.mkdir(exist_ok=True, parents=True)
                    anim.save(save_animation_path / f'{exp_nb}_{unc_name}_{idx_risk}_finding_threshold.gif',
                              writer='pillow', fps=4)

        print(f'Time since start: {time.time() - global_start}')

    if save_csv:
        results.to_csv(save_csv_path / 'results.csv')


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

if show_fig or save_fig:
    for exp_nb in exp_nbs:
        exp_nb = int(exp_nb)
        arguments = get_args(exp_nb, path)
        results_of_this_exp = results[results['exp'] == exp_nb]
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle(arguments, wrap=True)
        ax = fig.add_subplot(1, 1, 1)
        axs = {
            'vr': ax,
            'pe': ax,
            'mi': ax,
        }
        # axs = {
        #     'vr': fig.add_subplot(3, 1, 1),
        #     'pe': fig.add_subplot(3, 1, 2),
        #     'mi': fig.add_subplot(3, 1, 3),
        # }
        for unc, ax in axs.items():
            results_of_this_unc = results_of_this_exp[results_of_this_exp['unc'] == unc]
            # ax.set_title(unc)
            xs = results_of_this_unc['coverage'].astype(float).values
            ys = results_of_this_unc['acc'].astype(float).values
            ax.plot(xs[xs.argsort()], ys[xs.argsort()], linestyle='dotted', marker='.', label=unc)
            ax.set_xlabel('coverage')
            ax.set_ylabel('accuracy')
            ax.legend()
        if save_fig:
            save_fig_path = pathlib.Path(save_fig_path)
            save_fig_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_fig_path / f'{exp_nb}-acc-coverage.png')
        if show_fig:
            plt.show()


def f(x):
    if type(x) == str:
        if 'tensor' in x:
            return float(x.replace('tensor', '').replace('(', '').replace(')', ''))
        else:
            return x
    else:
        return x
