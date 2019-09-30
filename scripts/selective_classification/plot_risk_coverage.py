import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

from scripts.utils import get_args
from src.utils import save_to_file

######## TO CHANGE ###############

# exp_nbs = ['14617', '14746', '14681', '14627', '14748', '14689', '14633', '14754', '14695']
exp_nbs = ['4789', '4795']
number_of_tests_to_print = [10, 20]

show_fig = False
save_fig = True
figsize = (10, 6)

save_csv_path = 'results/risk_coverage/'
save_png_path = 'results/risk_coverage/png_figures'
save_pkl_path = 'results/risk_coverage/pyplot_figures'
GPUPATH = '/output/sicara/BayesianFewShotExperiments/groups/'
CPUPATH = 'polyaxon_results/groups'
###################################

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
save_png_path = pathlib.Path(save_png_path)
save_pkl_path = pathlib.Path(save_pkl_path)


def plot_acc_cov(number_of_tests_to_print, exp_nb, results, figsize=figsize, ):
    """
    Plots the accuracy / coverage function given the number of tests, the number of the experiment
    Args:
        number_of_tests_to_print (list): list of the number of tests we want to plot
        exp_nb (int || str): number of the experiment
        results (pandas.core.frame.DataFrame): dataframe containing the accuracy and coverage
        figsize (tuple): size of the figure

    Returns:
        matplotlib.figure.Figure: the figure with the plot inside
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    axs = {
        'vr': ax,
        'pe': ax,
        'mi': ax,
    }
    styles = ['dotted', (0, (5, 1)), ]  # size: len(number_of_tests_to_plot)
    # markers = ['D', 'v', ]  # size: len(number_of_tests_to_plot)
    cmaps = ['autumn', 'winter', 'summer']  # size: nb of measures
    cmaps_position = [0.4, 0.7]  # size: len(nb of tests to plot)
    for style, nb_of_test in enumerate(number_of_tests_to_print):
        exp_nb = int(exp_nb)
        results_of_this_exp = results.query(f'exp == {exp_nb} & number_of_tests == {nb_of_test}')

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
            cmap = cm.get_cmap(cmaps[unc_idx])
            ax.plot(
                xs[xs.argsort()],
                ys[xs.argsort()],
                linestyle=styles[style],
                marker='.',
                label=f'{unc}-{nb_of_test}',
                c=cmap(cmaps_position[style])
            )
    ax.set_xlabel('coverage')
    ax.set_ylabel('accuracy')
    ax.legend()
    return fig


results_train = pd.read_csv(save_csv_path / 'results_train.csv', index_col=False)
if show_fig or save_fig:
    for exp_nb in exp_nbs:
        fig = plot_acc_cov(number_of_tests_to_print, exp_nb, results_train, figsize=figsize)
        arguments = get_args(exp_nb, path)
        fig.suptitle(f'Trainset: Acc-Coverage, w.r.t. nb of tests and uncertainty measure\n'
                     f'{dict({k: v for k, v in arguments.items() if k not in ["type_of_unseen", "number_of_tests"]})}',
                     wrap=True)
        if save_fig:
            save_png_path.mkdir(exist_ok=True, parents=True)
            save_pkl_path.mkdir(exist_ok=True, parents=True)
            fig.savefig(save_png_path / f'{exp_nb}-acc-coverage-train.png')
            save_to_file(fig, save_pkl_path / f'{exp_nb}-acc-coverage-train.pkl')
        if show_fig:
            fig.show()

results_eval = pd.read_csv(save_csv_path / 'results_eval.csv')
if show_fig or save_fig:
    for exp_nb in exp_nbs:
        fig = plot_acc_cov(number_of_tests_to_print, exp_nb, results_eval, figsize=figsize)
        arguments = get_args(exp_nb, path)
        fig.suptitle(f'Testset: Acc-Coverage, w.r.t. nb of tests and uncertainty measure\n'
                     f'{dict({k: v for k, v in arguments.items() if k not in ["type_of_unseen", "number_of_tests"]})}',
                     wrap=True)
        if save_fig:
            save_png_path.mkdir(exist_ok=True, parents=True)
            save_pkl_path.mkdir(exist_ok=True, parents=True)
            fig.savefig(save_png_path / f'{exp_nb}-acc-coverage-eval.png')
            save_to_file(fig, save_pkl_path / f'{exp_nb}-acc-coverage-eval.pkl')
        if show_fig:
            fig.show()
