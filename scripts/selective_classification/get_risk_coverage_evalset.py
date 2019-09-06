"""
This file computes the useful graphs we want to extract from a selective classification experiment.
"""
import pathlib

import matplotlib.pyplot as plt

from src.utils import load_from_file

#### TO CHANGE ########

group_nb = '210'
show_fig = True
save_fig = True
save_path = 'results/selective_classification/risk_coverage/eval'
figsize = (10, 10)
#######################

results = load_from_file(f'results/raw_results/group{group_nb}_specific_results.pkl')
loss_types = set(results['loss_type'].values)
uncs = ['vr', 'pe', 'mi']


for unc in uncs:
    fig = plt.figure(figsize=figsize)
    fig.suptitle(unc)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    results_of_unc = results.filter(items=['loss_type', f'eval accuracy {unc}', f'eval coverage {unc}', 'risk'])
    for loss_type in loss_types:
        results_of_loss = results[results['loss_type'] == loss_type]
        xs1 = results_of_loss[f'eval coverage {unc}'].values
        ys1 = results_of_loss[f'eval accuracy {unc}'].values

        xs2 = xs1
        ys2 = results_of_loss['risk'].values

        xs3 = results_of_loss['risk'].values
        ys3 = results_of_loss[f'eval accuracy {unc}'].values

        ax1.scatter(xs1, ys1, label=loss_type)
        ax1.set_xlabel('coverage')
        ax1.set_ylabel('accuracy')

        ax2.scatter(xs2, ys2, label=loss_type)
        ax2.set_xlabel('coverage')
        ax2.set_ylabel('risk')

        ax3.plot(xs3[xs3.argsort()], ys3[xs3.argsort()], linestyle='dashed', marker='o', label=loss_type)
        ax3.set_xlabel('risk')
        ax3.set_ylabel('accuracy')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    if save_fig:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path / f'{group_nb}_{unc}_acc_cov.png')
    if show_fig:
        fig.show()
