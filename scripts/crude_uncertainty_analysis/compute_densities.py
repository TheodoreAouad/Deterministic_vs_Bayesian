"""
This script is used to create the figures of accuracies against uncertainty. This is used to get results of a single
experiment. It cannot be used to create a figure across multiple experiments.
To use it, change the arguments in the TO CHANGE box.
"""
import os
import pathlib
from time import time

import numpy as np
import torch

from scripts.utils import get_trained_model_and_args_and_groupnb

from importlib import reload
import scripts.utils as utils
from src.utils import save_to_file

reload(utils)

###### TO CHANGE ###########
# exp_nbs = ['3713', '3719', '3749', '3778', '3716', '3722', '3752', '3781', '3832', '3834', '3839',
#            '3840', '3842', '3851', '3861', '3864']
# exp_nbs = ['3842', ]  # '3851', '3861', '3864']
# exp_nbs = ['4792', '4789', '4795']
exp_nbs = ['3832',]
nb_of_batches = 1000
size_of_batch = 100
nb_of_random = 5000

do_computation = True
save_outputs = True

do_train_seen_unseen = False
do_train_correct_false = False
do_seen_correct_false = False
do_train_seen = False
do_train_seen_correct_false = False
do_train_seen_unseen_correct_false = True

figsize = (12, 12)
show_fig = True
save_fig = False
do_eval_mnist = True
save_path = 'results/uncertainty_density/'

GPUPATH = '/output/sicara/BayesianFewShotExperiments/groups/'
CPUPATH = 'polyaxon_results/groups'

############################


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


def get_save_path(save_path, dirname, arguments, group_nb, exp_nb):
    save_path_hists = pathlib.Path(save_path) / dirname / arguments.get("loss_type", "determinist")

    save_path_hists.mkdir(parents=True, exist_ok=True)
    save_path_hists = save_path_hists / f'{group_nb}_{exp_nb}_uncertainty_density.png'
    return save_path_hists


failors = []
start_time = time()
for exp_nb in exp_nbs:
    print(f'{exp_nb}')

    bay_net_trained, arguments, group_nb = get_trained_model_and_args_and_groupnb(exp_nb, exp_path=path)

    try:
        if do_computation and not os.path.exists(f'temp/density/{exp_nb}_all_outputs_train.pt'):

            all_outputs_seen, true_labels_seen = utils.get_seen_outputs_and_labels(
                bay_net_trained,
                arguments,
                device=device,
            )

            all_outputs_unseen = utils.get_unseen_outputs(
                bay_net_trained,
                arguments,
                nb_of_random,
                device=device,
            )

            all_outputs_train, true_labels_train = utils.get_train_outputs(
                bay_net_trained,
                arguments,
                device=device,
            )

            if save_outputs:
                torch.save(all_outputs_train, f'temp/density/{exp_nb}_all_outputs_train.pt')
                torch.save(true_labels_train, f'temp/density/{exp_nb}_true_labels_train.pt')

                torch.save(all_outputs_unseen, f'temp/density/{exp_nb}_all_outputs_unseen.pt')

                torch.save(all_outputs_seen, f'temp/density/{exp_nb}_all_outputs_seen.pt')
                torch.save(true_labels_seen, f'temp/density/{exp_nb}_true_labels_seen.pt')
        else:
            all_outputs_train = torch.load(f'temp/density/{exp_nb}_all_outputs_train.pt')
            true_labels_train = torch.load(f'temp/density/{exp_nb}_true_labels_train.pt')

            all_outputs_unseen = torch.load(f'temp/density/{exp_nb}_all_outputs_unseen.pt')

            all_outputs_seen = torch.load(f'temp/density/{exp_nb}_all_outputs_seen.pt')
            true_labels_seen = torch.load(f'temp/density/{exp_nb}_true_labels_seen.pt')

        if do_train_seen_unseen:
            print('Do train unseen seen ...')
            save_path_hists = get_save_path(save_path, 'train_seen_unseen', arguments, group_nb, exp_nb)

            fig1 = utils.compute_density_train_seen_unseen(
                arguments=arguments,
                all_outputs_train=all_outputs_train,
                all_outputs_seen=all_outputs_seen,
                all_outputs_unseen=all_outputs_unseen,
                show_fig=show_fig,
                save_fig=save_fig,
                save_path=save_path_hists,
                figsize=figsize,
            )
            if save_fig:
                print('Saving figure...')
                fig1.savefig(save_path_hists)
                save_to_file(fig1, save_path_hists.replace('png', 'pkl'))
                print('Figure saved.')
            if show_fig:
                print('Showing figures...')
                fig1.show()
                print('Figure shown.')
            print('Done')

        if do_train_correct_false:
            print('Do train correct false...')
            save_path_hists = get_save_path(save_path, 'train_correct_false', arguments, group_nb, exp_nb)

            fig2 = utils.compute_density_correct_false(
                arguments=arguments,
                all_outputs=all_outputs_train,
                true_labels=true_labels_train,
                show_fig=show_fig,
                save_fig=save_fig,
                save_path=save_path_hists,
                figsize=figsize,
            )

            if save_fig:
                assert save_path is not None
                print('Saving figure...')
                fig2.savefig(save_path_hists)
                save_to_file(fig2, save_path_hists.replace('png', 'pkl'))
                print('Figure saved.')
            if show_fig:
                print('Showing figures...')
                fig2.show()
                print('Figure shown.')
            print('Done')

        if do_seen_correct_false:
            print('Do seen correct false...')
            save_path_hists = get_save_path(save_path, 'eval_correct_false', arguments, group_nb, exp_nb)

            fig3 = utils.compute_density_correct_false(
                arguments=arguments,
                all_outputs=all_outputs_seen,
                true_labels=true_labels_seen,
                show_fig=show_fig,
                save_fig=save_fig,
                save_path=save_path_hists,
                figsize=figsize,
            )

            if save_fig:
                print('Saving figure...')
                fig3.savefig(save_path_hists)
                save_to_file(fig3, save_path_hists.replace('png', 'pkl'))
                print('Figure saved.')
            if show_fig:
                print('Showing figures...')
                fig3.show()
                print('Figure shown.')

            print('Done')

        if do_train_seen:
            print('Do train seen ...')
            save_path_hists = get_save_path(save_path, 'train_eval', arguments, group_nb, exp_nb)

            fig4 = utils.compute_density_train_seen(
                arguments=arguments,
                all_outputs_train=all_outputs_train,
                all_outputs_seen=all_outputs_seen,
                show_fig=show_fig,
                save_fig=save_fig,
                save_path=save_path_hists,
                figsize=figsize,
            )

            if save_fig:
                print('Saving figure...')
                fig4.savefig(save_path_hists)
                save_to_file(fig4, save_path_hists.replace('png', 'pkl'))
                print('Figure saved.')
            if show_fig:
                print('Showing figures...')
                fig4.show()
                print('Figure shown.')
            print('Done')

        if do_train_seen_correct_false:
            print('Do train seen correct false ...')
            save_path_hists = get_save_path(save_path, 'train_eval_correct_false', arguments, group_nb, exp_nb)

            fig5 = utils.compute_density_train_seen_correct_false(
                arguments=arguments,
                all_outputs_train=all_outputs_train,
                true_labels_train=true_labels_train,
                all_outputs_seen=all_outputs_seen,
                true_labels_seen=true_labels_seen,
                show_fig=show_fig,
                save_fig=save_fig,
                save_path=save_path_hists,
                figsize=figsize,
            )

            if save_fig:
                print('Saving figure...')
                fig5.savefig(save_path_hists)
                save_to_file(fig5, save_path_hists.replace('png', 'pkl'))
                print('Figure saved.')
            if show_fig:
                print('Showing figures...')
                fig5.show()
                print('Figure shown.')

        if do_train_seen_unseen_correct_false:
            print('Do train seen unseen correct false ...')
            save_path_hists = get_save_path(save_path, 'train_eval_unseen_correct_false', arguments, group_nb, exp_nb)

            fig6 = utils.compute_density_train_seen_unseen_correct_false(
                arguments=arguments,
                all_outputs_train=all_outputs_train,
                true_labels_train=true_labels_train,
                all_outputs_seen=all_outputs_seen,
                true_labels_seen=true_labels_seen,
                all_outputs_unseen=all_outputs_unseen,
                show_fig=show_fig,
                save_fig=save_fig,
                save_path=save_path_hists,
                figsize=figsize,
            )

            if save_fig:
                print('Saving figure...')
                fig6.savefig(save_path_hists)
                save_to_file(fig6, save_path_hists.replace('png', 'pkl'))
                print('Figure saved.')
            if show_fig:
                print('Showing figures...')
                fig6.show()
                print('Figure shown.')

            print('Done')

    except np.linalg.LinAlgError:
        print(exp_nb, 'failed.')
        failors.append(exp_nb)

    print(f'Time Elapsed:{round(time() - start_time)} s.')

print('Failors: ', failors)
