"""
This script is used to create the figures of accuracies against uncertainty. This is used to get results of a single
experiment. It cannot be used to create a figure across multiple experiments.
To use it, change the arguments in the TO CHANGE box.
"""
import pathlib
from time import time

import torch

from scripts.utils import compute_figures, compute_density, get_seen_outputs_and_labels, get_unseen_outputs, \
    get_trained_model_and_args_and_groupnb, get_train_outputs

###### TO CHANGE ###########
# exp_nbs = ['3713', '3719', '3749', '3778', '3716', '3722', '3752', '3781', '3832', '3834', '3839',
#            '3840', '3842', '3851', '3861', '3864']

exp_nbs = ['3713', '3832']
nb_of_batches = 1000
size_of_batch = 100
nb_of_random = 5000
do_compute_correlation = False
do_compute_histogram = True
show_fig = True
save_fig = False
do_eval_mnist = True


############################


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)
start_time = time()
for exp_nb in exp_nbs:
    print(f'{exp_nb}')

    bay_net_trained, arguments, group_nb = get_trained_model_and_args_and_groupnb(exp_nb)

    all_eval_outputs, true_labels_mnist = get_seen_outputs_and_labels(
        bay_net_trained,
        arguments,
        device=device,
    )

    all_outputs_unseen = get_unseen_outputs(
        bay_net_trained,
        arguments,
        nb_of_random,
        device=device,
    )

    if do_compute_correlation:
        save_path_cor = pathlib.Path(f'results/correlations_figures/{arguments.get("loss_type", "determinist")}')
        save_path_cor.mkdir(parents=True, exist_ok=True)
        save_path_cor = save_path_cor / f'{group_nb}_{exp_nb}_correlation_uncertainty_error.png'

        compute_figures(
            arguments=arguments,
            all_outputs_seen=all_eval_outputs,
            true_labels_seen=true_labels_mnist,
            all_outputs_unseen=all_outputs_unseen,
            nb_of_batches=nb_of_batches,
            size_of_batch=size_of_batch,
            scale='linear',
            show_fig=show_fig,
            save_fig=save_fig,
            save_path=save_path_cor,
            figsize=(10, 10),
        )

    if do_compute_histogram:
        all_outputs_train = get_train_outputs(
            bay_net_trained,
            arguments,
            device=device,
        )
        save_path_hists = pathlib.Path(f'results/uncertainty_density/{arguments.get("loss_type", "determinist")}')
        save_path_hists.mkdir(parents=True, exist_ok=True)
        save_path_hists = save_path_hists / f'{group_nb}_{exp_nb}_uncertainty_density.png'

        compute_density(
            arguments=arguments,
            all_outputs_train=all_outputs_train,
            all_outputs_seen=all_eval_outputs,
            all_outputs_unseen=all_outputs_unseen,
            show_fig=show_fig,
            save_fig=save_fig,
            save_path=save_path_hists,
            figsize=(12, 10),
        )

    print(f'Time Elapsed:{round(time() - start_time)} s.')


