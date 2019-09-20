"""
This file computes the deadzone of each uncertainty for each of the experiments.
To use:
- change the TO CHANGE parameters
- run in console
"""
import os
import pathlib
from time import time

import numpy as np
import pandas as pd
import torch
from math import sqrt

from scripts.utils import get_trained_model_and_args_and_groupnb, get_seen_outputs_and_labels, get_unseen_outputs, \
    get_res_args_groupnb, get_args
from src.uncertainty_measures import get_all_uncertainty_measures, get_all_uncertainty_measures_not_bayesian
from src.uncertainty_metric import get_deadzones, get_deadzone_from_unc
from src.utils import set_and_print_random_seed, get_unc_key, save_to_file, load_from_file

CPUPATH = 'polyaxon_results/groups'
GPUPATH = '/output/sicara/BayesianFewShotExperiments/groups/'

######## TO CHANGE ##############

# exp_nbs = ['3713', '3719', '3749', '3778', '3716', '3722', '3752', '3781',
#            '3832', '3834', '3839', '3840',
#            '3842', '3851', '3861', '3864',
#            ]
exp_nbs = [4789]
n = 100
nb_of_runs = 2
number_of_tests = 10
save_path = f'results/deadzones/todelete'

verbose = True
nb_of_random = 5000
do_recompute_outputs = True
save_csv = True
#################################

path_to_exps = CPUPATH



def main(
        exp_nbs=exp_nbs,
        path_to_results=save_path,
        path_to_exps=path_to_exps,
        n=n,
        nb_of_runs=nb_of_runs,
        number_of_tests=number_of_tests,
        verbose=verbose,
        nb_of_random=nb_of_random,
        do_recompute_outputs=do_recompute_outputs,
        save_csv=save_csv,
        device='cpu',
):
    save_path = pathlib.Path(path_to_results)
    if do_recompute_outputs:
        save_path = save_path / 'recomputed'
    else:
        save_path = save_path / 'saved_from_polyaxon'
    save_path.mkdir(exist_ok=True, parents=True)
    if not do_recompute_outputs:
        nb_of_runs = 1

    if not os.path.exists(save_path / 'deadzones.pkl'):
        deadzones = pd.DataFrame(columns=['group_nb', 'exp_nb', 'unc_name'])
    else:
        deadzones = load_from_file(save_path / 'deadzones.pkl', )
        deadzones.to_csv(save_path / 'deadzones.csv')

    recomputed_exps = []

    start_time = time()

    for repeat_idx in range(nb_of_runs):
        for exp_nb in exp_nbs:
            print(f'Repeat number {repeat_idx + 1} / {nb_of_runs}, Exp nb {exp_nb}')
            arguments = get_args(exp_nb, path_to_exps)
            determinist = arguments.get('rho', 'determinist') == 'determinist'

            def recompute_outputs(deadzones):
                bay_net_trained, arguments, group_nb = get_trained_model_and_args_and_groupnb(exp_nb, exp_path=path_to_exps)
                bay_net_trained.to(device)

                arguments['number_of_tests'] = number_of_tests

                all_eval_outputs, _ = get_seen_outputs_and_labels(
                    bay_net_trained,
                    arguments,
                    device=device,
                    verbose=verbose,
                )

                all_outputs_unseen = get_unseen_outputs(
                    bay_net_trained,
                    arguments,
                    nb_of_random,
                    device=device,
                    verbose=verbose,
                )

                if determinist:
                    dzs = get_deadzones(all_eval_outputs, all_outputs_unseen, get_all_uncertainty_measures_not_bayesian, n)
                    iterator = zip(['us', 'pe'], dzs)
                else:
                    dzs = get_deadzones(all_eval_outputs, all_outputs_unseen, get_all_uncertainty_measures, n)
                    iterator = zip(['vr', 'pe', 'mi'], dzs)
                for unc_name, dz in iterator:
                    deadzones = deadzones.append(pd.DataFrame.from_dict({
                        'group_nb': [group_nb],
                        'exp_nb': [exp_nb],
                        'type_of_unseen': [arguments['type_of_unseen']],
                        'epoch': [arguments['epoch']],
                        'number_of_tests': [arguments['number_of_tests']],
                        'unc_name': [unc_name],
                        f'dz_{n}': [dz],
                    }))
                return deadzones

            if do_recompute_outputs:
                deadzones = recompute_outputs(deadzones)

            else:
                try:
                    results, arguments, group_nb = get_res_args_groupnb(exp_nb, exp_path=path_to_exps)
                except RuntimeError as e:
                    if e.__str__() == "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() " \
                                      "is False. If you are running on a CPU-only machine, please use torch.load with " \
                                      "map_location='cpu' to map your storages to the CPU.":
                        recompute_outputs()
                        recomputed_exps.append(exp_nb)
                        continue
                    else:
                        raise e


                def seen_and_unseen_and_n(results, unc, n):
                    return (results.get(get_unc_key(results.columns, f'seen {unc}'), [torch.tensor([-1], dtype=torch.float)])[0],
                            results.get(get_unc_key(results.columns, f'unseen {unc}'), [torch.tensor([-1], dtype=torch.float)])[0],
                            n)

                try: dz_pe = get_deadzone_from_unc(*seen_and_unseen_and_n(results, 'pe', n))
                except: dz_pe = -1

                if determinist:
                    dz_us = get_deadzone_from_unc(*seen_and_unseen_and_n(results, 'us', n))
                    iterator = zip(['us', 'pe'], [dz_us, dz_pe])
                else:
                    dz_vr = get_deadzone_from_unc(*seen_and_unseen_and_n(results, 'vr', n))
                    dz_mi = get_deadzone_from_unc(*seen_and_unseen_and_n(results, 'mi', n))
                    iterator = zip(['vr', 'pe', 'mi'], [dz_vr, dz_pe, dz_mi])
                for unc_name, dz in iterator:
                    deadzones = deadzones.append(pd.DataFrame.from_dict({
                        'deadzone_number': [n],
                        'group_nb': [group_nb],
                        'exp_nb': [exp_nb],
                        'type_of_unseen': [arguments['type_of_unseen']],
                        'epoch': [arguments['epoch']],
                        'number_of_tests': [arguments['number_of_tests']],
                        'unc_name': [unc_name],
                        f'dz_{n}': [dz],
                    }))

            print(f'Time Elapsed:{round(time() - start_time)} s.')

    deadzones.exp_nb = deadzones.exp_nb.astype('int')

    if save_csv:

        save_to_file(arguments, save_path / 'arguments.pkl')
        deadzones.sort_values('exp_nb')
        deadzones.to_pickle(save_path / 'deadzones.pkl')
        deadzones.to_csv(save_path / 'deadzones.csv')

    print(deadzones)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
        path_to_exps = GPUPATH
        save_path = './output'
    else:
        device = "cpu"
        path_to_exps = CPUPATH
    device = torch.device(device)
    print(device)

    main(path_to_exps=path_to_exps, path_to_results=save_path, device=device,)
