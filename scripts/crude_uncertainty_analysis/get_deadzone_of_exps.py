"""
This file computes the deadzone of each uncertainty for each of the experiments.
To use:
- change the TO CHANGE parameters
- run in console
"""

import pathlib
from time import time

import pandas as pd
import torch
from math import sqrt

from scripts.utils import get_trained_model_and_args_and_groupnb, get_seen_outputs_and_labels, get_unseen_outputs, \
    get_res_args_groupnb
from src.uncertainty_measures import get_all_uncertainty_measures
from src.uncertainty_metric import get_deadzones, get_deadzone_from_unc
from src.utils import set_and_print_random_seed, get_unc_key, save_to_file

CPUPATH = 'polyaxon_results/groups'
GPUPATH = '/output/sicara/BayesianFewShotExperiments/groups/'

######## TO CHANGE ##############

exp_nbs = ['3713', '3719', '3749', '3778', '3716', '3722', '3752', '3781', '3832', '3834', '3839',
           '3840', '3842', '3851', '3861', '3864']
# exp_nbs = ['3851']
n = 100
exp_path = CPUPATH
nb_of_repeats = 20
verbose = False
nb_of_random = 5000
do_recompute_outputs = False
save_csv = True
save_path = f'results/deadzones/{n}/saved_from_polyaxon'
#################################

if not do_recompute_outputs:
    nb_of_repeats = 1

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

deadzones = pd.DataFrame(columns=['group', 'exp', 'vr', 'pe', 'mi'])
deadzones_aggregated = None
recomputed_exps = []

set_and_print_random_seed(1730311801)
start_time = time()

for repeat_idx in range(nb_of_repeats):
    for exp_nb in exp_nbs:
        print(f'Repeat number {repeat_idx + 1} / {nb_of_repeats}, Exp nb {exp_nb}')


        def recompute_outputs():
            global deadzones
            global arguments
            bay_net_trained, arguments, group_nb = get_trained_model_and_args_and_groupnb(exp_nb, exp_path=exp_path)
            bay_net_trained.to(device)

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
            dzs = get_deadzones(all_eval_outputs, all_outputs_unseen, get_all_uncertainty_measures, 100)
            deadzones = deadzones.append(pd.DataFrame.from_dict({
                'group': [group_nb],
                'exp': [exp_nb],
                'vr': [dzs[0]],
                'pe': [dzs[1]],
                'mi': [dzs[2]],
            }))


        if do_recompute_outputs:
            recompute_outputs()

        else:
            try:
                results, arguments, group_nb = get_res_args_groupnb(exp_nb, exp_path=exp_path)
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
                return (results.get(get_unc_key(results, f'seen {unc}'), [torch.tensor([-1], dtype=torch.float)])[0],
                        results.get(get_unc_key(results, f'unseen {unc}'), [torch.tensor([-1], dtype=torch.float)])[0],
                        n)


            dz_vr = get_deadzone_from_unc(*seen_and_unseen_and_n(results, 'vr', n))
            dz_pe = get_deadzone_from_unc(*seen_and_unseen_and_n(results, 'pe', n))
            dz_mi = get_deadzone_from_unc(*seen_and_unseen_and_n(results, 'mi', n))

            deadzones = deadzones.append(pd.DataFrame.from_dict({
                'group': [group_nb],
                'exp': [exp_nb],
                'vr': [dz_vr],
                'pe': [dz_pe],
                'mi': [dz_mi],
            }))
        print(f'Time Elapsed:{round(time() - start_time)} s.')

if save_csv:
    save_path = pathlib.Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    save_to_file(arguments, save_path / 'arguments.pkl')
    deadzones.sort(columns='exp')
    deadzones.to_pickle(save_path / 'deadzones.pkl')
    deadzones.to_csv(save_path / 'deadzones.csv')

    if nb_of_repeats > 1:
        grped = deadzones.groupby('exp')
        means = grped.mean()
        stds = grped.stds() / sqrt(nb_of_repeats)
        means_str = means.applymap(lambda x: str(round(x, 2)))
        stds_str = stds.applymap(lambda x: str(round(x, 2)))
        deadzones_aggregated = means_str + ' +- ' + stds_str
        deadzones_aggregated['group'] = means['group']
        deadzones_aggregated.drop(columns='Unnamed: 0', inplace=True)
        deadzones_aggregated['nb of rep'] = nb_of_repeats
        deadzones_aggregated.to_csv(save_path / 'deadzones_aggregated.csv')

print(recomputed_exps)
if deadzones_aggregated:
    print(deadzones_aggregated)
else:
    print(deadzones)
