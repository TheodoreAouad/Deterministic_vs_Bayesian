# %% Imports
import pathlib
from importlib import reload
import os
from time import time

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import scripts.utils as su
import src.uncertainty_measures as um
import src.utils as u
import src.tasks.evals as e
import src.grapher as g

def reload_modules():
    modules_to_reload = [su, u, um, e, g]
    for module in modules_to_reload:
        reload(module)

cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']
reload_modules()

if torch.cuda.is_available():
    device = "cuda"
    path_to_res = '/output/sicara/BayesianFewShotExperiments/groups/'
else:
    device = "cpu"
    path_to_res = 'polyaxon_results/groups'
device = torch.device(device)
print(device)



def get_ratio(df_det, unc_name):
    return df_det[f'unseen_uncertainty_{unc_name}'].iloc[0].mean() / df_det[f'seen_uncertainty_{unc_name}'].iloc[0].mean()


# %% Get bayesian args.

#CIFAR 10
# all_exps = [
#     20263, 20269, 20275, 20284, 20290, 20296, 20381, 20384, 20387,
#     20397, 20403, 20409, 20418, 20424, 20430, 20515, 20518, 20521,
#     20529, 20535, 20541, 20550, 20556, 20562, 20647, 20650, 20653,
# ]

#MNIST
all_exps = [
    21276, 21282, 21288, 21296, 21302, 21308, 21394, 21399, 21405,
    21400, 21406, 21412, 21438, 21444, 21450, 21526, 21531, 21537,
    21532, 21538, 21544, 21571, 21577, 21583, 21658, 21661, 21664,
]
# all_exps = [20263, 20269]
list_of_nb_of_tests = [10]
save_fig = True
path_to_outputs = pathlib.Path(f'temp/softmax_outputs/')
save_path = f'rapport/bayesian_results/histograms/'
save_path = pathlib.Path(save_path)
unc_names = ['sr', 'vr', 'pe', 'mi', 'au', 'eu']

start_tot = time()
start_exp = start_tot
for exp_idx, exp in enumerate(all_exps):
    print(f'========================')
    print(f'{exp_idx+1}/{len(all_exps)}')
    res = {}
    res_unseen_bay = {}
    reload_modules()
    verbose = True

    bay_net_trained, arguments, _ = su.get_trained_model_and_args_and_groupnb(exp, path_to_res)
    evalloader_seen = su.get_evalloader_seen(arguments, shuffle=False)
    arguments['type_of_unseen'] = 'unseen_dataset'
    evalloader_unseen = su.get_evalloader_unseen(arguments, shuffle=False)

    dont_show = ['save_loss', 'save_observables', 'save_outputs', 'type_of_unseen', 'unseen_evalset',
                 'split_labels', 'number_of_tests', 'split_train', 'exp_nb']

    is_determinist = arguments.get('determinist', False) or arguments.get('rho', 'determinist') == 'determinist'
    assert not is_determinist, 'network determinist'

    for number_of_tests in list_of_nb_of_tests:

        bay_net_trained, arguments, _ = su.get_trained_model_and_args_and_groupnb(exp, path_to_res)
        bay_net_trained.to(device)
        if number_of_tests < 100 and os.path.exists(path_to_outputs / '100' / f'{exp}/true_labels_seen.pt'):

            true_labels_seen = torch.load(path_to_outputs / f'100' / f'{exp}/true_labels_seen.pt')
            all_outputs_seen = torch.load(path_to_outputs / f'100' / f'{exp}/all_outputs_seen.pt')
            all_outputs_unseen = torch.load(path_to_outputs / f'100' / f'{exp}/all_outputs_unseen.pt')

            random_idx = np.arange(100)
            np.random.shuffle(random_idx)
            random_idx = random_idx[:number_of_tests]
            all_outputs_seen = all_outputs_seen[random_idx]
            all_outputs_unseen = all_outputs_unseen[random_idx]
        elif os.path.exists(path_to_outputs / f'{number_of_tests}' / f'{exp}/true_labels_seen.pt'):
            true_labels_seen = torch.load(path_to_outputs / f'{number_of_tests}' / f'{exp}/true_labels_seen.pt')
            all_outputs_seen = torch.load(path_to_outputs / f'{number_of_tests}' / f'{exp}/all_outputs_seen.pt')
            all_outputs_unseen = torch.load(path_to_outputs / f'{number_of_tests}' / f'{exp}/all_outputs_unseen.pt')
        else:
            (path_to_outputs / f'{number_of_tests}' / f'{exp}').mkdir(exist_ok=True, parents=True)
            evalloader_seen = su.get_evalloader_seen(arguments)
            # BE CAREFUL: in the paper, the process is tested on the enterity of the unseen classes
            evalloader_unseen = su.get_evalloader_unseen(arguments)
            true_labels_seen, all_outputs_seen = e.eval_bayesian(
                model=bay_net_trained,
                evalloader=evalloader_seen,
                number_of_tests=number_of_tests,
                return_accuracy=False,
                device=device,
                verbose=True,
            )

            _, all_outputs_unseen = e.eval_bayesian(
                model=bay_net_trained,
                evalloader=evalloader_unseen,
                number_of_tests=number_of_tests,
                device=device,
                verbose=True,
            )

            torch.save(true_labels_seen, path_to_outputs / f'{number_of_tests}' / f'{exp}/true_labels_seen.pt')
            torch.save(all_outputs_seen, path_to_outputs / f'{number_of_tests}' / f'{exp}/all_outputs_seen.pt')
            torch.save(all_outputs_unseen, path_to_outputs / f'{number_of_tests}' / f'{exp}/all_outputs_unseen.pt')


        all_outputs_seen, all_outputs_unseen = all_outputs_seen.cpu(), all_outputs_unseen.cpu()
        true_labels_seen = true_labels_seen.cpu()
        preds = um.get_predictions_from_multiple_tests(all_outputs_seen)

        res[number_of_tests] = pd.DataFrame()
        uncs = um.get_all_uncertainty_measures(all_outputs_seen)
        res[number_of_tests] = (
            res[number_of_tests]
            .assign(true=true_labels_seen)
            .assign(preds=preds)
            .assign(correct_pred=lambda df: (df.true == df.preds))
        )
        for unc, unc_name in zip(uncs, unc_names):
            res[number_of_tests][unc_name] = unc

        res_unseen_bay[number_of_tests] = pd.DataFrame()
        uncs_unseen = um.get_all_uncertainty_measures(all_outputs_unseen)
        for unc, unc_name in zip(uncs, unc_names):
            res_unseen_bay[number_of_tests][unc_name] = unc

        print(f'Nb Of Tests: {number_of_tests}. Accuracy: ', res[number_of_tests].correct_pred.mean())

    reload_modules()


    res_correct = res[number_of_tests].loc[res[number_of_tests].correct_pred == True]
    res_false = res[number_of_tests].loc[res[number_of_tests].correct_pred == False]
    res_unseen = res_unseen_bay[number_of_tests]

    all_uncs = [[res_correct[unc_name], res_false[unc_name], res_unseen[unc_name]] for unc_name in unc_names]

    unc_labels = ('true', 'false', 'unseen')
    for idx, (unc_name, uncs) in enumerate(zip(unc_names, all_uncs)):
        fig = plt.figure(figsize=(7, 5))
        # ax = fig.add_subplot(len(unc_names), 1, idx+1)
        ax = fig.add_subplot(111)
        ax.set_title(f'{unc_name}, rho={arguments["rho"]}, std={arguments["std_prior"]}, T={number_of_tests}, {arguments["loss_type"]}\n'
                     f'Accuracy: {round(res[number_of_tests].correct_pred.mean()*100, 2)}%')
        uncs_to_show = [unc.loc[unc < 1*unc.max()] for unc in uncs]
        g.plot_uncertainty(
            ax,
            unc_name,
            uncs,
            unc_labels,
        )
        ax.legend()

        (save_path / f'{arguments["trainset"]}/images/{arguments["loss_type"]}/{unc_name}/').mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path/f'{arguments["trainset"]}/images/{arguments["loss_type"]}/{unc_name}/{arguments["loss_type"]}_{unc_name}_rho{arguments["rho"]}_std{arguments["std_prior"]}_T{number_of_tests}.png')
        (save_path / f'{arguments["trainset"]}/pickles/{arguments["loss_type"]}/{unc_name}/').mkdir(exist_ok=True, parents=True)
        u.save_to_file(fig, save_path/f'{arguments["trainset"]}/pickles/{arguments["loss_type"]}/{unc_name}/{arguments["loss_type"]}_{unc_name}_rho{arguments["rho"]}_std{arguments["std_prior"]}_T{number_of_tests}.pkl')
        plt.close(fig)

    print('This exp time elapsed:', round(time() - start_exp), 's')
    print('Total time elapsed:', round(time() - start_tot), 's')
    start_exp = time()
