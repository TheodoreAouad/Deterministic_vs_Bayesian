import os
import pathlib

import pandas as pd
import torch

from scripts.utils import get_trained_model_and_args_and_groupnb
from src.dataset_manager.get_data import get_cifar10, get_mnist
from src.tasks.evals import eval_bayesian
from src.utils import save_to_file, load_from_file

CPUPATH = 'polyaxon_results/groups'
GPUPATH = '/output/sicara/BayesianFewShotExperiments/groups/'

###### TO CHANGE ########

exp_nbs = [4789]
nb_of_runs = 1
nb_of_tests = 2
save_output = True

#########################


if torch.cuda.is_available():
    device = "cuda"
    path = GPUPATH
    save_path = './output'
else:
    device = "cpu"
    path = CPUPATH
    save_path = './results/eval_acc/'
save_path = pathlib.Path(save_path)

args = {
    'exp_nbs': exp_nbs,
    'nb_of_runs': nb_of_runs,
    'nb_of_tests': nb_of_tests,
}
if save_output:
    save_path.mkdir(exist_ok=True, parents=True)
    save_to_file(args, save_path / 'args.pkl')

if not os.path.exists(save_path / 'all_eval_accs.pkl'):
    all_eval_accs = pd.DataFrame(columns=['exp_nb', 'group_nb', 'rho', 'std_prior', 'loss_type', 'number_of_tests'])
else:
    all_eval_accs = load_from_file(save_path / 'all_eval_accs.pkl', )
    all_eval_accs.to_csv(save_path / 'all_eval_accs_backup.csv')

def get_evalloader_seen(arguments):
    """
    Gets the evalloader for training set
    Args:
        arguments (dict): arguments given to the experiment. Contains all the info about trainset and split of classes.

    Returns:
        torch.utils.data.dataloader.DataLoader: dataloader of the test data on seen dataset

    """
    trainset = arguments['trainset']
    split_labels = arguments.get('split_labels', 10)
    if trainset == 'mnist':
        get_trainset = get_mnist
    elif trainset == 'cifar10':
        get_trainset = get_cifar10
    _, _, evalloader_seen = get_trainset(train_labels=(), eval_labels=range(split_labels), split_val=0)
    return evalloader_seen


for exp in exp_nbs:

    bay_net_trained, arguments, group_nb = get_trained_model_and_args_and_groupnb(exp, path)
    evalloader_seen = get_evalloader_seen(arguments)

    eval_acc, _ = eval_bayesian(
        bay_net_trained,
        evalloader_seen,
        number_of_tests=nb_of_tests if arguments['rho'] != 'determinist' else 1,
        return_accuracy=True,
        device=device,
        verbose=True,
    )

    all_eval_accs = all_eval_accs.append(pd.DataFrame.from_dict({
        'exp_nb': [exp],
        'group_nb': [group_nb],
        'split_labels': [arguments.get('split_labels', 10)],
        'trainset': [arguments['trainset']],
        'rho': [arguments['rho']],
        'std_prior': [arguments['std_prior']],
        'loss_type': [arguments['loss_type']],
        'number_of_tests': [arguments['number_of_tests']],
        'eval_acc': eval_acc,
    }))

print(all_eval_accs)
all_eval_accs.exp_nb = all_eval_accs.exp_nb.astype('int')

if save_output:
    all_eval_accs.to_csv(save_path / 'all_eval_accs.csv')
    all_eval_accs.to_pickle(save_path / 'all_eval_accs.pkl')
