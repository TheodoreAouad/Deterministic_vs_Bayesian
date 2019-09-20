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
save_path = './results/eval_acc/'

#########################

path_to_exps = CPUPATH

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


def main(
        exp_nbs=exp_nbs,
        path_to_exps=path_to_exps,
        path_to_results=save_path,
        nb_of_runs=nb_of_runs,
        nb_of_tests=nb_of_tests,
        device='cpu',
        **kwargs,
):
    """
    Evaluates the accuracy for the experiments given. Writes it in a csv.
    Args:
        exp_nbs (list): list of int or str. Experiments to evaluate
        path_to_exps (str): path to the experiments
        path_to_results (str): path to the directory to save the results
        nb_of_runs (int): number of times to run the same experiment for confidence interval
        nb_of_tests (int): number of tests for inference for each prediction
        device (torch.device): gpu or cpu, device to compute on
        **kwargs: args to be able to put any arguments in our functions and not raise an error.

    """

    save_path = pathlib.Path(path_to_results)

    if save_output:
        save_path.mkdir(exist_ok=True, parents=True)

    if not os.path.exists(save_path / 'all_eval_accs.pkl'):
        all_eval_accs = pd.DataFrame(columns=['exp_nb', 'group_nb', 'rho', 'std_prior', 'loss_type', 'number_of_tests'])
    else:
        all_eval_accs = load_from_file(save_path / 'all_eval_accs.pkl', )
        all_eval_accs.to_csv(save_path / 'all_eval_accs_backup.csv')

    for _ in range(nb_of_runs):
        for exp in exp_nbs:
            bay_net_trained, arguments, group_nb = get_trained_model_and_args_and_groupnb(exp, path_to_exps)
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
                'epoch': [arguments['epoch']],
                'loss_type': [arguments['loss_type']],
                'number_of_tests': [nb_of_tests],
                'eval_acc': eval_acc,
            }))

        all_eval_accs.exp_nb = all_eval_accs.exp_nb.astype('int')

        if save_output:
            all_eval_accs.to_csv(save_path / 'all_eval_accs.csv')
            all_eval_accs.to_pickle(save_path / 'all_eval_accs.pkl')

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
        path_to_exps = GPUPATH
        save_path = './output'
    else:
        device = "cpu"
        path_to_exps = CPUPATH
        save_path = './results/eval_acc/'

    main(path_to_exps=path_to_exps, path_to_results=save_path, device=device,)
