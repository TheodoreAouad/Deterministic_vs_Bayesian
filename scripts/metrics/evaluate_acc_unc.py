import os
import pathlib

import pandas as pd
import torch

from scripts.utils import get_trained_model_and_args_and_groupnb, get_evalloader_unseen
from src.dataset_manager.get_data import get_cifar10, get_mnist
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_all_uncertainty_measures_not_bayesian, get_all_uncertainty_measures
from src.utils import save_to_file, load_from_file, convert_tensor_to_float

CPUPATH = 'polyaxon_results/groups'
GPUPATH = '/output/sicara/BayesianFewShotExperiments/groups/'

###### TO CHANGE ########

# exp_nbs = ['19989', '20008', '19973', '20001', '19987', '19980', '20006', '19974', '20007', '19975',
#            '19981', '19986', '19972', '20000', '20009', '19988', '20013', '19995', '19992', '19966',
#            '20014', '20022', '19968', '19969', '19967', '20015', '19993', '19994', '20012', '19979',
#            '19983', '20005', '19977', '19970', '20002', '19984', '19985', '19971', '20003', '20004',
#            '19976', '19982', '19978', '19991', '19965', '20017', '20010', '19996', '20019', '19998',
#            '20021', '19999', '20020', '20018', '19997', '20011', '19963', '19964', '20016', '19990',
#            ]
# exp_nbs = ['14617', '14746', '14681', '14627', '14748', '14689', '14633', '14754', '14695']
exp_nbs = [14621]
nb_of_runs = 5
nb_of_tests = 10
save_output = True
save_path = './results/acc_unc_means/'

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
    trainset = arguments.get('trainset', 'mnist')
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

    if not os.path.exists(save_path / 'all_accs_uncs_mean.pkl'):
        all_accs_uncs_mean = pd.DataFrame(columns=['exp_nb', 'group_nb', 'rho', 'std_prior', 'loss_type', 'number_of_tests'])
    else:
        all_accs_uncs_mean = load_from_file(save_path / 'all_accs_uncs_mean.pkl', )
        all_accs_uncs_mean.to_csv(save_path / 'all_uncs_mean_backup.csv')

    for _ in range(nb_of_runs):
        for exp in exp_nbs:
            print(f'Run {_+1}, Exp {exp}, computing accuracy and uncertainty...')
            bay_net_trained, arguments, group_nb = get_trained_model_and_args_and_groupnb(exp, path_to_exps)
            evalloader_seen = get_evalloader_seen(arguments)
            evalloader_unseen = get_evalloader_unseen(arguments)

            eval_acc, all_outputs_seen = eval_bayesian(
                bay_net_trained,
                evalloader_seen,
                number_of_tests=nb_of_tests if arguments.get('rho', 'determinist') != 'determinist' else 1,
                return_accuracy=True,
                device=device,
                verbose=True,
            )

            _, all_outputs_unseen = eval_bayesian(
                bay_net_trained,
                evalloader_unseen,
                number_of_tests=nb_of_tests if arguments.get('rho', 'determinist') != 'determinist' else 1,
                return_accuracy=True,
                device=device,
                verbose=True,
            )

            determinist = (
                arguments.get('determinist', False) or
                arguments.get('rho', 'determinist') == 'determinist'
            )

            if determinist:
                get_unc_func = get_all_uncertainty_measures_not_bayesian
                unc_names = ['us', 'pe']
            else:
                get_unc_func = get_all_uncertainty_measures
                unc_names = ['vr', 'pe', 'mi']

            all_uncs_mean_seen = get_unc_func(all_outputs_seen)
            all_uncs_mean_unseen = get_unc_func(all_outputs_unseen)

            for unc_name, unc_seen, unc_unseen in zip(unc_names, all_uncs_mean_seen, all_uncs_mean_unseen):

                all_accs_uncs_mean = all_accs_uncs_mean.append(pd.DataFrame.from_dict({
                    'exp_nb': [exp],
                    'group_nb': [group_nb],
                    'split_labels': [arguments.get('split_labels', 10)],
                    'trainset': [arguments.get('trainset', 'mnist')],
                    'rho': [arguments.get('rho', 'determinist')],
                    'std_prior': [arguments.get('std_prior', -1)],
                    'epoch': [arguments['epoch']],
                    'loss_type': [arguments.get('loss_type', 'criterion')],
                    'number_of_tests': [nb_of_tests],
                    'eval_acc': [eval_acc],
                    'unc_name': [unc_name],
                    'unc_seen': [unc_seen.mean().item()],
                    'unc_unseen': [unc_unseen.mean().item()],
                    'ratio': [unc_unseen.mean().item()/unc_seen.mean().item()],
                }))

            all_accs_uncs_mean.exp_nb = all_accs_uncs_mean.exp_nb.astype('int')

            if save_output:
                all_accs_uncs_mean.to_csv(save_path / 'all_accs_uncs_mean.csv')
                all_accs_uncs_mean.to_pickle(save_path / 'all_accs_uncs_mean.pkl')

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
        path_to_exps = GPUPATH
        save_path = './output'
    else:
        device = "cpu"
        path_to_exps = CPUPATH
        save_path=save_path
    main(path_to_exps=path_to_exps, path_to_results=save_path, device=device,)
