import pathlib

import pandas as pd
import sklearn.metrics as metrics


###### TO CHANGE ########
from scripts.utils import get_args, get_res_args_groupnb

path_to_results = 'results/risk_coverage/cifar10'
local_path = 'polyaxon_results/groups'
nb_of_runs = 10

#########################

path = local_path

def compute_auc(results, nb_of_tests, exp_nb, unc_name):
    """
    Computes the Area Under Curve (AUC) of the accuracy / coverage function. The higher the auc, the
    better the model.
    Args:
        results (pandas.core.frame.DataFrame): the dataframe containing the accuracies, uncertainties
        nb_of_tests (int): the number of test we want to the uncertainty from
        exp_nb (str || int): the number of the experiment
        unc_name (str): the type of uncertainty

    Returns:
        float: the AUC

    """
    result_of_interest = (
        results.query(f'number_of_tests=={nb_of_tests}')
        .query(f'exp=={exp_nb}')
        .query(f'unc=="{unc_name}"')
    )

    grouped = result_of_interest.groupby('risk')

    nb_of_risks = grouped.coverage.size().iloc[0]

    covs = [grouped.coverage.apply(lambda df: df.iloc[i]) for i in range(nb_of_risks)]
    accs = [grouped.acc.apply(lambda df: df.iloc[i]) for i in range(nb_of_risks)]

    all_aucs = []
    for xs, ys in zip(covs, accs):
        all_aucs.append(metrics.auc(xs.iloc[xs.argsort()], ys.iloc[xs.argsort()]))

    return all_aucs

path_to_results = pathlib.Path(path_to_results)

for _ in nb_of_runs:
    for type in ['eval', 'train']:

        results = pd.read_csv(path_to_results / f'results_{type}.csv')

        exps = results.exp.unique()
        uncs = results.unc.unique()
        number_of_testss = results.number_of_tests.unique()

        aucs = pd.DataFrame()
        for exp_nb in exps:
            _, arguments, group_nb = get_res_args_groupnb(exp_nb, path)
            for unc in uncs:
                for number_of_tests in number_of_testss:
                    current_aucs = compute_auc(results, number_of_tests, exp_nb, unc)
                    for cur_auc in current_aucs:
                        aucs = aucs.append(pd.DataFrame.from_dict({
                            'exp_nb': [exp_nb],
                            'group_nb': [group_nb],
                            'trainset': [arguments.get('trainset', 'mnist')],
                            'rho': [arguments['rho']],
                            'std_prior': [arguments['std_prior']],
                            'loss_type': [arguments['loss_type']],
                            'unc_name': [unc],
                            'number_of_tests': [number_of_tests],
                            'auc': [cur_auc],
                        }))

        aucs.exp_nb = aucs.exp_nb.astype('int')
        aucs.to_csv(path_to_results / f'aucs_{type}.csv')
        aucs.to_pickle(path_to_results / f'aucs_{type}.pkl')
