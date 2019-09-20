import argparse
import pathlib

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--which_parameters',
                    help='List of the parameters to sort the dataframe on or path to a file with the parameters',
                    )
parser.add_argument('--which_values', help='List of the values of the dataframe to keep or path to file with values',
                    )
parser.add_argument('--results_dir_path', help='Path to the directory of the results', type=str,
                    default='results/')
parser.add_argument('--polyaxon_type', help='type of the experiments. Is either groups or experiments',
                    choices=['group', 'experiment'], type=str, default='group')
parser.add_argument('--exp_nb',
                    help='number of the experiment if polyaxon_type is experiments, number of groups if groups',
                    type=str)
parser.add_argument('--extra_info', help='extra info to write on the name of the csv.', type=str,
                    default='')
args = parser.parse_args()

which_parameters = args.which_parameters
which_values = args.which_values
results_dir_path = pathlib.Path(args.results_dir_path) / 'raw_results/all_columns'
save_dir_path = pathlib.Path(args.results_dir_path) / 'raw_results/specific_columns'
polyaxon_type = args.polyaxon_type
exp_nb = args.exp_nb
extra_info = args.extra_info

if type(which_parameters) == str:
    with open(which_parameters, 'r') as f:
        which_parameters_raw = f.read().splitlines()
        which_parameters = which_parameters_raw.copy()

if type(which_values) == str:
    with open(which_values, 'r') as f:
        which_values_raw = f.read().splitlines()
which_values = set(which_values_raw)-set(which_parameters_raw)

filename = polyaxon_type + exp_nb + extra_info
all_results = pd.read_pickle(results_dir_path / (filename + '.pkl'))
to_keep = set(which_parameters).intersection(set(all_results.keys()))
which_parameters = [p for p in which_parameters if p in to_keep]
all_results_sorted = all_results.sort_values(which_parameters)
which_values = which_values.intersection(set(all_results.keys()))
operations = {
    col: 'mean' for col in which_values
}
operations.update({
    "experiment": lambda x: list(x)
})
which_values.add('experiment')
specific_results = all_results_sorted.groupby(which_parameters,).agg(operations)
exps = all_results_sorted.groupby(which_parameters,).agg(lambda x: list(x)).experiment
specific_results = specific_results[which_values]
specific_results['experiment'] = exps
specific_results.reset_index(inplace=True)
specific_results = specific_results.reindex(['experiment'] + which_parameters_raw + which_values_raw, axis=1)

if specific_results.rho.iloc[0] == 'determinist':
    uncs = ['us', 'pe']
else:
    uncs = ['vr', 'pe', 'mi']
for unc in uncs:
    specific_results[f'{unc} unseen/seen'] = (specific_results[f'unseen uncertainty {unc}-mean'] /
                                              specific_results[f'seen uncertainty {unc}-mean'])
save_dir_path.mkdir(exist_ok=True, parents=True)
specific_results.to_pickle(save_dir_path / (filename + '_specific_results.pkl'))
specific_results.to_csv(save_dir_path / (filename + '_specific_results.csv'))
