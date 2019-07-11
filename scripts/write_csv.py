import argparse
import os
import sys

import torch

sys.path.append("BayesianFewShotExperiments")
from src.utils import get_interesting_result, write_results_in_csv, get_file_path_in_dir

parser = argparse.ArgumentParser()
parser.add_argument("--polyaxon_results_path")
parser.add_argument("--polyaxon_type")
parser.add_argument("--group_nb")
parser.add_argument("--exp_nb")
parser.add_argument("--which_file")
parser.add_argument("--extra_info")
args = parser.parse_args()

polyaxon_results_path = args.polyaxon_results_path
polyaxon_type = args.polyaxon_type
group_nb = args.group_nb
exp_nb = args.exp_nb
which_file = args.which_file
extra_info = args.extra_info

filename = "results/"
if group_nb is not None:
    all_files = get_file_path_in_dir(os.path.join(polyaxon_results_path, polyaxon_type, group_nb), which_file)
    filename = "results/group" + str(group_nb) + str(extra_info)
elif exp_nb is not None:
    all_files = get_file_path_in_dir(os.path.join(polyaxon_results_path, polyaxon_type, exp_nb), which_file)
    filename = "results/experiment" + str(exp_nb) + str(extra_info)

results = []
for file_path in all_files:
    result = torch.load(file_path, map_location="cpu")
    results.append(get_interesting_result(result))
    print(f'{len(results)} results loaded.')

write_results_in_csv(results, filename + '.csv')
