import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.append("BayesianFewShotExperiments")
from src.utils import compute_dkl_uniform


def open_experiment_results(type, exp_nb,  group_nb=None, polyaxon_path="polyaxon_results", filename = "results.pt"):
    '''

    Args:
        type (str): groups or experiment
        exp_nb (str): the number of the experiment
        group_nb (str || None): the number of the group (if type is groups)
        polyaxon_path (str): path of the parent folder of the results
        filename (str): name of the file we want to open

    Returns:
        content of the file. Most of the time it is a dict.

    '''
    if type == "groups":
        path_to_results = os.path.join(polyaxon_path, type, group_nb, exp_nb, filename)
    else:
        path_to_results = os.path.join(polyaxon_path, type, exp_nb, filename)
    return torch.load(path_to_results)


def get_interesting_result(result, number_of_classes):
    '''

    Args:
        result (dict): output of the polyaxon experiment
        number_of_classes (int): number of classes of the polyaxon experiment (ex: MNIST, 10)

    Returns:
        interesting_result (dict): dictionary of the desired parameters we would like to write in a csv

    '''
    interesting_result = dict()
    for key, value in result.items():
        if key != "random output":
            if "train" in key:
                interesting_result[key] = value[-1][-1]
            elif "uncertainty" in key or "dkls" in key:
                interesting_result[key + "-mean"] = value.mean().item()
                interesting_result[key + "-std"] = value.std().item()
            else:
                interesting_result[key] = value

        # This part is deprecated. Its use affects previous experiments, but should not affect future experiments.
        else:
            random_output = value.numpy().T
            dkls = np.zeros(random_output.shape[0])
            for i, output in enumerate(random_output):
                values, count = np.unique(output, return_counts=True)
                dkls[i] = compute_dkl_uniform(count, number_of_classes)
            interesting_result["DKL(p||uniform)"] = dkls.mean()
    return interesting_result


def write_results_in_csv(results, name="results/results.csv"):
    '''

    This function takes as input a tuple of results from multiple experiments
    and writes a csv.

    Args:
        results (dict): results we want to write in csv
        name (dict): path and name of the file we want to write in

    Returns:
        None

    '''

    file = open(name,"w")
    writer = csv.DictWriter(file, results[0].keys())
    writer.writeheader()
    for result in results:
        writer.writerow(result)
    file.close()


def get_file_path_in_dir(dir_path, file_name=""):
    '''
    Get all the paths of the files in a certain directory, with a restriction on the file_name
    Args:
        dir_path (str): directory we want to scan
        file_name (str): strign we want in the name

    Returns:
        all_files (list): list of the paths of all the files
    '''
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for file in filenames:
            if file_name in file:
                all_files.append(os.path.join(dirpath, file))
    return all_files


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
    results.append(get_interesting_result(result, 10))
write_results_in_csv(results, filename+'.csv')
