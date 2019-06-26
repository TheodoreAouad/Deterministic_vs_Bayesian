import csv
import math
import torch
import numpy as np
import os


def reset_parameters_conv(weight, bias=None):

    size = weight.size()
    in_channels, kernel_size = size[1], size[2:]

    n = in_channels
    for k in kernel_size:
        n *= k
    stdv = 1. / math.sqrt(n)
    weight.data.uniform_(-stdv, stdv)
    if bias is not None:
        bias.data.uniform_(-stdv, stdv)


def reset_parameters_linear(weight, bias=None):
    stdv = 1. / math.sqrt(weight.size(1))
    weight.data.uniform_(-stdv, stdv)
    if bias is not None:
        bias.data.uniform_(-stdv, stdv)


def set_and_print_random_seed(random_seed=None, show=False, save=False, checkpoint_dir='./'):
    '''
    Set and print numpy random seed, for reproducibility of the training,
    and set torch seed based on numpy random seed
    Args:
        random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
        save (bool): if True, the numpy random seed is saved in seeds.txt
        checkpoint_dir (str): output folder where the seed is saved
    Returns:
        int: numpy random seed

    '''
    if random_seed is None:
        random_seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)
    torch.manual_seed(np.random.randint(0, 2**32-1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if show:
        prompt = 'Random seed : {}\n'.format(random_seed)
        print(prompt)

    if save:
        with open(os.path.join(checkpoint_dir, 'seeds.txt'), 'a') as f:
            f.write(prompt)

    return random_seed


def compute_dkl_uniform(count, number_of_possibilities):
    normalized = count / count.sum()
    return np.sum(normalized * np.log(number_of_possibilities * normalized))


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
            else:
                interesting_result[key] = value
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

    This function takes as input a tuple of results from mulitple experiments
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




