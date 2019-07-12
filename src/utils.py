import csv

import math
import pickle

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


def aggregate_data(data):
    '''

    Args:
        data (torch.Tensor): size (number_of_tests, batch_size, number_of_classes)

    Returns:
        predicted (torch.Tensor): size (batch_size). Tensor of predictions for each element of the batch
        uncertainty (torch.Tensor): size (batch_size). Tensor of uncertainty for each element of the batch

    '''

    mean = data.mean(0)
    predicted = mean.argmax(1)

    std = data.std(0)
    uncertainty = std.mean(1)

    dkls = np.zeros(data.size(1))
    all_predicts = data.argmax(2).cpu().numpy().T
    for test_sample_idx, test_sample in enumerate(all_predicts):
        values, count = np.unique(test_sample, return_counts=True)
        dkls[test_sample_idx] = compute_dkl_uniform(count, data.size(2))

    return predicted, uncertainty, torch.tensor(dkls).float().to(data.device)


def vectorize(tensor):
    '''

    Args:
        tensor (torch.Tensor): the tensor we want to vectorize

    Returns:
        vector (torch.Tensor): has one dimension
    '''
    return tensor.view(tensor.nelement())


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


def get_interesting_result(result):
    '''
    Reads the results of a polyaxon experiment and extracts in a dict the desired information.
    Args:
        result (dict): output of the polyaxon experiment
        number_of_classes (int): number of classes of the polyaxon experiment (ex: MNIST, 10)

    Returns:
        dict: dictionary of the desired parameters we would like to write in a csv

    '''
    interesting_result = dict()
    for key, value in result.items():
        try:
            if key == "val accuracy":
                interesting_result[key + " max"] = np.array(value).max()
            else:
                if type(value) == list:
                    interesting_result[key] = value[-1][-1]
                elif "uncertainty" in key or "dkls" in key:
                    interesting_result[key + "-mean"] = value.mean().item()
                    interesting_result[key + "-std"] = value.std().item()
                else:
                    interesting_result[key] = value
        except Exception as e:
            print(str(key) + " recuperation not implemented, or unexpected error.")
            print(key, value)
            raise e

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

    file = open(name, "w")
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
        list: list of the paths of all the files
    '''
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for file in filenames:
            if file_name in file:
                all_files.append(os.path.join(dirpath, file))
    return all_files


def compute_weights_norm(model):
    norm = 0
    all_params = model.parameters()
    for param in model.parameters():
        norm += torch.norm(param.data).item()
    return norm/len(list(all_params))


def save_dict(to_save, path):
    """
    Saves a dictionary in a file. Warning: will overwrite the file.
    Args:
        to_save (dict): the dict we want to save.
        path (str): path to the file where to write the dict.

    """
    with open(path, "wb") as f:
        pickle.dump(to_save, f)


def load_dict(path):
    """
    Returns the dict saved in the path.
    Args:
        path (str): the path to the file where the dict is saved.

    Returns:
        dict: the dictionary loaded.

    """
    with open(path, "rb") as f:
        my_dict = pickle.load(f)
    return my_dict
