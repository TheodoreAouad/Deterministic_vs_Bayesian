import csv
import os
import pickle

import torch
import numpy as np
import pandas as pd


def set_and_print_random_seed(random_seed=None, show=False, save=False, checkpoint_dir='./'):
    """
    Set and print numpy random seed, for reproducibility of the training,
    and set torch seed based on numpy random seed
    Args:
        random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
        save (bool): if True, the numpy random seed is saved in seeds.txt
        checkpoint_dir (str): output folder where the seed is saved
    Returns:
        int: numpy random seed

    """
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


def vectorize(tensor):
    """

    Args:
        tensor (torch.Tensor): the tensor we want to vectorize

    Returns:
        torch.Tensor: has one dimension
    """
    return tensor.view(tensor.nelement())


def open_experiment_results(type, exp_nb,  group_nb=None, polyaxon_path="polyaxon_results", filename = "results.pt"):
    """

    Args:
        type (str): groups or experiment
        exp_nb (str): the number of the experiment
        group_nb (str || None): the number of the group (if type is groups)
        polyaxon_path (str): path of the parent folder of the results
        filename (str): name of the file we want to open

    Returns:
        content of the file. Most of the time it is a dict.

    """
    if type == "groups":
        path_to_results = os.path.join(polyaxon_path, type, group_nb, exp_nb, filename)
    else:
        path_to_results = os.path.join(polyaxon_path, type, exp_nb, filename)
    return torch.load(path_to_results)


def get_interesting_result(result):
    """
    Reads the results of a polyaxon experiment and extracts in a dict the desired information.
    Args:
        result (dict || pandas.core.frame.DataFrame): output of the polyaxon experiment

    Returns:
        dict: dictionary of the desired parameters we would like to write in a csv

    """
    if type(result) == dict:
        interesting_result = dict()
        for key, value in result.items():
            try:
                if key == "val accuracy":
                    interesting_result[key + " max"] = np.array(value).max()
                else:
                    if type(value) == list:
                        interesting_result[key] = torch.tensor(value[-1][-1]).float().mean().item()
                    elif type(value) == torch.Tensor and len(value.size()) > 1:
                        interesting_result[key + "-mean"] = value.mean().item()
                        interesting_result[key + "-std"] = value.std().item()
                    else:
                        interesting_result[key] = value
            except Exception as e:
                print(str(key) + " recuperation not implemented, or unexpected error.")
                print(key, value)
                raise e

        return interesting_result

    interesting_result = result.copy()
    interesting_result['val accuracy'] = interesting_result['val accuracy'].apply(
        lambda x: torch.tensor(x).max().item()
    )
    interesting_result = interesting_result.rename(columns={'val accuracy': 'val accuracy max'})
    uncertainty_keys = [key for key in result.keys() if 'uncertainty' in key]
    for key in uncertainty_keys:
        if type(result[key].iloc[3]) == str:
            print(result[key], key)
        interesting_result[key + "-mean"] = result[key].apply(lambda x: x.mean().item())
        interesting_result[key + "-std"] = result[key].apply(lambda x: x.std().item())
        interesting_result = interesting_result.drop(key, 1)

    return interesting_result




def write_dict_in_csv(results, name="results/results.csv"):
    """

    This function takes as input a tuple of results from multiple experiments
    and writes a csv.

    Args:
        results (dict): results we want to write in csv
        name (dict): path and name of the file we want to write in

    Returns:
        None

    """

    file = open(name, "w")
    writer = csv.DictWriter(file, results[0].keys())
    writer.writeheader()
    for result in results:
        writer.writerow(result)
    file.close()


def get_file_and_dir_path_in_dir(dir_path, file_name=""):
    """
    Get all the paths of the files in a certain directory, with a restriction on the file_name
    Args:
        dir_path (str): directory we want to scan
        file_name (str): strign we want in the name

    Returns:
        list: list of the paths of all the files
    """
    all_files = []
    all_dirs = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for file in filenames:
            if file_name in file:
                all_files.append(os.path.join(dirpath, file))
                all_dirs.append(dirpath)
    return all_files, all_dirs


def compute_weights_norm(model):
    """
    Compute the L2 norm of the weights of the model.
    Args:
        model (torch.nn.Module): the model we want to compute the norm of the weights of

    Returns:
        float: the norm of the weights parameters of the model

    """
    norm = 0
    all_params = model.parameters()
    for param in model.parameters():
        norm += torch.norm(param.data).item()
    return norm/len(list(all_params))


def save_to_file(to_save, path):
    """
    Saves an object in a file. Warning: will overwrite the file.
    Args:
        to_save : the object we want to save.
        path (str): path to the file where to write the dict.

    """
    with open(path, "wb") as f:
        pickle.dump(to_save, f)


def load_from_file(path):
    """
    Returns the object saved in the path.
    Args:
        path (str): the path to the file where the object is saved.

    Returns:
        the object loaded.

    """
    with open(path, "rb") as f:
        my_dict = pickle.load(f)
    return my_dict


def compute_memory_used_tensor(tensor):
    """

    Args:
        tensor (torch.Tensor): tensor we want to compute the memory of

    Returns:
        Dict: different information on the memory use
    """
    return dict({
        'number of elements': tensor.nelement(),
        'size of an element': tensor.element_size(),
        'total memory use': tensor.nelement() * tensor.element_size()
    })


def print_nicely_on_console(dic):
    """
    Prints nicely a dict on console.
    Args:
        dic (dict): the dictionary we want to print nicely on console.
    """
    to_print = ''
    for key, value in zip(dic.keys(), dic.values()):
        if type(value) == torch.Tensor:
            value_to_print = value.item()
        else:
            value_to_print = value

        if value is not None:
            if 'accuracy' in key:
                value_to_print = str(round(100 * value_to_print, 2)) + ' %'
            else:
                value_to_print = "{:.2E}".format(value)
            to_print += f'{key}: {value_to_print}, '
    print(to_print)


def convert_tensor_to_float(df):
    """
    Converts torch.Tensor types of a dataframe into a float
    Args:
        df (pandas.core.frame.DataFrame): dataframe to be converted
    """
    for key in list(df.columns):
        if type(df[key].iloc[0]) == torch.Tensor:
            try: df[key] = df[key].astype(float)
            except Exception as e:
                print(key, df[key])
                raise(e)

