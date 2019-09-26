import csv
import os
import pickle

import seaborn as sns
import torch
import numpy as np
from matplotlib import pyplot as plt


def set_and_print_random_seed(random_seed=None, show=False, save=False, checkpoint_dir='./'):
    """
    Set and print numpy random seed, for reproducibility of the training,
    and set torch seed based on numpy random seed
    Args:
        random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
        show (bool):
        save (bool): if True, the numpy random seed is saved in seeds.txt
        checkpoint_dir (str): output folder where the seed is saved
    Returns:
        int: numpy random seed

    """
    if random_seed is None:
        random_seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)
    torch.manual_seed(np.random.randint(0, 2 ** 32 - 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    prompt = 'Random seed : {}\n'.format(random_seed)

    if show:
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


def open_experiment_results(type_exp, exp_nb, group_nb=None, polyaxon_path="polyaxon_results", filename="results.pt"):
    """

    Args:
        type_exp (str): groups or experiment
        exp_nb (str): the number of the experiment
        group_nb (str || None): the number of the group (if type_exp is groups)
        polyaxon_path (str): path of the parent folder of the results
        filename (str): name of the file we want to open

    Returns:
        content of the file. Most of the time it is a dict.

    """
    if type_exp == "groups":
        path_to_results = os.path.join(polyaxon_path, type_exp, group_nb, exp_nb, filename)
    else:
        path_to_results = os.path.join(polyaxon_path, type_exp, exp_nb, filename)
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

    try:
        interesting_result = result.copy()
        if 'val accuracy' in result.keys():
            interesting_result['val accuracy'] = interesting_result['val accuracy'].apply(
                lambda x: torch.tensor(x).max().item()
            )
            interesting_result = interesting_result.rename(columns={'val accuracy': 'val accuracy max'})
        uncertainty_keys = [key for key in result.keys() if 'uncertainty' in key]
        for key in uncertainty_keys:
            if type(result[key].iloc[0]) == str:
                print(result[key], key)
            interesting_result[key + "-mean"] = result[key].apply(lambda x: x.mean().item())
            interesting_result[key + "-std"] = result[key].apply(lambda x: x.std().item())
            interesting_result = interesting_result.drop(key, 1)
    except KeyError as e:
        print(result['experiment'])
        raise(e)

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
            if file_name in os.path.join(dirpath, file):
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
    return norm / len(list(all_params))


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
            try:
                # df[key] = df[key].astype(float)
                df[key] = df[key].apply(lambda x: x.detach().numpy() if type(x) == torch.Tensor else x)
            except Exception as e:
                print(key, df[key])
                raise e


def convert_df_to_cpu(df):
    """
    Put dataframe in cpu.
    Args:
        df (pandas.core.frame.DataFrame): dataframe to be converted
    """
    for key in list(df.columns):
        if type(df[key].iloc[0]) == torch.Tensor:
            df[key] = df[key].apply(lambda x: x.to('cpu'))


def split_multiple_sep(s, separators):
    """
    Splits a string according to multiple separators.
    Args:
        s (str): string to split
        separators (list): list of strings, each string is a separator.

    Returns:
        list: list of string. separated list
    """
    res = [s]
    for sep in separators:
        temp = []
        for substr in res:
            temp += substr.split(sep)
        res = temp.copy()
    return res

def get_unc_key(keys, approximate_key):
    """
    This function gives the right key for the given approximated key. This function is used because of the
    different names of the same uncertainties, e.g. predictive entropy/pe/pes.
    Args:
        keys (list): the keys to search the right key in
        approximate_key (str): the approximated key.

    Returns:
        str: the right key
    """
    is_uncertainty = False

    vrs_possible_writings = {'vr', 'vrs', 'variation-ratio', 'variation-ratios', 'variation ratio',
                             'variation-ratios'}
    pes_possible_writings = {'pe', 'pes', 'predictive_entropies', 'predictive_entropy', 'predictive entropy',
                             'predictive entropies'}
    mis_possible_writings = {'mi', 'mis', 'mutual information'}
    us_possible_writings = {'us', 'uncertainty softmax', 'uncertainty_softmax', 'uncertainty-softmax'}
    nb_of_data_possible_writings = {'nb_of_data', 'split_train', 'nb of data'}

    if 'unseen' in approximate_key or 'random' in approximate_key:
        seen_or_unseen = ['unseen', 'random']
        is_uncertainty = True
    elif 'seen' in approximate_key:
        seen_or_unseen = ['seen']
        is_uncertainty = True
    elif 'nb_of_data' in approximate_key:
        this_is_the_keys = nb_of_data_possible_writings
    else:
        assert False, 'approximate_key not understood.'

    if is_uncertainty:
        found_the_uncertainty = False
        all_possible_writings = [vrs_possible_writings, pes_possible_writings, mis_possible_writings,
                                 us_possible_writings]
        for possible_writings in all_possible_writings:
            for possible_writing in possible_writings:
                if possible_writing in approximate_key:
                    this_is_the_keys = possible_writings
                    found_the_uncertainty = True
        if not found_the_uncertainty:
            assert False, 'Uncertainty not valid'

    all_keys = []
    for key in keys:
        is_correct_unc = sum([this_writing in key for this_writing in this_is_the_keys])
        is_correct_seen_or_unseen = sum([this_seen in split_multiple_sep(key, [' ', '-', '_'])
                                         for this_seen in seen_or_unseen]) if is_uncertainty else True
        if is_correct_seen_or_unseen and is_correct_unc:
            all_keys.append(key)
    if len(all_keys) == 1:
        return all_keys[0]
    return all_keys


def get_divisors(n):
    """
    Get the divisors of an integer.
    Args:
        n (int): number we want to get the divisors of

    Returns:
        list: list of divisors of n
    """
    res = []
    for k in range(1, int(n//2)):
        if n % k == 0:
            res.append(k)
    return res


def get_exact_batch_size(size_of_batch, total_nb_sample):
    """
    Does the computation of the exact size of batch (cf func compute_figures) depending on an approximate size
    Args:
        size_of_batch (int): the size of batch we would like optimally
        total_nb_sample (int): the number of images we want to divide into batches

    Returns:
        int: the size of batch we can divide the number of samples into
    """
    divisors = get_divisors(total_nb_sample)
    return min(divisors, key=lambda x: abs(x - size_of_batch))


def plot_density_on_ax(ax, uncs, labels, hist=False, **kwargs):
    """
    Plots density uncertainty on the given ax using kde smoothing.
    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): ax on which to plot the density
        uncs (tuple): tuple of arrays of the uncertainties. The number of elements is the number of uncertainties.
        labels (tuple): tuple of string. Each element is the label of the corresponding element of 'uncs'.
        **kwargs: kwargs to change the distplot.

    """
    for unc, label in zip(uncs, labels):
        sns.distplot(unc, hist=hist, kde_kws={"shade": True}, label=label, ax=ax, **kwargs)
        ax.set_xlim(left=0)


def get_fig_size(ax):
    return ax.figure.get_size_inches()*ax.figure.dpi


def aggregate_df(df, indexs):
    """
    Gives Mean and STD of df grouped by indexes
    Args:
        df (pandas.core.frame.DataFrame):
        indexs (list): list of string of column names on which to group by

    Returns:
        pandas.core.frame.DataFrame: aggregated dataframe

    """
    grped = df.groupby(indexs)

    def stringify(x):
        return str(round(x, 4))

    means = grped.agg('mean')
    stds = grped.agg(lambda df: 1.96 * df.std() / len(df))
    df_aggregated_show = means.applymap(stringify) + '+-' + stds.applymap(stringify)
    df_aggregated_data = grped.agg(lambda df: [df.mean(), 1.96 * df.std() / len(df), len(df)])
    return df_aggregated_show.reset_index(), df_aggregated_data.reset_index()
