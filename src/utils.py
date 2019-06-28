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


def get_bayesian_parameters(model):
    pass

