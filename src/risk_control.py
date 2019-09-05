"""
The first two functions are entirely copied (and slightly refactored) from this repo:
https://github.com/geifmany/selective_deep_learning, adaptation of the paper
Selective Classification for Deep Neural Networks (Geifman, El Yaniv, 2017)
"""

import numpy as np
import torch
from scipy.stats import binom
import scipy
import math
import random

from tqdm import tqdm

from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_predictions_from_multiple_tests


def calculate_bound(delta, m, erm, precision=1e-6, max_iter=1000,):
    """
    This function is a solver for the inverse of binomial CDF based on binary search. see eq. (4) of Selective
    Classification for Deep Neural Networks.
    Args:
        delta (float): the desired delta
        m (int): number of data
        erm (float): risk

    Returns:
        float: the inverse binomial CDF
    """

    def func(b):
        return (-1 * delta) + scipy.stats.binom.cdf(int(float(m) * float(erm)), float(m), float(b))

    a = erm  # start binary search from the empirical risk
    c = 1  # the upper bound is 1
    b = (a + c) / 2  # mid point
    funcval = func(b)
    # TODO: get rid of i and max_iter
    i = 0
    while abs(funcval) > precision:
        if a == 1.0 and c == 1.0:
            b = 1.0
            break
        elif funcval > 0:
            a = b
        else:
            c = b
        b = (a + c) / 2
        funcval = func(b)
        i += 1
        if i > max_iter:
            break
    return b.item() if type(b) == torch.Tensor else b


def bound_animate(rstar, delta, kappa, residuals, verbose=True, **kwargs):
    """
    See bound docstring. This is the same but we keep all thetas and bounds.
    Args:
        verbose:
        rstar (float): the requested risk bound
        delta (float): the desired delta
        kappa (array-like): size (batch_size): rating function over the points (higher values is
                            more confident prediction)
        residuals (array-like): a vector of the residuals of the samples 0 is correct prediction and 1 corresponding
                                to an error
    Returns:
        list, list: [thetas, bounds] (also prints latex text for the tables in the paper). Thetas are the threshold
                       for selection.
    """

    probs = kappa
    fy = residuals
    m = len(fy)
    probs_idx_sorted = probs.argsort()

    a = 0
    b = m - 1
    deltahat = delta / math.ceil(math.log2(m))

    thetas, bounds, risks, coverages = [], [], [], []

    iterator = range(math.ceil(math.log2(m)) + 1)
    if verbose:
        iterator = tqdm(iterator)
    for q in iterator:
        # the for runs log(m)+1 iterations but actually the bound calculated on only log(m) different candidate
        # thetas
        mid = math.ceil((a + b) / 2)

        mi = len(fy[probs_idx_sorted[mid:]])
        theta = probs[probs_idx_sorted[mid]]
        risk = sum(fy[probs_idx_sorted[mid:]]) / mi
        if type(risk) == torch.Tensor: risk = risk.item()

        bound = calculate_bound(deltahat, mi, risk, **kwargs)
        coverage = mi / m
        if bound > rstar:
            a = mid
        else:
            b = mid
        thetas.append(theta)
        bounds.append(bound)
        risks.append(risk)
        coverages.append(coverage)

    if verbose:
        print("%.5f & %.6f & %.4f & %.4f   \\\\" % (rstar, risk, coverage, bound))
    return [thetas, bounds, risks, coverages, ]


def bound(rstar, delta, kappa, residuals, verbose=True, **kwargs):
    """
    A function to calculate the risk bound proposed in the paper, the algorithm is based on algorithm 1
    from the paper. It is possible to add a validation, see repo.
    Args:
        verbose:
        rstar (float): the requested risk bound
        delta (float): the desired delta
        kappa (array-like): size (batch_size): rating function over the points (higher values is
                            more confident prediction)
        residuals (array-like): a vector of the residuals of the samples 0 is correct prediction and 1 corresponding
                                to an error
    Returns:
        float, float: [theta, bound] (also prints latex text for the tables in the paper). Theta is the threshold
                       for selection.
    """

    thetas, bounds = bound_animate(rstar, delta, kappa, residuals, verbose, **kwargs)[:2]
    return thetas[-1], bounds[-1]


def get_selection_threshold_one_unc(
        bay_net,
        trainloader,
        risk,
        delta,
        uncertainty_function,
        number_of_tests,
        verbose=False,
        device='cpu',
):
    """

    Args:
        bay_net (torch.nn.Module child):
        arguments (dict): must contain keys:
        risk (float): highest error we accept
        uncertainty_function (function): function to get uncertainty. Must be in src.uncertainty_measures

    Returns:
        float: threshold
    """

    true_labels, all_outputs_train = eval_bayesian(
        bay_net,
        trainloader,
        number_of_tests=number_of_tests,
        return_accuracy=False,
        device=device,
        verbose=verbose,
    )

    unc = uncertainty_function(all_outputs_train)
    labels_predicted = get_predictions_from_multiple_tests(all_outputs_train).int()

    correct_preds = (labels_predicted == true_labels)
    threshold, _ = bound(risk, delta, unc, correct_preds,)
    return threshold

def get_selection_threshold_all_unc(
        bay_net,
        true_labels,
        all_outputs_train,
        risk,
        delta,
        uncs,
        number_of_tests,
        verbose=False,
        device='cpu',
):
    """

    Args:
        bay_net (torch.nn.Module child):
        arguments (dict): must contain keys:
        risk (float): highest error we accept
        uncertainty_function (function): function to get uncertainty. Must be in src.uncertainty_measures

    Returns:
        float: threshold
    """


    thresholds = []
    for unc in uncs:
        labels_predicted = get_predictions_from_multiple_tests(all_outputs_train).float()

        correct_preds = (labels_predicted == true_labels)
        threshold, _ = bound(risk, delta, -unc, correct_preds,)
        thresholds.append(threshold)
    return thresholds
