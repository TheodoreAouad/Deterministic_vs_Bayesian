import numpy as np
import torch

from src.utils import get_exact_batch_size


def get_deadzones(all_outputs_seen, all_outputs_unseen, get_uncertainties_function, n):
    """
    This function gives the deadzones number n for each uncertainty measure
    Args:
        all_outputs_seen (torch.Tensor): size (nb_of_tests, batch_size, nb_of_classes): outputs of softmax layer for
                                         seen testset
        all_outputs_unseen (torch.Tensor): size (nb_of_tests, batch_size, nb_of_classes): outputs of softmax layer for
                                           unseen testset
        get_uncertainties_function (function): function that gives the uncertainties we want to compute

    Returns:

    """
    unc_seens = get_uncertainties_function(all_outputs_seen)
    unc_unseens = get_uncertainties_function(all_outputs_unseen)

    deadzones = []
    for unc_seen, unc_unseen in zip(unc_seens, unc_unseens):
        deadzones.append(get_deadzone_from_unc(unc_seen, unc_unseen, n))
    return deadzones


def get_deadzone_from_unc(unc_seen_original, unc_unseen_original, n):
    """
    This function gives the deadzone number n given the uncertainties
    Args:
        unc_seen_original (torch.Tensor): size (batch_size): uncertainty tensor for seen data
        unc_unseen_original (torch.Tensor): size (batch_size): uncertainty tensor for unseen data
        n (int): deadzone number. It is the size of the number of images we average the uncertainty on.

    Returns:
        float: the deadzone number n of this uncertainty
    """
    unc_seen = unc_seen_original.cpu()
    unc_unseen = unc_unseen_original.cpu()
    size_seen = unc_seen.size(0)
    size_unseen = unc_unseen.size(0)
    idx_exact_batch_seen = size_seen - size_seen % n
    idx_exact_batch_unseen = size_unseen - size_unseen % n

    unc_seen_regrouped = unc_seen[:idx_exact_batch_seen].reshape(idx_exact_batch_seen // n, n).mean(1)
    unc_unseen_regrouped = unc_unseen[:idx_exact_batch_unseen].reshape(idx_exact_batch_unseen // n, n).mean(1)

    if size_seen != idx_exact_batch_seen:
        unc_seen_regrouped = torch.cat(
            (unc_seen_regrouped, torch.tensor([unc_seen[idx_exact_batch_seen:].mean()],)))
    if size_unseen != idx_exact_batch_unseen:
        unc_unseen_regrouped = torch.cat(
            (unc_unseen_regrouped, torch.tensor([unc_unseen[idx_exact_batch_unseen:].mean()],)))

    max_seen = unc_seen_regrouped.max()
    min_unseen = unc_unseen_regrouped.min()
    res = ((min_unseen - max_seen) / np.abs(max_seen - unc_seen_regrouped.min()))
    if type(res) == torch.Tensor:
        return res.item()
    else:
        return res
