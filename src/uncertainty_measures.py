import numpy as np
import torch


def compute_dkl_uniform(count, number_of_possibilities):
    normalized = count / count.sum()
    return np.sum(normalized * np.log(number_of_possibilities * normalized))


def aggregate_data(data):
    '''

    Args:
        data (torch.Tensor): size (number_of_tests, batch_size, number_of_classes)

    Returns:
        torch.Tensor: size (batch_size). Tensor of predictions for each element of the batch
        torch.Tensor: size (batch_size). Tensor of uncertainty for each element of the batch

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


def compute_variation_ratio(data):
    """
    Computes the variation ratio for each sample. It computes the frequency of the most predicted label,
    then returns 1-this frquency.
    Args:
        data (torch.Tensor): size (number_of_tests, batch_size, number_of_classes). The output of the test on a batch.

    Returns:
        torch.Tensor: size (batch_size). The variation-ratio uncertainty measure for each sample.

    """
    batch_size = data.size(1)
    variation_ratios = torch.Tensor(batch_size).detach()
    predicted_labels = torch.transpose(data.argmax(2), 0, 1)
    for img_idx, img_labels in enumerate(predicted_labels):
        labels, counts = np.unique(img_labels, return_counts=True)
        highest_label_freq = counts.max() / counts.sum()
        variation_ratios[img_idx] = 1 - highest_label_freq

    return variation_ratios


def compute_predictive_entropy(data):
    """
    Computes the predictive entropy for each sample. It averages across all the tests,
    then computes the entropy for each sample.
    Args:
        data (torch.Tensor): size (number_of_tests, batch_size, number_of_classes). The output of the test on a batch.

    Returns:
        torch.Tensor: size (batch_size). The predictive entropy measure for each sample.
    """

    mean_of_distributions = data.mean(0).detach()
    predictive_entropies = torch.sum(-mean_of_distributions * torch.log(mean_of_distributions), 1)

    return predictive_entropies


def compute_mutual_information_uncertainty(data):
    """
    Computes the uncertainty linked to the mutual information. It computes the mutual information
    between the label prediction and the posterior on the weights.
    Args:
        data (torch.Tensor): size (number_of_tests, batch_size, number_of_classes). The output of the test on a batch.

    Returns:
        torch.Tensor: size (batch_size). The mutual information of label distribution and posterior weights for each
                                         sample.

    """
    number_of_tests = data.size(0)
    predictive_entropies = compute_predictive_entropy(data)
    x = data * torch.log(data)
    mutual_information_uncertainties = predictive_entropies + 1 / number_of_tests * x.sum(2).sum(0)

    return mutual_information_uncertainties
