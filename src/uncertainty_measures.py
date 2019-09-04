import numpy as np
import torch


def compute_dkl_uniform(count, number_of_possibilities):
    """
    This function computes the Kullback-Leibler divergence between a discrete distribution of probability
    and the uniform distribution.
    Args:
        count (array): size (number_of_labels) . Number of times each labels were predicted
        number_of_possibilities (int): number of classes

    Returns:
        float: kullback-leibler divergence
    """
    normalized = count / count.sum()
    return np.sum(normalized * np.log(number_of_possibilities * normalized))


def get_predictions_from_multiple_tests(data):
    """

    Args:
        data (torch.Tensor): size (number_of_tests, batch_size, number_of_classes)

    Returns:
        torch.Tensor: size (batch_size). The labels for each sample.
    """
    mean = data.mean(0)
    return mean.argmax(1)


def aggregate_data(data):
    """

    Args:
        data (torch.Tensor): size (number_of_tests, batch_size, number_of_classes)

    Returns:
        torch.Tensor: size (batch_size). Tensor of predictions for each element of the batch
        torch.Tensor: size (batch_size). Tensor of uncertainty for each element of the batch

    """

    predicted = get_predictions_from_multiple_tests(data)

    std = data.std(0)
    uncertainty = std.mean(1)

    dkls = np.zeros(data.size(1))
    all_predicts = data.argmax(2).cpu().numpy().T
    for test_sample_idx, test_sample in enumerate(all_predicts):
        values, count = np.unique(test_sample, return_counts=True)
        dkls[test_sample_idx] = compute_dkl_uniform(count, data.size(2))

    return predicted, uncertainty, torch.tensor(dkls).float().to(data.device)


def uncertainty_softmax(data):
    """
    Computes the variation ratio for each sample. It computes the frequency of the most predicted label,
    then returns 1-this frquency.
    Args:
        data (torch.Tensor): size (1, batch_size, number_of_classes). The output of the test on a batch.

    Returns:
        torch.Tensor: size (batch_size). The variation-ratio uncertainty measure for each sample.

    """
    data = data.squeeze()
    return 1 - data.max(1).values



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
    predicted_labels = torch.transpose(data.argmax(2), 0, 1).to('cpu')
    for img_idx, img_labels in enumerate(predicted_labels):
        labels, counts = np.unique(img_labels, return_counts=True)
        highest_label_freq = counts.max() / counts.sum()
        variation_ratios[img_idx] = 1 - highest_label_freq

    return variation_ratios.to(data.device).float()


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
    x = -mean_of_distributions * torch.log(mean_of_distributions)
    # put NaN values to 0
    x[x != x] = 0
    predictive_entropies = torch.sum(x, 1)

    return predictive_entropies.float()


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
    # put NaN values to 0
    x[x != x] = 0
    mutual_information_uncertainties = predictive_entropies + 1 / number_of_tests * x.sum(2).sum(0)

    return mutual_information_uncertainties.float()


def get_all_uncertainty_measures(data):
    """
    Gets all the uncertainty measures in this order: Variation-Ratio, Predictive entropy, Mutual information
    Args:
        data (torch.Tensor): size (number_of_tests, batch_size, number_of_classes). The output of the test on a batch.

    Returns:
        Tuple (torch.Tensor, torch.Tensor, torch.Tensor: all tensors are of size batch_size (=data.size(1))

    """
    return compute_variation_ratio(data), compute_predictive_entropy(data), compute_mutual_information_uncertainty(data)


def get_all_uncertainty_measures_not_bayesian(data):
    """
        Gets all the uncertainty measures in this order: Variation-Ratio, Predictive entropy, Mutual information
        Args:
            data (torch.Tensor): size (1, batch_size, number_of_classes). The output of the test on a batch.

        Returns:
            Tuple (torch.Tensor, torch.Tensor, torch.Tensor: all tensors are of size batch_size (=data.size(1))

        """
    return uncertainty_softmax(data), compute_predictive_entropy(data)
