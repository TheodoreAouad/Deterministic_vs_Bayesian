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


def predictive_entropy(data):
    """

    Args:
        data (torch.Tensor): size (number_of_tests, batch_size, number_of_classes). The output of the test on a batch.

    Returns:

    """

    pass
