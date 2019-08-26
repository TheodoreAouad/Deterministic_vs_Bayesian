import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.uncertainty_measures import get_predictions_from_multiple_tests, get_all_uncertainty_measures, \
    get_all_uncertainty_measures_not_bayesian


# TODO: refactor this function into modular functions
def compute_figures(
        arguments,
        all_outputs_seen,
        true_labels_seen,
        all_outputs_unseen,
        nb_of_batches,
        size_of_batch,
        type_of_unseen,
        show_fig=True,
        save_fig=True,
        save_path=None,
):
    """
    Show scatter plots of accuracy against the uncertainty, two for each uncertainty measure.
    Each point is a batch of size_of_batch images. The difference lays in the proportion of seen / unseen image in the
    batch.
    Args:
        arguments (dict): the arguments of the python executed line in the terminal
        all_outputs_seen (torch.Tensor): size (nb_of_tests, size_of_testset_seen, nb_of_classes): the output of the
                                         softmax of the seen test set
        true_labels_seen (torch.Tensor): size (size_of_testset): the true labels of the seen test set
        all_outputs_unseen (torch.Tensor): size (nb_of_tests, size_of_testset_unseen): the output of the softmax of the
                                           unseen dataset
        nb_of_batches (int): number of points to show
        size_of_batch (int): number of images per point
        type_of_unseen (str): Type of experiment. Either 'random', 'unseen_classes' or 'unseen_dataset'
        show_fig (bool): whether we show the figure
        save_fig (bool): whether we save the figure
        save_path (str): where to save the figures

    """
    arguments['type_of_unseen'] = type_of_unseen
    nb_of_seen = true_labels_seen.size(0)
    nb_of_unseen = all_outputs_unseen.size(1)
    # Get the labels of the evaluation on MNIST, and put -1 for the labels of Random
    all_outputs = torch.cat((all_outputs_seen, all_outputs_unseen), 1)
    total_nb_of_data = nb_of_seen + nb_of_unseen
    predicted_labels = get_predictions_from_multiple_tests(all_outputs).float()
    true_labels = torch.cat((true_labels_seen, -1 + torch.zeros(nb_of_unseen)))
    correct_labels = (predicted_labels == true_labels)

    # Shuffle data between rand and mnist
    all_outputs_shuffled = torch.Tensor()
    true_labels_shuffled = torch.Tensor()
    print(f'Shuffling MNIST and {type_of_unseen} ...')
    for i in tqdm(range(nb_of_batches)):
        idx_m = torch.randperm(nb_of_seen)
        idx_r = torch.randperm(nb_of_unseen)
        k = int(i / nb_of_batches * size_of_batch)
        batch_img = torch.cat((all_outputs_seen[:, idx_m[:size_of_batch - k], :], all_outputs_unseen[:, idx_r[:k], :]),
                              1)
        all_outputs_shuffled = torch.cat((all_outputs_shuffled, batch_img), 1)
        batch_label = torch.cat((true_labels_seen[idx_m[:size_of_batch - k]], -1 + torch.zeros(k)))
        true_labels_shuffled = torch.cat((true_labels_shuffled, batch_label))
    print('Shuffling over.')
    # Shuffled data, get predictions
    predicted_labels_shuffled = get_predictions_from_multiple_tests(all_outputs_shuffled).float()
    correct_labels_shuffled = (predicted_labels_shuffled == true_labels_shuffled)
    # Group by batch, compute accuracy
    correct_labels_shuffled = correct_labels_shuffled.reshape(size_of_batch, nb_of_batches).float()
    accuracies_shuffled = correct_labels_shuffled.mean(0)
    real_size_of_batch = get_exact_batch_size(size_of_batch, total_nb_of_data)
    correct_labels = correct_labels.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).float()
    accuracies = correct_labels.mean(1)
    # Compute proportion of random samples for each batch
    random_idxs = (true_labels_shuffled == -1).float()
    random_idxs_reshaped = random_idxs.reshape(size_of_batch, nb_of_batches)
    prop_of_unseen = random_idxs_reshaped.mean(0)
    labels_not_shuffled = (true_labels == -1).float().reshape(total_nb_of_data // real_size_of_batch,
                                                              real_size_of_batch).mean(1)

    # get uncertainties
    if 'rho' in arguments.keys():
        vr_shuffled, pe_shuffled, mi_shuffled = get_all_uncertainty_measures(all_outputs_shuffled)
        vr, pe, mi = get_all_uncertainty_measures(all_outputs)
        vr_regrouped_shuffled = vr_shuffled.reshape(size_of_batch, nb_of_batches).mean(0)
        pe_regrouped_shuffled = pe_shuffled.reshape(size_of_batch, nb_of_batches).mean(0)
        mi_regrouped_shuffled = mi_shuffled.reshape(size_of_batch, nb_of_batches).mean(0)
        vr_regrouped = vr.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).mean(1)
        pe_regrouped = pe.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).mean(1)
        mi_regrouped = mi.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).mean(1)

        if show_fig or save_fig:
            # plot graphs
            try:
                arguments['nb_of_tests'] = arguments.pop('number_of_tests')
            except KeyError:
                pass
            print('Plotting figures...')
            plt.figure(figsize=(8, 10))
            plt.suptitle(arguments, wrap=True)
            plt.subplot(321)
            plt.scatter(vr_regrouped, accuracies, c=labels_not_shuffled)
            plt.ylabel('accuracy')
            plt.title('VR - not shuffled')
            plt.subplot(322)
            plt.scatter(vr_regrouped_shuffled, accuracies_shuffled, c=prop_of_unseen)
            plt.ylabel('accuracy')
            plt.title('VR - shuffled')
            cbar = plt.colorbar()
            cbar.set_label(f'{type_of_unseen} ratio', rotation=270)
            plt.subplot(323)
            plt.scatter(pe_regrouped, accuracies, c=labels_not_shuffled)
            plt.ylabel('accuracy')
            plt.title('PE - not shuffled')
            plt.subplot(324)
            plt.scatter(pe_regrouped_shuffled, accuracies_shuffled, c=prop_of_unseen)
            plt.ylabel('accuracy')
            plt.title('PE - shuffled')
            cbar = plt.colorbar()
            cbar.set_label(f'{type_of_unseen} ratio', rotation=270)
            plt.subplot(325)
            plt.title('MI - not shuffled')
            plt.scatter(mi_regrouped, accuracies, c=labels_not_shuffled)
            plt.ylabel('accuracy')
            plt.subplot(326)
            plt.title('MI - shuffled')
            plt.scatter(mi_regrouped_shuffled, accuracies_shuffled, c=prop_of_unseen)
            plt.ylabel('accuracy')
            cbar = plt.colorbar()
            cbar.set_label(f'{type_of_unseen} ratio', rotation=270)
            print('Figures plotted.')

    else:
        unc_soft_shuffled, pe_shuffled = get_all_uncertainty_measures_not_bayesian(all_outputs_shuffled)
        unc_soft, pe = get_all_uncertainty_measures_not_bayesian(all_outputs)
        unc_soft_regrouped_shuffled = unc_soft_shuffled.reshape(size_of_batch, nb_of_batches).mean(0)
        pe_regrouped_shuffled = pe_shuffled.reshape(size_of_batch, nb_of_batches).mean(0)
        unc_soft_regrouped = unc_soft.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).mean(1)
        pe_regrouped = pe.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).mean(1)

        # plot graphs
        try:
            arguments['nb_of_tests'] = arguments.pop('number_of_tests')
        except KeyError:
            pass
        plt.figure(figsize=(8, 10))
        plt.suptitle(arguments, wrap=True)
        plt.subplot(221)
        plt.scatter(unc_soft_regrouped, accuracies, c=labels_not_shuffled)
        plt.ylabel('accuracy')
        plt.title('Uncertainty Softmax - not shuffled')
        plt.subplot(222)
        plt.scatter(unc_soft_regrouped_shuffled, accuracies_shuffled, c=prop_of_unseen)
        plt.ylabel('accuracy')
        plt.title('Uncertainty Softmax - shuffled')
        cbar = plt.colorbar()
        cbar.set_label('random ratio', rotation=270)
        plt.subplot(223)
        plt.scatter(pe_regrouped, accuracies, c=labels_not_shuffled)
        plt.ylabel('accuracy')
        plt.title('PE - not shuffled')
        plt.subplot(224)
        plt.scatter(pe_regrouped_shuffled, accuracies_shuffled, c=prop_of_unseen)
        plt.ylabel('accuracy')
        plt.title('PE - shuffled')
        cbar = plt.colorbar()
        cbar.set_label('random ratio', rotation=270)
    if save_fig:
        assert save_path is not None, 'Specify a file to save the figure'
        print('Saving figure...')
        plt.savefig(save_path)
        print('Figure saved.')
    if show_fig:
        print('Showing figures...')
        plt.show()
        print('Figure shown.')
        # pd.plotting.scatter_matrix(
        # #     pd.DataFrame({
        # #         'vr': np.array(vr_regrouped),
        # #         'pe': np.array(pe_regrouped),
        # #         'mi': np.array(mi_regrouped),
        # #     }),
        # # )


def get_divisors(n):
    """
    Get the divisors of an integer.
    Args:
        n (int): number we want to get the divisors of

    Returns:
        list: list of divisors of n
    """
    res = []
    for k in range(1, int(np.sqrt(n))):
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
