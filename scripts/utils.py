import pathlib

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms as transforms
from tqdm import tqdm

from src.dataset_manager.get_data import get_mnist, get_omniglot, get_cifar10, get_random
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.evals import eval_bayesian, eval_random

from src.uncertainty_measures import get_predictions_from_multiple_tests, get_all_uncertainty_measures, \
    get_all_uncertainty_measures_not_bayesian
from src.utils import get_exact_batch_size, get_file_and_dir_path_in_dir, load_from_file, plot_density_on_ax


# TODO: refa hist=True,ctor this function into modular functions
def compute_figures(
        arguments,
        all_outputs_seen,
        true_labels_seen,
        all_outputs_unseen,
        nb_of_batches,
        size_of_batch,
        scale='linear',
        show_fig=True,
        save_fig=True,
        save_path=None,
        figsize=(8, 10),
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
        scale (str): scale of the plot. Strongly advice 'linear'.
        show_fig (bool): whether we show the figure
        save_fig (bool): whether we save the figure
        save_path (str): where to save the figures

    """
    type_of_unseen = arguments['type_of_unseen']
    arguments['imgs per point'] = size_of_batch
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
        vr_regrouped_shuffled, pe_regrouped_shuffled, mi_regrouped_shuffled = reshape_shuffled(
            (vr_shuffled, pe_shuffled, mi_shuffled), size_of_batch, nb_of_batches,
        )
        vr_regrouped, pe_regrouped, mi_regrouped = reshape_for_not_shuffled(
            (vr, pe, mi), total_nb_of_data, real_size_of_batch,
        )

        # plot graphs
        try:
            arguments['nb_of_tests'] = arguments.pop('number_of_tests')
        except KeyError:
            pass
        print('Plotting figures...')
        plt.figure(figsize=figsize)
        plt.suptitle(arguments, wrap=True)
        plt.subplot(321)
        plt.yscale(scale)
        plt.scatter(vr_regrouped, accuracies, c=labels_not_shuffled)
        plt.ylabel('accuracy')
        max_seen, min_unseen, deadzone = get_deadzone(vr_regrouped, labels_not_shuffled)
        plt.title(f'VR - not shuffled, deadzone{real_size_of_batch}: {round(deadzone, 2)}')
        if max_seen < min_unseen:
            plt.axvspan(max_seen, min_unseen, color='green', alpha=0.5)
        else:
            plt.axvspan(min_unseen, max_seen, color='red', alpha=0.5)
        plt.subplot(322)
        plt.scatter(vr_regrouped_shuffled, accuracies_shuffled, c=prop_of_unseen)
        plt.ylabel('accuracy')
        plt.title(f'VR - shuffled ({size_of_batch} imgs/pt), '
                  f'corr: {round(correlation(vr_regrouped_shuffled, accuracies_shuffled), 2)}')
        cbar = plt.colorbar()
        cbar.set_label(f'{type_of_unseen} ratio', rotation=270)
        plt.subplot(323)
        plt.yscale(scale)
        plt.scatter(pe_regrouped, accuracies, c=labels_not_shuffled)
        plt.ylabel('accuracy')
        max_seen, min_unseen, deadzone = get_deadzone(pe_regrouped, labels_not_shuffled)
        plt.title(f'PE - not shuffled, deadzone{real_size_of_batch}: {round(deadzone, 2)}')
        if max_seen < min_unseen:
            plt.axvspan(max_seen, min_unseen, color='green', alpha=0.5)
        else:
            plt.axvspan(min_unseen, max_seen, color='red', alpha=0.5)
        plt.subplot(324)
        plt.scatter(pe_regrouped_shuffled, accuracies_shuffled, c=prop_of_unseen)
        plt.ylabel('accuracy')
        plt.title(f'PE - shuffled ({size_of_batch} imgs/pt), '
                  f'corr: {round(correlation(pe_regrouped_shuffled, accuracies_shuffled), 2)}')
        cbar = plt.colorbar()
        cbar.set_label(f'{type_of_unseen} ratio', rotation=270)
        plt.subplot(325)
        max_seen, min_unseen, deadzone = get_deadzone(mi_regrouped, labels_not_shuffled)
        plt.title(f'MI - not shuffled, deadzone{real_size_of_batch}: {round(deadzone, 2)}')
        if max_seen < min_unseen:
            plt.axvspan(max_seen, min_unseen, color='green', alpha=0.5)
        else:
            plt.axvspan(min_unseen, max_seen, color='red', alpha=0.5)
        plt.yscale(scale)
        plt.scatter(mi_regrouped, accuracies, c=labels_not_shuffled)
        plt.ylabel('accuracy')
        plt.subplot(326)
        plt.title(f'MI - shuffled ({size_of_batch} imgs/pt), '
                  f'corr: {round(correlation(mi_regrouped_shuffled, accuracies_shuffled), 2)}')
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
        plt.figure(figsize=figsize)
        plt.suptitle(arguments, wrap=True)
        plt.subplot(221)
        plt.yscale(scale)
        plt.scatter(unc_soft_regrouped, accuracies, c=labels_not_shuffled)
        plt.ylabel('accuracy')
        max_seen, min_unseen, deadzone = get_deadzone(unc_soft_regrouped, labels_not_shuffled)
        plt.title(f'US - not shuffled, deadzone{real_size_of_batch}: {round(deadzone, 2)}')
        if max_seen < min_unseen:
            plt.axvspan(max_seen, min_unseen, color='green', alpha=0.5)
        else:
            plt.axvspan(min_unseen, max_seen, color='red', alpha=0.5)
        plt.subplot(222)
        plt.scatter(unc_soft_regrouped_shuffled, accuracies_shuffled, c=prop_of_unseen)
        plt.ylabel('accuracy')
        plt.title(
            f'US - shuffled ({size_of_batch} imgs/pt), corr: '
            f'{round(correlation(unc_soft_regrouped_shuffled, accuracies_shuffled), 2)}')
        cbar = plt.colorbar()
        cbar.set_label('random ratio', rotation=270)
        plt.subplot(223)
        plt.yscale(scale)
        plt.scatter(pe_regrouped, accuracies, c=labels_not_shuffled)
        plt.ylabel('accuracy')
        max_seen, min_unseen, deadzone = get_deadzone(pe_regrouped, labels_not_shuffled)
        plt.title(f'PE - not shuffled, deadzone{real_size_of_batch}: {round(deadzone, 2)}')
        if max_seen < min_unseen:
            plt.axvspan(max_seen, min_unseen, color='green', alpha=0.5)
        else:
            plt.axvspan(min_unseen, max_seen, color='red', alpha=0.5)
        plt.subplot(224)
        plt.scatter(pe_regrouped_shuffled, accuracies_shuffled, c=prop_of_unseen)
        plt.ylabel('accuracy')
        plt.title(f'PE - shuffled ({size_of_batch} imgs/pt), '
                  f'corr: {round(correlation(pe_regrouped_shuffled, accuracies_shuffled), 2)}')
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


def correlation(a, b):
    x = a - a.mean()
    y = b - b.mean()

    return ((x * y).sum() / (np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum()))).item()


def get_deadzone(unc, is_false_label):
    unseen_uncs = unc[is_false_label == 1]
    seen_uncs = unc[is_false_label == 0]
    max_seen = seen_uncs.max()
    min_unseen = unseen_uncs.min()
    res = ((min_unseen - max_seen) / np.abs(max_seen - seen_uncs.min()))
    if type(res) == torch.Tensor:
        return max_seen, min_unseen, res.item()
    else:
        return max_seen, min_unseen, res
    #
    # nb_of_ambiguous_seen = (seen_uncs >= min_unc_unseen).sum()
    # nb_of_ambiguous_unseen = (unseen_uncs <= max_unc_seen).sum()
    #
    # return nb_of_ambiguous_seen, nb_of_ambiguous_unseen, nb_of_ambiguous_seen + nb_of_ambiguous_unseen


def reshape_shuffled(tensors_to_shuffle, size_of_batch, nb_of_batches):
    if type(tensors_to_shuffle) == torch.Tensor:
        return tensors_to_shuffle.reshape(size_of_batch, nb_of_batches).mean(0)
    tensors_reshaped = []
    for to_shuffle in tensors_to_shuffle:
        tensors_reshaped.append(to_shuffle.reshape(size_of_batch, nb_of_batches).mean(0))
    return tensors_reshaped


def reshape_for_not_shuffled(tensors_to_shuffle, total_nb_of_data, real_size_of_batch):
    if type(tensors_to_shuffle) == torch.Tensor:
        return tensors_to_shuffle.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).mean(1)
    tensors_reshaped = []
    for to_shuffle in tensors_to_shuffle:
        tensors_reshaped.append(to_shuffle.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).mean(1))
    return tensors_reshaped


# TODO: factorize the density computations
def compute_density_train_seen_unseen(arguments, all_outputs_train, all_outputs_seen, all_outputs_unseen, show_fig,
                                      save_fig, save_path=None, figsize=(8, 10), **kwargs):
    """
    Compute and show the density distribution of uncertainties.
    Args:
        arguments (dict): the arguments of the python executed line in the terminal
        all_outputs_seen (torch.Tensor): size (nb_of_tests, size_of_testset_seen, nb_of_classes): the output of the
                                         softmax of the seen test set
        all_outputs_unseen (torch.Tensor): size (nb_of_tests, size_of_testset_unseen): the output of the softmax of the
                                           unseen dataset
        show_fig:
        save_fig:
        save_path:
        **kwargs:



        scale (str): scale of the plot. Strongly advice 'linear'.
        show_fig (bool): whether we show the figure
        save_fig (bool): whether we save the figure
        save_path (str): where to save the figures
    Returns:

    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(f'Distribution of uncertainties - {arguments["type_of_unseen"]} - train seen unseen \n{arguments}',
                 wrap=True)
    if 'rho' in arguments.keys():

        vr_train, pe_train, mi_train = get_all_uncertainty_measures(all_outputs_train)
        vr_seen, pe_seen, mi_seen = get_all_uncertainty_measures(all_outputs_seen)
        vr_unseen, pe_unseen, mi_unseen = get_all_uncertainty_measures(all_outputs_unseen)

        uncs = {
            'VR': (vr_train, vr_seen, vr_unseen),
            'PE': (pe_train, pe_seen, pe_unseen),
            'MI': (mi_train, mi_seen, mi_unseen),
        }
        label = ('train', 'seen', 'unseen')

        axs = {
            'VR': fig.add_subplot(311),
            'PE': fig.add_subplot(312),
            'MI': fig.add_subplot(313),
        }

        plt.legend()

    else:
        us_train, pe_train = get_all_uncertainty_measures_not_bayesian(all_outputs_train)
        us_seen, pe_seen = get_all_uncertainty_measures_not_bayesian(all_outputs_seen)
        us_unseen, pe_unseen = get_all_uncertainty_measures_not_bayesian(all_outputs_unseen)

        uncs = {
            'US': (us_train, us_seen, us_unseen),
            'PE': (pe_train, pe_seen, pe_unseen),
        }
        label = ('train', 'seen', 'unseen')

        axs = {
            'US': fig.add_subplot(211),
            'PE': fig.add_subplot(212),
        }

    for unc, ax in axs.items():
        ax.set_title(unc)
        plot_density_on_ax(ax, uncs[unc], hist=True, labels=label, **kwargs)
        ax.legend()

    if save_fig:
        assert save_path is not None
        print('Saving figure...')
        plt.savefig(save_path)
        print('Figure saved.')
    if show_fig:
        print('Showing figures...')
        plt.show()
        print('Figure shown.')


def compute_density_train_seen(arguments, all_outputs_train, all_outputs_seen, show_fig,
                               save_fig, save_path=None, figsize=(8, 10), **kwargs):
    """
    Compute and show the density distribution of uncertainties.
    Args:
        arguments (dict): the arguments of the python executed line in the terminal
        all_outputs_seen (torch.Tensor): size (nb_of_tests, size_of_testset_seen, nb_of_classes): the output of the
                                         softmax of the seen test set
        all_outputs_unseen (torch.Tensor): size (nb_of_tests, size_of_testset_unseen): the output of the softmax of the
                                           unseen dataset
        show_fig:
        save_fig:
        save_path:
        **kwargs:



        scale (str): scale of the plot. Strongly advice 'linear'.
        show_fig (bool): whether we show the figure
        save_fig (bool): whether we save the figure
        save_path (str): where to save the figures
    Returns:

    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(f'Distribution of uncertainties - {arguments["type_of_unseen"]} - train seen\n{arguments}', wrap=True)
    if 'rho' in arguments.keys():

        vr_train, pe_train, mi_train = get_all_uncertainty_measures(all_outputs_train)
        vr_seen, pe_seen, mi_seen = get_all_uncertainty_measures(all_outputs_seen)

        uncs = {
            'VR': (vr_train, vr_seen,),
            'PE': (pe_train, pe_seen,),
            'MI': (mi_train, mi_seen,),
        }
        label = ('train', 'seen',)

        axs = {
            'VR': fig.add_subplot(311),
            'PE': fig.add_subplot(312),
            'MI': fig.add_subplot(313),
        }

        plt.legend()

    else:
        us_train, pe_train = get_all_uncertainty_measures_not_bayesian(all_outputs_train)
        us_seen, pe_seen = get_all_uncertainty_measures_not_bayesian(all_outputs_seen)

        uncs = {
            'US': (us_train, us_seen,),
            'PE': (pe_train, pe_seen,),
        }
        label = ('train', 'seen',)

        axs = {
            'US': fig.add_subplot(211),
            'PE': fig.add_subplot(212),
        }

    for unc, ax in axs.items():
        ax.set_title(unc)
        plot_density_on_ax(ax, uncs[unc], hist=True, labels=label, **kwargs)
        ax.legend()

    if save_fig:
        assert save_path is not None
        print('Saving figure...')
        plt.savefig(save_path)
        print('Figure saved.')
    if show_fig:
        print('Showing figures...')
        plt.show()
        print('Figure shown.')


def compute_density_correct_false(arguments, all_outputs, true_labels, show_fig,
                                  save_fig, save_path=None, figsize=(8, 10), **kwargs):
    """
    Compute and show the density distribution of uncertainties.
    Args:
        arguments (dict): the arguments of the python executed line in the terminal
        all_outputs_seen (torch.Tensor): size (nb_of_tests, size_of_testset_seen, nb_of_classes): the output of the
                                         softmax of the seen test set
        all_outputs_unseen (torch.Tensor): size (nb_of_tests, size_of_testset_unseen): the output of the softmax of the
                                           unseen dataset
        show_fig:
        save_fig:
        save_path:
        **kwargs:



        scale (str): scale of the plot. Strongly advice 'linear'.
        show_fig (bool): whether we show the figure
        save_fig (bool): whether we save the figure
        save_path (str): where to save the figures
    Returns:

    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(f'Distribution of uncertainties - {arguments["type_of_unseen"]} - correct false \n{arguments}',
                 wrap=True)
    preds = get_predictions_from_multiple_tests(all_outputs).float()
    correct_preds = (true_labels == preds)
    all_outputs_correct = all_outputs[:, correct_preds == 1, :]
    all_outputs_false = all_outputs[:, correct_preds == 0, :]
    if 'rho' in arguments.keys():

        vr_train, pe_train, mi_train = get_all_uncertainty_measures(all_outputs_correct)
        vr_seen, pe_seen, mi_seen = get_all_uncertainty_measures(all_outputs_false)

        uncs = {
            'VR': (vr_train, vr_seen,),
            'PE': (pe_train, pe_seen,),
            'MI': (mi_train, mi_seen,),
        }
        label = ('correct', 'false',)

        axs = {
            'VR': fig.add_subplot(311),
            'PE': fig.add_subplot(312),
            'MI': fig.add_subplot(313),
        }

        plt.legend()

    else:
        us_train, pe_train = get_all_uncertainty_measures_not_bayesian(all_outputs_correct)
        us_seen, pe_seen = get_all_uncertainty_measures_not_bayesian(all_outputs_false)

        uncs = {
            'US': (us_train, us_seen,),
            'PE': (pe_train, pe_seen,),
        }
        label = ('correct', 'false',)

        axs = {
            'US': fig.add_subplot(211),
            'PE': fig.add_subplot(212),
        }

    for unc, ax in axs.items():
        ax.set_title(unc)
        plot_density_on_ax(ax, uncs[unc], hist=True, labels=label, **kwargs)
        ax.legend()

    if save_fig:
        assert save_path is not None
        print('Saving figure...')
        plt.savefig(save_path)
        print('Figure saved.')
    if show_fig:
        print('Showing figures...')
        plt.show()
        print('Figure shown.')


def compute_density_train_seen_correct_false(arguments, all_outputs_train, true_labels_train, all_outputs_seen,
                                             true_labels_seen, show_fig,
                                             save_fig, save_path=None, figsize=(8, 10), **kwargs):
    """
    Compute and show the density distribution of uncertainties.
    Args:
        arguments (dict): the arguments of the python executed line in the terminal
        all_outputs_seen (torch.Tensor): size (nb_of_tests, size_of_testset_seen, nb_of_classes): the output of the
                                         softmax of the seen test set
        all_outputs_unseen (torch.Tensor): size (nb_of_tests, size_of_testset_unseen): the output of the softmax of the
                                           unseen dataset
        show_fig:
        save_fig:
        save_path:
        **kwargs:



        scale (str): scale of the plot. Strongly advice 'linear'.
        show_fig (bool): whether we show the figure
        save_fig (bool): whether we save the figure
        save_path (str): where to save the figures
    Returns:

    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(
        f'Distribution of uncertainties - {arguments["type_of_unseen"]} - train seen correct false\n{arguments}',
        wrap=True)
    preds_train = get_predictions_from_multiple_tests(all_outputs_train).float()
    correct_preds_train = (true_labels_train == preds_train)
    all_outputs_train_correct = all_outputs_train[:, correct_preds_train == 1, :]
    all_outputs_train_false = all_outputs_train[:, correct_preds_train == 0, :]

    preds_seen = get_predictions_from_multiple_tests(all_outputs_seen).float()
    correct_preds_seen = (true_labels_seen == preds_seen)
    all_outputs_seen_correct = all_outputs_seen[:, correct_preds_seen == 1, :]
    all_outputs_seen_false = all_outputs_seen[:, correct_preds_seen == 0, :]

    if 'rho' in arguments.keys():

        vr_train_correct, pe_train_correct, mi_train_correct = get_all_uncertainty_measures(all_outputs_train_correct)
        vr_train_false, pe_train_false, mi_train_false = get_all_uncertainty_measures(all_outputs_train_false)
        vr_seen_correct, pe_seen_correct, mi_seen_correct = get_all_uncertainty_measures(all_outputs_seen_correct)
        vr_seen_false, pe_seen_false, mi_seen_false = get_all_uncertainty_measures(all_outputs_seen_false)

        if arguments['loss_type'] == 'bbb' or arguments['loss_type'] == 'uniform':
            axs = {
                'VR - correct': fig.add_subplot(321),
                'VR - false': fig.add_subplot(322),
                'PE - correct': fig.add_subplot(323),
                'PE - false': fig.add_subplot(324),
                'MI - correct': fig.add_subplot(325),
                'MI - false': fig.add_subplot(326),
            }

            uncs = {
                'VR - correct': (vr_train_correct, vr_seen_correct),
                'PE - correct': (pe_train_correct, pe_seen_correct),
                'MI - correct': (mi_train_correct, mi_seen_correct),
                'VR - false': (vr_train_false, vr_seen_false,),
                'PE - false': (pe_train_false, pe_seen_false),
                'MI - false': (mi_train_false, mi_seen_false,),
            }

            label = ('train', 'seen')

        else:
            uncs = {
                'VR': (vr_train_correct, vr_train_false, vr_seen_correct, vr_seen_false,),
                'PE': (pe_train_correct, pe_train_false, pe_seen_correct, pe_seen_false,),
                'MI': (mi_train_correct, mi_train_false, mi_seen_correct, mi_seen_false,),
            }
            label = ('train correct', 'train false', 'seen correct', 'seen false',)

            axs = {
                'VR': fig.add_subplot(311),
                'PE': fig.add_subplot(312),
                'MI': fig.add_subplot(313),
            }

        plt.legend()

    else:
        us_train_correct, pe_train_correct = get_all_uncertainty_measures_not_bayesian(all_outputs_train_correct)
        us_train_false, pe_train_false = get_all_uncertainty_measures_not_bayesian(all_outputs_train_false)
        us_seen_correct, pe_seen_correct = get_all_uncertainty_measures_not_bayesian(all_outputs_seen_correct)
        us_seen_false, pe_seen_false = get_all_uncertainty_measures_not_bayesian(all_outputs_seen_false)

        uncs = {
            'US': (us_train_correct, us_train_false, us_seen_correct, us_seen_false),
            'PE': (pe_train_correct, pe_train_false, pe_seen_correct, pe_seen_false),
        }
        label = ('train correct', 'train false', 'seen correct', 'seen false',)

        axs = {
            'US': fig.add_subplot(211),
            'PE': fig.add_subplot(212),
        }

    for unc, ax in axs.items():
        ax.set_title(unc)
        plot_density_on_ax(ax, uncs[unc], hist=True, labels=label, **kwargs)
        ax.legend()

    if save_fig:
        assert save_path is not None
        print('Saving figure...')
        plt.savefig(save_path)
        print('Figure saved.')
    if show_fig:
        print('Showing figures...')
        plt.show()
        print('Figure shown.')


def compute_density_train_seen_unseen_correct_false(arguments, all_outputs_train, true_labels_train, all_outputs_seen,
                                                    true_labels_seen, all_outputs_unseen, show_fig,
                                                    save_fig, save_path=None, figsize=(8, 10), **kwargs):
    """
    Compute and show the density distribution of uncertainties.
    Args:
        arguments (dict): the arguments of the python executed line in the terminal
        all_outputs_seen (torch.Tensor): size (nb_of_tests, size_of_testset_seen, nb_of_classes): the output of the
                                         softmax of the seen test set
        all_outputs_unseen (torch.Tensor): size (nb_of_tests, size_of_testset_unseen): the output of the softmax of the
                                           unseen dataset
        show_fig:
        save_fig:
        save_path:
        **kwargs:



        scale (str): scale of the plot. Strongly advice 'linear'.
        show_fig (bool): whether we show the figure
        save_fig (bool): whether we save the figure
        save_path (str): where to save the figures
    Returns:

    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(
        f'Distribution of uncertainties - {arguments["type_of_unseen"]} - train seen correct false\n{arguments}',
        wrap=True)
    preds_train = get_predictions_from_multiple_tests(all_outputs_train).float()
    correct_preds_train = (true_labels_train == preds_train)
    all_outputs_train_correct = all_outputs_train[:, correct_preds_train == 1, :]
    all_outputs_train_false = all_outputs_train[:, correct_preds_train == 0, :]

    preds_seen = get_predictions_from_multiple_tests(all_outputs_seen).float()
    correct_preds_seen = (true_labels_seen == preds_seen)
    all_outputs_seen_correct = all_outputs_seen[:, correct_preds_seen == 1, :]
    all_outputs_seen_false = all_outputs_seen[:, correct_preds_seen == 0, :]

    if 'rho' in arguments.keys():

        vr_train_correct, pe_train_correct, mi_train_correct = get_all_uncertainty_measures(all_outputs_train_correct)
        vr_train_false, pe_train_false, mi_train_false = get_all_uncertainty_measures(all_outputs_train_false)
        vr_seen_correct, pe_seen_correct, mi_seen_correct = get_all_uncertainty_measures(all_outputs_seen_correct)
        vr_seen_false, pe_seen_false, mi_seen_false = get_all_uncertainty_measures(all_outputs_seen_false)
        vr_unseen, pe_unseen, mi_unseen = get_all_uncertainty_measures(all_outputs_unseen)

        uncs = {
            'VR': (vr_train_correct, vr_train_false, vr_seen_correct, vr_seen_false, vr_unseen),
            'PE': (pe_train_correct, pe_train_false, pe_seen_correct, pe_seen_false, pe_unseen),
            'MI': (mi_train_correct, mi_train_false, mi_seen_correct, mi_seen_false, mi_unseen),
        }
        label = ('train correct', 'train false', 'seen correct', 'seen false', 'unseen',)

        axs = {
            'VR': fig.add_subplot(311),
            'PE': fig.add_subplot(312),
            'MI': fig.add_subplot(313),
        }

        plt.legend()

    else:
        us_train_correct, pe_train_correct = get_all_uncertainty_measures_not_bayesian(all_outputs_train_correct)
        us_train_false, pe_train_false = get_all_uncertainty_measures_not_bayesian(all_outputs_train_false)
        us_seen_correct, pe_seen_correct = get_all_uncertainty_measures_not_bayesian(all_outputs_seen_correct)
        us_seen_false, pe_seen_false = get_all_uncertainty_measures_not_bayesian(all_outputs_seen_false)
        us_unseen, pe_unseen = get_all_uncertainty_measures_not_bayesian(all_outputs_unseen)

        uncs = {
            'US': (us_train_correct, us_train_false, us_seen_correct, us_seen_false, us_unseen),
            'PE': (pe_train_correct, pe_train_false, pe_seen_correct, pe_seen_false, pe_unseen),
        }
        label = ('train correct', 'train false', 'seen correct', 'seen false', 'unseen',)

        axs = {
            'US': fig.add_subplot(211),
            'PE': fig.add_subplot(212),
        }

    for unc, ax in axs.items():
        ax.set_title(unc)
        plot_density_on_ax(ax, uncs[unc], hist=True, labels=label, **kwargs)
        ax.legend()

    if save_fig:
        assert save_path is not None
        print('Saving figure...')
        plt.savefig(save_path)
        print('Figure saved.')
    if show_fig:
        print('Showing figures...')
        plt.show()
        print('Figure shown.')


def get_seen_outputs_and_labels(bay_net_trained, arguments, device='cpu', verbose=True, ):
    """
    Gives the outputs of the model on the seen testset
    Args:
        bay_net_trained (torch.nn.Module child): model trained to evaluate
        arguments (dict): arguments of the experiment that produced the trained model. Used to get the type of unseen,
                          the labels of the testset and the number of tests
        device (torch.device): device to compute on
        verbose (bool): show progress bar

    Returns:
        torch.Tensor, torch.Tensor: size (nb of tests, size of testset, nb of classes), size (nb of tests)
    """
    type_of_unseen = arguments['type_of_unseen']
    if arguments['trainset'] == 'mnist':
        get_trainset = get_mnist
    elif arguments['trainset'] == 'cifar10':
        get_trainset = get_cifar10
    else:
        assert False, 'trainset not recognized'
    if type_of_unseen == 'unseen_classes':
        _, _, evalloader_seen = get_trainset(train_labels=(), eval_labels=range(arguments['split_labels']), batch_size=128,
                                          split_val=0, shuffle=False)
    else:
        _, _, evalloader_seen = get_trainset(train_labels=(), split_val=0, batch_size=128, shuffle=False)
    shuffle_eval = torch.randperm(len(evalloader_seen.dataset))
    evalloader_seen.dataset.data = evalloader_seen.dataset.data[shuffle_eval]
    evalloader_seen.dataset.targets = evalloader_seen.dataset.targets[shuffle_eval]
    if type(evalloader_seen.dataset.targets) == torch.Tensor:
        true_labels_seen = evalloader_seen.dataset.targets.float()
    else:
        true_labels_seen = evalloader_seen.dataset.targets.astype(float)
    if verbose:
        print('Evaluation on seen ...')
    _, all_eval_outputs = eval_bayesian(bay_net_trained, evalloader_seen,
                                        number_of_tests=arguments.get('number_of_tests', 1), device=device,
                                        verbose=verbose)
    if verbose:
        print('Finished evaluation on seen.')
    return all_eval_outputs, true_labels_seen

# TODO: put this function in primary_results_bayesian.py
def get_evalloader_unseen(arguments):
    type_of_unseen = arguments['type_of_unseen']
    if arguments['trainset'] == 'mnist':
        get_trainset = get_mnist
        dim_input = 28
        dim_channels = 1
    elif arguments['trainset'] == 'cifar10':
        get_trainset = get_cifar10
        dim_input = 32
        dim_channels = 3
    if type_of_unseen == 'random':
        _, _, evalloader_unseen = get_random(number_of_channels=dim_channels, img_dim=dim_input, number_of_classes=10)
    if type_of_unseen == 'unseen_classes':
        split_labels = arguments['split_labels']
        _, _, evalloader_unseen = get_trainset(train_labels=(), eval_labels=range(split_labels, 10, ), )
    if type_of_unseen == 'unseen_dataset':
        unseen_evalset = arguments['unseen_evalset']
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=dim_channels),
            transforms.Resize(dim_input),
            transforms.ToTensor(),
        ])
        if unseen_evalset == 'cifar10':
            _, _, evalloader_unseen = get_cifar10(transform=transform)
        if unseen_evalset == 'mnist':
            _, _, evalloader_unseen = get_mnist(transform=transform)
        if unseen_evalset == 'omniglot':
            _, _, evalloader_unseen = get_omniglot(transform=transform, download=False)
    return evalloader_unseen


def get_unseen_outputs(bay_net_trained, arguments, nb_of_random=None, device='cpu', verbose=True, ):
    """
    Gives the outputs of the model on the unseen testset
    Args:
        bay_net_trained (torch.nn.Module child): model trained to evaluate
        arguments (dict): arguments of the experiment that produced the trained model. Used to get the type of unseen,
                          the labels of the testset and the number of tests
        nb_of_random (int): number of random generated images if type of unseen is random
        device (torch.device): device to compute on
        verbose (bool): show progress bar

    Returns:
        torch.Tensor: size (nb of tests, size of testset, nb of classes)
    """
    evalloader_unseen = get_evalloader_unseen(arguments)
    if verbose:
        print('Evaluation on',  arguments['type_of_unseen'], '...')
    _, all_unseen_outputs = eval_bayesian(bay_net_trained, evalloader_unseen,
                                          number_of_tests=arguments.get('number_of_tests', 1), device=device,
                                          verbose=verbose, )
    if verbose:
        print(f'Finished evaluation on {arguments["type_of_unseen"]}.')
    return all_unseen_outputs


def get_trained_model_and_args_and_groupnb(exp_nb, exp_path='polyaxon_results/groups'):
    """
    Gives the model, arguments.pkl output of the experiment and the group number of this experiment
    Args:
        exp_nb (str): experiment number
        exp_path (str): path to the polyaxon groups results

    Returns:
        dict, dict, int: the results.pkl, arguments used plus the type of unseen, grp nb and exp_nb
    """
    exp_path = pathlib.Path(exp_path)
    _, all_dirs = get_file_and_dir_path_in_dir(exp_path, f'{exp_nb}/argumen')
    dirpath = all_dirs[0]
    group_nb = dirpath.split('/')[-2]
    dirpath = pathlib.Path(dirpath)
    arguments = load_from_file(dirpath / 'arguments.pkl')
    if 'split_labels' in arguments.keys():
        type_of_unseen = 'unseen_classes'
    elif 'dataset' in arguments.keys():
        type_of_unseen = 'unseen_dataset'
    else:
        type_of_unseen = 'random'
    arguments['type_of_unseen'] = type_of_unseen
    arguments['group_nb'] = group_nb
    arguments['exp_nb'] = exp_nb
    final_weights = torch.load(dirpath / 'final_weights.pt', map_location='cpu')
    std_prior = arguments.get('std_prior', 0)
    if arguments.get('trainset', 'mnist') == 'cifar10':
        bay_net_trained = GaussianClassifier(
            rho=arguments.get('rho', 'determinist'),
            stds_prior=(std_prior, std_prior),
            number_of_classes=10,
            dim_input=32,
            dim_channels=3,
        )
    else:
        bay_net_trained = GaussianClassifier(
            rho=arguments.get('rho', 'determinist'),
            stds_prior=(std_prior, std_prior),
            number_of_classes=10,
            dim_input=28,
        )
    bay_net_trained.load_state_dict(final_weights)

    return bay_net_trained, arguments, group_nb


def get_res_args_groupnb(exp_nb, exp_path='polyaxon_results/groups'):
    """
    Gives the results.pkl, arguments.pkl output of the experiment and the group number of this experiment
    Args:
        exp_nb (str): experiment number
        exp_path (str): path to the polyaxon groups results

    Returns:
        dict, dict, int: the results.pkl, arguments used plus the type of unseen, grp nb and exp_nb
    """
    exp_path = pathlib.Path(exp_path)
    dirpath = get_file_and_dir_path_in_dir(exp_path, f'{exp_nb}/argumen')[1][0]
    group_nb = dirpath.split('/')[-2]
    dirpath = pathlib.Path(dirpath)
    arguments = load_from_file(dirpath / 'arguments.pkl')
    results = load_from_file(dirpath / 'results.pkl')
    if 'split_labels' in arguments.keys():
        type_of_unseen = 'unseen_classes'
    elif 'dataset' in arguments.keys():
        type_of_unseen = 'unseen_dataset'
    else:
        type_of_unseen = 'random'
    arguments['type_of_unseen'] = type_of_unseen
    arguments['group_nb'] = group_nb
    arguments['exp_nb'] = exp_nb

    return results, arguments, group_nb


# TODO: refactor all these get_args to factorize them
def get_args(exp_nb, exp_path='polyaxon_results/groups'):
    """
    Gives the arguments.pkl output of the experiment. Almost the same as get_res_args_groupnb but doesn't give the same
    output.
    Args:
        exp_nb (str): experiment number
        exp_path (str): path to the polyaxon groups results

    Returns:
        dict: the arguments used plus the type of unseen, grp nb and exp_nb
    """
    exp_path = pathlib.Path(exp_path)
    dirpath = get_file_and_dir_path_in_dir(exp_path, f'{exp_nb}/argumen')[1][0]
    group_nb = dirpath.split('/')[-2]
    dirpath = pathlib.Path(dirpath)
    arguments = load_from_file(dirpath / 'arguments.pkl')
    if 'split_labels' in arguments.keys():
        type_of_unseen = 'unseen_classes'
    elif 'dataset' in arguments.keys():
        type_of_unseen = 'unseen_dataset'
    else:
        type_of_unseen = 'random'
    arguments['type_of_unseen'] = type_of_unseen
    arguments['group_nb'] = group_nb
    arguments['exp_nb'] = exp_nb

    return arguments


def get_train_outputs(bay_net_trained, arguments, device='cpu', verbose=True):
    """
    Gives the outputs of the model on the trainset
    Args:
        bay_net_trained (torch.nn.Module child): model trained to evaluate
        arguments (dict): arguments of the experiment that produced the trained model. Used to get the type of unseen,
                          the labels of the trainset and the number of tests
        device (torch.device): device to compute on
        verbose (bool): shows progress bar

    Returns:
        torch.Tensor: size (nb of tests, size of trainset, nb of classes)
    """
    type_of_unseen = arguments['type_of_unseen']
    if arguments['trainset'] == 'mnist':
        get_trainset = get_mnist
    elif arguments['trainset'] == 'cifar10':
        get_trainset = get_cifar10
    if type_of_unseen == 'unseen_classes':
        trainloader, _, _ = get_trainset(train_labels=range(arguments['split_labels']), eval_labels=(), split_val=0,
                                      batch_size=128, shuffle=False)
    else:
        trainloader, _, _ = get_trainset(eval_labels=(), split_val=0, batch_size=128, shuffle=False)

    shuffle_train = torch.randperm(len(trainloader.dataset))
    trainloader.dataset.data = trainloader.dataset.data[shuffle_train]
    trainloader.dataset.targets = trainloader.dataset.targets[shuffle_train]
    if type(trainloader.dataset.targets) == torch.Tensor:
        true_labels_train = trainloader.dataset.targets.float()
    else:
        true_labels_train = trainloader.dataset.targets.astype(float)
    print('Evaluation on train ...')
    _, all_outputs_train = eval_bayesian(bay_net_trained, trainloader, arguments.get('number_of_tests', 1),
                                         device=device, verbose=verbose)
    print('Finished evaluation on train.')
    return all_outputs_train, true_labels_train
