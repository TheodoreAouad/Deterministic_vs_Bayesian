import pathlib

import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torchvision import transforms as transforms
from tqdm import tqdm

from src.dataset_manager.get_data import get_mnist, get_omniglot, get_cifar10
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.evals import eval_bayesian, eval_random

from src.uncertainty_measures import get_predictions_from_multiple_tests, get_all_uncertainty_measures, \
    get_all_uncertainty_measures_not_bayesian
from src.utils import get_exact_batch_size, get_file_and_dir_path_in_dir, load_from_file


# TODO: refactor this function into modular functions
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
        plt.title(f'VR - shuffled ({size_of_batch} imgs/pt), corr: {round(correlation(vr_regrouped_shuffled, accuracies_shuffled), 2)}')
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
        plt.title(f'PE - shuffled ({size_of_batch} imgs/pt), corr: {round(correlation(pe_regrouped_shuffled, accuracies_shuffled), 2)}')
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
        plt.title(f'MI - shuffled ({size_of_batch} imgs/pt), corr: {round(correlation(mi_regrouped_shuffled, accuracies_shuffled), 2)}')
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
        plt.title(f'PE - shuffled ({size_of_batch} imgs/pt), corr: {round(correlation(pe_regrouped_shuffled, accuracies_shuffled), 2)}')
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


def compute_density(
        arguments,
        all_outputs_train,
        all_outputs_seen,
        all_outputs_unseen,
        show_fig,
        save_fig,
        save_path=None,
        figsize=(8,10),
        **kwargs,
):
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
    plt.suptitle(f'Distribution of uncertainties - {arguments["type_of_unseen"]} \n{arguments}', wrap=True)
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

        # plt.subplot(311)
        # plt.title('VR')
        # sns.distplot(vr_train, hist=False, kde_kws={"shade": True}, label='train', **kwargs)
        # sns.distplot(vr_seen, hist=False, kde_kws={"shade": True}, label='seen', **kwargs)
        # sns.distplot(vr_unseen, hist=False, kde_kws={"shade": True}, label='unseen', **kwargs)
        # plt.legend()
        #
        # plt.subplot(312)
        # plt.title('PE')
        # sns.distplot(pe_train, hist=False, kde_kws={"shade": True}, label='train', **kwargs)
        # sns.distplot(pe_seen, hist=False, kde_kws={"shade": True}, label='seen', **kwargs)
        # sns.distplot(pe_unseen, hist=False, kde_kws={"shade": True}, label='unseen', **kwargs)
        # plt.legend()
        #
        # plt.subplot(313)
        # plt.title('MI')
        # sns.distplot(mi_train, hist=False, kde_kws={"shade": True}, label='train', **kwargs)
        # sns.distplot(mi_seen, hist=False, kde_kws={"shade": True}, label='seen', **kwargs)
        # sns.distplot(mi_unseen, hist=False, kde_kws={"shade": True}, label='unseen', **kwargs)
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
        #
        # plt.subplot(311)
        # plt.title('US')
        # sns.distplot(us_train, hist=False, kde_kws={"shade": True}, label='train', **kwargs)
        # sns.distplot(us_seen, hist=False, kde_kws={"shade": True}, label='seen', **kwargs)
        # sns.distplot(us_unseen, hist=False, kde_kws={"shade": True}, label='seen', **kwargs)
        # plt.legend()
        #
        # plt.subplot(312)
        # plt.title('PE')
        # sns.distplot(pe_train, hist=False, kde_kws={"shade": True}, label='train', **kwargs)
        # sns.distplot(pe_seen, hist=False, kde_kws={"shade": True}, label='seen', **kwargs)
        # sns.distplot(pe_unseen, hist=False, kde_kws={"shade": True}, label='unseen', **kwargs)
        # plt.legend()

    for unc, ax in axs.items():
        ax.set_title(unc)
        plot_density(ax, uncs[unc], label, **kwargs)

    if save_fig:
        assert save_path is not None
        print('Saving figure...')
        plt.savefig(save_path)
        print('Figure saved.')
    if show_fig:
        print('Showing figures...')
        plt.show()
        print('Figure shown.')


def plot_density(ax, uncs, labels, **kwargs):
    for unc, label in zip(uncs, labels):
        sns.distplot(unc, hist=False, kde_kws={"shade": True}, label=label, ax=ax, **kwargs)
    plt.legend()



def get_seen_outputs_and_labels(bay_net_trained, arguments, device='cpu', verbose=True,):
    type_of_unseen = arguments['type_of_unseen']
    if type_of_unseen == 'unseen_classes':
        _, _, evalloader_seen = get_mnist(train_labels=(), eval_labels=range(arguments['split_labels']), batch_size=128,
                                          split_val=0, shuffle=False)
    else:
        _, _, evalloader_seen = get_mnist(train_labels=(), split_val=0, batch_size=128, shuffle=False)
    shuffle_eval = torch.randperm(len(evalloader_seen.dataset))
    evalloader_seen.dataset.data = evalloader_seen.dataset.data[shuffle_eval]
    evalloader_seen.dataset.targets = evalloader_seen.dataset.targets[shuffle_eval]
    true_labels_seen = evalloader_seen.dataset.targets.float()
    if verbose:
        print('Evaluation on seen ...')
    _, all_eval_outputs = eval_bayesian(bay_net_trained, evalloader_seen,
                                        number_of_tests=arguments.get('number_of_tests', 1), device=device,
                                        verbose=verbose)
    if verbose:
        print('Finished evaluation on seen.')
    return all_eval_outputs, true_labels_seen


def get_unseen_outputs(bay_net_trained, arguments, nb_of_random=None, device='cpu', verbose=True,):
    global evalloader_unseen
    type_of_unseen = arguments['type_of_unseen']
    if type_of_unseen == 'random':
        assert nb_of_random is not None, 'Give a number of random samples'
        if verbose:
            print('Evaluation on random ...')
        output_random, _ = eval_random(bay_net_trained, batch_size=nb_of_random, img_channels=1, img_dim=28,
                                       number_of_tests=arguments.get('number_of_tests', 1), verbose=True, device=device)
        if verbose:
            print('Finished evaluation on random.')
        return output_random
    if type_of_unseen == 'unseen_classes':
        _, _, evalloader_unseen = get_mnist(train_labels=(), eval_labels=range(arguments['split_labels'], 10),
                                            batch_size=128, split_val=0)
    elif type_of_unseen == 'unseen_dataset':
        dataset = arguments['dataset']
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor()
        ])
        if dataset == "omniglot":
            evalloader_unseen = get_omniglot(transform=transform, batch_size=128, download=False)
        elif dataset == "cifar10":
            transform = transforms.Compose([
                transforms.Grayscale(),
                transform
            ])
            _, evalloader_unseen = get_cifar10(transform=transform, batch_size=128)
    else:
        raise TypeError('Unrecognized type_of_unseen. Is either "random", "unseen_classes", "unseen_dataset"')
    if verbose:
        print('Evaluation on', type_of_unseen, '...')
    _, all_unseen_outputs = eval_bayesian(bay_net_trained, evalloader_unseen,
                                          number_of_tests=arguments.get('number_of_tests', 1), device=device,
                                          verbose=verbose, )
    if verbose:
        print(f'Finished evaluation on {type_of_unseen}.')
    return all_unseen_outputs


def get_trained_model_and_args_and_groupnb(exp_nb, exp_path='polyaxon_results/groups'):
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
    bay_net_trained = GaussianClassifier(
        rho=arguments.get('rho', 'determinist'),
        stds_prior=(std_prior, std_prior),
        number_of_classes=10,
        dim_input=28,
    )
    bay_net_trained.load_state_dict(final_weights)

    return bay_net_trained, arguments, group_nb


def get_res_args_groupnb(exp_nb, exp_path='polyaxon_results/groups'):
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


def get_args(exp_nb, exp_path='polyaxon_results/groups'):
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

def get_train_outputs(bay_net_trained, arguments, device='cpu'):
    type_of_unseen = arguments['type_of_unseen']
    if type_of_unseen == 'unseen_classes':
        trainloader, _, _ = get_mnist(train_labels=range(arguments['split_labels']), eval_labels=(), split_val=0,
                                      batch_size=128)
    else:
        trainloader, _, _ = get_mnist(eval_labels=(), split_val=0, batch_size=128)
    print('Evaluation on train ...')
    _, all_outputs_train = eval_bayesian(bay_net_trained, trainloader, arguments.get('number_of_tests', 1),
                                         device=device, verbose=True)
    print('Finished evaluation on train.')
    return all_outputs_train
