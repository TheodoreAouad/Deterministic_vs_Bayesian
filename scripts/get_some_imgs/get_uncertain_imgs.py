import pathlib
import random

import torch
import matplotlib.pyplot as plt

from src.uncertainty_measures import get_predictions_from_multiple_tests, get_all_uncertainty_measures

from scripts.utils import get_trained_model_and_args_and_groupnb, get_evalloader_seen, get_args
import scripts.utils as su
from importlib import reload
reload(su)

###### TO CHANGE #########
# exp_nb = '20273'          #CIFAR rho -10, std 0.55
exp_nb = 21404            #MNIST rho -10, std 0.55
threshold_uncertain = .5        ##
threshold_certain = 0.05      ##
number_of_images_per_row = 5  ##
number_of_images_per_col = 5  ##
number_of_tests = 10
uncertain = True
# uncertain = False
##########################

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

def write_imgs_in_dir(imgs_dir, number_of_images_per_row, number_of_images_per_col, idx_high_unc, type_of_unc, labels,):
    """
    This function writes images of a certain label in a directory
    Args:
        imgs_dir (pathlib.PosixPath): parent directory of all images
        number_of_batches (int): number of examples per image
        idx_high_unc (torch.Tensor): tensor of indexes to sample from to get images
        type_of_unc (str): the type of uncertainty to write in the file name

    """
    imgs_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(number_of_images_per_row, number_of_images_per_col, figsize=(4.3*number_of_images_per_row, 4.3*number_of_images_per_col,))
    for k1 in range(number_of_images_per_row):
        for k2 in range(number_of_images_per_col):
            random_label_with_high_uncertainty = random.choice(idx_high_unc)
            axs[k1, k2].axis('off')
            img = evalloader.dataset.data[random_label_with_high_uncertainty]
            if img.shape == (1, 28, 28):
                img = img.squeeze()
            axs[k1, k2].imshow(img)
            title = f'------------------\n'
            # for unc, unc_name in zip(uncs, uncs_name):
            #     title += f'{unc_name}: {round(unc[random_label_with_high_uncertainty].item(), 4)} || '
            title += (
                f'Index: {random_label_with_high_uncertainty.item()}'
                f'\nTrue: {labels[evalloader.dataset.targets[random_label_with_high_uncertainty].item()]} || '
                f'Pred: {labels[predicted_labels[random_label_with_high_uncertainty].item()]} || '
            )
            axs[k1, k2].set_title(title)
    fig.suptitle(arguments, wrap=True)
    fig.savefig(imgs_dir / f'{type_of_unc}_{number_of_images_per_row*number_of_images_per_col}_{group_nb}_{exp_nb}.png')
    fig.tight_layout()
    plt.close(fig)


exp_path = pathlib.Path('polyaxon_results/groups')
arguments = get_args(exp_nb)
evalloader = get_evalloader_seen(arguments, shuffle=False)
bay_net_trained, arguments, group_nb = get_trained_model_and_args_and_groupnb(exp_nb, exp_path=exp_path)

true_labels, all_outputs, _ = su.get_saved_outputs_labels_seen_unseen(
    exp_nb=exp_nb,
    number_of_tests=number_of_tests,
    shuffle=False,
)
print('Testing finished.')
uncs = get_all_uncertainty_measures(all_outputs)
print('Uncertainty computed.')

if arguments['trainset'] == 'cifar10':
    labels_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']
else:
    labels_name = range(10)

predicted_labels = get_predictions_from_multiple_tests(all_outputs)

if arguments['determinist'] or arguments.get('rho', 'determinist') == 'determinist':
    uncs_name = ['sr', 'pe']
else:
    uncs_name = ['sr', 'vr', 'pe', 'mi']
img_dir = pathlib.Path('results/images')
for unc, unc_name in zip(uncs, uncs_name):
    if uncertain:
        idx_high_unc = (unc > threshold_uncertain * unc.max()).nonzero()
        imgs_dir_unc = img_dir / arguments['trainset'] / arguments['loss_type'] / f'high_{unc_name}'
        print(f'Writing imgs with high {unc_name}...')
    else:
        idx_high_unc = (unc < threshold_certain * unc.max()).nonzero()
        imgs_dir_unc = img_dir / arguments['trainset'] / arguments['loss_type'] / f'low_{unc_name}'
        print(f'Writing imgs with low {unc_name}...')
    write_imgs_in_dir(imgs_dir_unc, number_of_images_per_row, number_of_images_per_col, idx_high_unc, unc_name, labels_name)
