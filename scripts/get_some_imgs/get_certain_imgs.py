import pathlib
import random

import torch
import matplotlib.pyplot as plt

from src.dataset_manager.get_data import get_mnist
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_all_uncertainty_measures, get_predictions_from_multiple_tests
from src.utils import get_file_and_dir_path_in_dir, load_from_file

from scripts.utils import get_trained_model_and_args_and_groupnb, get_seen_outputs_and_labels, get_evalloader_seen

###### TO CHANGE #########
exp_nb = '14621'
threshold = .05         ##
number_of_batches = 10  ##
number_of_tests = 10
##########################

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

def write_imgs_in_dir(imgs_dir, number_of_batches, idx_high_unc, type_of_unc):
    """
    This function writes images of a certain label in a directory
    Args:
        imgs_dir (pathlib.PosixPath): parent directory of all images
        number_of_batches (int): number of examples per image
        idx_high_unc (torch.Tensor): tensor of indexes to sample from to get images
        type_of_unc (str): the type of uncertainty to write in the file name

    """
    imgs_dir.mkdir(parents=True, exist_ok=True)
    for batch_idx in range(number_of_batches):
        plt.figure(figsize=(13, 14))
        for k1 in range(3):
            for k2 in range(3):
                random_label_with_high_uncertainty = random.choice(idx_high_unc)
                plt.subplot(3, 3, 3 * k1 + k2 + 1)
                plt.axis('off')
                img = evalloader.dataset.data[random_label_with_high_uncertainty]
                plt.imshow(img)
                plt.title(f'------------------'
                          f'\nVR: {round(vr[random_label_with_high_uncertainty].item(), 4)} || '
                          f'PE: {round(pe[random_label_with_high_uncertainty].item(), 4)} || '
                          f'MI: {round(mi[random_label_with_high_uncertainty].item(), 4)}\n'
                          f'True: {evalloader.dataset.targets[random_label_with_high_uncertainty].item()} || '
                          f'Predicted: {predicted_labels[random_label_with_high_uncertainty].item()} || '
                          f'Index: {random_label_with_high_uncertainty.item()}')
        plt.suptitle(arguments, wrap=True)
        plt.savefig(imgs_dir / f'{type_of_unc}_{group_nb}_{exp_nb}_{batch_idx}.png')
        plt.close()


exp_path = pathlib.Path('polyaxon_results/groups')
bay_net_trained, arguments, group_nb = get_trained_model_and_args_and_groupnb(exp_nb)

print('Testing ...')
evalloader = get_evalloader_seen(arguments, shuffle=False)
labels, all_outputs = eval_bayesian(
    bay_net_trained,
    evalloader,
    number_of_tests=number_of_tests,
    return_accuracy=False,
    device=device,
    verbose=True,
)
print('Testing finished.')
vr, pe, mi = get_all_uncertainty_measures(all_outputs)
print('Uncertainty computed.')

predicted_labels = get_predictions_from_multiple_tests(all_outputs)

idx_high_vr = (vr < threshold * vr.max()).nonzero()
idx_high_pe = (pe < threshold * pe.max()).nonzero()
idx_high_mi = (mi < threshold * mi.max()).nonzero()


img_dir = pathlib.Path('results/images')
imgs_dir_vr = img_dir / arguments['loss_type'] / 'low_variation_ratio'
imgs_dir_pe = img_dir / arguments['loss_type'] / 'low_predictive_entropy'
imgs_dir_mi = img_dir / arguments['loss_type'] / 'low_mutual_information'


print('Writing imgs with low variation-ratio ...')
write_imgs_in_dir(imgs_dir_vr, number_of_batches, idx_high_vr, 'vr')

print('Writing imgs with low predictive entropy ...')
write_imgs_in_dir(imgs_dir_pe, number_of_batches, idx_high_pe, 'pe')

print('Writing imgs with low mutual information ...')
write_imgs_in_dir(imgs_dir_mi, number_of_batches, idx_high_mi, 'mi')
