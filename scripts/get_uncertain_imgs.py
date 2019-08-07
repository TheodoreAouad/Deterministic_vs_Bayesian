import pathlib
import random

import torch
import matplotlib.pyplot as plt

from src.dataset_manager.get_data import get_mnist
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_all_uncertainty_measures, get_predictions_from_multiple_tests
from src.utils import get_file_and_dir_path_in_dir, load_from_file

threshold = .8
number_of_batches = 10

exp_path = pathlib.Path('polyaxon_results/groups')
group_nb = '172'
_, all_dirs = get_file_and_dir_path_in_dir(exp_path/group_nb, 'argum')
dirpath = pathlib.Path(all_dirs[0])

arguments = load_from_file(dirpath/'arguments.pkl')
final_weigths = torch.load(dirpath/'final_weights.pt', map_location='cpu')
bay_net_trained = GaussianClassifier(
    rho=arguments['rho'],
    stds_prior=(arguments['std_prior'], arguments['std_prior']),
    number_of_classes=10,
    dim_input=28,
)
bay_net_trained.load_state_dict(final_weigths)

_, _, evalloader = get_mnist(split_val=0, batch_size=128, shuffle=False)

print('Training ...')
eval_acc, all_outputs = eval_bayesian(bay_net_trained, evalloader, number_of_tests=arguments['number_of_tests'])
print('Training finished.')
vr, pe, mi = get_all_uncertainty_measures(all_outputs)
print('Uncertainty computed.')

predicted_labels = get_predictions_from_multiple_tests(all_outputs)

idx_high_vr = (vr > threshold * vr.max()).nonzero()
idx_high_pe = (pe > threshold * pe.max()).nonzero()
idx_high_mi = (mi > threshold * mi.max()).nonzero()


img_dir = pathlib.Path('results/images')
imgs_dir_vr = img_dir / 'high_variation_ratio'
imgs_dir_pe = img_dir / 'high_predictive_entropy'
imgs_dir_mi = img_dir / 'high_mutual_information'


print('Writing imgs with high variation-ratio ...')
imgs_dir_vr.mkdir(parents=True, exist_ok=True)
for batch_idx in range(number_of_batches):
    plt.figure(figsize=(13, 14))
    for k1 in range(3):
        for k2 in range(3):
            random_label_with_high_vr = random.choice(idx_high_vr)
            plt.subplot(3, 3, 3*k1 + k2+1)
            plt.axis('off')
            plt.imshow(evalloader.dataset.data[random_label_with_high_vr][0])
            plt.title(f'------------------'
                      f'\nVR: {round(vr[random_label_with_high_vr].item(), 2)} || '
                      f'PE: {round(pe[random_label_with_high_vr].item(), 2)} || '
                      f'MI: {round(mi[random_label_with_high_vr].item(), 2)}\n'
                      f'True: {evalloader.dataset.targets[random_label_with_high_vr].item()} || '
                      f'Predicted: {predicted_labels[random_label_with_high_vr].item()} || '
                      f'Index: {random_label_with_high_vr.item()}')
    plt.savefig(imgs_dir_vr / f'vr_{group_nb}_{batch_idx}.png')
    plt.close()

print('Writing imgs with high predictive entropy ...')
imgs_dir_pe.mkdir(parents=True, exist_ok=True)
for batch_idx in range(10):
    plt.figure(figsize=(13, 14))
    for k1 in range(3):
        for k2 in range(3):
            random_label_with_high_pe = random.choice(idx_high_pe)
            plt.subplot(3, 3, 3*k1 + k2+1)
            plt.axis('off')
            plt.imshow(evalloader.dataset.data[random_label_with_high_pe][0])
            plt.title(f'------------------'
                      f'\nVR: {round(vr[random_label_with_high_pe].item(), 2)} || '
                      f'PE: {round(pe[random_label_with_high_pe].item(), 2)} || '
                      f'MI: {round(mi[random_label_with_high_pe].item(), 2)}\n'
                      f'True: {evalloader.dataset.targets[random_label_with_high_pe].item()} || '
                      f'Predicted: {predicted_labels[random_label_with_high_pe].item()} || '
                      f'Index: {random_label_with_high_pe.item()}')
    plt.savefig(imgs_dir_pe / f'pe_{group_nb}_{batch_idx}.png')
    plt.close()

print('Writing imgs with high mutual information ...')
imgs_dir_mi.mkdir(parents=True, exist_ok=True)
for batch_idx in range(10):
    plt.figure(figsize=(13, 14))
    for k1 in range(3):
        for k2 in range(3):
            random_label_with_high_mi = random.choice(idx_high_mi)
            plt.subplot(3, 3, 3*k1 + k2+1)
            plt.axis('off')
            plt.imshow(evalloader.dataset.data[random_label_with_high_mi][0])
            plt.title(f'------------------'
                      f'\nVR: {round(vr[random_label_with_high_mi].item(), 2)} || '
                      f'PE: {round(pe[random_label_with_high_mi].item(), 2)} || '
                      f'MI: {round(mi[random_label_with_high_mi].item(), 2)}\n'
                      f'True: {evalloader.dataset.targets[random_label_with_high_mi].item()} || '
                      f'Predicted: {predicted_labels[random_label_with_high_mi].item()} || '
                      f'Index: {random_label_with_high_mi.item()}')
    plt.savefig(imgs_dir_mi / f'mi_{group_nb}_{batch_idx}.png')
    plt.close()
