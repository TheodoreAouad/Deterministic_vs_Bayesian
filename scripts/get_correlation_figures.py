import pathlib

import matplotlib.pyplot as plt
import torch

from src.dataset_manager.get_data import get_mnist
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_predictions_from_multiple_tests, get_all_uncertainty_measures
from src.utils import load_from_file

exp_path = pathlib.Path('polyaxon_results/groups')
group_nb = '172'
_, all_dirs = u.get_file_and_dir_path_in_dir(exp_path/group_nb, 'argum')
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

eval_acc, all_outputs = eval_bayesian(bay_net_trained, evalloader, number_of_tests=arguments['number_of_tests'])
vr, pe, mi = get_all_uncertainty_measures(all_outputs)

predicted_labels = get_predictions_from_multiple_tests(all_outputs)
labels = evalloader.dataset.targets
size_of_batch = 200

correct_labels = predicted_labels == labels
correct_labels = correct_labels.reshape(10000//size_of_batch, size_of_batch).float()
accuracies = correct_labels.sum(1) / correct_labels.size(1)

vr_regrouped = vr.reshape(10000//size_of_batch, size_of_batch).mean(1)
pe_regrouped = pe.reshape(10000//size_of_batch, size_of_batch).mean(1)
mi_regrouped = mi.reshape(10000//size_of_batch, size_of_batch).mean(1)

plt.figure(figsize=(5,10))
plt.subplot(311)
plt.scatter(accuracies, vr_regrouped)
plt.title('VR')
plt.subplot(312)
plt.scatter(accuracies, pe_regrouped)
plt.title('PE')
plt.subplot(313)
plt.title('MI')
plt.scatter(accuracies, mi_regrouped)
plt.savefig('results/172_correlation_uncertainty_error.png')
