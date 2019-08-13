import pathlib

import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np

from src.dataset_manager.get_data import get_mnist
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.evals import eval_bayesian, eval_random
from src.uncertainty_measures import get_predictions_from_multiple_tests, get_all_uncertainty_measures
from src.utils import load_from_file, get_file_and_dir_path_in_dir

###### TO CHANGE ###########
group_nb = '172'
idx_in_group = 0
nb_of_batches = 100
size_of_batch = 100
nb_of_random = 1000
save_fig = False
show_correlation = False
do_eval_mnist = True
############################

def get_divisors(n):
    res = []
    for k in range(1, int(np.sqrt(n))):
        if n % k == 0:
            res.append(k)
    return res


def get_exact_batch_size(size_of_batch, total_nb_sample):
    divisors = get_divisors(total_nb_sample)
    return min(divisors, key=lambda x: abs(x - size_of_batch))


exp_path = pathlib.Path('polyaxon_results/groups')
_, all_dirs = get_file_and_dir_path_in_dir(exp_path / group_nb, 'argumen')
dirpath = all_dirs[idx_in_group]
exp_nb = dirpath.split('/')[-1]
dirpath = pathlib.Path(dirpath)

arguments = load_from_file(dirpath / 'arguments.pkl')
save_path = pathlib.Path(f'results/correlations_figures/{arguments["loss_type"]}')
save_path.mkdir(parents=True, exist_ok=True)
final_weigths = torch.load(dirpath / 'final_weights.pt', map_location='cpu')

_, _, evalloader = get_mnist(split_val=0, batch_size=128, shuffle=False)
true_labels_mnist = evalloader.dataset.targets.float()
nb_of_mnist = len(evalloader.dataset)

bay_net_trained = GaussianClassifier(
    rho=arguments['rho'],
    stds_prior=(arguments['std_prior'], arguments['std_prior']),
    number_of_classes=10,
    dim_input=28,
)
bay_net_trained.load_state_dict(final_weigths)

# Get outputs of evaluation on MNIST and on Random
if do_eval_mnist:
    shuffle_eval = torch.randperm(len(evalloader.dataset))
    evalloader.dataset.data = evalloader.dataset.data[shuffle_eval]
    evalloader.dataset.targets = evalloader.dataset.targets[shuffle_eval]
    print('Evaluation on MNIST .s..')
    eval_acc, all_eval_outputs = eval_bayesian(bay_net_trained, evalloader,
                                               number_of_tests=arguments['number_of_tests'])
    print('Finished evaluation on MNIST.')
output_random = torch.Tensor()
if nb_of_random > 0:
    print('Evaluation on random ...')
    output_random, _ = eval_random(
        bay_net_trained,
        batch_size=nb_of_random,
        img_channels=1,
        img_dim=28,
        number_of_tests=arguments['number_of_tests'],
        show_progress=True,
    )
    print('Finished evaluation on random.')
all_outputs = torch.cat((all_eval_outputs, output_random), 1)
total_nb_of_data = len(true_labels_mnist) + nb_of_random

# Shuffle data between rand and mnist
all_outputs_shuffled = torch.Tensor()
true_labels_shuffled = torch.Tensor()
for i in range(nb_of_batches):
    idx_m = torch.randperm(nb_of_mnist)
    idx_r = torch.randperm(nb_of_random)
    k = int(i / nb_of_batches * size_of_batch)
    batch_img = torch.cat((all_eval_outputs[:, idx_m[:size_of_batch - k], :], output_random[:, idx_r[:k], :]), 1)
    all_outputs_shuffled = torch.cat((all_outputs_shuffled, batch_img), 1)
    batch_label = torch.cat((true_labels_mnist[idx_m[:size_of_batch - k]], -1 + torch.zeros(k)))
    true_labels_shuffled = torch.cat((true_labels_shuffled, batch_label))

# Get the labels of the evaluation on MNIST, and put -1 for the labels of Random
predicted_labels = get_predictions_from_multiple_tests(all_outputs).float()
true_labels = torch.cat((true_labels_mnist, -1 + torch.zeros(nb_of_random)))
correct_labels = (predicted_labels == true_labels)
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
prop_of_rand = random_idxs_reshaped.mean(0)
labels_not_shuffled = true_labels.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).mean(1)

# get uncertainties
vr_shuffled, pe_shuffled, mi_shuffled = get_all_uncertainty_measures(all_outputs_shuffled)
vr, pe, mi = get_all_uncertainty_measures(all_outputs)
vr_regrouped_shuffled = vr_shuffled.reshape(size_of_batch, nb_of_batches).mean(1)
pe_regrouped_shuffled = pe_shuffled.reshape(size_of_batch, nb_of_batches).mean(1)
mi_regrouped_shuffled = mi_shuffled.reshape(size_of_batch, nb_of_batches).mean(1)
vr_regrouped = vr.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).mean(1)
pe_regrouped = pe.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).mean(1)
mi_regrouped = mi.reshape(total_nb_of_data // real_size_of_batch, real_size_of_batch).mean(1)


# plot graphs
try: arguments['nb_of_tests'] = arguments.pop('number_of_tests')
except KeyError: pass
plt.figure(figsize=(8, 10))
plt.suptitle(arguments, wrap=True)
plt.subplot(321)
plt.scatter(vr_regrouped, accuracies, c=labels_not_shuffled)
plt.ylabel('accuracy')
plt.title('VR - not shuffled')
plt.subplot(322)
plt.scatter(vr_regrouped_shuffled, accuracies_shuffled, c=prop_of_rand)
plt.ylabel('accuracy')
plt.title('VR - shuffled')
cbar = plt.colorbar()
cbar.set_label('random ratio', rotation=270)
plt.subplot(323)
plt.scatter(pe_regrouped, accuracies, c=labels_not_shuffled)
plt.ylabel('accuracy')
plt.title('PE - not shuffled')
plt.subplot(324)
plt.scatter(pe_regrouped_shuffled, accuracies_shuffled, c=prop_of_rand)
plt.ylabel('accuracy')
plt.title('PE - shuffled')
cbar = plt.colorbar()
cbar.set_label('random ratio', rotation=270)
plt.subplot(325)
plt.title('MI - not shuffled')
plt.scatter(mi_regrouped, accuracies, c=labels_not_shuffled)
plt.ylabel('accuracy')
plt.subplot(326)
plt.title('MI - shuffled')
plt.scatter(mi_regrouped_shuffled, accuracies_shuffled, c=prop_of_rand)
plt.ylabel('accuracy')
cbar = plt.colorbar()
cbar.set_label('random ratio', rotation=270)

if save_fig:
    plt.savefig(save_path / f'{group_nb}_{exp_nb}_correlation_uncertainty_error.png')
# pd.plotting.scatter_matrix(
# #     pd.DataFrame({
# #         'vr': np.array(vr_regrouped),
# #         'pe': np.array(pe_regrouped),
# #         'mi': np.array(mi_regrouped),
# #     }),
# # )
plt.show()
