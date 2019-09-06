from math import log, exp
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from src.loggers.losses.base_loss import BaseLoss
from src.loggers.losses.bbb_loss import BBBLoss
from src.loggers.observables import AccuracyAndUncertainty
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.risk_control import get_selection_threshold_all_unc
from src.tasks.trains import train_bayesian_modular, uniform
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_all_uncertainty_measures, get_predictions_from_multiple_tests
from src.utils import set_and_print_random_seed, save_to_file, convert_df_to_cpu
from src.dataset_manager.get_data import get_mnist

parser = argparse.ArgumentParser()
parser.add_argument('--rho', help='variable symbolizing the variance. std = log(1+exp(rho))',
                    type=float, default=-6)
parser.add_argument('--epoch', help='number of times we train the model on the same data',
                    type=int, default=30)
parser.add_argument('--batch_size', help='number of batches to split the data into',
                    type=int, default=32)
parser.add_argument('--number_of_tests', help='number of evaluations to perform for each each image to check for '
                                              'uncertainty', type=int, default=20)
parser.add_argument('--loss_type', help='which loss to use', choices=['exp', 'uniform', 'criterion'], type=str,
                    default='uniform')
parser.add_argument('--std_prior', help='the standard deviation of the prior', type=float, default=0.1)
parser.add_argument('--delta', help='probability upper bound of error higher that risk', type=float)

args = parser.parse_args()
save_to_file(vars(args), './output/arguments.pkl')

rho = args.rho
epoch = args.epoch
batch_size = args.batch_size
number_of_tests = args.number_of_tests
loss_type = args.loss_type
std_prior = args.std_prior
stds_prior = (std_prior, std_prior)
delta = args.delta
risks = np.linspace(0.01, 0.3, 20)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

trainloader, valloader, evalloader = get_mnist(train_labels=range(10), eval_labels=range(10),
                                               batch_size=batch_size)

seed_model = set_and_print_random_seed()
bay_net = GaussianClassifier(rho=rho, stds_prior=stds_prior, dim_input=28, number_of_classes=10)
bay_net.to(device)
criterion = CrossEntropyLoss()
if loss_type == 'uniform':
    step_function = uniform
    loss = BBBLoss(bay_net, criterion, step_function)
elif loss_type == 'exp':
    def step_function(batch_idx, number_of_batches):
        return 2 ** (number_of_batches - batch_idx) / (2 ** number_of_batches - 1)


    loss = BBBLoss(bay_net, criterion, step_function)
else:
    loss = BaseLoss(criterion)

optimizer = optim.Adam(bay_net.parameters())
observables = AccuracyAndUncertainty()
train_bayesian_modular(
    bay_net,
    optimizer,
    loss,
    observables,
    number_of_tests=number_of_tests,
    number_of_epochs=epoch,
    trainloader=trainloader,
    valloader=valloader,
    output_dir_tensorboard='./output',
    device=device,
    verbose=True,
)

true_labels_train, all_outputs_train = eval_bayesian(
    bay_net,
    trainloader,
    return_accuracy=False,
    number_of_tests=number_of_tests,
    device=device,
)

train_vr, train_pe, train_mi = get_all_uncertainty_measures(all_outputs_train)

true_eval_labels, all_outputs_eval = eval_bayesian(
    bay_net,
    evalloader,
    return_accuracy=False,
    number_of_tests=number_of_tests,
    device=device,
)
eval_vr, eval_pe, eval_mi = get_all_uncertainty_measures(all_outputs_eval)

eval_preds = get_predictions_from_multiple_tests(all_outputs_eval)
correct_preds = (eval_preds.float() == true_eval_labels.float()).float()

eval_acc_vrs = []
eval_acc_pes = []
eval_acc_mis = []
eval_coverage_vrs = []
eval_coverage_pes = []
eval_coverage_mis = []

for risk in tqdm(risks):
    threshold_vr, threshold_pe, threshold_mi = get_selection_threshold_all_unc(
        bay_net,
        true_labels_train,
        all_outputs_train,
        risk,
        delta,
        (train_vr, train_pe, train_mi),
        number_of_tests,
        verbose=False,
        device=device,
    )

    eval_acc_vrs.append(correct_preds[-eval_vr >= threshold_vr].mean().item())
    eval_acc_pes.append(correct_preds[-eval_pe >= threshold_pe].mean().item())
    eval_acc_mis.append(correct_preds[-eval_mi >= threshold_mi].mean().item())
    eval_coverage_vrs.append((correct_preds[-eval_vr >= threshold_vr].sum()/correct_preds.size(0)).item())
    eval_coverage_pes.append((correct_preds[-eval_pe >= threshold_pe].sum()/correct_preds.size(0)).item())
    eval_coverage_mis.append((correct_preds[-eval_mi >= threshold_mi].sum()/correct_preds.size(0)).item())

print(f'Eval acc vr: {round(100 * eval_acc_vrs, 2)} %')
print(f'Eval acc pe: {round(100 * eval_acc_pes, 2)} %')
print(f'Eval acc mi: {round(100 * eval_acc_mis, 2)} %')
print(f'Eval coverage vr: {round(100 * eval_coverage_vrs, 2)} %')
print(f'Eval coverage pe: {round(100 * eval_coverage_pes, 2)} %')
print(f'Eval coverage mi: {round(100 * eval_coverage_mis, 2)} %')
print(f'Variation-Ratio:{eval_vr.mean()}')
print(f'Predictive Entropy:{eval_pe.mean()}')
print(f'Mutual Information:{eval_mi.mean()}')

res = pd.DataFrame.from_dict({
    'loss_type': [loss_type],
    'number of epochs': [epoch],
    'batch_size': [batch_size],
    'split_train': [len(trainloader.dataset)],
    'number of tests': [number_of_tests],
    'seed_model': [seed_model],
    'stds_prior': [std_prior],
    'rho': [rho],
    'sigma initial': [log(1 + exp(rho))],
    'train accuracy': [observables.logs['train_accuracy_on_epoch']],
    'train max acc': [observables.max_train_accuracy_on_epoch],
    'train max acc epoch': [observables.epoch_with_max_train_accuracy],
    'train loss': [loss.logs.get('total_loss', -1)],
    'train loss llh': [loss.logs.get('likelihood', -1)],
    'train loss vp': [loss.logs.get('variational_posterior', -1)],
    'train loss pr': [loss.logs.get('prior', -1)],
    'val accuracy': [observables.logs['val_accuracy']],
    'val vr': [observables.logs['val_uncertainty_vr']],
    'val pe': [observables.logs['val_uncertainty_pe']],
    'val mi': [observables.logs['val_uncertainty_mi']],
    'risk': [risk],
    'delta': [delta],
    'eval accuracy vr': [eval_acc_vrs],
    'eval accuracy pe': [eval_acc_pes],
    'eval accuracy mi': [eval_acc_mis],
    'eval coverage vr': [eval_coverage_vrs],
    'eval coverage pe': [eval_coverage_pes],
    'eval coverage mi': [eval_coverage_mis],
    'seen uncertainty vr': [eval_vr],
    'seen uncertainty pe': [eval_pe],
    'seen uncertainty mi': [eval_mi],
    'threshold vr': [threshold_vr],
    'threshold pe': [threshold_pe],
    'threshold mi': [threshold_mi],
    'true labels': [true_eval_labels],
    'eval preds': [eval_preds],
})

convert_df_to_cpu(res)

save_to_file(loss, './output/loss.pkl')
save_to_file(observables, './output/TrainingLogs.pkl')
torch.save(all_outputs_eval, './output/softmax_outputs.pt')
# torch.save(res, './output/results.pt')
res.to_pickle('./output/results.pkl')
torch.save(bay_net.state_dict(), './output/final_weights.pt')
