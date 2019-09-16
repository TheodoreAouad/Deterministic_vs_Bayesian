from math import log, exp
import argparse

import pandas as pd
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from src.loggers.losses.base_loss import BaseLoss
from src.loggers.losses.bbb_loss import BBBLoss
from src.loggers.observables import AccuracyAndUncertainty
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.trains import train_bayesian_modular, uniform
from src.tasks.evals import eval_bayesian, eval_random
from src.uncertainty_measures import get_all_uncertainty_measures
from src.utils import set_and_print_random_seed, save_to_file, convert_df_to_cpu
from src.dataset_manager.get_data import get_mnist

parser = argparse.ArgumentParser()
parser.add_argument('--trainset', help='dataset on which we train', choices=['mnist','cifar10'], type=str)
parser.add_argument('--rho', help='variable symbolizing the variance. std = log(1+exp(rho))',
                    type=float, default=-5)
parser.add_argument('--epoch', help='number of times we train the model on the same data',
                    type=int, default=3)
parser.add_argument('--batch_size', help='number of batches to split the data into',
                    type=int, default=32)
parser.add_argument('--number_of_tests', help='number of evaluations to perform for each each image to check for '
                                              'uncertainty', type=int, default=10)
parser.add_argument('--loss_type', help='which loss to use', choices=['exp', 'uniform', 'criterion'], type=str,
                    default='uniform')
parser.add_argument('--std_prior', help='the standard deviation of the prior', type=float, default=1)
parser.add_argument('--split_train', help='the portion of training data we take', type=int)

args = parser.parse_args()
save_to_file(vars(args), './output/arguments.pkl')

rho = args.rho
epoch = args.epoch
batch_size = args.batch_size
number_of_tests = args.number_of_tests
loss_type = args.loss_type
std_prior = args.std_prior
split_train = args.split_train
stds_prior = (std_prior, std_prior)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

trainloader, valloader, evalloader = get_mnist(train_labels=range(10), eval_labels=range(10), split_train=split_train,
                                               batch_size=batch_size)

if trainset == 'mnist':
    trainloader, valloader, evalloader = get_mnist(train_labels=range(10), eval_labels=range(10),
                                                   batch_size=batch_size)
    dim_input = 28
    dim_channels = 1
if trainset == 'cifar10':
    trainloader, evalloader = get_cifar10(batch_size=batch_size)
    dim_input = 32
    dim_channels = 3

seed_model = set_and_print_random_seed()
bay_net = GaussianClassifier(rho=rho, stds_prior=stds_prior, dim_input=dim_input, number_of_classes=10, dim_channels=dim_channels)
bay_net.to(device)
criterion = CrossEntropyLoss()

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

eval_acc, all_outputs_eval = eval_bayesian(bay_net, evalloader, number_of_tests=number_of_tests, device=device)
eval_vr, eval_predictive_entropy, eval_mi = get_all_uncertainty_measures(all_outputs_eval)

output_random, seed = eval_random(bay_net, batch_size=32, img_channels=1, img_dim=28, number_of_tests=number_of_tests,
                                  device=device)

random_vr, random_predictive_entropy, random_mi = get_all_uncertainty_measures(output_random)

print(f'Eval acc: {round(100 * eval_acc, 2)} %, '
      f'Variation-Ratio:{eval_vr.mean()}, '
      f'Predictive Entropy:{eval_predictive_entropy.mean()}, '
      f'Mutual Information:{eval_mi.mean()}')
print(f'Random: '
      f'Variation-Ratio:{random_vr.mean()}, '
      f'Predictive Entropy:{random_predictive_entropy.mean()}, '
      f'Mutual Information:{random_mi.mean()}')
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
    'val predictive entropy': [observables.logs['val_uncertainty_pe']],
    'val mi': [observables.logs['val_uncertainty_mi']],
    'eval accuracy': [eval_acc],
    'seen uncertainty vr': [eval_vr],
    'seen uncertainty predictive entropy': [eval_predictive_entropy],
    'seen uncertainty mi': [eval_mi],
    'random uncertainty vr': [random_vr],
    'random uncertainty predictive entropy': [random_predictive_entropy],
    'random uncertainty mi': [random_mi],
})

convert_df_to_cpu(res)

save_to_file(loss, './output/loss.pkl')
save_to_file(observables, './output/TrainingLogs.pkl')
torch.save(output_random, './output/random_outputs.pt')
torch.save(all_outputs_eval, './output/softmax_outputs.pt')
# torch.save(res, './output/results.pt')
res.to_pickle('./output/results.pkl')
torch.save(bay_net.state_dict(), './output/final_weights.pt')
