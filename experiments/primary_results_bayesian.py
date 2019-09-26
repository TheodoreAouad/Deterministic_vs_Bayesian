from math import log, exp
import argparse

import pandas as pd
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torchvision.transforms as transforms

from src.loggers.losses.base_loss import BaseLoss
from src.loggers.losses.bbb_loss import BBBLoss
from src.loggers.observables import AccuracyAndUncertainty
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.trains import train_bayesian_modular, uniform
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_all_uncertainty_measures, get_all_uncertainty_measures_not_bayesian
from src.utils import set_and_print_random_seed, save_to_file, convert_df_to_cpu
from src.dataset_manager.get_data import get_mnist, get_cifar10, get_random, get_omniglot

parser = argparse.ArgumentParser()
parser.add_argument('--save_loss', help='whether to save the evolution of the loss during training. '
                                        'Note: the tensorboard is not affected.', action='store_true')
parser.add_argument('--save_observables', help='whether to save the evolution of the observables during training. '
                                               'Note: the tensorboard is not affected.', action='store_true')
parser.add_argument('--save_outputs', help='whether to save the soft max outputs', action='store_true')
parser.add_argument('--type_of_unseen', help='configuration set for unseen dataset.',
                    choices=['random', 'unseen_classes', 'unseen_dataset', ], type=str)
parser.add_argument('--trainset', help='dataset on which we train', choices=['mnist', 'cifar10'], type=str)
parser.add_argument('--unseen_evalset', help='unseen dataset to test uncertainty',
                    choices=['mnist', 'cifar10', 'omniglot'], type=str)
parser.add_argument('--split_labels', help='up to which label the training goes',
                    type=int, default=10)
parser.add_argument('--determinist', help='put this variable if you want the model to be determinist'
                    , action='store_true')
parser.add_argument('--rho', help='variable symbolizing the variance. std = log(1+exp(rho))',
                    type=float, default=-5)
parser.add_argument('--epoch', help='number of times we train the model on the same data',
                    type=int, default=3)
parser.add_argument('--batch_size', help='number of batches to split the data into',
                    type=int, default=32)
parser.add_argument('--number_of_tests', help='number of evaluations to perform for each each image to check for '
                                              'uncertainty', type=int, default=10)
parser.add_argument('--loss_type', help='which loss to use', choices=['uniform', 'exp', 'criterion'], type=str,
                    default='exp')
parser.add_argument('--std_prior', help='the standard deviation of the prior', type=float, default=1)
parser.add_argument('--split_train', help='the portion of training data we take', type=int)

args = parser.parse_args()
arguments = vars(args)

trainset = args.trainset
unseen_evalset = args.unseen_evalset
type_of_unseen = args.type_of_unseen
split_labels = args.split_labels
rho = args.rho
epoch = args.epoch
batch_size = args.batch_size
number_of_tests = args.number_of_tests
loss_type = args.loss_type
std_prior = args.std_prior
if type_of_unseen == 'unseen_classes':
    split_train = args.split_train
else:
    split_train = 10
    arguments['split_train'] = 10
stds_prior = (std_prior, std_prior)

save_to_file(arguments, './output/arguments.pkl')

if args.determinist:
    rho = 'determinist'
    number_of_tests = 1

res = pd.DataFrame()


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

def get_evalloader_unseen(arguments):
    type_of_unseen = arguments['type_of_unseen']
    dim_channels = arguments['dim_channels']
    dim_input = arguments['dim_input']
    split_labels = arguments['split_labels']
    if type_of_unseen == 'random':
        _, _, evalloader_unseen = get_random(number_of_channels=dim_channels, img_dim=dim_input, number_of_classes=10)
    if type_of_unseen == 'unseen_classes':
        _, _, evalloader_unseen = get_trainset(train_labels=(), eval_labels=range(split_labels, 10, ), )
    if type_of_unseen == 'unseen_dataset':
        res['unseen_dataset'] = unseen_evalset
        assert trainset != unseen_evalset, 'Train Set must be different from Unseen Test Set'
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

# Defining training set
if trainset == 'mnist':
    get_trainset = get_mnist
    arguments['dim_input'] = 28
    arguments['dim_channels'] = 1
if trainset == 'cifar10':
    get_trainset = get_cifar10
    arguments['dim_input'] = 32
    arguments['dim_channels'] = 3

# Defining training labels
if type_of_unseen != 'unseen_classes':
    split_labels = 10

trainloader_seen, valloader_seen, evalloader_seen = get_trainset(
    train_labels=range(split_labels),
    eval_labels=range(split_labels),
    batch_size=batch_size,
)

# Defining unseen evaluation set
evalloader_unseen = get_evalloader_unseen(arguments)

# Defining model
seed_model = set_and_print_random_seed()
bay_net = GaussianClassifier(
    rho=rho,
    stds_prior=stds_prior,
    dim_input=arguments['dim_input'],
    number_of_classes=10,
    dim_channels=arguments['dim_channels'],
)
bay_net.to(device)
# Defining loss
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
# Defining optimizer
optimizer = optim.Adam(bay_net.parameters())
observables = AccuracyAndUncertainty()
# Train
train_bayesian_modular(
    bay_net,
    optimizer,
    loss,
    observables,
    number_of_tests=number_of_tests,
    number_of_epochs=epoch,
    trainloader=trainloader_seen,
    valloader=valloader_seen,
    output_dir_tensorboard='./output',
    device=device,
    verbose=True,
)

# Evaluation on seen test set
eval_acc, all_outputs_eval = eval_bayesian(bay_net, evalloader_seen, number_of_tests=number_of_tests, device=device, )

# Evaluation on unseen test set
_, all_outputs_unseen = eval_bayesian(bay_net, evalloader_unseen, number_of_tests=number_of_tests, device=device)

res = pd.concat((res, pd.DataFrame.from_dict({
    'trainset': [trainset],
    'split_labels': [split_labels],
    'type_of_unseen': [type_of_unseen],
    'loss_type': [loss_type],
    'number_of_epochs': [epoch],
    'batch_size': [batch_size],
    'nb_of_data': [len(trainloader_seen.dataset)],
    'number_of_tests': [number_of_tests],
    'seed_model': [seed_model],
    'stds_prior': [std_prior],
    'rho': [rho],
    'train_accuracy': [observables.logs['train_accuracy_on_epoch']],
    'train_max_acc': [observables.max_train_accuracy_on_epoch],
    'train_max_acc_epoch': [observables.epoch_with_max_train_accuracy],
    'train_loss': [loss.logs.get('total_loss', -1)],
    'train_loss_llh': [loss.logs.get('likelihood', -1)],
    'train_loss_vp': [loss.logs.get('variational_posterior', -1)],
    'train_loss_pr': [loss.logs.get('prior', -1)],
    'val_accuracy': [observables.logs['val_accuracy']],
    'val_vr': [observables.logs['val_uncertainty_vr']],
    'val_pe': [observables.logs['val_uncertainty_pe']],
    'val_mi': [observables.logs['val_uncertainty_mi']],
    'eval_accuracy': [eval_acc],
})), axis=1)

if args.determinist:
    eval_us, eval_pe = get_all_uncertainty_measures_not_bayesian(all_outputs_eval)
    unseen_us, unseen_pe = get_all_uncertainty_measures_not_bayesian(all_outputs_unseen)
    print(f'Eval acc: {round(100 * eval_acc, 2)} %, '
          f'Uncertainty Softmax:{eval_us.mean()}, '
          f'Predictive Entropy:{eval_pe.mean()}, ')
    print(f'Unseen: '
          f'Uncertainty Softmax:{unseen_us.mean()}, '
          f'Predictive Entropy:{unseen_pe.mean()}, ')
    res = pd.concat((res, pd.DataFrame.from_dict({
        'seen_uncertainty_us': [eval_us],
        'seen_uncertainty_pe': [eval_pe],
        'unseen_uncertainty_us': [unseen_us],
        'unseen_uncertainty_pe': [unseen_pe],
    })), axis=1)
else:
    eval_vr, eval_pe, eval_mi = get_all_uncertainty_measures(all_outputs_eval)
    unseen_vr, unseen_pe, unseen_mi = get_all_uncertainty_measures(all_outputs_unseen)
    print(f'Eval acc: {round(100 * eval_acc, 2)} %, '
          f'Variation-Ratio:{eval_vr.mean()}, '
          f'Predictive Entropy:{eval_pe.mean()}, '
          f'Mutual Information:{eval_mi.mean()}')
    print(f'Unseen: '
          f'Variation-Ratio:{unseen_vr.mean()}, '
          f'Predictive Entropy:{unseen_pe.mean()}, '
          f'Mutual Information:{unseen_mi.mean()}')
    res = pd.concat((res, pd.DataFrame.from_dict({
        'sigma_initial': [log(1 + exp(rho))],
        'seen_uncertainty_vr': [eval_vr],
        'seen_uncertainty_pe': [eval_pe],
        'seen_uncertainty_mi': [eval_mi],
        'unseen_uncertainty_vr': [unseen_vr],
        'unseen_uncertainty_pe': [unseen_pe],
        'unseen_uncertainty_mi': [unseen_mi],
    })), axis=1)

convert_df_to_cpu(res)

if args.save_loss:
    save_to_file(loss, './output/loss.pkl')
if args.save_observables:
    save_to_file(observables, './output/TrainingLogs.pkl')
if args.save_outputs:
    torch.save(all_outputs_unseen, './output/unseen_outputs.pt')
    torch.save(all_outputs_eval, './output/seen_outputs.pt')
# torch.save(res, './output/results.pt')
res.to_pickle('./output/results.pkl')
torch.save(bay_net.state_dict(), './output/final_weights.pt')
torch.save(observables.max_weights, './output/best_weights.pt')
