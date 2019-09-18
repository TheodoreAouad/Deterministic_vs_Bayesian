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
from src.dataset_manager.get_data import get_mnist, get_cifar10, get_random

parser = argparse.ArgumentParser()
parser.add_argument('--save_loss', help='whether to save the evolution of the loss during training. '
                                        'Note: the tensorboard is not affected.', type=bool)
parser.add_argument('--save_observables', help='whether to save the evolution of the observables during training. '
                                               'Note: the tensorboard is not affected.', type=bool)
parser.add_argument('--save_outputs', help='whether to save the soft max outputs', type=bool)
parser.add_argument('--type_of_unseen', help='configuration set for unseen dataset.',
                    choices=['random', 'unseen_classes', 'unseen_dataset', ], type=str)
parser.add_argument('--trainset', help='dataset on which we train', choices=['mnist', 'cifar10'], type=str)
parser.add_argument('--unseen_evalset', help='unseen dataset to test uncertainty',
                    choices=['mnist', 'cifar10', 'omniglot'], type=str)
parser.add_argument('--split_labels', help='up to which label the training goes',
                    type=int, default=10)
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
parser.add_argument('--ratio_unseen', help='ratio of the evaluation data that is unseen in train', type=float)

args = parser.parse_args()
save_to_file(vars(args), './output/arguments.pkl')

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
stds_prior = (std_prior, std_prior)
delta = args.delta
ratio_unseen = args.ratio_unseen
risks = np.linspace(0.2, 0.5, 50)
res = pd.DataFrame()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

# Defining training set
if trainset == 'mnist':
    get_trainset = get_mnist
    dim_input = 28
    dim_channels = 1
if trainset == 'cifar10':
    get_trainset = get_cifar10
    dim_input = 32
    dim_channels = 3

# Defining training labels
if type_of_unseen != 'unseen_classes':
    split_labels = 10

trainloader_seen, valloader_seen, evalloader_seen = get_trainset(
    train_labels=range(split_labels),
    eval_labels=range(split_labels),
    batch_size=batch_size,
)

# Defining unseen evaluation set
if type_of_unseen == 'random':
    _, _, evalloader_unseen = get_random(number_of_channels=dim_channels, img_dim=dim_input, number_of_classes=10)
if type_of_unseen == 'unseen_classes':
    _, _, evalloader_unseen = get_trainset(train_labels=(), eval_labels=range(split_labels, 10, ), )
    res['split_labels'] = split_labels
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


seed_model = set_and_print_random_seed()
bay_net = GaussianClassifier(rho=rho, stds_prior=stds_prior, dim_input=dim_input, number_of_classes=10, dim_channels=dim_channels)
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
# Train
train_bayesian_modular(
    bay_net,
    optimizer,
    loss,
    observables,
    number_of_tests=number_of_tests,
    number_of_epochs=epoch,
    trainloader=trainloader_seen,
    # valloader=valloader,
    # output_dir_tensorboard='./output',
    device=device,
    verbose=True,
)

# Get uncertainty on train
true_labels_train, all_outputs_train = eval_bayesian(
    bay_net,
    trainloader_seen,
    return_accuracy=False,
    number_of_tests=number_of_tests,
    device=device,
)
train_vr, train_pe, train_mi = get_all_uncertainty_measures(all_outputs_train)

# Get uncertainty on seen
true_seen_labels, all_outputs_seen = eval_bayesian(
    bay_net,
    evalloader_seen,
    return_accuracy=False,
    number_of_tests=number_of_tests,
    device=device,
)
# eval_vr_seen, eval_pe_seen, eval_mi_seen = get_all_uncertainty_measures(all_outputs_seen)

# Get uncertainty on unseen
_, all_outputs_unseen = eval_bayesian(
    bay_net,
    evalloader_unseen,
    number_of_tests=number_of_tests,
    device=device,
)
# eval_vr_unseen, eval_pe_unseen, eval_mi_unseen = get_all_uncertainty_measures(all_outputs_unseen)

# Mixing up seen and unseen
idx = int(len(evalloader_seen.dataset) * ratio_unseen / (1 - ratio_unseen))
all_outputs_eval = torch.cat((all_outputs_seen, all_outputs_unseen[:, :idx, :]), 1)
true_eval_labels = torch.cat((true_seen_labels, -1 + torch.zeros(min(idx, all_outputs_unseen.size(1)))))
eval_vr, eval_pe, eval_mi = get_all_uncertainty_measures(all_outputs_eval)

# Computing predictions
eval_preds = get_predictions_from_multiple_tests(all_outputs_eval)
correct_preds = (eval_preds.float() == true_eval_labels.float()).float()

eval_acc_vrs = []
eval_acc_pes = []
eval_acc_mis = []
eval_coverage_vrs = []
eval_coverage_pes = []
eval_coverage_mis = []
thresholds_vr = []
thresholds_pe = []
thresholds_mi = []

failed = []
for risk in tqdm(risks):
    try:
        threshold_vr, threshold_pe, threshold_mi = get_selection_threshold_all_unc(
            true_labels_train,
            all_outputs_train,
            risk,
            delta,
            (train_vr, train_pe, train_mi),
        )
    except:
        print(f'{risk} failed!')
        failed.append(risk)
        continue

    eval_acc_vrs.append(correct_preds[-eval_vr >= threshold_vr].mean().item())
    eval_acc_pes.append(correct_preds[-eval_pe >= threshold_pe].mean().item())
    eval_acc_mis.append(correct_preds[-eval_mi >= threshold_mi].mean().item())
    eval_coverage_vrs.append((correct_preds[-eval_vr >= threshold_vr].sum()/correct_preds.size(0)).item())
    eval_coverage_pes.append((correct_preds[-eval_pe >= threshold_pe].sum()/correct_preds.size(0)).item())
    eval_coverage_mis.append((correct_preds[-eval_mi >= threshold_mi].sum()/correct_preds.size(0)).item())
    thresholds_vr.append(threshold_vr)
    thresholds_pe.append(threshold_pe)
    thresholds_mi.append(threshold_mi)



res = pd.DataFrame.from_dict({
    'type_of_unseen': [type_of_unseen],
    'trainset': [trainset],
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
    # 'val accuracy': [observables.logs['val_accuracy']],
    # 'val vr': [observables.logs['val_uncertainty_vr']],
    # 'val pe': [observables.logs['val_uncertainty_pe']],
    # 'val mi': [observables.logs['val_uncertainty_mi']],
    'risk': [risks],
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
    'threshold vr': [thresholds_vr],
    'threshold pe': [thresholds_pe],
    'threshold mi': [thresholds_mi],
    'true labels': [true_eval_labels],
    'eval preds': [eval_preds],
    'ratio_unseen': [ratio_unseen],
    'failed': [failed],
})

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
