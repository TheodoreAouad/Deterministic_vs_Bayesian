"""

Comparison, first batch only.
    p-values: acc: 0.04130769292470323
              vr: 1.1948968793878407e-06
              pe: 2.6380732753719537e-08
              mi: 1.5662983825396675e-09

Comparison, first image only.
    p-values: acc: 1.4266228896090916e-07)
              vr:  8.062383338395684e-16)
              pe:  1.0730306739099891e-19)
              mi:  2.0144309172705817e-19)


# Tester loi du chi2
"""
import argparse
from time import time

import pandas as pd
import torch
from scipy.stats import ttest_ind, chisquare
import numpy as np
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset_manager.get_data import get_mnist
from src.loggers.losses.base_loss import BaseLoss
from src.loggers.losses.bbb_loss import BBBLoss
from src.loggers.observables import AccuracyAndUncertainty
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.evals import eval_bayesian
from src.tasks.trains import train_bayesian_modular
from src.uncertainty_measures import get_all_uncertainty_measures_bayesian

parser = argparse.ArgumentParser()
parser.add_argument('--nb_of_runs', help='nb of runs for stat test', type=int)
parser.add_argument('--nb_of_epochs', help='nb of epochs for training', type=int)
parser.add_argument('--nb_of_tests', help='nb of tests for evaluation', type=int)
parser.add_argument('--rho', help='variance for bbb', type=float)

args = parser.parse_args()

##### TO CHANGE ######
nb_of_runs = args.nb_of_runs
nb_of_epochs = args.nb_of_epochs
nb_of_tests = args.nb_of_tests
rho = args.rho
######################

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

trainloader, valloader, evalloader = get_mnist(batch_size=32)
criterion = nn.CrossEntropyLoss()


def train_bayesian_modular_with_one_different(
        model,
        optimizer,
        loss,
        observables,
        number_of_epochs,
        trainloader,
        valloader=None,
        number_of_tests=10,
        output_dir_tensorboard=None,
        output_dir_results=None,
        device='cpu',
        verbose=False,
):
    """
    Train Bayesian with modular arguments
    Args:
        model (torch.nn.Module child): model we want to train
        optimizer (torch.optim optimizer): how do we update the weights
        loss (src.loggers.losses.base_loss.BaseLoss child): loss object
        observables (src.loggers.observables.Observables): observable object
        number_of_epochs (int): how long do we train our model
        trainloader (torch.utils.data.dataloader.DataLoader): dataloader of train set
        valloader (torch.utils.data.dataloader.DataLoader): dataloader of validation set
        number_of_tests (int): number of tests to perform during validation evaluation
        output_dir_results (str): output directory in which to save the results (NOT IMPLEMENTED)
        output_dir_tensorboard (str): output directory in which to save the tensorboard
        device (torch.device || str): cpu or gpu
        verbose (Bool): print training steps or not
    Returns
        NOT IMPLEMENTED YET
    """
    start_time = time()
    number_of_batch = len(trainloader)
    interval = max(number_of_batch // 10, 1)

    for logger in [loss, observables]:
        logger.set_number_of_epoch(number_of_epochs)
        logger.set_number_of_batch(number_of_batch)
        logger.init_tensorboard_writer(output_dir_tensorboard)
        logger.init_results_writer(output_dir_results)

    model.train()
    for epoch in range(number_of_epochs):
        loss.set_current_epoch(epoch)
        observables.set_current_epoch(epoch)

        loss.set_current_batch_idx(-1)

        idx_of_first_img = np.random.choice(len(trainloader.dataset))
        first_img = trainloader.dataset.data[idx_of_first_img].to(device)
        first_label = trainloader.dataset.targets[idx_of_first_img].to(device)
        optimizer.zero_grad()
        output = model(first_img.unsqueeze(0).unsqueeze(0).float())
        loss.compute(output, first_label.unsqueeze(0))
        loss.backward()
        optimizer.step()

        for batch_idx, data in enumerate(trainloader):
            loss.set_current_batch_idx(batch_idx)
            observables.set_current_batch_idx(batch_idx)

            inputs, labels = [x.to(device) for x in data]
            optimizer.zero_grad()
            outputs = model(inputs)

            observables.compute_train_on_batch(outputs, labels)
            loss.compute(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % interval == interval - 1:
                if valloader is not None:
                    val_acc, val_outputs = eval_bayesian(model, valloader, number_of_tests=number_of_tests,
                                                         device=device, verbose=verbose)
                    observables.compute_val(val_acc, val_outputs)

                if verbose:
                    print('======================================')
                    print(f'Epoch [{epoch + 1}/{number_of_epochs}]. Batch [{batch_idx}/{number_of_batch}].')
                    loss.show()
                    observables.show()
                    print(f'Time Elapsed: {round(time() - start_time)} s')

                loss.write_tensorboard()
                observables.write_tensorboard()
                if output_dir_results is not None:
                    loss.write_results()
                    observables.write_results()

        observables.compute_train_on_epoch(model, trainloader, device)

    loss.close_writer()
    observables.close_writer()
    if verbose:
        print('Finished Training')

    return loss.results(), observables.results()


def do_train_ce(verbose=True):
    bay_net = GaussianClassifier(rho, number_of_classes=10)
    bay_net.to(device)
    criterion = nn.CrossEntropyLoss()
    loss_bbb = BaseLoss(criterion)
    optimizer = optim.Adam(bay_net.parameters())
    observables = AccuracyAndUncertainty()
    train_bayesian_modular(
        bay_net,
        optimizer,
        loss_bbb,
        observables,
        number_of_epochs=nb_of_epochs,
        trainloader=trainloader,
        device=device,
        verbose=verbose,
    )

    return eval_bayesian(bay_net, evalloader, nb_of_tests, device=device, verbose=verbose)


def do_train_dirac_batch_same_size(verbose=True):
    def dirac(cur_batch, nb_of_batch):
        if cur_batch == 0:
            return 1
        else:
            return 0

    bay_net = GaussianClassifier(rho, number_of_classes=10)
    bay_net.to(device)
    loss_bbb = BBBLoss(bay_net, criterion, dirac)
    optimizer = optim.Adam(bay_net.parameters())
    observables = AccuracyAndUncertainty()
    train_bayesian_modular(
        bay_net,
        optimizer,
        loss_bbb,
        observables,
        number_of_epochs=nb_of_epochs,
        trainloader=trainloader,
        device=device,
        verbose=verbose,
    )

    return eval_bayesian(bay_net, evalloader, nb_of_tests, device=device, verbose=verbose)


def do_train_dirac_one_image(verbose=True):
    def dirac(cur_batch, nb_of_batch):
        if cur_batch == -1:
            return 1
        else:
            return 0

    bay_net = GaussianClassifier(rho, number_of_classes=10)
    bay_net.to(device)
    loss_bbb = BBBLoss(bay_net, criterion, dirac)
    optimizer = optim.Adam(bay_net.parameters())
    observables = AccuracyAndUncertainty()
    train_bayesian_modular_with_one_different(
        bay_net,
        optimizer,
        loss_bbb,
        observables,
        number_of_epochs=nb_of_epochs,
        trainloader=trainloader,
        device=device,
        verbose=verbose,
    )

    return eval_bayesian(bay_net, evalloader, nb_of_tests, device=device, verbose=verbose)


verbose = False
accs1 = np.zeros(nb_of_runs)
accs2 = np.zeros(nb_of_runs)
uncs1 = np.zeros((nb_of_runs, 3,))
uncs2 = np.zeros((nb_of_runs, 3,))

for i in tqdm(range(nb_of_runs)):
    eval_acc1, eval_output1 = do_train_dirac_one_image(verbose)
    eval_acc2, eval_output2 = do_train_ce(verbose)
    accs1[i] = eval_acc1
    accs2[i] = eval_acc2
    uncs1[i] = np.array([unc.mean() for unc in get_all_uncertainty_measures_bayesian(eval_output1)])
    uncs2[i] = np.array([unc.mean() for unc in get_all_uncertainty_measures_bayesian(eval_output2)])

# Do T test
pvalues = pd.DataFrame(columns=['acc'] + [f'unc_{i}' for i in range(uncs1.shape[1])], index=['ttest', 'chisquare'])
for name, stat_test in zip(['ttest', 'chisquare'], [ttest_ind, chisquare]):
    pvalues.loc[name, 'acc'] = stat_test(accs1, accs2).pvalue
    for i in range(3):
        print(ttest_ind(uncs1[:, i], uncs2[:, i]))
        pvalues.loc[name, f'unc_{i}'] = stat_test(uncs1[:, i], uncs2[:, i]).pvalue

accs = pd.DataFrame.from_dict({
    'accs1': accs1,
    'accs2': accs2,
})

uncs = pd.concat((
    pd.DataFrame(columns=['vr1', 'pe1', 'mi1'], data=uncs1), pd.DataFrame(columns=['vr2', 'pe2', 'mi2'], data=uncs2)
    ),
    axis=1)
accs_and_uncs = pd.concat((accs, uncs), axis=1)

pvalues.to_csv('./output/pvalues.csv')
pvalues.to_pickle('./output/pvalues.pkl')
accs_and_uncs.to_csv('./output/accs_and_uncs.csv')
accs_and_uncs.to_pickle('./output/accs_and_uncs.pkl')
