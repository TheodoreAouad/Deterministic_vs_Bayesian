import os
import shutil

import pytest
import torch.nn as nn
import torch.optim as optim

from src.dataset_manager.get_data import RandomLoader
from src.loggers.losses.base_loss import BaseLoss
from src.loggers.observables import AccuracyAndUncertainty
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.trains import train_bayesian_modular


@pytest.fixture(scope='class')
def data_folder_teardown(request):
    def fin():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        download_path_old = os.path.join(dir_path, 'tensorboard_results_old')
        if os.path.isdir(download_path_old):
            shutil.rmtree(download_path_old)
            print(f'{download_path_old} deleted.')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        download_path_new = os.path.join(dir_path, 'tensorboard_results_new')
        if os.path.isdir(download_path_new):
            shutil.rmtree(download_path_new)
            print(f'{download_path_new} deleted.')

    request.addfinalizer(fin)


class TestTrain:

    @staticmethod
    def test_training_no_validation():
        randomloader = RandomLoader()
        device = "cpu"
        det_net = GaussianClassifier("determinist", (0, 0), (1, 1), number_of_classes=10)
        det_net.to(device)
        criterion = nn.CrossEntropyLoss()
        loss = BaseLoss(criterion)
        observables = AccuracyAndUncertainty()
        optimizer = optim.Adam(det_net.parameters())

        train_bayesian_modular(det_net, optimizer, loss, observables, 1, trainloader=randomloader, number_of_tests=1,
                               device=device, verbose=False)

    @staticmethod
    def test_training_with_validation():
        randomloader = RandomLoader()
        randomloaderval = RandomLoader()
        device = "cpu"
        det_net = GaussianClassifier("determinist", (0, 0), (1, 1), number_of_classes=10)
        det_net.to(device)
        criterion = nn.CrossEntropyLoss()
        loss = BaseLoss(criterion)
        observables = AccuracyAndUncertainty()
        optimizer = optim.Adam(det_net.parameters())

        train_bayesian_modular(det_net, optimizer, loss, observables, 1, trainloader=randomloader,
                               valloader=randomloaderval,
                               number_of_tests=1, device=device, verbose=False)


class TestTrainBayesian:

    @staticmethod
    def test_training_no_validation():
        randomloader = RandomLoader()
        device = "cpu"
        bay_net = GaussianClassifier(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        loss = BaseLoss(criterion)
        observables = AccuracyAndUncertainty()
        optimizer = optim.Adam(bay_net.parameters())

        train_bayesian_modular(bay_net, optimizer, loss, observables, 1, trainloader=randomloader, device=device,
                               number_of_tests=2, verbose=False)

    @staticmethod
    def test_training_with_validation():
        randomloader = RandomLoader()
        randomloaderval = RandomLoader()
        device = "cpu"
        bay_net = GaussianClassifier(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        loss = BaseLoss(criterion)
        observables = AccuracyAndUncertainty()
        optimizer = optim.Adam(bay_net.parameters())

        train_bayesian_modular(bay_net, optimizer, loss, observables, 1, trainloader=randomloader,
                               valloader=randomloaderval,
                               number_of_tests=2, device=device, verbose=False)
