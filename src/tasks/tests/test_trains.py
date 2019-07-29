import filecmp
import os
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.bayesian_models.gaussian_classifiers import GaussianClassifierMNIST
from src.tasks.trains import train, train_bayesian, train_bayesian_refactored
from src.utils import set_and_print_random_seed, get_file_path_in_dir


class RandomTrainloader:

    def __init__(self, number_of_batches=2, batch_size=2, number_of_channels=1, img_dim=28, number_of_classes=10):
        self.batch_size = batch_size
        self.number_of_classes = number_of_classes
        self.dataset = torch.rand((number_of_batches, batch_size, number_of_channels, img_dim, img_dim))
        self.labels = torch.randint(0, number_of_classes, (number_of_batches, batch_size))

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

    def __len__(self):
        return len(self.dataset)


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
        randomloader = RandomTrainloader()
        device = "cpu"
        det_net = GaussianClassifierMNIST("determinist", (0, 0), (1, 1), number_of_classes=10)
        det_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(det_net.parameters())

        train(det_net, optimizer, criterion, 1, trainloader=randomloader, device=device, verbose=False)

    @staticmethod
    def test_training_with_validation():
        randomloader = RandomTrainloader()
        randomloaderval = RandomTrainloader()
        device = "cpu"
        det_net = GaussianClassifierMNIST("determinist", (0, 0), (1, 1), number_of_classes=10)
        det_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(det_net.parameters())

        train(det_net, optimizer, criterion, 1, trainloader=randomloader, valloader=randomloaderval,
              device=device, verbose=False)


class TestTrainBayesian:

    @staticmethod
    def test_training_no_validation():
        randomloader = RandomTrainloader()
        device = "cpu"
        bay_net = GaussianClassifierMNIST(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bay_net.parameters())

        train_bayesian(bay_net, optimizer, criterion, 1, trainloader=randomloader, device=device, verbose=False)

    @staticmethod
    def test_training_with_validation():
        randomloader = RandomTrainloader()
        randomloaderval = RandomTrainloader()
        device = "cpu"
        bay_net = GaussianClassifierMNIST(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bay_net.parameters())

        train_bayesian(bay_net, optimizer, criterion, 1, trainloader=randomloader, valloader=randomloaderval,
                       number_of_tests=2, device=device, verbose=False)


class TestBayesianRefactored:

    @staticmethod
    def test_identity_between_old_and_old_train_bayesian():
        device = 'cpu'
        randomloader = RandomTrainloader()

        seed1 = set_and_print_random_seed()
        bay_net = GaussianClassifierMNIST(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bay_net.parameters())
        output_old = train_bayesian(bay_net, optimizer, criterion, 1, trainloader=randomloader,
                                    device=device, verbose=False)

        set_and_print_random_seed(seed1)
        bay_net = GaussianClassifierMNIST(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bay_net.parameters())
        output_new = train_bayesian(bay_net, optimizer, criterion, 1, trainloader=randomloader,
                                    device=device, verbose=False)

        assert output_old == output_new

    @staticmethod
    def test_identity_between_new_and_old_train_bayesian():
        device = 'cpu'
        randomloader = RandomTrainloader()

        seed1 = set_and_print_random_seed()
        bay_net = GaussianClassifierMNIST(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bay_net.parameters())
        output_old = train_bayesian(bay_net, optimizer, criterion, 1, trainloader=randomloader,
                                    device=device, verbose=False)

        set_and_print_random_seed(seed1)
        bay_net = GaussianClassifierMNIST(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bay_net.parameters())
        output_new = train_bayesian_refactored(bay_net, optimizer, criterion, 1, trainloader=randomloader,
                                    device=device, verbose=False)

        assert output_old == output_new

    @staticmethod
    def test_identity_between_new_and_old_train_bayesian_with_tensorboard(data_folder_teardown):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        output_dir_tensorboard_old = os.path.join(dir_path, 'tensorboard_results_old')
        output_dir_tensorboard_new = os.path.join(dir_path, 'tensorboard_results_new')
        device = 'cpu'

        seed1 = set_and_print_random_seed()
        randomloader = RandomTrainloader()
        bay_net = GaussianClassifierMNIST(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bay_net.parameters())
        output_old = train_bayesian(bay_net, optimizer, criterion, 1, trainloader=randomloader,
                                    output_dir_tensorboard=output_dir_tensorboard_old, device=device, verbose=False)

        set_and_print_random_seed(seed1)
        randomloader = RandomTrainloader()
        bay_net = GaussianClassifierMNIST(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bay_net.parameters())
        output_new = train_bayesian(bay_net, optimizer, criterion, 1, trainloader=randomloader,
                                    output_dir_tensorboard=output_dir_tensorboard_new, device=device, verbose=False)

        assert output_old == output_new

    @staticmethod
    def test_identity_between_new_and_old_train_bayesian_with_validation(data_folder_teardown):
        device = 'cpu'
        randomloader = RandomTrainloader()
        randomloaderval = RandomTrainloader()

        seed1 = set_and_print_random_seed()
        bay_net = GaussianClassifierMNIST(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bay_net.parameters())
        output_old = train_bayesian(bay_net,
                                    optimizer,
                                    criterion,
                                    2,
                                    trainloader=randomloader,
                                    valloader=randomloaderval,
                                    device=device,
                                    verbose=False)

        set_and_print_random_seed(seed1)
        bay_net = GaussianClassifierMNIST(-3, (0, 0), (1, 1), number_of_classes=10)
        bay_net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bay_net.parameters())
        output_new = train_bayesian_refactored(bay_net,
                                               optimizer,
                                               criterion,
                                               2,
                                               trainloader=randomloader,
                                               valloader=randomloaderval,
                                               device=device,
                                               verbose=False)

        for old, new in zip(output_old, output_new):
            if type(old) in [int, float]:
                assert old == new
            elif type(old[0][0]) != torch.Tensor:
                assert old == new
            else:
                for o, n in zip(old,new):
                    assert torch.sum(torch.stack(o) - torch.stack(n)) == 0
