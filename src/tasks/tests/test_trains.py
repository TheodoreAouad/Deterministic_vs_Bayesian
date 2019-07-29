import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.bayesian_models.gaussian_classifiers import GaussianClassifierMNIST
from src.tasks.trains import train, train_bayesian


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
