import os
from math import ceil

import numpy as np

import torch
from torchvision.datasets import MNIST, CIFAR10


class MNISTSpecificLabels(MNIST):

    def __init__(self, root, labels=range(10), train=True, split=(0, 1), transform=None, target_transform=None,
                 download=False, ):
        """

        Args:
            root (str): path to directory where we want to download data
            labels (list || tuple || array): labels we want to keep
            train (bool): train data or test data
            split (tuple): the framing of the data we want to keep. Ex: (0.2, 0.5) gives data between the 20% index
                           and 50% index.
            transform (torchvision.transforms.transforms): transformation to apply to the data
            target_transform (torchvision.transforms.transforms): transformation to apply to the target
            download (bool): whether we want to download the data or not
        """
        super(MNISTSpecificLabels, self).__init__(root=root, train=train, transform=transform,
                                                  target_transform=target_transform, download=download)
        labels_to_keep = torch.from_numpy(np.isin(self.targets, labels))
        self.data, self.targets = self.data[labels_to_keep], self.targets[labels_to_keep]
        data_size = self.data.size(0)
        random_shuffle = torch.randperm(data_size)
        self.data, self.targets = self.data[random_shuffle], self.targets[random_shuffle]
        if type(split) == tuple:
            first_index = max(ceil(split[0] * data_size), 0)
            last_index = min(ceil(split[1] * data_size), data_size)
            self.data, self.targets = self.data[first_index:last_index], self.targets[first_index:last_index]
        elif type(split) == int:
            self.data, self.targets = self.data[:split], self.targets[:split]

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')


class CIFAR10SpecificLabels(CIFAR10):

    def __init__(self, root, labels=range(10), train=True, split=(0, 1), transform=None, target_transform=None,
                 download=False, ):
        """

        Args:
            root (str): path to directory where we want to download data
            labels (list || tuple || array): labels we want to keep
            train (bool): train data or test data
            split (tuple): the framing of the data we want to keep. Ex: (0.2, 0.5) gives data between the 20% index
                           and 50% index.
            transform (torchvision.transforms.transforms): transformation to apply to the data
            target_transform (torchvision.transforms.transforms): transformation to apply to the target
            download (bool): whether we want to download the data or not
        """
        super(CIFAR10SpecificLabels, self).__init__(root=root, train=train, transform=transform,
                                                    target_transform=target_transform, download=download)
        labels_to_keep = torch.from_numpy(np.isin(self.targets, labels))
        self.data, self.targets = (self.data[labels_to_keep],
                                   torch.tensor(np.array(self.targets)[labels_to_keep]))
        data_size = self.data.shape[0]
        random_shuffle = torch.randperm(data_size)
        self.data, self.targets = self.data[random_shuffle], self.targets[random_shuffle]
        if type(split) == tuple:
            first_index = max(ceil(split[0] * data_size), 0)
            last_index = min(ceil(split[1] * data_size), data_size)
            self.data, self.targets = self.data[first_index:last_index], self.targets[first_index:last_index]
        elif type(split) == int:
            self.data, self.targets = self.data[:split], self.targets[:split]

    # @property
    # def raw_folder(self):
    #     return os.path.join(self.root, 'CIFAR10', 'raw')
    #
    # @property
    # def processed_folder(self):
    #     return os.path.join(self.root, 'CIFAR10', 'processed')
