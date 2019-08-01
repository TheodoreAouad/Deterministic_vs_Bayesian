import os

import torch
import torchvision
import torchvision.transforms as transforms

from src.dataset_manager.datasets_creator import MNISTSpecificLabels

transform = transforms.ToTensor()

absolute_path = os.getcwd()
download_path = os.path.join(absolute_path, 'data')


class EmptyLoader:
    def __init__(self):
        self.dataset = []

    def __len__(self):
        return 0

    def __getitem__(self, index):
        self.dataset.__getitem__(index)


def get_mnist(root=download_path, train_labels=range(10), eval_labels=range(10), split_val=0.2, transform=transform,
              batch_size=16, shuffle=True):
    """

    Args:
        root (str): path to the directory where we want to download the data
        train_labels (list || tuple || array): labels we want to keep in the training set
        eval_labels (list || tuple || array): labels we want to keep in the testing set
        split_val (float): the proportion of the evaluation set we use for validation
        transform (torch.transform): which transformation to perform to the data
        batch_size (int): size of the batch
        shuffle (bool): whether or not we shuffle the data. Usually we shuffle the data.

    Returns:

        trainloader: loader of train data
        valloader: loader of val data
        evalloader: loader of eval data

    """
    print(root)
    trainloader, valloader, evalloader = EmptyLoader(), EmptyLoader(), EmptyLoader()
    if len(train_labels) > 0:
        trainset = MNISTSpecificLabels(root=root, labels=train_labels, train=True, transform=transform,
                                       download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

        if split_val > 0:
            valset = MNISTSpecificLabels(root=root, labels=train_labels, train=False, split=(0, split_val),
                                         transform=transform, download=True)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=shuffle)

    if len(eval_labels) > 0:
        evalset = MNISTSpecificLabels(root=root, labels=eval_labels, train=False, split=(split_val, 2),
                                      transform=transform, download=True)
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=shuffle)

    return trainloader, valloader, evalloader


def get_cifar10(transform=transform, batch_size=16, shuffle=True, download=False):
    '''

    Args:
        transform (torch.transform): which transformation to perform to the data
        batch_size (int): size of the batch
        shuffle (bool): whether or not we shuffle the data. Usually we shuffle the data.

    Returns:
        torch.utils.data.dataloader.DataLoader: loader of train data
        torch.utils.data.dataloader.DataLoader: loader of evaluation data

    '''

    absolute_path = os.getcwd()
    download_path = os.path.join(absolute_path, 'data')
    print(download_path)

    trainset = torchvision.datasets.CIFAR10(root=download_path, train=True, transform=transform, download=download)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

    testset = torchvision.datasets.CIFAR10(root=download_path, train=False, transform=transform, download=download)
    evalloader = torch.utils.data.DataLoader(testset, batch_size= batch_size, shuffle=shuffle)

    return trainloader, evalloader


def get_omniglot(root=download_path, transform=transform, batch_size=16, shuffle=True, download=True):
    """

    Args:
        root (str): path to the directory where we want to download the data
        transform (torch.transform): which transformation to perform to the data
        batch_size (int): size of the batch
        shuffle (bool): whether or not we shuffle the data. Usually we shuffle the data.
        download (bool): whether or not we download the data

    Returns:
        torch.utils.data.dataloader.DataLoader: loader of the omniglot dataset

    """

    dataset = torchvision.datasets.Omniglot(root=root, transform=transform, download=download)
    omniglot_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return omniglot_loader
