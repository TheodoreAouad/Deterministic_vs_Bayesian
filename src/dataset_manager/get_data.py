import os

import torch
import torchvision
import torchvision.transforms as transforms

from src.dataset_manager.datasets_creator import MNISTSpecificLabels

transform = transforms.ToTensor()

def get_mnist(train_labels=range(10), test_labels=range(10), transform=transform, batch_size=16, shuffle=True):
    '''

    Args:
        transform (torch.transform): which transformation to perform to the data
        batch_size (int): size of the batch
        shuffle (bool): whether or not we shuffle the data. Usually we shuffle the data.

    Returns:

    '''

    absolute_path = os.getcwd()
    download_path = os.path.join(absolute_path, 'data')
    print(download_path)

    trainset = MNISTSpecificLabels(root=download_path, labels=train_labels, train=True, transform=transform,
                                   download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

    testset = MNISTSpecificLabels(root=download_path, labels=test_labels, train=False, transform=transform,
                                   download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    return trainloader, testloader


def get_cifar10(transform=transform, batch_size=16, shuffle=True, download=False):
    '''

    Args:
        transform (torch.transform): which transformation to perform to the data
        batch_size (int): size of the batch
        shuffle (bool): whether or not we shuffle the data. Usually we shuffle the data.

    Returns:
        trainloader (torch.utils.data.dataloader.DataLoader)
        testloader (torch.utils.data.dataloader.DataLoader)

    '''

    absolute_path = os.getcwd()
    download_path = os.path.join(absolute_path, 'data')
    print(download_path)

    trainset = torchvision.datasets.CIFAR10(root=download_path, train=True, transform=transform, download=download)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

    testset = torchvision.datasets.CIFAR10(root=download_path, train=False, transform=transform, download=download)
    testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size, shuffle=shuffle)

    return trainloader, testloader
