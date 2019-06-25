import os

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor()

#TODO: docstring
def get_mnist(transform=transform, batch_size=16, shuffle=True):
    '''
    This function takes transformations and batch size as inputs and returns train and test loaders
    :param transform:
    :param batch_size:
    :param shuffle:
    :return:
    '''

    absolute_path = os.getcwd()
    download_path = os.path.join(absolute_path, 'data')
    print(download_path)

    trainset = torchvision.datasets.MNIST(root=download_path, train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

    testset = torchvision.datasets.MNIST(root=download_path, train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    return trainloader, testloader
