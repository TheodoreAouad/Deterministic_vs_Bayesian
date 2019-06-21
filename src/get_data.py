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
    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    return trainloader, testloader
