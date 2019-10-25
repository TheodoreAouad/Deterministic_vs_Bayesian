import os

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms as transforms

from src.dataset_manager.datasets_creator import MNISTSpecificLabels, CIFAR10SpecificLabels

transform = transforms.ToTensor()

absolute_path = os.getcwd()
download_path = os.path.join(absolute_path, 'data')


class EmptyLoader:
    def __init__(self):
        self.dataset = []

    def __len__(self):
        return 0

    def __getitem__(self, index):
        pass


def get_mnist(
        root=download_path,
        train_labels=range(10),
        eval_labels=range(10),
        split_train=(0, 1),
        split_val=0.2,
        transform=transform,
        batch_size=16,
        shuffle=True,
):
    """

    Args:
        root (str): path to the directory where we want to download the data
        train_labels (list || tuple || array): labels we want to keep in the training set
        eval_labels (list || tuple || array): labels we want to keep in the testing set
        split_train (Tuple): (beginning of the split: end of the split) split of the train data, in few shot we take
                              small splits.
        split_val (float): the proportion of the evaluation set we use for validation
        transform (torch.transform): which transformation to perform to the data
        batch_size (int): size of the batch
        shuffle (bool): whether or not we shuffle the data. Usually we shuffle the data.

    Returns:

        trainloader: loader of train data
        valloader: loader of val data
        evalloader: loader of eval data

    """
    trainloader, valloader, evalloader = EmptyLoader(), EmptyLoader(), EmptyLoader()
    if len(train_labels) > 0:
        trainset = MNISTSpecificLabels(root=root, labels=train_labels, train=True, split=split_train,
                                       transform=transform,
                                       download=True, shuffle_dataset=shuffle)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

        if split_val > 0:
            valset = MNISTSpecificLabels(root=root, labels=train_labels, train=False, split=(0, split_val),
                                         transform=transform, download=True, shuffle_dataset=shuffle)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=shuffle)

    if len(eval_labels) > 0:
        evalset = MNISTSpecificLabels(root=root, labels=eval_labels, train=False, split=(split_val, 2),
                                      transform=transform, download=True, shuffle_dataset=shuffle)
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=shuffle)

    return trainloader, valloader, evalloader


def get_cifar10(
        root=None,
        train_labels=range(10),
        eval_labels=range(10),
        split_train=(0, 1),
        split_val=0.2,
        transform=transform,
        batch_size=16,
        shuffle=True,
        download=False,
        **kwargs,
):
    """

    Args:
        transform (torch.transform): which transformation to perform to the data
        batch_size (int): size of the batch
        shuffle (bool): whether or not we shuffle the data. Usually we shuffle the data.

    Returns:
        torch.utils.data.dataloader.DataLoader: loader of train data
        torch.utils.data.dataloader.DataLoader: loader of evaluation data

    """
    global download_path
    if root is None:
        root = download_path
    #
    # trainset = CIFAR10SpecificLabels(root=root, labels=train_labels, train=True, transform=transform, download=download, shuffle_d  ataset=shuffle)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    #
    # testset = CIFAR10SpecificLabels(root=root, labels=eval_labels, train=False, transform=transform, download=download, shuffle_dataset=shuffle)
    # evalloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    trainloader, valloader, evalloader = EmptyLoader(), EmptyLoader(), EmptyLoader()
    if len(train_labels) > 0:
        trainset = CIFAR10SpecificLabels(root=root, labels=train_labels, train=True, split=split_train,
                                       transform=transform,
                                       download=download, shuffle_dataset=shuffle)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

        if split_val > 0:
            valset = CIFAR10SpecificLabels(root=root, labels=train_labels, train=False, split=(0, split_val),
                                         transform=transform, download=download, shuffle_dataset=shuffle)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=shuffle)

    if len(eval_labels) > 0:
        evalset = CIFAR10SpecificLabels(root=root, labels=eval_labels, train=False, split=(split_val, 2),
                                      transform=transform, download=download, shuffle_dataset=shuffle)
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=shuffle)

    return trainloader, valloader, evalloader

    # return trainloader, evalloader


def get_omniglot(
        root=download_path,
        transform=transform,
        batch_size=16,
        shuffle=True,
        download=True,
        **kwargs,
):
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

    dataset = torchvision.datasets.Omniglot(root=root, transform=transform, download=download,)
    omniglot_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return omniglot_loader


def get_random(
        batch_size=100,
        number_of_batches=10,
        number_of_channels=1,
        img_dim=28,
        number_of_classes=10,
        **kwargs,
):
    random_loader_train = RandomLoader(
        number_of_batches=number_of_batches,
        batch_size=batch_size,
        number_of_channels=number_of_channels,
        img_dim = img_dim,
        number_of_classes=number_of_classes,
    )

    random_loader_val = RandomLoader(
        number_of_batches=number_of_batches,
        batch_size=batch_size,
        number_of_channels=number_of_channels,
        img_dim = img_dim,
        number_of_classes=number_of_classes,
    )

    random_loader_eval = RandomLoader(
        number_of_batches=number_of_batches,
        batch_size=batch_size,
        number_of_channels=number_of_channels,
        img_dim = img_dim,
        number_of_classes=number_of_classes,
    )

    return random_loader_train, random_loader_val, random_loader_eval


class RandomLoader:

    def __init__(
            self,
            number_of_batches=2,
            batch_size=2,
            number_of_channels=1,
            img_dim=28,
            number_of_classes=10,
    ):
        self.batch_size = batch_size
        self.number_of_batches = number_of_batches
        self.number_of_classes = number_of_classes
        self.dataset = torch.randn((number_of_batches, batch_size, number_of_channels, img_dim, img_dim))
        self.labels = torch.randint(0, number_of_classes, (number_of_batches, batch_size))

    def __getitem__(self, idx):
        return (self.dataset[idx]), self.labels[idx]

    def __len__(self):
        return self.number_of_batches


class OneSampleLoader:

    def __init__(self, sample, target, transform):
        self.dataset = [(sample, target)]
        self.sample = sample
        self.target = target
        self.transform = transform
        self.num = 0

    def __len__(self):
        return 1

    def __getitem__(self, item):
        if self.num > 0:
            raise StopIteration
        self.num += 1
        return self.transform(self.sample).unsqueeze(0), self.target.unsqueeze(0)
