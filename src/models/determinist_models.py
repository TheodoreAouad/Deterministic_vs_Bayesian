import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import *


class DeterministClassifierSequential(nn.Module):

    def __init__(self, number_of_classes):
        super(DeterministClassifierSequential, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, number_of_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        output = F.softmax(self.fc1(x), dim=1)

        return output


class DeterministClassifierFunctional(nn.Module):

    def __init__(self, number_of_classes):
        super(DeterministClassifierFunctional, self).__init__()

        self.mu1 = nn.Parameter(data=torch.Tensor(16, 1, 3, 3), requires_grad=True)
        self.mu2 = nn.Parameter(data=torch.Tensor(32, 16, 3, 3), requires_grad=True)
        self.bias1 = nn.Parameter(data=torch.Tensor(16), requires_grad=True)
        self.bias2 = nn.Parameter(data=torch.Tensor(32), requires_grad=True)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.mu_fc = nn.Parameter(data=torch.Tensor(10, 32*7*7), requires_grad=True)
        self.bias_fc = nn.Parameter(data=torch.Tensor(10), requires_grad=True)

        reset_parameters_conv(self.mu1, self.bias1)
        reset_parameters_conv(self.mu2, self.bias2)
        reset_parameters_linear(self.mu_fc, self.bias_fc)

    def forward(self, x):
        x = self.pool1(F.relu(F.conv2d(x, weight=self.mu1, bias=self.bias1, padding=1)))
        x = self.pool2(F.relu(F.conv2d(x, weight=self.mu2, bias=self.bias2, padding=1)))
        x = x.view(-1, 32 * 7 * 7)
        output = F.softmax(F.linear(x, self.mu_fc, self.bias_fc), dim=1)
        return output


def init_same_baynet_detnet():
    '''
    This function returns the models, initiated the same way.
    :return: tuple: (BayNet, DetNet)
    '''
    DetNet = DeterministClassifierSequential(10)
    BayNet = DeterministClassifierFunctional(10)

    seed1 = set_and_print_random_seed()
    DetNet.conv1.reset_parameters()
    set_and_print_random_seed(seed1)
    reset_parameters_conv(BayNet.mu1, BayNet.bias1)

    set_and_print_random_seed(seed1)
    DetNet.conv2.reset_parameters()
    set_and_print_random_seed(seed1)
    reset_parameters_conv(BayNet.mu2, BayNet.bias2)

    set_and_print_random_seed(seed1)
    DetNet.fc1.reset_parameters()
    set_and_print_random_seed(seed1)
    reset_parameters_linear(BayNet.mu_fc, BayNet.bias_fc)

    return BayNet, DetNet


class DeterministClassifierCIFAR(nn.Module):

    def __init__(self, number_of_classes, dim_input=32):
        super().__init__()
        self.dim_input = dim_input

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(32 * self.dim_input//4*self.dim_input//4, number_of_classes)


    def forward(self, x):
        output = self.pool1(F.relu(self.conv1(x)))
        output = self.pool2(F.relu(self.conv2(output)))
        output = output.view(-1, 32 * self.dim_input // 4 * self.dim_input // 4)
        output = F.softmax(self.linear(output), dim=1)

        return output
