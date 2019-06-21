import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import *


class DeterministClassifier(nn.Module):

    def __init__(self, number_of_classes):
        super(DeterministClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, number_of_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        output = F.softmax(self.fc1(x))

        return output


class ProbabilistClassifier(nn.Module):

    def __init__(self, number_of_classes):
        super(ProbabilistClassifier, self).__init__()

        self.mu1 = nn.Parameter(data=torch.Tensor(16, 1, 3, 3), requires_grad=True)
        self.mu2 = nn.Parameter(data=torch.Tensor(32, 16, 3, 3), requires_grad=True)
        self.bias1 = nn.Parameter(data=torch.Tensor(16), requires_grad=True)
        self.bias2 = nn.Parameter(data=torch.Tensor(32), requires_grad=True)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.mu_fc = nn.Parameter(data=torch.Tensor(10, 16*14*14), requires_grad=True)
        self.bias_fc = nn.Parameter(data=torch.Tensor(10), requires_grad=True)

        reset_parameters_conv(self.mu1, self.bias1)
        reset_parameters_conv(self.mu2, self.bias2)
        reset_parameters_linear(self.mu_fc, self.bias_fc)

    def forward(self, x):
        x = self.pool1(F.relu(F.conv2d(x, weight=self.mu1, bias=self.bias1, padding=1)))
        x = self.pool2(F.relu(F.conv2d(x, weight=self.mu2, bias=self.bias2, padding=1)))
        x = x.view(-1, 32 * 7 * 7)
        output = F.softmax(F.linear(x, self.mu_fc, self.bias_fc))
        return output
