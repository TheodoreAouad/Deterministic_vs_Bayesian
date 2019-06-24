import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianCNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.mu = nn.Parameter(data=torch.Tensor(out_channels, in_channels, kernel_size, kernel_size),
                                   requires_grad=True)
        self.rho = nn.Parameter(data=torch.Tensor(out_channels, in_channels, kernel_size, kernel_size),
                                   requires_grad=True)
        self.reset_parameters()

    def forward(self, x, determinist=False):
        std = self.get_std()
        if determinist:
            weight = self.mu
        else:
            weight = self.mu + std*torch.randn_like(std)
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding,)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "mu"):
            self.mu.data = self.weight.data
            self.rho.data = torch.ones_like(self.rho.data)

    def get_std(self):
        return torch.log(1+torch.exp(self.rho))


class BayesianLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__(in_features, out_features)
        self.mu = nn.Parameter(data=torch.Tensor(out_features, in_features), requires_grad=True)
        self.rho = nn.Parameter(data=torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    def forward(self, x, determinist=False):
        std = self.get_std()
        if determinist:
            weight = self.mu
        else:
            weight = self.mu + std * torch.randn_like(std)
        return F.linear(x, weight, self.bias)

    def reset_parameters(self):
        super(BayesianLinear,self).reset_parameters()
        if hasattr(self, "mu"):
            self.mu.data = self.weight.data
            self.rho.data = torch.ones_like(self.rho.data)

    def get_std(self):
        return torch.log(1+torch.exp(self.rho))


class BayesianClassifier(nn.Module):

    def __init__(self, number_of_classes, determinist=False):
        super().__init__()
        self.determinist = determinist

        self.bay_conv1 = BayesianCNN(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bay_conv2 = BayesianCNN(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bay_linear = BayesianLinear(32*7*7, number_of_classes)

    def forward(self, x, determinist=None):
        if determinist is not None:
            do_determinist = determinist
        else:
            do_determinist = self.determinist
        output = self.pool1(F.relu(self.bay_conv1(x, determinist=do_determinist)))
        output = self.pool2(F.relu(self.bay_conv2(output, determinist=do_determinist)))
        output = output.view(-1, 32*7*7)
        output = F.softmax(self.bay_linear(output, determinist=do_determinist), dim=1)

        return output
