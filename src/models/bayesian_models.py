import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianCNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BayesianCNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.mu = nn.Parameter(data=torch.Tensor(out_channels,in_channels,kernel_size,kernel_size),
                                   requires_grad=True)
        self.rho = nn.Parameter(data=torch.Tensor(out_channels,in_channels,kernel_size,kernel_size),
                                   requires_grad=True)
        self.bias = nn.Parameter(data=torch.Tensor(out_channels), requires_grad=True)
        self.reset_parameters()

    def forward(self, x):
        std = self.get_std()
        weight = self.mu + std*torch.randn_like(std)
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding,)

    def reset_parameters(self):
        super(BayesianCNN,self).reset_parameters()
        self.mu.data = self.weight.data
        self.rho.data = torch.ones_like(self.rho.data)

    def get_std(self):
        return torch.log(1+torch.exp(self.rho))


class BayesianLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__(in_features, out_features)
        self.mu = nn.Parameter(data=torch.Tensor(out_features,in_features), requires_grad=True)
        self.rho = nn.Parameter(data=torch.tensor(out_features,in_features), requires_grad=True)
        self.reset_parameters()

    def forward(self, x):
        pass

    def reset_parameters(self):
        super(BayesianLinear,self).reset_parameters()
        self.mu.data = self.weight.data
        self.rho.data = torch.ones_like(self.rho.data)

    def get_std(self):
        return torch.log(1+torch.exp(self.rho))
