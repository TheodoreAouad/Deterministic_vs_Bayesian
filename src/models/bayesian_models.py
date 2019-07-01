import math

import torch
import torch.nn as nn
import torch.nn.functional as F


#TODO: Add get_bayesian_parameter to each bayesian module
class GaussianCNN(nn.Conv2d):

    def __init__(self, rho, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        if rho == "determinist" :
            self.determinist = True
        elif type(rho) in [int, float]:
            self.determinist = False
            self.rho_init = rho
            self.rho = nn.Parameter(data=torch.Tensor(out_channels, in_channels, kernel_size, kernel_size),
                                    requires_grad=True)
        else:
            print('rho not understood. Determinist classifier created. '
                          'To delete this warning, write rho as "determinist"')
            self.determinist = True

        self.mu = nn.Parameter(data=torch.Tensor(out_channels, in_channels, kernel_size, kernel_size),
                                   requires_grad=True)
        self.reset_parameters()

    def forward(self, x, determinist=None):
        if determinist is not None:
            do_determinist = determinist
        else:
            do_determinist = self.determinist
        if do_determinist:
            weight = self.mu
        else:
            std = self.get_std()
            weight = self.mu + std*torch.randn_like(std)
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding,)

    def reset_parameters(self, rho=None):
        super().reset_parameters()
        if hasattr(self, "mu"):
            self.mu.data = self.weight.data
            if not self.determinist:
                if rho is not None:
                    self.rho_init = rho
                self.rho.data = self.rho_init*torch.ones_like(self.rho.data)

    def get_std(self):
        if self.determinist:
            return 0
        return torch.log(1+torch.exp(self.rho))


class GaussianLinear(nn.Linear):

    def __init__(self, rho, in_features, out_features):
        super().__init__(in_features, out_features)
        if rho == "determinist":
            self.determinist = True
        elif type(rho) in [int, float]:
            self.determinist = False
            self.rho_init = rho
            self.rho = nn.Parameter(data=torch.Tensor(out_features, in_features), requires_grad=True)
        else:
            print('rho not understood. Determinist classifier created. '
                          'To delete this warning, write rho as "determinist"')
            self.determinist = True

        self.mu = nn.Parameter(data=torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    def forward(self, x, determinist=False):
        if determinist is not None:
            do_determinist = determinist
        else:
            do_determinist = self.determinist
        if do_determinist:
            weight = self.mu
        else:
            std = self.get_std()
            weight = self.mu + std * torch.randn_like(std)
        return F.linear(x, weight, self.bias)

    def reset_parameters(self, rho=None):
        super().reset_parameters()
        if hasattr(self, "mu"):
            self.mu.data = self.weight.data
            if not self.determinist:
                if rho is not None:
                    self.rho_init = rho
                self.rho.data = self.rho_init * torch.ones_like(self.rho.data)

    def get_std(self):
        if self.determinist:
            return 0
        return torch.log(1+torch.exp(self.rho))


class GaussianClassifierMNIST(nn.Module):

    def __init__(self, rho, number_of_classes=10, dim_input=28):
        super().__init__()
        if rho == "determinist" :
            self.determinist = True
        elif type(rho) in [int, float]:
            self.determinist = False
        else:
            print('rho not understood. Determinist classifier created. '
                          'To delete this warning, write rho as "determinist"')
            self.determinist = True

        self.dim_input = dim_input
        self.number_of_classes = number_of_classes

        self.gaussian_conv1 = GaussianCNN(rho, 1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.gaussian_conv2 = GaussianCNN(rho, 16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gaussian_linear = GaussianLinear(rho, 32 * self.dim_input//4*self.dim_input//4, number_of_classes)

    def forward_before_softmax(self, x, determinist=None):
        if determinist is not None:
            do_determinist = determinist
        else:
            do_determinist = self.determinist
        output = self.bn1(self.gaussian_conv1(x, determinist=do_determinist))
        output = self.pool1(F.relu(output))
        output = self.bn2(self.gaussian_conv2(output, determinist=do_determinist))
        output = self.pool2(F.relu(output))
        output = output.view(-1, 32*self.dim_input//4*self.dim_input//4)

        return self.gaussian_linear(output, determinist=do_determinist)

    def forward(self, x, determinist=None):
        if determinist is not None:
            do_determinist = determinist
        else:
            do_determinist = self.determinist
        output = self.forward_before_softmax(x, determinist)
        output = F.softmax(output, dim=1)

        return output


class GaussianClassifierCIFAR(nn.Module):

    def __init__(self, rho, number_of_classes, dim_input=32, determinist=False):
        super().__init__()
        self.determinist = determinist
        self.dim_input = dim_input

        self.gaussian_conv1 = GaussianCNN(rho, 3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.gaussian_conv2 = GaussianCNN(rho, 16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gaussian_linear = GaussianLinear(rho, 32 * self.dim_input//4*self.dim_input//4, number_of_classes)

    def forward(self, x, determinist=None):
        if determinist is not None:
            do_determinist = determinist
        else:
            do_determinist = self.determinist
        output = self.pool1(F.relu(self.gaussian_conv1(x, determinist=do_determinist)))
        output = self.pool2(F.relu(self.gaussian_conv2(output, determinist=do_determinist)))
        output = output.view(-1, 32 * self.dim_input // 4 * self.dim_input // 4)
        output = F.softmax(self.gaussian_linear(output, determinist=do_determinist), dim=1)

        return output
