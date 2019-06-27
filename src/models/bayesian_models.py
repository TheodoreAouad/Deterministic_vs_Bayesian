import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianCNN(nn.Conv2d):

    def __init__(self, rho, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.mu = nn.Parameter(data=torch.Tensor(out_channels, in_channels, kernel_size, kernel_size),
                                   requires_grad=True)
        self.rho = nn.Parameter(data=torch.Tensor(out_channels, in_channels, kernel_size, kernel_size),
                                   requires_grad=True)
        self.reset_parameters(rho)

    def forward(self, x, determinist=False):
        if determinist:
            weight = self.mu
        else:
            std = self.get_std()
            weight = self.mu + std*torch.randn_like(std)
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding,)

    def reset_parameters(self, rho=1):
        super().reset_parameters()
        if hasattr(self, "mu"):
            self.mu.data = self.weight.data
            self.rho.data = rho*torch.ones_like(self.rho.data)

    def get_std(self):
        return torch.log(1+torch.exp(self.rho))


class GaussianLinear(nn.Linear):

    def __init__(self, rho, in_features, out_features):
        super(GaussianLinear, self).__init__(in_features, out_features)
        self.mu = nn.Parameter(data=torch.Tensor(out_features, in_features), requires_grad=True)
        self.rho = nn.Parameter(data=torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters(rho)

    def forward(self, x, determinist=False):
        if determinist:
            weight = self.mu
        else:
            std = self.get_std()
            weight = self.mu + std * torch.randn_like(std)
        return F.linear(x, weight, self.bias)

    def reset_parameters(self, rho=1):
        super(GaussianLinear, self).reset_parameters()
        if hasattr(self, "mu"):
            self.mu.data = self.weight.data
            self.rho.data = rho*torch.ones_like(self.rho.data)

    def get_std(self):
        return torch.log(1+torch.exp(self.rho))


class GaussianClassifierMNIST(nn.Module):

    def __init__(self, rho, number_of_classes=10, dim_input=28, determinist=False):
        super().__init__()
        self.determinist = determinist
        self.dim_input = dim_input

        self.gaussian_conv1 = GaussianCNN(rho, 1, 16, 3, padding=1)
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
        output = output.view(-1, 32*self.dim_input//4*self.dim_input//4)
        output = F.softmax(self.gaussian_linear(output, determinist=do_determinist), dim=1)

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
