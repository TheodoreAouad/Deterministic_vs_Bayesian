import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from src.models.bayesian_models.bayesian_base_layers import GaussianCNN, GaussianLinear
from src.utils import vectorize


class GaussianClassifier(nn.Module):
    """
    Bayesian classifier on the dataset MNIST with gaussian weights.
    """

    def __init__(self, rho=-5, mus_prior=(0, 0), stds_prior=None,
                 number_of_classes=10, dim_input=28):
        """

        Args:
            rho (float || str): parameter to get the std. std = log(1+exp(rho))
            mus_prior (tuple): the means of the prior (weight and bias). Often will be (0,0)
            stds_prior (tuple): the stds of the prior (weight and bias)
            number_of_classes (int): number of different classes in the problem
            dim_input (int): dimension of a size of a squared image
        """
        super().__init__()
        if rho == "determinist":
            self.determinist = True
        elif type(rho) in [int, float]:
            self.determinist = False
        else:
            print('rho not understood. Determinist classifier created. '
                  'To delete this warning, write rho as "determinist"')
            self.determinist = True

        self.device = "cpu"
        self.dim_input = dim_input
        self.number_of_classes = number_of_classes

        if not self.determinist:
            mu_prior, mu_bias_prior = mus_prior
            self.mu_prior_init = mu_prior
            self.mu_bias_prior_init = mu_bias_prior
            if stds_prior is not None:
                std_prior, std_bias_prior = stds_prior
            else:
                std_prior = math.log(math.exp(rho) + 1)
                std_bias_prior = math.log(math.exp(rho) + 1)
            self.std_prior_init = std_prior
            self.std_bias_prior_init = std_bias_prior

        self.gaussian_conv1 = GaussianCNN(rho, 1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.gaussian_conv2 = GaussianCNN(rho, 16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gaussian_linear = GaussianLinear(rho, 32 * self.dim_input // 4 * self.dim_input // 4, number_of_classes)

        self.init_prior()

    def init_prior(self, mu_prior=None, std_prior=None, mu_bias_prior=None, std_bias_prior=None):
        if self.determinist:
            return 0

        if mu_prior is not None:
            self.mu_prior_init = mu_prior
            self.std_prior_init = std_prior
            self.mu_bias_prior_init = mu_bias_prior
            self.std_bias_prior_init = std_bias_prior

        dim_weights = torch.tensor(0, requires_grad=False).to(self.device)
        dim_bias = torch.tensor(0, requires_grad=False).to(self.device)
        for layer in self.children():
            if not getattr(layer, "determinist", True):
                for name, params in layer.named_bayesian_parameters():
                    if name == "mu":
                        dim_weights += params.nelement()
                    if name == "mu_bias":
                        dim_bias += params.nelement()

        self.mu_prior = self.mu_prior_init * torch.ones(dim_weights, requires_grad=False).to(self.device)
        self.std_prior = self.std_prior_init * torch.ones(dim_weights, requires_grad=False).to(self.device)
        self.mu_bias_prior = self.mu_bias_prior_init * torch.ones(dim_bias, requires_grad=False).to(self.device)
        self.std_bias_prior = self.std_bias_prior_init * torch.ones(dim_bias, requires_grad=False).to(self.device)

    def forward_before_softmax(self, x, determinist=None):
        if determinist is not None:
            is_forward_determinist = determinist
        else:
            is_forward_determinist = self.determinist
        output = self.bn1(self.gaussian_conv1(x, determinist=is_forward_determinist))
        output = self.pool1(F.relu(output))
        output = self.bn2(self.gaussian_conv2(output, determinist=is_forward_determinist))
        output = self.pool2(F.relu(output))
        output = output.view(-1, 32 * self.dim_input // 4 * self.dim_input // 4)

        return self.gaussian_linear(output, determinist=is_forward_determinist)

    def forward(self, x, determinist=None):
        if determinist is not None:
            is_forward_determinist = determinist
        else:
            is_forward_determinist = self.determinist
        output = self.forward_before_softmax(x, is_forward_determinist)
        output = F.softmax(output, dim=1)

        return output

    def variational_posterior(self, weights, bias):
        """
        log(q(w|D))
        Args:
            weights (torch.Tensor): 1 dimension tensor with size the number of weights parameters
            bias (torch.Tensor): 1 dimension tensor with size the number of bias parameters

        Returns:
            variational_posterior density evaluated on weights (torch.Tensor): size (1)
        """
        if self.determinist:
            return 0
        mu = torch.Tensor().to(self.device)
        rho = torch.Tensor().to(self.device)
        mu_bias = torch.Tensor().to(self.device)
        rho_bias = torch.Tensor().to(self.device)
        for layer in self.children():
            if not getattr(layer, "determinist", True):
                for name, params in layer.named_bayesian_parameters():
                    if name == "mu":
                        mu = torch.cat((mu, vectorize(params)))
                    if name == "rho":
                        rho = torch.cat((rho, vectorize(params)))
                    if name == "mu_bias":
                        mu_bias = torch.cat((mu_bias, vectorize(params)))
                    if name == "rho_bias":
                        rho_bias = torch.cat((rho_bias, vectorize(params)))

        std = torch.log(1 + torch.exp(rho))
        std_bias = torch.log(1 + torch.exp(rho_bias))

        return -1 / 2 * ((torch.sum((weights - mu) ** 2 / std) + torch.sum((bias - mu_bias) ** 2 / std_bias)) +
                         2 * torch.sum(torch.log(std)) + 2 * torch.sum(torch.log(std_bias)))

    def prior(self, weights, bias):
        '''
        logP(W). We choose the prior to be gaussian.
        Args:
            weights (torch.Tensor): 1 dimension tensor with size the number of weights parameters
            bias (torch.Tensor): 1 dimension tensor with size the number of bias paremeters

        Returns:
            prior_loss (torch.Tensor): size (1)
        '''
        return -1 / 2 * ((torch.sum((weights - self.mu_prior) ** 2 / self.std_prior) +
                          torch.sum((bias - self.mu_bias_prior) ** 2 / self.std_bias_prior)))

    def get_previous_weights(self, output='vector'):
        """
        Get the weights used to compute the previous sample.
        Args:
            output (str): Type of the output. Is either 'dict' or 'vector'.

        Returns:
            weights (output): the weights used last
            bias (output): the bias used last

        """

        if output == 'vector':
            weights = torch.Tensor().to(self.device)
            bias = torch.Tensor().to(self.device)
            for layer in self.children():
                if not getattr(layer, "determinist", True):
                    weight_to_add, bias_to_add = layer.get_previous_weights()
                    weights = torch.cat((weights, vectorize(weight_to_add)))
                    bias = torch.cat((bias, vectorize(bias_to_add)))
            return weights, bias

        if output == "dict":
            weights = dict()
            bias = dict()
            for name, layer in self.named_children():
                if not getattr(layer, "determinist", True):
                    weights[name], bias[name] = layer.get_previous_weights()
            return weights, bias

    def get_bayesian_number_of_weights_and_bias(self):
        number_of_weights = 0
        number_of_bias = 0
        for name, param in self.named_bayesian_parameters():
            if "bias" in name:
                number_of_bias += param.nelement()
            else:
                number_of_weights += param.nelement()
        return number_of_weights, number_of_bias

    def n_bayesian_element(self):
        return sum(self.get_bayesian_number_of_weights_and_bias())

    def bayesian_parameters(self):
        for params in self.parameters():
            if getattr(params, "bayesian", False):
                yield params

    def named_bayesian_parameters(self):
        for name, params in self.named_parameters():
            if getattr(params, "bayesian", False):
                yield name, params

    def to(self, device):
        super().to(device)
        self.device = device
        self.init_prior()


class GaussianClassifierNoBatchNorm(GaussianClassifier):
    """
    Same as GaussianClassifier but without batch norm layers
    """

    def __init__(self, rho, mus_prior=(0, 0), stds_prior=None,
                 number_of_classes=10, dim_input=28):
        """

        Args:
            rho (float): parameter to get the std. std = log(1+exp(rho))
            mus_prior (tuple): the means of the prior (weight and bias). Often will be (0,0)
            stds_prior (tuple): the stds of the prior (weight and bias)
            number_of_classes (int): number of different classes in the problem
            dim_input (int): dimension of a size of a squared image
        """
        super().__init__()
        if rho == "determinist":
            self.determinist = True
        elif type(rho) in [int, float]:
            self.determinist = False
        else:
            print('rho not understood. Determinist classifier created. '
                  'To delete this warning, write rho as "determinist"')
            self.determinist = True

        self.device = "cpu"
        self.dim_input = dim_input
        self.number_of_classes = number_of_classes

        mu_prior, mu_bias_prior = mus_prior
        self.mu_prior_init = mu_prior
        self.mu_bias_prior_init = mu_bias_prior
        if stds_prior is not None:
            std_prior, std_bias_prior = stds_prior
        else:
            std_prior = math.log(math.exp(rho) + 1)
            std_bias_prior = math.log(math.exp(rho) + 1)
        self.std_prior_init = std_prior
        self.std_bias_prior_init = std_bias_prior

        self.gaussian_conv1 = GaussianCNN(rho, 1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.gaussian_conv2 = GaussianCNN(rho, 16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gaussian_linear = GaussianLinear(rho, 32 * self.dim_input // 4 * self.dim_input // 4, number_of_classes)

        self.init_prior()

    def forward_before_softmax(self, x, determinist=None):
        if determinist is not None:
            is_forward_determinist = determinist
        else:
            is_forward_determinist = self.determinist
        output = self.gaussian_conv1(x, determinist=is_forward_determinist)
        output = self.pool1(F.relu(output))
        output = self.gaussian_conv2(output, determinist=is_forward_determinist)
        output = self.pool2(F.relu(output))
        output = output.view(-1, 32 * self.dim_input // 4 * self.dim_input // 4)

        return self.gaussian_linear(output, determinist=is_forward_determinist)
