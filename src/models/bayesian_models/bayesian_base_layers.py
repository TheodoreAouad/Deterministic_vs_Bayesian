import torch
from torch import nn as nn
from torch.nn import functional as F


class GaussianLinear(nn.Linear):
    """
    Bayesian Linear layer with gaussian weights.
    """

    def __init__(self, rho, in_features, out_features):
        """

        Args:
            rho (float): parameter to get the std. std = log(1+exp(rho))
        """
        super().__init__(in_features, out_features)
        if rho == "determinist":
            self.determinist = True
        elif type(rho) in [int, float]:
            self.determinist = False
            self.rho_init = rho
            self.rho = nn.Parameter(data=torch.Tensor(out_features, in_features), requires_grad=True)
            self.rho_bias = nn.Parameter(data=torch.Tensor(out_features), requires_grad=True)
        else:
            print('rho not understood. Determinist classifier created. '
                  'To delete this warning, write rho as "determinist"')
            self.determinist = True

        self.mu = nn.Parameter(data=torch.Tensor(out_features, in_features), requires_grad=True)
        self.mu_bias = nn.Parameter(data=torch.Tensor(out_features), requires_grad=True)
        self.reset_parameters()

        if not self.determinist:
            for params in [self.mu, self.rho, self.mu_bias, self.rho_bias]:
                params.bayesian = True

    def forward(self, x, determinist=None):
        if determinist is not None:
            is_forward_determinist = determinist
        else:
            is_forward_determinist = self.determinist
        if is_forward_determinist:
            weight = self.mu
            bias = self.mu_bias
        else:
            weight, bias = self.sample_weights()
        return F.linear(x, weight, bias)

    def reset_parameters(self, rho=None):
        super().reset_parameters()
        if hasattr(self, "mu"):
            self.mu.data = self.weight.data
            self.mu_bias.data = self.bias.data
            if not self.determinist:
                if rho is not None:
                    self.rho_init = rho
                self.rho.data = self.rho_init * torch.ones_like(self.rho.data)
                self.rho_bias.data = self.rho_init * torch.ones_like(self.rho_bias.data)

    def get_std(self):
        if self.determinist:
            return 0, 0
        return torch.log(1 + torch.exp(self.rho)), torch.log(1 + torch.exp(self.rho_bias))

    def sample_weights(self):
        std, std_bias = self.get_std()
        weight = self.mu + std * torch.randn_like(std)
        bias = self.mu_bias + std_bias * torch.randn_like(std_bias)
        self.previous_weight = weight
        self.previous_bias = bias
        return weight, bias

    def get_previous_weights(self):
        return self.previous_weight, self.previous_bias

    def bayesian_parameters(self):
        for params in self.parameters():
            if getattr(params, "bayesian", False):
                yield params

    def named_bayesian_parameters(self):
        for name, params in self.named_parameters():
            if getattr(params, "bayesian", False):
                yield name, params


class GaussianCNN(nn.Conv2d):
    """
    Bayesian 2D convolutional layer with bayesian weights.
    """

    def __init__(self, rho, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """

        Args:
            rho (float): parameter to get the std. std = log(1+exp(rho))
            rest: same as nn.Conv2d
        """
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         bias=bias)
        if rho == "determinist":
            self.determinist = True
        elif type(rho) in [int, float]:
            self.determinist = False
            self.rho_init = rho
            self.rho = nn.Parameter(data=torch.Tensor(out_channels, in_channels, kernel_size, kernel_size),
                                    requires_grad=True)
            self.rho_bias = nn.Parameter(data=torch.Tensor(out_channels), requires_grad=True)
        else:
            raise ValueError('rho should be "determinist" or type float.')

        self.mu = nn.Parameter(data=torch.Tensor(out_channels, in_channels, kernel_size, kernel_size),
                               requires_grad=True)
        self.mu_bias = nn.Parameter(data=torch.Tensor(out_channels), requires_grad=True)
        self.reset_parameters()

        if not self.determinist:
            for params in [self.mu, self.rho, self.mu_bias, self.rho_bias]:
                params.bayesian = True

    def forward(self, x, determinist=None):
        if determinist is not None:
            is_forward_determinist = determinist
        else:
            is_forward_determinist = self.determinist
        if is_forward_determinist:
            weight = self.mu
            bias = self.mu_bias
        else:
            weight, bias = self.sample_weights()
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding, )

    def reset_parameters(self, rho=None):
        super().reset_parameters()
        if hasattr(self, "mu"):
            self.mu.data = self.weight.data
            self.mu_bias.data = self.bias.data
            if not self.determinist:
                if rho is not None:
                    self.rho_init = rho
                self.rho.data = self.rho_init * torch.ones_like(self.rho.data)
                self.rho_bias.data = self.rho_init * torch.ones_like(self.rho_bias.data)

    def get_std(self):
        if self.determinist:
            return 0, 0
        return torch.log(1 + torch.exp(self.rho)), torch.log(1 + torch.exp(self.rho_bias))

    def sample_weights(self):
        std, std_bias = self.get_std()
        weight = self.mu + std * torch.randn_like(std)
        bias = self.mu_bias + std_bias * torch.randn_like(std_bias)
        self.previous_weight = weight
        self.previous_bias = bias

        return weight, bias

    def get_output_dim(self, input_dim):
        pass

    def get_previous_weights(self):
        return self.previous_weight, self.previous_bias

    def bayesian_parameters(self):
        for params in self.parameters():
            if getattr(params, "bayesian", False):
                yield params

    def named_bayesian_parameters(self):
        for name, params in self.named_parameters():
            if getattr(params, "bayesian", False):
                yield name, params
