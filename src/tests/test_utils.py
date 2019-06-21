import torch.nn as nn

from src.models import *
from src.utils import *


class ResetParametersConv:

    @staticmethod
    def test_BayNet_mu1_vs_mu1():
        mu1 = nn.Parameter(data=torch.Tensor(16, 1, 3, 3), requires_grad=True)
        bias1 = nn.Parameter(data=torch.Tensor(16), requires_grad=True)

        BayNet = ProbabilistClassifier(10)

        seed1 = set_and_print_random_seed()
        reset_parameters_conv(mu1, bias1)
        set_and_print_random_seed(seed1)
        reset_parameters_conv(BayNet.mu1, BayNet.bias1)

        assert torch.sum(torch.abs(mu1-BayNet.mu1)) == 0

    @staticmethod
    def test_DetNet_conv_vs_Conv2d():
        conv1 = nn.Conv2d(1, 16, 3)
        DetNet = DeterministClassifier(10)

        seed1 = set_and_print_random_seed()
        DetNet.conv1.reset_parameters()
        set_and_print_random_seed(seed1)
        conv1.reset_parameters()

        w1 = DetNet.conv1.weight.data
        w2 = conv1.weight.data

        assert torch.sum(torch.abs(w1-w2)) == 0

    @staticmethod
    def test_Conv2d_vs_custom_init():
        conv1 = nn.Conv2d(1, 16, 3)
        mu1 = nn.Parameter(data=torch.Tensor(16, 1, 3, 3), requires_grad=True)
        bias1 = nn.Parameter(data=torch.Tensor(16), requires_grad=True)

        seed1 = set_and_print_random_seed()
        conv1.reset_parameters()
        set_and_print_random_seed(seed1)
        reset_parameters_conv(mu1, bias1)

        w1 = conv1.weight.data
        w2 = mu1
        b1 = conv1.bias.data
        b2 = bias1

        assert torch.sum(torch.abs(w1-w2)) == 0
        assert torch.sum(torch.abs(b1-b2)) == 0

    @staticmethod
    def test_BayNet_ini_vs_DetNet_ini():
        DetNet = DeterministClassifier(10)
        BayNet = ProbabilistClassifier(10)

        seed1 = set_and_print_random_seed()
        DetNet.conv1.reset_parameters()
        set_and_print_random_seed(seed1)
        reset_parameters_conv(BayNet.mu1, BayNet.bias1)
        w1 = DetNet.conv1.weight.data
        w2 = BayNet.mu1
        b1 = DetNet.conv1.bias.data
        b2 = BayNet.bias1

        assert torch.sum(torch.abs(w1-w2)) == 0
        assert torch.sum(torch.abs(b1-b2)) == 0

    @staticmethod
    def test_conv_identity():
        DetNet = DeterministClassifier(10)
        BayNet = ProbabilistClassifier(10)

        seed1 = set_and_print_random_seed()
        DetNet.conv1.reset_parameters()
        set_and_print_random_seed(seed1)
        reset_parameters_conv(BayNet.mu1, BayNet.bias1)

        image = torch.rand(1,28,28)
        output1 = F.conv2d(image, weight=BayNet.mu1, bias=BayNet.bias1, padding=1)
        output2 = DetNet.conv1(image)

        assert torch.sum(torch.abs(output1-output2)) == 0

    @staticmethod
    def test_linear_identity():
        DetNet = DeterministClassifier(10)
        BayNet = ProbabilistClassifier(10)

        seed1 = set_and_print_random_seed()
        DetNet.fc1.reset_parameters()
        set_and_print_random_seed(seed1)
        reset_parameters_conv(BayNet.mu_fc, BayNet.bias_fc)

        embedding = torch.rand(32*7*7)
        output1 = F.linear(embedding, weight=BayNet.mu1, bias=BayNet.bias1, padding=1)
        output2 = DetNet.fc1(embedding)

        assert torch.sum(torch.abs(output1 - output2)) == 0

    @staticmethod
    def test_net_identity():
        DetNet = DeterministClassifier(10)
        BayNet = ProbabilistClassifier(10)

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

        image = torch.rand(1,28,28)

        output1 = DetNet(image)
        output2 = BayNet(image)

        assert torch.sum(torch.abs(output1 - output2)) == 0






