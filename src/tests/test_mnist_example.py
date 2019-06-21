from src.mnist_example import *


class TestResetParameterConv:

    @staticmethod
    def test_should_do_bla1():
        pass

    def test_should_do_bla2():
        pass


def test_baynet_init_vs_wild_init():

    mu1 = nn.Parameter(data=torch.Tensor(16, 1, 3, 3), requires_grad=True).to(device)
    bias1 = nn.Parameter(data=torch.Tensor(16), requires_grad=True).to(device)

    BayNet = ProbabilistClassifier(10).to(device)

    seed1 = set_and_print_random_seed(3616524511)
    reset_parameters_conv(mu1, bias1)
    set_and_print_random_seed(seed1)
    reset_parameters_conv(BayNet.mu1, BayNet.bias1)
