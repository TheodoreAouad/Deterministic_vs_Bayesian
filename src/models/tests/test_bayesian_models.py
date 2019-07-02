import torch
from torch import nn
import torch.nn.functional as F

from src.models.bayesian_models import GaussianCNN, GaussianLinear
from src.utils import set_and_print_random_seed


class TestBayesianCNN:


    @staticmethod
    def test_determinist_init():
        seed1 = set_and_print_random_seed()
        DetCNN = nn.Conv2d(1, 16, 3)
        BayCNN = GaussianCNN(1, 1, 16, 3)
        set_and_print_random_seed(seed1)
        BayCNN.reset_parameters()

        assert torch.sum(torch.abs(BayCNN.mu - DetCNN.weight)) == 0
        assert torch.sum(torch.abs(BayCNN.mu_bias - DetCNN.bias)) == 0

    @staticmethod
    def test_determinist_forward():
        seed1 = set_and_print_random_seed()
        DetCNN = nn.Conv2d(1, 16, 3)
        BayCNN = GaussianCNN(1, 1, 16, 3)
        set_and_print_random_seed(seed1)
        BayCNN.reset_parameters()
        image = torch.rand(1, 1, 28, 28)
        output1 = BayCNN(image, determinist=True)
        output2 = DetCNN(image)

        assert torch.sum(torch.abs(output1 - output2)) == 0

    @staticmethod
    def test_bayesian_parameters_determinist():
        bay_cnn = GaussianCNN("determinist", 1, 16, 3)
        assert len(list(bay_cnn.bayesian_parameters())) == 0

    @staticmethod
    def test_bayesian_parameters_non_determinist():
        bay_cnn = GaussianCNN(1, 1, 16, 3)
        assert len(list(bay_cnn.bayesian_parameters())) == 4



class TestBayesianLinear:

    @staticmethod
    def test_determinist_init():
        seed1 = set_and_print_random_seed()
        DetLin = nn.Linear(32*7*7,10)
        BayLin = GaussianLinear(1, 32 * 7 * 7, 10)
        set_and_print_random_seed(seed1)
        BayLin.reset_parameters()

        assert torch.sum(torch.abs(BayLin.mu - DetLin.weight)) == 0
        assert torch.sum(torch.abs(BayLin.mu_bias - DetLin.bias)) == 0


    @staticmethod
    def test_determinist_forward():
        seed1 = set_and_print_random_seed()
        DetLin = nn.Linear(32*7*7, 10)
        BayLin = GaussianLinear(1, 32 * 7 * 7, 10)
        set_and_print_random_seed(seed1)
        BayLin.reset_parameters()
        image = torch.rand(1, 32*7*7)
        output1 = BayLin(image, determinist=True)
        output2 = DetLin(image)

        assert torch.sum(torch.abs(output1 - output2)) == 0

    @staticmethod
    def test_bayesian_parameters_determinist():
        bay_linear = GaussianLinear("determinist", 32, 10)
        assert len(list(bay_linear.bayesian_parameters())) == 0

    @staticmethod
    def test_bayesian_parameters_non_determinist():
        bay_linear = GaussianLinear(1, 32, 10)
        assert len(list(bay_linear.bayesian_parameters())) == 4

