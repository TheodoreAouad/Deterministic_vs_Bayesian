import numpy as np
import torch

from src.uncertainty_measures import compute_epistemic_uncertainty, compute_aleatoric_uncertainty


class TestComputeEpistemicUncertainty:

    @staticmethod
    def test_one_image():

        data = torch.rand(10, 10)
        # data = data/data.sum(1)[:, None]

        eu1 = compute_epistemic_uncertainty(data.unsqueeze(1))[0]

        res = data.mean(0)
        p_hat = data
        tmp = p_hat - res
        epistemic = torch.mm(tmp.transpose(0,1),tmp)/tmp.shape[0]
        eu2 = torch.sum(epistemic)

        assert torch.abs((eu1 - eu2)/eu2) < 0.0001

    @staticmethod
    def test_two_images():

        data = torch.randn(10, 2, 10)
        data1 = data[:, 0, :]
        data2 = data[:, 1, :]

        eus = compute_epistemic_uncertainty(data)

        res = data1.mean(0)
        p_hat = data1
        tmp = p_hat - res
        epistemic = torch.mm(tmp.transpose(0,1),tmp)/tmp.shape[0]
        eu1 = torch.sum(epistemic)

        res = data2.mean(0)
        p_hat = data2
        tmp = p_hat - res
        epistemic = torch.mm(tmp.transpose(0,1),tmp)/tmp.shape[0]
        eu2 = torch.sum(epistemic)

        assert torch.abs((eus[0] - eu1)/eu1) < 0.0001
        assert torch.abs((eus[1] - eu2)/eu2) < 0.0001


class TestComputeAleatoricUncertainty:

    @staticmethod
    def test_one_image():

        data = torch.rand(10, 10)
        # data = data/data.sum(1)[:, None]

        au1 = compute_aleatoric_uncertainty(data.unsqueeze(1))[0]

        res = data.mean(0)
        p_hat = data
        aleatoric = torch.diag(res) - torch.mm(p_hat.transpose(0,1), p_hat)/p_hat.shape[0]
        au2 = torch.sum(aleatoric)

        assert torch.abs((au1 - au2)/au2) < 0.0001

    @staticmethod
    def test_two_images():

        data = torch.randn(10, 2, 10)
        data1 = data[:, 0, :]
        data2 = data[:, 1, :]

        aus = compute_aleatoric_uncertainty(data)

        res = data1.mean(0)
        p_hat = data1
        aleatoric = torch.diag(res) - torch.mm(p_hat.transpose(0,1), p_hat)/p_hat.shape[0]
        au1 = torch.sum(aleatoric)

        res = data2.mean(0)
        p_hat = data2
        aleatoric = torch.diag(res) - torch.mm(p_hat.transpose(0,1), p_hat)/p_hat.shape[0]
        au2 = torch.sum(aleatoric)

        assert torch.abs((aus[0] - au1)/au1) < 0.0001
        assert torch.abs((aus[1] - au2)/au2) < 0.0001
