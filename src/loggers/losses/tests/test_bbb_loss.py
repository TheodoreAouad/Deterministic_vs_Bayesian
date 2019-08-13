import numpy as np
import torch.nn as nn
import torch.optim as optim

from src.tasks.trains import uniform, get_loss
from src.loggers.losses.bbb_loss import BBBLoss
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.tests.test_trains import RandomTrainloader
from src.utils import set_and_print_random_seed


class TestBBBLoss:

    @staticmethod
    def test_identity_of_direct_losses():
        device='cpu'
        randomloader = RandomTrainloader()

        bay_net = GaussianClassifier(-3, number_of_classes=10)
        bay_net.to(device)
        get_train_data = iter(randomloader)
        inputs, labels = next(get_train_data)
        outputs = bay_net(inputs)

        criterion = nn.CrossEntropyLoss()
        bbb_loss = BBBLoss(bay_net, criterion, uniform)

        number_of_batch = 100
        batch_idx = np.random.randint(100)
        kl_weight = uniform(batch_idx, number_of_batch)

        bbb_loss.set_number_of_epoch(1)
        bbb_loss.set_current_epoch(0)
        bbb_loss.set_number_of_batch(number_of_batch)
        bbb_loss.set_current_batch_idx(batch_idx)

        bbb_loss.compute(outputs, labels)
        total_loss_old, llh_old, vp_old, pr_old = get_loss(bay_net, 'bbb', outputs, labels, criterion, kl_weight)

        assert round(total_loss_old.item()) == round(bbb_loss.logs['total_loss'].item())
        assert round(llh_old.item()) == round(bbb_loss.logs['likelihood'].item())
        assert round(vp_old.item()*kl_weight) == round(bbb_loss.logs['variational_posterior'].item())
        assert round(pr_old.item()*kl_weight) == round(bbb_loss.logs['prior'].item())

    @staticmethod
    def test_identity_of_optimizer_step():
        device = 'cpu'
        randomloader = RandomTrainloader()

        get_train_data = iter(randomloader)
        inputs, labels = next(get_train_data)
        number_of_batch = 100
        batch_idx = np.random.randint(100)

        seed1 = set_and_print_random_seed()
        bay_net_old = GaussianClassifier(-3, number_of_classes=10)
        bay_net_old.to(device)
        optimizer = optim.Adam(bay_net_old.parameters())
        criterion = nn.CrossEntropyLoss()
        outputs = bay_net_old(inputs)
        kl_weight = uniform(batch_idx, number_of_batch)
        total_loss_old, llh_old, vp_old, pr_old = get_loss(bay_net_old, 'bbb', outputs, labels, criterion, kl_weight)
        total_loss_old.backward()
        optimizer.step()
        outputs_old_final = bay_net_old(inputs)

        set_and_print_random_seed(seed1)
        bay_net_new = GaussianClassifier(-3, number_of_classes=10)
        bay_net_new.to(device)
        optimizer = optim.Adam(bay_net_new.parameters())
        criterion = nn.CrossEntropyLoss()
        outputs = bay_net_new(inputs)
        bbb_loss = BBBLoss(bay_net_new, criterion, uniform)

        bbb_loss.set_number_of_epoch(1)
        bbb_loss.set_current_epoch(0)
        bbb_loss.set_number_of_batch(number_of_batch)
        bbb_loss.set_current_batch_idx(batch_idx)
        bbb_loss.compute(outputs, labels)
        bbb_loss.backward()
        optimizer.step()
        outputs_new_final = bay_net_new(inputs)

        assert (outputs_new_final - outputs_old_final).sum() == 0





