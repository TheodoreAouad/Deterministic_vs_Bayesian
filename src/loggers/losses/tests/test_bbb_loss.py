import numpy as np
import torch.nn as nn
import torch.optim as optim

from src.tasks.trains import uniform
from src.loggers.losses.bbb_loss import BBBLoss
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.dataset_manager.get_data import RandomLoader
from src.utils import set_and_print_random_seed


class TestBBBLoss:

    @staticmethod
    def test_identity_of_direct_losses():
        device='cpu'
        randomloader = RandomLoader()

        bay_net = GaussianClassifier(-3, number_of_classes=10)
        bay_net.to(device)
        get_train_data = iter(randomloader)
        inputs, labels = next(get_train_data)
        outputs = bay_net(inputs)

        criterion = nn.CrossEntropyLoss()
        bbb_loss = BBBLoss(bay_net, criterion, uniform)

        number_of_batch = 100
        batch_idx = np.random.randint(100)

        bbb_loss.set_number_of_epoch(1)
        bbb_loss.set_current_epoch(0)
        bbb_loss.set_number_of_batch(number_of_batch)
        bbb_loss.set_current_batch_idx(batch_idx)

        bbb_loss.compute(outputs, labels)

    @staticmethod
    def test_identity_of_optimizer_step():
        device = 'cpu'
        randomloader = RandomLoader()

        get_train_data = iter(randomloader)
        inputs, labels = next(get_train_data)
        number_of_batch = 100
        batch_idx = np.random.randint(100)

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






