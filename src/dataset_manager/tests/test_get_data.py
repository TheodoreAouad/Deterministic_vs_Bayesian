import os
import shutil

import numpy as np
import pytest

from src.dataset_manager.get_data import get_mnist


@pytest.fixture()
def data_folder_teardown(request):
    def fin():
        absolute_path = os.getcwd()
        download_path = os.path.join(absolute_path, 'data')
        if os.path.isdir(download_path):
            shutil.rmtree(download_path)
    request.addfinalizer(fin)


#TODO: mock the MNIST to avoid downloading everything
class TestGetMnist:


    @staticmethod
    def test_label_specification(data_folder_teardown):
        train_labels = np.random.randint(0,10,5)
        test_labels = np.random.randint(0,10,5)

        trainloader, testloader = get_mnist(train_labels=train_labels, test_labels=test_labels)

        assert np.isin(trainloader.dataset.train_labels, train_labels).all()
        assert np.isin(testloader.dataset.test_labels, test_labels).all()
