import os
import shutil

import numpy as np
import pytest

from src.dataset_manager.get_data import get_mnist


@pytest.fixture(scope='class')
def data_folder_teardown(request):
    def fin():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        download_path = os.path.join(dir_path, 'data')
        if os.path.isdir(download_path):
            shutil.rmtree(download_path)
            print(f'{download_path} deleted.')
    request.addfinalizer(fin)


#TODO: mock the MNIST to avoid downloading everything
class TestGetMnist:


    @staticmethod
    def test_label_specification(data_folder_teardown):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.join(dir_path, 'data')
        train_labels = np.random.randint(0, 10, 5)
        eval_labels = np.random.randint(0, 10, 5)

        trainloader, valloader, evalloader = get_mnist(root=dir_path, train_labels=train_labels, split_val=0,
                                                       eval_labels=eval_labels)

        assert np.isin(trainloader.dataset.targets, train_labels).all()
        assert np.isin(evalloader.dataset.targets, eval_labels).all()

    @staticmethod
    def test_validation_split(data_folder_teardown):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.join(dir_path, 'data')

        trainloader, valloader, evalloader = get_mnist(root=dir_path, train_labels=range(6), eval_labels=range(6),
                                                       split_val=0)
        size_full_test = len(evalloader.dataset)
        trainloader, valloader, evalloader = get_mnist(root=dir_path, train_labels=range(6), eval_labels=range(6),
                                                               split_val=0.5)
        assert size_full_test == len(valloader.dataset) + len(evalloader.dataset)

