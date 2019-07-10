import os
import numpy as np

import torch
from torchvision.datasets import MNIST


class MNISTSpecificLabels(MNIST):

    def __init__(self, root, labels=range(10), train=True, transform=None, target_transform=None, download=False):
        super(MNISTSpecificLabels, self).__init__(root=root, train=train, transform=transform,
                                                  target_transform=target_transform, download=download)
        index_to_keep = torch.from_numpy(np.isin(self.targets, labels))
        self.data, self.targets = self.data[index_to_keep], self.targets[index_to_keep]

