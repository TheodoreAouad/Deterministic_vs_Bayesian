import torch

from src.loggers.logger import Logger


class BaseLoss(Logger):
    """
    Base class for losses. Losses should inherit this class.
    """
    def __init__(self, criterion):
        super(BaseLoss, self).__init__()
        self.criterion = criterion
        self.logs = {'total_loss': None}

    def compute(self, outputs, labels):
        self.logs['total_loss'] = self.criterion(outputs, labels)

    def backward(self):
        assert type(self.logs['total_loss']) == torch.Tensor, 'Loss type should be torch.Tensor'
        self.logs['total_loss'].backward()
