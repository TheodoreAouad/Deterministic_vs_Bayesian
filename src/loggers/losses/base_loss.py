import torch

from src.loggers.logger import Logger


class BaseLoss(Logger):
    """
    Base class for losses. Losses should inherit this class.
    """
    def __init__(self):
        super(BaseLoss, self).__init__()

        self.logs = {'total_loss': None}

    @property
    def total_loss(self):
        return self.logs['total_loss']

    def compute(self, outputs, labels):
        raise NotImplementedError

    def backward(self):
        assert type(self.total_loss) == torch.Tensor, 'Loss type should be torch.Tensor'
        self.total_loss.backward()
