import torch


class BaseLoss:

    def __init__(self):
        self.writer = None
        self.tensorboard_idx = None
        self.results_idx = None
        self.total_loss = None
        self.current_epoch = None

    def compute(self):
        raise NotImplementedError

    def backward(self):
        assert type(self.total_loss) == torch.Tensor, 'Loss is not a torch.Tensor'
        self.total_loss.backward()

    def show(self):
        pass

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def init_results_writer(self, path):
        self.results_idx = 0

    def init_tensorboard_writer(self, path):
        self.tensorboard_idx = 0

    def write_tensorboard(self):
        pass

    def close_writer(self):
        pass


