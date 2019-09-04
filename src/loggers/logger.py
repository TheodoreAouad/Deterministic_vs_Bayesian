import pathlib
import os
from collections import defaultdict

import torch

from torch.utils.tensorboard import SummaryWriter

from src.utils import print_nicely_on_console


class Logger:
    """
    This is a parent class to objects used to track information during training.
    """
    logs_history = defaultdict(list)

    def __init__(self):
        """
            self.writer (torch.utils.tensorboard.writer.SummaryWriter): torch object to write tensorboard
            self.tensorboard_idx (int): index of the tensorboards
            self.results_idx (int): Not implemented yet
            self.current_epoch (int): current epoch of training
            self.number_of_epoch (int): number of epochs in training
            self.current_batch_idx (int): current index of the training batch
            self.number_of_batch (int): number of batches for the training
            self.logs (dict): the current state of variables we want to keep logs of
            self.logs_history (dict): all the previous states of variables we want to log
            self.output_dir_results (str): output directory to store results (Not Implemented)
            self.output_dir_tensorboard (str): output directory to store tensorboards
        """
        self.writer = None
        self.tensorboard_idx = None
        self.results_idx = None
        self.current_epoch = None
        self.number_of_epoch = None
        self.current_batch_idx = None
        self.number_of_batch = None
        # self.logs = {}
        self.logs_history = {}
        self.output_dir_results = None
        self.output_dir_tensorboard = None

    def show(self):
        print_nicely_on_console(self.logs)

    def set_number_of_epoch(self, number_of_epoch):
        self.number_of_epoch = number_of_epoch
        for key in self.logs.keys():
            self.logs_history[key] = []

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch
        for key in self.logs_history.keys():
            self.logs_history[key].append([])

    def set_current_batch_idx(self, batch_idx):
        self.current_batch_idx = batch_idx

    def set_number_of_batch(self, number_of_batch):
        self.number_of_batch = number_of_batch

    def add_to_history(self, specific_keys=None):
        keys_to_add_to_history = specific_keys or self.logs.keys()
        for key in keys_to_add_to_history:
            self.logs_history[key][self.current_epoch].append(
                torch.tensor(self.logs[key]).detach()
            )

    def init_results_writer(self, path):
        """
        NOT IMPLEMENTED
        """
        pass

    def init_tensorboard_writer(self, path=None):
        if path is None:
            return

        self.tensorboard_idx = 0
        self.output_dir_tensorboard = pathlib.Path(path)
        self.output_dir_tensorboard.mkdir(parents=True, exist_ok=True)
        self.writer = {}
        for key in self.logs.keys():
            self.writer[key] = SummaryWriter(log_dir=self.output_dir_tensorboard / key)

    def write_tensorboard(self, **kwargs):
        """

        Args:
            **kwargs (dict): tell on which tensorboard to write each key.

        Returns:

        """
        if self.output_dir_tensorboard is None:
            return

        for key in self.writer.keys():
            self.writer[key].add_scalar(
                kwargs.get(key, key),
                self.logs[key],
                self.tensorboard_idx,
            )
        self.tensorboard_idx += 1

    def close_writer(self):
        if self.output_dir_tensorboard is None:
            return

        for key in self.writer.keys():
            self.writer[key].close()

    def results(self):
        """
        NOT IMPLEMENTED
        """
        pass

    def write_results(self):
        """
        NOT IMPLEMENTED
        """
        pass
