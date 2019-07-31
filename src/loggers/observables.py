class Observables:

    def __init__(self):
        self.writer = None
        self.tensorboard_idx = None
        self.results_idx = None

    def compute_train(self):
        pass

    def compute_val(self):
        pass

    def show(self):
        pass

    def init_results_writer(self, path):
        self.results_idx = 0

    def init_tensorboard_writer(self, path):
        self.tensorboard_idx = 0

    def write_tensorboard(self):
        pass

    def close_writer(self):
        pass
