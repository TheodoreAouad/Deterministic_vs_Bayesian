from src.loggers.logger import Logger
from src.uncertainty_measures import get_all_uncertainty_measures


class Observables(Logger):

    def compute_train_on_batch(self, outputs, labels):
        raise NotImplementedError

    def compute_train_on_epoch(self, model, trainloader, device):
        raise NotImplementedError

    def compute_val(self, val_accuracy, val_outputs):
        raise NotImplementedError


class AccuracyAndUncertainty(Observables):
    """
    Logger to store accuracies and uncertainties
    """

    def __init__(self):
        super(AccuracyAndUncertainty, self).__init__()
        self.logs = {
            'train_accuracy_on_batch': None,
            'train_accuracy_on_epoch': 0,
            'val_accuracy': None,
            'val_uncertainty_vr': None,
            'val_uncertainty_pe': None,
            'val_uncertainty_mi': None,
        }
        # self.logs_history = {
        #     'train_accuracy_on_batch': None,
        #     'train_accuracy_on_epoch': None,
        #     'val_accuracy': None,
        #     'val_uncertainty_vr': None,
        #     'val_uncertainty_pe': None,
        #     'val_uncertainty_mi': None,
        # }
        self.max_train_accuracy_on_epoch = 0
        self.epoch_with_max_train_accuracy = 0
        self.validation_logging = False

    def compute_train_on_batch(self, outputs, labels):
        """
        Logs we want to compute for each batch on train
        Args:
            outputs (torch.Tensor): size = (batch_size, number_of_classes): output of the model
            labels (torch.Tensor): size = (batch_size): true labels
        """
        predicted_labels = outputs.argmax(1)
        self.logs['train_accuracy_on_batch'] = (predicted_labels - labels == 0).sum().item() / labels.size(0)
        self.add_to_history(['train_accuracy_on_batch'])

    def compute_train_on_epoch(self, model, trainloader, device):
        """
        Logs we want to compute for each epoch on train
        Args:
            model (torch.nn.Module Child): model being trained
            trainloader (torch.utils.data.dataloader.DataLoader): dataloader of the train set
            device (torch.device || str): which device to compute on (either on GPU or CPU). Either torch.device type or
                                      specific string 'cpu' or 'gpu'. Will be the same device as the model.
        """
        number_of_correct_labels = 0
        number_of_labels = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            predicted_labels = model(inputs).argmax(1)
            number_of_correct_labels += (predicted_labels - labels == 0).sum().item()
            number_of_labels += labels.size(0)
        self.logs['train_accuracy_on_epoch'] = number_of_correct_labels / number_of_labels
        self.add_to_history(['train_accuracy_on_epoch'])
        if self.logs['train_accuracy_on_epoch'] > self.max_train_accuracy_on_epoch:
            self.max_train_accuracy_on_epoch = self.logs['train_accuracy_on_epoch']
            self.epoch_with_max_train_accuracy = self.current_epoch

    def compute_val(self, val_accuracy, val_outputs):
        """
        Logs we want to keep on the validation set
        Args:
            val_accuracy (float): accuracy on the validation set
            val_outputs (torch.Tensor): size = (number_of_tests, batch_size, number_of_classes):
            output of the evaluation on the validation set
        """
        if not self.validation_logging:
            self.validation_logging = True

        self.logs['val_accuracy'] = val_accuracy
        (
            self.logs['val_uncertainty_vr'],
            self.logs['val_uncertainty_pe'],
            self.logs['val_uncertainty_mi'],
        ) = get_all_uncertainty_measures(val_outputs)
        self.add_to_history([
            'val_accuracy',
            'val_uncertainty_vr',
            'val_uncertainty_pe',
            'val_uncertainty_mi',
        ])

        for key in ['val_uncertainty_vr', 'val_uncertainty_pe', 'val_uncertainty_mi']:
            self.logs[key] = self.logs[key].mean()

    def write_tensorboard(self):
        board_names = {key: 'accuracy' for key in self.logs.keys() if 'accuracy' in key}
        super(AccuracyAndUncertainty, self).write_tensorboard(**board_names)
