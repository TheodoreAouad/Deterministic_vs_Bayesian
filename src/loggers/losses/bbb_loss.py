from src.loggers.losses.base_loss import BaseLoss


class BBBLoss(BaseLoss):
    """
    Loss Bayes By Backprop. See paper 'Weight Uncertainty in Neural Networks' (2015, Blundell et al.) for more details.
    """

    def __init__(self, model, criterion, step_function, ):
        """

        Args:
            model (torch.module.nn child):
            criterion (function): takes 2 inputs, prediction, target, and gives a scalar
            step_function (function): takes 2 inputs, current batch idx, nb of batch, and gives a scalar
        """
        super(BBBLoss, self).__init__(criterion)
        self.model = model
        self.criterion = criterion
        self.step_function = step_function

        self.logs = {
            'total_loss': None,
            'likelihood': None,
            'variational_posterior': None,
            'prior': None,
        }

    # TODO: do we have to divide the loss by the batch size?
    def compute(self, outputs, labels):
        """
        Compute the loss L = kl_weight * KL[q(w | theta) || P(w)] - E_q(w)(logP(D|W))
        Args:
            outputs (torch.Tensor): size = (batch_size, number_of_classes): output of the model,
            will be the output of the softmax layer
            labels (torch.Tensor): size = (batch_size) True labels
        """
        kl_weight = self.step_function(self.current_batch_idx, self.number_of_batch)
        weights_used, bias_used = self.model.get_previous_weights()
        batch_size = outputs.size(0)

        self.logs['likelihood'] = self.criterion(outputs, labels)
        self.logs['variational_posterior'] = kl_weight * self.model.variational_posterior(weights_used,
                                                                                          bias_used)
        self.logs['prior'] = -kl_weight * self.model.prior(weights_used, bias_used)
        self.logs['total_loss'] = (self.logs['variational_posterior'] + self.logs['prior'] +
                                   self.logs['likelihood'])
        self.add_to_history()

    def write_tensorboard(self):
        board_names = {key: 'loss' for key in self.logs.keys()}
        super(BBBLoss, self).write_tensorboard(**board_names)
