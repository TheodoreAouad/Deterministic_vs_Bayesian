import os
from time import time

import torch
from torch.utils.tensorboard import SummaryWriter

from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_all_uncertainty_measures


def train(*args, **kwargs):
    return train_bayesian_refactored(
        *args,
        number_of_tests=1,
        loss_type='criterion',
        step_function=uniform,
        **kwargs,
    )


def uniform(_, number_of_batchs):
    return 1 / number_of_batchs


class TrainBayesian:

    def __init__(
            self,
            model,
            optimizer,
            criterion,
            number_of_epochs,
            trainloader,
            valloader=None,
            number_of_tests=10,
            loss_type='bbb',
            step_function=uniform,
            output_dir_tensorboard=None,
            output_dir_weights_results=None,
            device="cpu",
            verbose=False,
    ):
        """
        Train the model in a bayesian fashion, meaning the loss is different. How to use: init the model with
        the parameters. Then, use the __call__ method to train the model and return the output.
        Args:
            model (Torch.nn.Module child): the model we want to train
            optimizer (torch.optim optimizer): how do we update the weights
            criterion (function): how do we compute the likelihood
            number_of_epochs (int): how long do we train our model
            trainloader (torch.utils.data.dataloader.DataLoader): train data
            loss_type (str): which type of loss. "bbb" (Bayes By Backprop) or "criterion" (CrossEntropy)
            step_function (function): takes as args (number of batchs, length of batch) and returns the weight to give to KL
            output_dir_tensorboard (str): output directory in which to save the tensorboard
            device (torch.device || str): cpu or gpu
            verbose (Bool): print training steps or not

        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.number_of_epochs = number_of_epochs
        self.trainloader = trainloader
        self.valloader = valloader
        self.number_of_tests = number_of_tests
        self.loss_type = loss_type
        self.step_function = step_function
        self.output_dir_tensorboard = output_dir_tensorboard
        self.output_dir_weights_results = output_dir_weights_results
        self.device = device
        self.verbose = verbose
        self.number_of_batch = len(self.trainloader)

        # used during training
        self.weights_writer_idx = None
        self.tensorboard_idx = None
        self.writers = None
        self.val_acc = None
        self.val_vr = None
        self.val_predictive_entropy = None
        self.val_mi = None
        self.running_loss = None
        self.running_loss_llh = None
        self.running_loss_vp = None
        self.running_loss_pr = None
        self.number_of_correct_labels = None
        self.number_of_labels = None
        self.current_loss = None
        self.current_loss_llh = None
        self.current_loss_vp = None
        self.current_loss_pr = None
        self.current_train_acc = None

        # outputs
        self.loss_totals = None
        self.loss_llhs = None
        self.loss_vps = None
        self.loss_prs = None
        self.train_accs = None
        self.max_acc = None
        self.epoch_max_acc = None
        self.batch_idx_max_acc = None
        self.val_accs = None
        self.val_vrs = None
        self.val_predictive_entropies = None
        self.val_mis = None

    def fit(self):
        start_time = time()

        if self.output_dir_tensorboard is not None:
            self.tensorboard_idx = 0
            self.init_writers()
        self.init_save_weight_results()

        interval = max(self.number_of_batch // 10, 1)

        self.init_outputs()
        self.init_intermediate_values()

        self.model.train()
        for epoch in range(self.number_of_epochs):  # loop over the dataset multiple times

            self.reset_running_losses()
            self.number_of_correct_labels = 0
            self.number_of_labels = 0

            for batch_idx, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = [x.to(self.device) for x in data]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)

                kl_weight = self.step_function(batch_idx, self.number_of_batch)
                loss, loss_likelihood, loss_varational_posterior, loss_prior = self.get_loss(
                    outputs,
                    labels,
                    kl_weight
                )
                loss.backward()
                self.optimizer.step()
                self.update_running_losses(loss, loss_likelihood, loss_prior, loss_varational_posterior)
                predicted_labels = outputs.argmax(1)
                self.number_of_correct_labels += torch.sum(predicted_labels - labels == 0).item()
                self.number_of_labels += labels.size(0)

                if batch_idx % interval == interval - 1:
                    if self.valloader is not None:
                        self.val_acc, val_output = eval_bayesian(
                            self.model,
                            self.valloader,
                            number_of_tests=self.number_of_tests,
                            device=self.device,
                            val=True
                        )
                        self.val_vr, self.val_predictive_entropy, self.val_mi = get_all_uncertainty_measures(val_output)
                    self.set_current_losses()
                    self.current_train_acc = self.number_of_correct_labels / self.number_of_labels
                    if self.max_acc < self.current_train_acc:
                        self.max_acc = self.current_train_acc
                        self.epoch_max_acc = epoch
                        self.batch_idx_max_acc = batch_idx
                    if self.verbose:
                        self.show_progress_on_console(batch_idx, epoch, start_time)

                    self.update_output(epoch)

                    if self.output_dir_tensorboard is not None:
                        self.write_tensorboard()

                    self.save_weight_results()
                    self.reset_running_losses()

        if self.output_dir_tensorboard is not None:
            for writer in self.writers.value():
                writer.close()
        print('Finished Training')

    def update_running_losses(self, loss, loss_likelihood, loss_prior, loss_varational_posterior):
        self.running_loss += loss.item()
        if self.loss_type == 'bbb':
            self.running_loss_llh += loss_likelihood.item()
            self.running_loss_vp += loss_varational_posterior.item()
            self.running_loss_pr += loss_prior.item()

    def write_tensorboard(self):
        self.writers['loss'].add_scalar('loss', self.current_loss, self.tensorboard_idx)
        if self.loss_type == 'bbb':
            self.writers['loss_llh'].add_scalar('loss', self.current_loss_llh, self.tensorboard_idx)
            self.writers['loss_vp'].add_scalar('loss', self.current_loss_vp, self.tensorboard_idx)
            self.writers['loss_pr'].add_scalar('loss', self.current_loss_pr, self.tensorboard_idx)
        self.writers['vr'].add_scalar('variation ratio', self.val_vr.mean().item(), self.tensorboard_idx)
        self.writers['pe'].add_scalar('predictive entropy', self.val_predictive_entropy.mean().item(),
                                      self.tensorboard_idx)
        self.writers['mi'].add_scalar('mutual information', self.val_mi.mean().item(),
                                      self.tensorboard_idx)
        self.writers['accs_train'].add_scalar('accuracy', self.current_train_acc, self.tensorboard_idx)
        self.writers['accs_val'].add_scalar('accuracy', self.val_acc, self.tensorboard_idx)
        self.tensorboard_idx += 1

    def update_output(self, epoch):
        self.loss_totals[epoch].append(self.current_loss)
        if self.loss_type == 'bbb':
            self.loss_llhs[epoch].append(self.current_loss_llh)
            self.loss_vps[epoch].append(self.current_loss_vp)
            self.loss_prs[epoch].append(self.current_loss_pr)
        self.val_vrs[epoch].append(self.val_vr)
        self.val_predictive_entropies[epoch].append(self.val_predictive_entropy)
        self.val_mis[epoch].append(self.val_mi)
        self.train_accs[epoch].append(self.current_train_acc)
        self.val_accs[epoch].append(self.val_acc)

    def show_progress_on_console(self, batch_idx, epoch, start_time):
        if self.loss_type == 'bbb':
            print(f'Train: [{epoch + 1}, {batch_idx + 1}/{self.number_of_batch}] '
                  f'Acc: {round(100 * self.current_train_acc, 2)} %, '
                  f'loss: {round(self.current_loss, 2)}, '
                  f'loss_llh: {round(self.current_loss_llh, 2)}, '
                  f'loss_vp: {round(self.current_loss_vp, 2)}, '
                  f'loss_pr: {round(self.current_loss_pr, 2)}, '
                  f'Val Acc: {round(100 * self.val_acc, 2)} %, '
                  f'Val VR: {self.val_vr.mean().item()}, '
                  f'Val Pred Entropy: {self.val_predictive_entropy.mean().item()}, '
                  f'Val MI: {self.val_mi.mean().item()}, '
                  f'Time Elapsed: {round(time() - start_time)} s')
        else:
            print(f'Train: [{epoch + 1}, {batch_idx + 1}/{self.number_of_batch}] '
                  f'Acc: {round(100 * self.current_train_acc, 2)} %, '
                  f'loss: {round(self.current_loss, 2)}'
                  f'Val Acc: {round(100 * self.val_acc, 2)} %, '
                  f'Val VR: {self.val_vr.mean().item()}, '
                  f'Val Pred Entropy: {self.val_predictive_entropy.mean().item()}, '
                  f'Val MI: {self.val_mi.mean().item()}, '
                  f'Time Elapsed: {round(time() - start_time)} s')

    def set_current_losses(self):
        self.current_loss = self.running_loss / self.number_of_batch
        if self.loss_type == 'bbb':
            self.current_loss_llh = self.running_loss_llh / self.number_of_batch
            self.current_loss_vp = self.running_loss_vp / self.number_of_batch
            self.current_loss_pr = self.running_loss_pr / self.number_of_batch

    def reset_running_losses(self):
        self.running_loss = 0.0
        if self.loss_type == 'bbb':
            self.running_loss_llh = 0.0
            self.running_loss_vp = 0.0
            self.running_loss_pr = 0.0

    def init_intermediate_values(self):
        self.val_acc = -0.01
        self.val_vr = torch.tensor(-1).float()
        self.val_predictive_entropy = torch.tensor(-1).float()
        self.val_mi = torch.tensor(-1).float()

    def init_outputs(self):
        self.loss_totals = [[] for _ in range(self.number_of_epochs)]
        if self.loss_type == 'bbb':
            self.loss_llhs = [[] for _ in range(self.number_of_epochs)]
            self.loss_vps = [[] for _ in range(self.number_of_epochs)]
            self.loss_prs = [[] for _ in range(self.number_of_epochs)]
        self.train_accs = [[] for _ in range(self.number_of_epochs)]
        self.max_acc = -1
        self.epoch_max_acc = 0
        self.batch_idx_max_acc = 0
        self.val_accs = [[] for _ in range(self.number_of_epochs)]
        self.val_vrs = [[] for _ in range(self.number_of_epochs)]
        self.val_predictive_entropies = [[] for _ in range(self.number_of_epochs)]
        self.val_mis = [[] for _ in range(self.number_of_epochs)]

    def save_weight_results(self):
        if self.output_dir_weights_results is not None:
            file_idx = 'weight' + str(self.weights_writer_idx) + '.pt'
            torch.save(self.model.state_dict(), os.path.join(self.output_dir_weights_results, file_idx))
            self.weights_writer_idx += 1

    def init_save_weight_results(self):
        if self.output_dir_weights_results is not None:
            self.weights_writer_idx = 0
            if not os.path.exists(self.output_dir_weights_results):
                os.mkdir(self.output_dir_weights_results)

    def __call__(self):
        """
        Returns:
            loss_totals (list): list of the total loss for each epoch
            loss_llhs (list): list of the likelihood loss for each epoch (P(D|W))
            loss_vps (list): list of the variational posterior loss for each epoch (q(W|D))
            loss_prs (list): list of the prior loss for each epoch (P(W))
            train_accs (list): list of the accuracies for each epoch
            max_acc (float): the maximum accuracy obtained by the net
            epoch_max_acc (int): the epoch where the max acc is obtained
            batch_idx_max_acc (int): the batch_idx where the epoch is obtained
            val_accs (list): list of the validation accuracies
            val_vrs (list): list of the variation-ratios uncertainty for validation test
            val_predictive_entropies (list): list of the predictive entropy uncertainty for validation test
            val_mis (list): list of the mutual information uncertainty for validation test
        """
        self.fit()
        return (
            self.loss_totals,
            self.loss_llhs,
            self.loss_vps,
            self.loss_prs,
            self.train_accs,
            self.max_acc,
            self.epoch_max_acc,
            self.batch_idx_max_acc,
            self.val_accs,
            self.val_vrs,
            self.val_predictive_entropies,
            self.val_mis
        )

    def init_writers(self):

        self.writers = {
            'loss': SummaryWriter(log_dir=os.path.join(self.output_dir_tensorboard, 'total_loss')),
            'loss_llh': None,
            'loss_vp': None,
            'loss_pr': None,
            'vr': SummaryWriter(log_dir=os.path.join(self.output_dir_tensorboard, 'variation ratio')),
            'pe': SummaryWriter(log_dir=os.path.join(self.output_dir_tensorboard, 'predictive entropy')),
            'mi': SummaryWriter(log_dir=os.path.join(self.output_dir_tensorboard, "mutual information")),
            'accs_train': SummaryWriter(log_dir=os.path.join(self.output_dir_tensorboard, "train_accuracy")),
            'accs_val': SummaryWriter(log_dir=os.path.join(self.output_dir_tensorboard, "val_accuracy"))
        }

        if self.loss_type == 'bbb':
            self.writers['loss_llh'] = SummaryWriter(log_dir=os.path.join(self.output_dir_tensorboard, "loss_llh"))
            self.writers['loss_vp'] = SummaryWriter(log_dir=os.path.join(self.output_dir_tensorboard, "loss_vp"))
            self.writers['loss_pr'] = SummaryWriter(log_dir=os.path.join(self.output_dir_tensorboard, "loss_pr"))

    def get_loss(self, outputs, labels, kl_weight):
        """
        Returns the loss of the model
        Args:
            outputs:
            labels:
            kl_weight

        Returns:
            loss (torch.Float) :
            loss_likelihood (torch.Float) :
            loss_variational_posterior (torch.Float) :
            loss_prior (torch.Float) :

        """
        loss_likelihood = self.criterion(outputs, labels)
        if self.loss_type == 'bbb':
            weights_used, bias_used = self.model.get_previous_weights()
            loss_variational_posterior = self.model.variational_posterior(weights_used, bias_used)
            loss_prior = -self.model.prior(weights_used, bias_used)
            loss = kl_weight * (loss_variational_posterior + loss_prior) + loss_likelihood
            return loss, loss_likelihood, loss_variational_posterior, loss_prior
        elif self.loss_type == 'criterion':
            loss = loss_likelihood
            return loss, loss_likelihood, None, None
        else:
            raise ValueError('Loss must be either "bbb" for Bayes By Backprop,'
                             'or "criterion" for CrossEntropy. No other loss is implemented.')


def train_bayesian_refactored(model, optimizer, criterion, number_of_epochs, trainloader, valloader=None,
                              number_of_tests=10,
                              loss_type='bbb', step_function=uniform, output_dir_tensorboard=None,
                              output_dir_results=None,
                              device="cpu", verbose=False):
    trainer = TrainBayesian(model, optimizer, criterion, number_of_epochs, trainloader, valloader, number_of_tests,
                            loss_type, step_function, output_dir_tensorboard, output_dir_results,
                            device, verbose)
    return trainer()


def train_bayesian(
        model,
        optimizer,
        criterion,
        number_of_epochs,
        trainloader,
        valloader=None,
        number_of_tests=10,
        loss_type='bbb',
        step_function=uniform,
        output_dir_tensorboard=None,
        output_dir_results=None,
        device='cpu',
        verbose=False,
):
    """
    Train the model in a bayesian fashion, meaning the loss is different.
    Args:
        model (Torch.nn.Module child): the model we want to train
        optimizer (torch.optim optimizer): how do we update the weights
        criterion (function): how do we compute the likelihood
        number_of_epochs (int): how long do we train our model
        trainloader (torch.utils.data.dataloader.DataLoader): train data
        loss_type (str): which type of loss. "bbb" (Bayes By Backprop) or "criterion" (CrossEntropy)
        step_function (function): takes as args (number of batchs, length of batch) and returns the weight to give to KL
        output_dir_tensorboard (str): output directory in which to save the tensorboard
        device (torch.device || str): cpu or gpu
        verbose (Bool): print training steps or not

    Returns:
        loss_totals (list): list of the total loss for each epoch
        loss_llhs (list): list of the likelihood loss for each epoch (P(D|W))
        loss_vps (list): list of the variational posterior loss for each epoch (q(W|D))
        loss_prs (list): list of the prior loss for each epoch (P(W))
        train_accs (list): list of the accuracies for each epoch
        max_acc (float): the maximum accuracy obtained by the net
        epoch_max_acc (int): the epoch where the max acc is obtained
        batch_idx_max_acc (int): the batch_idx where the epoch is obtained
        val_accs (list): list of the validation accuracies
        val_vrs (list): list of the variation-ratios uncertainty for validation test
        val_predictive_entropies (list): list of the predictive entropy uncertainty for validation test
        val_mis (list): list of the mutual information uncertainty for validation test

    """
    start_time = time()

    if output_dir_tensorboard is not None:
        (writer_loss, writer_loss_llh, writer_loss_vp, writer_loss_pr, writer_accs_train, writer_accs_val, writer_vr,
         writer_predictive_entropy, writer_mi) = get_loss_writers(output_dir_tensorboard, loss_type)
        tensorboard_idx = 0
    if output_dir_results is not None:
        weights_writer_idx = 0
        os.makedirs(output_dir_results, exist_ok=True)

    max_acc = -1
    number_of_batch = len(trainloader)
    interval = max(number_of_batch // 10, 1)

    loss_totals = [[] for _ in range(number_of_epochs)]
    if loss_type == 'bbb':
        loss_llhs = [[] for _ in range(number_of_epochs)]
        loss_vps = [[] for _ in range(number_of_epochs)]
        loss_prs = [[] for _ in range(number_of_epochs)]
    train_accs = [[] for _ in range(number_of_epochs)]
    val_accs = [[] for _ in range(number_of_epochs)]
    val_vrs = [[] for _ in range(number_of_epochs)]
    val_predictive_entropies = [[] for _ in range(number_of_epochs)]
    val_mis = [[] for _ in range(number_of_epochs)]
    val_acc = -0.01
    val_vr = torch.tensor(-1).float()
    val_predictive_entropy = torch.tensor(-1).float()
    val_mi = torch.tensor(-1).float()

    model.train()
    for epoch in range(number_of_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        if loss_type == 'bbb':
            running_loss_llh = 0.0
            running_loss_vp = 0.0
            running_loss_pr = 0.0
        number_of_correct_labels = 0
        number_of_labels = 0

        for batch_idx, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = [x.to(device) for x in data]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            kl_weight = step_function(batch_idx, number_of_batch)
            loss, loss_likelihood, loss_variational_posterior, loss_prior = get_loss(
                model,
                loss_type,
                outputs,
                labels,
                criterion,
                kl_weight,
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if loss_type == 'bbb':
                running_loss_llh += loss_likelihood.item()
                running_loss_vp += loss_variational_posterior.item()
                running_loss_pr += loss_prior.item()
            predicted_labels = outputs.argmax(1)
            number_of_correct_labels += torch.sum(predicted_labels - labels == 0).item()
            number_of_labels += labels.size(0)

            if batch_idx % interval == interval - 1:
                if valloader is not None:
                    val_acc, val_outputs = eval_bayesian(model, valloader, number_of_tests=number_of_tests,
                                                         device=device, val=True)
                    val_vr, val_predictive_entropy, val_mi = get_all_uncertainty_measures(val_outputs)
                current_loss = running_loss / number_of_batch
                if loss_type == 'bbb':
                    current_loss_llh = running_loss_llh / number_of_batch
                    current_loss_vp = running_loss_vp / number_of_batch
                    current_loss_pr = running_loss_pr / number_of_batch
                current_train_acc = number_of_correct_labels / number_of_labels
                if max_acc < current_train_acc:
                    max_acc = current_train_acc
                    epoch_max_acc = epoch
                    batch_idx_max_acc = batch_idx
                if verbose:
                    if loss_type == 'bbb':
                        print(f'Train: [{epoch + 1}, {batch_idx + 1}/{number_of_batch}] '
                              f'Acc: {round(100 * current_train_acc, 2)} %, '
                              f'loss: {round(current_loss, 2)}, '
                              f'loss_llh: {round(current_loss_llh, 2)}, '
                              f'loss_vp: {round(current_loss_vp, 2)}, '
                              f'loss_pr: {round(current_loss_pr, 2)}, '
                              f'Val Acc: {round(100 * val_acc, 2)} %, '
                              f'Val VR: {val_vr.mean().item()}, '
                              f'Val Pred Entropy: {val_predictive_entropy.mean().item()}, '
                              f'Val MI: {val_mi.mean().item()}, '
                              f'Time Elapsed: {round(time() - start_time)} s')
                    else:
                        print(f'Train: [{epoch + 1}, {batch_idx + 1}/{number_of_batch}] '
                              f'Acc: {round(100 * current_train_acc, 2)} %, '
                              f'loss: {round(current_loss, 2)}'
                              f'Val Acc: {round(100 * val_acc, 2)} %, '
                              f'Val VR: {val_vr.mean().item()}, '
                              f'Val Pred Entropy: {val_predictive_entropy.mean().item()}, '
                              f'Val MI: {val_mi.mean().item()}, '
                              f'Time Elapsed: {round(time() - start_time)} s')

                loss_totals[epoch].append(current_loss)
                if loss_type == 'bbb':
                    loss_llhs[epoch].append(current_loss_llh)
                    loss_vps[epoch].append(current_loss_vp)
                    loss_prs[epoch].append(current_loss_pr)
                val_vrs[epoch].append(val_vr)
                val_predictive_entropies[epoch].append(val_predictive_entropy)
                val_mis[epoch].append(val_mi)
                train_accs[epoch].append(current_train_acc)
                val_accs[epoch].append(val_acc)

                if output_dir_tensorboard is not None:
                    writer_loss.add_scalar('loss', current_loss, tensorboard_idx)
                    if loss_type == 'bbb':
                        writer_loss_llh.add_scalar('loss', current_loss_llh, tensorboard_idx)
                        writer_loss_vp.add_scalar('loss', current_loss_vp, tensorboard_idx)
                        writer_loss_pr.add_scalar('loss', current_loss_pr, tensorboard_idx)
                    writer_vr.add_scalar('variation ratio', val_vr.mean().item(), tensorboard_idx)
                    writer_predictive_entropy.add_scalar('predictive entropy', val_predictive_entropy.mean().item(),
                                                         tensorboard_idx)
                    writer_mi.add_scalar('mutual information', val_mi.mean().item(), tensorboard_idx)
                    writer_accs_train.add_scalar('accuracy', current_train_acc, tensorboard_idx)
                    writer_accs_val.add_scalar('accuracy', val_acc, tensorboard_idx)
                    tensorboard_idx += 1

                if output_dir_results is not None:
                    file_idx = 'weight' + str(weights_writer_idx) + '.pt'
                    torch.save(model.state_dict(), os.path.join(output_dir_results, file_idx))
                    weights_writer_idx += 1

                running_loss = 0.0
                if loss_type == 'bbb':
                    running_loss_llh = 0.0
                    running_loss_vp = 0.0
                    running_loss_pr = 0.0

    if output_dir_tensorboard is not None:
        if loss_type == 'bbb':
            to_close = [writer_loss, writer_loss_llh, writer_loss_vp, writer_loss_pr, writer_accs_train,
                        writer_accs_val, writer_vr, writer_predictive_entropy, writer_mi]
        else:
            to_close = [writer_loss, writer_accs_train, writer_accs_val,
                        writer_vr, writer_predictive_entropy, writer_mi]
        for writer in to_close:
            writer.close()
    print('Finished Training')
    if loss_type == 'bbb':
        return (loss_totals, loss_llhs, loss_vps, loss_prs, train_accs, max_acc, epoch_max_acc, batch_idx_max_acc,
                val_accs, val_vrs, val_predictive_entropies, val_mis)
    else:
        return (loss_totals, [[0]], [[0]], [[0]], train_accs, max_acc, epoch_max_acc, batch_idx_max_acc,
                val_accs, val_vrs, val_predictive_entropies, val_mis)


def get_loss_writers(output_dir_tensorboard, loss_type):
    """
    Return initialized loss writers
    Args:
        loss_type (str): the typeof the loss

    Returns:
        Tuple[ (SummaryWriter) * 9]

    """
    writer_loss = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "total_loss"))
    if loss_type == 'bbb':
        writer_loss_llh = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "loss_llh"))
        writer_loss_vp = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "loss_vp"))
        writer_loss_pr = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "loss_pr"))
    else:
        writer_loss_llh, writer_loss_vp, writer_loss_pr = None, None, None
    writer_vr = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "variation ratio"))
    writer_predictive_entropy = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "predictive entropy"))
    writer_mi = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "mutual information"))
    writer_accs_train = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "train_accuracy"))
    writer_accs_val = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "val_accuracy"))
    return (writer_loss, writer_loss_llh, writer_loss_vp, writer_loss_pr,
            writer_accs_train, writer_accs_val, writer_vr, writer_predictive_entropy, writer_mi)


def get_loss(model, loss_type, outputs, labels, criterion, kl_weight):
    """
    Returns the loss of the model
    Args:
        model:
        loss_type:
        outputs:
        labels:
        criterion:
        kl_weight

    Returns:
        loss (torch.Float) :
        loss_likelihood (torch.Float) :
        loss_variational_posterior (torch.Float) :
        loss_prior (torch.Float) :

    """
    loss_likelihood = criterion(outputs, labels)
    if loss_type == 'bbb':
        weights_used, bias_used = model.get_previous_weights()
        loss_variational_posterior = model.variational_posterior(weights_used, bias_used)
        loss_prior = -model.prior(weights_used, bias_used)
        loss = kl_weight * (loss_variational_posterior + loss_prior) + loss_likelihood
        return loss, loss_likelihood, loss_variational_posterior, loss_prior
    elif loss_type == 'criterion':
        loss = loss_likelihood
        return loss, loss_likelihood, None, None
    else:
        raise ValueError('Loss must be either "bbb" for Bayes By Backprop,'
                         'or "criterion" for CrossEntropy. No other loss is implemented.')


def train_bayesian_modular(
        model,
        optimizer,
        loss,
        observables,
        number_of_epochs,
        trainloader,
        valloader,
        number_of_tests=10,
        output_dir_tensorboard=None,
        output_dir_results=None,
        device='cpu',
        verbose=False,
):
    """
    Train Bayesian with modular arguments
    Args:
        model (torch.nn.Module child): model we want to train
        optimizer (torch.optim optimizer): how do we update the weights
        loss (src.loggers.losses.base_loss.BaseLoss child): loss object
        observables (src.loggers.observables): observable object
        number_of_epochs:
        trainloader:
        valloader:
        number_of_epochs (int): how long do we train our model
        trainloader (torch.utils.data.dataloader.DataLoader): train data
        output_dir_tensorboard (str): output directory in which to save the tensorboard
        device (torch.device || str): cpu or gpu
        verbose (Bool): print training steps or not
    Returns
        NOT IMPLEMENTED YET
    """
    start_time = time()
    number_of_batch = len(trainloader)
    interval = max(number_of_batch // 10, 1)

    for logger in [loss, observables]:
        logger.set_number_of_epoch(number_of_epochs)
        logger.set_number_of_batch(number_of_batch)
        logger.init_tensorboard_writer(output_dir_tensorboard)
        logger.init_results_writer(output_dir_results)

    model.train()
    for epoch in range(number_of_epochs):
        loss.set_current_epoch(epoch)
        observables.set_current_epoch(epoch)

        for batch_idx, data in enumerate(trainloader):
            loss.set_current_batch_idx(batch_idx)
            observables.set_current_batch_idx(batch_idx)

            inputs, labels = [x.to(device) for x in data]
            optimizer.zero_grad()
            outputs = model(inputs)

            observables.compute_train_on_batch(outputs, labels)
            loss.compute(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % interval == interval - 1:
                if valloader is not None:
                    val_acc, val_outputs = eval_bayesian(
                        model,
                        valloader,
                        number_of_tests=number_of_tests,
                        device=device,
                        val=True,
                    )
                    observables.compute_val(val_acc, val_outputs)

                if verbose:
                    print('======================================')
                    print(f'Epoch [{epoch+1}/{number_of_epochs}]. Batch [{batch_idx}/{number_of_batch}].')
                    loss.show()
                    observables.show()
                    print(f'Time Elapsed: {round(time() - start_time)} s')

                loss.write_tensorboard()
                observables.write_tensorboard()
                if output_dir_results is not None:
                    loss.write_results()
                    observables.write_results()

        observables.compute_train_on_epoch(model, trainloader, device)

    loss.close_writer()
    observables.close_writer()
    if verbose:
        print('Finished Training')

    return loss.results(), observables.results()
