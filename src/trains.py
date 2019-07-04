import os

import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils import aggregate_data, set_and_print_random_seed


def train(model, optimizer, criterion, number_of_epochs, trainloader,
          output_dir_tensorboard=None, output_dir_results='sandbox_results', device="cpu", verbose = False):
    return train_bayesian(model, optimizer, criterion, number_of_epochs, trainloader,
                 loss_type='criterion',output_dir_tensorboard=output_dir_tensorboard,
                 output_dir_results= output_dir_results, device=device, verbose=verbose)


def test(model, testloader, device):
    return test_bayesian(model, testloader, number_of_tests=1, device=device)


def uniform(batch_index,number_of_batchs):
    return 1/number_of_batchs


def train_bayesian(model, optimizer, criterion, number_of_epochs, trainloader, loss_type='bbb',
                   step_function=uniform,
                   output_dir_tensorboard=None, output_dir_results=None, device="cpu", verbose=False):
    '''
    Train the model in a bayesian fashion, meaning the loss is different.
    Args:
        model (Torch.nn.Module child): the model we want to train
        optimizer (torch.optim optimizer): how do we update the weights
        criterion (function): how do we compute the likelihood
        number_of_epochs (int): how long do we train our model
        trainloader (torch.utils.data.dataloader.DataLoader): train data
        loss_type (str): which type of loss. Chose "bbb" (Bayes By Backprop) or "criterion" (CrossEntropy)
        step_function (function): takes as args (number of batchs, length of batch) and returns the weight to give to KL
        output_dir_tensorboard (str): output directory in which to save the tensorboard
        device (torch.device || str): cpu or gpu
        verbose (Bool): print training steps or not

    Returns:
        loss_accs (list): list of the total loss for each epoch
        loss_llhs (list): list of the lilelihood loss for each epoch (P(D|W))
        loss_vps (list): list of the variational posterior loss for each epoch (q(W|D))
        loss_prs (list): list of the prior loss for each epoch (P(W))
        train_accs (list): list of the accuracies for each epoch
        max_acc (float): the maximum accuracy obtained by the net
        epoch_max_acc (int): the epoch where the max acc is obtained
        i_max_acc (int): the batch_idx where the epoch is obtained

    '''

    if output_dir_tensorboard is not None:
        writer_loss = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "total_loss"))
        if loss_type == 'bbb':
            writer_loss_llh = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "loss_llh"))
            writer_loss_vp = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "loss_vp"))
            writer_loss_pr = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "loss_pr"))
        writer_accs = SummaryWriter(log_dir=os.path.join(output_dir_tensorboard, "accuracy"))
        tensorboard_idx = 0
    if output_dir_results is not None:
        weights_writer_idx = 0
        if not os.path.exists(output_dir_results):
            os.mkdir(output_dir_results)

    max_acc = 0

    model.train()
    loss_accs = [list() for _ in range(number_of_epochs)]
    if loss_type == 'bbb':
        loss_llhs = [list() for _ in range(number_of_epochs)]
        loss_vps = [list() for _ in range(number_of_epochs)]
        loss_prs = [list() for _ in range(number_of_epochs)]
    train_accs = [list() for _ in range(number_of_epochs)]
    for epoch in range(number_of_epochs):  # loop over the dataset multiple times

        number_of_data = len(trainloader)
        interval = number_of_data // 10
        running_loss = 0.0
        if loss_type == 'bbb':
            running_loss_llh = 0.0
            running_loss_vp = 0.0
            running_loss_pr = 0.0
        number_of_correct_labels = 0
        number_of_labels = 0

        for i, data in enumerate(trainloader, 0):
            M = len(trainloader)
            kl_weight = step_function(i, M)

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = [x.to(device) for x in data]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss_likelihood = criterion(outputs, labels)
            if loss_type == 'bbb':
                weights_used, bias_used = model.get_previous_weights()
                loss_varational_posterior = model.variational_posterior(weights_used, bias_used)
                loss_prior = -model.prior(weights_used, bias_used)
                loss = kl_weight*(loss_varational_posterior + loss_prior) + loss_likelihood
            elif loss_type == 'criterion':
                loss = loss_likelihood
            else:
                raise ValueError('Loss must be either "bbb" for Bayes By Backprop,'
                                 'or "criterion" for CrossEntropy. No other loss is implented.')

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if loss_type == 'bbb':
                running_loss_llh += loss_likelihood.item()
                running_loss_vp += loss_varational_posterior.item()
                running_loss_pr += loss_prior.item()
            predicted_labels = outputs.argmax(1)
            number_of_correct_labels += torch.sum(predicted_labels - labels == 0).item()
            number_of_labels += labels.size(0)
            if i % interval == interval - 1:
                current_loss = running_loss / number_of_data
                if loss_type == 'bbb':
                    current_loss_llh = running_loss_llh / number_of_data
                    current_loss_vp = running_loss_vp / number_of_data
                    current_loss_pr = running_loss_pr / number_of_data
                current_acc = number_of_correct_labels / number_of_labels
                if max_acc < current_acc:
                    max_acc = current_acc
                    epoch_max_acc = epoch
                    i_max_acc = i
                if verbose:
                    if loss_type == 'bbb':
                        print(f'Train: [{epoch + 1}, {i + 1}/{number_of_data}] '
                              f'Acc: {round(100 * current_acc, 2)} %, '
                              f'loss: {round(current_loss, 2)}, '
                              f'loss_llh: {round(current_loss_llh, 2)}, '
                              f'loss_vp: {round(current_loss_vp, 2)}, '
                              f'loss_pr: {round(current_loss_pr, 2)}')
                    else:
                        print(f'Train: [{epoch + 1}, {i + 1}/{number_of_data}] '
                              f'Acc: {round(100 * current_acc, 2)} %, '
                              f'loss: {round(current_loss, 2)}')

                loss_accs[epoch].append(current_loss)
                if loss_type == 'bbb':
                    loss_llhs[epoch].append(current_loss_llh)
                    loss_vps[epoch].append(current_loss_vp)
                    loss_prs[epoch].append(current_loss_pr)
                train_accs[epoch].append(current_acc)

                if output_dir_tensorboard is not None:
                    writer_loss.add_scalar('loss', current_loss, tensorboard_idx)
                    if loss_type == 'bbb':
                        writer_loss_llh.add_scalar('loss', current_loss_llh, tensorboard_idx)
                        writer_loss_vp.add_scalar('loss', current_loss_vp, tensorboard_idx)
                        writer_loss_pr.add_scalar('loss', current_loss_pr, tensorboard_idx)
                    writer_accs.add_scalar('Train accuracy', current_acc, tensorboard_idx)
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
            to_close = [writer_loss, writer_loss_llh, writer_loss_vp, writer_loss_pr, writer_accs]
        else:
            to_close = [writer_loss, writer_accs]
        for writer in to_close:
            writer.close()
    print('Finished Training')
    if loss_type == 'bbb':
        return loss_accs, loss_llhs, loss_vps, loss_prs, train_accs, max_acc, epoch_max_acc, i_max_acc
    else:
        return loss_accs, [[0]], [[0]], [[0]], train_accs, max_acc, epoch_max_acc, i_max_acc


def test_bayesian(model, testloader, number_of_tests, device):

    model.eval()
    number_of_samples = len(testloader.dataset)
    all_correct_labels = torch.zeros(1, requires_grad=False)
    all_uncertainties = torch.Tensor().to(device).detach()
    all_dkls = torch.Tensor().to(device).detach()

    for i, data in enumerate(testloader):
        inputs, labels = [x.to(device).detach() for x in data]
        batch_outputs = torch.Tensor(number_of_tests, inputs.size(0), model.number_of_classes).to(device).detach()
        for test_idx in range(number_of_tests):
            output = model(inputs)
            batch_outputs[test_idx] = output.detach()
        predicted_labels, uncertainty, dkls = aggregate_data(batch_outputs)

        all_uncertainties = torch.cat((all_uncertainties, uncertainty))
        all_dkls = torch.cat((all_dkls, dkls))
        all_correct_labels += torch.sum(predicted_labels.int() - labels.int() == 0)

    accuracy = (all_correct_labels / number_of_samples).item()

    return accuracy, all_uncertainties, all_dkls


def test_random(model, batch_size, img_channels, img_dim, number_of_tests, number_of_classes, random_seed=None, device='cpu'):
    seed = set_and_print_random_seed(random_seed)
    random_noise = torch.randn(batch_size, img_channels, img_dim, img_dim).to(device)
    output_random = torch.Tensor(number_of_tests, batch_size, number_of_classes)
    for test_idx in range(number_of_tests):
        output_random[test_idx] = model(random_noise).detach()
    _, random_uncertainty, random_dkl = aggregate_data(output_random)
    return random_uncertainty, random_dkl, seed
