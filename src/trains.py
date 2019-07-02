import torch

from src.utils import aggregate_data


def train(model, optimizer, criterion, number_of_epochs, trainloader, device="cpu", verbose = False):
    model.train()
    loss_accs = [list() for _ in range(number_of_epochs)]
    train_accs = [list() for _ in range(number_of_epochs)]
    for epoch in range(number_of_epochs):  # loop over the dataset multiple times

        number_of_data = len(trainloader)
        interval = number_of_data // 10
        running_loss = 0.0
        number_of_correct_labels = 0
        number_of_labels = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = [x.to(device) for x in data]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            predicted_labels = outputs.argmax(1)
            number_of_correct_labels += torch.sum(predicted_labels - labels == 0).item()
            number_of_labels += labels.size(0)
            if i % interval == interval - 1:
                if verbose:
                    print(f'Train: [{epoch + 1}, {i + 1}/{number_of_data}] loss: {running_loss / number_of_data}, '
                          f'Acc: {round(100 * number_of_correct_labels / number_of_labels, 2)} %')
                loss_accs[epoch].append(running_loss / number_of_data)
                train_accs[epoch].append(number_of_correct_labels / number_of_labels)
                running_loss = 0.0

    print('Finished Training')
    return loss_accs, train_accs


def test(model, testloader, device):
    return test_bayesian(model, testloader, number_of_tests=1, device=device)


#TODO: Add the loss of bayes by backprop: variational posterior and prior
def train_bayesian(model, optimizer, criterion, number_of_epochs, trainloader, device="cpu", verbose=False):
    model.train()
    loss_accs = [list() for _ in range(number_of_epochs)]
    train_accs = [list() for _ in range(number_of_epochs)]
    for epoch in range(number_of_epochs):  # loop over the dataset multiple times

        number_of_data = len(trainloader)
        interval = number_of_data // 10
        running_loss = 0.0
        number_of_correct_labels = 0
        number_of_labels = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = [x.to(device) for x in data]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss_likelihood = criterion(outputs, labels)
            weights_used, bias_used = model.get_previous_weights()
            loss_varational_posterior = model.variational_posterior(weights_used, bias_used)
            loss_prior = -model.prior(weights_used, bias_used)
            loss = loss_varational_posterior + loss_prior + loss_likelihood
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            predicted_labels = outputs.argmax(1)
            number_of_correct_labels += torch.sum(predicted_labels - labels == 0).item()
            number_of_labels += labels.size(0)
            if i % interval == interval - 1:
                if verbose:
                    print(f'Train: [{epoch + 1}, {i + 1}/{number_of_data}] loss: {running_loss / number_of_data}, '
                          f'Acc: {round(100 * number_of_correct_labels / number_of_labels, 2)} %')
                loss_accs[epoch].append(running_loss / number_of_data)
                train_accs[epoch].append(number_of_correct_labels / number_of_labels)
                running_loss = 0.0

    print('Finished Training')
    return loss_accs, train_accs


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
