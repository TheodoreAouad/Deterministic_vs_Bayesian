import torch


def train(model, optimizer, criterion, number_of_epochs, trainloader, device="cpu", verbose = False):
    model.train()
    loss_accs = [[]]*number_of_epochs
    train_accs = [[]]*number_of_epochs
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
                loss_accs[epoch].append([running_loss / number_of_data])
                train_accs[epoch].append([round(100 * number_of_correct_labels / number_of_labels, 2)])
                running_loss = 0.0

    print('Finished Training')
    return loss_accs, train_accs


def test(model, testloader, device):
    running_loss = 0.0
    number_of_correct_labels = 0
    number_of_labels = 0
    for i, data in enumerate(testloader, 0):

        inputs, labels = [x.to(device) for x in data]
        outputs = model(inputs)
        predicted_labels = outputs.argmax(1)
        number_of_correct_labels += torch.sum(predicted_labels - labels == 0).item()
        number_of_labels += labels.size(0)
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f' Test: {i + 1} loss: {running_loss / 2000}, '
                  f'Acc: {round(100 * number_of_correct_labels / number_of_labels, 2)} %')
            running_loss = 0.0
    print(f'Test accuracy: {round(100 * number_of_correct_labels / number_of_labels, 2)} %')
