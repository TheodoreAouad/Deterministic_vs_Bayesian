# %% Imports


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)
# %% md
## Get data
# %%
# %% Datasets

transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)
# %%
# %% Plot image

iter_train = iter(trainloader)
image, label = next(iter_train)
plt.imshow(image[0, 0, :, :])
print(label)
plt.show()
image = image.to(device)
label = label.to(device)


# %%
# %% Def training


def train(model, optimizer, criterion, number_of_epochs):
    model.train()
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
                print(f'Train: [{epoch + 1}, {i + 1}/{number_of_data}] loss: {running_loss / number_of_data}, '
                      f'Acc: {round(100 * number_of_correct_labels / number_of_labels, 2)} %')
                running_loss = 0.0

    print('Finished Training')


# %%

def test(model):
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


# %% md
## Models
# %% md
# %% Define modules


# %%
def init_weights(channel_out, channel_input, width, height):
    k = 1. / (channel_input * width * height)
    return (torch.rand(channel_out, channel_input, width, height) - 0.5) * torch.sqrt(torch.Tensor([k]))


# %%
class DeterministClassifier(nn.Module):

    def __init__(self, number_of_classes):
        super(DeterministClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, number_of_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        output = F.softmax(self.fc1(x))

        return output


class ProbabilistClassifier(nn.Module):

    def __init__(self, number_of_classes):
        super(ProbabilistClassifier, self).__init__()

        self.mu1 = nn.Parameter(data=init_weights(16, 1, 3, 3), requires_grad=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.mu2 = nn.Parameter(data=init_weights(32, 16, 3, 3), requires_grad=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # self.mufc = nn.Parameter(data=torch.rand(10,16*14*14),requires_grad=True)
        self.fc1 = nn.Linear(32 * 7 * 7, number_of_classes)

    def forward(self, x):
        weights =
        x = self.pool1(F.relu(F.conv2d(x, self.mu1 + self.sigma * random(), padding=1)))
        x = self.pool2(F.relu(F.conv2d(x, self.mu2, padding=1)))
        x = x.view(-1, 32 * 7 * 7)
        output = F.softmax(self.fc1(x))
        # output = F.softmax(F.linear(x, self.mufc))
        return output


class ProbabilistClassifier2(nn.Module):

    def __init__(self, number_of_classes):
        super(ProbabilistClassifier2, self).__init__()
        self.mu1 = nn.Parameter(data=torch.rand(16, 1, 3, 3), requires_grad=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.mu2 = nn.Parameter(data=torch.rand(32, 16, 3, 3), requires_grad=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.mufc = nn.Parameter(data=torch.rand(10, 32 * 7 * 7), requires_grad=True)

    # self.fc1 = nn.Linear(32*7*7, number_of_classes)

    def forward(self, x):
        conv1 = nn.Conv2d(1, 16, 3, padding=1).to(device)
        conv1.weight.data = self.mu1
        conv2 = nn.Conv2d(16, 32, 3, padding=1).to(device)
        conv2.weight.data = self.mu2

        x = self.pool1(F.relu(conv1(x)))
        x = self.pool2(F.relu(conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        # output = F.softmax(self.fc1(x))
        output = F.softmax(F.linear(x, self.mufc))
        return output


class DenseProbabilistClassifier(nn.Module):

    def __init__(self, number_of_classes):
        super(DenseProbabilistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # self.fc1 = nn.Linear(32*7*7, number_of_classes)
        self.mu1 = nn.Parameter(data=torch.rand(10, 32 * 7 * 7), requires_grad=True)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        # output = F.softmax(self.fc1(x))
        weights = self.mu1
        output = F.softmax(F.linear(x, weights))
        return output


# %%
model_proba = ProbabilistClassifier(10)
model_proba.to(device)
adam_proba = optim.Adam(model_proba.parameters())
criterion = nn.CrossEntropyLoss()
# %%
model_proba.mu1
# %%
train(model_proba, adam_proba, criterion, 1)

# %%
model_proba.zero_grad()
output = model_proba(image)
loss = criterion(output, label)
loss.backward()
[(name, torch.abs(k.grad).sum()) if k.grad is not None else (name, k.grad) for (name, k) in
 model_proba.named_parameters()]
# %%
model = DeterministClassifier(10)
model.conv1.weight.data
# %%
[(name, torch.abs(k.grad).sum()) if k.grad is not None else (name, k.grad) for (name, k) in
 model_proba.named_parameters()]
# %%
[(name, k.grad) if k is not None else (name, k) for (name, k) in model_dense_proba.named_parameters()]
# %%
model_dense_proba = DenseProbabilistClassifier(nn.Module)
model_dense_proba.to(device)
adam_proba_dense = optim.Adam(model_dense_proba.parameters())
criterion = nn.CrossEntropyLoss()
# %%
train(model_dense_proba, adam_proba_dense, criterion, 1)
# %%
[torch.abs(k.grad).sum() for k in list(model.fc1.parameters())]
# %%
# %% Definition of model

model = DeterministClassifier(number_of_classes=10)
model.to(device)
sgd = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
adam = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
# %%
# %% Training
number_of_epochs = 3
train(model, adam, criterion, number_of_epochs)

# %% Testing
test(model)
