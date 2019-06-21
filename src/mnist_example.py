#%% Imports

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from importlib import reload

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import src.utils as u
import src.models as m
reload(u)
reload(m)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

# %% Datasets

transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)
# %% Plot image

iter_train = iter(trainloader)
image, label = next(iter_train)
plt.imshow(image[0, 0, :, :])
print(label)
plt.show()
image = image.to(device)
label = label.to(device)


# %% Def training

def train(model, optimizer, criterion, number_of_epochs, verbose = False):
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
                running_loss = 0.0
                loss_accs[epoch].append([running_loss / number_of_data])
                train_accs[epoch].append([round(100 * number_of_correct_labels / number_of_labels, 2)])

    print('Finished Training')
    return loss_accs, train_accs


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


# %% gpu vs cpu

conv1 = nn.Conv2d(1,16,3)
conv2 = nn.Conv2d(1,16,3).to(device)

seed1 = u.set_and_print_random_seed(3616524511)
conv1.reset_parameters()
u.set_and_print_random_seed(seed1)
conv2.reset_parameters()

w1 = conv1.weight.data.to(device)
w2 = conv2.weight.data.to(device)

torch.sum(torch.abs(w1-w2))


# %% BayNet.mu1 vs mu1

mu1 = nn.Parameter(data=torch.Tensor(16, 1, 3, 3), requires_grad=True).to(device)
bias1 = nn.Parameter(data=torch.Tensor(16), requires_grad=True).to(device)

BayNet = m.ProbabilistClassifier(10).to(device)

seed1 = u.set_and_print_random_seed(3616524511)
u.reset_parameters_conv(mu1,bias1)
u.set_and_print_random_seed(seed1)
u.reset_parameters_conv(BayNet.mu1,BayNet.bias1)

print("BayNet.mu1 vs mu1: ", torch.sum(torch.abs(mu1-BayNet.mu1)))


# %% DetNet.conv vs nn.Conv2d

conv1 = nn.Conv2d(1,16,3)
DetNet = DeterministClassifier(10)

seed1 = u.set_and_print_random_seed(3616524511)
DetNet.conv1.reset_parameters()
u.set_and_print_random_seed(seed1)
conv1.reset_parameters()

w1_1 = DetNet.conv1.weight.data.to(device)
w1_2 = conv1.weight.data.to(device)

print("Detnet.conv vs nn.Conv2d: ", torch.sum(torch.abs(w1_1-w1_2)))


# %% nn.Conv2d vs custom init

conv1 = nn.Conv2d(1,16,3).to(device)
mu1 = nn.Parameter(data=torch.Tensor(16, 1, 3, 3), requires_grad=True).to(device)
bias1 = nn.Parameter(data=torch.Tensor(16), requires_grad=True).to(device)

seed1 = u.set_and_print_random_seed(3616524511)
conv1.reset_parameters()
u.set_and_print_random_seed(seed1)
reset_parameters_conv(mu1, bias1)

w2_1 = conv1.weight.data.to(device)
w2_2 = mu1.to(device)
b1 = conv1.bias.data.to(device)
b2 = bias1.to(device)

print("nn.Conv2d vs mu1")
print("Weight: ", torch.sum(torch.abs(w2_1-w2_2)))
print("Bias: ", torch.sum(torch.abs(b1-b2)))



# %% Baynet ini vs DetNet ini

DetNet = DeterministClassifier(10)
BayNet = m.ProbabilistClassifier(10)

seed1 = u.set_and_print_random_seed()
DetNet.conv1.reset_parameters()
u.set_and_print_random_seed(seed1)
reset_parameters_conv(BayNet.mu1, BayNet.bias1)
w11 = DetNet.conv1.weight.data
w21 = BayNet.mu1
b11 = DetNet.conv1.bias.data
b21 = BayNet.bias1

print("BayNet ini vs DetNet ini")
print("Weight: ", torch.sum(torch.abs(w11-w21)))
print("Bias: ", torch.sum(torch.abs(b11-b21)))

# %%Test conv identity

DetNet = m.DeterministClassifier(10)
BayNet = m.ProbabilistClassifier(10)

seed1 = u.set_and_print_random_seed()
DetNet.conv1.reset_parameters()
u.set_and_print_random_seed(seed1)
u.reset_parameters_conv(BayNet.mu1, BayNet.bias1)

x = torch.rand(16,1,3,3)

output1 = F.conv2d(x, weight = BayNet.mu1, bias = BayNet.bias1, padding=1)
output2 = DetNet.conv1(x)

torch.abs(torch.sum(output1-output2))



# %%Test net identity

DetNet = DeterministClassifier(10)
weight1, bias1 = DetNet.conv1.weight.data, DetNet.conv1.bias.data
weight2, bias2 = DetNet.conv2.weight.data, DetNet.conv2.bias.data

BayNet = ProbabilistClassifier(10, weights=(weight1,weight2), bias=(bias1,bias2))

DetNet.to(device)
BayNet.to(device)

output1 = DetNet(image)
output2 = BayNet(image)

torch.abs(torch.sum(output1-output2))

# %% Test Training identity

DetNet = DeterministClassifier(10)
weight1, bias1 = DetNet.conv1.weight.data, DetNet.conv1.bias.data
weight2, bias2 = DetNet.conv2.weight.data, DetNet.conv2.bias.data

BayNet = ProbabilistClassifier(10, weights=(weight1,weight2), bias=(bias1,bias2))

criterion = nn.CrossEntropyLoss()
seed1 = u.set_and_print_random_seed()

det_adam = optim.Adam(DetNet.parameters())
DetNet.to(device)
det_output = DetNet(image)
det_loss = criterion(det_output,label)
det_loss.backward()

u.set_and_print_random_seed(seed1)
bay_adam = optim.Adam(BayNet.parameters())
BayNet.to(device)
bay_output = BayNet(image)
bay_loss = criterion(bay_output,label)
bay_loss.backward()

u.set_and_print_random_seed(seed1)
det_losses, det_accs = train(DetNet,det_adam,criterion,1, verbose=True)
u.set_and_print_random_seed(seed1)
bay_losses, bay_accs = train(BayNet,bay_adam,criterion,1, verbose=True)
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
