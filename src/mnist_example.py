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
import src.determinist_models as m
reload(u)
reload(m)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

#%% Datasets

transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)
#%% Plot image

iter_train = iter(trainloader)
image, label = next(iter_train)
plt.imshow(image[0, 0, :, :])
print(label)
plt.show()
image = image.to(device)
label = label.to(device)


#%% Def training

def train(model, optimizer, criterion, number_of_epochs, device, verbose = False):
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


#%% gpu vs cpu

conv1 = nn.Conv2d(1,16,3)
conv2 = nn.Conv2d(1,16,3).to(device)

seed1 = u.set_and_print_random_seed(3616524511)
conv1.reset_parameters()
u.set_and_print_random_seed(seed1)
conv2.reset_parameters()

w1 = conv1.weight.data.to(device)
w2 = conv2.weight.data.to(device)

torch.sum(torch.abs(w1-w2))


#%% Test Training identity

DetNet = m.DeterministClassifierSequential(10)
weight1, bias1 = DetNet.conv1.weight.data, DetNet.conv1.bias.data
weight2, bias2 = DetNet.conv2.weight.data, DetNet.conv2.bias.data

BayNet = m.DeterministClassifierFunctional(10)

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
det_losses, det_accs = train(DetNet, det_adam, criterion, 1, verbose=True)
u.set_and_print_random_seed(seed1)
bay_losses, bay_accs = train(BayNet, bay_adam, criterion, 1, verbose=True)
#%%
model_proba = m.DeterministClassifierFunctional(10)
model_proba.to(device)
adam_proba = optim.Adam(model_proba.parameters())
criterion = nn.CrossEntropyLoss()

#%%
model_proba.zero_grad()
output = model_proba(image)
loss = criterion(output, label)
loss.backward()
[(name, torch.abs(k.grad).sum()) if k.grad is not None else (name, k.grad) for (name, k) in
 model_proba.named_parameters()]
# %%

BayNet, DetNet = m.init_same_baynet_detnet()
# BayNet.to(device)
# DetNet.to(device)
criterion = nn.CrossEntropyLoss()
adam_proba = optim.Adam(BayNet.parameters())
adam_det = optim.Adam(DetNet.parameters())

seed1 = u.set_and_print_random_seed()
train(BayNet,adam_proba,criterion,1, device="cpu", verbose=True)
u.set_and_print_random_seed(seed1)
train(DetNet,adam_det,criterion,1, device="cpu", verbose=True)

