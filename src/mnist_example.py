#%% Imports

import matplotlib.pyplot as plt
from importlib import reload

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import src.utils as u
from src.models.determinist_models import DeterministClassifierSequential, DeterministClassifierFunctional
from src.trains import train
from src.get_data import get_mnist

reload(u)
reload(m)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

#%% Datasets

trainloader, testloader = get_mnist()
#%% Plot image

iter_train = iter(trainloader)
image, label = next(iter_train)
plt.imshow(image[0, 0, :, :])
print(label)
plt.show()
image = image.to(device)
label = label.to(device)


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
train(BayNet, adam_proba, criterion, 1, device="cpu", verbose=True)
u.set_and_print_random_seed(seed1)
train(DetNet, adam_det, criterion, 1, device="cpu", verbose=True)

#%%
mu1 = nn.Parameter(data=torch.Tensor(16, 1, 3, 3)), requires_grad=True)
bias1 = nn.Parameter(data=torch.Tensor(16), requires_grad=True)
output1 = F.conv2d(image, weight=mu1, bias=bias1, padding=1)

#%%

