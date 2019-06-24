#%% Imports

import matplotlib.pyplot as plt
from importlib import reload

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import src.utils as u
import src.models.determinist_models as dm
import src.trains as t
from src.get_data import get_mnist
import src.models.bayesian_models as bm

reload(u)
reload(t)
reload(dm)
reload(bm)

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


# %% Test train accuracy

BayNet, DetNet = dm.init_same_baynet_detnet()
# BayNet.to(device)
# DetNet.to(device)
criterion = nn.CrossEntropyLoss()
adam_proba = optim.Adam(BayNet.parameters())
adam_det = optim.Adam(DetNet.parameters())

seed1 = u.set_and_print_random_seed()
t.train(BayNet, adam_proba, criterion, 1, trainloader, device="cpu", verbose=True)
u.set_and_print_random_seed(seed1)
t.train(DetNet, adam_det, criterion, 1, trainloader, device="cpu", verbose=True)


# %%

reload(bm)
BayNet = bm.BayesianClassifier(10, determinist=True)
criterion = nn.CrossEntropyLoss()
adam_proba = optim.Adam(BayNet.parameters())
t.train(BayNet, adam_proba, criterion, 1, trainloader, device="cpu", verbose=True)
