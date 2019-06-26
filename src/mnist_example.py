#%% Imports
import os

import matplotlib.pyplot as plt
from importlib import reload
import numpy as np

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
seed_random = u.set_and_print_random_seed()
random_noise = torch.randn(16,1,28,28).to(device)
rhos = [-5, -3, -1, 0, 1]
res = []
for rho in rhos:
    seed_model = u.set_and_print_random_seed()
    BayNet = bm.GaussianClassifier(rho=rho, dim_input=28, number_of_classes=10, determinist=False)
    BayNet.to(device)
    criterion = nn.CrossEntropyLoss()
    adam_proba = optim.Adam(BayNet.parameters())
    losses2, accs2 = t.train(BayNet, adam_proba, criterion, 10, trainloader, device=device, verbose=True)
    test_acc = t.test(BayNet, testloader, device)
    output_random = torch.Tensor(10,16)
    for i in range(10):
        output_random[i] = BayNet(random_noise).argmax(1)

    res.append(dict({
        "seed_random": seed_random,
        "seed_model": seed_model,
        "rho": rho,
        "train accuracy": accs2,
        "test accuracy": test_acc,
        "random output": output_random
    }))

torch.save(res, "results/experience01.pt")


# %%
reload(bm)
rho = -5
BayNet = bm.GaussianClassifier(rho=rho, dim_input=28, number_of_classes=10, determinist=False)
BayNet.to(device)
criterion = nn.CrossEntropyLoss()
adam_proba = optim.Adam(BayNet.parameters())
losses2, accs2 = t.train(BayNet, adam_proba, criterion, 5, trainloader, device=device, verbose=True)

# %%
t.test(BayNet, testloader, device)

#%%
random_noise = torch.randn(16,1,28,28).to(device)
#%%
polyaxon_results = "polyaxon_results"
single = "experiments"
group = "groups"
group_nb = "62"
exp_nb = "985"
path_to_results = os.path.join(polyaxon_results, group, group_nb, exp_nb, "experience01.pt")

res = torch.load(path_to_results)
print(u.get_interesting_results(res[0],10))
for key, value in res[0].items():
    if key != "random output":
        if "train" in key:
            print(key, value[-1][-1])
        else:
            print(key, value)


#%%

inpt = torch.ones(4,1,28,28)
BayNet = bm.GaussianClassifier(1,28,10)
outpt = BayNet(inpt)
print(outpt)

#%%

polyaxon_results = "polyaxon_results"
single = "experiments"
group = "groups"
group_nb = "62"
results = []
for exp_nb in [str(k) for k in range(984,993)]:
    result = u.open_experiment_results(group, exp_nb, group_nb, polyaxon_results)[0]
    results.append(u.get_interesting_result(result,10))
u.write_results_in_csv(results)
