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
import src.get_data as dataset
import src.models.bayesian_models as bm

reload(u)
reload(t)
reload(dm)
reload(bm)
reload(dataset)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

#%% Datasets

trainloader, testloader = dataset.get_mnist()
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
    BayNet = bm.GaussianClassifierMNIST(rho=rho, dim_input=28, number_of_classes=10, determinist=False)
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
BayNet = bm.GaussianClassifierMNIST(rho=rho, dim_input=28, number_of_classes=10, determinist=False)
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
BayNet = bm.GaussianClassifierMNIST(1, 28, 10)
outpt = BayNet(inpt)
print(outpt)

#%%
trainloader, testloader = get_cifar10()

#%%
reload(dm)
det_net = dm.DeterministClassifierCIFAR(number_of_classes=10)
det_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(det_net.parameters())
t.train(det_net, optimizer, criterion, 1, trainloader, device, verbose=True)

#%%
reload(bm)
bay_net = bm.GaussianClassifierCIFAR(rho=1, number_of_classes=10, determinist=True)
bay_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bay_net.parameters())
t.train(bay_net, optimizer, criterion, 1, trainloader, device, verbose=True)

#%%
reload(u)
reload(bm)
trainloader, testloader = dataset.get_mnist()
bay_net = bm.GaussianClassifierMNIST(rho=-2, number_of_classes=10)
bay_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bay_net.parameters())
t.train(bay_net, optimizer, criterion, 2, trainloader, device, verbose=True)

#%%
_,testloader = dataset.get_mnist(batch_size=16)
get_test_img = iter(testloader)
img, label = next(get_test_img)

#%%
def compute_memory_used_tensor(tensor):
    return dict({
        'number of elements': tensor.nelement(),
        'size of an element': tensor.element_size(),
        'total memory use': tensor.nelement() * tensor.element_size()
    })

#%%

reload(u)
bay_net.eval()
random_image = torch.rand(16,1,28,28).to(device)
number_of_tests = 10
data_random = torch.Tensor(20, 16, 10)
for test_idx in range(number_of_tests):
    data_random[test_idx] = bay_net(random_image)

data_mnist = torch.Tensor(20,16,10)
for test_idx in range(number_of_tests):
    data_mnist[test_idx] = bay_net(img.to(device))

#%%

_,testloader = dataset.get_mnist(batch_size=16)
get_test_img = iter(testloader)
# img, label = next(get_test_img)

number_of_tests = 5
model = bay_net

number_of_correct_labels = torch.zeros(1)
number_of_labels = torch.zeros(1)

all_outputs = torch.Tensor(number_of_tests, testloader.batch_size, model.number_of_classes).to(device)

i = 0
if True:# for i, data in enumerate(testloader, 0):
    data = next(get_test_img)
    inputs, labels = [x.to(device) for x in data]

    to_add_to_all_outputs = torch.Tensor(number_of_tests, inputs.size(0), model.number_of_classes).to(device)
    predicted = torch.FloatTensor(number_of_tests, testloader.batch_size).to(device)
    test_idx = 0
    if True:# for test_idx in range(number_of_tests):
        output = model(inputs)
        to_add_to_all_outputs[test_idx] = output
        predicted[test_idx] = output.argmax(1)
        test_idx += 1
    all_outputs = torch.cat((all_outputs, to_add_to_all_outputs), 1)
    predicted_labels = torch.round(predicted.mean(0)).int()
    number_of_correct_labels += torch.sum(predicted_labels - labels.int() == 0)
    number_of_labels += labels.size(0)

print(output.nelement(), output.element_size(), output.nelement() * output.element_size())

print(f'Test accuracy: {round(100 * (number_of_correct_labels / number_of_labels).item(), 2)} %')
returned1, returned2 = (number_of_correct_labels / number_of_labels).item(), all_outputs
#%%
reload(u)

_,testloader = dataset.get_mnist(batch_size=16)
number_of_tests = 1
model = bay_net

number_of_samples = torch.zeros(1, requires_grad=False)
all_correct_labels = torch.zeros(1, requires_grad=False)
all_uncertainties = torch.zeros(1, requires_grad=False)
all_dkls = torch.zeros(1, requires_grad=False)

for i, data in enumerate(testloader, 0):
    inputs, labels = [x.to(device).detach() for x in data]
    batch_outputs = torch.Tensor(number_of_tests, inputs.size(0), model.number_of_classes).to(
        device).detach()
    for test_idx in range(number_of_tests):
        output = model(inputs)
        batch_outputs[test_idx] = output.detach()
    predicted_labels, uncertainty, dkls = u.aggregate_data(batch_outputs)

    all_uncertainties += uncertainty.mean()
    all_dkls += dkls.mean()
    all_correct_labels += torch.sum(predicted_labels.int() - labels.int() == 0)
    number_of_samples += labels.size(0)

#%%

reload(t)
reload(u)
_,testloader = dataset.get_mnist(batch_size=16)
number_of_tests = 10
model = bay_net
t.test_bayesian(model, testloader, number_of_tests, device)

#%%
number_of_tests = 20
seed_random = u.set_and_print_random_seed()
random_noise = torch.randn(1000,1,28,28).to(device)
output_random = torch.Tensor(number_of_tests, 1000, 10)
for test_idx in range(number_of_tests):
    output_random[test_idx] = bay_net(random_noise).detach()
_, random_uncertainty, random_dkl = u.aggregate_data(output_random)
print(random_uncertainty.mean(), random_uncertainty.std())

#%%

bay_net = bm.GaussianClassifierMNIST(rho=1, number_of_classes=10)
