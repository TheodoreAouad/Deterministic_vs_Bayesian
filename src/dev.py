#%% Imports

import matplotlib.pyplot as plt
from importlib import reload
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import src.models.bayesian_models.bayesian_base_layers
import src.utils as u
import src.models.determinist_models as dm
import src.tasks.trains as t
import src.dataset_manager.get_data as dataset
import src.models.bayesian_models.gaussian_classifiers as gc

reload(u)
reload(t)
reload(dm)
reload(gc)
reload(dataset)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

#%% Datasets
reload(dataset)
trainloader, testloader = dataset.get_mnist()
get_train_img = iter(trainloader)
#%%
train_img, train_label = next(get_train_img)
print(np.unique(train_label))
plt.imshow(train_img[5][0])
plt.show()
#%%
trainloader, testloader = dataset.get_cifar10()

#%%
reload(u)
reload(gc)
reload(t)
trainloader, testloader = dataset.get_mnist(batch_size=128)
det_net = gc.GaussianClassifierMNIST("determinist", (0, 0), (1, 1), number_of_classes=10)
det_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(det_net.parameters())
t.train(det_net, optimizer, criterion, 3, output_dir_results='sandbox_results/det',
        trainloader=trainloader, device=device, verbose=True)

#%%

reload(u)
reload(gc)
reload(t)
trainloader, testloader = dataset.get_mnist(batch_size=1)
bay_net = gc.GaussianClassifierMNIST(-5, (0, 0), (1, 1), number_of_classes=10)
bay_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bay_net.parameters())
t.train_bayesian(bay_net, optimizer, criterion, 2, trainloader,
                 loss_type="bbb",
                 output_dir_results='sandbox_results/bbb_stepped',
                 output_dir_tensorboard="./output",
                 device=device, verbose=True);

#%%
reload(u)
weights_paths = u.get_file_path_in_dir('sandbox_results/bbb_stepped')
weights_norms = []
for path in weights_paths:
    model = gc.GaussianClassifierMNIST(rho=1)
    model.load_state_dict(torch.load(path))
    weights_norms.append(u.compute_weights_norm(model))

plt.plot(weights_norms)
plt.title("bbb_stepped")
plt.show()
#%%
reload(u)
weights_paths = u.get_file_path_in_dir('sandbox_results/ce')
weights_norms = []
for path in weights_paths:
    model = gc.GaussianClassifierMNIST()
    model.load_state_dict(torch.load(path))
    weights_norms.append(u.compute_weights_norm(model))

plt.plot(weights_norms)
plt.title("cross entropy")
plt.show()
#%%
weights_paths = u.get_file_path_in_dir('sandbox_results/det')
weights_norms = []
for path in weights_paths:
    model = gc.GaussianClassifierMNIST(rho="determinist")
    model.load_state_dict(torch.load(path))
    weights_norms.append(u.compute_weights_norm(model))

plt.plot(weights_norms)
plt.title("determinist")
plt.show()

# %%
t.eval_bayesian(bay_net, testloader, 15, device)

#%%

t.eval(det_net, testloader, device)


#%%




#%%

_, testloader = dataset.get_mnist(batch_size=128)
get_test_img = iter(testloader)
img, label = next(get_test_img)
#%%
outpt = bay_net(img)
print("predicted:", np.unique(outpt.argmax(1), return_counts=1))
print("true:", np.unique(label, return_counts=1))

#%%

weights = torch.Tensor().to(device)
bias = torch.Tensor().to(device)
all_layers = iter(bay_net.modules())
next(all_layers)
for layer in all_layers:
    print(layer)
    if not getattr(layer, "determinist", True):
        weight_to_add, bias_to_add = layer.sample_weights()
        weights = torch.cat((weights, u.vectorize(weight_to_add)))
        bias = torch.cat((bias, u.vectorize(bias_to_add)))

bay_net.variational_posterior(weights, bias)

#%%
_,testloader = dataset.get_mnist(batch_size=16)
get_test_img = iter(testloader)
img, label = next(get_test_img)

#%%
reload(t)
class MiniTestLoader:

    def __init__(self, img, label):
        self.dataset = img
        self.data = [(img, label)]

    def __getitem__(self, key):
        return self.data[key]


mini_testloader = MiniTestLoader(img, label)
acc, unc, dkls = t.eval_bayesian(bay_net, mini_testloader, 15, device)

#%%

q75, q25 = np.percentile(unc, [75 ,25])
q75-q25


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
t.eval_bayesian(model, testloader, number_of_tests, device)

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

bay_conv = src.models.bayesian_models.bayesian_base_layers.GaussianCNN(-1, 1, 16, 3)
lp = bay_conv.bayesian_parameters()
lpp = bay_conv.named_bayesian_parameters()

for name, params in bay_conv.named_parameters():
    if hasattr(params, "bayesian"):
        print(params.bayesian)
    else:
        print(name)
