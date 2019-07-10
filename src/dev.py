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
import src.tasks.evals as e
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
trainloader, valloader, evalloader = dataset.get_mnist(train_labels=range(6), eval_labels=range(6,10), split_val=0)
get_train_img = iter(trainloader)

#%%

trainloader, valloader, evalloader = dataset.get_mnist(train_labels=range(6), eval_labels=range(6), split_val=0)
print([len(set.dataset) for set in [trainloader, evalloader]])
print(len(evalloader.dataset))

trainloader, valloader, evalloader = dataset.get_mnist(train_labels=range(6), eval_labels=range(6), split_val=0.5)
print([len(set.dataset) for set in [trainloader, valloader, evalloader]])
print(len(valloader.dataset) + len(evalloader.dataset))

#%%
train_img, train_label = next(get_train_img)
print(np.unique(train_label))
plt.imshow(train_img[5][0])
plt.show()
#%%
trainloader, evalloader = dataset.get_cifar10()

#%%

reload(u)
reload(gc)
reload(t)
trainloader_0_5, valloader_0_5, evalloader_6_9 = dataset.get_mnist(train_labels=range(6), eval_labels=range(6,10),
                                                                   batch_size=32)
bbb_net = gc.GaussianClassifierMNIST(-5, (0, 0), (1, 1), number_of_classes=6)
bbb_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bbb_net.parameters())
t.train_bayesian(bbb_net, optimizer, criterion, 1, trainloader_0_5, valloader=valloader_0_5,
                 loss_type="bbb",
                 output_dir_results='sandbox_results/bbb_stepped',
                 output_dir_tensorboard="./output",
                 device=device, verbose=True);

#%%
reload(e)
_, evalloader_0_5 = dataset.get_mnist(eval_labels=range(6), batch_size=32)
_, bbb_softmax_unc_6_9, bbb_dkls_6_9 = e.eval_bayesian(bbb_net, evalloader_6_9, number_of_tests=10, device=device)
_, bbb_softmax_unc_0_5, bbb_dkls_0_5 = e.eval_bayesian(bbb_net, evalloader_0_5, number_of_tests=10, device=device)
print('Unseen: ', bbb_softmax_unc_6_9, bbb_dkls_6_9)
print('Seen: ', bbb_softmax_unc_0_5, bbb_dkls_0_5)

#%%
reload(u)
reload(gc)
reload(t)
trainloader_0_5, evalloader_6_9 = dataset.get_mnist(train_labels=range(6), eval_labels=range(6,10), batch_size=32)
ce_net = gc.GaussianClassifierMNIST(-5, (0, 0), (1, 1), number_of_classes=10)
ce_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ce_net.parameters())
t.train_bayesian(ce_net, optimizer, criterion, 2, trainloader,
                 loss_type="criterion",
                 output_dir_results='sandbox_results/ce_stepped',
                 output_dir_tensorboard="./output",
                 device=device, verbose=True);

#%%
reload(e)
_, evalloader_0_5 = dataset.get_mnist(eval_labels=range(6), batch_size=32)
_, ce_softmax_unc_6_9, ce_dkls_6_9 = e.eval_bayesian(ce_net, evalloader_6_9, number_of_tests=10, device=device)
_, ce_softmax_unc_0_5, ce_dkls_0_5 = e.eval_bayesian(ce_net, evalloader_0_5, number_of_tests=10, device=device)
print('Unseen: ', ce_softmax_unc_6_9, ce_dkls_6_9)
print('Seen: ', ce_softmax_unc_0_5, ce_dkls_0_5)

#%%
reload(u)
reload(gc)
reload(t)
trainloader, evalloader = dataset.get_mnist(batch_size=128)
det_net = gc.GaussianClassifierMNIST("determinist", (0, 0), (1, 1), number_of_classes=10)
det_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(det_net.parameters())
t.train(det_net, optimizer, criterion, 3, output_dir_results='sandbox_results/det',
        trainloader=trainloader, device=device, verbose=True)


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
t.eval_bayesian(bay_net, evalloader, 15, device)

#%%

t.eval(det_net, evalloader, device)


#%%




#%%

_, evalloader = dataset.get_mnist(batch_size=128)
get_eval_img = iter(evalloader)
img, label = next(get_eval_img)
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
_,evalloader = dataset.get_mnist(batch_size=16)
get_eval_img = iter(evalloader)
img, label = next(get_eval_img)

#%%
reload(t)
class MiniTestLoader:

    def __init__(self, img, label):
        self.dataset = img
        self.data = [(img, label)]

    def __getitem__(self, key):
        return self.data[key]


mini_evalloader = MiniTestLoader(img, label)
acc, unc, dkls = t.eval_bayesian(bay_net, mini_evalloader, 15, device)

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
number_of_evals = 10
data_random = torch.Tensor(20, 16, 10)
for eval_idx in range(number_of_evals):
    data_random[eval_idx] = bay_net(random_image)

data_mnist = torch.Tensor(20,16,10)
for eval_idx in range(number_of_evals):
    data_mnist[eval_idx] = bay_net(img.to(device))

#%%
reload(u)

_,evalloader = dataset.get_mnist(batch_size=16)
number_of_evals = 1
model = bay_net

number_of_samples = torch.zeros(1, requires_grad=False)
all_correct_labels = torch.zeros(1, requires_grad=False)
all_uncertainties = torch.zeros(1, requires_grad=False)
all_dkls = torch.zeros(1, requires_grad=False)

for i, data in enumerate(evalloader, 0):
    inputs, labels = [x.to(device).detach() for x in data]
    batch_outputs = torch.Tensor(number_of_evals, inputs.size(0), model.number_of_classes).to(
        device).detach()
    for eval_idx in range(number_of_evals):
        output = model(inputs)
        batch_outputs[eval_idx] = output.detach()
    predicted_labels, uncertainty, dkls = u.aggregate_data(batch_outputs)

    all_uncertainties += uncertainty.mean()
    all_dkls += dkls.mean()
    all_correct_labels += torch.sum(predicted_labels.int() - labels.int() == 0)
    number_of_samples += labels.size(0)

#%%

reload(t)
reload(u)
_,evalloader = dataset.get_mnist(batch_size=16)
number_of_evals = 10
model = bay_net
t.eval_bayesian(model, evalloader, number_of_evals, device)

#%%
number_of_evals = 20
seed_random = u.set_and_print_random_seed()
random_noise = torch.randn(1000,1,28,28).to(device)
output_random = torch.Tensor(number_of_evals, 1000, 10)
for eval_idx in range(number_of_evals):
    output_random[eval_idx] = bay_net(random_noise).detach()
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
