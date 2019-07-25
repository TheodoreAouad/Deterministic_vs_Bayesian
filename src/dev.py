#%% Imports

import matplotlib.pyplot as plt
from importlib import reload
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import src.models.bayesian_models.bayesian_base_layers
import src.utils as u
import src.models.determinist_models as dm
import src.tasks.trains as t
import src.tasks.evals as e
import src.dataset_manager.get_data as dataset
import src.models.bayesian_models.gaussian_classifiers as gc
import src.uncertainty_measures as um

reload(u)
reload(t)
reload(dm)
reload(gc)
reload(dataset)
reload(um)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

#%% Datasets
reload(dataset)
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
])
omniglot_loader = dataset.get_omniglot(transform=transform)
get_omniglot_img = iter(omniglot_loader)

#%%

F.dropout

#%%
omniglot_img, omniglot_label = next(get_omniglot_img)
plt.imshow(omniglot_img[0][0])
plt.show()
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
_, _, evalloader_0_5 = dataset.get_mnist(eval_labels=range(6), batch_size=32)
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

#%% Classic training
reload(u)
reload(gc)
reload(t)
trainloader, valloader, evalloader = dataset.get_mnist(batch_size=128, split_val=0)
bay_net = gc.GaussianClassifierMNIST("determinist", (0, 0), (1, 1), number_of_classes=10)
bay_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bay_net.parameters())
t.train(bay_net, optimizer, criterion, 1, output_dir_results='sandbox_results/det',
        trainloader=trainloader, device=device, verbose=True)

#%%
number_of_tests = 5
model = bay_net
get_eval_img = iter(evalloader)
data = next(get_eval_img)
inputs, labels = [x.to(device).detach() for x in data]
batch_outputs = torch.Tensor(number_of_tests, inputs.size(0), model.number_of_classes).to(device).detach()
for test_idx in range(number_of_tests):
    output = model(inputs)
    batch_outputs[test_idx] = output.detach()

#%%
reload(um)
batch_size = batch_outputs.size(1)
variation_ratios = torch.Tensor(batch_size)
predicted_labels = torch.transpose(batch_outputs.argmax(2), 0, 1)
for img_idx, img_labels in enumerate(predicted_labels):
    labels, counts = np.unique(img_labels, return_counts=True)
    highest_label_freq = counts.max() / counts.sum()
    variation_ratios[img_idx] = 1 - highest_label_freq

variation_ratio2 = um.compute_variation_ratio(batch_outputs)
print(variation_ratio2)

#%%
reload(um)
mean_of_distributions = batch_outputs.mean(0).detach()
predictive_entropies = torch.sum(-mean_of_distributions *torch.log(mean_of_distributions), 1)
predictive_entropies2 = um.compute_predictive_entropy(batch_outputs)

random_img = torch.rand_like(batch_outputs)
predictive_entropies_random = um.compute_predictive_entropy(random_img)
print(predictive_entropies2)

#%%
reload(um)
number_of_tests = batch_outputs.size(0)
predictive_entropies = um.compute_predictive_entropy(batch_outputs)
x = batch_outputs * torch.log(batch_outputs)
mutual_information_uncertainties = predictive_entropies + 1/number_of_tests * x.sum(2).sum(0)
mutual_information_uncertainties2 = um.compute_mutual_information_uncertainty(batch_outputs)
print(mutual_information_uncertainties2)
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
def compute_memory_used_tensor(tensor):
    return dict({
        'number of elements': tensor.nelement(),
        'size of an element': tensor.element_size(),
        'total memory use': tensor.nelement() * tensor.element_size()
    })


#%%

result = torch.load("output/results.pt", map_location="cpu")
print(result)


