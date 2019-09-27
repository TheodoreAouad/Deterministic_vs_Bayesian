# %% Imports
from importlib import reload

import pandas as pd
import numpy as np
import torch
from torchvision import transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt

import scripts.utils as su
import src.uncertainty_measures as um
import src.utils as u
import src.tasks.evals as e

def reload_modules():
    modules_to_reload = [su, u, um, e]
    for module in modules_to_reload:
        reload(module)

cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']
reload_modules()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(device)

# %% Accuracy VS (rho / std_prior)

path_to_accs = 'results/eval_acc/all_eval_accs.pkl'
df_ini = pd.read_pickle(path_to_accs)

df = df_ini.query('trainset=="mnist" & loss_type == "exp"')
# accs = df.groupby('rho').eval_acc.mean()
xs = df.rho
ys = df.eval_acc
xs.loc[xs == 'determinist'] = 0
fig1 = plt.figure()
ax = fig1.add_subplot(111)
# ax.hist(accs)
ax.scatter(xs, ys)
ax.set_xlabel('rho')
ax.set_ylabel('acc')
fig1.show()

print(np.unique(xs, return_counts=True))


# %%

path_to_results = 'results/raw_results/all_columns/group226.pkl'
trainset = 'cifar10'
acc_ini = 0.65
ratio_ini = 0

df_ini = pd.read_pickle(path_to_results)
for unc_name in ['vr', 'pe', 'mi']:
    df_ini[f'ratio_{unc_name}'] = df_ini[f'unseen uncertainty {unc_name}-mean'] / df_ini[f'seen uncertainty {unc_name}-mean']
df_ini= df_ini.rename(columns={
    'eval accuracy':'eval_accuracy',
    'number of tests': 'number_of_tests',
})

df = df_ini.query(f'eval_accuracy>= {acc_ini} & number_of_tests==20')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(df.rho+0.1*np.random.randn(len(df.rho)), df.stds_prior)
fig.show()

# good_dfs = []
# for unc_name in ['vr', 'pe', 'mi']:
#     df = df_ini.query(f'ratio_{unc_name}>={ratio_ini} & eval_accuracy>= {acc_ini}')
#     good_dfs.append(df)
#     print('=========')
#     print(unc_name)
#     print(df.rho.value_counts())
#     print(df.stds_prior.value_counts())
#     print(df['number of tests'].value_counts())
#     print(df.batch_size.value_counts())
#     print(df.loss_type.value_counts())
#     print(df.type_of_unseen.value_counts())
#     print(len(df), len(df_ini))

pass

#%% 3d plot accuracy vs (rho / std_prior)

path_to_results = 'results/raw_results/all_columns/group226.pkl'

df_ini = pd.read_pickle(path_to_results)
df = df_ini.groupby(['rho', 'stds_prior']).mean().reset_index()

xs = df.stds_prior
ys = df.rho
zs = df['eval accuracy']

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.plot_trisurf(xs, ys, zs)
ax3d.set_xlabel('std prior')
ax3d.set_ylabel('rho')
ax3d.set_zlabel('acc')
fig.show()


#%% Plot density

#NOTABLES:
# CIFAR10: 98, (604, 6),


reload_modules()
exp_nbs = ['14621', '14744', '14683', '14625', '14750', '14685', '14631', '14756', '14690']

# exp = '3713'
exp = '14621'
verbose = True
number_of_tests = 100

bay_net_trained2, arguments2, _ = su.get_trained_model_and_args_and_groupnb(exp)
evalloader_seen = su.get_evalloader_seen(arguments2, shuffle=False)
evalloader_unseen = su.get_evalloader_unseen(arguments2)

img_index_seen = np.random.randint(len(evalloader_seen))
img_index_unseen = np.random.randint(len(evalloader_unseen))

is_cifar = arguments2.get('trainset','mnist') == 'cifar10'

img_seen, target_seen = evalloader_seen.dataset.data[img_index_seen], evalloader_seen.dataset.targets[img_index_seen]
if arguments2['type_of_unseen'] == 'random':
    img_unseen = torch.randn_like(torch.tensor(img_seen).float()).numpy()
    transform = transforms.ToTensor()
else:
    img_unseen = evalloader_unseen.dataset.data[img_index_unseen].numpy()
    img_unseen = Image.fromarray(img_unseen)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=bay_net_trained2.dim_channels),
        transforms.Resize(bay_net_trained2.dim_input),
        transforms.ToTensor(),
    ])

inpt_seen = transforms.ToTensor()(img_seen)
inpt_unseen = transform(img_unseen)
def reshape_for_model(inpt):
    if len(inpt.shape) == 3:
        inpt = inpt.unsqueeze(0)
    if len(inpt.shape) == 2:
        inpt = inpt.unsqueeze(0).unsqueeze(0)
    return inpt
inpt_seen = reshape_for_model(inpt_seen)
inpt_unseen = reshape_for_model(inpt_unseen)

with torch.no_grad():
    bay_net_trained2.eval()
    sample_outputs_seen = torch.zeros((number_of_tests, 10)).to(device)
    sample_outputs_unseen = torch.zeros((number_of_tests, 10)).to(device)
    for test_idx in range(number_of_tests):
        sample_outputs_seen[test_idx] = bay_net_trained2(inpt_seen)
        sample_outputs_unseen[test_idx] = bay_net_trained2(inpt_unseen)

    labels = np.kron(np.arange(10), np.ones((number_of_tests,1))).flatten()
    densities_seen = sample_outputs_seen.numpy().flatten()
    densities_unseen = sample_outputs_unseen.numpy().flatten()

prediction = sample_outputs_seen.mean(0).argmax()

vr_seen, pe_seen, mi_seen = um.get_all_uncertainty_measures(sample_outputs_seen.unsqueeze(1))
vr_unseen, pe_unseen, mi_unseen = um.get_all_uncertainty_measures(sample_outputs_unseen.unsqueeze(1))

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221)
ax3 = fig.add_subplot(223)
ax2 = fig.add_subplot(222)
ax4 = fig.add_subplot(224)

fig.suptitle(f'Exp {exp}, Nb tests {number_of_tests} ')

ax1.imshow(img_seen)
ax2.imshow(img_unseen)

ax3.scatter(labels, densities_seen, marker='_')
ax3.set_title(f'softmax output seen. VR: {round(vr_seen.item(), 4)}, PE: {round(pe_seen.item(), 4)}, MI: {round(mi_seen.item(), 4)}')
if is_cifar:
    ax1.set_title(f'True: {cifar_labels[target_seen]}. Prediction: {cifar_labels[prediction]}. Id: {img_index_seen}')
    ax3.set_xticks(range(10))
    ax3.set_xticklabels(cifar_labels)
    ax3.tick_params(axis='x', rotation=45)
else:
    ax1.set_title(f'True: {target_seen}. Prediction: {prediction}. Id: {img_index_seen}')

ax2.set_title(f'Id: {img_index_unseen}')
ax4.scatter(labels, densities_unseen, marker='_')
ax4.set_title(f'softmax output unseen. VR: {round(vr_unseen.item(), 4)}, PE: {round(pe_unseen.item(), 4)}, MI: {round(mi_unseen.item(), 4)}')
if is_cifar:
    ax4.set_xticks(range(10))
    ax4.set_xticklabels(cifar_labels)
    ax4.tick_params(axis='x', rotation=45)

fig.show()

# %% Accuracy VS Uncertainty, histograms

reload_modules()
exp_nbs = ['14621', '14744', '14683', '14625', '14750', '14685', '14631', '14756', '14690']

# exp = '3713'
exp = '14621'
verbose = True
number_of_tests = 100

bay_net_trained, arguments, _ = su.get_trained_model_and_args_and_groupnb(exp)
evalloader_seen = su.get_evalloader_seen(arguments, shuffle=False)

all_outputs, labels = su.get_seen_outputs_and_labels(bay_net_trained, arguments, device=device, verbose=True)

vr, mi, pe = um.get_all_uncertainty_measures(all_outputs)
preds = um.get_predictions_from_multiple_tests(all_outputs)

df = (
    pd.DataFrame()
    .assign(true=labels)
    .assign(preds=preds)
    .assign(correct_pred= (df.true == df.preds))
    .assign(vr=vr)
    .assign(pe=pe)
    .assign(mi=mi)
)

df.correct_pred.sum()/len(df) ## ACCURACY TRES LOW ?  PUOURQUOIII??

# %% Accuracy vs rho, std prior 0.1

def get_ratio(df_det, unc_name):
    return df_det[f'unseen_uncertainty_{unc_name}'].iloc[0].mean() / df_det[f'seen_uncertainty_{unc_name}'].iloc[0].mean()

group_nb = 278
if group_nb == 279:
    trainset = 'mnist'
elif group_nb == 278:
    trainset = 'cifar10'
else:
    assert False, 'group not recognized'

path_to_res = f'results/raw_results/specific_columns/group{group_nb}_specific_results.pkl'
path_to_det = f'output/determinist_{trainset}/results.pkl'

df_det = pd.read_pickle(path_to_det)
u.convert_tensor_to_float(df_det)
det_acc = df_det.eval_accuracy.iloc[0]
det_us = get_ratio(df_det, 'us')
det_pe = get_ratio(df_det, 'pe')

df = pd.read_pickle(path_to_res)
df_r = df[df.type_of_unseen == 'random']
df_uc = df[df.type_of_unseen == 'unseen_classes']

df_an = df_r

fig = plt.figure(figsize=(7, 7))
fig.suptitle(f'Evolution according to rho. Trainset: {trainset}', y=1.)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


ax1.plot(df_an.rho, df_an.eval_accuracy, label='bayesian acc')
ax1.set_xlabel('rho')
ax1.set_ylabel('accuracy')
ax1.axhline(det_acc, c='r', label='determinist acc')
ax1.legend()

ax2.plot(df_an.rho, df_an['vr_unseen/seen'], label='vr')
ax2.plot(df_an.rho, df_an['pe_unseen/seen'], label='pe')
ax2.plot(df_an.rho, df_an['mi_unseen/seen'], label='mi')
ax2.axhline(det_us, label='det us', c='r')
ax2.axhline(det_pe, label='det pe', c='m')
ax2.legend()
ax2.set_xlabel('rho')
ax2.set_ylabel('uncertainty ratio unseen/seen')

fig.show()
fig.savefig(f'results/{trainset}_accuracy_vs_rho.png')

# %% Accuracy vs std_prior, rho -6

group_nb = 283
if group_nb == 282:
    trainset = 'cifar10'
elif group_nb == 283:
    trainset = 'mnist'
else:
    assert False, 'group not recognized'

path_to_res = f'results/raw_results/specific_columns/group{group_nb}_specific_results.pkl'
path_to_det = f'output/determinist_{trainset}/results.pkl'

df_det = pd.read_pickle(path_to_det)
det_acc = df_det.eval_accuracy.iloc[0]
det_us = get_ratio(df_det, 'us')
det_pe = get_ratio(df_det, 'pe')

df = pd.read_pickle(path_to_res)
df_r = df[df.type_of_unseen == 'random']
df_uc = df[df.type_of_unseen == 'unseen_classes']


df_an = df_r

fig = plt.figure(figsize=(7, 7))
fig.suptitle(f'Prior std. Trainset: {trainset}', y=1.)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(df_an.stds_prior, df_an.eval_accuracy, label='bayesian acc')
ax1.set_xlabel('stds_prior')
ax1.set_ylabel('accuracy')
ax1.axhline(det_acc, c='r', label='determinist acc')
ax1.legend()

ax2.plot(df_an.stds_prior, df_an['vr_unseen/seen'], label='vr')
ax2.plot(df_an.stds_prior, df_an['pe_unseen/seen'], label='pe')
ax2.plot(df_an.stds_prior, df_an['mi_unseen/seen'], label='mi')
ax2.axhline(det_us, label='det us', c='r')
ax2.axhline(det_pe, label='det pe', c='m')
ax2.legend()
ax2.set_xlabel('std_prior')
ax2.set_ylabel('uncertainty ratio unseen/seen')

fig.show()
fig.savefig(f'results/{trainset}_accuracy_vs_stds_prior.png')
