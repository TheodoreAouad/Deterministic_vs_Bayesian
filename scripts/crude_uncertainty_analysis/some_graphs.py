# %% Imports
import pathlib
from importlib import reload

import pandas as pd
import numpy as np
import torch
from torchvision import transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

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



def get_ratio(df_det, unc_name):
    return df_det[f'unseen_uncertainty_{unc_name}'].iloc[0].mean() / df_det[f'seen_uncertainty_{unc_name}'].iloc[0].mean()

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
path_to_res = f'output/'
exp = f'determinist_{trainset}' # DETERMINIST
# exp = '14621' # BAYESIAN
# path_to_res = 'polyaxon_results/groups'
verbose = True
number_of_tests = 1
nb_of_imgs = 1

bay_net_trained2, arguments2, _ = su.get_trained_model_and_args_and_groupnb(exp, path_to_res)
is_determinist = arguments2.get('determinist', False) or arguments2.get('rho', 'determinist') == 'determinist'

evalloader_seen = su.get_evalloader_seen(arguments2, shuffle=False)
evalloader_unseen = su.get_evalloader_unseen(arguments2)

for _ in range(nb_of_imgs):
    img_index_seen = np.random.randint(len(evalloader_seen))
    img_index_seen = 532
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


    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(211)
    ax3 = fig.add_subplot(223)
    # ax2 = fig.add_subplot(222)
    ax4 = fig.add_subplot(224)

    fig.suptitle(f'Exp {exp}, Nb tests {number_of_tests} ', x=0.16, y=0.8)

    ax1.imshow(img_seen)
    # ax2.imshow(img_unseen)
    sns.heatmap(sample_outputs_seen, ax=ax4)


    ax3.scatter(labels, densities_seen, marker='_')
    if is_determinist:
        sr_seen, pe_seen = um.get_all_uncertainty_measures_not_bayesian(sample_outputs_seen.unsqueeze(1))
        ax3.set_title(f'softmax output seen. SR: {round(sr_seen.item(), 4)}, PE: {round(pe_seen.item(), 4)}')
    else:
        vr_seen, pe_seen, mi_seen = um.get_all_uncertainty_measures(sample_outputs_seen.unsqueeze(1))
        vr_unseen, pe_unseen, mi_unseen = um.get_all_uncertainty_measures(sample_outputs_unseen.unsqueeze(1))
        ax3.set_title(f'softmax output seen. VR: {round(vr_seen.item(), 4)}, PE: {round(pe_seen.item(), 4)}, MI: {round(mi_seen.item(), 4)}')
    if is_cifar:
        ax1.set_title(f'True: {cifar_labels[target_seen]}. Prediction: {cifar_labels[prediction]}. Id: {img_index_seen}')
        ax3.set_xticks(range(10))
        ax3.set_xticklabels(cifar_labels)
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax1.set_title(f'True: {target_seen}. Prediction: {prediction}. Id: {img_index_seen}')

    # ax2.set_title(f'Id: {img_index_unseen}')
    # ax4.scatter(labels, densities_unseen, marker='_')
    # ax4.set_title(f'softmax output unseen. VR: {round(vr_unseen.item(), 4)}, PE: {round(pe_unseen.item(), 4)}, MI: {round(mi_unseen.item(), 4)}')
    if is_cifar:
        ax4.set_xticks(range(10))
        ax4.set_xticklabels(cifar_labels)
        ax4.tick_params(axis='x', rotation=45)

    fig.show()


    save_path = 'results/images/softmax_output'
    save_path = pathlib.Path(save_path)
    save_fig = False

    if save_fig:
        save_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path/f'softmax_output_{exp}_{img_index_seen}.png')
        u.save_to_file(fig, save_path/f'softmax_output_{exp}_{img_index_seen}.pkl')


# %% Accuracy vs rho, std prior 0.1

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

group_nb = 290
if group_nb == 291:
    trainset = 'cifar10'
elif group_nb == 290:
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
df_r = df[df.type_of_unseen == 'unseen_dataset']
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
# fig.savefig(f'results/{trainset}_accuracy_vs_stds_prior.png')

# %% Compute all outputs.

reload_modules()

trainset = 'cifar10'
exp_nbs = ['14621', '14744', '14683', '14625', '14750', '14685', '14631', '14756', '14690']


path_to_res = f'output/'
exp = f'determinist_{trainset}' # DETERMINIST
exp = '14621' # BAYESIAN
path_to_res = 'polyaxon_results/groups'

# exp = '3713'
verbose = True
number_of_tests = 10

bay_net_trained, arguments, _ = su.get_trained_model_and_args_and_groupnb(exp, path_to_res)
evalloader_seen = su.get_evalloader_seen(arguments, shuffle=False)

dont_show = ['save_loss', 'save_observables', 'save_outputs', 'type_of_unseen', 'unseen_evalset',
             'split_labels', 'number_of_tests', 'split_train', 'exp_nb']

is_determinist = arguments.get('determinist', False) or arguments.get('rho', 'determinist') == 'determinist'

if is_determinist:
    number_of_tests = 1
labels, all_outputs = e.eval_bayesian(
    bay_net_trained,
    evalloader_seen,
    number_of_tests=number_of_tests,
    return_accuracy=False,
    verbose=True,
)

preds = um.get_predictions_from_multiple_tests(all_outputs)
res = pd.DataFrame()
if is_determinist:
    sr, pe = um.get_all_uncertainty_measures_not_bayesian(all_outputs)
    res = (
        res
        .assign(true=labels)
        .assign(preds=preds)
        .assign(correct_pred=lambda df: (df.true == df.preds))
        .assign(sr=sr)
        .assign(pe=pe)
    )
else:
    vr, mi, pe = um.get_all_uncertainty_measures(all_outputs)
    res = (
        res
        .assign(true=labels)
        .assign(preds=preds)
        .assign(correct_pred=lambda df: (df.true == df.preds))
        .assign(vr=vr)
        .assign(pe=pe)
        .assign(mi=mi)
    )


res_correct = res.loc[res.correct_pred == True]
res_false = res.loc[res.correct_pred == False]
print('Accuracy: ', res.correct_pred.mean())

#%% Density graphs - determinist

if is_determinist:
    save_path = 'rapport/determinist_failure/'
else:
    save_path = 'rapport/bayesian_results/'
save_path = pathlib.Path(save_path)
save_fig = False

if is_determinist:
    uncs_sr = [res_correct.sr, res_false.sr]
    uncs_pe = [res_correct.pe, res_false.pe]
    all_uncs = [uncs_sr, uncs_pe]
    unc_names = ['sr', 'pe']
else:
    uncs_vr = [res_correct.vr, res_false.vr]
    uncs_pe = [res_correct.pe, res_false.pe]
    uncs_mi = [res_correct.mi, res_false.mi]
    all_uncs = [uncs_vr, uncs_pe, uncs_mi]
    unc_names = ['vr', 'pe', 'mi']

unc_labels = ('true', 'false')
fig = plt.figure()
for idx, (unc_name, uncs) in enumerate(zip(unc_names, all_uncs)):
    ax = fig.add_subplot(len(unc_names), 1, idx+1)
    ax.set_title(unc_name)
    ax.set_xlabel(unc_name)
    ax.set_ylabel('density')
    u.plot_density_on_ax(ax, uncs, unc_labels, hist=True,)
    ax.legend()

fig.show()

if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'density_uncertainty_{exp}.png')
    u.save_to_file(fig, save_path/f'density_uncertainty_{exp}.pkl')

# %% Density graph 2


if is_determinist:
    save_path = 'rapport/determinist_failure/'
else:
    save_path = 'rapport/bayesian_results/'
save_path = pathlib.Path(save_path)
save_fig = False

if is_determinist:
    unc_names = ['sr', 'pe']
else:
    unc_names = ['vr', 'pe', 'mi']


res.sample(frac=1)

fig = plt.figure(figsize=(5, 4))
axs = {}
for idx, unc_name in enumerate(unc_names):
    to_plot_true = res_correct[unc_name]
    to_plot_false = res_false[unc_name]
    ax = fig.add_subplot(len(unc_names), 1, idx+1)
    ax.set_title(unc_name)
    ax.scatter(to_plot_true, np.ones_like(to_plot_true), marker='|')
    ax.scatter(to_plot_false, np.zeros_like(to_plot_false), marker='|')
    ax.set_xlabel(unc_name)
    ax.set_yticks(range(2))
    ax.set_yticklabels(['False', 'True'])
    axs[unc_name] = ax

fig.suptitle(f'Uncertainty Repartition. Exp:{exp}. T={number_of_tests}'
             # f'\n {dict({k:v for k,v in arguments.items() if k not in dont_show})}',
             # wrap=True
             )
fig.show()

if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'scatterplot_uncertainty_{exp}.png')
    u.save_to_file(fig, save_path/f'scatterplot_uncertainty_{exp}.pkl')
    print(f'saved to f{save_path/f"scatterplot_uncertainty_{exp}.png"}')

# %% Acc vs unc


if is_determinist:
    save_path = 'rapport/determinist_failure/'
else:
    save_path = 'rapport/bayesian_results/'

save_path = pathlib.Path(save_path)
save_fig = False

nb_of_points = 10000
size_of_points = 30

interval = 20
nb_of_ratios = nb_of_points // interval
ratios = np.linspace(0, 1, nb_of_ratios)

to_plot = pd.DataFrame(columns=[
    'acc',
    'unc',
    'unc_name',
    'ratio',
    'size_of_points',
])

if is_determinist:
    unc_names = ['sr', 'pe']
else:
    unc_names = ['vr', 'pe', 'mi']

for ratio in ratios:
    nb_of_correct = int(ratio*size_of_points)
    nb_of_false = size_of_points - nb_of_correct
    current_points = (
        res_correct.sample(frac=1)[:nb_of_correct]
        .append(res_false.sample(frac=1)[:nb_of_false])
    )
    for unc_name in unc_names:
        to_plot = to_plot.append(pd.DataFrame.from_dict({
            'acc': [current_points.correct_pred.mean()],
            'unc': [current_points[unc_name].mean()],
            'unc_name': [unc_name],
            'size_of_points': [size_of_points],
        }), sort=True)

fig = plt.figure(figsize=(9, 9))

fig.suptitle(f'Uncertainty Repartition. Exp:{exp}. T={number_of_tests}'
             # f'\n {dict({k:v for k,v in arguments.items() if k not in dont_show})}',
             # wrap=True
             , y=1)
axs = {}
for idx, unc_name in enumerate(unc_names):
    to_plot_unc = to_plot.loc[to_plot.unc_name == unc_name]
    ax = fig.add_subplot(len(unc_names), 1, idx+1)
    ax.set_title(unc_name)
    ax.scatter(to_plot_unc.unc, to_plot_unc.acc)
    ax.set_xlabel(unc_name)
    ax.set_ylabel('acc')
    axs[unc_name] = ax

fig.show()

if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'acc_unc_{exp}.png')
    u.save_to_file(fig, save_path/f'acc_unc_{exp}.pkl')
    print(f'saved to f{save_path / f"acc_unc_{exp}.png"}')


# %% Selective classification

if is_determinist:
    save_path = 'rapport/determinist_failure/'
    unc_names = ['sr', 'pe']
else:
    save_path = 'rapport/bayesian_results/'
    unc_names = ['vr', 'pe', 'mi']
save_path = pathlib.Path(save_path)
save_fig = False

def oracle(x, acc):
    return np.minimum(1, acc/np.maximum(x, 0.0001))

coverage_array = np.arange(1, 1+len(res))/len(res)
model_acc = res.correct_pred.mean()

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
ax.plot([0, 1], [model_acc, model_acc], label='lower_bound')
ax.plot(coverage_array, oracle(coverage_array, model_acc), label='oracle')
for idx, unc_name in enumerate(unc_names):
    sorted_unc = res.sort_values(unc_name)
    sorted_unc.loc[sorted_unc[unc_name] < 0, unc_name] = 0

    sorted_unc = (
        sorted_unc
        .assign(cum_accs=lambda df: df.correct_pred.cumsum()/np.arange(1, len(sorted_unc)+1))
    )

    ax.plot(coverage_array, sorted_unc.cum_accs, label=unc_name)
ax.set_xlabel('coverage')
ax.set_ylabel('accuracy')
ax.legend()

fig.suptitle(f'Accuracy - Coverage. Exp:{exp}. T={number_of_tests}'
             # f'\n {dict({k:v for k,v in arguments.items() if k not in dont_show})}',
             # wrap=True, va='baseline',
             )
fig.show()
if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'acc_cov_{exp}.png')
    u.save_to_file(fig, save_path/f'acc_cov_{exp}.pkl')
    print(f'Saved to {save_path/f"acc_cov_{exp}.png"}')
