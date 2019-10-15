# %% Imports
import pathlib
from importlib import reload
import os

import pandas as pd
import numpy as np
import torch
from torchvision import transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm

import scripts.utils as su
import src.uncertainty_measures as um
import src.utils as u
import src.tasks.evals as e
import src.grapher as g

def reload_modules():
    modules_to_reload = [su, u, um, e, g]
    for module in modules_to_reload:
        reload(module)

cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']
reload_modules()

if torch.cuda.is_available():
    device = "cuda"
    path_to_res = '/output/sicara/BayesianFewShotExperiments/groups/'
else:
    device = "cpu"
    path_to_res = 'polyaxon_results/groups'
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

path_to_results = 'results/raw_results/all_columns/group323.pkl'

df_ini = pd.read_pickle(path_to_results)
df = df_ini.groupby(['rho', 'stds_prior']).mean().reset_index()

# df = df.loc[df['stds_prior'] < 5]

max_acc_params = df.sort_values('eval_accuracy').iloc[-1]
rho_star = max_acc_params.rho
std_star = max_acc_params.stds_prior
eval_star = max_acc_params.eval_accuracy

xs = df.stds_prior
ys = df.rho
zs = df['eval_accuracy']

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.plot((std_star, std_star), (rho_star, rho_star), (eval_star, 0.8), zorder=0, linestyle='solid', linewidth=5)
ax3d.text(std_star, rho_star, 0.8, f' Acc={round(100*eval_star, 2)}%, Rho={rho_star}, Std_prior={std_star}')
ax3d.plot_trisurf(xs, ys, zs, zorder=-1)
ax3d.set_xlabel('std prior')
ax3d.set_ylabel('rho')
ax3d.set_zlabel('acc')
# ax3d.annotate(f'Max Values: rho {max_acc_params.rho}, std prior {max_acc_params.stds_prior}',
#               xy=(max_acc_params.rho, max_acc_params.stds_prior), xycoords='data',
#               xytext=(1, 1), textcoords='offset points')
ax3d.view_init(15, 50)
fig.suptitle('Accuracy w.r.t. initial variance of posterior and prior')
fig.show()

#%% 3d plot ratio vs (rho / std_prior)

path_to_results = 'results/raw_results/specific_columns/group301_specific_results.pkl'

unc= 'vr'
param = f'{unc}_unseen/seen'
df_ini = pd.read_pickle(path_to_results)
df = df_ini.groupby(['rho', 'stds_prior']).mean().reset_index()

# df = df.loc[df['stds_prior'] < 5]

max_params = df.sort_values(param).iloc[-1]
rho_star = max_params.rho
std_star = max_params.stds_prior
eval_star = max_params[param]


xs = df.stds_prior
ys = df.rho
zs = df[param]

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.plot((std_star, std_star), (rho_star, rho_star), (eval_star, 0.8), zorder=0, linestyle='solid', linewidth=5)
ax3d.text(10, 0, 0.8, f' Ratio={round(eval_star,2)}, Rho={rho_star}, Std_prior={std_star}')
ax3d.plot_trisurf(xs, ys, zs, zorder=-1)
ax3d.set_xlabel('std prior')
ax3d.set_ylabel('rho')
ax3d.set_zlabel(param)
# ax3d.annotate(f'Max Values: rho {max_params.rho}, std prior {max_params.stds_prior}',
#               xy=(max_params.rho, max_params.stds_prior), xycoords='data',
#               xytext=(1, 1), textcoords='offset points')
ax3d.view_init(15, 50)
fig.suptitle(f'{param} w.r.t. initial variance of posterior and prior')
fig.show()


#%% Plot density



reload_modules()
exp_nbs = ['14621', '14744', '14683', '14625', '14750', '14685', '14631', '14756', '14690']

# exp = '3713'
# path_to_res = f'output/'
# exp = f'determinist_{trainset}' # DETERMINIST
exp = '20321' # BAYESIAN
# exp = 14621
path_to_res = 'polyaxon_results/groups'
verbose = True
number_of_tests = 20
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
        vr_seen, pe_seen, mi_seen = um.get_all_uncertainty_measures_bayesian(sample_outputs_seen.unsqueeze(1))
        vr_unseen, pe_unseen, mi_unseen = um.get_all_uncertainty_measures_bayesian(sample_outputs_unseen.unsqueeze(1))
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
save_fig = False

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
if save_fig:
    fig.savefig(f'results/{trainset}_accuracy_vs_stds_prior.png')

# %% Compute det outputs

reload_modules()

trainset = 'mnist'
res_det = pd.DataFrame()


det_net_trained, arguments, _ = su.get_trained_model_and_args_and_groupnb(f'determinist_{trainset}', f'output/')
evalloader_seen = su.get_evalloader_seen(arguments, shuffle=False)

labels, all_outputs = e.eval_bayesian(
        det_net_trained,
        evalloader_seen,
        number_of_tests=1,
        return_accuracy=False,
        verbose=True,
    )

preds = um.get_predictions_from_multiple_tests(all_outputs)
sr, pe = um.get_all_uncertainty_measures_not_bayesian(all_outputs)
res_det = (
    res_det
    .assign(true=labels)
    .assign(preds=preds)
    .assign(correct_pred=lambda df: (df.true == df.preds))
    .assign(sr=sr)
    .assign(pe=pe)
)

arguments['type_of_unseen'] = 'unseen_dataset'
evalloader_unseen = su.get_evalloader_unseen(arguments, shuffle=False)
_, all_outputs = e.eval_bayesian(
        det_net_trained,
        evalloader_unseen,
        number_of_tests=1,
        return_accuracy=False,
        verbose=True,
    )

sr_unseen, pe_unseen = um.get_all_uncertainty_measures_not_bayesian(all_outputs)
res_det_unseen = (
    pd.DataFrame()
    .assign(sr=sr_unseen)
    .assign(pe=pe_unseen)
)


# %% Get bayesian args.
exp = '20273' # BAYESIAN
res = {}
res_unseen_bay = {}
# %% Compute all outputs
pass
# exp = '3713'
path_to_outputs = pathlib.Path(f'temp/softmax_outputs/')
reload_modules()
verbose = True
list_of_nb_of_tests = [1, 5, 10, 15, 20, 50, 100]

bay_net_trained, arguments, _ = su.get_trained_model_and_args_and_groupnb(exp, path_to_res)
evalloader_seen = su.get_evalloader_seen(arguments, shuffle=False)
arguments['type_of_unseen'] = 'unseen_dataset'
evalloader_unseen = su.get_evalloader_unseen(arguments, shuffle=False)

unc_names_bay = ['sr', 'vr', 'pe', 'mi', 'au', 'eu',]

dont_show = ['save_loss', 'save_observables', 'save_outputs', 'type_of_unseen', 'unseen_evalset',
             'split_labels', 'number_of_tests', 'split_train', 'exp_nb']

is_determinist = arguments.get('determinist', False) or arguments.get('rho', 'determinist') == 'determinist'
assert not is_determinist, 'network determinist'

for number_of_tests in list_of_nb_of_tests:

    bay_net_trained, arguments, _ = su.get_trained_model_and_args_and_groupnb(exp, path_to_res)
    bay_net_trained.to(device)
    if number_of_tests < 100 and os.path.exists(path_to_outputs / '100' / f'{exp}/true_labels_seen.pt'):

        true_labels_seen = torch.load(path_to_outputs / f'100' / f'{exp}/true_labels_seen.pt')
        all_outputs_seen = torch.load(path_to_outputs / f'100' / f'{exp}/all_outputs_seen.pt')
        all_outputs_unseen = torch.load(path_to_outputs / f'100' / f'{exp}/all_outputs_unseen.pt')

        random_idx = np.arange(100)
        np.random.shuffle(random_idx)
        random_idx = random_idx[:number_of_tests]
        all_outputs_seen = all_outputs_seen[random_idx]
        all_outputs_unseen = all_outputs_unseen[random_idx]
    elif os.path.exists(path_to_outputs / f'{number_of_tests}' / f'{exp}/true_labels_seen.pt'):
        true_labels_seen = torch.load(path_to_outputs / f'{number_of_tests}' / f'{exp}/true_labels_seen.pt')
        all_outputs_seen = torch.load(path_to_outputs / f'{number_of_tests}' / f'{exp}/all_outputs_seen.pt')
        all_outputs_unseen = torch.load(path_to_outputs / f'{number_of_tests}' / f'{exp}/all_outputs_unseen.pt')
    else:
        (path_to_outputs / f'{number_of_tests}' / f'{exp}').mkdir(exist_ok=True, parents=True)
        evalloader_seen = su.get_evalloader_seen(arguments)
        # BE CAREFUL: in the paper, the process is tested on the enterity of the unseen classes
        evalloader_unseen = su.get_evalloader_unseen(arguments)
        true_labels_seen, all_outputs_seen = e.eval_bayesian(
            model=bay_net_trained,
            evalloader=evalloader_seen,
            number_of_tests=number_of_tests,
            return_accuracy=False,
            device=device,
            verbose=True,
        )

        _, all_outputs_unseen = e.eval_bayesian(
            model=bay_net_trained,
            evalloader=evalloader_unseen,
            number_of_tests=number_of_tests,
            device=device,
            verbose=True,
        )

        torch.save(true_labels_seen, path_to_outputs / f'{number_of_tests}' / f'{exp}/true_labels_seen.pt')
        torch.save(all_outputs_seen, path_to_outputs / f'{number_of_tests}' / f'{exp}/all_outputs_seen.pt')
        torch.save(all_outputs_unseen, path_to_outputs / f'{number_of_tests}' / f'{exp}/all_outputs_unseen.pt')
    #
    # true_labels_seen, all_outputs_seen = e.eval_bayesian(
    #     bay_net_trained,
    #     evalloader_seen,
    #     number_of_tests=number_of_tests,
    #     return_accuracy=False,
    #     verbose=True,
    # )

    all_outputs_seen, all_outputs_unseen = all_outputs_seen.cpu(), all_outputs_unseen.cpu()
    true_labels_seen = true_labels_seen.cpu()
    preds = um.get_predictions_from_multiple_tests(all_outputs_seen)

    res[number_of_tests] = pd.DataFrame()
    uncs = um.get_all_uncertainty_measures(all_outputs_seen)
    res[number_of_tests] = (
        res[number_of_tests]
        .assign(true=true_labels_seen)
        .assign(preds=preds)
        .assign(correct_pred=lambda df: (df.true == df.preds))
    )
    for unc, unc_name in zip(uncs, unc_names_bay):
        res[number_of_tests][unc_name] = unc
    #
    # _, all_outputs_unseen = e.eval_bayesian(
    #     bay_net_trained,
    #     evalloader_unseen,
    #     number_of_tests=number_of_tests,
    #     return_accuracy=False,
    #     verbose=True,
    # )

    res_unseen_bay[number_of_tests] = pd.DataFrame()
    uncs_unseen = um.get_all_uncertainty_measures(all_outputs_unseen)
    for unc, unc_name in zip(uncs, unc_names_bay):
        res_unseen_bay[number_of_tests][unc_name] = unc

    print(f'Nb Of Tests: {number_of_tests}. Accuracy: ', res[number_of_tests].correct_pred.mean())

#%% Density graphs

reload_modules()

show_determinist = False
number_of_tests = 10
show_determinist = True

res_correct = res[number_of_tests].loc[res[number_of_tests].correct_pred == True]
res_false = res[number_of_tests].loc[res[number_of_tests].correct_pred == False]
res_unseen = res_unseen_bay[number_of_tests]

if show_determinist:
    save_path = 'rapport/determinist_failure/'
    res_correct = res_det.loc[res_det.correct_pred == True]
    res_false = res_det.loc[res_det.correct_pred == False]
    res_unseen = res_det_unseen
else:
    save_path = f'rapport/bayesian_results/{exp}'
save_path = pathlib.Path(save_path)
save_fig = False

if show_determinist:
    unc_names = ['sr', 'pe']
else:
    unc_names = ['sr', 'vr', 'pe', 'mi', ]#'au', 'eu']
all_uncs = [[res_correct[unc_name], res_false[unc_name], res_unseen[unc_name]] for unc_name in unc_names]

unc_labels = ('true', 'false', 'unseen')
fig = plt.figure(figsize=(5, 9))
for idx, (unc_name, uncs) in enumerate(zip(unc_names, all_uncs)):
    ax = fig.add_subplot(len(unc_names), 1, idx+1)
    ax.set_title(unc_name)
    uncs_to_show = [unc.loc[unc < 1*unc.max()] for unc in uncs]
    g.plot_uncertainty(
        ax,
        unc_name,
        uncs,
        unc_labels,
    )
    ax.legend()

fig.show()

if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'density_uncertainty_{exp}.png')
    u.save_to_file(fig, save_path/f'density_uncertainty_{exp}.pkl')

# %% Density graph 2

show_determinist = False
number_of_tests = 10
# show_determinist = True

res_correct = res[number_of_tests].loc[res[number_of_tests].correct_pred == True]
res_false = res[number_of_tests].loc[res[number_of_tests].correct_pred == False]
res_unseen = res_unseen_bay[number_of_tests]

if show_determinist:
    save_path = 'rapport/determinist_failure/'
else:
    save_path = f'rapport/bayesian_results/{exp}'
save_path = pathlib.Path(save_path)
save_fig = False

if show_determinist:
    unc_names = ['sr', 'pe']
    res_correct = res_det.loc[res_det.correct_pred == True]
    res_false = res_det.loc[res_det.correct_pred == False]
    res_unseen = res_det_unseen
else:
    unc_names = ['sr', 'vr', 'pe', 'mi']


res[number_of_tests].sample(frac=1)

fig = plt.figure(figsize=(5, 8))
axs = {}
for idx, unc_name in enumerate(unc_names):
    to_plot_true = res_correct[unc_name]
    to_plot_false = res_false[unc_name]
    to_plot_unseen = res_unseen[unc_name]
    ax = fig.add_subplot(len(unc_names), 1, idx+1)
    ax.set_title(unc_name)
    ax.scatter(to_plot_true, 1 + np.ones_like(to_plot_true), marker='|')
    ax.scatter(to_plot_false, 1 + np.zeros_like(to_plot_false), marker='|')
    ax.scatter(to_plot_unseen, np.zeros_like(to_plot_unseen), marker='|')
    # ax.set_xlabel(unc_name)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['MNIST', 'CIFAR - False', 'CIFAR - True'])

# fig.suptitle(f'Uncertainty Repartition.  T={number_of_tests}'
#              # f'\n {dict({k:v for k,v in arguments.items() if k not in dont_show})}',
#              # wrap=True
#              , y=1)
fig.show()

if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'scatterplot_uncertainty_{exp}.png')
    u.save_to_file(fig, save_path/f'scatterplot_uncertainty_{exp}.pkl')
    print(f'saved to f{save_path/f"scatterplot_uncertainty_{exp}.png"}')

# %% Acc vs unc - mixed up

show_determinist = False
number_of_tests = 100
# show_determinist = True

x_size = 12
y_size = 7


if show_determinist:
    save_path = 'rapport/determinist_failure/'
    res_correct = res_det.loc[res_det.correct_pred == True]
    res_false = res_det.loc[res_det.correct_pred == False]
else:
    res_correct = res[number_of_tests].loc[res[number_of_tests].correct_pred == True]
    res_false = res[number_of_tests].loc[res[number_of_tests].correct_pred == False]
    save_path = f'rapport/bayesian_results/{exp}'

save_path = pathlib.Path(save_path)
save_fig = False

nb_of_points = 10000
size_of_points = 50

interval = 50
nb_of_ratios = nb_of_points // interval
ratios = np.linspace(0, 1, nb_of_ratios)

to_plot = pd.DataFrame(columns=[
    'acc',
    'unc',
    'unc_name',
    'ratio',
    'size_of_points',
])

if show_determinist:
    unc_names = ['sr', 'pe']
else:
    unc_names = ['sr', 'vr', 'pe', 'mi']#, 'au', 'eu',]

for ratio in tqdm(ratios):
    nb_of_correct = int(ratio*size_of_points)
    nb_of_false = size_of_points - nb_of_correct
    current_points = (
        res_correct.sample(frac=1)[:nb_of_correct]
        .append(res_false.sample(frac=1)[:nb_of_false])
    )
    for unc_name in unc_names:
        to_plot = to_plot.append(pd.DataFrame.from_dict({
            'acc': [current_points.correct_pred.mean()],
            'unc': [current_points[unc_name]],
            'unc_mean': [current_points[unc_name].mean()],
            'unc_median': [current_points[unc_name].median()],
            'unc_std': [current_points[unc_name].std()],
            'unc_name': [unc_name],
            'size_of_points': [size_of_points],
        }), sort=True)

if show_determinist:
    fig = plt.figure(figsize=(x_size, y_size))
else:
    fig = plt.figure(figsize=(x_size, 2*y_size))

# fig.suptitle(f'Uncertainty Repartition. Exp:{exp}. T={number_of_tests}'
#              # f'\n {dict({k:v for k,v in arguments.items() if k not in dont_show})}',
#              # wrap=True
#              , y=1)
axs = {}
if not show_determinist:
    unc_names = ['sr','pe', 'vr', 'mi']#, 'au', 'eu']
for idx, unc_name in enumerate(unc_names):
    to_plot_unc = to_plot.loc[to_plot.unc_name == unc_name]
    # initiate axes
    ax_mean = fig.add_subplot(len(unc_names), 2, 2*idx+1)
    ax_median = fig.add_subplot(len(unc_names), 2, 2*idx+2)
    # set titles
    ax_mean.set_title(f'{unc_name} - mean of {size_of_points} imgs - cor: {round(np.corrcoef(to_plot_unc.unc_mean, to_plot_unc.acc)[0,1], 2)}')
    ax_median.set_title(f'{unc_name} - median of {size_of_points} imgs - cor: {round(np.corrcoef(to_plot_unc.unc_median, to_plot_unc.acc)[0,1], 2)}')
    # plot scatterplots
    scat_mean = ax_mean.scatter(to_plot_unc.unc_mean, to_plot_unc.acc, c=to_plot_unc.unc_std)
    scat_median = ax_median.scatter(to_plot_unc.unc_median, to_plot_unc.acc, c=to_plot_unc.unc_std)
    # put labels
    ax_mean.set_xlabel(unc_name)
    ax_mean.set_ylabel('acc')
    ax_median.set_xlabel(unc_name)
    ax_median.set_ylabel('acc')
    # set frame
    xmax = max(to_plot_unc.unc_mean.max(), to_plot_unc.unc_median.max())
    ax_mean.set_xlim(left=0, right=xmax)
    ax_median.set_xlim(left=0, right=xmax)
    # but colorbars
    cbar_mean = plt.colorbar(scat_mean, ax=ax_mean)
    cbar_mean.set_label(f'std', rotation=270)
    cbar_median = plt.colorbar(scat_median, ax=ax_median)
    cbar_median.set_label(f'std', rotation=270)

fig.tight_layout()
fig.show()

if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'acc_unc_{exp}.png')
    u.save_to_file(fig, save_path/f'acc_unc_{exp}.pkl')
    print(f'saved to f{save_path / f"acc_unc_{exp}.png"}')

# %% Acc vs unc - correlation

show_determinist = False
list_of_number_of_tests = [1, 5, 10, 15, 20, 50, 100]
# show_determinist = True

x_size = 12
y_size = 7


if show_determinist:
    save_path = 'rapport/determinist_failure/'
    res_correct = res_det.loc[res_det.correct_pred == True]
    res_false = res_det.loc[res_det.correct_pred == False]
else:
    res_correct = res[number_of_tests].loc[res[number_of_tests].correct_pred == True]
    res_false = res[number_of_tests].loc[res[number_of_tests].correct_pred == False]
    save_path = f'rapport/bayesian_results/{exp}'

save_path = pathlib.Path(save_path)
save_fig = False

nb_of_points = 10000
size_of_points = 50

interval = 50
nb_of_ratios = nb_of_points // interval
ratios = np.linspace(0, 1, nb_of_ratios)

to_plot = pd.DataFrame(columns=[
    'acc',
    'unc',
    'unc_name',
    'ratio',
    'size_of_points',
])

if show_determinist:
    unc_names = ['sr', 'pe']
else:
    unc_names = ['sr', 'vr', 'pe', 'mi']#, 'au', 'eu',]

for number_of_tests in list_of_number_of_tests:
    for ratio in tqdm(ratios):
        nb_of_correct = int(ratio*size_of_points)
        nb_of_false = size_of_points - nb_of_correct
        current_points = (
            res_correct.sample(frac=1)[:nb_of_correct]
            .append(res_false.sample(frac=1)[:nb_of_false])
        )
        for unc_name in unc_names:
            to_plot = to_plot.append(
                pd.DataFrame.from_dict({
                    'acc': [current_points.correct_pred.mean()],
                    'unc': [current_points[unc_name]],
                    'unc_mean': [current_points[unc_name].mean()],
                    'unc_median': [current_points[unc_name].median()],
                    'unc_std': [current_points[unc_name].std()],
                    'unc_name': [unc_name],
                    'size_of_points': [size_of_points],
                    'number_of_tests': [number_of_tests],
                    'corr_mean': [np.corrcoef(current_points[unc_name].mean(), current_points.correct_pred.mean())[0, 1]],
                    'corr_median': [np.corrcoef(current_points[unc_name].median(), current_points.correct_pred.mean())[0, 1]],
            }), sort=True)


if show_determinist:
    fig = plt.figure(figsize=(x_size, y_size))
else:
    fig = plt.figure(figsize=(x_size, 2*y_size))

# fig.suptitle(f'Uncertainty Repartition. Exp:{exp}. T={number_of_tests}'
#              # f'\n {dict({k:v for k,v in arguments.items() if k not in dont_show})}',
#              # wrap=True
#              , y=1)
axs = {}
if not show_determinist:
    unc_names = ['sr','pe', 'vr', 'mi']#, 'au', 'eu']
for idx, unc_name in enumerate(unc_names):
    to_plot_unc = to_plot.loc[to_plot.unc_name == unc_name]
    grps = to_plot_unc.groupby('number_of_tests')
    corr_mean = grps.apply(lambda df: np.corrcoef(df.unc_mean, df.acc)[0,1])
    corr_median = grps.apply(lambda df: np.corrcoef(df.unc_median, df.acc)[0,1])
    # initiate axes
    ax_mean = fig.add_subplot(len(unc_names), 2, 2*idx+1)
    ax_median = fig.add_subplot(len(unc_names), 2, 2*idx+2)
    # set titles
    ax_mean.set_title(f'{unc_name} - mean of {size_of_points} imgs - cor: {round(np.corrcoef(to_plot_unc.unc_mean, to_plot_unc.acc)[0,1], 2)}')
    ax_median.set_title(f'{unc_name} - median of {size_of_points} imgs - cor: {round(np.corrcoef(to_plot_unc.unc_median, to_plot_unc.acc)[0,1], 2)}')
    # plot scatterplots
    scat_mean = ax_mean.plot(grps.number_of_tests.mean(), corr_mean)
    scat_median = ax_median.plot(grps.number_of_tests.mean(), corr_median)
    # put labels
    ax_mean.set_xlabel(unc_name)
    ax_mean.set_ylabel('acc')
    ax_median.set_xlabel(unc_name)
    ax_median.set_ylabel('acc')
    # set frame
    # xmax = max(to_plot_unc.unc_mean.max(), to_plot_unc.unc_median.max())
    # ax_mean.set_xlim(left=0, right=xmax)
    # ax_median.set_xlim(left=0, right=xmax)

fig.tight_layout()
fig.show()

if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'acc_unc_{exp}.png')
    u.save_to_file(fig, save_path/f'corr_unc_{exp}.pkl')
    print(f'saved to f{save_path / f"corr_unc_{exp}.png"}')


# %% Bayesian VS Determinist, nice layout of the graphs

save_path = 'rapport/bayesian_vs_det/'
save_path = pathlib.Path(save_path)
save_fig = False

fig = plt.figure(figsize=(12, 10))

fig.suptitle(f'Uncertainty Repartition. Exp:{exp}. T={number_of_tests}'
             # f'\n {dict({k:v for k,v in arguments.items() if k not in dont_show})}',
             # wrap=True
             , y=1)
axs = {}
# we want both to_plot_bbb and to_plot_det


def plot_on_figure_comparison(to_plot_bbb, to_plot_det):
    unc_names = ['sr', 'pe']
    for idx, unc_name in enumerate(unc_names):
        to_plot_unc_det = to_plot_det.loc[to_plot_det.unc_name == unc_name]
        to_plot_unc_bbb = to_plot_bbb.loc[to_plot_bbb.unc_name == unc_name]
        # initiate axes
        ax_mean_det = fig.add_subplot(4, 2, 4*idx+1)
        ax_median_det = fig.add_subplot(4, 2, 4*idx+3)
        ax_mean_bbb = fig.add_subplot(4, 2, 4*idx+2)
        ax_median_bbb = fig.add_subplot(4, 2, 4*idx+4)
        # set titles
        ax_mean_det.set_title(f'deterministic: {unc_name} - mean of {size_of_points} imgs')
        ax_median_det.set_title(f'deterministic: {unc_name} - median of {size_of_points} imgs')
        ax_mean_bbb.set_title(f'bayesian: {unc_name} - mean of {size_of_points} imgs')
        ax_median_bbb.set_title(f'bayesian: {unc_name} - median of {size_of_points} imgs')
        # plot scatterplots
        scat_mean_det = ax_mean_det.scatter(to_plot_unc_det.unc_mean, to_plot_unc_det.acc, c=to_plot_unc_det.unc_std)
        scat_median_det = ax_median_det.scatter(to_plot_unc_det.unc_median, to_plot_unc_det.acc, c=to_plot_unc_det.unc_std)
        scat_mean_bbb = ax_mean_bbb.scatter(to_plot_unc_bbb.unc_mean, to_plot_unc_bbb.acc, c=to_plot_unc_bbb.unc_std)
        scat_median_bbb = ax_median_bbb.scatter(to_plot_unc_bbb.unc_median, to_plot_unc_bbb.acc, c=to_plot_unc_bbb.unc_std)
        # put labels
        ax_mean_det.set_xlabel(unc_name)
        ax_mean_det.set_ylabel('acc')
        ax_median_det.set_xlabel(unc_name)
        ax_median_det.set_ylabel('acc')
        ax_mean_bbb.set_xlabel(unc_name)
        ax_mean_bbb.set_ylabel('acc')
        ax_median_bbb.set_xlabel(unc_name)
        ax_median_bbb.set_ylabel('acc')
        # set frame
        xmax = max([ax.get_xlim()[1] for ax in [ax_mean_det, ax_mean_bbb, ax_median_det, ax_median_bbb]])
        ax_mean_det.set_xlim(left=0, right=xmax)
        ax_median_det.set_xlim(left=-0.001, right=xmax)
        ax_mean_bbb.set_xlim(left=0, right=xmax)
        ax_median_bbb.set_xlim(left=-0.001, right=xmax)
        # but colorbars
        cbar_mean_det = plt.colorbar(scat_mean, ax=ax_mean_det)
        cbar_mean_det.set_label(f'std', rotation=270)
        cbar_median_det = plt.colorbar(scat_median, ax=ax_median_det)
        cbar_median_det.set_label(f'std', rotation=270)
        cbar_mean_bbb = plt.colorbar(scat_mean, ax=ax_mean_bbb)
        cbar_mean_bbb.set_label(f'std', rotation=270)
        cbar_median_bbb = plt.colorbar(scat_median, ax=ax_median_bbb)
        cbar_median_bbb.set_label(f'std', rotation=270)

plot_on_figure_comparison(to_plot_bbb, to_plot_det)
fig.tight_layout()
fig.show()

if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path / f'acc_unc_{exp}.png')
    u.save_to_file(fig, save_path / f'acc_unc_{exp}.pkl')
    print(f'saved to f{save_path / f"acc_unc_{exp}.png"}')


# %% Selective classification

show_determinist = False
number_of_tests = 10
# show_determinist = True

if show_determinist:
    save_path = 'rapport/determinist_failure/'
    # unc_names = ['sr', 'pe']
else:
    save_path = f'rapport/bayesian_results/{exp}'
    # unc_names = ['sr', 'vr', 'pe', 'mi']
unc_names = ['sr', 'pe', 'vr', 'mi', 'au', 'eu']
unc_names_det = ['sr', 'pe']
save_path = pathlib.Path(save_path)
save_fig = False

def oracle(x, acc):
    return np.minimum(1, acc/np.maximum(x, 0.0001))

coverage_array = np.arange(1, 1+len(res[number_of_tests]))/len(res[number_of_tests])
model_acc = res[number_of_tests].correct_pred.mean()
det_acc = res_det.correct_pred.mean()

cmaps = ['Reds', 'Blues', 'Greys', 'RdGy', 'Greens', 'Purples']  # size: nb of measures
cmaps_position = [0.4, 0.7]  # size: len(nb of tests to plot)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot([0, 1], [det_acc, det_acc], linestyle='dashed', label='lower_bound', c='b')
ax.plot(coverage_array, oracle(coverage_array, det_acc), linestyle='dashed', label='det_optimum', c='b')
for idx, unc_name in enumerate(unc_names):
    grouped_unc = res[number_of_tests].groupby(unc_name)
    to_plot = (
        grouped_unc
        .sum()
        .assign(n=grouped_unc.apply(lambda df: len(df)))
        .assign(cum_accs= lambda df: df.correct_pred.cumsum() / df.n.cumsum())
        .assign(coverage= lambda df: df.n.cumsum()/df.n.sum())

    ).reset_index()
    # sorted_unc = res[number_of_tests].sort_values(unc_name)
    # sorted_unc.loc[sorted_unc[unc_name] < 0, unc_name] = 0
    #
    # sorted_unc = (
    #     sorted_unc
    #     .assign(cum_accs=lambda df: df.correct_pred.cumsum()/np.arange(1, len(sorted_unc)+1))
    # )

    cmap = cm.get_cmap(cmaps[idx])
    ax.plot(to_plot.coverage, to_plot.cum_accs, label=unc_name, c=cmap(cmaps_position[1]))
for idx, unc_name_det in enumerate(unc_names_det):
    sorted_unc = res_det.sort_values(unc_name_det)
    sorted_unc.loc[sorted_unc[unc_name_det] < 0, unc_name_det] = 0

    sorted_unc = (
        sorted_unc
        .assign(cum_accs=lambda df: df.correct_pred.cumsum()/np.arange(1, len(sorted_unc)+1))
    )

    cmap = cm.get_cmap(cmaps[idx])
    ax.plot(
        coverage_array,
        sorted_unc.cum_accs,
        linestyle='dotted',
        label=unc_name_det+'-det',
        c=cmap(cmaps_position[0]),
    )
ax.set_ylim(bottom=0)
ax.set_xlabel('coverage')
ax.set_ylabel('accuracy')
ax.legend()

ax.set_title(f'Accuracy - Coverage. Exp:{exp}. T={number_of_tests}'
             # f'\n {dict({k:v for k,v in arguments.items() if k not in dont_show})}',
             # wrap=True, va='baseline',
             )
fig.show()
if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'acc_cov_{exp}.png')
    u.save_to_file(fig, save_path/f'acc_cov_{exp}.pkl')
    print(f'Saved to {save_path/f"acc_cov_{exp}.png"}')


# %% Selective classification - per unc measure

show_determinist = False
list_of_number_of_tests = list(res.keys())
list_of_number_of_tests.sort()
# show_determinist = True

if show_determinist:
    save_path = 'rapport/determinist_failure/'
    # unc_names = ['sr', 'pe']
else:
    save_path = f'rapport/bayesian_results/{exp}'
    # unc_names = ['sr', 'vr', 'pe', 'mi']

unc_names = ['sr', 'pe', 'mi', 'vr']
unc_names_det = ['sr', 'pe']
save_path = pathlib.Path(save_path)
save_fig = False

def oracle(x, acc):
    return np.minimum(1, acc/np.maximum(x, 0.0001))

coverage_array = np.arange(1, 1+len(res[number_of_tests]))/len(res[number_of_tests])
model_acc = res[number_of_tests].correct_pred.mean()
det_acc = res_det.correct_pred.mean()

cmaps = ['Purples', 'Blues', 'Greens', 'Greys']  # size: nb of measures
cmaps_position = np.linspace(0.3, 0.8, len(list_of_number_of_tests))  # size: len(nb of tests to plot)

fig, axs = plt.subplots(len(unc_names), 1, figsize=(10, 21))

for idx, unc_name in enumerate(unc_names):
    axs[idx].plot([0, 1], [det_acc, det_acc], linestyle='dashed', label='lower_bound', c='b')
    axs[idx].plot(coverage_array, oracle(coverage_array, det_acc), linestyle='dashed', label='det_optimum', c='b')
    if unc_name in unc_names_det:
        grouped_unc = res_det.groupby(unc_name)
        to_plot = (
            grouped_unc
            .sum()
            .assign(n=grouped_unc.apply(lambda df: len(df)))
            .assign(cum_accs=lambda df: df.correct_pred.cumsum() / df.n.cumsum())
            .assign(coverage=lambda df: df.n.cumsum() / df.n.sum())

        ).reset_index()

        cmap = cm.get_cmap(cmaps[idx])
        axs[idx].plot(to_plot.coverage, to_plot.cum_accs, label='det', c='r')
        axs[idx].plot([0, to_plot.coverage[0]], [to_plot.cum_accs[0], to_plot.cum_accs[0]], c='r', linestyle='dotted')
    for nb_test_idx, number_of_tests in enumerate(list_of_number_of_tests):
        grouped_unc = res[number_of_tests].groupby(unc_name)
        to_plot = (
            grouped_unc
            .sum()
            .assign(n=grouped_unc.apply(lambda df: len(df)))
            .assign(cum_accs= lambda df: df.correct_pred.cumsum() / df.n.cumsum())
            .assign(coverage= lambda df: df.n.cumsum()/df.n.sum())

        ).reset_index()

        cmap = cm.get_cmap(cmaps[idx])
        axs[idx].plot(to_plot.coverage, to_plot.cum_accs, label=f'T={number_of_tests}', c=cmap(cmaps_position[nb_test_idx]))
        axs[idx].plot(
            [0, to_plot.coverage[0]], [to_plot.cum_accs[0], to_plot.cum_accs[0]],
            c=cmap(cmaps_position[nb_test_idx]),
            linestyle='dotted',
        )
    axs[idx].set_ylim(bottom=0)
    axs[idx].set_xlabel('coverage')
    axs[idx].set_ylabel('accuracy')
    axs[idx].set_title(unc_name)
    axs[idx].legend()

fig.show()
if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'acc_cov_{exp}.png')
    u.save_to_file(fig, save_path/f'acc_cov_{exp}.pkl')
    print(f'Saved to {save_path/f"acc_cov_{exp}.png"}')

# %% Selective classification - thresholds

show_determinist = False
list_of_number_of_tests = list(res.keys())
list_of_number_of_tests.sort()
# show_determinist = True

if show_determinist:
    save_path = 'rapport/determinist_failure/'
    unc_names = ['sr', 'pe']
else:
    save_path = f'rapport/bayesian_results/{exp}'
    unc_names = ['sr', 'pe', 'vr', 'mi']

# unc_names = ['sr', 'pe', 'vr', 'mi']
unc_names_det = ['sr', 'pe']
save_path = pathlib.Path(save_path)
save_fig = False

coverage_array = np.arange(1, 1+len(res[number_of_tests]))/len(res[number_of_tests])
model_acc = res[number_of_tests].correct_pred.mean()
det_acc = res_det.correct_pred.mean()

cmaps = ['Purples', 'Blues', 'Greens', 'Greys']  # size: nb of measures
cmaps_position = np.linspace(0.3, 0.8, len(list_of_number_of_tests))  # size: len(nb of tests to plot)

fig, axs = plt.subplots(len(unc_names), 1, figsize=(10, 15))

for idx, unc_name in enumerate(unc_names):
    if unc_name in unc_names_det:
        grouped_unc = res_det.groupby(unc_name)
        to_plot = (
            grouped_unc
            .sum()
            .assign(n=grouped_unc.apply(lambda df: len(df)))
            .assign(cum_accs=lambda df: df.correct_pred.cumsum() / df.n.cumsum())
            .assign(coverage=lambda df: df.n.cumsum() / df.n.sum())

        ).reset_index()

        cmap = cm.get_cmap(cmaps[idx])
        axs[idx].plot(to_plot.coverage, to_plot[unc_name], label='det', c='r')
        axs[idx].axvline(to_plot.coverage.min(), linestyle='dashed', c='r')
    if not show_determinist:
        for nb_test_idx, number_of_tests in enumerate(list_of_number_of_tests):
            grouped_unc = res[number_of_tests].groupby(unc_name)
            to_plot = (
                grouped_unc
                .sum()
                .assign(n=grouped_unc.apply(lambda df: len(df)))
                .assign(cum_accs= lambda df: df.correct_pred.cumsum() / df.n.cumsum())
                .assign(coverage= lambda df: df.n.cumsum()/df.n.sum())

            ).reset_index()

            cmap = cm.get_cmap(cmaps[idx])
            axs[idx].plot(to_plot.coverage, to_plot[unc_name], label=number_of_tests, c=cmap(cmaps_position[nb_test_idx]))
            axs[idx].axvline(to_plot.coverage.min(), linestyle='dashed', c=cmap(cmaps_position[nb_test_idx]))

    axs[idx].set_ylim(bottom=0)
    axs[idx].set_xlim(left=0)
    axs[idx].set_xlabel('coverage')
    axs[idx].set_ylabel('uncertainty')
    axs[idx].set_title(unc_name)
    axs[idx].legend()

fig.show()
if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'threshold_cov_{exp}.png')
    u.save_to_file(fig, save_path/f'threshold_cov_{exp}.pkl')
    print(f'Saved to {save_path/f"threshold_cov_{exp}.png"}')


# %% Selective classification - ROC

show_determinist = False
list_of_number_of_tests = list(res.keys())
list_of_number_of_tests.sort()
# show_determinist = True

if show_determinist:
    save_path = 'rapport/determinist_failure/'
    # unc_names = ['sr', 'pe']
else:
    save_path = f'rapport/bayesian_results/{exp}'
    # unc_names = ['sr', 'vr', 'pe', 'mi']

unc_names = ['sr', 'pe', 'vr', 'mi']
unc_names_det = ['sr', 'pe']
save_path = pathlib.Path(save_path)
save_fig = False

def oracle(x, acc):
    return np.minimum(1, acc/np.maximum(x, 0.0001))

coverage_array = np.arange(1, 1+len(res[number_of_tests]))/len(res[number_of_tests])
model_acc = res[number_of_tests].correct_pred.mean()
det_acc = res_det.correct_pred.mean()

cmaps = ['Purples', 'Blues', 'Greens', 'Greys']  # size: nb of measures
cmaps_position = np.linspace(0.3, 0.8, len(list_of_number_of_tests))  # size: len(nb of tests to plot)

fig, axs = plt.subplots(len(unc_names), 1, figsize=(10, 15))

for idx, unc_name in enumerate(unc_names):
    # axs[idx].plot([0, 1], [det_acc, det_acc], linestyle='dashed', label='lower_bound', c='b')
    # axs[idx].plot(coverage_array, oracle(coverage_array, det_acc), linestyle='dashed', label='det_optimum', c='b')
    if unc_name in unc_names_det:

        grouped_unc = res_det.groupby(unc_name)
        # to_plot = (
        #     grouped_unc
        #     .sum()
        #     .assign(n=grouped_unc.apply(lambda df: len(df)))
        #     .assign(cum_accs=lambda df: df.correct_pred.cumsum() / df.n.cumsum())
        #     .assign(coverage=lambda df: df.n.cumsum() / df.n.sum())
        #
        # ).reset_index()

        to_plot = (
            grouped_unc
                .sum()
                .assign(n=grouped_unc.apply(lambda df: len(df)))
                .assign(tp=lambda df: df.correct_pred.cumsum())
                .assign(fp=lambda df: df.n.cumsum() - df.tp)

        ).reset_index()

        cmap = cm.get_cmap(cmaps[idx])
        this_df = res_det.loc[res_det[unc_name] == unc_name]
        positives = this_df.correct_pred.sum()
        negatives = (1-this_df.correct_pred).sum()
        axs[idx].plot(to_plot.fp/negatives, to_plot.tp/positives, label='det', c='r')
    for nb_test_idx, number_of_tests in enumerate(list_of_number_of_tests):
        grouped_unc = res[number_of_tests].groupby(unc_name)
        # to_plot = (
        #     grouped_unc
        #     .sum()
        #     .assign(n=grouped_unc.apply(lambda df: len(df)))
        #     .assign(cum _accs= lambda df: df.correct_pred.cumsum() / df.n.cumsum())
        #     .assign(coverage= lambda df: df.n.cumsum()/df.n.sum())
        #
        # ).reset_index()

        to_plot = (
            grouped_unc
                .sum()
                .assign(n=grouped_unc.apply(lambda df: len(df)))
                .assign(tp=lambda df: df.correct_pred.cumsum())
                .assign(fp=lambda df: df.n.cumsum() - df.tp)

        ).reset_index()

        cmap = cm.get_cmap(cmaps[idx])
        # axs[idx].plot(to_plot.coverage, to_plot[unc_name], label=number_of_tests, c=cmap(cmaps_position[nb_test_idx]))

    axs[idx].set_ylim(bottom=0)
    axs[idx].set_xlabel('coverage')
    axs[idx].set_ylabel('uncertainty')
    axs[idx].set_title(unc_name)
    # axs[idx].legend()

fig.show()
if save_fig:
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path/f'acc_cov_{exp}.png')
    u.save_to_file(fig, save_path/f'acc_cov_{exp}.pkl')
    print(f'Saved to {save_path/f"acc_cov_{exp}.png"}')
# %%

det_acc = res_det.correct_pred.mean()
path_to_results = 'results/raw_results/specific_columns/group323_specific_results.pkl'

unc= 'vr'
param = f'{unc}_unseen/seen'
df_ini = pd.read_pickle(path_to_results)
df = df_ini.groupby(['rho', 'stds_prior']).mean().reset_index()

df['dist'] = (df.eval_accuracy - det_acc).abs()
mlp = df.sort_values('dist').iloc[0]

std_star = mlp.stds_prior
rho_star = mlp.rho
eval_star = mlp.eval_accuracy
dist_star = mlp.dist

xs = df.stds_prior
ys = df.rho
zs = df.dist

fig = plt.figure(figsize=(9,5))
ax3d = fig.add_subplot(111, projection='3d')
ax3d.plot((std_star, std_star), (rho_star, rho_star), (dist_star, 0.8), zorder=0, linestyle='solid', linewidth=5)
ax3d.text(std_star, rho_star, 1.4, f' Acc={round(100*eval_star, 2)}%, Rho={rho_star}, Std={std_star}')
ax3d.plot_trisurf(xs, ys, zs, zorder=-1)
ax3d.scatter(std_star, rho_star, dist_star, zorder=0, linestyle='solid', linewidth=5)
ax3d.set_xlabel('std prior')
ax3d.set_ylabel('rho')
ax3d.set_zlabel('acc')
# ax3d.annotate(f'Max Values: rho {max_acc_params.rho}, std prior {max_acc_params.stds_prior}',
#               xy=(max_acc_params.rho, max_acc_params.stds_prior), xycoords='data',
#               xytext=(1, 1), textcoords='offset points')
ax3d.view_init(40, 230)
fig.suptitle('Accuracy w.r.t. initial variance of posterior and prior')
fig.show()

print(mlp.stds_prior, mlp.rho)

