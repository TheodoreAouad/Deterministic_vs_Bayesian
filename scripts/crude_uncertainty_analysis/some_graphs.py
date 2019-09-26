# %% Imports

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


#%% Plot output density

