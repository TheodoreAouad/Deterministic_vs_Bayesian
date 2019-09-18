"""
Merges multiple results to plot a final CSV with all the results we want.
"""
import pathlib


from src.utils import load_from_file

####### TO CHANGE #########

trainset = 'cifar10'
dz_nb = 100
path_to_dz = f'results/deadzones/{dz_nb}/recomputed/{trainset}/deadzones.pkl'
path_to_auc = f'results/risk_coverage/{trainset}/aucs_eval.pkl'
path_to_acc = f'results/eval_acc/all_eval_accs.pkl'
save_path = f'results/all_results/{trainset}'
###########################

save_path = pathlib.Path(save_path)


def aggregate_df(df, indexs):
    """
    Gives Mean and STD of df grouped by indexes
    Args:
        df (pandas.core.frame.DataFrame):
        indexs (list): list of string of column names on which to group by

    Returns:
        pandas.core.frame.DataFrame: aggregated dataframe

    """
    grped = df.groupby(indexs)
    means = grped.agg('mean').applymap(lambda x: str(round(x, 4)))
    stds = grped.agg(lambda df: 1.96*df.std() / len(df)).applymap(lambda x: str(round(x, 4)))
    df_aggregated = means + '+-' + stds
    return df_aggregated


df_dzs = load_from_file(path_to_dz)
df_aucs = load_from_file(path_to_auc)
df_accs = load_from_file(path_to_acc)

df_dzs_agg = aggregate_df(df_dzs, indexs=[
    'exp_nb',
    'group_nb',
    'number_of_tests',
    'type_of_unseen',
    'unc_name',
]).reset_index()

df_aucs_agg = aggregate_df(df_aucs, indexs=[
    'exp_nb',
    'group_nb',
    'trainset',
    'rho',
    'std_prior',
    'loss_type',
    'number_of_tests',
    'unc_name',
]).reset_index()

df_accs_agg = aggregate_df(df_accs, indexs=[
    'exp_nb',
    'group_nb',
    'trainset',
    'rho',
    'std_prior',
    'loss_type',
    'number_of_tests',
    'split_labels',
]).reset_index()

first_join_col = ['exp_nb', 'group_nb', 'number_of_tests', 'rho', 'std_prior', 'loss_type', 'trainset']

all_results = (
    df_accs_agg
    .merge(df_aucs_agg, on=first_join_col)
    .merge(df_dzs_agg, on=['group_nb', 'exp_nb', 'number_of_tests', 'unc_name'])
)

all_results = all_results.reindex([
    'group_nb',
    'exp_nb',
    'trainset',
    'type_of_unseen',
    'split_labels',
    'loss_type',
    'std_prior',
    'rho',
    'number_of_tests',
    'unc_name',
    'eval_acc',
    'auc',
    'dz_100',
], axis=1)

save_path.mkdir(exist_ok=True, parents=True)

all_results.to_csv(save_path / 'all_results.csv')
all_results.to_pickle(save_path / 'all_results.pkl')
