"""
Merges multiple results to plot a final CSV with all the results we want.
"""
import pathlib

from src.utils import load_from_file

####### TO CHANGE #########

path_to_dz = f'results/deadzones/'
path_to_auc = f'results/risk_coverage/'
path_to_acc = f'results/eval_acc/'
save_path = f'results/all_results/'
path_to_acc = path_to_acc + '/all_eval_accs.pkl'
path_to_auc = path_to_auc + f'/aucs_eval.pkl'
path_to_dz = path_to_dz + f'/recomputed/deadzones.pkl'
save_path = save_path


###########################


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

    def stringify(x):
        return str(round(x, 4))

    means = grped.agg('mean')
    stds = grped.agg(lambda df: 1.96 * df.std() / len(df))
    df_aggregated_show = means.applymap(stringify) + '+-' + stds.applymap(stringify)
    df_aggregated_data = grped.agg(lambda df: [df.mean(), 1.96 * df.std() / len(df), len(df)])
    return df_aggregated_show.reset_index(), df_aggregated_data.reset_index()


def main(
        path_to_acc=path_to_acc,
        path_to_dz=path_to_dz,
        path_to_auc=path_to_auc,
        save_path=save_path,
):
    save_path = pathlib.Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    df_dzs = load_from_file(path_to_dz)
    df_aucs = load_from_file(path_to_auc)
    df_accs = load_from_file(path_to_acc)

    df_dzs_agg_show, df_dzs_agg_data = aggregate_df(df_dzs, indexs=[
        'exp_nb',
        'group_nb',
        'epoch',
        'number_of_tests',
        'type_of_unseen',
        'unc_name',
    ])
    df_dzs_agg_show.to_csv(save_path / 'dzs_agg_show.csv')
    df_dzs_agg_data.to_csv(save_path / 'dzs_agg_data.csv')

    df_aucs_agg_show, df_aucs_agg_data = aggregate_df(df_aucs, indexs=[
        'exp_nb',
        'group_nb',
        'trainset',
        'rho',
        'std_prior',
        'loss_type',
        'epoch',
        'number_of_tests',
        'unc_name',
    ])
    df_aucs_agg_show.to_csv(save_path / 'aucs_agg_show.csv')
    df_aucs_agg_data.to_csv(save_path / 'aucs_agg_data.csv')

    df_accs_agg_show, df_accs_agg_data = aggregate_df(df_accs, indexs=[
        'exp_nb',
        'group_nb',
        'trainset',
        'rho',
        'std_prior',
        'loss_type',
        'epoch',
        'number_of_tests',
        'split_labels',
    ])
    df_accs_agg_show.to_csv(save_path / 'accs_agg_show.csv')
    df_accs_agg_data.to_csv(save_path / 'accs_agg_data.csv')

    first_join_col = ['exp_nb', 'group_nb', 'number_of_tests', 'rho', 'std_prior', 'epoch', 'loss_type', 'trainset']

    all_results_show = (
        df_accs_agg_show
        .merge(df_aucs_agg_show, on=first_join_col)
        .merge(df_dzs_agg_show, on=['group_nb', 'exp_nb', 'number_of_tests', 'unc_name', 'epoch'])
    )
    all_results_data = (
        df_accs_agg_data
        .merge(df_aucs_agg_data, on=first_join_col)
        .merge(df_dzs_agg_data, on=['group_nb', 'exp_nb', 'number_of_tests', 'unc_name', 'epoch'])
    )

    all_results_show = all_results_show.reindex([
        'group_nb',
        'exp_nb',
        'trainset',
        'type_of_unseen',
        'split_labels',
        'loss_type',
        'std_prior',
        'rho',
        'epoch',
        'number_of_tests',
        'unc_name',
        'eval_acc',
        'auc',
        'dz_100',
    ], axis=1)

    all_results_data = all_results_data.reindex([
        'group_nb',
        'exp_nb',
        'trainset',
        'type_of_unseen',
        'split_labels',
        'loss_type',
        'std_prior',
        'rho',
        'epoch',
        'number_of_tests',
        'unc_name',
        'eval_acc',
        'auc',
        'dz_100',
    ], axis=1)

    all_results_show.to_csv(save_path / 'all_results_show.csv')
    all_results_show.to_pickle(save_path / 'all_results_show.pkl')

    all_results_data.to_csv(save_path / 'all_results_data.csv')
    all_results_data.to_pickle(save_path / 'all_results_data.pkl')


if __name__ == '__main__':
    main()
