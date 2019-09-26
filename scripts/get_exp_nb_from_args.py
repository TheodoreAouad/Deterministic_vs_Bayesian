import pandas as pd

from scripts.utils import get_args, get_res_args_groupnb, get_trained_model_and_args_and_groupnb

####### TO CHANGE ########
from src.utils import get_file_and_dir_path_in_dir, load_from_file

group = 226
rho = -6

type_of_unseen_list = ['random', 'unseen_classes', 'unseen_dataset']
loss_type_list = ['exp', 'criterion', 'uniform']
batch_size = 32
stds_prior = 0.1

path_to_results = f'results/raw_results/all_columns/group{group}.pkl'
##########################

df = pd.read_pickle(path_to_results)

my_exps = []
for type_of_unseen in type_of_unseen_list:
    for loss_type in loss_type_list:
        my_exps.append(df.query(
            f'rho == {rho} & '
            f'type_of_unseen == "{type_of_unseen}" & '
            f'loss_type == "{loss_type}" & '
            f'batch_size == {batch_size} & '
            f'stds_prior == {stds_prior}'
        ).experiment.iloc[2])

print(my_exps)
