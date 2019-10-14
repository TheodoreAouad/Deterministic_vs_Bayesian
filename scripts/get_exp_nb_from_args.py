import pandas as pd

from scripts.utils import get_args, get_res_args_groupnb, get_trained_model_and_args_and_groupnb

####### TO CHANGE ########
from src.utils import get_file_and_dir_path_in_dir, load_from_file

groups = [322, 323, 324]
rhos = [-10, -5, 0]
stds_prior = [0.01, 0.1, 1]
loss_type = 'criterion'
trainset = 'mnist'
# type_of_unseen_list = ['random', 'unseen_classes', 'unseen_dataset']
# loss_type_list = ['exp', 'criterion', 'uniform']
type_of_unseen_list = ['']
loss_type_list = ['']
batch_size = 32

all_exps = []
for group in groups:
    for rho in rhos:
        for std in stds_prior:

            path_to_results = f'results/raw_results/all_columns/group{group}.pkl'
            ##########################

            df = pd.read_pickle(path_to_results)

            my_exps = []

            my_exps = (df.query(
                f'rho == {rho} '
                # f'type_of_unseen == "{type_of_unseen}" & '
                # f'& loss_type == "{loss_type}" '
                f'& batch_size == {batch_size}'
                f'& stds_prior == {std}'
                f'& trainset == "{trainset}"'
            ).experiment.astype(int))

            all_exps += list(my_exps.values)

print(len(all_exps))
print(len(set(all_exps)))
