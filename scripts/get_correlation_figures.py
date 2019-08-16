import pathlib

import torch
import torchvision.transforms as transforms

from src.dataset_manager.get_data import get_mnist, get_omniglot, get_cifar10
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.evals import eval_bayesian, eval_random
from src.utils import load_from_file, get_file_and_dir_path_in_dir, compute_figures

###### TO CHANGE ###########
group_nbs = ['189']
exp_nbs = ['3861', '3864']
type_of_unseen = 'unseen_dataset'
nb_of_batches = 1000
size_of_batch = 100
nb_of_random = 5000
save_fig = True
do_eval_mnist = True


############################

def get_seen_outputs_and_labels(bay_net_trained, type_of_unseen, arguments):
    if type_of_unseen == 'unseen_classes':
        _, _, evalloader_seen = get_mnist(train_labels=(), eval_labels=range(arguments['split_labels']), batch_size=128,
                                          split_val=0, shuffle=False)
    else:
        _, _, evalloader_seen = get_mnist(split_val=0, batch_size=128, shuffle=False)
    shuffle_eval = torch.randperm(len(evalloader_seen.dataset))
    evalloader_seen.dataset.data = evalloader_seen.dataset.data[shuffle_eval]
    evalloader_seen.dataset.targets = evalloader_seen.dataset.targets[shuffle_eval]
    true_labels_seen = evalloader_seen.dataset.targets.float()
    print('Evaluation on seen ...')
    _, all_eval_outputs = eval_bayesian(bay_net_trained, evalloader_seen,
                                        number_of_tests=arguments.get('number_of_tests', 1))
    print('Finished evaluation on seen.')
    return all_eval_outputs, true_labels_seen


def get_unseen_outputs(bay_net_trained, type_of_unseen, arguments, nb_of_random=None):
    global evalloader_unseen
    if type_of_unseen == 'random':
        assert nb_of_random is not None, 'Give a number of random samples'
        print('Evaluation on random ...')
        output_random, _ = eval_random(
            bay_net_trained,
            batch_size=nb_of_random,
            img_channels=1,
            img_dim=28,
            number_of_tests=arguments.get('number_of_tests', 1),
            show_progress=True,
        )
        print('Finished evaluation on random.')
        return output_random
    if type_of_unseen == 'unseen_classes':
        _, _, evalloader_unseen = get_mnist(train_labels=(), eval_labels=range(arguments['split_labels'], 10),
                                            batch_size=128, split_val=0)
    elif type_of_unseen == 'unseen_dataset':
        dataset = arguments['dataset']
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor()
        ])
        if dataset == "omniglot":
            evalloader_unseen = get_omniglot(transform=transform, batch_size=128, download=False)
        elif dataset == "cifar10":
            transform = transforms.Compose([
                transforms.Grayscale(),
                transform
            ])
            _, evalloader_unseen = get_cifar10(transform=transform, batch_size=128)
    else:
        raise TypeError('Unrecognized type_of_unseen. Is either "random", "unseen_classes", "unseen_dataset"')
    print('Evaluation on', type_of_unseen, '...')
    _, all_unseen_outputs = eval_bayesian(bay_net_trained, evalloader_unseen,
                                          number_of_tests=arguments.get('number_of_tests', 1))
    print(f'Finished evaluation on {type_of_unseen}.')
    return all_unseen_outputs


for group_nb in group_nbs:
    for exp_nb in exp_nbs:
        exp_path = pathlib.Path('polyaxon_results/groups')
        _, all_dirs = get_file_and_dir_path_in_dir(exp_path / group_nb / exp_nb, 'argumen')
        dirpath = all_dirs[0]
        exp_nb = dirpath.split('/')[-1]
        dirpath = pathlib.Path(dirpath)

        arguments = load_from_file(dirpath / 'arguments.pkl')
        final_weigths = torch.load(dirpath / 'final_weights.pt', map_location='cpu')
        std_prior = arguments.get('std_prior', 0)
        bay_net_trained = GaussianClassifier(
            rho=arguments.get('rho', 'determinist'),
            stds_prior=(std_prior, std_prior),
            number_of_classes=10,
            dim_input=28,
        )
        bay_net_trained.load_state_dict(final_weigths)

        save_path = pathlib.Path(f'results/correlations_figures/{arguments.get("loss_type", "determinist")}')
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f'{group_nb}_{exp_nb}_correlation_uncertainty_error.png'

        all_eval_outputs, true_labels_mnist = get_seen_outputs_and_labels(
            bay_net_trained,
            type_of_unseen,
            arguments
        )

        all_outputs_unseen = get_unseen_outputs(
            bay_net_trained,
            type_of_unseen,
            arguments,
            nb_of_random
        )

        compute_figures(
            arguments=arguments,
            all_outputs_seen=all_eval_outputs,
            true_labels_seen=true_labels_mnist,
            all_outputs_unseen=all_outputs_unseen,
            nb_of_batches=nb_of_batches,
            size_of_batch=size_of_batch,
            type_of_unseen=type_of_unseen,
            save_fig=save_fig,
            save_path=save_path,
        )
