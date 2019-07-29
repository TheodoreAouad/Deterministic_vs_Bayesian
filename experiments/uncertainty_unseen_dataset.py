import argparse
from math import log, exp

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss

from src.models.bayesian_models.gaussian_classifiers import GaussianClassifierMNIST
from src.tasks.trains import train_bayesian
from src.tasks.evals import eval_bayesian
from src.utils import set_and_print_random_seed, save_dict
from src.dataset_manager.get_data import get_mnist, get_omniglot, get_cifar10

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="which dataset to test the model", choices=["cifar10", "omniglot"], type=str,
                    default="omniglot")
parser.add_argument("--rho", help="variable symbolizing the variance. std = log(1+exp(rho))",
                    type=float, default=-5)
parser.add_argument("--epoch", help="number of times we train the model on the same data",
                    type=int, default=3)
parser.add_argument("--batch_size", help="number of batches to split the data into",
                    type=int, default=32)
parser.add_argument("--number_of_tests", help="number of evaluations to perform for each each image to check for "
                                              "uncertainty", type=int, default=10)
parser.add_argument("--loss_type", help="which loss to use", choices=["bbb", "criterion"], type=str,
                    default="bbb")
parser.add_argument("--std_prior", help="the standard deviation of the prior", type=float, default=1)
args = parser.parse_args()

save_dict(vars(args), './output/arguments.pkl')

dataset = args.dataset
rho = args.rho
epoch = args.epoch
batch_size = args.batch_size
number_of_tests = args.number_of_tests
loss_type = args.loss_type
std_prior = args.std_prior

stds_prior = (std_prior, std_prior)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)

transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
    ])
if dataset == "omniglot":
    unseen_loader = get_omniglot(transform=transform, batch_size=batch_size, download=False)
elif dataset == "cifar10":
    transform = transforms.Compose([
        transforms.Grayscale(),
        transform
    ])
    _, unseen_loader = get_cifar10(transform=transform, batch_size=batch_size)


trainloader, valloader, evalloader = get_mnist(batch_size=batch_size)


seed_model = set_and_print_random_seed()
bay_net = GaussianClassifierMNIST(rho=rho, stds_prior=stds_prior, dim_input=28, number_of_classes=10)
bay_net.to(device)
criterion = CrossEntropyLoss()
adam_proba = optim.Adam(bay_net.parameters())

(losses, loss_llhs, loss_vps, loss_prs, accs, max_acc, epoch_max_acc,
 batch_idx_max_acc, val_accs, val_vrs, val_pes, val_mis) = train_bayesian(
    bay_net,
    adam_proba,
    criterion,
    epoch,
    trainloader,
    valloader,
    loss_type=loss_type,
    output_dir_tensorboard='./output',
    output_dir_results="./output/weights_training",
    device=device,
    verbose=True
)

print("Evaluation on MNIST ...")
seen_eval_acc, seen_eval_vrs, seen_eval_pes, seen_eval_mis = eval_bayesian(
    bay_net,
    evalloader,
    number_of_tests=number_of_tests,
    device=device
)

print("Finished evaluation on MNIST.")
print(f"Evavuation on {dataset} ...")
_, unseen_eval_vrs, unseen_eval_pes, unseen_eval_mis = eval_bayesian(
    bay_net,
    unseen_loader,
    number_of_tests=number_of_tests,
    device=device
)

print("Finished evaluation on ", dataset)

print(f"MNIST: {round(100*seen_eval_acc,2)} %, "
      f"Variation-ratios:{seen_eval_vrs.mean()}, "
      f"Predictive Entropy:{seen_eval_pes.mean()}, "
      f"Mutual information:{seen_eval_mis.mean()}")
print(f"{dataset}: Variation-ratios:{unseen_eval_vrs.mean()}, "
      f"Predictive Entropy:{unseen_eval_pes.mean()}, ",
      f"Mutual Information:{unseen_eval_mis.mean()}")
res = dict({
    "dataset": dataset,
    "number of epochs": epoch,
    "batch_size": batch_size,
    "number of tests": number_of_tests,
    "seed_model": seed_model,
    "stds_prior": std_prior,
    "rho": rho,
    "sigma initial": log(1 + exp(rho)),
    "train accuracy": accs,
    "train max acc": max_acc,
    "train max acc epoch": epoch_max_acc,
    "train loss": losses,
    "train loss llh": loss_llhs,
    "train loss vp": loss_vps,
    "train loss pr": loss_prs,
    "val accuracy": val_accs,
    "val vrs": val_vrs,
    "val pes": val_pes,
    "val mis": val_mis,
    "eval accuracy": seen_eval_acc,
    "seen vrs": seen_eval_vrs,
    "seen pes": seen_eval_pes,
    "seen mis": seen_eval_mis,
    "unseen vrs": unseen_eval_vrs,
    "unseen pes": unseen_eval_pes,
    "unseen mis": unseen_eval_mis
})

torch.save(res, "./output/results.pt")
torch.save(bay_net.state_dict(), "./output/final_weights.pt")
