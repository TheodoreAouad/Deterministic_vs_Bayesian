from math import log, exp
import argparse

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torchvision.transforms as transforms

from src.models.bayesian_models.gaussian_classifiers import GaussianClassifierMNIST
from src.tasks.trains import train_bayesian
from src.tasks.evals import eval_bayesian
from src.utils import set_and_print_random_seed, save_dict
from src.dataset_manager.get_data import get_mnist, get_omniglot

parser = argparse.ArgumentParser()
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
omniglot_loader = get_omniglot(transform=transform)
trainloader, valloader, evalloader = get_mnist(batch_size=batch_size)


seed_model = set_and_print_random_seed()
bay_net = GaussianClassifierMNIST(rho=rho, stds_prior=stds_prior, dim_input=28, number_of_classes=10)
bay_net.to(device)
criterion = CrossEntropyLoss()
adam_proba = optim.Adam(bay_net.parameters())

(losses, loss_llhs, loss_vps, loss_prs, accs, max_acc, epoch_max_acc,
 batch_idx_max_acc, val_accs, val_uncs, val_dkls) = train_bayesian(bay_net,
                                                                   adam_proba,
                                                                   criterion,
                                                                   epoch,
                                                                   trainloader,
                                                                   valloader,
                                                                   loss_type=loss_type,
                                                                   output_dir_tensorboard='./output',
                                                                   output_dir_results="./output/weights_training",
                                                                   device=device,
                                                                   verbose=True)

print("Evaluation on MNIST ...")
seen_eval_acc, seen_eval_uncertainty, seen_eval_dkls = eval_bayesian(bay_net,
                                                                     evalloader,
                                                                     number_of_tests=number_of_tests,
                                                                     device=device)
print("Finished evaluation on MNIST.")
print("Evaluation on omniglot ...")
_, omniglot_eval_uncertainty, omniglot_eval_dkls = eval_bayesian(bay_net,
                                                                 omniglot_loader,
                                                                 number_of_tests=number_of_tests,
                                                                 device=device)
print("Finished evaluation on omniglot.")

print(f"MNIST: {round(100*seen_eval_acc,2)} %, "
      f"Softmax uncertainty:{seen_eval_uncertainty.mean()}, "
      f"Dkl:{seen_eval_dkls.mean()}")
print(f"Omniglot: Softmax uncertainty:{omniglot_eval_uncertainty.mean()}, "
      f"Dkl:{omniglot_eval_dkls.mean()}")
res = dict({
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
    "val uncertainty": val_uncs,
    "val dkls": val_dkls,
    "eval accuracy": seen_eval_acc,
    "seen uncertainty": seen_eval_uncertainty,
    "seen dkls": seen_eval_dkls,
    "omniglot uncertainty": omniglot_eval_uncertainty,
    "omniglot dkls": omniglot_eval_dkls
})

torch.save(res, "./output/results.pt")
torch.save(bay_net.state_dict(), "./output/final_weights.pt")
