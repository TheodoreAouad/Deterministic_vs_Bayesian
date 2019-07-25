from math import log, exp
import argparse
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from src.models.bayesian_models.gaussian_classifiers import GaussianClassifierMNIST
from src.tasks.trains import train_bayesian
from src.tasks.evals import eval_bayesian, eval_random
from src.utils import set_and_print_random_seed, save_dict
from src.dataset_manager.get_data import get_mnist


parser = argparse.ArgumentParser()
parser.add_argument("--rho")
parser.add_argument("--epoch")
parser.add_argument("--batch_size")
parser.add_argument("--number_of_tests")
parser.add_argument("--loss_type")
parser.add_argument("--mu_prior")
parser.add_argument("--std_prior")
args = parser.parse_args()
save_dict(vars(args), './output/arguments.pkl')

rho = float(args.rho)
epoch = int(args.epoch)
batch_size = int(args.batch_size)
number_of_tests = int(args.number_of_tests)
loss_type = args.loss_type
mu_prior = float(args.mu_prior)
std_prior = float(args.std_prior)

if mu_prior is None:
    mu_prior = 0

if std_prior is None:
    std_prior = 1

mus_prior = (mu_prior, mu_prior)
stds_prior = (std_prior, std_prior)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)

trainloader, testloader = get_mnist(batch_size=batch_size)

seed_model = set_and_print_random_seed()
bay_net = GaussianClassifierMNIST(rho=rho, mus_prior=mus_prior, stds_prior=stds_prior, dim_input=28, number_of_classes=10)
bay_net.to(device)
criterion = CrossEntropyLoss()
adam_proba = optim.Adam(bay_net.parameters())

losses, loss_llhs, loss_vps, loss_prs, accs, max_acc, epoch_max_acc, batch_idx_max_acc = train_bayesian(bay_net,
                                                                                                        adam_proba, criterion,
                                                                                                        epoch, trainloader, loss_type=loss_type,
                                                                                                        output_dir_tensorboard='./output',
                                                                                                        output_dir_results="./output/weights_training",
                                                                                                        device=device, verbose=True)
test_acc, test_uncertainty, test_dkls = eval_bayesian(bay_net, testloader, number_of_tests=number_of_tests,
                                                      device=device)
random_uncertainty, random_dkl, seed_random = eval_random(bay_net, batch_size, 1, 28, number_of_tests,
                                                          number_of_classes=10, device=device)

res = dict({
    "number of epochs": epoch,
    "batch_size": batch_size,
    "number of tests": number_of_tests,
    "seed_model": seed_model,
    "mu_prior": mu_prior,
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
    "test accuracy": test_acc,
    "test uncertainty": test_uncertainty,
    "test dkls": test_dkls,
    "seed_random": seed_random,
    "random uncertainty": random_uncertainty,
    "random dkls": random_dkl
})

torch.save(res, "./output/results.pt")
torch.save(bay_net.state_dict(), "./output/final_weights.pt")
