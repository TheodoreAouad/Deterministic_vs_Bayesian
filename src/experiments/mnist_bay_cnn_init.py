from math import log, exp
import argparse
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from src.models.bayesian_models import GaussianClassifierMNIST
from src.trains import train, test, test_bayesian, train_bayesian
from src.utils import set_and_print_random_seed, aggregate_data
from src.get_data import get_mnist


parser = argparse.ArgumentParser()
parser.add_argument("--rho")
parser.add_argument("--epoch")
parser.add_argument("--number_of_tests")
parser.add_argument("--loss_type")
parser.add_argument("--mu_prior")
parser.add_argument("--std_prior")
args = parser.parse_args()

rho = float(args.rho)
epoch = int(args.epoch)
number_of_tests = int(args.number_of_tests)
loss_type = args.loss_type
mu_prior = float(args.mu_prior)
std_prior = float(args.std_prior)


mus_prior = (mu_prior, mu_prior)
stds_prior = (std_prior, std_prior)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)

trainloader, testloader = get_mnist()

seed_model = set_and_print_random_seed()
bay_net = GaussianClassifierMNIST(rho=rho, mus_prior=mus_prior, stds_prior=stds_prior, dim_input=28, number_of_classes=10)
bay_net.to(device)
criterion = CrossEntropyLoss()
adam_proba = optim.Adam(bay_net.parameters())

losses, loss_llhs, loss_vps, loss_prs, accs = train_bayesian(bay_net, adam_proba, criterion,
                                                             epoch, trainloader, loss_type=loss_type,
                                                             output_dir_tensorboard='./output',
                                                             device=device, verbose=True)
test_acc, test_uncertainty, test_dkls = test_bayesian(bay_net, testloader,
                                                      number_of_tests=number_of_tests, device=device)


seed_random = set_and_print_random_seed()
random_noise = torch.randn(16,1,28,28).to(device)
output_random = torch.Tensor(number_of_tests, 16, 10)
for test_idx in range(number_of_tests):
    output_random[test_idx] = bay_net(random_noise).detach()
_, random_uncertainty, random_dkl = aggregate_data(output_random)

res = dict({
    "number of epochs": epoch,
    "number of tests": number_of_tests,
    "seed_model": seed_model,
    "mu_prior": 0,
    "stds_prior": 1,
    "rho": rho,
    "sigma initial": log(1 + exp(rho)),
    "train accuracy": accs,
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
torch.save(bay_net.state_dict(), "./output/weights.pt")
