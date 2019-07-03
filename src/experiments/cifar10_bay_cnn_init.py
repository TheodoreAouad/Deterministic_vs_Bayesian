from math import log, exp
import argparse
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from src.models.bayesian_models import GaussianClassifierCIFAR
from src.trains import train, test, test_bayesian
from src.utils import set_and_print_random_seed, aggregate_data
from src.get_data import get_mnist, get_cifar10

parser = argparse.ArgumentParser()
parser.add_argument("--rho")
parser.add_argument("--epoch")
parser.add_argument("--number_of_tests")
args = parser.parse_args()

rho = float(args.rho)
epoch = int(args.epoch)
number_of_tests = int(args.number_of_tests)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)

trainloader, testloader = get_cifar10()


seed_model = set_and_print_random_seed()
bay_net = GaussianClassifierCIFAR(rho=rho, dim_input=32, number_of_classes=10, determinist=False)
bay_net.to(device)
criterion = CrossEntropyLoss()
adam_proba = optim.Adam(bay_net.parameters())
losses2, accs2 = train(bay_net, adam_proba, criterion, epoch, trainloader, device=device, verbose=True)


test_acc, test_uncertainty, test_dkls = test_bayesian(bay_net, testloader, number_of_tests=number_of_tests, device=device)


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
    "rho": rho,
    "sigma initial": log(1 + exp(rho)),
    "train accuracy": accs2,
    "train loss": losses2,
    "test accuracy": test_acc,
    "test uncertainty": test_uncertainty,
    "test dkls": test_dkls,
    "seed_random": seed_random,
    "random uncertainty": random_uncertainty,
    "random dkls": random_dkl
})

torch.save(res, "./output/results.pt")
torch.save(bay_net.state_dict(), "./output/weights.pt")
