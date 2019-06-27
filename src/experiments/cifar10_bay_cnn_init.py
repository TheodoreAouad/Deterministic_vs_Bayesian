from math import log, exp
import argparse
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from src.models.bayesian_models import GaussianClassifierCIFAR
from src.trains import train, test
from src.utils import set_and_print_random_seed
from src.get_data import get_cifar10


parser = argparse.ArgumentParser()
parser.add_argument("--rho")
parser.add_argument("--epoch")
args = parser.parse_args()

rho = float(args.rho)
epoch = int(args.epoch)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)

trainloader, testloader = get_cifar10()

seed_random = set_and_print_random_seed()
random_noise = torch.randn(16, 3, 32, 32).to(device)
res = []

seed_model = set_and_print_random_seed()
bay_net = GaussianClassifierCIFAR(rho=rho, dim_input=32, number_of_classes=10, determinist=False)
bay_net.to(device)
criterion = CrossEntropyLoss()
adam_proba = optim.Adam(bay_net.parameters())
losses, accs = train(bay_net, adam_proba, criterion, epoch, trainloader, device=device, verbose=True)
test_acc = test(bay_net, testloader, device)
output_random = torch.Tensor(10,16)
for i in range(10):
    output_random[i] = bay_net(random_noise).argmax(1)

res = dict({
    "number of epochs": epoch,
    "seed_model": seed_model,
    "rho": rho,
    "sigma initial": log(1 + exp(rho)),
    "train accuracy": accs,
    "train loss": losses,
    "test accuracy": test_acc,
    "seed_random": seed_random,
    "random output": output_random
})

torch.save(res, "./output/results.pt")
torch.save(bay_net.state_dict(), "./output/weights.pt")
