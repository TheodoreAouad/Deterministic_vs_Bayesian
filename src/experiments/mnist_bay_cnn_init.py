import argparse
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from src.models.bayesian_models import GaussianClassifier
from src.trains import train, test
from src.utils import set_and_print_random_seed
from src.get_data import get_mnist


parser = argparse.ArgumentParser()
parser.add_argument("--rho")
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)

trainloader, testloader = get_mnist()
rho = float(args.rho)

seed_random = set_and_print_random_seed()
random_noise = torch.randn(16,1,28,28).to(device)
res = []

seed_model = set_and_print_random_seed()
BayNet = GaussianClassifier(rho=rho, dim_input=28, number_of_classes=10, determinist=False)
BayNet.to(device)
criterion = CrossEntropyLoss()
adam_proba = optim.Adam(BayNet.parameters())
losses2, accs2 = train(BayNet, adam_proba, criterion, 10, trainloader, device=device, verbose=True)
test_acc = test(BayNet, testloader, device)
output_random = torch.Tensor(10,16)
for i in range(10):
    output_random[i] = BayNet(random_noise).argmax(1)

res = dict({
    "seed_random": seed_random,
    "seed_model": seed_model,
    "rho": rho,
    "train accuracy": accs2,
    "train loss": losses2,
    "test accuracy": test_acc,
    "random output": output_random
})

torch.save(res, "./output/experience01.pt")
