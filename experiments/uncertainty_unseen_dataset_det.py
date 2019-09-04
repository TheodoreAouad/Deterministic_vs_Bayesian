import argparse
from math import log, exp

import pandas as pd
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss

from src.loggers.losses.base_loss import BaseLoss
from src.loggers.losses.bbb_loss import BBBLoss
from src.loggers.observables import AccuracyAndUncertainty
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.trains import train_bayesian_modular, uniform
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_all_uncertainty_measures, get_all_uncertainty_measures_not_bayesian
from src.utils import set_and_print_random_seed, save_to_file
from src.dataset_manager.get_data import get_mnist, get_omniglot, get_cifar10

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="which dataset to test the model", choices=["cifar10", "omniglot"], type=str,
                    default="omniglot")
parser.add_argument("--epoch", help="number of times we train the model on the same data",
                    type=int, default=3)
parser.add_argument("--batch_size", help="number of batches to split the data into",
                    type=int, default=32)
args = parser.parse_args()

save_to_file(vars(args), './output/arguments.pkl')

dataset = args.dataset.lower()
epoch = args.epoch
batch_size = args.batch_size

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
else:
    raise TypeError('Dataset not yet implemented')

trainloader, valloader, evalloader = get_mnist(batch_size=batch_size)


seed_model = set_and_print_random_seed()
det_net = GaussianClassifier(rho='determinist', stds_prior=0, dim_input=28, number_of_classes=10)
det_net.to(device)
criterion = CrossEntropyLoss()
loss = BaseLoss(criterion)

optimizer = optim.Adam(det_net.parameters())
observables = AccuracyAndUncertainty()
train_bayesian_modular(
    det_net,
    optimizer,
    loss,
    observables,
    number_of_tests=1,
    number_of_epochs=epoch,
    trainloader=trainloader,
    valloader=valloader,
    output_dir_tensorboard='./output',
    device=device,
    verbose=True,
)

print("Evaluation on MNIST ...")
seen_eval_acc, all_outputs_eval_seen = eval_bayesian(det_net, evalloader, number_of_tests=1, device=device)
seen_eval_uncertainty_softmax, seen_eval_pes = get_all_uncertainty_measures_not_bayesian(all_outputs_eval_seen)
print("Finished evaluation on MNIST.")

print(f"Evavuation on {dataset} ...")
_, all_outputs_eval_unseen = eval_bayesian(det_net, unseen_loader, number_of_tests=1, device=device)
unseen_eval_uncertainty_softmax, unseen_eval_pes = get_all_uncertainty_measures_not_bayesian(all_outputs_eval_unseen)
print("Finished evaluation on ", dataset)

print(f"MNIST: {round(100*seen_eval_acc,2)} %, "
      f"Uncertainty Softmax:{seen_eval_uncertainty_softmax.mean()}, "
      f"Predictive Entropy:{seen_eval_pes.mean()}, "
      )
print(f"{dataset}: Uncertainty Softmax:{unseen_eval_uncertainty_softmax.mean()}, "
      f"Predictive Entropy:{unseen_eval_pes.mean()}, ",
      )
res = pd.DataFrame.from_dict({
    'dataset': [dataset],
    'number of epochs': [epoch],
    'batch_size': [batch_size],
    'seed_model': [seed_model],
    'train accuracy': [observables.logs['train_accuracy_on_epoch']],
    'train max acc': [observables.max_train_accuracy_on_epoch],
    'train max acc epoch': [observables.epoch_with_max_train_accuracy],
    'train loss': [loss.logs.get('total_loss', -1)],
    'val accuracy': [observables.logs['val_accuracy']],
    "eval accuracy": [seen_eval_acc],
    "seen uncertainty softmax": [seen_eval_uncertainty_softmax],
    "seen uncertainty pes": [seen_eval_pes],
    "unseen uncertainty softmax": [unseen_eval_uncertainty_softmax],
    "unseen uncertainty pes": [unseen_eval_pes],
})


save_to_file(loss, './output/loss.pkl')
save_to_file(observables, './output/TrainingLogs.pkl')
torch.save(all_outputs_eval_unseen, './output/softmax_outputs_eval_unseen.pt')
torch.save(all_outputs_eval_seen, './output/softmax_outputs_eval_seen.pt')
# torch.save(res, "./output/results.pt")
res.to_pickle('./output/results.pkl')
torch.save(det_net.state_dict(), "./output/final_weights.pt")
