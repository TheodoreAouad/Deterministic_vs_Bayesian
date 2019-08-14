from math import log, exp
import argparse

import pandas as pd
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from src.loggers.losses.base_loss import BaseLoss
from src.loggers.losses.bbb_loss import BBBLoss
from src.loggers.observables import AccuracyAndUncertainty
from src.models.bayesian_models.gaussian_classifiers import GaussianClassifier
from src.tasks.trains import uniform, train_bayesian_modular
from src.tasks.evals import eval_bayesian
from src.uncertainty_measures import get_all_uncertainty_measures, get_all_uncertainty_measures_not_bayesian
from src.utils import set_and_print_random_seed, save_to_file
from src.dataset_manager.get_data import get_mnist


parser = argparse.ArgumentParser()
parser.add_argument("--split_labels", help="up to which label the training goes",
                    type=int, default=5)
parser.add_argument("--epoch", help="number of times we train the model on the same data",
                    type=int, default=3)
parser.add_argument("--batch_size", help="number of batches to split the data into",
                    type=int, default=32)
args = parser.parse_args()

save_to_file(vars(args), './output/arguments.pkl')

split_labels = args.split_labels
epoch = args.epoch
batch_size = args.batch_size

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)

trainloader, valloader, evalloader_unseen = get_mnist(
    train_labels=range(split_labels),
    eval_labels=range(split_labels, 10),
    batch_size=batch_size
)

_, _, evalloader_seen = get_mnist(train_labels=(), eval_labels=range(split_labels), batch_size=batch_size)


seed_model = set_and_print_random_seed()
bay_net = GaussianClassifier(rho='determinist', stds_prior=0, dim_input=28, number_of_classes=10)
bay_net.to(device)
criterion = CrossEntropyLoss()
loss = BaseLoss(criterion)

optimizer = optim.Adam(bay_net.parameters())
observables = AccuracyAndUncertainty()
train_bayesian_modular(
    bay_net,
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

_, all_outputs_eval_unseen = eval_bayesian(
    bay_net,
    evalloader_unseen,
    number_of_tests=1,
    device=device
)

unseen_eval_unc_soft, unseen_eval_predictive_entropy = get_all_uncertainty_measures_not_bayesian(all_outputs_eval_unseen)

seen_eval_acc, all_outputs_eval_seen = eval_bayesian(
    bay_net,
    evalloader_seen, number_of_tests=1,
    device=device
)

seen_eval_unc_soft, seen_eval_predictive_entropy = get_all_uncertainty_measures_not_bayesian(all_outputs_eval_seen)


print(f"Seen: {round(100*seen_eval_acc,2)} %, "
      f"Uncertainty Softmax:{seen_eval_unc_soft.mean()}, "
      f"Predictive Entropy:{seen_eval_predictive_entropy.mean()}, "
      )
print(f"Unseen: "
      f"Uncertainty Softmax:{unseen_eval_unc_soft.mean()}, "
      f"Predictive Entropy:{unseen_eval_predictive_entropy.mean()}, "
      )
res = pd.DataFrame.from_dict({
    'number of epochs': [epoch],
    'batch_size': [batch_size],
    'seed_model': [seed_model],
    'train accuracy': [observables.logs['train_accuracy_on_epoch']],
    'train max acc': [observables.max_train_accuracy_on_epoch],
    'train max acc epoch': [observables.epoch_with_max_train_accuracy],
    'train loss': [loss.logs.get('total_loss', -1)],
    'val accuracy': [observables.logs_history['val_accuracy']],
    "eval accuracy": [seen_eval_acc],
    "seen uncertainty uncertainty softmax": [seen_eval_unc_soft],
    "seen uncertainty predictive entropy": [seen_eval_predictive_entropy],
    "unseen uncertainty uncertainty softmax": [unseen_eval_unc_soft],
    "unseen uncertainty predictive entropy": [unseen_eval_predictive_entropy],
})


save_to_file(loss, './output/loss.pkl')
save_to_file(observables, './output/TrainingLogs.pkl')
torch.save(all_outputs_eval_unseen, './output/softmax_outputs_eval_unseen.pt')
torch.save(all_outputs_eval_seen, './output/softmax_outputs_eval_seen.pt')
# torch.save(res, "./output/results.pt")
res.to_pickle('./output/results.pkl')
torch.save(bay_net.state_dict(), "./output/final_weights.pt")
