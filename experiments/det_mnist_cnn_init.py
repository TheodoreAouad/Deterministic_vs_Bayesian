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
from src.tasks.trains import train_bayesian_modular, uniform
from src.tasks.evals import eval_bayesian, eval_random
from src.uncertainty_measures import get_all_uncertainty_measures, get_all_uncertainty_measures_not_bayesian
from src.utils import set_and_print_random_seed, save_to_file, convert_df_to_cpu
from src.dataset_manager.get_data import get_mnist


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', help='number of times we train the model on the same data',
                    type=int, default=10)
parser.add_argument('--batch_size', help='number of batches to split the data into',
                    type=int, default=128)
args = parser.parse_args()

save_to_file(vars(args), './output/arguments.pkl')

epoch = args.epoch
batch_size = args.batch_size

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

trainloader, valloader, evalloader = get_mnist(train_labels=range(10), eval_labels=range(10), batch_size=batch_size)

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

eval_acc, all_outputs_eval = eval_bayesian(det_net, evalloader, number_of_tests=1, device=device)
eval_unc_soft, eval_predictive_entropy = get_all_uncertainty_measures_not_bayesian(all_outputs_eval)

output_random, seed = eval_random(det_net, batch_size=32, img_channels=1, img_dim=28, number_of_tests=1, device=device)

random_unc_soft, random_predictive_entropy = get_all_uncertainty_measures_not_bayesian(output_random)


print(f'Eval acc: {round(100*eval_acc,2)} %, '
      f'Uncertainty Softmax:{eval_unc_soft.mean()}, '
      f'Predictive Entropy:{eval_predictive_entropy.mean()}, '
      )
print(f'Random: '
      f'Uncertainty Softmax:{random_unc_soft.mean()}, '
      f'Predictive Entropy:{random_predictive_entropy.mean()}, '
      )
res = pd.DataFrame.from_dict({
    'number of epochs': [epoch],
    'batch_size': [batch_size],
    'seed_model': [seed_model],
    'train accuracy': [observables.logs['train_accuracy_on_epoch']],
    'train max acc': [observables.max_train_accuracy_on_epoch],
    'train max acc epoch': [observables.epoch_with_max_train_accuracy],
    'train loss': [loss.logs.get('total_loss', -1)],
    'val accuracy': [observables.logs['val_accuracy']],
    'val predictive entropy': [observables.logs['val_uncertainty_pe']],
    'eval accuracy': [eval_acc],
    'seen uncertainty softmax': [eval_unc_soft],
    'seen uncertainty predictive entropy': [eval_predictive_entropy],
    'random uncertainty softmax': [random_unc_soft],
    'random uncertainty predictive entropy': [random_predictive_entropy],
})

convert_df_to_cpu(res)

save_to_file(loss, './output/loss.pkl')
save_to_file(observables, './output/TrainingLogs.pkl')
torch.save(output_random, './output/random_outputs.pt')
torch.save(all_outputs_eval, './output/softmax_outputs.pt')
res.to_pickle('./output/results.pkl')
torch.save(det_net.state_dict(), './output/final_weights.pt')
