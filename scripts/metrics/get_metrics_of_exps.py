

import numpy as np
import torch

from scripts.metrics.evaluate_acc_exps import main as main_eval_accs
from scripts.metrics.get_deadzone_of_exps import main as main_dzs
from scripts.selective_classification.get_risk_coverage import main as main_rc
from scripts.metrics.compute_auc import main as main_auc
from scripts.metrics.merge_csv_results import main as main_merge


CPUPATH = 'polyaxon_results/groups'
GPUPATH = '/output/sicara/BayesianFewShotExperiments/groups/'

####### TO CHANGE #########

# exp_nbs = ['14617', '14746', '14681', '14627', '14748', '14689', '14633', '14754', '14695']
exp_nbs = ['14804', '15031', '14866', '14810', '15037', '14872', '14820', '15039', '14880']
nb_of_runs = 2
nb_of_tests = 10

dz_nb = 100

rstars = np.linspace(0.1, 0.5, 50)
# rstars = [0.3, 0.2]
delta = 0.01

trainset = 'cifar10'
path_to_dz = f'results/deadzones/'
path_to_auc = f'results/risk_coverage/'
path_to_acc = f'results/eval_acc/'
save_path = f'results/all_results/'

###########################

if torch.cuda.is_available():
    device = "cuda"
    path_to_exps = GPUPATH
    save_path = './output'
else:
    device = "cpu"
    path_to_exps = CPUPATH
device = torch.device(device)
print(device)

print(f'Computing evaluation with {10} tests...')
main_eval_accs(
    exp_nbs=exp_nbs,
    path_to_exps=path_to_exps,
    path_to_results=path_to_acc,
    nb_of_runs=nb_of_runs,
    nb_of_tests=nb_of_tests,
    device=device,
)
print('Evaluation done.')

print('======================')
print(f'Computing deadzones {dz_nb} ...')
main_dzs(
    exp_nbs=exp_nbs,
    path_to_results=path_to_dz,
    path_to_exps=path_to_exps,
    n=dz_nb,
    nb_of_runs=nb_of_runs,
    number_of_tests=nb_of_tests,
    do_recompute_outputs=True,
    save_csv=True,
    device=device,
)
print('Deadzones computed.')

print('======================')
print('Computing risk-coverages ...')
main_rc(
    exp_nbs=exp_nbs,
    path_to_exps=path_to_exps,
    path_to_results=path_to_auc,
    nb_of_runs=nb_of_runs,
    number_of_tests=nb_of_tests,
    rstars=rstars,
    delta=delta,
    recompute_outputs=True,
    save_csv=True,
    do_save_animation=False,
    device=device,
)
print('Risk-coverage done.')

print('======================')
print('Computing auc...')
main_auc(
    path_to_exps=path_to_exps,
    path_to_results=path_to_auc,
)
print('AUC done.')

print('======================')
print('Merging all results...')
main_merge(
    path_to_acc=path_to_acc+'/all_eval_accs.pkl',
    path_to_auc=path_to_auc+f'/aucs_eval.pkl',
    path_to_dz=path_to_dz+f'/recomputed/deadzones.pkl',
    save_path=save_path,
)
print('======================')
print('All done.')
