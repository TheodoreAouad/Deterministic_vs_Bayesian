import matplotlib.pyplot as plt

from src.utils import load_from_file

path_to_accs = 'polyaxon_results/groups/249/19888/accs_and_uncs.pkl'
path_to_pvalues = 'polyaxon_results/groups/249/19888/pvalues.pkl'
nb_bins = 10

df = load_from_file(path_to_accs)

figure = plt.figure()
ax1 = figure.add_subplot(111)
ax1.hist(df.accs1, label='dirac', bins=nb_bins, density=True)
ax1.hist(df.accs2, label='ce', bins=nb_bins, density=True)
ax1.legend()
ax1.set_xlabel('accuracy')
ax1.set_ylabel('density')
figure.show()
figure.savefig('results/dirac_ce.png')
