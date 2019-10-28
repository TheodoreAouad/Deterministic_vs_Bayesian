# Deterministic_vs_Bayesian

This is the implementation of my master thesis "How to secure ap rediction in classification: a comparison between Bayesian and Deterministic Neural Networks".

The code is decomposed into two main modules: `src/` and `scripts/`. The directory `src/` contains the core of the code: the managing of datasets, the definition of models, the uncertainty computation and the training and evaluating tasks. It is sufficient to use our work in practice and for further research. The directory `scripts/` contains the scripts used to generate the figures obtained after having trained our models.

##To train the models:
 - If you have access to polyaxon, edit the yaml `experiments_launcher/run_polyaxon_grid_primary_results.yml` then run the experiment `polyaxon run -u -f experiments_launcher/run_polyaxon_grid_primary_results.yml`.
 - If you do not have polyaxon, run `experiments/primary_results_bayesian.py` with the desired arguments.
 

##Structure 
All our experiments were saved following the structure of `polyaxon_results` directory.
To reproduce our graphs, save the outputs of the experiments to ``polyaxon_results/groups/{number_of_group}/{number_of_exp}``,
then give the `number_of_exp` in arguments inside the `scripts/some_graphs.py` file.
