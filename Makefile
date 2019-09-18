pull-output-group:
	rsync -ave ssh muaddib:/output/sicara/BayesianFewShotExperiments/groups/$(GROUP_NB) ./polyaxon_results/groups/  --filter="merge makefile_specifications/filter_sync.txt"

pull-output-all:
	rsync -ave ssh muaddib:/output/sicara/BayesianFewShotExperiments/ ./polyaxon_results/  --exclude 'weights*'

pull-deadzone:
	rsync -ave ssh muaddib:/home/muaddib/Sicara/theodore/BayesianFewShotExperiments/results/deadzones ./results/

write-all-results-csv:
	python scripts/write_csv/write_all_results_csv.py --polyaxon_results_path polyaxon_results --polyaxon_type groups --group_nb $(GROUP_NB)

write-specific-results-csv-random-test:
	python scripts/write_csv/write_only_some_columns_in_csv.py --which_parameters 'makefile_specifications/parameters.txt' --which_values 'makefile_specifications/values_for_random_testset.txt' --exp_nb $(GROUP_NB)

write-specific-results-csv-unseen-test:
	python scripts/write_csv/write_only_some_columns_in_csv.py --which_parameters 'makefile_specifications/parameters.txt' --which_values 'makefile_specifications/values_for_unseen_testset.txt' --exp_nb $(GROUP_NB)

write-specific-results-csv:
	python scripts/write_csv/write_only_some_columns_in_csv.py --which_parameters 'makefile_specifications/parameters.txt' --which_values 'makefile_specifications/values.txt' --exp_nb $(GROUP_NB)

test:
	python -m pytest src/

run-polyaxon:
	polyaxon run -u -f experiments_launcher/run_polyaxon_grid.yml;
	polyaxon run -u -f experiments_launcher/run_polyaxon_grid_unseen_classes.yml;
	polyaxon run -u -f experiments_launcher/run_polyaxon_grid_unseen_dataset.yml;


