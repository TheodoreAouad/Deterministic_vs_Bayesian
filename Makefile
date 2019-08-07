pull-output-group:
	rsync -ave ssh muaddib:/output/sicara/BayesianFewShotExperiments/groups/$(GROUP_NB) ./polyaxon_results/groups/  --exclude 'weights*' --exclude 'softmax_*' --exclude '*.pkl'

write-results-csv:
	python scripts/write_csv.py --polyaxon_results_path polyaxon_results --polyaxon_type groups --group_nb $(GROUP_NB) --which_file results

pull-output-all:
	rsync -ave ssh muaddib:/output/sicara/BayesianFewShotExperiments/ ./polyaxon_results/  --exclude 'weights*'


test:
	python -m pytest src/
