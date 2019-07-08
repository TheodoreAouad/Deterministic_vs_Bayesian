pull-output-group:
	rsync -ave ssh muaddib:/output/sicara/BayesianFewShotExperiments/groups/$(GROUP_NB) ./polyaxon_results/groups/  --exclude 'weights*'

write-results-csv:
	python scripts/write_csv.py --polyaxon_results_path polyaxon_results --polyaxon_type groups --group_nb $(GROUP_NB) --which_file results

pull-and-write-csv:
	make pull-output
	make write-results-csv $(GROUP_NB)
