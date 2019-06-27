pull-output:
	rsync -ave ssh muaddib:/output/sicara/BayesianFewShotExperiments/ ./polyaxon_results/

write-results-csv:
	python src/write_csv.py --polyaxon_results_path polyaxon_results --polyaxon_type groups --group_nb 65 --which_file results

test:
	ENVIRONMENT=test python -m pytest -vv --cov-report term-missing --no-cov-on-fail --cov=src/

lint:
	pylint src/ --ignore=src/parsers/tests

doc-style:
	pydocstyle --config=./setup.cfg src

pull-data:
	rsync -ave ssh muaddib:/data/leto-yolo/ ./

s3-sync:
	./bash_scripts/sync_s3.sh
