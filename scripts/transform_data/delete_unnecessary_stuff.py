import os
import sys
sys.path.append('/home/muaddib/Sicara/theodore/BayesianFewShotExperiments')
from src.utils import get_file_and_dir_path_in_dir

groups_to_delete = ['/output/sicara/BayesianFewShotExperiments/groups']
files_to_delete = ['TrainingLogs.pkl', 'loss.pkl', 'seen_outputs.pt', 'unseen_outputs.pt',]

for group in groups_to_delete:
    for files_name in files_to_delete:
        files, dirs = get_file_and_dir_path_in_dir(group, files_name)
        print(files)
        i = 0
        print(f'Deleting {files_name} ...')
        for file in files:
            os.remove(file)
            i += 1
        print(f'{i} {files_name} deleted.')
