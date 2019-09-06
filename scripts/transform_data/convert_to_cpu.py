"""
This file converts the data that was saved with a GPU data-structure to a CPU data-structure.
It must be opened with a cuda gpu available. It will convert all the data to CPU data-structures.
To use it:
-change the 'groups' parameter to decide which experiment data you want to convert
-run the code in the console
"""
import os
from os.path import join

import pandas as pd

from src.utils import convert_df_to_cpu, get_file_and_dir_path_in_dir

groups = ['169', '170', '172', '180', '182', '185', '187', '188', '189']
cur_dir = os.getcwd()

for group in groups:
    all_files, _ = get_file_and_dir_path_in_dir(join(cur_dir, group), 'results')
    for file in all_files:
        exp_nb = file.split('/')[-1]
        df = pd.read_pickle(file)
        convert_df_to_cpu(df)
        df.to_pickle(file)
        print(file, 'converted')

