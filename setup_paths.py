import pandas as pd
import numpy as np
import sys
import os


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)


def change_path(old_path):
    all_backslashes = list(find_all(old_path, '/'))
    col_folder = old_path[all_backslashes[-3]:]
    return PATH_TO_PROJECT + DATA_FOLDER_NAME + col_folder


args = sys.argv
print(args)
assert len(args) == 2, 'Provide only one argument which is the name of the data folder'
DATA_FOLDER_NAME = str(args[1])
PATH_TO_PROJECT = os.path.abspath('.') + '/'
print('Path to the project:\n', PATH_TO_PROJECT)

recs_metadata = pd.read_csv(PATH_TO_PROJECT + '%s/recordings_metadata.csv' % DATA_FOLDER_NAME, )
recs_metadata_new = pd.DataFrame(np.copy(recs_metadata), columns=recs_metadata.columns)
recs_metadata_new['path'] = recs_metadata_new['path'].apply(change_path)
recs_metadata_new.to_csv(PATH_TO_PROJECT + '%s/recordings_metadata.csv' % DATA_FOLDER_NAME, index=False)