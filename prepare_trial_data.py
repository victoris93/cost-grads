from utils import *
import pandas as pd

data_path = '/home/victoria/server/data/COST/COST_mri/derivatives/rest'
task_data_path = './task_data'
compile_trial_data(data_path, task_data_path, save_to = './task_data/all_data_raw.csv')
task_data = pd.read_csv('./task_data/all_data_raw.csv')
data = prepare_distance_task_data('./task_data/all_data_raw.csv', 'distances.csv', False)

data['FlashType'] = data['Condition'].map({1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4})
data['NrBeeps'] = data['Condition'].map({1:0, 4:0, 7:0, 10:0, 2:1, 5:1, 8:1, 11:1, 3:2, 6:2, 9:2, 12:2})

data.to_csv('trial_data_distances.csv', index=False)
