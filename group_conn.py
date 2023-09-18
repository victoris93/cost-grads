
import os
import numpy as np


data_path = '/home/victoria/server/data/COST/COST_mri/derivatives/rest'
mat_dir = os.path.join(data_path, 'derivatives/connectivity_matrices')

mat_file_paths = [os.path.join(mat_dir, file) for file in os.listdir(mat_dir)]
mean_file_path = os.path.join(mat_dir, "COST_group_conn_matrix.npy")

for i, _ in enumerate(mat_file_paths):
    if not os.path.exists(mean_file_path):
        conn_mat = np.load(mat_file_paths[i])
    else:
        conn_mat = np.load(mean_file_path)
    print(f"Adding matrix {i + 2}")
    next_conn_mat = np.load(mat_file_paths[i+1])
    mean_conn_mat = conn_mat + next_conn_mat
    np.save(mean_file_path, mean_conn_mat)

del conn_mat, next_conn_mat, mean_conn_mat

print(f"Computing mean matrix")
conn_mat = np.load(mean_file_path)
mean_conn_mat = conn_mat / len(mat_file_paths)
print("Group matrix computed, saving...")
np.save(mean_file_path, mean_conn_mat)


