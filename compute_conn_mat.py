
import sys
import os
sys.path.append('/home/victoria/NeuroConn')
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
import tqdm
import concurrent.futures


data_path = '/home/victoria/server/data/COST/COST_mri/derivatives/rest'
l_surf = '/home/victoria/server/public/scratchpad/surfaces/fsLR/tpl-fsLR_den-32k_hemi-L_sphere.surf.gii'
r_surf = '/home/victoria/server/public/scratchpad/surfaces/fsLR/tpl-fsLR_den-32k_hemi-R_sphere.surf.gii'

matrices_output = os.path.join(data_path, 'derivatives/connectivity_matrices')

dataset = FmriPreppedDataSet(data_path)
subjects = dataset.subjects.tolist()
subjects.remove('82551')
subjects.remove('12731')
subjects.remove('71785')
print(f'Total of {len(subjects)} subjects')

for subject in subjects:
    if os.path.exists(os.path.join(matrices_output, f'z-conn-matrix-sub-{subject}-rest-fsLR_den-91k.npy')):
        subjects.remove(subject)
        print(f'Conn matrix for subject {subject} already exists, remaining subjects: {len(subjects)}')

for subject in tqdm.tqdm(subjects):
    try:
        print(f"Computing conn matrix for subjects {subject}")
        conn_matrix = dataset.get_conn_matrix(subject = subject, task = 'rest', output_space = 'fsLR_91k', bold_tr = 0.801, surf = True, smoothed = True, save = True, save_to = matrices_output)
    except Exception as e:
        print(f'Error with {subject}: {str(e)}')

print("Done.")