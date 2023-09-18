import sys
sys.path.append('/home/victoria/NeuroConn')
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
import tqdm

data_path = '/home/victoria/server/data/COST/COST_mri/derivatives/rest'
l_surf = '/home/victoria/server/public/scratchpad/surfaces/fsLR/tpl-fsLR_den-32k_hemi-L_sphere.surf.gii'
r_surf = '/home/victoria/server/public/scratchpad/surfaces/fsLR/tpl-fsLR_den-32k_hemi-R_sphere.surf.gii'
task = 'rest'
output_space = 'fsLR_91k'

dataset = FmriPreppedDataSet(data_path)

for subject in tqdm.tqdm(dataset.subjects):
    print('\n Smoothing subject %s' % subject)
    try:
        dataset.surf_smooth(subject, task, output_space, 4, 4, 'COLUMN', l_surf, r_surf)
    except Exception as e:
        print(e)

print('Smoothing done.')