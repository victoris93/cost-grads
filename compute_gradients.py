import sys
import os
sys.path.append('/home/victoria/NeuroConn')
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
from NeuroConn.gradient.gradient import get_gradients
import tqdm

data_path = f'/home/victoria/server/data/COST/COST_mri/derivatives/rest'
mat_path = os.path.join(data_path, "derivatives/connectivity_matrices")
grad_output = os.path.join(data_path, "derivatives/gradients")
align_to = os.path.join(grad_output, "COST_group_gradients.npy")

dataset = FmriPreppedDataSet(data_path)
existing_files = os.listdir(grad_output)

for subject in tqdm.tqdm(dataset.subjects):
    subject_file = [i for i in existing_files if subject in i]
    if len(subject_file) != 0:
        print(f"Gradients for subject {subject} already exist.")
    else:
        try:
            print(f"Computing gradients for subject {subject}...")
            get_gradients(dataset, subject = subject, task = "rest", n_components = 200, save = True, output_space = "fsLR_91k", align_to = align_to, mat_path = mat_path, save_to = grad_output)
            print(f"Gradients computed for subject {subject}.")
        except Exception as error:
            print(error)
            
print("Gradients computed for all subjects.")
