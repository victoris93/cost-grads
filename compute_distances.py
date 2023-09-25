import numpy as np
import nibabel as nib
import os
import sys
sys.path.append('/home/victoria/NeuroConn')
sys.path.append('/home/victoria/surfdist')
import pandas as pd
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
from surfdist import analysis
import tqdm
import csv

TpeakTempL = np.load('/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/COST_grad2_thresh10_L.npy')
TpeakTempR = np.load('/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/COST_grad2_thresh10_R.npy')
data_path = '/home/victoria/server/data/COST/COST_mri/derivatives/rest'
subjects = FmriPreppedDataSet(data_path).subjects.tolist()

fsLR_labels_L = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.L.label.gii').darrays[0].data
fsLR_labels_R = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.R.label.gii').darrays[0].data
cortex_L = np.where(fsLR_labels_L != 0)[0]
cortex_R = np.where(fsLR_labels_R != 0)[0]

for subject in tqdm.tqdm(subjects):
    try:
        surfL = nib.load(f"/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/sub-{subject}/anat/sub-{subject}_hemi-L_midthickness.32k_fs_LR.surf.gii")
        surfR = nib.load(f"/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/sub-{subject}/anat/sub-{subject}_hemi-R_midthickness.32k_fs_LR.surf.gii")

        nodesL = surfL.agg_data('NIFTI_INTENT_POINTSET')
        trianglesL = surfL.agg_data('NIFTI_INTENT_TRIANGLE')
        surfL = (nodesL, trianglesL)

        nodesR = surfR.agg_data('NIFTI_INTENT_POINTSET')
        trianglesR = surfR.agg_data('NIFTI_INTENT_TRIANGLE')
        surfR = (nodesR, trianglesR)

        parc_sub_L = nib.load("/home/victoria/server/data/COST/COST_mri/derivatives/freesurfer/sub-51902/label/lh.aparc.a2009s.annot.32k_fs_LR.label.gii").darrays[0].data
        parc_sub_R = nib.load("/home/victoria/server/data/COST/COST_mri/derivatives/freesurfer/sub-51902/label/rh.aparc.a2009s.annot.32k_fs_LR.label.gii").darrays[0].data

        a1L = np.where(parc_sub_L == 33)[0]
        a1R = np.where(parc_sub_R == 33)[0]
        v1L = np.where((parc_sub_L == 43) | (parc_sub_L == 45))[0]
        v1R = np.where((parc_sub_R == 43) | (parc_sub_R == 45))[0]

        dist_Tpeak_a1_L = analysis.calc_roi_dist(surfL, cortex_L, TpeakTempL, a1L, dist_type='min')
        dist_Tpeak_a1_R = analysis.calc_roi_dist(surfR, cortex_R, TpeakTempR, a1R, dist_type='min')

        dist_Tpeak_v1_L = analysis.calc_roi_dist(surfL, cortex_L, TpeakTempL, v1L, dist_type='min')
        dist_Tpeak_v1_R = analysis.calc_roi_dist(surfR, cortex_R, TpeakTempR, v1R, dist_type='min')

        if not os.path.exists(f'./distances_task.csv'):
            df = pd.DataFrame(columns=['participant_id', 'Dist to A1, LH', 'Dist to A1, RH', 'Dist to V1, LH', 'Dist to V1, RH'])
            df.to_csv('./distances_task.csv', index=False)

        print(f"Writing results for subject {subject}")
        with open('distances_task.csv', mode='a', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)

            # Write a new row to the CSV file
            writer.writerow([subject, dist_Tpeak_a1_L, dist_Tpeak_a1_R, dist_Tpeak_v1_L, dist_Tpeak_v1_R])
    except Exception as e:
        print(f"Error for subject {subject}: {e}")

print("Distances computed.")

    



