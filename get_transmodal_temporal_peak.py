import nibabel as nib
import numpy as np
import sys
sys.path.append('/home/victoria/NeuroConn')
sys.path.append('/home/victoria/surfdist')
import pandas as pd
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
import tqdm

individual = sys.argv[1]

if individual == 'ind':
    individual = True
else:
    individual = False

fsLR_labels_L = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.L.label.gii').darrays[0].data
fsLR_labels_R = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.R.label.gii').darrays[0].data

if not individual:
    grads = np.load('/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/COST_group_gradients.npy').T
    grad2L = grads[1][:32492]
    grad2L[np.where(grad2L > np.percentile(grad2L, 5))[0]] = np.nan
    grad2R = grads[1][32492:]
    grad2R[np.where(grad2R > np.percentile(grad2R, 5))[0]] = np.nan
    TpeakTempL = grad2L.copy()
    TpeakTempR = grad2R.copy()

    temp_zone_indL = np.where(fsLR_labels_L != 5)[0]
    temp_zone_indR = np.where(fsLR_labels_R != 2)[0]
    TpeakTempL[temp_zone_indL] = np.nan
    TpeakTempL = np.where(np.isnan(TpeakTempL) == False)[0]
    TpeakTempR[temp_zone_indR] = np.nan
    TpeakTempR = np.where(np.isnan(TpeakTempR) == False)[0]

    np.save('/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/thresholded/COST_grad2_thresh5_L.npy', TpeakTempL)
    np.save('/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/thresholded/COST_grad2_thresh5_R.npy', TpeakTempR)
else:
    data_path = '/home/victoria/server/data/COST/COST_mri/derivatives/rest'
    subjects = FmriPreppedDataSet(data_path).subjects.tolist()

    for subject in tqdm.tqdm(subjects):
        try:
            grads = np.load(f'/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/aligned-200gradients-sub-{subject}-rest--fsLR_den-91k.npy')[0].T
            grad2L = grads[1][:32492]
            grad2L[np.where(grad2L > np.percentile(grad2L, 5))[0]] = np.nan
            grad2R = grads[1][32492:]
            grad2R[np.where(grad2R > np.percentile(grad2R, 5))[0]] = np.nan
            TpeakTempL = grad2L.copy()
            TpeakTempR = grad2R.copy()

            temp_zone_indL = np.where(fsLR_labels_L != 5)[0]
            temp_zone_indR = np.where(fsLR_labels_R != 2)[0]
            TpeakTempL[temp_zone_indL] = np.nan
            TpeakTempL = np.where(np.isnan(TpeakTempL) == False)[0]
            TpeakTempR[temp_zone_indR] = np.nan
            TpeakTempR = np.where(np.isnan(TpeakTempR) == False)[0]

            np.save(f'/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/thresholded/aligned_grad2_thresh5_L_sub-{subject}.npy', TpeakTempL)
            np.save(f'/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/thresholded/aligned_grad2_thresh5_R_sub-{subject}.npy', TpeakTempR)
        except Exception as e:
            print(f"Error for subject {subject}: {e}")

print("Transmodal peaks saved.")