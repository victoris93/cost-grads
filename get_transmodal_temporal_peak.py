import nibabel as nib
import numpy as np
import sys
sys.path.append('/home/victoria/NeuroConn')
sys.path.append('/home/victoria/surfdist')
import pandas as pd
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
import tqdm

# for temporal: zones 5 (L) and 2 (R)
# for parietal: zones 2 (L) and 3 (R)

individual = sys.argv[1]
zone = sys.argv[2]
if zone == 'temporal':
    zoneL, zoneR = 5, 2
elif zone == 'parietal':
    zoneL, zoneR = 2, 3

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
    TpeakL = grad2L.copy()
    TpeakR = grad2R.copy()

    temp_zone_indL = np.where(fsLR_labels_L != 2)[0]
    temp_zone_indR = np.where(fsLR_labels_R != 3)[0]
    TpeakL[temp_zone_indL] = np.nan
    TpeakL = np.where(np.isnan(TpeakL) == False)[0]
    TpeakR[temp_zone_indR] = np.nan
    TpeakR = np.where(np.isnan(TpeakR) == False)[0]

    np.save('/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/thresholded/COST_grad2_thresh5_L.npy', TpeakL)
    np.save('/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/thresholded/COST_grad2_thresh5_R.npy', TpeakR)
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
            TpeakL = grad2L.copy()
            TpeakR = grad2R.copy()

            zone_indL = np.where(fsLR_labels_L != zoneL)[0]
            zone_indR = np.where(fsLR_labels_R != zoneR)[0]
            TpeakL[zone_indL] = np.nan
            TpeakL = np.where(np.isnan(TpeakL) == False)[0]
            TpeakR[zone_indR] = np.nan
            TpeakR = np.where(np.isnan(TpeakR) == False)[0]

            np.save(f'/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/thresholded/aligned_grad2_thresh5_L_{zone}_sub-{subject}.npy', TpeakL)
            np.save(f'/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/thresholded/aligned_grad2_thresh5_R_{zone}_sub-{subject}.npy', TpeakR)
        except Exception as e:
            print(f"Error for subject {subject}: {e}")

print("Transmodal peaks saved.")