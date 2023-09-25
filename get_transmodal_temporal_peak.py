import nibabel as nib
import numpy as np

grads = np.load('/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/COST_group_gradients.npy').T
grad2L = grads[1][:32492]
grad2L[np.where(grad2L > np.percentile(grad2L, 5))[0]] = np.nan
grad2R = grads[1][32492:]
grad2R[np.where(grad2R > np.percentile(grad2R, 5))[0]] = np.nan
TpeakTempL = grad2L.copy()
TpeakTempR = grad2R.copy()

fsLR_labels_L = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.L.label.gii').darrays[0].data
fsLR_labels_R = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.R.label.gii').darrays[0].data

temp_zone_indL = np.where(fsLR_labels_L != 5)[0]
temp_zone_indR = np.where(fsLR_labels_R != 2)[0]
TpeakTempL[temp_zone_indL] = np.nan
TpeakTempL = np.where(np.isnan(TpeakTempL) == False)[0]
TpeakTempR[temp_zone_indR] = np.nan
TpeakTempR = np.where(np.isnan(TpeakTempR) == False)[0]

np.save('/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/COST_grad2_thresh10_L.npy', TpeakTempL)
np.save('/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients/COST_grad2_thresh10_R.npy', TpeakTempR)

