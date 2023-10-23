import sys
import os
import numpy as np
import nibabel as nib
sys.path.append('/home/victoria/NeuroConn')
sys.path.append('/home/victoria/surfdist')
from austin_utils import save_gifti
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
import pandas as pd
from surfdist import analysis
import re


def get_subject_numbers(directory):
    filenames = os.listdir(directory)
    subject_numbers = []
    for filename in filenames:
        match = re.search(r'sub-(\d+)', filename)
        if match:
            subject_numbers.append(match.group(1))
    return subject_numbers

def vrtx_to_gifti(vertex_indices, fill_value, out_name):
    cortex_array = np.zeros((32492))
    cortex_array[:] = np.nan
    cortex_array[vertex_indices] = fill_value
    save_gifti(cortex_array, f"{out_name}")

# def get_transmodal_node(data_path, subject, V1_labels, peak = 'temporal'):

#     TpeakL = np.load(os.path.join(data_path, 'derivatives', 'gradients', 'thresholded', f'aligned_grad2_thresh5_L_{peak}_sub-{subject}.npy'))
#     TpeakR = np.load(os.path.join(data_path, 'derivatives', 'gradients', 'thresholded', f'aligned_grad2_thresh5_R_{peak}_sub-{subject}.npy'))

#     fsLR_labels_L = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.L.label.gii').darrays[0].data
#     fsLR_labels_R = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.R.label.gii').darrays[0].data

#     surfL = nib.load(os.path.join(data_path, 'derivatives', f'sub-{subject}', 'anat', f'sub-{subject}_hemi-L_midthickness.32k_fs_LR.surf.gii'))
#     surfR = nib.load(os.path.join(data_path, 'derivatives', f'sub-{subject}', 'anat', f'sub-{subject}_hemi-R_midthickness.32k_fs_LR.surf.gii'))

#     cortex_L = np.where(fsLR_labels_L != 0)[0]
#     cortex_R = np.where(fsLR_labels_R != 0)[0]

#     nodesL = surfL.agg_data('NIFTI_INTENT_POINTSET')
#     trianglesL = surfL.agg_data('NIFTI_INTENT_TRIANGLE')
#     surfL = (nodesL, trianglesL)


def calc_subj_shrtest_path(data_path, subject, target_labels, peak = 'temporal', target_name = None, save_to = None):

    TpeakL = np.load(os.path.join(data_path, 'derivatives', 'gradients', 'thresholded', f'aligned_grad2_thresh5_L_{peak}_sub-{subject}.npy'))
    TpeakR = np.load(os.path.join(data_path, 'derivatives', 'gradients', 'thresholded', f'aligned_grad2_thresh5_R_{peak}_sub-{subject}.npy'))

    fsLR_labels_L = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.L.label.gii').darrays[0].data
    fsLR_labels_R = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.R.label.gii').darrays[0].data

    surfL = nib.load(os.path.join(data_path, 'derivatives', f'sub-{subject}', 'anat', f'sub-{subject}_hemi-L_midthickness.32k_fs_LR.surf.gii'))
    surfR = nib.load(os.path.join(data_path, 'derivatives', f'sub-{subject}', 'anat', f'sub-{subject}_hemi-R_midthickness.32k_fs_LR.surf.gii'))

    cortex_L = np.where(fsLR_labels_L != 0)[0]
    cortex_R = np.where(fsLR_labels_R != 0)[0]

    nodesL = surfL.agg_data('NIFTI_INTENT_POINTSET')
    trianglesL = surfL.agg_data('NIFTI_INTENT_TRIANGLE')
    surfL = (nodesL, trianglesL)

    nodesR = surfR.agg_data('NIFTI_INTENT_POINTSET')
    trianglesR = surfR.agg_data('NIFTI_INTENT_TRIANGLE')
    surfR = (nodesR, trianglesR)

    parc_sub_L = nib.load(os.path.join(data_path, '..', 'freesurfer', f'sub-{subject}', 'label', 'lh.aparc.a2009s.annot.32k_fs_LR.label.gii')).darrays[0].data
    parc_sub_R = nib.load(os.path.join(data_path, '..', 'freesurfer', f'sub-{subject}', 'label', 'rh.aparc.a2009s.annot.32k_fs_LR.label.gii')).darrays[0].data

    if isinstance(target_labels, tuple):
        targetL = np.concatenate([np.where(parc_sub_L == idx)[0] for idx in target_labels])
        targetR = np.concatenate([np.where(parc_sub_R == idx)[0] for idx in target_labels])
    else:
        targetL = np.where(parc_sub_L == target_labels)[0]
        targetR = np.where(parc_sub_R == target_labels)[0]

    nodes_L = analysis.get_two_nodes(surfL, cortex_L, TpeakL, targetL)
    nodes_R = analysis.get_two_nodes(surfR, cortex_R, TpeakR, targetR)

    shortest_path_L = analysis.shortest_path(surfL, cortex_L, nodes_L[0], nodes_L[1])
    shortest_path_R = analysis.shortest_path(surfR, cortex_R, nodes_R[0], nodes_R[1])

    if save_to is not None and target_name is not None:
        
        vrtx_to_gifti(shortest_path_L, 1, os.path.join(save_to, f'sub-{subject}_L_{target_name}_shortest_path'))
        vrtx_to_gifti(shortest_path_R,1,  os.path.join(save_to, f'sub-{subject}_R_{target_name}_shortest_path'))

    return shortest_path_L, shortest_path_R


def prepare_distance_task_data(task_csv, distances_csv, long):
    task_data = pd.read_csv(task_csv)
    distances = pd.read_csv(distances_csv)
    distance_task_data = task_data.merge(distances, on='participant_id')
    distance_task_data["dist_ratio_L"] = distance_task_data["Dist to A1, LH"]/distance_task_data["Dist to V1, LH"]
    distance_task_data["dist_ratio_R"] = distance_task_data["Dist to A1, RH"]/distance_task_data["Dist to V1, RH"]
    if long:
        distance_task_data = pd.melt(distance_task_data, id_vars=['participant_id', 'FlashType', 'NrBeeps', 'correct', 'rt', 'correct_abs', 'correct_rel', 'rt_abs', 'rt_rel', 'rt_Cod', 'bl_correct', 'age', 'sex'], value_vars=['Dist to A1, LH', 'Dist to A1, RH', 'Dist to V1, LH', 'Dist to V1, RH', 'dist_ratio_L', 'dist_ratio_R'], var_name='Dist Type', value_name='Distance')
        distance_task_data = pd.melt(distance_task_data, id_vars=['participant_id', 'FlashType', 'NrBeeps', 'Dist Type', "Distance", "age", "sex"], value_vars=[ 'correct', 'rt', 'correct_abs', 'correct_rel', 'rt_abs', 'rt_rel', 'rt_Cod', 'bl_correct'], var_name = "Performance Var", value_name = "Measure")
        distance_task_data = pd.melt(distance_task_data, id_vars=['participant_id', 'FlashType', 'NrBeeps', 'Dist Type', 'Distance', 'Performance Var', 'Measure'], value_vars=[ 'sex', 'age'], var_name = "Demographic Var", value_name = "Dem Value")


    return distance_task_data