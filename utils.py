import sys
import os
import numpy as np
import nibabel as nib
sys.path.append('/home/victoria/NeuroConn')
sys.path.append('/home/victoria/surfdist')
from austin_utils import save_gifti
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
from brainspace.utils.parcellation import reduce_by_labels
from brainspace.datasets import load_parcellation
import pandas as pd
from surfdist import analysis
import re
import csv

def get_subject_numbers(directory):
    filenames = os.listdir(directory)
    subject_numbers = []
    for filename in filenames:
        match = re.search(r'sub-(\d+)', filename)
        if match:
            subject_numbers.append(match.group(1))
    return subject_numbers

def dat_to_csv(dat_file_path, save_to = None):
    with open(dat_file_path) as dat_file, open(save_to, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for line in dat_file:
            csv_writer.writerow(line.split())

def get_gradient_data(path, reduced = False, n_parcels = None):
    gradient_data = {}
    for filename in os.listdir(path):
        if "200gradients" in filename:
            subject_number = filename.split("sub-")[1].split("-")[0]
            file_path = os.path.join(path, filename)
            if reduced:
                schaefer_labels_1000 = load_parcellation('schaefer', scale=n_parcels, join=True)
                data = np.load(file_path)[0].T[:100]
                data = np.array([reduce_by_labels(i, schaefer_labels_1000) for i in data]).ravel()
            else:
                data = np.load(file_path)[0].T[:100].ravel()
            gradient_data[subject_number] = data

    return gradient_data

def compile_trial_data(data_path, task_data_path, save_to):
    subjects = FmriPreppedDataSet(data_path).subjects.tolist()
    for subject in subjects:
        try:
            dat_file = [i for i in os.listdir(os.path.join(task_data_path, subject)) if "DFI" in i][0]
            dat_file = os.path.join(task_data_path, subject, dat_file)
            csv_file = dat_file.replace(".dat", ".csv")
            if not os.path.exists(csv_file):
                dat_to_csv(dat_file, csv_file)
            subject_data = pd.read_csv(csv_file)
            if not os.path.exists(save_to):
                subject_data.to_csv(save_to, index = False)
            else:
                all_data = pd.read_csv(save_to)
                all_data = pd.concat([all_data, subject_data])
                all_data.to_csv(save_to, index = False)
        except Exception as e:
            print(e)

def vrtx_to_gifti(vertex_indices, fill_value, out_name):
    cortex_array = np.zeros((32492))
    cortex_array[:] = np.nan
    cortex_array[vertex_indices] = fill_value
    save_gifti(cortex_array, f"{out_name}")

def get_transmodal_node(data_path, subject, V1_labels, peak = 'temporal'):

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

    if isinstance(V1_labels, tuple):
        targetL = np.concatenate([np.where(parc_sub_L == idx)[0] for idx in V1_labels])
        targetR = np.concatenate([np.where(parc_sub_R == idx)[0] for idx in V1_labels])
    else:
        targetL = np.where(parc_sub_L == V1_labels)[0]
        targetR = np.where(parc_sub_R == V1_labels)[0]

    Tnode_L = analysis.get_node(surfL, cortex_L, targetL, TpeakL)
    Tnode_R = analysis.get_node(surfR, cortex_R, targetR, TpeakR)

    return Tnode_L, Tnode_R


def calc_subj_shortest_path(data_path, subject, target, target_name = None, save_to = None):

    Tpeaks = get_transmodal_node(data_path, subject, (43, 45))
    TpeakL = Tpeaks[0]
    TpeakR = Tpeaks[1]

    surfL = nib.load(os.path.join(data_path, 'derivatives', f'sub-{subject}', 'anat', f'sub-{subject}_hemi-L_midthickness.32k_fs_LR.surf.gii'))
    surfR = nib.load(os.path.join(data_path, 'derivatives', f'sub-{subject}', 'anat', f'sub-{subject}_hemi-R_midthickness.32k_fs_LR.surf.gii'))
    
    fsLR_labels_L = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.L.label.gii').darrays[0].data
    fsLR_labels_R = nib.load('/home/victoria/server/public/scratchpad/surfaces/fsLR/labels/fsLR.32k.R.label.gii').darrays[0].data

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

    if isinstance(target, tuple):
        targetL = np.concatenate([np.where(parc_sub_L == idx)[0] for idx in target])
        targetR = np.concatenate([np.where(parc_sub_R == idx)[0] for idx in target])
    else:
        targetL = np.where(parc_sub_L == target)[0]
        targetR = np.where(parc_sub_R == target)[0]

    target_node_L = analysis.get_node(surfL, cortex_L, TpeakL, targetL)
    target_node_R = analysis.get_node(surfR, cortex_R, TpeakR, targetR)

    shortest_path_L = analysis.shortest_path(surfL, cortex_L, TpeakL, target_node_L)
    shortest_path_R = analysis.shortest_path(surfR, cortex_R, TpeakR, target_node_R)

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