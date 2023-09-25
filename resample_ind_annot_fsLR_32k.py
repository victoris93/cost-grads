import sys
import os 
sys.path.append('/home/victoria/NeuroConn')
import tqdm
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet


hemi = sys.argv[1]

def parse_wb_surf_resample(subject, data_path, new_deformed_sphere, surf_type, hemi, output_nat_surf):
    nat_surf = os.path.join(data_path, "freesurfer", f"sub-{subject}", "surf", f"{hemi}.{surf_type}")
    white = os.path.join(data_path, "freesurfer", f"sub-{subject}", "surf", f"{hemi}.white")
    current_fs_sphere = os.path.join(data_path, "freesurfer", f"sub-{subject}", "surf", f"{hemi}.sphere.reg")
    temp_new_fs_surf = os.path.join(data_path, "freesurfer", f"sub-{subject}", "surf", f"{hemi}.{surf_type}.surf.gii")
    temp_new_fs_sphere = os.path.join(data_path, "freesurfer", f"sub-{subject}", "surf", f"{hemi}.sphere.reg.surf.gii")
    output_nat_surf = os.path.join(data_path, "rest", "derivatives", f"sub-{subject}", "anat", output_nat_surf)
    wb_surf_resample_cmd_lh = f"wb_shortcuts -freesurfer-resample-prep {white} {nat_surf} {current_fs_sphere} {new_deformed_sphere} {temp_new_fs_surf} {output_nat_surf} {temp_new_fs_sphere}"
    return wb_surf_resample_cmd_lh, temp_new_fs_surf, temp_new_fs_sphere, output_nat_surf

data_path = '/home/victoria/server/data/COST/COST_mri/derivatives'
dataset = FmriPreppedDataSet(data_path + "/rest")
subjects = dataset.subjects.tolist()
new_deformed_sphere_L = "/home/victoria/server/public/scratchpad/surfaces/resample_fsaverage/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii"
new_deformed_sphere_R = "/home/victoria/server/public/scratchpad/surfaces/resample_fsaverage/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii"
if hemi == 'lh':
    sphere = 'L'
elif hemi == 'rh':
    sphere = 'R'
i = 0
for subject in tqdm.tqdm(subjects):
    try:
        fs_annot = f"/home/victoria/server/data/COST/COST_mri/derivatives/freesurfer/sub-{subject}/label/{hemi}.aparc.a2009s.annot"
        fs_white = f"/home/victoria/server/data/COST/COST_mri/derivatives/freesurfer/sub-{subject}/surf/{hemi}.white"
        annot_gii = f"/home/victoria/server/data/COST/COST_mri/derivatives/freesurfer/sub-{subject}/label/{hemi}.aparc.a2009s.annot.gii"
        output_annot_fslr = f"/home/victoria/server/data/COST/COST_mri/derivatives/freesurfer/sub-{subject}/label/{hemi}.aparc.a2009s.annot.32k_fs_LR.label.gii"
        convert_annot_gii_cmd = f"mris_convert --annot {fs_annot} {fs_white} {annot_gii}"
        os.system(convert_annot_gii_cmd)
        if hemi == 'lh':
            output_nat_surf = f"sub-{subject}_hemi-L_midthickness.32k_fs_LR.surf.gii"
            wb_resample_prep_cmd, temp_new_fs_surf, temp_new_fs_sphere, output_nat_surf = parse_wb_surf_resample(subject, data_path, new_deformed_sphere_L, 'pial', hemi, output_nat_surf)
        else:
            output_nat_surf = f"sub-{subject}_hemi-R_midthickness.32k_fs_LR.surf.gii"
            wb_resample_prep_cmd, temp_new_fs_surf, temp_new_fs_sphere, output_nat_surf = parse_wb_surf_resample(subject, data_path, new_deformed_sphere_R, 'pial', hemi, output_nat_surf)
        os.system(wb_resample_prep_cmd)

        if hemi == 'lh':
            cmd = f"wb_command -label-resample {annot_gii} {temp_new_fs_sphere} {new_deformed_sphere_L} ADAP_BARY_AREA {output_annot_fslr} -area-surfs {temp_new_fs_surf} {output_nat_surf}"
        else:
            cmd = f"wb_command -label-resample {annot_gii} {temp_new_fs_sphere} {new_deformed_sphere_R} ADAP_BARY_AREA {output_annot_fslr} -area-surfs {temp_new_fs_surf} {output_nat_surf}"
        os.system(cmd)
        print(f"Annot of subject {subject}, {hemi} hemosphere resampled.")
    except Exception as e:
        print(e)

    
