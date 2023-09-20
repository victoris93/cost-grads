import sys
import os 
sys.path.append('/home/victoria/NeuroConn')
import tqdm
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet

surf_type = sys.argv[1] # pial or inflated

data_path = '/home/victoria/server/data/COST/COST_mri/derivatives'
dataset = FmriPreppedDataSet(data_path + "/rest")
subjects = dataset.subjects.tolist()
new_deformed_sphere_L = "/home/victoria/server/public/scratchpad/surfaces/resample_fsaverage/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii"
new_deformed_sphere_R = "/home/victoria/server/public/scratchpad/surfaces/resample_fsaverage/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii"
if surf_type == "pial":
    output_surf_type = "midthickness"
elif surf_type == "inflated":
    output_surf_type = "inflated"

def parse_wb_surf_resample(subject, data_path, new_deformed_sphere, surf_type, hemi, output_nat_surf):
    pial = os.path.join(data_path, "freesurfer", f"sub-{subject}", "surf", f"{hemi}.{surf_type}")
    white = os.path.join(data_path, "freesurfer", f"sub-{subject}", "surf", f"{hemi}.white")
    current_fs_sphere = os.path.join(data_path, "freesurfer", f"sub-{subject}", "surf", f"{hemi}.sphere.reg")
    temp_new_fs_surf = f"{hemi}.{surf_type}.surf.gii"
    temp_new_fs_sphere = f"{hemi}.sphere.reg.surf.gii"
    output_nat_surf = os.path.join(data_path, "rest", "derivatives", f"sub-{subject}", "anat", output_nat_surf)
    wb_surf_resample_cmd_lh = f"wb_shortcuts -freesurfer-resample-prep {white} {pial} {current_fs_sphere} {new_deformed_sphere} {temp_new_fs_surf} {output_nat_surf} {temp_new_fs_sphere}"
    return wb_surf_resample_cmd_lh
i = 0

for subject in tqdm.tqdm(subjects):
    output_nat_surf_L = f"sub-{subject}_hemi-L_{output_surf_type}.32k_fs_LR.surf.gii"
    output_nat_surf_R = f"sub-{subject}_hemi-R_{output_surf_type}.32k_fs_LR.surf.gii"

    wb_surf_resample_cmd_L = parse_wb_surf_resample(subject, data_path, new_deformed_sphere_L, surf_type, "lh", output_nat_surf_L)
    wb_surf_resample_cmd_R = parse_wb_surf_resample(subject, data_path, new_deformed_sphere_R, surf_type, "rh", output_nat_surf_R)
    if not os.path.exists(os.path.join(data_path, "rest", "derivatives", f"sub-{subject}", "anat", output_nat_surf_L)):
        try:
            os.system(wb_surf_resample_cmd_L)
            os.system(wb_surf_resample_cmd_R)
            i += 1
        except Exception as e:
            print(e)
    else:
        print(f"Subject {subject} already resampled")
print(f"Resampled {i} subjects")