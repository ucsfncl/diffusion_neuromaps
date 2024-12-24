# This script samples diffusion metrics to the fsLR 32k surface for a given subject
import numpy as np
import os
import nibabel as nib
from dipy.io import image
from nilearn import surface
import nilearn
=import nibabel as nib
from scipy import interpolate
import shutil
import copy


def clean_mapmri(arr: np.ndarray) -> np.ndarray:
    arr[arr < 0] = np.NaN
    arr[arr == np.inf] = np.NaN
    arr[arr == -np.inf] = np.NaN
    med_val = np.nanmedian(arr)
    mad_val = np.nanmedian(np.abs(arr - med_val))
    arr[arr > med_val + 4.5 * mad_val] = np.NaN
    arr[arr < med_val - 4.5 * mad_val] = np.NaN
    return arr


if __name__ == "__main__":
    data_path = "/data/HCP"  # path to diffusion data
    surf_path = "/data/HCP_SURFACE"  # path to surface data
    subjects = np.loadtxt("diffusion_neuromaps/data/hcp_qc_subjects.txt", dtype=str)

    subject_id = int(os.getenv("SGE_TASK_ID")) - 1
    subject = subjects[subject_id]
    print(subject)

    # shutil.rmtree(os.path.join(surf_path, subject, "32k"), ignore_errors=True)

    # Metrics to use
    metric_dict = {
        "dki-fa": os.path.join(data_path, subject, "DKI-mppca-debias", "fa.nii.gz"),
        "dki-ad": os.path.join(data_path, subject, "DKI-mppca-debias", "ad.nii.gz"),
        "dki-rd": os.path.join(data_path, subject, "DKI-mppca-debias", "rd.nii.gz"),
        "dki-md": os.path.join(data_path, subject, "DKI-mppca-debias", "md.nii.gz"),
        "mk": os.path.join(data_path, subject, "DKI-mppca-debias", "mk.nii.gz"),
        "ak": os.path.join(data_path, subject, "DKI-mppca-debias", "ak.nii.gz"),
        "rk": os.path.join(data_path, subject, "DKI-mppca-debias", "rk.nii.gz"),
        "kfa": os.path.join(data_path, subject, "DKI-mppca-debias", "kfa.nii.gz"),
        "mkt": os.path.join(data_path, subject, "DKI-mppca-debias", "mkt.nii.gz"),
        "odi": os.path.join(data_path, subject, "NODDI-mppca-debias-1.1", "odi.nii.gz"),
        "icvf": os.path.join(data_path, subject, "NODDI-mppca-debias-1.1", "ndi.nii.gz"),
        "isovf": os.path.join(data_path, subject, "NODDI-mppca-debias-1.1", "fiso.nii.gz"),
        "fa": os.path.join(data_path, subject, "DTI-mppca-debias", "fa.nii.gz"),
        "ad": os.path.join(data_path, subject, "DTI-mppca-debias", "ad.nii.gz"),
        "rd": os.path.join(data_path, subject, "DTI-mppca-debias", "rd.nii.gz"),
        "md": os.path.join(data_path, subject, "DTI-mppca-debias", "md.nii.gz"),
        "rtop": os.path.join(data_path, subject, "MAPMRI", "RTOP.nii.gz"),
        "rtap": os.path.join(data_path, subject, "MAPMRI", "RTAP.nii.gz"),
        "rtpp": os.path.join(data_path, subject, "MAPMRI", "RTPP.nii.gz"),
        "msd": os.path.join(data_path, subject, "MAPMRI", "MSD.nii.gz"),
        "qiv": os.path.join(data_path, subject, "MAPMRI", "QIV.nii.gz")
    }

    os.makedirs(os.path.join(surf_path, subject), exist_ok=True)
    os.makedirs(os.path.join(surf_path, subject, "32k"), exist_ok=True)

    # Ribbon and surface files
    ribbon_file = os.path.join(surf_path, subject, "ribbon.nii.gz")
    rh_pial_file = os.path.join(surf_path, subject, "rh_32k_pial.surf.gii")
    rh_wm_file = os.path.join(surf_path, subject, "rh_32k_white.surf.gii")
    lh_pial_file = os.path.join(surf_path, subject, "lh_32k_pial.surf.gii")
    lh_wm_file = os.path.join(surf_path, subject, "lh_32k_white.surf.gii")
    lh_mid_file = os.path.join(surf_path, subject, "lh_32k_midthickness.surf.gii")
    rh_mid_file = os.path.join(surf_path, subject, "rh_32k_midthickness.surf.gii")

    # load ribbon file and set ribbon as gm mask
    ribbon, affine = image.load_nifti(ribbon_file)
    ribbon_mask = np.zeros_like(ribbon)
    ribbon_mask[ribbon == 3] = 1.0
    ribbon_mask[ribbon == 42] = 1.0
    ribbon = nib.Nifti1Image(ribbon_mask, affine)

    # load misthickness surface file to be used by interpolator
    rh_mid = surface.load_surf_mesh(rh_mid_file)
    lh_mid = surface.load_surf_mesh(lh_mid_file)

    rh_mask = np.loadtxt(os.path.join("diffusion_neuromaps", "atlases", "fsLR_32k_cortex-rh_mask.txt"), dtype=bool)
    lh_mask = np.loadtxt(os.path.join("diffusion_neuromaps", "atlases", "fsLR_32k_cortex-lh_mask.txt"), dtype=bool)
    
    # resample ribbon to DWI resolution (1.25 mm isotropic) and set values to 0 or 1
    ribbon  = nilearn.image.resample_to_img(ribbon, os.path.join(data_path, subject, "DTI-mppca-debias", "fa.nii.gz"), interpolation="nearest")
    mask, affine = image.load_nifti(os.path.join(data_path, subject, "nodif_brain_mask.nii.gz"))
    lh_pts = lh_mid.coordinates[lh_mask, :]
    rh_pts = rh_mid.coordinates[rh_mask, :]

    for key, metric_file in metric_dict.items():
        print(key)
        metric, affine = image.load_nifti(metric_file)
        param_mask = copy.deepcopy(mask)

        # special outlier detection for MAP-MRI metrics
        if key in ["rtop", "rtap", "rtpp", "msd", "qiv"]:
            metric[param_mask.astype(bool)] = clean_mapmri(metric[param_mask.astype(bool)])
            param_mask[np.isnan(metric)] = 0.0

        # Clip metrics as necesary
        metric = np.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)
        metric = np.clip(metric, 0.0, None)
        if key in ["odi", "icvf", "isovf", "fa", "dki-fa", "kfa"]:
            metric = np.clip(metric, 0.0, 1.0)
        if key in ["ad", "rd", "md", "dki-ad", "dki-rd", "dki-md"]:
            metric = np.clip(metric, 0.0, 3e-3)

        # mask out values that fall +/- 1 std dev from the mean in the ribbon
        ribbon_mean = np.nanmean(metric[(param_mask * ribbon.get_fdata()).astype(bool)])
        ribbon_std = np.nanstd(metric[(param_mask * ribbon.get_fdata()).astype(bool)])
        param_mask[metric > ribbon_mean + ribbon_std] = 0.0
        param_mask[metric < ribbon_mean - ribbon_std] = 0.0
        param_mask = param_mask * ribbon.get_fdata()
        param_mask = nib.Nifti1Image(param_mask, affine)
        metric = np.clip(metric, ribbon_mean - ribbon_std, ribbon_mean + ribbon_std)

        # interpolate metric to surface
        metric_img = nib.Nifti1Image(metric, affine)
        lh_metric = surface.vol_to_surf(metric_img, surf_mesh=lh_pial_file, inner_mesh=lh_wm_file, mask_img=param_mask, n_samples=120)
        rh_metric = surface.vol_to_surf(metric_img, surf_mesh=rh_pial_file, inner_mesh=rh_wm_file, mask_img=param_mask, n_samples=120)
        lh_metric = lh_metric[lh_mask]
        rh_metric = rh_metric[rh_mask]

        print(np.sum(np.isnan(lh_metric)))
        print(np.sum(np.isnan(rh_metric)))

        # Fill in missing values with nearest neighbor
        lh_interp_nn = interpolate.NearestNDInterpolator(lh_pts[~np.isnan(lh_metric), :], lh_metric[~np.isnan(lh_metric)])
        rh_interp_nn = interpolate.NearestNDInterpolator(rh_pts[~np.isnan(rh_metric), :], rh_metric[~np.isnan(rh_metric)])
        lh_fill = lh_interp_nn(lh_pts[np.isnan(lh_metric), :])
        rh_fill = rh_interp_nn(rh_pts[np.isnan(rh_metric), :])
        lh_metric[np.isnan(lh_metric)] = lh_fill
        rh_metric[np.isnan(rh_metric)] = rh_fill

        print(np.sum(np.isnan(lh_metric)))
        print(np.sum(np.isnan(rh_metric)))

        print(np.mean(lh_metric))
        print(np.mean(rh_metric))
        print(np.std(lh_metric))
        print(np.std(rh_metric))

        print(np.min(lh_metric))
        print(np.min(rh_metric))
        print(np.max(lh_metric))
        print(np.max(rh_metric))

        np.save(os.path.join(surf_path, subject, "32k", f"lh_{key}.npy"), lh_metric)
        np.save(os.path.join(surf_path, subject, "32k", f"rh_{key}.npy"), rh_metric)