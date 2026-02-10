#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:07:26 2022
function for remove noise-related comp and create new file
refer to the regress methods implemented in tedana
@author: feng
"""

import nibabel as nib       
from nilearn.image import check_niimg
import numpy as np
from tedana import utils
from tedana.stats import get_coeffs
from nilearn.image import check_niimg
from nilearn.image import new_img_like
from scipy.stats import zscore
def new_nii_like(ref_img, data, affine=None, copy_header=True):
    """
    Coerces `data` into NiftiImage format like `ref_img`

    Parameters
    ----------
    ref_img : :obj:`str` or img_like
        Reference image
    data : (S [x T]) array_like
        Data to be saved
    affine : (4 x 4) array_like, optional
        Transformation matrix to be used. Default: `ref_img.affine`
    copy_header : :obj:`bool`, optional
        Whether to copy header from `ref_img` to new image. Default: True

    Returns
    -------
    nii : :obj:`nibabel.nifti1.Nifti1Image`
        NiftiImage
    """

    ref_img = check_niimg(ref_img)
    newdata = data.reshape(ref_img.shape[:3] + data.shape[1:])
    if ".nii" not in ref_img.valid_exts:
        # this is rather ugly and may lose some information...
        nii = nib.Nifti1Image(newdata, affine=ref_img.affine, header=ref_img.header)
    else:
        # nilearn's `new_img_like` is a very nice function
        nii = new_img_like(ref_img, newdata, affine=affine, copy_header=copy_header)
    nii.set_data_dtype(data.dtype)

    return nii      


def remove_comp(drive_loc1,current_folder,data_file,tica_type,reject_comp,acc_cmp):
    # load components
    fsl_components = nib.load(drive_loc1+'/'+current_folder+'/tensor_ICA/'+tica_type+'/melodic_IC.nii.gz')
    fsl_components = check_niimg(fsl_components)
    (nx, ny, nz) = fsl_components.shape[:3]
    fsl_components = fsl_components.get_fdata()
    fsl_components = fsl_components.reshape((-1,)+fsl_components.shape[3:]).squeeze()
    time_course = np.loadtxt(drive_loc1+'/'+current_folder+'/tensor_ICA/'+tica_type+'/melodic_Tmodes') 
    mask = check_niimg(drive_loc1+'/'+current_folder+'/tensor_ICA/'+tica_type+'/mask.nii.gz')  
    mask = mask.get_fdata().reshape(-1,).squeeze().astype(bool)
    time_length = np.size(time_course,0)
    for i in range(len(data_file)):
        data = check_niimg(nib.load(data_file[i]))
        (nx, ny), nz = data.shape[:2], data.shape[2]
        ref_img = data.__class__(
            np.zeros((nx, ny, nz, 1)), affine=data.affine, header=data.header, extra=data.extra
        )
        ref_img.header.extensions = []
        ref_img.header.set_sform(ref_img.header.get_sform(), code=1)
        
        
        data = data.get_fdata()
        data = data.reshape((-1,)+data.shape[3:]).squeeze()[:,0:time_length]
        
        data_z = data[mask,:].T - data[mask,:].T.mean(axis=0)
        # get mixing matrix for this echo's data for these components based on the spatial maps
        mmix = np.dot(np.linalg.pinv(fsl_components[mask,:]),data_z.T).T
        mdata = data[mask,:]
        dmdata = mdata.T - mdata.T.mean(axis=0)
        betas = get_coeffs(dmdata.T, mmix, mask=None)
        varexpl = (1 - ((dmdata.T - betas.dot(mmix.T)) ** 2.0).sum() / (dmdata ** 2.0).sum()) * 100
    
        # create component-based data
        hikts = utils.unmask(betas[:, acc_cmp].dot(mmix.T[acc_cmp, :]), mask)
        lowkts = utils.unmask(betas[:, reject_comp].dot(mmix.T[reject_comp, :]), mask)
        dnts = utils.unmask(data[mask] - lowkts[mask], mask)
        img = new_nii_like(ref_img, np.float32(dnts))
        img.to_filename(drive_loc1+'/'+current_folder+'/denoised_data/denoised_'+tica_type.split('_')[1]+'_'+str(i+1)+'.nii')
