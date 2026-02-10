#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:47:20 2022

The scripts is for running tensor-ICA, identifying rejected components, 
plotting components for visual check, denoising all echoes' data.
Here, we have preprocessed the data with relign and smooth with 4mm kernel with SPM12.
the kept ratio for select threshold: 0.3
som function are based on tedana, hence, tedana need to be installed in the enviorment.

@author: feng
"""


import os
import numpy as np
import shutil
from classify_based_tedana import classify_based_tedana
# if you'd like to run on your dataset,  you can chage the drive_loc and subject_folders
drive_loc = '/bif/home/tfeng/linux/projects/meco2010/meco_NF-FPS' # the path for data folder
subject_folders = ['S0003','S0004','S0005','S0006','S0007','S0008','S0009','S0010','S0011','S0013','S0014','S0015','S0016','S0017','S0018','S0020','S0021','S0022','S0023','S0025','S0027','S0028','S0029','S0030']#os.listdir(drive_loc)

for sub_name in subject_folders:
    print(sub_name)
    sess_folders = ['Session1','Session2','Session3']
    for sess_name in sess_folders:
        current_folder = sub_name+'/'+sess_name  # sub and sess

        
        # # 1. load data and run tnensor ICA to get independent spatial components
        # # the data are first motion_correction with the 6 motion parameters estimated from the the 1st echo's images and smoothed
        
        
        # # params need to be changed --------
        # how you named the data need to be changed
        data_file = [drive_loc+'/'+current_folder+'/processed_data/srdata_echo_1.nii',
                        drive_loc+'/'+current_folder+'/processed_data/srdata_echo_2.nii',
                        drive_loc+'/'+current_folder+'/processed_data/srdata_echo_3.nii',
                        drive_loc+'/'+current_folder+'/processed_data/srdata_echo_4.nii']
        out_dir = drive_loc+'/'+current_folder+'/tensor_ICA/'
        
        # what kind of input
        tica_type = 'tica_s4r'
        echo_times = [13,28,43,57]
        
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        # run tensor_ICA with fsl melodic
        os.system('melodic -i '+data_file[0]+','+data_file[1]+','+data_file[2]+','+data_file[3]+' -o ' +\
                    drive_loc+'/'+current_folder+'/tensor_ICA/'+tica_type+' -a tica --dimest=mdl --tr=1 ' +\
                    '--vn --sep_vn -v')
        
        # # 2. load TE distribution to identify noise-related comp
        from component_identify import component_identify
        print('------classifying components------------')
        [reject_comp,acc_cmp,reject_comp1,reject_comp2,reject_comp3] = component_identify(out_dir, tica_type, 0.3, echo_times)
        
        
        # 3. load components and plot components for visual inspection
        from plot_comp import plot_comp
        print('------plotting components------------')
        comp_type = np.zeros(len(reject_comp)+len(acc_cmp)).astype('int')
        comp_type[reject_comp] = 1# components' type: 1:reject component ,0 : accept components
        comp_type[reject_comp3] = 2
        np.savetxt(out_dir+tica_type+'/labels.txt',comp_type)# save the components labels
        plot_comp(drive_loc, current_folder, tica_type,comp_type, 1, echo_times)
        

        # 4. remove the noise-related components
        from remove_comp import remove_comp
        if not os.path.isdir(drive_loc+'/'+current_folder+'/denoised_data/'):
            os.mkdir(drive_loc+'/'+current_folder+'/denoised_data')
        print('------removing noise-related components------------')
        remove_comp(drive_loc, current_folder, data_file, tica_type, reject_comp, acc_cmp)
        
