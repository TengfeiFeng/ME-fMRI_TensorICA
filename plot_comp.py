#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:55:36 2022

@author: feng
"""
import matplotlib.pyplot as plt
import numpy as np
from tedana import io, stats, utils
import os
import nibabel as nib       
from nilearn.image import check_niimg
from nilearn.image import new_img_like

def plot_comp(drive_loc,current_folder,tica_type,comp_type,tr,echo_times):
    components = nib.load(drive_loc+'/'+current_folder+'/tensor_ICA/'+tica_type+'/melodic_IC.nii.gz')#spatial maps
    components = check_niimg(components)
    (nx, ny, nz) = components.shape[:3]
    ts_B = components.get_fdata()#spatial map
    mmix = np.loadtxt(drive_loc+'/'+current_folder+'/tensor_ICA/'+tica_type+'/melodic_Tmodes')#time course
    mmix = mmix[:,range(0,np.size(mmix,1),5)]
    TE_mode = np.loadtxt(drive_loc+'/'+current_folder+'/tensor_ICA/'+tica_type+'/melodic_Smodes')
    n_vols = len(mmix)
    cuts = [ts_B.shape[dim] // 6 for dim in range(3)]
    png_cmap = 'coolwarm'
    out_dir = drive_loc+'/'+current_folder+'/Comp_figures/'+tica_type
    if not os.path.isdir(drive_loc+'/'+current_folder+'/Comp_figures/'):
        os.mkdir(drive_loc+'/'+current_folder+'/Comp_figures/')
    if not os.path.isdir(out_dir):  
        os.mkdir(out_dir)
    for compnum in range(np.size(ts_B,3)):
        allplot = plt.figure(figsize=(10, 12))
        ax_ts = plt.subplot2grid((6, 6), (0, 0), rowspan=1, colspan=6, fig=allplot)
        
        ax_ts.set_xlabel("TRs")
        ax_ts.set_xlim(0, n_vols)
        plt.yticks([])
        # Make a second axis with units of time (s)
        max_xticks = 10
        xloc = plt.MaxNLocator(max_xticks)
        ax_ts.xaxis.set_major_locator(xloc)
        
        ax_ts2 = ax_ts.twiny()
        ax1Xs = ax_ts.get_xticks()
        
        ax2Xs = []
        for X in ax1Xs:
            # Limit to 2 decimal places
            seconds_val = round(X * tr, 2)
            ax2Xs.append(seconds_val)
        ax_ts2.set_xticks(ax1Xs)
        ax_ts2.set_xlim(ax_ts.get_xbound())
        ax_ts2.set_xticklabels(ax2Xs)
        ax_ts2.set_xlabel("seconds")
        line_color = "g"
        ax_ts.plot(mmix[:, compnum], color=line_color)
        
        # Title will include variance from comptable
        # comp_var = "{0:.2f}".format(comptable.loc[compnum, "variance explained"])
        # comp_kappa = "{0:.2f}".format(comptable.loc[compnum, "kappa"])
        # comp_rho = "{0:.2f}".format(comptable.loc[compnum, "rho"])
        # plt_title = "Comp. {}: variance: {}%, kappa: {}, rho: {}, {}".format(
        #     compnum, comp_var, comp_kappa, comp_rho, expl_text
        # )
        if len(comp_type)==1:
            plt_title = "comp. #{}".format(compnum)
        else:
            plt_title = "comp. #{}".format(compnum)+['   accepted','   rejected-low TE peak','   rejected-abnormal frequency band'][comp_type[compnum]]
        title = ax_ts.set_title(plt_title)
        title.set_y(1.5)
        
        # Set range to ~1/10th of max positive or negative beta
        imgmax = 2.3#0.1 * np.abs(ts_B[:, :, :, compnum]).max()
        imgmin = imgmax * -1
        
        for idx, _ in enumerate(cuts):
            for imgslice in range(1, 6):
                ax = plt.subplot2grid((6, 6), (idx + 1, imgslice - 1), rowspan=1, colspan=1)
                ax.axis("off")
        
                if idx == 0:
                    to_plot = np.rot90(ts_B[imgslice * cuts[idx], :, :, compnum])
                if idx == 1:
                    to_plot = np.rot90(ts_B[:, imgslice * cuts[idx], :, compnum])
                if idx == 2:
                    to_plot = ts_B[:, :, imgslice * cuts[idx], compnum]
        
                ax_im = ax.imshow(to_plot, vmin=imgmin, vmax=imgmax, aspect="equal", cmap=png_cmap)
        
        # Add a color bar to the plot.
        ax_cbar = allplot.add_axes([0.83, 0.37, 0.03, 0.37])
        cbar = allplot.colorbar(ax_im, ax_cbar)
        cbar.set_label("Component zscore", rotation=90)
        cbar.ax.yaxis.set_label_position("left")
        
        # Get fft and freqs for this subject
        # adapted from @dangom
        # spectrum, freqs = utils.get_spectrum(mmix[:, compnum], tr)
       
        # Plot it
        ax_te = plt.subplot2grid((6, 6), (4, 0), rowspan=1, colspan=6)
        ax_te.plot(echo_times, TE_mode[:,compnum])
        ax_te.set_title("TE_distribution")
        ax_te.set_xlabel("TEs")
        ax_te.set_xlim(10, 60)
        plt.yticks([])
        # Get fft and freqs for this subject
        # adapted from @dangom
        spectrum, freqs = utils.get_spectrum(mmix[:, compnum], tr)
    
        # Plot it
        ax_fft = plt.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=6)
        ax_fft.plot(freqs, spectrum)
        ax_fft.set_title("One Sided fft")
        ax_fft.set_xlabel("Hz")
        ax_fft.set_xlim(freqs[0], freqs[-1])
        plt.yticks([])
        
        
        # Fix spacing so TR label does overlap with other plots
        allplot.subplots_adjust(hspace=0.4)
        plot_name = "comp_{}.png".format(str(compnum).zfill(3))
        compplot_name = os.path.join(out_dir, plot_name)
        plt.savefig(compplot_name)
        plt.close()

    