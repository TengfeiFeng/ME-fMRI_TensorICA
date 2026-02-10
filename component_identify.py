#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:51:54 2022

@author: feng
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:07:51 2022
to identify noise related components
@author: tfeng
"""
import numpy as np
import math
from tedana import utils
from numpy import polyfit
def component_identify(out_dir,tica_type,keep_ratio,echo_times):
    num_echoes = len(echo_times)
    interp_degree = num_echoes -1
    
    TE_file = out_dir+'/'+tica_type+'/melodic_Smodes'
    Time_course = np.loadtxt(out_dir+'/'+tica_type+'/melodic_Tmodes')[:,0:-1:num_echoes+1]

    TE_mode = np.loadtxt(TE_file)
    reject_comp = []
    reject_comp1 = [] # decreased pattern components
    reject_comp2 = [] # low TE-peak components
    reject_comp3 = [] # with abnormal frequency bands
    second_level_comp = []
    acc_cmp = [] # accept comp
    
    coeff = polyfit(echo_times, TE_mode, interp_degree)
    peaks = -coeff[1]/(2*coeff[0])
    for i in range(len(TE_mode.transpose())):
        if peaks[i] < 15 or peaks[i]>55:
            reject_comp.append(i)
            reject_comp1.append(i)
        # if max(TE_mode[:, i]) == TE_mode[0, i]:  # identify the exponential decay pattern noise, need change with exponential decay fit
        #     reject_comp.append(i)
        else:
            second_level_comp.append(i)
    coeff = polyfit(echo_times, TE_mode[:,second_level_comp], interp_degree)
    peaks = -coeff[1]/(2*coeff[0])
    if math.ceil(np.size(TE_mode,1)*keep_ratio)<len(peaks)*0.7:
        threshold = sorted(peaks)[-math.ceil(np.size(TE_mode,1)*keep_ratio)]
    else:
        threshold = sorted(peaks)[-math.ceil(len(peaks)*0.7)]
    for i in range(np.size(coeff,1)):
        if peaks[i]>=threshold:
            spectrum, freqs = utils.get_spectrum(Time_course[:,second_level_comp[i]], 1)
            ratio = sum(spectrum[np.where((freqs>0.01)&(freqs<0.1))])/sum(spectrum[np.where(freqs>0.01)])
            # print(ratio)
            if ratio>0.7:
                acc_cmp.append(second_level_comp[i])
            else:
                reject_comp.append(second_level_comp[i])
                reject_comp3.append(second_level_comp[i])
        else:
            reject_comp.append(second_level_comp[i])
            reject_comp2.append(second_level_comp[i])
    return [reject_comp,acc_cmp,reject_comp1,reject_comp2,reject_comp3]
