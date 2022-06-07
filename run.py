# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:35:49 2022

@author: Douglas Harvey
"""



import pupil_fft_psf as pfp
import numpy as np
import matplotlib.pyplot as plt
from mpl_interactions import hyperslicer



"""
0: Round variable diameter aperture.
1: Round fixed diameter aperture with variable central obstruction.
2: Round fixed diameter aperture with variable grid.
"""
aperture_type = 0



if aperture_type == 0:

    aper_max_dia = 64
    pad_factor = 3
    steps = 64
    
    aperture_array = pfp.aperture_arr3d((0, aper_max_dia, steps), size = aper_max_dia*pad_factor)
    fft_array = pfp.fft_arr3d(aperture_array)
    
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8))
    aperture_image = hyperslicer(aperture_array, ax = ax1)
    fft_array = hyperslicer(np.abs(fft_array), ax = ax2, controls = aperture_image)
    plt.tight_layout()



if aperture_type == 1:

    aper_dia = 64
    pad_factor = 3
    obs_max_dia = 64
    steps = 64
    
    aperture_array = pfp.aperture_arr3d((0, aper_dia, steps), size = aper_dia*pad_factor, constant_diameter = True)
    obstruction_array = pfp.aperture_arr3d((0, obs_max_dia, steps), size = aper_dia*pad_factor)
    combined_array = aperture_array - obstruction_array
    fft_array = pfp.fft_arr3d(combined_array)
    
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8))
    combined_image = hyperslicer(combined_array, ax = ax1)
    fft_array = hyperslicer(np.abs(fft_array), ax = ax2, controls = combined_image)
    plt.tight_layout()



if aperture_type == 2:

    aper_dia = 64
    pad_factor = 3
    grid_max_spacing = 64
    steps = 64
    
    aperture_array = pfp.aperture_arr3d((0, aper_dia, steps), size = aper_dia*pad_factor, constant_diameter = True)
    grid_array = pfp.grid_arr3d((1, grid_max_spacing, steps), aper_dia*pad_factor)
    combined_array = aperture_array*(1 - grid_array)
    fft_array = pfp.fft_arr3d(combined_array)
    
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8))
    combined_image = hyperslicer(combined_array, ax = ax1)
    fft_array = hyperslicer(np.abs(fft_array), ax = ax2, controls = combined_image)
    plt.tight_layout()