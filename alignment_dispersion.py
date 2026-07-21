#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import os
from astropy.io import fits 
import numpy as np
import jwst123

outdir = 'alignment_output'

ref_image = glob.glob('../group_1/ref_9/coadd_corrected_1_9_f115w_i2d.fits')[0]

photfilename = jwst123.fix_phot(ref_image, telescope='jwst')

align_image = glob.glob('../jwst_data_M51/M51/F560W_144084448/mastDownload/JWST/jw01783007002_02101_00001_mirimage/jw01783007002_02101_00001_mirimage_cal.fits')[0]
align_photfile = jwst123.fix_phot(align_image, telescope='jwst')

jwst123.align_jwst_image(
    align_image=align_image,
    gaia=False,
    plot=True,
    outdir=outdir,
    verbose=True,
    photfilename=photfilename,
    Nbright=800)


# In[ ]:




