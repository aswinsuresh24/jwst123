#!/usr/bin/env python
# coding: utf-8

# In[18]:


import glob
import os
from astropy.io import fits
from jwst import datamodels
from mosaic import assign_gwcs

def apply_wcs_to_coadd(coadd_file):
    new_file = coadd_file.replace('coadd_', 'coadd_corrected_')

    with fits.open(coadd_file) as hdul:
        wcs_hdr = hdul['SCI'].header

    im = datamodels.open(coadd_file)
    
    wcsobj = assign_gwcs(box_outdir=os.path.dirname(coadd_file), wcs_hdr=wcs_hdr)

    im.meta.wcs = wcsobj
    im.save(new_file)

    return new_file


coadds = glob.glob('../group_*/ref_*/coadd*i2d.fits') ## change directory to existing references as needed

for coadd in coadds:
    apply_wcs_to_coadd(coadd)


# In[1]:


## checking WCS object type in the datamodel

from jwst import datamodels

dm = datamodels.open("../group_0/ref_0/coadd_corrected_0_0_f115w_i2d.fits")
print(type(dm.meta.wcs))


# In[6]:


from jwst import datamodels

dm = datamodels.open("f115w_i2d.fits")
print(type(dm.meta.wcs))

