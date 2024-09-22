import warnings
warnings.filterwarnings('ignore')
import stwcs
import glob
import sys
import os
import shutil
import time
import filecmp
import astroquery
import progressbar
import copy
import requests
import random
import astropy.wcs as wcs
import numpy as np
from contextlib import contextmanager
from astropy import units as u
from astropy.utils.data import clear_download_cache,download_file
from astropy.io import fits
from astropy.table import Table, Column, unique
from astropy.time import Time
from astroscrappy import detect_cosmics
from stwcs import updatewcs
from scipy.interpolate import interp1d
from jwst.pipeline import calwebb_image3
import jhat
from jhat import jwst_photclass,st_wcs_align

# Internal dependencies
from common import Constants
from common import Options
from common import Settings
from common import Util
from nbutils import get_filter, get_instrument, get_chip, get_filter, input_list
from nbutils import get_zpt, add_visit_info, organize_reduction_tables, pick_deepest_images

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

with suppress_stdout():
    from drizzlepac import tweakreg,astrodrizzle,catalogs,photeq
    from astroquery.mast import Observations
    from astropy.coordinates import SkyCoord

import jwst
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
print(f'JWST version: {jwst.__version__}')

def get_detector_chip(filename):
    fl_split = filename.split('_')
    mask = ['nrc' in x for x in fl_split]
    if any(mask):
        idx = mask.index(True)
        return fl_split[idx]
    
    return None

if __name__ == '__main__':
    root_dir = '.'
    
    input_images = glob.glob('jwstred_temp_mosaic/*cal.fits')
    table = input_list(input_images)
    tables = organize_reduction_tables(table, byvisit=False)

    align_images = np.array([r['image'] for r in table])

    ref_mask = ['0001' in r for r in align_images]
    ref_images = align_images[ref_mask]
    print(ref_images)

    dets = [get_detector_chip(r) for r in ref_images]
    ref_dict = dict(zip(dets, ref_images))
    print(ref_dict)

    for ref_image in ref_images:
        jwst_phot = jwst_photclass()
        jwst_phot.run_phot(imagename=ref_image,photfilename='auto',overwrite=True,ee_radius=80)

    wcs_align = st_wcs_align()

    for align_image in align_images:

        if align_image in ref_images:
            continue

        chip = get_detector_chip(align_image)
        ref_image = ref_dict[chip]
        ref_catname = ref_image.replace('.fits','.phot.txt')
        refcat = Table.read(ref_catname,format='ascii')

        wcs_align.run_all(align_image,
                telescope='jwst',
                outsubdir='jwstred_temp_mosaic',    
                refcat_racol='ra',
                refcat_deccol='dec',
                refcat_magcol='mag',
                refcat_magerrcol='dmag',
                overwrite=True,
                d2d_max=.25,
                showplots=0,
                find_stars_threshold=3,
                refcatname=ref_catname,
                histocut_order='dxdy',
                    sharpness_lim=(0.3,0.9),
                    roundness1_lim=(-0.7, 0.7),
                    SNR_min= 5,
                    dmag_max=0.1,
                    objmag_lim =(14,21))

    #gaia
    #     wcs_align.run_all(align_image,
    #                     telescope='jwst',
    #                     outsubdir='jwstred_temp+mosaic',
    #                     overwrite=True,
    #                     d2d_max=1,
    #                     find_stars_threshold = 3,
    #                     showplots=0,
    #                     refcatname='Gaia',
    #                     histocut_order='dxdy',
    #                         sharpness_lim=(0.3,0.9),
    #                         roundness1_lim=(-0.7, 0.7),
    #                         SNR_min= 3,
    #                         dmag_max=1,
    #                         Nbright = 150,
    #                         objmag_lim =(14,20))

    filter_name = table[0]['filter']
    nircam_asn_file = f'jwstred_temp_mosaic/{filter_name}.json'
    base_filenames = np.array([os.path.basename(r) for r in glob.glob('jwstred_temp_mosaic/*jhat.fits')])
    asn3 = asn_from_list.asn_from_list(base_filenames, 
        rule=DMS_Level3_Base, product_name=filter_name)
    
    with open(nircam_asn_file, 'w') as outfile:
        name, serialized = asn3.dump(format='json')
        outfile.write(serialized)

    image3 = calwebb_image3.Image3Pipeline()

    outdir_level3 = os.path.join('jwstred_temp_mosaic/', 'out')
    if not os.path.exists(outdir_level3):
        os.makedirs(outdir_level3)

    image3.output_dir = outdir_level3
    image3.save_results = True
    image3.tweakreg.skip = True
    image3.skymatch.skip = True
    image3.skymatch.match_down = False
    image3.source_catalog.skip=False

    image3.run(nircam_asn_file)