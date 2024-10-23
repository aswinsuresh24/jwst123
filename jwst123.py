import warnings
warnings.filterwarnings('ignore')
import stwcs
import glob
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
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
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astroscrappy import detect_cosmics
from stwcs import updatewcs
from scipy.interpolate import interp1d
from jwst.pipeline import calwebb_image3
import jhat
from jhat import jwst_photclass,st_wcs_align
import subprocess
from nbutils import input_list
from jwst.datamodels import ImageModel
from astroquery.gaia import Gaia

# Internal dependencies
from common import Constants
from common import Options
from common import Settings
from common import Util
from nircam_setttings import base_params, short_params, long_params
from nbutils import get_filter, get_instrument, get_chip, get_filter, input_list, xmatch_common
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

def create_parser():
    '''
    Create an argument parser

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser
    '''
    parser = argparse.ArgumentParser(description='Reduce JWST data')
    parser.add_argument('--workdir', type=str, default='.', help='Root directory to search for data')
    parser.add_argument('--align-method', default='relative', type=str,
                        help='Alignment method to use. Options are "relative" or "gaia".')
    return parser

def get_input_images(pattern=None, workdir=None):
    if workdir == None:
        workdir = '.'
    if pattern == None:
        pattern = ['*nrca*_cal.fits','*nrcb*_cal.fits']
    return([s for p in pattern for s in glob.glob(os.path.join(workdir,'raw',p))])

def get_detector_chip(filename):
    '''
    Get the detector chip from the file name

    Parameters
    ----------
    filename : str
        File name

    Returns
    -------
    detector_chip : str
        Detector chip
    '''
    fl_split = filename.split('_')
    mask = ['nrc' in x for x in fl_split]
    if any(mask):
        idx = mask.index(True)
        return fl_split[idx]
    
    return None

def apply_nircammask(files):
    '''
    Apply nircammask to input files
    
    Parameters
    ----------
    files : list
        List of files to apply nircammask to
        
    Returns
    -------
    None
    '''
    cmd = ['nircammask', '-etctime']
    for fl in files:
        # cmd += f' {fl}'
        cmd.append(fl)

    subprocess.run(cmd)  

def calc_cky(files):
    '''
    Calculate the sky for input files

    Parameters
    ----------
    files : list
        List of files to calculate the sky for

    Returns
    -------
    None
    '''
    cmd = 'calcsky {fits_base} {rin} {rout} {step} {sigma_low} {sigma_high}'
    for fl in files:
        #read params from options later
        fits_base = fl.replace('.fits','')
        print(fits_base)
        rin = 15
        rout = 25
        step = -64
        sigma_low = 2.25
        sigma_high = 2.00
        cmd_fl = cmd.format(fits_base=fits_base,rin=rin,rout=rout,
                            step=step,sigma_low=sigma_low,sigma_high=sigma_high)
        subprocess.run(cmd_fl,shell=True)

def generate_dolphot_paramfile(basedir):
    '''
    Generate the dolphot parameter file

    Parameters
    ----------
    basedir : str
        Base directory to search for files

    Returns
    -------
    None
    '''
    ref_image = glob.glob(basedir + '/*i2d.fits')
    if len(ref_image) > 1:
        raise ValueError('More than one i2d image found')
    
    ref_image = ref_image[0]
    phot_images = glob.glob(basedir + '/*cal.fits')
    phot_image_base = [os.path.basename(r).replace('.fits', '') for r in phot_images]
    phot_image_det = ['long' if 'long' in get_detector_chip(r) else 'short' for r in phot_images]
    N_img = len(phot_images)
    with open('jwstred_temp_dolphot/m92/nircam/dolphot.param', 'w') as f:
        f.write('Nimg = {}\n'.format(N_img))
        f.write('img0_file = {}\n'.format(os.path.basename(ref_image).replace('.fits', '')))
        # f.write('\n')
        for i, (img, det) in enumerate(zip(phot_image_base, phot_image_det)):
            f.write('img{}_file = {}\n'.format(i+1, img))
            if det == 'short':
                for key, val in short_params.items():
                    f.write('img{}_{} = {}\n'.format(i+1, key, val))
            if det == 'long':
                for key, val in long_params.items():
                    f.write('img{}_{} = {}\n'.format(i+1, key, val))
            # f.write('\n')
        for key, val in base_params.items():
            f.write('{} = {}\n'.format(key, val))

#ALIGNMENT BLOCK
def cut_gaia_sources(image, table_gaia):
    im = fits.open(image)
    hdr = im['SCI'].header
    w = wcs.WCS(hdr)
    nx, ny = hdr['NAXIS1'], hdr['NAXIS2']

    pix_coords = w.all_world2pix(np.array(table_gaia['ra']), np.array(table_gaia['dec']), 1)
    im_x, im_y = pix_coords[0], pix_coords[1]
    mask = (im_x > 0) & (im_x < nx) & (im_y > 0) & (im_y < ny)

    return table_gaia[mask]

def query_gaia(image, dr = 'gaiadr3', save_file = False):
    im = fits.open(image)

    hdr = im['SCI'].header
    nx = hdr['NAXIS1']
    ny = hdr['NAXIS2']

    image_model = ImageModel(im)
    
    #find radius of query region usinng image header
    ra0,dec0 = image_model.meta.wcs(nx/2.0-1,ny/2.0-1)
    coord0 = SkyCoord(ra0,dec0,unit=(u.deg, u.deg), frame='icrs')
    radius_deg = []
    for x in [0,nx-1]:        
        for y in [0,ny-1]:     
            ra,dec = image_model.meta.wcs(x,y)
            radius_deg.append(coord0.separation(SkyCoord(ra,dec,unit=(u.deg, u.deg), frame='icrs')).deg)
    radius_deg = np.amax(radius_deg)*1.1

    #query gaia
    query ="SELECT * FROM {}.gaia_source WHERE CONTAINS(POINT('ICRS',\
            {}.gaia_source.ra,{}.gaia_source.dec),\
            CIRCLE('ICRS',{},{} ,{}))=1;".format(dr,dr,dr,ra0,dec0,radius_deg)

    job5 = Gaia.launch_job_async(query)
    tb_gaia = job5.get_results() 
    tb_gaia = cut_gaia_sources(image, tb_gaia)
    print("Number of Gaia stars:",len(tb_gaia))
    
    if save_file:
        print(f'Saving Gaia query to {save_file}')
        np.savetxt(save_file, np.array(tb_gaia[['ra', 'dec']]), fmt = '%s')
    
    return tb_gaia

def calc_dispersion(gaia_table, phot_file, dist_limit = 1, plot = False):
    jhat_df = pd.read_csv(glob.glob(phot_file)[0], sep = '\s+') #read phot file with uneven spacing
    
    jh_ra, jh_dec = jhat_df['ra'].to_numpy()*u.degree, jhat_df['dec'].to_numpy()*u.degree
    jh_skycoord = SkyCoord(ra = jh_ra, dec = jh_dec)
    ga_ra, ga_dec = np.array(gaia_table['ra'])*u.degree, np.array(gaia_table['dec'])*u.degree
    ga_skycoord = SkyCoord(ra = ga_ra, dec = ga_dec)
    
    dist_matched_df = xmatch_common(jh_skycoord, ga_skycoord, dist_limit=dist_limit)

    clip_mean, clip_median, clip_std = sigma_clipped_stats(dist_matched_df['d2d']**2)
    dispersion = np.sqrt(clip_mean)
    
    if plot:
        plt.hist(dist_matched_df['d2d'], bins = 40, histtype = 'step', 
                 linestyle = '--', color = 'cornflowerblue')
        plt.xlabel('d2d (arcsec)')
        plt.ylabel('frequency')
        plt.title('JWST-Gaia xmatch')
        plt.grid(alpha = 0.2, linestyle = '--')
        plt.show()
        
    return dispersion

def jwst_phot(phot_img):
    jwst_phot = jwst_photclass()
    photfilename = phot_img.replace('.fits','.phot.txt')
    jwst_phot.run_phot(imagename=phot_img,
                        photfilename=photfilename,
                        overwrite=True,
                        ee_radius=70,
        #                    use_dq=True,
                        SNR_min = 10,
                        find_stars_threshold = 5)
    refcat = Table.read(photfilename,format='ascii')
    return refcat, photfilename

def align_to_gaia(align_image, outdir, verbose = False):
    wcs_align = st_wcs_align()
    print(f'Aligning {os.path.basename(align_image)} to Gaia')
    wcs_align.run_all(align_image,
          telescope='jwst',
          outsubdir=outdir,
          overwrite=True,
          d2d_max=2.0,
          find_stars_threshold = 3,
          showplots=2,
          refcatname='Gaia',
          refcat_pmflag = True, #reject proper motion stars
          histocut_order='dxdy',
          use_dq = False,
          verbose = verbose,
          iterate_with_xyshifts = True,
          xshift = -60,
          yshift = 10,
          sharpness_lim=(0.3,0.95),
          roundness1_lim=(-0.7, 0.7),
        #   Nbright = 1000,
          SNR_min= 5,
          dmag_max=.1,
          objmag_lim =(15,22),
          refmag_lim = (12,19),
          binsize_px = 1.0,
          saveplots = 0,
          slope_min = -20/2048,
          savephottable = 0)
    
    jhat_image = os.path.join(outdir, os.path.basename(align_image.replace('cal.fits', 'jhat.fits')))
    #compute dispersion
    gaia_table = query_gaia(align_image, save_file = None)
    dispersion_initial = calc_dispersion(gaia_table, jhat_image.replace('_jhat.fits', '.phot.txt'), dist_limit = 1, plot = True)
    print(f'Initial dispersion: {dispersion_initial}"')
    temp_cal_name = jhat_image.replace('jhat.fits', 'jhat_cal.fits')
    os.rename(jhat_image, temp_cal_name)
    refcat, photfilename = jwst_phot(jhat_image.replace('jhat.fits', 'jhat_cal.fits'))
    dispersion_final = calc_dispersion(gaia_table, photfilename, dist_limit = 1, plot = True)
    print(f'Final dispersion: {dispersion_final}"')
    os.rename(temp_cal_name, jhat_image)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    work_dir = args.workdir
    align_method = args.align_method

    input_images = get_input_images(workdir=work_dir)
    table = input_list(input_images)
    tables = organize_reduction_tables(table, byvisit=True)

    table = tables[0]
    align_images = np.array([r['image'] for r in table])
    align_to_gaia(align_images[0], os.path.join(work_dir, 'jhat'), verbose = True)