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
from multiprocessing import Pool
import astropy.wcs as wcs
import numpy as np
from contextlib import contextmanager
from astropy import units as u
from astropy.utils.data import clear_download_cache,download_file
from astropy.io import fits
from astropy.table import Table, Column, unique, vstack
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astroscrappy import detect_cosmics
from stwcs import updatewcs
from scipy.interpolate import interp1d
from jwst.pipeline import calwebb_image3
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.pipeline import calwebb_image3
import jhat
from jhat import jwst_photclass,st_wcs_align
import subprocess
from nbutils import input_list
from jwst.datamodels import ImageModel
from astroquery.gaia import Gaia
import shapely

# Internal dependencies
from common import Constants
from common import Options
from common import Settings
from common import Util
from nircam_settings import *
from nbutils import input_list, xmatch_common, get_detector_chip, create_filter_table

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
    parser.add_argument('--object', type=str, default='m92', help='Object to reduce')
    parser.add_argument('--ncores', type=int, default=1, help='Number of CPU cores')
    return parser

def mp_init(init_success: int = 0,
            init_failed: int = 0,
            init_success_files: list =[]):
    global success
    global failed
    global success_files
    success = init_success
    failed = init_failed
    success_files = init_success_files

def align_parallel_worker(align_image, outdir, gaia = False, photfilename = None, xshift = 0, 
                          yshift = 0, Nbright = 800, verbose = False, plot = False):
    try:
        align_jwst_image(align_image=align_image, outdir=outdir, gaia=gaia, photfilename=photfilename, xshift=xshift, 
                         yshift=yshift, Nbright=Nbright, verbose=verbose, plot=plot)
        
    except Exception as e:
        raise e

def get_input_images(pattern=None, workdir=None):
    '''
    Collect all images to be processed from the raw subdirectory
    in the work directory

    Parameters
    ----------
    pattern : list
        List of patterns to search for
    workdir : str
        Work directory

    Returns
    -------
    images : list
        List of images to process
    '''
    if workdir == None:
        workdir = '.'
    if pattern == None:
        pattern = ['*nrca*_cal.fits','*nrcb*_cal.fits']
    return([s for p in pattern for s in glob.glob(os.path.join(workdir,'raw',p))])


def pick_deepest_image(table):
    '''
    Pick the deepest image from a table

    Parameters
    ----------
    table : astropy.table.Table
        Table of images

    Returns
    -------
    deepest_image : astropy.table.Row
        Deepest image
    '''
    exptimes = [r['exptime'] for r in table]
    deepest_image = table[exptimes.index(max(exptimes))]
    return deepest_image

def add_alignment_groups(table, use_shapely=False):
    '''
    Add alignment groups to the table calculated from the image
    polygon overlap area

    Parameters
    ----------
    table : astropy.table.Table
        Input list table
    use_shapely: Bool
        find alignment groups using spatially distinct shapely
        geometries - fails in certain dither patterns

    Returns
    -------
    table : astropy.table.Table
        Table with alignment groups and overlap area added
    '''
    pgons, guide_star_id = [], []
    table_indices = np.arange(len(table))
    for im in table['image']:
        region = fits.open(im)['SCI'].header['S_REGION']
        coords = np.array(region.split('POLYGON ICRS  ')[1].split(' '), dtype = float)
        pgons.append(shapely.Polygon(coords.reshape(4, 2)))
        guide_star_id.append(fits.getval(im, 'GDSTARID', ext=0))

    #for each guide star, find a reference image for each image
    #and add the filename of the image to the table
    guide_star_id, pgons = np.array(guide_star_id), np.array(pgons)
    table.add_column(Column(name='gdstar', data=guide_star_id))
    table.add_column(Column(name='ref_img', data=[None]*len(table)))
    if use_shapely:
        for gdstar in np.unique(guide_star_id):
            align_groups = np.array([])
            gdstar_mask = guide_star_id == gdstar
            gdstar_pgons = pgons[gdstar_mask]
            footprint = shapely.unary_union(gdstar_pgons)
            #convert to multipolygon if needed (for single observation footprint)
            if type(footprint) == shapely.geometry.polygon.Polygon:
                footprint = shapely.MultiPolygon([footprint])
            
            #separate the footprint into spatially distinct groups
            for geom_ in footprint.geoms:
                align_groups = np.append(align_groups, geom_)
                
            align_idx, overlap = [], []
            for i, polygon_ in enumerate(gdstar_pgons):
                intersect_area = np.array([shapely.intersection(polygon_, gm_).area/polygon_.area for gm_ in align_groups])
                align_idx.append(np.argmax(intersect_area))
                overlap.append(np.max(intersect_area))

            for aln_idx in np.unique(align_idx):
                aln_mask = np.array(align_idx) == aln_idx
                ref_image = pick_deepest_image(table[gdstar_mask][aln_mask])['image']
                table['ref_img'][table_indices[gdstar_mask][aln_mask]] = ref_image
    else:        
        for gdstar in np.unique(guide_star_id):
            gdstar_mask = guide_star_id == gdstar
            for chip in np.unique(table[gdstar_mask]['chip']):
                chip_mask = table[gdstar_mask]['chip'] == chip
                ref_image = pick_deepest_image(table[gdstar_mask][chip_mask])['image']
                table['ref_img'][table_indices[gdstar_mask][chip_mask]] = ref_image

    return table

def visit_filter_dict(table):
    '''
    Find the filter in each visit which has the maximum spatial
    coverage for the visit

    Parameters
    ----------
    table : astropy.table.Table
        Input list table

    Returns
    -------
    flt_vis_dict : dict
        Dictionary mapping the filter in which the alignment
        mosaic will be created, to each visit
    '''
    visits = np.unique(table['visit']).value
    flt_vis_dict = dict.fromkeys(visits)
    for vis in visits:
        tbl = table[table['visit'] == vis]
        net_polygon = []
        filters = np.unique(tbl['filter']).value
        flt_mask = []
        for filt in filters:
            if 'N' in filt.upper():
                flt_mask.append(False)
                continue
            else:
                flt_mask.append(True)
            pgons = []
            ft = tbl[tbl['filter'] == filt]
            for im, pl in zip(ft['image'], ft['pupil']):
                if 'N' in pl.upper():
                    continue
                region = fits.open(im)['SCI'].header['S_REGION']
                coords = np.array(region.split('POLYGON ICRS  ')[1].split(' '), dtype = float)
                pgons.append(shapely.Polygon(coords.reshape(4, 2)))
            net_polygon.append(shapely.unary_union(pgons))

        area = []
        for pl_ in net_polygon:
            area.append(pl_.area)

        align_filter = filters[flt_mask][np.argmax(area)]
        flt_vis_dict[vis] = align_filter

        #implement recursive reordering of visits within a visit if maximum area mosaic doesn't cover all filters
        # intersection area is 96.8% for sw overlapping with lw filters
        # align_mosaic = net_polygon[np.argmax(area)]
        # intersection_area = [shapely.intersection(i, align_mosaic).area/i.area for i in net_polygon]
        # mask = intersection_area < 0.99
        # revisit_filters = filters[mask]

        # revisit_area =[]
        # for pl_ in net_polygon[mask]:
        #     revisit_area.append()
        
    return flt_vis_dict

def order_visits(table):
    visits = np.unique(table['visit']).value
    field = []
    for vis in visits:
        tbl = table[table['visit'] == vis]
        net_polygon = []
        filters = np.unique(tbl['filter']).value
        for filt in filters:
            pgons = []
            ft = tbl[tbl['filter'] == filt]
            for im in ft['image']:
                region = fits.open(im)['SCI'].header['S_REGION']
                coords = np.array(region.split('POLYGON ICRS  ')[1].split(' '), dtype = float)
                pgons.append(shapely.Polygon(coords.reshape(4, 2)))
            net_polygon.append(shapely.unary_union(pgons))
        field.append(shapely.unary_union(net_polygon))

    sort_order = np.argsort([i.area for i in field])[::-1]
    
    return sort_order

def jwst_phot(phot_img):
    '''
    Run photometry using jhat jwst_photclass

    Parameters
    ----------
    phot_img : str
        Photometry image

    Returns
    -------
    refcat : astropy.table.Table
        Reference catalog
    photfilename : str
        Photometry file name
    '''
    jwst_phot = jwst_photclass()
    photfilename = phot_img.replace('.fits','.phot.txt')
    jwst_phot.run_phot(imagename=phot_img,
                        photfilename=photfilename,
                        overwrite=True,
                        ee_radius=70)
                        # use_dq=True
    refcat = Table.read(photfilename,format='ascii')
    return refcat, photfilename


def generate_level3_mosaic(inputfiles, outdir):
    '''
    Create level3 drizzled mosaic from level2 input files
    using the JWST pipeline

    Parameters
    ----------
    input_files : list
        List of input level2 files
    outdir: str
        Output directory

    Returns
    -------
    mosaic_name : str
        Level3 mosaic file name
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    table = input_list(inputfiles)
    filters = np.unique([r['filter'] for r in table])
    filter_name = '_'.join(filters)
    nircam_asn_file = f'{outdir}/{filter_name}.json'
    base_filenames = np.array([os.path.basename(r['image']) for r in table])
    asn3 = asn_from_list.asn_from_list(base_filenames, 
        rule=DMS_Level3_Base, product_name=f'{filter_name}')
    
    with open(nircam_asn_file, 'w') as outfile:
        name, serialized = asn3.dump(format='json')
        outfile.write(serialized)

    image3 = calwebb_image3.Image3Pipeline()

    outdir_level3 = os.path.join(outdir, 'out')
    if not os.path.exists(outdir_level3):
        os.makedirs(outdir_level3)

    image3.output_dir = outdir_level3
    image3.save_results = True
    image3.tweakreg.skip = True
    image3.skymatch.skip = True
    image3.skymatch.match_down = False
    image3.source_catalog.skip=False
    image3.resample.pixfrac = 1.0
    # image3.pixel_scale = 0.0311
    image3.weight_type = 'ivm'

    image3.run(nircam_asn_file)
    mosaic_name = f'{outdir_level3}/{filter_name}_i2d.fits'
    return mosaic_name

def create_alignment_mosaic(filter_table, outdir, infilter=None, align_to='gaia', ncores = 10, Nbright=800):
    '''
    Create an alignement i2d mosiac from best available
    long wavelength filters or in the given input filter

    Parameters
    ----------
    filter_table : astropy.table.Table
        Input list table
    outdir : str
        Output directory for jhat aligned images
    infilter : str
        Input filter to create the mosaic
        If not given it is picked from a predefined list

    Returns
    -------
    aligned_mosaic : str
        Aligned mosaic file path
    '''
    if align_to is None:
        raise ValueError('Invalid alignment phot file')
    
    #best filters for Gaia alignment, in order
    good_filters = ['f277w', 'f322w2', 'f356w', 'f250m', 'f300m']

    #pick the best filter present in filter_table.keys() 
    #that appears first in good_filters
    align_table = None
    if infilter:
        if infilter in filter_table:
            align_table = filter_table[infilter]
    else:
        for filt in good_filters:
            if filt in filter_table:
                align_table = filter_table[filt]
                break
        
    if align_table is None:
        raise ValueError('No compatible alignment filter found')
    
    #add alignment groups to the table to pick corect
    #reference image for each module
    align_table = add_alignment_groups(align_table)
    jhat_success_images = [os.path.exists(os.path.join(outdir, os.path.basename(i.replace('cal.fits', 'jhat.fits')))) for i in align_table['image']]
    success_files = list(np.array(align_table['image'])[np.array(jhat_success_images)])
    p = Pool(initializer=mp_init, processes=ncores, initargs=(0, 0, success_files))
    jobs = []

    for row in align_table:
        image = row['image']
        ref_image = row['ref_img']
        if not os.path.exists(ref_image.replace('.fits','.phot.txt')):
            _, photfilename = jwst_phot(ref_image)
        else:
            photfilename = ref_image.replace('.fits','.phot.txt')

        jobs.append(p.apply_async(align_parallel_worker,
                                  args=(image, outdir, False, photfilename, 
                                        0, 0, 800, False, False)))
            
    for job in jobs:
        job.get()

    #create i2d mosaic from relative aligned images
    inputfiles = [os.path.join(outdir,os.path.basename(i.replace('cal.fits', 'jhat.fits'))) for i in align_table['image']]
    mosaic_name = generate_level3_mosaic(inputfiles, outdir)

    #align the mosaic to Gaia
    if align_to=='gaia':
        align_jwst_image(align_image=mosaic_name, outdir=outdir, gaia=True, photfilename=None, Nbright=Nbright)
        jh_image = os.path.join(outdir, os.path.basename(mosaic_name).replace('i2d.fits', 'jhat.fits'))
        shutil.move(jh_image, jh_image.replace('jhat.fits', 'jhat_i2d.fits'))
        aligned_mosaic = jh_image.replace('jhat.fits', 'jhat_i2d.fits')
    
    else:
        align_jwst_image(align_image=mosaic_name, outdir=outdir, gaia=False, photfilename=align_to, Nbright=Nbright)
        jh_image = os.path.join(outdir, os.path.basename(mosaic_name).replace('i2d.fits', 'jhat.fits'))
        shutil.move(jh_image, jh_image.replace('jhat.fits', 'jhat_i2d.fits'))
        aligned_mosaic = jh_image.replace('jhat.fits', 'jhat_i2d.fits')

    return aligned_mosaic

def cut_gaia_sources(image, table_gaia):
    ''''
    Remove Gaia sources that are outside the image

    Parameters
    ----------
    image : str
        Image file name
    table_gaia : astropy.table.Table
        Gaia table

    Returns
    -------
    table_gaia : astropy.table.Table
        Table with sources inside the image
    '''
    im = fits.open(image)
    hdr = im['SCI'].header
    w = wcs.WCS(hdr)
    nx, ny = hdr['NAXIS1'], hdr['NAXIS2']

    pix_coords = w.all_world2pix(np.array(table_gaia['ra']), np.array(table_gaia['dec']), 0)
    im_x, im_y = pix_coords[0], pix_coords[1]
    mask = (im_x > 0) & (im_x < nx) & (im_y > 0) & (im_y < ny)

    return table_gaia[mask]

def query_gaia(image, dr = 'gaiadr3', save_file = False):
    '''
    Query Gaia for sources in the image

    Parameters
    ----------
    image : str
        Image file name
    dr : str
        Gaia data release
    save_file : str
        File name to save the Gaia query to

    Returns
    -------
    tb_gaia : astropy.table.Table
        Gaia table
    '''
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

def calc_dispersion(ref_table, phot_file, w = False, dist_limit = 1, 
                     plot = False):
    '''
    Calculate disperson between JWST sources and a reference catalog

    Parameters
    ----------
    ref_table : astropy.table.Table
        Refernce photometry table
    phot_file : str
        Photometry file
    phot_image : str
        Photometry image
    dist_limit : float
        Distance limit for xmatch
    plot : bool
        Plot the histogram of distances

    Returns
    -------
    dispersion : float
        Dispersion between Gaia and JWST sources
    '''
    jhat_df = pd.read_csv(phot_file, sep = '\s+')    

    if w:
        jh_radec = w.all_pix2world(jhat_df['x'], jhat_df['y'], 0)
        jh_ra, jh_dec = np.array(jh_radec[0])*u.degree, np.array(jh_radec[1])*u.degree
        jh_skycoord = SkyCoord(ra = jh_ra, dec = jh_dec)        
    else:    
        jh_ra, jh_dec = jhat_df['ra'].to_numpy()*u.degree, jhat_df['dec'].to_numpy()*u.degree
        jh_skycoord = SkyCoord(ra = jh_ra, dec = jh_dec)
    ref_ra, ref_dec = np.array(ref_table['ra'])*u.degree, np.array(ref_table['dec'])*u.degree
    ref_skycoord = SkyCoord(ra = ref_ra, dec = ref_dec)
    
    dist_matched_df = xmatch_common(jh_skycoord, ref_skycoord, dist_limit=dist_limit)

    clip_mean, clip_median, clip_std = sigma_clipped_stats(dist_matched_df['d2d']**2, sigma_lower=None, sigma_upper=2)
    mean_dispersion, std_dispersion = np.sqrt(clip_mean), np.sqrt(clip_std)
    
    if plot:
        plt.hist(dist_matched_df['d2d'], bins = 40, histtype = 'step', 
                 linestyle = '--', color = 'cornflowerblue')
        plt.axvline(mean_dispersion, linestyle = '--', alpha = 0.8, color = 'royalblue', label = 'mean')
        plt.axvline(mean_dispersion + 2*std_dispersion, linestyle = '--', alpha = 0.8, color = 'purple', label = r'mean+2$\sigma$')
        plt.legend()
        plt.xlabel('d2d (arcsec)')
        plt.ylabel('frequency')
        plt.title('JWST-Gaia xmatch')
        plt.grid(alpha = 0.2, linestyle = '--')
        plt.show()
        
    return mean_dispersion, std_dispersion

def jwst_dispersion(align_image, outdir, photfile=None, gaia=False, plot=False):
    if 'cal.fits' in align_image:
        jhat_image = os.path.join(outdir, os.path.basename(align_image.replace('cal.fits', 'jhat.fits')))
        temp_cal_name = jhat_image.replace('jhat.fits', 'jhat_cal.fits')
        phot_image = False
    elif 'i2d.fits' in align_image:
        jhat_image = os.path.join(outdir, os.path.basename(align_image.replace('i2d.fits', 'jhat.fits')))
        temp_cal_name = jhat_image.replace('jhat.fits', 'jhat_i2d.fits')
        phot_image = temp_cal_name
    else:
        raise ValueError('Invalid image type')
    
    if gaia:
        refcat = query_gaia(align_image, save_file = None)
    else:
        if photfile is None:
            raise ValueError('Input photometric catalog is required')
        refcat = Table.read(photfile, format='ascii')

    disp_in_mean, disp_in_std = calc_dispersion(refcat, jhat_image.replace('_jhat.fits', '.phot.txt'), dist_limit = 0.5, plot = plot)
    print(f'Initial dispersion: {disp_in_mean*1000} mas')

    os.rename(jhat_image, temp_cal_name)
    align_cat, align_photfile = jwst_phot(temp_cal_name)
    disp_fn_mean, disp_fn_std = calc_dispersion(refcat, align_photfile, w = wcs.WCS(fits.getheader(phot_image, ext=1)),
                                                dist_limit = 0.5, plot = plot)
    print(f'Final dispersion: {disp_fn_mean*1000} mas')
    os.rename(temp_cal_name, jhat_image)

    with fits.open(jhat_image, mode='update') as filehandle:
        if gaia:
            filehandle[0].header['GADISPM'] = disp_fn_mean
            filehandle[0].header['GADISPS'] = disp_fn_std
        else:
            filehandle[0].header['JWDISPM'] = disp_fn_mean
            filehandle[0].header['JWDISPS'] = disp_fn_std
            filehandle[0].header['JWCAT'] = os.path.basename(photfile)

    return disp_fn_mean 

def guess_shift(align_image, ref_table, radius_px = 50, res = 5,  plot = False):
    sci_hdr = copy.copy(wcs.WCS(fits.getheader(align_image, ext=1)))
    align_photfile = fix_phot(align_image)
    crpix1, crpix2 = sci_hdr.wcs.crpix

    off, xshift, yshift = [], [], []
    xsh, ysh = np.arange(-radius_px, radius_px+res, res), np.arange(-radius_px, radius_px+res, res)
    ng = int(len(xsh))
    for xs in xsh:
        for ys in ysh:
            in_wcs = copy.copy(sci_hdr)
            in_wcs.wcs.crpix = [crpix1+xs, crpix2+ys]
            disp, _ = calc_dispersion(ref_table, align_photfile, w=in_wcs, dist_limit = 1, plot = False)
            off.append(disp)
            xshift.append(xs)
            yshift.append(ys)

    best_x, best_y = -xshift[np.argmin(off)], -yshift[np.argmin(off)]
    print(f'Best guess for {align_image}: ({best_x}, {best_y})')

    if plot:
        fig, ax = plt.subplots(1, 3, figsize = (24, 6))
        ax[0].scatter(np.array(xshift), np.array(off), s = 5)
        ax[0].grid(linestyle = '--', alpha = 0.5)
        ax[0].set_xlabel('XSHIFT')
        ax[0].set_ylabel('Offset (arcsec)')

        ax[1].scatter(np.array(yshift), np.array(off), s = 5)
        ax[1].grid(linestyle = '--', alpha = 0.5)
        ax[1].set_xlabel('YSHIFT')
        ax[1].set_ylabel('Offset (arcsec)')

        def fmt(x):
            s = f"{x*100:.1f}"
            if s.endswith("0"):
                s = f"{x*100:.0f}"
            return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"
        
        im = ax[2].imshow(np.array(off).reshape(ng,ng).T)
        cs = ax[2].contour(np.arange(0, ng, 1), np.arange(0, ng, 1), np.array(off).reshape(ng, ng).T, colors='white', levels = 3);
        ax[2].clabel(cs, cs.levels, fmt=fmt, fontsize=9)
        fig.colorbar(im, ax=ax[2], pad = 0.02)
        ax[2].set_xticks(np.linspace(0, ng-1, 8), np.round(np.linspace(-radius_px, radius_px, 8)))
        ax[2].set_yticks(np.linspace(0, ng-1, 8), np.round(np.linspace(-radius_px, radius_px, 8)))
        ax[2].set_xlabel('XSHIFT')
        ax[2].set_ylabel('YSHIFT')
        plt.show()

    return best_x, best_y
    
def run_jhat(align_image, outdir, params, gaia = False, photfilename = None, xshift = 0, yshift = 0, Nbright = 800, verbose = False):
    '''
    Run JHAT to align a JWST image to Gaia or a JWST source catalog

    Parameters
    ----------
    align_image : str
        Image to align
    outdir : str
        Output directory
    params: dict
        Parameters for JHAT
    gaia: bool
        If true, align to Gaia
    photfilename : str
        Photometry file name
    xshift, yshift : float
        x and y shift
    Nbright : int
        Number of bright stars to use
    verbose : bool
        Verbose output

    Returns
    -------
    dispersion_final : float
        Dispersion value
    '''
    wcs_align = st_wcs_align()
    if gaia:
        wcs_align.run_all(align_image,
            outsubdir = outdir,
            refcatname = 'Gaia',
            pmflag = True, # propagate proper motion to observation time
            use_dq = True,
            verbose = verbose,
            xshift = xshift,
            yshift = yshift,
            #   refmag_lim = (12,19),
            **params)  

    else:
        if photfilename is None:
            raise ValueError("Input photometric catalog is required")
        wcs_align.run_all(align_image,
            outsubdir=outdir,
            refcatname=photfilename,
            use_dq = True,
            verbose = verbose,
            xshift = xshift,
            yshift = yshift,
            Nbright=Nbright,
            **params)
        
def align_jwst_image(align_image, outdir, gaia = False, photfilename = None, xshift = 0, yshift = 0, Nbright = 800, verbose = False, plot = False):
    print(f"Aligning {os.path.basename(align_image)} to {'Gaia' if gaia else 'JWST'}")
    params = strict_gaia_params if gaia else strict_jwst_params
    if plot: params['showplots'] = 2
        
    try:
        pixscale = np.abs(fits.getval(align_image, 'CDELT1', ext=1)*3600)
    except:
        pixscale = np.abs(fits.getval(align_image, 'CD1_2', ext=1)*3600)
    pixscale = 0.031 if pixscale < 0.032 else 0.062

    try:
        run_jhat(align_image=align_image, 
                outdir=outdir, 
                params=params, 
                gaia=gaia, 
                photfilename=photfilename, 
                xshift=xshift, 
                yshift=yshift, 
                Nbright=Nbright, 
                verbose=verbose)
        disp = jwst_dispersion(align_image=align_image, outdir=outdir, photfile=photfilename, gaia=gaia, plot=plot)
    except Exception as e:
        print(e)
        disp = 99.99

    if disp/pixscale > 1:
        params = relaxed_gaia_params if gaia else relaxed_jwst_params
        if plot: params['showplots'] = 2 
        try:
            run_jhat(align_image=align_image, outdir=outdir, params=params, gaia=gaia, photfilename=photfilename, 
                    xshift=xshift, yshift=yshift, Nbright=Nbright, verbose=verbose)
            disp = jwst_dispersion(align_image=align_image, outdir=outdir, photfile=photfilename, gaia=gaia, plot=plot)
        except Exception as e:
            print(e)
            disp = 99.99

    if disp/pixscale > 1:
        if gaia:
            ref_table = query_gaia(align_image)
        else:
            ref_table = Table.read(photfilename, format='ascii')
        xsh, ysh = guess_shift(align_image, ref_table, radius_px=50, res=5, plot=plot)
        try:
            run_jhat(align_image=align_image, outdir=outdir, params=params, gaia=gaia, photfilename=photfilename, 
                    xshift=xsh, yshift=ysh, Nbright=Nbright, verbose=verbose)
            disp = jwst_dispersion(align_image=align_image, outdir=outdir, photfile=photfilename, gaia=gaia, plot=plot)
        except Exception as e:
            print(e)
            disp = 99.99

    if disp/pixscale > 1:
        if gaia:
            ref_table = query_gaia(align_image)
        else:
            ref_table = Table.read(photfilename, format='ascii')
        xsh, ysh = guess_shift(align_image, ref_table, radius_px=100, res=5, plot=plot)
        try:
            run_jhat(align_image=align_image, outdir=outdir, params=params, gaia=gaia, photfilename=photfilename, 
                    xshift=xsh, yshift=ysh, Nbright=Nbright, verbose=verbose)
            disp = jwst_dispersion(align_image=align_image, outdir=outdir, photfile=photfilename, gaia=gaia, plot=plot)
        except Exception as e:
            print(e)
            disp = 99.99

    print(f'''Final {'Gaia' if gaia else 'JWST'} dispersion for {align_image}: {disp*1000} mas''')

    if disp/pixscale > 1:
        print(f'Copying unaligned {align_image} to output, redo alignment')
        if 'cal.fits' in align_image:
            jhat_image = os.path.join(outdir, os.path.basename(align_image.replace('cal.fits', 'jhat.fits')))
        elif 'i2d.fits' in align_image:
            jhat_image = os.path.join(outdir, os.path.basename(align_image.replace('i2d.fits', 'jhat.fits')))

        shutil.copy(align_image, jhat_image)
        with fits.open(jhat_image, mode='update') as filehandle:
            if gaia:
                filehandle[0].header['GADISPM'] = disp
                filehandle[0].header['GADISPS'] = 'NaN'
            else:
                filehandle[0].header['JWDISPM'] = disp
                filehandle[0].header['JWDISPS'] = 'NaN'
                filehandle[0].header['JWCAT'] = 'NaN'

def fix_phot(mosaic):
    '''
    Fix jhat photometry for i2d files

    Parameters
    ----------
    mosaic : str
        Mosaic file name

    Returns
    -------
    photfile : str  
        Fixed photometry file name
    '''
    refcat, photfile = jwst_phot(mosaic)
    w = wcs.WCS(fits.open(mosaic)['SCI'].header)
    jh_radec = w.all_pix2world(refcat['x'], refcat['y'], 0)
    jh_ra, jh_dec = np.array(jh_radec[0]), np.array(jh_radec[1])
    refcat['ra'], refcat['dec'] = jh_ra, jh_dec
    refcat.write(photfile.replace('i2d', 'i2d.corr'), format = 'ascii', overwrite = True)
    return photfile.replace('i2d', 'i2d.corr')

def get_visit_geoms(table):
    visits = np.unique(table['visit']).value
    field = []
    for vis in visits:
        tbl = table[table['visit'] == vis]
        net_polygon = []
        filters = np.unique(tbl['filter']).value
        for filt in filters:
            pgons = []
            ft = tbl[tbl['filter'] == filt]
            for im in ft['image']:
                region = fits.open(im)['SCI'].header['S_REGION']
                coords = np.array(region.split('POLYGON ICRS  ')[1].split(' '), dtype = float)
                pgons.append(shapely.Polygon(coords.reshape(4, 2)))
            net_polygon.append(shapely.unary_union(pgons))
        field.append(shapely.unary_union(net_polygon))
    
    return dict(zip(visits, field))

def pick_visit(align_pgon, visit_geoms, flt_vis_dict):
    if align_pgon is None:
        narrowvis = np.array(list(flt_vis_dict.keys()))[np.array(['N' in i.upper() for i in flt_vis_dict.values()])]
        #BUG: narrowvis could have let narrow band pupils pass through
        for v_ in narrowvis:
            if (v_ in list(visit_geoms.keys())) and (len(list(visit_geoms.keys())) > 1):
                visit_geoms.pop(v_)
        vis_area = [visit_geoms[i].area for i in visit_geoms.keys()]
        arg = np.argmax(vis_area)
        ar_ = vis_area[arg]
        vis = list(visit_geoms.keys())[arg]

    else:
        intersect_area = [align_pgon.intersection(visit_geoms[i]).area/visit_geoms[i].area for i in visit_geoms.keys()]
        arg = np.argmax(intersect_area)
        ar_ = intersect_area[arg]
        vis = list(visit_geoms.keys())[arg]
    
    return vis, ar_

def update_refcat(mosaic_name, photfile, out_refcat, align_pgon):
    flt = fits.getval(mosaic_name, keyword='FILTER', ext=0)
    ppl = fits.getval(mosaic_name, keyword='PUPIL', ext=0)
    
    if not os.path.exists(out_refcat):
        refcat = Table.read(photfile, format='ascii')
        refcat[['ra', 'dec', 'mag', 'dmag']].write(out_refcat, format='ascii', overwrite=True)

    else:
        if ('N' in flt) or ('N' in ppl):
            #BUG: could be an edge case where a visit overlaps only with another narrowband visit
            return 0
        
        mastercat = Table.read(out_refcat, format='ascii')
        refcat = Table.read(photfile, format='ascii')

        pts = shapely.points(refcat['ra'], refcat['dec'])
        mask = shapely.contains(align_pgon, pts)
        refcat = refcat[~mask]

        mastercat = vstack([mastercat, refcat[['ra', 'dec', 'mag', 'dmag']]])
        mastercat.write(out_refcat, format='ascii', overwrite=True)
    
    return 0

def align_to_mosaic(mosaic_photfile, cal_images, outdir, gaia_offset = (0, 0), verbose = False, ncores = 10):
    '''
    Align JWST images to the alignment mosaic

    Parameters
    ----------
    mosaic_photfile : str
        Mosaic photometry file
    cal_images : list
        List of images to align
    outdir : str
        Output directory

    Returns
    -------
    None
    '''
    xshift, yshift = gaia_offset
    jhat_success_images = [os.path.exists(os.path.join(outdir, os.path.basename(i.replace('cal.fits', 'jhat.fits')))) for i in cal_images]
    success_files = list(np.array(cal_images)[np.array(jhat_success_images)])
    p = Pool(initializer=mp_init, processes=ncores, initargs=(0, 0,success_files))
    jobs = []
    for im in cal_images:
        jobs.append(p.apply_async(align_parallel_worker,
                                  args=(im, outdir, False, mosaic_photfile, xshift, 
                                        yshift, 800, verbose, False)))
        
    for job in jobs:
        job.get()

def create_dirs(work_dir, obj):
    '''
    Create directories for dolphot

    Parameters
    ----------
    work_dir : str
        Work directory
    obj : str
        Object name

    Returns
    -------
    outdir : str
        Output directory
    '''
    outdir = os.path.join(work_dir, obj)
    aligndir = os.path.join(work_dir, 'align')
    refdir = os.path.join(work_dir, 'reference')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(aligndir):
        os.makedirs(aligndir)
    if not os.path.exists(refdir):
        os.makedirs(refdir)
    return outdir

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    work_dir = args.workdir
    obj = args.object
    ncores = args.ncores

    dolphot_basedir = create_dirs(work_dir, obj)

    input_images = get_input_images(workdir=work_dir)
    table = input_list(input_images)
    ngroups, nvisits = np.unique(table['group']), np.unique(table['visit'])
    flt_vis_dict = visit_filter_dict(table)

    for g in ngroups:
        combined_photfile = os.path.join(work_dir, 'align', f'group_{g}', 'reference_catalog.txt')
        subtable = table[table['group'] == g]
        subvisits = np.unique(subtable['visit'])
        visit_geoms = get_visit_geoms(subtable)
        align_pgon = None

        for i in range(len(subvisits)):
            vis, ar_ = pick_visit(align_pgon, copy.copy(visit_geoms), flt_vis_dict)
            visit_tbl = subtable[subtable['visit'] == vis]
            
            filters = np.unique(visit_tbl['filter']).value
            filter_table = create_filter_table(visit_tbl, filters)
            infilter = flt_vis_dict[vis]

            if i == 0:
                mosaic_name = create_alignment_mosaic(filter_table, 
                                                      os.path.join(work_dir, 'align', f'group_{g}', f'visit_{vis}'), 
                                                      infilter = infilter, 
                                                      align_to='gaia',
                                                      ncores=ncores)

            else:
                if ar_ < 0.3:
                    Nbright = 50000
                else:
                    Nbright = 800
                mosaic_name = create_alignment_mosaic(filter_table, 
                                                      os.path.join(work_dir, 'align', f'group_{g}', f'visit_{vis}'), 
                                                      infilter = infilter, 
                                                      align_to=combined_photfile,
                                                      ncores=ncores,
                                                      Nbright=Nbright)
                
            mosaic_photfile = fix_phot(mosaic_name)
            _ = update_refcat(mosaic_name, mosaic_photfile, out_refcat=combined_photfile, align_pgon=align_pgon)

            print(f'Mosaic photfile: {mosaic_photfile}')
            for filt, tbl in filter_table.items():
                align_to_mosaic(mosaic_photfile, [r['image'] for r in tbl], os.path.join(work_dir, 'jhat'), 
                                gaia_offset = (0,0), verbose = False, ncores=ncores)

            if align_pgon is None:
                align_pgon = visit_geoms[vis]
            else:
                align_pgon = shapely.unary_union([align_pgon, visit_geoms[vis]])

            visit_geoms.pop(vis)