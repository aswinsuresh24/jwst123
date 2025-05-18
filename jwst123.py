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
    return parser

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
        for filt in filters:
            if 'N' in filt:
                continue
            pgons = []
            ft = tbl[tbl['filter'] == filt]
            for im, pl in zip(ft['image'], ft['pupil']):
                if 'N' in pl:
                    continue
                region = fits.open(im)['SCI'].header['S_REGION']
                coords = np.array(region.split('POLYGON ICRS  ')[1].split(' '), dtype = float)
                pgons.append(shapely.Polygon(coords.reshape(4, 2)))
            net_polygon.append(shapely.unary_union(pgons))

        area = []
        for pl_ in net_polygon:
            area.append(pl_.area)

        align_filter = filters[np.argmax(area)]
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

def create_alignment_mosaic(filter_table, outdir, work_dir, infilter=None, align_to='gaia'):
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

    for row in align_table:
        image = row['image']
        ref_image = row['ref_img']
        refcat, photfilename = jwst_phot(ref_image)
        dispersion = align_to_jwst(image, photfilename, outdir=outdir, verbose=False)
        print(f'Dispersion for {image}: {dispersion}"')

    #create i2d mosaic from relative aligned images
    inputfiles = [os.path.join(outdir,os.path.basename(i.replace('cal.fits', 'jhat.fits'))) for i in align_table['image']]
    mosaic_name = generate_level3_mosaic(inputfiles, outdir)

    #align the mosaic to Gaia
    if align_to=='gaia':
        dispersion = align_to_gaia(mosaic_name, outdir, xshift = 0, yshift = 0, verbose=False)
        jh_image = os.path.join(outdir, os.path.basename(mosaic_name).replace('i2d.fits', 'jhat.fits'))
        shutil.move(jh_image, jh_image.replace('jhat.fits', 'jhat_i2d.fits'))
        aligned_mosaic = jh_image.replace('jhat.fits', 'jhat_i2d.fits')
        print(f'Gaia Dispersion for {aligned_mosaic}: {dispersion}"')
    
    else:
        dispersion = align_to_jwst(mosaic_name, align_to, outdir=outdir, verbose=False)
        jh_image = os.path.join(outdir, os.path.basename(mosaic_name).replace('i2d.fits', 'jhat.fits'))
        shutil.move(jh_image, jh_image.replace('jhat.fits', 'jhat_i2d.fits'))
        aligned_mosaic = jh_image.replace('jhat.fits', 'jhat_i2d.fits')
        print(f'Dispersion for {mosaic_name}: {dispersion}"')

    return aligned_mosaic

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

def calc_sky(files):
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

def generate_dolphot_paramfile(basedir, files = None):
    '''
    Generate the dolphot parameter file

    Parameters
    ----------
    basedir : str
        Base directory to search for files
    files : list
        List of files to be included in the paramfile

    Returns
    -------
    None
    '''
    ref_image = glob.glob(basedir + '/*i2d.fits')
    if len(ref_image) > 1:
        raise ValueError('More than one i2d image found')
    
    ref_image = ref_image[0]
    if files:
        phot_images = files
    else:
        phot_images = glob.glob(basedir + '/*jhat.fits')
    phot_image_base = [os.path.basename(r).replace('.fits', '') for r in phot_images]
    phot_image_det = ['long' if 'long' in get_detector_chip(r) else 'short' for r in phot_images]
    N_img = len(phot_images)
    #dolphot can only process n images at a time
    #write the multiple param files in chunks of n images
    #with the same reference image
    paramfiles = []
    NMAX = 148
    if N_img > NMAX:
        N_chunks = int(N_img/NMAX) + 1
        for i in range(N_chunks):
            paramfiles.append(f'dolphot_{i}.param')
            n_file = NMAX if i < N_chunks - 1 else N_img - i*NMAX
            with open(f'{basedir}/dolphot_{i}.param', 'w') as f:
                f.write('Nimg = {}\n'.format(n_file))
                f.write('img0_file = {}\n'.format(os.path.basename(ref_image).replace('.fits', '')))

                for j, (img, det) in enumerate(zip(phot_image_base[i*NMAX:(i+1)*NMAX], phot_image_det[i*NMAX:(i+1)*NMAX])):
                    f.write('img{}_file = {}\n'.format(j+1, img))
                    if det == 'short':
                        for key, val in short_params.items():
                            f.write('img{}_{} = {}\n'.format(j+1, key, val))
                    if det == 'long':
                        for key, val in long_params.items():
                            f.write('img{}_{} = {}\n'.format(j+1, key, val))
                    # f.write('\n')
                for key, val in base_params.items():
                    f.write('{} = {}\n'.format(key, val))

    else:
        paramfiles.append(f'dolphot.param')
        with open(f'{basedir}/dolphot.param', 'w') as f:
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
    
    return paramfiles

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

def calc_dispersion(gaia_table, phot_file,
                     phot_image = False, dist_limit = 1, 
                     plot = False):
    '''
    Calculate disperson between Gaia and JWST sources

    Parameters
    ----------
    gaia_table : astropy.table.Table
        Gaia table
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

    if phot_image:
        hdr = fits.open(phot_image)['SCI'].header
        w = wcs.WCS(hdr)
        jh_radec = w.all_pix2world(jhat_df['x'], jhat_df['y'], 0)
        jh_ra, jh_dec = np.array(jh_radec[0])*u.degree, np.array(jh_radec[1])*u.degree
        jh_skycoord = SkyCoord(ra = jh_ra, dec = jh_dec)
    else:    
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

def align_to_gaia(align_image, outdir, xshift = 0, yshift = 0, verbose = False):
    '''
    Align JWST image to the Gaia frame

    Parameters
    ----------
    align_image : str
        Image to align
    outdir : str
        Output directory
    xshift, yshift : float
        x and y shift
    verbose : bool
        Verbose output

    Returns
    -------
    dispersion_final : float
        Dispersion value
    '''
    wcs_align = st_wcs_align()
    print(f'Aligning {os.path.basename(align_image)} to Gaia')
    try:
        wcs_align.run_all(align_image,
            outsubdir=outdir,
            refcatname='Gaia',
            refcat_pmflag = True, #reject proper motion stars
            use_dq = False,
            verbose = verbose,
            xshift = xshift,
            yshift = yshift,
            #   Nbright = 1500,
            #   refmag_lim = (12,19),
            **strict_gaia_params)
    except: #relaxed cuts
        wcs_align.run_all(align_image,
            outsubdir=outdir,
            refcatname='Gaia',
            refcat_pmflag = True, #reject proper motion stars
            use_dq = False,
            verbose = verbose,
            xshift = xshift,
            yshift = yshift,
            **relaxed_gaia_params)
    
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
    #compute dispersion
    gaia_table = query_gaia(align_image, save_file = None)
    dispersion_initial = calc_dispersion(gaia_table, jhat_image.replace('_jhat.fits', '.phot.txt'), dist_limit = 2, plot = False)
    print(f'Initial dispersion: {dispersion_initial}"')
    os.rename(jhat_image, temp_cal_name)
    refcat, photfilename = jwst_phot(temp_cal_name)
    dispersion_final = calc_dispersion(gaia_table, photfilename, phot_image = phot_image, dist_limit = 2, plot = False)
    print(f'Final dispersion: {dispersion_final}"')
    os.rename(temp_cal_name, jhat_image)

    with fits.open(jhat_image, mode='update') as filehandle:
        filehandle[0].header['GADISP'] = dispersion_final

    return dispersion_final

def align_to_jwst(align_image, photfilename, outdir, xshift = 0, yshift = 0, Nbright = 800, verbose = False):
    '''
    Align JWST image to another JWST image

    Parameters
    ----------
    align_image : str
        Image to align
    photfilename : str
        Photometry file name
    outdir : str
        Output directory
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
    print(f'Aligning {os.path.basename(align_image)} to JWST')
    try:
        wcs_align.run_all(align_image,
                        outsubdir=outdir,
                        refcatname=photfilename,
                        verbose = verbose,
                        xshift = xshift,
                        yshift = yshift,
                        Nbright=Nbright,
                        **strict_jwst_params)   

    except: #relaxed cuts
        wcs_align.run_all(align_image,
                        outsubdir=outdir,
                        refcatname=photfilename,
                        verbose = verbose,
                        xshift = xshift,
                        yshift = yshift,
                        Nbright=Nbright,
                        **relaxed_jwst_params)       

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

    refcat_in = Table.read(photfilename, format='ascii')
    dispersion_initial = calc_dispersion(refcat_in, jhat_image.replace('_jhat.fits', '.phot.txt'), dist_limit = 2, plot = False)
    print(f'Initial dispersion: {dispersion_initial}"')
    os.rename(jhat_image, temp_cal_name)
    aligned_refcat, aligned_photfilename = jwst_phot(temp_cal_name)
    dispersion_final = calc_dispersion(refcat_in, aligned_photfilename, phot_image = phot_image, dist_limit = 2, plot = False)
    print(f'Final dispersion: {dispersion_final}"')
    os.rename(temp_cal_name, jhat_image)

    with fits.open(jhat_image, mode='update') as filehandle:
        filehandle[0].header['JWDISP'] = dispersion_final
        filehandle[0].header['JWCAT'] = os.path.basename(photfilename)

    return dispersion_final

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

def pick_visit(align_pgon, visit_geoms):
    if align_pgon is None:
        #this could still pick a narrowband visit 
        arg = np.argmax([visit_geoms[i].area for i in visit_geoms.keys()])
        vis = list(visit_geoms.keys())[arg]

    else:
        intersect_area = [align_pgon.intersection(visit_geoms[i]).area/visit_geoms[i].area for i in visit_geoms.keys()]
        arg = np.argmax(intersect_area)
        vis = list(visit_geoms.keys())[arg]
    
    return vis

def update_refcat(mosaic_name, photfile, out_refcat, align_pgon):
    flt = fits.getval(mosaic_name, keyword='FILTER', ext=0)
    ppl = fits.getval(mosaic_name, keyword='PUPIL', ext=0)
    if ('N' in flt) or ('N' in ppl):
        #could be an edge case where a visit overlaps only with another narrowband visit
        return 0
    
    if not os.path.exists(out_refcat):
        refcat = Table.read(photfile, format='ascii')
        refcat[['ra', 'dec', 'mag', 'dmag']].write(out_refcat, format='ascii', overwrite=True)

    else:
        mastercat = Table.read(out_refcat, format='ascii')
        refcat = Table.read(photfile, format='ascii')

        pts = shapely.points(refcat['ra'], refcat['dec'])
        mask = shapely.contains(align_pgon, pts)
        refcat = refcat[~mask]

        mastercat = vstack([mastercat, refcat[['ra', 'dec', 'mag', 'dmag']]])
        mastercat.write(out_refcat, format='ascii', overwrite=True)
    
    return 0

def align_to_mosaic(mosaic_photfile, cal_images, outdir, gaia_offset = (0, 0), verbose = False):
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
    for im in cal_images:
        _ = align_to_jwst(im, mosaic_photfile, outdir, xshift = xshift, 
                          yshift = yshift, Nbright = None, verbose = verbose)

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

    dolphot_basedir = create_dirs(work_dir, obj)

    input_images = get_input_images(workdir=work_dir)
    table = input_list(input_images)
    ngroups, nvisits = np.unique(table['group'], np.unique(table['visit']))
    flt_vis_dict = visit_filter_dict(table)

    for g in ngroups:
        combined_photfile = os.path.join(work_dir, 'align', f'group_{g}', 'reference_catalog.txt')
        subtable = table[table['group'] == g]
        visit_geoms = get_visit_geoms(subtable)
        align_pgon = None

        for i in range(len(nvisits)):
            vis = pick_visit(align_pgon, visit_geoms)
            visit_tbl = subtable[subtable['visit'] == vis]
            
            filters = np.unique(visit_tbl['filter']).value
            filter_table = create_filter_table(visit_tbl, filters)
            infilter = flt_vis_dict[vis]

            if i == 0:
                mosaic_name = create_alignment_mosaic(filter_table, 
                                                      os.path.join(work_dir, 'align', f'group_{g}', f'visit_{vis}'), 
                                                      work_dir, 
                                                      infilter = infilter, 
                                                      align_to='gaia')

            else:
                mosaic_name = create_alignment_mosaic(filter_table, 
                                                      os.path.join(work_dir, 'align', f'group_{g}', f'visit_{vis}'), 
                                                      work_dir, 
                                                      infilter = infilter, 
                                                      align_to=combined_photfile)
                
            mosaic_photfile = fix_phot(mosaic_name)
            _ = update_refcat(mosaic_name, mosaic_photfile, out_refcat=combined_photfile, align_pgon=align_pgon)

            print(f'Mosaic photfile: {mosaic_photfile}')
            for filt, tbl in filter_table.items():
                align_to_mosaic(mosaic_photfile, [r['image'] for r in tbl], os.path.join(work_dir, 'jhat'), 
                                gaia_offset = (0,0), verbose = False)

            if align_pgon is None:
                align_pgon = visit_geoms[vis]
            else:
                align_pgon = shapely.unary_union([align_pgon, visit_geoms[vis]])

            visit_geoms.pop(vis)