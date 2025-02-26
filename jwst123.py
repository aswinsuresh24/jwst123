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

def add_alignment_groups(table):
    '''
    Add alignment groups to the table calculated from the image
    polygon overlap area

    Parameters
    ----------
    table : astropy.table.Table
        Input list table

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
    # for gdstar in np.unique(guide_star_id):
    #     align_groups = np.array([])
    #     gdstar_mask = guide_star_id == gdstar
    #     gdstar_pgons = pgons[gdstar_mask]
    #     footprint = shapely.unary_union(gdstar_pgons)
    #     #convert to multipolygon if needed (for single observation footprint)
    #     if type(footprint) == shapely.geometry.polygon.Polygon:
    #         footprint = shapely.MultiPolygon([footprint])
        
    #     #separate the footprint into spatially distinct groups
    #     for geom_ in footprint.geoms:
    #         align_groups = np.append(align_groups, geom_)
            
    #     align_idx, overlap = [], []
    #     for i, polygon_ in enumerate(gdstar_pgons):
    #         intersect_area = np.array([shapely.intersection(polygon_, gm_).area/polygon_.area for gm_ in align_groups])
    #         align_idx.append(np.argmax(intersect_area))
    #         overlap.append(np.max(intersect_area))

    #     for aln_idx in np.unique(align_idx):
    #         aln_mask = np.array(align_idx) == aln_idx
    #         ref_image = pick_deepest_image(table[gdstar_mask][aln_mask])['image']
    #         table['ref_img'][table_indices[gdstar_mask][aln_mask]] = ref_image
        
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
            pgons = []
            ft = tbl[tbl['filter'] == filt]
            for im in ft['image']:
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

def create_filter_table(tables, filters):
    '''
    Create a dictionary of (filter, table) pairs for
    the full observation set

    This is useful to collect all observations in a 
    patricular filter when the observation is a mix

    Parameters
    ----------
    tables : list
        List of tables
    filters : list
        Filter of each table

    Returns
    -------
    filter_table : dict
        Dictionary of (filter, table) pairs
    '''
    filter_table = dict.fromkeys(filters)
    for flt in filter_table.keys():
        # flt_tables = [tbl_ for tbl_ in tables if tbl_['filter'][0] == flt]
        filter_table[flt] = tables[tables['filter'] == flt] #vstack(flt_tables)

    return filter_table

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

    #move input files to outdir
    if 'reference' in outdir:
        for fl in inputfiles:
            shutil.move(fl, outdir)
        inputfiles = glob.glob(f'{outdir}/*nrc*jhat.fits')

    table = input_list(inputfiles)
    # tables = organize_reduction_tables(table, byvisit=False)
    # filter_name = table[0]['filter']
    # module_name = table[0]['module']
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

def create_alignment_mosaic(filter_table, outdir, work_dir, infilter=None):
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
    # inputfiles = glob.glob(f'{outdir}/*long*jhat*.fits') #this does not account for bymodule=True
    inputfiles = [os.path.join(outdir,os.path.basename(i.replace('cal.fits', 'jhat.fits'))) for i in tbl['image']]
    mosaic_name = generate_level3_mosaic(inputfiles, outdir)

    #align the mosaic to Gaia
    dispersion = align_to_gaia(mosaic_name, outdir, xshift = 0, yshift = 0, verbose=False)
    jh_image = os.path.join(outdir, os.path.basename(mosaic_name).replace('i2d.fits', 'jhat.fits'))
    shutil.move(jh_image, jh_image.replace('jhat.fits', 'jhat_i2d.fits'))
    aligned_mosaic = jh_image.replace('jhat.fits', 'jhat_i2d.fits')
    shutil.move(aligned_mosaic, work_dir)
    print(f'Dispersion for {aligned_mosaic}: {dispersion}"')

    return os.path.join(work_dir, os.path.basename(aligned_mosaic))

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
    # phot_images = glob.glob(basedir + '/*cal.fits')
    phot_images = glob.glob(basedir + '/*jhat.fits')
    phot_image_base = [os.path.basename(r).replace('.fits', '') for r in phot_images]
    phot_image_det = ['long' if 'long' in get_detector_chip(r) else 'short' for r in phot_images]
    N_img = len(phot_images)
    #dolphot can only process 99 images at a time
    #write the multiple param files in chunks of 99 images
    #with the same reference image
    paramfiles = []
    if N_img > 98:
        N_chunks = int(N_img/98) + 1
        for i in range(N_chunks):
            paramfiles.append(f'dolphot_{i}.param')
            n_file = 98 if i < N_chunks - 1 else N_img - i*98
            with open(f'{basedir}/dolphot_{i}.param', 'w') as f:
                f.write('Nimg = {}\n'.format(n_file))
                f.write('img0_file = {}\n'.format(os.path.basename(ref_image).replace('.fits', '')))

                for j, (img, det) in enumerate(zip(phot_image_base[i*98:(i+1)*98], phot_image_det[i*98:(i+1)*98])):
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

    pix_coords = w.all_world2pix(np.array(table_gaia['ra']), np.array(table_gaia['dec']), 1)
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
    wcs_align.run_all(align_image,
          telescope='jwst',
          outsubdir=outdir,
          overwrite=True,
          d2d_max=1.0,
          find_stars_threshold = 3,
          showplots=0,
          refcatname='Gaia',
          refcat_pmflag = True, #reject proper motion stars
          histocut_order='dxdy',
          use_dq = False,
          verbose = verbose,
          iterate_with_xyshifts = True,
          xshift = xshift,
          yshift = yshift,
          sharpness_lim=(0.3,0.95),
          roundness1_lim=(-0.7, 0.7),
        #   Nbright = 1500,
          SNR_min= 5,
          dmag_max=.1,
          objmag_lim =(15,22),
        #   refmag_lim = (12,19),
          binsize_px = 1.0,
          saveplots = 0,
          slope_min = -20/2048,
          savephottable = 0)
    
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
    dispersion_initial = calc_dispersion(gaia_table, jhat_image.replace('_jhat.fits', '.phot.txt'), dist_limit = 1, plot = False)
    print(f'Initial dispersion: {dispersion_initial}"')
    # temp_cal_name = jhat_image.replace('jhat.fits', 'jhat_cal.fits')
    os.rename(jhat_image, temp_cal_name)
    refcat, photfilename = jwst_phot(temp_cal_name)
    dispersion_final = calc_dispersion(gaia_table, photfilename, phot_image = phot_image, dist_limit = 1, plot = False)
    print(f'Final dispersion: {dispersion_final}"')
    os.rename(temp_cal_name, jhat_image)

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
    wcs_align.run_all(align_image,
                      telescope='jwst',
                      outsubdir=outdir,
                      refcat_racol='ra',
                      refcat_deccol='dec',
                      refcat_magcol='mag',
                      refcat_magerrcol='dmag',
                      overwrite=True,
                      d2d_max=1.0,
                      showplots=0,
                      find_stars_threshold=5,
                      refcatname=photfilename,
                      verbose = verbose,
                      iterate_with_xyshifts = True,
                      histocut_order='dxdy',
                      sharpness_lim=(0.3,0.95),
                      roundness1_lim=(-0.7, 0.7),
                      xshift = xshift,
                      yshift = yshift,
                      SNR_min= 5,
                      dmag_max=0.1,
                      Nbright=Nbright,
                      objmag_lim =(15,22),
                      slope_min = -20/2048,
                      binsize_px = 1.0,
                      savephottable=0)
    
    jhat_image = os.path.join(outdir, os.path.basename(align_image.replace('cal.fits', 'jhat.fits')))
    #compute dispersion
    # gaia_table = query_gaia(align_image, save_file = None)
    refcat = Table.read(photfilename, format='ascii')
    dispersion_initial = calc_dispersion(refcat, jhat_image.replace('_jhat.fits', '.phot.txt'), dist_limit = 1, plot = False)
    print(f'Initial dispersion: {dispersion_initial}"')
    temp_cal_name = jhat_image.replace('jhat.fits', 'jhat_cal.fits')
    os.rename(jhat_image, temp_cal_name)
    refcat, photfilename = jwst_phot(jhat_image.replace('jhat.fits', 'jhat_cal.fits'))
    dispersion_final = calc_dispersion(refcat, photfilename, dist_limit = 1, plot = False)
    print(f'Final dispersion: {dispersion_final}"')
    os.rename(temp_cal_name, jhat_image)

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
    # temp_mosaic_name = mosaic.replace('jhat.fits', 'i2d.fits')
    # os.rename(mosaic, temp_mosaic_name)
    refcat, photfile = jwst_phot(mosaic)
    w = wcs.WCS(fits.open(mosaic)['SCI'].header)
    jh_radec = w.all_pix2world(refcat['x'], refcat['y'], 0)
    jh_ra, jh_dec = np.array(jh_radec[0]), np.array(jh_radec[1])
    refcat['ra'], refcat['dec'] = jh_ra, jh_dec
    #photfile.replace('i2d', 'i2d_new')
    refcat.write(photfile.replace('i2d', 'i2d.corr'), format = 'ascii', overwrite = True)
    # os.rename(temp_mosaic_name, mosaic)
    return photfile.replace('i2d', 'i2d.corr')

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
    outdir = os.path.join(work_dir, obj, 'nircam')
    aligndir = os.path.join(work_dir, 'align')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(aligndir):
        os.makedirs(aligndir)
    return outdir

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    work_dir = args.workdir
    obj = args.object

    dolphot_basedir = create_dirs(work_dir, obj)

    input_images = get_input_images(workdir=work_dir)
    table = input_list(input_images)
    flt_vis_dict = visit_filter_dict(table)
    tables = organize_reduction_tables(table, byvisit=True, bymodule=False)
    for tbl in tables[0]:
        filters = np.unique(tbl['filter']).value
        filter_table = create_filter_table(tbl, filters)
        # filter_table = {k: v for k, v in filter_table.items() 
        #                 if k in ['f150w', 'f187n', 'f300m', 'f335m']}
        mosaic_name = create_alignment_mosaic(filter_table, os.path.join(work_dir, 'align'), work_dir,
                                              infilter = flt_vis_dict[np.unique(tbl['visit']).value[0]])
        mosaic_photfile = fix_phot(mosaic_name)
        print(f'Mosaic photfile: {mosaic_photfile}')
        for filt, tbl in filter_table.items():
            if filt == 'f277w':
                gaia_offset = (0, 0)
            else:
                gaia_offset = (0, 0)
            align_to_mosaic(mosaic_photfile, [r['image'] for r in tbl], os.path.join(work_dir, 'jhat'), 
                            gaia_offset = gaia_offset, verbose = False)
        aligned_images = glob.glob(os.path.join(work_dir, 'jhat', f'*nrc*jhat.fits'))
        align_list = input_list(aligned_images)
        refname = generate_level3_mosaic(align_list[align_list['filter'] == 'f150w']['image'],
                                        os.path.join(work_dir, 'jhat'))
        print('Dolphot reference image:', refname)
        
        #copy files to dolphot directory
        for fl in glob.glob(os.path.join(work_dir, 'jhat', '*jhat.fits')):
            shutil.copy(fl, dolphot_basedir)

        for fl in glob.glob(os.path.join(work_dir, 'jhat', 'out', '*i2d.fits')):
            shutil.copy(fl, dolphot_basedir)

        #generate dolphot param file
        paramfiles = generate_dolphot_paramfile(dolphot_basedir)
        #switch directory to basedir
        os.chdir(dolphot_basedir)
        #run dolphot
        dolphot_images = glob.glob('*fits')
        apply_nircammask(dolphot_images)
        calc_cky(dolphot_images)
        for i, paramfile in enumerate(paramfiles):
            subprocess.run(f'dolphot {obj}_{i}.phot -p{paramfile}', shell=True)