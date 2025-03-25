import glob, os
from nbutils import input_list, create_filter_table
import argparse
import numpy as np
from jwst.pipeline import calwebb_image3

import glob,os
from astropy.io import fits
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
import shapely
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

from astropy.modeling import models
from astropy import coordinates as coord
from astropy import units as u
from gwcs import coordinate_frames as cf
from astropy import wcs
from gwcs import WCS as g_wcs
from asdf import AsdfFile

from astropy.nddata import CCDData
from astropy import nddata
from ccdproc import Combiner

import stpsf
from astropy.stats import sigma_clipped_stats as scs
from photutils.psf.matching import resize_psf, SplitCosineBellWindow, create_matching_kernel, CosineBellWindow, TukeyWindow, TopHatWindow, HanningWindow
from astropy.convolution import convolve, convolve_fft
from reproject.mosaicking import find_optimal_celestial_wcs
import subprocess

def create_parser():
    '''
    Create an argument parser

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser
    '''
    parser = argparse.ArgumentParser(description='Reduce JWST data')
    parser.add_argument('--basedir', type=str, default='.', help='Root directory to search for data')
    # parser.add_argument('--filter', type=str, help='Filter to create mosaic of')
    return parser

def create_default_mosaic(inputfiles, outdir, filt):
    '''
    Create level3 drizzled mosaic from level2 input files
    using the JWST pipeline with default options

    Parameters
    ----------
    input_files : list
        List of input level2 files
    outdir: str
        Output directory

    Returns
    -------
    None
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    table = input_list(inputfiles)
    table = table[table['filter'] == filt]
    nircam_asn_file = f'{outdir}/{filt}.json'
    base_filenames = np.array([os.path.basename(r['image']) for r in table])
    asn3 = asn_from_list.asn_from_list(base_filenames,
        rule=DMS_Level3_Base, product_name=f'{filt}')

    with open(nircam_asn_file, 'w') as outfile:
        name, serialized = asn3.dump(format='json')
        outfile.write(serialized)

    image3 = calwebb_image3.Image3Pipeline()

    outdir_level3 = os.path.join(outdir, f'out_{filt}')
    if not os.path.exists(outdir_level3):
        os.makedirs(outdir_level3)

    image3.output_dir = outdir_level3
    image3.save_results = True
    image3.tweakreg.skip = True
    image3.skymatch.skip = True
    image3.skymatch.match_down = False
    image3.source_catalog.skip=False
    image3.resample.pixfrac = 1.0
    image3.pixel_scale = 0.0311
    image3.weight_type = 'ivm'

    image3.run(nircam_asn_file)

def create_coadd_mosaic(table, outdir, filt, centroid=None, 
                        output_shape=None, gwcs_file=None):
    '''
    Create level3 drizzled mosaic from level2 input files
    using the JWST pipeline. Centroid and output_shape or gwcs_file
    can be passed in to create uniformly drizzled mosaics

    Parameters
    ----------
    table : astropy.table.table.Table
        Table containing all images to be resampled
    outdir: str
        Output directory
    filt: str
        Filter of images to be resampled
    centroid: optional, shapely.geometry.point.Point
        Centroid of the drizzled field
    output_shape: optional, tuple
        (xbound, ybound) for the drizzled field
    gwcs_file: optional, str
        Path to gwcs file to drizzle uniformly

    Returns
    -------
    filepath: str
        Path to resampled i2d image
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # table = input_list(inputfiles)
    # table = table[table['filter'] == filt]
    nircam_asn_file = f'{outdir}/{filt}.json'
    base_filenames = np.array([os.path.basename(r['image']) for r in table])
    asn3 = asn_from_list.asn_from_list(base_filenames,
        rule=DMS_Level3_Base, product_name=f'{filt}')

    with open(nircam_asn_file, 'w') as outfile:
        name, serialized = asn3.dump(format='json')
        outfile.write(serialized)

    image3 = calwebb_image3.Image3Pipeline()

    outdir_level3 = os.path.join(outdir, f'out_{filt}')
    if not os.path.exists(outdir_level3):
        os.makedirs(outdir_level3)

    image3.output_dir = outdir_level3
    image3.save_results = True
    image3.tweakreg.skip = True
    image3.skymatch.skip = True
    image3.skymatch.match_down = False
    image3.source_catalog.skip=False
    image3.resample.pixfrac = 1.0
    image3.pixel_scale = 0.0311
    image3.weight_type = 'ivm'
    if gwcs_file:
        image3.resample.output_wcs = gwcs_file

    elif centroid:
        image3.resample.crval = (centroid.x , centroid.y)
        image3.resample.output_shape = output_shape[0], output_shape[1]

    image3.run(nircam_asn_file)
    
    filepath = f'out_{filt}/{filt}_i2d.fits'
    return filepath


def create_gwcs(wcs_out, shape_out, outdir):
    '''
    Convert astropy WCS to a GWCS object and write it into an asdf file

    Parameters
    ----------
    wcs_out : astropy.wcs.WCS
        Astropy WCS to be converted to gwcs
    shape_out : tuple
        (NAXIS2, NAXIS1) shape of output mosaic

    Returns
    -------
    gwcs_path: str
        Path to the GWCS asdf file
    '''
    sci_header = wcs_out.to_header()
    sci_header['NAXIS1'] = shape_out[1]
    sci_header['NAXIS2'] = shape_out[0]

    shift_by_crpix = models.Shift(-(sci_header['CRPIX1'] - 1)) & models.Shift(-(sci_header['CRPIX2'] - 1))
    matrix = np.array([[sci_header['PC1_1'], sci_header['PC1_2']],
                    [sci_header['PC2_1'] , sci_header['PC2_2']]])
    rotation = models.AffineTransformation2D(matrix , translation=[0, 0])

    tan = models.Pix2Sky_TAN()
    pixelscale = models.Scale(sci_header['CDELT1']) & models.Scale(sci_header['CDELT2'])
    celestial_rotation =  models.RotateNative2Celestial(sci_header['CRVAL1'], sci_header['CRVAL2'], 180)

    det2sky = shift_by_crpix | rotation | pixelscale | tan | celestial_rotation
    det2sky.name = "linear_transform"

    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs',
                                unit=(u.deg, u.deg))

    pipeline = [(detector_frame, det2sky),
                (sky_frame, None)
            ]
    wcsobj = g_wcs(pipeline)
    wcsobj.bounding_box = ((0, sci_header['NAXIS1']), (0, sci_header['NAXIS2']))
    
    #write gwcs to asdf file
    tree = {"wcs": wcsobj}
    wcs_file = AsdfFile(tree)
    gwcs_path = f"{outdir}/mosaic_gwcs.asdf"
    wcs_file.write_to(gwcs_path)

    return gwcs_path

def find_optimal_wcs(filter_table):
    images = np.hstack([filter_table[i]['image'].value for i in filter_table.keys()])
    image_hdus = [fits.open(i)[1] for i in images]
    wcs_out, shape_out = find_optimal_celestial_wcs(image_hdus, auto_rotate = True)

    return wcs_out, shape_out

def create_psf_kernel(ref_filter: str, in_filter: str, ovs=5, fov=81):
    nrc = stpsf.NIRCam()
    nrc.filter = in_filter.upper()
    nrc.detector = 'NRCA3'
    psf_src = nrc.calc_psf(oversample=ovs, fov_pixels=fov) 

    #use detector distorted version
    psf_src_dat = psf_src[3].data/psf_src[3].data.sum()

    nrc.filter = ref_filter.upper()
    psf_ref = nrc.calc_psf(oversample=ovs, fov_pixels=fov)
    psf_ref_dat = psf_ref[3].data/psf_ref[3].data.sum()

    window = SplitCosineBellWindow(1.5, 1.3)
    psf_kernel = create_matching_kernel(psf_src_dat, psf_ref_dat, window=window) 

    return psf_kernel

def convolve_images(filter_table, target_filter):
    for filt in filter_table.keys():
        if filt.upper() == target_filter.upper():
            continue
        psf_kernel = create_psf_kernel(target_filter, filt)
        tbl = filter_table[filt]
        for im in tbl['image']:
            hdu = fits.open(im)
            sci, err = hdu['SCI'].data, hdu['ERR'].data
            sci_con = convolve_fft(sci, psf_kernel, normalize_kernel=True)
            err_con = convolve_fft(err, psf_kernel, normalize_kernel=True)
            sci_header = hdu['SCI'].header
            sci_header['filter'] = filt.upper()

            hdu['SCI'].header, hdu['SCI'].data = sci_header, sci_con
            hdu['ERR'].data = err_con
            hdu.writeto(im, overwrite=True)

def create_ccddata(file):
    hdu = fits.open(file)
    sci_data = hdu['SCI'].data
    
    uncertainty = nddata.StdDevUncertainty(array = hdu['ERR'].data)
    data_unit = u.MJy/u.sr
    w = wcs.WCS(hdu['SCI'].header)
    mask = sci_data == 0
    ccd_data = CCDData(data = sci_data, uncertainty = uncertainty, 
                       wcs = w, unit = data_unit)
    
    return ccd_data

def update_photmjsr(ccddata, phots):
    ccd_mjsr = np.sum([ccd.data for ccd in ccddata], axis = 0)
    ccd_cps = np.sum([ccd.data/phot for ccd, phot in list(zip(ccddata, phots))], axis = 0)
    mjsr = ccd_mjsr/ccd_cps
    _, mjsr_med, _ = scs(mjsr)

    return mjsr_med

def coadd(ref_files, filt, filename = 'coadd_i2d.fits'):
    #edit specific header keys
    hdu_template = fits.open(ref_files[0])
    hdr_update = {'EFFEXPTM': [], 'TMEASURE': [], 'DURATION': []}
    filters, phots = [], []
    #WHT data for coadded image
    wht_data = []
    
    for file in ref_files:
        hdu_ = fits.open(file)
        for key in list(hdr_update.keys()):
            hdr_update[key].append(fits.getval(file, key, ext = 0))
        flt_ = fits.getval(file, 'FILTER', ext = 0)
        filters.append(flt_)
        phots.append(fits.getval(file, 'PHOTMJSR', ext = 1))
        # #inverse variance weighting
        wht_data.append(hdu_['WHT'].data/fits.getval(file, 'DURATION', ext = 0))
        hdu_.close()

    combiner_weights = np.array(wht_data)
    combiner_weights /= np.sum(combiner_weights, axis = 0)
    
    for i, wt_ in enumerate(combiner_weights):
        mask_ = np.isnan(wt_) | np.isinf(wt_)
        wt_[mask_] = 0
        combiner_weights[i] = wt_ 
    combiner_weights = np.array(combiner_weights)
    
    #coadd images using ccdproc
    ccddata_ = []
    for file in ref_files:
        ccddata_.append(create_ccddata(file))
        
    combiner = Combiner(ccddata_)
    combiner.weights = combiner_weights
    combined_sum = combiner.sum_combine()

    #SCI and ERR data for coadded image
    coadd_data = combined_sum.data
    det_mask = coadd_data == 0
    quad_err = np.sqrt(np.sum([(wht_*ccd.uncertainty.array)**2 for ccd, wht_ in zip(ccddata_, combiner_weights)], axis = 0))
        
    primary_header, sci_header = hdu_template['PRIMARY'].header, hdu_template['SCI'].header
    err_header, wht_header = hdu_template['ERR'].header, hdu_template['WHT'].header
    primary_header['FILENAME'] = filename
    primary_header['FILTER'] = filt.upper()
    
    exptime_wt = [np.nanmean(i) for i in combiner_weights]
    for key in list(hdr_update.keys()):
        hdr_update[key] = np.sum(hdr_update[key])
        primary_header[key] = hdr_update[key]

    sci_header['PHOTMJSR'] = update_photmjsr(ccddata_, phots)
    sci_header['XPOSURE'] = hdr_update['EFFEXPTM']
    sci_header['TELAPSE'] = hdr_update['DURATION']

    primary_hdu = fits.PrimaryHDU(header = primary_header)
    sci_hdu = fits.ImageHDU(data = coadd_data, header = sci_header, name = 'SCI')
    err_hdu = fits.ImageHDU(data = quad_err, header = err_header, name = 'ERR')
    wht_hdu = fits.ImageHDU(data = np.sum(wht_data, axis = 0), header = wht_header, name = 'WHT')
    
    coadd_hdul = fits.HDUList([primary_hdu, sci_hdu, err_hdu, wht_hdu])
    coadd_hdul.writeto(filename, overwrite = True)
    hdu_template.close()

def copy_files(filter_table, outdir):
    infiles = np.hstack([filter_table[i]['image'].value for i in filter_table.keys()])
    for file in infiles:
        shutil.copy(file, outdir)

def update_path(filter_table, outdir):
    for flt in filter_table.keys():
        tbl = filter_table[flt]
        filenames = [os.path.basename(i['image']) for i in tbl]
        tbl['image'] = [os.path.join(outdir, i) for i in filenames]
        filter_table[flt] = tbl
    
    return filter_table

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    base_dir = args.basedir
    # filt = args.filter
    filt = ['f115w', 'f150w', 'f200w']

    inputfiles = glob.glob(os.path.join(base_dir, '*jhat.fits'))
    table = input_list(inputfiles)
    filters = np.unique(table['filter']).value
    filter_table = create_filter_table(table, filters)
    filter_table = {k: v for k, v in filter_table.items() 
                    if k in filt}
    filter_keys = sorted(filter_table.keys())
    target_filter = filter_keys[-1]
    
    outdir = os.path.join(os.path.join(Path(base_dir).parent, 'reference'))
    copy_files(filter_table, outdir)
    filter_table = update_path(filter_table, outdir)
    base_dir = outdir
    
    wcs_out, shape_out = find_optimal_wcs(filter_table)
    gwcs_path = create_gwcs(wcs_out=wcs_out, shape_out=shape_out, outdir=base_dir)

    convolve_images(filter_table, target_filter)

    mosaics = []
    for flt_key in filter_keys:
        driz_image = create_coadd_mosaic(filter_table[flt_key], outdir=base_dir, filt=flt_key, 
                                        centroid=None, output_shape=None, gwcs_file=gwcs_path)
        mosaics.append(driz_image)
    mosaics = [os.path.join(base_dir, i) for i in mosaics]

    coadd_filename = os.path.join(base_dir, 'coadd_i2d.fits')
    coadd(mosaics, target_filter, coadd_filename)