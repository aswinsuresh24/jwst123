import warnings
warnings.filterwarnings('ignore')
import glob
import sys
import os
import argparse
import astropy.wcs as wcs
import numpy as np
from contextlib import contextmanager
from astropy import units as u

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
    from astroquery.mast import Observations
    from astropy.coordinates import SkyCoord

def create_parser():
    '''
    Create a parser for the command line arguments

    Returns:
    -------
    parser : argparse.ArgumentParser
        arg parser
    '''
    parser = argparse.ArgumentParser(description='Download JWST data')
    parser.add_argument('--ra', type=str, help='RA of the target', required=True)
    parser.add_argument('--dec', type=str, help='DEC of the target', required=True)
    parser.add_argument('--obj', type=str, help='Name of the object', required=True)  
    parser.add_argument('--radius', type=float, default = 3.0, help='Radius in arcminutes')
    parser.add_argument('--stage', type=int, default = 2, help='Stage of the reduction')
    return parser

def query_mast_jwst(coord):
    '''
    Download available data from MAST for JWST NIRCAM

    Parameters:
    ----------
    coord : astropy.coordinates.SkyCoord
        target coordinates

    Returns:
    -------
    None
    '''
    obsTable = Observations.query_region(coord, radius=radius)
    obsTable = obsTable.filled()

    #obsTable masks
    masks = []
    masks.append([t.upper()=='JWST' for t in obsTable['obs_collection']]) #JWST images
    masks.append([any(l) for l in list(map(list,zip(*[[det in inst.upper() #NIRCAM images
                for inst in obsTable['instrument_name']]
                for det in ['NIRCAM']])))])

    #add mask to remove entries with 1. null jpegURL/dataURL 2. private data rights(?)
    # Added mask to remove calibration data from search
    masks.append([f.upper()!='DETECTION' for f in obsTable['filters']])
    masks.append([i.upper()!='CALIBRATION' for i in obsTable['intentType']])

    mask = [all(l) for l in list(map(list, zip(*masks)))]
    obsTable_webb = obsTable[mask]

    for obs in obsTable_webb:
        filt, obsid = obs['filters'], obs['obsid']
        filt = filt.replace(';', '_')
        productList = Observations.get_product_list(obs)
        #product list masks
        productmasks = []
        productmasks.append([p.upper() == 'SCIENCE' for p in productList['productType']])
        if stage == 2:
            productmasks.append([t == 'CAL' for t in productList['productSubGroupDescription']])
            productmasks.append([c == 2 for c in productList['calib_level']])
        if stage == 3:
            productmasks.append([t == 'I2D' for t in productList['productSubGroupDescription']])
            productmasks.append([c == 3 for c in productList['calib_level']])


        productmask = [all(l) for l in list(map(list, zip(*productmasks)))]
        productList = productList[productmask]
        os.makedirs(f'jwst_data/{obj}/{filt}_{obsid}', exist_ok=True)
        download_dir = f'jwst_data/{obj}/{filt}_{obsid}'
        Observations.download_products(productList, download_dir=download_dir, extension='fits')

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ra, dec, radius = args.ra, args.dec, args.radius*u.arcmin
    obj = args.obj
    stage = args.stage

    os.makedirs(f'jwst_data/{obj}', exist_ok=True)

    coord = SkyCoord(ra, dec, frame = 'icrs', unit = 'deg')
    query_mast_jwst(coord)