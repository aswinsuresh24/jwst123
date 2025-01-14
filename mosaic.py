import glob, os
from nbutils import input_list
# from jwst123 import *
import argparse
import numpy as np
from jwst.pipeline import calwebb_image3

import glob,os
import jwst
from astropy.io import fits
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
import shapely
import matplotlib.pyplot as plt

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
    parser.add_argument('--filter', type=str, help='Filter to create mosaic of')
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

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    base_dir = args.basedir
    filt = args.filter

    inputfiles = glob.glob(os.path.join(base_dir, '*jhat.fits'))
    create_default_mosaic(inputfiles, base_dir, filt)