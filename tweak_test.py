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

acceptable_filters = [
    'F220W','F250W','F330W','F344N','F435W','F475W','F550M','F555W',
    'F606W','F625W','F660N','F660N','F775W','F814W','F850LP','F892N',
    'F098M','F105W','F110W','F125W','F126N','F127M','F128N','F130N','F132N',
    'F139M','F140W','F153M','F160W','F164N','F167N','F200LP','F218W','F225W',
    'F275W','F280N','F300X','F336W','F343N','F350LP','F373N','F390M','F390W',
    'F395N','F410M','F438W','F467M','F469N','F475X','F487N','F547M',
    'F600LP','F621M','F625W','F631N','F645N','F656N','F657N','F658N','F665N',
    'F673N','F680N','F689M','F763M','F845M','F953N','F122M','F160BW','F185W',
    'F218W','F255W','F300W','F375N','F380W','F390N','F437N','F439W','F450W',
    'F569W','F588N','F622W','F631N','F673N','F675W','F702W','F785LP','F791W',
    'F953N','F1042M']

catalog_pars = {
    'skysigma':0.0,
    'computesig':True,
    'conv_width':3.5,
    'sharplo':0.2,
    'sharphi':1.0,
    'roundlo':-1.0,
    'roundhi':1.0,
    'peakmin':None,
    'peakmax':None,
    'fluxmin':None,
    'fluxmax':None,
    'nsigma':1.5,
    'ratio':1.0,
    'theta':0.0,
    'use_sharp_round': True,
    'expand_refcat': False,
    'enforce_user_order': True,
    'clean': True,
    'interactive': False,
    'verbose': False,
    'updatewcs': False,
    'xyunits': 'pixels',
    '_RULES_': {'_rule_1':'True', '_rule2_':'False'}
}

workdir = '.'
input_images = [s for p in ['*flc.fits', '*flt.fits'] for s in glob.glob(os.path.join(workdir, p))]

obstable = input_list(input_images)[:12]
print(obstable)

def tweakreg_error(exception):
    message = '\n\n' + '#'*80 + '\n'
    message += 'WARNING: tweakreg failed: {e}\n'
    message += '#'*80 + '\n'
    print(message.format(e=exception.__class__.__name__))
    print('Error:', exception)
    print('Adjusting thresholds and images...')

# Apply TWEAKSUC header variable if tweakreg was successful
def apply_tweakreg_success(shifts):

    for row in shifts:
        if ~np.isnan(row['xoffset']) and ~np.isnan(row['yoffset']):
            file=row['file']
            if not os.path.exists(file):
                file=row['file']
                print(f'WARNING: {file} does not exist!')
                continue
            hdu = fits.open(file, mode='update')
            hdu[0].header['TWEAKSUC']=1
            hdu.close()

def check_images_for_tweakreg(run_images):

    if not run_images:
        return(None)

    images = copy.copy(run_images)

    for file in list(images):
        print('Checking {0} for TWEAKSUC=1'.format(file))
        hdu = fits.open(file, mode='readonly')
        remove_image = ('TWEAKSUC' in hdu[0].header.keys() and
            hdu[0].header['TWEAKSUC']==1)

        if remove_image:
            images.remove(file)

    # If run_images is now empty, return None instead
    if len(images)==0:
        return(None)

    return(images)

def fix_hdu_wcs_keys(image, change_keys, ref_url):

    hdu = fits.open(image, mode='update')
    ref = ref_url.strip('.old')
    outdir = '.'
    if not outdir:
        outdir = '.'

    for i,h in enumerate(hdu):
        for key in hdu[i].header.keys():
            if 'WCSNAME' in key:
                print('WCSNAME', hdu[i].header[key])
                hdu[i].header[key] = hdu[i].header[key].strip()
        for key in change_keys:
            if key in list(hdu[i].header.keys()):
                val = hdu[i].header[key]
                print(key, val)
            else:
                continue
            if val == 'N/A':
                continue
            if (ref+'$' in val):
                ref_file = val.split('$')[1]
            else:
                ref_file = val

            fullfile = os.path.join(outdir, ref_file)
            if not os.path.exists(fullfile):
                print(f'Grabbing: {fullfile}')
                # Try using both old cdbs database and new crds link
                urls = []
                url = 'https://hst-crds.stsci.edu/unchecked_get/references/hst/'
                urls.append(url+ref_file)

                url = 'ftp://ftp.stsci.edu/cdbs/'
                urls.append(url+ref_url+'/'+ref_file)

                for url in urls:
                    message = f'Downloading file: {url}'
                    sys.stdout.write(message)
                    sys.stdout.flush()
                    try:
                        dat = download_file(url, cache=False,
                            show_progress=False, timeout=120)
                        shutil.move(dat, fullfile)
                        message = '\r' + message
                        message += Constants.green+' [SUCCESS]'+Constants.end+'\n'
                        sys.stdout.write(message)
                        break
                    except:
                        message = '\r' + message
                        message += Constants.red+' [FAILURE]'+Constants.end+'\n'
                        sys.stdout.write(message)
                        print(message)

            message = f'Setting {image},{i} {key}={fullfile}'
            print(message)
            hdu[i].header[key] = fullfile

        # WFPC2 does not have residual distortion corrections and astrodrizzle
        # choke if DGEOFILE is in header but not NPOLFILE.  So do a final check
        # for this part of the WCS keys
        if 'wfpc2' in get_instrument(image).lower():
            keys = list(h.header.keys())
            if 'DGEOFILE' in keys and 'NPOLFILE' not in keys:
                del hdu[i].header['DGEOFILE']

    hdu.writeto(image, overwrite=True, output_verify='silentfix')
    hdu.close()

def fix_idcscale(image):

    det = '_'.join(get_instrument(image).split('_')[:2])

    if 'wfc3' in det:
        hdu = fits.open(image)
        idcscale = 0.1282500028610229
        for i,h in enumerate(hdu):
            if 'IDCSCALE' not in hdu[i].header.keys():
                hdu[i].header['IDCSCALE']=idcscale

        hdu.writeto(image, overwrite=True, output_verify='silentfix')

# Update image wcs using updatewcs routine
def update_image_wcs(image, use_db=True):

    hdu = fits.open(image, mode='readonly')
    # Check if tweakreg was successfully run.  If so, then skip
    if 'TWEAKSUC' in hdu[0].header.keys() and hdu[0].header['TWEAKSUC']==1:
        return(True)

    # Check for hierarchical alignment.  If image has been shifted with
    # hierarchical alignment, we don't want to shift it again
    if 'HIERARCH' in hdu[0].header.keys() and hdu[0].header['HIERARCH']==1:
        return(True)

    hdu.close()

    message = 'Updating WCS for {file}'
    print(message.format(file=image))

    change_keys = ['IDCTAB','DGEOFILE','NPOLEXT','NPOLFILE','D2IMFILE', 'D2IMEXT','OFFTAB']
    inst = get_instrument(image).split('_')[0]
    ref_url = 'jref.old'

    fix_hdu_wcs_keys(image, change_keys, ref_url)

    # Usually if updatewcs fails, that means it's already been done
    try:
        updatewcs.updatewcs(image, use_db=use_db) #probably not needed for jwst
        hdu = fits.open(image, mode='update')
        message = '\n\nupdatewcs success.  File info:'
        print(message)
        hdu.info()
        hdu.close()
        fix_hdu_wcs_keys(image, change_keys, ref_url)
        fix_idcscale(image)
        return(True)
    except:
        error = 'ERROR: failed to update WCS for image {file}'
        print(error.format(file=image))
        return(None)
    
def run_cosmic(image, options, output=None):
    message = 'Cleaning cosmic rays in image: {image}'
    print(message.format(image=image))
    hdulist = fits.open(image,mode='readonly')

    if output is None:
        output = image

    for i,hdu in enumerate(hdulist):
        if hdu.name=='SCI':
            mask = np.zeros(hdu.data.shape, dtype=np.bool_)

            crmask, crclean = detect_cosmics(hdu.data.copy().astype('<f4'),
                inmask=mask, readnoise=options['rdnoise'], gain=options['gain'],
                satlevel=options['saturate'], sigclip=options['sig_clip'],
                sigfrac=options['sig_frac'], objlim=options['obj_lim'])

            hdulist[i].data[:,:] = crclean[:,:]

            # Add crmask data to DQ array or DQ image
            if False:
                if 'flc' in image or 'flt' in image:
                    # Assume this hdu is corresponding DQ array
                    if len(hdulist)>=i+2 and hdulist[i+2].name=='DQ':
                        hdulist[i+2].data[np.where(crmask)]=4096
                elif 'c0m' in image:
                    maskfile = image.split('_')[0]+'_c1m.fits'
                    if os.path.exists(maskfile):
                        maskhdu = fits.open(maskfile)
                        maskhdu[i].data[np.where(crmask)]=4096
                        maskhdu.writeto(maskfile, overwrite=True)

    # This writes in place
    hdulist.writeto(output, overwrite=True, output_verify='silentfix')
    hdulist.close()

def get_nsources(image, thresh):
    imghdu = fits.open(image)
    nsources = 0
    message = '\n\nGetting number of sources in {im} at threshold={thresh}'
    print(message.format(im=image, thresh=thresh))
    for i,h in enumerate(imghdu):
        if h.name=='SCI' or (len(imghdu)==1 and h.name=='PRIMARY'):
            filename="{:s}[{:d}]".format(image, i)
            wcs = stwcs.wcsutil.HSTWCS(filename)
            catalog_mode = 'automatic'
            catalog = catalogs.generateCatalog(wcs, mode=catalog_mode,
                catalog=filename, threshold=thresh,
                **catalog_pars)
            try:
                catalog.buildCatalogs()
                nsources += catalog.num_objects
            except:
                pass

    message = 'Got {n} total sources'
    print(message.format(n=nsources))

    return(nsources)

def count_nsources(images):
    cat_str = '_sci*_xy_catalog.coo'
    # Tag cat files with the threshold so we can reference it later
    n = 0
    for image in images:
        for catalog in glob.glob(image.replace('.fits',cat_str)):
            with open(catalog, 'r+') as f:
                for line in f:
                    if 'threshold' not in line:
                        n += 1

    return(n)

# Given an input image, look for a matching catalog and estimate what the
# threshold should be for this image.  If no catalog exists, generate one
# on the fly and estimate threshold
def get_tweakreg_thresholds(image, target):

    message = 'Getting tweakreg threshold for {im}.  Target nobj={t}'
    print(message.format(im=image, t=target))

    inp_data = []
    # Cascade down in S/N threshold until we exceed the target number of objs
    for t in np.flip([3.0,4.0,5.0,6.0,8.0,10.0,15.0,20.0,25.0,30.0,40.0,80.0]):
        nobj = get_nsources(image, t)
        # If no data yet, just add and continue
        if len(inp_data)<3:
            inp_data.append((float(nobj), float(t)))
        # If we're going backward - i.e., more objects than last run, then
        # just break
        elif nobj < inp_data[-1][0]:
            break
        else:
            # Otherwise, add the data and if we've already hit the target then
            # break
            inp_data.append((float(nobj), float(t)))
            if nobj > target: break

    return(inp_data)

def add_thresh_data(thresh_data, image, inp_data):
    if not thresh_data:
        keys = []
        data = []
        for val in inp_data:
            keys.append('%2.1f'%float(val[1]))
            data.append([val[0]])

        keys.insert(0, 'file')
        data.insert(0, [image])

        thresh_data = Table(data, names=keys)
        return(thresh_data)

    keys = []
    data = []
    for val in inp_data:
        key = '%2.1f'%float(val[1])
        keys.append(key)
        data.append(float(val[0]))
        if key not in thresh_data.keys():
            thresh_data.add_column(Column([np.nan]*len(thresh_data),
                name=key))

    keys.insert(0, 'file')
    data.insert(0, image)

    # Recast as table to prevent complaint aobut thresh_data.keys()
    thresh_data = Table(thresh_data)

    for key in thresh_data.keys():
        if key not in keys:
            data.append(np.nan)

    thresh_data.add_row(data)
    return(thresh_data)

def get_best_tweakreg_threshold(thresh_data, target):

    thresh = []
    nsources = []
    thresh_data = Table(thresh_data)
    for key in thresh_data.keys():
        if key=='file': continue
        thresh.append(float(key))
        nsources.append(float(thresh_data[key]))

    thresh = np.array(thresh)
    nsources = np.array(nsources)

    mask = (~np.isnan(thresh)) & (~np.isnan(nsources))
    thresh = thresh[mask]
    nsources = nsources[mask]

    # Interpolate the data and check what S/N target we want to get obj number
    thresh_func = interp1d(nsources, thresh, kind='linear', bounds_error=False,
        fill_value='extrapolate')
    threshold = thresh_func(target)

    # Set minimum and maximum threshold
    if threshold<3.0: threshold=3.0
    if threshold>1000.0: threshold=1000.0

    message = 'Using threshold: {t}'
    print(message.format(t=threshold))

    return(threshold)

outdir = '.'
reference = ''
run_images = list(obstable['image'])
shift_table = Table([run_images, [np.nan]*len(run_images), 
                     [np.nan]*len(run_images)], names = ('file', 'xoffset', 'yoffset'))
tmp_images = []

# runs cosmic ray correction, probably not needed for jwst/nircam
for image in run_images:
        rawtmp = image.replace('.fits','.rawtmp.fits')
        tmp_images.append(rawtmp)
        
        # Check if rawtmp already exists
        if os.path.exists(rawtmp):
            message = '{file} exists. Skipping...'
            print(message.format(file=rawtmp))
            continue
        
        # Copy the raw data into a temporary file
        shutil.copyfile(image, rawtmp)

        # Clean cosmic rays so they aren't used for alignment
        inst = get_instrument(image).split('_')[0]
        crpars = {'rdnoise': 6.5,
                  'gain': 1.0,
                  'saturate': 70000.0,
                  'sig_clip': 3.0,
                  'sig_frac': 0.1,
                  'obj_lim': 5.0}
        run_cosmic(rawtmp, crpars)

modified = False
ref_images = pick_deepest_images(tmp_images)
deepest = sorted(ref_images, key=lambda im: fits.getval(im, 'EXPTIME'))[-1]
if (not reference or reference=='dummy.fits'):
    reference = 'dummy.fits'
    message = 'Copying {deep} to reference dummy.fits'
    print(message.format(deep=deepest))
    shutil.copyfile(deepest, reference)
else:
    modified = True

message = 'Tweakreg is executing...'
print(message)

start_tweak = time.time()

tweakreg_success = False
tweak_img = copy.copy(tmp_images)
ithresh = 10 ; rthresh = 10
shallow_img = []
thresh_data = None
tries = 0
nbr = 4000

while (not tweakreg_success and tries < 10):
    tweak_img = check_images_for_tweakreg(tweak_img)
    if not tweak_img: break
    if tweak_img:
        # Remove images from tweak_img if they are too shallow
        if shallow_img:
            for img in shallow_img:
                if img in tweak_img:
                    tweak_img.remove(img)

        if len(tweak_img)==0:
            error = 'ERROR: removed all images as shallow'
            print(error)
            tweak_img = copy.copy(tmp_images)
            tweak_img = check_images_for_tweakreg(tweak_img)

        # If we've tried multiple runs and there are images in input
        # list with TWEAKSUC and reference image=dummy.fits, we might need
        # to try a different reference image
        success = list(set(tmp_images) ^ set(tweak_img))
        if tries > 1 and reference=='dummy.fits' and len(success)>0:
            # Make random success image new dummy image
            n = len(success)-1
            shutil.copyfile(success[random.randint(0,n)],'dummy.fits')

        # This estimates what the input threshold should be and cuts
        # out images based on number of detected sources from previous
        # rounds of tweakreg
        message = '\n\nReference image: {ref} \n'
        message += 'Images: {im}'
        print(message.format(ref=reference, im=','.join(tweak_img)))

        # Get deepest image and use threshold from that
        deepest = sorted(tweak_img,
            key=lambda im: fits.getval(im, 'EXPTIME'))[-1]

        if not thresh_data or deepest not in thresh_data['file']:
            inp_data = get_tweakreg_thresholds(deepest, nbr*4)
            thresh_data = add_thresh_data(thresh_data, deepest, inp_data)
        mask = thresh_data['file']==deepest
        inp_thresh = thresh_data[mask][0]
        print('Getting image threshold...')
        new_ithresh = get_best_tweakreg_threshold(inp_thresh, nbr*4)

        if not thresh_data or reference not in thresh_data['file']:
            inp_data = get_tweakreg_thresholds(reference, nbr*4)
            thresh_data = add_thresh_data(thresh_data, reference, inp_data)
        mask = thresh_data['file']==reference
        inp_thresh = thresh_data[mask][0]
        print('Getting reference threshold...')
        new_rthresh = get_best_tweakreg_threshold(inp_thresh, nbr*4)

        if not rthresh: rthresh = 10
        if not ithresh: ithresh = 10

        # Other input options
        nbright = nbr
        minobj = 10
        search_rad = int(np.round(1.0))
        #if search_radius: search_rad = search_radius

        rconv = 3.5 ; iconv = 3.5 ; tol = 0.25
        if 'wfc3_ir' in get_instrument(reference):
            rconv = 2.5
        if all(['wfc3_ir' in get_instrument(i)
            for i in tweak_img]):
            iconv = 2.5 ; tol = 0.6
        if 'wfpc2' in get_instrument(reference):
            rconv = 2.5
        if all(['wfpc2' in get_instrument(i)
            for i in tweak_img]):
            iconv = 2.5 ; tol = 0.5


        # Don't want to keep trying same thing over and over
        if (new_ithresh>=ithresh or new_rthresh>=rthresh) and tries>1:
            # Decrease the threshold and increase tolerance
            message = 'Decreasing threshold and increasing tolerance...'
            print(message)
            ithresh = np.max([new_ithresh*(0.95**tries), 3.0])
            rthresh = np.max([new_rthresh*(0.95**tries), 3.0])
            tol = tol * 1.3**tries
            search_rad = search_rad * 1.2**tries
        else:
            ithresh = new_ithresh
            rthresh = new_rthresh

        if tries > 7:
            minobj = 7

        message = '\nAdjusting thresholds:\n'
        message += 'Reference threshold={rthresh}\n'
        message += 'Image threshold={ithresh}\n'
        message += 'Tolerance={tol}\n'
        message += 'Search radius={rad}\n'
        print(message.format(ithresh='%2.4f'%ithresh,
            rthresh='%2.4f'%rthresh, tol='%2.4f'%tol,
            rad='%2.4f'%search_rad))

        outshifts = os.path.join(outdir, 'drizzle_shifts.txt')

        try:
            tweakreg.TweakReg(files=tweak_img, refimage=reference,
                verbose=False, interactive=False, clean=True,
                writecat=True, updatehdr=True, reusename=True,
                rfluxunits='counts', minobj=minobj, wcsname='TWEAK',
                searchrad=search_rad, searchunits='arcseconds', runfile='',
                tolerance=tol, refnbright=nbright, nbright=nbright,
                separation=0.5, residplot='No plot', see2dplot=False,
                fitgeometry='shift',
                imagefindcfg = {'threshold': ithresh,
                    'conv_width': iconv, 'use_sharp_round': True},
                refimagefindcfg = {'threshold': rthresh,
                    'conv_width': rconv, 'use_sharp_round': True},
                shiftfile=True, outshifts=outshifts)

            # Reset shallow_img list
            shallow_img = []

        except AssertionError as e:
            tweakreg_error(e)

            message = 'Re-running tweakreg with shallow images removed:'
            print(message)
            for img in tweak_img:
                nsources = get_nsources(img, ithresh)
                if nsources < 1000:
                    shallow_img.append(img)

        # Occurs when all images fail alignment
        except TypeError as e:
            tweakreg_error(e)

        # Record what the shifts are for each of the files run
        message='Reading in shift file: {file}'
        print(message.format(file=outshifts))
        shifts = Table.read(outshifts, format='ascii', names=('file',
            'xoffset','yoffset','rotation1','rotation2','scale1','scale2'))

        apply_tweakreg_success(shifts)

        # Add data from output shiftfile to shift_table
        for row in shifts:
            filename = os.path.basename(row['file'])
            filename = filename.replace('.rawtmp.fits','')
            filename = filename.replace('.fits','')

            idx = [i for i,row in enumerate(shift_table)
                if filename in row['file']]

            if len(idx)==1:
                shift_table[idx[0]]['xoffset']=row['xoffset']
                shift_table[idx[0]]['yoffset']=row['yoffset']

        if not check_images_for_tweakreg(tmp_images):
            tweakreg_success = True

        tries += 1

message = 'Tweakreg took {time} seconds to execute.\n\n'
print(message.format(time = time.time()-start_tweak))

print(shift_table)