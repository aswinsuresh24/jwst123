# jwst123

An all-in-one script for downloading, registering, and drizzling JWST images, running dolphot, and scraping data from dolphot catalogs.  This script is optimized to obtain photometry of point sources across multiple JWST images.

## Installation

### Mac OS

It is easiest to install jwst123 dependencies using conda and pip:

```
conda create -n jwstphot python=3.11 astropy pip astroquery numpy shapely requests scipy jhat
conda activate jwstphot
pip install stpsf
```

### Linux

Follow the same instructions above with:

```
conda create -n jwstphot python=3.11 astropy pip astroquery numpy shapely requests scipy jhat
conda activate jwstphot
pip install stpsf
```

## Description

jwst123 is designed to be run in two steps in a working directory that contains your images: image alignment using `jwst123.py` and image coaddition using `mosaic.py`. 

If you want to download public archival images for a particular target, use `jwst_download.py` with the coordinates and search radius.

Currently, the script is capable of reducing JWST/NIRCam imaging (```cal.fits```). Support for JWST/MIRI will be added in the future.

## Options

```
usage: 

Image alignment
`jwst123.py --ra [ra] --dec [dec] --radius [radius] --object [object] --ncores [ncores]`

options:
  --ra                 Right ascension of target coordinates
  --dec                Declination of target coordinates
  --radius             Search radius for JWST observations
  --object             Target name 
  --ncores             Parallelize image alignment over multiple cores

Image co-addition and DOLPHOT setup
`mosaic.py --basedir [basedir] --object [object] --nmax [nmax] --spec_groups [spec_groups] --drizzle_all`

options:
  --basedir            Root directory to search for aligned images
  --object             Target name
  --nmax               Maximum number of images in a single dolphot run
  --spec_groups        List of images to be grouped together (this forces certain images to be in the same group)
  --drizzle_all        Create mosaics in all filters?
```

DOLPHOT can be run using `dolphot -pdolphot.param` and will produce data products in the directories set up by `mosaic.py`

## External dependencies

jwst123 requires a complete installation of dolphot to run PSF photometry, including all instrument-specific modules and filter PSFs.  To obtain these files, visit: http://americano.dolphinsim.com/dolphot/. 

## Contact

For all questions, comments, suggestions, and bugs related to this script, please contact Aswin Suresh at aswinsuresh2029@u.northwestern.edu or Charlie Kilpatrick at ckilpatrick@northwestern.edu.
