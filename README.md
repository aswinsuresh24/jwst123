# jwst123

Tools for downloading JWST/NIRCam imaging from MAST, aligning frames with
[JHAT](https://jhat.readthedocs.io/), building mosaics, and preparing DOLPHOT
runs. The repository also retains a legacy HST reduction script (`hst123.py`).

## Repository layout

| Path | Role |
| --- | --- |
| `jwst_download.py` | Download JWST NIRCam products from MAST (`--token`, `--outdir`) |
| `jwst123.py` | Align CAL frames with JHAT / Gaia and build Level-3 mosaics |
| `mosaic.py` | Mosaic / coadd / PSF-matching helpers and DOLPHOT prep |
| `nbutils.py` | Shared FITS bookkeeping (filters, visits, obstables) |
| `common/mast.py` | Shared MAST query, auth, product filtering, and download helpers |
| `link.py` | Symlink downloaded FITS into a reduction `raw/` directory |
| `nircam_settings.py` | JHAT alignment parameter sets |
| `scripts/jwst_relative_align.py` | Relative JHAT alignment of one image to a reference (+ diagnostic plots) |
| `hst123.py` | Legacy HST download / drizzle / DOLPHOT pipeline |
| `notebooks/` | Streamlined notebooks for MAST download, Gaia alignment, mosaics |
| `environment.yaml` | Recommended conda environment |
| `pyproject.toml` | Package metadata and editable install |

## Requirements

- **Python 3.11 or 3.12** (3.11 recommended)
- External **DOLPHOT** binaries if you run PSF photometry
  ([DOLPHOT](http://americano.dolphinsim.com/dolphot/))

Pinned `requirements.txt` versions target an older Python 3.10 stack and are
**not** recommended for new installs (they fail on Python 3.12 when building
`astropy==5.3.3`). Prefer `environment.yaml` / `pyproject.toml` below.

## Installation

### Recommended: conda + editable install

From the repository root:

```bash
conda env create -f environment.yaml
conda activate jwst123
pip install -e .
```

This creates a Python 3.11 environment with conda-forge binaries (including
HDF5 / blosc / pytables for drizzlepac) and installs the remaining STScI
packages (`jwst`, `jhat`, `drizzlepac`, …) via pip.

### Alternative: existing conda/venv + pip

```bash
conda create -n jwst123 python=3.11 pip hdf5 blosc pytables
conda activate jwst123
pip install -e .
```

If `tables` / drizzlepac fails to find HDF5 on macOS Homebrew:

```bash
export HDF5_DIR="$(brew --prefix hdf5)"
export BLOSC_DIR="$(brew --prefix c-blosc)"
pip install -e .
```

### Verify

```bash
python -c "import jwst, jhat, drizzlepac; print(jwst.__version__)"
jwst-download --help
```

## Quick start: JWST download

Download stage-2 `*_cal.fits` products near a target (public data by default):

```bash
python jwst_download.py \
  --ra "10:38:47.961" --dec "+53:30:34.10" \
  --obj NGC3310 \
  --outdir /path/to/NGC3310
```

For proprietary data, pass a MAST API token
([create one here](https://auth.mast.stsci.edu/info)):

```bash
python jwst_download.py \
  --ra 189.9976 --dec -11.623 \
  --obj NGC4536 \
  --outdir /path/to/NGC4536 \
  --token YOUR_MAST_TOKEN
```

Useful options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--outdir` | `jwst_data/<obj>` | Output directory |
| `--radius` | `3.0` | Search radius in arcminutes |
| `--stage` | `2` | `2` = CAL, `3` = I2D mosaics |
| `--token` | none | MAST auth for exclusive-access data |

After download, symlink FITS into a reduction tree:

```bash
python link.py --datadir /path/to/NGC3310 --symlinkdir /path/to/reduction
```

## JWST alignment and mosaics

Place CAL files under `<workdir>/raw/`, then:

```bash
python jwst123.py --workdir /path/to/reduction --object NGC3310 --ncores 4
```

Relative alignment of one frame to a reference image (writes JHAT diagnostics
under `--outdir`):

```bash
python scripts/jwst_relative_align.py \
  --ref /path/to/reference_i2d.fits \
  --align /path/to/target_cal.fits \
  --outdir /path/to/aligned
```

Interactive workflows live in:

- `notebooks/hst_download.ipynb` — MAST query / coverage / download
- `notebooks/jwst_gaia_align.ipynb` — JHAT + Gaia alignment
- `notebooks/jwst_mosaic.ipynb` — Level-3 mosaics, coadds, PSF matching

## Legacy HST pipeline (`hst123.py`)

`hst123.py` remains available for HST download, tweakreg/drizzle, DOLPHOT, and
catalog scraping. Typical usage:

```bash
python hst123.py <ra> <dec> --download --token YOUR_MAST_TOKEN
```

Supported HST products:

```
WFPC2: c0m.fits, c1m.fits (requires both)
ACS/WFC: flc.fits
ACS/HRC: flt.fits
WFC3/UVIS: flc.fits
WFC3/IR: flt.fits
```

Run `python hst123.py --help` for the full option list.

## External dependencies

DOLPHOT (including instrument modules and filter PSFs) is required for
`--run-dolphot` / mosaic DOLPHOT prep:
http://americano.dolphinsim.com/dolphot/

## Contact

Questions, bugs, and suggestions: Charlie Kilpatrick
(ckilpatrick@northwestern.edu).
