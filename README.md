# jwst123

Tools for downloading JWST imaging from MAST, aligning frames with
[JHAT](https://jhat.readthedocs.io/), building mosaics / coadds, and preparing
DOLPHOT runs.

## Repository layout

| Path | Role |
| --- | --- |
| `jwst123/` | Installable package (library, scripts, and notebooks) |
| `jwst123/scripts/` | Command-line entry points |
| `jwst123/notebooks/` | Generic notebooks for download, alignment, and mosaics |
| `pyproject.toml` | Package metadata; dependencies loaded from `requirements.txt` |
| `requirements.txt` | Pinned Python dependencies |

### Package modules

| Module | Role |
| --- | --- |
| `jwst123.align` | JHAT / Gaia alignment and visit grouping |
| `jwst123.mosaic` | Overlap splitting, PSF matching, coadds, GWCS, DOLPHOT prep |
| `jwst123.download` / `jwst123.mast` | MAST query and download helpers |
| `jwst123.utils` | FITS bookkeeping (filters, visits, obstables, xmatch) |
| `jwst123.settings` | JHAT and DOLPHOT parameter sets |
| `jwst123.illuminated_s_region` | Illuminated footprint / `S_REGION` from science+DQ |
| `jwst123.image_overlap` | MIRI vs reference footprint overlap |

### Scripts

| Script | Role |
| --- | --- |
| `jwst123/scripts/download.py` | Download JWST products from MAST |
| `jwst123/scripts/align.py` | Group / visit JHAT alignment pipeline |
| `jwst123/scripts/relative_align.py` | Align one image to a reference catalog (+ dispersion) |
| `jwst123/scripts/mosaic.py` | Mosaic / coadd / DOLPHOT prep |
| `jwst123/scripts/link_raw.py` | Symlink FITS into a reduction `raw/` directory |
| `jwst123/scripts/image_overlap.py` | Maximum-overlap reference selection |
| `jwst123/scripts/illuminated_s_region.py` | Illuminated `S_REGION` CLI |
| `jwst123/scripts/apply_gwcs.py` | Attach GWCS to coadd datamodels |
| `jwst123/scripts/catalog.py` | Combined photometry catalog CLI |

### Notebooks

| Notebook | Role |
| --- | --- |
| `jwst123/notebooks/download.ipynb` | Interactive MAST queries (HST or JWST) |
| `jwst123/notebooks/align.ipynb` | Relative / Gaia JHAT alignment |
| `jwst123/notebooks/mosaic.ipynb` | Level-3 mosaics and PSF-matched coadds |

## Requirements

- **Python 3.12** (3.11 also supported)
- External **DOLPHOT** binaries if you run PSF photometry
  ([DOLPHOT](http://americano.dolphinsim.com/dolphot/))

## Installation

The same conda + pip flow works on macOS and Linux/Ubuntu. Dependencies are
declared in `requirements.txt` and installed through `pyproject.toml`.

### macOS and Linux / Ubuntu

```bash
conda create -n jwst123 python=3.12 pip
conda activate jwst123
```

`drizzlepac` / `tables` need HDF5. Install the libraries with conda (recommended
on both platforms) before the editable install:

```bash
conda install -c conda-forge hdf5 blosc pytables -y
```

Then install jwst123 and its pinned dependencies from the repository root:

```bash
pip install -e .
```

If `tables` still cannot find HDF5 on macOS Homebrew:

```bash
brew install hdf5 c-blosc
export HDF5_DIR="$(brew --prefix hdf5)"
export BLOSC_DIR="$(brew --prefix c-blosc)"
pip install -e .
```

On Ubuntu, if you prefer system packages instead of conda HDF5:

```bash
sudo apt-get install -y libhdf5-dev libblosc-dev
pip install -e .
```

### Verify

```bash
python -c "import jwst, jhat, jwst123; print(jwst.__version__, jwst123.__version__)"
download --help
```

This install path was validated on macOS with Python 3.12 (`conda create -n jwst123 python=3.12 pip` then `pip install -e .`). The same steps apply on Linux/Ubuntu.
## Quick start

### Download JWST data

```bash
python jwst123/scripts/download.py \
  --ra "10:38:47.961" --dec "+53:30:34.10" \
  --obj NGC3310 \
  --outdir /path/to/NGC3310
```

For proprietary data, pass a MAST API token
([create one here](https://auth.mast.stsci.edu/info)), the same way hst123
used `--token` with `Observations.login`:

```bash
python jwst123/scripts/download.py \
  --ra 189.9976 --dec -11.623 \
  --obj NGC4536 \
  --outdir /path/to/NGC4536 \
  --token YOUR_MAST_TOKEN
```

Or export the token and omit `--token`:

```bash
export MAST_API_TOKEN=YOUR_MAST_TOKEN
python jwst123/scripts/download.py \
  --ra 189.9976 --dec -11.623 \
  --obj NGC4536 \
  --outdir /path/to/NGC4536
```

Useful options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--outdir` | `jwst_data/<obj>` | Output directory |
| `--radius` | `3.0` | Search radius in arcminutes |
| `--stage` | `2` | `2` = CAL, `3` = I2D |
| `--instruments` | `NIRCAM MIRI` | Instrument name filters |
| `--token` | `MAST_API_TOKEN` / `MAST_TOKEN` | MAST API token for proprietary data |

After install, console scripts from `pyproject.toml` are available (`download`, `align`, `mosaic`, `link-raw`, `image-overlap`, `illuminated-s-region`, `relative-align`, `apply-gwcs`, `catalog`).

### Stage files for a reduction

```bash
python jwst123/scripts/link_raw.py \
  --datadir /path/to/downloaded/data \
  --symlinkdir /path/to/reduction
```

### Relative alignment (one image → reference)

```bash
python jwst123/scripts/relative_align.py \
  --ref /path/to/coadd_i2d.fits \
  --align /path/to/cal_or_i2d.fits \
  --outdir /path/to/alignment_output
```

### Visit-level alignment pipeline

```bash
python jwst123/scripts/align.py --workdir /path/to/reduction --object TARGET
```

### Mosaics / DOLPHOT prep

```bash
python jwst123/scripts/mosaic.py --basedir /path/to/reduction --object TARGET
```

## Notes

- CRDS reference files are required for `jwst` pipeline steps; set `CRDS_PATH` /
  `CRDS_SERVER_URL` as recommended by STScI.
- JHAT alignment parameters live in `jwst123/settings.py` (`strict_*` /
  `relaxed_*` Gaia and JWST sets).
- HST MAST helpers remain in `jwst123.mast` for reference-image queries, but the
  legacy `hst123` reduction pipeline is not part of this repository.
