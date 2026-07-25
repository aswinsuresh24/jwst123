# jwst123

Tools for downloading JWST imaging from MAST, aligning frames with a
**custom, repository-local build of [JHAT](https://jhat.readthedocs.io/)**,
building mosaics / coadds, and preparing DOLPHOT runs.

## Repository layout

| Path | Role |
| --- | --- |
| `jwst123/` | Installable package (library, scripts, and notebooks) |
| `jwst123/scripts/` | Command-line entry points (required location for all `__main__` CLIs) |
| `jwst123/notebooks/` | Generic notebooks for download, alignment, and mosaics |
| `extdeps/` | Vendored / customized external packages (see below) |
| `extdeps/jhat/` | **Custom JHAT build for this repository** (not PyPI) |
| `pyproject.toml` | Package metadata; dependencies loaded from `requirements.txt` |
| `requirements.txt` | Pinned Python dependencies (JHAT excluded; install from `extdeps/jhat`) |

### Package modules

| Module | Role |
| --- | --- |
| `jwst123.alignment` | Alignment package (JHAT drivers, MIRI pipeline, calibrators) |
| `jwst123.alignment.align` | JHAT / Gaia alignment and visit grouping |
| `jwst123.alignment.relative_align` | Single-frame JHAT relative align + iterative refine |
| `jwst123.alignment.alignment_wrap` | MIRI↔reference overlap + filter-wave orchestration |
| `jwst123.alignment.alignment_fallback` | MIRI→MIRI parent ranking and provenance |
| `jwst123.alignment.alignment_parallel` | Spawn-safe REFERENCE / MIRI_REL workers |
| `jwst123.alignment.calibrators` | Per-filter JHAT/refine knobs and quality-hold thresholds |
| `jwst123.mosaic` | Overlap splitting, PSF matching, coadds, GWCS, DOLPHOT prep |
| `jwst123.mast` | MAST query and download helpers (`mast`, `download` submodules) |
| `jwst123.utils` | Shared utilities package (helpers, settings, link, constants) |
| `jwst123.utils.helpers` | Coordinates, FITS bookkeeping, visits, xmatch |
| `jwst123.utils.settings` | JHAT and DOLPHOT parameter sets |
| `jwst123.utils.link` | Symlink helpers for reduction ``raw/`` trees |
| `jwst123.utils.constants` | ANSI color strings for CLI messages |
| `jwst123.illuminated_s_region` | Illuminated footprint / `S_REGION` from science+DQ |
| `jwst123.image_overlap` | MIRI vs reference footprint overlap |

### Scripts

| Script | Role |
| --- | --- |
| `jwst123/scripts/download.py` | Download JWST products from MAST |
| `jwst123/scripts/align.py` | Group / visit JHAT alignment pipeline CLI |
| `jwst123/scripts/relative_align.py` | Single-frame relative-align CLI |
| `jwst123/scripts/alignment_wrap.py` | Full MIRI pipeline CLI (overlap → REFERENCE → MIRI_REL) |
| `jwst123/scripts/mosaic.py` | Mosaic / coadd / DOLPHOT prep |
| `jwst123/scripts/link_raw.py` | Symlink FITS into a reduction `raw/` directory |
| `jwst123/scripts/image_overlap.py` | Maximum-overlap reference selection |
| `jwst123/scripts/illuminated_s_region.py` | Illuminated `S_REGION` CLI |
| `jwst123/scripts/apply_gwcs.py` | Attach GWCS to coadd datamodels |
| `jwst123/scripts/catalog.py` | Combined photometry catalog CLI |

All CLI entry points with a `__main__` block live under `jwst123/scripts/`
(including `alignment_wrap.py` and `jwst_download.py`). See the local Cursor
rule in `.cursor/` (untracked) and `tests/test_entry_point_convention.py`.

### Notebooks

| Notebook | Role |
| --- | --- |
| `jwst123/notebooks/download.ipynb` | Interactive MAST queries (HST or JWST) |
| `jwst123/notebooks/align.ipynb` | Relative / Gaia JHAT alignment |
| `jwst123/notebooks/mosaic.ipynb` | Level-3 mosaics and PSF-matched coadds |

## Custom JHAT (`extdeps/jhat`)

This repository vendors a **custom JHAT build** under [`extdeps/jhat`](extdeps/jhat).
It is based on upstream
[arminrest/jhat](https://github.com/arminrest/jhat) but is **not** the
unmodified PyPI package (`jhat` on PyPI).

Use this tree for all JHAT-backed alignment code in jwst123, including:

- `jwst123.alignment.align` / `align_jwst_image` (`jhat_params`, soft-fail behavior)
- `jwst123.alignment.relative_align` (master catalogs, iterative refine, F560W/F770W knobs)
- `jwst123.alignment.alignment_wrap` (REFERENCE → MIRI_REL pipeline)

Install it from `extdeps/jhat` (see Installation). Do **not**
`pip install jhat` from PyPI for this project unless you intentionally want
upstream instead of the custom build. Details and version marking
(`0.3.7+jwst123`) are in [`extdeps/jhat/README.md`](extdeps/jhat/README.md).

## Requirements

- **Python 3.12** (3.11 also supported)
- External **DOLPHOT** binaries if you run PSF photometry
  ([DOLPHOT](http://americano.dolphinsim.com/dolphot/))
- The custom JHAT package under `extdeps/jhat` (installed with the steps below)

## Installation

The same conda + pip flow works on macOS and Linux/Ubuntu. Pinned dependencies
are declared in `requirements.txt` (via `pyproject.toml`). JHAT is installed
from `extdeps/jhat`, not from PyPI.

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

Then, from the repository root, install **custom JHAT** and **jwst123**
together:

```bash
pip install -e ./extdeps/jhat -e .
```

If `tables` still cannot find HDF5 on macOS Homebrew:

```bash
brew install hdf5 c-blosc
export HDF5_DIR="$(brew --prefix hdf5)"
export BLOSC_DIR="$(brew --prefix c-blosc)"
pip install -e ./extdeps/jhat -e .
```

On Ubuntu, if you prefer system packages instead of conda HDF5:

```bash
sudo apt-get install -y libhdf5-dev libblosc-dev
pip install -e ./extdeps/jhat -e .
```

### Verify

```bash
python -c "import jhat, jwst123; print(jhat.__version__, jhat.__file__); print(jwst123.__version__)"
# jhat.__version__ should be 0.3.7+jwst123
# jhat.__file__ should point under .../extdeps/jhat/jhat/
download --help
alignment-wrap --help
```

This install path was validated on macOS with Python 3.12
(`conda create -n jwst123 python=3.12 pip` then
`pip install -e ./extdeps/jhat -e .`). The same steps apply on Linux/Ubuntu.
## Quick start

### Download JWST data

```bash
python jwst123/scripts/download.py \
  --ra "10:38:47.961" --dec "+53:30:34.10" \
  --obj NGC3310 \
  --outdir /path/to/NGC3310
```

MIRI-only download into the `<FILTER>/<obsid>/mastDownload/...` layout used by
`alignment_wrap` (same as the `jwst_RSGs` `jwst_download.py` workflow):

```bash
python -m jwst123.scripts.jwst_download \
  --ra 159.694014 --dec 53.502851 --obj NGC3310 \
  --download-dir /data/rwisenbaker/jwst_data/NGC3310 \
  --radius 3 --stage 2
```

or equivalently:

```bash
python -m jwst123.scripts.download \
  --ra 159.694014 --dec 53.502851 --obj NGC3310 \
  --download-dir /data/rwisenbaker/jwst_data/NGC3310 \
  --radius 3 --stage 2 --instruments MIRI --layout filter/obsid
# after pip install -e .:  jwst-download ...   or   download ...
```

### MIRI ↔ NIRCam alignment pipeline

With MIRI cals under `--data-dir/<FILTER>/<obsid>/mastDownload/...` and NIRCam
coadds under `--data-dir/reference/`:

```bash
python -m jwst123.scripts.alignment_wrap \
  --data-dir /data/rwisenbaker/jwst_data/NGC3310 \
  --plot --continue-on-error --workers 8
```

or the `alignment-wrap` console script after `pip install -e .`.

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
