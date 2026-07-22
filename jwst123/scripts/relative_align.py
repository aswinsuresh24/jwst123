#!/usr/bin/env python3
"""
Relative JHAT alignment of one JWST image to a reference, with dispersion metrics.

Builds a photometry catalog from ``--ref``, then runs ``jwst123.align_jwst_image``
(which photometers ``--align`` internally) and reports initial/final dispersion.

Example:

    python -m jwst123.scripts.relative_align \\
        --ref /path/to/coadd_i2d.fits \\
        --align /path/to/mirimage_cal.fits \\
        --outdir alignment_output
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder

warnings.filterwarnings('ignore')

import jwst123  # noqa: E402


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Align --align to a photometry catalog from --ref with JHAT, '
            'and report alignment dispersion.'
        )
    )
    parser.add_argument(
        '--ref',
        required=True,
        help='Reference image used to build the photometry catalog (e.g. coadd i2d).',
    )
    parser.add_argument(
        '--align',
        required=True,
        help='Image to align (e.g. MIRI *_cal.fits or NIRCam *_i2d.fits).',
    )
    parser.add_argument(
        '--outdir',
        default='alignment_output',
        help='Output directory for JHAT products (default: alignment_output).',
    )
    parser.add_argument(
        '--photfile',
        default=None,
        help='Reuse an existing reference .phot.txt catalog instead of building one.',
    )
    parser.add_argument(
        '--nbright',
        type=int,
        default=800,
        help='Number of bright sources for JHAT (default: 800).',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Enable JHAT diagnostic plots (saved via Agg; non-interactive).',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose JHAT / alignment output.',
    )
    return parser


def has_jwst_gwcs(image: str) -> bool:
    """Return True if the FITS file has a JWST ASDF / GWCS extension."""
    with fits.open(image) as hdul:
        return any(hdu.name == 'ASDF' for hdu in hdul)


def phot_catalog_path(image: str, outdir: str) -> str:
    """Return ``outdir/<image-stem>.phot.txt``."""
    return str(Path(outdir) / f'{Path(image).stem}.phot.txt')


def write_jhat_phot_table(table: Table, photfilename: str) -> str:
    """
    Write a catalog JHAT can load.

    JHAT's pdastro loader does not treat '#' as a comment header, so write a
    plain space-separated table via pandas.
    """
    Path(photfilename).parent.mkdir(parents=True, exist_ok=True)
    table.to_pandas().to_string(photfilename, index=False)
    return photfilename


def load_sci_data_wcs(image: str) -> tuple[np.ndarray, WCS]:
    """Load science array + WCS from SCI if present, else the first 2-D HDU."""
    with fits.open(image) as hdul:
        if 'SCI' in hdul and hdul['SCI'].data is not None:
            hdu = hdul['SCI']
        else:
            hdu = next(
                (h for h in hdul if h.data is not None and getattr(h.data, 'ndim', 0) == 2),
                None,
            )
            if hdu is None:
                raise ValueError(f'No 2-D image HDU found in {image}')
        return hdu.data.astype(float), WCS(hdu.header)


def photutils_phot(image: str, photfilename: str, nsigma: float = 5.0, fwhm: float = 3.0) -> str:
    """Fallback DAOStarFinder photometry using the science WCS."""
    data, wcs = load_sci_data_wcs(image)
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    sources = DAOStarFinder(fwhm=fwhm, threshold=nsigma * std)(data - median)
    if sources is None or len(sources) == 0:
        raise RuntimeError(f'No sources found in {image}')

    ra, dec = wcs.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)
    flux = np.asarray(sources['flux'], dtype=float)
    flux = np.where(flux > 0, flux, np.nan)
    mag = -2.5 * np.log10(flux)
    dmag = np.full(len(mag), 0.05)

    catalog = Table(
        {
            'x': sources['xcentroid'],
            'y': sources['ycentroid'],
            'ra': ra,
            'dec': dec,
            'mag': mag,
            'dmag': dmag,
        }
    )
    return write_jhat_phot_table(catalog, photfilename)


def build_ref_catalog(image: str, outdir: str, photfile: str | None = None) -> str:
    """
    Build or stage a JHAT-compatible reference photometry catalog in ``outdir``.

    Preference order for new catalogs:
      1. ``jwst123.jwst_phot`` when JWST GWCS / ASDF is present
      2. ``jwst123.fix_phot`` for i2d mosaics (rewrites RA/Dec from SCI WCS)
      3. photutils DAOStarFinder for custom coadds without pipeline WCS
    """
    if photfile is not None:
        photfile = str(Path(photfile).expanduser().resolve())
        if not os.path.exists(photfile):
            raise FileNotFoundError(f'Reference photometry catalog not found: {photfile}')
        print(f'Using existing reference catalog: {photfile}')
        return stage_photfile(photfile, outdir)

    dest = phot_catalog_path(image, outdir)
    print(f'Running photometry on reference: {image}')

    if has_jwst_gwcs(image):
        print('  detected JWST ASDF/GWCS → jwst_phot')
        _, src = jwst123.jwst_phot(image)
        return stage_photfile(src, outdir, dest_name=Path(dest).name)

    if image.endswith(('i2d.fits', 'i2d.fits.gz')):
        try:
            print('  no JWST ASDF/GWCS; trying fix_phot')
            src = jwst123.fix_phot(image)
            return stage_photfile(src, outdir, dest_name=Path(dest).name)
        except Exception as exc:
            print(f'  fix_phot failed ({exc}); falling back to photutils')

    print('  falling back to photutils DAOStarFinder')
    return photutils_phot(image, dest)


def stage_photfile(
    photfile: str,
    outdir: str,
    dest_name: str | None = None,
) -> str:
    """Copy a catalog into ``outdir`` (no-op if already there) and return that path."""
    dest = os.path.join(outdir, dest_name or os.path.basename(photfile))
    if os.path.exists(dest) and os.path.samefile(photfile, dest):
        return dest
    shutil.copy2(photfile, dest)
    print(f'Copied reference catalog → {dest}')
    return dest


@contextmanager
def working_directory(path: str):
    """Temporarily change the process working directory."""
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextmanager
def plot_saver(outdir: str, prefix: str):
    """Save JHAT figures that only call ``plt.show()`` into ``outdir``."""
    counter = {'n': 0}
    already_saved: set[int] = set()
    original_show = plt.show
    original_savefig = plt.Figure.savefig

    def savefig_track(self, *args, **kwargs):
        already_saved.add(self.number)
        return original_savefig(self, *args, **kwargs)

    def show_and_save(*args, **kwargs):
        for num in plt.get_fignums():
            if num in already_saved:
                continue
            fig = plt.figure(num)
            counter['n'] += 1
            out = os.path.join(outdir, f'{prefix}.diag_{counter["n"]:02d}.png')
            # Use the original savefig so we do not mark this as a JHAT savefig.
            original_savefig(fig, out, dpi=150, bbox_inches='tight')
            print(f'Saved diagnostic plot: {out}')
        already_saved.clear()
        plt.close('all')

    plt.Figure.savefig = savefig_track
    plt.show = show_and_save
    try:
        yield
    finally:
        plt.show = original_show
        plt.Figure.savefig = original_savefig


def resolve_outdir(outdir: str) -> str:
    path = Path(outdir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def run_alignment(
    ref_image: str,
    align_image: str,
    outdir: str,
    photfile: str | None = None,
    nbright: int = 800,
    plot: bool = False,
    verbose: bool = False,
) -> tuple[object, str]:
    """Build a reference catalog and align ``align_image`` to it."""
    ref_image = str(Path(ref_image).expanduser().resolve())
    align_image = str(Path(align_image).expanduser().resolve())
    outdir = resolve_outdir(outdir)

    print(f'Output directory: {outdir}')
    print(f'Reference image:  {ref_image}')
    print(f'Align image:      {align_image}')

    ref_phot = build_ref_catalog(ref_image, outdir, photfile=photfile)

    # JHAT's outsubdir is relative to cwd; run from outdir so products land there.
    # Paths above are absolute so chdir is safe.
    stem = Path(align_image).stem.replace('_cal', '').replace('_i2d', '')
    with working_directory(outdir):
        if plot:
            with plot_saver(outdir, prefix=stem):
                guess_offset = jwst123.align_jwst_image(
                    align_image=align_image,
                    outdir='.',
                    gaia=False,
                    photfilename=ref_phot,
                    Nbright=nbright,
                    plot=True,
                    verbose=verbose,
                )
        else:
            guess_offset = jwst123.align_jwst_image(
                align_image=align_image,
                outdir='.',
                gaia=False,
                photfilename=ref_phot,
                Nbright=nbright,
                plot=False,
                verbose=verbose,
            )

    return guess_offset, outdir


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)

    for path, label in ((args.ref, '--ref'), (args.align, '--align')):
        if not os.path.exists(path):
            print(f'ERROR: {label} file not found: {path}', file=sys.stderr)
            return 1

    try:
        guess_offset, outdir = run_alignment(
            ref_image=args.ref,
            align_image=args.align,
            outdir=args.outdir,
            photfile=args.photfile,
            nbright=args.nbright,
            plot=args.plot,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f'ERROR: alignment failed: {exc}', file=sys.stderr)
        return 1

    print(f'Guess offset (x, y): {guess_offset}')
    print(f'Done. Products in {outdir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
