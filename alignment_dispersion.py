#!/usr/bin/env python3
"""
Relative JHAT alignment of one JWST image to a reference, with dispersion metrics.

Builds a photometry catalog from ``--ref``, optionally photometers ``--align``,
then runs ``jwst123.align_jwst_image`` and reports initial/final dispersion.

Example:

    python alignment_dispersion.py \\
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

# Ensure local package import when run from another cwd.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        '--skip-align-phot',
        action='store_true',
        help='Skip photometry on --align before running JHAT.',
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


def write_jhat_phot_table(table: Table, photfilename: str) -> str:
    """
    Write a catalog JHAT can load.

    JHAT's pdastro loader does not treat '#' as a comment header, so write a
    plain space-separated table via pandas.
    """
    table.to_pandas().to_string(photfilename, index=False)
    return photfilename


def photutils_phot(image: str, nsigma: float = 5.0, fwhm: float = 3.0) -> str:
    """Fallback DAOStarFinder photometry using the SCI WCS."""
    with fits.open(image) as hdul:
        data = hdul['SCI'].data.astype(float)
        wcs = WCS(hdul['SCI'].header)

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
    photfilename = image.replace('.fits', '.phot.txt')
    return write_jhat_phot_table(catalog, photfilename)


def build_phot_catalog(image: str, label: str = 'image') -> str:
    """
    Build a JHAT-compatible photometry catalog for ``image``.

    Preference order:
      1. ``jwst123.fix_phot`` / ``jwst_phot`` when JWST GWCS is present
      2. ``jwst123.fix_phot`` for i2d mosaics (rewrites RA/Dec)
      3. photutils DAOStarFinder fallback for custom coadds
    """
    print(f'Running photometry on {label}: {image}')

    if has_jwst_gwcs(image):
        print('  detected JWST ASDF/GWCS → jwst_phot')
        _, photfilename = jwst123.jwst_phot(image)
        return photfilename

    # Custom coadds / mosaics often lack ASDF; prefer fix_phot when possible.
    if image.endswith('i2d.fits'):
        try:
            print('  no JWST ASDF/GWCS; trying fix_phot')
            return jwst123.fix_phot(image)
        except Exception as exc:
            print(f'  fix_phot failed ({exc}); falling back to photutils')

    print('  falling back to photutils DAOStarFinder')
    return photutils_phot(image)


def install_plot_saver(outdir: str, prefix: str):
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
            path = os.path.join(outdir, f'{prefix}.diag_{counter["n"]:02d}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f'Saved diagnostic plot: {path}')
        already_saved.clear()
        plt.close('all')

    plt.Figure.savefig = savefig_track
    plt.show = show_and_save
    return original_show, original_savefig


def resolve_outdir(outdir: str) -> str:
    path = Path(outdir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def stage_photfile(photfile: str, outdir: str) -> str:
    """Copy the reference catalog into ``outdir`` and return that path."""
    dest = os.path.join(outdir, os.path.basename(photfile))
    same_file = (
        os.path.exists(dest)
        and os.path.samefile(photfile, dest)
    )
    if not same_file:
        shutil.copy2(photfile, dest)
        print(f'Copied reference catalog → {dest}')
    return dest


def run_alignment(
    ref_image: str,
    align_image: str,
    outdir: str,
    photfile: str | None = None,
    nbright: int = 800,
    skip_align_phot: bool = False,
    plot: bool = False,
    verbose: bool = False,
) -> tuple[object, str]:
    """Build catalogs and align ``align_image`` to ``ref_image``."""
    outdir = resolve_outdir(outdir)
    print(f'Output directory: {outdir}')
    print(f'Reference image:  {ref_image}')
    print(f'Align image:      {align_image}')

    if photfile is not None:
        if not os.path.exists(photfile):
            raise FileNotFoundError(f'Reference photometry catalog not found: {photfile}')
        print(f'Using existing reference catalog: {photfile}')
        ref_phot = photfile
    else:
        ref_phot = build_phot_catalog(ref_image, label='reference')

    ref_phot = stage_photfile(ref_phot, outdir)

    if not skip_align_phot:
        align_phot = build_phot_catalog(align_image, label='align')
        print(f'Align photometry catalog: {align_phot}')

    plot_hooks = None
    if plot:
        stem = Path(align_image).stem.replace('_cal', '').replace('_i2d', '')
        plot_hooks = install_plot_saver(outdir, prefix=stem)

    try:
        # JHAT treats outsubdir relative to cwd; chdir so products land in outdir.
        cwd = os.getcwd()
        os.chdir(outdir)
        try:
            guess_offset = jwst123.align_jwst_image(
                align_image=align_image,
                outdir='.',
                gaia=False,
                photfilename=ref_phot,
                Nbright=nbright,
                plot=plot,
                verbose=verbose,
            )
        finally:
            os.chdir(cwd)
    finally:
        if plot_hooks is not None:
            plt.show, plt.Figure.savefig = plot_hooks

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
            skip_align_phot=args.skip_align_phot,
            plot=args.plot,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f'ERROR: alignment failed: {exc}', file=sys.stderr)
        raise

    print(f'Guess offset (x, y): {guess_offset}')
    print(f'Done. Products in {outdir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
