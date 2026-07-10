#!/usr/bin/env python3
"""
Relative WCS alignment of a JWST image to a reference image with JHAT.

Builds a photometry catalog from ``--ref``, then aligns ``--align`` to that
catalog. Diagnostic plots and match tables are written under ``--outdir``.

Example:

    python scripts/jwst_relative_align.py \\
        --ref /Users/ckilpatrick/Downloads/data/coadd_1_10_f115w_i2d.fits \\
        --align /Users/ckilpatrick/Downloads/data/jw01783007003_02101_00001_mirimage_cal.fits \\
        --outdir /Users/ckilpatrick/Downloads/data/aligned
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path

# Non-interactive backend so JHAT diagnostics can be saved without blocking.
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# JHAT diagnostic PNGs written via saveplots / our plt.show() hook
DIAGNOSTIC_GLOBS = (
    '*.phot.prewcs.png',
    '*.phot.finalwcs.png',
    '*.diag_*.png',
    '*.goodmatches.csv',
    '*.goodmatches.phot.csv',
    '*.allmatches.phot.txt',
    '*.good.phot.txt',
    '*.refcat.txt',
    '*.phot.txt',
    '*_jhat.fits',
)


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Relative JHAT alignment: register --align to a photometry catalog '
            'built from --ref, and write diagnostic plots to --outdir.'
        ),
    )
    parser.add_argument(
        '--ref',
        type=str,
        required=True,
        help='Reference image used to build the photometry catalog (e.g. coadd i2d).',
    )
    parser.add_argument(
        '--align',
        type=str,
        required=True,
        help='Image to align (e.g. *_cal.fits).',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help='Output directory for JHAT products and plots (default: <align parent>/aligned).',
    )
    parser.add_argument('--nbright', type=int, default=800, help='Bright sources for JHAT.')
    parser.add_argument(
        '--ee-radius',
        type=float,
        default=70,
        help='Encircled-energy radius for JWST photometry.',
    )
    parser.add_argument(
        '--d2d-max',
        type=float,
        default=1.0,
        help='Max match radius (arcsec) for JHAT.',
    )
    parser.add_argument('--snr-min', type=float, default=3.0, help='Minimum SNR for JHAT sources.')
    parser.add_argument('--verbose', action='store_true', help='Verbose JHAT output.')
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip JHAT diagnostic plots (default: save all diagnostics).',
    )
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Also open interactive plot windows (requires a GUI backend).',
    )
    parser.add_argument(
        '--photfile',
        type=str,
        default=None,
        help='Reuse an existing .phot.txt catalog instead of running photometry on --ref.',
    )
    parser.add_argument(
        '--telescope',
        type=str,
        default='jwst',
        help='Telescope string passed to JHAT (default: jwst).',
    )
    return parser


def resolve_outdir(align_image: str, outdir: str | None) -> str:
    if outdir:
        path = Path(outdir)
    else:
        path = Path(align_image).resolve().parent / 'aligned'
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def install_plot_saver(outdir: str, prefix: str):
    """
    Replace ``plt.show`` so JHAT figures that only call ``show()`` are saved.

    JHAT's ``saveplots`` writes ``*.phot.prewcs.png`` / ``*.phot.finalwcs.png``.
    Initial match and histogram-cut figures only call ``plt.show()``; this hook
    captures those as ``{prefix}.diag_NN.png``. Figures already written via
    ``savefig`` are not duplicated.
    """
    counter = {'n': 0}
    already_saved: set[int] = set()
    original_show = plt.show
    original_savefig = plt.savefig

    def savefig_track(*args, **kwargs):
        already_saved.add(plt.gcf().number)
        return original_savefig(*args, **kwargs)

    def show_and_save(*args, **kwargs):
        fignums = plt.get_fignums()
        if not fignums:
            return
        for num in fignums:
            if num in already_saved:
                continue
            fig = plt.figure(num)
            counter['n'] += 1
            path = os.path.join(outdir, f'{prefix}.diag_{counter["n"]:02d}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f'Saved diagnostic plot: {path}')
        already_saved.clear()
        plt.close('all')

    plt.savefig = savefig_track
    plt.show = show_and_save
    return original_show, original_savefig


def _has_jwst_gwcs(image: str) -> bool:
    """Return True if the file has a JWST datamodel GWCS (ASDF extension)."""
    from astropy.io import fits

    with fits.open(image) as hdul:
        return any(h.name == 'ASDF' for h in hdul)


def jwst_phot(phot_img: str, ee_radius: float = 70.0):
    """Run JHAT JWST photometry and return (catalog table, phot filename)."""
    from astropy.table import Table
    from jhat import jwst_photclass

    phot = jwst_photclass()
    photfilename = phot_img.replace('.fits', '.phot.txt')
    phot.run_phot(
        imagename=phot_img,
        photfilename=photfilename,
        overwrite=True,
        ee_radius=ee_radius,
    )
    refcat = Table.read(photfilename, format='ascii')
    return refcat, photfilename


def hst_style_phot(phot_img: str, ee_radius: float = 70.0):
    """Photometry for images that lack JWST GWCS (custom coadds / mosaics)."""
    from astropy.table import Table
    from jhat import hst_photclass

    phot = hst_photclass()
    photfilename = phot_img.replace('.fits', '.phot.txt')
    try:
        phot.run_phot(
            imagename=phot_img,
            photfilename=photfilename,
            overwrite=True,
            aperture_radius=max(3.0, ee_radius / 10.0),
        )
    except TypeError:
        phot.run_phot(
            imagename=phot_img,
            photfilename=photfilename,
            overwrite=True,
        )
    refcat = Table.read(photfilename, format='ascii')
    return refcat, photfilename


def photutils_phot(phot_img: str, nsigma: float = 5.0, fwhm: float = 3.0):
    """Fallback DAOStarFinder photometry using the SCI WCS."""
    import numpy as np
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.table import Table
    from astropy.wcs import WCS
    from photutils.detection import DAOStarFinder

    with fits.open(phot_img) as hdul:
        data = hdul['SCI'].data.astype(float)
        w = WCS(hdul['SCI'].header)

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=nsigma * std)
    sources = finder(data - median)
    if sources is None or len(sources) == 0:
        raise RuntimeError(f'No sources found in {phot_img}')

    ra, dec = w.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)
    flux = np.array(sources['flux'], dtype=float)
    flux = np.where(flux > 0, flux, np.nan)
    mag = -2.5 * np.log10(flux)
    dmag = np.full(len(mag), 0.05)

    refcat = Table(
        {
            'x': sources['xcentroid'],
            'y': sources['ycentroid'],
            'ra': ra,
            'dec': dec,
            'mag': mag,
            'dmag': dmag,
        }
    )
    photfilename = phot_img.replace('.fits', '.phot.txt')
    # JHAT's pdastro.load() does not treat '#' as a commented header.
    # Write a plain space-separated header matching pdastro.write() / to_string().
    refcat.to_pandas().to_string(photfilename, index=False)
    return refcat, photfilename


def build_ref_catalog(
    ref_image: str,
    outdir: str,
    photfile: str | None = None,
    ee_radius: float = 70.0,
) -> str:
    """Build or reuse a reference catalog; copy it into ``outdir``."""
    if photfile is not None:
        if not os.path.exists(photfile):
            raise FileNotFoundError(f'Photometry catalog not found: {photfile}')
        print(f'Using existing photometry catalog: {photfile}')
        src = photfile
    else:
        print(f'Running photometry on reference: {ref_image}')
        if _has_jwst_gwcs(ref_image):
            print('  detected JWST ASDF/GWCS → jwst_photclass')
            refcat, src = jwst_phot(ref_image, ee_radius=ee_radius)
        else:
            print('  no JWST ASDF/GWCS (likely a custom coadd) → hst_photclass / photutils')
            try:
                refcat, src = hst_style_phot(ref_image, ee_radius=ee_radius)
            except Exception as exc:
                print(f'  hst_photclass failed ({exc}); falling back to photutils')
                refcat, src = photutils_phot(ref_image)
        print(f'Reference catalog: {src} ({len(refcat)} sources)')

    dest = os.path.join(outdir, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.copy2(src, dest)
        print(f'Copied reference catalog → {dest}')
    return dest


def run_relative_align(
    align_image: str,
    outdir: str,
    photfilename: str,
    telescope: str = 'jwst',
    d2d_max: float = 1.0,
    snr_min: float = 3.0,
    nbright: int = 800,
    verbose: bool = False,
    save_plots: bool = True,
):
    """Align ``align_image`` to the reference photometry catalog with JHAT."""
    from jhat import st_wcs_align

    # showplots=2: initial match + histogram-cut diagnostics
    # saveplots=2: write *.phot.prewcs.png and *.phot.finalwcs.png
    showplots = 2 if save_plots else 0
    saveplots = 2 if save_plots else 0

    print(f'Aligning {align_image}')
    print(f'  reference catalog: {photfilename}')
    print(f'  outdir: {outdir}')
    if save_plots:
        print('  diagnostics: showplots=2, saveplots=2')

    # JHAT joins outrootdir + outsubdir; pass the absolute path as outrootdir
    # so products land in ``outdir`` rather than ``./<outdir>``.
    wcs_align = st_wcs_align()
    wcs_align.run_all(
        align_image,
        telescope=telescope,
        outrootdir=outdir,
        refcat_racol='ra',
        refcat_deccol='dec',
        refcat_magcol='mag',
        refcat_magerrcol='dmag',
        overwrite=True,
        d2d_max=d2d_max,
        showplots=showplots,
        saveplots=saveplots,
        savephottable=1,
        refcatname=photfilename,
        histocut_order='dxdy',
        sharpness_lim=(0.3, 0.9),
        roundness1_lim=(-0.7, 0.7),
        SNR_min=snr_min,
        dmag_max=1.0,
        Nbright=nbright,
        use_dq=False,
        verbose=verbose,
    )
    return wcs_align


def find_jhat_product(align_image: str, outdir: str) -> str | None:
    """Locate the JHAT output for a cal/i2d input."""
    base = os.path.basename(align_image)
    candidates = []
    if base.endswith('cal.fits'):
        candidates.append(base.replace('cal.fits', 'jhat.fits'))
    if base.endswith('i2d.fits'):
        candidates.append(base.replace('i2d.fits', 'jhat.fits'))
        candidates.append(base.replace('i2d.fits', 'jhat_i2d.fits'))
    candidates.append(base.replace('.fits', '_jhat.fits'))

    for name in candidates:
        path = os.path.join(outdir, name)
        if os.path.exists(path):
            return path
    stem = Path(align_image).stem.split('_cal')[0].split('_i2d')[0]
    matches = sorted(Path(outdir).glob(f'*{stem}*jhat*.fits'))
    return str(matches[0]) if matches else None


def list_diagnostics(outdir: str) -> list[str]:
    """Return sorted paths of JHAT products / diagnostic files in ``outdir``."""
    found: set[str] = set()
    root = Path(outdir)
    for pattern in DIAGNOSTIC_GLOBS:
        found.update(str(p) for p in root.glob(pattern))
    return sorted(found)


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)

    for path, label in ((args.ref, '--ref'), (args.align, '--align')):
        if not os.path.exists(path):
            print(f'ERROR: {label} file not found: {path}', file=sys.stderr)
            return 1

    outdir = resolve_outdir(args.align, args.outdir)
    print(f'Output directory: {outdir}')

    save_plots = not args.no_plots
    plot_hooks = None
    if save_plots and not args.show_plots:
        stem = Path(args.align).stem.replace('_cal', '').replace('_i2d', '')
        plot_hooks = install_plot_saver(outdir, prefix=stem)
    elif args.show_plots:
        # Interactive mode: switch off Agg if a GUI is available.
        try:
            matplotlib.use('TkAgg', force=True)
        except Exception:
            print('WARNING: could not enable interactive backend; plots will be saved only')
            stem = Path(args.align).stem.replace('_cal', '').replace('_i2d', '')
            plot_hooks = install_plot_saver(outdir, prefix=stem)

    try:
        photfilename = build_ref_catalog(
            args.ref,
            outdir,
            photfile=args.photfile,
            ee_radius=args.ee_radius,
        )
        run_relative_align(
            args.align,
            outdir,
            photfilename,
            telescope=args.telescope,
            d2d_max=args.d2d_max,
            snr_min=args.snr_min,
            nbright=args.nbright,
            verbose=args.verbose,
            save_plots=save_plots,
        )
    finally:
        if plot_hooks is not None:
            plt.show, plt.savefig = plot_hooks

    product = find_jhat_product(args.align, outdir)
    if product:
        print(f'Relative JHAT product: {product}')
    else:
        print('WARNING: could not locate relative JHAT output product')

    diagnostics = list_diagnostics(outdir)
    if diagnostics:
        print('Products / diagnostics:')
        for path in diagnostics:
            print(f'  {path}')

    print('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
