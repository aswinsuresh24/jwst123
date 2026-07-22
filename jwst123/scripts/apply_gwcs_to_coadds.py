#!/usr/bin/env python3
"""Attach a GWCS object to coadd ``*_i2d.fits`` datamodels."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from astropy.io import fits
from jwst import datamodels

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwst123.mosaic import assign_gwcs  # noqa: E402


def apply_wcs_to_coadd(coadd_file: str) -> str:
    new_file = coadd_file.replace('coadd_', 'coadd_corrected_')
    with fits.open(coadd_file) as hdul:
        wcs_hdr = hdul['SCI'].header

    im = datamodels.open(coadd_file)
    wcsobj = assign_gwcs(box_outdir=os.path.dirname(coadd_file), wcs_hdr=wcs_hdr)
    im.meta.wcs = wcsobj
    im.save(new_file)
    return new_file


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Apply GWCS from mosaic helpers to coadd i2d products.',
    )
    parser.add_argument(
        'coadds',
        nargs='*',
        help='Coadd FITS files (default: group_*/ref_*/coadd*i2d.fits).',
    )
    parser.add_argument(
        '--glob',
        dest='pattern',
        default='group_*/ref_*/coadd*i2d.fits',
        help='Glob used when no coadd paths are given.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    coadds = list(args.coadds) if args.coadds else glob.glob(args.pattern)
    if not coadds:
        print(f'ERROR: no coadds found (pattern={args.pattern!r}).', file=sys.stderr)
        return 1
    for path in coadds:
        out = apply_wcs_to_coadd(path)
        print(f'{path} -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
