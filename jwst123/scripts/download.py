#!/usr/bin/env python3
"""Download JWST imaging products from MAST."""

from __future__ import annotations

import argparse
import sys

from astropy import units as u

from jwst123.download import query_mast_jwst, resolve_outdir
from jwst123.util import parse_coord


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Download JWST imaging from MAST. Pass --token (or set MAST_API_TOKEN) '
            'to authenticate and include proprietary data, as in hst123.'
        ),
    )
    parser.add_argument('--ra', type=str, help='RA of the target', required=True)
    parser.add_argument('--dec', type=str, help='DEC of the target', required=True)
    parser.add_argument('--obj', type=str, help='Name of the object', required=True)
    parser.add_argument(
        '--outdir',
        default=None,
        type=str,
        help='Output directory for downloads (default: jwst_data/<obj>).',
    )
    parser.add_argument('--radius', type=float, default=3.0, help='Radius in arcminutes')
    parser.add_argument('--stage', type=int, default=2, help='Stage of the reduction')
    parser.add_argument(
        '--instruments',
        nargs='+',
        default=None,
        help='JWST instruments to include (default: NIRCAM MIRI).',
    )
    parser.add_argument(
        '--token',
        default=None,
        type=str,
        help=(
            'MAST authorization token for proprietary data '
            '(see https://auth.mast.stsci.edu/info). '
            'Also read from MAST_API_TOKEN or MAST_TOKEN if unset.'
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    coord = parse_coord(args.ra, args.dec)
    if coord is None:
        return 1

    outdir = resolve_outdir(args.obj, outdir=args.outdir)
    try:
        n = query_mast_jwst(
            coord,
            outdir=outdir,
            radius=args.radius * u.arcmin,
            stage=args.stage,
            token=args.token,
            instruments=args.instruments,
        )
    except RuntimeError as exc:
        print(f'ERROR: {exc}')
        return 1

    return 0 if n else 1


if __name__ == '__main__':
    raise SystemExit(main())
