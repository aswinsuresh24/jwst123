#!/usr/bin/env python3
"""Download public JWST MIRI imaging products from MAST.

Directory layout (under ``--download-dir``)::

    <download-dir>/<FILTER>/<obsid>/mastDownload/JWST/<product>/...

Example::

    python jwst_download.py \\
      --ra 202.4699 --dec 47.1952 --obj M51 \\
      --download-dir /data/rwisenbaker/jwst_data/M51 \\
      --radius 6 --stage 2
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

from astropy import units as u

warnings.filterwarnings('ignore')


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
    from astroquery.mast import Observations
    from astropy.coordinates import SkyCoord


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Download JWST MIRI imaging data from MAST into '
            '<download-dir>/<FILTER>/<obsid>/mastDownload/...'
        )
    )
    parser.add_argument('--ra', type=str, required=True, help='RA of the target (deg)')
    parser.add_argument('--dec', type=str, required=True, help='Dec of the target (deg)')
    parser.add_argument(
        '--obj',
        type=str,
        default='target',
        help='Object name used only for logging (default: target)',
    )
    parser.add_argument(
        '--download-dir',
        type=Path,
        required=True,
        help=(
            'Base directory for downloads (e.g. /data/rwisenbaker/jwst_data/M51). '
            'Products land in <download-dir>/<FILTER>/<obsid>/mastDownload/...'
        ),
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=3.0,
        help='Search radius in arcminutes (default: 3.0)',
    )
    parser.add_argument(
        '--stage',
        type=int,
        default=2,
        choices=(2, 3),
        help='Calibration stage: 2=CAL, 3=I2D (default: 2)',
    )
    parser.add_argument(
        '--filters',
        type=str,
        default=None,
        help=(
            'Optional comma-separated filter list to keep '
            '(e.g. F560W,F770W). Default: all imaging filters.'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Query and list products without downloading.',
    )
    return parser


def normalize_filter_name(filt: str) -> str:
    """Turn MAST filter strings like ``F560W;CLEAR`` into a directory name."""
    name = str(filt).split(';')[0].strip().upper()
    name = re.sub(r'[^A-Z0-9_\-]+', '_', name)
    return name or 'UNKNOWN'


def filter_observations(obs_table, allowed_filters: set[str] | None):
    masks = [
        [str(t).upper() == 'JWST' for t in obs_table['obs_collection']],
        ['MIRI' in str(inst).upper() for inst in obs_table['instrument_name']],
        [str(f).upper() != 'DETECTION' for f in obs_table['filters']],
        [str(i).upper() != 'CALIBRATION' for i in obs_table['intentType']],
        [str(d).upper() == 'PUBLIC' for d in obs_table['dataRights']],
        [str(t).upper() == 'IMAGE' for t in obs_table['dataproduct_type']],
    ]
    keep = [all(row) for row in zip(*masks)]
    selected = obs_table[keep]

    if allowed_filters:
        filt_keep = [
            normalize_filter_name(f) in allowed_filters for f in selected['filters']
        ]
        selected = selected[filt_keep]
    return selected


def filter_products(product_list, stage: int):
    productmasks = [
        [str(p).upper() == 'SCIENCE' for p in product_list['productType']],
        # Imaging detector products only (exclude MRS / other MIRI modes).
        ['mirimage' in str(name).lower() for name in product_list['productFilename']],
    ]
    if stage == 2:
        productmasks.append(
            [str(t).upper() == 'CAL' for t in product_list['productSubGroupDescription']]
        )
        productmasks.append([int(c) == 2 for c in product_list['calib_level']])
    else:
        productmasks.append(
            [str(t).upper() == 'I2D' for t in product_list['productSubGroupDescription']]
        )
        productmasks.append([int(c) == 3 for c in product_list['calib_level']])

    keep = [all(row) for row in zip(*productmasks)]
    return product_list[keep]


def query_and_download(
    coord: SkyCoord,
    *,
    download_dir: Path,
    radius: u.Quantity,
    stage: int,
    obj: str,
    allowed_filters: set[str] | None = None,
    dry_run: bool = False,
) -> None:
    download_dir = Path(download_dir).expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f'Target: {obj}')
    print(f'Coordinates: {coord.to_string("hmsdms")}')
    print(f'Search radius: {radius}')
    print(f'Download directory: {download_dir}')
    print(f'Stage: {stage} ({"CAL" if stage == 2 else "I2D"})')
    if allowed_filters:
        print(f'Filters: {", ".join(sorted(allowed_filters))}')
    if dry_run:
        print('Dry run: no files will be downloaded')

    obs_table = Observations.query_region(coord, radius=radius)
    obs_table = obs_table.filled()
    obs_webb = filter_observations(obs_table, allowed_filters)
    print(f'Matched MIRI imaging observations: {len(obs_webb)}')

    n_obs_with_products = 0
    n_products = 0

    for obs in obs_webb:
        filt = normalize_filter_name(obs['filters'])
        obsid = str(obs['obsid'])
        product_list = Observations.get_product_list(obs)
        product_list = filter_products(product_list, stage)
        if len(product_list) == 0:
            continue

        n_obs_with_products += 1
        n_products += len(product_list)
        out_dir = download_dir / filt / obsid
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f'[{n_obs_with_products}] {filt}/{obsid}: '
            f'{len(product_list)} product(s) → {out_dir}'
        )
        for row in product_list:
            print(f'    {row["productFilename"]}')

        if dry_run:
            continue

        # astroquery writes <download_dir>/mastDownload/JWST/...
        Observations.download_products(
            product_list,
            download_dir=str(out_dir),
            extension='fits',
        )

    print(
        f'Done. Observations with products: {n_obs_with_products}; '
        f'total products: {n_products}'
    )


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    allowed = None
    if args.filters:
        allowed = {normalize_filter_name(f) for f in args.filters.split(',') if f.strip()}

    coord = SkyCoord(args.ra, args.dec, frame='icrs', unit='deg')
    query_and_download(
        coord,
        download_dir=args.download_dir,
        radius=args.radius * u.arcmin,
        stage=args.stage,
        obj=args.obj,
        allowed_filters=allowed,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
