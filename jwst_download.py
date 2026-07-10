import argparse
import os
import sys
from contextlib import contextmanager

from astropy import units as u

from common.mast import download_jwst_observations, query_jwst
from common.Util import parse_coord


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


def create_parser():
    '''
    Create a parser for the command line arguments

    Returns:
    -------
    parser : argparse.ArgumentParser
        arg parser
    '''
    parser = argparse.ArgumentParser(description='Download JWST data')
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
        '--token',
        default=None,
        type=str,
        help='MAST authorization token for proprietary data '
             '(see https://auth.mast.stsci.edu/info).',
    )
    return parser


def resolve_outdir(obj, outdir=None):
    '''
    Resolve the download output directory.

    Parameters:
    ----------
    obj : str
        Object name used for the default path
    outdir : str or None
        Explicit output directory. If None, uses ``jwst_data/<obj>``.

    Returns:
    -------
    str
        Absolute or relative output directory path
    '''
    if outdir is None:
        outdir = os.path.join('jwst_data', obj)
    return outdir


def query_mast_jwst(coord, outdir, radius, stage=2, token=None):
    '''
    Download available data from MAST for JWST NIRCAM

    Parameters:
    ----------
    coord : astropy.coordinates.SkyCoord
        target coordinates
    outdir : str
        output directory for downloads
    radius : astropy.units.Quantity
        search radius
    stage : int
        JWST calibration stage (2=CAL, 3=I2D)
    token : str or None
        Optional MAST API token. When set, authenticates via
        ``Observations.login`` and includes proprietary observations.

    Returns:
    -------
    None
    '''
    os.makedirs(outdir, exist_ok=True)
    obs_table = query_jwst(coord, radius=radius, token=token)
    # Session from mast_login persists; no need to pass token again for download.
    download_jwst_observations(obs_table, outdir=outdir, stage=stage)


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    coord = parse_coord(args.ra, args.dec)
    if coord is None:
        return 1

    outdir = resolve_outdir(args.obj, outdir=args.outdir)
    query_mast_jwst(
        coord,
        outdir=outdir,
        radius=args.radius * u.arcmin,
        stage=args.stage,
        token=args.token,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
