"""Helpers for downloading JWST imaging from MAST."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager


from jwst123.mast import (
    download_jwst_observations,
    query_jwst,
    resolve_mast_token,
)


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


def query_mast_jwst(coord, outdir, radius, stage=2, token=None, instruments=None):
    '''
    Query MAST and download available JWST imaging.

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
        Optional MAST API token. When set (or via MAST_API_TOKEN), authenticates
        with ``Observations.login`` and includes proprietary observations.
    instruments : sequence of str or None
        Instrument name substrings (e.g. NIRCAM, MIRI). None uses defaults.

    Returns:
    -------
    int
        Number of observation product sets downloaded.
    '''
    os.makedirs(outdir, exist_ok=True)
    token = resolve_mast_token(token)
    kwargs = {'radius': radius, 'token': token}
    if instruments is not None:
        kwargs['instruments'] = instruments

    obs_table = query_jwst(coord, **kwargs)
    print(f'Found {len(obs_table)} JWST observation(s)')
    if len(obs_table) == 0:
        return 0

    # Pass token again so download authenticates even if called standalone.
    return download_jwst_observations(
        obs_table, outdir=outdir, stage=stage, token=token
    )
