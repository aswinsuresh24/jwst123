#!/usr/bin/env python3
"""
End-to-end MIRI–reference overlap + alignment pipeline.

Expected dataset layout under ``--data-dir``::

    <data-dir>/<FILTER>/<obsid>/mastDownload/JWST/*_mirimage/*_cal.fits
    <data-dir>/reference/group_*/ref_*/coadd*i2d.fits

Steps
-----
1. Discover MIRI ``*_cal.fits`` and reference ``coadd*i2d.fits`` under
   ``--data-dir`` (default: ``/data/rwisenbaker/jwst_data/M51``).
2. For each MIRI frame, record:
   - the reference with maximum footprint overlap, and
   - every reference with any nonzero overlap.
3. Align MIRI frames filter-by-filter from blue→red (F560W, then F770W,
   then F1000W, …). Within each filter, frames are aligned in parallel
   (``--workers``).
4. If reference-image alignment fails, or succeeds but exceeds
   ``--max-nircam-dispersion-mas`` (default 70 mas), fall back to relative
   MIRI→MIRI alignment against the successfully aligned frame that is closest
   in wavelength and has the largest sky overlap (using completed bluer
   filters, then same-filter successes that passed the quality cut). Absolute
   dispersion is the quadrature sum of the parent absolute dispersion and the
   new relative dispersion. Provenance (``ALGNMODE``, ``ALGNREF``, ``ALGNTO``)
   is written to the JHAT header (``REFERENCE`` or ``MIRI_REL``). If MIRI_REL
   also fails, the prior reference alignment product is kept.
5. Reject MIRI frames whose cumulative (union) reference footprint coverage
   of the MIRI ROI is below ``--min-ref-overlap-frac`` (default 0.02) before
   alignment (logged to console / overlap summaries only; they never enter
   ``alignment_dispersion`` / JHAT, including MIRI→MIRI fallback, and are
   omitted from ``alignment_summary.txt``).
6. Write ``<data-dir>/<name>_alignment_summary.txt`` only for frames that
   proceed to alignment, with per-frame ``SUCCESS``/``FAILURE`` status,
   ``ref_overlap_frac`` (unique MIRI-ROI fraction covered by the union of all
   overlapping reference footprints), ``align_mode`` (``REFERENCE`` or
   ``MIRI_REL``), filter, calibrator count, dispersion, aligned JHAT path, and
   provenance (updated live after each finished frame).

Example::

    python alignment_wrap.py \\
      --data-dir /data/rwisenbaker/jwst_data/M51 \\
      --plot --continue-on-error --workers 8

Legacy sequential pair mode (original ``alignment_wrap`` behavior)::

    python alignment_wrap.py --legacy-overlap-file overlap_summary.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import traceback
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


def _resolve_repo_root(explicit: Path | None = None) -> Path:
    """Locate the jwst123 checkout that contains the overlap/alignment modules."""
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not (root / 'image_overlap.py').is_file():
            raise FileNotFoundError(f'--repo does not look like jwst123: {root}')
        return root

    here = Path(__file__).resolve().parent
    candidates = [
        here,
        Path.cwd(),
        Path('/data/rwisenbaker/jwst123'),
        here.parent / 'jwst123',
        here.parent / 'rwisenbaker' / 'jwst123',
    ]
    for cand in candidates:
        if (cand / 'image_overlap.py').is_file() and (
            cand / 'alignment_dispersion.py'
        ).is_file():
            return cand.resolve()
    raise FileNotFoundError(
        'Could not locate jwst123 repo (need image_overlap.py and '
        'alignment_dispersion.py). Pass --repo /path/to/jwst123.'
    )


def _bootstrap_imports(repo: Path) -> None:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


@dataclass(frozen=True)
class FrameOverlaps:
    """Best reference plus every reference with nonzero overlap for one MIRI frame."""

    miri_path: str
    best: object  # image_overlap.BestOverlap
    overlapping: list  # list[image_overlap.OverlapResult]
    # Unique MIRI-ROI fraction covered by the union of all overlapping refs.
    union_overlap_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            'miri_path': self.miri_path,
            'best': {
                'ref_path': self.best.ref_path,
                'overlap_area': asdict(self.best.overlap_area),
            },
            'overlapping': [
                {
                    'ref_path': r.ref_path,
                    'overlap_area': asdict(r.overlap_area),
                    'ref_area': asdict(r.ref_area),
                }
                for r in self.overlapping
            ],
            'union_overlap_fraction': float(self.union_overlap_fraction),
        }


@dataclass
class AlignmentSummaryRow:
    """One row of the galaxy alignment summary table."""

    miri_path: str
    filter: str
    status: str
    n_calibrators: int | str
    dispersion_mas: float | str
    aligned_path: str = 'NA'
    align_mode: str = 'NA'
    original_ref: str = 'NA'
    aligned_to: str = 'NA'
    # Cumulative unique fraction of MIRI ROI covered by all overlapping refs.
    ref_overlap_frac: float | str = 'NA'

    def format_line(self, widths: dict[str, int]) -> str:
        disp = (
            f'{self.dispersion_mas:.3f}'
            if isinstance(self.dispersion_mas, float)
            else str(self.dispersion_mas)
        )
        ov = (
            f'{self.ref_overlap_frac:.4f}'
            if isinstance(self.ref_overlap_frac, float)
            else str(self.ref_overlap_frac)
        )
        ncal = str(self.n_calibrators)
        return (
            f'{self.miri_path:<{widths["miri_path"]}}  '
            f'{self.filter:<{widths["filter"]}}  '
            f'{self.status:<{widths["status"]}}  '
            f'{ov:>{widths["ref_overlap_frac"]}}  '
            f'{ncal:>{widths["n_calibrators"]}}  '
            f'{disp:>{widths["dispersion_mas"]}}  '
            f'{self.align_mode:<{widths["align_mode"]}}  '
            f'{self.aligned_path:<{widths["aligned_path"]}}  '
            f'{self.original_ref:<{widths["original_ref"]}}  '
            f'{self.aligned_to:<{widths["aligned_to"]}}'
        )


def read_miri_filter(miri_path: str) -> str:
    """Return FILTER from the MIRI FITS primary (or SCI) header."""
    from astropy.io import fits

    with fits.open(miri_path) as hdul:
        filt = hdul[0].header.get('FILTER')
        if not filt and 'SCI' in hdul:
            filt = hdul['SCI'].header.get('FILTER')
    return str(filt) if filt else 'UNKNOWN'


def find_jhat_product(outdir: Path, miri_path: str) -> Path | None:
    """Locate the JHAT FITS product for a MIRI frame."""
    stem = Path(miri_path).name.replace('_cal.fits', '_jhat.fits').replace(
        '_i2d.fits', '_jhat.fits'
    )
    candidate = outdir / stem
    if candidate.is_file():
        return candidate
    matches = sorted(outdir.glob('*jhat*.fits'))
    return matches[0] if matches else None


def _normalize_align_mode(align_mode: str | None) -> str:
    """Map legacy ``NIRCAM`` labels to ``REFERENCE``; otherwise uppercase."""
    mode = str(align_mode or 'NA').upper()
    if mode == 'NIRCAM':
        return 'REFERENCE'
    return mode


def _is_reference_quality_hold(row: AlignmentSummaryRow) -> bool:
    """
    True for a REFERENCE solution held as FAILURE after the dispersion cut.

    These rows keep finite metrics / JHAT paths so MIRI_REL can be tried, and
    so the REFERENCE product can be restored to SUCCESS if fallback fails.
    """
    return (
        row.status == 'FAILURE'
        and _normalize_align_mode(row.align_mode) == 'REFERENCE'
        and isinstance(row.dispersion_mas, float)
        and bool(row.aligned_path)
        and str(row.aligned_path) != 'NA'
    )


def harvest_alignment_metrics(
    miri_path: str,
    outdir: Path,
    *,
    ran_ok: bool,
    default_align_mode: str = 'NA',
    default_original_ref: str = 'NA',
    default_aligned_to: str = 'NA',
    ref_overlap_frac: float | str = 'NA',
) -> AlignmentSummaryRow:
    """
    Build a summary row from alignment products / headers.

    SUCCESS requires a JHAT product with a finite final mean dispersion
    (``JWDISPM`` / ``GADISPM``, stored in arcsec, reported in mas). Method is
    recorded separately in ``align_mode`` (``REFERENCE`` or ``MIRI_REL``).
    ``ref_overlap_frac`` is the unique MIRI-ROI fraction covered by the union
    of all overlapping reference footprints (geometry; independent of JHAT).
    """
    from astropy.io import fits

    filt = read_miri_filter(miri_path)
    empty = dict(
        miri_path=miri_path,
        filter=filt,
        status='FAILURE',
        n_calibrators='NA',
        dispersion_mas='NA',
        aligned_path='NA',
        align_mode='NA',
        original_ref='NA',
        aligned_to='NA',
        ref_overlap_frac=ref_overlap_frac,
    )
    if not ran_ok:
        return AlignmentSummaryRow(**empty)

    jhat = find_jhat_product(outdir, miri_path)
    if jhat is None:
        return AlignmentSummaryRow(**empty)
    aligned_path = str(jhat.resolve())

    with fits.open(jhat) as hdul:
        hdr = hdul[0].header
        if hdr.get('FILTER'):
            filt = str(hdr['FILTER'])
        disp_std = hdr.get('JWDISPS', hdr.get('GADISPS'))
        disp_mean = hdr.get('JWDISPM', hdr.get('GADISPM'))
        n_cal = hdr.get('JWNCAL', hdr.get('GANCAL'))
        align_mode = _normalize_align_mode(
            hdr.get('ALGNMODE', default_align_mode) or default_align_mode
        )
        original_ref = str(hdr.get('ALGNREF', default_original_ref) or default_original_ref)
        aligned_to = str(hdr.get('ALGNTO', default_aligned_to) or default_aligned_to)

    # Soft-failure path in align_jwst_image writes JWDISPS as the string 'NaN'.
    rejected = isinstance(disp_std, str) and disp_std.upper() == 'NAN'

    dispersion_mas: float | str = 'NA'
    try:
        if disp_mean is not None and not (
            isinstance(disp_mean, str) and str(disp_mean).upper() == 'NAN'
        ):
            disp_val = float(disp_mean)
            if math.isfinite(disp_val):
                dispersion_mas = disp_val * 1000.0
    except (TypeError, ValueError):
        dispersion_mas = 'NA'

    n_calibrators: int | str = 'NA'
    try:
        if n_cal is not None and not (
            isinstance(n_cal, str) and str(n_cal).upper() == 'NAN'
        ):
            n_calibrators = int(n_cal)
    except (TypeError, ValueError):
        n_calibrators = 'NA'

    if rejected or dispersion_mas == 'NA':
        return AlignmentSummaryRow(**empty)

    return AlignmentSummaryRow(
        miri_path=miri_path,
        filter=filt,
        status='SUCCESS',
        n_calibrators=n_calibrators,
        dispersion_mas=dispersion_mas,
        aligned_path=aligned_path,
        align_mode=align_mode,
        original_ref=original_ref,
        aligned_to=aligned_to,
        ref_overlap_frac=ref_overlap_frac,
    )


def write_alignment_summary(
    rows: list[AlignmentSummaryRow],
    outfile: Path,
) -> Path:
    """Write a plain ASCII alignment summary table and return its path."""
    outfile = Path(outfile).expanduser().resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        'miri_path': 'miri_path',
        'filter': 'filter',
        'status': 'status',
        'ref_overlap_frac': 'ref_overlap_frac',
        'n_calibrators': 'n_calibrators',
        'dispersion_mas': 'dispersion_mas',
        'align_mode': 'align_mode',
        'aligned_path': 'aligned_path',
        'original_ref': 'original_ref',
        'aligned_to': 'aligned_to',
    }

    def _disp_len(r: AlignmentSummaryRow) -> int:
        if isinstance(r.dispersion_mas, float):
            return len(f'{r.dispersion_mas:.3f}')
        return len(str(r.dispersion_mas))

    def _ov_len(r: AlignmentSummaryRow) -> int:
        if isinstance(r.ref_overlap_frac, float):
            return len(f'{r.ref_overlap_frac:.4f}')
        return len(str(r.ref_overlap_frac))

    widths = {
        'miri_path': max(
            [len(headers['miri_path'])] + [len(r.miri_path) for r in rows] + [1]
        ),
        'filter': max(
            [len(headers['filter'])] + [len(r.filter) for r in rows] + [1]
        ),
        'status': max(
            [len(headers['status'])] + [len(r.status) for r in rows] + [1]
        ),
        'ref_overlap_frac': max(
            [len(headers['ref_overlap_frac'])]
            + [_ov_len(r) for r in rows]
            + [1]
        ),
        'n_calibrators': max(
            [len(headers['n_calibrators'])]
            + [len(str(r.n_calibrators)) for r in rows]
            + [1]
        ),
        'dispersion_mas': max(
            [len(headers['dispersion_mas'])]
            + [_disp_len(r) for r in rows]
            + [1]
        ),
        'align_mode': max(
            [len(headers['align_mode'])] + [len(r.align_mode) for r in rows] + [1]
        ),
        'aligned_path': max(
            [len(headers['aligned_path'])]
            + [len(r.aligned_path) for r in rows]
            + [1]
        ),
        'original_ref': max(
            [len(headers['original_ref'])]
            + [len(r.original_ref) for r in rows]
            + [1]
        ),
        'aligned_to': max(
            [len(headers['aligned_to'])] + [len(r.aligned_to) for r in rows] + [1]
        ),
    }

    header = (
        f'{headers["miri_path"]:<{widths["miri_path"]}}  '
        f'{headers["filter"]:<{widths["filter"]}}  '
        f'{headers["status"]:<{widths["status"]}}  '
        f'{headers["ref_overlap_frac"]:>{widths["ref_overlap_frac"]}}  '
        f'{headers["n_calibrators"]:>{widths["n_calibrators"]}}  '
        f'{headers["dispersion_mas"]:>{widths["dispersion_mas"]}}  '
        f'{headers["align_mode"]:<{widths["align_mode"]}}  '
        f'{headers["aligned_path"]:<{widths["aligned_path"]}}  '
        f'{headers["original_ref"]:<{widths["original_ref"]}}  '
        f'{headers["aligned_to"]:<{widths["aligned_to"]}}'
    )
    lines = [header, '-' * len(header)]
    lines.extend(r.format_line(widths) for r in rows)
    lines.append('')
    # Atomic replace so a live reader never sees a partially written table.
    tmp = outfile.with_name(outfile.name + '.tmp')
    tmp.write_text('\n'.join(lines))
    tmp.replace(outfile)
    return outfile


def _looks_like_filter(token: str) -> bool:
    """True for names like F560W / F1000W."""
    t = str(token).upper()
    return (
        len(t) >= 3
        and t.startswith('F')
        and t.endswith('W')
        and any(ch.isdigit() for ch in t)
    )


def discover_miri_images(data_dir: Path) -> list[str]:
    """
    Sorted MIRI cal images under ``data_dir``.

    Preferred layout::

        <data-dir>/<FILTER>/<obsid>/mastDownload/JWST/*_mirimage/*_cal.fits

    Also accepts the older combined directory name::

        <data-dir>/<FILTER>_<obsid>/mastDownload/JWST/*_mirimage/*_cal.fits
    """
    data_dir = Path(data_dir)
    patterns = (
        '*/*/mastDownload/JWST/*_mirimage/*_cal.fits',
        '*/mastDownload/JWST/*_mirimage/*_cal.fits',
    )
    found: set[str] = set()
    for pattern in patterns:
        for path in data_dir.glob(pattern):
            found.add(str(path.resolve()))
    return sorted(found)


def discover_ref_images(data_dir: Path) -> list[str]:
    """Sorted reference coadds under ``data_dir/reference/group_*/ref_*/``."""
    return sorted(
        str(p.resolve())
        for p in Path(data_dir).glob('reference/group_*/ref_*/coadd*i2d.fits')
    )


def parse_filters_arg(filters: str | None) -> list[str] | None:
    """
    Parse a comma-separated filter list (e.g. ``F560W`` or ``F560W,F770W``).

    Returns ``None`` when no filter restriction is requested.
    """
    if filters is None or not str(filters).strip():
        return None
    parsed = []
    for part in str(filters).split(','):
        name = part.strip().upper()
        if not name:
            continue
        parsed.append(name)
    return parsed or None


def filter_name_from_miri_path(miri_path: str) -> str | None:
    """
    Infer the MIRI filter from the MAST download path.

    Supports::

        .../<FILTER>/<obsid>/mastDownload/JWST/...
        .../<FILTER>_<obsid>/mastDownload/JWST/...
    """
    parts = Path(miri_path).parts
    for i, part in enumerate(parts):
        if part != 'mastDownload' or i < 1:
            continue
        parent = parts[i - 1]
        # Preferred: <FILTER>/<obsid>/mastDownload
        if i >= 2 and str(parent).isdigit():
            cand = parts[i - 2].upper()
            if _looks_like_filter(cand):
                return cand
        # Legacy: <FILTER>_<obsid>/mastDownload
        token = parent.split('_', 1)[0].upper()
        if _looks_like_filter(token):
            return token
    return None


def filter_miri_images(
    miri_images: list[str],
    filters: list[str] | None,
) -> list[str]:
    """Keep MIRI images whose path filter is in ``filters`` (case-insensitive)."""
    if not filters:
        return list(miri_images)
    wanted = {f.upper() for f in filters}
    selected = []
    for path in miri_images:
        name = filter_name_from_miri_path(path)
        if name is not None and name in wanted:
            selected.append(path)
    return selected


def find_frame_overlaps(
    miri_images: list[str],
    refs: list[str],
    *,
    MirIFootprint,
    BestOverlap,
    compute_overlap,
) -> list[FrameOverlaps]:
    """Compute best-overlap and all-nonzero-overlap refs for each MIRI frame."""
    results: list[FrameOverlaps] = []

    for image in miri_images:
        print(f'MIRI: {image}')
        miri = MirIFootprint.from_fits(image)
        print(f'  illuminated S_REGION: {miri.s_region.to_string()}')
        print(
            f'  WCS pixel solid angle: {miri.pixel_area_arcmin2:.8e} '
            f'arcmin^2 / pixel'
        )
        print(f'  illuminated area: {miri.area.format()}')

        overlapping = []
        best = None

        for ref in refs:
            try:
                result = compute_overlap(miri, ref)
            except Exception as exc:
                print(f'  FAILED for ref {ref}: {exc}')
                continue

            print(f'  ref: {ref}')
            print(f'    S_REGION: {result.ref_s_region.to_string()}')
            print(f'    ref area: {result.ref_area.format()}')
            print(f'    overlap area: {result.overlap_area.format()}')

            if result.overlap_area.pixels2 > 0.0:
                overlapping.append(result)

            if best is None or result.overlap_area.pixels2 > best.overlap_area.pixels2:
                best = BestOverlap(
                    miri_path=image,
                    ref_path=result.ref_path,
                    overlap_area=result.overlap_area,
                )

        if best is None or best.overlap_area.pixels2 <= 0.0:
            best = BestOverlap(
                miri_path=image,
                ref_path=None,
                overlap_area=miri.metrics(0.0),
            )

        overlapping.sort(key=lambda r: r.overlap_area.pixels2, reverse=True)

        from image_overlap import compute_cumulative_overlap_fraction

        union_frac = compute_cumulative_overlap_fraction(
            miri, [r.ref_path for r in overlapping]
        )

        print(
            f'Overlap maximized: MIRI image: {best.miri_path}, '
            f'Reference image: {best.ref_path}, '
            f'Max overlap area: {best.overlap_area.pixels2:.3f} pixels^2 '
            f'({best.overlap_area.arcmin2:.6f} arcmin^2, '
            f'{best.overlap_area.fraction_of_miri_roi:.4f} of MIRI illuminated ROI); '
            f'{len(overlapping)} reference(s) with any overlap; '
            f'union coverage {union_frac:.4f} of MIRI ROI'
        )
        if overlapping:
            print('  References with any overlap (largest first):')
            for result in overlapping:
                print(
                    f'    {result.ref_path}: '
                    f'{result.overlap_area.pixels2:.3f} pixels^2 '
                    f'({result.overlap_area.fraction_of_miri_roi:.4f} of MIRI ROI)'
                )
        print()

        results.append(
            FrameOverlaps(
                miri_path=image,
                best=best,
                overlapping=overlapping,
                union_overlap_fraction=union_frac,
            )
        )

    return results


def write_overlap_summaries(
    frames: list[FrameOverlaps],
    outdir: Path,
) -> tuple[Path, Path]:
    """Write text + JSON summaries of best and any-overlap references."""
    outdir.mkdir(parents=True, exist_ok=True)
    txt_path = outdir / 'overlap_summary.txt'
    json_path = outdir / 'overlap_summary.json'

    lines: list[str] = []
    for frame in frames:
        best = frame.best
        lines.append(
            f'Overlap maximized: MIRI image: {best.miri_path}, '
            f'Reference image: {best.ref_path}, '
            f'Max overlap area: {best.overlap_area.pixels2:.3f} pixels^2 '
            f'({best.overlap_area.arcmin2:.6f} arcmin^2, '
            f'{best.overlap_area.fraction_of_miri_roi:.4f} of MIRI illuminated ROI); '
            f'{len(frame.overlapping)} reference(s) with any overlap; '
            f'union coverage {frame.union_overlap_fraction:.4f} of MIRI ROI\n'
        )
        for result in frame.overlapping:
            lines.append(
                f'  any-overlap: {result.ref_path} '
                f'{result.overlap_area.pixels2:.3f} pixels^2 '
                f'({result.overlap_area.fraction_of_miri_roi:.4f} of MIRI ROI)\n'
            )
        lines.append('\n')

    txt_path.write_text(''.join(lines))
    payload = {
        'n_miri': len(frames),
        'n_with_overlap': sum(1 for f in frames if f.best.ref_path is not None),
        'frames': [f.to_dict() for f in frames],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f'Wrote overlap summary: {txt_path}')
    print(f'Wrote overlap JSON:    {json_path}')
    return txt_path, json_path


def alignment_outdir_for(miri_path: str) -> Path:
    """``<directory of MIRI cal file>/alignment_output``."""
    return Path(miri_path).resolve().parent / 'alignment_output'


def create_parser(default_data_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Run MIRI/reference overlap matching then align each MIRI frame '
            'to its best-overlap reference. Dataset root is ``--data-dir`` '
            'with layout <FILTER>/<obsid>/mastDownload/... plus reference/.'
        )
    )
    parser.add_argument(
        '--repo',
        type=Path,
        default=None,
        help='Path to the jwst123 repo (default: auto-detect).',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help=(
            'Dataset root containing <FILTER>/<obsid>/mastDownload/... MIRI '
            'cals and reference/ coadds '
            f'(default: {default_data_dir}).'
        ),
    )
    parser.add_argument(
        '--galaxy',
        default=None,
        help=(
            'Optional dataset label used in summary filenames '
            '(default: basename of --data-dir).'
        ),
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help=(
            'Deprecated. Parent of a galaxy subdirectory; used only when '
            '--data-dir is omitted as <data-root>/<galaxy>.'
        ),
    )
    parser.add_argument(
        '--overlap-outdir',
        type=Path,
        default=None,
        help=(
            'Where to write overlap_summary.txt/json '
            '(default: <data-dir>/overlap_output).'
        ),
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional cap on number of MIRI frames (useful for a smoke test).',
    )
    parser.add_argument(
        '--filters',
        type=str,
        default=None,
        help=(
            'Comma-separated MIRI filters to process '
            '(e.g. F560W or F560W,F770W). Default: all filters.'
        ),
    )
    parser.add_argument(
        '--overlap-only',
        action='store_true',
        help='Only compute overlaps; skip alignment.',
    )
    parser.add_argument(
        '--align-only',
        action='store_true',
        help='Skip overlap recompute; load overlap_summary.json and align.',
    )
    parser.add_argument(
        '--overlap-json',
        type=Path,
        default=None,
        help='Existing overlap_summary.json to use with --align-only.',
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
        help='Enable JHAT diagnostic plots during alignment.',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose JHAT / alignment output.',
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue aligning remaining frames if one fails.',
    )
    parser.add_argument(
        '--match-radius',
        type=float,
        default=0.1,
        help=(
            'Sky match radius in arcsec when merging overlapping reference '
            'catalogs (default: 0.1).'
        ),
    )
    parser.add_argument(
        '--no-clip-footprint',
        action='store_true',
        help=(
            'Do not clip the master catalog to the MIRI illuminated footprint '
            '(default: clip so Nbright prefers in-frame stars).'
        ),
    )
    parser.add_argument(
        '--no-refine',
        action='store_true',
        help='Disable iterative outlier-clipping refinement after the initial JHAT alignment.',
    )
    parser.add_argument(
        '--refine-sigma',
        type=float,
        default=2.0,
        help='Sigma threshold for iterative residual clipping (default: 2.0).',
    )
    parser.add_argument(
        '--refine-max-iter',
        type=int,
        default=5,
        help='Maximum iterative refinement iterations (default: 5).',
    )
    parser.add_argument(
        '--no-fallback',
        action='store_true',
        help=(
            'Disable MIRI→MIRI relative fallback when NIRCam alignment fails '
            '(default: try fallback using the closest-wavelength overlapping '
            'successfully aligned MIRI frame).'
        ),
    )
    parser.add_argument(
        '--max-nircam-dispersion-mas',
        type=float,
        default=70.0,
        help=(
            'If NIRCam alignment succeeds but final mean dispersion exceeds '
            'this threshold (mas), treat it as a quality failure and attempt '
            'MIRI→MIRI relative fallback (default: 70). Set <= 0 to disable.'
        ),
    )
    parser.add_argument(
        '--min-ref-overlap-frac',
        type=float,
        default=0.02,
        help=(
            'Minimum cumulative (union) fraction of the MIRI illuminated ROI '
            'that must be covered by overlapping reference footprints to '
            'attempt alignment (default: 0.02). Frames below this threshold '
            'are skipped and omitted from the alignment summary.'
        ),
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help=(
            'Number of parallel alignment workers within each filter wave '
            '(default: 1). Filters are still processed blue→red sequentially '
            'so fallback can use completed bluer alignments.'
        ),
    )
    parser.add_argument(
        '--legacy-overlap-file',
        type=Path,
        default=None,
        help=(
            'Run the original sequential MIRI/reference pair loop against an '
            'overlap text file (lines containing "Overlap maximized") instead '
            'of the full filter-wave pipeline. Writes successful_alignments.txt '
            'and failed_alignments.txt under the current working directory.'
        ),
    )
    parser.add_argument(
        '--legacy-outdir',
        type=Path,
        default=Path('alignment_output_auto'),
        help=(
            'Output root for --legacy-overlap-file pair products '
            '(default: alignment_output_auto).'
        ),
    )
    return parser


def run_legacy_overlap_file_pipeline(
    overlap_file: Path,
    *,
    outdir: Path = Path('alignment_output_auto'),
    success_file: Path = Path('successful_alignments.txt'),
    fail_file: Path = Path('failed_alignments.txt'),
    filters: tuple[str, ...] = ('F560W', 'F770W'),
    alignment_script: str = 'alignment_dispersion.py',
    plot: bool = True,
    verbose: bool = True,
) -> int:
    """
    Original ``alignment_wrap`` behavior: align each MIRI/reference pair listed
    in an overlap text file via ``alignment_dispersion.py``.
    """
    overlap_file = Path(overlap_file).expanduser().resolve()
    if not overlap_file.is_file():
        print(f'ERROR: legacy overlap file not found: {overlap_file}', file=sys.stderr)
        return 1

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    success_file = Path(success_file)
    fail_file = Path(fail_file)
    filt_tokens = tuple(f.upper() for f in filters)

    n_ok = 0
    n_fail = 0
    with open(success_file, 'w') as success, open(fail_file, 'w') as failed:
        with open(overlap_file) as file:
            for line in file:
                if 'Overlap maximized' not in line:
                    continue
                if not any(tok in line for tok in filt_tokens):
                    continue

                align_image = line.split('MIRI image: ')[1].split(
                    ', Reference image: '
                )[0]
                ref_image = line.split(', Reference image: ')[1].split(
                    ', Max overlap area'
                )[0]

                align_name = Path(align_image).stem
                ref_name = Path(ref_image).stem
                pair_outdir = os.path.join(
                    str(outdir), f'{align_name}_aligned_to_{ref_name}'
                )

                command = [
                    sys.executable,
                    alignment_script,
                    '--ref',
                    ref_image,
                    '--align',
                    align_image,
                    '--outdir',
                    pair_outdir,
                ]
                if plot:
                    command.append('--plot')
                if verbose:
                    command.append('--verbose')

                result = subprocess.run(command)
                if result.returncode == 0:
                    success.write(
                        f'MIRI image: {align_image}, \n'
                        f'Reference image: {ref_image}\n\n'
                    )
                    success.flush()
                    n_ok += 1
                else:
                    failed.write(
                        f'MIRI image: {align_image}, \n'
                        f'Reference image: {ref_image}\n\n'
                    )
                    failed.flush()
                    n_fail += 1

    print('Done')
    print(f'Successful pairs written to {success_file} ({n_ok})')
    print(f'Failed pairs written to {fail_file} ({n_fail})')
    return 1 if n_fail else 0


def resolve_data_dir(
    *,
    data_dir: Path | None,
    data_root: Path | None,
    galaxy: str | None,
    default_data_dir: Path,
) -> tuple[Path, str]:
    """
    Resolve ``(data_dir, dataset_label)``.

    Preference order for the dataset root:
      1. ``--data-dir``
      2. ``--data-root`` / ``--galaxy`` (legacy)
      3. ``default_data_dir``
    """
    if data_dir is not None:
        root = Path(data_dir).expanduser().resolve()
    elif data_root is not None:
        label = galaxy or 'M51'
        root = (Path(data_root).expanduser().resolve() / label).resolve()
    else:
        root = Path(default_data_dir).expanduser().resolve()

    label = galaxy or root.name
    return root, label


def run_overlaps(
    args: argparse.Namespace,
    *,
    MirIFootprint,
    BestOverlap,
    compute_overlap,
) -> list[FrameOverlaps]:
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    filters = parse_filters_arg(getattr(args, 'filters', None))
    miri_images = filter_miri_images(discover_miri_images(data_dir), filters)
    refs = discover_ref_images(data_dir)
    if args.limit is not None:
        miri_images = miri_images[: args.limit]

    if not miri_images:
        msg = (
            f'No MIRI *_cal.fits found under '
            f'{data_dir}/<FILTER>/<obsid>/mastDownload/JWST/'
        )
        if filters:
            msg += f' for filters {",".join(filters)}'
        raise FileNotFoundError(msg)
    if not refs:
        raise FileNotFoundError(
            f'No reference coadd*i2d.fits found under {data_dir}/reference/'
        )

    print(f'Repo:              {args.repo}')
    print(f'Data dir:          {data_dir}')
    print(f'Dataset label:     {args.galaxy}')
    print(f'Filters:           {", ".join(filters) if filters else "ALL"}')
    print(f'MIRI images:       {len(miri_images)}')
    print(f'Reference images:  {len(refs)}')
    print()

    frames = find_frame_overlaps(
        miri_images,
        refs,
        MirIFootprint=MirIFootprint,
        BestOverlap=BestOverlap,
        compute_overlap=compute_overlap,
    )
    overlap_outdir = Path(
        args.overlap_outdir or (data_dir / 'overlap_output')
    ).expanduser().resolve()
    write_overlap_summaries(frames, overlap_outdir)
    return frames


def _frame_ref_images(frame: FrameOverlaps | dict) -> tuple[str, list[str], str | None]:
    """Return ``(miri_path, ordered_ref_images, best_ref)`` for one overlap frame."""
    if isinstance(frame, dict):
        miri_path = frame['miri_path']
        best_ref = frame['best']['ref_path']
        overlapping = [r['ref_path'] for r in frame.get('overlapping', [])]
    else:
        miri_path = frame.miri_path
        best_ref = frame.best.ref_path
        overlapping = [r.ref_path for r in frame.overlapping]

    ref_images = overlapping or ([best_ref] if best_ref else [])
    ordered: list[str] = []
    for path in ([best_ref] if best_ref else []) + list(ref_images):
        if path and path not in ordered:
            ordered.append(path)
    return miri_path, ordered, best_ref


def _frame_ref_overlap_frac(frame: FrameOverlaps | dict) -> float:
    """
    Unique MIRI-ROI fraction covered by the union of overlapping references.

    Uses a cached ``union_overlap_fraction`` when present; otherwise recomputes
    from the overlapping reference paths.
    """
    if isinstance(frame, dict):
        cached = frame.get('union_overlap_fraction')
        if cached is not None:
            try:
                return float(cached)
            except (TypeError, ValueError):
                pass
        miri_path, ref_images, _best = _frame_ref_images(frame)
    else:
        try:
            return float(frame.union_overlap_fraction)
        except (TypeError, ValueError, AttributeError):
            pass
        miri_path, ref_images, _best = _frame_ref_images(frame)

    if not ref_images:
        return 0.0
    from image_overlap import MirIFootprint, compute_cumulative_overlap_fraction

    return compute_cumulative_overlap_fraction(
        MirIFootprint.from_fits(miri_path), ref_images
    )


def frame_has_nircam_overlap(
    frame: FrameOverlaps | dict,
    *,
    min_ref_overlap_frac: float = 0.02,
) -> bool:
    """
    True if cumulative reference coverage of the MIRI ROI meets the threshold.

    ``min_ref_overlap_frac`` is the unique (union) fraction of the MIRI
    illuminated footprint covered by all overlapping reference images.
    """
    _miri_path, ref_images, best_ref = _frame_ref_images(frame)
    if not ref_images or best_ref is None:
        return False
    return _frame_ref_overlap_frac(frame) >= float(min_ref_overlap_frac)


def reject_zero_nircam_overlap_frames(
    frames: list[FrameOverlaps] | list[dict],
    *,
    min_ref_overlap_frac: float = 0.02,
) -> tuple[list[FrameOverlaps] | list[dict], int]:
    """
    Drop MIRI frames with insufficient cumulative reference footprint overlap.

    Frames whose unique (union) reference coverage of the MIRI ROI is below
    ``min_ref_overlap_frac`` are rejected. Returns ``(kept_frames, n_rejected)``.
    Rejected frames remain in overlap summaries only — they are not written to
    ``alignment_summary.txt`` and are not passed to alignment (including
    MIRI→MIRI fallback).
    """
    min_frac = float(min_ref_overlap_frac)
    kept: list[FrameOverlaps | dict] = []
    n_rejected = 0

    for frame in frames:
        miri_path, _ref_images, best_ref = _frame_ref_images(frame)
        filt = filter_name_from_miri_path(miri_path) or read_miri_filter(miri_path)
        ov_frac = _frame_ref_overlap_frac(frame)
        if best_ref and ov_frac >= min_frac:
            kept.append(frame)
            continue

        n_rejected += 1
        print(
            f'REJECT {Path(miri_path).name}  {filt}  '
            f'ref_overlap_frac={ov_frac:.4f} < {min_frac:.4f} '
            f'(excluded from alignment)',
            flush=True,
        )

    if n_rejected:
        print(
            f'Rejected {n_rejected} MIRI frame(s) with '
            f'ref_overlap_frac < {min_frac:.4f}; '
            f'{len(kept)} frame(s) remain for alignment',
            flush=True,
        )
    return kept, n_rejected


def _group_frames_by_filter(
    frames: list[FrameOverlaps] | list[dict],
) -> OrderedDict[str, list[FrameOverlaps | dict]]:
    """Group frames by filter, preserving blue→red order of first appearance."""
    from alignment_fallback import sort_frames_blue_to_red

    ordered = sort_frames_blue_to_red(
        frames, filter_from_path=filter_name_from_miri_path
    )
    groups: OrderedDict[str, list[FrameOverlaps | dict]] = OrderedDict()
    for frame in ordered:
        miri_path, _, _ = _frame_ref_images(frame)
        filt = filter_name_from_miri_path(miri_path) or read_miri_filter(miri_path)
        groups.setdefault(filt, []).append(frame)
    return groups


def _format_worker_done(result) -> str:
    """One-line DONE status for an alignment worker result."""
    base = Path(result.miri_path).name
    filt = (
        result.row.get('filter')
        or getattr(result, 'filter', None)
        or 'UNKNOWN'
    )
    status = str(result.row.get('status', 'FAILURE'))
    mode = _normalize_align_mode(result.row.get('align_mode'))
    disp = result.row.get('dispersion_mas', 'NA')
    disp_s = f'{disp:.3f}' if isinstance(disp, float) else str(disp)
    if status == 'SUCCESS':
        return (
            f'DONE  {base}  {filt}  SUCCESS  align_mode={mode}  '
            f'dispersion_mas={disp_s}'
        )
    if (
        status == 'FAILURE'
        and mode == 'REFERENCE'
        and isinstance(disp, float)
        and str(result.row.get('aligned_path', 'NA')) != 'NA'
    ):
        return (
            f'DONE  {base}  {filt}  FAILURE  align_mode={mode}  '
            f'dispersion_mas={disp_s} (over threshold; try MIRI_REL)'
        )
    if status in ('SKIP', 'REJECTED'):
        return f'DONE  {base}  {filt}  {status}'
    return f'DONE  {base}  {filt}  FAILURE'


def _needs_miri_fallback(row: AlignmentSummaryRow) -> bool:
    """True if a frame should enter MIRI_REL after the reference-align pass."""
    return row.status not in ('SUCCESS', 'REJECTED', 'SKIP')


def _run_jobs_parallel(
    jobs: list[dict],
    worker,
    *,
    workers: int,
    label: str,
    on_result=None,
) -> list:
    """
    Run picklable worker jobs with up to ``workers`` processes (or serially).

    Emits a brief ``START`` line when each job begins and invokes ``on_result``
    (if given) as each job finishes so the caller can log ``DONE`` / update
    the summary without interleaving JHAT chatter.
    """
    import multiprocessing as mp

    from alignment_parallel import AlignWorkerResult

    if not jobs:
        return []

    n_workers = max(1, int(workers))
    print(
        f'{label}: {len(jobs)} job(s), workers={min(n_workers, len(jobs))}',
        flush=True,
    )

    def _start_line(job: dict) -> None:
        print(f'START {Path(job["miri_path"]).name}', flush=True)

    def _handle(result) -> None:
        if on_result is not None:
            on_result(result)

    if n_workers == 1 or len(jobs) == 1:
        results = []
        for job in jobs:
            _start_line(job)
            result = worker(job)
            results.append(result)
            _handle(result)
        return results

    results: list[AlignWorkerResult | None] = [None] * len(jobs)
    # spawn avoids fork+OpenMP/BLAS deadlocks after heavy scientific imports
    from alignment_parallel import worker_initializer

    repo = str(jobs[0].get('repo') or '')
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(
        max_workers=min(n_workers, len(jobs)),
        mp_context=ctx,
        initializer=worker_initializer,
        initargs=(repo,),
    ) as pool:
        future_map = {}
        for i, job in enumerate(jobs):
            _start_line(job)
            future_map[pool.submit(worker, job)] = i
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                job = jobs[idx]
                results[idx] = AlignWorkerResult(
                    miri_path=job['miri_path'],
                    filter=job['filter'],
                    mode=job.get('mode', label),
                    ok=False,
                    row=asdict(
                        AlignmentSummaryRow(
                            miri_path=job['miri_path'],
                            filter=job['filter'],
                            status='FAILURE',
                            n_calibrators='NA',
                            dispersion_mas='NA',
                            aligned_path='NA',
                            ref_overlap_frac=job.get('ref_overlap_frac', 'NA'),
                        )
                    ),
                    error=str(exc),
                    message=f'Worker crashed: {exc}',
                )
            _handle(results[idx])
    return [r for r in results if r is not None]


def align_from_frames(
    frames: list[FrameOverlaps] | list[dict],
    *,
    run_alignment,
    nbright: int,
    plot: bool,
    verbose: bool,
    continue_on_error: bool,
    cache_dir: Path | None = None,
    match_radius_arcsec: float = 0.1,
    clip_to_align_footprint: bool = True,
    refine: bool = True,
    refine_sigma: float = 2.0,
    refine_max_iter: int = 5,
    fallback: bool = True,
    max_nircam_dispersion_mas: float | None = 70.0,
    min_ref_overlap_frac: float = 0.02,
    summary_outfile: Path | None = None,
    workers: int = 1,
    repo: Path | None = None,
) -> tuple[int, list[AlignmentSummaryRow]]:
    """
    Align MIRI frames filter-by-filter (blue→red), parallel within each filter.

    For each filter wave:
      1. Align all frames to overlapping reference images in parallel
         (``align_mode=REFERENCE`` on success)
      2. Run MIRI→MIRI fallback in parallel for failures and for REFERENCE
         solutions whose dispersion exceeds ``max_nircam_dispersion_mas``
         (``align_mode=MIRI_REL``; parents = successes from bluer filters and
         from this filter's completed REFERENCE successes that passed the
         quality cut)
      3. Optionally repeat fallback once so same-filter MIRI_REL successes can
         parent remaining failures

    Summary ``status`` is binary ``SUCCESS``/``FAILURE``; method is in
    ``align_mode``. ``run_alignment`` is accepted for API compatibility;
    workers import it themselves. If ``summary_outfile`` is set, the summary
    is rewritten after each finished frame.
    """
    del run_alignment  # workers import alignment_dispersion.run_alignment

    from alignment_fallback import (
        SuccessfulAlignment,
        select_fallback_parent,
    )
    from alignment_parallel import run_fallback_align_job, run_nircam_align_job

    repo_str = str((repo or _resolve_repo_root(None)).resolve())
    workers = max(1, int(workers))
    if max_nircam_dispersion_mas is not None and max_nircam_dispersion_mas <= 0:
        max_nircam_dispersion_mas = None

    # Drop low-overlap frames before any alignment work. These remain in
    # overlap_summary* only and are omitted from alignment_summary.txt.
    frames, n_rejected = reject_zero_nircam_overlap_frames(
        frames, min_ref_overlap_frac=min_ref_overlap_frac
    )
    groups = _group_frames_by_filter(frames)

    failures = 0
    n_ok = 0
    n_fallback = 0
    rows: list[AlignmentSummaryRow] = []
    row_by_miri: dict[str, AlignmentSummaryRow] = {}
    successes: list[SuccessfulAlignment] = []

    def flush_summary() -> None:
        if summary_outfile is None:
            return
        write_alignment_summary(rows, summary_outfile)

    def record_result(result, *, count_fallback: bool = False) -> None:
        nonlocal n_ok, n_fallback, failures
        row = AlignmentSummaryRow(**result.row)
        prev = row_by_miri.get(result.miri_path)

        # Keep a prior REFERENCE quality-hold product when MIRI_REL fails.
        if (
            not result.ok
            and result.mode == 'fallback'
            and prev is not None
            and _is_reference_quality_hold(prev)
        ):
            kept = AlignmentSummaryRow(
                miri_path=prev.miri_path,
                filter=prev.filter,
                status='SUCCESS',
                n_calibrators=prev.n_calibrators,
                dispersion_mas=prev.dispersion_mas,
                aligned_path=prev.aligned_path,
                align_mode=_normalize_align_mode(prev.align_mode),
                original_ref=prev.original_ref,
                aligned_to=prev.aligned_to,
                ref_overlap_frac=prev.ref_overlap_frac,
            )
            idx = rows.index(prev)
            rows[idx] = kept
            row_by_miri[result.miri_path] = kept
            n_ok += 1
            print(
                f'DONE  {Path(result.miri_path).name}  {result.filter}  '
                f'SUCCESS  align_mode=REFERENCE  '
                f'dispersion_mas={kept.dispersion_mas:.3f} '
                f'(kept after MIRI_REL failed quality-cut fallback)',
                flush=True,
            )
            if verbose and result.error:
                print(f'  detail: {result.error}', file=sys.stderr, flush=True)
            flush_summary()
            return

        if prev is None:
            rows.append(row)
        else:
            idx = rows.index(prev)
            rows[idx] = row
        row_by_miri[result.miri_path] = row

        if result.ok and result.success is not None:
            successes.append(SuccessfulAlignment(**result.success))
            if prev is None or prev.status != 'SUCCESS':
                n_ok += 1
                if count_fallback or result.mode == 'fallback':
                    n_fallback += 1

        print(_format_worker_done(result), flush=True)
        if verbose and result.error:
            print(f'  detail: {result.error}', file=sys.stderr, flush=True)
        flush_summary()

    if summary_outfile is not None:
        summary_outfile = Path(summary_outfile).expanduser().resolve()
        write_alignment_summary(rows, summary_outfile)
        print(f'Live alignment summary → {summary_outfile}')

    print(
        f'Alignment plan: {len(groups)} filter wave(s), '
        f'{sum(len(v) for v in groups.values())} frame(s) after rejecting '
        f'{n_rejected} low-overlap '
        f'(ref_overlap_frac < {min_ref_overlap_frac:.4f}), workers={workers}'
    )
    for filt, group in groups.items():
        print(f'  {filt}: {len(group)} frame(s)')

    if not groups:
        print('No MIRI frames with NIRCam overlap remain to align.')
        return 0, rows

    common_job = dict(
        repo=repo_str,
        nbright=nbright,
        plot=plot,
        verbose=verbose,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        match_radius_arcsec=match_radius_arcsec,
        clip_to_align_footprint=clip_to_align_footprint,
        refine=refine,
        refine_sigma=refine_sigma,
        refine_max_iter=refine_max_iter,
        max_nircam_dispersion_mas=max_nircam_dispersion_mas,
    )

    if max_nircam_dispersion_mas is not None:
        print(
            f'REFERENCE quality cut: dispersion > '
            f'{max_nircam_dispersion_mas:.1f} mas → try MIRI_REL fallback'
        )

    for filt, group in groups.items():
        print()
        print('=' * 72)
        print(f'Filter wave {filt}: {len(group)} frame(s)')
        print('=' * 72)

        pending: dict[str, dict] = {}
        for frame in group:
            miri_path, ref_images, best_ref = _frame_ref_images(frame)
            # Safety: low-overlap frames are rejected above and must not reach
            # alignment_dispersion / JHAT.
            if not ref_images or not best_ref:
                continue
            outdir = alignment_outdir_for(miri_path)
            pending[miri_path] = {
                **common_job,
                'miri_path': miri_path,
                'filter': filt,
                'ref_images': ref_images,
                'best_ref': best_ref,
                'outdir': str(outdir),
                'ref_overlap_frac': _frame_ref_overlap_frac(frame),
            }

        # --- Pass 1: parallel REFERENCE alignment ---
        nircam_jobs = [
            {**job, 'mode': 'nircam'}
            for job in pending.values()
        ]

        _run_jobs_parallel(
            nircam_jobs,
            run_nircam_align_job,
            workers=workers,
            label=f'{filt} REFERENCE',
            on_result=record_result,
        )

        # --- Passes 2+: parallel MIRI fallback ---
        if fallback:
            for pass_idx in (1, 2):
                need_fallback = [
                    miri
                    for miri, row in row_by_miri.items()
                    if miri in pending and _needs_miri_fallback(row)
                ]
                for miri in pending:
                    if miri not in row_by_miri:
                        need_fallback.append(miri)
                seen: set[str] = set()
                ordered_need: list[str] = []
                for miri in need_fallback:
                    if miri not in seen:
                        seen.add(miri)
                        ordered_need.append(miri)

                fb_jobs = []
                for miri in ordered_need:
                    parent, ov_frac = select_fallback_parent(
                        miri, filt, successes
                    )
                    if parent is None:
                        continue
                    fb_jobs.append(
                        {
                            **pending[miri],
                            'mode': 'fallback',
                            'parent': asdict(parent),
                            'overlap_fraction': ov_frac,
                        }
                    )

                if not fb_jobs:
                    break

                any_new = False

                def _on_fallback(result, _pass=pass_idx) -> None:
                    nonlocal any_new
                    before = row_by_miri.get(result.miri_path)
                    record_result(result, count_fallback=True)
                    if result.ok and (
                        before is None or before.status != 'SUCCESS'
                    ):
                        any_new = True

                _run_jobs_parallel(
                    fb_jobs,
                    run_fallback_align_job,
                    workers=workers,
                    label=f'{filt} fallback pass {pass_idx}',
                    on_result=_on_fallback,
                )
                if not any_new:
                    break

        # Accept remaining REFERENCE quality-hold products when no MIRI_REL
        # parent was available (or fallback was disabled).
        for miri in list(pending):
            row = row_by_miri.get(miri)
            if row is None or not _is_reference_quality_hold(row):
                continue
            kept = AlignmentSummaryRow(
                miri_path=row.miri_path,
                filter=row.filter,
                status='SUCCESS',
                n_calibrators=row.n_calibrators,
                dispersion_mas=row.dispersion_mas,
                aligned_path=row.aligned_path,
                align_mode=_normalize_align_mode(row.align_mode),
                original_ref=row.original_ref,
                aligned_to=row.aligned_to,
                ref_overlap_frac=row.ref_overlap_frac,
            )
            idx = rows.index(row)
            rows[idx] = kept
            row_by_miri[miri] = kept
            n_ok += 1
            print(
                f'DONE  {Path(miri).name}  {kept.filter}  SUCCESS  '
                f'align_mode=REFERENCE  '
                f'dispersion_mas={kept.dispersion_mas:.3f} '
                f'(no MIRI_REL parent; keeping REFERENCE despite quality cut)',
                flush=True,
            )
            flush_summary()

        # Final failure tally for this filter wave.
        wave_failures = [
            miri
            for miri in pending
            if row_by_miri.get(miri) is not None
            and row_by_miri[miri].status not in ('SUCCESS', 'REJECTED')
        ]
        # Ensure every pending frame has a row.
        for miri, job in pending.items():
            if miri in row_by_miri:
                continue
            row = harvest_alignment_metrics(
                miri,
                Path(job['outdir']),
                ran_ok=False,
                ref_overlap_frac=job.get('ref_overlap_frac', 'NA'),
            )
            row.filter = filt
            rows.append(row)
            row_by_miri[miri] = row
            wave_failures.append(miri)
            flush_summary()

        failures += len(wave_failures)
        if wave_failures and not continue_on_error:
            raise RuntimeError(
                f'Alignment failed for {len(wave_failures)} {filt} frame(s); '
                f'first={wave_failures[0]}'
            )

        print(
            f'Filter wave {filt} done: '
            f'{sum(1 for m in pending if row_by_miri[m].status == "SUCCESS")} ok, '
            f'{len(wave_failures)} failed'
        )

    print()
    print(
        f'Alignment finished: {n_ok} ok ({n_fallback} via MIRI fallback), '
        f'{n_rejected} rejected (ref_overlap_frac < {min_ref_overlap_frac:.4f}), '
        f'{failures} failed'
    )
    return failures, rows


def main(argv: list[str] | None = None) -> int:
    # Pre-parse --repo / --help so --help does not require heavy imports.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--repo', type=Path, default=None)
    pre.add_argument('-h', '--help', action='store_true')
    pre_args, _ = pre.parse_known_args(argv)

    # Resolve repo for defaults even before importing science modules.
    try:
        repo = _resolve_repo_root(pre_args.repo)
    except FileNotFoundError:
        repo = Path('/data/rwisenbaker/jwst123')
    default_data_dir = Path('/data/rwisenbaker/jwst_data/M51')

    if pre_args.help:
        create_parser(default_data_dir).print_help()
        return 0

    repo = _resolve_repo_root(pre_args.repo)
    _bootstrap_imports(repo)

    # JHAT imports astroquery.gaia, which contacts ESA TAP on import
    # (GaiaClass show_server_messages). A short default timeout prevents that
    # network call from hanging pipeline startup when the archive is slow/down.
    import socket

    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(15)

    from alignment_dispersion import run_alignment
    from image_overlap import BestOverlap, MirIFootprint, compute_overlap

    args = create_parser(default_data_dir).parse_args(argv)
    args.repo = repo

    if args.legacy_overlap_file is not None:
        return run_legacy_overlap_file_pipeline(
            args.legacy_overlap_file,
            outdir=args.legacy_outdir,
            plot=args.plot,
            verbose=args.verbose,
        )

    data_dir, dataset_label = resolve_data_dir(
        data_dir=args.data_dir,
        data_root=args.data_root,
        galaxy=args.galaxy,
        default_data_dir=default_data_dir,
    )
    args.data_dir = data_dir
    args.galaxy = dataset_label
    # Keep data_root as an alias of data_dir for any older internal callers.
    args.data_root = data_dir
    summary_rows: list[AlignmentSummaryRow] = []
    summary_path = data_dir / f'{dataset_label}_alignment_summary.txt'

    if args.overlap_only and args.align_only:
        print(
            'ERROR: choose at most one of --overlap-only / --align-only',
            file=sys.stderr,
        )
        return 2

    try:
        if args.align_only:
            json_path = args.overlap_json
            if json_path is None:
                json_path = data_dir / 'overlap_output' / 'overlap_summary.json'
            json_path = Path(json_path).expanduser().resolve()
            if not json_path.is_file():
                print(f'ERROR: overlap JSON not found: {json_path}', file=sys.stderr)
                return 1
            print(f'Loading overlaps from {json_path}')
            payload = json.loads(json_path.read_text())
            frames: list[FrameOverlaps] | list[dict] = payload['frames']
            filters = parse_filters_arg(args.filters)
            if filters:
                wanted = {f.upper() for f in filters}
                frames = [
                    f
                    for f in frames
                    if filter_name_from_miri_path(
                        f['miri_path'] if isinstance(f, dict) else f.miri_path
                    )
                    in wanted
                ]
                print(
                    f'Filter restriction {", ".join(filters)}: '
                    f'{len(frames)} frame(s) from overlap JSON'
                )
            if args.limit is not None:
                frames = frames[: args.limit]
        else:
            frames = run_overlaps(
                args,
                MirIFootprint=MirIFootprint,
                BestOverlap=BestOverlap,
                compute_overlap=compute_overlap,
            )
            if args.overlap_only:
                return 0

        failures, summary_rows = align_from_frames(
            frames,
            run_alignment=run_alignment,
            nbright=args.nbright,
            plot=args.plot,
            verbose=args.verbose,
            continue_on_error=args.continue_on_error,
            cache_dir=data_dir / 'overlap_output' / 'ref_phot_cache',
            match_radius_arcsec=args.match_radius,
            clip_to_align_footprint=not args.no_clip_footprint,
            refine=not args.no_refine,
            refine_sigma=args.refine_sigma,
            refine_max_iter=args.refine_max_iter,
            fallback=not args.no_fallback,
            max_nircam_dispersion_mas=args.max_nircam_dispersion_mas,
            min_ref_overlap_frac=args.min_ref_overlap_frac,
            summary_outfile=summary_path,
            workers=args.workers,
            repo=args.repo,
        )
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        if getattr(args, 'verbose', False):
            traceback.print_exc()
        # Live summary is flushed per frame; rewrite once more if we have rows.
        if summary_rows:
            summary_path = write_alignment_summary(summary_rows, summary_path)
            print(f'Alignment summary: {summary_path}')
        return 1

    summary_path = write_alignment_summary(summary_rows, summary_path)
    print(f'Alignment summary: {summary_path}')

    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
