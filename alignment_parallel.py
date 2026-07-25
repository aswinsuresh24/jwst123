"""
Parallel per-filter alignment workers for ``alignment_wrap``.

Workers are module-level so they can be pickled by ``ProcessPoolExecutor``.
Each job handles either a NIRCam primary alignment or a MIRI→MIRI fallback.
JHAT / pipeline chatter is suppressed; the parent process logs brief START/DONE lines.

Heavy science imports (``jwst`` / ``jhat`` / ``pysiaf``) are preloaded once per
worker process via ``worker_initializer`` so spawn children do not repeatedly
trip fragile import-time logging.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_STACK_READY = False


@dataclass
class AlignWorkerResult:
    """Picklable result returned by a worker process."""

    miri_path: str
    filter: str
    mode: str  # 'nircam' | 'fallback' | 'skip'
    ok: bool
    row: dict[str, Any]
    success: dict[str, Any] | None = None
    error: str | None = None
    message: str = ''


def _bootstrap(repo: str) -> None:
    root = str(Path(repo).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _sanitize_logging() -> None:
    """
    Neutralize stpipe / jwst log handlers that TypeError on pysiaf warnings.

    ``pysiaf`` emits a multi-argument ``logger.warning(...)`` at import time.
    ``stpipe.log.LogHandler`` then raises during ``emit``, which can abort
    worker imports under spawn. Replace with quiet NullHandlers.
    """
    # Prevent handler emit side-effects while we reconfigure.
    logging.raiseExceptions = False

    def _quiet(logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.ERROR)

    _quiet(logging.getLogger())
    for name in (
        'stpipe',
        'stpipe.pipeline',
        'jwst',
        'jwst.associations',
        'pysiaf',
        'CRDS',
    ):
        _quiet(logging.getLogger(name))


@contextmanager
def suppress_output():
    """Silence stdout/stderr and logging (JHAT / stpipe are very chatty)."""
    devnull = open(os.devnull, 'w')
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            previous_disable = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                yield
            finally:
                logging.disable(previous_disable)
    finally:
        devnull.close()


def _preload_science_stack() -> None:
    """Import jwst/jhat/pysiaf once per process with sanitized logging."""
    global _STACK_READY
    if _STACK_READY:
        return

    _sanitize_logging()
    last_exc: Exception | None = None
    for _attempt in range(2):
        try:
            with suppress_output():
                _sanitize_logging()
                import alignment_dispersion  # noqa: F401
                import jwst123  # noqa: F401
            _STACK_READY = True
            return
        except Exception as exc:  # pragma: no cover - environment-dependent
            last_exc = exc
            _sanitize_logging()
    raise RuntimeError(f'Failed to preload science stack in worker: {last_exc}')


def worker_initializer(repo: str) -> None:
    """
    ProcessPoolExecutor initializer: path, timeouts, quiet logs, warm imports.

    Called once per spawn worker before any alignment jobs run.
    """
    _bootstrap(repo)
    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(15)
    # Limit BLAS/OpenMP oversubscription when many workers run together.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    _sanitize_logging()
    _preload_science_stack()


def _ensure_worker_ready(repo: str) -> None:
    """Idempotent bootstrap used at the start of each job (serial + pool)."""
    _bootstrap(repo)
    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(15)
    if not _STACK_READY:
        _sanitize_logging()
        _preload_science_stack()


def _row_to_dict(row: object) -> dict[str, Any]:
    return asdict(row)  # type: ignore[arg-type]


def _success_from_dict(data: dict[str, Any]):
    from alignment_fallback import SuccessfulAlignment

    return SuccessfulAlignment(**data)


def run_nircam_align_job(job: dict[str, Any]) -> AlignWorkerResult:
    """Worker: align one MIRI frame to overlapping NIRCam references."""
    _ensure_worker_ready(job['repo'])
    from alignment_dispersion import run_alignment
    from alignment_fallback import (
        SuccessfulAlignment,
        filter_wavelength_um,
        find_aligned_photfile,
        write_alignment_provenance,
    )
    from alignment_wrap import (
        AlignmentSummaryRow,
        harvest_alignment_metrics,
        read_miri_filter,
    )

    miri_path = job['miri_path']
    filt = job['filter']
    ref_images = list(job['ref_images'])
    best_ref = job.get('best_ref')
    outdir = Path(job['outdir'])
    ref_overlap_frac = job.get('ref_overlap_frac', 'NA')

    def fail(msg: str, exc: Exception | None = None) -> AlignWorkerResult:
        err = msg if exc is None else f'{msg}: {exc}'
        if job.get('verbose') and exc is not None:
            err = f'{err}\n{traceback.format_exc()}'
        row = harvest_alignment_metrics(
            miri_path,
            outdir,
            ran_ok=False,
            ref_overlap_frac=ref_overlap_frac,
        )
        row.filter = filt
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=filt,
            mode='nircam',
            ok=False,
            row=_row_to_dict(row),
            error=err,
        )

    if not ref_images:
        row = AlignmentSummaryRow(
            miri_path=miri_path,
            filter=filt,
            status='SKIP',
            n_calibrators='NA',
            dispersion_mas='NA',
            aligned_path='NA',
            ref_overlap_frac=ref_overlap_frac,
        )
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=filt,
            mode='skip',
            ok=False,
            row=_row_to_dict(row),
        )

    try:
        with suppress_output():
            run_alignment(
                ref_images=ref_images,
                align_image=miri_path,
                outdir=str(outdir),
                nbright=job['nbright'],
                plot=job['plot'],
                verbose=False,
                cache_dir=job.get('cache_dir'),
                match_radius_arcsec=job['match_radius_arcsec'],
                clip_to_align_footprint=job['clip_to_align_footprint'],
                refine=job['refine'],
                refine_sigma=job['refine_sigma'],
                refine_max_iter=job['refine_max_iter'],
            )
    except Exception as exc:
        return fail('NIRCam alignment failed', exc)

    original_ref = best_ref or ref_images[0]
    aligned_to = original_ref
    row = harvest_alignment_metrics(
        miri_path,
        outdir,
        ran_ok=True,
        default_align_mode='REFERENCE',
        default_original_ref=original_ref,
        default_aligned_to=aligned_to,
        ref_overlap_frac=ref_overlap_frac,
    )
    if row.status != 'SUCCESS' or not isinstance(row.dispersion_mas, float):
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=filt,
            mode='nircam',
            ok=False,
            row=_row_to_dict(row),
            error='REFERENCE alignment soft-failed',
        )

    write_alignment_provenance(
        row.aligned_path,
        align_mode='REFERENCE',
        original_ref=original_ref,
        aligned_to=aligned_to,
        relative_dispersion_mas=row.dispersion_mas,
        absolute_dispersion_mas=row.dispersion_mas,
        n_calibrators=(
            row.n_calibrators if isinstance(row.n_calibrators, int) else None
        ),
    )
    row = harvest_alignment_metrics(
        miri_path,
        outdir,
        ran_ok=True,
        default_align_mode='REFERENCE',
        default_original_ref=original_ref,
        default_aligned_to=aligned_to,
        ref_overlap_frac=ref_overlap_frac,
    )
    if not row.filter or row.filter == 'UNKNOWN':
        row.filter = filt or read_miri_filter(miri_path)

    max_disp = job.get('max_nircam_dispersion_mas')
    if (
        max_disp is not None
        and float(max_disp) > 0
        and float(row.dispersion_mas) > float(max_disp)
    ):
        # Keep REFERENCE products on disk, but mark FAILURE so the parent
        # process attempts MIRI_REL and does not use this frame as a parent.
        # Metrics stay on the row so MIRI_REL failure can restore SUCCESS.
        row.status = 'FAILURE'
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=row.filter,
            mode='nircam',
            ok=False,
            row=_row_to_dict(row),
            error=(
                f'REFERENCE dispersion {float(row.dispersion_mas):.3f} mas '
                f'exceeds quality threshold {float(max_disp):.3f} mas'
            ),
        )

    success = SuccessfulAlignment(
        miri_path=miri_path,
        jhat_path=row.aligned_path,
        filter=row.filter,
        wavelength_um=filter_wavelength_um(row.filter),
        dispersion_mas=float(row.dispersion_mas),
        relative_dispersion_mas=float(row.dispersion_mas),
        align_mode='REFERENCE',
        original_ref=original_ref,
        aligned_to=aligned_to,
        photfile=find_aligned_photfile(row.aligned_path),
    )
    return AlignWorkerResult(
        miri_path=miri_path,
        filter=row.filter,
        mode='nircam',
        ok=True,
        row=_row_to_dict(row),
        success=asdict(success),
    )


def _backup_alignment_products(outdir: Path, miri_path: str) -> list[tuple[Path, Path]]:
    """Copy existing JHAT/phot products aside so a failed MIRI_REL can restore them."""
    import shutil

    from alignment_wrap import find_jhat_product

    backups: list[tuple[Path, Path]] = []
    jhat = find_jhat_product(outdir, miri_path)
    candidates: list[Path] = []
    if jhat is not None:
        candidates.append(jhat)
        stem = jhat.name.replace('_jhat.fits', '')
        for pattern in (f'{stem}*phot*.fits', f'{stem}*phot*.ecsv', f'{stem}*.reg'):
            candidates.extend(outdir.glob(pattern))
    seen: set[Path] = set()
    for src in candidates:
        src = src.resolve()
        if src in seen or not src.is_file():
            continue
        seen.add(src)
        bak = src.with_name(src.name + '.nircam_bak')
        shutil.copy2(src, bak)
        backups.append((src, bak))
    return backups


def _restore_alignment_products(backups: list[tuple[Path, Path]]) -> None:
    import shutil

    for src, bak in backups:
        if bak.is_file():
            shutil.copy2(bak, src)
            bak.unlink(missing_ok=True)


def _cleanup_alignment_backups(backups: list[tuple[Path, Path]]) -> None:
    for _, bak in backups:
        bak.unlink(missing_ok=True)


def run_fallback_align_job(job: dict[str, Any]) -> AlignWorkerResult:
    """Worker: relative-align one MIRI frame to a chosen parent JHAT product."""
    _ensure_worker_ready(job['repo'])
    from alignment_dispersion import run_alignment
    from alignment_fallback import (
        SuccessfulAlignment,
        combine_dispersion_mas,
        filter_wavelength_um,
        find_aligned_photfile,
        write_alignment_provenance,
    )
    from alignment_wrap import harvest_alignment_metrics

    miri_path = job['miri_path']
    filt = job['filter']
    outdir = Path(job['outdir'])
    parent = _success_from_dict(job['parent'])
    backups = _backup_alignment_products(outdir, miri_path)
    ref_overlap_frac = job.get('ref_overlap_frac', 'NA')

    def _fail(msg: str, exc: Exception | None = None) -> AlignWorkerResult:
        _restore_alignment_products(backups)
        row = harvest_alignment_metrics(
            miri_path,
            outdir,
            ran_ok=False,
            ref_overlap_frac=ref_overlap_frac,
        )
        row.filter = filt
        err = msg if exc is None else f'{msg}: {exc}'
        if job.get('verbose') and exc is not None:
            err = f'{err}\n{traceback.format_exc()}'
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=filt,
            mode='fallback',
            ok=False,
            row=_row_to_dict(row),
            error=err,
        )

    try:
        parent_phot = parent.photfile or find_aligned_photfile(parent.jhat_path)
        common = dict(
            align_image=miri_path,
            outdir=str(outdir),
            nbright=job['nbright'],
            plot=job['plot'],
            verbose=False,
            refine=job['refine'],
            refine_sigma=job['refine_sigma'],
            refine_max_iter=job['refine_max_iter'],
        )
        with suppress_output():
            if parent_phot is not None:
                run_alignment(
                    photfile=parent_phot,
                    ref_image=parent.jhat_path,
                    **common,
                )
            else:
                run_alignment(
                    ref_image=parent.jhat_path,
                    **common,
                )
    except Exception as exc:
        return _fail('fallback alignment failed', exc)

    row = harvest_alignment_metrics(
        miri_path,
        outdir,
        ran_ok=True,
        default_align_mode='MIRI_REL',
        default_original_ref=parent.original_ref,
        default_aligned_to=parent.jhat_path,
        ref_overlap_frac=ref_overlap_frac,
    )
    if row.status != 'SUCCESS' or not isinstance(row.dispersion_mas, float):
        return _fail('fallback soft-failed')

    rel_mas = float(row.dispersion_mas)
    abs_mas = combine_dispersion_mas(parent.dispersion_mas, rel_mas)
    write_alignment_provenance(
        row.aligned_path,
        align_mode='MIRI_REL',
        original_ref=parent.original_ref,
        aligned_to=parent.jhat_path,
        relative_dispersion_mas=rel_mas,
        absolute_dispersion_mas=abs_mas,
        n_calibrators=(
            row.n_calibrators if isinstance(row.n_calibrators, int) else None
        ),
    )
    row = harvest_alignment_metrics(
        miri_path,
        outdir,
        ran_ok=True,
        default_align_mode='MIRI_REL',
        default_original_ref=parent.original_ref,
        default_aligned_to=parent.jhat_path,
        ref_overlap_frac=ref_overlap_frac,
    )
    success = SuccessfulAlignment(
        miri_path=miri_path,
        jhat_path=row.aligned_path,
        filter=row.filter,
        wavelength_um=filter_wavelength_um(row.filter),
        dispersion_mas=float(row.dispersion_mas),
        relative_dispersion_mas=rel_mas,
        align_mode='MIRI_REL',
        original_ref=parent.original_ref,
        aligned_to=parent.jhat_path,
        photfile=find_aligned_photfile(row.aligned_path),
    )
    _cleanup_alignment_backups(backups)
    return AlignWorkerResult(
        miri_path=miri_path,
        filter=row.filter,
        mode='fallback',
        ok=True,
        row=_row_to_dict(row),
        success=asdict(success),
    )
