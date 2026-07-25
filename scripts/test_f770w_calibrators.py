#!/usr/bin/env python3
"""
Re-align the worst high-ncal F770W frames with filter-specific calibrator settings.

Writes products under each frame's ``alignment_output_f770w_test/`` so the
production ``alignment_output/`` tree is not overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment_calibrators import (  # noqa: E402
    F770W_CALIBRATOR_SETTINGS,
    describe_calibrator_settings,
)
from alignment_dispersion import read_dispersion_mas, run_alignment  # noqa: E402

DEFAULT_TARGETS = [
    # worst dispersion among high-ncal REFERENCE successes (visit 03435010001)
    'jw03435010001_24101_00003_mirimage_cal.fits',  # 69.9 mas, 404 cals
    'jw03435010001_20101_00002_mirimage_cal.fits',  # 67.1 mas, 493 cals
    'jw03435010001_22101_00002_mirimage_cal.fits',  # 66.0 mas, 456 cals
    'jw03435010001_18101_00001_mirimage_cal.fits',  # 65.3 mas, 476 cals
    'jw03435010001_22101_00004_mirimage_cal.fits',  # 65.1 mas, 342 cals
]


def _load_frames(overlap_json: Path) -> dict[str, dict]:
    data = json.loads(overlap_json.read_text())
    frames = data['frames'] if isinstance(data, dict) else data
    by_name: dict[str, dict] = {}
    for fr in frames:
        miri = fr.get('miri_path') or fr.get('miri') or fr.get('path')
        if not miri:
            continue
        by_name[Path(miri).name] = fr
    return by_name


def _ref_images(frame: dict) -> list[str]:
    refs = (
        frame.get('ref_images')
        or frame.get('references')
        or frame.get('overlapping')
        or []
    )
    if refs and isinstance(refs[0], dict):
        out = []
        for r in refs:
            path = r.get('path') or r.get('ref_path') or r.get('image')
            if path:
                out.append(path)
        return out
    return list(refs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('/data/rwisenbaker/jwst_data/M51'),
    )
    parser.add_argument(
        '--overlap-json',
        type=Path,
        default=None,
        help='Defaults to <data-dir>/overlap_output/overlap_summary.json',
    )
    parser.add_argument(
        '--outdir-name',
        default='alignment_output_f770w_test',
        help='Per-frame output directory name (default: alignment_output_f770w_test)',
    )
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--targets',
        nargs='*',
        default=DEFAULT_TARGETS,
        help='Basename(s) of MIRI cal.fits frames to test',
    )
    args = parser.parse_args(argv)

    overlap_json = args.overlap_json or (
        args.data_dir / 'overlap_output' / 'overlap_summary.json'
    )
    if not overlap_json.is_file():
        print(f'ERROR: overlap JSON not found: {overlap_json}', file=sys.stderr)
        return 1

    by_name = _load_frames(overlap_json)
    settings = F770W_CALIBRATOR_SETTINGS
    print(f'F770W calibrator settings: {describe_calibrator_settings(settings)}')
    print(f'Testing {len(args.targets)} frame(s)')

    rows: list[tuple[str, float | None, int | None, str]] = []
    for name in args.targets:
        frame = by_name.get(name)
        if frame is None:
            # Fall back to a filesystem search under F770W/
            matches = list(args.data_dir.glob(f'F770W/**/{name}'))
            if not matches:
                print(f'ERROR: frame not in overlap JSON and not found: {name}')
                rows.append((name, None, None, 'MISSING'))
                continue
            miri_path = str(matches[0])
            refs = []
            print(f'WARNING: {name} not in overlap JSON; cannot resolve refs')
            rows.append((name, None, None, 'NO_REFS'))
            continue
        else:
            miri_path = frame.get('miri_path') or frame.get('miri')
            refs = _ref_images(frame)

        if not refs:
            print(f'ERROR: no refs for {name}')
            rows.append((name, None, None, 'NO_REFS'))
            continue

        miri = Path(miri_path)
        outdir = miri.parent / args.outdir_name
        # Baseline from production product if present.
        prod = miri.parent / 'alignment_output' / miri.name.replace(
            '_cal.fits', '_jhat.fits'
        )
        old_disp = old_ncal = None
        if prod.is_file():
            old_disp, old_ncal = read_dispersion_mas(str(prod))

        print()
        print('=' * 72)
        print(f'{name}')
        print(f'  miri: {miri}')
        print(f'  refs: {len(refs)}')
        print(f'  baseline: disp={old_disp} mas, ncal={old_ncal}')
        print(f'  outdir: {outdir}')

        try:
            run_alignment(
                ref_images=refs,
                align_image=str(miri),
                outdir=str(outdir),
                plot=args.plot,
                verbose=args.verbose,
                cache_dir=str(args.data_dir / 'overlap_output' / 'ref_phot_cache'),
                refine=True,
                **settings.as_run_kwargs(),
            )
            jhat = outdir / miri.name.replace('_cal.fits', '_jhat.fits')
            new_disp, new_ncal = read_dispersion_mas(str(jhat))
            status = 'OK'
            print(
                f'  RESULT: {old_disp} → {new_disp} mas '
                f'(ncal {old_ncal} → {new_ncal})'
            )
        except Exception as exc:
            new_disp = new_ncal = None
            status = f'FAIL:{exc}'
            print(f'  RESULT: FAILED ({exc})')

        rows.append((name, new_disp, new_ncal, status))

    print()
    print('=' * 72)
    print('SUMMARY')
    print(
        f'{"frame":48s}  {"old":>8s}  {"new":>8s}  {"ncal":>6s}  status'
    )
    for name in args.targets:
        frame = by_name.get(name)
        old_disp = None
        if frame is not None:
            miri = Path(frame.get('miri_path') or frame.get('miri'))
            prod = miri.parent / 'alignment_output' / miri.name.replace(
                '_cal.fits', '_jhat.fits'
            )
            if prod.is_file():
                old_disp, _ = read_dispersion_mas(str(prod))
        match = next((r for r in rows if r[0] == name), None)
        new_disp = match[1] if match else None
        new_ncal = match[2] if match else None
        status = match[3] if match else 'NA'
        print(
            f'{name:48s}  '
            f'{old_disp if old_disp is not None else float("nan"):8.3f}  '
            f'{new_disp if new_disp is not None else float("nan"):8.3f}  '
            f'{new_ncal if new_ncal is not None else -1:6d}  {status}'
        )
    return 0 if all(r[3] == 'OK' for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
