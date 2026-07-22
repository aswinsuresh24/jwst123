#!/usr/bin/env python
"""Compute MIRI / reference image footprint overlap from S_REGION polygons."""

from __future__ import annotations

import argparse
import glob
import warnings
from dataclasses import dataclass
from pathlib import Path

from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from shapely.geometry import Polygon

from illuminated_s_region import SRegionPolygon, illuminated_s_region_from_fits

warnings.filterwarnings('ignore')


@dataclass(frozen=True)
class AreaMetrics:
    """Area in MIRI pixel units, with solid-angle and ROI-fraction context."""

    pixels2: float
    arcmin2: float
    fraction_of_miri_roi: float

    @classmethod
    def from_pixels(
        cls,
        area_pix2: float,
        pix_area_arcmin2: float,
        miri_roi_pix2: float,
    ) -> AreaMetrics:
        frac = area_pix2 / miri_roi_pix2 if miri_roi_pix2 > 0 else float('nan')
        return cls(
            pixels2=float(area_pix2),
            arcmin2=float(area_pix2) * pix_area_arcmin2,
            fraction_of_miri_roi=frac,
        )

    def format(self) -> str:
        return (
            f'{self.pixels2:.3f} pixels^2 | {self.arcmin2:.6f} arcmin^2 | '
            f'{self.fraction_of_miri_roi:.4f} of MIRI illuminated ROI'
        )


@dataclass(frozen=True)
class MirIFootprint:
    """Illuminated MIRI footprint projected into the MIRI WCS pixel plane."""

    path: str
    s_region: SRegionPolygon
    wcs: WCS
    polygon: Polygon
    pixel_area_arcmin2: float

    @property
    def area(self) -> AreaMetrics:
        return AreaMetrics.from_pixels(
            self.polygon.area,
            self.pixel_area_arcmin2,
            self.polygon.area,
        )

    @classmethod
    def from_fits(cls, path: str) -> MirIFootprint:
        s_region, _, wcs, _, _, _ = illuminated_s_region_from_fits(path)
        return cls(
            path=path,
            s_region=s_region,
            wcs=wcs,
            polygon=s_region.to_pixel_polygon(wcs),
            pixel_area_arcmin2=float(
                wcs.proj_plane_pixel_area().to(u.arcmin**2).value
            ),
        )

    def metrics(self, area_pix2: float) -> AreaMetrics:
        return AreaMetrics.from_pixels(
            area_pix2,
            self.pixel_area_arcmin2,
            float(self.polygon.area),
        )


@dataclass(frozen=True)
class OverlapResult:
    """Overlap of one reference footprint with a MIRI illuminated footprint."""

    ref_path: str
    ref_s_region: SRegionPolygon
    ref_area: AreaMetrics
    overlap_area: AreaMetrics


@dataclass(frozen=True)
class BestOverlap:
    """Best-matching reference for a single MIRI frame."""

    miri_path: str
    ref_path: str | None
    overlap_area: AreaMetrics


def load_header_s_region(fits_path: str, extname: str = 'SCI') -> SRegionPolygon:
    """Parse S_REGION from a FITS science header."""
    with fits.open(fits_path) as hdul:
        return SRegionPolygon.parse(hdul[extname].header['S_REGION'])


def polygon_area(polygon: Polygon) -> float:
    """Return Shapely polygon area, or 0 for an empty geometry."""
    return 0.0 if polygon.is_empty else float(polygon.area)


def compute_overlap(miri: MirIFootprint, ref_path: str) -> OverlapResult:
    """Project a reference S_REGION into MIRI pixels and compute overlap."""
    ref_s_region = load_header_s_region(ref_path)
    ref_poly = ref_s_region.to_pixel_polygon(miri.wcs)
    overlap_poly = miri.polygon.intersection(ref_poly)
    miri_roi = float(miri.polygon.area)
    return OverlapResult(
        ref_path=ref_path,
        ref_s_region=ref_s_region,
        ref_area=AreaMetrics.from_pixels(
            float(ref_poly.area),
            miri.pixel_area_arcmin2,
            miri_roi,
        ),
        overlap_area=AreaMetrics.from_pixels(
            polygon_area(overlap_poly),
            miri.pixel_area_arcmin2,
            miri_roi,
        ),
    )


def overlap_area_pixels(
    miri_image: str,
    ref_image: str,
) -> tuple[float, Polygon, Polygon]:
    """
    Return overlap area (MIRI pixels²) and the two pixel-plane polygons.

    Kept for programmatic reuse of the previous API.
    """
    miri = MirIFootprint.from_fits(miri_image)
    ref_s_region = load_header_s_region(ref_image)
    ref_poly = ref_s_region.to_pixel_polygon(miri.wcs)
    overlap = miri.polygon.intersection(ref_poly)
    return polygon_area(overlap), miri.polygon, ref_poly


def find_best_refs(
    miri_images: list[str],
    refs: list[str],
    outfile: str | None = None,
) -> list[BestOverlap]:
    """For each MIRI image, find the reference with maximum overlap area."""
    results: list[BestOverlap] = []
    out_path = Path(outfile) if outfile else None
    out_lines: list[str] = []

    for image in miri_images:
        print(f'MIRI: {image}')
        miri = MirIFootprint.from_fits(image)
        print(f'  illuminated S_REGION: {miri.s_region.to_string()}')
        print(
            f'  WCS pixel solid angle: {miri.pixel_area_arcmin2:.8e} '
            f'arcmin^2 / pixel'
        )
        print(f'  illuminated area: {miri.area.format()}')

        best: BestOverlap | None = None
        for ref in refs:
            try:
                result = compute_overlap(miri, ref)
            except Exception as exc:
                msg = f'{image} {ref}\nimage failed: {exc}\n\n'
                print(f'  FAILED for ref {ref}: {exc}')
                out_lines.append(msg)
                continue

            print(f'  ref: {ref}')
            print(f'    S_REGION: {result.ref_s_region.to_string()}')
            print(f'    ref area: {result.ref_area.format()}')
            print(f'    overlap area: {result.overlap_area.format()}')

            if best is None or result.overlap_area.pixels2 > best.overlap_area.pixels2:
                best = BestOverlap(
                    miri_path=image,
                    ref_path=result.ref_path,
                    overlap_area=result.overlap_area,
                )

        if best is None:
            best = BestOverlap(
                miri_path=image,
                ref_path=None,
                overlap_area=miri.metrics(0.0),
            )

        line = (
            f'Overlap maximized: MIRI image: {best.miri_path}, '
            f'Reference image: {best.ref_path}, '
            f'Max overlap area: {best.overlap_area.pixels2:.3f} pixels^2 '
            f'({best.overlap_area.arcmin2:.6f} arcmin^2, '
            f'{best.overlap_area.fraction_of_miri_roi:.4f} of MIRI illuminated ROI)'
        )
        print(line)
        print()
        out_lines.append(line + '\n\n')
        results.append(best)

    if out_path is not None:
        out_path.write_text(''.join(out_lines))

    return results


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Find the reference image with maximum footprint overlap '
            'for each MIRI frame.'
        )
    )
    parser.add_argument(
        '--miri',
        nargs='+',
        default=None,
        help='MIRI *_cal.fits image(s). Default: jwst_data_M51/.../*_mirimage/*_cal.fits',
    )
    parser.add_argument(
        '--ref',
        nargs='+',
        default=None,
        help='Reference coadd *_i2d.fits image(s). Default: group_*/ref_*/coadd*i2d.fits',
    )
    parser.add_argument(
        '--outfile',
        type=str,
        default=None,
        help='Optional text file for summary lines (default: print only).',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)

    miri_images = args.miri or glob.glob(
        'jwst_data_M51/M51/*/mastDownload/JWST/*_mirimage/*_cal.fits'
    )
    refs = args.ref or glob.glob('group_*/ref_*/coadd*i2d.fits')

    if not miri_images:
        print('ERROR: no MIRI images found. Pass --miri PATH ...')
        return 1
    if not refs:
        print('ERROR: no reference images found. Pass --ref PATH ...')
        return 1

    print(f'MIRI images ({len(miri_images)}):')
    for path in miri_images:
        print(f'  {path}')
    print(f'Reference images ({len(refs)}):')
    for path in refs:
        print(f'  {path}')
    print()

    find_best_refs(miri_images, refs, outfile=args.outfile)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
