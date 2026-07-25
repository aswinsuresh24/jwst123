#!/usr/bin/env python
"""Compute MIRI / reference image footprint overlap from S_REGION polygons."""

from __future__ import annotations

import argparse
import glob
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from shapely.geometry import Polygon
from shapely.ops import unary_union

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
    """Illuminated MIRI footprint (sky S_REGION + on-detector pixel polygon)."""

    path: str
    s_region: SRegionPolygon
    wcs: WCS
    polygon: Polygon
    pixel_area_arcmin2: float
    center_ra_deg: float
    center_dec_deg: float
    sky_polygon_arcsec: Polygon

    @property
    def area(self) -> AreaMetrics:
        return AreaMetrics.from_pixels(
            self.polygon.area,
            self.pixel_area_arcmin2,
            self.polygon.area,
        )

    @property
    def pixel_area_arcsec2(self) -> float:
        return self.pixel_area_arcmin2 * 3600.0

    @classmethod
    def from_fits(cls, path: str) -> MirIFootprint:
        s_region, _, wcs, _, _, _ = illuminated_s_region_from_fits(path)
        verts = np.asarray(s_region.vertices, dtype=float)
        center_ra = float(np.mean(verts[:, 0]))
        center_dec = float(np.mean(verts[:, 1]))
        return cls(
            path=path,
            s_region=s_region,
            wcs=wcs,
            polygon=s_region.to_pixel_polygon(wcs),
            pixel_area_arcmin2=float(
                wcs.proj_plane_pixel_area().to(u.arcmin**2).value
            ),
            center_ra_deg=center_ra,
            center_dec_deg=center_dec,
            sky_polygon_arcsec=s_region.to_tangent_polygon(center_ra, center_dec),
        )

    def metrics(self, area_pix2: float) -> AreaMetrics:
        return AreaMetrics.from_pixels(
            area_pix2,
            self.pixel_area_arcmin2,
            float(self.polygon.area),
        )

    def metrics_from_sky_arcsec2(self, area_arcsec2: float) -> AreaMetrics:
        """Convert a tangent-plane area (arcsec²) into MIRI-pixel AreaMetrics."""
        pix_area = self.pixel_area_arcsec2
        area_pix2 = float(area_arcsec2) / pix_area if pix_area > 0 else 0.0
        miri_roi_pix2 = (
            float(self.sky_polygon_arcsec.area) / pix_area if pix_area > 0 else 0.0
        )
        return AreaMetrics.from_pixels(
            area_pix2,
            self.pixel_area_arcmin2,
            miri_roi_pix2,
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
    """
    Compute footprint overlap in a local sky tangent plane.

    Intersection is performed on ``S_REGION`` polygons expressed as
    arcsecond offsets from the MIRI footprint center.  This avoids false
    overlaps from WCS pixel extrapolation of off-FOV reference footprints.
    Reported areas are converted to MIRI pixels² via the MIRI pixel solid angle.
    """
    ref_s_region = load_header_s_region(ref_path)
    ref_sky = ref_s_region.to_tangent_polygon(
        miri.center_ra_deg,
        miri.center_dec_deg,
    )
    overlap_sky = miri.sky_polygon_arcsec.intersection(ref_sky)
    return OverlapResult(
        ref_path=ref_path,
        ref_s_region=ref_s_region,
        ref_area=miri.metrics_from_sky_arcsec2(polygon_area(ref_sky)),
        overlap_area=miri.metrics_from_sky_arcsec2(polygon_area(overlap_sky)),
    )


def compute_cumulative_overlap_fraction(
    miri: MirIFootprint,
    ref_paths: list[str],
) -> float:
    """
    Fraction of the MIRI illuminated ROI covered by the union of references.

    Overlapping reference footprints are merged (unique area only) before
    dividing by the MIRI sky footprint area. Returns 0.0 when there is no
    overlap.
    """
    miri_area = polygon_area(miri.sky_polygon_arcsec)
    if miri_area <= 0.0 or not ref_paths:
        return 0.0

    pieces = []
    for ref_path in ref_paths:
        try:
            ref_s_region = load_header_s_region(ref_path)
            ref_sky = ref_s_region.to_tangent_polygon(
                miri.center_ra_deg,
                miri.center_dec_deg,
            )
            overlap_sky = miri.sky_polygon_arcsec.intersection(ref_sky)
        except Exception:
            continue
        if not overlap_sky.is_empty and polygon_area(overlap_sky) > 0.0:
            pieces.append(overlap_sky)

    if not pieces:
        return 0.0
    return float(polygon_area(unary_union(pieces)) / miri_area)


def overlap_area_pixels(
    miri_image: str,
    ref_image: str,
) -> tuple[float, Polygon, Polygon]:
    """
    Return overlap area (MIRI pixels²) and the two sky-tangent polygons.

    Polygons are in local tangent-plane arcseconds (not detector pixels).
    Kept for programmatic reuse of the previous API.
    """
    miri = MirIFootprint.from_fits(miri_image)
    ref_s_region = load_header_s_region(ref_image)
    ref_sky = ref_s_region.to_tangent_polygon(
        miri.center_ra_deg,
        miri.center_dec_deg,
    )
    overlap = miri.sky_polygon_arcsec.intersection(ref_sky)
    return (
        miri.metrics_from_sky_arcsec2(polygon_area(overlap)).pixels2,
        miri.sky_polygon_arcsec,
        ref_sky,
    )


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
