#!/usr/bin/env python
"""Compute science / reference image footprint overlap from S_REGION polygons."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from shapely.geometry import Polygon

from jwst123.illuminated_s_region import SRegionPolygon, illuminated_s_region_from_fits

warnings.filterwarnings('ignore')


@dataclass(frozen=True)
class AreaMetrics:
    """Area in science-image pixel units, with solid-angle and ROI-fraction context."""

    pixels2: float
    arcmin2: float
    fraction_of_roi: float

    @classmethod
    def from_pixels(
        cls,
        area_pix2: float,
        pix_area_arcmin2: float,
        roi_pix2: float,
    ) -> AreaMetrics:
        frac = area_pix2 / roi_pix2 if roi_pix2 > 0 else float('nan')
        return cls(
            pixels2=float(area_pix2),
            arcmin2=float(area_pix2) * pix_area_arcmin2,
            fraction_of_roi=frac,
        )

    def format(self) -> str:
        return (
            f'{self.pixels2:.3f} pixels^2 | {self.arcmin2:.6f} arcmin^2 | '
            f'{self.fraction_of_roi:.4f} of illuminated ROI'
        )


@dataclass(frozen=True)
class ScienceFootprint:
    """Illuminated science footprint projected into the science WCS pixel plane."""

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
    def from_fits(cls, path: str) -> ScienceFootprint:
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


# Backward-compatible alias for older call sites / notebooks.
MirIFootprint = ScienceFootprint


@dataclass(frozen=True)
class OverlapResult:
    """Overlap of one reference footprint with a science illuminated footprint."""

    ref_path: str
    ref_s_region: SRegionPolygon
    ref_area: AreaMetrics
    overlap_area: AreaMetrics


@dataclass(frozen=True)
class BestOverlap:
    """Best-matching reference for a single science frame."""

    science_path: str
    ref_path: str | None
    overlap_area: AreaMetrics


def load_header_s_region(fits_path: str, extname: str = 'SCI') -> SRegionPolygon:
    """Parse S_REGION from a FITS science header."""
    with fits.open(fits_path) as hdul:
        return SRegionPolygon.parse(hdul[extname].header['S_REGION'])


def polygon_area(polygon: Polygon) -> float:
    """Return Shapely polygon area, or 0 for an empty geometry."""
    return 0.0 if polygon.is_empty else float(polygon.area)


def compute_overlap(science: ScienceFootprint, ref_path: str) -> OverlapResult:
    """Project a reference S_REGION into science pixels and compute overlap."""
    ref_s_region = load_header_s_region(ref_path)
    ref_poly = ref_s_region.to_pixel_polygon(science.wcs)
    overlap_poly = science.polygon.intersection(ref_poly)
    roi_area = float(science.polygon.area)
    return OverlapResult(
        ref_path=ref_path,
        ref_s_region=ref_s_region,
        ref_area=AreaMetrics.from_pixels(
            float(ref_poly.area),
            science.pixel_area_arcmin2,
            roi_area,
        ),
        overlap_area=AreaMetrics.from_pixels(
            polygon_area(overlap_poly),
            science.pixel_area_arcmin2,
            roi_area,
        ),
    )


def overlap_area_pixels(
    science_image: str,
    ref_image: str,
) -> tuple[float, Polygon, Polygon]:
    """
    Return overlap area (science pixels²) and the two pixel-plane polygons.

    Kept for programmatic reuse of the previous API.
    """
    science = ScienceFootprint.from_fits(science_image)
    ref_s_region = load_header_s_region(ref_image)
    ref_poly = ref_s_region.to_pixel_polygon(science.wcs)
    overlap = science.polygon.intersection(ref_poly)
    return polygon_area(overlap), science.polygon, ref_poly


def find_best_refs(
    science_images: list[str],
    refs: list[str],
    outfile: str | None = None,
) -> list[BestOverlap]:
    """For each science image, find the reference with maximum overlap area."""
    results: list[BestOverlap] = []
    out_path = Path(outfile) if outfile else None
    out_lines: list[str] = []

    for image in science_images:
        print(f'Science: {image}')
        science = ScienceFootprint.from_fits(image)
        print(f'  illuminated S_REGION: {science.s_region.to_string()}')
        print(
            f'  WCS pixel solid angle: {science.pixel_area_arcmin2:.8e} '
            f'arcmin^2 / pixel'
        )
        print(f'  illuminated area: {science.area.format()}')

        best: BestOverlap | None = None
        for ref in refs:
            try:
                result = compute_overlap(science, ref)
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
                    science_path=image,
                    ref_path=result.ref_path,
                    overlap_area=result.overlap_area,
                )

        if best is None:
            best = BestOverlap(
                science_path=image,
                ref_path=None,
                overlap_area=science.metrics(0.0),
            )

        line = (
            f'Overlap maximized: Science image: {best.science_path}, '
            f'Reference image: {best.ref_path}, '
            f'Max overlap area: {best.overlap_area.pixels2:.3f} pixels^2 '
            f'({best.overlap_area.arcmin2:.6f} arcmin^2, '
            f'{best.overlap_area.fraction_of_roi:.4f} of illuminated ROI)'
        )
        print(line)
        print()
        out_lines.append(line + '\n\n')
        results.append(best)

    if out_path is not None:
        out_path.write_text(''.join(out_lines))

    return results
