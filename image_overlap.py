#!/usr/bin/env python
# coding: utf-8

# In[8]:


import warnings 

warnings.filterwarnings('ignore')

from __future__ import annotations
import argparse
from pathlib import Path
import shapely
from shapely.geometry import Polygon
from shapely import simplify
from shapely.ops import unary_union
import shapely.ops
import matplotlib.pyplot as plt
import numpy as np
import glob
import illuminated_s_region
from scipy import ndimage
from skimage import measure
from astropy.io import fits
from astropy.wcs import WCS
from dataclasses import dataclass
#from matplotlib.patches import Polygon

@dataclass(frozen=True)
class SRegionPolygon:
    """Sky polygon in S_REGION format."""

    frame: str
    vertices: np.ndarray  # shape (N, 2), columns are (ra, dec) in degrees

    @classmethod
    def parse(cls, s_region: str) -> SRegionPolygon:
        tokens = s_region.strip().split()
        if len(tokens) < 4 or tokens[0].upper() != "POLYGON":
            raise ValueError(f"Unsupported S_REGION format: {s_region!r}")
        frame = tokens[1].upper()
        values = [float(token) for token in tokens[2:]]
        if len(values) % 2:
            raise ValueError(f"S_REGION has an odd number of coordinate values: {s_region!r}")
        vertices = np.asarray(values, dtype=float).reshape(-1, 2)        return cls(frame=frame, vertices=vertices)

    def to_string(self, precision: int = 9) -> str:
        coord_text = " ".join(
            f"{ra:.{precision}f} {dec:.{precision}f}" for ra, dec in self.vertices)
        return f"POLYGON {self.frame} {coord_text}"

    def to_pixel_polygon(self, wcs: WCS) -> Polygon:
        x, y = wcs.world_to_pixel_values(self.vertices[:, 0], self.vertices[:, 1])
        return Polygon(np.column_stack([x, y]))

refs = glob.glob('group_*/ref_*/coadd*i2d.fits')

def all_miri_images(images):
    f = open('image_data_maximized_pt2', 'w')
    
    for image in images:
        s_region_polygon, region_mask, wcs, header, data, _ = illuminated_s_region.illuminated_s_region_from_fits(image)
	#print(s_region_polygon) 

        ## defining 'best' variables
        best_ref = None
        best_overlap = 0.0
        
        for ref in refs:
            with fits.open(ref) as hdu:
                header = hdu['SCI'].header
                original_s_region = SRegionPolygon.parse(header["S_REGION"])
            
                poly1 = s_region_polygon.to_pixel_polygon(wcs)        ## move outside this for loop?
                poly2 = original_s_region.to_pixel_polygon(wcs)
                #print(poly1, poly2)
                
                try:
                    overlap = poly1.intersection(poly2)

                    if overlap.is_empty or overlap.area == 0:
                        continue

                    #f.write(f'{image} {ref} \n')
                    #f.write(f'{overlap} {overlap.area} \n')
                    #f.write('\n')
                    
                    #print(image, ref)
                    #print(overlap, overlap.area)
                    #print()

                    # selecting single best reference image w/ max overlap 
                    if overlap.area > best_overlap:
                        best_overlap = overlap.area
                        best_ref = ref

                except:
                    f.write(f'{image} {ref} \n')
                    f.write(f'image failed \n')
                    f.write('\n')
                    
                    #print("image failed")
                    #print(image, ref)

        f.write(f'Overlap maximized: MIRI image: {image}, Reference image: {best_ref}, Max overlap area: {best_overlap}\n\n')
        #print(f'Overlap maximized: MIRI image: {image}, Reference image: {best_ref}, Max overlap area: {best_overlap}')

    f.close()

images = glob.glob('jwst_data_M51/M51/*/mastDownload/JWST/*_mirimage/*_cal.fits')

all_miri_images(images)
