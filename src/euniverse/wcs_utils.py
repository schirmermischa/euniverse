#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# euniverse.py - A program to display MER colour images created with eummy

# MIT License

# Copyright (c) [2026] [Mischa Schirmer]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from astropy.wcs import WCS

class WCSConverter:
    def __init__(self, metadata):
        self.metadata = metadata
        self.wcs = WCS(naxis=2)

        self.wcs.wcs.crpix = [metadata['CRPIX1'], metadata['CRPIX2']]
        self.wcs.wcs.cd = [[metadata['CD1_1'], metadata['CD1_2']],
                           [metadata['CD2_1'], metadata['CD2_2']]]
        self.wcs.wcs.crval = [metadata['CRVAL1'], metadata['CRVAL2']]
        self.wcs.wcs.ctype = [metadata['CTYPE1'], metadata['CTYPE2']]
        self.wcs.wcs.cunit = [metadata['CUNIT1'], metadata['CUNIT2']]
        self.wcs.wcs.radesys = metadata.get('RADESYS', 'ICRS')
        self.wcs.wcs.equinox = metadata.get('EQUINOX', 2000.0)

    def pixel_to_world(self, x, y):
        ra, dec = self.wcs.wcs_pix2world([[x, y]], 0)[0]
        return ra, dec


    def world_to_pixel(self, sky_coord):
        """
        Convert world coordinates (RA, Dec) to pixel coordinates (x, y).
        
        Args:
        sky_coord (SkyCoord): Astropy SkyCoord object (single or array of coordinates).
        
        Returns:
        tuple: Arrays (x, y) of pixel coordinates.
        """
        ra = sky_coord.ra.deg
        dec = sky_coord.dec.deg
        # Ensure ra, dec are arrays
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        # Create input array of shape (N, 2) for all_world2pix
        coords = np.column_stack((ra, dec))
        # Convert to pixel coordinates (origin=0 for FITS convention)
        x, y = self.wcs.all_world2pix(coords, 0).T
        return x, y
