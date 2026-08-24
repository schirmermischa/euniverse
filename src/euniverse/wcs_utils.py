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

"""
wcs_utils.py — WCS construction and validation for Euniverse Explorer
=======================================================================
TIFF metadata (the JSON blob stored in the ImageDescription tag) may
express the astrometric solution using more than one FITS convention:

  Reference point
    CRVAL1/CRVAL2           preferred.
    RA/DEC                  fallback. Assumed to coincide with the
                             pixel given by CRPIX1/CRPIX2 — i.e. RA/DEC
                             is treated as a direct alias for CRVAL, not
                             as an independent "image center" keyword.
                             If that assumption doesn't hold for a given
                             file, the resulting WCS will be silently
                             wrong rather than absent; there is no way
                             to detect that case from the keywords alone.

  Pixel scale / rotation
    CD1_1/CD1_2/CD2_1/CD2_2  takes precedence if present (full linear
                             transform: scale + rotation + skew).
    CDELT1/CDELT2            fallback, optionally combined with a PC
    (+ PC*_* or CROTA2)      matrix or a CROTA2 rotation angle. These
                             keywords are handed to astropy.wcs.WCS
                             as-is; astropy resolves PC-vs-CROTA2 per
                             the FITS standard itself rather than this
                             module reimplementing that logic.

Construction builds an astropy.io.fits.Header from the resolved
keywords and passes it to WCS(header=...), then validates the result
with a pixel round-trip test. Anything that doesn't produce a usable,
self-consistent celestial WCS raises InvalidWCSError with a message
describing what was missing or inconsistent — callers should treat
this as "no usable WCS for this image", not as a fatal load error.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


class InvalidWCSError(ValueError):
    """
    Raised when TIFF metadata does not yield a usable celestial WCS —
    either because required keywords are missing (under any of the
    supported conventions) or because the resulting transform fails a
    round-trip self-consistency check (degenerate/singular matrix,
    non-finite output, etc.).
    """
    pass


class WCSConverter:
    """
    Builds and validates an astropy WCS from Euclid TIFF metadata.

    Raises
    ------
    InvalidWCSError
        If no usable WCS can be constructed from *metadata*. Callers
        should catch this specifically and continue without a WCS
        (image display does not depend on it) rather than aborting
        the whole load.
    """

    # Round-trip tolerance for the validation check, in pixels. A
    # correctly-specified WCS round-trips to within numerical noise
    # (<< 1e-3 px); this is deliberately generous so it only rejects
    # genuinely degenerate transforms, not fine numerical differences.
    _ROUNDTRIP_TOL_PX = 1.0

    def __init__(self, metadata: dict):
        self.metadata = metadata
        header = self._build_header(metadata)

        try:
            # fix=False is deliberate: astropy's default fix=True will
            # silently repair a structurally-broken CD/PC matrix (e.g. an
            # all-zero matrix becomes an identity matrix — a "valid" but
            # wrong 1 deg/pixel WCS) instead of raising. With fix=False,
            # wcslib raises a clear SingularMatrixError for exactly that
            # case, which is far more useful than a WCS that constructs
            # cleanly but is silently wrong.
            self.wcs = WCS(header, fix=False)
        except InvalidWCSError:
            raise
        except Exception as e:
            raise InvalidWCSError(f"astropy/wcslib could not construct a WCS: {e}") from e

        self._validate()

    # ------------------------------------------------------------------
    # Header assembly — this is where convention/alias resolution lives.
    # Everything past this method deals with a normal FITS header, so
    # convention-specific logic never leaks into construction/validation.
    # ------------------------------------------------------------------

    def _build_header(self, metadata: dict) -> fits.Header:
        header = fits.Header()

        # --- Reference pixel + axis types: required, no alternate names ---
        for key in ('CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2'):
            if key not in metadata:
                raise InvalidWCSError(f"Missing required WCS keyword: {key}")
            header[key] = metadata[key]

        # --- Reference value: CRVAL preferred, RA/DEC as a fallback alias ---
        if 'CRVAL1' in metadata and 'CRVAL2' in metadata:
            header['CRVAL1'] = metadata['CRVAL1']
            header['CRVAL2'] = metadata['CRVAL2']
        elif 'RA' in metadata and 'DEC' in metadata:
            header['CRVAL1'] = metadata['RA']
            header['CRVAL2'] = metadata['DEC']
        else:
            raise InvalidWCSError(
                "Missing reference coordinate: need CRVAL1/CRVAL2 or RA/DEC"
            )

        # --- Pixel scale / rotation: CD matrix takes precedence over CDELT ---
        cd_keys = ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')
        if all(k in metadata for k in cd_keys):
            for k in cd_keys:
                header[k] = metadata[k]
        elif 'CDELT1' in metadata and 'CDELT2' in metadata:
            header['CDELT1'] = metadata['CDELT1']
            header['CDELT2'] = metadata['CDELT2']
            # Pass through whichever decomposed rotation term is present
            # (PC matrix or CROTA2) and let astropy's WCS resolve it —
            # both are valid FITS conventions and astropy already
            # implements the CROTA2 -> rotation-matrix conversion
            # correctly, so there's no reason to reimplement it here.
            for k in ('PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CROTA2'):
                if k in metadata:
                    header[k] = metadata[k]
        else:
            raise InvalidWCSError(
                "Missing pixel scale: need a CD matrix "
                "(CD1_1/CD1_2/CD2_1/CD2_2) or CDELT1/CDELT2"
            )

        # --- Optional keywords ---
        if 'CUNIT1' in metadata and 'CUNIT2' in metadata:
            header['CUNIT1'] = metadata['CUNIT1']
            header['CUNIT2'] = metadata['CUNIT2']

        radesys = metadata.get('RADESYS', 'ICRS')
        header['RADESYS'] = radesys

        # EQUINOX handling — two bugs fixed here vs. the original code:
        #
        # 1. metadata.get('EQUINOX', 2000.0) only falls back to 2000.0
        #    when the key is *absent*. When the JSON has "EQUINOX": null
        #    (Python None) the key IS present, so .get() returns None,
        #    and header['EQUINOX'] = None writes a blank-value FITS card.
        #    astropy then emits FITSFixedWarning: "a floating-point value
        #    was expected" when it parses that card back — not because
        #    the WCS is invalid, but because we asked it to parse a
        #    keyword we told it existed but gave no value for.
        #
        # 2. EQUINOX is only meaningful for equinox-based reference frames
        #    (FK4, FK5). ICRS/ICRF/GCRS/CIRS are equinox-independent by
        #    definition, so a file with RADESYS=ICRS and no EQUINOX isn't
        #    missing anything — the keyword should simply be omitted,
        #    not defaulted to 2000.0.
        equinox = metadata.get('EQUINOX')
        if equinox is not None:
            header['EQUINOX'] = equinox
        elif radesys.upper() not in ('ICRS', 'ICRF', 'GCRS', 'CIRS'):
            header['EQUINOX'] = 2000.0
        # else: equinox-independent frame, explicit null, or simply
        # absent — leave EQUINOX out of the header entirely.

        return header

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self):
        """
        Round-trip a couple of pixels through pix->world->pix.

        Catches the failure mode that construction alone does not:
        astropy.wcs.WCS() accepts a structurally-complete but numerically
        degenerate header (e.g. an all-zero or singular CD matrix)
        without complaint, and only fails — often with a low-level
        wcslib error — the first time the transform is actually used.
        Surfacing that here, at load time, means every downstream
        consumer (cursor readout, catalog overlay, exports, ...) can
        keep trusting "self.wcs is not None" the way the code already
        does, instead of each needing its own error handling.
        """
        if not self.wcs.has_celestial:
            raise InvalidWCSError("WCS has no celestial axes")

        crpix1, crpix2 = self.wcs.wcs.crpix[0], self.wcs.wcs.crpix[1]
        # CRPIX itself, plus an offset point — a degenerate (e.g. all-zero)
        # CD/PC matrix maps every offset to the same sky position, which a
        # single test point at CRPIX would not catch since CRPIX always
        # maps to CRVAL by definition regardless of the matrix.
        test_pixels = [(crpix1, crpix2), (crpix1 + 100, crpix2 + 100)]

        for x, y in test_pixels:
            try:
                ra, dec = self.wcs.wcs_pix2world([[x, y]], 0)[0]
                if not np.all(np.isfinite([ra, dec])):
                    raise InvalidWCSError(
                        f"WCS produced non-finite sky coordinates at pixel ({x}, {y})"
                    )
                x2, y2 = self.wcs.wcs_world2pix([[ra, dec]], 0)[0]
                if not np.all(np.isfinite([x2, y2])):
                    raise InvalidWCSError(
                        "WCS round-trip produced non-finite pixel coordinates"
                    )
                if abs(x2 - x) > self._ROUNDTRIP_TOL_PX or abs(y2 - y) > self._ROUNDTRIP_TOL_PX:
                    raise InvalidWCSError(
                        f"WCS failed round-trip self-consistency: "
                        f"({x:.2f}, {y:.2f}) -> ({x2:.2f}, {y2:.2f}), "
                        f"tolerance {self._ROUNDTRIP_TOL_PX} px"
                    )
            except InvalidWCSError:
                raise
            except Exception as e:
                # Catches wcslib-level failures (e.g. singular matrix)
                # that astropy raises lazily, only once the transform
                # is actually evaluated.
                raise InvalidWCSError(f"WCS transform failed: {e}") from e

    # ------------------------------------------------------------------
    # Public conversion API — unchanged from before
    # ------------------------------------------------------------------

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
        ra = np.atleast_1d(sky_coord.ra.deg)
        dec = np.atleast_1d(sky_coord.dec.deg)
        coords = np.column_stack((ra, dec))
        x, y = self.wcs.all_world2pix(coords, 0).T
        return x, y
