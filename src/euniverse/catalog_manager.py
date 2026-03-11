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
catalog_manager.py — Euclid MER catalog loading and overlay creation
=====================================================================
CatalogManager loads the FITS MER (Multi-Extension Result) catalog for a
given Euclid tile and converts its source entries into QGraphicsEllipseItems
that ImageViewer can add to the scene.

Design principle — signals instead of back-references
------------------------------------------------------
Earlier versions held a direct reference to ImageViewer and called its
methods directly (e.g. self.image_viewer.update_status(...)).  This created
tight coupling: CatalogManager could not be instantiated or tested without
a live ImageViewer, and any refactor of ImageViewer risked breaking the
catalog code.

The new design emits Qt signals instead:

  status_updated(str, int)              → connect to viewer.update_status
  selection_display_requested(list)     → connect to viewer.display_selected_MER
  view_center_requested(float, float)   → connect to viewer.centerOn

ImageViewer connects these signals inside on_image_loaded when it constructs
a new CatalogManager.  Any other widget (e.g. a future progress dialog) can
also connect without modifying this class.

The optional image_viewer parameter is retained for convenience: if provided,
all three signals are wired automatically, preserving the old single-call
construction pattern.
"""

import os
import gc
import warnings

import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from PyQt5.QtCore import QObject, pyqtSignal, QRectF
from PyQt5.QtWidgets import QGraphicsEllipseItem
from PyQt5.QtGui import QPen, QColor


# ---------------------------------------------------------------------------
# Astropy 'NA' unit — present in some Euclid FITS column headers
# ---------------------------------------------------------------------------
try:
    NA_UNIT = u.def_unit('NA', u.dimensionless_unscaled)
except ValueError:
    NA_UNIT = u.Unit('NA')


class CatalogManager(QObject):
    """
    Loads, pre-processes, and exposes the MER catalog for a single Euclid tile.

    Signals
    -------
    status_updated(str, int)
        Human-readable status message + display timeout in ms.
        Connect to ImageViewer.update_status or any QStatusBar slot.

    selection_display_requested(list)
        Emitted with a list of OBJECT_IDs when the scatter-plot lasso requests
        that the viewer highlight a subset of sources.
        Connect to ImageViewer.display_selected_MER.

    view_center_requested(float, float)
        Emitted with (scene_x, scene_y) so the viewer pans to the first
        selected source.
        Connect to ImageViewer.centerOn.

    Parameters
    ----------
    tileID : str
        Tile identifier extracted from the TIFF filename, e.g. 'TILE101794875'.
    wcs : WCSConverter
        WCS object for RA/Dec ↔ pixel conversion.
    search_dir : str
        Directory to search for a matching FITS catalog file.
    image_viewer : object, optional
        If provided, the three signals above are automatically connected to
        the corresponding ImageViewer methods.  Pass None to wire signals
        manually (e.g. in unit tests).
    """

    status_updated              = pyqtSignal(str, int)
    selection_display_requested = pyqtSignal(list)
    view_center_requested       = pyqtSignal(float, float)

    def __init__(self, tileID: str, wcs, search_dir: str = ".",
                 image_viewer=None):
        super().__init__()

        self.tileID     = tileID
        self.wcs        = wcs
        self.search_dir = search_dir

        # Catalog data
        self.catalog        = None   # Astropy Table; None until load_catalog succeeds
        self.catalog_name   = None   # Filename stem, used for display
        self.catalog_path   = None   # Directory where catalog was found
        self.has_magnitudes = False  # Guard to prevent re-computing MAG columns

        # QGraphicsEllipseItem lists — added to / removed from the scene by ImageViewer
        self.MER_items          = []   # Full catalog overlay
        self.selected_MER_items = []   # Lasso-selected subset

        # Wire signals → ImageViewer if one was provided at construction time
        if image_viewer is not None:
            self.status_updated.connect(image_viewer.update_status)
            self.selection_display_requested.connect(image_viewer.display_selected_MER)
            self.view_center_requested.connect(image_viewer.centerOn)

        # Load the catalog immediately so numsources is valid after __init__
        self.load_catalog()
        self.numsources = self.get_catalog_row_count()

    # ------------------------------------------------------------------
    # Catalog loading
    # ------------------------------------------------------------------

    def load_catalog(self):
        """
        Scan search_dir for a FITS catalog matching this tile and load it.

        Match criteria (all must be satisfied):
          - filename contains self.tileID
          - filename contains 'EUC_MER_FINAL-CAT'
          - extension is .fits (case-insensitive)

        On success  : self.catalog is populated, columns pruned, MAG cols added.
        On failure  : self.catalog stays None; status_updated is emitted.
        """
        self.catalog = None
        gc.collect()

        try:
            # Ensure the 'NA' unit is available in the current Astropy session
            try:
                na_unit = u.Unit('NA')
            except ValueError:
                na_unit = u.def_unit('NA', u.dimensionless_unscaled)

            if not os.path.isdir(self.search_dir):
                return

            for filename in os.listdir(self.search_dir):
                if (self.tileID in filename and
                        'EUC_MER_FINAL-CAT' in filename and
                        filename.lower().endswith('.fits')):

                    self.catalog_name = filename[:-5] + "\n"
                    filepath = os.path.join(self.search_dir, filename)

                    # Suppress harmless Astropy unit / FITS-verify warnings
                    with u.add_enabled_units([na_unit]), warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=u.UnitsWarning)
                        warnings.simplefilter('ignore', category=fits.verify.VerifyWarning)
                        self.catalog = Table.read(filepath, format='fits')

                    self.catalog_path = self.search_dir
                    self.delete_empty_columns()
                    self.compute_magnitudes()
                    self.status_updated.emit(f"Loaded MER catalog: {filename}", 3000)
                    return

            raise FileNotFoundError(f"MER catalog not present for {self.tileID}")

        except Exception as e:
            self.status_updated.emit(f"Error loading catalog: {e}", 10000)
            print(f"INFO: {e}")
            self.catalog = None

    # ------------------------------------------------------------------
    # Catalog introspection
    # ------------------------------------------------------------------

    def get_catalog_row_count(self) -> int:
        """Return the number of rows in the loaded catalog, or 0 if not loaded."""
        return len(self.catalog) if self.catalog is not None else 0

    def get_non_empty_columns(self) -> list:
        """Return names of columns that are not entirely masked."""
        if self.catalog is None:
            return []
        return [col for col in self.catalog.colnames
                if not all(self.catalog[col].mask)]

    def get_all_column_names(self) -> list:
        """Return all column names, or an empty list if no catalog is loaded."""
        return self.catalog.colnames if self.catalog is not None else []

    # ------------------------------------------------------------------
    # Catalog pre-processing
    # ------------------------------------------------------------------

    def delete_empty_columns(self):
        """
        Drop columns that carry no useful data:
          - Entirely masked Astropy MaskedColumns
          - Float columns where every value is NaN
            (integer / bool columns skip the NaN test to avoid TypeError)
          - Columns where every value is Python None
        """
        to_remove = []
        for col_name in self.catalog.colnames:
            col = self.catalog[col_name]
            if hasattr(col, 'mask') and np.all(col.mask):
                to_remove.append(col_name)
            elif (np.issubdtype(col.dtype, np.floating) and
                  np.all(np.isnan(col.data))):
                to_remove.append(col_name)
            elif all(item is None for item in col):
                to_remove.append(col_name)

        if to_remove:
            self.catalog.remove_columns(to_remove)
            self.status_updated.emit(
                f"Removed {len(to_remove)} empty columns.", 2000
            )

    def compute_magnitudes(self):
        """
        Compute AB magnitudes from FLUX columns (assumed to be in µJy).

        For each FLUX_* column a new MAG_* column is added:
          positive flux : MAG = -2.5 * log10(flux_µJy * 1e-6) + 8.90
          zero or negative : MAG = 99  (standard sentinel for undetected)

        Guarded by self.has_magnitudes so it runs at most once.
        """
        if self.has_magnitudes:
            return

        for col in self.catalog.colnames:
            if col.startswith('FLUX'):
                new_name = col.replace('FLUX', 'MAG')
                flux_si  = 1e-6 * self.catalog[col].data   # µJy → Jy
                positive = flux_si > 0
                mag      = np.full_like(flux_si, 99.0, dtype=np.float32)
                mag[positive] = -2.5 * np.log10(flux_si[positive]) + 8.90
                self.catalog[new_name] = mag

        self.has_magnitudes = True

    # ------------------------------------------------------------------
    # Ellipse factory
    # ------------------------------------------------------------------

    def _make_ellipse(self, x: float, y: float,
                      a: float, b: float, pa: float,
                      color: QColor, width: float,
                      obj_id) -> QGraphicsEllipseItem:
        """
        Create a rotated QGraphicsEllipseItem for a single catalog source.

        Parameters
        ----------
        x, y   : scene pixel coords of the ellipse centre
        a, b   : semi-major and semi-minor axes in pixels
        pa     : position angle in degrees (FITS convention: N through E)
        color  : QPen colour
        width  : pen width
        obj_id : stored in item.data(0) for click-to-select lookup

        The rotation formula 90 - pa converts the FITS position angle
        (measured CCW from North = up) to Qt's clockwise rotation from the
        positive x-axis (East = right).
        """
        item = QGraphicsEllipseItem(QRectF(x - a, y - b, 2 * a, 2 * b))
        item.setPen(QPen(color, width))
        item.setTransformOriginPoint(x, y)
        item.setRotation(90 - pa)
        item.setData(0, obj_id)
        return item

    # ------------------------------------------------------------------
    # Full catalog overlay
    # ------------------------------------------------------------------

    def get_MER(self, image_height: int):
        """
        Build QGraphicsEllipseItems for every catalog source and store them
        in self.MER_items.

        The items are NOT added to the scene here — that is the responsibility
        of ImageViewer.toggle_MER(), which must run on the main thread.

        Parameters
        ----------
        image_height : int
            Full image height in pixels, used to flip y from FITS (bottom-up)
            to Qt (top-down) convention.
        """
        if self.catalog is None:
            print("No MER catalog available for plotting")
            return

        if self.wcs is None:
            print("No WCS — cannot overlay MER catalog")
            return

        self.MER_items = []
        try:
            required = ['OBJECT_ID', 'RIGHT_ASCENSION', 'DECLINATION',
                        'SEMIMAJOR_AXIS', 'POSITION_ANGLE', 'ELLIPTICITY']
            for col in required:
                if col not in self.catalog.colnames:
                    raise KeyError(f"Required column '{col}' missing from catalog")

            ra  = np.array(self.catalog['RIGHT_ASCENSION'].data, dtype=float)
            dec = np.array(self.catalog['DECLINATION'].data,     dtype=float)
            a   = self.catalog['SEMIMAJOR_AXIS'].data
            e   = self.catalog['ELLIPTICITY'].data
            pa  = self.catalog['POSITION_ANGLE'].data
            ids = self.catalog['OBJECT_ID'].data

            sky  = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
            x, y = self.wcs.world_to_pixel(sky)
            y    = image_height - y   # FITS y-flip

            # Scale SourceExtractor semi-axes to better approximate visual extent
            a_px = 3 * a
            b_px = a_px * (1 - e)

            for i in range(len(x)):
                self.MER_items.append(
                    self._make_ellipse(x[i], y[i], a_px[i], b_px[i], pa[i],
                                       QColor(255, 0, 0), 1, ids[i])
                )

            self.status_updated.emit(
                f"Retrieved {len(self.MER_items)} sources from MER catalog", 3000
            )

        except Exception as e:
            self.status_updated.emit(f"Error processing MER catalog: {e}", 5000)
            print(f"Error processing MER catalog: {e}")

    # ------------------------------------------------------------------
    # Selected-subset overlay (scatter-plot lasso)
    # ------------------------------------------------------------------

    def handle_selected_objects(self, object_ids: list):
        """
        Called by PlotDialog after a lasso selection.
        Emits selection_display_requested so ImageViewer can show the subset.
        CatalogManager does not access the scene directly.
        """
        self.selection_display_requested.emit(object_ids)

    def get_selected_MER(self, selected_object_ids: list,
                         image_height: int) -> list:
        """
        Build yellow QGraphicsEllipseItems for a specified subset of OBJECT_IDs.

        Returns the list for ImageViewer to add to the scene.  Also emits
        view_center_requested so the viewer pans to the first source.

        Parameters
        ----------
        selected_object_ids : list
            OBJECT_ID values to highlight (e.g. from a lasso selection).
        image_height : int
            Image height in pixels for the FITS y-flip.
        """
        if self.catalog is None or self.wcs is None or not selected_object_ids:
            return []

        self.selected_MER_items = []
        try:
            mask = np.isin(self.catalog['OBJECT_ID'].data, selected_object_ids)
            if not np.any(mask):
                return []

            sub  = self.catalog[mask]
            ra   = np.array(sub['RIGHT_ASCENSION'].data, dtype=float)
            dec  = np.array(sub['DECLINATION'].data,     dtype=float)
            a    = sub['SEMIMAJOR_AXIS'].data
            e    = sub['ELLIPTICITY'].data
            pa   = sub['POSITION_ANGLE'].data
            ids  = sub['OBJECT_ID'].data

            sky  = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
            x, y = self.wcs.world_to_pixel(sky)
            y    = image_height - y

            a_px = 3 * a
            b_px = a_px * (1 - e)

            for i in range(len(x)):
                self.selected_MER_items.append(
                    self._make_ellipse(x[i], y[i], a_px[i], b_px[i], pa[i],
                                       QColor(255, 255, 0), 1, ids[i])
                )

            # Signal the viewer to pan to the first result
            if len(x) > 0:
                self.view_center_requested.emit(float(x[0]), float(y[0]))

        except Exception as e:
            print(f"Error in get_selected_MER: {e}")

        return self.selected_MER_items

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def clear_MER(self):
        """
        Reset the MER item list.
        Scene removal is handled by ImageViewer.clear_MER(); this method
        only clears the Python-side reference list.
        """
        self.MER_items = []

    def clear_selected_MER(self):
        """
        Reset the selected-subset item list.
        Scene removal is handled by ImageViewer.clear_selected_MER().
        """
        self.selected_MER_items = []
