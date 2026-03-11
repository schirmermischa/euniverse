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
image_viewer.py — Main image display and interaction widget
===========================================================
ImageViewer is a QGraphicsView subclass that displays a Euclid MER tile TIFF
and handles all user interaction on it.

Responsibilities
----------------
  - TIFF loading via a background TiffLoader thread (see workers.py)
  - Contrast adjustment engine (LUT-based full pass + viewport-crop preview)
  - MER catalog overlay management (add / remove / toggle QGraphicsEllipseItems)
  - User annotation circles (right-click to classify, left-click to select,
    Delete key to remove); stored as Annotation dataclass instances (annotations.py)
  - Mouse / keyboard / wheel event handling
  - Zoom helpers and viewport navigation
  - Distance measurement ruler (middle-mouse drag, drawn in drawForeground)
  - PNG export delegated to ImageExporter (image_exporter.py)

What is NOT here
----------------
  - File saving / PNG export  →  image_exporter.py  (ImageExporter)
  - Background thread workers →  workers.py          (TiffLoader, CsvUploader)
  - Annotation data model     →  annotations.py      (Annotation dataclass)
  - Control panel UI          →  control_dock.py     (ControlDock)
  - WCS math                  →  wcs_utils.py        (WCSConverter)
"""

import os
import re
import gc

import numpy as np
from PIL import Image
from astropy.coordinates import SkyCoord
import astropy.units as u

from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint, QThread, pyqtSignal, QObject, QSize, QSizeF
from PyQt5.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QFont,
                         QCursor, QTransform)
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QFileDialog, QApplication, QMessageBox, QMenu,
    QGraphicsEllipseItem, QWidget, QListWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QLabel, QDialog, QStatusBar,
    QGraphicsRectItem, QGraphicsLineItem, QGraphicsItemGroup,
)

from .annotations     import Annotation
from .catalog_manager import CatalogManager
from .image_exporter  import ImageExporter
from .workers         import TiffLoader
from .wcs_utils       import WCSConverter

# Astropy sometimes does not recognise the 'NA' unit used in Euclid FITS files.
# Define it once at import time so every module that reads those files benefits.
try:
    u.def_unit('NA', u.dimensionless_unscaled, register_to_subclass=True)
except (TypeError, ValueError):
    u.def_unit('NA', u.dimensionless_unscaled)


class ImageViewer(QGraphicsView):
    """
    Central display widget for a Euclid MER tile.

    Signals
    -------
    image_loaded(np.ndarray, dict, str)
        Emitted after a TIFF has been successfully loaded and processed.
        Carries (image_array, metadata_dict, file_path) for any listener
        that needs to react to a new image (e.g. external analysis tools).
    """

    image_loaded = pyqtSignal(np.ndarray, dict, str)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, main_window=None):
        super().__init__()

        # ---- Qt scene setup ----
        self.main_window = main_window
        self.scene = QGraphicsScene()
        # BspTreeIndex speeds up hit-testing when many overlay items are present
        self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet(self._scrollbar_stylesheet())

        # ---- Image state ----
        self.image_item     = None   # QGraphicsPixmapItem currently in the scene
        self.original_image = None   # Raw uint16 numpy array from TIFF
        self.metadata       = None   # JSON metadata dict from TIFF ImageDescription tag
        self.qimage         = None   # Full-res uint8 QImage (post-LUT), kept alive for raw-pointer safety
        self.last_pixmap    = None   # Most recent full-image QPixmap
        self.preview_image  = None   # Downscaled thumbnail numpy array for the dock navigator
        self.wcs            = None   # WCSConverter for this tile
        self.scale_factor   = 1.0   # Cumulative zoom factor; used by reset_zoom()
        self.is_displaying_preview = False  # True while contrast slider preview crop is shown

        # ---- File / tile identity ----
        self.filepath   = None
        self.dirpath    = None
        self.tileID     = ""
        self.title      = ""

        # ---- Catalog and user annotations ----
        self.catalog_manager = None
        # annotations replaces the old 'circles' list of 5-tuples.
        # Each element is an Annotation dataclass (see annotations.py).
        self.annotations: list = []

        # ---- Control dock reference (injected via set_control_dock) ----
        self.control_dock = None

        # ---- Background load thread ----
        self.load_thread = None
        self.load_worker = None

        # ---- Contrast engine state ----
        # Maps (min_val, max_val) -> uint8 numpy LUT array
        self.contrast_luts16: dict = {}
        # Keeps the numpy buffer alive while QImage holds a raw pointer to it
        self._contrast_buffer         = None
        # Viewport crop captured once on slider_pressed for the live preview path
        self._preview_crop_raw        = None  # uint16 numpy view / downsampled copy
        self._preview_crop_scene_pos  = None  # QPointF: top-left of crop in scene coords
        self._preview_crop_scene_size = None  # QSizeF: scene extent when downsampled

        # ---- Mouse interaction state ----
        self.start_point        = None  # QPointF: scene pos at mouse-press
        self.end_point          = None  # QPointF: scene pos at mouse-release
        self.current_point      = None  # QPointF: scene pos updated on every move
        self.start_point_screen = None   # set on left-press; None until first press
        self.end_point_screen   = None
        self.start_ra_dec       = None  # (ra, dec) cached at press for ruler + annotation
        self.rubber_band        = None  # QGraphicsRectItem shown in screenshot mode
        self.crosshair          = None  # QGraphicsItemGroup crosshair inside rubber-band
        self.crosshair_ra       = 0.0
        self.crosshair_dec      = 0.0
        self.callback_on_selection = None  # called with QRectF after rubber-band drag

        # ---- Rectangle-selection / screenshot mode ----
        self.rectangle_selection = False
        self.selection_rect      = None

        # ---- Distance measurement (middle-mouse drag) ----
        self.is_measuring       = False
        self.angular_offset     = None
        self.offset_unit        = None
        self.horizontal_offset  = None
        self.horizontal_unit    = None
        self.vertical_offset    = None
        self.vertical_unit      = None

        # ---- Sub-components ----
        # ImageExporter handles all PNG save / screenshot operations
        self.exporter   = ImageExporter(self)
        self.status_bar = QStatusBar()

        # Refresh the navigator thumbnail whenever the user scrolls
        self.horizontalScrollBar().valueChanged.connect(self.refresh_preview)
        self.verticalScrollBar().valueChanged.connect(self.refresh_preview)

        QApplication.setOverrideCursor(Qt.ArrowCursor)

    # ------------------------------------------------------------------
    # Scrollbar stylesheet
    # ------------------------------------------------------------------

    def _scrollbar_stylesheet(self) -> str:
        """CSS string that gives the scroll bars a blue handle on a grey trough."""
        h = "#87CEEB"
        return f"""
            QScrollBar:vertical   {{ background: #E0E0E0; }}
            QScrollBar:horizontal {{ background: #E0E0E0; }}
            QScrollBar::handle:vertical {{
                background: {h}; min-height: 20px; border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background: {h}; min-width: 20px; border-radius: 6px;
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{ border: none; background: none; }}
        """

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def update_status(self, message: str, timeout: int = 5000):
        """Push *message* to the main window status bar for *timeout* ms."""
        if self.main_window:
            self.main_window.statusBar().showMessage(message, timeout)

    def set_main_window_title(self, title: str):
        if self.main_window:
            self.main_window.setWindowTitle(title)

    def extract_tileID(self, filepath: str):
        """
        Extract the TILE identifier from a Euclid TIFF filename.

        Returns (raw, spaced) e.g. ('TILE101794875', 'TILE 101794875'),
        or (None, None) if the pattern is not found.
        """
        match = re.search(r'(TILE\d+)\D', os.path.basename(filepath))
        if match:
            raw = match.group(1)
            return raw, raw.replace('TILE', 'TILE ')
        return None, None

    # ------------------------------------------------------------------
    # Control dock wiring
    # ------------------------------------------------------------------

    def set_control_dock(self, dock):
        """
        Inject the ControlDock and wire up all callbacks.
        Called once from MainWindow after both widgets have been created.
        """
        self.control_dock = dock
        dock.set_load_callback(self.open_file_dialog)
        dock.set_slider_press_callback(self.capture_preview_crop)
        dock.set_contrast_callback(self.apply_preview_contrast)
        dock.set_full_contrast_callback(self.apply_contrast)

    def discard_preview_crop(self):
        """
        Release the temporary contrast-preview crop.
        Called by ControlDock.slider_released so it does not poke
        private attributes directly.
        """
        self._preview_crop_raw        = None
        self._preview_crop_scene_pos  = None
        self._preview_crop_scene_size = None

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def open_file_dialog(self):
        """Show a Qt file-open dialog and start loading the chosen TIFF."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MER TILE TIFF", "",
            "TIFF files (*TILE*.tif *TILE*.tiff)"
        )
        if path:
            self.filepath = path
            self.dirpath  = os.path.dirname(path)
            self.load_image(path)

    def load_image(self, path: str):
        """
        Spin up a TiffLoader worker thread for *path*.
        Any already-running load is cleanly aborted first to prevent
        two threads from racing to call on_image_loaded.
        """
        if self.load_thread and self.load_thread.isRunning():
            self.load_thread.quit()
            self.load_thread.wait()
            self.load_thread = None
            self.load_worker = None

        self.load_thread = QThread()
        self.load_worker = TiffLoader(path)
        self.load_worker.moveToThread(self.load_thread)

        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.finished.connect(self.load_thread.quit)
        self.load_worker.finished.connect(self.load_worker.deleteLater)
        self.load_worker.finished.connect(self.on_image_loaded)
        self.load_worker.error.connect(self.on_load_error)
        self.load_thread.start()

    def on_image_loaded(self, image: np.ndarray, metadata: dict, path: str):
        """
        Main-thread slot called by TiffLoader.finished.

        Builds the initial contrast display, creates the navigator thumbnail,
        and loads the matching MER catalog.
        """
        try:
            self.reset()

            self.update_status("Processing TIFF …")
            self.original_image = image
            self.metadata       = metadata
            self.wcs            = WCSConverter(metadata)
            if self.wcs is None:
                raise ValueError("WCSConverter returned None")

            h, w = image.shape[:2]

            # Build a downscaled navigator thumbnail (longest axis ≤ 1000 px)
            scale = min(1000 / max(h, w), 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            if image.ndim == 3:
                self.update_status("Building navigator thumbnail …")
                img8    = (image >> 8).astype(np.uint8) if image.dtype == np.uint16 else image.astype(np.uint8)
                pil_img = Image.fromarray(img8).resize((new_w, new_h), Image.LANCZOS)
                self.preview_image = np.array(pil_img)
            else:
                from scipy.ndimage import zoom as scipy_zoom
                self.preview_image = scipy_zoom(image, scale, order=1)

            # Apply default full-range contrast and show the image
            self.update_status("Applying contrast …")
            self.control_dock.max_slider.setValue(65535)
            self.apply_contrast(0, 65535)
            self.centerOn(QPointF(w / 2, h / 2))

            # Populate the navigator with the current viewport crop
            self.update_status("Updating navigator …")
            self.get_visible_qimage_pixmap()

            # Tile identity and MER catalog
            self.tileID, self.title = self.extract_tileID(path)
            if self.title:
                self.set_main_window_title(f"Euniverse Explorer – {self.title}")
            if self.tileID:
                self.update_status("Loading MER catalog …")
                self.catalog_manager = CatalogManager(
                    self.tileID, self.wcs, os.path.dirname(path),
                    image_viewer=self
                )
            else:
                self.catalog_manager = None
                self.update_status("No TILE id in filename — catalog skipped.")

            self.default_image = path
            self.image_loaded.emit(image, metadata, path)
            n = self.catalog_manager.numsources if self.catalog_manager else 0
            self.update_status(f"TIFF loaded — {n} MER sources found.")

        except ValueError as ve:
            self.on_load_error(str(ve))
        except Exception as e:
            self.on_load_error(f"Unexpected error in on_image_loaded: {e}")

        QApplication.restoreOverrideCursor()

    def on_load_error(self, message: str):
        """Slot for TiffLoader.error and internal on_image_loaded exceptions."""
        self.update_status(f"Error loading TIFF: {message}")
        print(f"Error loading TIFF: {message}")
        if self.load_thread:
            self.load_thread.quit()
            self.load_thread.wait()
            self.load_thread = None
        if self.load_worker:
            self.load_worker.deleteLater()
            self.load_worker = None

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset(self):
        """
        Reset all viewer state in preparation for loading a new image.

        Order matters:
          0. Abort any in-flight background thread first.
          1. Remove annotation circles before scene.clear() to avoid dangling
             C++ wrappers (QGraphicsEllipseItem destroyed by scene.clear while
             Python still holds a reference to the wrapper object → crash).
          2. Remove MER overlays.
          3. Release large numpy arrays so memory is reclaimed promptly.
          4. Call scene.clear() — safe now that all items are removed.
          5. Force a GC cycle.
        """
        # 0. Abort in-flight load
        if self.load_thread and self.load_thread.isRunning():
            self.load_thread.quit()
            self.load_thread.wait()
        self.load_thread = None
        self.load_worker = None

        # 1. Annotation circles (must precede scene.clear)
        self.clear_annotations()
        if self.control_dock:
            self.control_dock.set_black_squares()

        # 2. MER overlays
        self.clear_MER()
        self.image_item = None

        # 3. Catalog reference
        if self.catalog_manager:
            self.catalog_manager.catalog = None
        self.catalog_manager = None

        # 4. Measurement state
        self.clear_measuring_state()

        # 5. Large arrays and contrast buffers
        self.original_image           = None
        self._contrast_buffer         = None
        self._preview_crop_raw        = None
        self._preview_crop_scene_pos  = None
        self._preview_crop_scene_size = None

        # 6. Scene
        self.scene.clear()
        gc.collect()

    # ------------------------------------------------------------------
    # MER catalog overlay
    # ------------------------------------------------------------------

    def toggle_MER(self):
        """
        Show or hide the MER catalog ellipse overlays.

        First call builds the item list from CatalogManager and adds them
        to the scene.  Subsequent calls flip their visibility flag.
        """
        if self.catalog_manager is None or self.original_image is None:
            return

        if not self.catalog_manager.MER_items:
            # First activation — build and add all ellipses
            self.catalog_manager.get_MER(self.original_image.shape[0])
            self.setUpdatesEnabled(False)
            try:
                for ellipse in self.catalog_manager.MER_items:
                    self.scene.addItem(ellipse)
                    ellipse.setVisible(True)
            finally:
                self.setUpdatesEnabled(True)
                self.scene.update()
        else:
            # Toggle existing items
            new_vis = not self.catalog_manager.MER_items[0].isVisible()
            for ellipse in self.catalog_manager.MER_items:
                ellipse.setVisible(new_vis)
            self.scene.update()

    def clear_MER(self):
        """
        Permanently remove all MER overlay items and reset the item list.
        The next toggle_MER call will rebuild everything from scratch.
        """
        if self.catalog_manager and self.catalog_manager.MER_items:
            for ellipse in self.catalog_manager.MER_items:
                if ellipse.scene() == self.scene:
                    self.scene.removeItem(ellipse)
            self.catalog_manager.MER_items = []
            self.scene.update()

    def display_selected_MER(self, object_ids: list):
        """
        Show ellipses for a specific subset of catalog IDs.
        Called by CatalogManager.handle_selected_objects after a lasso selection
        in the scatter plot.
        """
        if not self.catalog_manager or self.original_image is None:
            return
        self.clear_selected_MER()
        items = self.catalog_manager.get_selected_MER(object_ids, self.original_image.shape[0])
        if items:
            self.setUpdatesEnabled(False)
            try:
                for item in items:
                    self.scene.addItem(item)
                    item.setVisible(True)
                self.centerOn(items[0].rect().center())
            finally:
                self.setUpdatesEnabled(True)
                self.scene.update()

    def clear_selected_MER(self):
        """Remove the lasso-selected subset of ellipses from the scene."""
        if self.catalog_manager and self.catalog_manager.selected_MER_items:
            for item in self.catalog_manager.selected_MER_items:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            self.catalog_manager.selected_MER_items = []
            self.scene.update()

    def toggle_selected_MER(self):
        """Toggle visibility of the current lasso-selected subset."""
        items = getattr(self.catalog_manager, 'selected_MER_items', [])
        if items:
            new_vis = not items[0].isVisible()
            for item in items:
                item.setVisible(new_vis)
            self.scene.update()

    # ------------------------------------------------------------------
    # User annotation circles
    # ------------------------------------------------------------------

    def clear_annotations(self):
        """
        Remove all user annotation circles from the scene and reset the list.

        Must be called before scene.clear() to prevent dangling C++ object
        wrappers.  sip.isdeleted() is used as a defensive backstop in case
        some items were already destroyed by an earlier scene.clear().
        """
        import sip
        for ann in self.annotations:
            try:
                if not sip.isdeleted(ann.item) and ann.item.scene() == self.scene:
                    self.scene.removeItem(ann.item)
            except RuntimeError:
                pass  # C++ object already gone — nothing to do
        self.annotations = []

        if self.control_dock:
            self.control_dock.selected_circle = None
            self.control_dock.coord_list.clear()
            try:
                self.control_dock.submit_targets_button.setEnabled(False)
                self.control_dock.save_targets_button.setEnabled(False)
            except RuntimeError:
                pass

    # ------------------------------------------------------------------
    # Navigator thumbnail extraction
    # ------------------------------------------------------------------

    def get_visible_qimage_pixmap(self):
        """
        Extract the currently visible scene area from self.qimage and store
        it as a numpy array in self.preview_image.

        Called after the initial load and after each full contrast pass so
        the navigator thumbnail always reflects the current display state.
        Operates on the rendered uint8 QImage so the thumbnail matches exactly
        what is shown on screen.
        """
        if not self.scene or self.qimage is None:
            return

        visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        image_item   = next(
            (i for i in self.scene.items() if isinstance(i, QGraphicsPixmapItem)), None
        )
        if image_item is None:
            return

        intersection = visible_rect.intersected(image_item.boundingRect())
        if intersection.isEmpty():
            return

        tl = image_item.mapFromScene(intersection.topLeft()).toPoint()
        br = image_item.mapFromScene(intersection.bottomRight()).toPoint()
        x  = max(0, tl.x())
        y  = max(0, tl.y())
        w  = min(br.x(), self.qimage.width()  - 1) - x + 1
        h  = min(br.y(), self.qimage.height() - 1) - y + 1

        if w <= 0 or h <= 0:
            return

        qimg = self.qimage.copy(x, y, w, h)
        fmt  = qimg.format()

        if fmt in (QImage.Format_RGB32, QImage.Format_ARGB32):
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)
            self.preview_image = arr[:, :, :3][:, :, ::-1]   # BGRA → RGB
        elif fmt == QImage.Format_RGB888:
            ptr    = qimg.bits()
            ptr.setsize(qimg.byteCount())
            stride = qimg.bytesPerLine()
            arr    = np.frombuffer(ptr, dtype=np.uint8).reshape(qimg.height(), stride)
            self.preview_image = arr[:, :qimg.width() * 3].reshape(qimg.height(), qimg.width(), 3)
        else:
            print(f"Warning: unsupported QImage format {fmt} in get_visible_qimage_pixmap")
            self.preview_image = None

    # ------------------------------------------------------------------
    # Contrast engine
    # ------------------------------------------------------------------
    #
    # Two modes:
    #
    # FULL mode (slider released / initial load)
    #   apply_contrast(min, max)
    #   Builds a uint16→uint8 LUT (cached), applies lut[original_image],
    #   stores self.qimage, updates the scene pixmap, resets sceneRect.
    #
    # PREVIEW mode (slider held)
    #   1. capture_preview_crop() — called once on slider_pressed.
    #      Slices the visible region from original_image (zero-copy view),
    #      optionally nearest-neighbour downsamples to viewport size.
    #   2. apply_preview_contrast(min, max) — called on every slider tick.
    #      Stretches the crop with a reusable float32 scratch buffer,
    #      updates image_item pixmap in-place. sceneRect never changes.
    # ------------------------------------------------------------------

    _MAX_LUT_CACHE = 64   # Evict oldest entries beyond this many cached LUTs

    def _trim_lut_cache(self, cache: dict):
        """Evict oldest cache entries when cache exceeds _MAX_LUT_CACHE entries."""
        while len(cache) > self._MAX_LUT_CACHE:
            cache.pop(next(iter(cache)))

    def create_contrast_lut(self, min_val, max_val,
                            input_dtype, output_dtype=np.uint8) -> np.ndarray:
        """
        Build a linear-stretch lookup table: input_dtype values → uint8 [0..255].

        Values below min_val → 0, above max_val → 255.
        The 65 536-entry float32 intermediate fits in L2 cache, so plain numpy
        is sufficient here.
        """
        n    = np.iinfo(input_dtype).max + 1
        diff = float(max_val - min_val)
        if diff == 0:
            return np.full(n, 255, dtype=output_dtype)
        indices = np.arange(n, dtype=np.float32)
        return np.clip((indices - min_val) * (255.0 / diff), 0, 255).astype(output_dtype)

    def capture_preview_crop(self):
        """
        Slice the currently visible viewport region from original_image and
        cache it in _preview_crop_raw for use by apply_preview_contrast.

        Called once on slider_pressed before any dragging starts, so each tick
        operates on the same crop without re-reading the image.

        If the crop is larger than the physical viewport (user is zoomed out),
        it is nearest-neighbour downsampled to viewport pixel dimensions.
        This keeps per-tick work proportional to the pixels actually rendered,
        regardless of zoom level.
        """
        if self.original_image is None or self.image_item is None:
            return

        # Compute crop bounds, clamped to image extent
        visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        img_h, img_w = self.original_image.shape[:2]
        crop_rect    = visible_rect.intersected(QRectF(0, 0, img_w, img_h))
        if crop_rect.isEmpty():
            return

        x0 = max(0, int(crop_rect.left()))
        y0 = max(0, int(crop_rect.top()))
        x1 = min(img_w, int(crop_rect.right())  + 1)
        y1 = min(img_h, int(crop_rect.bottom()) + 1)

        crop = self.original_image[y0:y1, x0:x1]   # zero-copy numpy view

        # Downsample to viewport resolution when zoomed out
        vp    = self.viewport()
        dpr   = vp.devicePixelRatio() if hasattr(vp, 'devicePixelRatio') else 1.0
        vp_h  = vp.height() * dpr
        vp_w  = vp.width()  * dpr
        crop_h, crop_w = crop.shape[:2]
        scale = min(vp_h / crop_h, vp_w / crop_w, 1.0)

        if scale < 1.0:
            new_h   = max(1, int(round(crop_h * scale)))
            new_w   = max(1, int(round(crop_w * scale)))
            # Integer index arrays give nearest-neighbour without cv2/scipy
            row_idx = (np.arange(new_h) * (crop_h / new_h)).astype(np.intp)
            col_idx = (np.arange(new_w) * (crop_w / new_w)).astype(np.intp)
            # np.ix_ produces a copy — intentional; _preview_float_buf must not
            # alias original_image memory
            crop = crop[np.ix_(row_idx, col_idx)]
            self._preview_crop_scene_size = QSizeF(x1 - x0, y1 - y0)
        else:
            self._preview_crop_scene_size = None   # 1:1, no transform needed

        self._preview_crop_raw       = crop
        self._preview_crop_scene_pos = QPointF(x0, y0)

    def apply_preview_contrast(self, min_val: int, max_val: int):
        """
        Stretch _preview_crop_raw in-place and update image_item without
        touching sceneRect, preserving scroll position and zoom level.

        Key implementation detail: the uint16 → float32 widening MUST happen
        before the subtraction.  Using casting='unsafe' in np.subtract would
        evaluate (uint16 - min_val) in uint16 arithmetic first, causing wrap-
        around for pixels below min_val (e.g. 1000 - 5000 → 61536).
        Instead we widen with np.copyto first, then do arithmetic in float32.
        """
        if self._preview_crop_raw is None:
            return

        crop = self._preview_crop_raw
        diff = float(max_val - min_val)
        h, w = crop.shape[:2]
        c    = 3 if crop.ndim == 3 else 1

        # Allocate or reuse the float32 scratch buffer
        if (not hasattr(self, '_preview_float_buf') or
                self._preview_float_buf.shape != crop.shape):
            self._preview_float_buf = np.empty(crop.shape, dtype=np.float32)

        if diff == 0:
            out = np.full(crop.shape, 255, dtype=np.uint8)
        else:
            scale = 255.0 / diff
            np.copyto(self._preview_float_buf, crop, casting='unsafe')  # uint16 → float32
            self._preview_float_buf -= min_val   # in-place; no extra allocation
            self._preview_float_buf *= scale
            np.clip(self._preview_float_buf, 0, 255, out=self._preview_float_buf)
            out = self._preview_float_buf.astype(np.uint8)  # final copy for QImage

        stride = c * w
        # Keep buffer alive — QImage holds a raw pointer, not a copy
        self._contrast_buffer = out
        qimg   = QImage(self._contrast_buffer.data, w, h, stride,
                        QImage.Format_RGB888 if c == 3 else QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        self.image_item.setPixmap(pixmap)
        self.image_item.setPos(self._preview_crop_scene_pos)

        # If the crop was downsampled, scale the item to cover its scene extent
        if self._preview_crop_scene_size is not None:
            sx = self._preview_crop_scene_size.width()  / w
            sy = self._preview_crop_scene_size.height() / h
            self.image_item.setTransform(QTransform.fromScale(sx, sy))
        else:
            self.image_item.setTransform(QTransform())   # identity

        self.is_displaying_preview = True

    def apply_contrast(self, min_val: int, max_val: int):
        """
        Full-image LUT pass: build / look up a uint16→uint8 LUT, apply it to
        original_image, store the result as self.qimage, and update the scene.

        Called on slider release and on initial image load.
        Always restores image_item to position (0,0) and rebuilds sceneRect.
        """
        if self.original_image is None:
            return

        lut_key = (min_val, max_val)
        if lut_key not in self.contrast_luts16:
            self.contrast_luts16[lut_key] = self.create_contrast_lut(
                min_val, max_val, self.original_image.dtype, np.uint8
            )
            self._trim_lut_cache(self.contrast_luts16)
        lut = self.contrast_luts16[lut_key]

        stretched = lut[self.original_image]   # uint8, same shape as original_image
        h, w      = stretched.shape[:2]
        c         = 3 if stretched.ndim == 3 else 1
        stride    = c * w

        # Keep buffer alive — QImage does NOT copy the data
        self._contrast_buffer = stretched
        self.qimage = QImage(
            self._contrast_buffer.data, w, h, stride,
            QImage.Format_RGB888 if c == 3 else QImage.Format_Grayscale8
        )
        pixmap           = QPixmap.fromImage(self.qimage)
        self.last_pixmap = pixmap
        self.is_displaying_preview = False

        if self.image_item:
            self.image_item.setPixmap(pixmap)
            self.image_item.setPos(0, 0)
            self.image_item.setTransform(QTransform())   # clear any preview scale
        else:
            self.image_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.image_item)

        self.setSceneRect(QRectF(0, 0, w, h))

        if self.control_dock:
            self.control_dock.update_preview(self.last_pixmap)

    # ------------------------------------------------------------------
    # Zoom and viewport navigation
    # ------------------------------------------------------------------

    def zoom_in(self):
        self.scale_factor *= 1.2
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(1.2, 1.2)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.refresh_preview()

    def zoom_out(self):
        self.scale_factor /= 1.2
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(1 / 1.2, 1 / 1.2)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.refresh_preview()

    def reset_zoom(self):
        """Return to 1:1 pixel mapping."""
        if self.scale_factor != 1.0:
            self.scale(1.0 / self.scale_factor, 1.0 / self.scale_factor)
            self.scale_factor = 1.0
            self.refresh_preview()

    def reset_transform(self):
        """Hard-reset the view transform (loses scale_factor tracking)."""
        self.resetTransform()
        self.scale_factor = 1.0

    def fit_to_view(self):
        """Scale and centre the image so the whole tile fits the viewport."""
        if self.last_pixmap and self.image_item:
            self.reset_zoom()
            view_rect  = self.viewport().rect()
            scene_rect = self.sceneRect()
            scale = min(view_rect.width()  / scene_rect.width(),
                        view_rect.height() / scene_rect.height())
            self.scale_factor = scale
            self.scale(scale, scale)
            self.centerOn(scene_rect.center())
            self.refresh_preview()

    def refresh_preview(self):
        """Ask the dock to redraw the navigator thumbnail."""
        if self.control_dock and self.last_pixmap:
            self.control_dock.update_preview(self.last_pixmap)

    def get_current_view_center(self) -> QPointF:
        """Return the scene point currently at the physical viewport centre."""
        return self.mapToScene(self.viewport().rect().center())

    def restore_view_center(self, scene_center: QPointF):
        """Centre the view on *scene_center*."""
        self.centerOn(scene_center)

    # ------------------------------------------------------------------
    # Export / screenshot — thin wrappers around ImageExporter
    # ------------------------------------------------------------------

    def start_selection(self):
        """Enter rubber-band screenshot mode (delegates to ImageExporter)."""
        self.exporter.start_selection()

    def save_full_image_with_overlays(self):
        self.exporter.save_full_image_with_overlays()

    def save_visible_area_with_overlays(self):
        self.exporter.save_visible_area_with_overlays()

    def save_area_from_selection(self, rect: QRectF):
        self.exporter.save_area_from_selection(rect)

    # ------------------------------------------------------------------
    # Crosshair helper (rubber-band mode)
    # ------------------------------------------------------------------

    def create_crosshair(self, parent_item) -> QGraphicsItemGroup:
        """
        Build a small red crosshair QGraphicsItemGroup parented to parent_item.
        Repositioned in mouseMoveEvent while rubber-band selection is active.
        """
        length = 5.0
        pen    = QPen(QColor("red"), 1, Qt.SolidLine)
        v_line = QGraphicsLineItem(0, -length, 0, length)
        h_line = QGraphicsLineItem(-length, 0, length, 0)
        v_line.setPen(pen)
        h_line.setPen(pen)
        group = QGraphicsItemGroup(parent_item)
        group.addToGroup(v_line)
        group.addToGroup(h_line)
        group.setPos(0, 0)
        return group

    # ------------------------------------------------------------------
    # Measurement state
    # ------------------------------------------------------------------

    def clear_measuring_state(self):
        """Reset all distance-measurement variables (called on middle-button release)."""
        self.is_measuring       = False
        self.start_point        = None
        self.start_ra_dec       = None
        self.angular_offset     = None
        self.offset_unit        = None
        self.horizontal_offset  = None
        self.horizontal_unit    = None
        self.vertical_offset    = None
        self.vertical_unit      = None

    # ------------------------------------------------------------------
    # Qt event handlers
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        self.start_point = scene_pos

        # Cache WCS coordinates at press for the ruler and annotation placement
        if self.wcs and self.original_image is not None:
            fy = self.original_image.shape[0] - scene_pos.y()
            self.start_ra_dec = self.wcs.pixel_to_world(scene_pos.x(), fy)

        # Record screen-space press position for the click-vs-drag test in
        # mouseReleaseEvent.  Done before the rubber-band early-return so the
        # value is always fresh for the matching release.
        if event.button() == Qt.LeftButton:
            self.start_point_screen = event.pos()

        # Rubber-band / screenshot mode
        if self.rectangle_selection:
            if event.button() == Qt.LeftButton:
                self.rubber_band = QGraphicsRectItem(QRectF(self.start_point, QPointF()))
                pen = QPen(QColor("white"), 1, Qt.DashLine)
                self.rubber_band.setPen(pen)
                self.scene.addItem(self.rubber_band)
                self.crosshair = self.create_crosshair(self.rubber_band)
            return   # Don't call super(); that would activate scroll drag

        elif event.button() == Qt.MidButton:
            self.is_measuring = True
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().update()
            return

        elif event.button() == Qt.RightButton:
            import sip
            # Remove an existing circle if the click is close enough to one
            for ann in self.annotations[:]:
                if sip.isdeleted(ann.item):
                    self.annotations.remove(ann)
                    continue
                centre = ann.item.rect().center()
                dist   = ((scene_pos.x() - centre.x()) ** 2 +
                          (scene_pos.y() - centre.y()) ** 2) ** 0.5
                if dist <= 10:
                    self.scene.removeItem(ann.item)
                    self.annotations.remove(ann)
                    if self.control_dock:
                        self.control_dock.update_coord_list(self.annotations)
                        if self.control_dock.selected_circle == ann.item:
                            self.control_dock.selected_circle = None
                    self.viewport().update()
                    return
            # No circle hit — show the classification context menu
            self._show_annotation_menu(event, scene_pos)

        super().mousePressEvent(event)

    def _show_annotation_menu(self, event, scene_pos: QPointF):
        """Build and execute the right-click annotation classification menu."""
        menu = QMenu(self)

        categories = {
            "GL":  ["lens", "arc", "multiple image", "Einstein ring", "DSPL"],
            "AGN": ["Seyfert 1", "outflow"],
            "Gx":  ["Emissionline", "Ring", "Polar ring", "Stream",
                    "Merger", "Irregular", "Dwarf", "weird"],
        }
        color_map = {
            "GL":  QColor(0, 200, 255),
            "AGN": QColor(220, 220, 0),
            "Gx":  QColor(255, 50, 50),
        }

        action_to_label: dict = {}
        for cat, entries in categories.items():
            for label in entries:
                full_label = f"{cat}: {label}"
                action_to_label[menu.addAction(full_label)] = full_label
            menu.addSeparator()

        chosen = menu.exec_(self.mapToGlobal(event.pos()))
        if chosen and self.wcs and self.original_image is not None and self.start_ra_dec:
            classifier = action_to_label[chosen]
            cat_prefix = classifier.split(":")[0]
            color      = color_map.get(cat_prefix, QColor(255, 255, 255))
            thickness  = 2.0

            item = QGraphicsEllipseItem(scene_pos.x() - 10, scene_pos.y() - 10, 20, 20)
            item.setPen(QPen(color, thickness))
            item.setZValue(10)
            self.scene.addItem(item)

            self.annotations.append(Annotation(
                item=item,
                ra=self.start_ra_dec[0],
                dec=self.start_ra_dec[1],
                classifier=classifier,
                normal_thickness=thickness,
            ))
            if self.control_dock:
                self.control_dock.update_coord_list(self.annotations)
            self.viewport().update()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

        if not self.wcs or not self.control_dock or self.original_image is None:
            return

        self.current_point = self.mapToScene(event.pos())
        # FITS images are stored bottom-up; Qt renders top-down — flip y before WCS calls
        fy = self.original_image.shape[0] - self.current_point.y()

        # Only update the magnifier when the cursor is actually over the image;
        # outside the image rect the magnifier would show edge pixels, which is
        # more confusing than showing nothing.
        if self.sceneRect().contains(self.current_point):
            self.control_dock.update_magnifier(self.current_point)
            try:
                ra, dec = self.wcs.pixel_to_world(self.current_point.x(), fy)
                self.control_dock.update_cursor_display(self.current_point.x(), fy, ra, dec)
            except Exception:
                pass
        else:
            self.control_dock.update_cursor_display(None, None, None, None)
            return

        # Rubber-band and measurement are mutually exclusive modes; the
        # rubber-band check comes first so it returns before measurement logic.
        if self.rectangle_selection and self.rubber_band:
            rect = QRectF(self.start_point, self.current_point).normalized()
            self.rubber_band.setRect(rect)
            if self.crosshair and self.crosshair.scene():
                self.crosshair.setPos(rect.center())
                ch_x = rect.center().x()
                ch_y = self.original_image.shape[0] - rect.center().y()
                self.crosshair_ra, self.crosshair_dec = self.wcs.pixel_to_world(ch_x, ch_y)
            return

        # Distance measurement ruler update
        if self.is_measuring and self.start_point and self.start_ra_dec:
            try:
                ra, dec = self.wcs.pixel_to_world(self.current_point.x(), fy)
                self._update_measurement(ra, dec)
            except Exception:
                pass
            self.viewport().update()

    def _update_measurement(self, ra: float, dec: float):
        """
        Recompute the ruler labels from the start WCS position to (ra, dec).

        Separations are converted to arcseconds or arcminutes depending on
        magnitude and stored as (value, unit_string) pairs for drawForeground.
        """
        def _to_unit(deg: float):
            if deg * 3600 < 60:
                return deg * 3600, '"'
            elif deg < 1:
                return deg * 60, "'"
            return None, None

        s0 = SkyCoord(self.start_ra_dec[0] * u.deg, self.start_ra_dec[1] * u.deg, frame='icrs')
        s1 = SkyCoord(ra * u.deg,                   dec * u.deg,                   frame='icrs')
        sh = SkyCoord(ra * u.deg,                   self.start_ra_dec[1] * u.deg, frame='icrs')
        sv = SkyCoord(self.start_ra_dec[0] * u.deg, dec * u.deg,                  frame='icrs')

        self.angular_offset,    self.offset_unit    = _to_unit(s0.separation(s1).deg)
        self.horizontal_offset, self.horizontal_unit = _to_unit(s0.separation(sh).deg)
        self.vertical_offset,   self.vertical_unit   = _to_unit(s0.separation(sv).deg)

    def mouseReleaseEvent(self, event):
        self.end_point        = self.mapToScene(event.pos())
        self.end_point_screen = event.pos()

        # Rubber-band completion: only fire on left-button release so that an
        # accidental right- or middle-click during a rubber-band drag does not
        # prematurely trigger the screenshot callback.
        if self.rectangle_selection:
            if event.button() == Qt.LeftButton:
                self._handle_screenshot_release()
            return

        if event.button() == Qt.LeftButton:
            # Guard: start_point_screen is None if no left-press was recorded
            # (e.g. the press happened outside the widget).
            if self.start_point_screen is not None:
                was_click = (self.end_point_screen - self.start_point_screen).manhattanLength() < 5
                self.start_point_screen = None   # consumed; reset for next press
                if was_click:
                    # Always call super() first so Qt cancels the ScrollHandDrag
                    # that it armed on press — skipping it leaves the view in a
                    # dragging state and the image pans on the next mouse move.
                    super().mouseReleaseEvent(event)
                    self._handle_object_click(self.end_point)
                    return
            self.refresh_preview()

        elif event.button() == Qt.MidButton:
            self.clear_measuring_state()
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().update()

        super().mouseReleaseEvent(event)

    def _handle_screenshot_release(self):
        """Finalise the rubber-band rectangle and fire callback_on_selection."""
        if self.callback_on_selection:
            if self.start_point is None or self.end_point is None:
                return   # press happened outside the widget — nothing to do
            self.selection_rect = QRectF(self.start_point, self.end_point).normalized()
            if self.rubber_band:
                self.scene.removeItem(self.rubber_band)
                self.rubber_band = None
            self.setCursor(Qt.ArrowCursor)
            cb = self.callback_on_selection
            # Clear state before calling cb so re-entrant calls are safe
            self.callback_on_selection = None
            self.start_point           = None
            self.end_point             = None
            self.rectangle_selection   = False
            cb(self.selection_rect)

    def _handle_object_click(self, scene_pos: QPointF) -> bool:
        """
        Test whether a left-click at *scene_pos* hits a MER ellipse or an
        annotation circle.  Applies highlight styling and returns True if an
        object was handled (suppresses the default preview refresh).
        """
        import sip

        # Hit radius in scene space.  10 screen pixels divided by the current
        # zoom factor converts to the equivalent scene-space radius so the
        # click target stays the same physical size regardless of zoom level.
        hit_radius = 10.0 / max(self.scale_factor, 0.01)

        # MER catalog ellipses — use the scene's BspTree spatial index to get
        # only the items near the click point instead of scanning all N items.
        if self.catalog_manager and self.catalog_manager.MER_items:
            search_rect = QRectF(
                scene_pos.x() - hit_radius, scene_pos.y() - hit_radius,
                hit_radius * 2,             hit_radius * 2,
            )
            nearby = self.scene.items(search_rect)
            for ellipse in nearby:
                if not isinstance(ellipse, QGraphicsEllipseItem):
                    continue
                if sip.isdeleted(ellipse):
                    continue
                if ellipse not in self.catalog_manager.MER_items:
                    continue   # could be an annotation or selection-overlay item
                centre = ellipse.rect().center()
                dist   = ((scene_pos.x() - centre.x()) ** 2 +
                          (scene_pos.y() - centre.y()) ** 2) ** 0.5
                if dist <= hit_radius:
                    oid = ellipse.data(0)
                    if oid is not None and self.control_dock:
                        self.control_dock.select_table_row(oid)
                        # Reset all visible MER items to red, then highlight hit
                        for other in self.catalog_manager.MER_items:
                            if not sip.isdeleted(other) and other.isVisible():
                                other.setPen(QPen(QColor(255, 0, 0), 1.0))
                        ellipse.setPen(QPen(QColor(255, 255, 0), 1.5))
                        self.scene.update()
                    return True

        # User annotation circles
        for ann in self.annotations:
            if sip.isdeleted(ann.item):
                continue
            centre = ann.item.rect().center()
            dist   = ((scene_pos.x() - centre.x()) ** 2 +
                      (scene_pos.y() - centre.y()) ** 2) ** 0.5
            if dist <= hit_radius:
                if self.control_dock:
                    self.control_dock.select_coord_list_item(ann.ra, ann.dec)
                    # Reset the previously selected circle
                    if (self.control_dock.selected_circle and
                            not sip.isdeleted(self.control_dock.selected_circle) and
                            self.control_dock.selected_circle != ann.item):
                        for a in self.annotations:
                            if a.item == self.control_dock.selected_circle:
                                pen = a.item.pen()
                                pen.setWidthF(a.normal_thickness)
                                a.item.setPen(pen)
                                break
                    # Thicken the newly selected circle
                    new_pen = QPen(ann.item.pen().color(), ann.normal_thickness * 1.5)
                    ann.item.setPen(new_pen)
                    self.control_dock.selected_circle = ann.item
                    self.scene.update()
                return True

        return False

    def keyPressEvent(self, event):
        """Delete key removes the currently selected annotation circle."""
        if event.key() == Qt.Key_Delete and self.control_dock:
            import sip
            target_ann = None

            # Priority 1: delete by item identity — the visually thickened
            # (selected_circle) item.  Identity is exact and unambiguous even
            # when two annotations share the same sky position.
            if (self.control_dock.selected_circle and
                    not sip.isdeleted(self.control_dock.selected_circle)):
                for ann in self.annotations:
                    if ann.item is self.control_dock.selected_circle:
                        target_ann = ann
                        break

            # Priority 2: coord_list keyboard selection — fall back to
            # RA/Dec proximity only when no item is highlighted.
            if target_ann is None:
                ra, dec = self.control_dock.get_selected_coord()
                if ra is not None and dec is not None:
                    for ann in self.annotations:
                        if abs(ann.ra - ra) < 1e-5 and abs(ann.dec - dec) < 1e-5:
                            target_ann = ann
                            break

            if target_ann is not None:
                if not sip.isdeleted(target_ann.item):
                    self.scene.removeItem(target_ann.item)
                if self.control_dock.selected_circle is target_ann.item:
                    self.control_dock.selected_circle = None
                self.annotations.remove(target_ann)
                self.control_dock.update_coord_list(self.annotations)
                self.viewport().update()

        super().keyPressEvent(event)

    def wheelEvent(self, event):
        """Scroll wheel zooms, anchored to the cursor position."""
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale_factor *= factor
        self.scale(factor, factor)
        self.refresh_preview()
        event.accept()

    # ------------------------------------------------------------------
    # drawForeground — measurement ruler (Qt virtual hook)
    # ------------------------------------------------------------------

    def drawForeground(self, painter: QPainter, rect):
        """
        Qt virtual hook called automatically each time the viewport is redrawn.
        Draws the distance-measurement ruler overlay when the middle mouse
        button is held down.

        The ruler consists of:
          - A solid diagonal line from start to current cursor position
          - Dashed horizontal and vertical legs forming a right triangle
          - Text labels for the total, horizontal, and vertical separations
        """
        super().drawForeground(painter, rect)

        if not (self.is_measuring and self.start_point and self.current_point):
            return

        painter.setRenderHint(QPainter.Antialiasing)

        # Diagonal line
        solid = QPen(QColor(255, 255, 0), 1, Qt.SolidLine)
        solid.setCosmetic(True)
        painter.setPen(solid)
        painter.drawLine(self.start_point, self.current_point)

        # Right-triangle legs
        dashed = QPen(QColor(255, 255, 0), 1, Qt.DashLine)
        dashed.setCosmetic(True)
        painter.setPen(dashed)
        corner = QPointF(self.current_point.x(), self.start_point.y())
        painter.drawLine(self.start_point, corner)
        painter.drawLine(corner, self.current_point)

        # Text labels — drawn in viewport (screen) coordinates.
        # All mapFromScene() calls must happen BEFORE setWorldMatrixEnabled(False)
        # because mapFromScene() uses the current view transform.  Calling it
        # after disabling the world matrix would give wrong positions if the
        # painter has a non-identity render offset.
        mid_scene = QPointF((self.start_point.x() + self.current_point.x()) / 2,
                             (self.start_point.y() + self.current_point.y()) / 2)
        hor_scene = QPointF(mid_scene.x(), self.start_point.y())
        ver_scene = QPointF(self.current_point.x(), mid_scene.y())

        mid_vp = self.mapFromScene(mid_scene)
        hor_vp = self.mapFromScene(hor_scene)
        ver_vp = self.mapFromScene(ver_scene)

        painter.setWorldMatrixEnabled(False)
        painter.setFont(QFont("Arial", 10))
        painter.setPen(QPen(QColor(255, 255, 0), 1))

        if self.angular_offset is not None:
            painter.drawText(QPointF(mid_vp.x(),       mid_vp.y() - 10),
                             f"{self.angular_offset:.2f} {self.offset_unit}")
        if self.horizontal_offset is not None:
            painter.drawText(QPointF(hor_vp.x(),       hor_vp.y() + 20),
                             f"{self.horizontal_offset:.2f} {self.horizontal_unit}")
        if self.vertical_offset is not None:
            painter.drawText(QPointF(ver_vp.x() + 10,  ver_vp.y()),
                             f"{self.vertical_offset:.2f} {self.vertical_unit}")

        painter.setWorldMatrixEnabled(True)
