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
image_exporter.py — Save and screenshot helpers for ImageViewer
================================================================
All functions that render the scene to a PNG file have been moved here to
keep ImageViewer focused on display and interaction.

The module exposes a single class, ImageExporter, which is instantiated by
ImageViewer and holds a back-reference to it.  Every method delegates the
actual Qt rendering to viewer.scene.render() and hands off file I/O to the
shared _save_image() helper.

Public interface (called from ImageViewer)
------------------------------------------
  exporter.save_full_image_with_overlays()
      Renders the *entire* scene at native resolution with all active
      overlays (MER ellipses, annotation circles).

  exporter.save_visible_area_with_overlays()
      Renders only the currently visible viewport region.

  exporter.save_area_from_selection(rect)
      Renders the user-drawn rubber-band selection rectangle.

  exporter.start_selection()
      Activates the rubber-band mode in the viewer so the user can drag
      a rectangle; on release, save_area_from_selection is called back.
"""

import os

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox


class ImageExporter:
    """
    Handles all PNG export operations for an ImageViewer instance.

    Parameters
    ----------
    viewer : ImageViewer
        The viewer whose scene will be rendered.  The exporter reads
        viewer.scene, viewer.sceneRect(), viewer.viewport(), viewer.wcs,
        viewer.original_image, viewer.dirpath, viewer.tileID, and
        viewer.control_dock.
    """

    def __init__(self, viewer):
        # Keep a reference to the viewer so we can access its scene and state.
        # This is intentionally a plain reference (not a weak ref) — the
        # exporter lifetime is the same as the viewer's.
        self._v = viewer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_image(self, qimage: QImage, filename: str) -> bool:
        """
        Save *qimage* to *filename* as PNG.

        If the file already exists the user is shown a Save-As dialog so they
        can choose a different name or overwrite intentionally.  Returns True
        on success, False if the user cancelled or the write failed.
        """
        v = self._v

        if not os.path.exists(filename):
            if qimage.save(filename, "png"):
                v.update_status(f"Saved: {filename}")
                return True
            QMessageBox.critical(None, "Error", f"Failed to save to: {filename}")
            return False

        # File exists — ask for a new name
        new_name, _ = QFileDialog.getSaveFileName(
            None,
            "File Already Exists — Choose New Name",
            filename,
            "PNG Files (*.png);;All Files (*)",
        )
        if new_name:
            if qimage.save(new_name, "png"):
                v.update_status(f"Saved: {new_name}")
                return True
            QMessageBox.critical(None, "Error", f"Failed to save to: {new_name}")
            return False

        v.update_status("Save cancelled.")
        return False

    def _make_wcs_filename(self, scene_x: float, scene_y: float) -> str:
        """
        Build an output filename that encodes the sky coordinates of *scene_x*,
        *scene_y* (in image-pixel space, before y-flip).

        The FITS y-axis runs bottom-to-top, Qt's top-to-bottom, so the flip is
        applied here before calling pixel_to_world.
        """
        v = self._v
        flipped_y = v.original_image.shape[0] - scene_y
        ra, dec    = v.wcs.pixel_to_world(scene_x, flipped_y)
        return os.path.join(v.dirpath, f"{v.tileID}_{ra:.6f}_{dec:.5f}.png")

    # ------------------------------------------------------------------
    # Public export methods
    # ------------------------------------------------------------------

    def save_full_image_with_overlays(self):
        """
        Render the entire scene at native 1:1 pixel resolution and save as PNG.

        Steps:
          1. Zoom out to fit the whole image in the scene.
          2. Allocate a QImage sized to the scene's bounding rect.
          3. Use QPainter + scene.render() to paint image + overlays.
          4. Save to a WCS-named file in viewer.dirpath.
          5. Restore the user's previous zoom/scroll position.
        """
        v = self._v
        if v.original_image is None:
            return

        if v.control_dock:
            v.control_dock.photoPushButton.setEnabled(False)

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Remember current view so we can restore it afterwards
            current_rect = v.sceneRect()

            # Zoom to show the full image
            v.fit_to_view()

            # Build output filename from scene centre WCS coordinates
            centre   = v.sceneRect().center()
            filename = self._make_wcs_filename(centre.x(), centre.y())

            # Allocate render buffer sized to the full scene
            size   = v.scene.itemsBoundingRect().size().toSize()
            buffer = QImage(size, QImage.Format_ARGB32_Premultiplied)
            buffer.fill(Qt.black)

            painter = QPainter(buffer)
            painter.setRenderHint(QPainter.Antialiasing)
            v.scene.render(painter)
            painter.end()

            self._save_image(buffer, filename)

            # Restore the previous viewport
            v.fitInView(current_rect)

        except Exception as e:
            v.update_status(f"Error saving full image: {e}")

        finally:
            if v.control_dock:
                v.control_dock.photoPushButton.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def save_visible_area_with_overlays(self):
        """
        Render only the currently visible viewport area and save as PNG.

        The output pixel dimensions match the viewport so the file is a
        1:1 screenshot of what the user sees.
        """
        v = self._v
        if v.original_image is None or not v.qimage:
            return

        if v.control_dock:
            v.control_dock.photoPushButton.setEnabled(False)

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Determine the visible scene region, clipped to the image boundary
            visible_rect = (
                v.mapToScene(v.viewport().rect())
                 .boundingRect()
                 .intersected(v.sceneRect())
            )

            # Build output filename from visible-area centre WCS coordinates
            centre   = visible_rect.center()
            filename = self._make_wcs_filename(centre.x(), centre.y())

            # Render at viewport resolution (1:1 with the screen)
            output_size = v.viewport().size()
            buffer      = QImage(output_size, QImage.Format_ARGB32_Premultiplied)
            buffer.fill(Qt.black)

            painter = QPainter(buffer)
            painter.setRenderHint(QPainter.Antialiasing)
            # Map the visible scene rect onto the full output buffer
            v.scene.render(painter, QRectF(buffer.rect()), visible_rect)
            painter.end()

            self._save_image(buffer, filename)

        except Exception as e:
            v.update_status(f"Error saving visible area: {e}")

        finally:
            if v.control_dock:
                v.control_dock.photoPushButton.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def save_area_from_selection(self, rect: QRectF):
        """
        Render the rubber-band *rect* (in scene coordinates) and save as PNG.

        Called back by handle_selection_rect after the user completes the
        rubber-band drag.  The output uses the crosshair centre WCS position
        set during the drag (viewer.crosshair_ra / crosshair_dec) so the
        filename matches the visual centre of the selection.
        """
        v = self._v
        if v.original_image is None or not v.qimage:
            return

        buffer = QImage(rect.size().toSize(), QImage.Format_ARGB32_Premultiplied)
        buffer.fill(Qt.black)

        painter = QPainter(buffer)
        painter.setRenderHint(QPainter.Antialiasing)
        v.scene.render(painter, QRectF(buffer.rect()), rect)
        painter.end()

        filename = os.path.join(
            v.dirpath,
            f"{v.tileID}_{v.crosshair_ra:.6f}_{v.crosshair_dec:.5f}.png"
        )
        self._save_image(buffer, filename)

    # ------------------------------------------------------------------
    # Selection-mode helpers (called from ImageViewer event handlers)
    # ------------------------------------------------------------------

    def start_selection(self):
        """
        Put the viewer into rubber-band selection mode.

        After the user draws the rectangle, handle_selection_rect is called
        back which calls save_area_from_selection.
        """
        v = self._v
        v.rectangle_selection     = True
        v.callback_on_selection   = self._handle_selection_rect
        v.start_point             = None
        v.current_point           = None
        v.selection_rect          = None
        QApplication.setOverrideCursor(Qt.CrossCursor)

    def _handle_selection_rect(self, rect: QRectF):
        """
        Internal callback registered in start_selection().
        Disables the photo button, saves the selection, then re-enables.
        """
        v = self._v

        # Guard against double-firing (e.g. accidental double-click)
        if v.control_dock and not v.control_dock.photoPushButton.isEnabled():
            return

        if v.control_dock:
            v.control_dock.photoPushButton.setEnabled(False)

        try:
            v.setCursor(Qt.CrossCursor)
            self.save_area_from_selection(rect)
        finally:
            v.rectangle_selection = False
            if v.control_dock:
                v.control_dock.photoPushButton.setEnabled(True)
            QApplication.restoreOverrideCursor()
            QApplication.setOverrideCursor(Qt.ArrowCursor)
