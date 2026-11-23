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
control_dock.py — Left-side control panel widget
=================================================
ControlDock is a QWidget loaded from control_dock.ui that provides all the
controls for the main viewer: image loading, contrast sliders, zoom buttons,
the navigator thumbnail, the magnifier, coordinate display, the MER overlay
toggle, the scatter-plot dialog trigger, annotation management, and the
photo / screenshot button.

Design notes
------------
- ControlDock holds a reference to ImageViewer (self.viewer) so it can call
  its public API.  It must NOT access private attributes (names starting with
  _) of ImageViewer directly; instead, it uses the named public methods such
  as viewer.discard_preview_crop().
- Annotation circles are stored in viewer.annotations as Annotation dataclass
  instances (annotations.py).  This file uses named field access (ann.ra,
  ann.dec, ann.classifier) instead of positional tuple unpacking.
- Network upload (on_submit_targets) runs in a CsvUploader background thread
  defined in workers.py, keeping the UI responsive during the POST request.
"""

from PyQt5.QtWidgets import (
    QApplication, QLabel, QListWidget, QSlider, QWidget, QPushButton,
    QHBoxLayout, QFrame, QVBoxLayout,
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor, QIcon
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize
from PyQt5 import uic
import os
import getpass
import threading
from datetime import datetime
import csv
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

from .generate_icons import *
from .table_dialog import TableDialog
from .catalog_plotter import PlotDialog
from .curves_dialog import CurvesDialog
from .workers import CsvUploader

# Define NA as a recognized unit so the parser doesn't complain
try:
    # Try the version 5.x/6.x way
    u.def_unit('NA', u.dimensionless_unscaled, register_to_subclass=True)
except (TypeError, ValueError):
    # Fallback for Astropy < 5.0 OR Astropy >= 7.0
    u.def_unit('NA', u.dimensionless_unscaled)


class CustomSlider(QSlider):
    def __init__(self, orientation):
        super().__init__(orientation)
        self.released_callback = None

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.released_callback:
            self.released_callback()

class ControlDock(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.viewcenter = None
        self.table_dialog = None
        self.plot_dialog = None
        self.curves_dialog = None
        self.is_viewer_displaying_preview = False
        self.wcs_available = True
        self.init_ui()

    def init_ui(self):
        # Prevent black background on the dock widget
        self.setAutoFillBackground(False)

        pale_yellow = QColor(255, 255, 204)


        # Load UI directly onto 'self'
        from euniverse import get_resource
        ui_file = get_resource('control_dock.ui')
        if not os.path.exists(ui_file):
            print(f"CRITICAL: UI file not found at {ui_file}")
            # Fallback: Create a basic label so the app doesn't just vanish
            self.layout = QVBoxLayout(self)
            self.layout.addWidget(QLabel("Error: control_dock.ui missing."))
            return
        uic.loadUi(ui_file, self)

        # Designer's scaledContents=True on these two labels stretches
        # whatever pixmap is assigned to fill the full 240x240 label
        # regardless of the pixmap's own aspect ratio — that silently
        # undoes the Qt.KeepAspectRatio scaling done in update_preview()
        # and update_magnifier() below, and is what causes visible
        # distortion for non-square (e.g. very wide/tall tile) images.
        # Alignment is left at Designer's default (top-left) rather than
        # centred, since preview_mouse_move()/handle_preview_drag() map
        # click positions assuming the pixmap is drawn at the label's
        # origin with no centering offset.
        self.preview_label.setScaledContents(False)
        self.magnifier_label.setScaledContents(False)

        # Configure sliders
        self.min_slider.setMinimum(0)
        self.min_slider.setMaximum(65535)
        self.min_slider.setValue(0)
        self.max_slider.setMinimum(0)
        self.max_slider.setMaximum(65535)
        self.max_slider.setValue(65535)

        # Log-scale display state for the sliders — see
        # _slider_pos_to_value/_value_to_slider_pos for the mapping.
        # A UI display preference, not image-derived state, so it
        # deliberately does NOT get reset on new image load (unlike e.g.
        # curve_sessions in ImageViewer).
        self.log_scale_enabled = False

        # Button stylesheet
        button_style = """
            QPushButton, QPushButton:checked, QPushButton:pressed {
                border-width: 0px !important;
                outline: none !important;
                padding: 0px !important;
                margin: 0px !important;
                background: none !important;
            }
            QPushButton:checked {
                background: #d0d0d0;
            }
            QPushButton:hover {
                background: #e0e0e0;
            }
        """

        button_style_photo = """
            QPushButton {
                 border: 1px solid #444444;
                 outline: none;
                 padding: 1px;
                 margin: 1px;
                 background-color: transparent; 
             }
             QPushButton:hover, QPushButton:checked {
                 background-color: #ffffe0;
                 border: 1px solid #444444;
             }
             QPushButton:pressed {
                 background-color: #ffffe0;
             }
         """

        # Assign icons and stylesheet
        self.MER_PushButton.setIcon(create_MER_icon())
        self.MER_PushButton.setIconSize(QSize(32, 32))
        self.MER_PushButton.setStyleSheet(button_style)
        self.MER_PushButton.toggled.connect(self.on_MER_toggled)

        self.plotPushButton.setIcon(create_scatter_plot_icon())
        self.plotPushButton.setIconSize(QSize(32, 32))
        self.plotPushButton.setStyleSheet(button_style)
        self.plotPushButton.toggled.connect(self.on_plot_toggled)

        self.photoPushButton.setIcon(create_camera_icon())
        self.photoPushButton.setIconSize(QSize(32,32))
        self.photoPushButton.setStyleSheet(button_style_photo)
        self.photoPushButton.toggled.connect(self.photoPushButton_requested)

        self.curvesPushButton.setIcon(create_curves_icon())
        self.curvesPushButton.setIconSize(QSize(32, 32))
        self.curvesPushButton.setStyleSheet(button_style)
        self.curvesPushButton.toggled.connect(self.on_curves_toggled)

        self.lookupPushButton.setIcon(create_lookup_icon())
        self.lookupPushButton.setIconSize(QSize(32, 32))
        self.lookupPushButton.setStyleSheet(button_style)
        self.lookupPushButton.toggled.connect(self.on_lookup_toggled)
        self.lookupPushButton.setToolTip("Look up known objects in the field of view: NED (green circles), "
                                         "SIMBAD (yellow squares) and galaxy-cluster catalogues (cyan diamonds)")

        self.logScalePushButton.setStyleSheet(button_style)
        self.logScalePushButton.toggled.connect(self.on_log_scale_toggled)

        # Helper to create the dual-state icon
        def make_toggle_icon(self, icon_func):
            """Creates a QIcon with two states: Normal (transparent) and On (pale yellow)."""
            icon = QIcon()
            # Passive/Off State
            icon.addPixmap(icon_func(Qt.transparent), QIcon.Normal, QIcon.Off)
            # Active/On State
            icon.addPixmap(icon_func(pale_yellow), QIcon.Normal, QIcon.On)
            return icon

        # Override coord_list keyPressEvent
        def coord_list_key_press(event):
            if event.key() == Qt.Key_Delete:
                # Forwarding Delete key from coord_list to viewer
                self.viewer.keyPressEvent(event)
            else:
                QListWidget.keyPressEvent(self.coord_list, event)
        self.coord_list.keyPressEvent = coord_list_key_press  # Install the handler

#        self.setLayout(main_layout)

        # Connect signals
        self.load_button.clicked.connect(self.on_load_image)
        self.zoom_in_button.clicked.connect(self.on_zoom_in)
        self.zoom_out_button.clicked.connect(self.on_zoom_out)
        self.reset_zoom_button.clicked.connect(self.on_reset_zoom)
        self.fit_button.clicked.connect(self.on_fit)
        self.submit_targets_button.clicked.connect(self.on_submit_targets)
        self.save_targets_button.clicked.connect(self.on_save_targets)
        self.submit_targets_button.setEnabled(False)
        self.save_targets_button.setEnabled(False)
        self.min_slider.valueChanged.connect(self.slider_changed)
        self.max_slider.valueChanged.connect(self.slider_changed)
        self.min_slider.sliderPressed.connect(self.slider_pressed)
        self.max_slider.sliderPressed.connect(self.slider_pressed)
        self.min_slider.sliderReleased.connect(self.slider_released)
        self.max_slider.sliderReleased.connect(self.slider_released)
        self.min_slider.released_callback = self.slider_released
        self.max_slider.released_callback = self.slider_released
        self.preview_label.setMouseTracking(True)
        self.preview_label.setCursor(Qt.CrossCursor)
        self.dragging = False
        self.preview_label.mousePressEvent = self.preview_mouse_press
        self.preview_label.mouseMoveEvent = self.preview_mouse_move
        self.preview_label.mouseReleaseEvent = self.preview_mouse_release
        self.load_callback = None
        self.contrast_callback = None
        self.full_contrast_callback = None
        self.slider_press_callback = None
        self.coord_list.itemClicked.connect(self.on_coord_list_item_clicked)
        self.selected_circle = None
        self.set_black_squares()

    def update_status(self, message, timeout=5000):
        if self.viewer.main_window:
            self.viewer.main_window.statusBar().showMessage(message, timeout)

    def set_black_squares(self):
        black_square = QPixmap(240, 240)
        black_square.fill(Qt.black)
        self.preview_label.setPixmap(black_square)
        self.magnifier_label.setPixmap(black_square)

    def set_load_callback(self, callback):
        self.load_callback = callback

    def set_slider_press_callback(self, callback):
        self.slider_press_callback = callback

    def set_contrast_callback(self, callback):
        self.contrast_callback = callback

    def set_full_contrast_callback(self, callback):
        self.full_contrast_callback = callback

    def on_load_image(self):
        if self.load_callback:
            self.load_callback()

    def on_zoom_in(self):
        if self.viewer:
            self.viewer.zoom_in()

    def on_zoom_out(self):
        if self.viewer:
            self.viewer.zoom_out()

    def on_reset_zoom(self):
        if self.viewer:
            self.viewer.reset_zoom()

    def on_fit(self):
        if self.viewer:
            self.viewer.fit_to_view()

    def on_save_targets(self):
        if not self.viewer or not self.viewer.annotations:
            return
        image_path = self.viewer.tileID
        base_name  = os.path.splitext(os.path.basename(image_path))[0]
        username   = getpass.getuser()
        timestamp  = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        csv_filename = f"{self.viewer.dirpath}/{base_name}_{username}_{timestamp}.csv"

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['RA', 'Dec', 'Classifier'])
            # ann is an Annotation dataclass (annotations.py) — use named fields
            for ann in self.viewer.annotations:
                writer.writerow([ann.ra, ann.dec, ann.classifier])
        self.update_status(f"Saved targets to {csv_filename}")

    def on_submit_targets(self):
        if not self.viewer or not self.viewer.annotations:
            return

        username     = getpass.getuser()
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{self.viewer.dirpath}/image_{username}_{timestamp}.csv"

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['RA', 'Dec', 'Classifier'])
            # ann is an Annotation dataclass (annotations.py) — use named fields
            for ann in self.viewer.annotations:
                writer.writerow([ann.ra, ann.dec, ann.classifier])

        # Upload in a background thread so the UI stays responsive.
        # CsvUploader is defined in workers.py.  A plain Python thread is used,
        # not a QThread: destroying a QThread from Python (e.g. when a second
        # upload replaces the first) can deadlock on the GIL.  The uploader is
        # a QObject of this (GUI) thread, so its `done` signal is queued here.
        self._uploads = [(u, t) for u, t in getattr(self, "_uploads", []) if t.is_alive()]
        uploader = CsvUploader(csv_filename)
        uploader.done.connect(self.update_status)
        thread = threading.Thread(target=uploader.run, name="euniverse-upload", daemon=True)
        self._uploads.append((uploader, thread))       # keeps the uploader alive until its thread has ended
        thread.start()

    def slider_pressed(self):
        # Save the current view centre so we can restore it after the full pass
        self.viewcenter = self.viewer.get_current_view_center()
        # Hide the MER catalog overlay (too many items slows redraws during drag)
        if self.MER_PushButton.isChecked():
            self.viewer.toggle_MER()
        # Capture the visible uint16 crop once, before any slider movement
        if self.slider_press_callback:
            self.slider_press_callback()

    def slider_changed(self):
        # Fast numexpr preview on the pre-captured crop
        if self.contrast_callback:
            min_val = self._slider_pos_to_value(self.min_slider.value(), is_min=True)
            max_val = self._slider_pos_to_value(self.max_slider.value(), is_min=False)
            self.contrast_callback(min_val, max_val)

    def slider_released(self):
        # Full LUT pass on the entire original_image
        if self.full_contrast_callback:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            QApplication.processEvents()
            min_val = self._slider_pos_to_value(self.min_slider.value(), is_min=True)
            max_val = self._slider_pos_to_value(self.max_slider.value(), is_min=False)
            self.full_contrast_callback(min_val, max_val)
            QApplication.restoreOverrideCursor()
        # Discard the preview crop now that the full-image LUT pass is done.
        # Using the named method keeps us from poking private attributes directly.
        self.viewer.discard_preview_crop()
        # Restore MER overlay if it was visible before
        if self.MER_PushButton.isChecked():
            self.viewer.toggle_MER()
        # Restore scroll position
        self.viewer.restore_view_center(self.viewcenter)

    # ------------------------------------------------------------------
    # Log-scale slider mapping
    # ------------------------------------------------------------------
    # QSlider is inherently linear/integer, so its internal range stays
    # 0..65535 in both modes — what changes is how that raw position maps
    # to the actual pixel value used for the contrast stretch.
    #
    # Only min_slider (the black point) gets the asinh treatment; max_slider
    # (the white point) always stays linear. The motivation for log-like
    # precision is specifically that shadow/black-point detail gets
    # compressed near zero — the white point doesn't have that problem, and
    # giving it the same treatment made it uncomfortably twitchy: any
    # single mapping that buys extra precision at the low end necessarily
    # spends it by making the high end coarser (a slider only has 65535
    # discrete positions to divide up), which is a bad trade for a control
    # that's usually parked somewhere in the upper half of the range.
    #
    # Uses an asinh (inverse hyperbolic sine) stretch for the min slider —
    # the same family used for astronomical display stretches (Lupton et
    # al. 2004) — rather than a true logarithm, for the same reason it's
    # used there: sinh/asinh are smooth and well-behaved through v=0 (no
    # singularity, unlike log), so position 0 maps to value 0 exactly, no
    # special-casing needed, while still approaching logarithmic behavior
    # away from zero.
    #
    # _ASINH_BETA is the softening parameter — also the approximate
    # crossover value below which the mapping behaves close to linear and
    # above which it starts compressing logarithmically (asinh(x) ~ x for
    # x << beta, ~ln(2x) for x >> beta). 3600 gives roughly a 5x precision
    # improvement over linear for values in the low end (e.g. ~7.6% of the
    # slider's travel for values 0-1000, vs. ~1.5% in linear mode).
    _SLIDER_RANGE = 65535
    _ASINH_BETA   = 3600.0
    _ASINH_SPAN   = np.arcsinh(_SLIDER_RANGE / _ASINH_BETA)

    def _slider_pos_to_value(self, pos: int, is_min: bool) -> int:
        """
        Raw QSlider position (0..65535) -> actual pixel value.

        is_min selects which slider this position came from: only the
        min (black-point) slider is affected by log_scale_enabled — the
        max (white-point) slider is always linear, position == value.
        """
        if not self.log_scale_enabled or not is_min:
            return pos
        p_norm = pos / self._SLIDER_RANGE
        return int(round(self._ASINH_BETA * np.sinh(p_norm * self._ASINH_SPAN)))

    def _value_to_slider_pos(self, value: int, is_min: bool) -> int:
        """Inverse of _slider_pos_to_value — actual pixel value -> raw QSlider position."""
        if not self.log_scale_enabled or not is_min:
            return int(value)
        value = max(0, min(self._SLIDER_RANGE, value))
        p_norm = np.arcsinh(value / self._ASINH_BETA) / self._ASINH_SPAN
        return int(round(p_norm * self._SLIDER_RANGE))

    def on_log_scale_toggled(self, checked):
        """
        Switch the black-point (min) slider between linear and logarithmic
        (asinh) display — the white-point (max) slider is always linear,
        so it's unaffected by this toggle.

        Repositions min_slider so the actual pixel value it represents
        stays the same across the switch — toggling this is a display-mode
        change only, not a contrast change, so the currently rendered
        image doesn't need to be touched at all.
        """
        old_min_value = self._slider_pos_to_value(self.min_slider.value(), is_min=True)

        self.log_scale_enabled = checked

        self.min_slider.blockSignals(True)
        self.min_slider.setValue(self._value_to_slider_pos(old_min_value, is_min=True))
        self.min_slider.blockSignals(False)

    def degrees_to_sexagesimal(self, ra, dec):
        ra_hours = ra / 15.0
        ra_h = int(ra_hours)
        ra_m = int((ra_hours - ra_h) * 60)
        ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60
        ra_str = f"{ra_h:02d}:{ra_m:02d}:{ra_s:05.2f}"
        dec_sign = '+' if dec >= 0 else '-'
        dec = abs(dec)
        dec_d = int(dec)
        dec_m = int((dec - dec_d) * 60)
        dec_s = ((dec - dec_d) * 60 - dec_m) * 60
        dec_str = f"{dec_sign}{dec_d:02d}:{dec_m:02d}:{dec_s:04.1f}"
        return ra_str, dec_str

    def update_cursor_display(self, x, y, ra, dec):
        # If coordinates are None (mouse outside image)
        if x is None or y is None:
            # If 'none' propagates, then the coordinate conversion below crashes
            return

        self.cartesianxLabel.setText(f"x = {x:.1f}")
        self.cartesianyLabel.setText(f"y = {y:.1f}")

        # ra/dec are None when no valid WCS is available for the loaded
        # image (see set_wcs_available). Pixel coordinates above are still
        # meaningful and get updated regardless; the equatorial labels are
        # left as-is (set_wcs_available already blanked/disabled them).
        if ra is None or dec is None:
            return

        self.equatorialRALabel.setText(f"α = {ra:.6f}")
        self.equatorialDecLabel.setText(f"δ = {dec:.6f}")
        ra_str, dec_str = self.degrees_to_sexagesimal(ra, dec)
        self.equatorialRAHexLabel.setText(f"α = {ra_str}")
        self.equatorialDecHexLabel.setText(f"δ = {dec_str}")

    def set_wcs_available(self, available: bool):
        """
        Enable or disable the sky-coordinate (RA/Dec) readout labels.

        Called by ImageViewer.on_image_loaded once a WCS has been
        validated (available=True) or found to be missing/invalid
        (available=False). Pixel (x, y) coordinates are unaffected —
        they don't depend on the WCS and keep updating either way.
        """
        self.wcs_available = available
        labels = (
            self.equatorialRALabel, self.equatorialDecLabel,
            self.equatorialRAHexLabel, self.equatorialDecHexLabel,
        )
        for label in labels:
            label.setEnabled(available)
        if not available:
            self.equatorialRALabel.setText("α = —")
            self.equatorialDecLabel.setText("δ = —")
            self.equatorialRAHexLabel.setText("α = —")
            self.equatorialDecHexLabel.setText("δ = —")

    def update_coord_list(self, entries):
        """
        Repopulate the coordinate list widget from a list of Annotation objects.
        *entries* is viewer.annotations — a list of Annotation dataclass instances
        (see annotations.py).  Named field access replaces the old tuple unpacking.
        """
        self.coord_list.clear()
        for ann in entries:
            self.coord_list.addItem(f"{ann.ra:.6f}, {ann.dec:.6f}, {ann.classifier}")
        self.coord_list.repaint()
        self.submit_targets_button.setEnabled(bool(entries))
        self.save_targets_button.setEnabled(bool(entries))

    def select_coord_list_item(self, ra, dec):
        for index in range(self.coord_list.count()):
            item = self.coord_list.item(index)
            text = item.text()
            try:
                ra_str, dec_str, _ = text.split(", ", 2)
                item_ra = float(ra_str)
                item_dec = float(dec_str)
                if abs(item_ra - ra) < 1e-6 and abs(item_dec - dec) < 1e-6:
                    self.coord_list.setCurrentItem(item)
                    break
            except (ValueError, IndexError):
                continue

    def get_selected_coord(self):
        selected_item = self.coord_list.currentItem()
        if selected_item:
            text = selected_item.text()
            try:
                ra_str, dec_str, _ = text.split(", ", 2)
                return float(ra_str), float(dec_str)
            except (ValueError, IndexError):
                print(f"Failed to parse coordinates from: {text}")
        return None, None

    def _highlight_selected_circle(self, ra, dec):
        """
        Visually highlight the annotation circle at (ra, dec) and reset the
        previously selected one.

        viewer.annotations is a list of Annotation dataclass instances
        (annotations.py).  Named field access replaces the old tuple unpacking.
        """
        if not self.viewer or not self.viewer.annotations:
            return

        # Reset the previously selected circle if it still exists
        if self.selected_circle:
            try:
                for ann in self.viewer.annotations:
                    if ann.item == self.selected_circle:
                        pen = ann.item.pen()
                        pen.setWidthF(ann.normal_thickness)
                        ann.item.setPen(pen)
                        break
            except Exception:
                pass

        # Find and highlight the new circle
        for ann in self.viewer.annotations:
            if abs(ann.ra - ra) < 1e-6 and abs(ann.dec - dec) < 1e-6:
                pen = ann.item.pen()
                pen.setWidthF(ann.normal_thickness * 1.5)
                ann.item.setPen(pen)
                self.selected_circle = ann.item
                self.viewer.scene.update()
                break

    def on_coord_list_item_clicked(self, item):
        """Safely centers the viewer on a selected catalog object."""
        if not (self.viewer and self.viewer.wcs and self.viewer.original_image is not None):
            self.update_status("Cannot center: No image or WCS loaded.")
            return

        text = item.text()
        try:
            # Safer parsing using split and strip
            parts = text.split(",")
            ra = float(parts[0].strip())
            dec = float(parts[1].strip())
            
            sky_coord = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
            x, y = self.viewer.wcs.world_to_pixel(sky_coord)
            # Extract the first element if they are arrays
            if isinstance(x, np.ndarray):
                x = float(x.item() if x.size == 1 else x[0])
                y = float(y.item() if y.size == 1 else y[0])

            # Use pathlib-style shape access
            img_height = self.viewer.original_image.shape[0]
            y = img_height - y
            
            self.viewer.reset_zoom()
            self.viewer.centerOn(QPointF(x, y))
            self.viewer.refresh_preview()
            
            # Highlight logic...
            self._highlight_selected_circle(ra, dec)

        except (ValueError, IndexError) as e:
            self.update_status(f"Error parsing coordinates: {e}")

    def update_preview(self, pixmap):
        if self.viewer and self.viewer.is_displaying_preview:
            return  # Do not update the preview if the viewer is showing its internal preview

        if not pixmap or pixmap.isNull():
            self.set_black_squares()
            return
        # Rescaling the full-resolution image is O(pixels) (70-350 ms for
        # 100-250 Mpx) and update_preview() runs on every scrollbar step, so
        # the thumbnail is built once per (image, label size) and only the
        # view rectangle is redrawn on top of a copy of it afterwards.
        key = (pixmap.cacheKey(), self.preview_label.width(), self.preview_label.height())
        cached = getattr(self, "_thumb_cache", None)
        if cached is None or cached[0] != key:
            thumb = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self._thumb_cache = cached = (key, thumb)
        scaled_pixmap = cached[1].copy()
        painter = QPainter(scaled_pixmap)
        painter.setPen(QPen(Qt.white, 2))
        if self.viewer:
            view_rect = self.viewer.viewport().rect()
            scene_rect = self.viewer.mapToScene(view_rect).boundingRect()
            img_rect = pixmap.rect()
            x_scale = scaled_pixmap.width() / img_rect.width()
            y_scale = scaled_pixmap.height() / img_rect.height()
            x = scene_rect.left() * x_scale
            y = scene_rect.top() * y_scale
            w = scene_rect.width() * x_scale
            h = scene_rect.height() * y_scale
            painter.drawRect(int(x), int(y), int(w), int(h))
        painter.end()
        self.preview_label.setPixmap(scaled_pixmap)

    def update_magnifier(self, scene_pos):
        if not self.viewer or not self.viewer.last_pixmap:
            self.set_black_squares()
            return
        img_x = int(scene_pos.x())
        img_y = int(scene_pos.y())
        # pixmap -> QImage conversion of the whole frame is costly (up to
        # ~10 ms at 250 Mpx) and this runs on every mouse move: do it once.
        pm = self.viewer.last_pixmap
        cached = getattr(self, "_magnifier_image", None)
        if cached is None or cached[0] != pm.cacheKey():
            self._magnifier_image = cached = (pm.cacheKey(), pm.toImage())
        image = cached[1]
        img_width = image.width()
        img_height = image.height()
        box_size = 50
        x = max(0, min(img_width - box_size, int(img_x - box_size / 2)))
        y = max(0, min(img_height - box_size, int(img_y - box_size / 2)))
        cropped = image.copy(x, y, box_size, box_size)
        magnified = QPixmap.fromImage(cropped).scaled(
            self.magnifier_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        painter = QPainter(magnified)
        painter.setPen(QPen(QColor(255, 0, 0), 1, Qt.SolidLine))
        crosshair_length = int(0.1 * min(magnified.width(), magnified.height()))
        center_x = magnified.width() // 2
        center_y = magnified.height() // 2
        painter.drawLine(center_x, center_y - crosshair_length // 2, center_x, center_y + crosshair_length // 2)
        painter.drawLine(center_x - crosshair_length // 2, center_y, center_x + crosshair_length // 2, center_y)
        painter.end()
        self.magnifier_label.setPixmap(magnified)

    def preview_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.handle_preview_drag(event.pos())

    def preview_mouse_move(self, event):
        if self.dragging:
            self.handle_preview_drag(event.pos())

        # Coordinate display — works independently of dragging
        if not self.viewer or not self.viewer.wcs or self.viewer.original_image is None:
            return
        pm = self.preview_label.pixmap()
        if not pm or pm.isNull():
            return

        # Map label pixel → full-resolution image pixel (same ratio used in handle_preview_drag)
        scaled_w = pm.width()
        scaled_h = pm.height()
        image_w = self.viewer.last_pixmap.width()
        image_h = self.viewer.last_pixmap.height()
        img_x = event.pos().x() * (image_w / scaled_w)
        img_y = event.pos().y() * (image_h / scaled_h)

        # Guard: only update if the mapped position is inside the image
        if not (0 <= img_x < image_w and 0 <= img_y < image_h):
            return

        # Apply the same FITS y-flip as in the main viewer's mouseMoveEvent
        flipped_y = self.viewer.original_image.shape[0] - img_y

        try:
            ra, dec = self.viewer.wcs.pixel_to_world(img_x, flipped_y)
            self.update_cursor_display(img_x, flipped_y, ra, dec)
        except Exception:
            pass

    def preview_mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def handle_preview_drag(self, pos):
        if not self.viewer or not self.viewer.last_pixmap:
            return
        scaled_width = self.preview_label.pixmap().width() if self.preview_label.pixmap() else 1
        scaled_height = self.preview_label.pixmap().height() if self.preview_label.pixmap() else 1
        image_width = self.viewer.last_pixmap.width()
        image_height = self.viewer.last_pixmap.height()
        x_ratio = image_width / scaled_width
        y_ratio = image_height / scaled_height
        img_x = pos.x() * x_ratio
        img_y = pos.y() * y_ratio
        center_point = QPointF(img_x, img_y)
        self.viewer.centerOn(center_point)
        self.viewer.refresh_preview()

    def on_plot_toggled(self, checked):
        if checked:
            if self.viewer and self.viewer.catalog_manager:
                # Close any existing dialog before creating a new one
                if self.plot_dialog is not None:
                    self.plot_dialog.close()
                    self.plot_dialog = None
                self.plot_dialog = PlotDialog(
                    self.viewer.catalog_manager,
                    self.viewer,
                    parent=self
                )
                self.plot_dialog.finished.connect(self.on_plot_dialog_closed)
                self.plot_dialog.show()
            else:
                print("Error: Could not locate ImageViewer or CatalogManager.")
                self.plotPushButton.blockSignals(True)
                self.plotPushButton.setChecked(False)
                self.plotPushButton.blockSignals(False)
        else:
            if self.plot_dialog is not None:
                self.plot_dialog.close()
                self.plot_dialog = None


    def on_plot_dialog_closed(self):
        """
        Slot to uncheck the plotPushButton when the PlotDialog is closed via its 'Accept' role button.
        """
        if self.plotPushButton:
            self.plotPushButton.setChecked(False)

    def on_curves_toggled(self, checked):
        if checked:
            if self.viewer and self.viewer.original_image is not None:
                if self.curves_dialog is not None:
                    self.curves_dialog.close()
                    self.curves_dialog = None
                self.curves_dialog = CurvesDialog(self.viewer, parent=self)
                self.curves_dialog.finished.connect(self.on_curves_dialog_closed)
                self.curves_dialog.show()
            else:
                print("Error: No image loaded — nothing for Curves to adjust.")
                self.curvesPushButton.blockSignals(True)
                self.curvesPushButton.setChecked(False)
                self.curvesPushButton.blockSignals(False)
        else:
            if self.curves_dialog is not None:
                self.curves_dialog.close()
                self.curves_dialog = None

    def on_curves_dialog_closed(self):
        """Uncheck curvesPushButton when the CurvesDialog is closed via its own Close button."""
        if self.curvesPushButton:
            self.curvesPushButton.setChecked(False)

    def on_lookup_toggled(self, checked):
        """
        Look-up button: ON queries NED and SIMBAD for the field of view and
        overlays green circles (NED), yellow squares (SIMBAD) and cyan diamonds
        (cluster catalogues); OFF removes them and discards the catalogs.  The queries run in the background
        (see simbad_lookup.SimbadManager).
        """
        if not self.viewer:
            return
        if checked:
            if not self.viewer.simbad.start_lookup():
                self.set_lookup_checked(False)
        else:
            self.viewer.simbad.clear()
            self.viewer.update_status("Look-up overlay cleared.", 3000)

    def set_lookup_checked(self, checked: bool):
        """Set the look-up button state without triggering on_lookup_toggled."""
        if self.lookupPushButton:
            self.lookupPushButton.blockSignals(True)
            self.lookupPushButton.setChecked(checked)
            self.lookupPushButton.blockSignals(False)

    def close_curves_dialog(self):
        """
        Called by ImageViewer.reset() before a new image replaces the
        current one — a CurvesDialog holds a histogram and dtype captured
        from the outgoing image, so it can't be left open across a load.
        """
        if self.curves_dialog is not None:
            self.curves_dialog.close()
            self.curves_dialog = None
        if self.curvesPushButton:
            self.curvesPushButton.blockSignals(True)
            self.curvesPushButton.setChecked(False)
            self.curvesPushButton.blockSignals(False)


#    def on_load_image(self):
#        """Trigger the file dialog in the image viewer."""
#        if self.viewer:
#            print("shit")
#            self.viewer.open_file_dialog()


    def on_MER_toggled(self, checked):
        """
        Handles the display of the MER catalog and associated table dialog.
        Refactored for robustness with explicit state cleanup and visual feedback.
        """
        if not self.viewer:
            return

        if checked:
            # 1. Provide immediate visual feedback for heavy processing
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            QApplication.processEvents() # Force cursor update
            
            try:
                # 2. Trigger the catalog overlay
                self.viewer.toggle_MER()
                
                # 3. Check if a valid, non-empty catalog actually exists
                cat_manager = self.viewer.catalog_manager
                if cat_manager and cat_manager.catalog is not None and len(cat_manager.catalog) > 0:
                    # Only create the dialog if it doesn't already exist
                    if self.table_dialog is None:
                        # Fetch the path from the catalog manager
                        catalog_name = getattr(cat_manager, 'catalog_name', None)
                        self.table_dialog = TableDialog(cat_manager.catalog, self.viewer, catalog_name, self)
                        self.table_dialog.show()
                else:
                    # 4. Handle missing/empty data gracefully
                    self.update_status("No MER catalog data available for this tile.")
                    print("Warning: Attempted to show MER, but catalog is missing or empty.")
                    
                    # Uncheck the button since we can't fulfill the request
                    self.MER_PushButton.blockSignals(True)
                    self.MER_PushButton.setChecked(False)
                    self.MER_PushButton.blockSignals(False)
            
            except Exception as e:
                self.update_status(f"Error displaying MER: {e}")
                print(f"Critical error in on_MER_toggled: {e}")
            
            finally:
                # Always restore the cursor, even if an error occurs
                QApplication.restoreOverrideCursor()
        
        else:
            # 5. Robust cleanup when toggled off
            self.viewer.toggle_MER()
            if self.table_dialog:
                self.table_dialog.close()
                self.table_dialog = None
            self.update_status("MER catalog hidden.")


    def reset_ui_state(self):
        """Clears all overlays and dialogs when a new image is loaded."""
        self.set_black_squares()
        self.coord_list.clear()
        
        if self.table_dialog:
            self.table_dialog.close()
            self.table_dialog = None
            
        if self.plot_dialog:
            self.plot_dialog.close()
            self.plot_dialog = None

        self.close_curves_dialog()

        # Ensure buttons reflect the new empty state
        self.MER_PushButton.setChecked(False)
        self.plotPushButton.setChecked(False)
        self.submit_targets_button.setEnabled(False)
        self.save_targets_button.setEnabled(False)

    def select_table_row(self, object_id):
        if self.table_dialog:
            self.table_dialog.select_row_by_object_id(object_id)

    def photoPushButton_requested(self):
        if self.viewer:
            self.viewer.start_selection()
        self.photoPushButton.setChecked(False)

