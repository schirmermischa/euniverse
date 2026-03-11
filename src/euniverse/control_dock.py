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
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize, QThread
from PyQt5 import uic
import os
import getpass
from datetime import datetime
import csv
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

from .generate_icons import *
from .table_dialog import TableDialog
from .catalog_plotter import PlotDialog
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
        self.is_viewer_displaying_preview = False
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

        # Configure sliders
        self.min_slider.setMinimum(0)
        self.min_slider.setMaximum(65535)
        self.min_slider.setValue(0)
        self.max_slider.setMinimum(0)
        self.max_slider.setMaximum(65535)
        self.max_slider.setValue(65535)

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
        self.sunglassesPushButton.setIcon(create_sunglasses_icon())
        self.sunglassesPushButton.setIconSize(QSize(32, 32))
        self.sunglassesPushButton.setStyleSheet(button_style)
        self.sunglassesPushButton.toggled.connect(self.on_sunglasses_toggled)

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
        # CsvUploader is defined in workers.py.
        self._upload_thread  = QThread()
        self._uploader       = CsvUploader(csv_filename)
        self._uploader.moveToThread(self._upload_thread)
        self._upload_thread.started.connect(self._uploader.run)
        self._uploader.done.connect(self.update_status)
        self._uploader.done.connect(self._upload_thread.quit)
        self._upload_thread.start()

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
            self.contrast_callback(self.min_slider.value(), self.max_slider.value())

    def slider_released(self):
        # Full LUT pass on the entire original_image
        if self.full_contrast_callback:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            QApplication.processEvents()
            self.full_contrast_callback(self.min_slider.value(), self.max_slider.value())
            QApplication.restoreOverrideCursor()
        # Discard the preview crop now that the full-image LUT pass is done.
        # Using the named method keeps us from poking private attributes directly.
        self.viewer.discard_preview_crop()
        # Restore MER overlay if it was visible before
        if self.MER_PushButton.isChecked():
            self.viewer.toggle_MER()
        # Restore scroll position
        self.viewer.restore_view_center(self.viewcenter)
                
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
        self.equatorialRALabel.setText(f"α = {ra:.6f}")
        self.equatorialDecLabel.setText(f"δ = {dec:.6f}")
        ra_str, dec_str = self.degrees_to_sexagesimal(ra, dec)
        self.equatorialRAHexLabel.setText(f"α = {ra_str}")
        self.equatorialDecHexLabel.setText(f"δ = {dec_str}")

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
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
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
        image = self.viewer.last_pixmap.toImage()
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

    def on_sunglasses_toggled(self, checked):
        pass

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

