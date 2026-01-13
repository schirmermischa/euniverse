from PyQt5.QtWidgets import (
    QApplication, QLabel, QListWidget, QSlider, QWidget, QPushButton, QHBoxLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize
from PyQt5 import uic
import os
import getpass
from datetime import datetime
import requests
import csv
from astropy.coordinates import SkyCoord
import astropy.units as u

from .generate_icons import *
from .table_dialog import TableDialog
from .catalog_plotter import PlotDialog

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

        # Load UI directly onto 'self'
#        ui_file = os.path.join(os.path.dirname(__file__), "control_dock.ui")
        from euniverse import get_resource
        ui_file = get_resource('control_dock.ui')
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

        # Button stylesheet
        button_style_photo = """
            QPushButton, QPushButton:checked, QPushButton:pressed {
                border-width: 1px !important;
                outline: none !important;
                padding: 1px !important;
                margin: 1px !important;
                background: none !important;
            }
            QPushButton:checked {
                background: #d0d0d0;
            }
            QPushButton:hover {
                background: #666666;
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

        self.photoFrame.setFrameStyle(QFrame.NoFrame)
        icon_label = QLabel(self.photoFrame)
        camera_icon = create_camera_icon()
        icon_label.setPixmap(camera_icon.pixmap(32, 32))

        self.photoLargePushButton.setIcon(create_largephoto_icon())
        self.photoLargePushButton.setIconSize(QSize(28, 28))
        self.photoLargePushButton.setStyleSheet(button_style_photo)
        self.photoLargePushButton.toggled.connect(self.on_photoLargePushButton_clicked)

        self.photoMediumPushButton.setIcon(create_mediumphoto_icon())
        self.photoMediumPushButton.setIconSize(QSize(28, 28))
        self.photoMediumPushButton.setStyleSheet(button_style_photo)
        self.photoMediumPushButton.toggled.connect(self.on_photoMediumPushButton_clicked)

        self.photoSmallPushButton.setIcon(create_smallphoto_icon())
        self.photoSmallPushButton.setIconSize(QSize(28, 28))
        self.photoSmallPushButton.setStyleSheet(button_style_photo)
        self.photoSmallPushButton.toggled.connect(self.on_photoSmallPushButton_clicked)
    
        # Override coord_list keyPressEvent
        def coord_list_key_press(event):
            if event.key() == Qt.Key_Delete:
                # Forwarding Delete key from coord_list to viewer
                self.viewer.keyPressEvent(event)
            else:
                QListWidget.keyPressEvent(self.coord_list, event)
        self.coord_list.keyPressEvent = coord_list_key_press

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
        if not self.viewer or not self.viewer.circles:
            # No targets to save
            return
        image_path = self.viewer.tileID
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        username = getpass.getuser()
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        csv_filename = f"{base_name}_{username}_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['RA', 'Dec', 'Classifier'])
            for _, ra, dec, classifier, _ in self.viewer.circles:
                writer.writerow([ra, dec, classifier])
        self.update_status(f"Saved targets to {csv_filename}")
        csv_filename_rel = f"{self.viewer.dirpath}/{base_name}_{username}_{timestamp}.csv"
        with open(csv_filename_rel, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['RA', 'Dec', 'Classifier'])
            for _, ra, dec, classifier, _ in self.viewer.circles:
                writer.writerow([ra, dec, classifier])
        self.update_status(f"Saved targets to {csv_filename}")
#        print(f"Saved targets to {csv_filename}")

    def on_submit_targets(self):
        if not self.viewer or not self.viewer.circles:
            # No targets to submit.
            return
#        image_path = self.viewer.default_image if self.viewer.default_image else "image"
        image_path = "image"
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        username = getpass.getuser()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{base_name}_{username}_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['RA', 'Dec', 'Classifier'])
            for _, ra, dec, classifier, _ in self.viewer.circles:
                writer.writerow([ra, dec, classifier])
        try:
            with open(csv_filename, 'rb') as f:
                files = {'file': (csv_filename, f, 'text/csv')}
                response = requests.post("https://www.euclid-ec.org/target_receiver", files=files)
            if response.status_code == 200:
                print(f"Successfully uploaded {csv_filename} to server.")
            else:
                print(f"Failed to upload {csv_filename}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error uploading {csv_filename}: {e}")

    def slider_pressed(self):
        # Hide the MER catalog if it is shown
        if self.MER_PushButton.isChecked():
            self.viewer.hide_MER()
        self.viewcenter = self.viewer.get_current_view_center()
            
    def slider_changed(self):
        if self.contrast_callback:
            self.contrast_callback(self.min_slider.value(), self.max_slider.value())

    def slider_released(self):
        self.viewer.is_preview_updated = False
        if self.full_contrast_callback:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            QApplication.processEvents()
            self.full_contrast_callback(self.min_slider.value(), self.max_slider.value())
            QApplication.restoreOverrideCursor()
            # Show the catalog again if it was shown before
            if self.MER_PushButton.isChecked():
                self.viewer.show_MER()
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
        self.cartesianxLabel.setText(f"x = {x:.1f}")
        self.cartesianyLabel.setText(f"y = {y:.1f}")
        self.equatorialRALabel.setText(f"α = {ra:.6f}")
        self.equatorialDecLabel.setText(f"δ = {dec:.6f}")
        ra_str, dec_str = self.degrees_to_sexagesimal(ra, dec)
        self.equatorialRAHexLabel.setText(f"α = {ra_str}")
        self.equatorialDecHexLabel.setText(f"δ = {dec_str}")

    def update_coord_list(self, entries):
        self.coord_list.clear()
        for _, ra, dec, classifier, _ in entries:
            item_text = f"{ra:.6f}, {dec:.6f}, {classifier}"
            self.coord_list.addItem(item_text)
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

    def on_coord_list_item_clicked(self, item):
        if self.viewer and self.viewer.wcs and self.viewer.original_image is not None:
            text = item.text()
            try:
                ra_str, dec_str, _ = text.split(", ", 2)
                ra = float(ra_str)
                dec = float(dec_str)
                sky_coord = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
                x, y = self.viewer.wcs.world_to_pixel(sky_coord)
                y = self.viewer.original_image.shape[0] - y
                self.viewer.reset_zoom()
                self.viewer.centerOn(QPointF(x, y))
                self.viewer.refresh_preview()
                for circle, circle_ra, circle_dec, _, normal_thickness in self.viewer.circles:
                    if abs(circle_ra - ra) < 1e-6 and abs(circle_dec - dec) < 1e-6:
                        if self.selected_circle and self.selected_circle != circle:
                            normal_pen = self.selected_circle.pen()
                            normal_pen.setWidthF(normal_thickness)
                            self.selected_circle.setPen(normal_pen)
                        pen = circle.pen()
                        pen.setWidthF(normal_thickness * 1.5)
                        circle.setPen(pen)
                        self.selected_circle = circle
                        self.viewer.scene.update()
                        break
            except (ValueError, IndexError):
                print(f"Failed to parse coordinates from: {text}")

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
            if self.viewer.catalog_manager.catalog is not None:
                if not self.plot_dialog:
                    self.plot_dialog = PlotDialog(self.viewer.catalog_manager, self)
                    # Connect the dialog's accepted signal to our new slot
                    self.plot_dialog.finished.connect(self.on_plot_dialog_closed)
                    self.plot_dialog.lasso_points_selected.connect(self.viewer.catalog_manager.handle_selected_objects)
                self.plot_dialog.show()
            else:
                print("Warning: No catalog loaded. Cannot open plotter.")
                # self.plotPushButton should be part of ControlDock
                if self.plotPushButton:
                    self.plotPushButton.setChecked(False) # Uncheck the button
        else:
            if self.plot_dialog:
                self.plot_dialog.hide()

    def on_plot_dialog_closed(self):
        """
        Slot to uncheck the plotPushButton when the PlotDialog is closed via its 'Accept' role button.
        """
        if self.plotPushButton:
            self.plotPushButton.setChecked(False)


    def on_MER_toggled(self, checked):
        if self.viewer:
            if checked:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                self.viewer.show_MER()
                if self.viewer.catalog_manager and self.viewer.catalog_manager.catalog:
                    if self.table_dialog is None:
                        self.table_dialog = TableDialog(self.viewer.catalog_manager.catalog, self.viewer, self)
                        self.table_dialog.show()
                else:
                    print("No catalog available to display")
                    self.MER_PushButton.setChecked(False)  # Optionally, uncheck the button
                QApplication.restoreOverrideCursor()
            else:
                self.viewer.hide_MER()
                if self.table_dialog:
                    self.table_dialog.close()
                    self.table_dialog = None

    def select_table_row(self, object_id):
        if self.table_dialog:
            self.table_dialog.select_row_by_object_id(object_id)

    def on_photoLargePushButton_clicked(self):
        if self.viewer:
            self.viewer.save_full_image_with_overlays()

    def on_photoMediumPushButton_clicked(self):
        if self.viewer:
            self.viewer.save_visible_area_with_overlays()

    def on_photoSmallPushButton_clicked(self):
        if self.viewer:
            self.viewer.start_selection()

