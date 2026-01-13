from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, QApplication, QMessageBox, QMenu, QGraphicsEllipseItem, QWidget, QListWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QDialog, QStatusBar, QGraphicsRectItem, QGraphicsLineItem, QGraphicsItemGroup
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QCursor
from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint, QThread, pyqtSignal, QObject, QSize
import tifffile
import json
import os
import re
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from astropy.coordinates import SkyCoord
import astropy.units as u

from .catalog_manager import CatalogManager
from .wcs_utils import WCSConverter

# Define NA as a recognized unit so the parser doesn't complain
try:
    # Try the version 5.x/6.x way
    u.def_unit('NA', u.dimensionless_unscaled, register_to_subclass=True)
except (TypeError, ValueError):
    # Fallback for Astropy < 5.0 OR Astropy >= 7.0
    u.def_unit('NA', u.dimensionless_unscaled)


class ImageViewer(QGraphicsView):
    # Add signal for load completion
    image_loaded = pyqtSignal(np.ndarray, dict, str)  # Emits image, metadata, path

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.image_item = None
        self.circles = []  # List of (QGraphicsEllipseItem, RA, Dec, classifier, normal_thickness)
        self.wcs = None
        self.filepath = None
        self.dirpath = None
        self.qimage = None
        self.preview_image = None
        self.control_dock = None
        self.annotations = []
        self.setMouseTracking(True)
        self.original_image = None
        self.metadata = None
        self.scale_factor = 1.0
        self.last_pixmap = None
        self.last_preview_pixmap = None
        self.press_pos = QPoint()
        self.release_pos = QPoint()
        self.dragging = False
        self.catalog_manager = None
        self.is_displaying_preview = False
        self.is_preview_updated = False
        # Measurement attributes
        self.is_measuring = False
        self.measure_start = None
        self.measure_end = None
        self.start_ra_dec = None
        self.end_ra_dec = None
        self.angular_offset = None
        self.offset_unit = None
        self.horizontal_offset = None
        self.horizontal_unit = None
        self.vertical_offset = None
        self.vertical_unit = None
        self.crosshair = None
        self.crosshair_ra = 0
        self.crosshair_dec = 0
        self.tileID = ""
        self.title = ""
        # Ensure the view can receive key events
        self.setFocusPolicy(Qt.StrongFocus)
        # Connect scrollbar signals
        self.horizontalScrollBar().valueChanged.connect(self.refresh_preview)
        self.verticalScrollBar().valueChanged.connect(self.refresh_preview)
        # Add status bar
        self.status_bar = QStatusBar()
        # Ensure status bar is added to the main window or layout later if needed
        self.load_thread = None
        self.load_worker = None
        self.contrast_luts16 = {}  # Cache for LUTs
        self.contrast_luts8 = {}  # Cache for LUTs
        self.rectangle_selection = False
        self.rubber_band = None
        self.start_point = None
        self.end_point = None
        self.selection_rect = None
        self.callback_on_selection = None
        QApplication.setOverrideCursor(Qt.ArrowCursor)
        self.setStyleSheet(self._get_scrollbar_stylesheet())

    def _get_scrollbar_stylesheet(self):
        # The color for the movable handle/slider
        handle_color = "#87CEEB"  # Blue

        return f"""
            /* 1. Define the overall Scrollbar dimensions and background (Trough) */
            QScrollBar:vertical {{
                background: #E0E0E0; /* Set a recognizable light gray background for the trough */
            }}
            QScrollBar:horizontal {{
                background: #E0E0E0; /* Set a recognizable light gray background for the trough */
            }}

            /* 2. Target the movable Handle (the Slider) and apply the blue color */
            QScrollBar::handle:vertical {{
                background: {handle_color};
                min-height: 20px;
                border-radius: 6px; /* Half of the width to make it rounded */
            }}
            QScrollBar::handle:horizontal {{
                background: {handle_color};
                min-width: 20px;
                border-radius: 6px; /* Half of the height to make it rounded */
            }}

            /* 3. Remove default styling for arrows (optional but recommended) */
            QScrollBar::add-line, QScrollBar::sub-line {{
                border: none;
                background: none;
            }}
        """

    def extract_tileID(self, filepath):
        """
        Extracts the substring starting with 'TILE' and ending after the digit sequence,
        stopping when a non-digit character is encountered.
        
        Returns:
        - the raw segment (e.g., 'TILE101794875')
        - the same segment with a space after 'TILE' (e.g., 'TILE 101794875')
        """
        filename = os.path.basename(filepath)
        match = re.search(r'(TILE\d+)\D', filename)
        if match:
            raw = match.group(1)
            spaced = raw.replace('TILE', 'TILE ')
            return raw, spaced
        else:
            return None, None
        
    ###############################################
    # File loading functions
    ###############################################

    # Worker class for loading TIFF in a separate thread
    class LoadWorker(QObject):
        finished = pyqtSignal(np.ndarray, dict, str)  # Signal to emit image, metadata, path
        error = pyqtSignal(str)  # Signal for errors

        def __init__(self, path):
            super().__init__()
            self.path = path

        def run(self):
            """
            Loads the TIFF image and extracts metadata.
            Emits the 'finished' signal with the image data, metadata, and path on success.
            Emits the 'error' signal if an exception occurs.
            """
            try:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                with tifffile.TiffFile(self.path) as tif:
                    image = tif.pages[0].asarray()
                    if 'ImageDescription' not in tif.pages[0].tags:
                        raise ValueError("No ImageDescription tag found in TIFF metadata")
                    desc = tif.pages[0].tags['ImageDescription'].value
                    try:
                        metadata = json.loads(desc)
                    except json.JSONDecodeError:
                        raise ValueError("Invalid JSON in ImageDescription tag")
                    self.finished.emit(image, metadata, self.path)
                                    
            except ValueError as ve:
                self.error.emit(str(ve))
                QApplication.restoreOverrideCursor()
            except Exception as e:
                self.error.emit(f"An unexpected error occurred: {str(e)}")
                QApplication.restoreOverrideCursor()


    def reset(self):
        # Clear the catalogs
        self.control_dock.coord_list.clear()
        self.hide_MER()
        self.catalog_manager = None
        self.control_dock.MER_PushButton.setChecked(False)

        # Clear the windows
        self.control_dock.set_black_squares()
            

    def load_image(self, path):
        # Show status bar message
#        self.status_bar.showMessage("Loading TIFF ... this will take a while")

        # Create thread and worker
        self.load_thread = QThread()
        self.load_worker = self.LoadWorker(path)
        self.load_worker.moveToThread(self.load_thread)

        # Connects signals
        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.finished.connect(self.load_thread.quit)
        self.load_worker.finished.connect(self.load_worker.deleteLater)
        self.load_worker.finished.connect(self.on_image_loaded)
        self.load_worker.error.connect(self.on_load_error)
        self.load_thread.start()

    def on_image_loaded(self, image, metadata, path):
        """
        Handles the image data after loading is complete.  This runs in the main thread.
        """

        try:
            self.reset()
            
            self.update_status("Processing TIFF ...")
            self.original_image = image
            self.metadata = metadata
            self.wcs = WCSConverter(self.metadata)
            if self.wcs is None:
                raise ValueError("WCSConverter returned None")

            h, w = image.shape[:2]
            scale = min(1000 / max(h, w), 1.0)
            new_h, new_w = int(h * scale), int(w * scale)

            if len(image.shape) == 3:
                self.update_status("Making preview image ...")
                if image.dtype == np.uint16:
                    image_uint8 = (image // 256).astype(np.uint8)
                else:
                    image_uint8 = image.astype(np.uint8)
                pil_img = Image.fromarray(image_uint8).resize((new_w, new_h), Image.LANCZOS)
                self.preview_image = np.array(pil_img)
            else:
                from scipy.ndimage import zoom
                self.preview_image = zoom(image, scale, order=1)

            self.update_status("Adjusting contrast ...")
            self.control_dock.max_slider.setValue(65535)
            self.apply_contrast(0, 65535, full_image=True)
            self.centerOn(QPointF(w / 2, h / 2))
            self.update_status("Getting preview ...")
            self.get_visible_qimage_pixmap()
            self.update_status("Getting preview done.")

            self.tileID, self.title = self.extract_tileID(path)
            if self.title:
                self.set_main_window_title(f"Euniverse Explorer – {self.title}")
            if self.tileID:
                self.update_status("Loading MER catalog ...")
                self.catalog_manager = CatalogManager(self.tileID, self.wcs, os.path.dirname(path),
                                                    image_viewer=self)  # Pass self
            else:
                print("No TILE number found in image filename")
                self.catalog_manager = None

            self.default_image = path
            self.image_loaded.emit(image, metadata, path)  # Emit signal
            self.update_status(f"TIFF loaded, {self.catalog_manager.numsources} MER sources found.")

        except ValueError as ve:
            self.on_load_error(str(ve))
        except Exception as e:
            self.on_load_error(f"An unexpected error occurred in on_image_loaded: {e}")

        QApplication.restoreOverrideCursor()


    def on_load_error(self, error_msg):
        self.update_status(f"Error loading TIFF: {error_msg}")
        print(f"Error loading TIFF: {error_msg}")
        self.load_thread.quit()  # Ensure thread is stopped on error
        self.load_thread.wait()
        self.load_worker.deleteLater()
        self.load_worker = None

    # Update open_file_dialog to use the threaded load_image
    def open_file_dialog(self):
        self.filepath, _ = QFileDialog.getOpenFileName(self, "Select MER TILE TIFF", "", "TIFF files (*TILE*.tif *TILE*.tiff)")
        if self.filepath:
            self.dirpath = os.path.dirname(self.filepath)
            self.load_image(self.filepath)

    def update_status(self, message, timeout=5000):
        if self.main_window:
            self.main_window.statusBar().showMessage(message, timeout)

    def set_main_window_title(self, title):
        if self.main_window:
            self.main_window.setWindowTitle(title)

    ###############################################
    # MER catalog display
    ###############################################
    def show_MER(self):
        if self.catalog_manager is None:
            print("No MER catalog available")
            return
        if self.original_image is None:
            print("No image loaded for MER catalog overlay")
            return
        # Clear previous display
        self.hide_MER()
        self.catalog_manager.get_MER(self.original_image.shape[0])
        for ellipse in self.catalog_manager.MER_items:
            self.scene.addItem(ellipse)
        self.scene.update()

    def hide_MER(self):
        if self.catalog_manager and self.catalog_manager.MER_items:
            for ellipse in self.catalog_manager.MER_items:
                if ellipse.scene() == self.scene:
                    self.scene.removeItem(ellipse)
                #        self.catalog_manager.MER_items = []
            self.scene.update()

    def show_selected_MER(self):
        if self.catalog_manager is None:
            print("No MER catalog available")
            return
        if self.original_image is None:
            print("No image loaded for MER catalog overlay")
            return
        # Clear previous display
        self.hide_selected_MER()
        if self.catalog_manager.selected_MER_items:
            for ellipse in self.catalog_manager.selected_MER_items:
                self.scene.addItem(ellipse)
            self.scene.update()

    def hide_selected_MER(self):
        if self.catalog_manager and self.catalog_manager.selected_MER_items:
            for ellipse in self.catalog_manager.selected_MER_items:
                if ellipse.scene() == self.scene:
                    self.scene.removeItem(ellipse)
                #        self.catalog_manager.MER_items = []
            self.scene.update()

    ###############################################
    # Process image
    ###############################################
    def get_visible_qimage_pixmap(self):
        """
        Determines the currently visible area in a QGraphicsView, extracts that area
        from the loaded 8-bit RGB QImage displayed in the scene, and stores it as an
        internal numpy array in the ImageViewer class.
        We use this new image to quickly display interactive contrast changes

        Args:
        view: The QGraphicsView instance displaying the 8-bit RGB QImage.
        """
        if not self.scene:
            return

        self.is_preview_updated = False
        
        # Get the visible rectangle in scene coordinates
        visible_rect_scene = self.mapToScene(self.viewport().rect()).boundingRect()

        # Find the QGraphicsPixmapItem containing the image
        image_item = None
        for item in self.scene.items():
            if isinstance(item, QGraphicsPixmapItem):
                image_item = item
                break
        if not image_item:
            return

        # Get the bounding rect of the image item in scene coordinates
        image_rect_scene = image_item.boundingRect()

        # Calculate the intersection of the visible area and the image area
        intersection_rect_scene = visible_rect_scene.intersected(image_rect_scene)

        if intersection_rect_scene.isEmpty():
            return

        # Map the intersection rectangle from scene coordinates to image coordinates
        top_left_scene = intersection_rect_scene.topLeft()
        bottom_right_scene = intersection_rect_scene.bottomRight()
        center_scene = intersection_rect_scene.center()

        top_left_image = image_item.mapFromScene(top_left_scene).toPoint()
        bottom_right_image = image_item.mapFromScene(bottom_right_scene).toPoint()
        # center_image = image_item.mapFromScene(bottom_right_scene).toPoint()
        center_image = image_item.mapFromScene(center_scene).toPoint()

        # Ensure the extracted rectangle is within the bounds of the original image
        x = max(0, top_left_image.x())
        y = max(0, top_left_image.y())

        width = min(bottom_right_image.x(), self.qimage.width() - 1) - x + 1
        height = min(bottom_right_image.y(), self.qimage.height() - 1) - y + 1

        if width <= 0 or height <= 0:
            return

        # Extract the visible portion as a new QImage
        qimg = self.qimage.copy(x, y, width, height)

        # Convert the QImage to a NumPy array (assuming 8-bit RGB - likely Format_RGB32 or Format_RGB888)
        format = qimg.format()
        if format == QImage.Format_RGB32 or format == QImage.Format_ARGB32:
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)  # BGRA
            self.preview_image = arr[:, :, 0:3][:, :, ::-1]  # Convert BGRA to RGB
            self.is_preview_updated = True
        elif format == QImage.Format_RGB888:
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            self.preview_image = np.array(ptr).reshape(qimg.height(), qimg.width(), 3) # RGB
            self.is_preview_updated = True
        else:
            print(f"Warning: Unsupported QImage format for RGB NumPy conversion: {format}")
            self.preview_image = None

        
    def create_contrast_lut(self, min_val, max_val, input_dtype, output_dtype=np.uint8):
        lut = np.arange(np.iinfo(input_dtype).max + 1, dtype=np.float32)
        diff = max_val - min_val
        if diff == 0:
            lut.fill(255)
        else:
            lut = np.clip((lut - min_val) / diff, 0, 1)
        lut = (255 * lut).astype(output_dtype)
        return lut
    
    def apply_contrast_lut(self, image, lut):
        """Applies contrast adjustment using a lookup table."""
        return lut[image]


    def apply_contrast(self, min_val, max_val, full_image=False):
        if self.original_image is None:
            return

        if full_image:
            image = self.original_image
            input_dtype = image.dtype
            output_dtype = np.uint8
            lut_key = (min_val, max_val, input_dtype.name, output_dtype.__name__)
            if lut_key not in self.contrast_luts16:
                self.contrast_luts16[lut_key] = self.create_contrast_lut(min_val, max_val, input_dtype, output_dtype)
            lut = self.contrast_luts16[lut_key]
            
        else:
            # in 8-bit mode for quick preview
            if not self.is_preview_updated:
                self.get_visible_qimage_pixmap()
            min_val = min_val / 255
            max_val = max_val / 255
            image = self.preview_image
            input_dtype = image.dtype
            output_dtype = np.uint8
            lut_key = (min_val, max_val, input_dtype.name, output_dtype.__name__)
            if lut_key not in self.contrast_luts8:
                self.contrast_luts8[lut_key] = self.create_contrast_lut(min_val, max_val, input_dtype, output_dtype)
            lut = self.contrast_luts8[lut_key]
            
        image = self.apply_contrast_lut(image, lut)
        h, w = image.shape[:2]
        c = 3 if len(image.shape) == 3 else 1
        stride = c * w
        
        self.qimage = QImage(image.data, w, h, stride,
                             QImage.Format_RGB888 if c == 3 else QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(self.qimage)

        if full_image:
            self.last_pixmap = pixmap
            self.is_displaying_preview = False
        else:
            self.last_preview_pixmap = pixmap
            self.last_pixmap = pixmap
            self.is_displaying_preview = True
            
        if self.image_item:
            self.image_item.setPixmap(pixmap)
        else:
            self.image_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.image_item)

        self.setSceneRect(QRectF(0, 0, w, h))

        # Only update the control dock's preview if we are NOT displaying the preview in the main view
        if self.control_dock and not self.is_displaying_preview:
            self.control_dock.update_preview(self.last_pixmap)


    def set_control_dock(self, dock):
        """
        Connects the control dock to this image viewer and wires up all necessary callbacks
        including contrast sliders and load image functionality.
        """
        self.control_dock = dock
        
        # Callback for loading an image
        self.control_dock.set_load_callback(self.open_file_dialog)
        
        # Live contrast adjustment (called while slider is moving)
        def preview_contrast(min_val, max_val):
            if self.original_image is not None:
                self.apply_contrast(min_val, max_val, full_image=False)
                
        # Full contrast computation (called after slider is released)
        def final_contrast(min_val, max_val):
            if self.original_image is not None:
                self.apply_contrast(min_val, max_val, full_image=True)

        self.control_dock.set_contrast_callback(preview_contrast)
        self.control_dock.set_full_contrast_callback(final_contrast)


    # Display a cross-hair in selection rectangles
    def create_crosshair(self, parent_item):
        # Crosshair dimensions
        length = 5.0
        
        # Pen style for the crosshair
        crosshair_pen = QPen(QColor("red"), 1, Qt.SolidLine)
        
        # 1. Vertical line: from (0, -length) to (0, length)
        v_line = QGraphicsLineItem(0, -length, 0, length)
        v_line.setPen(crosshair_pen)
        
        # 2. Horizontal line: from (-length, 0) to (length, 0)
        h_line = QGraphicsLineItem(-length, 0, length, 0)
        h_line.setPen(crosshair_pen)
        
        # 3. Create a QGraphicsItemGroup to hold both lines
        # This makes it easy to move the whole crosshair together
        crosshair_group = QGraphicsItemGroup(parent_item)
        crosshair_group.addToGroup(v_line)
        crosshair_group.addToGroup(h_line)
        
        # The crosshair is created, but it won't be positioned until the 
        # rectangle size is defined (usually in mouseMoveEvent).
        # We set its initial position (relative to the parent) to the start point
        # In a typical setup, the parent_item (rubber band) is positioned 
        # at the start point.
        crosshair_group.setPos(0, 0) 
        
        return crosshair_group

    ############################################################
    # Event handling
    ############################################################
    def mousePressEvent(self, event):
        # If screenshotting a sub-area
        if self.rectangle_selection:
            if self.start_point is None and self.callback_on_selection:
                self.start_point = self.mapToScene(event.pos())
                self.rubber_band = QGraphicsRectItem(QRectF(self.start_point, QPointF()))
                pen = QPen(QColor("white"), 1, Qt.DashLine)
                self.rubber_band.setPen(pen)
                self.scene.addItem(self.rubber_band)
                # Create crosshair; don't add it yet (done in mousemove event)
                self.crosshair = self.create_crosshair(self.rubber_band)
            else:
                super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        # Left button: 
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.press_pos = event.pos()  # Store press position for click detection
        elif event.button() == Qt.MidButton:
            self.is_measuring = True
            self.measure_start = scene_pos
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().update()
        elif event.button() == Qt.RightButton:
            # Check if right-click is on an existing circle
            self.dragging = False
            for circle, ra, dec, classifier, normal_thickness in self.circles[:]:
                circle_rect = circle.rect()
                circle_center = circle_rect.center()
                distance = ((scene_pos.x() - circle_center.x())**2 + (scene_pos.y() - circle_center.y())**2)**0.5
                if distance <= 10:
                    self.scene.removeItem(circle)
                    self.circles.remove((circle, ra, dec, classifier, normal_thickness))
                    if self.control_dock:
                        self.control_dock.update_coord_list(self.circles)
                        if self.control_dock.selected_circle == circle:
                            self.control_dock.selected_circle = None
                    self.viewport().update()
                    return
            # Show context menu
            menu = QMenu(self)
            gl_arc = menu.addAction("GL: lens")
            gl_arc = menu.addAction("GL: arc")
            gl_multi = menu.addAction("GL: multiple image")
            gl_lens = menu.addAction("GL: strong lens")
            gl_einstein = menu.addAction("GL: Einstein ring")
            gl_einstein = menu.addAction("GL: DSLP")
            menu.addSeparator()
            agn_outflow = menu.addAction("AGN: outflow")
            menu.addSeparator()
            gx_ring = menu.addAction("Gx: Ring")
            gx_polar = menu.addAction("Gx: Polar ring")
            gx_stream = menu.addAction("Gx: Stream")
            gx_merger = menu.addAction("Gx: Merger")
            gx_irregular = menu.addAction("Gx: Irregular")
            gx_weird = menu.addAction("Gx: Dwarf")
            gx_weird = menu.addAction("Gx: weird")
            gl_arc.setData({"gl": True})
            gl_multi.setData({"gl": True})
            gl_lens.setData({"gl": True})
            gl_einstein.setData({"gl": True})
            agn_outflow.setData({"agn": True})
            gx_ring.setData({"gx": True})
            gx_polar.setData({"gx": True})
            gx_stream.setData({"gx": True})
            gx_merger.setData({"gx": True})
            gx_irregular.setData({"gx": True})
            gx_weird.setData({"gx": True})
            action = menu.exec_(self.mapToGlobal(event.pos()))
            if action and self.wcs and self.original_image is not None:
                flipped_y = self.original_image.shape[0] - scene_pos.y()
                ra, dec = self.wcs.pixel_to_world(scene_pos.x(), flipped_y)
                classifier = action.text()
                if classifier.startswith("GL"):
                    color = QColor(0, 200, 255)  # Bright blue
                elif classifier.startswith("AGN"):
                    color = QColor(220, 220, 0)  # Yellow
                else:  # Gx
                    color = QColor(255, 50, 50)  # Red
                normal_thickness = 2.0
                circle = QGraphicsEllipseItem(scene_pos.x() - 10, scene_pos.y() - 10, 20, 20)
                circle.setPen(QPen(color, normal_thickness))
                circle.setZValue(10)
                self.scene.addItem(circle)
                self.circles.append((circle, ra, dec, classifier, normal_thickness))
                if self.control_dock:
                    self.control_dock.update_coord_list(self.circles)
                self.viewport().update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # If screenshotting a sub-area
        if self.rectangle_selection:
            if self.start_point and self.rubber_band:
                self.end_point = self.mapToScene(event.pos())
                rect = QRectF(self.start_point, self.end_point).normalized()
                self.rubber_band.setRect(rect)
                if self.crosshair is not None:
                    self.crosshair.setPos(rect.center())
                    ch_x = rect.center().x()
                    ch_y = self.original_image.shape[0] - rect.center().y()
                    self.crosshair_ra, self.crosshair_dec = self.wcs.pixel_to_world(ch_x, ch_y)
            else:
                super().mouseMoveEvent(event)
            return
        
        scene_pos = self.mapToScene(event.pos())
        # Measuring distances in image (middle-mouse-button drag)
        if self.is_measuring:
            self.measure_end = scene_pos
            if self.wcs and self.original_image is not None:
                flipped_y_start = self.original_image.shape[0] - self.measure_start.y()
                flipped_y_end = self.original_image.shape[0] - self.measure_end.y()
                self.start_ra_dec = self.wcs.pixel_to_world(self.measure_start.x(), flipped_y_start)
                self.end_ra_dec = self.wcs.pixel_to_world(self.measure_end.x(), flipped_y_end)
                start_coord = SkyCoord(self.start_ra_dec[0] * u.deg, self.start_ra_dec[1] * u.deg, frame='icrs')
                end_coord = SkyCoord(self.end_ra_dec[0] * u.deg, self.end_ra_dec[1] * u.deg, frame='icrs')
                angular_offset_deg = start_coord.separation(end_coord).to(u.deg).value
                if angular_offset_deg * 3600 < 60:
                    self.angular_offset = angular_offset_deg * 3600
                    self.offset_unit = "\""
                elif angular_offset_deg < 1:
                    self.angular_offset = angular_offset_deg * 60
                    self.offset_unit = "\'"
                else:
                    self.angular_offset = None
                    self.offset_unit = None
                horizontal_coord = SkyCoord(self.end_ra_dec[0] * u.deg, self.start_ra_dec[1] * u.deg, frame='icrs')
                horizontal_offset_deg = start_coord.separation(horizontal_coord).to(u.deg).value
                if horizontal_offset_deg * 3600 < 60:
                    self.horizontal_offset = horizontal_offset_deg * 3600
                    self.horizontal_unit = "\""
                elif horizontal_offset_deg < 1:
                    self.horizontal_offset = horizontal_offset_deg * 60
                    self.horizontal_unit = "\'"
                else:
                    self.horizontal_offset = None
                    self.horizontal_unit = None
                vertical_coord = SkyCoord(self.start_ra_dec[0] * u.deg, self.end_ra_dec[1] * u.deg, frame='icrs')
                vertical_offset_deg = start_coord.separation(vertical_coord).to(u.deg).value
                if vertical_offset_deg * 3600 < 60:
                    self.vertical_offset = vertical_offset_deg * 3600
                    self.vertical_unit = "\""
                elif vertical_offset_deg < 1:
                    self.vertical_offset = vertical_offset_deg * 60
                    self.vertical_unit = "\'"
                else:
                    self.vertical_offset = None
                    self.vertical_unit = None
            self.viewport().update()
        # Update cursor coordinates display
        if self.wcs and self.control_dock and self.original_image is not None:
            flipped_y = self.original_image.shape[0] - scene_pos.y()
            ra, dec = self.wcs.pixel_to_world(scene_pos.x(), flipped_y)
            self.control_dock.update_cursor_display(scene_pos.x(), flipped_y, ra, dec)
        super().mouseMoveEvent(event)
        if self.control_dock:
            self.control_dock.update_magnifier(scene_pos)

    def mouseReleaseEvent(self, event):
        # If screenshotting a sub-area
        if self.rectangle_selection:
            if self.start_point and self.end_point and self.callback_on_selection:
                self.selection_rect = QRectF(self.start_point, self.end_point).normalized()
                self.scene.removeItem(self.rubber_band)
                self.rubber_band = None
                self.setCursor(Qt.ArrowCursor)
                if self.callback_on_selection:
                    self.callback_on_selection(self.selection_rect)
                    self.callback_on_selection = None
                self.start_point = None
                self.end_point = None
            else:
                super().mouseReleaseEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        self.dragging = False
        if event.button() == Qt.LeftButton:
            self.release_pos = event.pos()
            # Determine if this was a click (small movement) or a drag
            was_click = (abs(self.press_pos.x() - self.release_pos.x()) < 5 and
                         abs(self.press_pos.y() - self.release_pos.y()) < 5)

            if was_click:
                # Check for ellipse click
                for ellipse in self.catalog_manager.MER_items:
                    rect = ellipse.rect()
                    center = rect.center()
                    distance = ((scene_pos.x() - center.x())**2 + (scene_pos.y() - center.y())**2)**0.5
                    if distance <= 10:
                        object_id = ellipse.data(0)
                        if object_id is not None and self.control_dock:
                            self.control_dock.select_table_row(object_id)
                            # Highlight the ellipse with yellow color and 1.5x thickness
                            pen = QPen(QColor(255, 255, 0), 1.5)  # Yellow
                            ellipse.setPen(pen)
                            # Reset other ellipses to red
                            for other_ellipse in self.catalog_manager.MER_items:
                                if other_ellipse != ellipse:
                                    other_pen = QPen(QColor(255, 0, 0), 1.0)  # Red
                                    other_ellipse.setPen(other_pen)
                            self.scene.update()
                            self.dragging = False  # Prevent drag
                            event.accept()
                            return
                # Check for circle click
                for circle, ra, dec, classifier, normal_thickness in self.circles:
                    circle_rect = circle.rect()
                    circle_center = circle_rect.center()
                    distance = ((scene_pos.x() - circle_center.x())**2 + (scene_pos.y() - circle_center.y())**2)**0.5
                    if distance <= 10:
                        if self.control_dock:
                            self.control_dock.select_coord_list_item(ra, dec)
                            if self.control_dock.selected_circle and self.control_dock.selected_circle != circle:
                                normal_pen = self.control_dock.selected_circle.pen()
                                normal_pen.setWidthF(normal_thickness)
                                self.control_dock.selected_circle.setPen(normal_pen)
                            pen = QPen(circle.pen().color(), normal_thickness * 1.5)
                            circle.setPen(pen)
                            self.control_dock.selected_circle = circle
                            self.scene.update()
                        self.dragging = False  # Prevent drag
                        event.accept()
                        return

            # Handle drag (only if not a click on an ellipse/circle)
            if not was_click:
                self.refresh_preview()

        elif event.button() == Qt.MidButton:
            self.is_measuring = False
            self.measure_start = None
            self.measure_end = None
            self.start_ra_dec = None
            self.end_ra_dec = None
            self.angular_offset = None
            self.offset_unit = None
            self.horizontal_offset = None
            self.horizontal_unit = None
            self.vertical_offset = None
            self.vertical_unit = None
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().update()

        self.dragging = False
        super().mouseReleaseEvent(event)


    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
        super().mouseDoubleClickEvent(event)
        
    def wheelEvent(self, event):
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        angle = event.angleDelta().y()
        factor = 1.2 if angle > 0 else 1 / 1.2
        self.scale_factor *= factor
        self.scale(factor, factor)
        if self.control_dock:
            self.control_dock.update_preview(self.last_pixmap)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete and self.control_dock:
            ra = None
            dec = None
            # Priority 1: Use thickened circle (selected_circle) if it exists
            if self.control_dock.selected_circle:
                for circle, circle_ra, circle_dec, classifier, normal_thickness in self.circles:
                    if circle == self.control_dock.selected_circle:
                        ra, dec = circle_ra, circle_dec
                        break
            # Priority 2: Fall back to coord_list selection
            if ra is None or dec is None:
                ra, dec = self.control_dock.get_selected_coord()
            # Delete the matching circle
            if ra is not None and dec is not None:
                for circle, circle_ra, circle_dec, classifier, normal_thickness in self.circles[:]:
                    if abs(circle_ra - ra) < 1e-5 and abs(circle_dec - dec) < 1e-5:
                        self.scene.removeItem(circle)
                        self.circles.remove((circle, circle_ra, circle_dec, classifier, normal_thickness))
                        if self.control_dock.selected_circle == circle:
                            self.control_dock.selected_circle = None
                        self.control_dock.update_coord_list(self.circles)
                        self.viewport().update()
                        break
        super().keyPressEvent(event)

    def drawForeground(self, painter, rect):
        # this draws a ruler when dragging with the middle mouse button
        super().drawForeground(painter, rect)
        if self.is_measuring and self.measure_start and self.measure_end:
            painter.setRenderHint(QPainter.Antialiasing)
            solid_pen = QPen(QColor(255, 255, 0), 1, Qt.SolidLine)
            solid_pen.setCosmetic(True)
            painter.setPen(solid_pen)
            painter.drawLine(self.measure_start, self.measure_end)
            dashed_pen = QPen(QColor(255, 255, 0), 1, Qt.DashLine)
            dashed_pen.setCosmetic(True)
            painter.setPen(dashed_pen)
            painter.drawLine(self.measure_start, QPointF(self.measure_end.x(), self.measure_start.y()))
            painter.drawLine(QPointF(self.measure_end.x(), self.measure_start.y()), self.measure_end)
            painter.setWorldMatrixEnabled(False)
            font = QFont("Arial", 10)
            pen = QPen(QColor(255, 255, 0), 1)
            painter.setFont(font)
            painter.setPen(pen)
            if self.angular_offset is not None and self.offset_unit:
                mid_x = (self.measure_start.x() + self.measure_end.x()) / 2
                mid_y = (self.measure_start.y() + self.measure_end.y()) / 2
                mid_scene = QPointF(mid_x, mid_y)
                mid_viewport = self.mapFromScene(mid_scene)
                label_pos = QPointF(mid_viewport.x(), mid_viewport.y() - 10)
                text = f"{self.angular_offset:.2f} {self.offset_unit}"
                painter.drawText(label_pos, text)
            if self.horizontal_offset is not None and self.horizontal_unit:
                hor_x = (self.measure_start.x() + self.measure_end.x()) / 2
                hor_y = self.measure_start.y()
                hor_scene = QPointF(hor_x, hor_y)
                hor_viewport = self.mapFromScene(hor_scene)
                label_pos = QPointF(hor_viewport.x(), hor_viewport.y() + 20)
                text = f"{self.horizontal_offset:.2f} {self.horizontal_unit}"
                painter.drawText(label_pos, text)
            if self.vertical_offset is not None and self.vertical_unit:
                ver_x = self.measure_end.x()
                ver_y = (self.measure_start.y() + self.measure_end.y()) / 2
                ver_scene = QPointF(ver_x, ver_y)
                ver_viewport = self.mapFromScene(ver_scene)
                label_pos = QPointF(ver_viewport.x() + 10, ver_viewport.y())
                text = f"{self.vertical_offset:.2f} {self.vertical_unit}"
                painter.drawText(label_pos, text)
            painter.setWorldMatrixEnabled(True)


    ######################################################
    # Zoom functions
    ######################################################
    def zoom_in(self):
        self.scale_factor *= 1.2
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(1.2, 1.2)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        if self.control_dock:
            self.control_dock.update_preview(self.last_pixmap)

    def zoom_out(self):
        self.scale_factor /= 1.2
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(1/1.2, 1/1.2)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        if self.control_dock:
            self.control_dock.update_preview(self.last_pixmap)

    def reset_zoom(self):
        if self.scale_factor != 1.0:
            inverse_scale = 1.0 / self.scale_factor
            self.scale(inverse_scale, inverse_scale)
            self.scale_factor = 1.0
            if self.control_dock:
                self.control_dock.update_preview(self.last_pixmap)

    def reset_transform(self):
        self.resetTransform()
        self.scale_factor = 1.0

    def fit_to_view(self):
        if self.last_pixmap and self.image_item:
            self.reset_zoom()
            view_rect = self.viewport().rect()
            scene_rect = self.sceneRect()
            scale_x = view_rect.width() / scene_rect.width()
            scale_y = view_rect.height() / scene_rect.height()
            scale = min(scale_x, scale_y)
            self.scale_factor = scale
            self.scale(scale, scale)
            self.centerOn(scene_rect.center())
            if self.control_dock:
                self.control_dock.update_preview(self.last_pixmap)

    def refresh_preview(self):
        if self.control_dock and self.last_pixmap:
            self.control_dock.update_preview(self.last_pixmap)

    def get_current_view_center(self):
        """Gets the current center point of the visible scene."""
        viewport_center = self.viewport().rect().center()
        scene_center = self.mapToScene(viewport_center)
        return scene_center

    def restore_view_center(self, scene_center):
        """Centers the view on the given scene point."""
        self.centerOn(scene_center)

    def image_saver(self, targetImage, filename):
        if not os.path.exists(filename):
            if targetImage.save(filename, "png"):
                self.update_status(f"Image saved as: {filename}")
                return True
            else:
                QMessageBox.critical(None, "Error", f"Failed to save image to: {filename}")
                return False
        else:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(None, "File Already Exists - Choose New Name",
                                                        filename,
                                                        "PNG Files (*.png);;All Files (*)",
                                                        options=options)
            if filename:
                if targetImage.save(filename, "png"):
                    self.update_status(f"Image saved as: {filename}")
                    return True
                else:
                    QMessageBox.critical(None, "Error", f"Failed to save image to: {filename}")
                    return False
            else:
                print("Saving cancelled by user.")
                return False
        
    def save_full_image_with_overlays(self):
        # To store a very large image with overlays, we must zoom out so that the full image is visible.
        # Otherwise, the code crashes, or the catalog items are not displayed
        current_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        self.fit_to_view()   # we are not updating the view, so this runs in the background 
        self.save_visible_area_with_overlays()
        # restore the view
        self.fitInView(current_rect)
        

    def save_visible_area_with_overlays(self):
        """
        Saves the currently visible area of the graphics view, including all
        overlays, as a PNG file at the native resolution (zoom level 1.0).
        """
        if not self.qimage:
            print("No image loaded.")
            return

        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        # Get the visible rectangle in scene coordinates
        visible_rect_scene = self.mapToScene(self.viewport().rect()).boundingRect()

        # Ensure the visible rect is within the bounds of the original image
        image_rect = QRectF(0, 0, self.qimage.width(), self.qimage.height())
        clipped_rect_scene = visible_rect_scene.intersected(image_rect)

        if clipped_rect_scene.isEmpty():
            print("Visible area is empty or outside image bounds.")
            return

        # Convert the visible scene rectangle to image coordinates (at zoom level 1.0)
        target_size = clipped_rect_scene.size().toSize()
        target = QImage(target_size, QImage.Format_ARGB32_Premultiplied)
        target.fill(Qt.transparent)

        painter = QPainter(target)
        painter.setRenderHint(QPainter.Antialiasing) # Optional: Enable anti-aliasing

        # Render the scene onto the target image, translating so the visible area aligns
        self.scene.render(painter, QRectF(target.rect()), clipped_rect_scene)
        rect_x = clipped_rect_scene.center().x()
        rect_y = clipped_rect_scene.center().y()
        rect_y = self.original_image.shape[0] - rect_y
        cen_ra, cen_dec = self.wcs.pixel_to_world(rect_x, rect_y)

        painter.end()

        filename = f"{self.dirpath}/cutout_{cen_ra:.6f}_{cen_dec:.5f}.png"
        self.image_saver(target, filename)
        QApplication.restoreOverrideCursor()

    def save_area_from_selection(self, rect: QRectF):
        """
        Saves the area within the given rectangle (in scene coordinates),
        including all overlays, as a PNG file at the native resolution (zoom level 1.0).
        """
        if not self.qimage:
            print("No image loaded.")
            return

        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self.rectangle_selection = True

        # Ensure the selection rect is within the bounds of the original image
        image_rect = QRectF(0, 0, self.qimage.width(), self.qimage.height())
        clipped_rect_scene = rect.intersected(image_rect)

        if clipped_rect_scene.isEmpty():
            print("Selected area is empty or outside image bounds.")
            self.rectangle_selection = False
            return

        # Convert the selected scene rectangle to image coordinates (at zoom level 1.0)
        target_size = clipped_rect_scene.size().toSize()
        target = QImage(target_size, QImage.Format_ARGB32_Premultiplied)
        target.fill(Qt.transparent)

        painter = QPainter(target)
        painter.setRenderHint(QPainter.Antialiasing) # Optional: Enable anti-aliasing

        # Render the scene onto the target image, translating so the selected area aligns
        source_rect_scene = clipped_rect_scene
        target_rect = QRectF(0, 0, target.width(), target.height())
        self.scene.render(painter, target_rect, source_rect_scene)

        painter.end()
        filename = f"{self.dirpath}/{self.tileID}_cutout_{self.crosshair_ra:.6f}_{self.crosshair_dec:.5f}.png"
        self.image_saver(target, filename)

        self.rectangle_selection = False
        QApplication.restoreOverrideCursor()
        QApplication.setOverrideCursor(Qt.ArrowCursor)

        
    def handle_selection_rect(self, rect: QRectF):  # Internal callback
        self.setCursor(Qt.CrossCursor)
        """Handles the selected rectangle and saves the area."""
        self.save_area_from_selection(rect)

    def set_image(self, qimage: QImage):
        self.qimage = qimage
        if qimage:
            pixmap = QPixmap.fromImage(qimage)
            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.setSceneRect(self.scene.itemsBoundingRect())
            self.reset_selection()

    def add_graphics_item(self, item):
        self.scene.addItem(item)

    def reset_selection(self):
        if self.rubber_band:
            self.scene.removeItem(self.rubber_band)
            self.rubber_band = None
        self.start_point = None
        self.end_point = None
        self.selection_rect = None
        self.setCursor(Qt.ArrowCursor)
        
    def start_selection(self):  # Removed callback argument
        self.rectangle_selection = True
        QApplication.setOverrideCursor(Qt.CrossCursor)
        self.start_point = None
        self.end_point = None
        self.selection_rect = None
        self.callback_on_selection = self.handle_selection_rect  # Set the callback here

