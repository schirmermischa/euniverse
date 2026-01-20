import os
from PyQt5.QtWidgets import QDialog, QAbstractItemView, QTableView, QVBoxLayout, QHeaderView, QGraphicsEllipseItem, QGraphicsView 
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QPointF
from PyQt5.QtGui import QColor, QPen
from astropy.table import Table, MaskedColumn
import numpy as np

class CatalogTableModel(QAbstractTableModel):
    """
    A table model to display an astropy Table in a QTableView,
    with support for filtering columns and preserving original data types.
    """

    def __init__(self, catalog, required_columns=None, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.required_columns = required_columns
        self.columns = self._get_visible_columns()
        self.data_types = self._get_column_data_types()

    def sort(self, column, order):
        """Sort table by a column index."""
        if self.catalog is None or len(self.catalog) == 0:
            return

        # Notify the view that the layout is about to change
        self.layoutAboutToBeChanged.emit()

        # Get the name of the column to sort by
        col_name = self.columns[column]
        
        # Astropy tables can be sorted in-place
        # order == 0 is Ascending, order == 1 is Descending
        reverse = (order == Qt.DescendingOrder)
        self.catalog.sort(col_name, reverse=reverse)

        # Notify the view that the layout has changed
        self.layoutChanged.emit()

    def _get_visible_columns(self):
        """
        Determines which columns to display from the catalog,
        respecting the required_columns if provided.
        Excludes columns with object types.
        """
        visible_columns = []
        all_columns = self.catalog.colnames
        if self.required_columns:
            for col in self.required_columns:
                if col in all_columns and self.catalog[col].dtype.kind not in ['O', 'U', 'S']:
                    visible_columns.append(col)
        else:
            for col in all_columns:
                if self.catalog[col].dtype.kind not in ['O', 'U', 'S']:
                    visible_columns.append(col)
        return visible_columns

    def _get_column_data_types(self):
        """
        Gets the data types of the visible columns.
        """
        data_types = {}
        for col in self.columns:
            data_types[col] = self.catalog[col].dtype
        return data_types

    def rowCount(self, parent=QModelIndex()):
        return len(self.catalog) if self.catalog is not None else 0

    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not self.catalog:
            return None

        row = index.row()
        col = index.column()
        col_name = self.columns[col]
        value = self.catalog[row][col_name]

        if role == Qt.DisplayRole:
            # Handle Masked Values
            if hasattr(value, 'mask') and value.mask:
                return "" # Or "NaN" / "null" depending on preference

            # Maintain Numeric Precision for Floats
            if isinstance(value, (np.float32, np.float64)):
                # trim=None ensures we don't truncate necessary precision
                # positional=True avoids scientific notation unless necessary
                return np.format_float_positional(value, trim='-') 
            
            # Fallback for integers and strings
            return str(value)

        elif role == Qt.TextAlignmentRole:
            # Ensure we are checking the native numpy dtype to align columns
            if np.issubdtype(self.catalog[col_name].dtype, np.number):
                return Qt.AlignRight | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter

        elif role == Qt.BackgroundRole:
            if row % 2 == 0:
                return QColor(240, 240, 240)
        
        return None


    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns[section]
        return None


class TableDialog(QDialog):
    def __init__(self, catalog, viewer, catalog_name=None, parent=None):
        super().__init__(parent)
        # Determine window title based on filename

        if catalog_name:
            self.setWindowTitle(f"{catalog_name}")
        else:
            self.setWindowTitle("Catalog Table")

        self.catalog = catalog
        self.viewer = viewer

        # Initial sizing
        self.resize(1000, 400)

        self.table_view = QTableView()

        # --- Enable sorting ---
        self.table_view.setSortingEnabled(True)
        
        # The subset of columns to display
        self.required_columns = [
            'OBJECT_ID', 'RIGHT_ASCENSION', 'DECLINATION', 'MAG_VIS_PSF',
            'MAG_Y_TEMPLFIT', 'MAG_J_TEMPLFIT', 'MAG_H_TEMPLFIT', 'FWHM', 'KRON_RADIUS'
        ]
        
        self.table_model = CatalogTableModel(catalog, required_columns=self.required_columns)
        self.table_view.setModel(self.table_model)

        # --- PERFORMANCE OPTIMIZATIONS ---
        header = self.table_view.horizontalHeader()
        
        # 1. Fit columns to content once
        self.table_view.resizeColumnsToContents()
        
        # 2. Switch to Interactive mode to freeze widths
        header.setSectionResizeMode(QHeaderView.Interactive) 
        header.setStretchLastSection(False) 

        # 3. Optimize Vertical Header (Row heights)
        v_header = self.table_view.verticalHeader()
        v_header.setDefaultSectionSize(25)
        v_header.setSectionResizeMode(QHeaderView.Fixed)

        # 4. General View Settings
        self.table_view.setWordWrap(False)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)

        # Connect signals
        self.table_view.clicked.connect(self.on_row_selected)

        # Create layout (keeping default margins)
        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        self.setLayout(layout)

        # --- DYNAMIC WIDTH CALCULATION ---
        # We need to process events to ensure the headers and scrollbars have 
        # calculated their own sizes before we ask for them.
        self.table_view.horizontalHeader().setMinimumSectionSize(0)
        
        # Calculate sum of all column widths
        content_width = 0
        for i in range(len(self.required_columns)):
            content_width += self.table_view.columnWidth(i)
        
        # Add width of the vertical header (the row index column)
        content_width += v_header.width()
        
        # Add width of the vertical scrollbar
        content_width += self.table_view.verticalScrollBar().sizeHint().width()
        
        # Add the frame width of the QTableView itself (left + right borders)
        content_width += self.table_view.frameWidth() * 2
        
        # Add the layout margins (left + right)
        margins = layout.contentsMargins()
        total_width = content_width + margins.left() + margins.right()

        # Resize the dialog to the exact content width
        self.resize(total_width, 400)

        self.selected_row = None


    def on_row_selected(self, index):
        """
        Handles row selection: highlights the object in the viewer and centers it.
        """
        if not index.isValid() or self.catalog is None or self.viewer is None:
            return

        self.selected_row = index.row()
        # Retrieve the OBJECT_ID from the actual catalog data using the row index
        object_id = self.catalog['OBJECT_ID'][index.row()]

        # Iterate through scene items to find the matching QGraphicsEllipseItem
        for item in self.viewer.scene.items():
            if isinstance(item, QGraphicsEllipseItem) and item.data(0) == object_id:
                # Highlight the selected ellipse in yellow
                pen = QPen(QColor(255, 255, 0), 1.5)
                item.setPen(pen)

                # Center the ImageViewer on this object
                rect = item.rect()
                center_scene = item.mapToScene(rect.center())
                self.viewer.centerOn(center_scene)
                
                # Reset cursor state for the viewport
                self.viewer.setDragMode(QGraphicsView.NoDrag)
                self.viewer.viewport().setCursor(Qt.ArrowCursor)
            
            elif isinstance(item, QGraphicsEllipseItem):
                # Reset other ellipses to red
                pen = QPen(QColor(255, 0, 0), 1.0)
                item.setPen(pen)

        self.viewer.scene.update()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create table view
        self.table_view = QTableView()
        self.table_model = CatalogTableModel(self.catalog, self.required_columns)  # Pass required_columns
        self.table_view.setModel(self.table_model)

        # Configure table appearance
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Make it read-only
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        layout.addWidget(self.table_view)
        self.setLayout(layout)
        self.resize(800, 600)

        # Connect row selection
        self.table_view.clicked.connect(self.on_row_clicked)

    def on_row_clicked(self, index):
        """
        Handles row clicks in the table: highlights ellipse and centers the *view* on it.
        Does not change the drag mode.
        """
        if not self.viewer or not self.catalog or not self.viewer.wcs or self.viewer.original_image is None:
            print("Invalid state: viewer, catalog, wcs, or original_image missing")
            return

        self.selected_row = index.row()
        object_id = self.catalog['OBJECT_ID'][index.row()]

        # Highlight corresponding ellipse
        for item in self.viewer.scene.items():
            if isinstance(item, QGraphicsEllipseItem) and item.data(0) == object_id:
                pen = QPen(QColor(255, 255, 0), 1.5)  # Yellow
                item.setPen(pen)

                # Center the *view* on the ellipse's center
                rect = item.rect()
                center_scene = item.mapToScene(rect.center())
                self.viewer.centerOn(center_scene)
                self.viewer.setDragMode(QGraphicsView.NoDrag)
                self.viewer.viewport().setCursor(Qt.ArrowCursor)  # Change the cursor too
            elif isinstance(item, QGraphicsEllipseItem) and item.data(0) != object_id:
                pen = QPen(QColor(255, 0, 0), 1.0)  # Red
                item.setPen(pen)

        self.viewer.scene.update()

    def select_row_by_object_id(self, object_id):
        if not self.catalog:
            return

        # Find row with matching OBJECT_ID
        for row in range(len(self.catalog)):
            if self.catalog['OBJECT_ID'][row] == object_id:
                index = self.table_model.index(row, 0)  # Get index for the first column
                self.table_view.selectRow(row)
                self.table_view.setCurrentIndex(index)
                self.selected_row = row
                return
